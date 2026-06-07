from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


def _segments():
    return [
        {"start": 0.0, "end": 5.0, "text": "Here is the first hook."},
        {"start": 5.0, "end": 11.0, "text": "This explains why the trick works."},
        {"start": 11.0, "end": 18.0, "text": "The result is faster and cleaner."},
    ]


def _poll_job(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}")
        data = resp.get_json()
        if data["status"] in ("complete", "error", "cancelled"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_plan_from_cached_transcript_is_deterministic(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "interview.mp4"
    media.write_bytes(b"fake")
    payload = {
        "transcript_segments": _segments(),
        "max_candidates": 2,
        "min_duration": 10,
        "platform_presets": ["youtube_shorts", "tiktok"],
    }

    first = build_magic_clips_plan(str(media), payload)
    second = build_magic_clips_plan(str(media), payload)
    first_json = {key: first[key] for key in first.keys()}
    second_json = {key: second[key] for key in second.keys()}

    assert first_json == second_json
    assert first.dry_run is True
    assert first.requires_analysis is False
    assert first.source_path_hash.startswith("src:")
    assert first.config_hash.startswith("cfg:")
    assert first.candidates[0].candidate_id.startswith("mcc:")
    assert len(first.candidates[0].estimated_outputs) == 2
    assert {output["preset_id"] for output in first.candidates[0].estimated_outputs} == {
        "youtube_shorts",
        "tiktok",
    }
    assert all(step.status == "planned" for step in first.steps)


def test_plan_from_cached_highlights_includes_reasons_and_steps(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "episode.mp4"
    media.write_bytes(b"fake")
    plan = build_magic_clips_plan(
        str(media),
        {
            "transcript_segments": _segments(),
            "highlights": [
                {
                    "start": 5.0,
                    "end": 18.0,
                    "title": "Why it works",
                    "score": 87.4,
                    "reason": "strong explanation",
                }
            ],
            "caption_style": "bold_yellow",
            "burn_captions": True,
            "min_duration": 5,
        },
    )

    candidate = plan.candidates[0]
    assert candidate.title == "Why it works"
    assert candidate.score > 0
    assert candidate.score_breakdown["highlight_score"] == 87.4
    assert candidate.selection_reason.startswith("strong explanation")
    assert candidate.fallback_mode == "heuristic_no_llm"
    assert "strong explanation" in candidate.reasons
    assert "why the trick works" in candidate.transcript_excerpt
    assert [step.step_type for step in plan.steps] == ["trim", "reframe", "captions", "export"]
    assert plan.steps[-1].depends_on == [plan.steps[-2].step_id]


def test_candidates_are_ranked_with_deterministic_tie_breaks(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "ranking.mp4"
    media.write_bytes(b"fake")
    plan = build_magic_clips_plan(
        str(media),
        {
            "transcript_segments": [
                {"start": 0.0, "end": 12.0, "text": "watch this exact idea"},
                {"start": 20.0, "end": 32.0, "text": "watch this exact idea"},
            ],
            "highlights": [
                {"start": 20.0, "end": 32.0, "title": "Second", "score": 70},
                {"start": 0.0, "end": 12.0, "title": "First", "score": 70},
            ],
            "min_duration": 5,
            "max_candidates": 2,
        },
    )

    assert [candidate.title for candidate in plan.candidates] == ["First", "Second"]
    assert plan.candidates[0].score == plan.candidates[1].score


def test_rejected_candidates_record_too_short_overlap_and_malformed_inputs(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "rejects.mp4"
    media.write_bytes(b"fake")
    plan = build_magic_clips_plan(
        str(media),
        {
            "transcript_segments": [
                {"start": 0.0, "end": 12.0, "text": "why this clip works", "speaker": "A"},
                {"start": 7.0, "end": 3.0, "text": "bad"},
            ],
            "highlights": [
                {"start": 0.0, "end": 12.0, "title": "Keeper", "score": 95},
                {"start": 2.0, "end": 10.0, "title": "Overlap", "score": 80},
                {"start": 30.0, "end": 32.0, "title": "Too short", "score": 99},
                {"start": 50.0, "end": 40.0, "title": "Bad"},
            ],
            "min_duration": 5,
            "max_candidates": 3,
        },
    )

    assert [candidate.title for candidate in plan.candidates] == ["Keeper"]
    reasons = {item["reason"] for item in plan.rejected_candidates}
    assert "malformed_transcript_segment" in reasons
    assert "malformed_highlight" in reasons
    assert "too_short" in reasons
    assert "overlap_with_selected" in reasons


def test_heuristic_fallback_metadata_surfaces_without_llm(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "fallback.mp4"
    media.write_bytes(b"fake")
    plan = build_magic_clips_plan(
        str(media),
        {
            "transcript_segments": [
                {"start": 0.0, "end": 16.0, "text": "why you should stop making this mistake now"},
            ],
            "min_duration": 5,
        },
    )

    assert plan.fallback_mode == "heuristic_no_llm"
    candidate = plan.candidates[0]
    assert candidate.fallback_mode == "heuristic_no_llm"
    assert candidate.score_breakdown["transcript_hook"] > 0
    assert "duration_fit" in candidate.score_breakdown


def test_plan_platform_contracts_cover_social_targets(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "platforms.mp4"
    media.write_bytes(b"fake")
    presets = ["youtube_shorts", "tiktok", "instagram_reels", "instagram_feed"]
    plan = build_magic_clips_plan(
        str(media),
        {
            "transcript_segments": _segments(),
            "platform_presets": presets,
            "min_duration": 10,
        },
    )

    candidate = plan.candidates[0]
    assert [platform["preset_id"] for platform in candidate.platform_presets] == presets
    outputs = {output["preset_id"]: output for output in candidate.estimated_outputs}
    assert outputs["youtube_shorts"]["width"] == 1080
    assert outputs["youtube_shorts"]["height"] == 1920
    assert outputs["youtube_shorts"]["max_duration"] == 60
    assert outputs["tiktok"]["max_duration"] == 600
    assert outputs["instagram_reels"]["max_duration"] == 90
    assert outputs["instagram_feed"]["width"] == 1080
    assert outputs["instagram_feed"]["height"] == 1080
    assert all(candidate.candidate_id.replace(":", "_") in item["filename_template"] for item in outputs.values())


def test_no_cached_data_returns_analysis_required_steps(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "long.mp4"
    media.write_bytes(b"fake")
    before = sorted(path.name for path in tmp_path.iterdir())
    plan = build_magic_clips_plan(str(media), {"max_candidates": 2})
    after = sorted(path.name for path in tmp_path.iterdir())

    assert before == after
    assert plan.requires_analysis is True
    assert plan.candidates == []
    assert [step.status for step in plan.steps] == ["requires_analysis", "requires_analysis"]
    assert [step.step_type for step in plan.steps] == ["transcribe", "highlight_extract"]


def test_invalid_config_rejected(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    try:
        build_magic_clips_plan(str(media), {"platform_presets": ["unknown"]})
    except ValueError as exc:
        assert "Unsupported platform preset" in str(exc)
    else:
        raise AssertionError("expected invalid preset to fail")


def test_result_subscript_protocol(tmp_path):
    from opencut.core.magic_clips import build_magic_clips_plan

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    plan = build_magic_clips_plan(str(media), {"transcript_segments": _segments(), "min_duration": 10})

    assert "candidates" in plan.keys()
    assert plan["candidates"][0]["candidate_id"] == plan.candidates[0].candidate_id
    assert plan["steps"][0]["step_id"] == plan.steps[0].step_id


def test_magic_clips_plan_route(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    media = tmp_path / "route.mp4"
    media.write_bytes(b"fake")
    resp = client.post(
        "/video/magic-clips/plan",
        data=json.dumps({
            "filepath": str(media),
            "transcript_segments": _segments(),
            "platform_presets": ["youtube_shorts"],
            "min_duration": 10,
        }),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["dry_run"] is True
    assert body["requires_analysis"] is False
    assert body["candidates"][0]["candidate_id"].startswith("mcc:")
    assert body["steps"][-1]["step_type"] == "export"


def test_magic_clips_plan_route_rejects_invalid_config(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    media = tmp_path / "route.mp4"
    media.write_bytes(b"fake")
    resp = client.post(
        "/video/magic-clips/plan",
        data=json.dumps({"filepath": str(media), "max_duration": 3, "min_duration": 10}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400


def test_shorts_pipeline_renders_only_approved_magic_candidates(tmp_path, monkeypatch):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")
    output_dir = tmp_path / "out"
    trim_calls = []
    conform_calls = []

    def fake_trim(_input_path, start, end, output_path):
        trim_calls.append((start, end))
        Path(output_path).write_bytes(b"clip")

    def fake_conform(input_path, width, height, output_path):
        conform_calls.append((Path(input_path).name, width, height, Path(output_path).name))
        Path(output_path).write_bytes(Path(input_path).read_bytes())

    monkeypatch.setattr("opencut.core.shorts_pipeline._trim_clip", fake_trim)
    monkeypatch.setattr("opencut.core.shorts_pipeline._conform_clip_dimensions", fake_conform)
    monkeypatch.setattr("opencut.core.shorts_pipeline._probe_duration", lambda _path: 0.0)

    clips = generate_shorts(
        str(media),
        config=ShortsPipelineConfig(
            max_shorts=5,
            face_track=False,
            burn_captions=False,
            approved_plan_id="mcplan:test",
            approved_candidates=[
                {"candidate_id": "mcc:first", "start": 10.0, "end": 25.0, "title": "First hook", "score": 88.0},
                {"candidate_id": "mcc:second", "start": 40.0, "end": 52.5, "title": "Second turn", "score": 74.0},
            ],
        ),
        output_dir=str(output_dir),
    )

    assert trim_calls == [(10.0, 25.0), (40.0, 52.5)]
    assert conform_calls == []
    assert [clip.title for clip in clips] == ["First hook", "Second turn"]
    assert [clip.score for clip in clips] == [88.0, 74.0]
    assert sorted(path.name for path in output_dir.glob("*.mp4")) == [
        "source_short_1_First_hook.mp4",
        "source_short_2_Second_turn.mp4",
    ]


def test_shorts_pipeline_platform_presets_render_variants_and_clamp_duration(tmp_path, monkeypatch):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")
    output_dir = tmp_path / "out"
    trim_calls = []
    conform_calls = []

    def fake_trim(_input_path, start, end, output_path):
        trim_calls.append((start, end, Path(output_path).name))
        Path(output_path).write_bytes(b"clip")

    def fake_conform(input_path, width, height, output_path):
        conform_calls.append((Path(input_path).name, width, height, Path(output_path).name))
        Path(output_path).write_bytes(Path(input_path).read_bytes())

    monkeypatch.setattr("opencut.core.shorts_pipeline._trim_clip", fake_trim)
    monkeypatch.setattr("opencut.core.shorts_pipeline._conform_clip_dimensions", fake_conform)
    monkeypatch.setattr("opencut.core.shorts_pipeline._probe_duration", lambda _path: 0.0)

    clips = generate_shorts(
        str(media),
        config=ShortsPipelineConfig(
            max_shorts=1,
            face_track=False,
            burn_captions=False,
            approved_candidates=[
                {"candidate_id": "mcc:first", "start": 0.0, "end": 120.0, "title": "Long hook", "score": 88.0},
            ],
            platform_presets=["youtube_shorts", "tiktok", "instagram_reels", "instagram_feed"],
        ),
        output_dir=str(output_dir),
    )

    assert [(start, end) for start, end, _name in trim_calls] == [
        (0.0, 60.0),
        (0.0, 120.0),
        (0.0, 90.0),
        (0.0, 60.0),
    ]
    assert [clip.platform_preset for clip in clips] == [
        "youtube_shorts",
        "tiktok",
        "instagram_reels",
        "instagram_feed",
    ]
    assert [(clip.width, clip.height) for clip in clips] == [
        (1080, 1920),
        (1080, 1920),
        (1080, 1920),
        (1080, 1080),
    ]
    assert [(width, height) for _name, width, height, _output in conform_calls] == [
        (1080, 1920),
        (1080, 1920),
        (1080, 1920),
        (1080, 1080),
    ]
    assert sorted(path.name for path in output_dir.glob("*.mp4")) == [
        "source_short_1_instagram_feed_Long_hook.mp4",
        "source_short_1_instagram_reels_Long_hook.mp4",
        "source_short_1_tiktok_Long_hook.mp4",
        "source_short_1_youtube_shorts_Long_hook.mp4",
    ]


def test_shorts_pipeline_manifest_survives_cancel_after_transcribe(tmp_path, monkeypatch):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")
    output_dir = tmp_path / "out"
    calls = {"transcribe": 0, "highlight": 0}

    def fake_transcribe(_input_path, config):
        calls["transcribe"] += 1
        return {"segments": _segments()}

    def fail_after_transcribe(**_kwargs):
        calls["highlight"] += 1
        raise RuntimeError("cancel after transcribe")

    monkeypatch.setattr("opencut.core.captions.transcribe", fake_transcribe)
    monkeypatch.setattr("opencut.core.highlights.extract_highlights", fail_after_transcribe)

    config = ShortsPipelineConfig(
        face_track=False,
        burn_captions=False,
        checkpoint_manifest_path=str(
            output_dir / "magic_clips_runs" / "cancel-after-transcribe" / "magic_clips_run_manifest.json"
        ),
    )
    with pytest.raises(RuntimeError, match="cancel after transcribe"):
        generate_shorts(str(media), config=config, output_dir=str(output_dir))

    manifest_path = Path(config.checkpoint_manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == {"transcribe": 1, "highlight": 1}
    assert manifest["status"] == "interrupted"
    assert manifest["next_step"] == "highlight"
    assert manifest["transcript_cache_id"]
    assert len(manifest["transcript_segments"]) == len(_segments())
    assert Path(manifest["intermediates_dir"]).is_dir()


def test_shorts_pipeline_resume_skips_completed_render_outputs(tmp_path, monkeypatch):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")
    output_dir = tmp_path / "out"
    trim_calls = []
    copy_calls = {"count": 0}

    def fake_trim(_input_path, start, end, output_path):
        trim_calls.append((start, end, Path(output_path).name))
        Path(output_path).write_bytes(b"clip")

    def fail_after_first_copy(input_path, output_path):
        copy_calls["count"] += 1
        if copy_calls["count"] == 2:
            raise RuntimeError("cancel after first render")
        Path(output_path).write_bytes(Path(input_path).read_bytes())

    monkeypatch.setattr("opencut.core.shorts_pipeline._trim_clip", fake_trim)
    monkeypatch.setattr("opencut.core.shorts_pipeline._probe_duration", lambda _path: 0.0)
    monkeypatch.setattr("opencut.core.shorts_pipeline.shutil.copy2", fail_after_first_copy)

    config = ShortsPipelineConfig(
        max_shorts=2,
        face_track=False,
        burn_captions=False,
        approved_candidates=[
            {"candidate_id": "mcc:first", "start": 10.0, "end": 25.0, "title": "First hook", "score": 88.0},
            {"candidate_id": "mcc:second", "start": 40.0, "end": 52.5, "title": "Second turn", "score": 74.0},
        ],
    )
    with pytest.raises(RuntimeError, match="cancel after first render"):
        generate_shorts(str(media), config=config, output_dir=str(output_dir))

    manifest_path = Path(config.checkpoint_manifest_path)
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["status"] == "interrupted"
    assert len(first_manifest["final_exports"]) == 1
    assert Path(first_manifest["final_exports"][0]["output_path"]).is_file()

    def copy_ok(input_path, output_path):
        copy_calls["count"] += 1
        Path(output_path).write_bytes(Path(input_path).read_bytes())

    trim_calls.clear()
    copy_calls["count"] = 0
    monkeypatch.setattr("opencut.core.shorts_pipeline.shutil.copy2", copy_ok)
    resume_config = ShortsPipelineConfig(
        max_shorts=2,
        face_track=False,
        burn_captions=False,
        checkpoint_manifest_path=str(manifest_path),
        approved_candidates=[
            {"candidate_id": "mcc:first", "start": 10.0, "end": 25.0, "title": "First hook", "score": 88.0},
            {"candidate_id": "mcc:second", "start": 40.0, "end": 52.5, "title": "Second turn", "score": 74.0},
        ],
    )

    clips = generate_shorts(str(media), config=resume_config, output_dir=str(output_dir))
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(clips) == 2
    assert trim_calls == []
    assert copy_calls["count"] == 1
    assert final_manifest["status"] == "complete"
    assert final_manifest["next_step"] == "complete"
    assert len(final_manifest["final_exports"]) == 2


def test_shorts_pipeline_resume_restarts_on_config_mismatch(tmp_path, monkeypatch):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")
    output_dir = tmp_path / "out"
    trim_calls = []

    def fake_trim(_input_path, start, end, output_path):
        trim_calls.append((start, end, Path(output_path).name))
        Path(output_path).write_bytes(b"clip")

    def fake_copy(input_path, output_path):
        Path(output_path).write_bytes(Path(input_path).read_bytes())

    monkeypatch.setattr("opencut.core.shorts_pipeline._trim_clip", fake_trim)
    monkeypatch.setattr("opencut.core.shorts_pipeline._probe_duration", lambda _path: 0.0)
    monkeypatch.setattr("opencut.core.shorts_pipeline.shutil.copy2", fake_copy)

    base_candidates = [
        {"candidate_id": "mcc:first", "start": 10.0, "end": 25.0, "title": "First hook", "score": 88.0},
    ]
    config = ShortsPipelineConfig(
        face_track=False,
        burn_captions=False,
        caption_style="default",
        approved_candidates=base_candidates,
    )
    generate_shorts(str(media), config=config, output_dir=str(output_dir))
    manifest_path = Path(config.checkpoint_manifest_path)

    trim_calls.clear()
    changed = ShortsPipelineConfig(
        face_track=False,
        burn_captions=False,
        caption_style="bold_yellow",
        checkpoint_manifest_path=str(manifest_path),
        approved_candidates=base_candidates,
    )
    generate_shorts(str(media), config=changed, output_dir=str(output_dir))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert trim_calls == [(10.0, 25.0, "trim_1.mp4")]
    assert manifest["resume"]["accepted"] is False
    assert manifest["resume"]["mismatch"] == "config"
    assert manifest["status"] == "complete"


def test_shorts_pipeline_writes_grouped_magic_bundle_manifest(tmp_path, monkeypatch):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")
    output_dir = tmp_path / "out"

    def fake_trim(_input_path, _start, _end, output_path):
        Path(output_path).write_bytes(b"clip")

    def fake_conform(input_path, _width, _height, output_path):
        Path(output_path).write_bytes(Path(input_path).read_bytes())

    monkeypatch.setattr("opencut.core.shorts_pipeline._trim_clip", fake_trim)
    monkeypatch.setattr("opencut.core.shorts_pipeline._conform_clip_dimensions", fake_conform)
    monkeypatch.setattr("opencut.core.shorts_pipeline._probe_duration", lambda _path: 0.0)

    clips = generate_shorts(
        str(media),
        config=ShortsPipelineConfig(
            max_shorts=1,
            face_track=False,
            burn_captions=False,
            approved_plan_id="mcplan:bundle",
            approved_candidates=[
                {
                    "candidate_id": "mcc:first",
                    "start": 5.0,
                    "end": 25.0,
                    "title": "First hook",
                    "score": 88.0,
                    "transcript_excerpt": "This is the hook that explains the result.",
                    "score_breakdown": {"hook": 0.9, "duration": 0.8},
                    "hashtags": ["#editing"],
                },
            ],
            platform_presets=["youtube_shorts", "instagram_feed"],
        ),
        output_dir=str(output_dir),
    )

    bundle_path = output_dir / "magic_clips_manifest.json"
    csv_path = output_dir / "magic_clips_manifest.csv"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert [clip.bundle_manifest_path for clip in clips] == [str(bundle_path), str(bundle_path)]
    assert [clip.bundle_csv_path for clip in clips] == [str(csv_path), str(csv_path)]
    assert bundle["schema_version"] == "opencut.magic_clips.bundle.v1"
    assert bundle["candidate_count"] == 1
    assert bundle["output_count"] == 2
    assert bundle["plan_id"] == "mcplan:bundle"
    assert bundle["candidates"][0]["candidate_id"] == "mcc:first"
    assert bundle["candidates"][0]["score_breakdown"] == {"duration": 0.8, "hook": 0.9}
    assert bundle["candidates"][0]["transcript_excerpt"] == "This is the hook that explains the result."
    assert {output["platform_preset"] for output in bundle["candidates"][0]["outputs"]} == {
        "youtube_shorts",
        "instagram_feed",
    }
    assert all(Path(output["export_path"]).is_file() for output in bundle["candidates"][0]["outputs"])
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "candidate_id,platform_preset,start,end,duration,title,score,export_path" in csv_text
    assert csv_text.count("mcc:first") == 2


def test_shorts_pipeline_rejects_unknown_platform_preset(tmp_path):
    from opencut.core.shorts_pipeline import ShortsPipelineConfig, generate_shorts

    media = tmp_path / "source.mp4"
    media.write_bytes(b"fake-media")

    try:
        generate_shorts(
            str(media),
            config=ShortsPipelineConfig(
                face_track=False,
                burn_captions=False,
                approved_candidates=[
                    {"candidate_id": "mcc:first", "start": 0.0, "end": 30.0, "title": "Hook"},
                ],
                platform_presets=["unknown"],
            ),
            output_dir=str(tmp_path / "out"),
        )
    except ValueError as exc:
        assert "Unknown platform preset" in str(exc)
    else:
        raise AssertionError("expected unknown platform preset to fail")


def test_shorts_pipeline_route_filters_magic_plan_candidate_ids(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    media = tmp_path / "route.mp4"
    media.write_bytes(b"fake")
    captured = {}

    def fake_generate(_filepath, config, output_dir, on_progress):
        captured["approved_plan_id"] = config.approved_plan_id
        captured["approved_candidates"] = config.approved_candidates
        captured["platform_presets"] = config.platform_presets
        captured["checkpoint_manifest_path"] = config.checkpoint_manifest_path
        captured["resume"] = config.resume
        captured["output_dir"] = output_dir
        on_progress(100, "done")
        return []

    plan = {
        "plan_id": "mcplan:abc",
        "candidates": [
            {
                "candidate_id": "mcc:keep",
                "start": 5.0,
                "end": 20.0,
                "title": "Keep",
                "platform_presets": [
                    {"preset_id": "youtube_shorts"},
                    {"preset_id": "instagram_feed"},
                ],
            },
            {"candidate_id": "mcc:skip", "start": 25.0, "end": 40.0, "title": "Skip"},
        ],
    }

    with patch("opencut.routes.video_specialty.rate_limit", return_value=True), \
            patch("opencut.routes.video_specialty.rate_limit_release"), \
            patch("opencut.core.shorts_pipeline.generate_shorts", side_effect=fake_generate):
        resp = client.post(
            "/video/shorts-pipeline",
            data=json.dumps({
                "filepath": str(media),
                "output_dir": str(tmp_path / "out"),
                "magic_clips_plan": plan,
                "candidate_ids": ["mcc:keep"],
                "face_track": "false",
                "burn_captions": "false",
            }),
            headers=csrf_headers(csrf_token),
        )
        job = _poll_job(client, resp.get_json()["job_id"])

    assert resp.status_code == 200
    assert job["status"] == "complete"
    assert captured["approved_plan_id"] == "mcplan:abc"
    assert captured["approved_candidates"] == [plan["candidates"][0]]
    assert captured["platform_presets"] == ["youtube_shorts", "instagram_feed"]
    assert captured["checkpoint_manifest_path"].endswith("magic_clips_run_manifest.json")
    assert captured["resume"] is True
    assert job["result"]["magic_clips_manifest"] == captured["checkpoint_manifest_path"]


def test_shorts_pipeline_route_returns_magic_bundle_manifest(client, csrf_token, tmp_path):
    from types import SimpleNamespace

    from tests.conftest import csrf_headers

    media = tmp_path / "route.mp4"
    media.write_bytes(b"fake")
    bundle_path = tmp_path / "out" / "magic_clips_manifest.json"
    csv_path = tmp_path / "out" / "magic_clips_manifest.csv"
    export_path = tmp_path / "out" / "out.mp4"
    bundle_payload = {
        "schema_version": "opencut.magic_clips.bundle.v1",
        "output_dir": str(tmp_path / "out"),
        "candidate_count": 1,
        "output_count": 1,
        "candidates": [
            {
                "candidate_id": "mcc:keep",
                "title": "Keep",
                "social_metadata": {"suggested_caption": "Ship the useful part", "hashtags": ["#editing"]},
                "outputs": [{"platform_preset": "youtube_shorts", "export_path": str(export_path)}],
            },
        ],
    }

    def fake_generate(_filepath, config, output_dir, on_progress):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_bytes(b"clip")
        bundle_path.write_text(json.dumps(bundle_payload), encoding="utf-8")
        csv_path.write_text("candidate_id,platform_preset\nmcc:keep,youtube_shorts\n", encoding="utf-8")
        on_progress(100, "done")
        return [
            SimpleNamespace(
                index=1,
                output_path=str(export_path),
                start=5.0,
                end=20.0,
                duration=15.0,
                title="Keep",
                score=90.0,
                engagement=None,
                platform_preset="youtube_shorts",
                width=1080,
                height=1920,
                manifest_path=config.checkpoint_manifest_path,
                bundle_manifest_path=str(bundle_path),
                bundle_csv_path=str(csv_path),
            ),
        ]

    plan = {
        "plan_id": "mcplan:abc",
        "candidates": [
            {"candidate_id": "mcc:keep", "start": 5.0, "end": 20.0, "title": "Keep"},
        ],
    }

    with patch("opencut.routes.video_specialty.rate_limit", return_value=True), \
            patch("opencut.routes.video_specialty.rate_limit_release"), \
            patch("opencut.core.shorts_pipeline.generate_shorts", side_effect=fake_generate):
        resp = client.post(
            "/video/shorts-pipeline",
            data=json.dumps({
                "filepath": str(media),
                "magic_clips_plan": plan,
                "candidate_ids": ["mcc:keep"],
                "face_track": "false",
                "burn_captions": "false",
            }),
            headers=csrf_headers(csrf_token),
        )
        job = _poll_job(client, resp.get_json()["job_id"])

    result = job["result"]
    assert resp.status_code == 200
    assert result["magic_clips_bundle_manifest"] == str(bundle_path)
    assert result["magic_clips_bundle_csv"] == str(csv_path)
    assert result["magic_clips_bundle"] == bundle_payload
    assert result["clips"][0]["bundle_manifest_path"] == str(bundle_path)
    handoff = result["magic_clips_downstream_handoff"]
    assert handoff["schema_version"] == "opencut.magic_clips.downstream.v1"
    assert handoff["timeline_import_count"] == 1
    assert handoff["social_upload_count"] == 1
    assert handoff["timeline_imports"][0]["import_path"] == str(export_path)
    assert handoff["social_uploads"][0]["upload_payload"]["platform"] == "youtube"


def test_shorts_pipeline_route_rejects_unmatched_magic_candidate_ids(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    media = tmp_path / "route.mp4"
    media.write_bytes(b"fake")
    plan = {
        "plan_id": "mcplan:abc",
        "candidates": [
            {"candidate_id": "mcc:keep", "start": 5.0, "end": 20.0, "title": "Keep"},
        ],
    }

    with patch("opencut.routes.video_specialty.rate_limit", return_value=True), \
            patch("opencut.routes.video_specialty.rate_limit_release"), \
            patch("opencut.core.shorts_pipeline.generate_shorts"):
        resp = client.post(
            "/video/shorts-pipeline",
            data=json.dumps({
                "filepath": str(media),
                "magic_clips_plan": plan,
                "candidate_ids": ["mcc:missing"],
            }),
            headers=csrf_headers(csrf_token),
        )
        job = _poll_job(client, resp.get_json()["job_id"])

    assert resp.status_code == 200
    assert job["status"] == "error"
    assert "approved Magic Clips handoff requires candidate windows" in job["error"]


def test_magic_clips_bundle_builds_social_upload_plan(tmp_path):
    from opencut.core.social_post import build_magic_clips_social_upload_plan

    export_path = tmp_path / "first_youtube.mp4"
    export_path.write_bytes(b"clip")
    bundle_path = tmp_path / "magic_clips_manifest.json"
    bundle_path.write_text(json.dumps({
        "schema_version": "opencut.magic_clips.bundle.v1",
        "plan_id": "mcplan:social",
        "candidates": [
            {
                "candidate_id": "mcc:first",
                "title": "First hook",
                "transcript_excerpt": "This hook explains the result.",
                "social_metadata": {
                    "suggested_caption": "Fast edit breakdown",
                    "hashtags": ["#editing", "#premiere"],
                },
                "outputs": [
                    {
                        "platform_preset": "youtube_shorts",
                        "export_path": str(export_path),
                        "caption_path": str(tmp_path / "first.srt"),
                        "thumbnail_path": str(tmp_path / "thumb.jpg"),
                        "start": 5.0,
                        "end": 20.0,
                        "duration": 15.0,
                    },
                    {
                        "platform_preset": "instagram_feed",
                        "export_path": str(tmp_path / "first_instagram.mp4"),
                    },
                ],
            },
        ],
    }), encoding="utf-8")

    plan = build_magic_clips_social_upload_plan(
        str(bundle_path),
        platform="youtube",
        candidate_ids=["mcc:first"],
        privacy="unlisted",
    )

    assert plan["schema_version"] == "opencut.magic_clips.social_upload_plan.v1"
    assert plan["source_manifest"] == str(bundle_path)
    assert plan["plan_id"] == "mcplan:social"
    assert plan["upload_count"] == 1
    assert plan["platforms"] == ["youtube"]
    upload = plan["uploads"][0]
    assert upload["candidate_id"] == "mcc:first"
    assert upload["platform"] == "youtube"
    assert upload["platform_preset"] == "youtube_shorts"
    assert upload["filepath"] == str(export_path)
    assert upload["title"] == "First hook"
    assert upload["description"] == "Fast edit breakdown"
    assert upload["tags"] == ["#editing", "#premiere"]
    assert upload["privacy"] == "unlisted"
    assert upload["ready"] is True
    assert upload["warnings"] == []


def test_magic_clips_downstream_handoff_skips_outputs_outside_bundle_root(tmp_path):
    from opencut.core.shorts_pipeline import build_magic_clips_downstream_handoff

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"secret")
    bundle_path = output_dir / "magic_clips_manifest.json"
    bundle_path.write_text(json.dumps({
        "schema_version": "opencut.magic_clips.bundle.v1",
        "output_dir": str(output_dir),
        "candidates": [
            {
                "candidate_id": "mcc:first",
                "title": "First hook",
                "outputs": [{"platform_preset": "youtube_shorts", "export_path": str(outside)}],
            },
        ],
    }), encoding="utf-8")

    handoff = build_magic_clips_downstream_handoff(str(bundle_path))

    assert handoff["timeline_imports"] == []
    assert handoff["social_uploads"] == []
    assert handoff["warnings"][0]["type"] == "output_outside_bundle_root"


def test_social_upload_dry_run_consumes_magic_clips_bundle_manifest(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    youtube_path = tmp_path / "first_youtube.mp4"
    instagram_path = tmp_path / "first_instagram.mp4"
    youtube_path.write_bytes(b"yt")
    instagram_path.write_bytes(b"ig")
    bundle_path = tmp_path / "magic_clips_manifest.json"
    bundle_path.write_text(json.dumps({
        "schema_version": "opencut.magic_clips.bundle.v1",
        "plan_id": "mcplan:social-route",
        "candidates": [
            {
                "candidate_id": "mcc:first",
                "title": "First hook",
                "social_metadata": {"suggested_caption": "Ready for the feed"},
                "outputs": [
                    {"platform_preset": "youtube_shorts", "export_path": str(youtube_path)},
                    {"platform_preset": "instagram_feed", "export_path": str(instagram_path)},
                ],
            },
        ],
    }), encoding="utf-8")

    resp = client.post(
        "/social/upload",
        data=json.dumps({
            "magic_clips_bundle_manifest": str(bundle_path),
            "dry_run": True,
            "platform": "instagram",
            "candidate_ids": ["mcc:first"],
            "privacy": "public",
        }),
        headers=csrf_headers(csrf_token),
    )
    job = _poll_job(client, resp.get_json()["job_id"])

    assert resp.status_code == 200
    assert job["status"] == "complete"
    result = job["result"]
    assert result["dry_run"] is True
    assert result["source_manifest"] == str(bundle_path)
    assert result["upload_count"] == 1
    assert result["platforms"] == ["instagram"]
    assert result["uploads"][0]["filepath"] == str(instagram_path)
    assert result["uploads"][0]["platform"] == "instagram"
    assert result["uploads"][0]["privacy"] == "public"
    assert result["uploads"][0]["ready"] is True


def test_social_upload_rejects_magic_bundle_without_dry_run(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    bundle_path = tmp_path / "magic_clips_manifest.json"
    bundle_path.write_text(json.dumps({
        "schema_version": "opencut.magic_clips.bundle.v1",
        "candidates": [],
    }), encoding="utf-8")

    resp = client.post(
        "/social/upload",
        data=json.dumps({"magic_clips_bundle_manifest": str(bundle_path)}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    assert "dry_run=true" in resp.get_json()["error"]


def test_timeline_magic_clips_import_plan_reads_bundle_manifest(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    export_path = output_dir / "first_youtube.mp4"
    export_path.write_bytes(b"yt")
    bundle_path = output_dir / "magic_clips_manifest.json"
    bundle_path.write_text(json.dumps({
        "schema_version": "opencut.magic_clips.bundle.v1",
        "plan_id": "mcplan:timeline-route",
        "output_dir": str(output_dir),
        "candidates": [
            {
                "candidate_id": "mcc:first",
                "title": "First hook",
                "outputs": [
                    {"platform_preset": "youtube_shorts", "export_path": str(export_path), "start": 1.0, "end": 9.0},
                ],
            },
        ],
    }), encoding="utf-8")

    resp = client.post(
        "/timeline/magic-clips-import-plan",
        data=json.dumps({"bundle_manifest_path": str(bundle_path)}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["schema_version"] == "opencut.magic_clips.downstream.v1"
    assert payload["timeline_import_count"] == 1
    assert payload["timeline_imports"][0]["import_path"] == str(export_path)
    assert payload["timeline_imports"][0]["source"] == "magic_clips_bundle"
