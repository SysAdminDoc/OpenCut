from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch


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
    assert sorted(path.name for path in output_dir.iterdir()) == [
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
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "source_short_1_instagram_feed_Long_hook.mp4",
        "source_short_1_instagram_reels_Long_hook.mp4",
        "source_short_1_tiktok_Long_hook.mp4",
        "source_short_1_youtube_shorts_Long_hook.mp4",
    ]


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
            patch("opencut.core.shorts_pipeline.generate_shorts") as mock_generate:
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
    mock_generate.assert_not_called()
