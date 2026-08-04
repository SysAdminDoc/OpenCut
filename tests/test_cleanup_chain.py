"""Contracts for the reviewable silence-to-captions cleanup verb."""

def _plan(tmp_path, **kwargs):
    from opencut.core.workflow import build_cleanup_plan

    return build_cleanup_plan(
        filepath=str(tmp_path / "interview.mp4"),
        duration=12.0,
        speech_segments=[
            {"start": 1.0, "end": 4.0},
            {"start": 6.0, "end": 11.0},
        ],
        output_dir=str(tmp_path / "renders"),
        capabilities={"ffmpeg": True, "captions_available": True, "captions_backend": "faster-whisper"},
        **kwargs,
    )


def test_cleanup_plan_is_side_effect_free_and_lists_every_step(tmp_path):
    from opencut.core.workflow import CLEANUP_STEP_IDS, validate_cleanup_plan

    output_dir = tmp_path / "renders"
    plan = _plan(tmp_path, options={"preset": "podcast", "target_lufs": -16.0})

    assert not output_dir.exists()
    assert [step["id"] for step in plan["steps"]] == list(CLEANUP_STEP_IDS)
    assert all(step["state"] == "ready" for step in plan["steps"])
    assert plan["removed_ranges"] == [
        {"start": 0.0, "end": 1.0},
        {"start": 4.0, "end": 6.0},
        {"start": 11.0, "end": 12.0},
    ]
    assert plan["summary"]["removed_seconds"] == 4.0
    assert plan["preview_only"] is True
    assert plan["requires_confirmation"] is True
    assert validate_cleanup_plan(plan, filepath=plan["source"]["filepath"]) == (True, "")


def test_cleanup_plan_degrades_optional_steps_honestly(tmp_path):
    from opencut.core.workflow import build_cleanup_plan

    plan = build_cleanup_plan(
        filepath=str(tmp_path / "interview.wav"),
        duration=5.0,
        speech_segments=[{"start": 0.0, "end": 5.0}],
        capabilities={
            "ffmpeg": False,
            "captions_available": False,
            "captions_backend": "none",
            "captions_reason": "Install a caption backend",
        },
    )

    by_id = {step["id"]: step for step in plan["steps"]}
    assert by_id["silence_trim"]["state"] == "ready"
    assert by_id["denoise"]["state"] == "skipped"
    assert by_id["denoise"]["reason"] == "FFmpeg is unavailable"
    assert by_id["loudness"]["state"] == "skipped"
    assert by_id["captions"]["state"] == "skipped"
    assert by_id["captions"]["reason"] == "Install a caption backend"
    assert plan["summary"]["skipped_steps"] == 3


def test_cleanup_plan_normalizes_invalid_loudness_targets(tmp_path):
    plan = _plan(tmp_path, options={"target_lufs": "not-a-number"})
    assert plan["options"]["target_lufs"] == -16.0

    clamped = _plan(tmp_path, options={"target_lufs": 8})
    assert clamped["options"]["target_lufs"] == 0.0


def test_cleanup_plan_id_covers_preview_contents(tmp_path):
    from opencut.core.workflow import cleanup_plan_id, validate_cleanup_plan

    plan = _plan(tmp_path)
    changed = dict(plan)
    changed["segments_data"] = list(plan["segments_data"]) + [{"start": 11.5, "end": 12.0}]

    assert cleanup_plan_id(plan) == plan["plan_id"]
    assert cleanup_plan_id(changed) != plan["plan_id"]
    assert validate_cleanup_plan(changed, filepath=plan["source"]["filepath"])[0] is False


def test_cleanup_routes_are_registered(app):
    routes = {
        (rule.rule, method)
        for rule in app.url_map.iter_rules()
        for method in rule.methods
        if method not in {"HEAD", "OPTIONS"}
    }

    assert ("/cleanup/preview", "POST") in routes
    assert ("/cleanup/apply", "POST") in routes
