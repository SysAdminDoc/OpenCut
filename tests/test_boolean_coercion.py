import json
import math
import time
from unittest.mock import patch

import pytest

from opencut.security import safe_bool
from tests.conftest import csrf_headers


# ---------------------------------------------------------------------------
# Unit tests for safe_bool() edge cases (v1.9.22 hardening)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (True, True),
    (False, False),
    (None, False),
    (1, True),
    (0, False),
    (-1, True),
    (1.0, True),
    (0.0, False),
    ("true", True),
    ("True", True),
    ("false", False),
    ("FALSE", False),
    ("yes", True),
    ("no", False),
    ("on", True),
    ("off", False),
    ("1", True),
    ("0", False),
    ("", False),
    ("null", False),
    ("none", False),
    (b"true", True),
    (b"false", False),
    (bytearray(b"yes"), True),
    (bytearray(b"no"), False),
])
def test_safe_bool_accepts_common_forms(value, expected):
    assert safe_bool(value, default=False) is expected


@pytest.mark.parametrize("value", [
    float("nan"),
    float("inf"),
    float("-inf"),
    [True],
    [],
    {"key": "value"},
    {},
    (1, 2),
    set(),
    object(),
])
def test_safe_bool_rejects_unsafe_inputs(value):
    # Ambiguous / unsafe inputs must fall back to the caller-supplied default
    assert safe_bool(value, default=False) is False
    assert safe_bool(value, default=True) is True


def test_safe_bool_unknown_string_uses_default():
    assert safe_bool("maybe", default=True) is True
    assert safe_bool("maybe", default=False) is False


def test_safe_bool_nan_float_uses_default():
    assert safe_bool(math.nan, default=True) is True
    assert safe_bool(math.nan, default=False) is False


def poll_job(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}")
        data = resp.get_json()
        if data["status"] in ("complete", "error", "cancelled"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def test_whisper_settings_string_false_stays_false(client, csrf_token):
    resp = client.post(
        "/whisper/settings",
        data=json.dumps({"cpu_mode": "false", "model": "base"}),
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["settings"]["cpu_mode"] is False


def test_audio_enhance_string_false_flags_fail_validation(client, csrf_token, tmp_path):
    media = tmp_path / "dialog.wav"
    media.write_bytes(b"RIFF-test")

    with patch("opencut.checks.check_resemble_enhance_available", return_value=True):
        resp = client.post(
            "/audio/enhance",
            data=json.dumps({
                "filepath": str(media),
                "backend": "resemble",
                "denoise": "false",
                "enhance": "false",
            }),
            headers=csrf_headers(csrf_token),
        )
        job = poll_job(client, resp.get_json()["job_id"])

    assert resp.status_code == 200
    assert job["status"] == "error"
    assert "At least one of 'denoise' or 'enhance' must be True" in job["error"]


def test_video_auto_zoom_string_false_does_not_apply_file(client, csrf_token, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake-mp4")

    keyframes = [{"time": 0.0, "zoom": 1.15}]
    with patch("opencut.core.auto_zoom.generate_zoom_keyframes", return_value=keyframes), \
            patch("opencut.routes.video_editing._sp.run") as mock_run:
        resp = client.post(
            "/video/auto-zoom",
            data=json.dumps({
                "filepath": str(media),
                "apply_to_file": "false",
            }),
            headers=csrf_headers(csrf_token),
        )
        job = poll_job(client, resp.get_json()["job_id"])

    assert resp.status_code == 200
    assert job["status"] == "complete"
    assert job["result"]["keyframes"] == keyframes
    assert "output" not in job["result"]
    assert not any(
        call.args and any(str(part) == "-vf" or "zoompan" in str(part) for part in call.args[0])
        for call in mock_run.call_args_list
    )


def test_video_shorts_pipeline_string_false_flags_stay_false(client, csrf_token, tmp_path):
    media = tmp_path / "interview.mp4"
    media.write_bytes(b"fake-mp4")

    captured = {}

    def fake_generate(_filepath, config, output_dir, on_progress):
        captured["face_track"] = config.face_track
        captured["burn_captions"] = config.burn_captions
        captured["output_dir"] = output_dir
        on_progress(100, "done")
        return []

    with patch("opencut.routes.video_specialty.rate_limit", return_value=True), \
            patch("opencut.routes.video_specialty.rate_limit_release"), \
            patch("opencut.core.shorts_pipeline.generate_shorts", side_effect=fake_generate):
        resp = client.post(
            "/video/shorts-pipeline",
            data=json.dumps({
                "filepath": str(media),
                "face_track": "false",
                "burn_captions": "false",
            }),
            headers=csrf_headers(csrf_token),
        )
        job = poll_job(client, resp.get_json()["job_id"])

    assert resp.status_code == 200
    assert job["status"] == "complete"
    assert captured["face_track"] is False
    assert captured["burn_captions"] is False
