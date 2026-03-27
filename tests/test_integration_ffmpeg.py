"""
Integration tests for OpenCut FFmpeg-based video processing.

These tests use real FFmpeg to generate test media and exercise the Flask API
end-to-end.  They are intentionally excluded from the fast unit-test suite.

Run with:
    python -m pytest tests/test_integration_ffmpeg.py -v --tb=short

Skip automatically when FFmpeg is not installed.
"""

import json
import shutil
import subprocess
import time

import pytest

from tests.conftest import csrf_headers

# ---------------------------------------------------------------------------
# Module-level markers — every test in this file is tagged "integration"
# and skipped when ffmpeg is absent.
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def poll_job(client, job_id, csrf_token, timeout=30):
    """Poll a job until it completes or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/status/{job_id}", headers={"X-OpenCut-Token": csrf_token})
        data = resp.get_json()
        if data["status"] in ("complete", "error", "cancelled"):
            return data
        time.sleep(0.5)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def _probe_duration(path):
    """Return duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Fixtures — generate real media files with FFmpeg
# ---------------------------------------------------------------------------
@pytest.fixture
def test_audio(tmp_path):
    """Generate a 5-second 440 Hz sine-wave WAV file."""
    out = tmp_path / "test_audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=5",
            "-af", "volume=0.8",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return str(out)


@pytest.fixture
def test_video(tmp_path):
    """Generate a 5-second 320x240 MP4 with audio."""
    out = tmp_path / "test_video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "aac", "-b:a", "64k",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return str(out)


@pytest.fixture
def test_video_pair(tmp_path):
    """Generate two short test videos for merge testing."""
    paths = []
    for i, dur in enumerate([3, 4]):
        out = tmp_path / f"clip_{i}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"testsrc=duration={dur}:size=320x240:rate=24",
                "-f", "lavfi", "-i", f"sine=frequency={440 + i * 100}:duration={dur}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "64k",
                "-shortest",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        paths.append(str(out))
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVideoTrim:
    def test_video_trim(self, client, csrf_token, test_video):
        """POST /video/trim with start/end, poll job, verify output duration."""
        resp = client.post(
            "/video/trim",
            data=json.dumps({
                "filepath": test_video,
                "start": "0:00:01",
                "end": "0:00:03",
                "quality": "low",
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        result = poll_job(client, job_id, csrf_token, timeout=30)
        assert result["status"] == "complete", f"Job failed: {result.get('error')}"
        output_path = result["result"]["output_path"]

        duration = _probe_duration(output_path)
        assert 1.5 <= duration <= 2.5, f"Expected ~2s, got {duration:.2f}s"


class TestVideoMerge:
    def test_video_merge(self, client, csrf_token, test_video_pair):
        """POST /video/merge with two clips, verify merged duration."""
        resp = client.post(
            "/video/merge",
            data=json.dumps({
                "files": test_video_pair,
                "mode": "concat_demux",
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        result = poll_job(client, job_id, csrf_token, timeout=60)
        assert result["status"] == "complete", f"Job failed: {result.get('error')}"
        output_path = result["result"]["output_path"]

        duration = _probe_duration(output_path)
        # clips are 3s + 4s = 7s; allow some tolerance
        assert 5.5 <= duration <= 8.5, f"Expected ~7s, got {duration:.2f}s"


class TestVideoScenes:
    def test_video_scenes(self, client, csrf_token, test_video):
        """POST /video/scenes, verify returns scene list."""
        resp = client.post(
            "/video/scenes",
            data=json.dumps({
                "filepath": test_video,
                "method": "ffmpeg",
                "threshold": 0.3,
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        result = poll_job(client, job_id, csrf_token, timeout=30)
        assert result["status"] == "complete", f"Job failed: {result.get('error')}"
        res = result["result"]
        # Should have at least the scenes key (could be boundaries or scenes)
        assert isinstance(res, dict)


class TestExportVideo:
    def test_export_video(self, client, csrf_token, test_video):
        """POST /export-video with a single full-file segment."""
        resp = client.post(
            "/export-video",
            data=json.dumps({
                "filepath": test_video,
                "segments": [{"start": 0.0, "end": 5.0, "label": "speech"}],
                "video_codec": "libx264",
                "audio_codec": "aac",
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        result = poll_job(client, job_id, csrf_token, timeout=60)
        assert result["status"] == "complete", f"Job failed: {result.get('error')}"
        output_path = result["result"]["output_path"]

        import os
        assert os.path.isfile(output_path), f"Output not found: {output_path}"


class TestPreviewFrame:
    def test_preview_frame(self, client, csrf_token, test_video):
        """POST /video/preview-frame, verify returns base64 image data."""
        resp = client.post(
            "/video/preview-frame",
            data=json.dumps({
                "filepath": test_video,
                "timestamp": "00:00:02",
                "width": 320,
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        result = poll_job(client, job_id, csrf_token, timeout=15)
        assert result["status"] == "complete", f"Job failed: {result.get('error')}"
        res = result["result"]
        # Should contain base64 image data
        assert "image" in res or "frame" in res or "base64" in json.dumps(res).lower(), \
            f"Expected base64 image data in result, got keys: {list(res.keys())}"


class TestVideoFxList:
    def test_video_fx_list(self, client):
        """GET /video/fx/list returns a list of effects."""
        resp = client.get("/video/fx/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, (dict, list))


class TestVideoSpeedChange:
    def test_video_speed_change(self, client, csrf_token, test_video):
        """POST /video/speed/change with speed=2.0, verify output."""
        resp = client.post(
            "/video/speed/change",
            data=json.dumps({
                "filepath": test_video,
                "speed": 2.0,
            }),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        result = poll_job(client, job_id, csrf_token, timeout=30)
        assert result["status"] == "complete", f"Job failed: {result.get('error')}"
        output_path = result["result"]["output_path"]

        duration = _probe_duration(output_path)
        # 5s video at 2x should be ~2.5s
        assert 1.5 <= duration <= 3.5, f"Expected ~2.5s, got {duration:.2f}s"
