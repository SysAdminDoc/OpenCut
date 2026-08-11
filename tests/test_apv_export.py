"""F311 — APV (IETF RFC 9924) mezzanine encode via liboapv."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import apv_export  # noqa: E402

APV_UNAVAILABLE = not apv_export.check_apv_available()


@pytest.fixture(autouse=True)
def _reset_cache():
    yield
    apv_export.clear_availability_cache()


def _make_clip(tmp_path, name="src.mp4", seconds=1):
    ff = apv_export.get_ffmpeg_path()
    src = str(tmp_path / name)
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=320x240:rate=25:duration={seconds}",
         "-pix_fmt", "yuv420p", src],
        check=True, timeout=120,
    )
    return src


class TestCapabilityReport:
    def test_info_reports_the_standard_and_presets(self):
        info = apv_export.apv_info()
        assert info["standard"] == "IETF RFC 9924"
        assert info["encoder"] == "liboapv"
        assert {p["name"] for p in info["presets"]} == set(apv_export.APV_PRESETS)
        assert isinstance(info["available"], bool)

    def test_install_hint_only_when_unavailable(self):
        info = apv_export.apv_info()
        if info["available"]:
            assert info["install_hint"] is None
        else:
            assert "liboapv" in info["install_hint"]

    def test_availability_is_cached_then_clearable(self):
        first = apv_export.check_apv_available()
        assert apv_export.check_apv_available() is first
        apv_export.clear_availability_cache()
        assert apv_export.check_apv_available() is first


class TestArgumentValidation:
    def test_missing_input_is_rejected(self):
        with pytest.raises(ValueError, match="not found"):
            apv_export.encode_apv("/no/such/file.mp4")

    def test_unknown_preset_is_rejected(self, tmp_path):
        clip = tmp_path / "x.mp4"
        clip.write_bytes(b"stub")
        with pytest.raises(ValueError, match="preset"):
            apv_export.encode_apv(str(clip), preset="ludicrous")

    def test_unknown_container_is_rejected(self, tmp_path):
        clip = tmp_path / "x.mp4"
        clip.write_bytes(b"stub")
        with pytest.raises(ValueError, match="container"):
            apv_export.encode_apv(str(clip), container=".mkv")

    def test_missing_encoder_raises_a_dependency_error(self, tmp_path, monkeypatch):
        clip = tmp_path / "x.mp4"
        clip.write_bytes(b"stub")
        monkeypatch.setattr(apv_export, "check_apv_available", lambda: False)
        with pytest.raises(RuntimeError, match="liboapv"):
            apv_export.encode_apv(str(clip))


@pytest.mark.skipif(APV_UNAVAILABLE, reason="FFmpeg build lacks the liboapv encoder")
@pytest.mark.skipif(not shutil.which("ffprobe") and not apv_export.get_ffmpeg_path(),
                    reason="FFmpeg/ffprobe unavailable")
class TestRealEncode:
    def _probe_codec(self, path):
        # Not str.replace on the ffmpeg path: it also rewrites the parent
        # directory, yielding .../ffprobe/ffprobe.exe.
        from opencut.helpers import get_ffprobe_path

        probe = get_ffprobe_path()
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        )
        return out.stdout.strip()

    def test_encodes_a_real_apv_stream(self, tmp_path):
        src = _make_clip(tmp_path)
        result = apv_export.encode_apv(src, preset="fast", container=".mp4")
        assert os.path.isfile(result["output"])
        assert result["size_bytes"] > 0
        assert result["preset"] == "fast"
        assert self._probe_codec(result["output"]).startswith("apv")

    def test_raw_container_is_supported(self, tmp_path):
        src = _make_clip(tmp_path, name="raw.mp4")
        result = apv_export.encode_apv(src, preset="fast", container=".apv")
        assert result["output"].endswith(".apv")
        assert os.path.getsize(result["output"]) > 0

    def test_qp_override_is_honoured(self, tmp_path):
        src = _make_clip(tmp_path, name="qp.mp4")
        result = apv_export.encode_apv(src, preset="fast", qp_override=40)
        assert result["qp"] == 40

    def test_progress_is_reported(self, tmp_path):
        src = _make_clip(tmp_path, name="prog.mp4")
        seen = []
        apv_export.encode_apv(src, preset="fast", on_progress=lambda pct, msg="": seen.append(pct))
        assert seen and seen[-1] == 100


class TestRoutes:
    def test_info_route_reports_capability(self, client):
        body = client.get("/video/encode/apv/info").get_json()
        assert body["encoder"] == "liboapv"
        assert "presets" in body

    def test_encode_route_is_queueable(self):
        from opencut.routes.jobs_routes import _ALLOWED_QUEUE_ENDPOINTS

        assert "/video/encode/apv" in _ALLOWED_QUEUE_ENDPOINTS
