"""
Unit tests for OpenCut processing features:
  - Audio analysis (loudness, spectrum, platform compliance)
  - Deinterlacing (detection + processing)
  - Lens distortion correction (presets, correction, horizon levelling)

Tests pure logic with mocked FFmpeg subprocess calls.
"""

import math
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import csrf_headers


# ========================================================================
# Audio Analysis — Loudness
# ========================================================================
class TestMeasureLoudness:
    """Tests for opencut.core.audio_analysis.measure_loudness"""

    def test_parses_ebur128_summary(self):
        """measure_loudness should parse integrated, peak, LRA from ebur128 output."""
        ebur128_stderr = (
            "[Parsed_ebur128_0 @ 0xabc]   M: -14.2  S: -15.1\n"
            "[Parsed_ebur128_0 @ 0xabc]   M: -12.8  S: -13.5\n"
            "[Parsed_ebur128_0 @ 0xabc]   M: -16.0  S: -17.2\n"
            "[Parsed_ebur128_0 @ 0xabc] Summary:\n"
            "  Integrated loudness:\n"
            "    I:         -14.0 LUFS\n"
            "    Threshold: -24.5 LUFS\n"
            "  Loudness range:\n"
            "    LRA:         7.2 LU\n"
            "  True peak:\n"
            "    True peak:  -1.5 dBFS\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ebur128_stderr

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import measure_loudness
            result = measure_loudness("test.mp4")

        assert result.integrated_lufs == -14.0
        assert result.true_peak_dbtp == -1.5
        assert result.lra == 7.2
        assert result.momentary_max == -12.8
        assert result.short_term_max == -13.5

    def test_handles_empty_stderr(self):
        """Should return defaults when ebur128 produces no output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "some irrelevant output\n"

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import measure_loudness
            result = measure_loudness("test.mp4")

        assert result.integrated_lufs == -70.0
        assert result.true_peak_dbtp == -70.0
        assert result.lra == 0.0

    def test_progress_callback(self):
        """Progress callback should be called during processing."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "  I: -14.0 LUFS\n"
        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append((pct, msg))

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import measure_loudness
            measure_loudness("test.mp4", on_progress=on_progress)

        assert len(progress_calls) >= 2
        assert progress_calls[-1][0] == 100


# ========================================================================
# Audio Analysis — Spectrum
# ========================================================================
class TestAnalyzeSpectrum:
    """Tests for opencut.core.audio_analysis.analyze_spectrum"""

    def test_parses_astats_rms(self):
        """analyze_spectrum should parse RMS levels from astats output."""
        rms_output = (
            "lavfi.astats.Overall.RMS_level=-25.3\n"
            "lavfi.astats.Overall.RMS_level=-24.1\n"
            "lavfi.astats.Overall.RMS_level=-26.0\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = rms_output
        mock_result.stderr = ""

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import analyze_spectrum
            result = analyze_spectrum("test.mp4")

        # All bands should have parsed values (all use the same mock)
        assert result.sub_bass == -25.1  # avg of -25.3, -24.1, -26.0
        assert result.bass == -25.1
        assert result.mid == -25.1

    def test_handles_timeout(self):
        """Should default to -70 on timeout."""
        with patch("opencut.core.audio_analysis.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)):
            from opencut.core.audio_analysis import analyze_spectrum
            result = analyze_spectrum("test.mp4")

        assert result.sub_bass == -70.0
        assert result.bass == -70.0
        assert result.brilliance == -70.0

    def test_handles_no_rms_data(self):
        """Should return -70 when no RMS data is found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "no useful data here\n"
        mock_result.stderr = ""

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import analyze_spectrum
            result = analyze_spectrum("test.mp4")

        assert result.sub_bass == -70.0


# ========================================================================
# Audio Analysis — Platform Loudness Check
# ========================================================================
class TestCheckPlatformLoudness:
    """Tests for opencut.core.audio_analysis.check_platform_loudness"""

    def test_youtube_pass(self):
        """File at -14.0 LUFS should pass YouTube check."""
        ebur128_stderr = "  I: -14.0 LUFS\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ebur128_stderr

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import check_platform_loudness
            result = check_platform_loudness("test.mp4", platform="youtube")

        assert result.platform == "youtube"
        assert result.target_lufs == -14.0
        assert result.actual_lufs == -14.0
        assert result.passes is True
        assert result.adjustment_needed_db == 0.0

    def test_youtube_fail_too_loud(self):
        """File at -8.0 LUFS should fail YouTube check."""
        ebur128_stderr = "  I: -8.0 LUFS\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ebur128_stderr

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import check_platform_loudness
            result = check_platform_loudness("test.mp4", platform="youtube")

        assert result.passes is False
        assert result.adjustment_needed_db == -6.0  # needs to come down 6 dB

    def test_broadcast_target(self):
        """Broadcast target should be -24 LUFS."""
        ebur128_stderr = "  I: -24.0 LUFS\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ebur128_stderr

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import check_platform_loudness
            result = check_platform_loudness("test.mp4", platform="broadcast")

        assert result.target_lufs == -24.0
        assert result.passes is True

    def test_unknown_platform_defaults_to_youtube(self):
        """Unknown platform should default to -14 LUFS target."""
        ebur128_stderr = "  I: -14.0 LUFS\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ebur128_stderr

        with patch("opencut.core.audio_analysis.subprocess.run", return_value=mock_result):
            from opencut.core.audio_analysis import check_platform_loudness
            result = check_platform_loudness("test.mp4", platform="unknown_platform")

        assert result.target_lufs == -14.0

    def test_all_platforms_have_targets(self):
        """All defined platforms should have a target."""
        from opencut.core.audio_analysis import PLATFORM_TARGETS
        for platform in ("youtube", "spotify", "apple_podcasts", "broadcast", "tiktok"):
            assert platform in PLATFORM_TARGETS
            assert isinstance(PLATFORM_TARGETS[platform], float)


# ========================================================================
# Deinterlace — Detection
# ========================================================================
class TestDetectInterlaced:
    """Tests for opencut.core.deinterlace.detect_interlaced"""

    def test_detects_tff_interlaced(self):
        """Should detect TFF interlaced content."""
        idet_stderr = (
            "[Parsed_idet_0 @ 0x123] Repeated Fields: Neither: 500 Top:    0 Bottom:    0\n"
            "[Parsed_idet_0 @ 0x123] Single frame detection: TFF:  400 BFF:   10 Progressive:   90 Undetermined:    0\n"
            "[Parsed_idet_0 @ 0x123] Multi frame detection:  TFF:  420 BFF:    5 Progressive:   75 Undetermined:    0\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = idet_stderr

        with patch("opencut.core.deinterlace.subprocess.run", return_value=mock_result):
            from opencut.core.deinterlace import detect_interlaced
            info = detect_interlaced("test.mp4")

        assert info.is_interlaced is True
        assert info.field_order == "tff"
        assert info.detection_confidence > 0.5

    def test_detects_progressive(self):
        """Should detect progressive (non-interlaced) content."""
        idet_stderr = (
            "[Parsed_idet_0 @ 0x123] Multi frame detection:  TFF:    5 BFF:    3 Progressive:  490 Undetermined:    2\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = idet_stderr

        with patch("opencut.core.deinterlace.subprocess.run", return_value=mock_result):
            from opencut.core.deinterlace import detect_interlaced
            info = detect_interlaced("test.mp4")

        assert info.is_interlaced is False
        assert info.field_order == "unknown"

    def test_detects_bff(self):
        """Should detect BFF interlaced content."""
        idet_stderr = (
            "[Parsed_idet_0 @ 0x123] Multi frame detection:  TFF:   10 BFF:  400 Progressive:   90 Undetermined:    0\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = idet_stderr

        with patch("opencut.core.deinterlace.subprocess.run", return_value=mock_result):
            from opencut.core.deinterlace import detect_interlaced
            info = detect_interlaced("test.mp4")

        assert info.is_interlaced is True
        assert info.field_order == "bff"

    def test_handles_timeout(self):
        """Should return defaults on timeout."""
        with patch("opencut.core.deinterlace.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)):
            from opencut.core.deinterlace import detect_interlaced
            info = detect_interlaced("test.mp4")

        assert info.is_interlaced is False
        assert info.field_order == "unknown"
        assert info.detection_confidence == 0.0

    def test_handles_no_idet_output(self):
        """Should return defaults when idet produces no parseable output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "some other ffmpeg output\n"

        with patch("opencut.core.deinterlace.subprocess.run", return_value=mock_result):
            from opencut.core.deinterlace import detect_interlaced
            info = detect_interlaced("test.mp4")

        assert info.is_interlaced is False
        assert info.detection_confidence == 0.0

    def test_falls_back_to_single_frame(self):
        """Should use single-frame detection if multi-frame is absent."""
        idet_stderr = (
            "[Parsed_idet_0 @ 0x123] Single frame detection: TFF:  400 BFF:   10 Progressive:   90 Undetermined:    0\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = idet_stderr

        with patch("opencut.core.deinterlace.subprocess.run", return_value=mock_result):
            from opencut.core.deinterlace import detect_interlaced
            info = detect_interlaced("test.mp4")

        assert info.is_interlaced is True
        assert info.field_order == "tff"


# ========================================================================
# Deinterlace — Processing
# ========================================================================
class TestDeinterlace:
    """Tests for opencut.core.deinterlace.deinterlace"""

    def test_bwdif_default(self):
        """Default method should be bwdif with auto field order."""
        idet_stderr = (
            "[Parsed_idet_0 @ 0x123] Multi frame detection:  TFF:  400 BFF:    5 Progressive:   95 Undetermined:    0\n"
        )
        mock_idet = MagicMock(returncode=0, stderr=idet_stderr)

        with patch("opencut.core.deinterlace.subprocess.run", return_value=mock_idet), \
             patch("opencut.core.deinterlace.run_ffmpeg") as mock_run:
            from opencut.core.deinterlace import deinterlace
            result = deinterlace("test.mp4", output_path_override="/tmp/out.mp4")

        assert result["output_path"] == "/tmp/out.mp4"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "bwdif" in cmd_str

    def test_yadif_method(self):
        """yadif method should build correct filter."""
        with patch("opencut.core.deinterlace.subprocess.run") as mock_sub, \
             patch("opencut.core.deinterlace.run_ffmpeg") as mock_run:
            # Mock idet detection for auto field order
            mock_sub.return_value = MagicMock(
                returncode=0,
                stderr="[Parsed_idet_0 @ 0x1] Multi frame detection:  TFF:  400 BFF:    5 Progressive:   95 Undetermined:    0\n"
            )
            from opencut.core.deinterlace import deinterlace
            deinterlace(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                method="yadif",
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "yadif" in cmd_str

    def test_explicit_field_order_skips_detection(self):
        """Explicit field_order should skip idet auto-detection."""
        with patch("opencut.core.deinterlace.run_ffmpeg") as mock_run:
            from opencut.core.deinterlace import deinterlace
            deinterlace(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                field_order="bff",
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "parity=1" in cmd_str  # bff = parity 1

    def test_invalid_method_defaults_to_bwdif(self):
        """Invalid method should fall back to bwdif."""
        with patch("opencut.core.deinterlace.run_ffmpeg") as mock_run:
            from opencut.core.deinterlace import deinterlace
            deinterlace(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                method="invalid_method",
                field_order="tff",
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "bwdif" in cmd_str

    def test_nnedi_fallback_on_missing_weights(self):
        """nnedi should fall back to bwdif when weights file is missing."""
        with patch("opencut.core.deinterlace.run_ffmpeg") as mock_run:
            # First call (nnedi) raises error about missing weights
            # Second call (bwdif fallback) succeeds
            mock_run.side_effect = [
                RuntimeError("FFmpeg error: nnedi3_weights.bin not found"),
                "",  # success
            ]
            from opencut.core.deinterlace import deinterlace
            deinterlace(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                method="nnedi",
                field_order="tff",
            )

        assert mock_run.call_count == 2
        second_cmd = mock_run.call_args_list[1][0][0]
        cmd_str = " ".join(second_cmd)
        assert "bwdif" in cmd_str


# ========================================================================
# Lens Correction — Presets
# ========================================================================
class TestLensPresets:
    """Tests for opencut.core.lens_correction.list_lens_presets"""

    def test_list_presets_returns_all(self):
        """list_lens_presets should return all defined presets."""
        from opencut.core.lens_correction import LENS_PRESETS, list_lens_presets
        presets = list_lens_presets()
        assert len(presets) == len(LENS_PRESETS)
        for p in presets:
            assert "id" in p
            assert "name" in p
            assert "k1" in p
            assert "k2" in p
            assert p["id"] in LENS_PRESETS

    def test_preset_values_are_numeric(self):
        """All preset k1/k2 values should be floats."""
        from opencut.core.lens_correction import LENS_PRESETS
        for name, data in LENS_PRESETS.items():
            assert isinstance(data["k1"], (int, float)), f"{name}: k1 is not numeric"
            assert isinstance(data["k2"], (int, float)), f"{name}: k2 is not numeric"
            assert isinstance(data["name"], str), f"{name}: name is not string"

    def test_expected_presets_exist(self):
        """Standard camera presets should be present."""
        from opencut.core.lens_correction import LENS_PRESETS
        expected = {"gopro_wide", "gopro_linear", "dji_mini", "dji_mavic",
                    "iphone_wide", "fisheye_moderate", "fisheye_strong"}
        assert expected.issubset(set(LENS_PRESETS.keys()))


# ========================================================================
# Lens Correction — Processing
# ========================================================================
class TestCorrectLensDistortion:
    """Tests for opencut.core.lens_correction.correct_lens_distortion"""

    def test_with_preset(self):
        """Should resolve preset coefficients and run ffmpeg."""
        with patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import correct_lens_distortion
            result = correct_lens_distortion(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                preset="gopro_wide",
            )

        assert result["output_path"] == "/tmp/out.mp4"
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "lenscorrection" in cmd_str
        assert "k1=-0.3" in cmd_str
        assert "k2=0.0" in cmd_str

    def test_with_explicit_k1_k2(self):
        """Should use explicit k1/k2 values."""
        with patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import correct_lens_distortion
            correct_lens_distortion(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                k1=-0.25,
                k2=0.05,
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "k1=-0.25" in cmd_str
        assert "k2=0.05" in cmd_str

    def test_preset_overrides_k1_k2(self):
        """Preset should override any explicit k1/k2."""
        with patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import correct_lens_distortion
            correct_lens_distortion(
                "test.mp4",
                output_path_override="/tmp/out.mp4",
                k1=-0.99,
                k2=-0.99,
                preset="iphone_wide",
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "k1=-0.05" in cmd_str  # iphone_wide preset value, not -0.99

    def test_unknown_preset_raises(self):
        """Unknown preset should raise ValueError."""
        from opencut.core.lens_correction import correct_lens_distortion
        with pytest.raises(ValueError, match="Unknown lens preset"):
            correct_lens_distortion("test.mp4", preset="nonexistent_camera")

    def test_missing_k1_and_preset_raises(self):
        """Neither preset nor k1 should raise ValueError."""
        from opencut.core.lens_correction import correct_lens_distortion
        with pytest.raises(ValueError, match="Either preset or k1"):
            correct_lens_distortion("test.mp4")

    def test_k2_defaults_to_zero(self):
        """k2 should default to 0.0 if only k1 is specified."""
        with patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import correct_lens_distortion
            correct_lens_distortion(
                "test.mp4", output_path_override="/tmp/out.mp4", k1=-0.2,
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "k2=0.0" in cmd_str


# ========================================================================
# Lens Correction — Horizon Levelling
# ========================================================================
class TestLevelHorizon:
    """Tests for opencut.core.lens_correction.level_horizon"""

    def test_positive_angle(self):
        """Positive angle should rotate clockwise."""
        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}

        with patch("opencut.core.lens_correction.get_video_info", return_value=mock_info), \
             patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import level_horizon
            result = level_horizon(
                "test.mp4", output_path_override="/tmp/out.mp4", angle=5.0,
            )

        assert result["output_path"] == "/tmp/out.mp4"
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "rotate" in cmd_str
        # Check angle is roughly 5 degrees in radians
        expected_rad = 5.0 * (math.pi / 180.0)
        assert f"{expected_rad}" in cmd_str

    def test_zero_angle(self):
        """Zero angle should still run (no-op rotation)."""
        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}

        with patch("opencut.core.lens_correction.get_video_info", return_value=mock_info), \
             patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import level_horizon
            level_horizon(
                "test.mp4", output_path_override="/tmp/out.mp4", angle=0.0,
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "rotate=0.0" in cmd_str

    def test_negative_angle(self):
        """Negative angle should rotate counter-clockwise."""
        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}

        with patch("opencut.core.lens_correction.get_video_info", return_value=mock_info), \
             patch("opencut.core.lens_correction.run_ffmpeg") as mock_run:
            from opencut.core.lens_correction import level_horizon
            level_horizon(
                "test.mp4", output_path_override="/tmp/out.mp4", angle=-3.0,
            )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "rotate=" in cmd_str
        # Negative radians
        expected_rad = -3.0 * (math.pi / 180.0)
        assert str(expected_rad) in cmd_str


# ========================================================================
# Route Smoke Tests
# ========================================================================
class TestProcessingRoutes:
    """Smoke tests for opencut/routes/processing_routes.py"""

    def test_lens_presets_get(self, client):
        """GET /video/lens-presets should return a list of presets."""
        resp = client.get("/video/lens-presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) >= 7
        for p in data:
            assert "id" in p
            assert "name" in p
            assert "k1" in p

    def test_loudness_missing_filepath(self, client, csrf_token):
        """POST /audio/loudness without filepath should return 400."""
        resp = client.post(
            "/audio/loudness",
            json={},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_spectrum_missing_filepath(self, client, csrf_token):
        """POST /audio/spectrum without filepath should return 400."""
        resp = client.post(
            "/audio/spectrum",
            json={},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_loudness_check_missing_filepath(self, client, csrf_token):
        """POST /audio/loudness-check without filepath should return 400."""
        resp = client.post(
            "/audio/loudness-check",
            json={"platform": "youtube"},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_detect_interlace_missing_filepath(self, client, csrf_token):
        """POST /video/detect-interlace without filepath should return 400."""
        resp = client.post(
            "/video/detect-interlace",
            json={},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_deinterlace_missing_filepath(self, client, csrf_token):
        """POST /video/deinterlace without filepath should return 400."""
        resp = client.post(
            "/video/deinterlace",
            json={},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_lens_correct_missing_filepath(self, client, csrf_token):
        """POST /video/lens-correct without filepath should return 400."""
        resp = client.post(
            "/video/lens-correct",
            json={"preset": "gopro_wide"},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_horizon_level_missing_filepath(self, client, csrf_token):
        """POST /video/horizon-level without filepath should return 400."""
        resp = client.post(
            "/video/horizon-level",
            json={"angle": 5.0},
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 400

    def test_loudness_no_csrf(self, client):
        """POST /audio/loudness without CSRF token should return 403."""
        resp = client.post(
            "/audio/loudness",
            json={"filepath": "/some/file.mp4"},
        )
        assert resp.status_code == 403

    def test_lens_correct_no_csrf(self, client):
        """POST /video/lens-correct without CSRF token should return 403."""
        resp = client.post(
            "/video/lens-correct",
            json={"filepath": "/some/file.mp4", "preset": "gopro_wide"},
        )
        assert resp.status_code == 403
