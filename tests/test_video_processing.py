"""
Tests for OpenCut video processing features:
- HDR/SDR Tone Mapping (hdr_tools)
- Multi-Camera Audio Sync (audio_sync)
- Corrupted Video Repair (video_repair)
- Rolling Shutter Correction (rolling_shutter)
- Advanced Stabilization Modes (advanced_stabilize)
- Color Space Auto-Detection & Conversion (colorspace)
- All corresponding routes (video_processing_routes)
"""

import inspect
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# HDR Tools Tests
# ============================================================
class TestHDRTools(unittest.TestCase):
    """Tests for opencut.core.hdr_tools module."""

    def test_hdr_info_dataclass_defaults(self):
        from opencut.core.hdr_tools import HDRInfo
        info = HDRInfo()
        self.assertFalse(info.is_hdr)
        self.assertEqual(info.transfer, "sdr")
        self.assertEqual(info.primaries, "bt709")
        self.assertEqual(info.max_cll, 0)
        self.assertEqual(info.max_fall, 0)

    def test_hdr_info_dataclass_custom(self):
        from opencut.core.hdr_tools import HDRInfo
        info = HDRInfo(is_hdr=True, transfer="pq", primaries="bt2020", max_cll=1000, max_fall=400)
        self.assertTrue(info.is_hdr)
        self.assertEqual(info.transfer, "pq")
        self.assertEqual(info.primaries, "bt2020")
        self.assertEqual(info.max_cll, 1000)
        self.assertEqual(info.max_fall, 400)

    def test_detect_hdr_file_not_found(self):
        from opencut.core.hdr_tools import detect_hdr
        with self.assertRaises(FileNotFoundError):
            detect_hdr("/nonexistent/video.mp4")

    def test_detect_hdr_pq_transfer(self):
        """Probe output with smpte2084 transfer should be detected as HDR PQ."""
        from opencut.core.hdr_tools import detect_hdr
        probe_output = json.dumps({
            "streams": [{
                "color_transfer": "smpte2084",
                "color_primaries": "bt2020",
                "side_data_list": [{"max_content": 1000, "max_average": 400}],
            }]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            info = detect_hdr("/fake/video.mp4")
            self.assertTrue(info.is_hdr)
            self.assertEqual(info.transfer, "pq")
            self.assertEqual(info.primaries, "bt2020")
            self.assertEqual(info.max_cll, 1000)
            self.assertEqual(info.max_fall, 400)

    def test_detect_hdr_hlg_transfer(self):
        from opencut.core.hdr_tools import detect_hdr
        probe_output = json.dumps({
            "streams": [{
                "color_transfer": "arib-std-b67",
                "color_primaries": "bt2020",
                "side_data_list": [],
            }]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            info = detect_hdr("/fake/video.mp4")
            self.assertTrue(info.is_hdr)
            self.assertEqual(info.transfer, "hlg")

    def test_detect_hdr_sdr_content(self):
        from opencut.core.hdr_tools import detect_hdr
        probe_output = json.dumps({
            "streams": [{
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "side_data_list": [],
            }]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            info = detect_hdr("/fake/video.mp4")
            self.assertFalse(info.is_hdr)
            self.assertEqual(info.transfer, "sdr")
            self.assertEqual(info.primaries, "bt709")

    def test_valid_tonemap_algorithms(self):
        from opencut.core.hdr_tools import VALID_TONEMAP_ALGORITHMS
        self.assertIn("hable", VALID_TONEMAP_ALGORITHMS)
        self.assertIn("reinhard", VALID_TONEMAP_ALGORITHMS)
        self.assertIn("mobius", VALID_TONEMAP_ALGORITHMS)
        self.assertIn("linear", VALID_TONEMAP_ALGORITHMS)

    def test_tonemap_invalid_algorithm(self):
        from opencut.core.hdr_tools import tonemap_hdr_to_sdr
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                tonemap_hdr_to_sdr("/fake/video.mp4", algorithm="invalid")

    def test_tonemap_file_not_found(self):
        from opencut.core.hdr_tools import tonemap_hdr_to_sdr
        with self.assertRaises(FileNotFoundError):
            tonemap_hdr_to_sdr("/nonexistent/video.mp4")

    def test_tonemap_calls_ffmpeg(self):
        from opencut.core.hdr_tools import tonemap_hdr_to_sdr
        probe_output = json.dumps({
            "streams": [{"color_transfer": "smpte2084", "color_primaries": "bt2020", "side_data_list": []}]
        }).encode()
        mock_probe = MagicMock(returncode=0, stdout=probe_output)
        mock_ffmpeg = MagicMock(returncode=0, stderr=b"")
        progress_calls = []

        def track_progress(pct, msg):
            progress_calls.append((pct, msg))

        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", side_effect=[mock_probe, mock_ffmpeg]), \
             patch("opencut.core.hdr_tools.run_ffmpeg"):
            result = tonemap_hdr_to_sdr("/fake/video.mp4", algorithm="hable", on_progress=track_progress)
            self.assertIn("output_path", result)
            self.assertEqual(result["algorithm"], "hable")
            self.assertTrue(len(progress_calls) > 0)

    def test_detect_and_suggest_hdr(self):
        from opencut.core.hdr_tools import detect_and_suggest
        probe_output = json.dumps({
            "streams": [{"color_transfer": "smpte2084", "color_primaries": "bt2020", "side_data_list": []}]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = detect_and_suggest("/fake/video.mp4")
            self.assertTrue(result["is_hdr"])
            self.assertEqual(result["suggested_action"], "tonemap_to_sdr")

    def test_detect_and_suggest_sdr(self):
        from opencut.core.hdr_tools import detect_and_suggest
        probe_output = json.dumps({
            "streams": [{"color_transfer": "bt709", "color_primaries": "bt709", "side_data_list": []}]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = detect_and_suggest("/fake/video.mp4")
            self.assertFalse(result["is_hdr"])
            self.assertEqual(result["suggested_action"], "none")


# ============================================================
# Audio Sync Tests
# ============================================================
class TestAudioSync(unittest.TestCase):
    """Tests for opencut.core.audio_sync module."""

    def test_sync_result_dataclass(self):
        from opencut.core.audio_sync import SyncResult
        result = SyncResult()
        self.assertEqual(result.offset_seconds, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.method, "cross_correlation")

    def test_sync_result_custom(self):
        from opencut.core.audio_sync import SyncResult
        result = SyncResult(offset_seconds=1.5, confidence=0.85, method="cross_correlation")
        self.assertEqual(result.offset_seconds, 1.5)
        self.assertEqual(result.confidence, 0.85)

    def test_compute_sync_offset_file_not_found(self):
        from opencut.core.audio_sync import compute_sync_offset
        with self.assertRaises(FileNotFoundError):
            compute_sync_offset("/nonexistent/ref.mp4", "/nonexistent/tgt.mp4")

    def test_compute_envelope(self):
        from opencut.core.audio_sync import _compute_envelope
        samples = [100] * 320 + [200] * 320
        env = _compute_envelope(samples, window=160)
        # 640 samples, window=160, range(0, 480, 160) = 3 windows
        self.assertEqual(len(env), 3)
        self.assertEqual(env[0], 100)
        self.assertEqual(env[2], 200)

    def test_cross_correlate_identical(self):
        from opencut.core.audio_sync import _cross_correlate
        env = [0, 100, 200, 100, 0, 50, 150, 250, 150, 50]
        lag, _, confidence = _cross_correlate(env, env, max_lag=5)
        self.assertEqual(lag, 0)
        self.assertGreater(confidence, 0.9)

    def test_cross_correlate_shifted(self):
        from opencut.core.audio_sync import _cross_correlate
        ref = [0, 0, 100, 200, 100, 0, 0, 0, 0, 0]
        tgt = [0, 0, 0, 0, 100, 200, 100, 0, 0, 0]
        lag, _, confidence = _cross_correlate(ref, tgt, max_lag=5)
        # Target is shifted +2 relative to reference
        self.assertEqual(lag, 2)

    def test_read_wav_samples_too_short(self):
        from opencut.core.audio_sync import _read_wav_samples
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"\x00" * 10)
            f.flush()
            path = f.name
        try:
            with self.assertRaises(RuntimeError):
                _read_wav_samples(path)
        finally:
            os.unlink(path)

    def test_read_wav_samples_valid(self):
        from opencut.core.audio_sync import _read_wav_samples
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write 44-byte dummy header + 4 samples
            f.write(b"\x00" * 44)
            for val in [100, -100, 200, -200]:
                f.write(struct.pack("<h", val))
            f.flush()
            path = f.name
        try:
            samples = _read_wav_samples(path)
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0], 100)
            self.assertEqual(samples[1], -100)
        finally:
            os.unlink(path)

    def test_sync_multiple_empty_targets(self):
        from opencut.core.audio_sync import sync_multiple
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                sync_multiple("/fake/ref.mp4", [])

    def test_apply_sync_offset_file_not_found(self):
        from opencut.core.audio_sync import apply_sync_offset
        with self.assertRaises(FileNotFoundError):
            apply_sync_offset("/nonexistent/video.mp4", 1.0)

    def test_apply_sync_zero_offset(self):
        from opencut.core.audio_sync import apply_sync_offset
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.audio_sync.run_ffmpeg"):
            result = apply_sync_offset("/fake/video.mp4", 0.0)
            self.assertEqual(result["applied_offset_seconds"], 0.0)
            self.assertIn("output_path", result)

    def test_apply_sync_positive_offset(self):
        from opencut.core.audio_sync import apply_sync_offset
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.audio_sync.run_ffmpeg") as mock_ffmpeg:
            result = apply_sync_offset("/fake/video.mp4", 2.5)
            self.assertEqual(result["applied_offset_seconds"], 2.5)
            # Should use adelay filter
            call_args = mock_ffmpeg.call_args[0][0]
            self.assertTrue(any("adelay" in str(a) for a in call_args))

    def test_apply_sync_negative_offset(self):
        from opencut.core.audio_sync import apply_sync_offset
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.audio_sync.run_ffmpeg") as mock_ffmpeg:
            result = apply_sync_offset("/fake/video.mp4", -1.0)
            self.assertEqual(result["applied_offset_seconds"], -1.0)
            # Should use atrim filter
            call_args = mock_ffmpeg.call_args[0][0]
            self.assertTrue(any("atrim" in str(a) for a in call_args))

    def test_compute_sync_offset_signature(self):
        from opencut.core.audio_sync import compute_sync_offset
        sig = inspect.signature(compute_sync_offset)
        self.assertIn("reference_path", sig.parameters)
        self.assertIn("target_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)


# ============================================================
# Video Repair Tests
# ============================================================
class TestVideoRepair(unittest.TestCase):
    """Tests for opencut.core.video_repair module."""

    def test_corruption_diagnosis_defaults(self):
        from opencut.core.video_repair import CorruptionDiagnosis
        diag = CorruptionDiagnosis()
        self.assertEqual(diag.corruption_type, "no_corruption")
        self.assertEqual(diag.severity, "none")
        self.assertTrue(diag.recoverable)

    def test_corruption_types_constant(self):
        from opencut.core.video_repair import CORRUPTION_TYPES
        self.assertIn("missing_moov", CORRUPTION_TYPES)
        self.assertIn("broken_container", CORRUPTION_TYPES)
        self.assertIn("partial_file", CORRUPTION_TYPES)
        self.assertIn("bitstream_error", CORRUPTION_TYPES)
        self.assertIn("no_corruption", CORRUPTION_TYPES)

    def test_diagnose_file_not_found(self):
        from opencut.core.video_repair import diagnose_corruption
        with self.assertRaises(FileNotFoundError):
            diagnose_corruption("/nonexistent/video.mp4")

    def test_diagnose_missing_moov(self):
        from opencut.core.video_repair import diagnose_corruption
        mock_result = MagicMock(returncode=1, stderr=b"moov atom not found", stdout=b"{}")
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            diag = diagnose_corruption("/fake/corrupt.mp4")
            self.assertEqual(diag.corruption_type, "missing_moov")
            self.assertEqual(diag.severity, "high")
            self.assertTrue(diag.recoverable)
            self.assertEqual(diag.suggested_action, "remux")

    def test_diagnose_broken_container(self):
        from opencut.core.video_repair import diagnose_corruption
        mock_result = MagicMock(returncode=1, stderr=b"Invalid data found when processing input", stdout=b"{}")
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            diag = diagnose_corruption("/fake/broken.mp4")
            self.assertEqual(diag.corruption_type, "broken_container")
            self.assertFalse(diag.recoverable)

    def test_diagnose_no_corruption(self):
        from opencut.core.video_repair import diagnose_corruption
        probe_output = json.dumps({"format": {"duration": "120.0"}, "streams": [{"codec_type": "video"}]}).encode()
        mock_probe = MagicMock(returncode=0, stderr=b"", stdout=probe_output)
        mock_scan = MagicMock(returncode=0, stderr=b"", stdout=b"")
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", side_effect=[mock_probe, mock_scan]):
            diag = diagnose_corruption("/fake/good.mp4")
            self.assertEqual(diag.corruption_type, "no_corruption")
            self.assertEqual(diag.severity, "none")

    def test_diagnose_bitstream_errors(self):
        from opencut.core.video_repair import diagnose_corruption
        probe_output = json.dumps({"format": {"duration": "120.0"}, "streams": [{"codec_type": "video"}]}).encode()
        mock_probe = MagicMock(returncode=0, stderr=b"", stdout=probe_output)
        mock_scan = MagicMock(returncode=0, stderr=b"error error error corrupt missing error error error", stdout=b"")
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", side_effect=[mock_probe, mock_scan]):
            diag = diagnose_corruption("/fake/bitstream.mp4")
            self.assertEqual(diag.corruption_type, "bitstream_error")
            self.assertIn(diag.severity, ("low", "medium", "high"))
            self.assertEqual(diag.suggested_action, "error_conceal")

    def test_repair_video_file_not_found(self):
        from opencut.core.video_repair import repair_video
        with self.assertRaises(FileNotFoundError):
            repair_video("/nonexistent/video.mp4")

    def test_repair_video_calls_ffmpeg(self):
        from opencut.core.video_repair import repair_video
        probe_output = json.dumps({"format": {"duration": "10"}, "streams": [{"codec_type": "video"}]}).encode()
        mock_probe = MagicMock(returncode=0, stderr=b"", stdout=probe_output)
        mock_scan = MagicMock(returncode=0, stderr=b"", stdout=b"")
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        progress_calls = []

        with patch("os.path.isfile", return_value=True), \
             patch("os.path.getsize", return_value=5_000_000), \
             patch("subprocess.run", side_effect=[mock_probe, mock_scan]), \
             patch("opencut.core.video_repair.run_ffmpeg"), \
             patch("opencut.core.video_repair.get_video_info", return_value=video_info):
            result = repair_video("/fake/video.mp4", on_progress=lambda p, m: progress_calls.append((p, m)))
            self.assertIn("output_path", result)
            self.assertIn("recovered_duration", result)
            self.assertIn("recovery_percentage", result)
            self.assertTrue(len(progress_calls) > 0)


# ============================================================
# Rolling Shutter Tests
# ============================================================
class TestRollingShutter(unittest.TestCase):
    """Tests for opencut.core.rolling_shutter module."""

    def test_correct_file_not_found(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter
        with self.assertRaises(FileNotFoundError):
            correct_rolling_shutter("/nonexistent/video.mp4")

    def test_correct_strength_clamped(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.rolling_shutter.get_video_info", return_value=video_info), \
             patch("opencut.core.rolling_shutter.run_ffmpeg"), \
             patch("os.unlink"):
            # Strength > 1 should be clamped
            result = correct_rolling_shutter("/fake/video.mp4", strength=2.0)
            self.assertEqual(result["strength"], 1.0)

            # Strength < 0 should be clamped
            result = correct_rolling_shutter("/fake/video.mp4", strength=-0.5)
            self.assertEqual(result["strength"], 0.0)

    def test_correct_returns_expected_keys(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.rolling_shutter.get_video_info", return_value=video_info), \
             patch("opencut.core.rolling_shutter.run_ffmpeg"), \
             patch("os.unlink"):
            result = correct_rolling_shutter("/fake/video.mp4", strength=0.5)
            self.assertIn("output_path", result)
            self.assertIn("strength", result)
            self.assertIn("shakiness", result)
            self.assertIn("smoothing", result)

    def test_correct_signature(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter
        sig = inspect.signature(correct_rolling_shutter)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("strength", sig.parameters)
        self.assertIn("output_path_override", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_correct_progress_callback(self):
        from opencut.core.rolling_shutter import correct_rolling_shutter
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        progress_calls = []
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.rolling_shutter.get_video_info", return_value=video_info), \
             patch("opencut.core.rolling_shutter.run_ffmpeg"), \
             patch("os.unlink"):
            correct_rolling_shutter(
                "/fake/video.mp4", strength=0.5,
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )
            self.assertTrue(len(progress_calls) > 0)
            self.assertEqual(progress_calls[-1][0], 100)


# ============================================================
# Advanced Stabilization Tests
# ============================================================
class TestAdvancedStabilize(unittest.TestCase):
    """Tests for opencut.core.advanced_stabilize module."""

    def test_valid_modes(self):
        from opencut.core.advanced_stabilize import VALID_MODES
        self.assertIn("smooth", VALID_MODES)
        self.assertIn("lockdown", VALID_MODES)
        self.assertIn("perspective", VALID_MODES)

    def test_invalid_mode(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                stabilize_advanced("/fake/video.mp4", mode="invalid")

    def test_file_not_found(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        with self.assertRaises(FileNotFoundError):
            stabilize_advanced("/nonexistent/video.mp4")

    def test_smooth_mode(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.advanced_stabilize.run_ffmpeg"), \
             patch("builtins.open", MagicMock()), \
             patch("os.unlink"):
            result = stabilize_advanced("/fake/video.mp4", mode="smooth", smoothing=30)
            self.assertEqual(result["mode"], "smooth")
            self.assertEqual(result["smoothing"], 30)
            self.assertIn("output_path", result)
            self.assertIn("max_shift", result)
            self.assertIn("avg_rotation", result)

    def test_lockdown_mode(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.advanced_stabilize.run_ffmpeg"), \
             patch("builtins.open", MagicMock()), \
             patch("os.unlink"):
            result = stabilize_advanced("/fake/video.mp4", mode="lockdown")
            self.assertEqual(result["mode"], "lockdown")
            self.assertEqual(result["smoothing"], 0)

    def test_perspective_mode(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.advanced_stabilize.run_ffmpeg"), \
             patch("builtins.open", MagicMock()), \
             patch("os.unlink"):
            result = stabilize_advanced("/fake/video.mp4", mode="perspective", smoothing=20)
            self.assertEqual(result["mode"], "perspective")

    def test_smoothing_clamped(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.advanced_stabilize.run_ffmpeg"), \
             patch("builtins.open", MagicMock()), \
             patch("os.unlink"):
            result = stabilize_advanced("/fake/video.mp4", mode="smooth", smoothing=200)
            self.assertEqual(result["smoothing"], 100)

    def test_stabilize_signature(self):
        from opencut.core.advanced_stabilize import stabilize_advanced
        sig = inspect.signature(stabilize_advanced)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("mode", sig.parameters)
        self.assertIn("smoothing", sig.parameters)
        self.assertIn("output_path_override", sig.parameters)
        self.assertIn("on_progress", sig.parameters)


# ============================================================
# Color Space Tests
# ============================================================
class TestColorSpace(unittest.TestCase):
    """Tests for opencut.core.colorspace module."""

    def test_colorspace_info_defaults(self):
        from opencut.core.colorspace import ColorSpaceInfo
        info = ColorSpaceInfo()
        self.assertEqual(info.primaries, "unknown")
        self.assertEqual(info.transfer, "unknown")
        self.assertEqual(info.matrix, "unknown")
        self.assertEqual(info.bit_depth, 8)
        self.assertFalse(info.is_hdr)
        self.assertFalse(info.is_wide_gamut)
        self.assertEqual(info.profile_name, "unknown")

    def test_detect_colorspace_file_not_found(self):
        from opencut.core.colorspace import detect_colorspace
        with self.assertRaises(FileNotFoundError):
            detect_colorspace("/nonexistent/video.mp4")

    def test_detect_colorspace_bt709(self):
        from opencut.core.colorspace import detect_colorspace
        probe_output = json.dumps({
            "streams": [{
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_space": "bt709",
                "bits_per_raw_sample": "8",
                "pix_fmt": "yuv420p",
            }]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            info = detect_colorspace("/fake/video.mp4")
            self.assertEqual(info.primaries, "bt709")
            self.assertEqual(info.transfer, "bt709")
            self.assertFalse(info.is_hdr)
            self.assertFalse(info.is_wide_gamut)
            self.assertEqual(info.bit_depth, 8)
            self.assertEqual(info.profile_name, "bt709_sdr")

    def test_detect_colorspace_bt2020_pq(self):
        from opencut.core.colorspace import detect_colorspace
        probe_output = json.dumps({
            "streams": [{
                "color_primaries": "bt2020",
                "color_transfer": "smpte2084",
                "color_space": "bt2020nc",
                "bits_per_raw_sample": "10",
                "pix_fmt": "yuv420p10le",
            }]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            info = detect_colorspace("/fake/hdr.mp4")
            self.assertTrue(info.is_hdr)
            self.assertTrue(info.is_wide_gamut)
            self.assertEqual(info.bit_depth, 10)
            self.assertEqual(info.profile_name, "bt2020_pq")

    def test_detect_colorspace_10bit_pix_fmt(self):
        from opencut.core.colorspace import detect_colorspace
        probe_output = json.dumps({
            "streams": [{
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_space": "bt709",
                "pix_fmt": "yuv420p10le",
            }]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            info = detect_colorspace("/fake/video.mp4")
            self.assertEqual(info.bit_depth, 10)

    def test_convert_colorspace_file_not_found(self):
        from opencut.core.colorspace import convert_colorspace
        with self.assertRaises(FileNotFoundError):
            convert_colorspace("/nonexistent/video.mp4")

    def test_convert_colorspace_sdr_to_sdr(self):
        from opencut.core.colorspace import convert_colorspace
        probe_output = json.dumps({
            "streams": [{
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_space": "bt709",
                "bits_per_raw_sample": "8",
                "pix_fmt": "yuv420p",
            }]
        }).encode()
        mock_probe = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_probe), \
             patch("opencut.core.colorspace.run_ffmpeg"):
            result = convert_colorspace("/fake/video.mp4", target_primaries="bt709")
            self.assertIn("output_path", result)
            self.assertEqual(result["target_primaries"], "bt709")

    def test_batch_detect_empty(self):
        from opencut.core.colorspace import batch_detect_colorspace
        result = batch_detect_colorspace([])
        self.assertEqual(result, [])

    def test_batch_detect_multiple(self):
        from opencut.core.colorspace import batch_detect_colorspace
        probe_output = json.dumps({
            "streams": [{"color_primaries": "bt709", "color_transfer": "bt709", "color_space": "bt709", "pix_fmt": "yuv420p"}]
        }).encode()
        mock_result = MagicMock(returncode=0, stdout=probe_output)
        with patch("os.path.isfile", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            results = batch_detect_colorspace(["/fake/a.mp4", "/fake/b.mp4"])
            self.assertEqual(len(results), 2)
            for info in results:
                self.assertEqual(info.primaries, "bt709")

    def test_known_profiles_map(self):
        from opencut.core.colorspace import KNOWN_PROFILES
        self.assertIn(("bt709", "bt709", "bt709"), KNOWN_PROFILES)
        self.assertEqual(KNOWN_PROFILES[("bt709", "bt709", "bt709")], "bt709_sdr")


# ============================================================
# Route Tests
# ============================================================
class TestVideoProcessingRoutes(unittest.TestCase):
    """Tests for video_processing_routes blueprint registration and endpoints."""

    def test_blueprint_import(self):
        from opencut.routes.video_processing_routes import video_proc_bp
        self.assertEqual(video_proc_bp.name, "video_proc")

    def test_blueprint_registered(self):
        """The video_proc_bp should be registered in register_blueprints."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/video/hdr/detect", rules)
        self.assertIn("/video/hdr/tonemap", rules)
        self.assertIn("/video/audio-sync", rules)
        self.assertIn("/video/audio-sync/multi", rules)
        self.assertIn("/video/repair/diagnose", rules)
        self.assertIn("/video/repair", rules)
        self.assertIn("/video/rolling-shutter", rules)
        self.assertIn("/video/stabilize-advanced", rules)
        self.assertIn("/video/colorspace/detect", rules)
        self.assertIn("/video/colorspace/convert", rules)
        self.assertIn("/video/colorspace/batch-detect", rules)

    def test_hdr_detect_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/hdr/detect", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_colorspace_detect_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/colorspace/detect", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_hdr_tonemap_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/hdr/tonemap", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_audio_sync_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/audio-sync", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_repair_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/repair", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_rolling_shutter_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/rolling-shutter", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_stabilize_advanced_no_filepath(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
        resp = client.post("/video/stabilize-advanced", json={}, headers=headers)
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
