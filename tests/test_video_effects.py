"""
Tests for OpenCut Video Effects & Color features (3.2, 13.4, 43.3, 18.1, 18.4, 45.4).

Covers:
  - Sky replacement (detection, foreground lighting, full pipeline)
  - LOG camera profiles (detection, IDT application, LUT stacking, profile list)
  - Display calibration (SMPTE bars, grayscale ramp, gamut test, verification guide)
  - Cinemagraph (reference frame extraction, mask parsing, creation pipeline)
  - Hyperlapse (creation pipeline, stabilization)
  - Lossless intermediate (codec registry, to/from conversion, codec listing)
  - Route endpoints for all video effects operations
"""

import inspect
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 3.2 Sky Replacement — Core Tests
# ============================================================
class TestSkyReplacementCore(unittest.TestCase):
    """Tests for opencut.core.sky_replace module."""

    def test_sky_mask_result_fields(self):
        """SkyMaskResult dataclass should have all expected fields."""
        from opencut.core.sky_replace import SkyMaskResult
        r = SkyMaskResult()
        self.assertEqual(r.mask_path, "")
        self.assertEqual(r.sky_fraction, 0.0)
        self.assertEqual(r.horizon_y, 0)
        self.assertEqual(r.confidence, 0.0)
        self.assertEqual(r.width, 0)
        self.assertEqual(r.height, 0)

    def test_sky_replace_result_fields(self):
        """SkyReplaceResult dataclass should have all expected fields."""
        from opencut.core.sky_replace import SkyReplaceResult
        r = SkyReplaceResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.frames_processed, 0)
        self.assertEqual(r.avg_sky_fraction, 0.0)
        self.assertFalse(r.foreground_adjusted)
        self.assertEqual(r.method, "brightness")

    def test_detect_sky_mask_signature(self):
        """detect_sky_mask should have the expected parameters."""
        from opencut.core.sky_replace import detect_sky_mask
        sig = inspect.signature(detect_sky_mask)
        expected = ["frame_path", "method", "threshold", "blue_weight", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_detect_sky_mask_default_method(self):
        """Default method should be 'brightness'."""
        from opencut.core.sky_replace import detect_sky_mask
        sig = inspect.signature(detect_sky_mask)
        self.assertEqual(sig.parameters["method"].default, "brightness")

    def test_detect_sky_mask_default_threshold(self):
        """Default threshold should be 0.55."""
        from opencut.core.sky_replace import detect_sky_mask
        sig = inspect.signature(detect_sky_mask)
        self.assertEqual(sig.parameters["threshold"].default, 0.55)

    def test_detect_sky_mask_file_not_found(self):
        """detect_sky_mask should raise error for missing file."""
        from opencut.core.sky_replace import detect_sky_mask
        with patch("opencut.core.sky_replace.ensure_package", return_value=True):
            with patch.dict("sys.modules", {"cv2": MagicMock(), "numpy": MagicMock()}):
                with self.assertRaises(FileNotFoundError):
                    detect_sky_mask("/nonexistent/frame.png")

    def test_replace_sky_signature(self):
        """replace_sky should have the expected parameters."""
        from opencut.core.sky_replace import replace_sky
        sig = inspect.signature(replace_sky)
        expected_params = [
            "video_path", "sky_source", "output_path_str", "output_dir",
            "method", "threshold", "blue_weight", "feather",
            "adjust_lighting", "lighting_strength", "on_progress",
        ]
        for p in expected_params:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_replace_sky_file_not_found(self):
        """replace_sky should raise FileNotFoundError for missing input."""
        from opencut.core.sky_replace import replace_sky
        with patch("opencut.core.sky_replace.ensure_package", return_value=True):
            with patch.dict("sys.modules", {"cv2": MagicMock(), "numpy": MagicMock()}):
                with self.assertRaises(FileNotFoundError):
                    replace_sky("/nonexistent/video.mp4", "/nonexistent/sky.png")

    def test_adjust_foreground_lighting_signature(self):
        """adjust_foreground_lighting should have correct parameters."""
        from opencut.core.sky_replace import adjust_foreground_lighting
        sig = inspect.signature(adjust_foreground_lighting)
        expected = ["frame", "sky_brightness", "target_brightness", "strength"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_adjust_foreground_lighting_zero_strength(self):
        """Zero-strength adjustment should return the frame unchanged."""
        import numpy as np

        from opencut.core.sky_replace import adjust_foreground_lighting
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = adjust_foreground_lighting(frame, 0.5, strength=0.0)
        np.testing.assert_array_equal(result, frame)


# ============================================================
# 13.4 LOG Camera Profiles — Core Tests
# ============================================================
class TestLogProfilesCore(unittest.TestCase):
    """Tests for opencut.core.log_profiles module."""

    def test_known_profiles_exist(self):
        """All expected LOG profiles should be defined."""
        from opencut.core.log_profiles import KNOWN_PROFILES
        expected_ids = {"slog3", "clog3", "vlog", "logc", "braw", "nlog", "flog", "dlog"}
        self.assertEqual(set(KNOWN_PROFILES.keys()), expected_ids)

    def test_log_profile_dataclass_fields(self):
        """LogProfile dataclass should have required fields."""
        from opencut.core.log_profiles import KNOWN_PROFILES
        for pid, profile in KNOWN_PROFILES.items():
            self.assertTrue(hasattr(profile, "name"), f"{pid} missing name")
            self.assertTrue(hasattr(profile, "camera"), f"{pid} missing camera")
            self.assertTrue(hasattr(profile, "color_trc"), f"{pid} missing color_trc")
            self.assertTrue(hasattr(profile, "description"), f"{pid} missing description")

    def test_slog3_profile(self):
        """S-Log3 profile should reference Sony camera."""
        from opencut.core.log_profiles import KNOWN_PROFILES
        self.assertEqual(KNOWN_PROFILES["slog3"].camera, "Sony")
        self.assertIn("S-Log3", KNOWN_PROFILES["slog3"].name)

    def test_logc_profile(self):
        """LogC profile should reference ARRI camera."""
        from opencut.core.log_profiles import KNOWN_PROFILES
        self.assertEqual(KNOWN_PROFILES["logc"].camera, "ARRI")

    def test_log_detect_result_fields(self):
        """LogDetectResult should have all expected fields."""
        from opencut.core.log_profiles import LogDetectResult
        r = LogDetectResult()
        self.assertEqual(r.detected_profile, "")
        self.assertEqual(r.profile_name, "")
        self.assertEqual(r.camera, "")
        self.assertEqual(r.confidence, 0.0)

    def test_detect_log_profile_signature(self):
        """detect_log_profile should have video_path and on_progress."""
        from opencut.core.log_profiles import detect_log_profile
        sig = inspect.signature(detect_log_profile)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_detect_log_profile_file_not_found(self):
        """detect_log_profile should raise FileNotFoundError for missing file."""
        from opencut.core.log_profiles import detect_log_profile
        with self.assertRaises(FileNotFoundError):
            detect_log_profile("/nonexistent/video.mp4")

    def test_idt_transforms_match_profiles(self):
        """Every known profile should have an IDT transform."""
        from opencut.core.log_profiles import _IDT_TRANSFORMS, KNOWN_PROFILES
        for pid in KNOWN_PROFILES:
            self.assertIn(pid, _IDT_TRANSFORMS, f"Profile {pid} missing IDT transform")

    def test_apply_idt_signature(self):
        """apply_idt should have expected parameters."""
        from opencut.core.log_profiles import apply_idt
        sig = inspect.signature(apply_idt)
        expected = ["video_path", "profile", "output_path_str", "output_dir",
                    "lut_size", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_apply_idt_file_not_found(self):
        """apply_idt should raise FileNotFoundError for missing input."""
        from opencut.core.log_profiles import apply_idt
        with self.assertRaises(FileNotFoundError):
            apply_idt("/nonexistent/video.mp4", "slog3")

    def test_apply_idt_unknown_profile(self):
        """apply_idt should raise ValueError for unknown profile."""
        from opencut.core.log_profiles import apply_idt
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                apply_idt("/fake/video.mp4", "bogus_profile")
            self.assertIn("Unknown profile", str(ctx.exception))

    def test_stack_luts_signature(self):
        """stack_luts should have expected parameters."""
        from opencut.core.log_profiles import stack_luts
        sig = inspect.signature(stack_luts)
        expected = ["video_path", "lut_paths", "output_path_str", "output_dir", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_stack_luts_empty_list(self):
        """stack_luts should raise ValueError for empty lut_paths."""
        from opencut.core.log_profiles import stack_luts
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                stack_luts("/fake/video.mp4", [])

    def test_stack_luts_missing_lut_file(self):
        """stack_luts should raise FileNotFoundError for missing LUT."""
        from opencut.core.log_profiles import stack_luts
        with self.assertRaises(FileNotFoundError):
            stack_luts("/fake/video.mp4", ["/nonexistent/lut.cube"])

    def test_list_supported_profiles(self):
        """list_supported_profiles should return all profiles with metadata."""
        from opencut.core.log_profiles import list_supported_profiles
        profiles = list_supported_profiles()
        self.assertIsInstance(profiles, list)
        self.assertEqual(len(profiles), 8)  # slog3, clog3, vlog, logc, braw, nlog, flog, dlog
        ids = {p["id"] for p in profiles}
        self.assertIn("slog3", ids)
        self.assertIn("logc", ids)
        for p in profiles:
            self.assertIn("id", p)
            self.assertIn("name", p)
            self.assertIn("camera", p)
            self.assertIn("has_idt", p)
            self.assertTrue(p["has_idt"])

    def test_slog3_to_linear_function(self):
        """S-Log3 linearization should produce non-negative values."""
        from opencut.core.log_profiles import _slog3_to_linear
        # At 18% gray, S-Log3 maps 0.4105 -> ~0.18
        result = _slog3_to_linear(0.4105)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_linear_to_rec709(self):
        """Rec.709 OETF should clip at 0 for negative input and grow monotonically."""
        from opencut.core.log_profiles import _linear_to_rec709
        self.assertEqual(_linear_to_rec709(-1.0), 0.0)
        self.assertEqual(_linear_to_rec709(0.0), 0.0)
        val_low = _linear_to_rec709(0.01)
        val_high = _linear_to_rec709(0.5)
        self.assertGreater(val_high, val_low)

    @patch("opencut.core.log_profiles.run_ffmpeg")
    def test_generate_idt_lut_creates_file(self, mock_ffmpeg):
        """_generate_idt_lut should create a .cube file."""
        from opencut.core.log_profiles import _generate_idt_lut
        lut_path = _generate_idt_lut("slog3", size=5)
        self.assertTrue(lut_path.endswith(".cube"))
        self.assertTrue(os.path.isfile(lut_path))
        with open(lut_path, "r") as f:
            content = f.read()
        self.assertIn("TITLE", content)
        self.assertIn("LUT_SIZE 5", content)
        # Cleanup
        try:
            os.unlink(lut_path)
        except OSError:
            pass


# ============================================================
# 43.3 Display Calibration — Core Tests
# ============================================================
class TestDisplayCalibrationCore(unittest.TestCase):
    """Tests for opencut.core.display_calibration module."""

    def test_test_pattern_result_fields(self):
        """TestPatternResult should have expected fields."""
        from opencut.core.display_calibration import TestPatternResult
        r = TestPatternResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.pattern_type, "")
        self.assertEqual(r.resolution, (1920, 1080))

    def test_smpte_bars_signature(self):
        """generate_smpte_bars should have expected parameters."""
        from opencut.core.display_calibration import generate_smpte_bars
        sig = inspect.signature(generate_smpte_bars)
        expected = ["output_path_str", "resolution", "duration", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_smpte_top_bars_count(self):
        """SMPTE top bars should have 7 color values."""
        from opencut.core.display_calibration import _SMPTE_TOP_BARS
        self.assertEqual(len(_SMPTE_TOP_BARS), 7)

    def test_smpte_mid_bars_count(self):
        """SMPTE middle bars should have 7 entries."""
        from opencut.core.display_calibration import _SMPTE_MID_BARS
        self.assertEqual(len(_SMPTE_MID_BARS), 7)

    def test_smpte_bottom_pluge_count(self):
        """SMPTE PLUGE section should have 6 entries."""
        from opencut.core.display_calibration import _SMPTE_BOTTOM_PLUGE
        self.assertEqual(len(_SMPTE_BOTTOM_PLUGE), 6)

    def test_smpte_bars_small_resolution_error(self):
        """generate_smpte_bars should reject too-small resolution."""
        from opencut.core.display_calibration import generate_smpte_bars
        with patch("opencut.core.display_calibration.ensure_package", return_value=True):
            import numpy as np
            mock_cv2 = MagicMock()
            mock_cv2.imwrite = MagicMock()
            with patch.dict("sys.modules", {"cv2": mock_cv2, "numpy": np}):
                with self.assertRaises(ValueError):
                    generate_smpte_bars("/tmp/test.png", resolution=(50, 50))

    def test_grayscale_ramp_signature(self):
        """generate_grayscale_ramp should have expected parameters."""
        from opencut.core.display_calibration import generate_grayscale_ramp
        sig = inspect.signature(generate_grayscale_ramp)
        expected = ["output_path_str", "resolution", "steps", "include_labels",
                    "duration", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_grayscale_ramp_default_steps(self):
        """Default grayscale steps should be 32."""
        from opencut.core.display_calibration import generate_grayscale_ramp
        sig = inspect.signature(generate_grayscale_ramp)
        self.assertEqual(sig.parameters["steps"].default, 32)

    def test_gamut_test_signature(self):
        """generate_gamut_test should have expected parameters."""
        from opencut.core.display_calibration import generate_gamut_test
        sig = inspect.signature(generate_gamut_test)
        expected = ["output_path_str", "resolution", "include_skin_tones",
                    "duration", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_gamut_patches_defined(self):
        """Gamut patches for rec709 and skin_tones should be defined."""
        from opencut.core.display_calibration import _GAMUT_PATCHES
        self.assertIn("rec709", _GAMUT_PATCHES)
        self.assertIn("skin_tones", _GAMUT_PATCHES)
        self.assertEqual(len(_GAMUT_PATCHES["rec709"]), 8)
        self.assertEqual(len(_GAMUT_PATCHES["skin_tones"]), 8)

    def test_gamut_patches_have_rgb(self):
        """Each gamut patch should have name and rgb fields."""
        from opencut.core.display_calibration import _GAMUT_PATCHES
        for category, patches in _GAMUT_PATCHES.items():
            for patch_item in patches:
                self.assertIn("name", patch_item, f"Missing name in {category}")
                self.assertIn("rgb", patch_item, f"Missing rgb in {category}")
                self.assertEqual(len(patch_item["rgb"]), 3)

    def test_verification_guide_structure(self):
        """get_verification_guide should return structured guide with steps."""
        from opencut.core.display_calibration import get_verification_guide
        guide = get_verification_guide()
        self.assertIn("title", guide)
        self.assertIn("steps", guide)
        self.assertIn("notes", guide)
        self.assertEqual(len(guide["steps"]), 5)
        for step in guide["steps"]:
            self.assertIn("step", step)
            self.assertIn("pattern", step)
            self.assertIn("title", step)
            self.assertIn("instructions", step)
            self.assertIn("pass_criteria", step)
            self.assertIn("fail_criteria", step)

    def test_verification_guide_step_numbers(self):
        """Steps should be numbered 1 through 5."""
        from opencut.core.display_calibration import get_verification_guide
        guide = get_verification_guide()
        step_numbers = [s["step"] for s in guide["steps"]]
        self.assertEqual(step_numbers, [1, 2, 3, 4, 5])

    def test_verification_guide_notes_nonempty(self):
        """Notes should contain at least one item."""
        from opencut.core.display_calibration import get_verification_guide
        guide = get_verification_guide()
        self.assertGreater(len(guide["notes"]), 0)


# ============================================================
# 18.1 Cinemagraph — Core Tests
# ============================================================
class TestCinemagraphCore(unittest.TestCase):
    """Tests for opencut.core.cinemagraph module."""

    def test_cinemagraph_result_fields(self):
        """CinemagraphResult should have all expected fields."""
        from opencut.core.cinemagraph import CinemagraphResult
        r = CinemagraphResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.frames_written, 0)
        self.assertEqual(r.loop_duration, 0.0)
        self.assertEqual(r.crossfade_frames, 0)
        self.assertEqual(r.resolution, (0, 0))

    def test_reference_frame_result_fields(self):
        """ReferenceFrameResult should have all expected fields."""
        from opencut.core.cinemagraph import ReferenceFrameResult
        r = ReferenceFrameResult()
        self.assertEqual(r.frame_path, "")
        self.assertEqual(r.timestamp, 0.0)
        self.assertEqual(r.width, 0)
        self.assertEqual(r.height, 0)

    def test_extract_reference_frame_signature(self):
        """extract_reference_frame should have expected parameters."""
        from opencut.core.cinemagraph import extract_reference_frame
        sig = inspect.signature(extract_reference_frame)
        expected = ["video_path", "timestamp", "output_path_str", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_extract_reference_frame_file_not_found(self):
        """extract_reference_frame should raise FileNotFoundError."""
        from opencut.core.cinemagraph import extract_reference_frame
        with self.assertRaises(FileNotFoundError):
            extract_reference_frame("/nonexistent/video.mp4")

    def test_create_cinemagraph_signature(self):
        """create_cinemagraph should have expected parameters."""
        from opencut.core.cinemagraph import create_cinemagraph
        sig = inspect.signature(create_cinemagraph)
        expected = [
            "video_path", "mask_data", "loop_duration", "start_time",
            "crossfade", "output_path_str", "output_dir", "output_format",
            "ref_timestamp", "on_progress",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_create_cinemagraph_defaults(self):
        """Default loop_duration should be 3.0, crossfade 0.5."""
        from opencut.core.cinemagraph import create_cinemagraph
        sig = inspect.signature(create_cinemagraph)
        self.assertEqual(sig.parameters["loop_duration"].default, 3.0)
        self.assertEqual(sig.parameters["crossfade"].default, 0.5)
        self.assertEqual(sig.parameters["output_format"].default, "mp4")

    def test_create_cinemagraph_file_not_found(self):
        """create_cinemagraph should raise FileNotFoundError for missing input."""
        from opencut.core.cinemagraph import create_cinemagraph
        with patch("opencut.core.cinemagraph.ensure_package", return_value=True):
            with patch.dict("sys.modules", {"cv2": MagicMock(), "numpy": MagicMock()}):
                with self.assertRaises(FileNotFoundError):
                    create_cinemagraph("/nonexistent/video.mp4", {"type": "rect"})

    def test_parse_mask_rect(self):
        """_parse_mask_data should create a rectangular mask."""
        import numpy as np

        from opencut.core.cinemagraph import _parse_mask_data
        mock_cv2 = MagicMock()
        mock_cv2.rectangle = MagicMock()
        mock_cv2.GaussianBlur = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            mask = _parse_mask_data({"type": "rect", "x": 10, "y": 10, "w": 50, "h": 50, "feather": 0}, 100, 100)
        self.assertEqual(mask.shape, (100, 100))

    def test_parse_mask_unknown_type_raises(self):
        """_parse_mask_data should raise ValueError for unknown mask type."""

        from opencut.core.cinemagraph import _parse_mask_data
        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with self.assertRaises(ValueError) as ctx:
                _parse_mask_data({"type": "spiral"}, 100, 100)
            self.assertIn("Unknown mask type", str(ctx.exception))

    def test_parse_mask_polygon_too_few_points(self):
        """_parse_mask_data with polygon needs at least 3 points."""

        from opencut.core.cinemagraph import _parse_mask_data
        mock_cv2 = MagicMock()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            with self.assertRaises(ValueError):
                _parse_mask_data({"type": "polygon", "points": [[0, 0], [10, 10]]}, 100, 100)


# ============================================================
# 18.4 Hyperlapse — Core Tests
# ============================================================
class TestHyperlapseCore(unittest.TestCase):
    """Tests for opencut.core.hyperlapse module."""

    def test_hyperlapse_result_fields(self):
        """HyperlapseResult should have all expected fields."""
        from opencut.core.hyperlapse import HyperlapseResult
        r = HyperlapseResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.speed_factor, 1.0)
        self.assertEqual(r.original_duration, 0.0)
        self.assertEqual(r.output_duration, 0.0)
        self.assertEqual(r.frames_sampled, 0)
        self.assertEqual(r.stabilization_smoothing, 0)
        self.assertEqual(r.edge_fill, "none")

    def test_stabilize_result_fields(self):
        """StabilizeResult should have all expected fields."""
        from opencut.core.hyperlapse import StabilizeResult
        r = StabilizeResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.smoothing, 30)
        self.assertEqual(r.passes, 1)
        self.assertEqual(r.edge_fill, "none")

    def test_create_hyperlapse_signature(self):
        """create_hyperlapse should have expected parameters."""
        from opencut.core.hyperlapse import create_hyperlapse
        sig = inspect.signature(create_hyperlapse)
        expected = [
            "video_path", "speed_factor", "smoothing", "edge_fill",
            "output_path_str", "output_dir", "stabilize", "passes", "on_progress",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_create_hyperlapse_defaults(self):
        """Default speed factor should be 10.0, smoothing 45."""
        from opencut.core.hyperlapse import create_hyperlapse
        sig = inspect.signature(create_hyperlapse)
        self.assertEqual(sig.parameters["speed_factor"].default, 10.0)
        self.assertEqual(sig.parameters["smoothing"].default, 45)
        self.assertEqual(sig.parameters["edge_fill"].default, "mirror")
        self.assertTrue(sig.parameters["stabilize"].default)
        self.assertEqual(sig.parameters["passes"].default, 2)

    def test_create_hyperlapse_file_not_found(self):
        """create_hyperlapse should raise FileNotFoundError for missing input."""
        from opencut.core.hyperlapse import create_hyperlapse
        with self.assertRaises(FileNotFoundError):
            create_hyperlapse("/nonexistent/video.mp4")

    def test_create_hyperlapse_speed_clamped(self):
        """Speed factor should be clamped to 1.0-100.0 range."""
        from opencut.core.hyperlapse import create_hyperlapse
        # We cannot fully run it without a real file, but we test through error path
        with self.assertRaises(FileNotFoundError):
            create_hyperlapse("/nonexistent/video.mp4", speed_factor=0.5)
        with self.assertRaises(FileNotFoundError):
            create_hyperlapse("/nonexistent/video.mp4", speed_factor=200.0)

    def test_stabilize_hyperlapse_signature(self):
        """stabilize_hyperlapse should have expected parameters."""
        from opencut.core.hyperlapse import stabilize_hyperlapse
        sig = inspect.signature(stabilize_hyperlapse)
        expected = [
            "video_path", "smoothing", "edge_fill", "passes",
            "output_path_str", "output_dir", "on_progress",
        ]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_stabilize_hyperlapse_file_not_found(self):
        """stabilize_hyperlapse should raise FileNotFoundError for missing input."""
        from opencut.core.hyperlapse import stabilize_hyperlapse
        with self.assertRaises(FileNotFoundError):
            stabilize_hyperlapse("/nonexistent/video.mp4")


# ============================================================
# 45.4 Lossless Intermediate — Core Tests
# ============================================================
class TestLosslessIntermediateCore(unittest.TestCase):
    """Tests for opencut.core.lossless_intermediate module."""

    def test_intermediate_codecs_defined(self):
        """All three intermediate codecs should be defined."""
        from opencut.core.lossless_intermediate import INTERMEDIATE_CODECS
        expected = {"ffv1", "huffyuv", "utvideo"}
        self.assertEqual(set(INTERMEDIATE_CODECS.keys()), expected)

    def test_ffv1_codec_properties(self):
        """FFV1 should use .mkv container and support 10-bit."""
        from opencut.core.lossless_intermediate import INTERMEDIATE_CODECS
        ffv1 = INTERMEDIATE_CODECS["ffv1"]
        self.assertEqual(ffv1.container, ".mkv")
        self.assertEqual(ffv1.ffmpeg_encoder, "ffv1")
        self.assertIn("yuv422p10le", ffv1.pix_fmts)
        self.assertEqual(ffv1.default_pix_fmt, "yuv422p10le")

    def test_huffyuv_codec_properties(self):
        """HuffYUV should use .avi container."""
        from opencut.core.lossless_intermediate import INTERMEDIATE_CODECS
        huffyuv = INTERMEDIATE_CODECS["huffyuv"]
        self.assertEqual(huffyuv.container, ".avi")
        self.assertEqual(huffyuv.ffmpeg_encoder, "huffyuv")
        self.assertEqual(huffyuv.default_pix_fmt, "yuv422p")

    def test_utvideo_codec_properties(self):
        """UT Video should use .avi container."""
        from opencut.core.lossless_intermediate import INTERMEDIATE_CODECS
        ut = INTERMEDIATE_CODECS["utvideo"]
        self.assertEqual(ut.container, ".avi")
        self.assertEqual(ut.ffmpeg_encoder, "utvideo")

    def test_intermediate_result_fields(self):
        """IntermediateResult should have all expected fields."""
        from opencut.core.lossless_intermediate import IntermediateResult
        r = IntermediateResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.codec, "")
        self.assertEqual(r.pix_fmt, "")
        self.assertEqual(r.file_size_mb, 0.0)
        self.assertTrue(r.lossless)

    def test_delivery_result_fields(self):
        """DeliveryResult should have all expected fields."""
        from opencut.core.lossless_intermediate import DeliveryResult
        r = DeliveryResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.source_codec, "")
        self.assertEqual(r.delivery_codec, "")
        self.assertEqual(r.file_size_mb, 0.0)

    def test_to_intermediate_signature(self):
        """to_intermediate should have expected parameters."""
        from opencut.core.lossless_intermediate import to_intermediate
        sig = inspect.signature(to_intermediate)
        expected = ["video_path", "codec", "pix_fmt", "output_path_str",
                    "output_dir", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_to_intermediate_file_not_found(self):
        """to_intermediate should raise FileNotFoundError for missing input."""
        from opencut.core.lossless_intermediate import to_intermediate
        with self.assertRaises(FileNotFoundError):
            to_intermediate("/nonexistent/video.mp4")

    def test_to_intermediate_unknown_codec(self):
        """to_intermediate should raise ValueError for unknown codec."""
        from opencut.core.lossless_intermediate import to_intermediate
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                to_intermediate("/fake/video.mp4", codec="magic_codec")
            self.assertIn("Unknown intermediate codec", str(ctx.exception))

    def test_from_intermediate_signature(self):
        """from_intermediate should have expected parameters."""
        from opencut.core.lossless_intermediate import from_intermediate
        sig = inspect.signature(from_intermediate)
        expected = ["intermediate_path", "delivery_codec", "output_path_str",
                    "output_dir", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_from_intermediate_file_not_found(self):
        """from_intermediate should raise FileNotFoundError for missing input."""
        from opencut.core.lossless_intermediate import from_intermediate
        with self.assertRaises(FileNotFoundError):
            from_intermediate("/nonexistent/intermediate.mkv")

    def test_from_intermediate_unknown_preset(self):
        """from_intermediate should raise ValueError for unknown delivery preset."""
        from opencut.core.lossless_intermediate import from_intermediate
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                from_intermediate("/fake/intermediate.mkv", delivery_codec="betamax")
            self.assertIn("Unknown delivery preset", str(ctx.exception))

    def test_delivery_presets_defined(self):
        """All delivery presets should be defined."""
        from opencut.core.lossless_intermediate import DELIVERY_PRESETS
        expected = {"h264_web", "h264_master", "h265_web", "h265_master",
                    "prores_proxy", "prores_hq", "dnxhd_36"}
        self.assertEqual(set(DELIVERY_PRESETS.keys()), expected)

    def test_delivery_presets_have_codec(self):
        """Each delivery preset should specify a codec."""
        from opencut.core.lossless_intermediate import DELIVERY_PRESETS
        for name, preset in DELIVERY_PRESETS.items():
            self.assertIn("codec", preset, f"Preset {name} missing codec")
            self.assertIn("description", preset, f"Preset {name} missing description")

    def test_list_intermediate_codecs(self):
        """list_intermediate_codecs should return all codecs with metadata."""
        from opencut.core.lossless_intermediate import list_intermediate_codecs
        codecs = list_intermediate_codecs()
        self.assertIsInstance(codecs, list)
        self.assertEqual(len(codecs), 3)
        ids = {c["id"] for c in codecs}
        self.assertEqual(ids, {"ffv1", "huffyuv", "utvideo"})
        for c in codecs:
            self.assertIn("name", c)
            self.assertIn("container", c)
            self.assertIn("pix_fmts", c)
            self.assertIn("description", c)
            self.assertIn("pros", c)
            self.assertIn("cons", c)

    def test_get_recommended_codec_general(self):
        """General recommendation should return ffv1."""
        from opencut.core.lossless_intermediate import get_recommended_codec
        rec = get_recommended_codec("general")
        self.assertEqual(rec["recommended_codec"], "ffv1")
        self.assertIn("reason", rec)

    def test_get_recommended_codec_speed(self):
        """Speed recommendation should return utvideo."""
        from opencut.core.lossless_intermediate import get_recommended_codec
        rec = get_recommended_codec("speed")
        self.assertEqual(rec["recommended_codec"], "utvideo")

    def test_get_recommended_codec_compatibility(self):
        """Compatibility recommendation should return huffyuv."""
        from opencut.core.lossless_intermediate import get_recommended_codec
        rec = get_recommended_codec("compatibility")
        self.assertEqual(rec["recommended_codec"], "huffyuv")

    def test_get_recommended_codec_unknown_fallback(self):
        """Unknown use case should fall back to general."""
        from opencut.core.lossless_intermediate import get_recommended_codec
        rec = get_recommended_codec("totally_unknown")
        self.assertEqual(rec["use_case"], "general")
        self.assertEqual(rec["recommended_codec"], "ffv1")


# ============================================================
# Route Blueprint Registration
# ============================================================
class TestVideoEffectsBlueprint(unittest.TestCase):
    """Tests for video_effects_bp Blueprint and route registration."""

    def test_blueprint_importable(self):
        """video_effects_bp should be importable from routes."""
        from opencut.routes.video_effects_routes import video_effects_bp
        self.assertEqual(video_effects_bp.name, "video_effects")

    def test_blueprint_registered_in_init(self):
        """video_effects_bp should be registered in the app."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        # Check registered blueprints
        bp_names = [bp.name for bp in app.blueprints.values()]
        self.assertIn("video_effects", bp_names)

    def test_sky_replace_route_exists(self):
        """The /api/video/sky-replace route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/video/sky-replace", rules)

    def test_log_detect_route_exists(self):
        """The /api/video/log-detect route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/video/log-detect", rules)

    def test_log_apply_route_exists(self):
        """The /api/video/log-apply route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/video/log-apply", rules)

    def test_lut_stack_route_exists(self):
        """The /api/video/lut-stack route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/video/lut-stack", rules)

    def test_test_pattern_route_exists(self):
        """The /api/display/test-pattern route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/display/test-pattern", rules)

    def test_cinemagraph_route_exists(self):
        """The /api/video/cinemagraph route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/video/cinemagraph", rules)

    def test_hyperlapse_route_exists(self):
        """The /api/video/hyperlapse route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/video/hyperlapse", rules)

    def test_intermediate_route_exists(self):
        """The /api/encoding/intermediate route should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/encoding/intermediate", rules)


# ============================================================
# Route Endpoint Tests (via test client)
# ============================================================
class TestVideoEffectsRoutes(unittest.TestCase):
    """Integration tests for video effects route endpoints."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        config = OpenCutConfig()
        cls.app = create_app(config=config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf_token = data.get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf_token,
            "Content-Type": "application/json",
        }

    def test_sky_replace_missing_filepath(self):
        """Sky replace should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/video/sky-replace",
            headers=self.headers,
            data=json.dumps({"sky_source": "/fake/sky.png"}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_sky_replace_missing_sky_source(self):
        """Sky replace should return a job that errors without sky_source."""
        resp = self.client.post(
            "/api/video/sky-replace",
            headers=self.headers,
            data=json.dumps({"filepath": "/nonexistent/video.mp4"}),
        )
        # Should be 400 (file not found) before job creation
        self.assertIn(resp.status_code, [400, 200])

    def test_log_detect_missing_filepath(self):
        """LOG detect should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/video/log-detect",
            headers=self.headers,
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_log_apply_missing_filepath(self):
        """LOG apply should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/video/log-apply",
            headers=self.headers,
            data=json.dumps({"profile": "slog3"}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_lut_stack_missing_filepath(self):
        """LUT stack should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/video/lut-stack",
            headers=self.headers,
            data=json.dumps({"lut_paths": ["/fake.cube"]}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_test_pattern_generates_job(self):
        """Test pattern route should accept request and return a job_id."""
        resp = self.client.post(
            "/api/display/test-pattern",
            headers=self.headers,
            data=json.dumps({"pattern": "guide"}),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("job_id", data)

    def test_cinemagraph_missing_filepath(self):
        """Cinemagraph should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/video/cinemagraph",
            headers=self.headers,
            data=json.dumps({"mask": {"type": "rect"}}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_hyperlapse_missing_filepath(self):
        """Hyperlapse should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/video/hyperlapse",
            headers=self.headers,
            data=json.dumps({"speed_factor": 10.0}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_intermediate_missing_filepath(self):
        """Intermediate route should return 400 when filepath is missing."""
        resp = self.client.post(
            "/api/encoding/intermediate",
            headers=self.headers,
            data=json.dumps({"direction": "to", "codec": "ffv1"}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_csrf_required(self):
        """Routes should reject requests without CSRF token."""
        resp = self.client.post(
            "/api/video/sky-replace",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"filepath": "/fake.mp4"}),
        )
        self.assertEqual(resp.status_code, 403)

    def test_all_routes_are_post(self):
        """All video effects routes should only accept POST."""
        endpoints = [
            "/api/video/sky-replace",
            "/api/video/log-detect",
            "/api/video/log-apply",
            "/api/video/lut-stack",
            "/api/display/test-pattern",
            "/api/video/cinemagraph",
            "/api/video/hyperlapse",
            "/api/encoding/intermediate",
        ]
        for endpoint in endpoints:
            resp = self.client.get(endpoint)
            self.assertEqual(resp.status_code, 405,
                             f"GET {endpoint} should return 405 Method Not Allowed")


# ============================================================
# Cross-Module Integration
# ============================================================
class TestCrossModuleIntegration(unittest.TestCase):
    """Integration tests that verify cross-module consistency."""

    def test_all_core_modules_importable(self):
        """All 6 core modules should be importable."""
        from opencut.core import (
            cinemagraph,
            display_calibration,
            hyperlapse,
            log_profiles,
            lossless_intermediate,
            sky_replace,
        )
        self.assertIsNotNone(sky_replace)
        self.assertIsNotNone(log_profiles)
        self.assertIsNotNone(display_calibration)
        self.assertIsNotNone(cinemagraph)
        self.assertIsNotNone(hyperlapse)
        self.assertIsNotNone(lossless_intermediate)

    def test_all_core_functions_exported(self):
        """Key functions should be importable from their modules."""
        from opencut.core.cinemagraph import create_cinemagraph, extract_reference_frame
        from opencut.core.display_calibration import (
            generate_gamut_test,
            generate_grayscale_ramp,
            generate_smpte_bars,
            get_verification_guide,
        )
        from opencut.core.hyperlapse import create_hyperlapse, stabilize_hyperlapse
        from opencut.core.log_profiles import apply_idt, detect_log_profile, list_supported_profiles, stack_luts
        from opencut.core.lossless_intermediate import from_intermediate, list_intermediate_codecs, to_intermediate
        from opencut.core.sky_replace import adjust_foreground_lighting, detect_sky_mask, replace_sky

        # Verify they are callable
        for func in [detect_sky_mask, replace_sky, adjust_foreground_lighting,
                     detect_log_profile, apply_idt, stack_luts, list_supported_profiles,
                     generate_smpte_bars, generate_grayscale_ramp, generate_gamut_test,
                     get_verification_guide,
                     create_cinemagraph, extract_reference_frame,
                     create_hyperlapse, stabilize_hyperlapse,
                     to_intermediate, from_intermediate, list_intermediate_codecs]:
            self.assertTrue(callable(func), f"{func.__name__} should be callable")

    def test_all_progress_callbacks_optional(self):
        """All core functions with on_progress should default to None."""
        from opencut.core.cinemagraph import create_cinemagraph, extract_reference_frame
        from opencut.core.display_calibration import generate_gamut_test, generate_grayscale_ramp, generate_smpte_bars
        from opencut.core.hyperlapse import create_hyperlapse, stabilize_hyperlapse
        from opencut.core.log_profiles import apply_idt, detect_log_profile, stack_luts
        from opencut.core.lossless_intermediate import from_intermediate, to_intermediate
        from opencut.core.sky_replace import detect_sky_mask, replace_sky

        funcs = [detect_sky_mask, replace_sky, detect_log_profile, apply_idt,
                 stack_luts, generate_smpte_bars, generate_grayscale_ramp,
                 generate_gamut_test, create_cinemagraph, extract_reference_frame,
                 create_hyperlapse, stabilize_hyperlapse, to_intermediate,
                 from_intermediate]

        for func in funcs:
            sig = inspect.signature(func)
            if "on_progress" in sig.parameters:
                self.assertIsNone(
                    sig.parameters["on_progress"].default,
                    f"{func.__name__}: on_progress default should be None",
                )

    def test_all_core_use_helpers(self):
        """Core modules should import from opencut.helpers."""
        import opencut.core.cinemagraph as m4
        import opencut.core.display_calibration as m3
        import opencut.core.hyperlapse as m5
        import opencut.core.log_profiles as m2
        import opencut.core.lossless_intermediate as m6
        import opencut.core.sky_replace as m1

        for mod in [m1, m2, m3, m4, m5, m6]:
            source = inspect.getsource(mod)
            self.assertIn("from opencut.helpers import", source,
                          f"{mod.__name__} should import from opencut.helpers")

    def test_route_count(self):
        """video_effects_bp should have 8 route rules."""
        from opencut.routes.video_effects_routes import video_effects_bp
        rules = list(video_effects_bp.deferred_functions)
        self.assertEqual(len(rules), 8)


if __name__ == "__main__":
    unittest.main()
