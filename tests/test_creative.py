"""
Tests for OpenCut creative features.

Covers: video_compare, retro_effects (retro, tilt-shift, light leaks),
        accessibility (color blindness simulation, flash detection),
        and creative_routes blueprint.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Video Comparison Tests
# ============================================================
class TestVideoCompare(unittest.TestCase):
    """Tests for opencut.core.video_compare."""

    def setUp(self):
        self.tmp_a = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp_a.write(b"\x00" * 100)
        self.tmp_a.close()
        self.tmp_b = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp_b.write(b"\x00" * 100)
        self.tmp_b.close()

    def tearDown(self):
        for f in (self.tmp_a.name, self.tmp_b.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_videos_sidebyside(self, mock_info, mock_ffmpeg):
        """Side-by-side comparison should use hstack filter."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import compare_videos

        result = compare_videos(self.tmp_a.name, self.tmp_b.name, mode="sidebyside")

        self.assertIn("output_path", result)
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        self.assertIn("hstack", cmd_str)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_videos_difference(self, mock_info, mock_ffmpeg):
        """Difference mode should use blend=all_mode=difference."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import compare_videos

        result = compare_videos(self.tmp_a.name, self.tmp_b.name, mode="difference")

        self.assertIn("output_path", result)
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("difference", cmd_str)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_videos_overlay(self, mock_info, mock_ffmpeg):
        """Overlay mode should use blend=all_mode=average."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import compare_videos

        compare_videos(self.tmp_a.name, self.tmp_b.name, mode="overlay")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("average", cmd_str)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_videos_wipe(self, mock_info, mock_ffmpeg):
        """Wipe mode should use xfade transition."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import compare_videos

        compare_videos(self.tmp_a.name, self.tmp_b.name, mode="wipe")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("xfade", cmd_str)

    def test_compare_videos_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        from opencut.core.video_compare import compare_videos

        with self.assertRaises(ValueError):
            compare_videos(self.tmp_a.name, self.tmp_b.name, mode="invalid")

    def test_compare_videos_missing_file(self):
        """Missing input file should raise FileNotFoundError."""
        from opencut.core.video_compare import compare_videos

        with self.assertRaises(FileNotFoundError):
            compare_videos("/nonexistent/a.mp4", self.tmp_b.name)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_videos_progress_callback(self, mock_info, mock_ffmpeg):
        """Progress callback should be invoked during comparison."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import compare_videos

        progress = MagicMock()
        compare_videos(self.tmp_a.name, self.tmp_b.name, on_progress=progress)

        self.assertTrue(progress.called)
        # Should call at least once with 100 for completion
        pct_values = [c[0][0] for c in progress.call_args_list]
        self.assertIn(100, pct_values)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_frames_returns_output_path(self, mock_info, mock_ffmpeg):
        """compare_frames should return dict with output_path."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import compare_frames

        result = compare_frames(self.tmp_a.name, self.tmp_b.name, timestamp=5.0)

        self.assertIn("output_path", result)
        self.assertTrue(result["output_path"].endswith(".png"))

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_compare_frames_all_modes(self, mock_info, mock_ffmpeg):
        """compare_frames should work for all valid modes."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.video_compare import VALID_MODES, compare_frames

        for mode in VALID_MODES:
            result = compare_frames(self.tmp_a.name, self.tmp_b.name, mode=mode)
            self.assertIn("output_path", result)


# ============================================================
# Retro Effects Tests
# ============================================================
class TestRetroEffects(unittest.TestCase):
    """Tests for opencut.core.retro_effects.apply_retro_effect."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_vhs_effect(self, mock_info, mock_ffmpeg):
        """VHS effect should include chromashift and noise filters."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_retro_effect

        result = apply_retro_effect(self.tmp.name, effect="vhs")
        self.assertIn("output_path", result)
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("chromashift", cmd_str)
        self.assertIn("noise", cmd_str)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_super8_effect(self, mock_info, mock_ffmpeg):
        """Super 8 effect should include vignette and colorbalance."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_retro_effect

        apply_retro_effect(self.tmp.name, effect="super8")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("vignette", cmd_str)
        self.assertIn("colorbalance", cmd_str)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_film_damage_effect(self, mock_info, mock_ffmpeg):
        """Film damage effect should include grain and vignette."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_retro_effect

        apply_retro_effect(self.tmp.name, effect="film_damage")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("noise", cmd_str)
        self.assertIn("vignette", cmd_str)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_old_tv_effect(self, mock_info, mock_ffmpeg):
        """Old TV effect should include interlace simulation and noise."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_retro_effect

        apply_retro_effect(self.tmp.name, effect="old_tv")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("tinterlace", cmd_str)
        self.assertIn("noise", cmd_str)

    def test_invalid_effect_raises(self):
        """Invalid effect name should raise ValueError."""
        from opencut.core.retro_effects import apply_retro_effect

        with self.assertRaises(ValueError):
            apply_retro_effect(self.tmp.name, effect="betamax")

    def test_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.retro_effects import apply_retro_effect

        with self.assertRaises(FileNotFoundError):
            apply_retro_effect("/nonexistent/video.mp4")

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_intensity_clamped(self, mock_info, mock_ffmpeg):
        """Intensity should be clamped to [0.0, 1.0]."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_retro_effect

        # Should not raise even with out-of-range intensity
        result = apply_retro_effect(self.tmp.name, effect="vhs", intensity=5.0)
        self.assertIn("output_path", result)

        result = apply_retro_effect(self.tmp.name, effect="vhs", intensity=-1.0)
        self.assertIn("output_path", result)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_progress_callback(self, mock_info, mock_ffmpeg):
        """Progress callback should fire at least at start and end."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_retro_effect

        progress = MagicMock()
        apply_retro_effect(self.tmp.name, effect="vhs", on_progress=progress)

        self.assertTrue(progress.called)
        pct_values = [c[0][0] for c in progress.call_args_list]
        self.assertIn(100, pct_values)


# ============================================================
# Tilt-Shift Tests
# ============================================================
class TestTiltShift(unittest.TestCase):
    """Tests for opencut.core.retro_effects.apply_tilt_shift."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_tilt_shift_default_params(self, mock_info, mock_ffmpeg):
        """Tilt-shift with defaults should produce output_path with boxblur."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_tilt_shift

        result = apply_tilt_shift(self.tmp.name)

        self.assertIn("output_path", result)
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("boxblur", cmd_str)
        self.assertIn("saturation", cmd_str)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_tilt_shift_custom_params(self, mock_info, mock_ffmpeg):
        """Custom tilt-shift parameters should be applied."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_tilt_shift

        result = apply_tilt_shift(
            self.tmp.name, focus_y=0.3, focus_width=0.5,
            blur_amount=20, saturation=2.0,
        )
        self.assertIn("output_path", result)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_tilt_shift_clamps_params(self, mock_info, mock_ffmpeg):
        """Out-of-range parameters should be clamped, not error."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_tilt_shift

        result = apply_tilt_shift(
            self.tmp.name, focus_y=5.0, blur_amount=999, saturation=-1.0,
        )
        self.assertIn("output_path", result)

    def test_tilt_shift_missing_file(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.retro_effects import apply_tilt_shift

        with self.assertRaises(FileNotFoundError):
            apply_tilt_shift("/nonexistent/video.mp4")


# ============================================================
# Light Leak Tests
# ============================================================
class TestLightLeak(unittest.TestCase):
    """Tests for opencut.core.retro_effects.apply_light_leak."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_all_styles(self, mock_info, mock_ffmpeg):
        """All light leak styles should produce valid output."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import VALID_LIGHT_LEAK_STYLES, apply_light_leak

        for style in VALID_LIGHT_LEAK_STYLES:
            result = apply_light_leak(self.tmp.name, style=style)
            self.assertIn("output_path", result)

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_light_leak_uses_screen_blend(self, mock_info, mock_ffmpeg):
        """Light leak should use screen blend mode."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_light_leak

        apply_light_leak(self.tmp.name, style="warm_amber")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("screen", cmd_str)

    def test_invalid_style_raises(self):
        """Invalid style should raise ValueError."""
        from opencut.core.retro_effects import apply_light_leak

        with self.assertRaises(ValueError):
            apply_light_leak(self.tmp.name, style="neon_green")

    def test_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.retro_effects import apply_light_leak

        with self.assertRaises(FileNotFoundError):
            apply_light_leak("/nonexistent/video.mp4")

    @patch("opencut.core.retro_effects.run_ffmpeg")
    @patch("opencut.core.retro_effects.get_video_info")
    def test_intensity_affects_opacity(self, mock_info, mock_ffmpeg):
        """Different intensity values should change the blend opacity."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.retro_effects import apply_light_leak

        apply_light_leak(self.tmp.name, style="warm_amber", intensity=0.2)
        cmd_low = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])

        apply_light_leak(self.tmp.name, style="warm_amber", intensity=0.9)
        cmd_high = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])

        # The opacity values in the filter strings should differ
        self.assertNotEqual(cmd_low, cmd_high)


# ============================================================
# Color Blindness Simulation Tests
# ============================================================
class TestColorBlindSim(unittest.TestCase):
    """Tests for opencut.core.accessibility.simulate_color_blindness."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.accessibility.run_ffmpeg")
    def test_all_conditions(self, mock_ffmpeg):
        """All color blindness conditions should produce output."""
        from opencut.core.accessibility import VALID_CONDITIONS, simulate_color_blindness

        for condition in VALID_CONDITIONS:
            result = simulate_color_blindness(self.tmp.name, condition=condition)
            self.assertIn("output_path", result)

    @patch("opencut.core.accessibility.run_ffmpeg")
    def test_deuteranopia_uses_colorchannelmixer(self, mock_ffmpeg):
        """Deuteranopia should use colorchannelmixer filter."""
        from opencut.core.accessibility import simulate_color_blindness

        simulate_color_blindness(self.tmp.name, condition="deuteranopia")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("colorchannelmixer", cmd_str)

    @patch("opencut.core.accessibility.run_ffmpeg")
    def test_achromatopsia_grayscale_matrix(self, mock_ffmpeg):
        """Achromatopsia should use equal-weight grayscale matrix."""
        from opencut.core.accessibility import simulate_color_blindness

        simulate_color_blindness(self.tmp.name, condition="achromatopsia")
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        # All rows should have the same coefficients (luminance)
        self.assertIn("rr=0.299", cmd_str)
        self.assertIn("gr=0.299", cmd_str)
        self.assertIn("br=0.299", cmd_str)

    def test_invalid_condition_raises(self):
        """Invalid condition should raise ValueError."""
        from opencut.core.accessibility import simulate_color_blindness

        with self.assertRaises(ValueError):
            simulate_color_blindness(self.tmp.name, condition="tetrachromacy")

    def test_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.accessibility import simulate_color_blindness

        with self.assertRaises(FileNotFoundError):
            simulate_color_blindness("/nonexistent/video.mp4")


# ============================================================
# Flash Detection Tests
# ============================================================
class TestFlashDetection(unittest.TestCase):
    """Tests for opencut.core.accessibility.detect_flashing."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_safe_video_no_flashes(self, mock_info, mock_lum):
        """Video with stable luminance should be assessed as 'safe'."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}
        # Steady luminance values
        mock_lum.return_value = [128.0] * 150

        from opencut.core.accessibility import detect_flashing

        result = detect_flashing(self.tmp.name)

        self.assertEqual(result["risk_assessment"], "safe")
        self.assertEqual(result["total_flashes"], 0)
        self.assertEqual(result["events"], [])

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_dangerous_rapid_flashing(self, mock_info, mock_lum):
        """Rapid alternating luminance should be flagged as dangerous."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.0}
        # Alternating high/low luminance every frame (extreme flashing)
        values = []
        for i in range(60):
            values.append(230.0 if i % 2 == 0 else 20.0)
        mock_lum.return_value = values

        from opencut.core.accessibility import detect_flashing

        result = detect_flashing(self.tmp.name, max_flashes_per_sec=3, min_luminance_change=0.2)

        self.assertIn(result["risk_assessment"], ("warning", "dangerous"))
        self.assertGreater(result["total_flashes"], 0)
        self.assertGreater(len(result["events"]), 0)

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_flash_event_structure(self, mock_info, mock_lum):
        """Flash events should have the expected fields."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.0}
        values = []
        for i in range(60):
            values.append(200.0 if i % 2 == 0 else 50.0)
        mock_lum.return_value = values

        from opencut.core.accessibility import detect_flashing

        result = detect_flashing(self.tmp.name)

        for event in result["events"]:
            self.assertIn("start", event)
            self.assertIn("end", event)
            self.assertIn("flash_count", event)
            self.assertIn("peak_luminance_change", event)
            self.assertIn("severity", event)
            self.assertIn(event["severity"], ("low", "medium", "high"))

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_empty_luminance_returns_safe(self, mock_info, mock_lum):
        """Empty luminance extraction should return safe result."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}
        mock_lum.return_value = []

        from opencut.core.accessibility import detect_flashing

        result = detect_flashing(self.tmp.name)

        self.assertEqual(result["risk_assessment"], "safe")

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_progress_callback_fires(self, mock_info, mock_lum):
        """Progress callback should be called during detection."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 1.0}
        mock_lum.return_value = [128.0] * 30

        from opencut.core.accessibility import detect_flashing

        progress = MagicMock()
        detect_flashing(self.tmp.name, on_progress=progress)

        self.assertTrue(progress.called)
        pct_values = [c[0][0] for c in progress.call_args_list]
        self.assertIn(100, pct_values)

    def test_missing_file_raises(self):
        """Missing file should raise FileNotFoundError."""
        from opencut.core.accessibility import detect_flashing

        with self.assertRaises(FileNotFoundError):
            detect_flashing("/nonexistent/video.mp4")

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_result_keys(self, mock_info, mock_lum):
        """Result dict should have all expected top-level keys."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}
        mock_lum.return_value = [128.0] * 150

        from opencut.core.accessibility import detect_flashing

        result = detect_flashing(self.tmp.name)

        for key in ("events", "total_flashes", "risk_assessment",
                     "max_flashes_per_sec", "duration_analyzed"):
            self.assertIn(key, result)

    @patch("opencut.core.accessibility._extract_frame_luminance")
    @patch("opencut.core.accessibility.get_video_info")
    def test_luminance_threshold_controls_sensitivity(self, mock_info, mock_lum):
        """Higher min_luminance_change should detect fewer flashes."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 2.0}
        # Moderate oscillation (about 40% change)
        values = []
        for i in range(60):
            values.append(180.0 if i % 2 == 0 else 80.0)
        mock_lum.return_value = values

        from opencut.core.accessibility import detect_flashing

        result_sensitive = detect_flashing(self.tmp.name, min_luminance_change=0.1)
        result_strict = detect_flashing(self.tmp.name, min_luminance_change=0.5)

        self.assertGreaterEqual(
            result_sensitive["total_flashes"],
            result_strict["total_flashes"],
        )


# ============================================================
# FlashingResult Dataclass Tests
# ============================================================
class TestFlashingResultDataclass(unittest.TestCase):
    """Tests for FlashEvent and FlashingResult dataclasses."""

    def test_flash_event_creation(self):
        """FlashEvent should store all fields correctly."""
        from opencut.core.accessibility import FlashEvent

        event = FlashEvent(
            start=1.5, end=2.3, flash_count=5,
            peak_luminance_change=0.45, severity="high",
        )
        self.assertEqual(event.start, 1.5)
        self.assertEqual(event.end, 2.3)
        self.assertEqual(event.flash_count, 5)
        self.assertAlmostEqual(event.peak_luminance_change, 0.45)
        self.assertEqual(event.severity, "high")

    def test_flashing_result_defaults(self):
        """FlashingResult should have sensible defaults."""
        from opencut.core.accessibility import FlashingResult

        result = FlashingResult()
        self.assertEqual(result.events, [])
        self.assertEqual(result.total_flashes, 0)
        self.assertEqual(result.risk_assessment, "safe")
        self.assertEqual(result.max_flashes_per_sec, 0.0)
        self.assertEqual(result.duration_analyzed, 0.0)


# ============================================================
# Creative Routes Blueprint Tests
# ============================================================
class TestCreativeRoutes(unittest.TestCase):
    """Smoke tests for creative_routes blueprint registration and endpoints."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        cls.app = create_app(config=test_config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Fetch CSRF token
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf_token = data.get("csrf_token", "")

    def _headers(self):
        return {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    def test_blueprint_registered(self):
        """Creative blueprint should be registered with the app."""
        rules = [rule.rule for rule in self.app.url_map.iter_rules()]
        self.assertIn("/video/compare", rules)
        self.assertIn("/video/compare-frame", rules)
        self.assertIn("/effects/retro", rules)
        self.assertIn("/effects/tilt-shift", rules)
        self.assertIn("/effects/light-leak", rules)
        self.assertIn("/accessibility/colorblind-sim", rules)
        self.assertIn("/accessibility/flash-detect", rules)

    def test_compare_no_filepath_returns_400(self):
        """POST /video/compare without filepath should return 400."""
        resp = self.client.post(
            "/video/compare",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_compare_frame_no_filepath_returns_400(self):
        """POST /video/compare-frame without filepath should return 400."""
        resp = self.client.post(
            "/video/compare-frame",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_retro_no_filepath_returns_400(self):
        """POST /effects/retro without filepath should return 400."""
        resp = self.client.post(
            "/effects/retro",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_tilt_shift_no_filepath_returns_400(self):
        """POST /effects/tilt-shift without filepath should return 400."""
        resp = self.client.post(
            "/effects/tilt-shift",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_light_leak_no_filepath_returns_400(self):
        """POST /effects/light-leak without filepath should return 400."""
        resp = self.client.post(
            "/effects/light-leak",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_colorblind_sim_no_filepath_returns_400(self):
        """POST /accessibility/colorblind-sim without filepath should return 400."""
        resp = self.client.post(
            "/accessibility/colorblind-sim",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_flash_detect_no_filepath_returns_400(self):
        """POST /accessibility/flash-detect without filepath should return 400."""
        resp = self.client.post(
            "/accessibility/flash-detect",
            headers=self._headers(),
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_csrf_required(self):
        """Endpoints should reject requests without CSRF token."""
        resp = self.client.post(
            "/effects/retro",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"filepath": "/tmp/test.mp4"}),
        )
        # Should be 403 (CSRF missing)
        self.assertIn(resp.status_code, (400, 403))

    def test_compare_frame_missing_input_b_returns_400(self):
        """POST /video/compare-frame without input_b should return 400."""
        resp = self.client.post(
            "/video/compare-frame",
            headers=self._headers(),
            data=json.dumps({"filepath": "/tmp/test.mp4"}),
        )
        self.assertEqual(resp.status_code, 400)


# ============================================================
# Color Blindness Matrix Validation
# ============================================================
class TestColorBlindMatrices(unittest.TestCase):
    """Validate the color blindness simulation matrices."""

    def test_all_conditions_have_matrices(self):
        """Every valid condition should have a matrix defined."""
        from opencut.core.accessibility import _CB_MATRICES, VALID_CONDITIONS

        for condition in VALID_CONDITIONS:
            self.assertIn(condition, _CB_MATRICES)

    def test_matrix_keys_complete(self):
        """Each matrix should have all 9 channel mixer keys."""
        from opencut.core.accessibility import _CB_MATRICES

        expected_keys = {"rr", "rg", "rb", "gr", "gg", "gb", "br", "bg", "bb"}
        for condition, matrix in _CB_MATRICES.items():
            self.assertEqual(set(matrix.keys()), expected_keys,
                             f"Incomplete matrix for {condition}")

    def test_matrix_values_in_range(self):
        """All matrix values should be in [0.0, 1.0]."""
        from opencut.core.accessibility import _CB_MATRICES

        for condition, matrix in _CB_MATRICES.items():
            for key, value in matrix.items():
                self.assertGreaterEqual(value, 0.0,
                                        f"{condition}.{key} = {value} < 0")
                self.assertLessEqual(value, 1.0,
                                     f"{condition}.{key} = {value} > 1")

    def test_achromatopsia_is_grayscale(self):
        """Achromatopsia should produce identical R/G/B output (grayscale)."""
        from opencut.core.accessibility import _CB_MATRICES

        m = _CB_MATRICES["achromatopsia"]
        # All three rows should be identical
        self.assertEqual(m["rr"], m["gr"])
        self.assertEqual(m["gr"], m["br"])
        self.assertEqual(m["rg"], m["gg"])
        self.assertEqual(m["gg"], m["bg"])
        self.assertEqual(m["rb"], m["gb"])
        self.assertEqual(m["gb"], m["bb"])


if __name__ == "__main__":
    unittest.main()
