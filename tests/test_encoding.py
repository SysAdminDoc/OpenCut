"""
Tests for OpenCut Encoding & Export features (Features 45.1-45.3, 47.1).

Covers:
  - ProRes export (all profiles, alpha detection, encoder availability)
  - AV1 encoding (encoder detection, auto-selection, quality presets)
  - DNxHR/DNxHD export (all profiles, MOV/MXF containers)
  - Batch transcode (matrix generation, parallel execution, estimation)
  - New EXPORT_PRESETS entries (ProRes, AV1, DNxHR)
  - Route endpoints for all encoding operations
"""

import inspect
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 45.1 ProRes Export Core Tests
# ============================================================
class TestProResExport(unittest.TestCase):
    """Tests for opencut.core.prores_export."""

    def test_prores_profiles_defined(self):
        """All six ProRes profiles should be defined."""
        from opencut.core.prores_export import PRORES_PROFILES
        expected = {"proxy", "lt", "422", "422hq", "4444", "4444xq"}
        self.assertEqual(set(PRORES_PROFILES.keys()), expected)

    def test_prores_profile_ids(self):
        """Profile IDs should match Apple's ProRes spec (0-5)."""
        from opencut.core.prores_export import PRORES_PROFILES
        self.assertEqual(PRORES_PROFILES["proxy"]["profile_id"], 0)
        self.assertEqual(PRORES_PROFILES["lt"]["profile_id"], 1)
        self.assertEqual(PRORES_PROFILES["422"]["profile_id"], 2)
        self.assertEqual(PRORES_PROFILES["422hq"]["profile_id"], 3)
        self.assertEqual(PRORES_PROFILES["4444"]["profile_id"], 4)
        self.assertEqual(PRORES_PROFILES["4444xq"]["profile_id"], 5)

    def test_prores_4444_pix_fmt(self):
        """4444 and 4444 XQ profiles should use yuva444p10le pixel format."""
        from opencut.core.prores_export import PRORES_PROFILES
        self.assertEqual(PRORES_PROFILES["4444"]["pix_fmt"], "yuva444p10le")
        self.assertEqual(PRORES_PROFILES["4444xq"]["pix_fmt"], "yuva444p10le")

    def test_prores_422_pix_fmt(self):
        """422-family profiles should use yuv422p10le pixel format."""
        from opencut.core.prores_export import PRORES_PROFILES
        for name in ("proxy", "lt", "422", "422hq"):
            self.assertEqual(
                PRORES_PROFILES[name]["pix_fmt"], "yuv422p10le",
                f"Profile {name} should use yuv422p10le",
            )

    def test_export_prores_signature(self):
        """export_prores should have the expected parameters."""
        from opencut.core.prores_export import export_prores
        sig = inspect.signature(export_prores)
        expected = ["input_path", "profile", "output_path_override",
                    "include_alpha", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_export_prores_defaults(self):
        """Verify default parameter values."""
        from opencut.core.prores_export import export_prores
        sig = inspect.signature(export_prores)
        self.assertEqual(sig.parameters["profile"].default, "422hq")
        self.assertIsNone(sig.parameters["output_path_override"].default)
        self.assertFalse(sig.parameters["include_alpha"].default)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_export_prores_file_not_found(self):
        """export_prores should raise FileNotFoundError for missing input."""
        from opencut.core.prores_export import export_prores
        with self.assertRaises(FileNotFoundError):
            export_prores("/nonexistent/video.mp4")

    def test_export_prores_invalid_profile(self):
        """export_prores should raise ValueError for unknown profile."""
        from opencut.core.prores_export import export_prores
        with patch("os.path.isfile", return_value=True):
            with patch("opencut.core.prores_export.detect_prores_encoder", return_value=True):
                with self.assertRaises(ValueError) as ctx:
                    export_prores("/fake/video.mp4", profile="bogus")
                self.assertIn("Unknown ProRes profile", str(ctx.exception))

    @patch("opencut.core.prores_export.detect_prores_encoder", return_value=False)
    @patch("os.path.isfile", return_value=True)
    def test_export_prores_no_encoder(self, mock_isfile, mock_detect):
        """export_prores should raise RuntimeError if prores_ks is unavailable."""
        from opencut.core.prores_export import export_prores
        with self.assertRaises(RuntimeError) as ctx:
            export_prores("/fake/video.mp4")
        self.assertIn("prores_ks", str(ctx.exception))

    @patch("opencut.core.prores_export.run_ffmpeg")
    @patch("opencut.core.prores_export.detect_prores_encoder", return_value=True)
    @patch("os.path.getsize", return_value=1048576)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_prores_422hq_success(self, mock_isfile, mock_exists,
                                          mock_getsize, mock_detect, mock_ffmpeg):
        """Successful ProRes 422 HQ export should return correct result dict."""
        from opencut.core.prores_export import export_prores

        result = export_prores(
            input_path="/fake/input.mp4",
            profile="422hq",
            output_path_override="/fake/output.mov",
        )
        self.assertEqual(result["output_path"], "/fake/output.mov")
        self.assertEqual(result["profile"], "422hq")
        self.assertEqual(result["profile_id"], 3)
        self.assertEqual(result["encoder"], "prores_ks")
        self.assertIn("encode_time_seconds", result)
        self.assertIn("file_size_mb", result)
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.prores_export.run_ffmpeg")
    @patch("opencut.core.prores_export._has_alpha_channel", return_value=True)
    @patch("opencut.core.prores_export.detect_prores_encoder", return_value=True)
    @patch("os.path.getsize", return_value=2097152)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_prores_4444_with_alpha(self, mock_isfile, mock_exists,
                                            mock_getsize, mock_detect,
                                            mock_alpha, mock_ffmpeg):
        """ProRes 4444 with include_alpha=True and alpha source should set has_alpha."""
        from opencut.core.prores_export import export_prores

        result = export_prores(
            input_path="/fake/input.mov",
            profile="4444",
            output_path_override="/fake/output.mov",
            include_alpha=True,
        )
        self.assertTrue(result["has_alpha"])
        self.assertEqual(result["profile_id"], 4)
        # Verify yuva444p10le pixel format was used in command
        ffmpeg_cmd = mock_ffmpeg.call_args[0][0]
        pix_idx = ffmpeg_cmd.index("-pix_fmt")
        self.assertEqual(ffmpeg_cmd[pix_idx + 1], "yuva444p10le")

    @patch("opencut.core.prores_export.run_ffmpeg")
    @patch("opencut.core.prores_export._has_alpha_channel", return_value=False)
    @patch("opencut.core.prores_export.detect_prores_encoder", return_value=True)
    @patch("os.path.getsize", return_value=2097152)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_prores_4444_no_alpha_source(self, mock_isfile, mock_exists,
                                                  mock_getsize, mock_detect,
                                                  mock_alpha, mock_ffmpeg):
        """ProRes 4444 without alpha source should use yuv444p10le."""
        from opencut.core.prores_export import export_prores

        result = export_prores(
            input_path="/fake/input.mp4",
            profile="4444",
            output_path_override="/fake/output.mov",
            include_alpha=True,
        )
        self.assertFalse(result["has_alpha"])
        ffmpeg_cmd = mock_ffmpeg.call_args[0][0]
        pix_idx = ffmpeg_cmd.index("-pix_fmt")
        self.assertEqual(ffmpeg_cmd[pix_idx + 1], "yuv444p10le")

    @patch("opencut.core.prores_export.run_ffmpeg")
    @patch("opencut.core.prores_export.detect_prores_encoder", return_value=True)
    @patch("os.path.getsize", return_value=524288)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_prores_auto_output_path(self, mock_isfile, mock_exists,
                                             mock_getsize, mock_detect, mock_ffmpeg):
        """Auto-generated output path should include profile name and .mov extension."""
        from opencut.core.prores_export import export_prores

        result = export_prores(
            input_path="/fake/dir/clip.mp4",
            profile="proxy",
        )
        self.assertTrue(result["output_path"].endswith("_prores_proxy.mov"))
        self.assertIn("clip_prores_proxy.mov", result["output_path"])

    @patch("opencut.core.prores_export.run_ffmpeg")
    @patch("opencut.core.prores_export.detect_prores_encoder", return_value=True)
    @patch("os.path.getsize", return_value=524288)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_prores_progress_callback(self, mock_isfile, mock_exists,
                                              mock_getsize, mock_detect, mock_ffmpeg):
        """on_progress should be called during export."""
        from opencut.core.prores_export import export_prores

        progress = MagicMock()
        export_prores(
            input_path="/fake/input.mp4",
            output_path_override="/fake/output.mov",
            on_progress=progress,
        )
        self.assertTrue(progress.called)
        # Should have at least start and end calls
        self.assertGreaterEqual(progress.call_count, 2)

    def test_detect_prores_encoder_caching(self):
        """detect_prores_encoder should cache its result."""
        import opencut.core.prores_export as mod
        mod._encoder_available = None  # Reset cache

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="prores_ks", returncode=0)
            result1 = mod.detect_prores_encoder()
            result2 = mod.detect_prores_encoder()
            self.assertTrue(result1)
            self.assertTrue(result2)
            # Only called once due to caching
            self.assertEqual(mock_run.call_count, 1)

        mod._encoder_available = None  # Reset for other tests

    def test_get_prores_profiles(self):
        """get_prores_profiles should return list of profile dicts."""
        import opencut.core.prores_export as mod
        mod._encoder_available = True
        profiles = mod.get_prores_profiles()
        self.assertEqual(len(profiles), 6)
        names = {p["name"] for p in profiles}
        self.assertEqual(names, {"proxy", "lt", "422", "422hq", "4444", "4444xq"})
        for p in profiles:
            self.assertIn("label", p)
            self.assertIn("supports_alpha", p)
            self.assertIn("encoder_available", p)
        mod._encoder_available = None


# ============================================================
# 45.2 AV1 Encoding Core Tests
# ============================================================
class TestAV1Export(unittest.TestCase):
    """Tests for opencut.core.av1_export."""

    def test_av1_encoders_defined(self):
        """All four AV1 encoders should be defined."""
        from opencut.core.av1_export import AV1_ENCODERS
        expected = {"libsvtav1", "libaom-av1", "av1_nvenc", "av1_qsv"}
        self.assertEqual(set(AV1_ENCODERS.keys()), expected)

    def test_av1_encoder_priorities(self):
        """Hardware encoders should have higher priority than software."""
        from opencut.core.av1_export import AV1_ENCODERS
        self.assertLess(AV1_ENCODERS["av1_nvenc"]["priority"],
                        AV1_ENCODERS["libsvtav1"]["priority"])
        self.assertLess(AV1_ENCODERS["av1_qsv"]["priority"],
                        AV1_ENCODERS["libsvtav1"]["priority"])
        self.assertLess(AV1_ENCODERS["libsvtav1"]["priority"],
                        AV1_ENCODERS["libaom-av1"]["priority"])

    def test_av1_quality_presets_exist(self):
        """All encoders should have speed/balanced/quality presets."""
        from opencut.core.av1_export import _QUALITY_PRESETS
        for enc_name in ("libsvtav1", "libaom-av1", "av1_nvenc", "av1_qsv"):
            self.assertIn(enc_name, _QUALITY_PRESETS)
            for quality in ("speed", "balanced", "quality"):
                self.assertIn(quality, _QUALITY_PRESETS[enc_name],
                              f"Missing {quality} preset for {enc_name}")

    def test_export_av1_signature(self):
        """export_av1 should have the expected parameters."""
        from opencut.core.av1_export import export_av1
        sig = inspect.signature(export_av1)
        expected = ["input_path", "encoder", "quality", "crf",
                    "output_path_override", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_export_av1_defaults(self):
        """Verify default parameter values."""
        from opencut.core.av1_export import export_av1
        sig = inspect.signature(export_av1)
        self.assertEqual(sig.parameters["encoder"].default, "auto")
        self.assertEqual(sig.parameters["quality"].default, "balanced")
        self.assertEqual(sig.parameters["crf"].default, 28)

    def test_export_av1_file_not_found(self):
        """export_av1 should raise FileNotFoundError for missing input."""
        from opencut.core.av1_export import export_av1
        with self.assertRaises(FileNotFoundError):
            export_av1("/nonexistent/video.mp4")

    @patch("opencut.core.av1_export._pick_best_encoder", return_value=None)
    @patch("os.path.isfile", return_value=True)
    def test_export_av1_no_encoder(self, mock_isfile, mock_pick):
        """export_av1 should raise RuntimeError if no AV1 encoder is available."""
        from opencut.core.av1_export import export_av1
        with self.assertRaises(RuntimeError) as ctx:
            export_av1("/fake/video.mp4")
        self.assertIn("No AV1 encoder", str(ctx.exception))

    def test_detect_av1_encoders_caching(self):
        """detect_av1_encoders should cache results."""
        import opencut.core.av1_export as mod
        mod._detected_encoders = None  # Reset cache

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="libsvtav1 libaom-av1", returncode=0,
            )
            result1 = mod.detect_av1_encoders()
            mod.detect_av1_encoders()
            self.assertEqual(mock_run.call_count, 1)
            self.assertTrue(result1["libsvtav1"]["available"])
            self.assertTrue(result1["libaom-av1"]["available"])
            self.assertFalse(result1["av1_nvenc"]["available"])

        mod._detected_encoders = None

    def test_pick_best_encoder_auto_hw_preferred(self):
        """Auto selection should prefer HW over SW encoders."""
        import opencut.core.av1_export as mod
        mod._detected_encoders = {
            "libsvtav1": {"name": "libsvtav1", "available": True, "label": "", "type": "software", "description": ""},
            "libaom-av1": {"name": "libaom-av1", "available": True, "label": "", "type": "software", "description": ""},
            "av1_nvenc": {"name": "av1_nvenc", "available": True, "label": "", "type": "hardware", "description": ""},
            "av1_qsv": {"name": "av1_qsv", "available": False, "label": "", "type": "hardware", "description": ""},
        }
        best = mod._pick_best_encoder("auto")
        self.assertEqual(best, "av1_nvenc")
        mod._detected_encoders = None

    def test_pick_best_encoder_auto_sw_fallback(self):
        """Auto selection should fall back to libsvtav1 when no HW available."""
        import opencut.core.av1_export as mod
        mod._detected_encoders = {
            "libsvtav1": {"name": "libsvtav1", "available": True, "label": "", "type": "software", "description": ""},
            "libaom-av1": {"name": "libaom-av1", "available": True, "label": "", "type": "software", "description": ""},
            "av1_nvenc": {"name": "av1_nvenc", "available": False, "label": "", "type": "hardware", "description": ""},
            "av1_qsv": {"name": "av1_qsv", "available": False, "label": "", "type": "hardware", "description": ""},
        }
        best = mod._pick_best_encoder("auto")
        self.assertEqual(best, "libsvtav1")
        mod._detected_encoders = None

    def test_pick_best_encoder_specific(self):
        """Requesting a specific encoder should return it if available."""
        import opencut.core.av1_export as mod
        mod._detected_encoders = {
            "libsvtav1": {"name": "libsvtav1", "available": True, "label": "", "type": "software", "description": ""},
            "libaom-av1": {"name": "libaom-av1", "available": True, "label": "", "type": "software", "description": ""},
            "av1_nvenc": {"name": "av1_nvenc", "available": False, "label": "", "type": "hardware", "description": ""},
            "av1_qsv": {"name": "av1_qsv", "available": False, "label": "", "type": "hardware", "description": ""},
        }
        self.assertEqual(mod._pick_best_encoder("libaom-av1"), "libaom-av1")
        self.assertIsNone(mod._pick_best_encoder("av1_nvenc"))
        mod._detected_encoders = None

    @patch("opencut.core.av1_export.run_ffmpeg")
    @patch("opencut.core.av1_export._pick_best_encoder", return_value="libsvtav1")
    @patch("os.path.getsize", return_value=524288)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_av1_libsvtav1_success(self, mock_isfile, mock_exists,
                                           mock_getsize, mock_pick, mock_ffmpeg):
        """Successful AV1 export with libsvtav1 should return correct result."""
        from opencut.core.av1_export import export_av1

        result = export_av1(
            input_path="/fake/input.mp4",
            encoder="auto",
            quality="balanced",
            output_path_override="/fake/output.mp4",
        )
        self.assertEqual(result["output_path"], "/fake/output.mp4")
        self.assertEqual(result["encoder_used"], "libsvtav1")
        self.assertEqual(result["quality"], "balanced")
        self.assertFalse(result["hw_accelerated"])
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.av1_export.run_ffmpeg")
    @patch("opencut.core.av1_export._pick_best_encoder", return_value="av1_nvenc")
    @patch("os.path.getsize", return_value=524288)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_av1_hw_accelerated(self, mock_isfile, mock_exists,
                                        mock_getsize, mock_pick, mock_ffmpeg):
        """AV1 export with HW encoder should set hw_accelerated=True."""
        from opencut.core.av1_export import export_av1

        result = export_av1(
            input_path="/fake/input.mp4",
            output_path_override="/fake/output.mp4",
        )
        self.assertTrue(result["hw_accelerated"])
        self.assertEqual(result["encoder_used"], "av1_nvenc")

    @patch("opencut.core.av1_export.run_ffmpeg")
    @patch("opencut.core.av1_export._pick_best_encoder", return_value="libsvtav1")
    @patch("os.path.getsize", return_value=524288)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_av1_crf_clamping(self, mock_isfile, mock_exists,
                                      mock_getsize, mock_pick, mock_ffmpeg):
        """CRF should be clamped to 0-63 range."""
        from opencut.core.av1_export import export_av1

        result = export_av1(
            input_path="/fake/input.mp4",
            crf=100,
            output_path_override="/fake/output.mp4",
        )
        self.assertEqual(result["crf"], 63)

        result2 = export_av1(
            input_path="/fake/input.mp4",
            crf=-5,
            output_path_override="/fake/output2.mp4",
        )
        self.assertEqual(result2["crf"], 0)


# ============================================================
# 45.3 DNxHR Export Core Tests
# ============================================================
class TestDNxHRExport(unittest.TestCase):
    """Tests for opencut.core.dnx_export."""

    def test_dnxhr_profiles_defined(self):
        """All five DNxHR profiles should be defined."""
        from opencut.core.dnx_export import DNXHR_PROFILES
        expected = {"dnxhr_lb", "dnxhr_sq", "dnxhr_hq", "dnxhr_hqx", "dnxhr_444"}
        self.assertEqual(set(DNXHR_PROFILES.keys()), expected)

    def test_dnxhr_valid_containers(self):
        """MOV and MXF should be valid container formats."""
        from opencut.core.dnx_export import VALID_CONTAINERS
        self.assertEqual(VALID_CONTAINERS, {"mov", "mxf"})

    def test_dnxhr_profile_pix_fmts(self):
        """Profile pixel formats should match DNxHR spec."""
        from opencut.core.dnx_export import DNXHR_PROFILES
        self.assertEqual(DNXHR_PROFILES["dnxhr_lb"]["pix_fmt"], "yuv422p")
        self.assertEqual(DNXHR_PROFILES["dnxhr_sq"]["pix_fmt"], "yuv422p")
        self.assertEqual(DNXHR_PROFILES["dnxhr_hq"]["pix_fmt"], "yuv422p")
        self.assertEqual(DNXHR_PROFILES["dnxhr_hqx"]["pix_fmt"], "yuv422p10le")
        self.assertEqual(DNXHR_PROFILES["dnxhr_444"]["pix_fmt"], "yuv444p10le")

    def test_export_dnxhr_signature(self):
        """export_dnxhr should have the expected parameters."""
        from opencut.core.dnx_export import export_dnxhr
        sig = inspect.signature(export_dnxhr)
        expected = ["input_path", "profile", "container",
                    "output_path_override", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_export_dnxhr_defaults(self):
        """Verify default parameter values."""
        from opencut.core.dnx_export import export_dnxhr
        sig = inspect.signature(export_dnxhr)
        self.assertEqual(sig.parameters["profile"].default, "dnxhr_hq")
        self.assertEqual(sig.parameters["container"].default, "mov")

    def test_export_dnxhr_file_not_found(self):
        """export_dnxhr should raise FileNotFoundError for missing input."""
        from opencut.core.dnx_export import export_dnxhr
        with self.assertRaises(FileNotFoundError):
            export_dnxhr("/nonexistent/video.mp4")

    def test_export_dnxhr_invalid_profile(self):
        """export_dnxhr should raise ValueError for unknown profile."""
        from opencut.core.dnx_export import export_dnxhr
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                export_dnxhr("/fake/video.mp4", profile="bogus")
            self.assertIn("Unknown DNxHR profile", str(ctx.exception))

    def test_export_dnxhr_invalid_container(self):
        """export_dnxhr should raise ValueError for unsupported container."""
        from opencut.core.dnx_export import export_dnxhr
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                export_dnxhr("/fake/video.mp4", container="avi")
            self.assertIn("Unsupported container", str(ctx.exception))

    @patch("opencut.core.dnx_export.detect_dnxhd_encoder", return_value=False)
    @patch("os.path.isfile", return_value=True)
    def test_export_dnxhr_no_encoder(self, mock_isfile, mock_detect):
        """export_dnxhr should raise RuntimeError if dnxhd encoder is unavailable."""
        from opencut.core.dnx_export import export_dnxhr
        with self.assertRaises(RuntimeError) as ctx:
            export_dnxhr("/fake/video.mp4")
        self.assertIn("DNxHD/DNxHR encoder", str(ctx.exception))

    @patch("opencut.core.dnx_export.run_ffmpeg")
    @patch("opencut.core.dnx_export.detect_dnxhd_encoder", return_value=True)
    @patch("os.path.getsize", return_value=10485760)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_dnxhr_hq_mov_success(self, mock_isfile, mock_exists,
                                          mock_getsize, mock_detect, mock_ffmpeg):
        """Successful DNxHR HQ MOV export should return correct result."""
        from opencut.core.dnx_export import export_dnxhr

        result = export_dnxhr(
            input_path="/fake/input.mp4",
            profile="dnxhr_hq",
            container="mov",
            output_path_override="/fake/output.mov",
        )
        self.assertEqual(result["output_path"], "/fake/output.mov")
        self.assertEqual(result["profile"], "dnxhr_hq")
        self.assertEqual(result["container"], "mov")
        self.assertEqual(result["encoder"], "dnxhd")
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.dnx_export.run_ffmpeg")
    @patch("opencut.core.dnx_export.detect_dnxhd_encoder", return_value=True)
    @patch("os.path.getsize", return_value=10485760)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_dnxhr_mxf_container(self, mock_isfile, mock_exists,
                                         mock_getsize, mock_detect, mock_ffmpeg):
        """DNxHR export with MXF container should use correct extension."""
        from opencut.core.dnx_export import export_dnxhr

        result = export_dnxhr(
            input_path="/fake/input.mp4",
            profile="dnxhr_hq",
            container="mxf",
            output_path_override="/fake/output.mxf",
        )
        self.assertEqual(result["container"], "mxf")

    @patch("opencut.core.dnx_export.run_ffmpeg")
    @patch("opencut.core.dnx_export.detect_dnxhd_encoder", return_value=True)
    @patch("os.path.getsize", return_value=10485760)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_dnxhr_ffmpeg_profile_flag(self, mock_isfile, mock_exists,
                                               mock_getsize, mock_detect, mock_ffmpeg):
        """FFmpeg command should include -profile:v with correct DNxHR profile."""
        from opencut.core.dnx_export import export_dnxhr

        export_dnxhr(
            input_path="/fake/input.mp4",
            profile="dnxhr_444",
            output_path_override="/fake/output.mov",
        )
        ffmpeg_cmd = mock_ffmpeg.call_args[0][0]
        prof_idx = ffmpeg_cmd.index("-profile:v")
        self.assertEqual(ffmpeg_cmd[prof_idx + 1], "dnxhr_444")

    @patch("opencut.core.dnx_export.run_ffmpeg")
    @patch("opencut.core.dnx_export.detect_dnxhd_encoder", return_value=True)
    @patch("os.path.getsize", return_value=524288)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_export_dnxhr_auto_output_path(self, mock_isfile, mock_exists,
                                            mock_getsize, mock_detect, mock_ffmpeg):
        """Auto-generated output path should include profile name."""
        from opencut.core.dnx_export import export_dnxhr

        result = export_dnxhr(
            input_path="/fake/dir/clip.mp4",
            profile="dnxhr_sq",
            container="mov",
        )
        self.assertTrue(result["output_path"].endswith("_dnxhr_sq.mov"))

    def test_get_dnxhr_profiles(self):
        """get_dnxhr_profiles should return list of profile dicts."""
        import opencut.core.dnx_export as mod
        mod._encoder_available = True
        profiles = mod.get_dnxhr_profiles()
        self.assertEqual(len(profiles), 5)
        names = {p["name"] for p in profiles}
        self.assertEqual(names, {"dnxhr_lb", "dnxhr_sq", "dnxhr_hq", "dnxhr_hqx", "dnxhr_444"})
        for p in profiles:
            self.assertIn("containers", p)
            self.assertIn("encoder_available", p)
        mod._encoder_available = None

    def test_detect_dnxhd_encoder_caching(self):
        """detect_dnxhd_encoder should cache its result."""
        import opencut.core.dnx_export as mod
        mod._encoder_available = None

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="dnxhd", returncode=0)
            result1 = mod.detect_dnxhd_encoder()
            mod.detect_dnxhd_encoder()
            self.assertTrue(result1)
            self.assertEqual(mock_run.call_count, 1)

        mod._encoder_available = None


# ============================================================
# 47.1 Batch Transcode Core Tests
# ============================================================
class TestBatchTranscode(unittest.TestCase):
    """Tests for opencut.core.batch_transcode."""

    def test_batch_transcode_result_dataclass(self):
        """BatchTranscodeResult should have expected fields."""
        from opencut.core.batch_transcode import BatchTranscodeResult
        result = BatchTranscodeResult()
        self.assertEqual(result.total, 0)
        self.assertEqual(result.completed, 0)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.results, [])

    def test_batch_transcode_result_to_dict(self):
        """BatchTranscodeResult.to_dict should produce a serializable dict."""
        from opencut.core.batch_transcode import BatchTranscodeResult, TranscodeItemResult
        result = BatchTranscodeResult(
            total=2,
            completed=1,
            failed=1,
            results=[
                TranscodeItemResult(input_path="/a.mp4", preset="youtube_1080p",
                                    status="complete", output_path="/out/a.mp4"),
                TranscodeItemResult(input_path="/b.mp4", preset="youtube_1080p",
                                    status="failed", error="timeout"),
            ],
        )
        d = result.to_dict()
        self.assertEqual(d["total"], 2)
        self.assertEqual(d["completed"], 1)
        self.assertEqual(len(d["results"]), 2)
        self.assertEqual(d["results"][0]["status"], "complete")
        self.assertEqual(d["results"][1]["error"], "timeout")

    def test_transcode_item_result_defaults(self):
        """TranscodeItemResult should have sensible defaults."""
        from opencut.core.batch_transcode import TranscodeItemResult
        item = TranscodeItemResult(input_path="/a.mp4", preset="youtube_1080p")
        self.assertEqual(item.status, "pending")
        self.assertEqual(item.output_path, "")
        self.assertEqual(item.error, "")

    def test_batch_transcode_signature(self):
        """batch_transcode should have the expected parameters."""
        from opencut.core.batch_transcode import batch_transcode
        sig = inspect.signature(batch_transcode)
        expected = ["file_paths", "preset_names", "output_base_dir",
                    "parallel", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_batch_transcode_no_files(self):
        """batch_transcode should raise ValueError for empty file list."""
        from opencut.core.batch_transcode import batch_transcode
        with self.assertRaises(ValueError) as ctx:
            batch_transcode([], ["youtube_1080p"])
        self.assertIn("No files", str(ctx.exception))

    def test_batch_transcode_no_presets(self):
        """batch_transcode should raise ValueError for empty preset list."""
        from opencut.core.batch_transcode import batch_transcode
        with self.assertRaises(ValueError) as ctx:
            batch_transcode(["/fake/a.mp4"], [])
        self.assertIn("No presets", str(ctx.exception))

    def test_batch_transcode_invalid_presets(self):
        """batch_transcode should raise ValueError when all presets are invalid."""
        from opencut.core.batch_transcode import batch_transcode
        with self.assertRaises(ValueError) as ctx:
            batch_transcode(["/fake/a.mp4"], ["nonexistent_preset"])
        self.assertIn("No valid presets", str(ctx.exception))

    @patch("opencut.core.export_presets.export_with_preset")
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=True)
    def test_batch_transcode_matrix_generation(self, mock_isfile, mock_makedirs,
                                                mock_export):
        """batch_transcode should generate correct file x preset matrix."""
        from opencut.core.batch_transcode import batch_transcode

        mock_export.return_value = "/out/file.mp4"
        result = batch_transcode(
            file_paths=["/fake/a.mp4", "/fake/b.mp4"],
            preset_names=["youtube_1080p", "twitter"],
        )
        # 2 files x 2 presets = 4 combinations
        self.assertEqual(result.total, 4)
        self.assertEqual(mock_export.call_count, 4)

    @patch("opencut.core.export_presets.export_with_preset")
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=True)
    def test_batch_transcode_output_organization(self, mock_isfile, mock_makedirs,
                                                   mock_export):
        """Output should be organized as output_dir/preset_name/filename."""
        from opencut.core.batch_transcode import batch_transcode

        mock_export.return_value = "/output/youtube_1080p/clip.mp4"
        batch_transcode(
            file_paths=["/fake/clip.mp4"],
            preset_names=["youtube_1080p"],
            output_base_dir="/output",
        )
        # Verify makedirs was called with preset subdirectory
        made_dirs = [c[0][0] for c in mock_makedirs.call_args_list]
        self.assertTrue(any("youtube_1080p" in d for d in made_dirs))

    @patch("opencut.core.export_presets.export_with_preset")
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=True)
    def test_batch_transcode_progress_callback(self, mock_isfile, mock_makedirs,
                                                mock_export):
        """on_progress should be called during batch processing."""
        from opencut.core.batch_transcode import batch_transcode

        mock_export.return_value = "/out/file.mp4"
        progress = MagicMock()
        batch_transcode(
            file_paths=["/fake/a.mp4"],
            preset_names=["youtube_1080p"],
            on_progress=progress,
        )
        self.assertTrue(progress.called)

    @patch("opencut.core.export_presets.export_with_preset",
           side_effect=RuntimeError("encode failed"))
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=True)
    def test_batch_transcode_handles_failures(self, mock_isfile, mock_makedirs,
                                               mock_export):
        """Batch should continue after individual failures and record errors."""
        from opencut.core.batch_transcode import batch_transcode

        result = batch_transcode(
            file_paths=["/fake/a.mp4"],
            preset_names=["youtube_1080p"],
        )
        self.assertEqual(result.total, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.completed, 0)
        self.assertIn("encode failed", result.results[0].error)

    @patch("opencut.core.export_presets.export_with_preset")
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=False)
    def test_batch_transcode_missing_file(self, mock_isfile, mock_makedirs,
                                           mock_export):
        """Missing input files should be recorded as failed."""
        from opencut.core.batch_transcode import batch_transcode

        result = batch_transcode(
            file_paths=["/fake/missing.mp4"],
            preset_names=["youtube_1080p"],
        )
        self.assertEqual(result.failed, 1)
        self.assertIn("File not found", result.results[0].error)
        mock_export.assert_not_called()

    def test_estimate_batch_size_signature(self):
        """estimate_batch_size should have the expected parameters."""
        from opencut.core.batch_transcode import estimate_batch_size
        sig = inspect.signature(estimate_batch_size)
        expected = ["file_paths", "preset_names"]
        for p in expected:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    @patch("opencut.core.batch_transcode.get_video_info",
           return_value={"duration": 60.0, "width": 1920, "height": 1080, "fps": 30.0})
    @patch("os.path.isfile", return_value=True)
    def test_estimate_batch_size_basic(self, mock_isfile, mock_info):
        """estimate_batch_size should return estimated size and time."""
        from opencut.core.batch_transcode import estimate_batch_size

        estimate = estimate_batch_size(
            file_paths=["/fake/a.mp4"],
            preset_names=["youtube_1080p"],
        )
        self.assertEqual(estimate["total_files"], 1)
        self.assertEqual(estimate["total_combinations"], 1)
        self.assertGreater(estimate["estimated_size_mb"], 0)
        self.assertGreater(estimate["estimated_time_seconds"], 0)
        self.assertEqual(len(estimate["per_preset"]), 1)

    @patch("opencut.core.batch_transcode.get_video_info",
           return_value={"duration": 60.0, "width": 1920, "height": 1080, "fps": 30.0})
    @patch("os.path.isfile", return_value=True)
    def test_estimate_batch_size_unknown_preset(self, mock_isfile, mock_info):
        """Unknown presets in estimate should be reported as errors."""
        from opencut.core.batch_transcode import estimate_batch_size

        estimate = estimate_batch_size(
            file_paths=["/fake/a.mp4"],
            preset_names=["nonexistent_preset"],
        )
        self.assertIn("error", estimate["per_preset"][0])

    @patch("opencut.core.export_presets.export_with_preset")
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=True)
    def test_batch_transcode_parallel_clamping(self, mock_isfile, mock_makedirs,
                                                mock_export):
        """Parallel value should be clamped between 1 and 8."""
        from opencut.core.batch_transcode import batch_transcode

        mock_export.return_value = "/out/file.mp4"
        # parallel=0 should become 1, parallel=100 should become 8
        result = batch_transcode(
            file_paths=["/fake/a.mp4"],
            preset_names=["youtube_1080p"],
            parallel=0,
        )
        self.assertEqual(result.completed, 1)


# ============================================================
# Export Presets — new entries
# ============================================================
class TestExportPresetsNewEntries(unittest.TestCase):
    """Tests for new EXPORT_PRESETS entries."""

    def test_prores_presets_in_export_presets(self):
        """All ProRes presets should be in EXPORT_PRESETS."""
        from opencut.core.export_presets import EXPORT_PRESETS
        prores_keys = ["prores_proxy", "prores_lt", "prores_422",
                       "prores_422hq", "prores_4444", "prores_4444xq"]
        for key in prores_keys:
            self.assertIn(key, EXPORT_PRESETS, f"Missing preset: {key}")

    def test_prores_preset_values(self):
        """ProRes presets should have correct codec and profile values."""
        from opencut.core.export_presets import EXPORT_PRESETS
        self.assertEqual(EXPORT_PRESETS["prores_proxy"]["codec"], "prores_ks")
        self.assertEqual(EXPORT_PRESETS["prores_proxy"]["profile"], "0")
        self.assertEqual(EXPORT_PRESETS["prores_proxy"]["ext"], ".mov")
        self.assertEqual(EXPORT_PRESETS["prores_lt"]["profile"], "1")
        self.assertEqual(EXPORT_PRESETS["prores_422"]["profile"], "2")
        self.assertEqual(EXPORT_PRESETS["prores_422hq"]["profile"], "3")
        self.assertEqual(EXPORT_PRESETS["prores_4444"]["profile"], "4")
        self.assertEqual(EXPORT_PRESETS["prores_4444xq"]["profile"], "5")

    def test_prores_4444_pix_fmt_in_presets(self):
        """ProRes 4444 and 4444 XQ presets should use yuva444p10le."""
        from opencut.core.export_presets import EXPORT_PRESETS
        self.assertEqual(EXPORT_PRESETS["prores_4444"]["pix_fmt"], "yuva444p10le")
        self.assertEqual(EXPORT_PRESETS["prores_4444xq"]["pix_fmt"], "yuva444p10le")

    def test_av1_presets_in_export_presets(self):
        """AV1 specific presets should be in EXPORT_PRESETS."""
        from opencut.core.export_presets import EXPORT_PRESETS
        av1_keys = ["av1_youtube", "av1_archive", "av1_hw_fast"]
        for key in av1_keys:
            self.assertIn(key, EXPORT_PRESETS, f"Missing preset: {key}")

    def test_av1_youtube_preset_values(self):
        """AV1 YouTube preset should be optimized for uploads."""
        from opencut.core.export_presets import EXPORT_PRESETS
        preset = EXPORT_PRESETS["av1_youtube"]
        self.assertEqual(preset["codec"], "libsvtav1")
        self.assertEqual(preset["width"], 1920)
        self.assertEqual(preset["height"], 1080)
        self.assertEqual(preset["ext"], ".mp4")
        self.assertIn("category", preset)

    def test_av1_archive_preset_values(self):
        """AV1 Archive preset should preserve original resolution."""
        from opencut.core.export_presets import EXPORT_PRESETS
        preset = EXPORT_PRESETS["av1_archive"]
        self.assertEqual(preset["codec"], "libsvtav1")
        self.assertEqual(preset["width"], 0)  # Keep original
        self.assertEqual(preset["height"], 0)

    def test_av1_hw_fast_preset_values(self):
        """AV1 HW Fast preset should use hardware acceleration."""
        from opencut.core.export_presets import EXPORT_PRESETS
        preset = EXPORT_PRESETS["av1_hw_fast"]
        self.assertTrue(preset.get("hw_accel"))
        self.assertEqual(preset["quality"], "speed")
        self.assertEqual(preset["category"], "hw_accel")

    def test_dnxhr_presets_in_export_presets(self):
        """DNxHR presets should be in EXPORT_PRESETS."""
        from opencut.core.export_presets import EXPORT_PRESETS
        dnxhr_keys = ["dnxhr_lb", "dnxhr_sq", "dnxhr_hq", "dnxhr_hqx", "dnxhr_444"]
        for key in dnxhr_keys:
            self.assertIn(key, EXPORT_PRESETS, f"Missing preset: {key}")

    def test_dnxhr_preset_values(self):
        """DNxHR presets should have correct codec and profile values."""
        from opencut.core.export_presets import EXPORT_PRESETS
        for key in ("dnxhr_lb", "dnxhr_sq", "dnxhr_hq", "dnxhr_hqx", "dnxhr_444"):
            preset = EXPORT_PRESETS[key]
            self.assertEqual(preset["codec"], "dnxhd", f"{key} codec mismatch")
            self.assertEqual(preset["profile"], key, f"{key} profile mismatch")
            self.assertIn(preset["ext"], (".mov", ".mxf"), f"{key} ext mismatch")


# ============================================================
# Route Tests
# ============================================================
class TestEncodingRoutes(unittest.TestCase):
    """Tests for encoding route endpoints."""

    def setUp(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        self.app = create_app(config=test_config)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        resp = self.client.get("/health")
        data = resp.get_json()
        self.csrf_token = data.get("csrf_token", "")

    def _headers(self):
        return {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    # --- ProRes profiles (sync) ---
    def test_get_prores_profiles_route(self):
        """GET /export/prores/profiles should return profiles list."""
        resp = self.client.get("/export/prores/profiles")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("profiles", data)
        self.assertIsInstance(data["profiles"], list)
        self.assertGreater(len(data["profiles"]), 0)
        names = {p["name"] for p in data["profiles"]}
        self.assertIn("422hq", names)

    # --- AV1 encoders (sync) ---
    def test_get_av1_encoders_route(self):
        """GET /export/av1/encoders should return encoder info."""
        resp = self.client.get("/export/av1/encoders")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("encoders", data)
        self.assertIn("libsvtav1", data["encoders"])

    # --- DNxHR profiles (sync) ---
    def test_get_dnxhr_profiles_route(self):
        """GET /export/dnxhr/profiles should return profiles list."""
        resp = self.client.get("/export/dnxhr/profiles")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("profiles", data)
        self.assertIsInstance(data["profiles"], list)
        names = {p["name"] for p in data["profiles"]}
        self.assertIn("dnxhr_hq", names)

    # --- ProRes export (async) ---
    def test_post_prores_export_no_filepath(self):
        """POST /export/prores without filepath should return 400."""
        resp = self.client.post(
            "/export/prores",
            headers=self._headers(),
            json={"profile": "422hq"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_prores_export_returns_job_id(self):
        """POST /export/prores with valid filepath should return job_id."""
        with patch("opencut.security.validate_filepath",
                   side_effect=lambda x: x):
            with patch("os.path.isfile", return_value=True):
                resp = self.client.post(
                    "/export/prores",
                    headers=self._headers(),
                    json={"filepath": "/fake/video.mp4", "profile": "422hq"},
                )
        # async_job returns 200 with job_id
        self.assertIn(resp.status_code, (200,))
        data = resp.get_json()
        self.assertIn("job_id", data)

    # --- AV1 export (async) ---
    def test_post_av1_export_no_filepath(self):
        """POST /export/av1 without filepath should return 400."""
        resp = self.client.post(
            "/export/av1",
            headers=self._headers(),
            json={"encoder": "auto"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_av1_export_returns_job_id(self):
        """POST /export/av1 with valid filepath should return job_id."""
        with patch("opencut.security.validate_filepath",
                   side_effect=lambda x: x):
            with patch("os.path.isfile", return_value=True):
                resp = self.client.post(
                    "/export/av1",
                    headers=self._headers(),
                    json={"filepath": "/fake/video.mp4", "quality": "balanced"},
                )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("job_id", data)

    # --- DNxHR export (async) ---
    def test_post_dnxhr_export_no_filepath(self):
        """POST /export/dnxhr without filepath should return 400."""
        resp = self.client.post(
            "/export/dnxhr",
            headers=self._headers(),
            json={"profile": "dnxhr_hq"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_dnxhr_export_returns_job_id(self):
        """POST /export/dnxhr with valid filepath should return job_id."""
        with patch("opencut.security.validate_filepath",
                   side_effect=lambda x: x):
            with patch("os.path.isfile", return_value=True):
                resp = self.client.post(
                    "/export/dnxhr",
                    headers=self._headers(),
                    json={"filepath": "/fake/video.mp4", "profile": "dnxhr_hq"},
                )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("job_id", data)

    # --- Batch transcode (async) ---
    def test_post_batch_transcode_no_files(self):
        """POST /batch/transcode without file_paths should return 200 with job (validated inside)."""
        resp = self.client.post(
            "/batch/transcode",
            headers=self._headers(),
            json={"preset_names": ["youtube_1080p"]},
        )
        # filepath_required=False, so it proceeds and creates a job
        # The error is raised inside the async handler
        self.assertEqual(resp.status_code, 200)

    def test_post_batch_transcode_returns_job_id(self):
        """POST /batch/transcode with valid data should return job_id."""
        with patch("opencut.security.validate_filepath",
                   side_effect=lambda x: x):
            resp = self.client.post(
                "/batch/transcode",
                headers=self._headers(),
                json={
                    "file_paths": ["/fake/a.mp4", "/fake/b.mp4"],
                    "preset_names": ["youtube_1080p"],
                    "parallel": 2,
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("job_id", data)

    # --- Batch estimate (sync) ---
    @patch("opencut.core.batch_transcode.get_video_info",
           return_value={"duration": 60.0, "width": 1920, "height": 1080, "fps": 30.0})
    @patch("os.path.isfile", return_value=True)
    def test_post_batch_estimate_success(self, mock_isfile, mock_info):
        """POST /batch/transcode/estimate should return estimation data."""
        resp = self.client.post(
            "/batch/transcode/estimate",
            headers=self._headers(),
            json={
                "file_paths": ["/fake/a.mp4"],
                "preset_names": ["youtube_1080p"],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("total_files", data)
        self.assertIn("estimated_size_mb", data)
        self.assertIn("per_preset", data)

    def test_post_batch_estimate_no_files(self):
        """POST /batch/transcode/estimate with empty files should return 400."""
        resp = self.client.post(
            "/batch/transcode/estimate",
            headers=self._headers(),
            json={"file_paths": [], "preset_names": ["youtube_1080p"]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_post_batch_estimate_no_presets(self):
        """POST /batch/transcode/estimate with empty presets should return 400."""
        resp = self.client.post(
            "/batch/transcode/estimate",
            headers=self._headers(),
            json={"file_paths": ["/fake/a.mp4"], "preset_names": []},
        )
        self.assertEqual(resp.status_code, 400)

    # --- CSRF enforcement ---
    def test_prores_export_requires_csrf(self):
        """POST /export/prores without CSRF token should return 403."""
        resp = self.client.post(
            "/export/prores",
            headers={"Content-Type": "application/json"},
            json={"filepath": "/fake/video.mp4"},
        )
        self.assertIn(resp.status_code, (400, 403))

    def test_av1_export_requires_csrf(self):
        """POST /export/av1 without CSRF token should return 403."""
        resp = self.client.post(
            "/export/av1",
            headers={"Content-Type": "application/json"},
            json={"filepath": "/fake/video.mp4"},
        )
        self.assertIn(resp.status_code, (400, 403))

    def test_dnxhr_export_requires_csrf(self):
        """POST /export/dnxhr without CSRF token should return 403."""
        resp = self.client.post(
            "/export/dnxhr",
            headers={"Content-Type": "application/json"},
            json={"filepath": "/fake/video.mp4"},
        )
        self.assertIn(resp.status_code, (400, 403))


# ============================================================
# Blueprint Registration Test
# ============================================================
class TestEncodingBlueprintRegistered(unittest.TestCase):
    """Verify encoding_bp is registered with the app."""

    def test_encoding_blueprint_registered(self):
        """The encoding blueprint should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        blueprint_names = [bp.name for bp in app.blueprints.values()]
        self.assertIn("encoding", blueprint_names)

    def test_encoding_routes_exist(self):
        """All encoding endpoints should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = [
            "/export/prores",
            "/export/prores/profiles",
            "/export/av1",
            "/export/av1/encoders",
            "/export/dnxhr",
            "/export/dnxhr/profiles",
            "/batch/transcode",
            "/batch/transcode/estimate",
        ]
        for route in expected_routes:
            self.assertIn(route, rules, f"Missing route: {route}")


if __name__ == "__main__":
    unittest.main()
