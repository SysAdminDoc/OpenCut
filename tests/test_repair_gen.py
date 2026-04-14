"""
Tests for OpenCut Repair & AI Generation features (Sections 53-54).

Covers:
  53.1  Corrupted File Recovery (recover_moov_atom, salvage_frames)
  53.2  Adaptive Deinterlacing (auto_detect_interlacing, adaptive_deinterlace)
  53.3  Old Footage Restoration Pipeline
  53.4  SDR-to-HDR Upconversion
  53.5  Frame Rate Conversion with Optical Flow
  54.1  AI Outpainting / Frame Extension
  54.2  Image-to-Video Animation
  54.3  AI Scene Extension
  54.4  AI Video Summary / Condensed Recap
  54.5  Background Replacement with AI Generation
  Routes: repair_gen_routes.py smoke tests

90+ tests total.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Helper: create a fake video file on disk
# ============================================================
def _make_fake_file(suffix=".mp4", size=1024):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"\x00" * size)
    f.close()
    return f.name


def _make_fake_image(suffix=".png"):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    # minimal valid-ish content
    f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    f.close()
    return f.name


# ============================================================
# 53.1 — Corrupted File Recovery
# ============================================================
class TestRecoverMoovAtom(unittest.TestCase):
    """Tests for opencut.core.video_repair.recover_moov_atom."""

    def setUp(self):
        self.fake = _make_fake_file()

    def tearDown(self):
        try:
            os.unlink(self.fake)
        except OSError:
            pass

    def test_missing_file_raises(self):
        from opencut.core.video_repair import recover_moov_atom
        with self.assertRaises(FileNotFoundError):
            recover_moov_atom("/nonexistent/video.mp4")

    @patch("opencut.core.video_repair.run_ffmpeg")
    @patch("opencut.core.video_repair.get_video_info",
           return_value={"duration": 120.0, "fps": 30})
    @patch("opencut.core.video_repair._sp.run")
    def test_recover_no_reference(self, mock_sp, mock_info, mock_ff):
        from opencut.core.video_repair import recover_moov_atom
        # Create a fake output so os.path.isfile check passes
        out = self.fake.replace(".mp4", "_moov_recovered.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00" * 100)
        try:
            result = recover_moov_atom(self.fake)
            self.assertIn("output_path", result)
            self.assertIn("method", result)
            self.assertIn("success", result)
            mock_ff.assert_called()
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    @patch("opencut.core.video_repair.run_ffmpeg")
    @patch("opencut.core.video_repair.get_video_info",
           return_value={"duration": 60.0, "fps": 30})
    @patch("opencut.core.video_repair._sp.run")
    def test_recover_with_reference(self, mock_sp, mock_info, mock_ff):
        from opencut.core.video_repair import recover_moov_atom
        ref = _make_fake_file()
        out = self.fake.replace(".mp4", "_moov_recovered.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00" * 100)
        try:
            result = recover_moov_atom(self.fake, reference_path=ref)
            self.assertIn("method", result)
            self.assertTrue(result["method"].startswith("reference"))
        finally:
            for p in (ref, out):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @patch("opencut.core.video_repair.run_ffmpeg",
           side_effect=[RuntimeError("fail"), None])
    @patch("opencut.core.video_repair.get_video_info",
           return_value={"duration": 30.0, "fps": 30})
    @patch("opencut.core.video_repair._sp.run")
    def test_recover_fallback_transcode(self, mock_sp, mock_info, mock_ff):
        from opencut.core.video_repair import recover_moov_atom
        out = self.fake.replace(".mp4", "_moov_recovered.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00" * 100)
        try:
            result = recover_moov_atom(self.fake)
            self.assertEqual(result["method"], "transcode")
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    def test_progress_callback(self):
        from opencut.core.video_repair import recover_moov_atom
        progress = []
        with patch("opencut.core.video_repair.run_ffmpeg"):
            with patch("opencut.core.video_repair.get_video_info",
                       return_value={"duration": 10}):
                out = self.fake.replace(".mp4", "_moov_recovered.mp4")
                with open(out, "wb") as f:
                    f.write(b"\x00" * 100)
                try:
                    recover_moov_atom(self.fake,
                                     on_progress=lambda p, m: progress.append(p))
                    self.assertTrue(len(progress) > 0)
                    self.assertEqual(progress[-1], 100)
                finally:
                    try:
                        os.unlink(out)
                    except OSError:
                        pass


class TestSalvageFrames(unittest.TestCase):
    """Tests for opencut.core.video_repair.salvage_frames."""

    def setUp(self):
        self.fake = _make_fake_file(size=5000)

    def tearDown(self):
        try:
            os.unlink(self.fake)
        except OSError:
            pass

    def test_missing_file_raises(self):
        from opencut.core.video_repair import salvage_frames
        with self.assertRaises(FileNotFoundError):
            salvage_frames("/nonexistent/video.mp4")

    @patch("opencut.core.video_repair.run_ffmpeg")
    @patch("opencut.core.video_repair.get_video_info",
           return_value={"duration": 45.0, "fps": 30})
    def test_salvage_returns_expected_keys(self, mock_info, mock_ff):
        from opencut.core.video_repair import salvage_frames
        out = self.fake.replace(".mp4", "_salvaged.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00" * 2000)
        try:
            result = salvage_frames(self.fake)
            for key in ("output_path", "recovered_duration",
                        "estimated_original_duration",
                        "recovery_percentage", "frames_recovered"):
                self.assertIn(key, result)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    @patch("opencut.core.video_repair.run_ffmpeg",
           side_effect=[RuntimeError("copy fail"), None])
    @patch("opencut.core.video_repair.get_video_info",
           return_value={"duration": 20.0, "fps": 30})
    def test_salvage_fallback_transcode(self, mock_info, mock_ff):
        from opencut.core.video_repair import salvage_frames
        out = self.fake.replace(".mp4", "_salvaged.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00" * 2000)
        try:
            result = salvage_frames(self.fake)
            self.assertEqual(mock_ff.call_count, 2)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    @patch("opencut.core.video_repair.run_ffmpeg")
    @patch("opencut.core.video_repair.get_video_info",
           return_value={"duration": 10.0, "fps": 30})
    def test_salvage_frames_recovered_count(self, mock_info, mock_ff):
        from opencut.core.video_repair import salvage_frames
        out = self.fake.replace(".mp4", "_salvaged.mp4")
        with open(out, "wb") as f:
            f.write(b"\x00" * 2000)
        try:
            result = salvage_frames(self.fake)
            self.assertEqual(result["frames_recovered"], 300)  # 10s * 30fps
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass


# ============================================================
# 53.2 — Adaptive Deinterlacing
# ============================================================
class TestAutoDetectInterlacing(unittest.TestCase):
    """Tests for opencut.core.deinterlace.auto_detect_interlacing."""

    @patch("opencut.core.deinterlace.subprocess.run")
    @patch("opencut.core.deinterlace.detect_interlaced")
    def test_returns_detailed_info(self, mock_detect, mock_run):
        from opencut.core.deinterlace import auto_detect_interlacing, DetailedInterlaceInfo
        mock_detect.return_value = MagicMock(
            is_interlaced=True, field_order="tff", detection_confidence=0.9)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"field_order":"tt"}]}',
            stderr="Multi frame detection:  TFF:  200 BFF:    5 Progressive:   95 Undetermined:    0")

        result = auto_detect_interlacing("dummy.mp4")
        self.assertIsInstance(result, DetailedInterlaceInfo)
        self.assertTrue(result.is_interlaced)
        self.assertEqual(result.field_order, "tff")

    @patch("opencut.core.deinterlace.subprocess.run")
    @patch("opencut.core.deinterlace.detect_interlaced")
    def test_progressive_detection(self, mock_detect, mock_run):
        from opencut.core.deinterlace import auto_detect_interlacing
        mock_detect.return_value = MagicMock(
            is_interlaced=False, field_order="unknown", detection_confidence=0.8)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"field_order":"progressive"}]}',
            stderr="Multi frame detection:  TFF:    2 BFF:    0 Progressive:  498 Undetermined:    0")

        result = auto_detect_interlacing("dummy.mp4")
        self.assertFalse(result.is_interlaced)
        self.assertEqual(result.field_order, "progressive")
        self.assertGreaterEqual(result.detection_confidence, 0.9)

    @patch("opencut.core.deinterlace.subprocess.run")
    @patch("opencut.core.deinterlace.detect_interlaced")
    def test_bff_detection(self, mock_detect, mock_run):
        from opencut.core.deinterlace import auto_detect_interlacing
        mock_detect.return_value = MagicMock(
            is_interlaced=True, field_order="bff", detection_confidence=0.85)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"field_order":"bb"}]}',
            stderr="Multi frame detection:  TFF:    3 BFF:  200 Progressive:   97 Undetermined:    0")

        result = auto_detect_interlacing("dummy.mp4")
        self.assertTrue(result.is_interlaced)
        self.assertEqual(result.field_order, "bff")

    @patch("opencut.core.deinterlace.detect_interlaced")
    def test_handles_probe_failure(self, mock_detect):
        from opencut.core.deinterlace import auto_detect_interlacing
        mock_detect.return_value = MagicMock(
            is_interlaced=False, field_order="unknown", detection_confidence=0.5)

        # First call (ffprobe) raises, second call (idet) returns ok
        normal_return = MagicMock(
            returncode=0, stdout="", stderr="")
        with patch("opencut.core.deinterlace.subprocess.run",
                   side_effect=[Exception("probe failed"), normal_return]):
            result = auto_detect_interlacing("dummy.mp4")
            self.assertIsNotNone(result)

    @patch("opencut.core.deinterlace.subprocess.run")
    @patch("opencut.core.deinterlace.detect_interlaced")
    def test_recommended_method_heavy(self, mock_detect, mock_run):
        from opencut.core.deinterlace import auto_detect_interlacing
        mock_detect.return_value = MagicMock(
            is_interlaced=True, field_order="tff", detection_confidence=0.95)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"streams":[{"field_order":"tt"}]}',
            stderr="Multi frame detection:  TFF:  400 BFF:    5 Progressive:   90 Undetermined:    5")

        result = auto_detect_interlacing("dummy.mp4")
        self.assertEqual(result.recommended_method, "bwdif")


class TestAdaptiveDeinterlace(unittest.TestCase):
    """Tests for opencut.core.deinterlace.adaptive_deinterlace."""

    def setUp(self):
        self.fake = _make_fake_file()

    def tearDown(self):
        try:
            os.unlink(self.fake)
        except OSError:
            pass

    @patch("opencut.core.deinterlace.deinterlace",
           return_value={"output_path": "/tmp/out.mp4"})
    @patch("opencut.core.deinterlace.auto_detect_interlacing")
    def test_auto_progressive_skips(self, mock_detect, mock_deint):
        from opencut.core.deinterlace import adaptive_deinterlace
        mock_detect.return_value = MagicMock(
            is_interlaced=False, field_order="progressive",
            detection_confidence=0.95, recommended_method="bwdif")

        result = adaptive_deinterlace(self.fake, method="auto")
        self.assertFalse(result["was_interlaced"])
        self.assertEqual(result["method_used"], "none")
        mock_deint.assert_not_called()

    @patch("opencut.core.deinterlace.deinterlace",
           return_value={"output_path": "/tmp/out.mp4"})
    @patch("opencut.core.deinterlace.auto_detect_interlacing")
    def test_auto_interlaced_deinterlaces(self, mock_detect, mock_deint):
        from opencut.core.deinterlace import adaptive_deinterlace
        mock_detect.return_value = MagicMock(
            is_interlaced=True, field_order="tff",
            detection_confidence=0.9, recommended_method="bwdif")

        result = adaptive_deinterlace(self.fake, method="auto")
        self.assertTrue(result["was_interlaced"])
        mock_deint.assert_called_once()

    @patch("opencut.core.deinterlace.deinterlace",
           return_value={"output_path": "/tmp/out.mp4"})
    @patch("opencut.core.deinterlace.auto_detect_interlacing")
    def test_explicit_method(self, mock_detect, mock_deint):
        from opencut.core.deinterlace import adaptive_deinterlace
        mock_detect.return_value = MagicMock(
            is_interlaced=True, field_order="tff",
            detection_confidence=0.8, recommended_method="bwdif")

        result = adaptive_deinterlace(self.fake, method="yadif")
        self.assertEqual(result["method_used"], "yadif")

    @patch("opencut.core.deinterlace.deinterlace",
           return_value={"output_path": "/tmp/out.mp4"})
    @patch("opencut.core.deinterlace.auto_detect_interlacing")
    def test_invalid_method_defaults(self, mock_detect, mock_deint):
        from opencut.core.deinterlace import adaptive_deinterlace
        mock_detect.return_value = MagicMock(
            is_interlaced=False, field_order="progressive",
            detection_confidence=0.9, recommended_method="bwdif")

        result = adaptive_deinterlace(self.fake, method="invalid_method")
        self.assertEqual(result["method_used"], "none")

    @patch("opencut.core.deinterlace.deinterlace",
           return_value={"output_path": "/tmp/out.mp4"})
    @patch("opencut.core.deinterlace.auto_detect_interlacing")
    def test_result_keys(self, mock_detect, mock_deint):
        from opencut.core.deinterlace import adaptive_deinterlace
        mock_detect.return_value = MagicMock(
            is_interlaced=True, field_order="tff",
            detection_confidence=0.9, recommended_method="bwdif")

        result = adaptive_deinterlace(self.fake, method="auto")
        for key in ("output_path", "method_used", "field_order",
                    "was_interlaced", "detection_confidence"):
            self.assertIn(key, result)


# ============================================================
# 53.3 — Old Footage Restoration Pipeline
# ============================================================
class TestOldRestoration(unittest.TestCase):
    """Tests for opencut.core.old_restoration."""

    def test_list_presets(self):
        from opencut.core.old_restoration import list_presets
        presets = list_presets()
        self.assertIsInstance(presets, list)
        self.assertGreaterEqual(len(presets), 3)
        names = [p["name"] for p in presets]
        self.assertIn("VHS", names)
        self.assertIn("8mm Film", names)
        self.assertIn("Early Digital", names)

    def test_preset_has_description(self):
        from opencut.core.old_restoration import RESTORATION_PRESETS
        for name, cfg in RESTORATION_PRESETS.items():
            self.assertIn("description", cfg, f"Preset {name} missing description")
            self.assertTrue(len(cfg["description"]) > 5)

    def test_missing_file_raises(self):
        from opencut.core.old_restoration import restore_old_footage
        with self.assertRaises(FileNotFoundError):
            restore_old_footage("/nonexistent/video.mp4")

    def test_invalid_preset_raises(self):
        from opencut.core.old_restoration import restore_old_footage
        fake = _make_fake_file()
        try:
            with self.assertRaises(ValueError):
                restore_old_footage(fake, preset="BETAMAX")
        finally:
            os.unlink(fake)

    @patch("opencut.core.old_restoration.run_ffmpeg")
    @patch("opencut.core.old_restoration.get_video_info",
           return_value={"width": 640, "height": 480, "fps": 29.97, "duration": 60})
    def test_vhs_preset_stages(self, mock_info, mock_ff):
        from opencut.core.old_restoration import restore_old_footage
        fake = _make_fake_file()
        try:
            result = restore_old_footage(fake, preset="VHS")
            self.assertEqual(result["preset"], "VHS")
            self.assertIn("stages_applied", result)
            self.assertIsInstance(result["stages_applied"], list)
        finally:
            os.unlink(fake)

    @patch("opencut.core.old_restoration.run_ffmpeg")
    @patch("opencut.core.old_restoration.get_video_info",
           return_value={"width": 720, "height": 480, "fps": 18, "duration": 120})
    def test_8mm_preset(self, mock_info, mock_ff):
        from opencut.core.old_restoration import restore_old_footage
        fake = _make_fake_file()
        try:
            result = restore_old_footage(fake, preset="8mm Film")
            self.assertEqual(result["preset"], "8mm Film")
        finally:
            os.unlink(fake)

    @patch("opencut.core.old_restoration.run_ffmpeg")
    @patch("opencut.core.old_restoration.get_video_info",
           return_value={"width": 320, "height": 240, "fps": 15, "duration": 30})
    def test_early_digital_preset(self, mock_info, mock_ff):
        from opencut.core.old_restoration import restore_old_footage
        fake = _make_fake_file()
        try:
            result = restore_old_footage(fake, preset="Early Digital")
            self.assertEqual(result["preset"], "Early Digital")
        finally:
            os.unlink(fake)

    @patch("opencut.core.old_restoration.run_ffmpeg")
    @patch("opencut.core.old_restoration.get_video_info",
           return_value={"width": 640, "height": 480, "fps": 30, "duration": 60})
    def test_result_keys(self, mock_info, mock_ff):
        from opencut.core.old_restoration import restore_old_footage
        fake = _make_fake_file()
        try:
            result = restore_old_footage(fake, preset="VHS")
            for key in ("output_path", "preset", "stages_applied",
                        "original_resolution", "output_resolution",
                        "original_fps", "output_fps", "duration"):
                self.assertIn(key, result)
        finally:
            os.unlink(fake)

    @patch("opencut.core.old_restoration.run_ffmpeg")
    @patch("opencut.core.old_restoration.get_video_info",
           return_value={"width": 640, "height": 480, "fps": 30, "duration": 60})
    def test_progress_callback(self, mock_info, mock_ff):
        from opencut.core.old_restoration import restore_old_footage
        fake = _make_fake_file()
        progress = []
        try:
            restore_old_footage(fake, preset="VHS",
                               on_progress=lambda p, m: progress.append(p))
            self.assertTrue(len(progress) > 0)
            self.assertEqual(progress[-1], 100)
        finally:
            os.unlink(fake)


# ============================================================
# 53.4 — SDR-to-HDR Upconversion
# ============================================================
class TestSdrToHdr(unittest.TestCase):
    """Tests for opencut.core.sdr_to_hdr."""

    def test_missing_file_raises(self):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        with self.assertRaises(FileNotFoundError):
            sdr_to_hdr("/nonexistent/video.mp4")

    def test_invalid_transfer_raises(self):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        fake = _make_fake_file()
        try:
            with self.assertRaises(ValueError):
                sdr_to_hdr(fake, transfer="invalid")
        finally:
            os.unlink(fake)

    @patch("opencut.core.sdr_to_hdr.run_ffmpeg")
    @patch("opencut.core.sdr_to_hdr.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30})
    def test_pq_conversion(self, mock_info, mock_ff):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        fake = _make_fake_file()
        try:
            result = sdr_to_hdr(fake, transfer="pq")
            self.assertEqual(result["transfer_function"], "pq")
            self.assertEqual(result["color_primaries"], "bt2020")
            self.assertTrue(result["has_metadata"])
        finally:
            os.unlink(fake)

    @patch("opencut.core.sdr_to_hdr.run_ffmpeg")
    @patch("opencut.core.sdr_to_hdr.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30})
    def test_hlg_conversion(self, mock_info, mock_ff):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        fake = _make_fake_file()
        try:
            result = sdr_to_hdr(fake, transfer="hlg")
            self.assertEqual(result["transfer_function"], "hlg")
        finally:
            os.unlink(fake)

    @patch("opencut.core.sdr_to_hdr.run_ffmpeg")
    @patch("opencut.core.sdr_to_hdr.get_video_info",
           return_value={"width": 3840, "height": 2160, "fps": 60})
    def test_custom_luminance(self, mock_info, mock_ff):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        fake = _make_fake_file()
        try:
            result = sdr_to_hdr(fake, max_luminance=4000, min_luminance=0.001)
            self.assertEqual(result["max_luminance"], 4000)
            self.assertEqual(result["min_luminance"], 0.001)
        finally:
            os.unlink(fake)

    def test_list_transfer_functions(self):
        from opencut.core.sdr_to_hdr import list_transfer_functions
        tfs = list_transfer_functions()
        self.assertIsInstance(tfs, list)
        self.assertEqual(len(tfs), 2)
        ids = [t["id"] for t in tfs]
        self.assertIn("pq", ids)
        self.assertIn("hlg", ids)

    @patch("opencut.core.sdr_to_hdr.run_ffmpeg")
    @patch("opencut.core.sdr_to_hdr.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30})
    def test_result_keys(self, mock_info, mock_ff):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        fake = _make_fake_file()
        try:
            result = sdr_to_hdr(fake)
            for key in ("output_path", "transfer_function",
                        "color_primaries", "max_luminance",
                        "min_luminance", "has_metadata"):
                self.assertIn(key, result)
        finally:
            os.unlink(fake)

    @patch("opencut.core.sdr_to_hdr.run_ffmpeg")
    @patch("opencut.core.sdr_to_hdr.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30})
    def test_ffmpeg_gets_x265_params(self, mock_info, mock_ff):
        from opencut.core.sdr_to_hdr import sdr_to_hdr
        fake = _make_fake_file()
        try:
            sdr_to_hdr(fake, transfer="pq")
            cmd = mock_ff.call_args[0][0]
            cmd_str = " ".join(cmd)
            self.assertIn("libx265", cmd_str)
            self.assertIn("master-display", cmd_str)
            self.assertIn("max-cll", cmd_str)
        finally:
            os.unlink(fake)


# ============================================================
# 53.5 — Frame Rate Conversion
# ============================================================
class TestFrameRateConversion(unittest.TestCase):
    """Tests for opencut.core.framerate_convert."""

    def test_missing_file_raises(self):
        from opencut.core.framerate_convert import convert_framerate
        with self.assertRaises(FileNotFoundError):
            convert_framerate("/nonexistent/video.mp4")

    def test_invalid_preset_raises(self):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            with self.assertRaises(ValueError):
                convert_framerate(fake, preset="turbo")
        finally:
            os.unlink(fake)

    @patch("opencut.core.framerate_convert.get_video_info",
           return_value={"fps": 30.0, "duration": 60.0})
    def test_invalid_fps_raises(self, mock_info):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            with self.assertRaises(ValueError):
                convert_framerate(fake, target_fps=-1)
        finally:
            os.unlink(fake)

    @patch("opencut.core.framerate_convert.run_ffmpeg")
    @patch("opencut.core.framerate_convert.get_video_info",
           return_value={"fps": 30.0, "duration": 60.0, "width": 1920, "height": 1080})
    def test_smooth_preset(self, mock_info, mock_ff):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            result = convert_framerate(fake, preset="smooth")
            self.assertEqual(result["preset"], "smooth")
            self.assertEqual(result["target_fps"], 60.0)
        finally:
            os.unlink(fake)

    @patch("opencut.core.framerate_convert.run_ffmpeg")
    @patch("opencut.core.framerate_convert.get_video_info",
           return_value={"fps": 29.97, "duration": 120.0, "width": 1920, "height": 1080})
    def test_cinematic_telecine(self, mock_info, mock_ff):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            result = convert_framerate(fake, preset="cinematic")
            self.assertEqual(result["method"], "inverse_telecine")
        finally:
            os.unlink(fake)

    @patch("opencut.core.framerate_convert.run_ffmpeg")
    @patch("opencut.core.framerate_convert.get_video_info",
           return_value={"fps": 30.0, "duration": 60.0, "width": 1920, "height": 1080})
    def test_sport_preset(self, mock_info, mock_ff):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            result = convert_framerate(fake, preset="sport")
            self.assertEqual(result["target_fps"], 120.0)
        finally:
            os.unlink(fake)

    @patch("opencut.core.framerate_convert.run_ffmpeg")
    @patch("opencut.core.framerate_convert.get_video_info",
           return_value={"fps": 60.0, "duration": 60.0, "width": 1920, "height": 1080})
    def test_downsample(self, mock_info, mock_ff):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            result = convert_framerate(fake, target_fps=24)
            self.assertEqual(result["method"], "fps_filter")
        finally:
            os.unlink(fake)

    def test_list_presets(self):
        from opencut.core.framerate_convert import list_presets
        presets = list_presets()
        self.assertIsInstance(presets, list)
        self.assertEqual(len(presets), 3)
        names = [p["name"] for p in presets]
        self.assertIn("smooth", names)
        self.assertIn("cinematic", names)
        self.assertIn("sport", names)

    @patch("opencut.core.framerate_convert.run_ffmpeg")
    @patch("opencut.core.framerate_convert.get_video_info",
           return_value={"fps": 30.0, "duration": 60.0, "width": 1920, "height": 1080})
    def test_result_keys(self, mock_info, mock_ff):
        from opencut.core.framerate_convert import convert_framerate
        fake = _make_fake_file()
        try:
            result = convert_framerate(fake)
            for key in ("output_path", "original_fps", "target_fps",
                        "preset", "method", "duration"):
                self.assertIn(key, result)
        finally:
            os.unlink(fake)


# ============================================================
# 54.1 — AI Outpainting / Frame Extension
# ============================================================
class TestOutpaintAspectRatio(unittest.TestCase):
    """Tests for opencut.core.frame_extension.outpaint_aspect_ratio."""

    @patch("opencut.core.frame_extension.ensure_package", return_value=True)
    @patch("opencut.core.frame_extension.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30})
    def test_same_ratio_passthrough(self, mock_info, mock_pkg):
        from opencut.core.frame_extension import outpaint_aspect_ratio
        fake = _make_fake_file()
        try:
            result = outpaint_aspect_ratio(fake, target_ratio="16:9")
            self.assertEqual(result["fill_method"], "none")
            self.assertEqual(result["frames_processed"], 0)
        finally:
            os.unlink(fake)

    def test_missing_file_raises(self):
        from opencut.core.frame_extension import outpaint_aspect_ratio
        with patch("opencut.core.frame_extension.ensure_package", return_value=True):
            with self.assertRaises(FileNotFoundError):
                outpaint_aspect_ratio("/nonexistent/video.mp4")

    @patch("opencut.core.frame_extension.run_ffmpeg")
    @patch("opencut.core.frame_extension.ensure_package", return_value=True)
    @patch("opencut.core.frame_extension.get_video_info",
           return_value={"width": 640, "height": 480, "fps": 30})
    def test_wider_ratio_extends(self, mock_info, mock_pkg, mock_ff):
        from opencut.core.frame_extension import outpaint_aspect_ratio
        import numpy as np

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30, 7: 10}.get(x, 0)
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8))
        ] + [(False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0

        fake = _make_fake_file()
        try:
            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                # Mock _extend_frame to avoid deep cv2 calls
                with patch("opencut.core.frame_extension._extend_frame",
                           return_value=np.zeros((480, 854, 3), dtype=np.uint8)):
                    result = outpaint_aspect_ratio(fake, target_ratio="16:9",
                                                  fill_method="reflect")
                    self.assertIn("output_path", result)
                    self.assertEqual(result["frames_processed"], 1)
        finally:
            os.unlink(fake)

    @patch("opencut.core.frame_extension.ensure_package", return_value=True)
    @patch("opencut.core.frame_extension.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30})
    def test_result_keys(self, mock_info, mock_pkg):
        from opencut.core.frame_extension import outpaint_aspect_ratio
        fake = _make_fake_file()
        try:
            result = outpaint_aspect_ratio(fake, target_ratio="16:9")
            for key in ("output_path", "original_size", "output_size",
                        "frames_processed", "fill_method", "ai_enhanced"):
                self.assertIn(key, result)
        finally:
            os.unlink(fake)


# ============================================================
# 54.2 — Image-to-Video Animation
# ============================================================
class TestImgToVideo(unittest.TestCase):
    """Tests for opencut.core.img_to_video."""

    def test_missing_image_raises(self):
        from opencut.core.img_to_video import animate_image
        with self.assertRaises(FileNotFoundError):
            animate_image("/nonexistent/image.png")

    @patch("opencut.core.img_to_video.run_ffmpeg")
    def test_ken_burns_default(self, mock_ff):
        from opencut.core.img_to_video import animate_image
        fake = _make_fake_image()
        try:
            result = animate_image(fake, method="ken_burns", duration=3)
            self.assertEqual(result["method"], "ken_burns")
            self.assertEqual(result["duration"], 3)
            mock_ff.assert_called_once()
        finally:
            os.unlink(fake)

    def test_list_motion_presets(self):
        from opencut.core.img_to_video import list_motion_presets
        presets = list_motion_presets()
        self.assertIsInstance(presets, list)
        self.assertGreaterEqual(len(presets), 7)
        names = [p["name"] for p in presets]
        self.assertIn("zoom_in", names)
        self.assertIn("pan_left", names)
        self.assertIn("parallax", names)

    @patch("opencut.core.img_to_video.run_ffmpeg")
    def test_invalid_preset_defaults(self, mock_ff):
        from opencut.core.img_to_video import animate_image
        fake = _make_fake_image()
        try:
            result = animate_image(fake, motion_preset="nonexistent",
                                  method="ken_burns")
            self.assertEqual(result["motion_preset"], "zoom_in")
        finally:
            os.unlink(fake)

    @patch("opencut.core.img_to_video.run_ffmpeg")
    def test_custom_dimensions(self, mock_ff):
        from opencut.core.img_to_video import animate_image
        fake = _make_fake_image()
        try:
            result = animate_image(fake, width=3840, height=2160,
                                  method="ken_burns")
            self.assertEqual(result["width"], 3840)
            self.assertEqual(result["height"], 2160)
        finally:
            os.unlink(fake)

    @patch("opencut.core.img_to_video.run_ffmpeg")
    def test_result_keys(self, mock_ff):
        from opencut.core.img_to_video import animate_image
        fake = _make_fake_image()
        try:
            result = animate_image(fake, method="ken_burns")
            for key in ("output_path", "duration", "fps",
                        "motion_preset", "method", "width", "height"):
                self.assertIn(key, result)
        finally:
            os.unlink(fake)

    @patch("opencut.core.img_to_video.run_ffmpeg")
    def test_output_is_mp4(self, mock_ff):
        from opencut.core.img_to_video import animate_image
        fake = _make_fake_image()
        try:
            result = animate_image(fake, method="ken_burns")
            self.assertTrue(result["output_path"].endswith(".mp4"))
        finally:
            os.unlink(fake)

    @patch("opencut.core.img_to_video.ensure_package", return_value=False)
    @patch("opencut.core.img_to_video.run_ffmpeg")
    def test_ai_fallback_to_ken_burns(self, mock_ff, mock_pkg):
        from opencut.core.img_to_video import animate_image
        fake = _make_fake_image()
        try:
            result = animate_image(fake, method="auto")
            self.assertEqual(result["method"], "ken_burns")
        finally:
            os.unlink(fake)


# ============================================================
# 54.3 — AI Scene Extension
# ============================================================
class TestSceneExtend(unittest.TestCase):
    """Tests for opencut.core.scene_extend."""

    def test_missing_file_raises(self):
        from opencut.core.scene_extend import extend_scene
        with self.assertRaises(FileNotFoundError):
            extend_scene("/nonexistent/video.mp4")

    def test_negative_seconds_raises(self):
        from opencut.core.scene_extend import extend_scene
        fake = _make_fake_file()
        try:
            with self.assertRaises(ValueError):
                extend_scene(fake, extra_seconds=-1)
        finally:
            os.unlink(fake)

    @patch("opencut.core.scene_extend.run_ffmpeg")
    @patch("opencut.core.scene_extend.ensure_package", return_value=True)
    @patch("opencut.core.scene_extend.get_video_info",
           return_value={"duration": 10.0, "fps": 30, "width": 640, "height": 480})
    def test_optical_flow_method(self, mock_info, mock_pkg, mock_ff):
        from opencut.core.scene_extend import extend_scene
        with patch("opencut.core.scene_extend._extrapolate_optical_flow",
                   return_value={"frames_generated": 90, "blend_frames": 10}):
            fake = _make_fake_file()
            try:
                result = extend_scene(fake, extra_seconds=3.0, method="optical_flow")
                self.assertEqual(result["extra_seconds"], 3.0)
                self.assertEqual(result["method"], "optical_flow")
            finally:
                os.unlink(fake)

    @patch("opencut.core.scene_extend.run_ffmpeg")
    @patch("opencut.core.scene_extend.ensure_package", return_value=True)
    @patch("opencut.core.scene_extend.get_video_info",
           return_value={"duration": 10.0, "fps": 30, "width": 640, "height": 480})
    def test_result_keys(self, mock_info, mock_pkg, mock_ff):
        from opencut.core.scene_extend import extend_scene
        with patch("opencut.core.scene_extend._extrapolate_optical_flow",
                   return_value={"frames_generated": 90, "blend_frames": 10}):
            fake = _make_fake_file()
            try:
                result = extend_scene(fake, extra_seconds=3.0)
                for key in ("output_path", "original_duration",
                            "extended_duration", "extra_seconds",
                            "frames_generated", "method", "blend_frames"):
                    self.assertIn(key, result)
            finally:
                os.unlink(fake)


# ============================================================
# 54.4 — AI Video Summary / Condensed Recap
# ============================================================
class TestVideoCondensed(unittest.TestCase):
    """Tests for opencut.core.video_condensed."""

    def test_missing_file_raises(self):
        from opencut.core.video_condensed import condense_video
        with self.assertRaises(FileNotFoundError):
            condense_video("/nonexistent/video.mp4")

    @patch("opencut.core.video_condensed.get_video_info",
           return_value={"duration": 30.0, "fps": 30})
    def test_short_video_passthrough(self, mock_info):
        from opencut.core.video_condensed import condense_video
        fake = _make_fake_file()
        try:
            result = condense_video(fake, max_duration=60)
            self.assertEqual(result["method"], "passthrough")
            self.assertEqual(result["output_path"], fake)
        finally:
            os.unlink(fake)

    @patch("opencut.core.video_condensed.run_ffmpeg")
    @patch("opencut.core.video_condensed._detect_scenes",
           return_value=[
               {"start": 0, "end": 5, "duration": 5},
               {"start": 5, "end": 15, "duration": 10},
               {"start": 15, "end": 25, "duration": 10},
               {"start": 25, "end": 40, "duration": 15},
               {"start": 40, "end": 55, "duration": 15},
               {"start": 55, "end": 60, "duration": 5},
           ])
    @patch("opencut.core.video_condensed.get_video_info",
           return_value={"duration": 300.0, "fps": 30})
    def test_condense_long_video(self, mock_info, mock_scenes, mock_ff):
        from opencut.core.video_condensed import condense_video
        fake = _make_fake_file()
        try:
            result = condense_video(fake, target_duration=45)
            self.assertIn("output_path", result)
            self.assertGreater(result["shots_selected"], 0)
            self.assertEqual(result["method"], "scene_score")
        finally:
            os.unlink(fake)

    def test_score_scenes_favors_intro(self):
        from opencut.core.video_condensed import _score_scenes
        scenes = [
            {"start": 0, "end": 5, "duration": 5},
            {"start": 50, "end": 55, "duration": 5},
            {"start": 95, "end": 100, "duration": 5},
        ]
        scored = _score_scenes(scenes, 100)
        # First scene should have highest score
        self.assertGreater(scored[0].score, scored[1].score)

    def test_score_scenes_favors_closing(self):
        from opencut.core.video_condensed import _score_scenes
        scenes = [
            {"start": 0, "end": 5, "duration": 5},
            {"start": 50, "end": 55, "duration": 5},
            {"start": 95, "end": 100, "duration": 5},
        ]
        scored = _score_scenes(scenes, 100)
        # Last scene should score higher than middle
        self.assertGreater(scored[2].score, scored[1].score)

    @patch("opencut.core.video_condensed.run_ffmpeg")
    @patch("opencut.core.video_condensed._detect_scenes",
           return_value=[
               {"start": 0, "end": 10, "duration": 10},
               {"start": 10, "end": 20, "duration": 10},
           ])
    @patch("opencut.core.video_condensed.get_video_info",
           return_value={"duration": 600.0, "fps": 30})
    def test_result_keys(self, mock_info, mock_scenes, mock_ff):
        from opencut.core.video_condensed import condense_video
        fake = _make_fake_file()
        try:
            result = condense_video(fake)
            for key in ("output_path", "original_duration",
                        "condensed_duration", "compression_ratio",
                        "shots_selected", "total_shots", "method"):
                self.assertIn(key, result)
        finally:
            os.unlink(fake)


# ============================================================
# 54.5 — Background Replacement
# ============================================================
class TestBGReplaceAI(unittest.TestCase):
    """Tests for opencut.core.bg_replace_ai."""

    def test_missing_file_raises(self):
        from opencut.core.bg_replace_ai import replace_background
        with patch("opencut.core.bg_replace_ai.ensure_package", return_value=True):
            with self.assertRaises(FileNotFoundError):
                replace_background("/nonexistent/video.mp4")

    def test_bg_presets_defined(self):
        from opencut.core.bg_replace_ai import BG_PRESETS
        self.assertIn("solid", BG_PRESETS)
        self.assertIn("blur", BG_PRESETS)
        self.assertIn("gradient", BG_PRESETS)
        self.assertIn("image", BG_PRESETS)
        self.assertIn("video", BG_PRESETS)
        self.assertIn("ai_generated", BG_PRESETS)

    @patch("opencut.core.bg_replace_ai.run_ffmpeg")
    @patch("opencut.core.bg_replace_ai.ensure_package", return_value=True)
    @patch("opencut.core.bg_replace_ai.get_video_info",
           return_value={"width": 640, "height": 480, "fps": 30})
    def test_blur_bg_type(self, mock_info, mock_pkg, mock_ff):
        import numpy as np

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30, 7: 5}.get(x, 0)
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8))
        ] + [(False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0
        mock_cv2.GaussianBlur.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_cv2.inRange.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_cv2.bitwise_not.return_value = np.ones((480, 640), dtype=np.uint8) * 255
        mock_cv2.getStructuringElement.return_value = np.ones((5, 5), dtype=np.uint8)
        mock_cv2.morphologyEx.return_value = np.ones((480, 640), dtype=np.uint8) * 255
        mock_cv2.COLOR_BGR2HSV = 40
        mock_cv2.COLOR_BGR2BGRA = 2
        mock_cv2.MORPH_CLOSE = 3
        mock_cv2.MORPH_OPEN = 2
        mock_cv2.MORPH_ELLIPSE = 2

        fake = _make_fake_file()
        try:
            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                with patch("opencut.core.bg_replace_ai._remove_bg_chroma",
                           return_value=np.zeros((480, 640, 4), dtype=np.uint8)):
                    with patch("opencut.core.bg_replace_ai._composite_rgba_on_bg",
                               return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
                        from opencut.core.bg_replace_ai import replace_background
                        result = replace_background(fake, bg_type="blur",
                                                   removal_method="chroma")
                        self.assertEqual(result["bg_type"], "blur")
        finally:
            os.unlink(fake)

    @patch("opencut.core.bg_replace_ai.run_ffmpeg")
    @patch("opencut.core.bg_replace_ai.ensure_package", return_value=True)
    @patch("opencut.core.bg_replace_ai.get_video_info",
           return_value={"width": 640, "height": 480, "fps": 30})
    def test_result_keys(self, mock_info, mock_pkg, mock_ff):
        import numpy as np

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30, 7: 5}.get(x, 0)
        mock_cap.read.side_effect = [(False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0

        fake = _make_fake_file()
        try:
            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                from opencut.core.bg_replace_ai import replace_background
                result = replace_background(fake, bg_type="solid",
                                           removal_method="chroma")
                for key in ("output_path", "bg_type", "prompt",
                            "frames_processed", "method", "ai_generated"):
                    self.assertIn(key, result)
        finally:
            os.unlink(fake)

    def test_invalid_bg_type_defaults(self):
        import numpy as np

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30, 7: 1}.get(x, 0)
        mock_cap.read.side_effect = [(False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 0

        fake = _make_fake_file()
        try:
            with patch.dict("sys.modules", {"cv2": mock_cv2}):
                with patch("opencut.core.bg_replace_ai.ensure_package",
                           return_value=True):
                    with patch("opencut.core.bg_replace_ai.get_video_info",
                               return_value={"width": 640, "height": 480, "fps": 30}):
                        with patch("opencut.core.bg_replace_ai.run_ffmpeg"):
                            from opencut.core.bg_replace_ai import replace_background
                            result = replace_background(fake, bg_type="fantasy",
                                                       removal_method="chroma")
                            self.assertEqual(result["bg_type"], "blur")
        finally:
            os.unlink(fake)


# ============================================================
# Routes smoke tests
# ============================================================
class TestRepairGenRoutes(unittest.TestCase):
    """Smoke tests for repair_gen_routes blueprint registration."""

    def test_blueprint_exists(self):
        from opencut.routes.repair_gen_routes import repair_gen_bp
        self.assertEqual(repair_gen_bp.name, "repair_gen")

    def test_repair_recover_registered(self):
        from opencut.routes.repair_gen_routes import repair_recover
        self.assertTrue(callable(repair_recover))

    def test_repair_deinterlace_registered(self):
        from opencut.routes.repair_gen_routes import repair_deinterlace
        self.assertTrue(callable(repair_deinterlace))

    def test_repair_restore_registered(self):
        from opencut.routes.repair_gen_routes import repair_restore
        self.assertTrue(callable(repair_restore))

    def test_repair_sdr_to_hdr_registered(self):
        from opencut.routes.repair_gen_routes import repair_sdr_to_hdr
        self.assertTrue(callable(repair_sdr_to_hdr))

    def test_repair_framerate_registered(self):
        from opencut.routes.repair_gen_routes import repair_framerate
        self.assertTrue(callable(repair_framerate))

    def test_aigen_outpaint_registered(self):
        from opencut.routes.repair_gen_routes import aigen_outpaint
        self.assertTrue(callable(aigen_outpaint))

    def test_aigen_img_to_video_registered(self):
        from opencut.routes.repair_gen_routes import aigen_img_to_video
        self.assertTrue(callable(aigen_img_to_video))

    def test_aigen_extend_scene_registered(self):
        from opencut.routes.repair_gen_routes import aigen_extend_scene
        self.assertTrue(callable(aigen_extend_scene))

    def test_aigen_summarize_registered(self):
        from opencut.routes.repair_gen_routes import aigen_summarize
        self.assertTrue(callable(aigen_summarize))

    def test_aigen_replace_bg_registered(self):
        from opencut.routes.repair_gen_routes import aigen_replace_bg
        self.assertTrue(callable(aigen_replace_bg))


# ============================================================
# Cross-module integration tests
# ============================================================
class TestModuleImports(unittest.TestCase):
    """Verify all new modules import cleanly."""

    def test_import_video_repair(self):
        from opencut.core.video_repair import (
            recover_moov_atom, salvage_frames,
            diagnose_corruption, repair_video,
        )
        self.assertTrue(callable(recover_moov_atom))
        self.assertTrue(callable(salvage_frames))

    def test_import_deinterlace(self):
        from opencut.core.deinterlace import (
            auto_detect_interlacing, adaptive_deinterlace,
            detect_interlaced, deinterlace,
        )
        self.assertTrue(callable(auto_detect_interlacing))
        self.assertTrue(callable(adaptive_deinterlace))

    def test_import_old_restoration(self):
        from opencut.core.old_restoration import (
            restore_old_footage, list_presets, RESTORATION_PRESETS,
        )
        self.assertTrue(callable(restore_old_footage))
        self.assertIsInstance(RESTORATION_PRESETS, dict)

    def test_import_sdr_to_hdr(self):
        from opencut.core.sdr_to_hdr import (
            sdr_to_hdr, list_transfer_functions, TRANSFER_FUNCTIONS,
        )
        self.assertTrue(callable(sdr_to_hdr))

    def test_import_framerate_convert(self):
        from opencut.core.framerate_convert import (
            convert_framerate, list_presets, FRAMERATE_PRESETS,
        )
        self.assertTrue(callable(convert_framerate))

    def test_import_frame_extension(self):
        from opencut.core.frame_extension import (
            extend_frame_spatial, extend_frame_temporal,
            outpaint_aspect_ratio,
        )
        self.assertTrue(callable(outpaint_aspect_ratio))

    def test_import_img_to_video(self):
        from opencut.core.img_to_video import (
            animate_image, list_motion_presets, MOTION_PRESETS,
        )
        self.assertTrue(callable(animate_image))
        self.assertGreaterEqual(len(MOTION_PRESETS), 7)

    def test_import_scene_extend(self):
        from opencut.core.scene_extend import extend_scene
        self.assertTrue(callable(extend_scene))

    def test_import_video_condensed(self):
        from opencut.core.video_condensed import condense_video
        self.assertTrue(callable(condense_video))

    def test_import_bg_replace_ai(self):
        from opencut.core.bg_replace_ai import replace_background, BG_PRESETS
        self.assertTrue(callable(replace_background))
        self.assertGreaterEqual(len(BG_PRESETS), 5)

    def test_import_routes(self):
        from opencut.routes.repair_gen_routes import repair_gen_bp
        self.assertIsNotNone(repair_gen_bp)


# ============================================================
# Data validation tests
# ============================================================
class TestPresetCompleteness(unittest.TestCase):
    """Verify all presets have required fields."""

    def test_restoration_presets_complete(self):
        from opencut.core.old_restoration import RESTORATION_PRESETS
        required = {"description", "stabilize", "deinterlace",
                    "denoise_filter", "upscale", "color_restore",
                    "frame_interpolate"}
        for name, cfg in RESTORATION_PRESETS.items():
            for field in required:
                self.assertIn(field, cfg, f"Preset {name} missing {field}")

    def test_framerate_presets_complete(self):
        from opencut.core.framerate_convert import FRAMERATE_PRESETS
        required = {"description", "mi_mode", "mc_mode",
                    "me_mode", "vsbmc", "default_fps"}
        for name, cfg in FRAMERATE_PRESETS.items():
            for field in required:
                self.assertIn(field, cfg, f"Preset {name} missing {field}")

    def test_motion_presets_complete(self):
        from opencut.core.img_to_video import MOTION_PRESETS
        required = {"description", "start_scale", "end_scale",
                    "start_x", "start_y", "end_x", "end_y"}
        for name, cfg in MOTION_PRESETS.items():
            for field in required:
                self.assertIn(field, cfg, f"Preset {name} missing {field}")

    def test_transfer_functions_complete(self):
        from opencut.core.sdr_to_hdr import TRANSFER_FUNCTIONS
        required = {"name", "transfer", "max_luminance", "zscale_transfer"}
        for name, cfg in TRANSFER_FUNCTIONS.items():
            for field in required:
                self.assertIn(field, cfg, f"Transfer {name} missing {field}")

    def test_bg_presets_are_strings(self):
        from opencut.core.bg_replace_ai import BG_PRESETS
        for name, desc in BG_PRESETS.items():
            self.assertIsInstance(desc, str)
            self.assertTrue(len(desc) > 3)


if __name__ == "__main__":
    unittest.main()
