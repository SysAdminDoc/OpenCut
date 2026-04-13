"""
Tests for OpenCut content production features.

Covers:
  - Scrolling credits generator (credits_gen.py)
  - Image sequence import & assembly (image_sequence.py)
  - Timelapse deflicker (timelapse.py)
  - Audiogram generator (audiogram.py)
  - Production routes (production_routes.py)
"""

import inspect
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Credits Generator Tests
# ============================================================
class TestCreditsGen(unittest.TestCase):
    """Tests for opencut.core.credits_gen module."""

    def test_parse_credits_file_basic(self):
        """parse_credits_file should parse sections from a text file."""
        from opencut.core.credits_gen import parse_credits_file

        content = (
            "DIRECTED BY\n"
            "Jane Smith\n"
            "\n"
            "CAST\n"
            "Actor A as Character One\n"
            "Actor B as Character Two\n"
            "\n"
            "MUSIC BY\n"
            "Composer X\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            sections = parse_credits_file(path)
            self.assertEqual(len(sections), 3)
            self.assertEqual(sections[0]["section"], "Directed By")
            self.assertEqual(sections[0]["names"], ["Jane Smith"])
            self.assertEqual(sections[1]["section"], "Cast")
            self.assertEqual(len(sections[1]["names"]), 2)
            self.assertIn("Actor A as Character One", sections[1]["names"])
            self.assertEqual(sections[2]["section"], "Music By")
        finally:
            os.unlink(path)

    def test_parse_credits_file_not_found(self):
        """parse_credits_file should raise FileNotFoundError for missing files."""
        from opencut.core.credits_gen import parse_credits_file
        with self.assertRaises(FileNotFoundError):
            parse_credits_file("/nonexistent/credits.txt")

    def test_parse_credits_empty_file(self):
        """parse_credits_file should handle empty files."""
        from opencut.core.credits_gen import parse_credits_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name

        try:
            sections = parse_credits_file(path)
            self.assertEqual(sections, [])
        finally:
            os.unlink(path)

    def test_generate_credits_signature(self):
        """generate_credits should accept all documented parameters."""
        from opencut.core.credits_gen import generate_credits

        sig = inspect.signature(generate_credits)
        params = list(sig.parameters.keys())
        self.assertIn("credits_data", params)
        self.assertIn("output_path_str", params)
        self.assertIn("width", params)
        self.assertIn("height", params)
        self.assertIn("fps", params)
        self.assertIn("scroll_speed", params)
        self.assertIn("font_size", params)
        self.assertIn("font_color", params)
        self.assertIn("bg_color", params)
        self.assertIn("font_path", params)
        self.assertIn("on_progress", params)

    def test_generate_credits_requires_pillow(self):
        """generate_credits should raise RuntimeError if Pillow is missing."""
        from opencut.core.credits_gen import generate_credits

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None,
                                         "PIL.ImageDraw": None,
                                         "PIL.ImageFont": None}):
            # Force the import to fail
            with patch("builtins.__import__", side_effect=ImportError("No module named 'PIL'")):
                with self.assertRaises((RuntimeError, ImportError)):
                    generate_credits(
                        [{"section": "Test", "names": ["Name"]}],
                        "/tmp/test.mp4",
                    )

    def test_parse_credits_multiple_blank_lines(self):
        """parse_credits_file should handle multiple consecutive blank lines."""
        from opencut.core.credits_gen import parse_credits_file

        content = "TITLE\nName One\n\n\n\nANOTHER SECTION\nName Two\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            sections = parse_credits_file(path)
            self.assertEqual(len(sections), 2)
        finally:
            os.unlink(path)


# ============================================================
# Image Sequence Tests
# ============================================================
class TestImageSequence(unittest.TestCase):
    """Tests for opencut.core.image_sequence module."""

    def test_detect_image_sequence_basic(self):
        """detect_image_sequence should find numbered PNG files."""
        from opencut.core.image_sequence import detect_image_sequence

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create numbered image files (empty files, just for detection)
            for i in range(1, 11):
                open(os.path.join(tmpdir, f"frame_{i:04d}.png"), "w").close()

            info = detect_image_sequence(tmpdir)
            self.assertEqual(info.first_frame, 1)
            self.assertEqual(info.last_frame, 10)
            self.assertEqual(info.total_frames, 10)
            self.assertEqual(info.extension, "png")
            self.assertIn("%04d", info.pattern)
            self.assertEqual(info.gaps, [])

    def test_detect_image_sequence_with_gaps(self):
        """detect_image_sequence should detect gaps in frame numbering."""
        from opencut.core.image_sequence import detect_image_sequence

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in [1, 2, 3, 5, 7, 8, 9, 10]:
                open(os.path.join(tmpdir, f"img_{i:04d}.tif"), "w").close()

            info = detect_image_sequence(tmpdir)
            self.assertEqual(info.first_frame, 1)
            self.assertEqual(info.last_frame, 10)
            self.assertEqual(info.total_frames, 8)
            self.assertIn(4, info.gaps)
            self.assertIn(6, info.gaps)

    def test_detect_image_sequence_no_images(self):
        """detect_image_sequence should raise ValueError for empty folders."""
        from opencut.core.image_sequence import detect_image_sequence

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-image files
            open(os.path.join(tmpdir, "readme.txt"), "w").close()
            with self.assertRaises(ValueError):
                detect_image_sequence(tmpdir)

    def test_detect_image_sequence_folder_not_found(self):
        """detect_image_sequence should raise FileNotFoundError."""
        from opencut.core.image_sequence import detect_image_sequence
        with self.assertRaises(FileNotFoundError):
            detect_image_sequence("/nonexistent/folder")

    def test_sequence_info_dataclass(self):
        """SequenceInfo should have all expected fields."""
        from opencut.core.image_sequence import SequenceInfo

        field_names = {f.name for f in fields(SequenceInfo)}
        expected = {"pattern", "first_frame", "last_frame", "total_frames",
                    "extension", "detected_pattern_str", "gaps", "folder"}
        self.assertTrue(expected.issubset(field_names))

    def test_assemble_image_sequence_signature(self):
        """assemble_image_sequence should accept all documented parameters."""
        from opencut.core.image_sequence import assemble_image_sequence

        sig = inspect.signature(assemble_image_sequence)
        params = list(sig.parameters.keys())
        self.assertIn("folder_path", params)
        self.assertIn("output_path_str", params)
        self.assertIn("fps", params)
        self.assertIn("pattern", params)
        self.assertIn("start_frame", params)
        self.assertIn("end_frame", params)
        self.assertIn("codec", params)
        self.assertIn("quality", params)
        self.assertIn("on_progress", params)

    def test_detect_multiple_extensions(self):
        """detect_image_sequence should pick the largest group of images."""
        from opencut.core.image_sequence import detect_image_sequence

        with tempfile.TemporaryDirectory() as tmpdir:
            # 3 PNGs and 7 TIFFs - should detect TIFFs
            for i in range(1, 4):
                open(os.path.join(tmpdir, f"img_{i:04d}.png"), "w").close()
            for i in range(1, 8):
                open(os.path.join(tmpdir, f"img_{i:04d}.tif"), "w").close()

            info = detect_image_sequence(tmpdir)
            self.assertEqual(info.extension, "tif")
            self.assertEqual(info.total_frames, 7)


# ============================================================
# Timelapse Deflicker Tests
# ============================================================
class TestTimelapse(unittest.TestCase):
    """Tests for opencut.core.timelapse module."""

    def test_flicker_analysis_dataclass(self):
        """FlickerAnalysis should have all expected fields."""
        from opencut.core.timelapse import FlickerAnalysis

        field_names = {f.name for f in fields(FlickerAnalysis)}
        expected = {"per_frame_luminance", "flicker_score", "needs_deflicker",
                    "frame_count", "avg_luminance", "min_luminance",
                    "max_luminance", "std_dev"}
        self.assertTrue(expected.issubset(field_names))

    def test_analyze_flicker_signature(self):
        """analyze_flicker should accept input_path and on_progress."""
        from opencut.core.timelapse import analyze_flicker

        sig = inspect.signature(analyze_flicker)
        params = list(sig.parameters.keys())
        self.assertIn("input_path", params)
        self.assertIn("on_progress", params)

    def test_deflicker_signature(self):
        """deflicker should accept all documented parameters."""
        from opencut.core.timelapse import deflicker

        sig = inspect.signature(deflicker)
        params = list(sig.parameters.keys())
        self.assertIn("input_path", params)
        self.assertIn("output_path_str", params)
        self.assertIn("window_size", params)
        self.assertIn("strength", params)
        self.assertIn("method", params)
        self.assertIn("on_progress", params)

    def test_rolling_average_basic(self):
        """_rolling_average should smooth values."""
        from opencut.core.timelapse import _rolling_average

        values = [100.0, 200.0, 100.0, 200.0, 100.0]
        result = _rolling_average(values, 3)
        self.assertEqual(len(result), 5)
        # Middle values should be smoothed toward 150
        self.assertAlmostEqual(result[1], 133.33, places=1)
        self.assertAlmostEqual(result[2], 166.67, places=1)
        self.assertAlmostEqual(result[3], 133.33, places=1)

    def test_rolling_average_window_1(self):
        """_rolling_average with window=1 should return original values."""
        from opencut.core.timelapse import _rolling_average

        values = [10.0, 20.0, 30.0]
        result = _rolling_average(values, 1)
        self.assertEqual(result, values)

    def test_rolling_average_large_window(self):
        """_rolling_average with window > len should still work."""
        from opencut.core.timelapse import _rolling_average

        values = [10.0, 20.0, 30.0]
        result = _rolling_average(values, 99)
        self.assertEqual(len(result), 3)
        # All values should converge toward the mean
        mean = 20.0
        self.assertAlmostEqual(result[0], mean, places=1)
        self.assertAlmostEqual(result[1], mean, places=1)
        self.assertAlmostEqual(result[2], mean, places=1)

    def test_deflicker_method_validation(self):
        """deflicker should handle invalid method gracefully."""
        from opencut.core.timelapse import deflicker

        # Should not crash on import; actual execution needs FFmpeg
        sig = inspect.signature(deflicker)
        self.assertEqual(sig.parameters["method"].default, "auto")

    def test_flicker_analysis_defaults(self):
        """FlickerAnalysis should have sensible defaults."""
        from opencut.core.timelapse import FlickerAnalysis

        analysis = FlickerAnalysis()
        self.assertEqual(analysis.flicker_score, 0.0)
        self.assertFalse(analysis.needs_deflicker)
        self.assertEqual(analysis.per_frame_luminance, [])
        self.assertEqual(analysis.frame_count, 0)


# ============================================================
# Audiogram Generator Tests
# ============================================================
class TestAudiogramGen(unittest.TestCase):
    """Tests for opencut.core.audiogram module."""

    def test_generate_audiogram_signature(self):
        """generate_audiogram should accept all documented parameters."""
        from opencut.core.audiogram import generate_audiogram

        sig = inspect.signature(generate_audiogram)
        params = list(sig.parameters.keys())
        self.assertIn("audio_path", params)
        self.assertIn("output_path_str", params)
        self.assertIn("style", params)
        self.assertIn("width", params)
        self.assertIn("height", params)
        self.assertIn("bg_color", params)
        self.assertIn("wave_color", params)
        self.assertIn("duration", params)
        self.assertIn("artwork_path", params)
        self.assertIn("title_text", params)
        self.assertIn("caption_srt", params)
        self.assertIn("on_progress", params)

    def test_audiogram_styles_constant(self):
        """AUDIOGRAM_STYLES should contain all three styles."""
        from opencut.core.audiogram import AUDIOGRAM_STYLES

        self.assertIn("bars", AUDIOGRAM_STYLES)
        self.assertIn("wave", AUDIOGRAM_STYLES)
        self.assertIn("circular", AUDIOGRAM_STYLES)

    def test_escape_ffmpeg_text(self):
        """_escape_ffmpeg_text should escape colons and backslashes."""
        from opencut.core.audiogram import _escape_ffmpeg_text

        self.assertEqual(_escape_ffmpeg_text("Hello: World"), "Hello\\: World")
        self.assertEqual(_escape_ffmpeg_text("50%"), "50%%")
        self.assertEqual(_escape_ffmpeg_text("C:\\path"), "C\\:\\\\path")

    def test_generate_audiogram_invalid_style(self):
        """generate_audiogram should fallback gracefully with bad style."""
        from opencut.core.audiogram import generate_audiogram

        # Mock ffprobe to return 0 duration to trigger the ValueError early
        with patch("opencut.core.audiogram._get_audio_duration", return_value=0.0):
            with self.assertRaises(ValueError) as ctx:
                generate_audiogram("/fake/audio.mp3", style="invalid_style")
            self.assertIn("duration", str(ctx.exception).lower())

    def test_get_audio_duration_handles_error(self):
        """_get_audio_duration should return 0.0 on ffprobe failure."""
        from opencut.core.audiogram import _get_audio_duration

        with patch("subprocess.run", side_effect=Exception("ffprobe failed")):
            result = _get_audio_duration("/nonexistent.mp3")
            self.assertEqual(result, 0.0)

    def test_audiogram_default_params(self):
        """generate_audiogram defaults should match spec."""
        from opencut.core.audiogram import generate_audiogram

        sig = inspect.signature(generate_audiogram)
        self.assertEqual(sig.parameters["style"].default, "bars")
        self.assertEqual(sig.parameters["width"].default, 1080)
        self.assertEqual(sig.parameters["height"].default, 1080)
        self.assertEqual(sig.parameters["bg_color"].default, "#1a1a2e")
        self.assertEqual(sig.parameters["wave_color"].default, "#e94560")


# ============================================================
# Production Routes Tests
# ============================================================
class TestProductionRoutes(unittest.TestCase):
    """Tests for production route blueprint registration and endpoints."""

    def test_blueprint_registered(self):
        """production_bp should be importable."""
        from opencut.routes.production_routes import production_bp
        self.assertEqual(production_bp.name, "production")

    def test_blueprint_in_register_blueprints(self):
        """production_bp should be registered by register_blueprints."""
        import opencut.routes as routes_mod
        source = inspect.getsource(routes_mod.register_blueprints)
        self.assertIn("production_bp", source)
        self.assertIn("production_routes", source)

    def test_credits_generate_route_exists(self):
        """POST /credits/generate route should exist on the app."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/credits/generate", rules)
        self.assertIn("/credits/parse", rules)
        self.assertIn("/image-sequence/detect", rules)
        self.assertIn("/image-sequence/assemble", rules)
        self.assertIn("/timelapse/analyze-flicker", rules)
        self.assertIn("/timelapse/deflicker", rules)
        self.assertIn("/audiogram/generate", rules)

    def test_credits_parse_no_filepath(self):
        """POST /credits/parse should return 400 without filepath."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        # Get CSRF token
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        resp = client.post("/credits/parse",
                           json={},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 400)

    def test_image_sequence_detect_no_folder(self):
        """POST /image-sequence/detect should return 400 without folder_path."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        resp = client.post("/image-sequence/detect",
                           json={},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 400)

    def test_credits_generate_no_data(self):
        """POST /credits/generate should return job_id (async) but fail in worker."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        # Empty credits_data should still get a job_id (validation is in worker)
        resp = client.post("/credits/generate",
                           json={"credits_data": []},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        # async_job returns job_id even if it will fail later
        data = resp.get_json()
        self.assertIn("job_id", data)

    def test_timelapse_analyze_no_filepath(self):
        """POST /timelapse/analyze-flicker should return 400 without filepath."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        resp = client.post("/timelapse/analyze-flicker",
                           json={},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 400)

    def test_audiogram_generate_no_audio(self):
        """POST /audiogram/generate should return 400 without audio_path."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        resp = client.post("/audiogram/generate",
                           json={},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 400)

    def test_timelapse_deflicker_no_filepath(self):
        """POST /timelapse/deflicker should return 400 without filepath."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        resp = client.post("/timelapse/deflicker",
                           json={},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        self.assertEqual(resp.status_code, 400)

    def test_image_sequence_assemble_no_folder(self):
        """POST /image-sequence/assemble should get job_id but fail without folder."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app

        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")

        resp = client.post("/image-sequence/assemble",
                           json={},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        # filepath_required=False so it gets a job_id
        data = resp.get_json()
        self.assertIn("job_id", data)


if __name__ == "__main__":
    unittest.main()
