"""
Tests for OpenCut format export features:
  - GIF export with two-pass palette optimization
  - Animated WebP export
  - Animated APNG export
  - Metadata reading, stripping, selective preservation, and copy
  - Format routes (gif, webp, apng, metadata read/strip/copy)
"""

import inspect
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# GIF Export Core Tests
# ============================================================
class TestGifExport(unittest.TestCase):
    """Tests for opencut.core.gif_export.export_gif."""

    def test_export_gif_signature(self):
        """export_gif should have the expected parameters."""
        from opencut.core.gif_export import export_gif
        sig = inspect.signature(export_gif)
        expected_params = [
            "input_path", "output_path", "max_width", "fps",
            "max_colors", "dither", "loop", "max_file_size_mb", "on_progress",
        ]
        for p in expected_params:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_export_gif_defaults(self):
        """Verify default parameter values."""
        from opencut.core.gif_export import export_gif
        sig = inspect.signature(export_gif)
        self.assertIsNone(sig.parameters["output_path"].default)
        self.assertEqual(sig.parameters["max_width"].default, 480)
        self.assertEqual(sig.parameters["fps"].default, 15)
        self.assertEqual(sig.parameters["max_colors"].default, 256)
        self.assertEqual(sig.parameters["dither"].default, "sierra2_4a")
        self.assertEqual(sig.parameters["loop"].default, 0)
        self.assertIsNone(sig.parameters["max_file_size_mb"].default)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_export_gif_file_not_found(self):
        """export_gif should raise FileNotFoundError for missing input."""
        from opencut.core.gif_export import export_gif
        with self.assertRaises(FileNotFoundError):
            export_gif("/nonexistent/video.mp4")

    def test_export_gif_invalid_dither_fallback(self):
        """Invalid dither value should fall back to sierra2_4a."""
        from opencut.core.gif_export import VALID_DITHERS
        self.assertNotIn("bogus_dither", VALID_DITHERS)

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 480, "height": 270})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=524288)
    @patch("os.path.isfile", return_value=True)
    def test_export_gif_two_pass(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_gif should call FFmpeg twice (palettegen + paletteuse)."""
        from opencut.core.gif_export import export_gif

        result = export_gif(
            input_path="/fake/input.mp4",
            output_path="/fake/output.gif",
        )
        # Two FFmpeg calls: palette generation + palette application
        self.assertEqual(mock_ffmpeg.call_count, 2)
        self.assertEqual(result["output_path"], "/fake/output.gif")
        self.assertEqual(result["file_size"], 524288)
        self.assertEqual(result["width"], 480)
        self.assertEqual(result["height"], 270)

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 320, "height": 180})
    @patch("opencut.core.gif_export._file_size_bytes")
    @patch("os.path.isfile", return_value=True)
    def test_export_gif_size_constraint_retries(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """With max_file_size_mb, export_gif should retry with reduced quality."""
        from opencut.core.gif_export import export_gif

        # First attempt: 5MB, second attempt: 0.5MB (under 1MB target)
        mock_size.side_effect = [5 * 1024 * 1024, 500 * 1024]

        result = export_gif(
            input_path="/fake/input.mp4",
            output_path="/fake/output.gif",
            max_file_size_mb=1.0,
        )
        # Should have called FFmpeg 4 times (2 passes x 2 attempts)
        self.assertEqual(mock_ffmpeg.call_count, 4)
        self.assertEqual(result["file_size"], 500 * 1024)

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 480, "height": 270})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=100000)
    @patch("os.path.isfile", return_value=True)
    def test_export_gif_progress_callback(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_gif should invoke on_progress callbacks."""
        progress_calls = []

        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        export_gif_fn = __import__("opencut.core.gif_export", fromlist=["export_gif"]).export_gif
        export_gif_fn(
            input_path="/fake/input.mp4",
            output_path="/fake/output.gif",
            on_progress=on_progress,
        )
        self.assertTrue(len(progress_calls) >= 3)
        # First call should be early in progress
        self.assertLessEqual(progress_calls[0][0], 10)
        # Last call should be 100
        self.assertEqual(progress_calls[-1][0], 100)

    def test_valid_dithers_set(self):
        """VALID_DITHERS should contain all expected algorithms."""
        from opencut.core.gif_export import VALID_DITHERS
        expected = {"bayer", "floyd_steinberg", "sierra2", "sierra2_4a", "none"}
        for d in expected:
            self.assertIn(d, VALID_DITHERS)


# ============================================================
# WebP Export Core Tests
# ============================================================
class TestWebPExport(unittest.TestCase):
    """Tests for opencut.core.gif_export.export_webp."""

    def test_export_webp_signature(self):
        """export_webp should have the expected parameters."""
        from opencut.core.gif_export import export_webp
        sig = inspect.signature(export_webp)
        expected = ["input_path", "output_path", "max_width", "fps", "quality", "loop", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_export_webp_defaults(self):
        """Verify default parameter values for WebP."""
        from opencut.core.gif_export import export_webp
        sig = inspect.signature(export_webp)
        self.assertEqual(sig.parameters["quality"].default, 75)
        self.assertEqual(sig.parameters["max_width"].default, 480)

    def test_export_webp_file_not_found(self):
        """export_webp should raise FileNotFoundError for missing input."""
        from opencut.core.gif_export import export_webp
        with self.assertRaises(FileNotFoundError):
            export_webp("/nonexistent/video.mp4")

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 480, "height": 270})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=262144)
    @patch("os.path.isfile", return_value=True)
    def test_export_webp_uses_libwebp_anim(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_webp should use libwebp_anim codec."""
        from opencut.core.gif_export import export_webp

        result = export_webp(
            input_path="/fake/input.mp4",
            output_path="/fake/output.webp",
        )
        self.assertEqual(mock_ffmpeg.call_count, 1)
        cmd_args = mock_ffmpeg.call_args[0][0]
        self.assertIn("libwebp_anim", cmd_args)
        self.assertEqual(result["output_path"], "/fake/output.webp")

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 480, "height": 270})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=100000)
    @patch("os.path.isfile", return_value=True)
    def test_export_webp_quality_in_cmd(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_webp should pass quality parameter to FFmpeg."""
        from opencut.core.gif_export import export_webp

        export_webp(
            input_path="/fake/input.mp4",
            output_path="/fake/output.webp",
            quality=90,
        )
        cmd_args = mock_ffmpeg.call_args[0][0]
        quality_idx = cmd_args.index("-quality")
        self.assertEqual(cmd_args[quality_idx + 1], "90")

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 320, "height": 180})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=100000)
    @patch("os.path.isfile", return_value=True)
    def test_export_webp_auto_output_path(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_webp should auto-generate output path with _animated.webp suffix."""
        from opencut.core.gif_export import export_webp

        result = export_webp(input_path="/fake/myvideo.mp4")
        self.assertTrue(result["output_path"].endswith("_animated.webp"))
        self.assertIn("myvideo", result["output_path"])


# ============================================================
# APNG Export Core Tests
# ============================================================
class TestAPNGExport(unittest.TestCase):
    """Tests for opencut.core.gif_export.export_apng."""

    def test_export_apng_signature(self):
        """export_apng should have the expected parameters."""
        from opencut.core.gif_export import export_apng
        sig = inspect.signature(export_apng)
        expected = ["input_path", "output_path", "max_width", "fps", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_export_apng_file_not_found(self):
        """export_apng should raise FileNotFoundError for missing input."""
        from opencut.core.gif_export import export_apng
        with self.assertRaises(FileNotFoundError):
            export_apng("/nonexistent/video.mp4")

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 480, "height": 270})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=1048576)
    @patch("os.path.isfile", return_value=True)
    def test_export_apng_uses_apng_format(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_apng should use -f apng format flag."""
        from opencut.core.gif_export import export_apng

        result = export_apng(
            input_path="/fake/input.mp4",
            output_path="/fake/output.apng",
        )
        self.assertEqual(mock_ffmpeg.call_count, 1)
        cmd_args = mock_ffmpeg.call_args[0][0]
        self.assertIn("-f", cmd_args)
        fmt_idx = cmd_args.index("-f")
        self.assertEqual(cmd_args[fmt_idx + 1], "apng")
        self.assertEqual(result["output_path"], "/fake/output.apng")

    @patch("opencut.core.gif_export.run_ffmpeg")
    @patch("opencut.core.gif_export._get_dimensions", return_value={"width": 480, "height": 270})
    @patch("opencut.core.gif_export._file_size_bytes", return_value=100000)
    @patch("os.path.isfile", return_value=True)
    def test_export_apng_auto_output_path(self, mock_isfile, mock_size, mock_dims, mock_ffmpeg):
        """export_apng should auto-generate output path with _animated.apng suffix."""
        from opencut.core.gif_export import export_apng

        result = export_apng(input_path="/fake/clip.mov")
        self.assertTrue(result["output_path"].endswith("_animated.apng"))


# ============================================================
# Metadata Tools Core Tests
# ============================================================
class TestGetMetadata(unittest.TestCase):
    """Tests for opencut.core.metadata_tools.get_metadata."""

    def test_get_metadata_signature(self):
        """get_metadata should accept input_path."""
        from opencut.core.metadata_tools import get_metadata
        sig = inspect.signature(get_metadata)
        self.assertIn("input_path", sig.parameters)

    def test_get_metadata_file_not_found(self):
        """get_metadata should raise FileNotFoundError for missing input."""
        from opencut.core.metadata_tools import get_metadata
        with self.assertRaises(FileNotFoundError):
            get_metadata("/nonexistent/video.mp4")

    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_get_metadata_parses_ffprobe_output(self, mock_isfile, mock_run):
        """get_metadata should parse ffprobe JSON output correctly."""
        from opencut.core.metadata_tools import get_metadata

        ffprobe_output = json.dumps({
            "format": {
                "filename": "/fake/video.mp4",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "format_long_name": "QuickTime / MOV",
                "duration": "120.5",
                "size": "10485760",
                "bit_rate": "696000",
                "nb_streams": 2,
                "tags": {
                    "title": "My Video",
                    "creation_time": "2026-01-01T00:00:00.000000Z",
                    "artist": "Test User",
                },
            },
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "tags": {"language": "und"},
                },
                {
                    "index": 1,
                    "codec_name": "aac",
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 2,
                    "tags": {"language": "eng"},
                },
            ],
        })
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=ffprobe_output.encode(),
            stderr=b"",
        )

        result = get_metadata("/fake/video.mp4")
        self.assertIn("format_tags", result)
        self.assertIn("streams", result)
        self.assertIn("format", result)
        self.assertEqual(result["format_tags"]["title"], "My Video")
        self.assertEqual(len(result["streams"]), 2)
        self.assertEqual(result["streams"][0]["codec_name"], "h264")
        self.assertEqual(result["format"]["duration"], "120.5")

    @patch("subprocess.run")
    @patch("os.path.isfile", return_value=True)
    def test_get_metadata_ffprobe_failure(self, mock_isfile, mock_run):
        """get_metadata should raise RuntimeError on ffprobe failure."""
        from opencut.core.metadata_tools import get_metadata

        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"ffprobe error: invalid data",
        )
        with self.assertRaises(RuntimeError):
            get_metadata("/fake/video.mp4")


class TestStripMetadata(unittest.TestCase):
    """Tests for opencut.core.metadata_tools.strip_metadata."""

    def test_strip_metadata_signature(self):
        """strip_metadata should have the expected parameters."""
        from opencut.core.metadata_tools import strip_metadata
        sig = inspect.signature(strip_metadata)
        expected = ["input_path", "output_path", "preserve_fields",
                     "strip_fields", "mode", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_strip_metadata_default_mode(self):
        """Default mode should be strip_all."""
        from opencut.core.metadata_tools import strip_metadata
        sig = inspect.signature(strip_metadata)
        self.assertEqual(sig.parameters["mode"].default, "strip_all")

    def test_strip_metadata_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        from opencut.core.metadata_tools import strip_metadata
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                strip_metadata("/fake/video.mp4", mode="bogus")
            self.assertIn("bogus", str(ctx.exception))

    def test_strip_metadata_file_not_found(self):
        """strip_metadata should raise FileNotFoundError for missing input."""
        from opencut.core.metadata_tools import strip_metadata
        with self.assertRaises(FileNotFoundError):
            strip_metadata("/nonexistent/video.mp4")

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("opencut.core.metadata_tools.get_metadata")
    @patch("os.path.getsize", return_value=1048576)
    @patch("os.path.isfile", return_value=True)
    def test_strip_all_uses_map_metadata_minus1(self, mock_isfile, mock_size, mock_meta, mock_ffmpeg):
        """strip_all mode should use -map_metadata -1."""
        from opencut.core.metadata_tools import strip_metadata

        mock_meta.return_value = {
            "format_tags": {"title": "Test", "artist": "Me"},
            "streams": [],
        }

        result = strip_metadata(
            input_path="/fake/video.mp4",
            output_path="/fake/output.mp4",
            mode="strip_all",
        )
        cmd_args = mock_ffmpeg.call_args[0][0]
        self.assertIn("-map_metadata", cmd_args)
        idx = cmd_args.index("-map_metadata")
        self.assertEqual(cmd_args[idx + 1], "-1")
        self.assertEqual(result["mode"], "strip_all")
        self.assertEqual(result["stripped_count"], 2)
        self.assertEqual(result["preserved_count"], 0)

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("opencut.core.metadata_tools.get_metadata")
    @patch("os.path.getsize", return_value=1048576)
    @patch("os.path.isfile", return_value=True)
    def test_preserve_all_uses_map_metadata_0(self, mock_isfile, mock_size, mock_meta, mock_ffmpeg):
        """preserve_all mode should use -map_metadata 0."""
        from opencut.core.metadata_tools import strip_metadata

        mock_meta.return_value = {
            "format_tags": {"title": "Test"},
            "streams": [],
        }

        result = strip_metadata(
            input_path="/fake/video.mp4",
            output_path="/fake/output.mp4",
            mode="preserve_all",
        )
        cmd_args = mock_ffmpeg.call_args[0][0]
        idx = cmd_args.index("-map_metadata")
        self.assertEqual(cmd_args[idx + 1], "0")
        self.assertEqual(result["preserved_count"], 1)
        self.assertEqual(result["stripped_count"], 0)

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("opencut.core.metadata_tools.get_metadata")
    @patch("os.path.getsize", return_value=1048576)
    @patch("os.path.isfile", return_value=True)
    def test_selective_mode_preserves_specified_fields(self, mock_isfile, mock_size, mock_meta, mock_ffmpeg):
        """selective mode should keep only the specified fields."""
        from opencut.core.metadata_tools import strip_metadata

        mock_meta.return_value = {
            "format_tags": {
                "title": "My Title",
                "artist": "Secret Artist",
                "creation_time": "2026-01-01",
            },
            "streams": [],
        }

        result = strip_metadata(
            input_path="/fake/video.mp4",
            output_path="/fake/output.mp4",
            mode="selective",
            preserve_fields=["title", "creation_time"],
        )
        cmd_args = mock_ffmpeg.call_args[0][0]
        # Should have metadata entries for title and creation_time
        metadata_pairs = []
        for i, arg in enumerate(cmd_args):
            if arg == "-metadata":
                metadata_pairs.append(cmd_args[i + 1])
        self.assertTrue(any("title=" in p for p in metadata_pairs))
        self.assertTrue(any("creation_time=" in p for p in metadata_pairs))
        self.assertFalse(any("artist=" in p for p in metadata_pairs))
        self.assertEqual(result["preserved_count"], 2)
        self.assertEqual(result["stripped_count"], 1)

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("opencut.core.metadata_tools.get_metadata")
    @patch("os.path.getsize", return_value=1048576)
    @patch("os.path.isfile", return_value=True)
    def test_legal_mode_strips_privacy_fields(self, mock_isfile, mock_size, mock_meta, mock_ffmpeg):
        """legal mode should strip GPS and serial fields but keep timestamps."""
        from opencut.core.metadata_tools import strip_metadata

        mock_meta.return_value = {
            "format_tags": {
                "title": "My Title",
                "creation_time": "2026-01-01",
                "location": "+40.7128-074.0060/",
                "serial_number": "ABC123",
                "artist": "John Doe",
            },
            "streams": [],
        }

        result = strip_metadata(
            input_path="/fake/video.mp4",
            output_path="/fake/output.mp4",
            mode="legal",
        )
        cmd_args = mock_ffmpeg.call_args[0][0]
        metadata_pairs = []
        for i, arg in enumerate(cmd_args):
            if arg == "-metadata":
                metadata_pairs.append(cmd_args[i + 1])
        # title and creation_time should be preserved
        self.assertTrue(any("title=" in p for p in metadata_pairs))
        self.assertTrue(any("creation_time=" in p for p in metadata_pairs))
        # location, serial_number, artist should be stripped
        self.assertFalse(any("location=" in p for p in metadata_pairs))
        self.assertFalse(any("serial_number=" in p for p in metadata_pairs))
        self.assertFalse(any("artist=" in p for p in metadata_pairs))
        # 3 stripped (location, serial_number, artist), 2 preserved
        self.assertEqual(result["stripped_count"], 3)
        self.assertEqual(result["preserved_count"], 2)

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("opencut.core.metadata_tools.get_metadata")
    @patch("os.path.getsize", return_value=500000)
    @patch("os.path.isfile", return_value=True)
    def test_strip_metadata_progress_callback(self, mock_isfile, mock_size, mock_meta, mock_ffmpeg):
        """strip_metadata should invoke on_progress callbacks."""
        from opencut.core.metadata_tools import strip_metadata

        mock_meta.return_value = {"format_tags": {}, "streams": []}
        progress_calls = []

        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        strip_metadata(
            input_path="/fake/video.mp4",
            output_path="/fake/output.mp4",
            on_progress=on_progress,
        )
        self.assertTrue(len(progress_calls) >= 2)
        self.assertEqual(progress_calls[-1][0], 100)


class TestCopyWithMetadata(unittest.TestCase):
    """Tests for opencut.core.metadata_tools.copy_with_metadata."""

    def test_copy_with_metadata_signature(self):
        """copy_with_metadata should have the expected parameters."""
        from opencut.core.metadata_tools import copy_with_metadata
        sig = inspect.signature(copy_with_metadata)
        expected = ["input_path", "output_path", "metadata_overrides", "on_progress"]
        for p in expected:
            self.assertIn(p, sig.parameters)

    def test_copy_with_metadata_file_not_found(self):
        """copy_with_metadata should raise FileNotFoundError for missing input."""
        from opencut.core.metadata_tools import copy_with_metadata
        with self.assertRaises(FileNotFoundError):
            copy_with_metadata("/nonexistent/video.mp4", "/fake/output.mp4")

    def test_copy_with_metadata_requires_output(self):
        """copy_with_metadata should raise ValueError when output_path is empty."""
        from opencut.core.metadata_tools import copy_with_metadata
        with patch("os.path.isfile", return_value=True):
            with self.assertRaises(ValueError):
                copy_with_metadata("/fake/video.mp4", "")

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("os.path.getsize", return_value=2097152)
    @patch("os.path.isfile", return_value=True)
    def test_copy_with_overrides(self, mock_isfile, mock_size, mock_ffmpeg):
        """copy_with_metadata should apply metadata overrides via -metadata flags."""
        from opencut.core.metadata_tools import copy_with_metadata

        result = copy_with_metadata(
            input_path="/fake/input.mp4",
            output_path="/fake/output.mp4",
            metadata_overrides={"title": "New Title", "artist": "New Artist"},
        )
        cmd_args = mock_ffmpeg.call_args[0][0]
        metadata_pairs = []
        for i, arg in enumerate(cmd_args):
            if arg == "-metadata":
                metadata_pairs.append(cmd_args[i + 1])
        self.assertTrue(any("title=New Title" in p for p in metadata_pairs))
        self.assertTrue(any("artist=New Artist" in p for p in metadata_pairs))
        self.assertEqual(len(result["fields_modified"]), 2)
        self.assertIn("title", result["fields_modified"])

    @patch("opencut.core.metadata_tools.run_ffmpeg")
    @patch("os.path.getsize", return_value=1000000)
    @patch("os.path.isfile", return_value=True)
    def test_copy_sanitizes_metadata_keys(self, mock_isfile, mock_size, mock_ffmpeg):
        """copy_with_metadata should sanitize metadata key names."""
        from opencut.core.metadata_tools import copy_with_metadata

        result = copy_with_metadata(
            input_path="/fake/input.mp4",
            output_path="/fake/output.mp4",
            metadata_overrides={
                "valid_key": "value1",
                "also-valid.key": "value2",
                "": "empty_key_ignored",
                "has spaces bad": "value3",  # spaces stripped
            },
        )
        # Empty key should be excluded, key with spaces has spaces stripped
        self.assertIn("valid_key", result["fields_modified"])
        self.assertIn("also-valid.key", result["fields_modified"])
        self.assertNotIn("", result["fields_modified"])


class TestMetadataValidModes(unittest.TestCase):
    """Tests for metadata mode validation."""

    def test_valid_modes(self):
        """VALID_MODES should contain all documented modes."""
        from opencut.core.metadata_tools import VALID_MODES
        self.assertIn("strip_all", VALID_MODES)
        self.assertIn("preserve_all", VALID_MODES)
        self.assertIn("selective", VALID_MODES)
        self.assertIn("legal", VALID_MODES)

    def test_privacy_fields_set(self):
        """_PRIVACY_FIELDS should contain GPS and serial-related fields."""
        from opencut.core.metadata_tools import _PRIVACY_FIELDS
        self.assertIn("location", _PRIVACY_FIELDS)
        self.assertIn("gps_latitude", _PRIVACY_FIELDS)
        self.assertIn("serial_number", _PRIVACY_FIELDS)
        self.assertIn("artist", _PRIVACY_FIELDS)


# ============================================================
# Route Tests
# ============================================================
class TestFormatRoutes(unittest.TestCase):
    """Smoke tests for format_routes Blueprint endpoints."""

    @classmethod
    def setUpClass(cls):
        """Create Flask test app and client."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        cls.app = create_app(config=test_config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Get CSRF token
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf_token = data.get("csrf_token", "")

    def _headers(self):
        return {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    def test_export_gif_no_filepath(self):
        """POST /export/gif without filepath should return 400."""
        resp = self.client.post(
            "/export/gif",
            json={},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_webp_no_filepath(self):
        """POST /export/webp without filepath should return 400."""
        resp = self.client.post(
            "/export/webp",
            json={},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_apng_no_filepath(self):
        """POST /export/apng without filepath should return 400."""
        resp = self.client.post(
            "/export/apng",
            json={},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_metadata_read_no_filepath(self):
        """POST /metadata/read without filepath should return 400."""
        resp = self.client.post(
            "/metadata/read",
            json={},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_metadata_strip_no_filepath(self):
        """POST /metadata/strip without filepath should return 400."""
        resp = self.client.post(
            "/metadata/strip",
            json={},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_metadata_copy_no_filepath(self):
        """POST /metadata/copy without filepath should return 400."""
        resp = self.client.post(
            "/metadata/copy",
            json={},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_gif_missing_csrf(self):
        """POST /export/gif without CSRF token should return 403."""
        resp = self.client.post(
            "/export/gif",
            json={"filepath": "/fake/test.mp4"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)

    def test_metadata_read_missing_csrf(self):
        """POST /metadata/read without CSRF token should return 403."""
        resp = self.client.post(
            "/metadata/read",
            json={"filepath": "/fake/test.mp4"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)

    def test_export_gif_nonexistent_file(self):
        """POST /export/gif with nonexistent file should return 400."""
        resp = self.client.post(
            "/export/gif",
            json={"filepath": "/nonexistent/path/video.mp4"},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_metadata_strip_nonexistent_file(self):
        """POST /metadata/strip with nonexistent file should return 400."""
        resp = self.client.post(
            "/metadata/strip",
            json={"filepath": "/nonexistent/path/video.mp4"},
            headers=self._headers(),
        )
        self.assertEqual(resp.status_code, 400)


# ============================================================
# Blueprint Registration Test
# ============================================================
class TestBlueprintRegistration(unittest.TestCase):
    """Verify the format blueprint is registered properly."""

    def test_format_bp_exists(self):
        """format_bp should be importable."""
        from opencut.routes.format_routes import format_bp
        self.assertEqual(format_bp.name, "format")

    def test_format_bp_registered_in_app(self):
        """format_bp should be registered in the Flask app."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        # Check that our routes are registered
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/export/gif", rules)
        self.assertIn("/export/webp", rules)
        self.assertIn("/export/apng", rules)
        self.assertIn("/metadata/read", rules)
        self.assertIn("/metadata/strip", rules)
        self.assertIn("/metadata/copy", rules)


if __name__ == "__main__":
    unittest.main()
