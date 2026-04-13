"""
Tests for OpenCut generative features.

Covers: talking_head (face detection, simple FFmpeg pipeline, SadTalker/LivePortrait
        backends, config/result dataclasses), gaussian_splat (PLY parsing, camera
        path math, orbit generation, interpolation, fallback renderer, validation),
        and generative_routes blueprint.
"""

import json
import math
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Helper: Create minimal PLY files for testing
# ============================================================

def _make_ascii_ply(path, num_vertices=10, include_colors=False, include_sh=False):
    """Create a minimal ASCII PLY file for testing."""
    lines = ["ply", "format ascii 1.0", f"element vertex {num_vertices}"]
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    if include_colors:
        lines.append("property uchar red")
        lines.append("property uchar green")
        lines.append("property uchar blue")
    if include_sh:
        lines.append("property float f_dc_0")
        lines.append("property float f_dc_1")
        lines.append("property float f_dc_2")
        lines.append("property float opacity")
    lines.append("end_header")
    for i in range(num_vertices):
        row = f"{float(i)} {float(i * 2)} {float(i * 3)}"
        if include_colors:
            row += f" {min(255, i * 25)} {min(255, i * 50)} {min(255, i * 75)}"
        if include_sh:
            row += f" {0.5 + i * 0.01} {0.3 + i * 0.01} {0.2 + i * 0.01} {0.9}"
        lines.append(row)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _make_binary_ply(path, num_vertices=10, include_sh=False):
    """Create a minimal binary little-endian PLY file for testing."""
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if include_sh:
        header_lines.extend([
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
        ])
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(num_vertices):
            f.write(struct.pack("<fff", float(i), float(i * 2), float(i * 3)))
            if include_sh:
                f.write(struct.pack("<ffff", 0.5, 0.3, 0.2, 0.9))


def _make_dummy_image(path):
    """Create a minimal dummy image file."""
    # Write a tiny valid file (doesn't need to be a real image for mocked tests)
    with open(path, "wb") as f:
        f.write(b"\x00" * 100)


def _make_dummy_audio(path):
    """Create a minimal dummy audio file."""
    with open(path, "wb") as f:
        f.write(b"\x00" * 100)


# ============================================================
# Talking Head Core Tests
# ============================================================

class TestTalkingHeadConfig(unittest.TestCase):
    """Tests for TalkingHeadConfig dataclass."""

    def test_default_config(self):
        from opencut.core.talking_head import TalkingHeadConfig
        config = TalkingHeadConfig()
        self.assertEqual(config.backend, "simple")
        self.assertEqual(config.fps, 25)
        self.assertEqual(config.resolution, (512, 512))
        self.assertEqual(config.expression_scale, 1.0)
        self.assertEqual(config.pose_style, 0)

    def test_custom_config(self):
        from opencut.core.talking_head import TalkingHeadConfig
        config = TalkingHeadConfig(
            image_path="/test/img.png",
            audio_path="/test/audio.wav",
            backend="sadtalker",
            fps=30,
            resolution=(1024, 1024),
            expression_scale=1.5,
            pose_style=3,
        )
        self.assertEqual(config.image_path, "/test/img.png")
        self.assertEqual(config.backend, "sadtalker")
        self.assertEqual(config.resolution, (1024, 1024))

    def test_config_still_mode(self):
        from opencut.core.talking_head import TalkingHeadConfig
        config = TalkingHeadConfig(still_mode=True)
        self.assertTrue(config.still_mode)

    def test_config_enhancer(self):
        from opencut.core.talking_head import TalkingHeadConfig
        config = TalkingHeadConfig(enhancer="gfpgan")
        self.assertEqual(config.enhancer, "gfpgan")


class TestTalkingHeadResult(unittest.TestCase):
    """Tests for TalkingHeadResult dataclass."""

    def test_default_result(self):
        from opencut.core.talking_head import TalkingHeadResult
        result = TalkingHeadResult()
        self.assertEqual(result.output_path, "")
        self.assertEqual(result.duration, 0.0)
        self.assertEqual(result.frames, 0)
        self.assertFalse(result.face_detected)

    def test_populated_result(self):
        from opencut.core.talking_head import TalkingHeadResult
        result = TalkingHeadResult(
            output_path="/out/video.mp4",
            duration=10.5,
            frames=262,
            backend_used="simple",
            face_detected=True,
            message="Done",
        )
        self.assertEqual(result.frames, 262)
        self.assertEqual(result.backend_used, "simple")


class TestFaceDetection(unittest.TestCase):
    """Tests for detect_face_in_image."""

    def test_missing_image(self):
        from opencut.core.talking_head import detect_face_in_image
        with self.assertRaises(FileNotFoundError):
            detect_face_in_image("/nonexistent/image.png")

    @patch("opencut.core.talking_head.ensure_package", return_value=True)
    def test_detect_face_with_faces(self, mock_pkg):
        import numpy as np

        from opencut.core.talking_head import detect_face_in_image

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        _make_dummy_image(tmp.name)

        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)
        mock_faces = np.array([[10, 10, 50, 50], [60, 60, 30, 30]])

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = mock_faces

        with patch("cv2.imread", return_value=mock_img), \
             patch("cv2.cvtColor", return_value=mock_gray), \
             patch("cv2.CascadeClassifier", return_value=mock_cascade), \
             patch("cv2.data") as mock_data, \
             patch("os.path.isfile", side_effect=lambda p: True):
            mock_data.haarcascades = "/mock/"
            result = detect_face_in_image(tmp.name)

        self.assertTrue(result["detected"])
        self.assertEqual(result["count"], 2)
        self.assertIsNotNone(result["primary_face"])
        os.unlink(tmp.name)

    @patch("opencut.core.talking_head.ensure_package", return_value=True)
    def test_detect_face_no_faces(self, mock_pkg):
        import numpy as np

        from opencut.core.talking_head import detect_face_in_image

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        _make_dummy_image(tmp.name)

        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_gray = np.zeros((100, 100), dtype=np.uint8)

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([]).reshape(0, 4)

        with patch("cv2.imread", return_value=mock_img), \
             patch("cv2.cvtColor", return_value=mock_gray), \
             patch("cv2.CascadeClassifier", return_value=mock_cascade), \
             patch("cv2.data") as mock_data, \
             patch("os.path.isfile", side_effect=lambda p: True):
            mock_data.haarcascades = "/mock/"
            result = detect_face_in_image(tmp.name)

        self.assertFalse(result["detected"])
        self.assertEqual(result["count"], 0)
        os.unlink(tmp.name)

    @patch("opencut.core.talking_head.ensure_package", return_value=True)
    def test_detect_face_unreadable_image(self, mock_pkg):
        from opencut.core.talking_head import detect_face_in_image

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        _make_dummy_image(tmp.name)

        with patch("cv2.imread", return_value=None):
            with self.assertRaises(ValueError):
                detect_face_in_image(tmp.name)
        os.unlink(tmp.name)

    @patch("opencut.core.talking_head.ensure_package", return_value=False)
    def test_detect_face_no_opencv(self, mock_pkg):
        from opencut.core.talking_head import detect_face_in_image

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        _make_dummy_image(tmp.name)

        with self.assertRaises(RuntimeError):
            detect_face_in_image(tmp.name)
        os.unlink(tmp.name)


class TestListBackends(unittest.TestCase):
    """Tests for list_available_backends."""

    def test_list_backends_includes_simple(self):
        from opencut.core.talking_head import list_available_backends
        backends = list_available_backends()
        names = [b["name"] for b in backends]
        self.assertIn("simple", names)
        # Simple should always be available
        simple = next(b for b in backends if b["name"] == "simple")
        self.assertTrue(simple["available"])

    def test_list_backends_structure(self):
        from opencut.core.talking_head import list_available_backends
        backends = list_available_backends()
        for b in backends:
            self.assertIn("name", b)
            self.assertIn("label", b)
            self.assertIn("available", b)
            self.assertIn("description", b)

    def test_list_backends_sadtalker_unavailable(self):
        """SadTalker should be unavailable in test environment."""
        from opencut.core.talking_head import list_available_backends
        backends = list_available_backends()
        sadtalker = next((b for b in backends if b["name"] == "sadtalker"), None)
        self.assertIsNotNone(sadtalker)
        self.assertFalse(sadtalker["available"])

    def test_list_backends_has_all_three(self):
        from opencut.core.talking_head import list_available_backends
        backends = list_available_backends()
        names = {b["name"] for b in backends}
        self.assertIn("sadtalker", names)
        self.assertIn("liveportrait", names)
        self.assertIn("simple", names)


class TestAudioDuration(unittest.TestCase):
    """Tests for _get_audio_duration helper."""

    @patch("subprocess.run")
    def test_get_audio_duration_success(self, mock_run):
        from opencut.core.talking_head import _get_audio_duration
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "5.5"}}).encode(),
        )
        dur = _get_audio_duration("/test/audio.wav")
        self.assertAlmostEqual(dur, 5.5)

    @patch("subprocess.run")
    def test_get_audio_duration_failure(self, mock_run):
        from opencut.core.talking_head import _get_audio_duration
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        dur = _get_audio_duration("/test/audio.wav")
        self.assertEqual(dur, 0.0)

    @patch("subprocess.run", side_effect=Exception("boom"))
    def test_get_audio_duration_exception(self, mock_run):
        from opencut.core.talking_head import _get_audio_duration
        dur = _get_audio_duration("/test/audio.wav")
        self.assertEqual(dur, 0.0)


class TestSimpleTalkingHead(unittest.TestCase):
    """Tests for generate_simple_talking_head."""

    def setUp(self):
        self.img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.img.write(b"\x00" * 100)
        self.img.close()
        self.audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.audio.write(b"\x00" * 100)
        self.audio.close()

    def tearDown(self):
        for f in (self.img.name, self.audio.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    def test_missing_image(self):
        from opencut.core.talking_head import generate_simple_talking_head
        with self.assertRaises(FileNotFoundError):
            generate_simple_talking_head("/nonexistent.png", self.audio.name)

    def test_missing_audio(self):
        from opencut.core.talking_head import generate_simple_talking_head
        with self.assertRaises(FileNotFoundError):
            generate_simple_talking_head(self.img.name, "/nonexistent.wav")

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_generates_video(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import generate_simple_talking_head
        result = generate_simple_talking_head(self.img.name, self.audio.name)
        self.assertTrue(result.endswith("_talking.mp4"))
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_custom_output_path(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import generate_simple_talking_head
        out = os.path.join(tempfile.gettempdir(), "custom_output.mp4")
        result = generate_simple_talking_head(self.img.name, self.audio.name, output=out)
        self.assertEqual(result, out)

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_ffmpeg_cmd_contains_zoompan(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import generate_simple_talking_head
        generate_simple_talking_head(self.img.name, self.audio.name)
        cmd = mock_ffmpeg.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        self.assertIn("zoompan", cmd_str)

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_progress_callback(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import generate_simple_talking_head
        progress_calls = []
        def _progress(pct, msg):
            progress_calls.append((pct, msg))
        generate_simple_talking_head(self.img.name, self.audio.name, on_progress=_progress)
        self.assertTrue(len(progress_calls) > 0)
        self.assertEqual(progress_calls[-1][0], 100)

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=0.0)
    def test_zero_duration_fallback(self, mock_dur, mock_ffmpeg):
        """Zero duration from ffprobe should fallback to 10s."""
        from opencut.core.talking_head import generate_simple_talking_head
        generate_simple_talking_head(self.img.name, self.audio.name)
        cmd = mock_ffmpeg.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        self.assertIn("-t 10.0", cmd_str)

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_custom_fps(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import generate_simple_talking_head
        generate_simple_talking_head(self.img.name, self.audio.name, fps=30)
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("fps=30", cmd_str)


class TestGenerateTalkingHead(unittest.TestCase):
    """Tests for the main generate_talking_head entry point."""

    def setUp(self):
        self.img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.img.write(b"\x00" * 100)
        self.img.close()
        self.audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.audio.write(b"\x00" * 100)
        self.audio.close()

    def tearDown(self):
        for f in (self.img.name, self.audio.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    def test_missing_image(self):
        from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head
        config = TalkingHeadConfig(image_path="/no.png", audio_path=self.audio.name)
        with self.assertRaises(FileNotFoundError):
            generate_talking_head(config)

    def test_missing_audio(self):
        from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head
        config = TalkingHeadConfig(image_path=self.img.name, audio_path="/no.wav")
        with self.assertRaises(FileNotFoundError):
            generate_talking_head(config)

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_simple_backend_fallback(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head
        config = TalkingHeadConfig(
            image_path=self.img.name,
            audio_path=self.audio.name,
            backend="sadtalker",  # not available, should fall back to simple
        )
        result = generate_talking_head(config)
        self.assertEqual(result.backend_used, "simple")

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_explicit_simple_backend(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head
        config = TalkingHeadConfig(
            image_path=self.img.name,
            audio_path=self.audio.name,
            backend="simple",
        )
        result = generate_talking_head(config)
        self.assertEqual(result.backend_used, "simple")
        self.assertGreater(result.duration, 0)

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=8.0)
    def test_auto_output_path(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head
        config = TalkingHeadConfig(
            image_path=self.img.name,
            audio_path=self.audio.name,
        )
        result = generate_talking_head(config)
        self.assertTrue(result.output_path.endswith("_talking_head.mp4"))

    @patch("opencut.core.talking_head.run_ffmpeg")
    @patch("opencut.core.talking_head._get_audio_duration", return_value=5.0)
    def test_custom_output(self, mock_dur, mock_ffmpeg):
        from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head
        out = os.path.join(tempfile.gettempdir(), "my_vid.mp4")
        config = TalkingHeadConfig(
            image_path=self.img.name,
            audio_path=self.audio.name,
        )
        result = generate_talking_head(config, output=out)
        self.assertEqual(result.output_path, out)


# ============================================================
# Gaussian Splat Core Tests
# ============================================================

class TestPLYParsing(unittest.TestCase):
    """Tests for PLY file parsing."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_parse_ascii_header(self):
        from opencut.core.gaussian_splat import _parse_ply_header
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_ascii_ply(ply, 5)
        header = _parse_ply_header(ply)
        self.assertEqual(header["vertex_count"], 5)
        self.assertEqual(header["format"], "ascii")
        self.assertTrue(len(header["properties"]) >= 3)

    def test_parse_binary_header(self):
        from opencut.core.gaussian_splat import _parse_ply_header
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_binary_ply(ply, 20)
        header = _parse_ply_header(ply)
        self.assertEqual(header["vertex_count"], 20)
        self.assertEqual(header["format"], "binary_little_endian")

    def test_extract_property_names(self):
        from opencut.core.gaussian_splat import _extract_property_names
        props = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
        ]
        names = _extract_property_names(props)
        self.assertEqual(names, ["x", "y", "z", "red"])

    def test_property_struct_format(self):
        from opencut.core.gaussian_splat import _get_property_struct_format
        props = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
        ]
        fmt = _get_property_struct_format(props)
        self.assertEqual(fmt, "<fffB")

    def test_read_ascii_positions(self):
        from opencut.core.gaussian_splat import _parse_ply_header, _read_positions_ascii
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_ascii_ply(ply, 5)
        header = _parse_ply_header(ply)
        positions = _read_positions_ascii(ply, header)
        self.assertEqual(len(positions), 5)
        self.assertAlmostEqual(positions[0][0], 0.0)
        self.assertAlmostEqual(positions[1][0], 1.0)
        self.assertAlmostEqual(positions[1][1], 2.0)

    def test_read_binary_positions(self):
        from opencut.core.gaussian_splat import _parse_ply_header, _read_positions_binary
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_binary_ply(ply, 5)
        header = _parse_ply_header(ply)
        positions = _read_positions_binary(ply, header)
        self.assertEqual(len(positions), 5)
        self.assertAlmostEqual(positions[0][0], 0.0)
        self.assertAlmostEqual(positions[2][2], 6.0)

    def test_compute_bounds(self):
        from opencut.core.gaussian_splat import _compute_bounds
        positions = [(0, 0, 0), (10, 20, 30), (5, 10, 15)]
        bounds = _compute_bounds(positions)
        self.assertEqual(bounds["min"], (0, 0, 0))
        self.assertEqual(bounds["max"], (10, 20, 30))
        self.assertEqual(bounds["center"], (5.0, 10.0, 15.0))
        self.assertEqual(bounds["extent"], (10, 20, 30))

    def test_compute_bounds_empty(self):
        from opencut.core.gaussian_splat import _compute_bounds
        bounds = _compute_bounds([])
        self.assertEqual(bounds["min"], (0.0, 0.0, 0.0))

    def test_compute_bounds_single_point(self):
        from opencut.core.gaussian_splat import _compute_bounds
        bounds = _compute_bounds([(5.0, 3.0, 1.0)])
        self.assertEqual(bounds["min"], (5.0, 3.0, 1.0))
        self.assertEqual(bounds["max"], (5.0, 3.0, 1.0))
        self.assertEqual(bounds["extent"], (0.0, 0.0, 0.0))


class TestSplatValidation(unittest.TestCase):
    """Tests for validate_splat."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_ascii_ply(self):
        from opencut.core.gaussian_splat import validate_splat
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_ascii_ply(ply, 10, include_sh=True)
        result = validate_splat(ply)
        self.assertTrue(result["valid"])
        self.assertEqual(result["point_count"], 10)
        self.assertTrue(result["has_sh"])
        self.assertTrue(result["has_opacity"])

    def test_valid_binary_ply(self):
        from opencut.core.gaussian_splat import validate_splat
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_binary_ply(ply, 20, include_sh=True)
        result = validate_splat(ply)
        self.assertTrue(result["valid"])
        self.assertEqual(result["point_count"], 20)

    def test_missing_file(self):
        from opencut.core.gaussian_splat import validate_splat
        result = validate_splat("/nonexistent.ply")
        self.assertFalse(result["valid"])
        self.assertIn("File not found", result["errors"][0])

    def test_not_ply_file(self):
        from opencut.core.gaussian_splat import validate_splat
        txt = os.path.join(self.tmp_dir, "test.txt")
        with open(txt, "w") as f:
            f.write("not a ply file")
        result = validate_splat(txt)
        self.assertFalse(result["valid"])

    def test_ply_without_xyz(self):
        """PLY file without x,y,z properties should fail validation."""
        from opencut.core.gaussian_splat import validate_splat
        ply = os.path.join(self.tmp_dir, "bad.ply")
        with open(ply, "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nproperty float a\nend_header\n1.0\n")
        result = validate_splat(ply)
        self.assertFalse(result["valid"])

    def test_non_ply_extension(self):
        from opencut.core.gaussian_splat import validate_splat
        ply = os.path.join(self.tmp_dir, "test.obj")
        with open(ply, "wb") as f:
            f.write(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n")
        result = validate_splat(ply)
        # Valid PLY data but wrong extension — should have warning
        self.assertTrue(any("extension" in e.lower() for e in result["errors"]))


class TestLoadSplat(unittest.TestCase):
    """Tests for load_splat."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_ascii(self):
        from opencut.core.gaussian_splat import load_splat
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_ascii_ply(ply, 10, include_colors=True)
        scene = load_splat(ply)
        self.assertEqual(scene.point_count, 10)
        self.assertTrue(scene.has_colors)
        self.assertTrue(scene.format_valid)
        self.assertIn("x", scene.properties)

    def test_load_binary(self):
        from opencut.core.gaussian_splat import load_splat
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_binary_ply(ply, 20, include_sh=True)
        scene = load_splat(ply)
        self.assertEqual(scene.point_count, 20)
        self.assertTrue(scene.has_sh_coeffs)

    def test_load_missing_file(self):
        from opencut.core.gaussian_splat import load_splat
        with self.assertRaises(FileNotFoundError):
            load_splat("/nonexistent.ply")

    def test_load_bounds_computed(self):
        from opencut.core.gaussian_splat import load_splat
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_ascii_ply(ply, 5)
        scene = load_splat(ply)
        self.assertIn("min", scene.bounds)
        self.assertIn("max", scene.bounds)
        self.assertIn("center", scene.bounds)

    def test_load_message_includes_count(self):
        from opencut.core.gaussian_splat import load_splat
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_ascii_ply(ply, 100)
        scene = load_splat(ply)
        self.assertIn("100", scene.message)


# ============================================================
# Camera Path Tests
# ============================================================

class TestCameraKeyframe(unittest.TestCase):
    """Tests for CameraKeyframe dataclass."""

    def test_defaults(self):
        from opencut.core.gaussian_splat import CameraKeyframe
        kf = CameraKeyframe()
        self.assertEqual(kf.position, (0.0, 0.0, 0.0))
        self.assertEqual(kf.fov, 60.0)
        self.assertEqual(kf.time, 0.0)

    def test_custom(self):
        from opencut.core.gaussian_splat import CameraKeyframe
        kf = CameraKeyframe(position=(1, 2, 3), rotation=(10, 20, 0), fov=90, time=1.5)
        self.assertEqual(kf.position, (1, 2, 3))
        self.assertEqual(kf.fov, 90)


class TestCameraPath(unittest.TestCase):
    """Tests for CameraPath and define_camera_path."""

    def test_empty_keyframes_raises(self):
        from opencut.core.gaussian_splat import define_camera_path
        with self.assertRaises(ValueError):
            define_camera_path([])

    def test_single_keyframe(self):
        from opencut.core.gaussian_splat import CameraKeyframe, define_camera_path
        path = define_camera_path([CameraKeyframe(time=0.0)])
        self.assertEqual(len(path.keyframes), 1)
        self.assertEqual(path.duration, 0.0)

    def test_sorts_by_time(self):
        from opencut.core.gaussian_splat import CameraKeyframe, define_camera_path
        kfs = [
            CameraKeyframe(time=2.0),
            CameraKeyframe(time=0.0),
            CameraKeyframe(time=1.0),
        ]
        path = define_camera_path(kfs)
        times = [kf.time for kf in path.keyframes]
        self.assertEqual(times, [0.0, 1.0, 2.0])

    def test_duration(self):
        from opencut.core.gaussian_splat import CameraKeyframe, define_camera_path
        kfs = [CameraKeyframe(time=0.0), CameraKeyframe(time=4.0)]
        path = define_camera_path(kfs, fps=30)
        self.assertEqual(path.duration, 4.0)
        self.assertEqual(path.total_frames, 120)

    def test_invalid_interpolation_defaults_to_linear(self):
        from opencut.core.gaussian_splat import CameraKeyframe, define_camera_path
        path = define_camera_path([CameraKeyframe()], interpolation="invalid")
        self.assertEqual(path.interpolation, "linear")

    def test_cubic_interpolation(self):
        from opencut.core.gaussian_splat import CameraKeyframe, define_camera_path
        kfs = [CameraKeyframe(time=i) for i in range(5)]
        path = define_camera_path(kfs, interpolation="cubic")
        self.assertEqual(path.interpolation, "cubic")


class TestOrbitPath(unittest.TestCase):
    """Tests for create_orbit_path."""

    def test_basic_orbit(self):
        from opencut.core.gaussian_splat import create_orbit_path
        path = create_orbit_path(center=(0, 0, 0), radius=5, frames=60, fps=30)
        self.assertEqual(len(path.keyframes), 60)
        self.assertTrue(path.loop)

    def test_orbit_radius(self):
        from opencut.core.gaussian_splat import create_orbit_path
        path = create_orbit_path(center=(0, 0, 0), radius=5, frames=60)
        # First keyframe should be at radius distance from center
        kf0 = path.keyframes[0]
        dist = math.sqrt(kf0.position[0]**2 + kf0.position[2]**2)
        self.assertAlmostEqual(dist, 5.0, places=3)

    def test_orbit_height(self):
        from opencut.core.gaussian_splat import create_orbit_path
        path = create_orbit_path(center=(0, 0, 0), radius=5, height=3.0, frames=60)
        for kf in path.keyframes:
            self.assertAlmostEqual(kf.position[1], 3.0, places=3)

    def test_orbit_loops(self):
        from opencut.core.gaussian_splat import create_orbit_path
        path = create_orbit_path(frames=60, loops=2)
        self.assertEqual(len(path.keyframes), 120)

    def test_orbit_invalid_radius(self):
        from opencut.core.gaussian_splat import create_orbit_path
        with self.assertRaises(ValueError):
            create_orbit_path(radius=0)

    def test_orbit_too_few_frames(self):
        from opencut.core.gaussian_splat import create_orbit_path
        with self.assertRaises(ValueError):
            create_orbit_path(frames=1)

    def test_orbit_custom_fov(self):
        from opencut.core.gaussian_splat import create_orbit_path
        path = create_orbit_path(fov=90.0, frames=10)
        for kf in path.keyframes:
            self.assertEqual(kf.fov, 90.0)

    def test_orbit_center_offset(self):
        from opencut.core.gaussian_splat import create_orbit_path
        path = create_orbit_path(center=(10, 5, 10), radius=3, frames=30)
        kf0 = path.keyframes[0]
        dx = kf0.position[0] - 10
        dz = kf0.position[2] - 10
        dist = math.sqrt(dx**2 + dz**2)
        self.assertAlmostEqual(dist, 3.0, places=3)


class TestInterpolation(unittest.TestCase):
    """Tests for camera interpolation helpers."""

    def test_lerp(self):
        from opencut.core.gaussian_splat import _lerp
        self.assertAlmostEqual(_lerp(0, 10, 0.5), 5.0)
        self.assertAlmostEqual(_lerp(0, 10, 0.0), 0.0)
        self.assertAlmostEqual(_lerp(0, 10, 1.0), 10.0)

    def test_lerp_tuple(self):
        from opencut.core.gaussian_splat import _lerp_tuple
        result = _lerp_tuple((0, 0, 0), (10, 20, 30), 0.5)
        self.assertAlmostEqual(result[0], 5.0)
        self.assertAlmostEqual(result[1], 10.0)
        self.assertAlmostEqual(result[2], 15.0)

    def test_catmull_rom_midpoint(self):
        from opencut.core.gaussian_splat import _catmull_rom
        # At t=0 should return p1
        val = _catmull_rom(0, 1, 2, 3, 0.0)
        self.assertAlmostEqual(val, 1.0)
        # At t=1 should return p2
        val = _catmull_rom(0, 1, 2, 3, 1.0)
        self.assertAlmostEqual(val, 2.0)

    def test_interpolate_single_keyframe(self):
        from opencut.core.gaussian_splat import CameraKeyframe, CameraPath, _interpolate_camera
        path = CameraPath(keyframes=[CameraKeyframe(position=(5, 5, 5), time=0)])
        cam = _interpolate_camera(path, 0.0)
        self.assertEqual(cam.position, (5, 5, 5))

    def test_interpolate_linear(self):
        from opencut.core.gaussian_splat import CameraKeyframe, CameraPath, _interpolate_camera
        path = CameraPath(
            keyframes=[
                CameraKeyframe(position=(0, 0, 0), time=0.0),
                CameraKeyframe(position=(10, 20, 30), time=2.0),
            ],
            interpolation="linear",
        )
        cam = _interpolate_camera(path, 1.0)
        self.assertAlmostEqual(cam.position[0], 5.0)
        self.assertAlmostEqual(cam.position[1], 10.0)

    def test_interpolate_before_start(self):
        from opencut.core.gaussian_splat import CameraKeyframe, CameraPath, _interpolate_camera
        path = CameraPath(
            keyframes=[CameraKeyframe(position=(1, 2, 3), time=1.0)],
        )
        cam = _interpolate_camera(path, 0.0)
        self.assertEqual(cam.position, (1, 2, 3))

    def test_interpolate_after_end(self):
        from opencut.core.gaussian_splat import CameraKeyframe, CameraPath, _interpolate_camera
        path = CameraPath(
            keyframes=[
                CameraKeyframe(position=(0, 0, 0), time=0.0),
                CameraKeyframe(position=(10, 10, 10), time=1.0),
            ],
        )
        cam = _interpolate_camera(path, 5.0)
        self.assertEqual(cam.position, (10, 10, 10))

    def test_interpolate_empty_path(self):
        from opencut.core.gaussian_splat import CameraPath, _interpolate_camera
        path = CameraPath(keyframes=[])
        cam = _interpolate_camera(path, 0.0)
        self.assertEqual(cam.position, (0.0, 0.0, 0.0))


class TestPointProjection(unittest.TestCase):
    """Tests for _project_point."""

    def test_point_in_front(self):
        from opencut.core.gaussian_splat import _project_point
        result = _project_point(
            point=(0, 0, 5),
            cam_pos=(0, 0, 0),
            cam_rot=(0, 0, 0),
            fov=60, width=800, height=600,
        )
        self.assertIsNotNone(result)
        sx, sy, depth = result
        # Should project near center
        self.assertTrue(350 < sx < 450)
        self.assertTrue(250 < sy < 350)

    def test_point_behind_camera(self):
        from opencut.core.gaussian_splat import _project_point
        result = _project_point(
            point=(0, 0, -5),
            cam_pos=(0, 0, 0),
            cam_rot=(0, 0, 0),
            fov=60, width=800, height=600,
        )
        self.assertIsNone(result)

    def test_point_off_screen(self):
        from opencut.core.gaussian_splat import _project_point
        result = _project_point(
            point=(1000, 0, 1),
            cam_pos=(0, 0, 0),
            cam_rot=(0, 0, 0),
            fov=60, width=800, height=600,
        )
        # Very far to the side — should be None (off screen)
        self.assertIsNone(result)


class TestRenderSplatToVideo(unittest.TestCase):
    """Tests for render_splat_to_video."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_missing_ply(self):
        from opencut.core.gaussian_splat import CameraKeyframe, CameraPath, render_splat_to_video
        path = CameraPath(keyframes=[CameraKeyframe(time=0)])
        with self.assertRaises(FileNotFoundError):
            render_splat_to_video("/nonexistent.ply", path)

    def test_empty_camera_path(self):
        from opencut.core.gaussian_splat import CameraPath, render_splat_to_video
        ply = os.path.join(self.tmp_dir, "test.ply")
        _make_ascii_ply(ply, 5)
        with self.assertRaises(ValueError):
            render_splat_to_video(ply, CameraPath())

    @patch("opencut.core.gaussian_splat.ensure_package", return_value=True)
    @patch("opencut.core.gaussian_splat.run_ffmpeg")
    def test_render_fallback_renderer(self, mock_ffmpeg, mock_pkg):
        from opencut.core.gaussian_splat import (
            CameraKeyframe,
            define_camera_path,
            render_splat_to_video,
        )
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_ascii_ply(ply, 5)

        kfs = [
            CameraKeyframe(position=(0, 0, 5), time=0.0),
            CameraKeyframe(position=(5, 0, 5), time=0.5),
        ]
        camera_path = define_camera_path(kfs, fps=10)

        # Mock PIL Image to avoid actual rendering
        mock_img = MagicMock()
        with patch("opencut.core.gaussian_splat._render_frame_pillow", return_value=mock_img):
            result = render_splat_to_video(
                ply_path=ply,
                camera_path=camera_path,
                resolution=(320, 240),
            )

        self.assertEqual(result.renderer, "pillow_fallback")
        self.assertGreater(result.frames, 0)
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.gaussian_splat.ensure_package", return_value=True)
    @patch("opencut.core.gaussian_splat.run_ffmpeg")
    def test_render_progress_callback(self, mock_ffmpeg, mock_pkg):
        from opencut.core.gaussian_splat import (
            CameraKeyframe,
            define_camera_path,
            render_splat_to_video,
        )
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_ascii_ply(ply, 5)

        kfs = [
            CameraKeyframe(position=(0, 0, 5), time=0.0),
            CameraKeyframe(position=(5, 0, 5), time=0.3),
        ]
        camera_path = define_camera_path(kfs, fps=10)

        progress_calls = []
        def _on_progress(pct, msg):
            progress_calls.append((pct, msg))

        mock_img = MagicMock()
        with patch("opencut.core.gaussian_splat._render_frame_pillow", return_value=mock_img):
            render_splat_to_video(
                ply_path=ply,
                camera_path=camera_path,
                on_progress=_on_progress,
                resolution=(320, 240),
            )

        self.assertTrue(len(progress_calls) > 0)
        self.assertEqual(progress_calls[-1][0], 100)


class TestRenderSplatFrame(unittest.TestCase):
    """Tests for render_splat_frame."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_missing_ply(self):
        from opencut.core.gaussian_splat import CameraKeyframe, render_splat_frame
        with self.assertRaises(FileNotFoundError):
            render_splat_frame("/nonexistent.ply", CameraKeyframe())

    @patch("opencut.core.gaussian_splat.ensure_package", return_value=True)
    def test_render_frame_returns_png(self, mock_pkg):
        from opencut.core.gaussian_splat import CameraKeyframe, render_splat_frame
        ply = os.path.join(self.tmp_dir, "scene.ply")
        _make_ascii_ply(ply, 5)

        mock_img = MagicMock()
        with patch("opencut.core.gaussian_splat._render_frame_pillow", return_value=mock_img):
            result = render_splat_frame(
                ply_path=ply,
                camera=CameraKeyframe(position=(0, 0, 5)),
                resolution=(320, 240),
            )
        self.assertTrue(result.endswith(".png"))


# ============================================================
# SplatScene Dataclass Tests
# ============================================================

class TestSplatSceneDataclass(unittest.TestCase):
    """Tests for SplatScene dataclass defaults."""

    def test_defaults(self):
        from opencut.core.gaussian_splat import SplatScene
        scene = SplatScene()
        self.assertEqual(scene.ply_path, "")
        self.assertEqual(scene.point_count, 0)
        self.assertFalse(scene.has_colors)
        self.assertFalse(scene.format_valid)

    def test_custom(self):
        from opencut.core.gaussian_splat import SplatScene
        scene = SplatScene(
            ply_path="/test.ply",
            point_count=1000,
            has_colors=True,
            format_valid=True,
        )
        self.assertEqual(scene.point_count, 1000)
        self.assertTrue(scene.has_colors)


class TestSplatRenderResult(unittest.TestCase):
    """Tests for SplatRenderResult dataclass."""

    def test_defaults(self):
        from opencut.core.gaussian_splat import SplatRenderResult
        result = SplatRenderResult()
        self.assertEqual(result.output_path, "")
        self.assertEqual(result.frames, 0)
        self.assertEqual(result.resolution, (1280, 720))

    def test_custom(self):
        from opencut.core.gaussian_splat import SplatRenderResult
        result = SplatRenderResult(
            output_path="/out.mp4",
            duration=4.0,
            frames=120,
            renderer="pillow_fallback",
        )
        self.assertEqual(result.renderer, "pillow_fallback")


if __name__ == "__main__":
    unittest.main()
