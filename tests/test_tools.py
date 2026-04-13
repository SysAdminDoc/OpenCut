"""
Tests for OpenCut production tools features.

Covers:
  - Cursor zoom detection & application (cursor_zoom.py)
  - Lower-thirds generation & batch (lower_thirds.py)
  - Beat-synced auto-cuts (beat_cuts.py)
  - Selective redaction (redaction.py)
  - Vertical-first intelligent reframe (smart_reframe.py)
  - Telemetry data overlay (telemetry_overlay.py)
  - Tools routes (tools_routes.py)
"""

import csv
import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Cursor Zoom Tests
# ============================================================
class TestCursorZoom(unittest.TestCase):
    """Tests for opencut.core.cursor_zoom module."""

    def test_click_region_dataclass(self):
        """ClickRegion should hold timestamp, x, y, confidence."""
        from opencut.core.cursor_zoom import ClickRegion
        cr = ClickRegion(timestamp=5.0, x=400, y=300, confidence=0.85)
        self.assertEqual(cr.timestamp, 5.0)
        self.assertEqual(cr.x, 400)
        self.assertEqual(cr.y, 300)
        self.assertEqual(cr.confidence, 0.85)

    def test_click_region_defaults(self):
        """ClickRegion confidence should default to 0.0."""
        from opencut.core.cursor_zoom import ClickRegion
        cr = ClickRegion(timestamp=1.0, x=100, y=200)
        self.assertEqual(cr.confidence, 0.0)

    def test_cursor_zoom_result_dataclass(self):
        """CursorZoomResult should have expected fields."""
        from opencut.core.cursor_zoom import CursorZoomResult
        r = CursorZoomResult(output_path="/tmp/out.mp4", zoom_factor=2.5)
        self.assertEqual(r.output_path, "/tmp/out.mp4")
        self.assertEqual(r.zoom_factor, 2.5)
        self.assertIsInstance(r.click_regions, list)

    def test_ease_in_out(self):
        """Ease function should return 0 at t=0 and 1 at t=1."""
        from opencut.core.cursor_zoom import _ease_in_out
        self.assertAlmostEqual(_ease_in_out(0.0), 0.0)
        self.assertAlmostEqual(_ease_in_out(1.0), 1.0)
        # Midpoint should be 0.5
        self.assertAlmostEqual(_ease_in_out(0.5), 0.5)

    def test_ease_in_out_clamping(self):
        """Ease function should clamp values outside [0,1]."""
        from opencut.core.cursor_zoom import _ease_in_out
        self.assertAlmostEqual(_ease_in_out(-0.5), 0.0)
        self.assertAlmostEqual(_ease_in_out(1.5), 1.0)

    def test_center_of_mass(self):
        """Center of mass should be weighted average of block positions."""
        from opencut.core.cursor_zoom import _center_of_mass
        blocks = [
            {"x": 100, "y": 100, "energy": 10},
            {"x": 200, "y": 200, "energy": 10},
        ]
        com = _center_of_mass(blocks)
        self.assertIsNotNone(com)
        self.assertEqual(com["x"], 150)
        self.assertEqual(com["y"], 150)

    def test_center_of_mass_empty(self):
        """Center of mass should return None for empty blocks."""
        from opencut.core.cursor_zoom import _center_of_mass
        self.assertIsNone(_center_of_mass([]))

    def test_center_of_mass_weighted(self):
        """Center of mass should weight toward higher energy."""
        from opencut.core.cursor_zoom import _center_of_mass
        blocks = [
            {"x": 0, "y": 0, "energy": 1},
            {"x": 100, "y": 100, "energy": 99},
        ]
        com = _center_of_mass(blocks)
        self.assertIsNotNone(com)
        self.assertGreater(com["x"], 90)
        self.assertGreater(com["y"], 90)

    def test_build_zoom_segments_empty_clicks(self):
        """With no click regions, only a passthrough segment should be returned."""
        from opencut.core.cursor_zoom import _build_zoom_segments
        info = {"width": 1920, "height": 1080, "fps": 30, "duration": 10.0}
        segs = _build_zoom_segments("input.mp4", [], 2.0, 1.5, info)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["type"], "passthrough")

    def test_build_zoom_segments_single_click(self):
        """A single click should produce zoom segments plus passthrough."""
        from opencut.core.cursor_zoom import ClickRegion, _build_zoom_segments
        info = {"width": 1920, "height": 1080, "fps": 30, "duration": 10.0}
        clicks = [ClickRegion(timestamp=5.0, x=960, y=540, confidence=0.9)]
        segs = _build_zoom_segments("input.mp4", clicks, 2.0, 1.5, info)
        self.assertGreater(len(segs), 1)
        zoom_segs = [s for s in segs if s["type"] == "zoom"]
        self.assertGreater(len(zoom_segs), 0)

    def test_detect_click_regions_signature(self):
        """detect_click_regions should accept input_path and on_progress."""
        from opencut.core.cursor_zoom import detect_click_regions
        sig = inspect.signature(detect_click_regions)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_apply_cursor_zoom_signature(self):
        """apply_cursor_zoom should accept required parameters."""
        from opencut.core.cursor_zoom import apply_cursor_zoom
        sig = inspect.signature(apply_cursor_zoom)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("click_regions", sig.parameters)
        self.assertIn("zoom_factor", sig.parameters)
        self.assertIn("zoom_duration", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_diff_frames_zero_diff(self):
        """Identical frames should produce no significant diff blocks."""
        from opencut.core.cursor_zoom import _diff_frames
        w, h = 64, 64
        frame = bytes([128] * w * h * 3)
        blocks = _diff_frames(frame, frame, w, h, block_size=16)
        self.assertEqual(len(blocks), 0)

    def test_diff_frames_detects_change(self):
        """Different frames should produce diff blocks."""
        from opencut.core.cursor_zoom import _diff_frames
        w, h = 64, 64
        frame_a = bytes([0] * w * h * 3)
        frame_b = bytes([255] * w * h * 3)
        blocks = _diff_frames(frame_a, frame_b, w, h, block_size=16)
        self.assertGreater(len(blocks), 0)


# ============================================================
# Lower-Thirds Tests
# ============================================================
class TestLowerThirds(unittest.TestCase):
    """Tests for opencut.core.lower_thirds module."""

    def test_styles_defined(self):
        """All four style presets should be defined."""
        from opencut.core.lower_thirds import STYLES
        for style in ("modern", "corporate", "news", "minimal"):
            self.assertIn(style, STYLES)
            self.assertIn("bg_color", STYLES[style])
            self.assertIn("name_font_size", STYLES[style])

    def test_lower_third_entry_dataclass(self):
        """LowerThirdEntry should store name, title, organization, timestamp."""
        from opencut.core.lower_thirds import LowerThirdEntry
        entry = LowerThirdEntry(name="Jane Smith", title="CEO", organization="Acme", timestamp=10.5)
        self.assertEqual(entry.name, "Jane Smith")
        self.assertEqual(entry.title, "CEO")
        self.assertEqual(entry.timestamp, 10.5)

    def test_batch_lower_third_result_dataclass(self):
        """BatchLowerThirdResult should have expected fields."""
        from opencut.core.lower_thirds import BatchLowerThirdResult
        r = BatchLowerThirdResult(output_dir="/tmp", count=3, style="news")
        self.assertEqual(r.count, 3)
        self.assertEqual(r.style, "news")
        self.assertIsInstance(r.files, list)

    def test_parse_csv_data_basic(self):
        """parse_csv_data should parse a valid CSV file."""
        from opencut.core.lower_thirds import parse_csv_data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Title", "Organization", "Timestamp"])
            writer.writerow(["Jane Smith", "CEO", "Acme Corp", "5.0"])
            writer.writerow(["John Doe", "CTO", "Acme Corp", "15.0"])
            path = f.name

        try:
            entries = parse_csv_data(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["name"], "Jane Smith")
            self.assertEqual(entries[0]["title"], "CEO")
            self.assertEqual(entries[0]["organization"], "Acme Corp")
            self.assertAlmostEqual(entries[0]["timestamp"], 5.0)
            self.assertEqual(entries[1]["name"], "John Doe")
        finally:
            os.unlink(path)

    def test_parse_csv_data_missing_name_skipped(self):
        """Rows without a name should be skipped."""
        from opencut.core.lower_thirds import parse_csv_data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Title"])
            writer.writerow(["Jane", "CEO"])
            writer.writerow(["", "Intern"])
            writer.writerow(["Bob", "CTO"])
            path = f.name

        try:
            entries = parse_csv_data(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0]["name"], "Jane")
            self.assertEqual(entries[1]["name"], "Bob")
        finally:
            os.unlink(path)

    def test_parse_csv_data_file_not_found(self):
        """parse_csv_data should raise FileNotFoundError for missing files."""
        from opencut.core.lower_thirds import parse_csv_data
        with self.assertRaises(FileNotFoundError):
            parse_csv_data("/nonexistent/data.csv")

    def test_parse_csv_data_case_insensitive(self):
        """Column headers should be case-insensitive."""
        from opencut.core.lower_thirds import parse_csv_data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["NAME", "TITLE", "ORGANIZATION"])
            writer.writerow(["Alice", "VP", "BigCo"])
            path = f.name

        try:
            entries = parse_csv_data(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["name"], "Alice")
        finally:
            os.unlink(path)

    def test_generate_lower_third_signature(self):
        """generate_lower_third should accept all expected parameters."""
        from opencut.core.lower_thirds import generate_lower_third
        sig = inspect.signature(generate_lower_third)
        for param in ("name", "title", "organization", "style", "duration", "width", "height", "on_progress"):
            self.assertIn(param, sig.parameters)

    def test_batch_lower_thirds_signature(self):
        """batch_lower_thirds should accept data_source and style."""
        from opencut.core.lower_thirds import batch_lower_thirds
        sig = inspect.signature(batch_lower_thirds)
        self.assertIn("data_source", sig.parameters)
        self.assertIn("style", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_batch_lower_thirds_empty_raises(self):
        """batch_lower_thirds should raise ValueError for empty data source."""
        from opencut.core.lower_thirds import batch_lower_thirds
        with self.assertRaises(ValueError):
            batch_lower_thirds(data_source=[], style="modern")


# ============================================================
# Beat-Synced Cuts Tests
# ============================================================
class TestBeatCuts(unittest.TestCase):
    """Tests for opencut.core.beat_cuts module."""

    def test_beat_marker_dataclass(self):
        """BeatMarker should store time, strength, bar_position."""
        from opencut.core.beat_cuts import BeatMarker
        bm = BeatMarker(time=1.5, strength=0.9, bar_position=2)
        self.assertEqual(bm.time, 1.5)
        self.assertEqual(bm.strength, 0.9)
        self.assertEqual(bm.bar_position, 2)

    def test_beat_marker_defaults(self):
        """BeatMarker should have sensible defaults."""
        from opencut.core.beat_cuts import BeatMarker
        bm = BeatMarker(time=2.0)
        self.assertEqual(bm.strength, 1.0)
        self.assertEqual(bm.bar_position, 0)

    def test_cut_entry_dataclass(self):
        """CutEntry should store clip_path, start, duration, beat_time."""
        from opencut.core.beat_cuts import CutEntry
        ce = CutEntry(clip_path="/tmp/clip.mp4", start=0.0, duration=0.5, beat_time=2.5)
        self.assertEqual(ce.clip_path, "/tmp/clip.mp4")
        self.assertEqual(ce.duration, 0.5)
        self.assertEqual(ce.beat_time, 2.5)

    def test_beat_cut_result_dataclass(self):
        """BeatCutResult should have cuts, beats, total_duration, density, bpm."""
        from opencut.core.beat_cuts import BeatCutResult
        r = BeatCutResult()
        self.assertIsInstance(r.cuts, list)
        self.assertIsInstance(r.beats, list)
        self.assertEqual(r.total_duration, 0.0)
        self.assertEqual(r.density, "every_beat")
        self.assertEqual(r.bpm, 0.0)

    def test_generate_beat_cut_list_no_clips_raises(self):
        """generate_beat_cut_list should raise ValueError if no clips."""
        from opencut.core.beat_cuts import generate_beat_cut_list
        with self.assertRaises(ValueError):
            generate_beat_cut_list("/tmp/music.mp3", clip_paths=[])

    def test_generate_beat_cut_list_signature(self):
        """generate_beat_cut_list should accept expected parameters."""
        from opencut.core.beat_cuts import generate_beat_cut_list
        sig = inspect.signature(generate_beat_cut_list)
        for param in ("music_path", "clip_paths", "density", "on_progress"):
            self.assertIn(param, sig.parameters)

    def test_assemble_beat_synced_no_cuts_raises(self):
        """assemble_beat_synced should raise ValueError if cut_list is empty."""
        from opencut.core.beat_cuts import assemble_beat_synced
        with self.assertRaises(ValueError):
            assemble_beat_synced("/tmp/music.mp3", cut_list=[])

    def test_assemble_beat_synced_signature(self):
        """assemble_beat_synced should accept expected parameters."""
        from opencut.core.beat_cuts import assemble_beat_synced
        sig = inspect.signature(assemble_beat_synced)
        for param in ("music_path", "cut_list", "output_path_str", "transition", "on_progress"):
            self.assertIn(param, sig.parameters)

    @patch("subprocess.run")
    def test_get_audio_duration_invalid(self, mock_run):
        """_get_audio_duration should return 0 for failed ffprobe."""
        from opencut.core.beat_cuts import _get_audio_duration
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        dur = _get_audio_duration("/nonexistent/file.mp3")
        self.assertEqual(dur, 0.0)


# ============================================================
# Redaction Tests
# ============================================================
class TestRedaction(unittest.TestCase):
    """Tests for opencut.core.redaction module."""

    def test_redaction_region_dataclass(self):
        """RedactionRegion should store x, y, w, h, start_time, end_time."""
        from opencut.core.redaction import RedactionRegion
        rr = RedactionRegion(x=100, y=200, w=50, h=50, start_time=1.0, end_time=5.0)
        self.assertEqual(rr.x, 100)
        self.assertEqual(rr.y, 200)
        self.assertEqual(rr.w, 50)
        self.assertEqual(rr.end_time, 5.0)

    def test_redaction_region_defaults(self):
        """RedactionRegion end_time should default to -1 (end of video)."""
        from opencut.core.redaction import RedactionRegion
        rr = RedactionRegion(x=0, y=0, w=100, h=100)
        self.assertEqual(rr.start_time, 0.0)
        self.assertEqual(rr.end_time, -1.0)

    def test_redaction_result_dataclass(self):
        """RedactionResult should have expected fields."""
        from opencut.core.redaction import RedactionResult
        r = RedactionResult(output_path="/tmp/out.mp4", regions_count=3, method="blur")
        self.assertEqual(r.regions_count, 3)
        self.assertEqual(r.method, "blur")

    @patch("opencut.core.redaction.get_video_info")
    def test_redact_region_no_regions_raises(self, mock_info):
        """redact_region should raise ValueError with empty regions."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 10.0}
        from opencut.core.redaction import redact_region
        with self.assertRaises(ValueError):
            redact_region("/tmp/input.mp4", regions=[])

    def test_redact_region_signature(self):
        """redact_region should accept expected parameters."""
        from opencut.core.redaction import redact_region
        sig = inspect.signature(redact_region)
        for param in ("input_path", "regions", "method", "output_path_str", "on_progress"):
            self.assertIn(param, sig.parameters)

    def test_redact_faces_signature(self):
        """redact_faces should accept expected parameters."""
        from opencut.core.redaction import redact_faces
        sig = inspect.signature(redact_faces)
        for param in ("input_path", "method", "output_path_str", "on_progress"):
            self.assertIn(param, sig.parameters)

    def test_build_blur_filter_single_region(self):
        """Blur filter should produce split/crop/boxblur/overlay chain."""
        from opencut.core.redaction import RedactionRegion, _build_blur_filter
        regions = [RedactionRegion(x=100, y=100, w=200, h=200, start_time=0, end_time=10)]
        result = _build_blur_filter(regions, 1920, 1080, 30.0, 20)
        self.assertIn("split", result)
        self.assertIn("boxblur", result)
        self.assertIn("overlay", result)
        self.assertIn("[out]", result)

    def test_build_blur_filter_multiple_regions(self):
        """Blur filter should chain multiple regions."""
        from opencut.core.redaction import RedactionRegion, _build_blur_filter
        regions = [
            RedactionRegion(x=0, y=0, w=100, h=100, start_time=0, end_time=5),
            RedactionRegion(x=500, y=500, w=100, h=100, start_time=2, end_time=8),
        ]
        result = _build_blur_filter(regions, 1920, 1080, 30.0)
        self.assertIn("[out]", result)
        # Should have two split operations
        self.assertIn("split[base0]", result)
        self.assertIn("split[base1]", result)

    def test_build_pixelate_filter(self):
        """Pixelate filter should produce crop/scale-down/scale-up/overlay chain."""
        from opencut.core.redaction import RedactionRegion, _build_pixelate_filter
        regions = [RedactionRegion(x=100, y=100, w=200, h=200, start_time=0, end_time=10)]
        result = _build_pixelate_filter(regions, 1920, 1080, 30.0)
        self.assertIn("scale=", result)
        self.assertIn("overlay", result)
        self.assertIn("[out]", result)

    def test_generate_redaction_log(self):
        """generate_redaction_log should produce audit log dict."""
        from opencut.core.redaction import generate_redaction_log
        regions = [
            {"x": 100, "y": 100, "w": 200, "h": 200, "start_time": 0, "end_time": 10},
        ]
        with patch("opencut.core.redaction.get_video_info") as mock_info:
            mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 30.0}
            tmp = tempfile.mktemp(suffix=".json")
            try:
                result = generate_redaction_log(regions, "/tmp/video.mp4", method="blur", output_path_str=tmp)
                self.assertEqual(result["region_count"], 1)
                self.assertIn("log", result)
                self.assertIn("redaction_log", result["log"])
                self.assertEqual(result["log"]["redaction_log"]["method"], "blur")
                # Verify file was written
                self.assertTrue(os.path.isfile(tmp))
                with open(tmp) as f:
                    saved = json.load(f)
                self.assertIn("redaction_log", saved)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def test_generate_redaction_log_with_dataclass(self):
        """generate_redaction_log should accept RedactionRegion objects."""
        from opencut.core.redaction import RedactionRegion, generate_redaction_log
        regions = [RedactionRegion(x=50, y=50, w=100, h=100, start_time=1, end_time=5)]
        with patch("opencut.core.redaction.get_video_info") as mock_info:
            mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 30.0}
            tmp = tempfile.mktemp(suffix=".json")
            try:
                result = generate_redaction_log(regions, "/tmp/video.mp4", output_path_str=tmp)
                self.assertEqual(result["region_count"], 1)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)


# ============================================================
# Smart Reframe Tests
# ============================================================
class TestSmartReframe(unittest.TestCase):
    """Tests for opencut.core.smart_reframe module."""

    def test_parse_aspect_standard(self):
        """Standard aspect ratios should parse correctly."""
        from opencut.core.smart_reframe import _parse_aspect
        self.assertEqual(_parse_aspect("9:16"), (9, 16))
        self.assertEqual(_parse_aspect("4:5"), (4, 5))
        self.assertEqual(_parse_aspect("1:1"), (1, 1))
        self.assertEqual(_parse_aspect("16:9"), (16, 9))

    def test_parse_aspect_invalid(self):
        """Invalid aspect strings should return default 9:16."""
        from opencut.core.smart_reframe import _parse_aspect
        self.assertEqual(_parse_aspect("invalid"), (9, 16))
        self.assertEqual(_parse_aspect(""), (9, 16))

    def test_calc_crop_dims_landscape_to_portrait(self):
        """Crop dimensions for 1920x1080 to 9:16 should crop width."""
        from opencut.core.smart_reframe import _calc_crop_dims
        crop_w, crop_h = _calc_crop_dims(1920, 1080, 9, 16)
        self.assertEqual(crop_h, 1080)
        self.assertLess(crop_w, 1920)
        # Verify aspect ratio
        actual_ratio = crop_w / crop_h
        expected_ratio = 9 / 16
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=1)

    def test_calc_crop_dims_square(self):
        """Crop dimensions for 1:1 from landscape should crop width to height."""
        from opencut.core.smart_reframe import _calc_crop_dims
        crop_w, crop_h = _calc_crop_dims(1920, 1080, 1, 1)
        self.assertEqual(crop_w, crop_h)
        self.assertLessEqual(crop_w, 1080)

    def test_calc_crop_dims_already_portrait(self):
        """Portrait source to portrait target should crop height if needed."""
        from opencut.core.smart_reframe import _calc_crop_dims
        crop_w, crop_h = _calc_crop_dims(1080, 1920, 9, 16)
        self.assertLessEqual(crop_w, 1080)
        self.assertLessEqual(crop_h, 1920)

    def test_calc_crop_dims_even(self):
        """Crop dimensions should always be even numbers."""
        from opencut.core.smart_reframe import _calc_crop_dims
        for src_w, src_h in [(1920, 1080), (1280, 720), (3840, 2160), (1001, 501)]:
            crop_w, crop_h = _calc_crop_dims(src_w, src_h, 9, 16)
            self.assertEqual(crop_w % 2, 0, f"crop_w={crop_w} not even for {src_w}x{src_h}")
            self.assertEqual(crop_h % 2, 0, f"crop_h={crop_h} not even for {src_w}x{src_h}")

    def test_reframe_result_dataclass(self):
        """ReframeResult should have expected fields."""
        from opencut.core.smart_reframe import ReframeResult
        r = ReframeResult(output_path="/tmp/out.mp4", method="center", target_aspect="9:16")
        self.assertEqual(r.method, "center")
        self.assertEqual(r.target_aspect, "9:16")

    def test_smooth_positions_single(self):
        """Smoothing single position should return it unchanged."""
        from opencut.core.smart_reframe import _smooth_positions
        positions = [{"time": 0, "x": 100, "y": 200, "w": 50, "h": 50}]
        smoothed = _smooth_positions(positions, 10.0)
        self.assertEqual(len(smoothed), 1)
        self.assertEqual(smoothed[0]["x"], 100)

    def test_smooth_positions_multiple(self):
        """Smoothing multiple positions should average neighbors."""
        from opencut.core.smart_reframe import _smooth_positions
        positions = [
            {"time": 0, "x": 0, "y": 0, "w": 50, "h": 50},
            {"time": 1, "x": 100, "y": 100, "w": 50, "h": 50},
            {"time": 2, "x": 200, "y": 200, "w": 50, "h": 50},
        ]
        smoothed = _smooth_positions(positions, 10.0)
        self.assertEqual(len(smoothed), 3)
        # Middle point should be averaged with neighbors
        self.assertEqual(smoothed[1]["x"], 100)  # (0+100+200)/3 = 100

    def test_reframe_vertical_signature(self):
        """reframe_vertical should accept expected parameters."""
        from opencut.core.smart_reframe import reframe_vertical
        sig = inspect.signature(reframe_vertical)
        for param in ("input_path", "target_aspect", "method", "output_path_str", "on_progress"):
            self.assertIn(param, sig.parameters)

    @patch("opencut.core.smart_reframe.run_ffmpeg")
    @patch("opencut.core.smart_reframe.get_video_info")
    def test_reframe_center_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        """Center reframe should call FFmpeg with crop filter."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 10.0}
        from opencut.core.smart_reframe import _reframe_center
        result = _reframe_center("/tmp/in.mp4", mock_info.return_value, 608, 1080, "/tmp/out.mp4")
        self.assertTrue(mock_ffmpeg.called)
        # Check crop was in the command
        cmd_args = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd_args.index("-vf")
        self.assertIn("crop=", cmd_args[vf_idx + 1])
        self.assertEqual(result["method"], "center")


# ============================================================
# Telemetry Overlay Tests
# ============================================================
class TestTelemetryOverlay(unittest.TestCase):
    """Tests for opencut.core.telemetry_overlay module."""

    def test_telemetry_frame_dataclass(self):
        """TelemetryFrame should store all expected fields."""
        from opencut.core.telemetry_overlay import TelemetryFrame
        tf = TelemetryFrame(
            timestamp=1.0, latitude=40.7128, longitude=-74.0060,
            altitude=100.5, speed=15.3, battery=85.0, gimbal_angle=-30.0,
        )
        self.assertAlmostEqual(tf.latitude, 40.7128)
        self.assertAlmostEqual(tf.altitude, 100.5)
        self.assertEqual(tf.battery, 85.0)

    def test_telemetry_frame_defaults(self):
        """TelemetryFrame should default all numeric fields to 0."""
        from opencut.core.telemetry_overlay import TelemetryFrame
        tf = TelemetryFrame()
        self.assertEqual(tf.timestamp, 0.0)
        self.assertEqual(tf.latitude, 0.0)
        self.assertEqual(tf.speed, 0.0)

    def test_parse_dji_srt_basic(self):
        """parse_dji_srt should extract telemetry from DJI SRT format."""
        from opencut.core.telemetry_overlay import parse_dji_srt
        srt_content = (
            "1\n"
            "00:00:01,000 --> 00:00:02,000\n"
            "[latitude: 40.7128] [longitude: -74.0060] [altitude: 100.5] [speed: 15.3] [battery: 85%]\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,000\n"
            "[latitude: 40.7130] [longitude: -74.0058] [altitude: 105.2] [speed: 16.1] [battery: 84%]\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(srt_content)
            path = f.name

        try:
            frames = parse_dji_srt(path)
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0].timestamp, 1.0)
            self.assertAlmostEqual(frames[0].latitude, 40.7128)
            self.assertAlmostEqual(frames[0].longitude, -74.0060)
            self.assertAlmostEqual(frames[0].altitude, 100.5)
            self.assertAlmostEqual(frames[0].speed, 15.3)
            self.assertAlmostEqual(frames[0].battery, 85.0)
            self.assertAlmostEqual(frames[1].altitude, 105.2)
        finally:
            os.unlink(path)

    def test_parse_dji_srt_file_not_found(self):
        """parse_dji_srt should raise FileNotFoundError."""
        from opencut.core.telemetry_overlay import parse_dji_srt
        with self.assertRaises(FileNotFoundError):
            parse_dji_srt("/nonexistent/drone.srt")

    def test_parse_dji_srt_empty_file(self):
        """parse_dji_srt should handle empty files."""
        from opencut.core.telemetry_overlay import parse_dji_srt
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            frames = parse_dji_srt(path)
            self.assertEqual(frames, [])
        finally:
            os.unlink(path)

    def test_parse_dji_srt_with_gimbal(self):
        """parse_dji_srt should parse gimbal_angle field."""
        from opencut.core.telemetry_overlay import parse_dji_srt
        srt_content = (
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "[latitude: 40.71] [longitude: -74.00] [altitude: 50] [gimbal_angle: -45.5]\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(srt_content)
            path = f.name
        try:
            frames = parse_dji_srt(path)
            self.assertEqual(len(frames), 1)
            self.assertAlmostEqual(frames[0].gimbal_angle, -45.5)
        finally:
            os.unlink(path)

    def test_parse_telemetry_csv_basic(self):
        """parse_telemetry_csv should parse a standard telemetry CSV."""
        from opencut.core.telemetry_overlay import parse_telemetry_csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "latitude", "longitude", "altitude", "speed"])
            writer.writerow(["0.0", "40.7128", "-74.006", "100", "15"])
            writer.writerow(["1.0", "40.713", "-74.005", "105", "16"])
            path = f.name

        try:
            frames = parse_telemetry_csv(path)
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0].latitude, 40.7128)
            self.assertAlmostEqual(frames[1].altitude, 105.0)
        finally:
            os.unlink(path)

    def test_parse_telemetry_csv_aliases(self):
        """parse_telemetry_csv should accept column aliases like 'lat', 'alt'."""
        from opencut.core.telemetry_overlay import parse_telemetry_csv
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "lat", "lon", "alt", "speed"])
            writer.writerow(["0.5", "35.0", "139.0", "200", "20"])
            path = f.name

        try:
            frames = parse_telemetry_csv(path)
            self.assertEqual(len(frames), 1)
            self.assertAlmostEqual(frames[0].timestamp, 0.5)
            self.assertAlmostEqual(frames[0].latitude, 35.0)
            self.assertAlmostEqual(frames[0].longitude, 139.0)
        finally:
            os.unlink(path)

    def test_parse_telemetry_csv_file_not_found(self):
        """parse_telemetry_csv should raise FileNotFoundError."""
        from opencut.core.telemetry_overlay import parse_telemetry_csv
        with self.assertRaises(FileNotFoundError):
            parse_telemetry_csv("/nonexistent/telem.csv")

    def test_field_formatters(self):
        """Field formatters should produce expected string formats."""
        from opencut.core.telemetry_overlay import FIELD_FORMATTERS, TelemetryFrame
        tf = TelemetryFrame(altitude=150.3, speed=12.5, latitude=40.7, longitude=-74.0, battery=80, gimbal_angle=-30)

        self.assertIn("150.3", FIELD_FORMATTERS["altitude"](tf))
        self.assertIn("12.5", FIELD_FORMATTERS["speed"](tf))
        self.assertIn("40.7", FIELD_FORMATTERS["gps"](tf))
        self.assertIn("80", FIELD_FORMATTERS["battery"](tf))
        self.assertIn("-30", FIELD_FORMATTERS["gimbal"](tf))

    def test_escape_drawtext(self):
        """Drawtext escaper should handle colons and special chars."""
        from opencut.core.telemetry_overlay import _escape_drawtext
        self.assertEqual(_escape_drawtext("GPS: 40.7, -74.0"), "GPS\\: 40.7, -74.0")
        self.assertEqual(_escape_drawtext("100%"), "100%%")

    def test_position_presets(self):
        """All four position presets should be defined."""
        from opencut.core.telemetry_overlay import POSITION_PRESETS
        for pos in ("bottom-left", "bottom-right", "top-left", "top-right"):
            self.assertIn(pos, POSITION_PRESETS)

    def test_overlay_telemetry_signature(self):
        """overlay_telemetry should accept expected parameters."""
        from opencut.core.telemetry_overlay import overlay_telemetry
        sig = inspect.signature(overlay_telemetry)
        for param in ("video_path", "telemetry", "fields", "position", "font_size", "on_progress"):
            self.assertIn(param, sig.parameters)

    @patch("opencut.core.telemetry_overlay.get_video_info")
    def test_overlay_telemetry_no_data_raises(self, mock_info):
        """overlay_telemetry should raise ValueError with empty telemetry."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 10.0}
        from opencut.core.telemetry_overlay import overlay_telemetry
        with self.assertRaises(ValueError):
            overlay_telemetry("/tmp/video.mp4", telemetry=[])

    def test_build_drawtext_filters_basic(self):
        """Drawtext filter builder should produce valid filter chain."""
        from opencut.core.telemetry_overlay import TelemetryFrame, _build_drawtext_filters
        frames = [TelemetryFrame(timestamp=0, altitude=100, speed=10)]
        result = _build_drawtext_filters(frames, ["altitude", "speed"], "bottom-left", 18, "white", "black@0.5", 10.0)
        self.assertIn("drawtext=", result)
        self.assertIn("ALT", result)
        self.assertIn("SPD", result)

    def test_build_drawtext_filters_empty(self):
        """Empty telemetry should produce null filter."""
        from opencut.core.telemetry_overlay import _build_drawtext_filters
        result = _build_drawtext_filters([], ["altitude"], "bottom-left", 18, "white", "black@0.5", 10.0)
        self.assertEqual(result, "null")


# ============================================================
# Routes Tests
# ============================================================
class TestToolsRoutes(unittest.TestCase):
    """Tests for tools route registration and endpoint availability."""

    def test_tools_bp_exists(self):
        """tools_bp should be importable."""
        from opencut.routes.tools_routes import tools_bp
        self.assertEqual(tools_bp.name, "tools")

    def test_tools_bp_registered(self):
        """tools_bp should be registered in the app."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        # Collect all registered blueprint names
        bp_names = [bp.name for bp in app.iter_blueprints()]
        self.assertIn("tools", bp_names)

    def test_cursor_zoom_endpoint_exists(self):
        """POST /screen/cursor-zoom should be a registered endpoint."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/screen/cursor-zoom", rules)

    def test_lower_thirds_generate_endpoint_exists(self):
        """POST /lower-thirds/generate should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/lower-thirds/generate", rules)

    def test_lower_thirds_batch_endpoint_exists(self):
        """POST /lower-thirds/batch should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/lower-thirds/batch", rules)

    def test_beat_cuts_generate_endpoint_exists(self):
        """POST /beat-cuts/generate should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/beat-cuts/generate", rules)

    def test_beat_cuts_assemble_endpoint_exists(self):
        """POST /beat-cuts/assemble should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/beat-cuts/assemble", rules)

    def test_redact_region_endpoint_exists(self):
        """POST /redact/region should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/redact/region", rules)

    def test_redact_faces_endpoint_exists(self):
        """POST /redact/faces should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/redact/faces", rules)

    def test_reframe_vertical_endpoint_exists(self):
        """POST /reframe/vertical should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/reframe/vertical", rules)

    def test_telemetry_parse_srt_endpoint_exists(self):
        """POST /telemetry/parse-srt should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/telemetry/parse-srt", rules)

    def test_telemetry_overlay_endpoint_exists(self):
        """POST /telemetry/overlay should be registered."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertIn("/telemetry/overlay", rules)

    def test_telemetry_parse_srt_no_filepath(self):
        """POST /telemetry/parse-srt without filepath should return 400."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/health")
        token = resp.get_json().get("csrf_token", "")
        resp = client.post(
            "/telemetry/parse-srt",
            json={"filepath": ""},
            headers={"X-OpenCut-Token": token, "Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)


# ============================================================
# Cross-Module Integration Tests
# ============================================================
class TestToolsIntegration(unittest.TestCase):
    """Cross-module integration sanity checks."""

    def test_all_core_modules_importable(self):
        """All 6 core tool modules should import without error."""

    def test_all_core_modules_have_docstrings(self):
        """All core modules should have module-level docstrings."""
        import opencut.core.beat_cuts
        import opencut.core.cursor_zoom
        import opencut.core.lower_thirds
        import opencut.core.redaction
        import opencut.core.smart_reframe
        import opencut.core.telemetry_overlay

        for mod in [
            opencut.core.cursor_zoom,
            opencut.core.lower_thirds,
            opencut.core.beat_cuts,
            opencut.core.redaction,
            opencut.core.smart_reframe,
            opencut.core.telemetry_overlay,
        ]:
            self.assertIsNotNone(mod.__doc__, f"{mod.__name__} missing docstring")

    def test_routes_module_importable(self):
        """tools_routes should import without error."""

    def test_all_route_functions_have_docstrings(self):
        """All route handler functions should have docstrings."""
        from opencut.routes import tools_routes
        for name in dir(tools_routes):
            obj = getattr(tools_routes, name)
            if callable(obj) and not name.startswith("_") and hasattr(obj, "__wrapped__"):
                self.assertIsNotNone(obj.__doc__, f"{name} missing docstring")


if __name__ == "__main__":
    unittest.main()
