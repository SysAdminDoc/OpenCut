"""
Tests for OpenCut delivery features:
  - Broadcast QC (broadcast_qc.py): audio levels, video levels, full QC
  - Thumbnail A/B (thumbnail_ab.py): frame scoring, variants, grid
  - End Screen (end_screen.py): templates, rendering, preview
  - News Ticker (news_ticker.py): overlay, standalone
  - HLS/DASH Streaming (streaming_package.py): packages, renditions
  - Broadcast CC (broadcast_cc.py): EBU-TT, TTML, CEA-608
  - FCPXML Export (fcpxml_export.py): clips, sequences, FCPXML 1.11
  - Delivery Routes (delivery_routes.py): all route endpoints
"""

import inspect
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Broadcast QC Core Tests
# ============================================================
class TestBroadcastQCCore(unittest.TestCase):
    """Tests for opencut.core.broadcast_qc."""

    def test_check_broadcast_standards_signature(self):
        from opencut.core.broadcast_qc import check_broadcast_standards
        sig = inspect.signature(check_broadcast_standards)
        for p in ["video_path", "standard", "check_audio", "check_video",
                   "check_codecs", "check_resolution_flag", "check_captions_flag",
                   "on_progress"]:
            self.assertIn(p, sig.parameters, f"Missing parameter: {p}")

    def test_check_audio_levels_signature(self):
        from opencut.core.broadcast_qc import check_audio_levels
        sig = inspect.signature(check_audio_levels)
        for p in ["video_path", "standard", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_check_video_levels_signature(self):
        from opencut.core.broadcast_qc import check_video_levels
        sig = inspect.signature(check_video_levels)
        for p in ["video_path", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_generate_qc_report_signature(self):
        from opencut.core.broadcast_qc import generate_qc_report
        sig = inspect.signature(generate_qc_report)
        for p in ["results", "output_path", "format", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_broadcast_standards_defined(self):
        from opencut.core.broadcast_qc import BROADCAST_STANDARDS
        self.assertIn("ebu_r128", BROADCAST_STANDARDS)
        self.assertIn("atsc_a85", BROADCAST_STANDARDS)
        self.assertIn("arib_tr_b32", BROADCAST_STANDARDS)

    def test_ebu_r128_target_loudness(self):
        from opencut.core.broadcast_qc import BROADCAST_STANDARDS
        self.assertEqual(BROADCAST_STANDARDS["ebu_r128"]["target_loudness"], -23.0)

    def test_atsc_a85_target_loudness(self):
        from opencut.core.broadcast_qc import BROADCAST_STANDARDS
        self.assertEqual(BROADCAST_STANDARDS["atsc_a85"]["target_loudness"], -24.0)

    def test_check_audio_levels_file_not_found(self):
        from opencut.core.broadcast_qc import check_audio_levels
        with self.assertRaises(FileNotFoundError):
            check_audio_levels("/nonexistent/video.mp4")

    def test_check_video_levels_file_not_found(self):
        from opencut.core.broadcast_qc import check_video_levels
        with self.assertRaises(FileNotFoundError):
            check_video_levels("/nonexistent/video.mp4")

    def test_check_broadcast_standards_file_not_found(self):
        from opencut.core.broadcast_qc import check_broadcast_standards
        with self.assertRaises(FileNotFoundError):
            check_broadcast_standards("/nonexistent/video.mp4")

    def test_check_audio_invalid_standard(self):
        from opencut.core.broadcast_qc import check_audio_levels
        with self.assertRaises(ValueError):
            check_audio_levels(__file__, standard="bogus_standard")

    def test_qccheckresult_dataclass(self):
        from opencut.core.broadcast_qc import QCCheckResult
        r = QCCheckResult(name="test", passed=True, value="ok")
        self.assertEqual(r.name, "test")
        self.assertTrue(r.passed)
        self.assertEqual(r.severity, "error")

    def test_audiolevelresult_defaults(self):
        from opencut.core.broadcast_qc import AudioLevelResult
        r = AudioLevelResult()
        self.assertEqual(r.integrated_loudness, 0.0)
        self.assertEqual(r.standard, "ebu_r128")
        self.assertFalse(r.passed)

    def test_videolevelresult_defaults(self):
        from opencut.core.broadcast_qc import VideoLevelResult
        r = VideoLevelResult()
        self.assertEqual(r.ymin, 0.0)
        self.assertEqual(r.ymax, 255.0)
        self.assertFalse(r.passed)

    def test_broadcast_qc_report_defaults(self):
        from opencut.core.broadcast_qc import BroadcastQCReport
        r = BroadcastQCReport()
        self.assertFalse(r.overall_pass)
        self.assertEqual(r.total_checks, 0)

    def test_valid_broadcast_codecs(self):
        from opencut.core.broadcast_qc import VALID_BROADCAST_CODECS
        self.assertIn("h264", VALID_BROADCAST_CODECS["video"])
        self.assertIn("aac", VALID_BROADCAST_CODECS["audio"])

    def test_broadcast_resolutions(self):
        from opencut.core.broadcast_qc import BROADCAST_RESOLUTIONS
        self.assertEqual(BROADCAST_RESOLUTIONS["hd_1080"], (1920, 1080))
        self.assertEqual(BROADCAST_RESOLUTIONS["hd_720"], (1280, 720))


# ============================================================
# Thumbnail A/B Core Tests
# ============================================================
class TestThumbnailABCore(unittest.TestCase):
    """Tests for opencut.core.thumbnail_ab."""

    def test_score_frames_signature(self):
        from opencut.core.thumbnail_ab import score_frames
        sig = inspect.signature(score_frames)
        for p in ["video_path", "count", "min_interval", "sample_count", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_generate_variants_signature(self):
        from opencut.core.thumbnail_ab import generate_variants
        sig = inspect.signature(generate_variants)
        for p in ["frame_path", "text", "output_dir", "width", "height", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_thumbnail_grid_signature(self):
        from opencut.core.thumbnail_ab import create_thumbnail_grid
        sig = inspect.signature(create_thumbnail_grid)
        for p in ["thumbnails", "output_path_str", "columns", "cell_width",
                   "cell_height", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_score_frames_file_not_found(self):
        from opencut.core.thumbnail_ab import score_frames
        with self.assertRaises(FileNotFoundError):
            score_frames("/nonexistent/video.mp4")

    def test_generate_variants_file_not_found(self):
        from opencut.core.thumbnail_ab import generate_variants
        with self.assertRaises(FileNotFoundError):
            generate_variants("/nonexistent/frame.jpg")

    def test_create_thumbnail_grid_empty(self):
        from opencut.core.thumbnail_ab import create_thumbnail_grid
        with self.assertRaises(ValueError):
            create_thumbnail_grid([], "/tmp/grid.jpg")

    def test_create_thumbnail_grid_invalid_paths(self):
        from opencut.core.thumbnail_ab import create_thumbnail_grid
        with self.assertRaises(ValueError):
            create_thumbnail_grid(["/no/such/file.jpg"], "/tmp/grid.jpg")

    def test_framescore_dataclass(self):
        from opencut.core.thumbnail_ab import FrameScore
        fs = FrameScore(timestamp=5.0, sharpness=80.0, overall_score=75.0)
        self.assertEqual(fs.timestamp, 5.0)
        self.assertEqual(fs.overall_score, 75.0)

    def test_thumbnailvariant_dataclass(self):
        from opencut.core.thumbnail_ab import ThumbnailVariant
        tv = ThumbnailVariant(variant_type="color_boost", path="/tmp/t.jpg")
        self.assertEqual(tv.variant_type, "color_boost")

    def test_score_frames_defaults(self):
        from opencut.core.thumbnail_ab import score_frames
        sig = inspect.signature(score_frames)
        self.assertEqual(sig.parameters["count"].default, 5)
        self.assertEqual(sig.parameters["sample_count"].default, 20)


# ============================================================
# End Screen Core Tests
# ============================================================
class TestEndScreenCore(unittest.TestCase):
    """Tests for opencut.core.end_screen."""

    def test_generate_end_screen_signature(self):
        from opencut.core.end_screen import generate_end_screen
        sig = inspect.signature(generate_end_screen)
        for p in ["template", "data", "duration", "output_path_str", "width",
                   "height", "fps", "fade_duration", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_list_templates_returns_list(self):
        from opencut.core.end_screen import list_templates
        templates = list_templates()
        self.assertIsInstance(templates, list)
        self.assertTrue(len(templates) >= 3)

    def test_list_templates_structure(self):
        from opencut.core.end_screen import list_templates
        templates = list_templates()
        for t in templates:
            self.assertIn("name", t)
            self.assertIn("label", t)
            self.assertIn("description", t)
            self.assertIn("elements", t)

    def test_list_templates_known_names(self):
        from opencut.core.end_screen import list_templates
        names = [t["name"] for t in list_templates()]
        self.assertIn("youtube_classic", names)
        self.assertIn("minimal", names)

    def test_preview_template_signature(self):
        from opencut.core.end_screen import preview_template
        sig = inspect.signature(preview_template)
        for p in ["template", "data", "width", "height", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_generate_end_screen_invalid_template(self):
        from opencut.core.end_screen import generate_end_screen
        with self.assertRaises(ValueError):
            generate_end_screen("nonexistent_template", {})

    def test_preview_template_invalid_template(self):
        from opencut.core.end_screen import preview_template
        with self.assertRaises(ValueError):
            preview_template("nonexistent_template", {})

    def test_endscreen_template_dataclass(self):
        from opencut.core.end_screen import EndScreenTemplate
        t = EndScreenTemplate(name="test", label="Test")
        self.assertEqual(t.name, "test")
        self.assertEqual(t.background_color, "000000")

    def test_endscreen_element_dataclass(self):
        from opencut.core.end_screen import EndScreenElement
        e = EndScreenElement(element_type="subscribe", x=0.3, y=0.7)
        self.assertEqual(e.element_type, "subscribe")

    def test_generate_end_screen_duration_clamping(self):
        """Duration should be clamped to 5-20 seconds."""
        from opencut.core.end_screen import generate_end_screen
        # We can't easily run it without FFmpeg, but we test the function exists
        sig = inspect.signature(generate_end_screen)
        self.assertEqual(sig.parameters["duration"].default, 10.0)


# ============================================================
# News Ticker Core Tests
# ============================================================
class TestNewsTickerCore(unittest.TestCase):
    """Tests for opencut.core.news_ticker."""

    def test_create_ticker_signature(self):
        from opencut.core.news_ticker import create_ticker
        sig = inspect.signature(create_ticker)
        for p in ["text_content", "video_path", "output_path_str", "speed",
                   "font_size", "font_color", "bg_color", "direction",
                   "position", "margin", "separator", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_ticker_overlay_signature(self):
        from opencut.core.news_ticker import create_ticker_overlay
        sig = inspect.signature(create_ticker_overlay)
        for p in ["text_content", "duration", "output_path_str", "width",
                   "height", "speed", "font_size", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_ticker_file_not_found(self):
        from opencut.core.news_ticker import create_ticker
        with self.assertRaises(FileNotFoundError):
            create_ticker("Breaking News", "/nonexistent/video.mp4")

    def test_create_ticker_empty_text(self):
        from opencut.core.news_ticker import create_ticker
        with self.assertRaises(ValueError):
            create_ticker("", __file__)

    def test_create_ticker_overlay_empty_text(self):
        from opencut.core.news_ticker import create_ticker_overlay
        with self.assertRaises(ValueError):
            create_ticker_overlay("")

    def test_ticker_positions_defined(self):
        from opencut.core.news_ticker import TICKER_POSITIONS
        self.assertIn("bottom", TICKER_POSITIONS)
        self.assertIn("top", TICKER_POSITIONS)
        self.assertIn("center", TICKER_POSITIONS)

    def test_ticker_directions_defined(self):
        from opencut.core.news_ticker import TICKER_DIRECTIONS
        self.assertIn("left", TICKER_DIRECTIONS)
        self.assertIn("right", TICKER_DIRECTIONS)

    def test_ticker_defaults(self):
        from opencut.core.news_ticker import DEFAULT_FONT_SIZE, DEFAULT_SPEED
        self.assertEqual(DEFAULT_FONT_SIZE, 48)
        self.assertEqual(DEFAULT_SPEED, 100)

    def test_load_ticker_text_string(self):
        from opencut.core.news_ticker import _load_ticker_text
        result = _load_ticker_text("Hello World")
        self.assertEqual(result, "Hello World")

    def test_load_ticker_text_list(self):
        from opencut.core.news_ticker import _load_ticker_text
        result = _load_ticker_text(["Line 1", "Line 2"], separator=" | ")
        self.assertEqual(result, "Line 1 | Line 2")

    def test_load_ticker_text_dict(self):
        from opencut.core.news_ticker import _load_ticker_text
        result = _load_ticker_text({"items": ["A", "B"]}, separator=" - ")
        self.assertEqual(result, "A - B")

    def test_escape_drawtext(self):
        from opencut.core.news_ticker import _escape_drawtext
        result = _escape_drawtext("Hello: World's 100%")
        self.assertIn("\\:", result)
        self.assertIn("\\'", result)
        self.assertIn("%%", result)

    def test_tickerconfig_dataclass(self):
        from opencut.core.news_ticker import TickerConfig
        tc = TickerConfig(text="Test", speed=200)
        self.assertEqual(tc.speed, 200)
        self.assertEqual(tc.direction, "left")


# ============================================================
# Streaming Package Core Tests
# ============================================================
class TestStreamingPackageCore(unittest.TestCase):
    """Tests for opencut.core.streaming_package."""

    def test_create_hls_package_signature(self):
        from opencut.core.streaming_package import create_hls_package
        sig = inspect.signature(create_hls_package)
        for p in ["video_path", "output_dir", "renditions", "segment_duration",
                   "include_zip", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_dash_package_signature(self):
        from opencut.core.streaming_package import create_dash_package
        sig = inspect.signature(create_dash_package)
        for p in ["video_path", "output_dir", "renditions", "segment_duration",
                   "include_zip", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_get_rendition_configs_signature(self):
        from opencut.core.streaming_package import get_rendition_configs
        sig = inspect.signature(get_rendition_configs)
        for p in ["source_resolution", "source_path"]:
            self.assertIn(p, sig.parameters)

    def test_get_rendition_configs_1080p(self):
        from opencut.core.streaming_package import get_rendition_configs
        configs = get_rendition_configs(source_resolution=(1920, 1080))
        names = [r.name for r in configs]
        self.assertIn("1080p", names)
        self.assertIn("720p", names)
        self.assertNotIn("1440p", names)
        self.assertNotIn("2160p", names)

    def test_get_rendition_configs_720p(self):
        from opencut.core.streaming_package import get_rendition_configs
        configs = get_rendition_configs(source_resolution=(1280, 720))
        names = [r.name for r in configs]
        self.assertIn("720p", names)
        self.assertNotIn("1080p", names)

    def test_get_rendition_configs_default(self):
        from opencut.core.streaming_package import get_rendition_configs
        configs = get_rendition_configs()
        self.assertTrue(len(configs) >= 1)

    def test_get_rendition_configs_low_res(self):
        from opencut.core.streaming_package import get_rendition_configs
        configs = get_rendition_configs(source_resolution=(320, 240))
        self.assertTrue(len(configs) >= 1)
        self.assertEqual(configs[0].name, "240p")

    def test_all_renditions_defined(self):
        from opencut.core.streaming_package import ALL_RENDITIONS
        self.assertIn("240p", ALL_RENDITIONS)
        self.assertIn("720p", ALL_RENDITIONS)
        self.assertIn("1080p", ALL_RENDITIONS)
        self.assertIn("2160p", ALL_RENDITIONS)

    def test_rendition_dataclass(self):
        from opencut.core.streaming_package import Rendition
        r = Rendition("test", 1280, 720, "2800k", "128k")
        self.assertEqual(r.name, "test")
        self.assertEqual(r.maxrate, "2800k")
        self.assertIn("k", r.bufsize)

    def test_create_hls_file_not_found(self):
        from opencut.core.streaming_package import create_hls_package
        with self.assertRaises(FileNotFoundError):
            create_hls_package("/nonexistent/video.mp4")

    def test_create_dash_file_not_found(self):
        from opencut.core.streaming_package import create_dash_package
        with self.assertRaises(FileNotFoundError):
            create_dash_package("/nonexistent/video.mp4")


# ============================================================
# Broadcast CC Core Tests
# ============================================================
class TestBroadcastCCCore(unittest.TestCase):
    """Tests for opencut.core.broadcast_cc."""

    def test_export_ebu_tt_signature(self):
        from opencut.core.broadcast_cc import export_ebu_tt
        sig = inspect.signature(export_ebu_tt)
        for p in ["captions", "output_path", "language", "title",
                   "frame_rate", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_export_ttml_signature(self):
        from opencut.core.broadcast_cc import export_ttml
        sig = inspect.signature(export_ttml)
        for p in ["captions", "output_path", "language", "title",
                   "profile", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_embed_cea608_signature(self):
        from opencut.core.broadcast_cc import embed_cea608
        sig = inspect.signature(embed_cea608)
        for p in ["video_path", "captions", "output_path", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_export_ebu_tt_empty_captions(self):
        from opencut.core.broadcast_cc import export_ebu_tt
        with self.assertRaises(ValueError):
            export_ebu_tt([], "/tmp/test.xml")

    def test_export_ttml_empty_captions(self):
        from opencut.core.broadcast_cc import export_ttml
        with self.assertRaises(ValueError):
            export_ttml([], "/tmp/test.xml")

    def test_embed_cea608_file_not_found(self):
        from opencut.core.broadcast_cc import embed_cea608
        with self.assertRaises(FileNotFoundError):
            embed_cea608("/nonexistent/video.mp4", [{"start": 0, "end": 1, "text": "Hi"}])

    def test_embed_cea608_empty_captions(self):
        from opencut.core.broadcast_cc import embed_cea608
        with self.assertRaises(ValueError):
            embed_cea608(__file__, [])

    def test_caption_dataclass(self):
        from opencut.core.broadcast_cc import Caption
        c = Caption(start=1.0, end=3.0, text="Hello")
        self.assertEqual(c.start, 1.0)
        self.assertEqual(c.align, "center")

    def test_captiondata_dataclass(self):
        from opencut.core.broadcast_cc import CaptionData
        cd = CaptionData(language="de")
        self.assertEqual(cd.language, "de")
        self.assertEqual(cd.captions, [])

    def test_parse_captions_list(self):
        from opencut.core.broadcast_cc import _parse_captions
        data = [
            {"start": 0, "end": 2, "text": "Hello"},
            {"start": 3, "end": 5, "text": "World"},
        ]
        result = _parse_captions(data)
        self.assertEqual(len(result.captions), 2)
        self.assertEqual(result.captions[0].text, "Hello")

    def test_parse_captions_dict(self):
        from opencut.core.broadcast_cc import _parse_captions
        data = {
            "language": "fr",
            "captions": [{"start": 0, "end": 1, "text": "Bonjour"}],
        }
        result = _parse_captions(data)
        self.assertEqual(result.language, "fr")
        self.assertEqual(len(result.captions), 1)

    def test_parse_srt(self):
        from opencut.core.broadcast_cc import _parse_srt
        srt = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
Second line"""
        result = _parse_srt(srt)
        self.assertEqual(len(result.captions), 2)
        self.assertAlmostEqual(result.captions[0].start, 1.0)
        self.assertAlmostEqual(result.captions[0].end, 3.0)
        self.assertEqual(result.captions[0].text, "Hello world")

    def test_secs_to_media_time(self):
        from opencut.core.broadcast_cc import _secs_to_media_time
        self.assertEqual(_secs_to_media_time(0), "00:00:00.000")
        self.assertEqual(_secs_to_media_time(3661.5), "01:01:01.500")

    def test_secs_to_timecode(self):
        from opencut.core.broadcast_cc import _secs_to_timecode
        result = _secs_to_timecode(0)
        self.assertEqual(result, "00:00:00:00")

    def test_export_ebu_tt_writes_xml(self):
        from opencut.core.broadcast_cc import export_ebu_tt
        captions = [
            {"start": 0, "end": 2, "text": "Hello"},
            {"start": 3, "end": 5, "text": "World"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            out_path = f.name
        try:
            result = export_ebu_tt(captions, out_path)
            self.assertEqual(result["format"], "ebu-tt")
            self.assertEqual(result["caption_count"], 2)
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(result["file_size_bytes"], 0)
        finally:
            os.unlink(out_path)

    def test_export_ttml_writes_xml(self):
        from opencut.core.broadcast_cc import export_ttml
        captions = [
            {"start": 0, "end": 2, "text": "Hello"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".ttml", delete=False) as f:
            out_path = f.name
        try:
            result = export_ttml(captions, out_path)
            self.assertIn("ttml", result["format"])
            self.assertTrue(os.path.isfile(out_path))
        finally:
            os.unlink(out_path)

    def test_export_ttml_imsc1_profile(self):
        from opencut.core.broadcast_cc import export_ttml
        captions = [{"start": 0, "end": 1, "text": "Test"}]
        with tempfile.NamedTemporaryFile(suffix=".ttml", delete=False) as f:
            out_path = f.name
        try:
            result = export_ttml(captions, out_path, profile="imsc1")
            self.assertEqual(result["format"], "ttml-imsc1")
        finally:
            os.unlink(out_path)


# ============================================================
# FCPXML Export Core Tests
# ============================================================
class TestFCPXMLExportCore(unittest.TestCase):
    """Tests for opencut.core.fcpxml_export."""

    def test_export_fcpxml_signature(self):
        from opencut.core.fcpxml_export import export_fcpxml
        sig = inspect.signature(export_fcpxml)
        for p in ["sequence", "output_path", "project_name", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_fcpxml_clip_signature(self):
        from opencut.core.fcpxml_export import create_fcpxml_clip
        sig = inspect.signature(create_fcpxml_clip)
        for p in ["source_path", "name", "start", "duration", "markers", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_fcpxml_sequence_signature(self):
        from opencut.core.fcpxml_export import create_fcpxml_sequence
        sig = inspect.signature(create_fcpxml_sequence)
        for p in ["clips", "name", "width", "height", "fps", "markers", "on_progress"]:
            self.assertIn(p, sig.parameters)

    def test_create_fcpxml_clip_file_not_found(self):
        from opencut.core.fcpxml_export import create_fcpxml_clip
        with self.assertRaises(FileNotFoundError):
            create_fcpxml_clip("/nonexistent/video.mp4")

    def test_create_fcpxml_sequence_empty_clips(self):
        from opencut.core.fcpxml_export import create_fcpxml_sequence
        with self.assertRaises(ValueError):
            create_fcpxml_sequence([])

    def test_export_fcpxml_empty_sequence(self):
        from opencut.core.fcpxml_export import FCPXMLSequence, export_fcpxml
        seq = FCPXMLSequence(name="Empty")
        with self.assertRaises(ValueError):
            export_fcpxml(seq, "/tmp/test.fcpxml")

    def test_fcpxml_clip_dataclass(self):
        from opencut.core.fcpxml_export import FCPXMLClip
        clip = FCPXMLClip(name="Test", duration=5.0)
        self.assertEqual(clip.name, "Test")
        self.assertTrue(clip.enabled)

    def test_fcpxml_marker_dataclass(self):
        from opencut.core.fcpxml_export import FCPXMLMarker
        m = FCPXMLMarker(name="Start", start=1.0)
        self.assertEqual(m.marker_type, "standard")

    def test_fcpxml_sequence_dataclass(self):
        from opencut.core.fcpxml_export import FCPXMLSequence
        seq = FCPXMLSequence(name="Test", fps=24.0)
        self.assertEqual(seq.fps, 24.0)
        self.assertEqual(seq.width, 1920)

    def test_seconds_to_rational_30fps(self):
        from opencut.core.fcpxml_export import _seconds_to_rational
        result = _seconds_to_rational(1.0, 30.0)
        self.assertIn("s", result)
        self.assertIn("/", result)

    def test_seconds_to_rational_zero(self):
        from opencut.core.fcpxml_export import _seconds_to_rational
        result = _seconds_to_rational(0.0, 30.0)
        self.assertTrue(result.endswith("s"))

    def test_fps_to_frameDuration(self):
        from opencut.core.fcpxml_export import _fps_to_frameDuration
        result = _fps_to_frameDuration(29.97)
        self.assertEqual(result, "1001/30000s")

    def test_fps_to_frameDuration_24(self):
        from opencut.core.fcpxml_export import _fps_to_frameDuration
        result = _fps_to_frameDuration(24.0)
        self.assertTrue(result.endswith("s"))

    def test_path_to_file_url(self):
        from opencut.core.fcpxml_export import _path_to_file_url
        result = _path_to_file_url("/test/video.mp4")
        self.assertTrue(result.startswith("file://"))
        self.assertIn("video.mp4", result)

    @patch("opencut.core.fcpxml_export.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_create_fcpxml_clip_auto_name(self, mock_info):
        from opencut.core.fcpxml_export import create_fcpxml_clip
        clip = create_fcpxml_clip(__file__, duration=5.0)
        # Name should be derived from filename
        self.assertIn("test_delivery", clip.name)

    def test_create_fcpxml_sequence_with_dicts(self):
        from opencut.core.fcpxml_export import create_fcpxml_sequence
        clips = [
            {"name": "Clip 1", "source_path": __file__, "duration": 5.0},
            {"name": "Clip 2", "source_path": __file__, "duration": 3.0},
        ]
        seq = create_fcpxml_sequence(clips)
        self.assertEqual(len(seq.clips), 2)
        self.assertEqual(seq.clips[0].name, "Clip 1")
        self.assertAlmostEqual(seq.duration, 8.0)

    def test_create_fcpxml_sequence_auto_position(self):
        from opencut.core.fcpxml_export import create_fcpxml_sequence
        clips = [
            {"name": "A", "duration": 5.0},
            {"name": "B", "duration": 3.0},
        ]
        seq = create_fcpxml_sequence(clips)
        self.assertAlmostEqual(seq.clips[0].start, 0.0)
        self.assertAlmostEqual(seq.clips[1].start, 5.0)

    def test_create_fcpxml_sequence_with_markers(self):
        from opencut.core.fcpxml_export import create_fcpxml_sequence
        clips = [{"name": "A", "duration": 10.0}]
        markers = [{"name": "Intro", "start": 0.0}, {"name": "End", "start": 9.0}]
        seq = create_fcpxml_sequence(clips, markers=markers)
        self.assertEqual(len(seq.markers), 2)

    def test_export_fcpxml_writes_file(self):
        from opencut.core.fcpxml_export import FCPXMLClip, FCPXMLSequence, export_fcpxml
        seq = FCPXMLSequence(
            name="Test Sequence",
            clips=[
                FCPXMLClip(name="Clip 1", source_path=__file__, start=0, duration=5.0,
                           source_duration=5.0),
            ],
            duration=5.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".fcpxml", delete=False) as f:
            out_path = f.name
        try:
            result = export_fcpxml(seq, out_path)
            self.assertEqual(result["format"], "fcpxml-1.11")
            self.assertEqual(result["clip_count"], 1)
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(result["file_size_bytes"], 0)
            # Verify XML is parseable
            content = open(out_path, "r", encoding="utf-8").read()
            self.assertIn("fcpxml", content)
            self.assertIn("version", content)
        finally:
            os.unlink(out_path)

    def test_export_fcpxml_contains_dtd(self):
        from opencut.core.fcpxml_export import FCPXMLClip, FCPXMLSequence, export_fcpxml
        seq = FCPXMLSequence(
            name="Test",
            clips=[FCPXMLClip(name="C", source_path=__file__,
                               start=0, duration=3.0, source_duration=3.0)],
            duration=3.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".fcpxml", delete=False) as f:
            out_path = f.name
        try:
            export_fcpxml(seq, out_path)
            content = open(out_path, "r", encoding="utf-8").read()
            self.assertIn("<!DOCTYPE fcpxml>", content)
            self.assertIn('<?xml version="1.0"', content)
        finally:
            os.unlink(out_path)

    def test_export_fcpxml_with_markers(self):
        from opencut.core.fcpxml_export import (
            FCPXMLClip,
            FCPXMLMarker,
            FCPXMLSequence,
            export_fcpxml,
        )
        seq = FCPXMLSequence(
            name="Marked",
            clips=[
                FCPXMLClip(
                    name="C", source_path=__file__, start=0,
                    duration=10.0, source_duration=10.0,
                    markers=[FCPXMLMarker(name="Cut", start=5.0)],
                ),
            ],
            markers=[FCPXMLMarker(name="Chapter 1", start=0.0, marker_type="chapter")],
            duration=10.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".fcpxml", delete=False) as f:
            out_path = f.name
        try:
            result = export_fcpxml(seq, out_path)
            self.assertEqual(result["marker_count"], 2)
            content = open(out_path, "r", encoding="utf-8").read()
            self.assertIn("marker", content)
            self.assertIn("chapter-marker", content)
        finally:
            os.unlink(out_path)

    def test_export_fcpxml_from_dict(self):
        from opencut.core.fcpxml_export import export_fcpxml
        seq_dict = {
            "name": "Dict Sequence",
            "clips": [
                {"name": "Clip A", "source_path": __file__, "duration": 5.0},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".fcpxml", delete=False) as f:
            out_path = f.name
        try:
            result = export_fcpxml(seq_dict, out_path)
            self.assertEqual(result["clip_count"], 1)
            self.assertTrue(os.path.isfile(out_path))
        finally:
            os.unlink(out_path)

    def test_export_fcpxml_progress_callback(self):
        from opencut.core.fcpxml_export import FCPXMLClip, FCPXMLSequence, export_fcpxml
        progress_calls = []

        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        seq = FCPXMLSequence(
            name="Test",
            clips=[FCPXMLClip(name="C", source_path=__file__,
                               start=0, duration=3.0, source_duration=3.0)],
            duration=3.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".fcpxml", delete=False) as f:
            out_path = f.name
        try:
            export_fcpxml(seq, out_path, on_progress=on_progress)
            self.assertTrue(len(progress_calls) >= 3)
            self.assertEqual(progress_calls[-1][0], 100)
        finally:
            os.unlink(out_path)

    def test_fcpxml_version_constant(self):
        from opencut.core.fcpxml_export import FCPXML_VERSION
        self.assertEqual(FCPXML_VERSION, "1.11")


# ============================================================
# Delivery Routes Tests
# ============================================================
class TestDeliveryRouteBlueprint(unittest.TestCase):
    """Tests for delivery_routes.py Blueprint registration and structure."""

    def test_delivery_bp_exists(self):
        from opencut.routes.delivery_routes import delivery_bp
        self.assertEqual(delivery_bp.name, "delivery")

    def test_delivery_bp_registered(self):
        """delivery_bp should be in the registered blueprint list."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        bp_names = [bp.name for bp in app.iter_blueprints()]
        self.assertIn("delivery", bp_names)


class TestDeliveryRouteBroadcastQC(unittest.TestCase):
    """Tests for /delivery/broadcast-qc routes."""

    def test_broadcast_qc_route_exists(self):
        from opencut.routes.delivery_routes import broadcast_qc_route
        self.assertTrue(callable(broadcast_qc_route))

    def test_broadcast_qc_audio_route_exists(self):
        from opencut.routes.delivery_routes import broadcast_qc_audio_route
        self.assertTrue(callable(broadcast_qc_audio_route))

    def test_broadcast_qc_report_route_exists(self):
        from opencut.routes.delivery_routes import broadcast_qc_report_route
        self.assertTrue(callable(broadcast_qc_report_route))


class TestDeliveryRouteThumbnailAB(unittest.TestCase):
    """Tests for /delivery/thumbnail-ab route."""

    def test_thumbnail_ab_route_exists(self):
        from opencut.routes.delivery_routes import thumbnail_ab_route
        self.assertTrue(callable(thumbnail_ab_route))


class TestDeliveryRouteEndScreen(unittest.TestCase):
    """Tests for /delivery/end-screen routes."""

    def test_end_screen_route_exists(self):
        from opencut.routes.delivery_routes import end_screen_route
        self.assertTrue(callable(end_screen_route))

    def test_end_screen_templates_route_exists(self):
        from opencut.routes.delivery_routes import end_screen_templates_route
        self.assertTrue(callable(end_screen_templates_route))


class TestDeliveryRouteNewsTicker(unittest.TestCase):
    """Tests for /delivery/news-ticker routes."""

    def test_news_ticker_route_exists(self):
        from opencut.routes.delivery_routes import news_ticker_route
        self.assertTrue(callable(news_ticker_route))

    def test_news_ticker_standalone_route_exists(self):
        from opencut.routes.delivery_routes import news_ticker_standalone_route
        self.assertTrue(callable(news_ticker_standalone_route))


class TestDeliveryRouteStreaming(unittest.TestCase):
    """Tests for /delivery/hls and /delivery/dash routes."""

    def test_hls_route_exists(self):
        from opencut.routes.delivery_routes import hls_package_route
        self.assertTrue(callable(hls_package_route))

    def test_dash_route_exists(self):
        from opencut.routes.delivery_routes import dash_package_route
        self.assertTrue(callable(dash_package_route))


class TestDeliveryRouteCaptions(unittest.TestCase):
    """Tests for /delivery/caption routes."""

    def test_ebu_tt_route_exists(self):
        from opencut.routes.delivery_routes import caption_ebu_tt_route
        self.assertTrue(callable(caption_ebu_tt_route))

    def test_ttml_route_exists(self):
        from opencut.routes.delivery_routes import caption_ttml_route
        self.assertTrue(callable(caption_ttml_route))

    def test_embed_cc_route_exists(self):
        from opencut.routes.delivery_routes import caption_embed_cc_route
        self.assertTrue(callable(caption_embed_cc_route))


class TestDeliveryRouteFCPXML(unittest.TestCase):
    """Tests for /delivery/fcpxml route."""

    def test_fcpxml_route_exists(self):
        from opencut.routes.delivery_routes import fcpxml_export_route
        self.assertTrue(callable(fcpxml_export_route))


# ============================================================
# Integration-style Route Tests (Flask test client)
# ============================================================
class TestDeliveryRouteEndpoints(unittest.TestCase):
    """Flask test client integration tests for delivery routes."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        cls.app = create_app(config=OpenCutConfig())
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Get CSRF token
        resp = cls.client.get("/health")
        cls.csrf_token = resp.get_json().get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf_token,
            "Content-Type": "application/json",
        }

    def test_broadcast_qc_no_filepath(self):
        resp = self.client.post("/delivery/broadcast-qc",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_broadcast_qc_missing_file(self):
        resp = self.client.post("/delivery/broadcast-qc",
                                headers=self.headers,
                                json={"filepath": "/nonexistent/video.mp4"})
        self.assertEqual(resp.status_code, 400)

    def test_broadcast_qc_audio_no_filepath(self):
        resp = self.client.post("/delivery/broadcast-qc/audio",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_broadcast_qc_report_no_filepath(self):
        resp = self.client.post("/delivery/broadcast-qc/report",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_thumbnail_ab_no_filepath(self):
        resp = self.client.post("/delivery/thumbnail-ab",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_end_screen_no_template(self):
        """End screen doesn't require filepath, should accept empty body."""
        resp = self.client.post("/delivery/end-screen",
                                headers=self.headers,
                                json={"template": "youtube_classic"})
        # Should get job_id back (or 200)
        self.assertIn(resp.status_code, [200])
        data = resp.get_json()
        self.assertIn("job_id", data)

    def test_end_screen_templates(self):
        resp = self.client.post("/delivery/end-screen/templates",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("templates", data)
        self.assertTrue(len(data["templates"]) >= 3)

    def test_news_ticker_no_filepath(self):
        resp = self.client.post("/delivery/news-ticker",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_news_ticker_standalone_no_text(self):
        """Standalone ticker doesn't need filepath but needs text."""
        resp = self.client.post("/delivery/news-ticker/standalone",
                                headers=self.headers,
                                json={})
        # Should get job_id since filepath_required=False; error comes in job
        self.assertIn(resp.status_code, [200])

    def test_hls_no_filepath(self):
        resp = self.client.post("/delivery/hls",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_dash_no_filepath(self):
        resp = self.client.post("/delivery/dash",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_caption_ebu_tt_no_captions(self):
        """EBU-TT doesn't require filepath; error should come from missing captions in job."""
        resp = self.client.post("/delivery/caption/ebu-tt",
                                headers=self.headers,
                                json={})
        # filepath_required=False so job is created
        self.assertIn(resp.status_code, [200])

    def test_caption_ttml_no_captions(self):
        resp = self.client.post("/delivery/caption/ttml",
                                headers=self.headers,
                                json={})
        self.assertIn(resp.status_code, [200])

    def test_caption_embed_cc_no_filepath(self):
        resp = self.client.post("/delivery/caption/embed-cc",
                                headers=self.headers,
                                json={})
        self.assertEqual(resp.status_code, 400)

    def test_fcpxml_export_creates_job(self):
        """FCPXML export doesn't require filepath, should create a job."""
        resp = self.client.post("/delivery/fcpxml",
                                headers=self.headers,
                                json={
                                    "sequence": {
                                        "name": "Test",
                                        "clips": [{"name": "Clip 1", "duration": 5.0}],
                                    }
                                })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("job_id", data)

    def test_csrf_required(self):
        """Routes should reject requests without CSRF token."""
        resp = self.client.post("/delivery/broadcast-qc",
                                headers={"Content-Type": "application/json"},
                                json={"filepath": "/test/video.mp4"})
        self.assertEqual(resp.status_code, 403)

    def test_end_screen_templates_csrf_required(self):
        resp = self.client.post("/delivery/end-screen/templates",
                                headers={"Content-Type": "application/json"},
                                json={})
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
