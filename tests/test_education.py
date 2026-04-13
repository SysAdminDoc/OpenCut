"""
Tests for OpenCut Education & Tutorial features.

Covers:
  - Click & Keystroke Overlay (11.2): data types, parsing, filter building
  - Callout & Annotation Generator (11.3): step callouts, annotation types
  - Screenshot-to-Video with Ken Burns (11.4): ROI detection, keyframes
  - Slide Change Detection (33.1): detection, extraction, chapters
  - PiP Lecture Processing (33.2): region detection, stream extraction
  - Auto-Quiz Overlay (33.3): question generation, rendering
  - Education routes (smoke tests)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Click & Keystroke Overlay (11.2)
# ============================================================
class TestClickOverlayDataTypes(unittest.TestCase):
    """Tests for click_overlay data types."""

    def test_click_event_defaults(self):
        from opencut.core.click_overlay import ClickEvent
        e = ClickEvent(timestamp=1.0, x=100, y=200)
        self.assertEqual(e.button, "left")
        self.assertAlmostEqual(e.duration, 0.4)

    def test_click_event_custom(self):
        from opencut.core.click_overlay import ClickEvent
        e = ClickEvent(timestamp=2.5, x=50, y=75, button="right", duration=0.8)
        self.assertEqual(e.button, "right")
        self.assertAlmostEqual(e.duration, 0.8)

    def test_keystroke_event_defaults(self):
        from opencut.core.click_overlay import KeystrokeEvent
        e = KeystrokeEvent(timestamp=3.0, keys="Ctrl+S")
        self.assertEqual(e.keys, "Ctrl+S")
        self.assertAlmostEqual(e.duration, 1.5)

    def test_click_overlay_result_to_dict(self):
        from opencut.core.click_overlay import ClickOverlayResult
        r = ClickOverlayResult(output_path="/out.mp4", click_count=5, duration=30.0)
        d = r.to_dict()
        self.assertEqual(d["output_path"], "/out.mp4")
        self.assertEqual(d["click_count"], 5)

    def test_keystroke_overlay_result_to_dict(self):
        from opencut.core.click_overlay import KeystrokeOverlayResult
        r = KeystrokeOverlayResult(output_path="/out.mp4", keystroke_count=3, duration=20.0)
        d = r.to_dict()
        self.assertEqual(d["keystroke_count"], 3)


class TestParseClickLog(unittest.TestCase):
    """Tests for parse_click_log."""

    def test_parse_json_log(self):
        from opencut.core.click_overlay import parse_click_log
        data = {
            "clicks": [
                {"timestamp": 1.0, "x": 100, "y": 200},
                {"timestamp": 3.5, "x": 300, "y": 400, "button": "right"},
            ],
            "keystrokes": [
                {"timestamp": 2.0, "keys": "Ctrl+Z"},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            result = parse_click_log(path)
            self.assertEqual(len(result["clicks"]), 2)
            self.assertEqual(len(result["keystrokes"]), 1)
            self.assertEqual(result["clicks"][0].x, 100)
            self.assertEqual(result["clicks"][1].button, "right")
            self.assertEqual(result["keystrokes"][0].keys, "Ctrl+Z")
        finally:
            os.unlink(path)

    def test_parse_text_log(self):
        from opencut.core.click_overlay import parse_click_log
        content = "# Comment\nclick 1.0 100 200 left\nkey 2.5 Ctrl+S\nclick 3.0 300 400\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = parse_click_log(path)
            self.assertEqual(len(result["clicks"]), 2)
            self.assertEqual(len(result["keystrokes"]), 1)
        finally:
            os.unlink(path)

    def test_parse_missing_file(self):
        from opencut.core.click_overlay import parse_click_log
        with self.assertRaises(FileNotFoundError):
            parse_click_log("/nonexistent/log.txt")

    def test_parse_empty_file(self):
        from opencut.core.click_overlay import parse_click_log
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                parse_click_log(path)
        finally:
            os.unlink(path)

    def test_parse_invalid_text_line(self):
        from opencut.core.click_overlay import parse_click_log
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("click 1.0\n")  # missing x, y
            path = f.name
        try:
            with self.assertRaises(ValueError):
                parse_click_log(path)
        finally:
            os.unlink(path)


class TestClickOverlayRender(unittest.TestCase):
    """Tests for render_click_overlay."""

    def test_render_missing_video(self):
        from opencut.core.click_overlay import ClickEvent, render_click_overlay
        events = [ClickEvent(timestamp=1.0, x=100, y=100)]
        with self.assertRaises(FileNotFoundError):
            render_click_overlay("/nonexistent.mp4", events)

    def test_render_empty_events(self):
        from opencut.core.click_overlay import render_click_overlay
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                render_click_overlay(path, [])
        finally:
            os.unlink(path)


class TestKeystrokeOverlayRender(unittest.TestCase):
    """Tests for render_keystroke_overlay."""

    def test_render_missing_video(self):
        from opencut.core.click_overlay import KeystrokeEvent, render_keystroke_overlay
        events = [KeystrokeEvent(timestamp=1.0, keys="Ctrl+S")]
        with self.assertRaises(FileNotFoundError):
            render_keystroke_overlay("/nonexistent.mp4", events)

    def test_render_empty_events(self):
        from opencut.core.click_overlay import render_keystroke_overlay
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                render_keystroke_overlay(path, [])
        finally:
            os.unlink(path)

    def test_render_invalid_position(self):
        from opencut.core.click_overlay import KeystrokeEvent, render_keystroke_overlay
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            events = [KeystrokeEvent(timestamp=1.0, keys="X")]
            with self.assertRaises(ValueError):
                render_keystroke_overlay(path, events, position="invalid-pos")
        finally:
            os.unlink(path)


class TestEscapeDrawtext(unittest.TestCase):
    """Tests for drawtext escaping."""

    def test_escape_colon(self):
        from opencut.core.click_overlay import _escape_drawtext
        self.assertIn("\\:", _escape_drawtext("A:B"))

    def test_escape_percent(self):
        from opencut.core.click_overlay import _escape_drawtext
        self.assertIn("%%", _escape_drawtext("50%"))


# ============================================================
# Callout & Annotation Generator (11.3)
# ============================================================
class TestCalloutDataTypes(unittest.TestCase):
    """Tests for callout_gen data types."""

    def test_region_creation(self):
        from opencut.core.callout_gen import Region
        r = Region(x=10, y=20, w=100, h=50)
        self.assertEqual(r.x, 10)
        self.assertEqual(r.h, 50)

    def test_annotation_defaults(self):
        from opencut.core.callout_gen import Annotation
        a = Annotation(type="callout", start_time=1.0, end_time=3.0)
        self.assertIsNone(a.region)
        self.assertEqual(a.color, "yellow")

    def test_callout_result_to_dict(self):
        from opencut.core.callout_gen import CalloutResult
        r = CalloutResult(
            output_path="/out.mp4", annotation_count=3,
            types_used=["callout", "blur"], duration=60.0,
        )
        d = r.to_dict()
        self.assertEqual(d["annotation_count"], 3)
        self.assertIn("blur", d["types_used"])


class TestStepCallout(unittest.TestCase):
    """Tests for create_step_callout."""

    def test_valid_step(self):
        from opencut.core.callout_gen import create_step_callout
        sc = create_step_callout("Click here", number=1, style="circle")
        self.assertEqual(sc.text, "Click here")
        self.assertEqual(sc.number, 1)

    def test_invalid_number(self):
        from opencut.core.callout_gen import create_step_callout
        with self.assertRaises(ValueError):
            create_step_callout("Step", number=0)

    def test_invalid_style(self):
        from opencut.core.callout_gen import create_step_callout
        with self.assertRaises(ValueError):
            create_step_callout("Step", number=1, style="hexagon")

    def test_empty_text(self):
        from opencut.core.callout_gen import create_step_callout
        with self.assertRaises(ValueError):
            create_step_callout("   ", number=1)

    def test_step_callout_to_dict(self):
        from opencut.core.callout_gen import create_step_callout
        sc = create_step_callout("Do this", number=2, style="badge")
        d = sc.to_dict()
        self.assertEqual(d["number"], 2)
        self.assertEqual(d["style"], "badge")


class TestGenerateCallout(unittest.TestCase):
    """Tests for generate_callout."""

    def test_missing_video(self):
        from opencut.core.callout_gen import Annotation, generate_callout
        anns = [Annotation(type="callout", start_time=0, end_time=1, text="Hi")]
        with self.assertRaises(FileNotFoundError):
            generate_callout("/nonexistent.mp4", anns)

    def test_empty_annotations(self):
        from opencut.core.callout_gen import generate_callout
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                generate_callout(path, [])
        finally:
            os.unlink(path)


class TestSpotlight(unittest.TestCase):
    """Tests for create_spotlight."""

    def test_missing_video(self):
        from opencut.core.callout_gen import Region, create_spotlight
        with self.assertRaises(FileNotFoundError):
            create_spotlight("/nonexistent.mp4", Region(0, 0, 100, 100), (0, 5))

    def test_invalid_time_range(self):
        from opencut.core.callout_gen import Region, create_spotlight
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_spotlight(path, Region(0, 0, 100, 100), (5, 2))
        finally:
            os.unlink(path)

    def test_negative_start_time(self):
        from opencut.core.callout_gen import Region, create_spotlight
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_spotlight(path, Region(0, 0, 100, 100), (-1, 5))
        finally:
            os.unlink(path)


# ============================================================
# Screenshot-to-Video with Ken Burns (11.4)
# ============================================================
class TestScreenshotVideo(unittest.TestCase):
    """Tests for screenshot_video module."""

    def test_roi_data(self):
        from opencut.core.screenshot_video import ROI
        r = ROI(x=10, y=20, w=100, h=80, weight=1.5, label="face")
        c = r.center()
        self.assertEqual(c, (60, 60))
        d = r.to_dict()
        self.assertEqual(d["label"], "face")

    def test_keyframe_data(self):
        from opencut.core.screenshot_video import KenBurnsKeyframe
        kf = KenBurnsKeyframe(time=1.5, x=0.5, y=0.3, zoom=1.2)
        d = kf.to_dict()
        self.assertAlmostEqual(d["time"], 1.5)
        self.assertAlmostEqual(d["zoom"], 1.2)

    def test_detect_roi_missing_image(self):
        from opencut.core.screenshot_video import detect_roi
        with self.assertRaises(FileNotFoundError):
            detect_roi("/nonexistent.png")

    def test_detect_roi_fallback(self):
        """detect_roi should return at least center ROIs for any image."""
        from opencut.core.screenshot_video import _detect_roi_center
        rois = _detect_roi_center("dummy.png")
        self.assertTrue(len(rois) >= 1)
        self.assertEqual(rois[0].label, "center")


class TestKenBurnsKeyframes(unittest.TestCase):
    """Tests for generate_ken_burns_keyframes."""

    def test_basic_keyframes(self):
        from opencut.core.screenshot_video import generate_ken_burns_keyframes
        rois = [
            {"x": 0, "y": 0, "w": 100, "h": 100, "weight": 1.0},
            {"x": 200, "y": 200, "w": 100, "h": 100, "weight": 1.0},
        ]
        kfs = generate_ken_burns_keyframes(rois, duration=10.0)
        self.assertTrue(len(kfs) >= 2)
        self.assertAlmostEqual(kfs[0]["time"], 0.0)

    def test_zero_duration(self):
        from opencut.core.screenshot_video import generate_ken_burns_keyframes
        with self.assertRaises(ValueError):
            generate_ken_burns_keyframes([{"x": 0, "y": 0, "w": 10, "h": 10}], 0)

    def test_empty_rois(self):
        from opencut.core.screenshot_video import generate_ken_burns_keyframes
        with self.assertRaises(ValueError):
            generate_ken_burns_keyframes([], 5.0)

    def test_custom_zoom_range(self):
        from opencut.core.screenshot_video import generate_ken_burns_keyframes
        rois = [{"x": 0, "y": 0, "w": 50, "h": 50, "weight": 1.0}]
        kfs = generate_ken_burns_keyframes(rois, 5.0, min_zoom=1.0, max_zoom=2.0)
        zooms = [kf["zoom"] for kf in kfs]
        self.assertTrue(all(1.0 <= z <= 2.0 for z in zooms))


class TestCreateScreenshotVideo(unittest.TestCase):
    """Tests for create_screenshot_video."""

    def test_empty_images(self):
        from opencut.core.screenshot_video import create_screenshot_video
        with self.assertRaises(ValueError):
            create_screenshot_video([], "/tmp/out.mp4")

    def test_missing_image(self):
        from opencut.core.screenshot_video import create_screenshot_video
        with self.assertRaises(FileNotFoundError):
            create_screenshot_video(["/nonexistent.png"], "/tmp/out.mp4")

    def test_unsupported_format(self):
        from opencut.core.screenshot_video import create_screenshot_video
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not an image")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_screenshot_video([path], "/tmp/out.mp4")
        finally:
            os.unlink(path)


# ============================================================
# Slide Change Detection (33.1)
# ============================================================
class TestSlideDetectDataTypes(unittest.TestCase):
    """Tests for slide_detect data types."""

    def test_slide_change_to_dict(self):
        from opencut.core.slide_detect import SlideChange
        sc = SlideChange(timestamp=5.123, frame_number=153, score=0.85, slide_index=2)
        d = sc.to_dict()
        self.assertAlmostEqual(d["timestamp"], 5.123)
        self.assertEqual(d["slide_index"], 2)

    def test_slide_chapter_to_dict(self):
        from opencut.core.slide_detect import SlideChapter
        ch = SlideChapter(title="Slide 1", start_time=0.0, end_time=30.0, slide_index=0)
        d = ch.to_dict()
        self.assertEqual(d["title"], "Slide 1")

    def test_slide_detection_result_to_dict(self):
        from opencut.core.slide_detect import SlideDetectionResult
        r = SlideDetectionResult(changes=[], slide_count=0, duration=60.0, threshold_used=0.3)
        d = r.to_dict()
        self.assertAlmostEqual(d["threshold_used"], 0.3)

    def test_slide_extraction_result_to_dict(self):
        from opencut.core.slide_detect import SlideExtractionResult
        r = SlideExtractionResult(output_dir="/tmp/slides", image_paths=["/a.png"], slide_count=1)
        d = r.to_dict()
        self.assertEqual(d["slide_count"], 1)


class TestDetectSlideChanges(unittest.TestCase):
    """Tests for detect_slide_changes."""

    def test_missing_video(self):
        from opencut.core.slide_detect import detect_slide_changes
        with self.assertRaises(FileNotFoundError):
            detect_slide_changes("/nonexistent.mp4")

    def test_invalid_threshold(self):
        from opencut.core.slide_detect import detect_slide_changes
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                detect_slide_changes(path, threshold=0.0)
            with self.assertRaises(ValueError):
                detect_slide_changes(path, threshold=1.0)
        finally:
            os.unlink(path)


class TestExtractSlideImages(unittest.TestCase):
    """Tests for extract_slide_images."""

    def test_missing_video(self):
        from opencut.core.slide_detect import extract_slide_images
        with self.assertRaises(FileNotFoundError):
            extract_slide_images("/nonexistent.mp4", [1.0], "/tmp/slides")

    def test_empty_timestamps(self):
        from opencut.core.slide_detect import extract_slide_images
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                extract_slide_images(path, [], "/tmp/slides")
        finally:
            os.unlink(path)

    def test_invalid_format(self):
        from opencut.core.slide_detect import extract_slide_images
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                extract_slide_images(path, [1.0], "/tmp/slides", image_format="bmp")
        finally:
            os.unlink(path)


class TestGenerateSlideChapters(unittest.TestCase):
    """Tests for generate_slide_chapters."""

    def test_basic_chapters(self):
        from opencut.core.slide_detect import generate_slide_chapters
        timestamps = [0.0, 30.0, 60.0]
        chapters = generate_slide_chapters(timestamps, video_duration=90.0)
        self.assertEqual(len(chapters), 3)
        self.assertEqual(chapters[0]["title"], "Slide 1")
        self.assertAlmostEqual(chapters[0]["start_time"], 0.0)
        self.assertAlmostEqual(chapters[0]["end_time"], 30.0)
        self.assertAlmostEqual(chapters[2]["end_time"], 90.0)

    def test_custom_titles(self):
        from opencut.core.slide_detect import generate_slide_chapters
        timestamps = [0.0, 10.0]
        chapters = generate_slide_chapters(
            timestamps, custom_titles=["Intro", "Main Content"]
        )
        self.assertEqual(chapters[0]["title"], "Intro")
        self.assertEqual(chapters[1]["title"], "Main Content")

    def test_custom_prefix(self):
        from opencut.core.slide_detect import generate_slide_chapters
        chapters = generate_slide_chapters([0.0], title_prefix="Chapter")
        self.assertEqual(chapters[0]["title"], "Chapter 1")

    def test_empty_timestamps(self):
        from opencut.core.slide_detect import generate_slide_chapters
        with self.assertRaises(ValueError):
            generate_slide_chapters([])

    def test_mismatched_custom_titles(self):
        from opencut.core.slide_detect import generate_slide_chapters
        with self.assertRaises(ValueError):
            generate_slide_chapters([0, 10], custom_titles=["Only One"])


# ============================================================
# PiP Lecture Processing (33.2)
# ============================================================
class TestPipDataTypes(unittest.TestCase):
    """Tests for pip_lecture data types."""

    def test_pip_region_to_dict(self):
        from opencut.core.pip_lecture import PipRegion
        r = PipRegion(x=100, y=200, w=320, h=240, confidence=0.85, position="bottom-right")
        d = r.to_dict()
        self.assertEqual(d["x"], 100)
        self.assertAlmostEqual(d["confidence"], 0.85)

    def test_pip_detection_result(self):
        from opencut.core.pip_lecture import PipDetectionResult
        r = PipDetectionResult(
            region={"x": 0, "y": 0, "w": 100, "h": 100},
            video_width=1920, video_height=1080, method="contour",
        )
        d = r.to_dict()
        self.assertEqual(d["method"], "contour")

    def test_pip_extraction_result(self):
        from opencut.core.pip_lecture import PipExtractionResult
        r = PipExtractionResult(
            speaker_path="/out/speaker.mp4",
            screen_path="/out/screen.mp4",
            output_dir="/out",
        )
        d = r.to_dict()
        self.assertIn("speaker.mp4", d["speaker_path"])

    def test_side_by_side_result(self):
        from opencut.core.pip_lecture import SideBySideResult
        r = SideBySideResult(output_path="/out.mp4", layout="speaker-left", resolution=(1920, 1080))
        d = r.to_dict()
        self.assertEqual(d["layout"], "speaker-left")


class TestPipFallback(unittest.TestCase):
    """Tests for PiP fallback detection."""

    def test_fallback_produces_region(self):
        from opencut.core.pip_lecture import _detect_pip_fallback
        pip = _detect_pip_fallback(1920, 1080)
        self.assertEqual(pip.position, "bottom-right")
        self.assertGreater(pip.w, 0)
        self.assertGreater(pip.h, 0)
        self.assertAlmostEqual(pip.confidence, 0.3)

    def test_fallback_scales_with_resolution(self):
        from opencut.core.pip_lecture import _detect_pip_fallback
        pip_hd = _detect_pip_fallback(1920, 1080)
        pip_4k = _detect_pip_fallback(3840, 2160)
        self.assertGreater(pip_4k.w, pip_hd.w)


class TestDetectPipRegion(unittest.TestCase):
    """Tests for detect_pip_region."""

    def test_missing_video(self):
        from opencut.core.pip_lecture import detect_pip_region
        with self.assertRaises(FileNotFoundError):
            detect_pip_region("/nonexistent.mp4")


class TestExtractPipStreams(unittest.TestCase):
    """Tests for extract_pip_streams."""

    def test_missing_video(self):
        from opencut.core.pip_lecture import extract_pip_streams
        with self.assertRaises(FileNotFoundError):
            extract_pip_streams(
                "/nonexistent.mp4",
                {"x": 0, "y": 0, "w": 100, "h": 100},
                "/tmp/pip_out",
            )

    def test_invalid_region(self):
        from opencut.core.pip_lecture import extract_pip_streams
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                extract_pip_streams(path, {"x": 0, "y": 0}, "/tmp/pip_out")
        finally:
            os.unlink(path)

    def test_zero_size_region(self):
        from opencut.core.pip_lecture import extract_pip_streams
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                extract_pip_streams(
                    path, {"x": 0, "y": 0, "w": 0, "h": 100}, "/tmp/pip_out"
                )
        finally:
            os.unlink(path)


class TestCreateSideBySide(unittest.TestCase):
    """Tests for create_side_by_side."""

    def test_missing_speaker(self):
        from opencut.core.pip_lecture import create_side_by_side
        with self.assertRaises(FileNotFoundError):
            create_side_by_side("/nonexistent.mp4", "/also_missing.mp4", "/out.mp4")

    def test_invalid_layout(self):
        from opencut.core.pip_lecture import create_side_by_side
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f1:
            f1.write(b"fake")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f2:
            f2.write(b"fake")
            p2 = f2.name
        try:
            with self.assertRaises(ValueError):
                create_side_by_side(p1, p2, "/tmp/out.mp4", layout="diagonal")
        finally:
            os.unlink(p1)
            os.unlink(p2)


# ============================================================
# Auto-Quiz Overlay (33.3)
# ============================================================
class TestQuizDataTypes(unittest.TestCase):
    """Tests for quiz_overlay data types."""

    def test_quiz_question_to_dict(self):
        from opencut.core.quiz_overlay import QuizQuestion
        q = QuizQuestion(
            question="What is X?",
            options=["A", "B", "C", "D"],
            correct_index=2,
            source_sentence="X is a concept.",
            keyword="concept",
        )
        d = q.to_dict()
        self.assertEqual(d["correct_index"], 2)
        self.assertEqual(len(d["options"]), 4)

    def test_quiz_overlay_result(self):
        from opencut.core.quiz_overlay import QuizOverlayResult
        r = QuizOverlayResult(output_path="/out.mp4", question_count=5, resolution=(1920, 1080))
        d = r.to_dict()
        self.assertEqual(d["question_count"], 5)

    def test_quiz_insert_result(self):
        from opencut.core.quiz_overlay import QuizInsertResult
        r = QuizInsertResult(output_path="/out.mp4", questions_inserted=3, chapter_count=5)
        d = r.to_dict()
        self.assertEqual(d["questions_inserted"], 3)


class TestKeywordExtraction(unittest.TestCase):
    """Tests for keyword extraction helpers."""

    def test_tokenize(self):
        from opencut.core.quiz_overlay import _tokenize
        tokens = _tokenize("Hello World! Testing 123.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)

    def test_split_sentences(self):
        from opencut.core.quiz_overlay import _split_sentences
        text = "First sentence here. Second one too! And a third?"
        sentences = _split_sentences(text)
        self.assertTrue(len(sentences) >= 2)

    def test_extract_keywords(self):
        from opencut.core.quiz_overlay import _extract_keywords
        text = (
            "Machine learning algorithms process data to find patterns. "
            "Deep learning is a subset of machine learning that uses neural networks. "
            "Neural networks consist of layers of interconnected nodes. "
            "Training involves adjusting weights through backpropagation. "
            "The algorithm learns from labeled training data."
        )
        keywords = _extract_keywords(text, top_n=5)
        self.assertTrue(len(keywords) >= 1)
        # Should find domain-relevant terms
        kw_names = [kw for kw, _ in keywords]
        self.assertTrue(any("learn" in k for k in kw_names))

    def test_extract_keywords_empty(self):
        from opencut.core.quiz_overlay import _extract_keywords
        result = _extract_keywords("")
        self.assertEqual(result, [])


class TestGenerateQuizQuestions(unittest.TestCase):
    """Tests for generate_quiz_questions."""

    def test_basic_generation(self):
        from opencut.core.quiz_overlay import generate_quiz_questions
        transcript = (
            "Photosynthesis is the process by which plants convert sunlight into energy. "
            "Chlorophyll is the green pigment responsible for absorbing light. "
            "Carbon dioxide and water are the main reactants in photosynthesis. "
            "Oxygen is released as a byproduct of the photosynthesis reaction. "
            "The Calvin cycle is the light-independent stage of photosynthesis."
        )
        questions = generate_quiz_questions(transcript, count=3)
        self.assertTrue(len(questions) >= 1)
        self.assertTrue(len(questions) <= 3)
        q = questions[0]
        self.assertIn("question", q)
        self.assertIn("options", q)
        self.assertEqual(len(q["options"]), 4)
        self.assertIn("correct_index", q)

    def test_short_transcript(self):
        from opencut.core.quiz_overlay import generate_quiz_questions
        with self.assertRaises(ValueError):
            generate_quiz_questions("Too short")

    def test_invalid_count(self):
        from opencut.core.quiz_overlay import generate_quiz_questions
        with self.assertRaises(ValueError):
            generate_quiz_questions("x" * 100, count=0)

    def test_invalid_difficulty(self):
        from opencut.core.quiz_overlay import generate_quiz_questions
        with self.assertRaises(ValueError):
            generate_quiz_questions("x" * 100, difficulty="extreme")

    def test_difficulty_levels(self):
        from opencut.core.quiz_overlay import generate_quiz_questions
        transcript = (
            "Distributed systems coordinate multiple computers to work together. "
            "Consensus algorithms ensure all nodes agree on the system state. "
            "The Raft protocol is a popular consensus algorithm for replication. "
            "Leader election selects a coordinator node among the cluster. "
            "Log replication ensures data consistency across nodes."
        )
        for diff in ("easy", "medium", "hard"):
            qs = generate_quiz_questions(transcript, count=1, difficulty=diff)
            self.assertTrue(len(qs) >= 1)
            self.assertEqual(qs[0]["difficulty"], diff)


class TestRenderQuizOverlay(unittest.TestCase):
    """Tests for render_quiz_overlay."""

    def test_empty_questions(self):
        from opencut.core.quiz_overlay import render_quiz_overlay
        with self.assertRaises(ValueError):
            render_quiz_overlay([], "/tmp/out.mp4")


class TestInsertQuizAtChapters(unittest.TestCase):
    """Tests for insert_quiz_at_chapters."""

    def test_missing_video(self):
        from opencut.core.quiz_overlay import insert_quiz_at_chapters
        with self.assertRaises(FileNotFoundError):
            insert_quiz_at_chapters(
                "/nonexistent.mp4",
                [{"question": "Q?", "options": ["A", "B", "C", "D"], "correct_index": 0}],
                [{"start_time": 0, "end_time": 10}],
                "/tmp/out.mp4",
            )

    def test_empty_questions(self):
        from opencut.core.quiz_overlay import insert_quiz_at_chapters
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                insert_quiz_at_chapters(
                    path, [], [{"start_time": 0, "end_time": 10}], "/tmp/out.mp4"
                )
        finally:
            os.unlink(path)

    def test_empty_chapters(self):
        from opencut.core.quiz_overlay import insert_quiz_at_chapters
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                insert_quiz_at_chapters(
                    path,
                    [{"question": "Q?", "options": ["A", "B", "C", "D"], "correct_index": 0}],
                    [],
                    "/tmp/out.mp4",
                )
        finally:
            os.unlink(path)

    def test_invalid_position(self):
        from opencut.core.quiz_overlay import insert_quiz_at_chapters
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                insert_quiz_at_chapters(
                    path,
                    [{"question": "Q?", "options": ["A", "B", "C", "D"], "correct_index": 0}],
                    [{"start_time": 0, "end_time": 10}],
                    "/tmp/out.mp4",
                    position="middle",
                )
        finally:
            os.unlink(path)


# ============================================================
# Education Routes (smoke tests)
# ============================================================
class TestEducationRoutesSmoke(unittest.TestCase):
    """Smoke tests for education route registration and basic validation."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        config = OpenCutConfig()
        cls.app = create_app(config=config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        resp = cls.client.get("/health")
        cls.csrf = resp.get_json().get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf,
            "Content-Type": "application/json",
        }

    def test_click_overlay_no_events(self):
        resp = self.client.post(
            "/api/tutorial/click-overlay",
            headers=self.headers,
            data=json.dumps({"filepath": "/tmp/fake.mp4", "click_events": []}),
        )
        # Should fail on validation (missing file or empty events)
        self.assertIn(resp.status_code, (400, 200))

    def test_keystroke_overlay_no_events(self):
        resp = self.client.post(
            "/api/tutorial/keystroke-overlay",
            headers=self.headers,
            data=json.dumps({"filepath": "/tmp/fake.mp4", "keystroke_events": []}),
        )
        self.assertIn(resp.status_code, (400, 200))

    def test_parse_click_log_no_path(self):
        resp = self.client.post(
            "/api/tutorial/parse-click-log",
            headers=self.headers,
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_callout_no_annotations(self):
        resp = self.client.post(
            "/api/tutorial/callout",
            headers=self.headers,
            data=json.dumps({"filepath": "/tmp/fake.mp4", "annotations": []}),
        )
        self.assertIn(resp.status_code, (400, 200))

    def test_step_callout_sync(self):
        resp = self.client.post(
            "/api/tutorial/step-callout",
            headers=self.headers,
            data=json.dumps({"text": "Click the button", "number": 1}),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["text"], "Click the button")
        self.assertEqual(data["number"], 1)

    def test_step_callout_empty_text(self):
        resp = self.client.post(
            "/api/tutorial/step-callout",
            headers=self.headers,
            data=json.dumps({"text": "", "number": 1}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_screenshot_video_no_images(self):
        resp = self.client.post(
            "/api/tutorial/screenshot-video",
            headers=self.headers,
            data=json.dumps({"image_paths": [], "output_path": "/tmp/out.mp4"}),
        )
        # Returns job_id; error happens async
        self.assertIn(resp.status_code, (200, 400))

    def test_detect_roi_no_path(self):
        resp = self.client.post(
            "/api/tutorial/detect-roi",
            headers=self.headers,
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_ken_burns_keyframes_sync(self):
        resp = self.client.post(
            "/api/tutorial/ken-burns-keyframes",
            headers=self.headers,
            data=json.dumps({
                "rois": [{"x": 0, "y": 0, "w": 100, "h": 100, "weight": 1.0}],
                "duration": 5.0,
            }),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("keyframes", data)
        self.assertGreater(data["count"], 0)

    def test_slide_chapters_sync(self):
        resp = self.client.post(
            "/api/education/slide-chapters",
            headers=self.headers,
            data=json.dumps({
                "timestamps": [0, 15, 30],
                "video_duration": 45,
            }),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["count"], 3)

    def test_slide_chapters_no_timestamps(self):
        resp = self.client.post(
            "/api/education/slide-chapters",
            headers=self.headers,
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_quiz_generate_sync(self):
        resp = self.client.post(
            "/api/education/quiz-generate",
            headers=self.headers,
            data=json.dumps({
                "transcript": (
                    "Photosynthesis converts sunlight into chemical energy. "
                    "Chlorophyll absorbs light in plant cells. "
                    "Carbon dioxide combines with water during photosynthesis. "
                    "Glucose is produced as the main product. "
                    "Oxygen is released into the atmosphere."
                ),
                "count": 2,
            }),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("questions", data)
        self.assertGreater(data["count"], 0)

    def test_quiz_generate_no_transcript(self):
        resp = self.client.post(
            "/api/education/quiz-generate",
            headers=self.headers,
            data=json.dumps({}),
        )
        self.assertEqual(resp.status_code, 400)

    def test_pip_side_by_side_missing_paths(self):
        resp = self.client.post(
            "/api/education/pip-side-by-side",
            headers=self.headers,
            data=json.dumps({}),
        )
        # Returns job_id; error happens async
        self.assertIn(resp.status_code, (200, 400))

    def test_csrf_required(self):
        """Routes should reject requests without CSRF token."""
        resp = self.client.post(
            "/api/tutorial/step-callout",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": "Test", "number": 1}),
        )
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
