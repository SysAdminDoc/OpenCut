"""
Tests for OpenCut editing workflow features.

Covers: script_storyboard, multi_publish, paper_edit, template_assembly,
        timeline_copilot, programmatic_video, multilang_subtitle,
        speaker_layout, ceremony_autoedit, and editing_workflow_routes.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 4.7 — Script/Storyboard Integration Tests
# ============================================================
class TestScriptParsing(unittest.TestCase):
    """Tests for opencut.core.script_storyboard.parse_script."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        self.tmp.write(
            "INT. OFFICE - DAY\n\n"
            "JOHN walks in and sits down.\n\n"
            "JOHN\n"
            "    Hello, how are you?\n\n"
            "JANE\n"
            "    I'm good, thanks.\n\n"
            "CUT TO:\n\n"
            "EXT. PARK - EVENING\n\n"
            "Birds fly over the sunset.\n"
        )
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_parse_basic(self):
        from opencut.core.script_storyboard import parse_script
        lines = parse_script(self.tmp.name)
        self.assertGreater(len(lines), 0)

    def test_parse_scene_headings(self):
        from opencut.core.script_storyboard import parse_script
        lines = parse_script(self.tmp.name)
        headings = [ln for ln in lines if ln.line_type == "scene_heading"]
        self.assertGreaterEqual(len(headings), 2)
        self.assertIn("INT.", headings[0].text)

    def test_parse_dialogue(self):
        from opencut.core.script_storyboard import parse_script
        lines = parse_script(self.tmp.name)
        dialogue = [ln for ln in lines if ln.line_type == "dialogue"]
        self.assertGreater(len(dialogue), 0)

    def test_parse_transitions(self):
        from opencut.core.script_storyboard import parse_script
        lines = parse_script(self.tmp.name)
        transitions = [ln for ln in lines if ln.line_type == "transition"]
        self.assertGreaterEqual(len(transitions), 1)

    def test_parse_missing_file(self):
        from opencut.core.script_storyboard import parse_script
        with self.assertRaises(FileNotFoundError):
            parse_script("/nonexistent/script.txt")

    def test_parse_with_progress(self):
        from opencut.core.script_storyboard import parse_script
        progress_calls = []
        def on_progress(pct, msg):
            progress_calls.append((pct, msg))
        parse_script(self.tmp.name, on_progress=on_progress)
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 100)

    def test_parse_action_lines(self):
        from opencut.core.script_storyboard import parse_script
        lines = parse_script(self.tmp.name)
        actions = [ln for ln in lines if ln.line_type == "action"]
        self.assertGreater(len(actions), 0)


class TestScriptAlignment(unittest.TestCase):
    """Tests for align_script_to_transcript."""

    def test_align_basic(self):
        from opencut.core.script_storyboard import ScriptLine, align_script_to_transcript
        script = [
            ScriptLine(0, "dialogue", "Hello how are you", character="JOHN"),
            ScriptLine(1, "dialogue", "I am good thanks", character="JANE"),
        ]
        transcript = [
            {"text": "Hello, how are you?", "start": 0.0, "end": 2.5},
            {"text": "I am good, thanks!", "start": 3.0, "end": 5.0},
        ]
        alignment = align_script_to_transcript(script, transcript)
        self.assertEqual(len(alignment), 2)
        self.assertTrue(alignment[0].covered)
        self.assertTrue(alignment[1].covered)

    def test_align_no_match(self):
        from opencut.core.script_storyboard import ScriptLine, align_script_to_transcript
        script = [ScriptLine(0, "dialogue", "something completely unrelated")]
        transcript = [{"text": "different words entirely", "start": 0.0, "end": 2.0}]
        alignment = align_script_to_transcript(script, transcript)
        self.assertFalse(alignment[0].covered)

    def test_align_scene_heading_skipped(self):
        from opencut.core.script_storyboard import ScriptLine, align_script_to_transcript
        script = [ScriptLine(0, "scene_heading", "INT. OFFICE - DAY")]
        transcript = [{"text": "Hello there", "start": 0.0, "end": 1.0}]
        alignment = align_script_to_transcript(script, transcript)
        self.assertFalse(alignment[0].covered)

    def test_align_with_progress(self):
        from opencut.core.script_storyboard import ScriptLine, align_script_to_transcript
        script = [ScriptLine(0, "dialogue", "Hello world")]
        transcript = [{"text": "Hello world", "start": 0.0, "end": 1.0}]
        progress = []
        align_script_to_transcript(
            script, transcript,
            on_progress=lambda p, m: progress.append(p),
        )
        self.assertGreater(len(progress), 0)


class TestMissingCoverage(unittest.TestCase):
    """Tests for find_missing_coverage."""

    def test_finds_missing(self):
        from opencut.core.script_storyboard import (
            ScriptAlignment,
            ScriptLine,
            find_missing_coverage,
        )
        alignment = [
            ScriptAlignment(ScriptLine(0, "dialogue", "Found"), covered=True),
            ScriptAlignment(ScriptLine(1, "dialogue", "Missing"), covered=False),
            ScriptAlignment(ScriptLine(2, "action", "Also missing"), covered=False),
        ]
        missing = find_missing_coverage(alignment)
        self.assertEqual(len(missing), 2)

    def test_no_missing(self):
        from opencut.core.script_storyboard import (
            ScriptAlignment,
            ScriptLine,
            find_missing_coverage,
        )
        alignment = [
            ScriptAlignment(ScriptLine(0, "dialogue", "Found"), covered=True),
        ]
        missing = find_missing_coverage(alignment)
        self.assertEqual(len(missing), 0)


class TestBrollSuggestions(unittest.TestCase):
    """Tests for suggest_broll_from_script."""

    def test_suggest_basic(self):
        from opencut.core.script_storyboard import ScriptLine, suggest_broll_from_script
        lines = [
            ScriptLine(0, "action", "Cars drive down the city street at sunset"),
            ScriptLine(1, "dialogue", "Hello there"),
        ]
        suggestions = suggest_broll_from_script(lines)
        self.assertGreater(len(suggestions), 0)
        self.assertGreater(len(suggestions[0].keywords), 0)

    def test_no_keywords(self):
        from opencut.core.script_storyboard import ScriptLine, suggest_broll_from_script
        lines = [ScriptLine(0, "action", "He smiles")]
        suggestions = suggest_broll_from_script(lines)
        self.assertEqual(len(suggestions), 0)

    def test_media_library_matching(self):
        from opencut.core.script_storyboard import ScriptLine, suggest_broll_from_script
        lines = [ScriptLine(0, "action", "A sunset over the ocean")]
        suggestions = suggest_broll_from_script(
            lines, media_library=["/clips/sunset_timelapse.mp4"]
        )
        self.assertGreater(len(suggestions), 0)
        self.assertIn("matching clip", suggestions[0].suggestion)


# ============================================================
# 5.1 — Multi-Platform Batch Publish Tests
# ============================================================
class TestPublishQueue(unittest.TestCase):
    """Tests for multi_publish.create_publish_queue."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_create_queue(self):
        from opencut.core.multi_publish import create_publish_queue
        queue = create_publish_queue(self.tmp.name, ["youtube", "tiktok"])
        self.assertEqual(queue.total, 2)
        self.assertEqual(len(queue.items), 2)

    def test_invalid_platform(self):
        from opencut.core.multi_publish import create_publish_queue
        with self.assertRaises(ValueError):
            create_publish_queue(self.tmp.name, ["nonexistent"])

    def test_missing_video(self):
        from opencut.core.multi_publish import create_publish_queue
        with self.assertRaises(FileNotFoundError):
            create_publish_queue("/nonexistent.mp4", ["youtube"])

    def test_queue_with_config(self):
        from opencut.core.multi_publish import PublishConfig, create_publish_queue
        config = {"youtube": PublishConfig(platform="youtube", title="My Video")}
        queue = create_publish_queue(self.tmp.name, ["youtube"], config=config)
        self.assertEqual(queue.items[0].config.title, "My Video")


class TestExportForPlatform(unittest.TestCase):
    """Tests for multi_publish.export_for_platform."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.multi_publish.run_ffmpeg")
    @patch("opencut.core.multi_publish.get_video_info")
    def test_export_youtube(self, mock_info, mock_ffmpeg):
        from opencut.core.multi_publish import export_for_platform
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        mock_ffmpeg.return_value = ""
        result = export_for_platform(self.tmp.name, "youtube")
        self.assertTrue(result.endswith(".mp4"))
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.multi_publish.run_ffmpeg")
    @patch("opencut.core.multi_publish.get_video_info")
    def test_export_tiktok(self, mock_info, mock_ffmpeg):
        from opencut.core.multi_publish import export_for_platform
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 30}
        mock_ffmpeg.return_value = ""
        result = export_for_platform(self.tmp.name, "tiktok")
        self.assertIn("tiktok", result)

    def test_export_invalid_platform(self):
        from opencut.core.multi_publish import export_for_platform
        with self.assertRaises(ValueError):
            export_for_platform(self.tmp.name, "fakebook")

    def test_export_missing_file(self):
        from opencut.core.multi_publish import export_for_platform
        with self.assertRaises(FileNotFoundError):
            export_for_platform("/nonexistent.mp4", "youtube")


class TestPublishToPlatform(unittest.TestCase):
    """Tests for multi_publish.publish_to_platform."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.multi_publish.get_video_info")
    def test_publish_stub(self, mock_info):
        from opencut.core.multi_publish import publish_to_platform
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        result = publish_to_platform(self.tmp.name, "youtube")
        self.assertEqual(result["platform"], "youtube")
        self.assertIn("status", result)

    @patch("opencut.core.multi_publish.get_video_info")
    def test_publish_exceeds_duration(self, mock_info):
        from opencut.core.multi_publish import publish_to_platform
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 99999}
        with self.assertRaises(ValueError):
            publish_to_platform(self.tmp.name, "twitter")


# ============================================================
# 14.1 — Paper Edit Tests
# ============================================================
class TestPaperEdit(unittest.TestCase):
    """Tests for opencut.core.paper_edit."""

    def test_create_basic(self):
        from opencut.core.paper_edit import create_paper_edit
        transcript = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "Goodbye", "start": 5.0, "end": 7.0},
        ]
        selections = [
            {"start": 0.0, "end": 2.0},
            {"start": 5.0, "end": 7.0},
        ]
        edit = create_paper_edit(transcript, selections)
        self.assertEqual(len(edit.selections), 2)
        self.assertAlmostEqual(edit.total_duration, 4.0)

    def test_create_empty_selections(self):
        from opencut.core.paper_edit import create_paper_edit
        with self.assertRaises(ValueError):
            create_paper_edit([], [])

    def test_create_invalid_range(self):
        from opencut.core.paper_edit import create_paper_edit
        with self.assertRaises(ValueError):
            create_paper_edit([], [{"start": 5.0, "end": 3.0}])

    def test_reorder(self):
        from opencut.core.paper_edit import PaperEdit, PaperEditSelection
        edit = PaperEdit(selections=[
            PaperEditSelection(start=0, end=2, order=0, label="A"),
            PaperEditSelection(start=5, end=7, order=1, label="B"),
        ])
        edit.reorder([1, 0])
        self.assertEqual(edit.selections[0].label, "B")
        self.assertEqual(edit.selections[1].label, "A")

    def test_reorder_invalid(self):
        from opencut.core.paper_edit import PaperEdit, PaperEditSelection
        edit = PaperEdit(selections=[
            PaperEditSelection(start=0, end=2, order=0),
        ])
        with self.assertRaises(ValueError):
            edit.reorder([1, 0])

    def test_text_enrichment(self):
        from opencut.core.paper_edit import create_paper_edit
        transcript = [{"text": "Hello world", "start": 0.0, "end": 2.0}]
        selections = [{"start": 0.0, "end": 2.0}]
        edit = create_paper_edit(transcript, selections)
        self.assertIn("Hello", edit.selections[0].text)


class TestPaperEditExport(unittest.TestCase):
    """Tests for paper_edit.export_paper_edit."""

    def test_export_json(self):
        from opencut.core.paper_edit import PaperEdit, PaperEditSelection, export_paper_edit
        edit = PaperEdit(selections=[
            PaperEditSelection(start=0, end=2, text="Hello", order=0),
        ])
        edit._recalc()
        result = export_paper_edit(edit, format="json")
        self.assertTrue(os.path.isfile(result["output_path"]))
        os.unlink(result["output_path"])

    def test_export_txt(self):
        from opencut.core.paper_edit import PaperEdit, PaperEditSelection, export_paper_edit
        edit = PaperEdit(selections=[
            PaperEditSelection(start=0, end=2, text="Hello", order=0),
        ])
        edit._recalc()
        result = export_paper_edit(edit, format="txt")
        self.assertTrue(os.path.isfile(result["output_path"]))
        with open(result["output_path"], "r") as f:
            content = f.read()
        self.assertIn("Paper Edit", content)
        os.unlink(result["output_path"])

    def test_export_edl(self):
        from opencut.core.paper_edit import PaperEdit, PaperEditSelection, export_paper_edit
        edit = PaperEdit(selections=[
            PaperEditSelection(start=0, end=2, text="Hello", order=0),
        ])
        edit._recalc()
        result = export_paper_edit(edit, format="edl")
        self.assertTrue(os.path.isfile(result["output_path"]))
        os.unlink(result["output_path"])

    def test_export_invalid_format(self):
        from opencut.core.paper_edit import PaperEdit, PaperEditSelection, export_paper_edit
        edit = PaperEdit(selections=[
            PaperEditSelection(start=0, end=2, text="Hello", order=0),
        ])
        edit._recalc()
        with self.assertRaises(ValueError):
            export_paper_edit(edit, format="csv")


# ============================================================
# 15.3 — Template Assembly Tests
# ============================================================
class TestTemplateList(unittest.TestCase):
    """Tests for template_assembly.list_templates."""

    def test_list_all(self):
        from opencut.core.template_assembly import list_templates
        templates = list_templates()
        self.assertGreater(len(templates), 0)

    def test_list_by_category(self):
        from opencut.core.template_assembly import list_templates
        templates = list_templates(category="general")
        self.assertGreater(len(templates), 0)
        for t in templates:
            self.assertEqual(t["category"], "general")

    def test_list_empty_category(self):
        from opencut.core.template_assembly import list_templates
        templates = list_templates(category="nonexistent_category")
        self.assertEqual(len(templates), 0)


class TestTemplateLoad(unittest.TestCase):
    """Tests for template_assembly.load_template."""

    def test_load_valid(self):
        from opencut.core.template_assembly import load_template
        tmpl = load_template("split_screen")
        self.assertEqual(tmpl.name, "split_screen")
        self.assertGreater(len(tmpl.placeholders), 0)

    def test_load_invalid(self):
        from opencut.core.template_assembly import load_template
        with self.assertRaises(ValueError):
            load_template("nonexistent_template")


class TestTemplateFill(unittest.TestCase):
    """Tests for template_assembly.fill_template."""

    def setUp(self):
        self.tmp1 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp1.write(b"\x00" * 100)
        self.tmp1.close()
        self.tmp2 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp2.write(b"\x00" * 100)
        self.tmp2.close()

    def tearDown(self):
        for f in (self.tmp1.name, self.tmp2.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    def test_fill_split_screen(self):
        from opencut.core.template_assembly import fill_template, load_template
        tmpl = load_template("split_screen")
        filled = fill_template(tmpl, {
            "left_video": self.tmp1.name,
            "right_video": self.tmp2.name,
        })
        self.assertEqual(len(filled.assignments), 2)

    def test_fill_missing_required(self):
        from opencut.core.template_assembly import fill_template, load_template
        tmpl = load_template("split_screen")
        with self.assertRaises(ValueError):
            fill_template(tmpl, {"left_video": self.tmp1.name})

    def test_fill_bad_file(self):
        from opencut.core.template_assembly import fill_template, load_template
        tmpl = load_template("split_screen")
        with self.assertRaises(FileNotFoundError):
            fill_template(tmpl, {
                "left_video": "/nonexistent.mp4",
                "right_video": self.tmp2.name,
            })

    def test_fill_unknown_placeholder(self):
        from opencut.core.template_assembly import fill_template, load_template
        tmpl = load_template("split_screen")
        with self.assertRaises(ValueError):
            fill_template(tmpl, {
                "left_video": self.tmp1.name,
                "right_video": self.tmp2.name,
                "nonexistent": self.tmp1.name,
            })


# ============================================================
# 21.1 — Timeline Copilot Tests
# ============================================================
class TestCopilotQuery(unittest.TestCase):
    """Tests for timeline_copilot.process_copilot_query."""

    def test_trim_start(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("trim the first 5 seconds")
        self.assertEqual(action.action_type, "trim")
        self.assertEqual(action.parameters["seconds"], 5.0)

    def test_trim_end(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("cut the last 10 seconds")
        self.assertEqual(action.action_type, "trim")
        self.assertEqual(action.parameters["seconds"], 10.0)

    def test_cut_range(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("cut from 10s to 20s")
        self.assertEqual(action.action_type, "cut")
        self.assertAlmostEqual(action.parameters["start"], 10.0)
        self.assertAlmostEqual(action.parameters["end"], 20.0)

    def test_speed_up(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("speed up 2x")
        self.assertEqual(action.action_type, "speed")
        self.assertAlmostEqual(action.parameters["factor"], 2.0)

    def test_mute(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("mute the audio")
        self.assertEqual(action.action_type, "volume")
        self.assertAlmostEqual(action.parameters["level"], 0.0)

    def test_volume(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("set volume to 50%")
        self.assertEqual(action.action_type, "volume")
        self.assertAlmostEqual(action.parameters["level"], 0.5)

    def test_duration_query(self):
        from opencut.core.timeline_copilot import (
            TimelineContext,
            process_copilot_query,
        )
        ctx = TimelineContext(duration=120.5, width=1920, height=1080, fps=30)
        action = process_copilot_query("how long is the video", context=ctx)
        self.assertEqual(action.action_type, "info")
        self.assertIn("duration", str(action.result))

    def test_find_text(self):
        from opencut.core.timeline_copilot import (
            TimelineContext,
            process_copilot_query,
        )
        ctx = TimelineContext(
            transcript=[{"text": "Hello world", "start": 0, "end": 2}],
        )
        action = process_copilot_query("find where someone says 'Hello'", context=ctx)
        self.assertEqual(action.action_type, "find")

    def test_empty_query(self):
        from opencut.core.timeline_copilot import process_copilot_query
        with self.assertRaises(ValueError):
            process_copilot_query("")

    def test_unrecognized_query(self):
        from opencut.core.timeline_copilot import process_copilot_query
        action = process_copilot_query("what is the meaning of life")
        self.assertEqual(action.action_type, "info")
        self.assertLess(action.confidence, 0.5)


class TestCopilotContext(unittest.TestCase):
    """Tests for timeline_copilot.build_timeline_context."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.timeline_copilot.get_video_info")
    def test_build_context(self, mock_info):
        from opencut.core.timeline_copilot import build_timeline_context
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}
        ctx = build_timeline_context(self.tmp.name)
        self.assertEqual(ctx.width, 1920)
        self.assertEqual(ctx.duration, 60)

    def test_build_context_missing_file(self):
        from opencut.core.timeline_copilot import build_timeline_context
        with self.assertRaises(FileNotFoundError):
            build_timeline_context("/nonexistent.mp4")


# ============================================================
# 21.5 — Programmatic Video Tests
# ============================================================
class TestDataVideoTemplate(unittest.TestCase):
    """Tests for programmatic_video.DataVideoTemplate."""

    def test_validate_ok(self):
        from opencut.core.programmatic_video import DataVideoTemplate
        t = DataVideoTemplate(name="test", width=1920, height=1080, duration=10)
        t.validate()  # should not raise

    def test_validate_bad_width(self):
        from opencut.core.programmatic_video import DataVideoTemplate
        t = DataVideoTemplate(name="test", width=0, height=1080, duration=10)
        with self.assertRaises(ValueError):
            t.validate()

    def test_validate_bad_duration(self):
        from opencut.core.programmatic_video import DataVideoTemplate
        t = DataVideoTemplate(name="test", width=1920, height=1080, duration=0)
        with self.assertRaises(ValueError):
            t.validate()

    def test_validate_odd_dimensions(self):
        from opencut.core.programmatic_video import DataVideoTemplate
        t = DataVideoTemplate(name="test", width=1921, height=1080, duration=10)
        with self.assertRaises(ValueError):
            t.validate()

    def test_validate_missing_key(self):
        from opencut.core.programmatic_video import DataVideoTemplate
        t = DataVideoTemplate(
            name="test", width=1920, height=1080, duration=10,
            text_fields=[{"no_key": "oops"}],
        )
        with self.assertRaises(ValueError):
            t.validate()


class TestCreateDataVideo(unittest.TestCase):
    """Tests for programmatic_video.create_data_video."""

    @patch("opencut.core.programmatic_video.run_ffmpeg")
    def test_create_basic(self, mock_ffmpeg):
        from opencut.core.programmatic_video import DataVideoTemplate, create_data_video
        mock_ffmpeg.return_value = ""
        template = DataVideoTemplate(
            name="test", width=1920, height=1080, duration=5,
            text_fields=[{"key": "name", "fontsize": 48}],
        )
        fd, out = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            result = create_data_video(template, {"name": "John"}, out)
            self.assertEqual(result["output_path"], out)
            self.assertIn("name", result["fields_used"])
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass

    @patch("opencut.core.programmatic_video.run_ffmpeg")
    def test_create_with_progress(self, mock_ffmpeg):
        from opencut.core.programmatic_video import DataVideoTemplate, create_data_video
        mock_ffmpeg.return_value = ""
        template = DataVideoTemplate(name="test", width=1920, height=1080, duration=5)
        fd, out = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        progress = []
        try:
            create_data_video(
                template, {"name": "Test"}, out,
                on_progress=lambda p, m: progress.append(p),
            )
            self.assertGreater(len(progress), 0)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass


class TestBatchDataVideos(unittest.TestCase):
    """Tests for programmatic_video.batch_data_videos."""

    def test_missing_csv(self):
        from opencut.core.programmatic_video import DataVideoTemplate, batch_data_videos
        template = DataVideoTemplate(name="test", width=1920, height=1080, duration=5)
        with self.assertRaises(FileNotFoundError):
            batch_data_videos(template, "/nonexistent.csv", "/tmp")

    @patch("opencut.core.programmatic_video.run_ffmpeg")
    def test_batch_basic(self, mock_ffmpeg):
        from opencut.core.programmatic_video import DataVideoTemplate, batch_data_videos
        mock_ffmpeg.return_value = ""
        template = DataVideoTemplate(
            name="test", width=1920, height=1080, duration=5,
            text_fields=[{"key": "name"}],
        )
        csv_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        csv_file.write("name\nAlice\nBob\n")
        csv_file.close()
        out_dir = tempfile.mkdtemp()
        try:
            result = batch_data_videos(template, csv_file.name, out_dir)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["succeeded"], 2)
        finally:
            os.unlink(csv_file.name)
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)


# ============================================================
# 24.2 — Multi-Language Subtitle Tests
# ============================================================
class TestMultilangProject(unittest.TestCase):
    """Tests for multilang_subtitle.create_multilang_project."""

    def test_create_basic(self):
        from opencut.core.multilang_subtitle import create_multilang_project
        timing = [
            {"start": 0, "end": 2},
            {"start": 3, "end": 5},
        ]
        project = create_multilang_project(timing, ["en", "es"])
        self.assertEqual(project.entry_count(), 2)
        self.assertEqual(project.language_count(), 2)

    def test_create_with_source(self):
        from opencut.core.multilang_subtitle import create_multilang_project
        timing = [{"start": 0, "end": 2, "text": "Hello"}]
        project = create_multilang_project(
            timing, ["en", "es"], source_language="en",
        )
        self.assertEqual(project.translations["en"][0], "Hello")
        self.assertEqual(project.translations["es"][0], "")

    def test_create_empty_timing(self):
        from opencut.core.multilang_subtitle import create_multilang_project
        with self.assertRaises(ValueError):
            create_multilang_project([], ["en"])

    def test_create_empty_languages(self):
        from opencut.core.multilang_subtitle import create_multilang_project
        with self.assertRaises(ValueError):
            create_multilang_project([{"start": 0, "end": 1}], [])

    def test_create_bad_timing(self):
        from opencut.core.multilang_subtitle import create_multilang_project
        with self.assertRaises(ValueError):
            create_multilang_project([{"start": 5, "end": 3}], ["en"])


class TestMultilangUpdate(unittest.TestCase):
    """Tests for multilang_subtitle.update_language_text."""

    def test_update_basic(self):
        from opencut.core.multilang_subtitle import (
            create_multilang_project,
            update_language_text,
        )
        project = create_multilang_project(
            [{"start": 0, "end": 2}, {"start": 3, "end": 5}],
            ["en", "es"],
        )
        result = update_language_text(project, "es", ["Hola", "Adios"])
        self.assertEqual(result["entries_updated"], 2)
        self.assertEqual(project.translations["es"][0], "Hola")

    def test_update_wrong_language(self):
        from opencut.core.multilang_subtitle import (
            create_multilang_project,
            update_language_text,
        )
        project = create_multilang_project(
            [{"start": 0, "end": 2}], ["en"],
        )
        with self.assertRaises(ValueError):
            update_language_text(project, "fr", ["Bonjour"])

    def test_update_wrong_count(self):
        from opencut.core.multilang_subtitle import (
            create_multilang_project,
            update_language_text,
        )
        project = create_multilang_project(
            [{"start": 0, "end": 2}], ["en"],
        )
        with self.assertRaises(ValueError):
            update_language_text(project, "en", ["one", "two"])


class TestMultilangExport(unittest.TestCase):
    """Tests for multilang_subtitle.export_language."""

    def _make_project(self):
        from opencut.core.multilang_subtitle import (
            create_multilang_project,
            update_language_text,
        )
        project = create_multilang_project(
            [{"start": 0, "end": 2}, {"start": 3, "end": 5}],
            ["en", "es"],
        )
        update_language_text(project, "en", ["Hello", "World"])
        return project

    def test_export_srt(self):
        from opencut.core.multilang_subtitle import export_language
        project = self._make_project()
        result = export_language(project, "en", format="srt")
        self.assertTrue(os.path.isfile(result["output_path"]))
        with open(result["output_path"], "r") as f:
            content = f.read()
        self.assertIn("Hello", content)
        self.assertIn("-->", content)
        os.unlink(result["output_path"])

    def test_export_vtt(self):
        from opencut.core.multilang_subtitle import export_language
        project = self._make_project()
        result = export_language(project, "en", format="vtt")
        self.assertTrue(os.path.isfile(result["output_path"]))
        with open(result["output_path"], "r") as f:
            content = f.read()
        self.assertIn("WEBVTT", content)
        os.unlink(result["output_path"])

    def test_export_json(self):
        from opencut.core.multilang_subtitle import export_language
        project = self._make_project()
        result = export_language(project, "en", format="json")
        self.assertTrue(os.path.isfile(result["output_path"]))
        with open(result["output_path"], "r") as f:
            data = json.load(f)
        self.assertEqual(data["language"], "en")
        self.assertEqual(len(data["entries"]), 2)
        os.unlink(result["output_path"])

    def test_export_bad_format(self):
        from opencut.core.multilang_subtitle import export_language
        project = self._make_project()
        with self.assertRaises(ValueError):
            export_language(project, "en", format="xml")


class TestMultilangSync(unittest.TestCase):
    """Tests for multilang_subtitle.sync_timing_change."""

    def _make_project(self):
        from opencut.core.multilang_subtitle import create_multilang_project
        return create_multilang_project(
            [{"start": 0, "end": 2}, {"start": 3, "end": 5}],
            ["en", "es"],
        )

    def test_shift(self):
        from opencut.core.multilang_subtitle import sync_timing_change
        project = self._make_project()
        result = sync_timing_change(project, {"operation": "shift", "offset_seconds": 1.0})
        self.assertEqual(result["operation"], "shift")
        self.assertAlmostEqual(project.timing[0].start, 1.0)
        self.assertAlmostEqual(project.timing[0].end, 3.0)

    def test_update_entry(self):
        from opencut.core.multilang_subtitle import sync_timing_change
        project = self._make_project()
        sync_timing_change(
            project, {"operation": "update", "index": 0, "start": 0.5, "end": 2.5}
        )
        self.assertAlmostEqual(project.timing[0].start, 0.5)

    def test_insert(self):
        from opencut.core.multilang_subtitle import sync_timing_change
        project = self._make_project()
        sync_timing_change(
            project, {"operation": "insert", "start": 2.0, "end": 2.8}
        )
        self.assertEqual(len(project.timing), 3)
        self.assertEqual(len(project.translations["en"]), 3)

    def test_delete(self):
        from opencut.core.multilang_subtitle import sync_timing_change
        project = self._make_project()
        sync_timing_change(project, {"operation": "delete", "index": 0})
        self.assertEqual(len(project.timing), 1)
        self.assertEqual(len(project.translations["en"]), 1)

    def test_invalid_operation(self):
        from opencut.core.multilang_subtitle import sync_timing_change
        project = self._make_project()
        with self.assertRaises(ValueError):
            sync_timing_change(project, {"operation": "explode"})


# ============================================================
# 40.2 — Speaker Layout Engine Tests
# ============================================================
class TestLayoutCalculation(unittest.TestCase):
    """Tests for speaker_layout layout calculations."""

    def test_grid_single(self):
        from opencut.core.speaker_layout import _calc_grid
        positions = _calc_grid(1, 1920, 1080, 4)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["w"], 1920)

    def test_grid_four(self):
        from opencut.core.speaker_layout import _calc_grid
        positions = _calc_grid(4, 1920, 1080, 4)
        self.assertEqual(len(positions), 4)
        for p in positions:
            self.assertGreater(p["w"], 0)
            self.assertGreater(p["h"], 0)

    def test_grid_zero(self):
        from opencut.core.speaker_layout import _calc_grid
        positions = _calc_grid(0, 1920, 1080, 4)
        self.assertEqual(len(positions), 0)

    def test_spotlight(self):
        from opencut.core.speaker_layout import _calc_spotlight
        positions = _calc_spotlight(3, 1920, 1080, 0)
        self.assertEqual(len(positions), 3)
        # First should be the spotlight (larger)
        self.assertGreater(positions[0]["h"], positions[1]["h"])

    def test_pip(self):
        from opencut.core.speaker_layout import _calc_pip
        positions = _calc_pip(2, 1920, 1080, 25, "bottom_right")
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0]["w"], 1920)
        self.assertLess(positions[1]["w"], 1920)


class TestSpeakerLayout(unittest.TestCase):
    """Tests for speaker_layout.create_speaker_layout."""

    def test_empty_paths(self):
        from opencut.core.speaker_layout import create_speaker_layout
        with self.assertRaises(ValueError):
            create_speaker_layout([])

    def test_missing_file(self):
        from opencut.core.speaker_layout import create_speaker_layout
        with self.assertRaises(FileNotFoundError):
            create_speaker_layout(["/nonexistent.mp4"])


class TestActiveSpeaker(unittest.TestCase):
    """Tests for speaker_layout.apply_active_speaker."""

    def test_empty_paths(self):
        from opencut.core.speaker_layout import apply_active_speaker
        with self.assertRaises(ValueError):
            apply_active_speaker([], [{"speaker": "A", "start": 0, "end": 5}])

    def test_empty_diarization(self):
        from opencut.core.speaker_layout import apply_active_speaker
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00" * 100)
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                apply_active_speaker([tmp.name], [])
        finally:
            os.unlink(tmp.name)


# ============================================================
# 48.1 — Ceremony Auto-Edit Tests
# ============================================================
class TestCeremonyConfig(unittest.TestCase):
    """Tests for ceremony_autoedit.CeremonyConfig."""

    def test_defaults(self):
        from opencut.core.ceremony_autoedit import CeremonyConfig
        config = CeremonyConfig()
        self.assertEqual(config.segment_duration, 5.0)
        self.assertAlmostEqual(
            config.audio_weight + config.motion_weight + config.variety_weight, 1.0,
        )


class TestCeremonyScoring(unittest.TestCase):
    """Tests for ceremony_autoedit.score_camera_angles."""

    def test_empty_cameras(self):
        from opencut.core.ceremony_autoedit import score_camera_angles
        with self.assertRaises(ValueError):
            score_camera_angles([], 0.0)

    def test_missing_camera(self):
        from opencut.core.ceremony_autoedit import score_camera_angles
        with self.assertRaises(FileNotFoundError):
            score_camera_angles(["/nonexistent.mp4"], 0.0)


class TestEditDecision(unittest.TestCase):
    """Tests for ceremony_autoedit.EditDecision."""

    def test_duration(self):
        from opencut.core.ceremony_autoedit import EditDecision
        d = EditDecision(camera_index=0, start=5.0, end=10.0)
        self.assertAlmostEqual(d.duration, 5.0)


class TestGenerateMulticamEdit(unittest.TestCase):
    """Tests for ceremony_autoedit.generate_multicam_edit."""

    def test_generate_basic(self):
        from opencut.core.ceremony_autoedit import (
            CameraScore,
            CeremonyConfig,
            generate_multicam_edit,
        )
        scores = {
            0.0: [CameraScore(0, 0.0, total_score=0.8),
                  CameraScore(1, 0.0, total_score=0.5)],
            5.0: [CameraScore(1, 5.0, total_score=0.9),
                  CameraScore(0, 5.0, total_score=0.4)],
        }
        config = CeremonyConfig(segment_duration=5.0)
        decisions = generate_multicam_edit(scores, config)
        self.assertGreater(len(decisions), 0)

    def test_generate_empty(self):
        from opencut.core.ceremony_autoedit import CeremonyConfig, generate_multicam_edit
        with self.assertRaises(ValueError):
            generate_multicam_edit({}, CeremonyConfig())

    def test_merge_same_camera(self):
        from opencut.core.ceremony_autoedit import (
            CameraScore,
            CeremonyConfig,
            generate_multicam_edit,
        )
        scores = {
            0.0: [CameraScore(0, 0.0, total_score=0.8)],
            5.0: [CameraScore(0, 5.0, total_score=0.7)],
        }
        config = CeremonyConfig(segment_duration=5.0, max_segment=30)
        decisions = generate_multicam_edit(scores, config)
        # Same camera should be merged
        self.assertEqual(len(decisions), 1)
        self.assertAlmostEqual(decisions[0].start, 0.0)
        self.assertAlmostEqual(decisions[0].end, 10.0)


class TestAutoEditCeremony(unittest.TestCase):
    """Tests for ceremony_autoedit.auto_edit_ceremony."""

    def test_too_few_cameras(self):
        from opencut.core.ceremony_autoedit import auto_edit_ceremony
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00" * 100)
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                auto_edit_ceremony([tmp.name])
        finally:
            os.unlink(tmp.name)

    def test_missing_camera_file(self):
        from opencut.core.ceremony_autoedit import auto_edit_ceremony
        with self.assertRaises(FileNotFoundError):
            auto_edit_ceremony(["/a.mp4", "/b.mp4"])


# ============================================================
# Route Blueprint Tests
# ============================================================
class TestEditingWorkflowRoutes(unittest.TestCase):
    """Integration tests for editing_workflow_routes blueprint."""

    @classmethod
    def setUpClass(cls):
        """Create a minimal Flask test app."""
        from flask import Flask
        cls.app = Flask(__name__)
        cls.app.config["TESTING"] = True

        from opencut.routes.editing_workflow_routes import editing_wf_bp
        cls.app.register_blueprint(editing_wf_bp)

        # Disable CSRF for testing
        import time

        from opencut.security import _csrf_tokens
        _csrf_tokens["test_token"] = time.monotonic() + 99999

        cls.headers = {
            "Content-Type": "application/json",
            "X-OpenCut-Token": "test_token",
        }

    def test_script_parse_route_missing_file(self):
        with self.app.test_client() as c:
            resp = c.post("/script/parse", headers=self.headers,
                          data=json.dumps({"script_path": ""}))
            self.assertIn(resp.status_code, (400, 200))

    def test_publish_queue_route_no_platforms(self):
        with self.app.test_client() as c:
            resp = c.post("/publish/queue", headers=self.headers,
                          data=json.dumps({"video_path": "/test.mp4"}))
            # Will get a job_id then error in the job
            self.assertIn(resp.status_code, (200, 400))

    def test_paper_edit_create_route(self):
        with self.app.test_client() as c:
            resp = c.post("/paper-edit/create", headers=self.headers,
                          data=json.dumps({
                              "selections": [{"start": 0, "end": 2}],
                              "transcript": [{"text": "Hi", "start": 0, "end": 2}],
                          }))
            self.assertIn(resp.status_code, (200, 400))

    def test_template_list_route(self):
        with self.app.test_client() as c:
            resp = c.post("/template/list", headers=self.headers,
                          data=json.dumps({}))
            self.assertIn(resp.status_code, (200,))

    def test_copilot_query_route(self):
        with self.app.test_client() as c:
            resp = c.post("/copilot/query", headers=self.headers,
                          data=json.dumps({"query": "how long is the video"}))
            self.assertIn(resp.status_code, (200,))

    def test_copilot_query_empty(self):
        with self.app.test_client() as c:
            resp = c.post("/copilot/query", headers=self.headers,
                          data=json.dumps({"query": ""}))
            self.assertIn(resp.status_code, (200, 400))

    def test_multilang_create_route(self):
        with self.app.test_client() as c:
            resp = c.post("/multilang/create", headers=self.headers,
                          data=json.dumps({
                              "base_timing": [{"start": 0, "end": 2}],
                              "languages": ["en", "es"],
                          }))
            self.assertIn(resp.status_code, (200,))

    def test_multilang_create_no_timing(self):
        with self.app.test_client() as c:
            resp = c.post("/multilang/create", headers=self.headers,
                          data=json.dumps({"languages": ["en"]}))
            self.assertIn(resp.status_code, (200, 400))

    def test_csrf_required(self):
        with self.app.test_client() as c:
            resp = c.post("/template/list",
                          data=json.dumps({}),
                          content_type="application/json")
            self.assertEqual(resp.status_code, 403)

    def test_data_video_create_route(self):
        with self.app.test_client() as c:
            resp = c.post("/data-video/create", headers=self.headers,
                          data=json.dumps({
                              "template": {"name": "test", "width": 1920, "height": 1080},
                              "data_row": {"name": "Test"},
                          }))
            self.assertIn(resp.status_code, (200,))

    def test_ceremony_auto_edit_route_no_cameras(self):
        with self.app.test_client() as c:
            resp = c.post("/ceremony/auto-edit", headers=self.headers,
                          data=json.dumps({"camera_paths": []}))
            self.assertIn(resp.status_code, (200, 400))


if __name__ == "__main__":
    unittest.main()
