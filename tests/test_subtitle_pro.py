"""
Tests for OpenCut Professional Subtitling features.

Covers:
  - Shot-aware timing (split at cuts, gap enforcement, min duration, profiles)
  - Multi-language subtitles (CRUD, import/export, timing sync, SRT/VTT gen)
  - Broadcast captions (all 6 formats, character encoding, validation)
  - SDH formatting (speaker labels, sound events, music detection)
  - Subtitle positioning (obstruction detection, zone fallback, ASS generation)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Shot-Aware Subtitle Timing
# ============================================================
class TestShotAwareProfiles(unittest.TestCase):
    """Tests for timing profile management."""

    def test_list_profiles_returns_all(self):
        from opencut.core.subtitle_shot_aware import list_profiles
        profiles = list_profiles()
        names = [p["name"] for p in profiles]
        self.assertIn("netflix", names)
        self.assertIn("bbc", names)
        self.assertIn("fcc", names)
        self.assertIn("custom", names)

    def test_list_profiles_has_required_fields(self):
        from opencut.core.subtitle_shot_aware import list_profiles
        for p in list_profiles():
            self.assertIn("name", p)
            self.assertIn("description", p)
            self.assertIn("max_chars_per_line", p)
            self.assertIn("min_duration", p)
            self.assertIn("fps", p)

    def test_get_profile_netflix(self):
        from opencut.core.subtitle_shot_aware import get_profile
        p = get_profile("netflix")
        self.assertEqual(p["max_chars_per_line"], 42)
        self.assertEqual(p["max_lines"], 2)

    def test_get_profile_bbc(self):
        from opencut.core.subtitle_shot_aware import get_profile
        p = get_profile("bbc")
        self.assertEqual(p["max_chars_per_line"], 37)
        self.assertEqual(p["fps"], 25)

    def test_get_profile_fcc(self):
        from opencut.core.subtitle_shot_aware import get_profile
        p = get_profile("fcc")
        self.assertEqual(p["max_chars_per_line"], 32)
        self.assertEqual(p["max_lines"], 4)

    def test_get_profile_unknown_raises(self):
        from opencut.core.subtitle_shot_aware import get_profile
        with self.assertRaises(ValueError):
            get_profile("nonexistent")

    def test_get_profile_case_insensitive(self):
        from opencut.core.subtitle_shot_aware import get_profile
        p = get_profile("Netflix")
        self.assertEqual(p["max_chars_per_line"], 42)


class TestShotAwareSplitting(unittest.TestCase):
    """Tests for splitting subtitles at cut points."""

    def test_no_cuts_no_change(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [SubtitleSegment(index=1, start=1.0, end=3.0, text="Hello world")]
        result = process_shot_aware(subs, cuts=[], profile="netflix")
        self.assertEqual(len(result.adjusted_subtitles), 1)
        self.assertEqual(result.splits_made, 0)

    def test_split_at_single_cut(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [SubtitleSegment(index=1, start=1.0, end=5.0, text="Hello world today")]
        result = process_shot_aware(subs, cuts=[3.0], profile="netflix")
        self.assertGreater(len(result.adjusted_subtitles), 1)
        self.assertGreater(result.splits_made, 0)

    def test_subtitle_not_crossing_cut_unchanged(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [SubtitleSegment(index=1, start=1.0, end=2.5, text="Before cut")]
        result = process_shot_aware(subs, cuts=[5.0], profile="netflix")
        self.assertEqual(len(result.adjusted_subtitles), 1)

    def test_multiple_cuts_split_correctly(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [SubtitleSegment(
            index=1, start=1.0, end=10.0,
            text="word1 word2 word3 word4 word5 word6",
        )]
        result = process_shot_aware(subs, cuts=[3.0, 6.0], profile="netflix")
        self.assertGreater(len(result.adjusted_subtitles), 2)

    def test_empty_subtitles_returns_empty(self):
        from opencut.core.subtitle_shot_aware import process_shot_aware
        result = process_shot_aware([], cuts=[5.0])
        self.assertEqual(len(result.adjusted_subtitles), 0)
        self.assertEqual(result.profile_used, "netflix")


class TestShotAwareGapEnforcement(unittest.TestCase):
    """Tests for gap enforcement near cut points."""

    def test_gap_enforced_near_cut(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        # Subtitle ends exactly at cut - gap should be enforced
        subs = [SubtitleSegment(index=1, start=1.0, end=5.0, text="End near cut")]
        result = process_shot_aware(subs, cuts=[5.01], profile="netflix")
        for s in result.adjusted_subtitles:
            self.assertLessEqual(s.end, 5.01)

    def test_gaps_enforced_count(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [
            SubtitleSegment(index=1, start=1.0, end=4.98, text="First"),
            SubtitleSegment(index=2, start=5.02, end=8.0, text="Second"),
        ]
        result = process_shot_aware(subs, cuts=[5.0], profile="netflix")
        self.assertIsInstance(result.gaps_enforced, int)


class TestShotAwareMinDuration(unittest.TestCase):
    """Tests for minimum duration enforcement."""

    def test_short_subtitle_extended(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [SubtitleSegment(index=1, start=1.0, end=1.2, text="Short")]
        result = process_shot_aware(subs, cuts=[], profile="netflix")
        for s in result.adjusted_subtitles:
            self.assertGreaterEqual(s.duration, 0.833)

    def test_violations_fixed_counted(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware
        subs = [
            SubtitleSegment(index=1, start=1.0, end=1.1, text="A"),
            SubtitleSegment(index=2, start=5.0, end=5.1, text="B"),
        ]
        result = process_shot_aware(subs, cuts=[], profile="netflix")
        self.assertGreaterEqual(result.violations_fixed, 2)


class TestShotAwareLineWrap(unittest.TestCase):
    """Tests for line wrapping enforcement."""

    def test_long_line_wrapped(self):
        from opencut.core.subtitle_shot_aware import _wrap_text
        text = "This is a very long subtitle line that should definitely be wrapped to fit"
        wrapped = _wrap_text(text, 42, 2)
        for line in wrapped.split("\n"):
            self.assertLessEqual(len(line), 42)

    def test_short_line_not_wrapped(self):
        from opencut.core.subtitle_shot_aware import _wrap_text
        text = "Short text"
        wrapped = _wrap_text(text, 42, 2)
        self.assertEqual(wrapped, "Short text")

    def test_max_lines_enforced(self):
        from opencut.core.subtitle_shot_aware import _wrap_text
        text = "Line one\nLine two\nLine three\nLine four"
        wrapped = _wrap_text(text, 42, 2)
        self.assertLessEqual(len(wrapped.split("\n")), 2)


class TestShotAwareExport(unittest.TestCase):
    """Tests for SRT/VTT/ASS export."""

    def test_export_srt_format(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, export_srt
        subs = [SubtitleSegment(index=1, start=1.0, end=3.0, text="Hello")]
        srt = export_srt(subs)
        self.assertIn("1", srt)
        self.assertIn("-->", srt)
        self.assertIn(",", srt)  # SRT uses comma
        self.assertIn("Hello", srt)

    def test_export_vtt_format(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, export_vtt
        subs = [SubtitleSegment(index=1, start=1.0, end=3.0, text="Hello")]
        vtt = export_vtt(subs)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("-->", vtt)
        self.assertIn(".", vtt)  # VTT uses dot

    def test_export_ass_format(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, export_ass
        subs = [SubtitleSegment(index=1, start=1.0, end=3.0, text="Hello")]
        ass = export_ass(subs)
        self.assertIn("[Script Info]", ass)
        self.assertIn("[V4+ Styles]", ass)
        self.assertIn("[Events]", ass)
        self.assertIn("Dialogue:", ass)

    def test_export_to_file(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, export_to_file
        subs = [SubtitleSegment(index=1, start=1.0, end=3.0, text="Test")]
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = f.name
        try:
            result = export_to_file(subs, path, fmt="srt")
            self.assertEqual(result, path)
            self.assertTrue(os.path.isfile(path))
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("Test", content)
        finally:
            os.unlink(path)

    def test_export_unsupported_format_raises(self):
        from opencut.core.subtitle_shot_aware import SubtitleSegment, export_to_file
        subs = [SubtitleSegment(index=1, start=1.0, end=3.0, text="Test")]
        with self.assertRaises(ValueError):
            export_to_file(subs, "/tmp/test.xyz", fmt="xyz")


class TestShotAwareParsing(unittest.TestCase):
    """Tests for SRT/VTT parsing."""

    def test_parse_srt(self):
        from opencut.core.subtitle_shot_aware import parse_srt
        content = "1\n00:00:01,000 --> 00:00:03,000\nHello\n\n2\n00:00:04,000 --> 00:00:06,000\nWorld\n"
        segments = parse_srt(content)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hello")
        self.assertAlmostEqual(segments[0].start, 1.0)

    def test_parse_vtt(self):
        from opencut.core.subtitle_shot_aware import parse_vtt
        content = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello\n\n00:00:04.000 --> 00:00:06.000\nWorld\n"
        segments = parse_vtt(content)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hello")

    def test_parse_srt_empty(self):
        from opencut.core.subtitle_shot_aware import parse_srt
        segments = parse_srt("")
        self.assertEqual(len(segments), 0)


class TestShotAwareDicts(unittest.TestCase):
    """Tests for process_shot_aware_dicts convenience function."""

    def test_process_from_dicts(self):
        from opencut.core.subtitle_shot_aware import process_shot_aware_dicts
        dicts = [
            {"start": 1.0, "end": 5.0, "text": "Hello world today"},
            {"start": 6.0, "end": 9.0, "text": "Second subtitle"},
        ]
        result = process_shot_aware_dicts(dicts, [3.0], profile="netflix")
        self.assertGreater(result.total_segments, 0)
        self.assertEqual(result.profile_used, "netflix")

    def test_custom_profile_with_settings(self):
        from opencut.core.subtitle_shot_aware import process_shot_aware_dicts
        dicts = [{"start": 1.0, "end": 3.0, "text": "Test"}]
        custom = {"max_chars_per_line": 30, "min_duration": 0.5}
        result = process_shot_aware_dicts(dicts, [], profile="custom", custom_settings=custom)
        self.assertEqual(result.profile_used, "custom")


# ============================================================
# Multi-Language Subtitles
# ============================================================
class TestMultiLangCreate(unittest.TestCase):
    """Tests for multi-language project creation."""

    def setUp(self):
        self._cleanup_ids = []

    def tearDown(self):
        from opencut.core.multilang_subtitle import delete_project
        for pid in self._cleanup_ids:
            try:
                delete_project(pid)
            except Exception:
                pass

    def test_create_project(self):
        from opencut.core.multilang_subtitle import create_project
        timing = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 4.0}]
        data = create_project("Test", timing, base_language="en")
        self._cleanup_ids.append(data.project_id)
        self.assertEqual(data.name, "Test")
        self.assertEqual(data.segment_count, 2)
        self.assertIn("en", data.languages)

    def test_create_with_base_texts(self):
        from opencut.core.multilang_subtitle import create_project
        timing = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 4.0}]
        texts = ["Hello", "World"]
        data = create_project("Test", timing, base_language="en", base_texts=texts)
        self._cleanup_ids.append(data.project_id)
        self.assertEqual(data.texts["en"], ["Hello", "World"])

    def test_create_empty_timing(self):
        from opencut.core.multilang_subtitle import create_project
        data = create_project("Empty")
        self._cleanup_ids.append(data.project_id)
        self.assertEqual(data.segment_count, 0)


class TestMultiLangCRUD(unittest.TestCase):
    """Tests for language add/remove/update operations."""

    def setUp(self):
        from opencut.core.multilang_subtitle import create_project
        timing = [{"start": 0.0, "end": 2.0}, {"start": 2.5, "end": 4.0}]
        self.project = create_project("CRUD Test", timing, base_language="en",
                                      base_texts=["Hello", "World"])
        self.pid = self.project.project_id

    def tearDown(self):
        from opencut.core.multilang_subtitle import delete_project
        try:
            delete_project(self.pid)
        except Exception:
            pass

    def test_add_language(self):
        from opencut.core.multilang_subtitle import add_language
        data = add_language(self.pid, "es")
        self.assertIn("es", data.languages)
        self.assertEqual(len(data.texts["es"]), 2)

    def test_add_duplicate_language_raises(self):
        from opencut.core.multilang_subtitle import add_language
        with self.assertRaises(ValueError):
            add_language(self.pid, "en")

    def test_remove_language(self):
        from opencut.core.multilang_subtitle import add_language, remove_language
        add_language(self.pid, "fr")
        data = remove_language(self.pid, "fr")
        self.assertNotIn("fr", data.languages)

    def test_remove_nonexistent_language_raises(self):
        from opencut.core.multilang_subtitle import remove_language
        with self.assertRaises(ValueError):
            remove_language(self.pid, "zz")

    def test_update_text(self):
        from opencut.core.multilang_subtitle import update_text
        data = update_text(self.pid, "en", 0, "Updated")
        self.assertEqual(data.texts["en"][0], "Updated")

    def test_update_text_invalid_index_raises(self):
        from opencut.core.multilang_subtitle import update_text
        with self.assertRaises(IndexError):
            update_text(self.pid, "en", 99, "Bad index")

    def test_update_text_invalid_language_raises(self):
        from opencut.core.multilang_subtitle import update_text
        with self.assertRaises(ValueError):
            update_text(self.pid, "zz", 0, "Bad lang")

    def test_bulk_update_texts(self):
        from opencut.core.multilang_subtitle import bulk_update_texts
        data = bulk_update_texts(self.pid, "en", ["New1", "New2"])
        self.assertEqual(data.texts["en"], ["New1", "New2"])

    def test_bulk_update_wrong_count_raises(self):
        from opencut.core.multilang_subtitle import bulk_update_texts
        with self.assertRaises(ValueError):
            bulk_update_texts(self.pid, "en", ["Only one"])


class TestMultiLangPersistence(unittest.TestCase):
    """Tests for project save/load/delete."""

    def test_save_and_load(self):
        from opencut.core.multilang_subtitle import create_project, load_project
        timing = [{"start": 0.0, "end": 2.0}]
        data = create_project("Persist", timing, base_language="en",
                              base_texts=["Test"])
        pid = data.project_id
        try:
            loaded = load_project(pid)
            self.assertEqual(loaded.name, "Persist")
            self.assertEqual(loaded.texts["en"][0], "Test")
        finally:
            from opencut.core.multilang_subtitle import delete_project
            delete_project(pid)

    def test_load_nonexistent_raises(self):
        from opencut.core.multilang_subtitle import load_project
        with self.assertRaises(FileNotFoundError):
            load_project("nonexistent_project_id_12345")

    def test_delete_project(self):
        from opencut.core.multilang_subtitle import create_project, delete_project
        data = create_project("DeleteMe")
        self.assertTrue(delete_project(data.project_id))
        self.assertFalse(delete_project(data.project_id))

    def test_list_projects(self):
        from opencut.core.multilang_subtitle import (
            create_project,
            delete_project,
            list_projects,
        )
        data = create_project("ListTest")
        try:
            projects = list_projects()
            ids = [p.project_id for p in projects]
            self.assertIn(data.project_id, ids)
        finally:
            delete_project(data.project_id)


class TestMultiLangImport(unittest.TestCase):
    """Tests for SRT/VTT import."""

    def setUp(self):
        from opencut.core.multilang_subtitle import create_project
        timing = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.5, "end": 4.5},
        ]
        self.project = create_project("Import Test", timing, base_language="en",
                                      base_texts=["Hello", "World"])
        self.pid = self.project.project_id

    def tearDown(self):
        from opencut.core.multilang_subtitle import delete_project
        delete_project(self.pid)

    def test_import_srt_aligned(self):
        from opencut.core.multilang_subtitle import bulk_import
        srt = "1\n00:00:00,000 --> 00:00:02,000\nHola\n\n2\n00:00:02,500 --> 00:00:04,500\nMundo\n"
        data = bulk_import(self.pid, "es", srt, fmt="srt")
        self.assertIn("es", data.languages)
        self.assertEqual(data.texts["es"][0], "Hola")

    def test_import_vtt_aligned(self):
        from opencut.core.multilang_subtitle import bulk_import
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nBonjour\n\n00:00:02.500 --> 00:00:04.500\nMonde\n"
        data = bulk_import(self.pid, "fr", vtt, fmt="vtt")
        self.assertIn("fr", data.languages)
        self.assertEqual(data.texts["fr"][0], "Bonjour")

    def test_import_unsupported_format_raises(self):
        from opencut.core.multilang_subtitle import bulk_import
        with self.assertRaises(ValueError):
            bulk_import(self.pid, "de", "content", fmt="xyz")

    def test_import_empty_content_raises(self):
        from opencut.core.multilang_subtitle import bulk_import
        with self.assertRaises(ValueError):
            bulk_import(self.pid, "de", "")


class TestMultiLangExport(unittest.TestCase):
    """Tests for SRT/VTT/ASS export."""

    def setUp(self):
        from opencut.core.multilang_subtitle import create_project
        timing = [{"start": 1.0, "end": 3.0}, {"start": 4.0, "end": 6.0}]
        self.project = create_project("Export Test", timing, base_language="en",
                                      base_texts=["Hello", "World"])
        self.pid = self.project.project_id

    def tearDown(self):
        from opencut.core.multilang_subtitle import delete_project
        delete_project(self.pid)

    def test_export_srt(self):
        from opencut.core.multilang_subtitle import export_srt, load_project
        data = load_project(self.pid)
        srt = export_srt(data, "en")
        self.assertIn("Hello", srt)
        self.assertIn("-->", srt)
        self.assertIn(",", srt)

    def test_export_vtt(self):
        from opencut.core.multilang_subtitle import export_vtt, load_project
        data = load_project(self.pid)
        vtt = export_vtt(data, "en")
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("Hello", vtt)

    def test_export_ass(self):
        from opencut.core.multilang_subtitle import export_ass, load_project
        data = load_project(self.pid)
        ass = export_ass(data, "en")
        self.assertIn("[Script Info]", ass)
        self.assertIn("Dialogue:", ass)

    def test_export_invalid_language_raises(self):
        from opencut.core.multilang_subtitle import export_srt, load_project
        data = load_project(self.pid)
        with self.assertRaises(ValueError):
            export_srt(data, "zz")

    def test_export_language_files(self):
        from opencut.core.multilang_subtitle import (
            add_language,
            export_language_files,
        )
        add_language(self.pid, "es")
        with tempfile.TemporaryDirectory() as td:
            files = export_language_files(self.pid, td, fmt="srt")
            self.assertIn("en", files)
            self.assertIn("es", files)
            for path in files.values():
                self.assertTrue(os.path.isfile(path))


class TestMultiLangTimingOps(unittest.TestCase):
    """Tests for timing shift and segment operations."""

    def setUp(self):
        from opencut.core.multilang_subtitle import create_project
        timing = [{"start": 1.0, "end": 3.0}, {"start": 4.0, "end": 6.0}]
        self.project = create_project("Timing Test", timing, base_language="en",
                                      base_texts=["A", "B"])
        self.pid = self.project.project_id

    def tearDown(self):
        from opencut.core.multilang_subtitle import delete_project
        delete_project(self.pid)

    def test_timing_shift_positive(self):
        from opencut.core.multilang_subtitle import timing_shift
        data = timing_shift(self.pid, 1.0)
        self.assertAlmostEqual(data.timing[0].start, 2.0)
        self.assertAlmostEqual(data.timing[0].end, 4.0)

    def test_timing_shift_negative_clamps(self):
        from opencut.core.multilang_subtitle import timing_shift
        data = timing_shift(self.pid, -5.0)
        self.assertGreaterEqual(data.timing[0].start, 0.0)

    def test_add_segment(self):
        from opencut.core.multilang_subtitle import add_segment
        data = add_segment(self.pid, 7.0, 9.0, {"en": "New"})
        self.assertEqual(data.segment_count, 3)
        self.assertEqual(data.texts["en"][2], "New")

    def test_remove_segment(self):
        from opencut.core.multilang_subtitle import remove_segment
        data = remove_segment(self.pid, 0)
        self.assertEqual(data.segment_count, 1)
        self.assertEqual(data.texts["en"][0], "B")

    def test_remove_invalid_segment_raises(self):
        from opencut.core.multilang_subtitle import remove_segment
        with self.assertRaises(IndexError):
            remove_segment(self.pid, 99)


# ============================================================
# Broadcast Caption Export
# ============================================================
class TestBroadcastFormats(unittest.TestCase):
    """Tests for broadcast format listing and validation."""

    def test_list_formats(self):
        from opencut.core.broadcast_caption import list_formats
        formats = list_formats()
        names = [f["name"] for f in formats]
        self.assertIn("cea608", names)
        self.assertIn("cea708", names)
        self.assertIn("ebu_tt", names)
        self.assertIn("ttml", names)
        self.assertIn("imsc1", names)
        self.assertIn("webvtt_pos", names)

    def test_list_formats_have_description(self):
        from opencut.core.broadcast_caption import list_formats
        for f in list_formats():
            self.assertTrue(f["description"])

    def test_segments_from_dicts(self):
        from opencut.core.broadcast_caption import segments_from_dicts
        dicts = [{"start": 1.0, "end": 3.0, "text": "Hello"}]
        segments = segments_from_dicts(dicts)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "Hello")
        self.assertEqual(segments[0].index, 1)


class TestBroadcastCEA608(unittest.TestCase):
    """Tests for CEA-608 export."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [
            CaptionSegment(index=1, start=1.0, end=3.0, text="Hello world"),
            CaptionSegment(index=2, start=4.0, end=6.0, text="Second line"),
        ]

    def test_export_cea608_creates_file(self):
        from opencut.core.broadcast_caption import export_cea608
        with tempfile.NamedTemporaryFile(suffix=".scc", delete=False) as f:
            path = f.name
        try:
            result = export_cea608(self._segments(), path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result.format, "cea608")
            self.assertEqual(result.segments_exported, 2)
        finally:
            os.unlink(path)

    def test_cea608_scc_header(self):
        from opencut.core.broadcast_caption import export_cea608
        with tempfile.NamedTemporaryFile(suffix=".scc", delete=False) as f:
            path = f.name
        try:
            export_cea608(self._segments(), path)
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("Scenarist_SCC V1.0", content)
        finally:
            os.unlink(path)

    def test_cea608_char_encoding(self):
        from opencut.core.broadcast_caption import _encode_cea608_char
        self.assertEqual(_encode_cea608_char("A"), ord("A"))
        self.assertEqual(_encode_cea608_char(" "), 0x20)


class TestBroadcastCEA708(unittest.TestCase):
    """Tests for CEA-708 export."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [CaptionSegment(index=1, start=1.0, end=3.0, text="Test")]

    def test_export_cea708_creates_file(self):
        from opencut.core.broadcast_caption import export_cea708
        with tempfile.NamedTemporaryFile(suffix=".mcc", delete=False) as f:
            path = f.name
        try:
            result = export_cea708(self._segments(), path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result.format, "cea708")
        finally:
            os.unlink(path)

    def test_cea708_invalid_channel_warning(self):
        from opencut.core.broadcast_caption import export_cea708
        with tempfile.NamedTemporaryFile(suffix=".mcc", delete=False) as f:
            path = f.name
        try:
            result = export_cea708(self._segments(), path, service_channel=99)
            self.assertTrue(any("out of range" in w for w in result.warnings))
        finally:
            os.unlink(path)


class TestBroadcastEBUTT(unittest.TestCase):
    """Tests for EBU-TT export."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [CaptionSegment(index=1, start=1.0, end=3.0, text="Test caption")]

    def test_export_ebu_tt_creates_xml(self):
        from opencut.core.broadcast_caption import export_ebu_tt
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            path = f.name
        try:
            result = export_ebu_tt(self._segments(), path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result.format, "ebu_tt")
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("ebuttm:documentEbuttVersion", content)
        finally:
            os.unlink(path)


class TestBroadcastTTML(unittest.TestCase):
    """Tests for TTML export."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [CaptionSegment(index=1, start=1.0, end=3.0, text="TTML test")]

    def test_export_ttml_creates_xml(self):
        from opencut.core.broadcast_caption import export_ttml
        with tempfile.NamedTemporaryFile(suffix=".ttml", delete=False) as f:
            path = f.name
        try:
            result = export_ttml(self._segments(), path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result.format, "ttml")
        finally:
            os.unlink(path)

    def test_ttml_contains_timing(self):
        from opencut.core.broadcast_caption import export_ttml
        with tempfile.NamedTemporaryFile(suffix=".ttml", delete=False) as f:
            path = f.name
        try:
            export_ttml(self._segments(), path)
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("begin=", content)
            self.assertIn("end=", content)
        finally:
            os.unlink(path)


class TestBroadcastIMSC1(unittest.TestCase):
    """Tests for IMSC1 export."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [CaptionSegment(index=1, start=1.0, end=3.0, text="IMSC test")]

    def test_export_imsc1_creates_xml(self):
        from opencut.core.broadcast_caption import export_imsc1
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            path = f.name
        try:
            result = export_imsc1(self._segments(), path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result.format, "imsc1")
        finally:
            os.unlink(path)

    def test_imsc1_has_cell_resolution(self):
        from opencut.core.broadcast_caption import export_imsc1
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            path = f.name
        try:
            export_imsc1(self._segments(), path)
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("cellResolution", content)
        finally:
            os.unlink(path)


class TestBroadcastWebVTT(unittest.TestCase):
    """Tests for WebVTT with positioning export."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [
            CaptionSegment(index=1, start=1.0, end=3.0, text="VTT test", row=15),
        ]

    def test_export_webvtt_positioned(self):
        from opencut.core.broadcast_caption import export_webvtt_positioned
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as f:
            path = f.name
        try:
            result = export_webvtt_positioned(self._segments(), path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(result.format, "webvtt_pos")
        finally:
            os.unlink(path)

    def test_webvtt_has_positioning(self):
        from opencut.core.broadcast_caption import export_webvtt_positioned
        with tempfile.NamedTemporaryFile(suffix=".vtt", delete=False) as f:
            path = f.name
        try:
            export_webvtt_positioned(self._segments(), path)
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("WEBVTT", content)
            self.assertIn("line:", content)
            self.assertIn("position:", content)
        finally:
            os.unlink(path)


class TestBroadcastUnified(unittest.TestCase):
    """Tests for the unified export_broadcast function."""

    def _segments(self):
        from opencut.core.broadcast_caption import CaptionSegment
        return [CaptionSegment(index=1, start=1.0, end=3.0, text="Unified")]

    def test_export_broadcast_unknown_format_raises(self):
        from opencut.core.broadcast_caption import export_broadcast
        with self.assertRaises(ValueError):
            export_broadcast(self._segments(), "/tmp/out.txt", fmt="invalid")

    def test_export_broadcast_all_formats(self):
        from opencut.core.broadcast_caption import SUPPORTED_FORMATS, export_broadcast
        ext_map = {"cea608": ".scc", "cea708": ".mcc", "ebu_tt": ".xml",
                    "ttml": ".ttml", "imsc1": ".xml", "webvtt_pos": ".vtt"}
        for fmt_name in SUPPORTED_FORMATS:
            ext = ext_map.get(fmt_name, ".txt")
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                path = f.name
            try:
                result = export_broadcast(self._segments(), path, fmt=fmt_name)
                self.assertEqual(result.format, fmt_name)
                self.assertTrue(os.path.isfile(path))
            finally:
                os.unlink(path)


class TestBroadcastValidation(unittest.TestCase):
    """Tests for segment validation."""

    def test_validation_catches_long_lines(self):
        from opencut.core.broadcast_caption import CaptionSegment, _validate_segments
        seg = CaptionSegment(index=1, start=1.0, end=3.0, text="A" * 50)
        errors = _validate_segments([seg], {"max_chars_per_line": 32, "max_lines": 4})
        self.assertTrue(any("chars" in e for e in errors))

    def test_validation_catches_too_many_lines(self):
        from opencut.core.broadcast_caption import CaptionSegment, _validate_segments
        seg = CaptionSegment(index=1, start=1.0, end=3.0, text="A\nB\nC\nD\nE")
        errors = _validate_segments([seg], {"max_chars_per_line": 42, "max_lines": 2})
        self.assertTrue(any("lines" in e for e in errors))

    def test_validation_catches_invalid_timing(self):
        from opencut.core.broadcast_caption import CaptionSegment, _validate_segments
        seg = CaptionSegment(index=1, start=5.0, end=3.0, text="Bad")
        errors = _validate_segments([seg], {"max_chars_per_line": 42, "max_lines": 4})
        self.assertTrue(any("timing" in e.lower() for e in errors))


# ============================================================
# SDH Formatting
# ============================================================
class TestSDHSpeakerLabels(unittest.TestCase):
    """Tests for speaker identification."""

    def test_speaker_from_diarization(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hello there"}]
        diar = [{"start": 0.0, "end": 3.0, "speaker": "John"}]
        result = format_sdh(subs, diarization=diar)
        self.assertEqual(result.speaker_labels_added, 1)
        self.assertIn("JOHN", result.formatted_subtitles[0].formatted_text)

    def test_speaker_uppercase_default(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hi", "speaker": "alice"}]
        result = format_sdh(subs, config=SDHConfig(uppercase_speakers=True))
        self.assertIn("ALICE:", result.formatted_subtitles[0].formatted_text)

    def test_speaker_lowercase_config(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hi", "speaker": "Alice"}]
        result = format_sdh(subs, config=SDHConfig(uppercase_speakers=False))
        self.assertIn("Alice:", result.formatted_subtitles[0].formatted_text)

    def test_no_diarization_no_labels(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
        result = format_sdh(subs)
        self.assertEqual(result.speaker_labels_added, 0)

    def test_speaker_separator(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hi", "speaker": "Bob"}]
        result = format_sdh(subs, config=SDHConfig(speaker_separator=" -"))
        self.assertIn("BOB -", result.formatted_subtitles[0].formatted_text)


class TestSDHSoundEvents(unittest.TestCase):
    """Tests for sound event detection and insertion."""

    def test_sound_event_inserted(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "What was that?"}]
        events = [{"start": 1.0, "end": 1.5, "type": "door_slam"}]
        result = format_sdh(subs, audio_events=events)
        self.assertGreater(result.sound_events_added, 0)
        self.assertIn("[door slams]", result.formatted_subtitles[0].formatted_text)

    def test_custom_sound_event_description(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Look"}]
        events = [{"start": 0.5, "end": 1.0, "type": "custom", "description": "glass breaking"}]
        result = format_sdh(subs, audio_events=events)
        self.assertIn("[glass breaking]", result.formatted_subtitles[0].formatted_text)

    def test_no_events_no_insertion(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Quiet"}]
        result = format_sdh(subs, audio_events=[])
        self.assertEqual(result.sound_events_added, 0)

    def test_sound_events_disabled(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Test"}]
        events = [{"start": 1.0, "end": 1.5, "type": "explosion"}]
        config = SDHConfig(include_sound_events=False)
        result = format_sdh(subs, audio_events=events, config=config)
        self.assertEqual(result.sound_events_added, 0)

    def test_event_position_after(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Before"}]
        events = [{"start": 1.0, "end": 1.5, "type": "gunshot"}]
        config = SDHConfig(sound_event_position="after")
        result = format_sdh(subs, audio_events=events, config=config)
        text = result.formatted_subtitles[0].formatted_text
        self.assertTrue(text.index("Before") < text.index("[gunshot]"))


class TestSDHMusicDetection(unittest.TestCase):
    """Tests for music detection and notation."""

    def test_music_detected_from_stems(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 5.0, "text": "La la la"}]
        stems = [{"start": 0.0, "end": 5.0, "has_vocals": False, "has_music": True}]
        result = format_sdh(subs, stem_metadata=stems)
        self.assertGreater(result.music_segments_marked, 0)

    def test_music_notation_symbol(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 5.0, "text": "Song lyrics"}]
        stems = [{"start": 0.0, "end": 5.0, "has_vocals": False, "has_music": True}]
        config = SDHConfig(music_symbol="\u266a")
        result = format_sdh(subs, stem_metadata=stems, config=config)
        text = result.formatted_subtitles[0].formatted_text
        self.assertIn("\u266a", text)

    def test_music_notation_disabled(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 5.0, "text": "Song"}]
        stems = [{"start": 0.0, "end": 5.0, "has_vocals": False, "has_music": True}]
        config = SDHConfig(include_music_notation=False)
        result = format_sdh(subs, stem_metadata=stems, config=config)
        text = result.formatted_subtitles[0].formatted_text
        self.assertNotIn("\u266a", text)

    def test_vocals_present_not_marked_music(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 5.0, "text": "Talking"}]
        stems = [{"start": 0.0, "end": 5.0, "has_vocals": True, "has_music": True}]
        result = format_sdh(subs, stem_metadata=stems)
        text = result.formatted_subtitles[0].formatted_text
        self.assertNotIn("\u266a", text)


class TestSDHToneMarkers(unittest.TestCase):
    """Tests for emotional tone markers."""

    def test_tone_marker_applied(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Get out"}]
        tones = [{"start": 0.0, "end": 3.0, "tone": "shouting"}]
        result = format_sdh(subs, tone_annotations=tones)
        self.assertEqual(result.tone_markers_added, 1)
        self.assertIn("[shouting]", result.formatted_subtitles[0].formatted_text)

    def test_custom_tone_marker(self):
        from opencut.core.sdh_formatter import format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Please"}]
        tones = [{"start": 0.0, "end": 3.0, "tone": "begging"}]
        result = format_sdh(subs, tone_annotations=tones)
        self.assertIn("[begging]", result.formatted_subtitles[0].formatted_text)

    def test_tone_markers_disabled(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Test"}]
        tones = [{"start": 0.0, "end": 3.0, "tone": "whispering"}]
        config = SDHConfig(include_tone_markers=False)
        result = format_sdh(subs, tone_annotations=tones, config=config)
        self.assertEqual(result.tone_markers_added, 0)


class TestSDHBracketStyle(unittest.TestCase):
    """Tests for bracket style normalization."""

    def test_round_brackets(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
        events = [{"start": 1.0, "end": 1.5, "type": "phone_ring"}]
        config = SDHConfig(bracket_style="round")
        result = format_sdh(subs, audio_events=events, config=config)
        text = result.formatted_subtitles[0].formatted_text
        self.assertIn("(phone rings)", text)
        self.assertNotIn("[phone rings]", text)

    def test_angle_brackets(self):
        from opencut.core.sdh_formatter import SDHConfig, format_sdh
        subs = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
        events = [{"start": 1.0, "end": 1.5, "type": "door_knock"}]
        config = SDHConfig(bracket_style="angle")
        result = format_sdh(subs, audio_events=events, config=config)
        text = result.formatted_subtitles[0].formatted_text
        self.assertIn("<knocking on door>", text)


class TestSDHExport(unittest.TestCase):
    """Tests for SDH export to SRT/VTT."""

    def test_export_sdh_srt(self):
        from opencut.core.sdh_formatter import export_sdh_srt, format_sdh
        subs = [{"start": 1.0, "end": 3.0, "text": "Hello"}]
        result = format_sdh(subs)
        srt = export_sdh_srt(result)
        self.assertIn("-->", srt)
        self.assertIn(",", srt)

    def test_export_sdh_vtt(self):
        from opencut.core.sdh_formatter import export_sdh_vtt, format_sdh
        subs = [{"start": 1.0, "end": 3.0, "text": "Hello"}]
        result = format_sdh(subs)
        vtt = export_sdh_vtt(result)
        self.assertTrue(vtt.startswith("WEBVTT"))

    def test_empty_subtitles(self):
        from opencut.core.sdh_formatter import format_sdh
        result = format_sdh([])
        self.assertEqual(result.total_segments, 0)


class TestSDHClassification(unittest.TestCase):
    """Tests for audio segment classification."""

    def test_silence_classification(self):
        from opencut.core.sdh_formatter import _classify_audio_segment
        result = _classify_audio_segment(0.005, 0, 0, 0)
        self.assertEqual(result, "silence")

    def test_speech_classification(self):
        from opencut.core.sdh_formatter import _classify_audio_segment
        result = _classify_audio_segment(0.3, 2000, 1500, 0.15)
        self.assertEqual(result, "speech")

    def test_music_classification(self):
        from opencut.core.sdh_formatter import _classify_audio_segment
        result = _classify_audio_segment(0.2, 1000, 4000, 0.05)
        self.assertEqual(result, "music")


class TestSDHEventLists(unittest.TestCase):
    """Tests for sound event and tone marker listing."""

    def test_list_sound_events(self):
        from opencut.core.sdh_formatter import list_sound_events
        events = list_sound_events()
        self.assertGreater(len(events), 10)
        types = [e["type"] for e in events]
        self.assertIn("door_slam", types)

    def test_list_tone_markers(self):
        from opencut.core.sdh_formatter import list_tone_markers
        markers = list_tone_markers()
        self.assertGreater(len(markers), 5)
        tones = [m["tone"] for m in markers]
        self.assertIn("whispering", tones)


# ============================================================
# Subtitle Positioning
# ============================================================
class TestPositionZones(unittest.TestCase):
    """Tests for zone definitions and coordinate conversion."""

    def test_all_zones_defined(self):
        from opencut.core.subtitle_position import ZONES
        expected = {"bottom_center", "top_center", "bottom_left",
                    "bottom_right", "top_left", "top_right"}
        self.assertEqual(set(ZONES.keys()), expected)

    def test_zone_to_pixels_bottom_center(self):
        from opencut.core.subtitle_position import zone_to_pixels
        x, y = zone_to_pixels("bottom_center", 1920, 1080)
        self.assertEqual(x, 960)
        self.assertGreater(y, 900)

    def test_zone_to_pixels_top_center(self):
        from opencut.core.subtitle_position import zone_to_pixels
        x, y = zone_to_pixels("top_center", 1920, 1080)
        self.assertEqual(x, 960)
        self.assertLess(y, 200)

    def test_zone_to_pixels_unknown_defaults(self):
        from opencut.core.subtitle_position import zone_to_pixels
        x, y = zone_to_pixels("unknown_zone", 1920, 1080)
        # Should use bottom_center defaults
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)


class TestPositionFrameAnalysis(unittest.TestCase):
    """Tests for frame obstruction analysis."""

    def test_empty_frame_no_obstructions(self):
        from opencut.core.subtitle_position import analyze_frame
        result = analyze_frame(None, 1920, 1080)
        self.assertEqual(len(result.obstructions), 0)
        self.assertEqual(result.best_zone, "bottom_center")

    def test_face_in_bottom_third(self):
        from opencut.core.subtitle_position import analyze_frame
        frame_data = {
            "faces": [{"x": 800, "y": 800, "w": 200, "h": 200}],
        }
        result = analyze_frame(frame_data, 1920, 1080)
        self.assertGreater(len(result.obstructions), 0)
        self.assertEqual(result.obstructions[0].obstruction_type, "face")

    def test_text_region_detection(self):
        from opencut.core.subtitle_position import analyze_frame
        frame_data = {
            "text_regions": [{"x": 100, "y": 900, "w": 300, "h": 50}],
        }
        result = analyze_frame(frame_data, 1920, 1080)
        self.assertGreater(len(result.obstructions), 0)

    def test_bright_region_high_intensity(self):
        from opencut.core.subtitle_position import analyze_frame
        frame_data = {
            "bright_regions": [
                {"x": 500, "y": 900, "w": 400, "h": 100, "intensity": 0.95},
            ],
        }
        result = analyze_frame(frame_data, 1920, 1080)
        types = [o.obstruction_type for o in result.obstructions]
        self.assertIn("bright", types)

    def test_edge_density_busy(self):
        from opencut.core.subtitle_position import analyze_frame
        frame_data = {"edge_density": 0.8}
        result = analyze_frame(frame_data, 1920, 1080)
        types = [o.obstruction_type for o in result.obstructions]
        self.assertIn("busy", types)

    def test_obstruction_triggers_repositioning(self):
        from opencut.core.subtitle_position import analyze_frame
        frame_data = {
            "faces": [{"x": 800, "y": 800, "w": 300, "h": 300}],
        }
        result = analyze_frame(frame_data, 1920, 1080)
        self.assertNotEqual(result.best_zone, "bottom_center")


class TestPositionZoneFallback(unittest.TestCase):
    """Tests for zone fallback logic."""

    def test_bottom_center_fallback_to_top(self):
        from opencut.core.subtitle_position import _find_best_zone
        result = _find_best_zone(["bottom_center"])
        self.assertEqual(result, "top_center")

    def test_all_obstructed_returns_top_center(self):
        from opencut.core.subtitle_position import _find_best_zone
        all_zones = ["bottom_center", "top_center", "bottom_left",
                     "bottom_right", "top_left", "top_right"]
        result = _find_best_zone(all_zones)
        self.assertEqual(result, "top_center")

    def test_no_obstructions_returns_default(self):
        from opencut.core.subtitle_position import _find_best_zone
        result = _find_best_zone([])
        self.assertEqual(result, "bottom_center")

    def test_partial_obstruction_skips_blocked(self):
        from opencut.core.subtitle_position import _find_best_zone
        result = _find_best_zone(["bottom_center", "top_center"])
        self.assertEqual(result, "bottom_left")


class TestPositionBatch(unittest.TestCase):
    """Tests for batch subtitle positioning."""

    def test_position_subtitles_no_analyses(self):
        from opencut.core.subtitle_position import position_subtitles
        subs = [{"start": 1.0, "end": 3.0, "text": "Hello"}]
        result = position_subtitles(subs)
        self.assertEqual(result.total_segments, 1)
        self.assertEqual(result.repositioned_count, 0)

    def test_position_subtitles_with_obstruction(self):
        from opencut.core.subtitle_position import position_subtitles
        subs = [{"start": 1.0, "end": 3.0, "text": "Hello"}]
        analyses = {
            1.0: {"faces": [{"x": 800, "y": 800, "w": 300, "h": 300}]},
        }
        result = position_subtitles(subs, frame_analyses=analyses)
        self.assertEqual(result.repositioned_count, 1)
        self.assertNotEqual(
            result.positioned_subtitles[0].zone, "bottom_center",
        )

    def test_position_empty_subtitles(self):
        from opencut.core.subtitle_position import position_subtitles
        result = position_subtitles([])
        self.assertEqual(result.total_segments, 0)

    def test_position_multiple_subtitles(self):
        from opencut.core.subtitle_position import position_subtitles
        subs = [
            {"start": 1.0, "end": 3.0, "text": "First"},
            {"start": 4.0, "end": 6.0, "text": "Second"},
            {"start": 7.0, "end": 9.0, "text": "Third"},
        ]
        result = position_subtitles(subs)
        self.assertEqual(result.total_segments, 3)
        self.assertEqual(result.frames_analyzed, 3)

    def test_obstruction_types_counted(self):
        from opencut.core.subtitle_position import position_subtitles
        subs = [
            {"start": 1.0, "end": 3.0, "text": "A"},
            {"start": 4.0, "end": 6.0, "text": "B"},
        ]
        analyses = {
            1.0: {"faces": [{"x": 800, "y": 800, "w": 300, "h": 300}]},
            4.0: {"text_regions": [{"x": 100, "y": 900, "w": 300, "h": 50}]},
        }
        result = position_subtitles(subs, frame_analyses=analyses)
        self.assertIn("face", result.obstruction_types)


class TestPositionPreview(unittest.TestCase):
    """Tests for single frame position preview."""

    def test_preview_no_obstruction(self):
        from opencut.core.subtitle_position import preview_position
        result = preview_position("Hello", None)
        self.assertEqual(result["zone"], "bottom_center")
        self.assertIn("ass_override", result)

    def test_preview_with_obstruction(self):
        from opencut.core.subtitle_position import preview_position
        frame_data = {
            "faces": [{"x": 800, "y": 800, "w": 300, "h": 300}],
        }
        result = preview_position("Hello", frame_data)
        self.assertNotEqual(result["zone"], "bottom_center")
        self.assertIn("\\pos(", result["ass_override"])

    def test_preview_returns_coordinates(self):
        from opencut.core.subtitle_position import preview_position
        result = preview_position("Test", None, 1920, 1080)
        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIsInstance(result["x"], int)
        self.assertIsInstance(result["y"], int)


class TestPositionASSExport(unittest.TestCase):
    """Tests for positioned ASS export."""

    def test_export_positioned_ass(self):
        from opencut.core.subtitle_position import (
            PositionResult,
            PositionedSubtitle,
            export_positioned_ass,
        )
        subs = [PositionedSubtitle(
            index=1, start=1.0, end=3.0, text="Hello",
            zone="top_center", x=960, y=108,
        )]
        result = PositionResult(positioned_subtitles=subs)
        ass = export_positioned_ass(result)
        self.assertIn("[Script Info]", ass)
        self.assertIn("\\pos(960,108)", ass)
        self.assertIn("Hello", ass)

    def test_export_to_file(self):
        from opencut.core.subtitle_position import (
            PositionResult,
            PositionedSubtitle,
            export_to_file,
        )
        subs = [PositionedSubtitle(
            index=1, start=1.0, end=3.0, text="Test",
            zone="bottom_center", x=960, y=972,
        )]
        result = PositionResult(positioned_subtitles=subs)
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            path = f.name
        try:
            out = export_to_file(result, path)
            self.assertEqual(out, path)
            self.assertTrue(os.path.isfile(path))
        finally:
            os.unlink(path)

    def test_ass_contains_pos_tags(self):
        from opencut.core.subtitle_position import position_subtitles
        subs = [{"start": 1.0, "end": 3.0, "text": "With position"}]
        analyses = {
            1.0: {"faces": [{"x": 800, "y": 800, "w": 300, "h": 300}]},
        }
        result = position_subtitles(subs, frame_analyses=analyses)
        from opencut.core.subtitle_position import export_positioned_ass
        ass = export_positioned_ass(result)
        self.assertIn("\\pos(", ass)


if __name__ == "__main__":
    unittest.main()
