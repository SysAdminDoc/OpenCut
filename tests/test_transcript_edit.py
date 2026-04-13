"""
Tests for Transcript-Based Editing (1.1) and AI Rough Cut Assembly (21.3).

80+ tests covering:
- TranscriptMap building from multiple formats
- Word-level editing (delete, rearrange)
- Bidirectional text<->time mapping
- Cut segment computation
- Export (EDL, OTIO, JSON)
- RoughCutBrief parsing
- Footage analysis
- Plan generation (with mocked LLM)
- Plan execution (with mocked FFmpeg)
- Full pipeline (with all mocks)
- Route smoke tests
- Edge cases and error handling
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

# ───────────────────────────────────────────────────────────────────
# Helpers for building test data
# ───────────────────────────────────────────────────────────────────

def _whisperx_transcript():
    """Sample WhisperX-format transcript."""
    return {
        "segments": [
            {
                "text": "Hello world this is a test",
                "start": 0.0,
                "end": 3.0,
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.95},
                    {"word": "world", "start": 0.5, "end": 1.0, "score": 0.92},
                    {"word": "this", "start": 1.0, "end": 1.3, "score": 0.90},
                    {"word": "is", "start": 1.3, "end": 1.5, "score": 0.88},
                    {"word": "a", "start": 1.5, "end": 1.6, "score": 0.99},
                    {"word": "test", "start": 1.6, "end": 2.0, "score": 0.94},
                ],
            },
            {
                "text": "Second paragraph here",
                "start": 3.5,
                "end": 5.5,
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Second", "start": 3.5, "end": 4.0, "score": 0.91},
                    {"word": "paragraph", "start": 4.0, "end": 4.5, "score": 0.87},
                    {"word": "here", "start": 4.5, "end": 5.0, "score": 0.93},
                ],
            },
            {
                "text": "Final section of speech",
                "start": 6.0,
                "end": 8.0,
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Final", "start": 6.0, "end": 6.5, "score": 0.90},
                    {"word": "section", "start": 6.5, "end": 7.0, "score": 0.88},
                    {"word": "of", "start": 7.0, "end": 7.2, "score": 0.95},
                    {"word": "speech", "start": 7.2, "end": 8.0, "score": 0.92},
                ],
            },
        ],
        "language": "en",
    }


def _simple_segments():
    """Simple segment list format."""
    return [
        {"text": "Hello world", "start": 0.0, "end": 2.0,
         "words": [
             {"text": "Hello", "start": 0.0, "end": 1.0},
             {"text": "world", "start": 1.0, "end": 2.0},
         ]},
        {"text": "Goodbye moon", "start": 3.0, "end": 5.0,
         "words": [
             {"text": "Goodbye", "start": 3.0, "end": 4.0},
             {"text": "moon", "start": 4.0, "end": 5.0},
         ]},
    ]


def _words_only_transcript():
    """Words-only format."""
    return {
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "there", "start": 0.5, "end": 1.0},
            {"word": "friend", "start": 3.0, "end": 3.5},
        ],
    }


def _no_word_timestamps():
    """Segments without word-level timestamps."""
    return {
        "segments": [
            {"text": "Hello world foo bar", "start": 0.0, "end": 2.0},
            {"text": "Second segment", "start": 3.0, "end": 4.0},
        ],
    }


# ===================================================================
# Feature 1: Transcript-Based Editing Tests
# ===================================================================

class TestWordMapping(unittest.TestCase):
    """Tests for WordMapping dataclass."""

    def test_duration_positive(self):
        from opencut.core.transcript_edit import WordMapping
        w = WordMapping(index=0, text="hello", start=1.0, end=2.5)
        self.assertAlmostEqual(w.duration, 1.5)

    def test_duration_zero(self):
        from opencut.core.transcript_edit import WordMapping
        w = WordMapping(index=0, text="hi", start=1.0, end=1.0)
        self.assertAlmostEqual(w.duration, 0.0)

    def test_duration_negative_clamped(self):
        from opencut.core.transcript_edit import WordMapping
        w = WordMapping(index=0, text="x", start=2.0, end=1.0)
        self.assertAlmostEqual(w.duration, 0.0)

    def test_defaults(self):
        from opencut.core.transcript_edit import WordMapping
        w = WordMapping(index=0, text="a", start=0, end=1)
        self.assertEqual(w.confidence, 1.0)
        self.assertEqual(w.speaker, "")
        self.assertFalse(w.is_deleted)
        self.assertEqual(w.paragraph_index, 0)


class TestParagraphMapping(unittest.TestCase):

    def test_word_count(self):
        from opencut.core.transcript_edit import ParagraphMapping
        p = ParagraphMapping(index=0, text="hello world", start=0, end=2,
                             word_start_index=0, word_end_index=5)
        self.assertEqual(p.word_count, 5)

    def test_duration(self):
        from opencut.core.transcript_edit import ParagraphMapping
        p = ParagraphMapping(index=0, text="x", start=1.5, end=4.0)
        self.assertAlmostEqual(p.duration, 2.5)


class TestTimeRange(unittest.TestCase):

    def test_duration(self):
        from opencut.core.transcript_edit import TimeRange
        t = TimeRange(start=0.5, end=3.0)
        self.assertAlmostEqual(t.duration, 2.5)

    def test_duration_zero(self):
        from opencut.core.transcript_edit import TimeRange
        t = TimeRange(start=1.0, end=1.0)
        self.assertAlmostEqual(t.duration, 0.0)


class TestTextEdit(unittest.TestCase):

    def test_valid_types(self):
        from opencut.core.transcript_edit import TextEdit
        for t in ("delete", "rearrange", "keep"):
            e = TextEdit(edit_type=t)
            self.assertEqual(e.edit_type, t)

    def test_invalid_type(self):
        from opencut.core.transcript_edit import TextEdit
        with self.assertRaises(ValueError):
            TextEdit(edit_type="invalid")


class TestCutSegment(unittest.TestCase):

    def test_duration(self):
        from opencut.core.transcript_edit import CutSegment
        s = CutSegment(start=1.0, end=5.0)
        self.assertAlmostEqual(s.duration, 4.0)


class TestBuildTranscriptMap(unittest.TestCase):
    """Tests for build_transcript_map()."""

    def test_whisperx_format(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        self.assertEqual(len(tmap.words), 13)
        self.assertEqual(len(tmap.paragraphs), 3)
        self.assertEqual(tmap.language, "en")
        self.assertGreater(tmap.total_duration, 0)

    def test_simple_segment_list(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_simple_segments())
        self.assertEqual(len(tmap.words), 4)
        self.assertEqual(len(tmap.paragraphs), 2)

    def test_words_only_format(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_words_only_transcript())
        # Should group into segments based on gaps
        self.assertGreater(len(tmap.words), 0)
        self.assertGreater(len(tmap.paragraphs), 0)

    def test_no_word_timestamps_interpolated(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_no_word_timestamps())
        # Words should be interpolated from segment text
        self.assertGreater(len(tmap.words), 0)
        # Check words have valid timestamps
        for w in tmap.words:
            self.assertGreaterEqual(w.start, 0)
            self.assertGreater(w.end, w.start)
            self.assertEqual(w.confidence, 0.5)

    def test_empty_transcript_raises(self):
        from opencut.core.transcript_edit import build_transcript_map
        with self.assertRaises(ValueError):
            build_transcript_map({"segments": []})

    def test_empty_list_raises(self):
        from opencut.core.transcript_edit import build_transcript_map
        with self.assertRaises(ValueError):
            build_transcript_map([])

    def test_word_indices_sequential(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        for i, w in enumerate(tmap.words):
            self.assertEqual(w.index, i)

    def test_paragraph_word_ranges(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        # First paragraph has 6 words (indices 0-5)
        self.assertEqual(tmap.paragraphs[0].word_start_index, 0)
        self.assertEqual(tmap.paragraphs[0].word_end_index, 6)
        # Second paragraph has 3 words (indices 6-8)
        self.assertEqual(tmap.paragraphs[1].word_start_index, 6)
        self.assertEqual(tmap.paragraphs[1].word_end_index, 9)

    def test_source_file_stored(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_simple_segments(), source_file="/test/video.mp4")
        self.assertEqual(tmap.source_file, "/test/video.mp4")

    def test_speaker_assigned_to_words(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        self.assertEqual(tmap.words[0].speaker, "SPEAKER_00")
        self.assertEqual(tmap.words[6].speaker, "SPEAKER_01")

    def test_to_dict_roundtrip(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        d = tmap.to_dict()
        self.assertEqual(len(d["words"]), 13)
        self.assertEqual(len(d["paragraphs"]), 3)
        self.assertIn("total_duration", d)
        self.assertIn("word_count", d)

    def test_progress_callback_called(self):
        from opencut.core.transcript_edit import build_transcript_map
        progress_calls = []
        def cb(pct, msg=""):
            progress_calls.append((pct, msg))
        build_transcript_map(_whisperx_transcript(), on_progress=cb)
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 100)

    def test_single_segment_dict(self):
        from opencut.core.transcript_edit import build_transcript_map
        data = {"text": "solo segment", "start": 0.0, "end": 2.0,
                "words": [{"word": "solo", "start": 0.0, "end": 1.0},
                          {"word": "segment", "start": 1.0, "end": 2.0}]}
        tmap = build_transcript_map(data)
        self.assertEqual(len(tmap.words), 2)
        self.assertEqual(len(tmap.paragraphs), 1)


class TestTextSelectionToTimerange(unittest.TestCase):

    def setUp(self):
        from opencut.core.transcript_edit import build_transcript_map
        self.tmap = build_transcript_map(_whisperx_transcript())

    def test_single_word(self):
        from opencut.core.transcript_edit import text_selection_to_timerange
        tr = text_selection_to_timerange(self.tmap, 0, 0)
        self.assertAlmostEqual(tr.start, 0.0)
        self.assertAlmostEqual(tr.end, 0.5)

    def test_word_range(self):
        from opencut.core.transcript_edit import text_selection_to_timerange
        tr = text_selection_to_timerange(self.tmap, 0, 5)
        self.assertAlmostEqual(tr.start, 0.0)
        self.assertAlmostEqual(tr.end, 2.0)

    def test_cross_paragraph(self):
        from opencut.core.transcript_edit import text_selection_to_timerange
        tr = text_selection_to_timerange(self.tmap, 4, 7)
        self.assertAlmostEqual(tr.start, 1.5)
        self.assertAlmostEqual(tr.end, 4.5)

    def test_invalid_range(self):
        from opencut.core.transcript_edit import text_selection_to_timerange
        with self.assertRaises(ValueError):
            text_selection_to_timerange(self.tmap, 5, 2)

    def test_clamp_to_bounds(self):
        from opencut.core.transcript_edit import text_selection_to_timerange
        tr = text_selection_to_timerange(self.tmap, -5, 100)
        self.assertAlmostEqual(tr.start, 0.0)

    def test_empty_map_raises(self):
        from opencut.core.transcript_edit import TranscriptMap, text_selection_to_timerange
        empty = TranscriptMap()
        with self.assertRaises(ValueError):
            text_selection_to_timerange(empty, 0, 0)


class TestTimerangeToTextSelection(unittest.TestCase):

    def setUp(self):
        from opencut.core.transcript_edit import build_transcript_map
        self.tmap = build_transcript_map(_whisperx_transcript())

    def test_first_segment(self):
        from opencut.core.transcript_edit import timerange_to_text_selection
        start_idx, end_idx = timerange_to_text_selection(self.tmap, 0.0, 2.0)
        self.assertEqual(start_idx, 0)
        self.assertGreaterEqual(end_idx, 4)

    def test_exact_word_boundaries(self):
        from opencut.core.transcript_edit import timerange_to_text_selection
        start_idx, end_idx = timerange_to_text_selection(self.tmap, 3.5, 5.0)
        self.assertEqual(start_idx, 6)
        self.assertEqual(end_idx, 8)

    def test_no_words_in_range(self):
        from opencut.core.transcript_edit import timerange_to_text_selection
        with self.assertRaises(ValueError):
            timerange_to_text_selection(self.tmap, 2.5, 3.0)

    def test_invalid_range(self):
        from opencut.core.transcript_edit import timerange_to_text_selection
        with self.assertRaises(ValueError):
            timerange_to_text_selection(self.tmap, 5.0, 2.0)


class TestDeleteWords(unittest.TestCase):

    def test_delete_single_word(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_whisperx_transcript())
        segments = delete_words(tmap, [2])  # delete "this"
        self.assertTrue(tmap.words[2].is_deleted)
        self.assertGreater(len(segments), 0)

    def test_delete_multiple_words(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_whisperx_transcript())
        segments = delete_words(tmap, [0, 1, 2, 3, 4, 5])
        # All first paragraph words deleted
        for i in range(6):
            self.assertTrue(tmap.words[i].is_deleted)
        # Remaining words form segments
        self.assertGreater(len(segments), 0)

    def test_delete_no_words(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_whisperx_transcript())
        segments = delete_words(tmap, [])
        # Should return all content as segments
        self.assertGreater(len(segments), 0)

    def test_delete_creates_gap(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_simple_segments())
        # Delete second word - should split into segments
        segments = delete_words(tmap, [1])
        # Word 1 is "world" (1.0-2.0), gap to word 2 "Goodbye" (3.0-4.0)
        # After deletion, word 0 (0.0-1.0) and word 2 (3.0-4.0) are separate
        self.assertGreaterEqual(len(segments), 2)

    def test_delete_out_of_range_ignored(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_simple_segments())
        segments = delete_words(tmap, [999])
        self.assertGreater(len(segments), 0)

    def test_delete_all_words(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_simple_segments())
        segments = delete_words(tmap, [0, 1, 2, 3])
        self.assertEqual(len(segments), 0)

    def test_progress_callback(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_simple_segments())
        calls = []
        delete_words(tmap, [0], on_progress=lambda p, m="": calls.append(p))
        self.assertGreater(len(calls), 0)


class TestRearrangeParagraphs(unittest.TestCase):

    def test_reverse_order(self):
        from opencut.core.transcript_edit import build_transcript_map, rearrange_paragraphs
        tmap = build_transcript_map(_whisperx_transcript())
        segments = rearrange_paragraphs(tmap, [2, 1, 0])
        self.assertEqual(len(segments), 3)
        # First segment should be from paragraph 2 (starts at 6.0)
        self.assertAlmostEqual(segments[0].start, 6.0)

    def test_subset_paragraphs(self):
        from opencut.core.transcript_edit import build_transcript_map, rearrange_paragraphs
        tmap = build_transcript_map(_whisperx_transcript())
        segments = rearrange_paragraphs(tmap, [1])
        self.assertEqual(len(segments), 1)

    def test_duplicate_paragraphs(self):
        from opencut.core.transcript_edit import build_transcript_map, rearrange_paragraphs
        tmap = build_transcript_map(_whisperx_transcript())
        segments = rearrange_paragraphs(tmap, [0, 0, 1])
        self.assertEqual(len(segments), 3)

    def test_empty_order_raises(self):
        from opencut.core.transcript_edit import build_transcript_map, rearrange_paragraphs
        tmap = build_transcript_map(_whisperx_transcript())
        with self.assertRaises(ValueError):
            rearrange_paragraphs(tmap, [])

    def test_invalid_index_raises(self):
        from opencut.core.transcript_edit import build_transcript_map, rearrange_paragraphs
        tmap = build_transcript_map(_whisperx_transcript())
        with self.assertRaises(ValueError):
            rearrange_paragraphs(tmap, [99])

    def test_rearrange_with_deleted_words(self):
        from opencut.core.transcript_edit import (
            build_transcript_map,
            delete_words,
            rearrange_paragraphs,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        delete_words(tmap, [0, 1])  # delete "Hello" and "world"
        segments = rearrange_paragraphs(tmap, [0, 1, 2])
        # First paragraph should still produce a segment (4 remaining words)
        self.assertEqual(len(segments), 3)


class TestTranscriptMapProperties(unittest.TestCase):

    def test_text_property(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        text = tmap.text
        self.assertIn("Hello", text)
        self.assertIn("speech", text)

    def test_text_excludes_deleted(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_whisperx_transcript())
        delete_words(tmap, [0])  # delete "Hello"
        self.assertNotIn("Hello", tmap.text.split()[0] if tmap.text else "")

    def test_word_count(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        self.assertEqual(tmap.word_count, 13)

    def test_word_count_after_delete(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_whisperx_transcript())
        delete_words(tmap, [0, 1])
        self.assertEqual(tmap.word_count, 11)

    def test_get_word(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        w = tmap.get_word(0)
        self.assertIsNotNone(w)
        self.assertEqual(w.text, "Hello")

    def test_get_word_out_of_range(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        self.assertIsNone(tmap.get_word(-1))
        self.assertIsNone(tmap.get_word(999))

    def test_get_paragraph(self):
        from opencut.core.transcript_edit import build_transcript_map
        tmap = build_transcript_map(_whisperx_transcript())
        p = tmap.get_paragraph(1)
        self.assertIsNotNone(p)
        self.assertIn("Second", p.text)

    def test_get_active_words(self):
        from opencut.core.transcript_edit import build_transcript_map, delete_words
        tmap = build_transcript_map(_whisperx_transcript())
        delete_words(tmap, [0])
        active = tmap.get_active_words()
        self.assertEqual(len(active), 12)
        self.assertEqual(active[0].text, "world")


class TestApplyTextEdits(unittest.TestCase):

    @patch("opencut.core.transcript_edit.run_ffmpeg")
    @patch("opencut.core.transcript_edit.get_video_info",
           return_value={"duration": 10.0, "width": 1920, "height": 1080, "fps": 30})
    @patch("os.path.isfile", return_value=True)
    def test_apply_delete_edit(self, mock_isfile, mock_info, mock_ffmpeg):
        from opencut.core.transcript_edit import (
            TextEdit,
            apply_text_edits,
            build_transcript_map,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="delete", word_indices=[0, 1])
        result = apply_text_edits(
            "/test/video.mp4", tmap, [edit], out_path="/test/out.mp4"
        )
        self.assertEqual(result.output_path, "/test/out.mp4")
        self.assertGreater(result.cut_count, 0)

    @patch("opencut.core.transcript_edit.run_ffmpeg")
    @patch("os.path.isfile", return_value=True)
    def test_apply_rearrange_edit(self, mock_isfile, mock_ffmpeg):
        from opencut.core.transcript_edit import (
            TextEdit,
            apply_text_edits,
            build_transcript_map,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="rearrange", new_order=[2, 0, 1])
        result = apply_text_edits(
            "/test/video.mp4", tmap, [edit], out_path="/test/out.mp4"
        )
        self.assertGreater(result.cut_count, 0)

    def test_apply_no_edits_raises(self):
        from opencut.core.transcript_edit import (
            apply_text_edits,
            build_transcript_map,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        with self.assertRaises(ValueError):
            apply_text_edits("/test/video.mp4", tmap, [])

    def test_apply_file_not_found(self):
        from opencut.core.transcript_edit import (
            TextEdit,
            apply_text_edits,
            build_transcript_map,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="delete", word_indices=[0])
        with self.assertRaises(FileNotFoundError):
            apply_text_edits("/nonexistent/video.mp4", tmap, [edit])


class TestExportEditedSequence(unittest.TestCase):

    def test_export_edl(self):
        from opencut.core.transcript_edit import (
            TextEdit,
            build_transcript_map,
            export_edited_sequence,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="keep")

        with tempfile.NamedTemporaryFile(suffix=".edl", delete=False) as f:
            out_path = f.name
        try:
            result = export_edited_sequence(
                "", tmap, [edit], format="edl", out_path=out_path
            )
            self.assertEqual(result["format"], "edl")
            self.assertGreater(result["segment_count"], 0)
            with open(out_path, "r") as f:
                content = f.read()
            self.assertIn("TITLE: Transcript Edit", content)
        finally:
            os.unlink(out_path)

    @patch("opencut.core.transcript_edit.get_video_info",
           return_value={"fps": 24.0})
    def test_export_otio(self, mock_info):
        from opencut.core.transcript_edit import (
            TextEdit,
            build_transcript_map,
            export_edited_sequence,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="keep")

        with tempfile.NamedTemporaryFile(suffix=".otio", delete=False) as f:
            out_path = f.name
        try:
            result = export_edited_sequence(
                "/test/video.mp4", tmap, [edit],
                format="otio", out_path=out_path,
            )
            self.assertEqual(result["format"], "otio")
            with open(out_path, "r") as f:
                otio = json.load(f)
            self.assertEqual(otio["OTIO_SCHEMA"], "Timeline.1")
            tracks = otio["tracks"]["children"]
            self.assertEqual(len(tracks), 1)
            clips = tracks[0]["children"]
            self.assertGreater(len(clips), 0)
        finally:
            os.unlink(out_path)

    def test_export_json(self):
        from opencut.core.transcript_edit import (
            TextEdit,
            build_transcript_map,
            export_edited_sequence,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="keep")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            result = export_edited_sequence(
                "", tmap, [edit], format="json", out_path=out_path,
            )
            self.assertEqual(result["format"], "json")
            with open(out_path, "r") as f:
                data = json.load(f)
            self.assertIn("segments", data)
            self.assertIn("transcript_map", data)
        finally:
            os.unlink(out_path)

    def test_export_invalid_format(self):
        from opencut.core.transcript_edit import (
            TextEdit,
            build_transcript_map,
            export_edited_sequence,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="keep")
        with self.assertRaises(ValueError):
            export_edited_sequence("", tmap, [edit], format="invalid")

    def test_export_with_deletions(self):
        from opencut.core.transcript_edit import (
            TextEdit,
            build_transcript_map,
            export_edited_sequence,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edits = [
            TextEdit(edit_type="delete", word_indices=[0, 1]),
            TextEdit(edit_type="keep"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".edl", delete=False) as f:
            out_path = f.name
        try:
            result = export_edited_sequence(
                "", tmap, edits, format="edl", out_path=out_path
            )
            self.assertGreater(result["segment_count"], 0)
        finally:
            os.unlink(out_path)

    def test_export_auto_path(self):
        from opencut.core.transcript_edit import (
            TextEdit,
            build_transcript_map,
            export_edited_sequence,
        )
        tmap = build_transcript_map(_whisperx_transcript())
        edit = TextEdit(edit_type="keep")
        result = export_edited_sequence("", tmap, [edit], format="edl")
        self.assertTrue(result["output_path"].endswith(".edl"))
        # Clean up
        try:
            os.unlink(result["output_path"])
        except OSError:
            pass


class TestFmtTc(unittest.TestCase):

    def test_zero(self):
        from opencut.core.transcript_edit import _fmt_tc
        self.assertEqual(_fmt_tc(0), "00:00:00:00")

    def test_normal(self):
        from opencut.core.transcript_edit import _fmt_tc
        self.assertEqual(_fmt_tc(3661.5), "01:01:01:15")

    def test_fractional(self):
        from opencut.core.transcript_edit import _fmt_tc
        tc = _fmt_tc(0.1)
        self.assertEqual(tc, "00:00:00:03")


# ===================================================================
# Feature 2: AI Rough Cut Assembly Tests
# ===================================================================

class TestRoughCutBrief(unittest.TestCase):

    def test_defaults(self):
        from opencut.core.rough_cut import RoughCutBrief
        b = RoughCutBrief()
        self.assertEqual(b.style, "narrative")
        self.assertEqual(b.duration, 60.0)
        self.assertEqual(b.tone, "neutral")

    def test_from_dict(self):
        from opencut.core.rough_cut import RoughCutBrief
        d = {"goal": "test goal", "style": "documentary",
             "duration": 120, "keywords": ["nature"]}
        b = RoughCutBrief.from_dict(d)
        self.assertEqual(b.goal, "test goal")
        self.assertEqual(b.style, "documentary")
        self.assertEqual(b.duration, 120.0)
        self.assertEqual(b.keywords, ["nature"])

    def test_to_dict(self):
        from opencut.core.rough_cut import RoughCutBrief
        b = RoughCutBrief(goal="my goal", keywords=["a", "b"])
        d = b.to_dict()
        self.assertEqual(d["goal"], "my goal")
        self.assertEqual(d["keywords"], ["a", "b"])


class TestAnalyzedClip(unittest.TestCase):

    def test_to_dict(self):
        from opencut.core.rough_cut import AnalyzedClip
        c = AnalyzedClip(file_path="/test.mp4", duration=30.0,
                         has_speech=True, quality_score=0.7)
        d = c.to_dict()
        self.assertEqual(d["file_path"], "/test.mp4")
        self.assertEqual(d["duration"], 30.0)


class TestPlannedClip(unittest.TestCase):

    def test_duration(self):
        from opencut.core.rough_cut import PlannedClip
        c = PlannedClip(source_file="/test.mp4", start=5.0, end=15.0)
        self.assertAlmostEqual(c.duration, 10.0)

    def test_to_dict(self):
        from opencut.core.rough_cut import PlannedClip
        c = PlannedClip(source_file="/test.mp4", start=0, end=10,
                        justification="good clip")
        d = c.to_dict()
        self.assertEqual(d["justification"], "good clip")


class TestRoughCutPlan(unittest.TestCase):

    def test_to_dict(self):
        from opencut.core.rough_cut import PlannedClip, RoughCutBrief, RoughCutPlan
        plan = RoughCutPlan(
            clips=[PlannedClip(source_file="/a.mp4", start=0, end=10)],
            brief=RoughCutBrief(goal="test"),
            total_duration=10.0,
        )
        d = plan.to_dict()
        self.assertEqual(len(d["clips"]), 1)
        self.assertIsNotNone(d["brief"])


class TestRoughCutResult(unittest.TestCase):

    def test_to_dict(self):
        from opencut.core.rough_cut import RoughCutResult
        r = RoughCutResult(output_path="/out.mp4", duration=30.0, clip_count=3)
        d = r.to_dict()
        self.assertEqual(d["output_path"], "/out.mp4")
        self.assertEqual(d["clip_count"], 3)


class TestScoreClipQuality(unittest.TestCase):

    def test_empty_clip(self):
        from opencut.core.rough_cut import AnalyzedClip, _score_clip_quality
        c = AnalyzedClip()
        score = _score_clip_quality(c)
        self.assertEqual(score, 0.0)

    def test_speech_clip(self):
        from opencut.core.rough_cut import AnalyzedClip, _score_clip_quality
        c = AnalyzedClip(has_speech=True, duration=60.0)
        score = _score_clip_quality(c)
        self.assertGreater(score, 0.3)

    def test_keywords_boost(self):
        from opencut.core.rough_cut import AnalyzedClip, _score_clip_quality
        c = AnalyzedClip(keywords_found=["nature", "wildlife", "forest"])
        score = _score_clip_quality(c)
        self.assertGreater(score, 0.0)

    def test_max_score_capped(self):
        from opencut.core.rough_cut import AnalyzedClip, _score_clip_quality
        c = AnalyzedClip(
            has_speech=True, duration=120.0,
            keywords_found=["a", "b", "c", "d", "e"],
            speech_segments=[{"start": 0, "end": 100}],
        )
        score = _score_clip_quality(c)
        self.assertLessEqual(score, 1.0)


class TestAnalyzeFootage(unittest.TestCase):

    @patch("opencut.core.rough_cut.get_video_info",
           return_value={"duration": 30.0, "width": 1920, "height": 1080, "fps": 30})
    @patch("opencut.core.rough_cut._analyze_single_clip")
    @patch("os.path.isfile", return_value=True)
    def test_analyze_multiple_clips(self, mock_isfile, mock_analyze, mock_info):
        from opencut.core.rough_cut import AnalyzedClip, analyze_footage
        mock_analyze.return_value = AnalyzedClip(
            file_path="/test.mp4", duration=30.0, has_speech=True,
        )
        result = analyze_footage(["/test.mp4", "/test2.mp4"])
        self.assertEqual(len(result), 2)
        self.assertEqual(mock_analyze.call_count, 2)

    def test_empty_paths_raises(self):
        from opencut.core.rough_cut import analyze_footage
        with self.assertRaises(ValueError):
            analyze_footage([])

    @patch("os.path.isfile", return_value=False)
    def test_nonexistent_file_raises(self, mock_isfile):
        from opencut.core.rough_cut import analyze_footage
        with self.assertRaises(FileNotFoundError):
            analyze_footage(["/nonexistent.mp4"])


class TestGeneratePlan(unittest.TestCase):

    @patch("opencut.core.llm.query_llm")
    def test_plan_from_llm(self, mock_llm):
        from opencut.core.llm import LLMResponse
        from opencut.core.rough_cut import (
            AnalyzedClip,
            RoughCutBrief,
            generate_plan,
        )

        mock_llm.return_value = LLMResponse(
            text=json.dumps({
                "clips": [
                    {"source_file": "clip1.mp4", "start": 0, "end": 10,
                     "justification": "great opening", "score": 0.9,
                     "clip_type": "content"},
                    {"source_file": "clip2.mp4", "start": 5, "end": 15,
                     "justification": "good b-roll", "score": 0.7,
                     "clip_type": "broll"},
                ],
                "narrative_summary": "Test narrative",
            }),
            provider="ollama",
            model="llama3.2",
        )

        brief = RoughCutBrief(goal="test", duration=20)
        footage = [
            AnalyzedClip(file_path="/clip1.mp4", duration=30.0, has_speech=True),
            AnalyzedClip(file_path="/clip2.mp4", duration=20.0, has_speech=False),
        ]

        plan = generate_plan(brief, footage)
        self.assertEqual(len(plan.clips), 2)
        self.assertEqual(plan.narrative_summary, "Test narrative")
        self.assertGreater(plan.total_duration, 0)

    @patch("opencut.core.llm.query_llm")
    def test_fallback_plan_on_llm_error(self, mock_llm):
        from opencut.core.llm import LLMResponse
        from opencut.core.rough_cut import (
            AnalyzedClip,
            RoughCutBrief,
            generate_plan,
        )

        mock_llm.return_value = LLMResponse(
            text="LLM error: connection refused",
            provider="ollama",
            model="llama3.2",
        )

        brief = RoughCutBrief(goal="test", duration=30)
        footage = [
            AnalyzedClip(file_path="/clip.mp4", duration=60.0,
                         quality_score=0.5),
        ]

        plan = generate_plan(brief, footage)
        self.assertGreater(len(plan.clips), 0)

    def test_empty_footage_raises(self):
        from opencut.core.rough_cut import RoughCutBrief, generate_plan
        brief = RoughCutBrief(goal="test")
        with self.assertRaises(ValueError):
            generate_plan(brief, [])


class TestFallbackPlan(unittest.TestCase):

    def test_fallback_respects_duration(self):
        from opencut.core.rough_cut import (
            AnalyzedClip,
            RoughCutBrief,
            _fallback_plan,
        )
        brief = RoughCutBrief(goal="test", duration=10.0)
        footage = [
            AnalyzedClip(file_path="/a.mp4", duration=60.0, quality_score=0.8),
        ]
        plan = _fallback_plan(brief, footage)
        total = sum(c.duration for c in plan.clips)
        self.assertLessEqual(total, 15.0)  # some tolerance

    def test_fallback_empty_footage(self):
        from opencut.core.rough_cut import RoughCutBrief, _fallback_plan
        brief = RoughCutBrief(goal="test")
        plan = _fallback_plan(brief, [])
        self.assertEqual(len(plan.clips), 0)


class TestExecutePlan(unittest.TestCase):

    @patch("opencut.core.rough_cut.run_ffmpeg")
    @patch("os.path.isfile", return_value=True)
    def test_execute_basic(self, mock_isfile, mock_ffmpeg):
        from opencut.core.rough_cut import (
            PlannedClip,
            RoughCutPlan,
            execute_plan,
        )
        plan = RoughCutPlan(
            clips=[
                PlannedClip(source_file="/clip1.mp4", start=0, end=10, order=0),
                PlannedClip(source_file="/clip2.mp4", start=5, end=15, order=1),
            ],
            total_duration=20.0,
        )
        result = execute_plan(plan, out_path="/out.mp4")
        self.assertEqual(result.output_path, "/out.mp4")
        self.assertGreater(result.clip_count, 0)

    def test_execute_empty_plan_raises(self):
        from opencut.core.rough_cut import RoughCutPlan, execute_plan
        with self.assertRaises(ValueError):
            execute_plan(RoughCutPlan())

    @patch("os.path.isfile", return_value=False)
    def test_execute_missing_source_raises(self, mock_isfile):
        from opencut.core.rough_cut import PlannedClip, RoughCutPlan, execute_plan
        plan = RoughCutPlan(clips=[
            PlannedClip(source_file="/missing.mp4", start=0, end=5),
        ])
        with self.assertRaises(FileNotFoundError):
            execute_plan(plan)


class TestParseBrief(unittest.TestCase):

    @patch("opencut.core.llm.query_llm")
    def test_parse_brief_with_llm(self, mock_llm):
        from opencut.core.llm import LLMResponse
        from opencut.core.rough_cut import _parse_brief

        mock_llm.return_value = LLMResponse(
            text=json.dumps({
                "goal": "Create a nature documentary",
                "style": "documentary",
                "duration": 120,
                "keywords": ["nature", "wildlife"],
                "tone": "calm",
                "pacing": "slow",
            }),
            provider="ollama",
            model="llama3.2",
        )

        brief = _parse_brief("Create a 2 minute nature documentary")
        self.assertEqual(brief.style, "documentary")
        self.assertEqual(brief.duration, 120)

    @patch("opencut.core.llm.query_llm")
    def test_parse_brief_fallback(self, mock_llm):
        from opencut.core.llm import LLMResponse
        from opencut.core.rough_cut import _parse_brief

        mock_llm.return_value = LLMResponse(text="LLM error: fail")

        brief = _parse_brief("A fast highlight reel about 30 seconds long")
        # Fallback keyword parsing
        self.assertEqual(brief.pacing, "fast")
        self.assertEqual(brief.style, "highlight")
        self.assertEqual(brief.duration, 30.0)

    @patch("opencut.core.llm.query_llm")
    def test_parse_brief_minute_duration(self, mock_llm):
        from opencut.core.llm import LLMResponse
        from opencut.core.rough_cut import _parse_brief

        mock_llm.return_value = LLMResponse(text="LLM error: fail")

        brief = _parse_brief("2 minute documentary")
        self.assertEqual(brief.duration, 120.0)
        self.assertEqual(brief.style, "documentary")


class TestExtractJson(unittest.TestCase):

    def test_plain_json(self):
        from opencut.core.rough_cut import _extract_json
        result = _extract_json('{"key": "value"}')
        self.assertEqual(result["key"], "value")

    def test_markdown_fenced(self):
        from opencut.core.rough_cut import _extract_json
        text = 'Here is the plan:\n```json\n{"clips": []}\n```'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["clips"], [])

    def test_invalid_json(self):
        from opencut.core.rough_cut import _extract_json
        result = _extract_json("not json at all")
        self.assertIsNone(result)

    def test_embedded_json(self):
        from opencut.core.rough_cut import _extract_json
        text = 'Some text before {"result": true} and after'
        result = _extract_json(text)
        self.assertIsNotNone(result)


class TestRoughCutFromBrief(unittest.TestCase):

    @patch("opencut.core.rough_cut.execute_plan")
    @patch("opencut.core.rough_cut.generate_plan")
    @patch("opencut.core.rough_cut.analyze_footage")
    @patch("opencut.core.rough_cut._parse_brief")
    def test_full_pipeline(self, mock_parse, mock_analyze, mock_plan, mock_exec):
        from opencut.core.rough_cut import (
            AnalyzedClip,
            PlannedClip,
            RoughCutBrief,
            RoughCutPlan,
            RoughCutResult,
            rough_cut_from_brief,
        )

        mock_parse.return_value = RoughCutBrief(goal="test", duration=30)
        mock_analyze.return_value = [
            AnalyzedClip(file_path="/clip.mp4", duration=60.0),
        ]
        mock_plan.return_value = RoughCutPlan(
            clips=[PlannedClip(source_file="/clip.mp4", start=0, end=30)],
            total_duration=30.0,
        )
        mock_exec.return_value = RoughCutResult(
            output_path="/out.mp4", duration=30.0, clip_count=1,
        )

        result = rough_cut_from_brief(
            ["/clip.mp4"], "Make a 30 second highlight"
        )
        self.assertEqual(result.output_path, "/out.mp4")
        self.assertEqual(result.duration, 30.0)

    def test_empty_files_raises(self):
        from opencut.core.rough_cut import rough_cut_from_brief
        with self.assertRaises(ValueError):
            rough_cut_from_brief([], "test brief")

    @patch("opencut.core.rough_cut.execute_plan")
    @patch("opencut.core.rough_cut.generate_plan")
    @patch("opencut.core.rough_cut.analyze_footage")
    @patch("opencut.core.rough_cut._parse_brief")
    def test_pipeline_progress_callbacks(self, mock_parse, mock_analyze,
                                         mock_plan, mock_exec):
        from opencut.core.rough_cut import (
            AnalyzedClip,
            PlannedClip,
            RoughCutBrief,
            RoughCutPlan,
            RoughCutResult,
            rough_cut_from_brief,
        )

        mock_parse.return_value = RoughCutBrief(goal="test")
        mock_analyze.return_value = [
            AnalyzedClip(file_path="/a.mp4", duration=30.0),
        ]
        mock_plan.return_value = RoughCutPlan(
            clips=[PlannedClip(source_file="/a.mp4", start=0, end=10)],
            total_duration=10.0,
        )
        mock_exec.return_value = RoughCutResult(output_path="/out.mp4")

        calls = []
        rough_cut_from_brief(
            ["/a.mp4"], "test",
            on_progress=lambda p, m="": calls.append(p),
        )
        self.assertGreater(len(calls), 0)


class TestExportPlanEdl(unittest.TestCase):

    def test_edl_export(self):
        from opencut.core.rough_cut import PlannedClip, RoughCutPlan, _export_plan_edl

        plan = RoughCutPlan(clips=[
            PlannedClip(source_file="/clip.mp4", start=0, end=10,
                        justification="opening"),
            PlannedClip(source_file="/clip2.mp4", start=5, end=15,
                        justification="b-roll"),
        ])

        with tempfile.NamedTemporaryFile(suffix=".edl", delete=False, mode="w") as f:
            edl_path = f.name
        try:
            _export_plan_edl(plan, edl_path)
            with open(edl_path, "r") as f:
                content = f.read()
            self.assertIn("TITLE: AI Rough Cut", content)
            self.assertIn("opening", content)
            self.assertIn("b-roll", content)
        finally:
            os.unlink(edl_path)


class TestFmtSrtTime(unittest.TestCase):

    def test_zero(self):
        from opencut.core.rough_cut import _fmt_srt_time
        self.assertEqual(_fmt_srt_time(0), "00:00:00,000")

    def test_normal(self):
        from opencut.core.rough_cut import _fmt_srt_time
        self.assertEqual(_fmt_srt_time(3661.5), "01:01:01,500")


class TestFmtEdlTc(unittest.TestCase):

    def test_zero(self):
        from opencut.core.rough_cut import _fmt_edl_tc
        self.assertEqual(_fmt_edl_tc(0), "00:00:00:00")


# ===================================================================
# Route Smoke Tests
# ===================================================================

class TestTranscriptEditRoutes(unittest.TestCase):
    """Smoke tests for transcript edit and rough cut routes."""

    def setUp(self):
        from opencut.config import OpenCutConfig
        from opencut.routes.transcript_edit_routes import transcript_edit_bp
        from opencut.server import create_app
        config = OpenCutConfig()
        self.app = create_app(config=config)
        self.app.config["TESTING"] = True
        # Register our blueprint for testing (user handles this in production)
        if "transcript_edit" not in self.app.blueprints:
            self.app.register_blueprint(transcript_edit_bp)
        self.client = self.app.test_client()
        resp = self.client.get("/health")
        data = resp.get_json()
        self.token = data.get("csrf_token", "")
        self.headers = {
            "X-OpenCut-Token": self.token,
            "Content-Type": "application/json",
        }

    def test_build_map_no_data(self):
        """POST without transcript_json should fail in the async job."""
        resp = self.client.post(
            "/transcript-edit/build-map",
            headers=self.headers,
            json={},
        )
        # Should get a job_id (error happens async) or 400
        self.assertIn(resp.status_code, (200, 400))

    def test_delete_words_no_data(self):
        resp = self.client.post(
            "/transcript-edit/delete-words",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_rearrange_no_data(self):
        resp = self.client.post(
            "/transcript-edit/rearrange",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_export_no_data(self):
        resp = self.client.post(
            "/transcript-edit/export",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_rough_cut_analyze_no_data(self):
        resp = self.client.post(
            "/rough-cut/analyze",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_rough_cut_plan_no_data(self):
        resp = self.client.post(
            "/rough-cut/plan",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_rough_cut_execute_no_data(self):
        resp = self.client.post(
            "/rough-cut/execute",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_rough_cut_auto_no_data(self):
        resp = self.client.post(
            "/rough-cut/auto",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (200, 400))

    def test_csrf_required_build_map(self):
        resp = self.client.post(
            "/transcript-edit/build-map",
            json={"transcript_json": {"segments": []}},
        )
        self.assertEqual(resp.status_code, 403)

    def test_csrf_required_rough_cut(self):
        resp = self.client.post(
            "/rough-cut/auto",
            json={"file_paths": ["/test.mp4"]},
        )
        self.assertEqual(resp.status_code, 403)


# ===================================================================
# Reconstruct Map Helper Tests
# ===================================================================

class TestReconstructMap(unittest.TestCase):

    def test_roundtrip(self):
        from opencut.core.transcript_edit import build_transcript_map
        from opencut.routes.transcript_edit_routes import _reconstruct_map

        tmap = build_transcript_map(_whisperx_transcript())
        d = tmap.to_dict()
        reconstructed = _reconstruct_map(d)
        self.assertEqual(len(reconstructed.words), len(tmap.words))
        self.assertEqual(len(reconstructed.paragraphs), len(tmap.paragraphs))
        self.assertEqual(reconstructed.language, tmap.language)

    def test_empty_map(self):
        from opencut.routes.transcript_edit_routes import _reconstruct_map
        tmap = _reconstruct_map({})
        self.assertEqual(len(tmap.words), 0)
        self.assertEqual(len(tmap.paragraphs), 0)


if __name__ == "__main__":
    unittest.main()
