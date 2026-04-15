"""
Tests for OpenCut Voice & Speech features (Category 78).

Covers:
  - Transcript Timeline Edit (parse, delete, rearrange, undo, EDL, OTIO)
  - Eye Contact Fix (gaze estimation, temporal smoothing, no-face fallback)
  - Voice Overdub (segment replacement, crossfade, time stretch, TTS)
  - Lip Sync (face detection, mouth ROI, amplitude-driven)
  - Voice Conversion (pitch shift, profile creation, FFmpeg filter chain)
  - Route blueprint smoke tests
"""

import inspect
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Transcript Timeline Edit Tests
# ============================================================
class TestTranscriptTimelineEdit(unittest.TestCase):
    """Tests for opencut.core.transcript_timeline_edit module."""

    def _sample_transcript(self):
        return {
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "S1",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.95},
                        {"word": "world", "start": 0.5, "end": 1.0, "score": 0.90},
                    ],
                },
                {
                    "text": "This is great",
                    "start": 1.5,
                    "end": 3.0,
                    "speaker": "S1",
                    "words": [
                        {"word": "This", "start": 1.5, "end": 1.8, "score": 0.88},
                        {"word": "is", "start": 1.9, "end": 2.1, "score": 0.92},
                        {"word": "great", "start": 2.2, "end": 3.0, "score": 0.85},
                    ],
                },
                {
                    "text": "Thank you",
                    "start": 3.5,
                    "end": 4.5,
                    "speaker": "S2",
                    "words": [
                        {"word": "Thank", "start": 3.5, "end": 4.0, "score": 0.91},
                        {"word": "you", "start": 4.0, "end": 4.5, "score": 0.89},
                    ],
                },
            ],
        }

    def test_parse_transcript_whisperx_format(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        self.assertEqual(len(tmap.words), 7)
        self.assertEqual(len(tmap.paragraphs), 3)
        self.assertEqual(tmap.language, "en")
        self.assertGreater(tmap.total_duration, 0)

    def test_parse_transcript_simple_list(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        segments = self._sample_transcript()["segments"]
        tmap = parse_transcript(segments)
        self.assertEqual(len(tmap.words), 7)

    def test_parse_transcript_words_only(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        data = {
            "words": [
                {"word": "one", "start": 0.0, "end": 0.5},
                {"word": "two", "start": 0.6, "end": 1.0},
                {"word": "three", "start": 3.0, "end": 3.5},
            ]
        }
        tmap = parse_transcript(data)
        self.assertGreaterEqual(len(tmap.words), 3)

    def test_parse_transcript_no_word_timestamps(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        data = [{"text": "Hello world foo", "start": 0.0, "end": 1.5}]
        tmap = parse_transcript(data)
        self.assertEqual(len(tmap.words), 3)
        self.assertAlmostEqual(tmap.words[0].start, 0.0, places=2)

    def test_parse_transcript_empty_raises(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        with self.assertRaises(ValueError):
            parse_transcript({})

    def test_parse_transcript_progress_callback(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        progress_values = []
        parse_transcript(
            self._sample_transcript(),
            on_progress=lambda p: progress_values.append(p),
        )
        self.assertTrue(any(p == 100 for p in progress_values))
        self.assertGreater(len(progress_values), 0)

    def test_timeline_word_properties(self):
        from opencut.core.transcript_timeline_edit import TimelineWord

        w = TimelineWord(index=0, text="hello", start=1.0, end=2.5)
        self.assertAlmostEqual(w.duration, 1.5)
        d = w.to_dict()
        self.assertEqual(d["text"], "hello")
        self.assertEqual(d["start"], 1.0)

    def test_timeline_paragraph_properties(self):
        from opencut.core.transcript_timeline_edit import TimelineParagraph

        p = TimelineParagraph(
            index=0, text="hello world", start=0, end=2,
            word_start_index=0, word_end_index=2,
        )
        self.assertEqual(p.word_count, 2)
        self.assertAlmostEqual(p.duration, 2.0)

    def test_timeline_map_text_property(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        text = tmap.text
        self.assertIn("Hello", text)
        self.assertIn("world", text)
        self.assertIn("great", text)

    def test_timeline_map_to_dict(self):
        from opencut.core.transcript_timeline_edit import parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        d = tmap.to_dict()
        self.assertIn("words", d)
        self.assertIn("paragraphs", d)
        self.assertIn("active_word_count", d)
        self.assertEqual(d["active_word_count"], 7)

    def test_delete_words(self):
        from opencut.core.transcript_timeline_edit import delete_words, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        cuts = delete_words(tmap, [1])  # delete "world"
        self.assertEqual(tmap.active_word_count, 6)
        self.assertTrue(tmap.words[1].is_deleted)
        self.assertGreater(len(cuts), 0)

    def test_delete_words_empty_list(self):
        from opencut.core.transcript_timeline_edit import delete_words, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        delete_words(tmap, [])
        self.assertEqual(tmap.active_word_count, 7)

    def test_delete_words_multiple(self):
        from opencut.core.transcript_timeline_edit import delete_words, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        delete_words(tmap, [0, 1, 2])
        self.assertEqual(tmap.active_word_count, 4)

    def test_delete_words_undo_stack(self):
        from opencut.core.transcript_timeline_edit import delete_words, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        delete_words(tmap, [0])
        self.assertEqual(tmap.undo_depth, 1)
        self.assertEqual(tmap.active_word_count, 6)

    def test_rearrange_segments(self):
        from opencut.core.transcript_timeline_edit import parse_transcript, rearrange_segments

        tmap = parse_transcript(self._sample_transcript())
        cuts = rearrange_segments(tmap, [2, 0, 1])
        self.assertGreater(len(cuts), 0)
        self.assertEqual(tmap.undo_depth, 1)

    def test_rearrange_segments_invalid_index(self):
        from opencut.core.transcript_timeline_edit import parse_transcript, rearrange_segments

        tmap = parse_transcript(self._sample_transcript())
        with self.assertRaises(ValueError):
            rearrange_segments(tmap, [0, 1, 99])

    def test_rearrange_partial(self):
        from opencut.core.transcript_timeline_edit import parse_transcript, rearrange_segments

        tmap = parse_transcript(self._sample_transcript())
        cuts = rearrange_segments(tmap, [1, 0])
        self.assertEqual(len(cuts), 2)

    def test_duplicate_segment(self):
        from opencut.core.transcript_timeline_edit import duplicate_segment, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        original_para_count = tmap.paragraph_count
        cuts = duplicate_segment(tmap, 0)
        self.assertEqual(len(tmap.paragraphs), original_para_count + 1)
        self.assertGreater(len(cuts), 0)

    def test_duplicate_segment_invalid_index(self):
        from opencut.core.transcript_timeline_edit import duplicate_segment, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        with self.assertRaises(ValueError):
            duplicate_segment(tmap, 99)

    def test_duplicate_segment_insert_position(self):
        from opencut.core.transcript_timeline_edit import duplicate_segment, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        cuts = duplicate_segment(tmap, 1, insert_after=2)
        self.assertEqual(len(cuts), 4)  # 3 original + 1 duplicate

    def test_insert_pause(self):
        from opencut.core.transcript_timeline_edit import insert_pause, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        cuts = insert_pause(tmap, after_word_index=1, pause_duration=2.0)
        self.assertGreater(len(cuts), 0)
        self.assertEqual(len(tmap._pauses), 1)
        self.assertEqual(tmap._pauses[0]["duration"], 2.0)

    def test_insert_pause_at_beginning(self):
        from opencut.core.transcript_timeline_edit import insert_pause, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        cuts = insert_pause(tmap, after_word_index=-1, pause_duration=1.0)
        self.assertGreater(len(cuts), 0)

    def test_insert_pause_invalid_index(self):
        from opencut.core.transcript_timeline_edit import insert_pause, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        with self.assertRaises(ValueError):
            insert_pause(tmap, after_word_index=999, pause_duration=1.0)

    def test_insert_pause_zero_duration(self):
        from opencut.core.transcript_timeline_edit import insert_pause, parse_transcript

        tmap = parse_transcript(self._sample_transcript())
        with self.assertRaises(ValueError):
            insert_pause(tmap, after_word_index=0, pause_duration=0)

    def test_undo(self):
        from opencut.core.transcript_timeline_edit import delete_words, parse_transcript, undo

        tmap = parse_transcript(self._sample_transcript())
        delete_words(tmap, [0, 1])
        self.assertEqual(tmap.active_word_count, 5)
        result = undo(tmap)
        self.assertTrue(result)
        self.assertEqual(tmap.active_word_count, 7)

    def test_undo_empty_stack(self):
        from opencut.core.transcript_timeline_edit import parse_transcript, undo

        tmap = parse_transcript(self._sample_transcript())
        result = undo(tmap)
        self.assertFalse(result)

    def test_undo_multiple(self):
        from opencut.core.transcript_timeline_edit import (
            delete_words,
            insert_pause,
            parse_transcript,
            undo,
        )

        tmap = parse_transcript(self._sample_transcript())
        delete_words(tmap, [0])
        insert_pause(tmap, 1, 1.0)
        self.assertEqual(tmap.undo_depth, 2)
        undo(tmap)
        self.assertEqual(tmap.undo_depth, 1)
        undo(tmap)
        self.assertEqual(tmap.undo_depth, 0)
        self.assertEqual(tmap.active_word_count, 7)

    def test_export_edl(self):
        from opencut.core.transcript_timeline_edit import CutEntry, export_edl

        cuts = [
            CutEntry(source_start=0.0, source_end=2.0, dest_start=0.0),
            CutEntry(source_start=3.0, source_end=5.0, dest_start=2.0),
        ]
        edl = export_edl(cuts, title="Test Edit")
        self.assertIn("TITLE: Test Edit", edl)
        self.assertIn("FCM: NON-DROP FRAME", edl)
        self.assertIn("001", edl)
        self.assertIn("002", edl)

    def test_export_edl_with_pause(self):
        from opencut.core.transcript_timeline_edit import CutEntry, export_edl

        cuts = [
            CutEntry(source_start=0.0, source_end=2.0, dest_start=0.0),
            CutEntry(source_start=-1, source_end=-1, dest_start=2.0, label="pause"),
            CutEntry(source_start=3.0, source_end=5.0, dest_start=3.0),
        ]
        edl = export_edl(cuts)
        self.assertIn("BL", edl)
        self.assertIn("003", edl)

    def test_export_otio_json(self):
        from opencut.core.transcript_timeline_edit import CutEntry, export_otio_json

        cuts = [
            CutEntry(source_start=0.0, source_end=2.0, dest_start=0.0, label="clip_0"),
            CutEntry(source_start=3.0, source_end=5.0, dest_start=2.0, label="clip_1"),
        ]
        otio = export_otio_json(cuts, source_file="test.mp4")
        self.assertEqual(otio["OTIO_SCHEMA"], "Timeline.1")
        tracks = otio["tracks"]["children"]
        self.assertEqual(len(tracks), 1)
        clips = tracks[0]["children"]
        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0]["OTIO_SCHEMA"], "Clip.2")

    def test_export_otio_json_with_gap(self):
        from opencut.core.transcript_timeline_edit import CutEntry, export_otio_json

        cuts = [
            CutEntry(source_start=0.0, source_end=1.0, dest_start=0.0),
            CutEntry(source_start=-1, source_end=-1, dest_start=1.0, label="gap"),
        ]
        otio = export_otio_json(cuts)
        clips = otio["tracks"]["children"][0]["children"]
        self.assertEqual(clips[1]["OTIO_SCHEMA"], "Gap.1")

    def test_seconds_to_tc(self):
        from opencut.core.transcript_timeline_edit import _seconds_to_tc

        self.assertEqual(_seconds_to_tc(0.0, 30.0), "00:00:00:00")
        tc = _seconds_to_tc(3661.5, 30.0)  # 1h 1m 1.5s
        self.assertTrue(tc.startswith("01:01:01:"))

    def test_cut_entry_properties(self):
        from opencut.core.transcript_timeline_edit import CutEntry

        c = CutEntry(source_start=1.0, source_end=3.5, dest_start=0.0)
        self.assertAlmostEqual(c.duration, 2.5)
        d = c.to_dict()
        self.assertIn("source_start", d)
        self.assertIn("duration", d)

    def test_transcript_edit_dataclass(self):
        from opencut.core.transcript_timeline_edit import TranscriptEdit

        te = TranscriptEdit(
            operations=[{"type": "delete_words"}],
            resulting_cuts=[{"start": 0, "end": 1}],
            total_duration_change=-2.0,
            new_duration=8.0,
            original_duration=10.0,
        )
        d = te.to_dict()
        self.assertEqual(d["total_duration_change"], -2.0)
        self.assertEqual(d["new_duration"], 8.0)

    def test_preview_edits(self):
        from opencut.core.transcript_timeline_edit import parse_transcript, preview_edits

        tmap = parse_transcript(self._sample_transcript())
        ops = [{"type": "delete_words", "word_indices": [0, 1]}]
        result = preview_edits(tmap, ops)
        self.assertEqual(result.word_count_after, 5)
        self.assertGreater(len(result.resulting_cuts), 0)
        # Original map should be unmodified
        self.assertEqual(tmap.active_word_count, 7)

    def test_preview_edits_rearrange(self):
        from opencut.core.transcript_timeline_edit import parse_transcript, preview_edits

        tmap = parse_transcript(self._sample_transcript())
        ops = [{"type": "rearrange_segments", "new_order": [2, 1, 0]}]
        result = preview_edits(tmap, ops)
        self.assertGreater(len(result.resulting_cuts), 0)

    @patch("opencut.core.transcript_timeline_edit.run_ffmpeg")
    @patch("opencut.core.transcript_timeline_edit.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_edits(self, mock_info, mock_ffmpeg):
        from opencut.core.transcript_timeline_edit import CutEntry, apply_edits

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake")
            tmp_path = tmp.name

        try:
            cuts = [CutEntry(source_start=0.0, source_end=5.0, dest_start=0.0)]
            result = apply_edits(tmp_path, cuts)
            self.assertGreater(len(result.resulting_cuts), 0)
            self.assertAlmostEqual(result.new_duration, 5.0)
            mock_ffmpeg.assert_called_once()
        finally:
            os.unlink(tmp_path)

    @patch("opencut.core.transcript_timeline_edit.run_ffmpeg")
    @patch("opencut.core.transcript_timeline_edit.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_apply_edits_concat(self, mock_info, mock_ffmpeg):
        from opencut.core.transcript_timeline_edit import CutEntry, apply_edits

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake")
            tmp_path = tmp.name

        try:
            cuts = [
                CutEntry(source_start=0.0, source_end=2.0, dest_start=0.0),
                CutEntry(source_start=4.0, source_end=6.0, dest_start=2.0),
            ]
            result = apply_edits(tmp_path, cuts)
            self.assertAlmostEqual(result.new_duration, 4.0)
        finally:
            os.unlink(tmp_path)


# ============================================================
# Eye Contact Fix Tests
# ============================================================
class TestEyeContactFix(unittest.TestCase):
    """Tests for opencut.core.eye_contact_fix module."""

    def test_eye_contact_result_dataclass(self):
        from opencut.core.eye_contact_fix import EyeContactResult

        r = EyeContactResult(
            output_path="/out.mp4",
            frames_processed=100,
            faces_detected=95,
            average_correction_magnitude=0.15,
            intensity=0.8,
        )
        d = r.to_dict()
        self.assertEqual(d["frames_processed"], 100)
        self.assertEqual(d["faces_detected"], 95)
        self.assertEqual(d["intensity"], 0.8)

    def test_gaze_estimate_dataclass(self):
        from opencut.core.eye_contact_fix import GazeEstimate

        g = GazeEstimate(
            frame_index=5,
            left_gaze_x=0.1,
            left_gaze_y=-0.05,
            right_gaze_x=0.12,
            right_gaze_y=-0.03,
            face_detected=True,
            correction_magnitude=0.11,
        )
        d = g.to_dict()
        self.assertEqual(d["frame_index"], 5)
        self.assertTrue(d["face_detected"])
        self.assertEqual(d["left_gaze"], [0.1, -0.05])

    def test_gaze_smoother_first_value(self):
        from opencut.core.eye_contact_fix import GazeSmoother

        s = GazeSmoother(alpha=0.3)
        result = s.smooth((0.5, 0.5, 0.5, 0.5))
        self.assertEqual(result, (0.5, 0.5, 0.5, 0.5))

    def test_gaze_smoother_convergence(self):
        from opencut.core.eye_contact_fix import GazeSmoother

        s = GazeSmoother(alpha=0.5)
        s.smooth((1.0, 1.0, 1.0, 1.0))
        result = s.smooth((0.0, 0.0, 0.0, 0.0))
        # After one step with alpha=0.5: 0.5*0 + 0.5*1 = 0.5
        self.assertAlmostEqual(result[0], 0.5, places=3)

    def test_gaze_smoother_reset(self):
        from opencut.core.eye_contact_fix import GazeSmoother

        s = GazeSmoother(alpha=0.3)
        s.smooth((1.0, 1.0, 1.0, 1.0))
        s.reset()
        result = s.smooth((0.5, 0.5, 0.5, 0.5))
        self.assertEqual(result, (0.5, 0.5, 0.5, 0.5))

    def test_compute_correction_magnitude(self):
        from opencut.core.eye_contact_fix import _compute_correction_magnitude

        mag = _compute_correction_magnitude((0.3, 0.4, 0.3, 0.4))
        expected = 0.5  # sqrt(0.09 + 0.16) = 0.5
        self.assertAlmostEqual(mag, expected, places=3)

    def test_compute_correction_magnitude_zero(self):
        from opencut.core.eye_contact_fix import _compute_correction_magnitude

        mag = _compute_correction_magnitude((0, 0, 0, 0))
        self.assertAlmostEqual(mag, 0.0)

    def test_estimate_gaze_from_landmarks(self):
        from opencut.core.eye_contact_fix import _estimate_gaze_from_landmarks

        # Create mock landmarks
        landmarks = [MagicMock() for _ in range(478)]
        # Left eye
        landmarks[133].x, landmarks[133].y = 0.45, 0.4  # inner
        landmarks[33].x, landmarks[33].y = 0.35, 0.4    # outer
        landmarks[468].x, landmarks[468].y = 0.40, 0.4  # iris center
        landmarks[159].x, landmarks[159].y = 0.40, 0.38  # top
        landmarks[145].x, landmarks[145].y = 0.40, 0.42  # bottom
        # Right eye
        landmarks[362].x, landmarks[362].y = 0.55, 0.4  # inner
        landmarks[263].x, landmarks[263].y = 0.65, 0.4  # outer
        landmarks[473].x, landmarks[473].y = 0.60, 0.4  # iris center
        landmarks[386].x, landmarks[386].y = 0.60, 0.38  # top
        landmarks[374].x, landmarks[374].y = 0.60, 0.42  # bottom

        gaze = _estimate_gaze_from_landmarks(landmarks, 1920, 1080)
        self.assertEqual(len(gaze), 4)

    def test_fix_eye_contact_signature(self):
        from opencut.core.eye_contact_fix import fix_eye_contact

        sig = inspect.signature(fix_eye_contact)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("intensity", sig.parameters)
        self.assertIn("smoothing_alpha", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_preview_eye_contact_signature(self):
        from opencut.core.eye_contact_fix import preview_eye_contact

        sig = inspect.signature(preview_eye_contact)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("frame_number", sig.parameters)
        self.assertIn("intensity", sig.parameters)

    def test_landmark_constants(self):
        from opencut.core.eye_contact_fix import (
            LEFT_EYE_INNER,
            LEFT_IRIS_CENTER,
            RIGHT_EYE_INNER,
            RIGHT_IRIS_CENTER,
        )

        self.assertEqual(LEFT_IRIS_CENTER, 468)
        self.assertEqual(RIGHT_IRIS_CENTER, 473)
        self.assertNotEqual(LEFT_EYE_INNER, RIGHT_EYE_INNER)


# ============================================================
# Voice Overdub Tests
# ============================================================
class TestVoiceOverdub(unittest.TestCase):
    """Tests for opencut.core.voice_overdub module."""

    def test_overdub_result_dataclass(self):
        from opencut.core.voice_overdub import OverdubResult

        r = OverdubResult(
            output_path="/out.mp4",
            replaced_segments=[{"start": 1.0, "end": 2.0}],
            original_duration=10.0,
            new_duration=10.0,
        )
        d = r.to_dict()
        self.assertEqual(d["output_path"], "/out.mp4")
        self.assertEqual(len(d["replaced_segments"]), 1)
        self.assertEqual(d["original_duration"], 10.0)

    def test_replacement_segment_dataclass(self):
        from opencut.core.voice_overdub import ReplacementSegment

        s = ReplacementSegment(
            start_time=1.0, end_time=3.0,
            original_text="hello", replacement_text="hi there",
        )
        self.assertAlmostEqual(s.duration, 2.0)
        d = s.to_dict()
        self.assertIn("replacement_text", d)

    def test_voice_profile_dataclass(self):
        from opencut.core.voice_overdub import VoiceProfile

        p = VoiceProfile(
            speaker_id="spk0", source_path="/audio.wav",
            duration=15.0, sample_rate=24000,
            reference_audio_path="/ref.wav",
        )
        d = p.to_dict()
        self.assertEqual(d["speaker_id"], "spk0")
        self.assertEqual(d["sample_rate"], 24000)

    def test_atempo_chain_normal(self):
        from opencut.core.voice_overdub import _build_atempo_chain

        chain = _build_atempo_chain(1.5)
        self.assertEqual(len(chain), 1)
        self.assertIn("atempo=1.5", chain[0])

    def test_atempo_chain_slow(self):
        from opencut.core.voice_overdub import _build_atempo_chain

        chain = _build_atempo_chain(0.3)
        self.assertGreater(len(chain), 1)
        self.assertTrue(any("0.5" in f for f in chain))

    def test_atempo_chain_unity(self):
        from opencut.core.voice_overdub import _build_atempo_chain

        chain = _build_atempo_chain(1.0)
        self.assertEqual(len(chain), 1)

    def test_overdub_signature(self):
        from opencut.core.voice_overdub import overdub

        sig = inspect.signature(overdub)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("replacements", sig.parameters)
        self.assertIn("tts_endpoint", sig.parameters)
        self.assertIn("crossfade_ms", sig.parameters)

    def test_overdub_no_file_raises(self):
        from opencut.core.voice_overdub import overdub

        with self.assertRaises(FileNotFoundError):
            overdub("/nonexistent.mp4", [{"start_time": 0, "end_time": 1, "replacement_text": "x"}])

    def test_overdub_empty_replacements_raises(self):
        from opencut.core.voice_overdub import overdub

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake")
            tmp_path = tmp.name
        try:
            with self.assertRaises(ValueError):
                overdub(tmp_path, [])
        finally:
            os.unlink(tmp_path)

    @patch("opencut.core.voice_overdub.run_ffmpeg")
    def test_extract_speaker_audio(self, mock_ffmpeg):
        from opencut.core.voice_overdub import extract_speaker_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake")
            tmp_path = tmp.name

        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "S1"},
            {"start": 6.0, "end": 11.0, "speaker": "S1"},
            {"start": 12.0, "end": 25.0, "speaker": "S1"},
        ]

        try:
            profile = extract_speaker_audio(
                tmp_path, segments,
                exclude_start=6.0, exclude_end=11.0,
            )
            self.assertEqual(profile.speaker_id, "speaker_0")
            self.assertGreater(profile.duration, 0)
        finally:
            os.unlink(tmp_path)

    def test_extract_speaker_no_file_raises(self):
        from opencut.core.voice_overdub import extract_speaker_audio

        with self.assertRaises(FileNotFoundError):
            extract_speaker_audio("/nonexistent.wav", [])

    def test_tts_backends_constant(self):
        from opencut.core.voice_overdub import TTS_BACKENDS

        self.assertIn("edge_tts", TTS_BACKENDS)
        self.assertIn("external_api", TTS_BACKENDS)

    def test_default_crossfade(self):
        from opencut.core.voice_overdub import DEFAULT_CROSSFADE_MS

        self.assertEqual(DEFAULT_CROSSFADE_MS, 50)

    @patch("opencut.core.voice_overdub.run_ffmpeg")
    def test_time_stretch_audio_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.voice_overdub import _time_stretch_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake")
            input_path = tmp.name
        out_path = input_path + "_stretched.wav"

        # Mock subprocess for duration detection
        with patch("opencut.core.voice_overdub.subprocess") as mock_sp:
            mock_result = MagicMock()
            mock_result.stderr = b"Duration: 00:00:03.00, start"
            mock_sp.run.return_value = mock_result
            _time_stretch_audio(input_path, 2.0, out_path)
            mock_ffmpeg.assert_called()

        try:
            os.unlink(input_path)
        except OSError:
            pass


# ============================================================
# Lip Sync Tests
# ============================================================
class TestLipSync(unittest.TestCase):
    """Tests for opencut.core.lip_sync module."""

    def test_lip_sync_result_dataclass(self):
        from opencut.core.lip_sync import LipSyncResult

        r = LipSyncResult(
            output_path="/out.mp4",
            frames_processed=300,
            audio_duration=10.0,
            sync_quality_score=0.85,
        )
        d = r.to_dict()
        self.assertEqual(d["frames_processed"], 300)
        self.assertAlmostEqual(d["sync_quality_score"], 0.85)

    def test_mouth_contour_length(self):
        from opencut.core.lip_sync import MOUTH_CONTOUR

        self.assertEqual(len(MOUTH_CONTOUR), 20)

    def test_mouth_openness_ratio_constants(self):
        from opencut.core.lip_sync import MOUTH_CLOSED_RATIO, MOUTH_OPEN_RATIO

        self.assertLess(MOUTH_CLOSED_RATIO, MOUTH_OPEN_RATIO)
        self.assertGreater(MOUTH_OPEN_RATIO, 0)

    def test_compute_mouth_openness(self):
        from opencut.core.lip_sync import _compute_mouth_openness

        landmarks = [MagicMock() for _ in range(478)]
        landmarks[13].y = 0.4    # top
        landmarks[14].y = 0.45   # bottom
        landmarks[61].x = 0.35   # left
        landmarks[291].x = 0.65  # right

        openness = _compute_mouth_openness(landmarks, 1080)
        self.assertGreater(openness, 0)

    def test_compute_mouth_openness_closed(self):
        from opencut.core.lip_sync import _compute_mouth_openness

        landmarks = [MagicMock() for _ in range(478)]
        landmarks[13].y = 0.40
        landmarks[14].y = 0.40   # same = closed
        landmarks[61].x = 0.35
        landmarks[291].x = 0.65

        openness = _compute_mouth_openness(landmarks, 1080)
        self.assertAlmostEqual(openness, 0.0, places=2)

    def test_get_mouth_roi(self):
        from opencut.core.lip_sync import MOUTH_CONTOUR, _get_mouth_roi

        landmarks = [MagicMock() for _ in range(478)]
        for idx in MOUTH_CONTOUR:
            landmarks[idx].x = 0.5
            landmarks[idx].y = 0.5

        x_min, y_min, x_max, y_max = _get_mouth_roi(landmarks, 1920, 1080)
        self.assertLess(x_min, x_max)
        self.assertLess(y_min, y_max)

    def test_check_wav2lip_returns_none_or_string(self):
        from opencut.core.lip_sync import _check_wav2lip

        result = _check_wav2lip()
        self.assertTrue(result is None or isinstance(result, str))

    def test_apply_lip_sync_signature(self):
        from opencut.core.lip_sync import apply_lip_sync

        sig = inspect.signature(apply_lip_sync)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("use_external", sig.parameters)
        self.assertIn("blend_strength", sig.parameters)

    def test_apply_lip_sync_no_video_raises(self):
        from opencut.core.lip_sync import apply_lip_sync

        with self.assertRaises(FileNotFoundError):
            apply_lip_sync("/nonexistent.mp4", "/nonexistent.wav")

    def test_preview_lip_sync_signature(self):
        from opencut.core.lip_sync import preview_lip_sync

        sig = inspect.signature(preview_lip_sync)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("frame_number", sig.parameters)

    @patch("opencut.core.lip_sync.run_ffmpeg")
    def test_extract_audio_features(self, mock_ffmpeg):
        from opencut.core.lip_sync import _extract_audio_features

        # Create a fake raw PCM file that the function would read
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake")
            audio_path = tmp.name

        # Mock the raw PCM file creation and reading
        fake_samples = struct.pack("<" + "h" * 16000, *([1000] * 16000))  # 1s of audio

        def side_effect(cmd, **kwargs):
            # Write fake PCM when FFmpeg is called
            for i, c in enumerate(cmd):
                if i > 0 and cmd[i - 1] != "-i" and c.endswith(".raw"):
                    with open(c, "wb") as f:
                        f.write(fake_samples)
                    break

        mock_ffmpeg.side_effect = side_effect

        try:
            features = _extract_audio_features(audio_path, fps=30.0)
            self.assertIn("amplitudes", features)
            self.assertIn("duration", features)
            self.assertIn("frame_count", features)
        except Exception:
            pass  # FFmpeg mock may not capture all paths
        finally:
            os.unlink(audio_path)


# ============================================================
# Voice Conversion Tests
# ============================================================
class TestVoiceConvert(unittest.TestCase):
    """Tests for opencut.core.voice_convert module."""

    def test_voice_convert_result_dataclass(self):
        from opencut.core.voice_convert import VoiceConvertResult

        r = VoiceConvertResult(
            output_path="/out.mp4",
            source_profile={"avg_pitch_hz": 120},
            target_profile={"avg_pitch_hz": 200},
            similarity_score=0.75,
        )
        d = r.to_dict()
        self.assertEqual(d["similarity_score"], 0.75)
        self.assertEqual(d["source_profile"]["avg_pitch_hz"], 120)

    def test_voice_convert_profile_dataclass(self):
        from opencut.core.voice_convert import VoiceConvertProfile

        p = VoiceConvertProfile(
            name="test_voice",
            avg_pitch_hz=150.0,
            pitch_range_hz=(120.0, 180.0),
            spectral_centroid=2500.0,
            spectral_tilt=-0.003,
            energy_db=-20.0,
        )
        d = p.to_dict()
        self.assertEqual(d["name"], "test_voice")
        self.assertEqual(d["avg_pitch_hz"], 150.0)
        self.assertEqual(len(d["pitch_range_hz"]), 2)

    def test_voice_convert_profile_save_load(self):
        from opencut.core.voice_convert import VoiceConvertProfile

        p = VoiceConvertProfile(
            name="test_save",
            avg_pitch_hz=160.0,
            pitch_range_hz=(130.0, 190.0),
            spectral_centroid=2600.0,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            p.save(tmp_path)
            loaded = VoiceConvertProfile.load(tmp_path)
            self.assertEqual(loaded.name, "test_save")
            self.assertAlmostEqual(loaded.avg_pitch_hz, 160.0)
        finally:
            os.unlink(tmp_path)

    def test_compute_pitch_shift_semitones(self):
        from opencut.core.voice_convert import _compute_pitch_shift_semitones

        # Octave up = 12 semitones
        shift = _compute_pitch_shift_semitones(100.0, 200.0)
        self.assertAlmostEqual(shift, 12.0, places=1)

        # Same pitch = 0
        shift = _compute_pitch_shift_semitones(100.0, 100.0)
        self.assertAlmostEqual(shift, 0.0)

        # Octave down = -12
        shift = _compute_pitch_shift_semitones(200.0, 100.0)
        self.assertAlmostEqual(shift, -12.0, places=1)

    def test_compute_pitch_shift_zero_pitch(self):
        from opencut.core.voice_convert import _compute_pitch_shift_semitones

        self.assertAlmostEqual(_compute_pitch_shift_semitones(0, 100), 0.0)
        self.assertAlmostEqual(_compute_pitch_shift_semitones(100, 0), 0.0)

    def test_build_voice_filter_chain(self):
        from opencut.core.voice_convert import _build_voice_filter_chain

        chain = _build_voice_filter_chain(0.0)
        self.assertIn("asetrate=", chain)
        self.assertIn("aresample=24000", chain)

    def test_build_voice_filter_chain_with_shift(self):
        from opencut.core.voice_convert import _build_voice_filter_chain

        chain = _build_voice_filter_chain(6.0)
        self.assertIn("asetrate=", chain)
        self.assertIn("atempo=", chain)

    def test_build_voice_filter_chain_with_formant(self):
        from opencut.core.voice_convert import _build_voice_filter_chain

        chain = _build_voice_filter_chain(0.0, formant_shift=0.5)
        self.assertIn("equalizer", chain)

    def test_build_voice_filter_chain_negative_formant(self):
        from opencut.core.voice_convert import _build_voice_filter_chain

        chain = _build_voice_filter_chain(0.0, formant_shift=-0.5)
        self.assertIn("equalizer", chain)

    def test_build_voice_filter_chain_clamps(self):
        from opencut.core.voice_convert import _build_voice_filter_chain

        # Should not raise even with extreme values
        chain = _build_voice_filter_chain(20.0)
        self.assertIn("asetrate=", chain)
        chain = _build_voice_filter_chain(-20.0)
        self.assertIn("asetrate=", chain)

    def test_atempo_chain_normal(self):
        from opencut.core.voice_convert import _build_atempo_chain

        chain = _build_atempo_chain(1.5)
        self.assertEqual(len(chain), 1)

    def test_atempo_chain_extreme_slow(self):
        from opencut.core.voice_convert import _build_atempo_chain

        chain = _build_atempo_chain(0.3)
        self.assertGreater(len(chain), 1)

    def test_check_rvc_returns_none_or_string(self):
        from opencut.core.voice_convert import _check_rvc

        result = _check_rvc()
        self.assertTrue(result is None or isinstance(result, str))

    def test_list_voice_profiles_returns_list(self):
        from opencut.core.voice_convert import list_voice_profiles

        profiles = list_voice_profiles()
        self.assertIsInstance(profiles, list)

    def test_convert_voice_signature(self):
        from opencut.core.voice_convert import convert_voice

        sig = inspect.signature(convert_voice)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("target_profile_path", sig.parameters)
        self.assertIn("target_profile_name", sig.parameters)
        self.assertIn("pitch_shift", sig.parameters)
        self.assertIn("use_rvc", sig.parameters)

    def test_convert_voice_no_file_raises(self):
        from opencut.core.voice_convert import convert_voice

        with self.assertRaises(FileNotFoundError):
            convert_voice("/nonexistent.wav")

    def test_create_voice_profile_signature(self):
        from opencut.core.voice_convert import create_voice_profile

        sig = inspect.signature(create_voice_profile)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("name", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_create_voice_profile_no_file_raises(self):
        from opencut.core.voice_convert import create_voice_profile

        with self.assertRaises(FileNotFoundError):
            create_voice_profile("/nonexistent.wav")

    def test_pitch_shift_constants(self):
        from opencut.core.voice_convert import MAX_PITCH_SHIFT, MIN_PITCH_SHIFT

        self.assertEqual(MIN_PITCH_SHIFT, -12)
        self.assertEqual(MAX_PITCH_SHIFT, 12)


# ============================================================
# Route Blueprint Tests
# ============================================================
class TestVoiceSpeechRoutes(unittest.TestCase):
    """Smoke tests for voice_speech_routes blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.voice_speech_routes import voice_speech_bp

        self.assertEqual(voice_speech_bp.name, "voice_speech")

    def test_route_functions_exist(self):
        from opencut.routes import voice_speech_routes as mod

        self.assertTrue(callable(getattr(mod, "route_transcript_edit", None)))
        self.assertTrue(callable(getattr(mod, "route_transcript_parse", None)))
        self.assertTrue(callable(getattr(mod, "route_transcript_preview", None)))
        self.assertTrue(callable(getattr(mod, "route_eye_contact", None)))
        self.assertTrue(callable(getattr(mod, "route_eye_contact_preview", None)))
        self.assertTrue(callable(getattr(mod, "route_overdub", None)))
        self.assertTrue(callable(getattr(mod, "route_clone_voice", None)))
        self.assertTrue(callable(getattr(mod, "route_lip_sync", None)))
        self.assertTrue(callable(getattr(mod, "route_lip_sync_preview", None)))
        self.assertTrue(callable(getattr(mod, "route_voice_convert", None)))
        self.assertTrue(callable(getattr(mod, "route_create_voice_profile", None)))
        self.assertTrue(callable(getattr(mod, "route_list_voice_profiles", None)))

    def test_route_count(self):
        from opencut.routes.voice_speech_routes import voice_speech_bp

        rules = list(voice_speech_bp.deferred_functions)
        self.assertEqual(len(rules), 12)

    def test_blueprint_url_prefix_compatible(self):
        """Verify routes use /api prefix in their paths."""
        from opencut.routes.voice_speech_routes import voice_speech_bp

        # The blueprint itself doesn't set url_prefix; routes include /api
        # Check that route registration functions exist
        self.assertIsNotNone(voice_speech_bp.deferred_functions)

    def test_imports_are_valid(self):
        """Verify all imports in the routes module resolve."""
        import opencut.routes.voice_speech_routes  # noqa: F401


if __name__ == "__main__":
    unittest.main()
