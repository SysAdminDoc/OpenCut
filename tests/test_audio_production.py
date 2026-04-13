"""
Tests for OpenCut Audio Production features.

Covers:
  - Audio Restoration Toolkit (declip, dehum, decrackle, dewind, dereverb)
  - Audio Fingerprinting & Copyright Detection
  - Room Tone Matching & Generation
  - M&E Mix Export
  - Automated Dialogue Premix
  - AI Show Notes & Transcript Summary
  - Audio Production Routes (smoke tests)
"""

import inspect
import json
import os
import struct
import sys
import tempfile
import unittest
from dataclasses import fields
from unittest.mock import MagicMock, mock_open, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Audio Restoration Toolkit Tests
# ============================================================
class TestAudioRestore(unittest.TestCase):
    """Tests for opencut.core.audio_restore module."""

    def test_declip_signature(self):
        from opencut.core.audio_restore import declip
        sig = inspect.signature(declip)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["output_path"].default)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_dehum_signature(self):
        from opencut.core.audio_restore import dehum
        sig = inspect.signature(dehum)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("frequency", sig.parameters)
        self.assertIn("harmonics", sig.parameters)
        self.assertEqual(sig.parameters["frequency"].default, 60.0)
        self.assertEqual(sig.parameters["harmonics"].default, 4)

    def test_decrackle_signature(self):
        from opencut.core.audio_restore import decrackle
        sig = inspect.signature(decrackle)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_dewind_signature(self):
        from opencut.core.audio_restore import dewind
        sig = inspect.signature(dewind)
        self.assertIn("cutoff_hz", sig.parameters)
        self.assertEqual(sig.parameters["cutoff_hz"].default, 80.0)

    def test_dereverb_signature(self):
        from opencut.core.audio_restore import dereverb
        sig = inspect.signature(dereverb)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_declip_calls_ffmpeg(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import declip
        result = declip("/fake/input.wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("adeclip", " ".join(cmd))
        self.assertEqual(result["filter_used"], "adeclip")
        self.assertIn("output_path", result)

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_dehum_builds_correct_filters(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import dehum
        result = dehum("/fake/input.wav", frequency=50.0, harmonics=3)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        af_str = " ".join(cmd)
        # Should have bandreject at 50, 100, 150 Hz
        self.assertIn("bandreject", af_str)
        self.assertEqual(result["frequencies_removed"], [50.0, 100.0, 150.0])

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_dehum_clamps_frequency(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import dehum
        result = dehum("/fake/input.wav", frequency=5.0, harmonics=1)
        # Should clamp to 20Hz minimum
        self.assertEqual(result["frequencies_removed"], [20.0])

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_decrackle_uses_afftdn(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import decrackle
        result = decrackle("/fake/input.wav")
        cmd = mock_run.call_args[0][0]
        af_str = " ".join(cmd)
        self.assertIn("afftdn", af_str)
        self.assertEqual(result["filter_used"], "afftdn")

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_dewind_uses_highpass(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import dewind
        result = dewind("/fake/input.wav", cutoff_hz=100.0)
        cmd = mock_run.call_args[0][0]
        af_str = " ".join(cmd)
        self.assertIn("highpass", af_str)
        self.assertIn("f=100.0", af_str)
        self.assertEqual(result["cutoff_hz"], 100.0)

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_dereverb_uses_gate_approach(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import dereverb
        dereverb("/fake/input.wav")
        cmd = mock_run.call_args[0][0]
        af_str = " ".join(cmd)
        self.assertIn("agate", af_str)
        self.assertIn("acompressor", af_str)

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_declip_respects_custom_output(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import declip
        result = declip("/fake/input.wav", output_path="/custom/out.wav")
        self.assertEqual(result["output_path"], "/custom/out.wav")

    @patch("opencut.core.audio_restore.run_ffmpeg")
    @patch("opencut.core.audio_restore.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_declip_fires_progress(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_restore import declip
        progress_calls = []
        declip("/fake/input.wav", on_progress=lambda p, m: progress_calls.append((p, m)))
        self.assertTrue(len(progress_calls) >= 2)
        self.assertEqual(progress_calls[-1][0], 100)


# ============================================================
# Audio Fingerprinting Tests
# ============================================================
class TestAudioFingerprint(unittest.TestCase):
    """Tests for opencut.core.audio_fingerprint module."""

    def test_dataclass_fields(self):
        from opencut.core.audio_fingerprint import AudioFingerprint, FingerprintMatch
        fp_fields = {f.name for f in fields(AudioFingerprint)}
        self.assertIn("hash_data", fp_fields)
        self.assertIn("duration", fp_fields)
        self.assertIn("sample_rate", fp_fields)

        match_fields = {f.name for f in fields(FingerprintMatch)}
        self.assertIn("match_score", match_fields)
        self.assertIn("matched_chunks", match_fields)
        self.assertIn("is_match", match_fields)

    def test_compare_identical_fingerprints(self):
        from opencut.core.audio_fingerprint import AudioFingerprint, compare_fingerprints
        fp = AudioFingerprint(hash_data=["aabb", "ccdd", "eeff", "1122"], duration=2.0)
        result = compare_fingerprints(fp, fp)
        self.assertEqual(result.match_score, 1.0)
        self.assertTrue(result.is_match)

    def test_compare_empty_fingerprints(self):
        from opencut.core.audio_fingerprint import AudioFingerprint, compare_fingerprints
        fp1 = AudioFingerprint(hash_data=[], duration=0.0)
        fp2 = AudioFingerprint(hash_data=[], duration=0.0)
        result = compare_fingerprints(fp1, fp2)
        self.assertEqual(result.match_score, 0.0)
        self.assertFalse(result.is_match)

    def test_compare_different_fingerprints(self):
        from opencut.core.audio_fingerprint import AudioFingerprint, compare_fingerprints
        fp1 = AudioFingerprint(hash_data=["aaaa", "bbbb", "cccc", "dddd"], duration=2.0)
        fp2 = AudioFingerprint(hash_data=["xxxx", "yyyy", "zzzz", "wwww"], duration=2.0)
        result = compare_fingerprints(fp1, fp2)
        self.assertLessEqual(result.match_score, 0.5)
        self.assertFalse(result.is_match)

    def test_compare_partial_match(self):
        from opencut.core.audio_fingerprint import AudioFingerprint, compare_fingerprints
        fp1 = AudioFingerprint(hash_data=["aaaa", "bbbb", "cccc", "dddd"], duration=2.0)
        fp2 = AudioFingerprint(hash_data=["aaaa", "bbbb", "xxxx", "yyyy"], duration=2.0)
        result = compare_fingerprints(fp1, fp2)
        self.assertEqual(result.match_score, 0.5)
        self.assertEqual(result.matched_chunks, 2)

    def test_compare_custom_threshold(self):
        from opencut.core.audio_fingerprint import AudioFingerprint, compare_fingerprints
        fp1 = AudioFingerprint(hash_data=["aaaa", "bbbb", "cccc"], duration=1.5)
        fp2 = AudioFingerprint(hash_data=["aaaa", "xxxx", "yyyy"], duration=1.5)
        result = compare_fingerprints(fp1, fp2, threshold=0.2)
        self.assertTrue(result.is_match)  # 1/3 >= 0.2

    def test_hash_chunk_deterministic(self):
        from opencut.core.audio_fingerprint import _hash_chunk
        bins = [100, 200, 50, 300, 150, 250, 75, 175]
        h1 = _hash_chunk(bins)
        h2 = _hash_chunk(bins)
        self.assertEqual(h1, h2)

    def test_hash_chunk_empty(self):
        from opencut.core.audio_fingerprint import _hash_chunk
        result = _hash_chunk([])
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_compute_spectral_bins(self):
        from opencut.core.audio_fingerprint import _NUM_BINS, _compute_spectral_bins
        samples = list(range(1000))
        bins = _compute_spectral_bins(samples)
        self.assertEqual(len(bins), _NUM_BINS)
        # All bins should be non-negative
        for b in bins:
            self.assertGreaterEqual(b, 0)

    def test_compute_spectral_bins_empty(self):
        from opencut.core.audio_fingerprint import _NUM_BINS, _compute_spectral_bins
        bins = _compute_spectral_bins([])
        self.assertEqual(len(bins), _NUM_BINS)
        self.assertTrue(all(b == 0 for b in bins))

    @patch("opencut.core.audio_fingerprint.run_ffmpeg")
    @patch("opencut.core.audio_fingerprint.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_generate_fingerprint_calls_ffmpeg(self, mock_ffmpeg_path, mock_run):
        from opencut.core.audio_fingerprint import generate_fingerprint
        # Create fake PCM data (100 samples of 16-bit audio)
        pcm = struct.pack("<100h", *([1000] * 100))
        with patch("builtins.open", mock_open(read_data=pcm)):
            with patch("os.path.exists", return_value=True):
                with patch("os.unlink"):
                    fp = generate_fingerprint("/fake/input.wav")
        self.assertIsNotNone(fp.hash_data)
        self.assertGreater(fp.sample_rate, 0)

    def test_load_database_missing_file(self):
        from opencut.core.audio_fingerprint import _load_database
        db = _load_database("/nonexistent/path/db.json")
        self.assertEqual(db, {"tracks": []})

    def test_load_database_valid(self):
        from opencut.core.audio_fingerprint import _load_database
        data = json.dumps({"tracks": [{"label": "test", "hash_data": ["abc"]}]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(data)
            f.flush()
            db = _load_database(f.name)
        os.unlink(f.name)
        self.assertEqual(len(db["tracks"]), 1)
        self.assertEqual(db["tracks"][0]["label"], "test")

    def test_save_and_load_database(self):
        from opencut.core.audio_fingerprint import _load_database, _save_database
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "test_db.json")
            db = {"tracks": [{"label": "Song A", "hash_data": ["a1", "a2"]}]}
            _save_database(db, db_path)
            loaded = _load_database(db_path)
            self.assertEqual(loaded["tracks"][0]["label"], "Song A")


# ============================================================
# Room Tone Tests
# ============================================================
class TestRoomTone(unittest.TestCase):
    """Tests for opencut.core.room_tone module."""

    def test_extract_room_tone_signature(self):
        from opencut.core.room_tone import extract_room_tone
        sig = inspect.signature(extract_room_tone)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("duration", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertEqual(sig.parameters["duration"].default, 5.0)

    def test_generate_room_tone_signature(self):
        from opencut.core.room_tone import generate_room_tone
        sig = inspect.signature(generate_room_tone)
        self.assertIn("reference_path", sig.parameters)
        self.assertIn("duration", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertEqual(sig.parameters["duration"].default, 30.0)

    def test_fill_gaps_with_tone_signature(self):
        from opencut.core.room_tone import fill_gaps_with_tone
        sig = inspect.signature(fill_gaps_with_tone)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("tone_path", sig.parameters)
        self.assertIn("gap_threshold_db", sig.parameters)
        self.assertEqual(sig.parameters["gap_threshold_db"].default, -50.0)

    def test_detect_silence_segments_parsing(self):
        """Test the silence detection output parsing logic."""
        from opencut.core.room_tone import _detect_silence_segments
        stderr_output = (
            "[silencedetect @ 0x1234] silence_start: 1.5\n"
            "[silencedetect @ 0x1234] silence_end: 3.2 | silence_duration: 1.7\n"
            "[silencedetect @ 0x1234] silence_start: 5.0\n"
            "[silencedetect @ 0x1234] silence_end: 6.5 | silence_duration: 1.5\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr_output

        with patch("opencut.core.room_tone.get_ffmpeg_path", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run", return_value=mock_result):
                segments = _detect_silence_segments("/fake/input.wav")

        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0]["start"], 1.5)
        self.assertAlmostEqual(segments[0]["end"], 3.2)
        self.assertAlmostEqual(segments[1]["start"], 5.0)

    @patch("opencut.core.room_tone.run_ffmpeg")
    @patch("opencut.core.room_tone.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.room_tone._get_duration", return_value=60.0)
    @patch("opencut.core.room_tone._detect_silence_segments", return_value=[])
    def test_fill_gaps_no_gaps(self, mock_detect, mock_dur, mock_ffmpeg, mock_run):
        from opencut.core.room_tone import fill_gaps_with_tone
        result = fill_gaps_with_tone("/fake/input.wav", "/fake/tone.wav")
        self.assertEqual(result["gaps_filled"], 0)
        self.assertEqual(result["total_gap_duration"], 0.0)

    @patch("opencut.core.room_tone.run_ffmpeg")
    @patch("opencut.core.room_tone.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.room_tone._find_quietest_segments", return_value=[(10.0, -45.0)])
    def test_extract_room_tone_result(self, mock_quiet, mock_ffmpeg, mock_run):
        from opencut.core.room_tone import extract_room_tone
        result = extract_room_tone("/fake/input.wav", duration=3.0)
        self.assertIn("output_path", result)
        self.assertIn("tone_profile", result)
        self.assertEqual(result["tone_profile"]["rms_level"], -45.0)
        self.assertEqual(result["tone_profile"]["duration"], 3.0)

    @patch("opencut.core.room_tone.run_ffmpeg")
    @patch("opencut.core.room_tone.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.room_tone._find_quietest_segments", return_value=[])
    def test_extract_room_tone_no_segments(self, mock_quiet, mock_ffmpeg, mock_run):
        from opencut.core.room_tone import extract_room_tone
        with self.assertRaises(RuntimeError) as ctx:
            extract_room_tone("/fake/input.wav")
        self.assertIn("No suitable room tone", str(ctx.exception))


# ============================================================
# M&E Mix Tests
# ============================================================
class TestMEMix(unittest.TestCase):
    """Tests for opencut.core.me_mix module."""

    def test_generate_me_mix_signature(self):
        from opencut.core.me_mix import generate_me_mix
        sig = inspect.signature(generate_me_mix)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("output_path", sig.parameters)
        self.assertIn("method", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertEqual(sig.parameters["method"].default, "subtract")

    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_center_subtract_method(self, mock_ffmpeg_path, mock_run):
        from opencut.core.me_mix import generate_me_mix
        result = generate_me_mix("/fake/input.wav", method="subtract")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        af_str = " ".join(cmd)
        self.assertIn("pan=stereo", af_str)
        self.assertIn("c0=c0-c1", af_str)
        self.assertIn("output_path", result)
        self.assertIn("subtract", result["method_used"])

    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_stems_fallback_to_subtract(self, mock_ffmpeg_path, mock_run):
        from opencut.core.me_mix import generate_me_mix
        # stems method should fall back when stem separation is unavailable
        result = generate_me_mix("/fake/input.wav", method="stems")
        self.assertIn("subtract", result["method_used"])
        self.assertIn("not available", result["notes"].lower())

    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_invalid_method_defaults_to_subtract(self, mock_ffmpeg_path, mock_run):
        from opencut.core.me_mix import generate_me_mix
        result = generate_me_mix("/fake/input.wav", method="invalid")
        self.assertIn("subtract", result["method_used"])

    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_me_mix_custom_output(self, mock_ffmpeg_path, mock_run):
        from opencut.core.me_mix import generate_me_mix
        result = generate_me_mix("/fake/input.wav", output_path="/custom/me.wav")
        self.assertEqual(result["output_path"], "/custom/me.wav")

    @patch("opencut.core.me_mix.run_ffmpeg")
    @patch("opencut.core.me_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_me_mix_progress_callback(self, mock_ffmpeg_path, mock_run):
        from opencut.core.me_mix import generate_me_mix
        progress = []
        generate_me_mix("/fake/input.wav", on_progress=lambda p, m: progress.append(p))
        self.assertTrue(len(progress) >= 2)
        self.assertEqual(progress[-1], 100)


# ============================================================
# Dialogue Premix Tests
# ============================================================
class TestDialoguePremix(unittest.TestCase):
    """Tests for opencut.core.dialogue_premix module."""

    def test_premix_dialogue_signature(self):
        from opencut.core.dialogue_premix import premix_dialogue
        sig = inspect.signature(premix_dialogue)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("target_lufs", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertEqual(sig.parameters["target_lufs"].default, -23.0)

    def test_premix_multi_speaker_signature(self):
        from opencut.core.dialogue_premix import premix_multi_speaker
        sig = inspect.signature(premix_multi_speaker)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("diarization_segments", sig.parameters)
        self.assertIn("target_lufs", sig.parameters)

    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_dialogue_chain(self, mock_ffmpeg_path, mock_run):
        from opencut.core.dialogue_premix import premix_dialogue
        result = premix_dialogue("/fake/input.wav", target_lufs=-23.0)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        af_str = " ".join(cmd)
        # Should contain de-ess, EQ, compressor, loudnorm
        self.assertIn("equalizer", af_str)
        self.assertIn("highpass", af_str)
        self.assertIn("acompressor", af_str)
        self.assertIn("loudnorm", af_str)
        self.assertIn("I=-23.0", af_str)
        self.assertEqual(result["target_lufs"], -23.0)
        self.assertIn("processing_chain", result)
        self.assertEqual(len(result["processing_chain"]), 4)

    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_dialogue_clamps_lufs(self, mock_ffmpeg_path, mock_run):
        from opencut.core.dialogue_premix import premix_dialogue
        result = premix_dialogue("/fake/input.wav", target_lufs=-50.0)
        self.assertEqual(result["target_lufs"], -36.0)

    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_multi_no_segments(self, mock_ffmpeg_path, mock_run):
        from opencut.core.dialogue_premix import premix_multi_speaker
        result = premix_multi_speaker("/fake/input.wav")
        self.assertEqual(result.get("segments_processed", -1), 0)
        # Should have fallen back to single premix
        self.assertIn("output_path", result)

    @patch("opencut.core.dialogue_premix.run_ffmpeg")
    @patch("opencut.core.dialogue_premix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_premix_progress_callback(self, mock_ffmpeg_path, mock_run):
        from opencut.core.dialogue_premix import premix_dialogue
        progress = []
        premix_dialogue("/fake/input.wav", on_progress=lambda p, m: progress.append(p))
        self.assertTrue(len(progress) >= 4)
        self.assertEqual(progress[-1], 100)


# ============================================================
# Show Notes Tests
# ============================================================
class TestShowNotes(unittest.TestCase):
    """Tests for opencut.core.show_notes module."""

    def test_show_notes_dataclass(self):
        from opencut.core.show_notes import ShowNotes
        notes = ShowNotes()
        self.assertEqual(notes.summary, "")
        self.assertEqual(notes.key_topics, [])
        self.assertEqual(notes.quotes, [])
        self.assertEqual(notes.chapter_markers, [])
        self.assertEqual(notes.resources_mentioned, [])

    def test_generate_show_notes_empty_input(self):
        from opencut.core.show_notes import generate_show_notes
        notes = generate_show_notes("")
        self.assertIn("No transcript", notes.summary)

    def test_fallback_show_notes(self):
        from opencut.core.show_notes import _fallback_show_notes
        transcript = (
            "Today we discuss machine learning and artificial intelligence. "
            "Machine learning is used in many applications. "
            "Check out https://example.com for more information. "
            "Artificial intelligence will change the world. "
            "Machine learning models can be trained on large datasets. "
            "This concludes our discussion on machine learning."
        )
        notes = _fallback_show_notes(transcript)
        self.assertTrue(len(notes.summary) > 0)
        self.assertTrue(len(notes.key_topics) > 0)
        self.assertIn("https://example.com", notes.resources_mentioned)

    def test_fallback_extracts_urls(self):
        from opencut.core.show_notes import _fallback_show_notes
        text = "Visit https://example.com and http://test.org for resources."
        notes = _fallback_show_notes(text)
        self.assertTrue(len(notes.resources_mentioned) >= 2)

    def test_parse_llm_response(self):
        from opencut.core.show_notes import _parse_llm_response
        response = """## Summary
This is a test summary about technology.

## Key Topics
- [00:01:00] Introduction to AI
- [00:05:30] Machine learning basics

## Notable Quotes
- "AI will transform everything"
- "Data is the new oil"

## Chapter Markers
- [00:00:00] Introduction
- [00:05:00] Main content

## Resources Mentioned
- https://example.com
- Book: Deep Learning by Goodfellow"""

        notes = _parse_llm_response(response)
        self.assertIn("technology", notes.summary)
        self.assertEqual(len(notes.key_topics), 2)
        self.assertEqual(notes.key_topics[0]["timestamp"], "00:01:00")
        self.assertEqual(len(notes.quotes), 2)
        self.assertEqual(len(notes.chapter_markers), 2)
        self.assertEqual(len(notes.resources_mentioned), 2)

    def test_parse_llm_response_no_timestamps(self):
        from opencut.core.show_notes import _parse_llm_response
        response = """## Summary
A short summary.

## Key Topics
- Topic without timestamp

## Notable Quotes
None

## Chapter Markers
None

## Resources Mentioned
None"""
        notes = _parse_llm_response(response)
        self.assertEqual(len(notes.key_topics), 1)
        self.assertEqual(notes.key_topics[0]["timestamp"], "")
        self.assertEqual(len(notes.quotes), 0)

    def test_export_markdown(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(
            summary="Test summary.",
            key_topics=[{"timestamp": "00:01:00", "topic": "Topic 1"}],
            quotes=["Great quote"],
            chapter_markers=[{"timestamp": "00:00:00", "title": "Intro"}],
            resources_mentioned=["https://example.com"],
        )
        md = export_show_notes(notes, format="markdown")
        self.assertIn("# Show Notes", md)
        self.assertIn("Test summary.", md)
        self.assertIn("[00:01:00]", md)
        self.assertIn("Great quote", md)

    def test_export_html(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(
            summary="Test summary.",
            key_topics=[{"timestamp": "", "topic": "Topic 1"}],
        )
        html = export_show_notes(notes, format="html")
        self.assertIn("<h1>Show Notes</h1>", html)
        self.assertIn("Test summary.", html)
        self.assertIn("<li>", html)

    def test_export_text(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(summary="Test summary.")
        txt = export_show_notes(notes, format="text")
        self.assertIn("SHOW NOTES", txt)
        self.assertIn("SUMMARY", txt)
        self.assertIn("Test summary.", txt)

    def test_export_invalid_format_defaults_to_markdown(self):
        from opencut.core.show_notes import ShowNotes, export_show_notes
        notes = ShowNotes(summary="Test.")
        result = export_show_notes(notes, format="xml")
        self.assertIn("# Show Notes", result)  # markdown

    def test_generate_show_notes_fallback_when_no_llm(self):
        """generate_show_notes should use fallback when LLM is unavailable."""
        from opencut.core.show_notes import generate_show_notes
        transcript = "This is a test transcript about programming and software development. " * 10
        with patch.dict("sys.modules", {"opencut.core.llm": None}):
            # Force ImportError on LLM import
            notes = generate_show_notes(transcript)
        # Should still produce results via fallback
        self.assertTrue(len(notes.summary) > 0)

    def test_html_escape(self):
        from opencut.core.show_notes import _html_escape
        self.assertEqual(_html_escape("<script>"), "&lt;script&gt;")
        self.assertEqual(_html_escape('a&b"c'), "a&amp;b&quot;c")


# ============================================================
# Route Smoke Tests
# ============================================================
class TestAudioProductionRoutes(unittest.TestCase):
    """Smoke tests for audio production route registration."""

    def test_blueprint_exists(self):
        from opencut.routes.audio_production_routes import audio_prod_bp
        self.assertEqual(audio_prod_bp.name, "audio_prod")

    def test_blueprint_has_deferred_functions(self):
        from opencut.routes.audio_production_routes import audio_prod_bp
        # Blueprint should have deferred function registrations
        self.assertIsNotNone(audio_prod_bp.deferred_functions)
        self.assertGreater(len(audio_prod_bp.deferred_functions), 0)

    def test_route_functions_importable(self):
        from opencut.routes.audio_production_routes import (
            route_declip,
            route_decrackle,
            route_dehum,
            route_dereverb,
            route_dewind,
            route_dialogue_premix,
            route_fingerprint,
            route_fingerprint_add,
            route_fingerprint_scan,
            route_me_mix,
            route_room_tone_extract,
            route_room_tone_fill,
            route_room_tone_generate,
            route_show_notes,
        )
        # All route functions should be callable
        self.assertTrue(callable(route_declip))
        self.assertTrue(callable(route_dehum))
        self.assertTrue(callable(route_decrackle))
        self.assertTrue(callable(route_dewind))
        self.assertTrue(callable(route_dereverb))
        self.assertTrue(callable(route_fingerprint))
        self.assertTrue(callable(route_fingerprint_scan))
        self.assertTrue(callable(route_fingerprint_add))
        self.assertTrue(callable(route_room_tone_extract))
        self.assertTrue(callable(route_room_tone_generate))
        self.assertTrue(callable(route_room_tone_fill))
        self.assertTrue(callable(route_me_mix))
        self.assertTrue(callable(route_dialogue_premix))
        self.assertTrue(callable(route_show_notes))

    def test_blueprint_registered_in_init(self):
        """Verify audio_prod_bp is imported in routes/__init__.py."""
        import opencut.routes as routes_mod
        source_file = inspect.getfile(routes_mod)
        with open(source_file, "r") as f:
            content = f.read()
        self.assertIn("audio_production_routes", content)
        self.assertIn("audio_prod_bp", content)


# ============================================================
# Integration-style tests (all mocked)
# ============================================================
class TestCrossModuleIntegration(unittest.TestCase):
    """Verify cross-module interactions work correctly."""

    def test_all_core_modules_importable(self):
        """All new core modules should import without error."""
        from opencut.core import audio_fingerprint, audio_restore, dialogue_premix, me_mix, room_tone, show_notes
        self.assertIsNotNone(audio_restore)
        self.assertIsNotNone(audio_fingerprint)
        self.assertIsNotNone(room_tone)
        self.assertIsNotNone(me_mix)
        self.assertIsNotNone(dialogue_premix)
        self.assertIsNotNone(show_notes)

    def test_all_functions_have_docstrings(self):
        """All public functions should have docstrings."""
        from opencut.core.audio_fingerprint import (
            add_to_database,
            compare_fingerprints,
            generate_fingerprint,
            scan_against_database,
        )
        from opencut.core.audio_restore import declip, decrackle, dehum, dereverb, dewind
        from opencut.core.dialogue_premix import premix_dialogue, premix_multi_speaker
        from opencut.core.me_mix import generate_me_mix
        from opencut.core.room_tone import extract_room_tone, fill_gaps_with_tone, generate_room_tone
        from opencut.core.show_notes import export_show_notes, generate_show_notes

        functions = [
            declip, dehum, decrackle, dewind, dereverb,
            generate_fingerprint, compare_fingerprints, scan_against_database, add_to_database,
            extract_room_tone, generate_room_tone, fill_gaps_with_tone,
            generate_me_mix,
            premix_dialogue, premix_multi_speaker,
            generate_show_notes, export_show_notes,
        ]
        for fn in functions:
            self.assertIsNotNone(fn.__doc__, f"{fn.__name__} missing docstring")

    def test_fingerprint_roundtrip(self):
        """Test fingerprint -> compare -> match roundtrip."""
        from opencut.core.audio_fingerprint import AudioFingerprint, compare_fingerprints
        fp1 = AudioFingerprint(hash_data=["aa", "bb", "cc", "dd", "ee"], duration=2.5)
        fp2 = AudioFingerprint(hash_data=["aa", "bb", "cc", "dd", "ee"], duration=2.5)
        match = compare_fingerprints(fp1, fp2)
        self.assertTrue(match.is_match)
        self.assertEqual(match.match_score, 1.0)

    def test_show_notes_full_pipeline(self):
        """Test generate -> export pipeline."""
        from opencut.core.show_notes import export_show_notes, generate_show_notes
        transcript = (
            "Welcome to the podcast about technology. "
            "Today we cover three topics: cloud computing, AI, and security. "
            "Cloud computing has revolutionized how businesses operate. "
            "Visit https://cloud.example.com for more details. "
            "Artificial intelligence is changing healthcare and finance. "
            "Security remains the biggest concern for enterprises. "
            "In conclusion, technology continues to evolve rapidly."
        )
        # Use fallback (no LLM)
        with patch.dict("sys.modules", {"opencut.core.llm": None}):
            notes = generate_show_notes(transcript)

        for fmt in ("markdown", "html", "text"):
            output = export_show_notes(notes, format=fmt)
            self.assertTrue(len(output) > 0, f"Empty output for format {fmt}")


if __name__ == "__main__":
    unittest.main()
