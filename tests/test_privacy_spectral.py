"""
Tests for OpenCut Privacy & Spectral features.

Covers:
  - License Plate Detection & Blur (plate_blur.py)
  - OCR-Based PII Redaction (pii_redact.py)
  - Profanity Bleep Automation (profanity_bleep.py)
  - Document & Screen Redaction (doc_redact.py)
  - Audio Redaction & Speaker Anonymization (audio_anon.py)
  - Visual Spectrogram Editor (spectrogram_edit.py)
  - Spectral Repair / Frequency Removal (spectral_repair.py)
  - AI Environmental Noise Classifier (noise_classify.py)
  - Room Tone Auto-Generation enhancements (room_tone.py)
  - Privacy & Spectral Routes (privacy_spectral_routes.py)
"""

import inspect
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# License Plate Detection & Blur Tests
# ============================================================
class TestPlateBlur(unittest.TestCase):
    """Tests for opencut.core.plate_blur module."""

    def test_detect_plates_signature(self):
        from opencut.core.plate_blur import detect_plates
        sig = inspect.signature(detect_plates)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("sample_fps", sig.parameters)
        self.assertIn("iou_threshold", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_blur_plates_signature(self):
        from opencut.core.plate_blur import blur_plates
        sig = inspect.signature(blur_plates)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("detections", sig.parameters)
        self.assertIn("blur_strength", sig.parameters)
        self.assertIn("output_path_str", sig.parameters)

    def test_iou_perfect_overlap(self):
        from opencut.core.plate_blur import _iou
        a = {"x": 10, "y": 10, "w": 50, "h": 30}
        self.assertAlmostEqual(_iou(a, a), 1.0)

    def test_iou_no_overlap(self):
        from opencut.core.plate_blur import _iou
        a = {"x": 0, "y": 0, "w": 10, "h": 10}
        b = {"x": 100, "y": 100, "w": 10, "h": 10}
        self.assertAlmostEqual(_iou(a, b), 0.0)

    def test_iou_partial_overlap(self):
        from opencut.core.plate_blur import _iou
        a = {"x": 0, "y": 0, "w": 20, "h": 20}
        b = {"x": 10, "y": 10, "w": 20, "h": 20}
        iou_val = _iou(a, b)
        self.assertGreater(iou_val, 0.0)
        self.assertLess(iou_val, 1.0)

    def test_track_detections_assigns_ids(self):
        from opencut.core.plate_blur import _track_detections, PlateDetection
        dets = [
            PlateDetection(x=10, y=10, w=50, h=30, frame=0),
            PlateDetection(x=12, y=11, w=50, h=30, frame=1),  # overlaps with first
            PlateDetection(x=200, y=200, w=50, h=30, frame=1),  # new track
        ]
        tracked = _track_detections(dets, iou_threshold=0.3)
        self.assertEqual(tracked[0].track_id, tracked[1].track_id)
        self.assertNotEqual(tracked[0].track_id, tracked[2].track_id)

    def test_plate_detection_dataclass(self):
        from opencut.core.plate_blur import PlateDetection
        d = PlateDetection(x=10, y=20, w=100, h=30, confidence=0.9, text="ABC123")
        self.assertEqual(d.x, 10)
        self.assertEqual(d.text, "ABC123")
        self.assertEqual(d.track_id, -1)

    @patch("opencut.core.plate_blur.run_ffmpeg")
    @patch("opencut.core.plate_blur.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.plate_blur.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 10.0, "fps": 25})
    def test_blur_plates_no_detections_copies(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.plate_blur import blur_plates
        result = blur_plates("/fake/video.mp4", detections=[])
        self.assertEqual(result["plates_found"], 0)
        mock_run.assert_called_once()

    @patch("opencut.core.plate_blur.run_ffmpeg")
    @patch("opencut.core.plate_blur.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.plate_blur.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 10.0, "fps": 25})
    def test_blur_plates_with_detections(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.plate_blur import blur_plates
        dets = [{"x": 100, "y": 200, "w": 150, "h": 40, "timestamp": 1.0, "track_id": 0}]
        with patch("builtins.open", mock_open()):
            result = blur_plates("/fake/video.mp4", detections=dets)
        self.assertEqual(result["plates_found"], 1)
        self.assertIn("output_path", result)


# ============================================================
# OCR-Based PII Redaction Tests
# ============================================================
class TestPIIRedact(unittest.TestCase):
    """Tests for opencut.core.pii_redact module."""

    def test_detect_pii_signature(self):
        from opencut.core.pii_redact import detect_pii
        sig = inspect.signature(detect_pii)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("sample_fps", sig.parameters)
        self.assertIn("pii_types", sig.parameters)

    def test_redact_pii_signature(self):
        from opencut.core.pii_redact import redact_pii
        sig = inspect.signature(redact_pii)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("detections", sig.parameters)
        self.assertIn("blur_strength", sig.parameters)

    def test_pii_patterns_ssn(self):
        from opencut.core.pii_redact import PII_PATTERNS
        self.assertIn("ssn", PII_PATTERNS)
        self.assertTrue(PII_PATTERNS["ssn"].search("123-45-6789"))
        self.assertIsNone(PII_PATTERNS["ssn"].search("12-345-6789"))

    def test_pii_patterns_email(self):
        from opencut.core.pii_redact import PII_PATTERNS
        self.assertTrue(PII_PATTERNS["email"].search("user@example.com"))
        self.assertIsNone(PII_PATTERNS["email"].search("not an email"))

    def test_pii_patterns_phone(self):
        from opencut.core.pii_redact import PII_PATTERNS
        self.assertTrue(PII_PATTERNS["phone"].search("(555) 123-4567"))
        self.assertTrue(PII_PATTERNS["phone"].search("555-123-4567"))

    def test_pii_patterns_credit_card(self):
        from opencut.core.pii_redact import PII_PATTERNS
        self.assertTrue(PII_PATTERNS["credit_card"].search("4111-1111-1111-1111"))
        self.assertTrue(PII_PATTERNS["credit_card"].search("4111111111111111"))

    def test_pii_detection_dataclass(self):
        from opencut.core.pii_redact import PIIDetection
        d = PIIDetection(x=10, y=20, w=100, h=15, pii_type="email", matched_text="a@b.com")
        self.assertEqual(d.pii_type, "email")
        self.assertEqual(d.matched_text, "a@b.com")

    def test_scan_for_pii_finds_ssn(self):
        from opencut.core.pii_redact import _scan_for_pii
        ocr = [{"text": "SSN: 123-45-6789", "x": 10, "y": 10, "w": 100, "h": 15, "conf": 90}]
        results = _scan_for_pii(ocr, frame_idx=0, timestamp=1.0, pii_types=["ssn"])
        self.assertTrue(any(d.pii_type == "ssn" for d in results))

    def test_scan_for_pii_filters_types(self):
        from opencut.core.pii_redact import _scan_for_pii
        ocr = [{"text": "user@example.com 123-45-6789", "x": 10, "y": 10, "w": 200, "h": 15, "conf": 90}]
        results = _scan_for_pii(ocr, frame_idx=0, timestamp=0, pii_types=["email"])
        types = set(d.pii_type for d in results)
        self.assertIn("email", types)
        self.assertNotIn("ssn", types)

    @patch("opencut.core.pii_redact.run_ffmpeg")
    @patch("opencut.core.pii_redact.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.pii_redact.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 10.0})
    def test_redact_pii_no_detections(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.pii_redact import redact_pii
        result = redact_pii("/fake/video.mp4", detections=[])
        self.assertEqual(result["pii_found"], 0)


# ============================================================
# Profanity Bleep Automation Tests
# ============================================================
class TestProfanityBleep(unittest.TestCase):
    """Tests for opencut.core.profanity_bleep module."""

    def test_find_profanity_signature(self):
        from opencut.core.profanity_bleep import find_profanity
        sig = inspect.signature(find_profanity)
        self.assertIn("transcript_path", sig.parameters)
        self.assertIn("custom_words", sig.parameters)

    def test_bleep_profanity_signature(self):
        from opencut.core.profanity_bleep import bleep_profanity
        sig = inspect.signature(bleep_profanity)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("transcript_path", sig.parameters)
        self.assertIn("bleep_frequency", sig.parameters)
        self.assertIn("bleep_volume", sig.parameters)
        self.assertIn("mouth_blur", sig.parameters)

    def test_load_profanity_list_returns_list(self):
        from opencut.core.profanity_bleep import _load_profanity_list
        words = _load_profanity_list()
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)

    def test_generate_bleep_tone(self):
        from opencut.core.profanity_bleep import _generate_bleep_tone
        path = _generate_bleep_tone(0.5, frequency=1000.0)
        try:
            self.assertTrue(os.path.isfile(path))
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getframerate(), 48000)
                self.assertGreater(wf.getnframes(), 0)
        finally:
            os.unlink(path)

    def test_parse_transcript_srt(self):
        from opencut.core.profanity_bleep import _parse_transcript
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:04,000 --> 00:00:06,000\nGoodbye\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt)
            f.flush()
            path = f.name
        try:
            words = _parse_transcript(path)
            self.assertGreater(len(words), 0)
            self.assertIn("start", words[0])
            self.assertIn("end", words[0])
        finally:
            os.unlink(path)

    def test_parse_transcript_json(self):
        from opencut.core.profanity_bleep import _parse_transcript
        data = [{"word": "hello", "start": 0.0, "end": 0.5}, {"word": "world", "start": 0.5, "end": 1.0}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            words = _parse_transcript(path)
            self.assertEqual(len(words), 2)
            self.assertEqual(words[0]["word"], "hello")
        finally:
            os.unlink(path)

    def test_find_profanity_detects_words(self):
        from opencut.core.profanity_bleep import find_profanity
        data = [{"word": "damn", "start": 1.0, "end": 1.5}, {"word": "hello", "start": 2.0, "end": 2.5}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            result = find_profanity(path)
            self.assertEqual(result["total_found"], 1)
            self.assertIn("damn", result["unique_words"])
        finally:
            os.unlink(path)

    def test_find_profanity_custom_words(self):
        from opencut.core.profanity_bleep import find_profanity
        data = [{"word": "foobar", "start": 0.0, "end": 0.5}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            result = find_profanity(path, custom_words=["foobar"])
            self.assertEqual(result["total_found"], 1)
        finally:
            os.unlink(path)

    @patch("opencut.core.profanity_bleep.run_ffmpeg")
    @patch("opencut.core.profanity_bleep.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.profanity_bleep.get_video_info", return_value={"duration": 10.0, "width": 1920, "height": 1080})
    def test_bleep_profanity_no_matches(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.profanity_bleep import bleep_profanity
        data = [{"word": "hello", "start": 0.0, "end": 0.5}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            result = bleep_profanity("/fake/audio.wav", transcript_path=path)
            self.assertEqual(result["bleeps_applied"], 0)
        finally:
            os.unlink(path)


# ============================================================
# Document & Screen Redaction Tests
# ============================================================
class TestDocRedact(unittest.TestCase):
    """Tests for opencut.core.doc_redact module."""

    def test_detect_surfaces_signature(self):
        from opencut.core.doc_redact import detect_surfaces
        sig = inspect.signature(detect_surfaces)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("sample_fps", sig.parameters)
        self.assertIn("min_area_ratio", sig.parameters)

    def test_redact_surfaces_signature(self):
        from opencut.core.doc_redact import redact_surfaces
        sig = inspect.signature(redact_surfaces)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("surface_types", sig.parameters)
        self.assertIn("redaction_mode", sig.parameters)
        self.assertIn("blur_strength", sig.parameters)

    def test_surface_detection_dataclass(self):
        from opencut.core.doc_redact import SurfaceDetection
        d = SurfaceDetection(x=0, y=0, w=400, h=300, surface_type="screen", confidence=0.8)
        self.assertEqual(d.surface_type, "screen")
        self.assertEqual(d.confidence, 0.8)

    def test_classify_surface_screen(self):
        from opencut.core.doc_redact import _classify_surface
        result = _classify_surface(
            {"aspect": 1.6, "area_ratio": 0.05, "mean_brightness": 200.0, "edge_density": 0.05},
            frame_brightness=120.0,
        )
        self.assertEqual(result, "screen")

    def test_classify_surface_document(self):
        from opencut.core.doc_redact import _classify_surface
        result = _classify_surface(
            {"aspect": 0.7, "area_ratio": 0.03, "mean_brightness": 220.0, "edge_density": 0.15},
            frame_brightness=120.0,
        )
        self.assertEqual(result, "document")

    def test_classify_surface_whiteboard(self):
        from opencut.core.doc_redact import _classify_surface
        # Whiteboard: large, very bright, low edges, aspect not in screen range
        result = _classify_surface(
            {"aspect": 2.5, "area_ratio": 0.15, "mean_brightness": 230.0, "edge_density": 0.03},
            frame_brightness=120.0,
        )
        self.assertEqual(result, "whiteboard")

    def test_classify_surface_unknown(self):
        from opencut.core.doc_redact import _classify_surface
        result = _classify_surface(
            {"aspect": 5.0, "area_ratio": 0.001, "mean_brightness": 50.0, "edge_density": 0.01},
            frame_brightness=120.0,
        )
        self.assertEqual(result, "unknown")

    @patch("opencut.core.doc_redact.run_ffmpeg")
    @patch("opencut.core.doc_redact.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.doc_redact.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 10.0})
    def test_redact_surfaces_no_detections(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.doc_redact import redact_surfaces
        result = redact_surfaces("/fake/video.mp4", detections=[])
        self.assertEqual(result["surfaces_found"], 0)

    @patch("opencut.core.doc_redact.run_ffmpeg")
    @patch("opencut.core.doc_redact.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.doc_redact.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 10.0})
    def test_redact_surfaces_filters_by_type(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.doc_redact import redact_surfaces
        dets = [
            {"x": 0, "y": 0, "w": 400, "h": 300, "timestamp": 1.0, "surface_type": "screen"},
            {"x": 500, "y": 100, "w": 200, "h": 300, "timestamp": 1.0, "surface_type": "document"},
        ]
        result = redact_surfaces("/fake/video.mp4", detections=dets, surface_types=["screen"])
        self.assertEqual(result["surfaces_found"], 1)


# ============================================================
# Audio Redaction & Speaker Anonymization Tests
# ============================================================
class TestAudioAnon(unittest.TestCase):
    """Tests for opencut.core.audio_anon module."""

    def test_diarize_speakers_signature(self):
        from opencut.core.audio_anon import diarize_speakers
        sig = inspect.signature(diarize_speakers)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("num_speakers", sig.parameters)

    def test_anonymize_speaker_signature(self):
        from opencut.core.audio_anon import anonymize_speaker
        sig = inspect.signature(anonymize_speaker)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("target_speaker", sig.parameters)
        self.assertIn("pitch_semitones", sig.parameters)

    def test_speaker_segment_dataclass(self):
        from opencut.core.audio_anon import SpeakerSegment
        s = SpeakerSegment(speaker="speaker_0", start=0.0, end=5.0, duration=5.0)
        self.assertEqual(s.speaker, "speaker_0")
        self.assertEqual(s.duration, 5.0)

    @patch("opencut.core.audio_anon.subprocess.run")
    @patch("opencut.core.audio_anon.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.audio_anon.get_video_info", return_value={"duration": 30.0, "width": 0, "height": 0})
    def test_diarize_simple_returns_segments(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.audio_anon import _diarize_simple
        mock_run.return_value = MagicMock(
            stderr="silence_start: 5.0\nsilence_end: 6.0 | silence_duration: 1.0\n"
                   "silence_start: 15.0\nsilence_end: 16.0 | silence_duration: 1.0\n",
            returncode=0,
        )
        segments = _diarize_simple("/fake/audio.wav", num_speakers=2)
        self.assertGreater(len(segments), 0)
        self.assertTrue(all(hasattr(s, "speaker") for s in segments))

    @patch("opencut.core.audio_anon.run_ffmpeg")
    @patch("opencut.core.audio_anon.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.audio_anon.get_video_info", return_value={"duration": 10.0, "width": 0, "height": 0})
    @patch("opencut.core.audio_anon.diarize_speakers")
    def test_anonymize_speaker_no_target(self, mock_diarize, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.audio_anon import anonymize_speaker
        mock_diarize.return_value = {
            "segments": [{"speaker": "speaker_0", "start": 0, "end": 5, "duration": 5}],
            "speaker_count": 1,
            "speakers": ["speaker_0"],
            "method": "energy",
        }
        result = anonymize_speaker("/fake/audio.wav", target_speaker="nonexistent")
        self.assertEqual(result["segments_processed"], 0)


# ============================================================
# Visual Spectrogram Editor Tests
# ============================================================
class TestSpectrogramEdit(unittest.TestCase):
    """Tests for opencut.core.spectrogram_edit module."""

    def test_generate_spectrogram_signature(self):
        from opencut.core.spectrogram_edit import generate_spectrogram
        sig = inspect.signature(generate_spectrogram)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("n_fft", sig.parameters)
        self.assertIn("hop_length", sig.parameters)

    def test_apply_spectrogram_mask_signature(self):
        from opencut.core.spectrogram_edit import apply_spectrogram_mask
        sig = inspect.signature(apply_spectrogram_mask)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("mask", sig.parameters)
        self.assertIn("n_fft", sig.parameters)

    @patch("opencut.core.spectrogram_edit.run_ffmpeg")
    @patch("opencut.core.spectrogram_edit.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_apply_mask_empty_copies(self, mock_ffmpeg, mock_run):
        from opencut.core.spectrogram_edit import apply_spectrogram_mask
        result = apply_spectrogram_mask("/fake/audio.wav", mask=[])
        self.assertEqual(result["regions_applied"], 0)

    @patch("opencut.core.spectrogram_edit._generate_spectrogram_librosa", return_value=None)
    @patch("opencut.core.spectrogram_edit.run_ffmpeg")
    @patch("opencut.core.spectrogram_edit.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_apply_mask_ffmpeg_fallback(self, mock_ffmpeg, mock_run, mock_lib):
        from opencut.core.spectrogram_edit import apply_spectrogram_mask
        with patch("opencut.helpers.get_video_info", return_value={"duration": 10.0}):
            mask = [{"time_start": 0, "time_end": 5, "freq_low": 100, "freq_high": 500, "gain": 0.0}]
            result = apply_spectrogram_mask("/fake/audio.wav", mask=mask)
            self.assertEqual(result["method"], "ffmpeg_bandreject")
            self.assertEqual(result["regions_applied"], 1)

    @patch("opencut.core.spectrogram_edit._generate_spectrogram_librosa", return_value=None)
    @patch("opencut.core.spectrogram_edit._generate_spectrogram_image")
    @patch("opencut.core.spectrogram_edit.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_generate_spectrogram_ffmpeg_fallback(self, mock_ffmpeg, mock_image, mock_lib):
        from opencut.core.spectrogram_edit import generate_spectrogram
        result = generate_spectrogram("/fake/audio.wav")
        self.assertEqual(result["method"], "ffmpeg")
        mock_image.assert_called_once()


# ============================================================
# Spectral Repair / Frequency Removal Tests
# ============================================================
class TestSpectralRepair(unittest.TestCase):
    """Tests for opencut.core.spectral_repair module."""

    def test_analyze_frequencies_signature(self):
        from opencut.core.spectral_repair import analyze_frequencies
        sig = inspect.signature(analyze_frequencies)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("n_fft", sig.parameters)
        self.assertIn("threshold_db", sig.parameters)

    def test_repair_frequencies_signature(self):
        from opencut.core.spectral_repair import repair_frequencies
        sig = inspect.signature(repair_frequencies)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("target_frequencies", sig.parameters)
        self.assertIn("auto_detect", sig.parameters)
        self.assertIn("attenuation_db", sig.parameters)
        self.assertIn("bandwidth", sig.parameters)

    @patch("opencut.core.spectral_repair._analyze_frequencies_librosa", return_value=None)
    def test_analyze_frequencies_ffmpeg_fallback(self, mock_lib):
        from opencut.core.spectral_repair import analyze_frequencies
        result = analyze_frequencies("/fake/audio.wav")
        self.assertEqual(result["method"], "ffmpeg")
        self.assertIn("peaks", result)

    @patch("opencut.core.spectral_repair.run_ffmpeg")
    @patch("opencut.core.spectral_repair.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.spectral_repair._analyze_frequencies_librosa", return_value=None)
    def test_repair_no_frequencies(self, mock_lib, mock_ffmpeg, mock_run):
        from opencut.core.spectral_repair import repair_frequencies
        result = repair_frequencies("/fake/audio.wav", target_frequencies=[], auto_detect=False)
        self.assertEqual(result["frequencies_removed"], [])

    @patch("opencut.core.spectral_repair.run_ffmpeg")
    @patch("opencut.core.spectral_repair.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.spectral_repair._analyze_frequencies_librosa", return_value=None)
    def test_repair_ffmpeg_fallback(self, mock_lib, mock_ffmpeg, mock_run):
        from opencut.core.spectral_repair import repair_frequencies
        result = repair_frequencies(
            "/fake/audio.wav", target_frequencies=[60.0, 120.0], auto_detect=False
        )
        self.assertEqual(result["method"], "ffmpeg_bandreject")
        self.assertEqual(len(result["frequencies_removed"]), 2)

    def test_repair_ffmpeg_builds_filters(self):
        from opencut.core.spectral_repair import _repair_ffmpeg
        with patch("opencut.core.spectral_repair.run_ffmpeg") as mock_run, \
             patch("opencut.core.spectral_repair.get_ffmpeg_path", return_value="/usr/bin/ffmpeg"):
            _repair_ffmpeg("/fake/audio.wav", [50.0, 100.0], bandwidth=10.0)
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)
            self.assertIn("bandreject", cmd_str)


# ============================================================
# AI Environmental Noise Classifier Tests
# ============================================================
class TestNoiseClassify(unittest.TestCase):
    """Tests for opencut.core.noise_classify module."""

    def test_classify_noise_signature(self):
        from opencut.core.noise_classify import classify_noise
        sig = inspect.signature(classify_noise)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("segment_duration", sig.parameters)

    def test_remove_classified_noise_signature(self):
        from opencut.core.noise_classify import remove_classified_noise
        sig = inspect.signature(remove_classified_noise)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("noise_types", sig.parameters)
        self.assertIn("segments", sig.parameters)

    def test_noise_classes_defined(self):
        from opencut.core.noise_classify import NOISE_CLASSES
        self.assertIn("traffic", NOISE_CLASSES)
        self.assertIn("wind", NOISE_CLASSES)
        self.assertIn("hvac", NOISE_CLASSES)
        self.assertIn("typing", NOISE_CLASSES)
        self.assertIn("electrical_hum", NOISE_CLASSES)

    def test_noise_segment_dataclass(self):
        from opencut.core.noise_classify import NoiseSegment
        s = NoiseSegment(start=0.0, end=1.0, duration=1.0, noise_type="wind", confidence=0.7)
        self.assertEqual(s.noise_type, "wind")

    @patch("opencut.core.noise_classify._classify_yamnet", return_value=None)
    @patch("opencut.core.noise_classify.run_ffmpeg")
    @patch("opencut.core.noise_classify.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.noise_classify.get_video_info", return_value={"duration": 5.0})
    def test_classify_falls_back_to_heuristic(self, mock_info, mock_ffmpeg, mock_run, mock_yamnet):
        from opencut.core.noise_classify import classify_noise
        # Mock heuristic to return empty since raw audio file won't exist
        with patch("opencut.core.noise_classify._classify_heuristic", return_value=[]):
            result = classify_noise("/fake/audio.wav")
            self.assertEqual(result["method"], "heuristic")

    @patch("opencut.core.noise_classify.run_ffmpeg")
    @patch("opencut.core.noise_classify.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.noise_classify.get_video_info", return_value={"duration": 5.0})
    def test_remove_noise_no_matching_segments(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.noise_classify import remove_classified_noise
        segs = [{"noise_type": "traffic", "start": 0, "end": 1, "duration": 1}]
        result = remove_classified_noise("/fake/audio.wav", noise_types=["wind"], segments=segs)
        self.assertEqual(result["segments_removed"], 0)

    @patch("opencut.core.noise_classify.run_ffmpeg")
    @patch("opencut.core.noise_classify.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    @patch("opencut.core.noise_classify.get_video_info", return_value={"duration": 5.0})
    def test_remove_noise_matching_segments(self, mock_info, mock_ffmpeg, mock_run):
        from opencut.core.noise_classify import remove_classified_noise
        segs = [{"noise_type": "traffic", "start": 0, "end": 1, "duration": 1}]
        result = remove_classified_noise("/fake/audio.wav", noise_types=["traffic"], segments=segs)
        self.assertEqual(result["segments_removed"], 1)
        self.assertIn("traffic", result["noise_types_removed"])


# ============================================================
# Room Tone Enhanced Tests
# ============================================================
class TestRoomToneEnhanced(unittest.TestCase):
    """Tests for enhanced room_tone.py functions."""

    def test_analyze_room_tone_signature(self):
        from opencut.core.room_tone import analyze_room_tone
        sig = inspect.signature(analyze_room_tone)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("n_fft", sig.parameters)
        self.assertIn("hop_length", sig.parameters)

    def test_synthesize_room_tone_signature(self):
        from opencut.core.room_tone import synthesize_room_tone
        sig = inspect.signature(synthesize_room_tone)
        self.assertIn("profile", sig.parameters)
        self.assertIn("duration", sig.parameters)
        self.assertIn("output_path_str", sig.parameters)

    def test_fill_cuts_with_room_tone_signature(self):
        from opencut.core.room_tone import fill_cuts_with_room_tone
        sig = inspect.signature(fill_cuts_with_room_tone)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("cut_points", sig.parameters)
        self.assertIn("output_path_str", sig.parameters)

    @patch("opencut.core.room_tone._find_quietest_segments", return_value=[(2.0, -45.0)])
    @patch("opencut.core.room_tone._get_duration", return_value=30.0)
    def test_analyze_room_tone_ffmpeg_fallback(self, mock_dur, mock_quiet):
        from opencut.core.room_tone import analyze_room_tone
        result = analyze_room_tone("/fake/audio.wav")
        self.assertIn("spectral_envelope", result)
        self.assertIn("rms_db", result)
        self.assertEqual(result["method"], "ffmpeg_basic")
        self.assertIsInstance(result["spectral_envelope"], list)

    def test_synthesize_room_tone_ffmpeg(self):
        from opencut.core.room_tone import synthesize_room_tone
        profile = {
            "spectral_envelope": [{"freq": 100, "magnitude_db": -30}],
            "rms_db": -50,
            "sample_rate": 48000,
        }
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            with patch("opencut.core.room_tone.run_ffmpeg"), \
                 patch("opencut.core.room_tone.get_ffmpeg_path", return_value="/usr/bin/ffmpeg"):
                result = synthesize_room_tone(profile, duration=2.0, output_path_str=out_path)
                self.assertIn("output_path", result)
                self.assertIn("method", result)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    @patch("opencut.core.room_tone.run_ffmpeg")
    @patch("opencut.core.room_tone.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_fill_cuts_empty_list(self, mock_ffmpeg, mock_run):
        from opencut.core.room_tone import fill_cuts_with_room_tone
        result = fill_cuts_with_room_tone("/fake/audio.wav", cut_points=[])
        self.assertEqual(result["cuts_filled"], 0)


# ============================================================
# Privacy & Spectral Routes Tests
# ============================================================
class TestPrivacySpectralRoutes(unittest.TestCase):
    """Tests for privacy_spectral_routes.py blueprint registration and routes."""

    def test_blueprint_exists(self):
        from opencut.routes.privacy_spectral_routes import privacy_spectral_bp
        self.assertEqual(privacy_spectral_bp.name, "privacy_spectral")

    def test_plate_blur_route_exists(self):
        from opencut.routes.privacy_spectral_routes import plate_blur
        self.assertTrue(callable(plate_blur))

    def test_pii_redact_route_exists(self):
        from opencut.routes.privacy_spectral_routes import pii_redact
        self.assertTrue(callable(pii_redact))

    def test_profanity_bleep_route_exists(self):
        from opencut.routes.privacy_spectral_routes import profanity_bleep
        self.assertTrue(callable(profanity_bleep))

    def test_doc_redact_route_exists(self):
        from opencut.routes.privacy_spectral_routes import doc_redact
        self.assertTrue(callable(doc_redact))

    def test_anonymize_speaker_route_exists(self):
        from opencut.routes.privacy_spectral_routes import anonymize_speaker
        self.assertTrue(callable(anonymize_speaker))

    def test_spectral_edit_route_exists(self):
        from opencut.routes.privacy_spectral_routes import spectral_edit
        self.assertTrue(callable(spectral_edit))

    def test_spectral_repair_route_exists(self):
        from opencut.routes.privacy_spectral_routes import spectral_repair
        self.assertTrue(callable(spectral_repair))

    def test_classify_noise_route_exists(self):
        from opencut.routes.privacy_spectral_routes import classify_noise
        self.assertTrue(callable(classify_noise))

    def test_room_tone_fill_route_exists(self):
        from opencut.routes.privacy_spectral_routes import room_tone_fill
        self.assertTrue(callable(room_tone_fill))

    def test_all_routes_use_post(self):
        from opencut.routes.privacy_spectral_routes import privacy_spectral_bp
        # All routes should be POST
        for rule_func in privacy_spectral_bp.deferred_functions:
            # The deferred_functions are registration callables, not rules
            pass
        # Verify by importing the module and checking the endpoint functions have route decorators
        import opencut.routes.privacy_spectral_routes as mod
        src = inspect.getsource(mod)
        self.assertIn('methods=["POST"]', src)

    def test_route_paths_defined(self):
        import opencut.routes.privacy_spectral_routes as mod
        src = inspect.getsource(mod)
        expected = [
            "/privacy/plate-blur",
            "/privacy/pii-redact",
            "/privacy/profanity-bleep",
            "/privacy/doc-redact",
            "/privacy/anonymize-speaker",
            "/spectral/edit",
            "/spectral/repair",
            "/spectral/classify-noise",
            "/spectral/room-tone-fill",
        ]
        for path in expected:
            self.assertIn(path, src, f"Route {path} not found in source")


# ============================================================
# Cross-module integration pattern checks
# ============================================================
class TestModulePatterns(unittest.TestCase):
    """Verify modules follow codebase conventions."""

    def test_plate_blur_uses_helpers(self):
        import opencut.core.plate_blur as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)
        self.assertIn("get_ffmpeg_path", src)
        self.assertIn("run_ffmpeg", src)

    def test_pii_redact_uses_helpers(self):
        import opencut.core.pii_redact as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_profanity_bleep_uses_helpers(self):
        import opencut.core.profanity_bleep as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_doc_redact_uses_helpers(self):
        import opencut.core.doc_redact as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_audio_anon_uses_helpers(self):
        import opencut.core.audio_anon as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_spectrogram_edit_uses_helpers(self):
        import opencut.core.spectrogram_edit as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_spectral_repair_uses_helpers(self):
        import opencut.core.spectral_repair as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_noise_classify_uses_helpers(self):
        import opencut.core.noise_classify as mod
        src = inspect.getsource(mod)
        self.assertIn("from opencut.helpers import", src)

    def test_routes_use_async_job(self):
        import opencut.routes.privacy_spectral_routes as mod
        src = inspect.getsource(mod)
        self.assertIn("@async_job", src)
        self.assertIn("@require_csrf", src)

    def test_routes_use_safe_helpers(self):
        import opencut.routes.privacy_spectral_routes as mod
        src = inspect.getsource(mod)
        self.assertIn("safe_float", src)
        self.assertIn("safe_int", src)
        self.assertIn("safe_bool", src)

    def test_all_core_modules_have_docstring(self):
        import opencut.core.plate_blur
        import opencut.core.pii_redact
        import opencut.core.profanity_bleep
        import opencut.core.doc_redact
        import opencut.core.audio_anon
        import opencut.core.spectrogram_edit
        import opencut.core.spectral_repair
        import opencut.core.noise_classify
        for mod in [
            opencut.core.plate_blur,
            opencut.core.pii_redact,
            opencut.core.profanity_bleep,
            opencut.core.doc_redact,
            opencut.core.audio_anon,
            opencut.core.spectrogram_edit,
            opencut.core.spectral_repair,
            opencut.core.noise_classify,
        ]:
            self.assertIsNotNone(mod.__doc__, f"{mod.__name__} missing module docstring")

    def test_all_core_modules_use_logger(self):
        import opencut.core.plate_blur
        import opencut.core.pii_redact
        import opencut.core.profanity_bleep
        import opencut.core.doc_redact
        import opencut.core.audio_anon
        import opencut.core.spectrogram_edit
        import opencut.core.spectral_repair
        import opencut.core.noise_classify
        for mod in [
            opencut.core.plate_blur,
            opencut.core.pii_redact,
            opencut.core.profanity_bleep,
            opencut.core.doc_redact,
            opencut.core.audio_anon,
            opencut.core.spectrogram_edit,
            opencut.core.spectral_repair,
            opencut.core.noise_classify,
        ]:
            self.assertTrue(
                hasattr(mod, "logger"),
                f"{mod.__name__} missing logger",
            )


if __name__ == "__main__":
    unittest.main()
