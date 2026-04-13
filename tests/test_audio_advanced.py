"""
Tests for OpenCut Audio Advanced features.

Covers:
  - Podcast Production Suite (podcast_suite.py)
  - Real-Time Voice Conversion (realtime_voice.py)
  - ADR Cueing System (adr_cueing.py)
  - Surround Sound Panning & Upmix (surround_mix.py)
  - Audio-Reactive Visualizers (audio_visualizer.py)
  - Audio Description Track Generator (audio_description.py)
  - Voice Commands (voice_commands.py)
  - Audio Advanced Routes (smoke tests)
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Podcast Production Suite Tests
# ============================================================
class TestPodcastSuite(unittest.TestCase):
    """Tests for opencut.core.podcast_suite module."""

    def test_podcast_config_dataclass(self):
        from opencut.core.podcast_suite import PodcastConfig
        config = PodcastConfig()
        self.assertEqual(config.target_lufs, -16.0)
        self.assertIsNone(config.intro_path)
        self.assertIsNone(config.outro_path)
        self.assertEqual(config.crossfade_duration, 2.0)
        self.assertTrue(config.generate_chapters)
        self.assertEqual(config.highpass_hz, 80.0)
        self.assertEqual(config.compressor_ratio, 3.0)

    def test_podcast_config_custom_values(self):
        from opencut.core.podcast_suite import PodcastConfig
        config = PodcastConfig(target_lufs=-23.0, intro_path="/fake/intro.wav")
        self.assertEqual(config.target_lufs, -23.0)
        self.assertEqual(config.intro_path, "/fake/intro.wav")

    def test_podcast_result_dataclass(self):
        from opencut.core.podcast_suite import PodcastResult
        result = PodcastResult()
        self.assertEqual(result.output_path, "")
        self.assertEqual(result.speakers_detected, 0)
        self.assertIsInstance(result.chapters, list)
        self.assertEqual(result.loudness_lufs, -16.0)

    def test_speaker_segment_dataclass(self):
        from opencut.core.podcast_suite import SpeakerSegment
        seg = SpeakerSegment(speaker_id="speaker_0", start=1.0, end=5.0, text="Hello")
        self.assertEqual(seg.speaker_id, "speaker_0")
        self.assertEqual(seg.start, 1.0)
        self.assertEqual(seg.end, 5.0)
        self.assertEqual(seg.text, "Hello")

    def test_chapter_marker_dataclass(self):
        from opencut.core.podcast_suite import ChapterMarker
        ch = ChapterMarker(title="Intro", start_time=0.0, end_time=60.0)
        self.assertEqual(ch.title, "Intro")
        self.assertEqual(ch.start_time, 0.0)

    def test_polish_podcast_signature(self):
        from opencut.core.podcast_suite import polish_podcast
        sig = inspect.signature(polish_podcast)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("config", sig.parameters)
        self.assertIn("output_path_val", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_detect_speakers_signature(self):
        from opencut.core.podcast_suite import detect_speakers
        sig = inspect.signature(detect_speakers)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_apply_per_speaker_processing_signature(self):
        from opencut.core.podcast_suite import apply_per_speaker_processing
        sig = inspect.signature(apply_per_speaker_processing)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("speaker_segments", sig.parameters)
        self.assertIn("config", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    @patch("opencut.core.podcast_suite.run_ffmpeg")
    @patch("opencut.core.podcast_suite.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_detect_speakers_returns_list(self, mock_ffmpeg, mock_run):
        from opencut.core.podcast_suite import detect_speakers
        mock_run.return_value = ""
        segments = detect_speakers("/fake/podcast.wav")
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        self.assertEqual(segments[0].speaker_id, "speaker_0")

    @patch("opencut.core.podcast_suite.run_ffmpeg")
    @patch("opencut.core.podcast_suite.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_detect_speakers_with_silence(self, mock_ffmpeg, mock_run):
        from opencut.core.podcast_suite import detect_speakers
        mock_run.return_value = (
            "silence_start: 5.0\nsilence_end: 7.0\n"
            "silence_start: 12.0\nsilence_end: 14.0\n"
        )
        segments = detect_speakers("/fake/podcast.wav")
        self.assertIsInstance(segments, list)
        # Should have segments between silences
        self.assertGreater(len(segments), 0)


# ============================================================
# Real-Time Voice Conversion Tests
# ============================================================
class TestRealtimeVoice(unittest.TestCase):
    """Tests for opencut.core.realtime_voice module."""

    def test_voice_conversion_config_dataclass(self):
        from opencut.core.realtime_voice import VoiceConversionConfig
        config = VoiceConversionConfig()
        self.assertEqual(config.pitch_shift_semitones, 0.0)
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.model_name, "default")
        self.assertTrue(config.apply_noise_gate)

    def test_voice_conversion_session_dataclass(self):
        from opencut.core.realtime_voice import VoiceConversionSession
        session = VoiceConversionSession()
        self.assertEqual(session.status, "idle")
        self.assertEqual(session.session_id, "")
        self.assertEqual(session.samples_processed, 0)

    def test_list_voice_models_returns_builtins(self):
        from opencut.core.realtime_voice import list_voice_models
        models = list_voice_models()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 5)
        names = [m["name"] for m in models]
        self.assertIn("pitch_up", names)
        self.assertIn("pitch_down", names)
        self.assertIn("chipmunk", names)
        self.assertIn("deep", names)
        self.assertIn("robot", names)

    def test_list_voice_models_structure(self):
        from opencut.core.realtime_voice import list_voice_models
        models = list_voice_models()
        for m in models:
            self.assertIn("name", m)
            self.assertIn("display_name", m)
            self.assertIn("type", m)
            self.assertIn("path", m)
            self.assertIn("description", m)

    def test_start_voice_conversion_signature(self):
        from opencut.core.realtime_voice import start_voice_conversion
        sig = inspect.signature(start_voice_conversion)
        self.assertIn("model_path", sig.parameters)
        self.assertIn("config", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_stop_voice_conversion_signature(self):
        from opencut.core.realtime_voice import stop_voice_conversion
        sig = inspect.signature(stop_voice_conversion)
        self.assertIn("on_progress", sig.parameters)

    def test_start_creates_session(self):
        # Clean up any prior session
        import opencut.core.realtime_voice as mod
        from opencut.core.realtime_voice import (
            start_voice_conversion,
            stop_voice_conversion,
        )
        with mod._session_lock:
            mod._active_session = None

        session = start_voice_conversion("pitch_up")
        self.assertEqual(session.status, "recording")
        self.assertIn("vc_", session.session_id)
        self.assertEqual(session.model_path, "pitch_up")

        # Clean up
        stop_voice_conversion()

    def test_stop_without_start_raises(self):
        import opencut.core.realtime_voice as mod
        with mod._session_lock:
            mod._active_session = None
        with self.assertRaises(RuntimeError):
            mod.stop_voice_conversion()


# ============================================================
# ADR Cueing System Tests
# ============================================================
class TestADRCueing(unittest.TestCase):
    """Tests for opencut.core.adr_cueing module."""

    def test_adr_cue_dataclass(self):
        from opencut.core.adr_cueing import ADRCue
        cue = ADRCue()
        self.assertEqual(cue.status, "pending")
        self.assertEqual(cue.priority, "normal")
        self.assertEqual(cue.takes_recorded, 0)
        field_names = [f.name for f in fields(ADRCue)]
        self.assertIn("cue_id", field_names)
        self.assertIn("character", field_names)
        self.assertIn("line_text", field_names)

    def test_adr_cue_sheet_dataclass(self):
        from opencut.core.adr_cueing import ADRCueSheet
        sheet = ADRCueSheet()
        self.assertEqual(sheet.total_lines, 0)
        self.assertIsInstance(sheet.cues, list)
        self.assertEqual(sheet.export_format, "json")

    def test_adr_guide_result_dataclass(self):
        from opencut.core.adr_cueing import ADRGuideResult
        result = ADRGuideResult()
        self.assertEqual(result.preroll_seconds, 3.0)
        self.assertEqual(result.postroll_seconds, 2.0)

    def test_adr_sync_result_dataclass(self):
        from opencut.core.adr_cueing import ADRSyncResult
        result = ADRSyncResult()
        self.assertEqual(result.sync_quality, "good")

    def test_create_adr_cue_sheet_signature(self):
        from opencut.core.adr_cueing import create_adr_cue_sheet
        sig = inspect.signature(create_adr_cue_sheet)
        self.assertIn("transcript", sig.parameters)
        self.assertIn("marked_lines", sig.parameters)
        self.assertIn("output_path_val", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_create_adr_cue_sheet_basic(self):
        from opencut.core.adr_cueing import create_adr_cue_sheet
        transcript = [
            {"text": "Hello there", "start": 1.0, "end": 2.5, "character": "Alice"},
            {"text": "How are you", "start": 3.0, "end": 4.5, "character": "Bob"},
            {"text": "I am fine", "start": 5.0, "end": 6.5, "character": "Alice"},
        ]
        result = create_adr_cue_sheet(transcript, [0, 2])
        self.assertEqual(result.total_lines, 2)
        self.assertEqual(result.cues[0].character, "Alice")
        self.assertEqual(result.cues[1].character, "Alice")
        self.assertEqual(result.cues[0].cue_id, "ADR-0001")

    def test_create_adr_cue_sheet_json_export(self):
        from opencut.core.adr_cueing import create_adr_cue_sheet
        transcript = [
            {"text": "Line one", "start": 0, "end": 1, "character": "A"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        try:
            create_adr_cue_sheet(transcript, [0], output_path_val=out_path, export_format="json")
            with open(out_path) as f:
                data = json.load(f)
            self.assertIn("cues", data)
            self.assertEqual(len(data["cues"]), 1)
        finally:
            os.unlink(out_path)

    def test_create_adr_cue_sheet_csv_export(self):
        from opencut.core.adr_cueing import create_adr_cue_sheet
        transcript = [
            {"text": "Line one", "start": 0, "end": 1, "character": "A"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out_path = f.name
        try:
            create_adr_cue_sheet(transcript, [0], output_path_val=out_path, export_format="csv")
            with open(out_path) as f:
                content = f.read()
            self.assertIn("Cue ID", content)
            self.assertIn("ADR-0001", content)
        finally:
            os.unlink(out_path)

    def test_create_adr_cue_sheet_txt_export(self):
        from opencut.core.adr_cueing import create_adr_cue_sheet
        transcript = [
            {"text": "Line one", "start": 0, "end": 1, "character": "Bob"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            out_path = f.name
        try:
            create_adr_cue_sheet(transcript, [0], output_path_val=out_path, export_format="txt")
            with open(out_path) as f:
                content = f.read()
            self.assertIn("ADR CUE SHEET", content)
            self.assertIn("Bob", content)
        finally:
            os.unlink(out_path)

    def test_generate_adr_guide_signature(self):
        from opencut.core.adr_cueing import generate_adr_guide
        sig = inspect.signature(generate_adr_guide)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("cue", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    @patch("opencut.core.adr_cueing.run_ffmpeg")
    @patch("opencut.core.adr_cueing.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_generate_adr_guide_calls_ffmpeg(self, mock_ffmpeg, mock_run):
        from opencut.core.adr_cueing import ADRCue, generate_adr_guide
        cue = ADRCue(cue_id="ADR-0001", character="Alice", line_text="Hello",
                     start_time=5.0, end_time=8.0)
        result = generate_adr_guide("/fake/video.mp4", cue)
        mock_run.assert_called_once()
        self.assertEqual(result.cue_id, "ADR-0001")
        self.assertGreater(result.duration, 0)

    def test_sync_adr_recording_signature(self):
        from opencut.core.adr_cueing import sync_adr_recording
        sig = inspect.signature(sync_adr_recording)
        self.assertIn("original_path", sig.parameters)
        self.assertIn("recorded_path", sig.parameters)
        self.assertIn("cue", sig.parameters)
        self.assertIn("on_progress", sig.parameters)


# ============================================================
# Surround Sound Tests
# ============================================================
class TestSurroundMix(unittest.TestCase):
    """Tests for opencut.core.surround_mix module."""

    def test_surround_position_dataclass(self):
        from opencut.core.surround_mix import SurroundPosition
        pos = SurroundPosition()
        self.assertEqual(pos.angle, 0.0)
        self.assertEqual(pos.distance, 1.0)
        self.assertEqual(pos.lfe_amount, 0.0)

    def test_channel_layouts_defined(self):
        from opencut.core.surround_mix import CHANNEL_LAYOUTS
        self.assertIn("5.1", CHANNEL_LAYOUTS)
        self.assertIn("7.1", CHANNEL_LAYOUTS)
        self.assertEqual(CHANNEL_LAYOUTS["5.1"]["channels"], 6)
        self.assertEqual(CHANNEL_LAYOUTS["7.1"]["channels"], 8)

    def test_export_formats_defined(self):
        from opencut.core.surround_mix import EXPORT_FORMATS
        self.assertIn("wav", EXPORT_FORMATS)
        self.assertIn("flac", EXPORT_FORMATS)
        self.assertIn("ac3", EXPORT_FORMATS)
        self.assertIn("eac3", EXPORT_FORMATS)

    def test_upmix_signature(self):
        from opencut.core.surround_mix import upmix_to_surround
        sig = inspect.signature(upmix_to_surround)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("channels", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_upmix_rejects_invalid_layout(self):
        from opencut.core.surround_mix import upmix_to_surround
        with self.assertRaises(ValueError):
            upmix_to_surround("/fake/audio.wav", channels="9.1")

    @patch("opencut.core.surround_mix.run_ffmpeg")
    @patch("opencut.core.surround_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_upmix_51_calls_ffmpeg(self, mock_ffmpeg, mock_run):
        from opencut.core.surround_mix import upmix_to_surround
        result = upmix_to_surround("/fake/audio.wav", channels="5.1")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("pan=5.1", " ".join(cmd))
        self.assertEqual(result.target_channels, 6)

    @patch("opencut.core.surround_mix.run_ffmpeg")
    @patch("opencut.core.surround_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_upmix_71_calls_ffmpeg(self, mock_ffmpeg, mock_run):
        from opencut.core.surround_mix import upmix_to_surround
        result = upmix_to_surround("/fake/audio.wav", channels="7.1")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("pan=7.1", " ".join(cmd))
        self.assertEqual(result.target_channels, 8)

    def test_pan_signature(self):
        from opencut.core.surround_mix import pan_in_surround
        sig = inspect.signature(pan_in_surround)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("position", sig.parameters)
        self.assertIn("channels", sig.parameters)

    def test_calculate_surround_gains(self):
        from opencut.core.surround_mix import SurroundPosition, _calculate_surround_gains
        pos = SurroundPosition(angle=0.0, distance=1.0)
        gains = _calculate_surround_gains(pos, "5.1")
        self.assertIn("FL", gains)
        self.assertIn("FR", gains)
        self.assertIn("FC", gains)
        self.assertIn("LFE", gains)
        self.assertIn("BL", gains)
        self.assertIn("BR", gains)
        # Center position should have FC as strongest non-LFE
        total = sum(v for k, v in gains.items() if k != "LFE")
        self.assertAlmostEqual(total, 1.0, places=1)

    @patch("opencut.core.surround_mix.run_ffmpeg")
    @patch("opencut.core.surround_mix.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_downmix_preview(self, mock_ffmpeg, mock_run):
        from opencut.core.surround_mix import downmix_preview
        downmix_preview("/fake/surround.wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("-ac", cmd)
        self.assertIn("2", cmd)

    def test_export_multichannel_signature(self):
        from opencut.core.surround_mix import export_multichannel
        sig = inspect.signature(export_multichannel)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("format", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_export_rejects_invalid_format(self):
        from opencut.core.surround_mix import export_multichannel
        with self.assertRaises(ValueError):
            export_multichannel("/fake/audio.wav", format="mp3")


# ============================================================
# Audio Visualizer Tests
# ============================================================
class TestAudioVisualizer(unittest.TestCase):
    """Tests for opencut.core.audio_visualizer module."""

    def test_visualizer_config_dataclass(self):
        from opencut.core.audio_visualizer import VisualizerConfig
        config = VisualizerConfig()
        self.assertEqual(config.width, 1920)
        self.assertEqual(config.height, 1080)
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.style, "waveform_bars")
        self.assertFalse(config.as_overlay)

    def test_visualizer_styles_defined(self):
        from opencut.core.audio_visualizer import VISUALIZER_STYLES
        self.assertIn("waveform_bars", VISUALIZER_STYLES)
        self.assertIn("spectrum_circle", VISUALIZER_STYLES)
        self.assertIn("waterfall", VISUALIZER_STYLES)
        self.assertIn("waveform_line", VISUALIZER_STYLES)
        self.assertIn("volume_meter", VISUALIZER_STYLES)

    def test_generate_visualizer_signature(self):
        from opencut.core.audio_visualizer import generate_visualizer
        sig = inspect.signature(generate_visualizer)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("style", sig.parameters)
        self.assertIn("config", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_generate_visualizer_rejects_invalid_style(self):
        from opencut.core.audio_visualizer import generate_visualizer
        with self.assertRaises(ValueError):
            generate_visualizer("/fake/audio.wav", style="nonexistent")

    def test_create_waveform_bars_signature(self):
        from opencut.core.audio_visualizer import create_waveform_bars
        sig = inspect.signature(create_waveform_bars)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("config", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    @patch("opencut.core.audio_visualizer.run_ffmpeg")
    @patch("opencut.core.audio_visualizer.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_create_waveform_bars_calls_ffmpeg(self, mock_ffmpeg, mock_run):
        from opencut.core.audio_visualizer import create_waveform_bars
        result = create_waveform_bars("/fake/audio.wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("showwaves", " ".join(cmd))
        self.assertEqual(result.style, "waveform_bars")

    def test_create_spectrum_circle_signature(self):
        from opencut.core.audio_visualizer import create_spectrum_circle
        sig = inspect.signature(create_spectrum_circle)
        self.assertIn("audio_path", sig.parameters)

    @patch("opencut.core.audio_visualizer.run_ffmpeg")
    @patch("opencut.core.audio_visualizer.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_create_spectrum_circle_calls_ffmpeg(self, mock_ffmpeg, mock_run):
        from opencut.core.audio_visualizer import create_spectrum_circle
        create_spectrum_circle("/fake/audio.wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("showfreqs", " ".join(cmd))


# ============================================================
# Audio Description Tests
# ============================================================
class TestAudioDescription(unittest.TestCase):
    """Tests for opencut.core.audio_description module."""

    def test_description_gap_dataclass(self):
        from opencut.core.audio_description import DescriptionGap
        gap = DescriptionGap(start=5.0, end=8.0, duration=3.0)
        self.assertEqual(gap.start, 5.0)
        self.assertEqual(gap.duration, 3.0)
        self.assertTrue(gap.suitable)

    def test_visual_description_dataclass(self):
        from opencut.core.audio_description import VisualDescription
        desc = VisualDescription(timestamp=10.0, description="A person walks")
        self.assertEqual(desc.importance, "normal")

    def test_ad_result_dataclass(self):
        from opencut.core.audio_description import ADResult
        result = ADResult()
        self.assertEqual(result.gaps_found, 0)
        self.assertEqual(result.descriptions_added, 0)

    def test_find_description_gaps_signature(self):
        from opencut.core.audio_description import find_description_gaps
        sig = inspect.signature(find_description_gaps)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("transcript", sig.parameters)
        self.assertIn("min_gap_seconds", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_find_gaps_from_transcript(self):
        from opencut.core.audio_description import find_description_gaps
        transcript = [
            {"text": "Hello", "start": 0, "end": 2},
            {"text": "World", "start": 5, "end": 7},
            {"text": "Done", "start": 12, "end": 14},
        ]
        gaps = find_description_gaps(
            "/fake/audio.wav", transcript=transcript, min_gap_seconds=2.0,
        )
        self.assertIsInstance(gaps, list)
        self.assertGreater(len(gaps), 0)
        # Gap between 2.0 and 5.0 = 3 seconds
        self.assertAlmostEqual(gaps[0].start, 2.0)
        self.assertAlmostEqual(gaps[0].end, 5.0)

    def test_describe_visual_content_signature(self):
        from opencut.core.audio_description import describe_visual_content
        sig = inspect.signature(describe_visual_content)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("timestamp", sig.parameters)

    def test_synthesize_description_signature(self):
        from opencut.core.audio_description import synthesize_description
        sig = inspect.signature(synthesize_description)
        self.assertIn("text", sig.parameters)
        self.assertIn("voice", sig.parameters)
        self.assertIn("speed", sig.parameters)
        self.assertEqual(sig.parameters["speed"].default, 1.1)

    def test_generate_audio_description_signature(self):
        from opencut.core.audio_description import generate_audio_description
        sig = inspect.signature(generate_audio_description)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("descriptions", sig.parameters)
        self.assertIn("transcript", sig.parameters)
        self.assertIn("voice", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    @patch("opencut.core.audio_description.run_ffmpeg")
    @patch("opencut.core.audio_description.get_ffmpeg_path", return_value="/usr/bin/ffmpeg")
    def test_synthesize_description_fallback(self, mock_ffmpeg, mock_run):
        from opencut.core.audio_description import synthesize_description
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            synthesize_description(
                "A character enters the room",
                output_path_val=out_path,
            )
            # Should have called FFmpeg as fallback
            mock_run.assert_called_once()
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


# ============================================================
# Voice Commands Tests
# ============================================================
class TestVoiceCommands(unittest.TestCase):
    """Tests for opencut.core.voice_commands module."""

    def test_voice_command_config_dataclass(self):
        from opencut.core.voice_commands import VoiceCommandConfig
        config = VoiceCommandConfig()
        self.assertEqual(config.wake_word, "opencut")
        self.assertEqual(config.language, "en-US")
        self.assertTrue(config.require_wake_word)
        self.assertTrue(config.continuous)

    def test_voice_command_dataclass(self):
        from opencut.core.voice_commands import VoiceCommand
        cmd = VoiceCommand()
        self.assertEqual(cmd.raw_text, "")
        self.assertFalse(cmd.matched)
        self.assertEqual(cmd.confidence, 0.0)

    def test_voice_listener_session_dataclass(self):
        from opencut.core.voice_commands import VoiceListenerSession
        session = VoiceListenerSession()
        self.assertEqual(session.status, "idle")
        self.assertEqual(session.commands_processed, 0)

    def test_get_command_mapping(self):
        from opencut.core.voice_commands import get_command_mapping
        mapping = get_command_mapping()
        self.assertIsInstance(mapping, dict)
        self.assertIn("playback", mapping)
        self.assertIn("edit", mapping)
        self.assertIn("audio", mapping)
        self.assertIn("timeline", mapping)
        self.assertIn("export", mapping)

    def test_get_command_mapping_structure(self):
        from opencut.core.voice_commands import get_command_mapping
        mapping = get_command_mapping()
        for category, info in mapping.items():
            self.assertIn("actions", info)
            self.assertIn("description", info)
            self.assertIsInstance(info["actions"], list)

    def test_parse_play_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("play")
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "play")
        self.assertIn("category", cmd.parameters)
        self.assertEqual(cmd.parameters["category"], "playback")

    def test_parse_pause_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("pause the video")
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "pause")

    def test_parse_cut_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("cut here")
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "cut")

    def test_parse_undo_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("undo that")
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "undo")

    def test_parse_zoom_in_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("zoom in on the timeline")
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "zoom_in")

    def test_parse_export_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("export the video")
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "render")

    def test_parse_unrecognized_command(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("make me a sandwich")
        self.assertFalse(cmd.matched)
        self.assertEqual(cmd.command, "")

    def test_parse_numeric_parameters(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("volume up 50 percent")
        self.assertTrue(cmd.matched)
        self.assertIn("percentage", cmd.parameters)
        self.assertEqual(cmd.parameters["percentage"], 50.0)

    def test_parse_seconds_parameter(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("rewind 5 seconds")
        self.assertTrue(cmd.matched)
        self.assertIn("seconds", cmd.parameters)
        self.assertEqual(cmd.parameters["seconds"], 5.0)

    def test_parse_custom_command(self):
        from opencut.core.voice_commands import parse_voice_command
        custom = {r"\bflatten\b": "flatten_tracks"}
        cmd = parse_voice_command("flatten all tracks", custom_commands=custom)
        self.assertTrue(cmd.matched)
        self.assertEqual(cmd.action, "flatten_tracks")

    def test_parse_empty_input(self):
        from opencut.core.voice_commands import parse_voice_command
        cmd = parse_voice_command("")
        self.assertFalse(cmd.matched)

    def test_start_voice_listener_signature(self):
        from opencut.core.voice_commands import start_voice_listener
        sig = inspect.signature(start_voice_listener)
        self.assertIn("config", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_stop_voice_listener_signature(self):
        from opencut.core.voice_commands import stop_voice_listener
        sig = inspect.signature(stop_voice_listener)
        self.assertIn("on_progress", sig.parameters)

    def test_start_and_stop_listener(self):
        import opencut.core.voice_commands as mod
        # Clean state
        with mod._listener_lock:
            mod._active_listener = None

        session = mod.start_voice_listener()
        self.assertEqual(session.status, "listening")
        self.assertIn("voice_", session.session_id)

        stopped = mod.stop_voice_listener()
        self.assertEqual(stopped.status, "stopped")

    def test_stop_without_start_raises(self):
        import opencut.core.voice_commands as mod
        with mod._listener_lock:
            mod._active_listener = None
        with self.assertRaises(RuntimeError):
            mod.stop_voice_listener()


# ============================================================
# Route Smoke Tests
# ============================================================
class TestAudioAdvancedRoutes(unittest.TestCase):
    """Smoke tests for audio_advanced_routes.py Blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.audio_advanced_routes import audio_adv_bp
        self.assertEqual(audio_adv_bp.name, "audio_adv")

    def test_route_functions_importable(self):
        from opencut.routes.audio_advanced_routes import (
            route_audio_visualizer,
            route_parse_voice_command,
            route_podcast_polish,
        )
        # All route functions should be callable
        self.assertTrue(callable(route_podcast_polish))
        self.assertTrue(callable(route_audio_visualizer))
        self.assertTrue(callable(route_parse_voice_command))

    def test_blueprint_registered(self):
        """Verify audio_adv_bp is included in register_blueprints."""
        import opencut.routes
        source = inspect.getsource(opencut.routes.register_blueprints)
        self.assertIn("audio_adv_bp", source)
        self.assertIn("audio_advanced_routes", source)

    def test_route_count(self):
        """Verify expected number of routes in the blueprint."""
        from opencut.routes.audio_advanced_routes import audio_adv_bp
        rules = list(audio_adv_bp.deferred_functions)
        # 22 route functions registered
        self.assertGreaterEqual(len(rules), 20)


if __name__ == "__main__":
    unittest.main()
