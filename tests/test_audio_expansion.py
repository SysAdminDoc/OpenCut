"""
Tests for OpenCut Audio Expansion features.

Covers:
  - AI Sound Effects Generation (2.4)
  - Spatial Audio (2.5)
  - Stem Remix (2.7)
  - SFX Library (19.2)
  - Transition SFX (18.7)
  - Podcast Audio Extract (40.3)
  - Google Fonts Browser (19.3)
  - Audio Expansion Routes (smoke tests)
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# AI Sound Effects (2.4) Tests
# ============================================================
class TestAISFX(unittest.TestCase):
    """Tests for opencut.core.ai_sfx module."""

    def test_detect_sfx_cues_signature(self):
        from opencut.core.ai_sfx import detect_sfx_cues
        sig = inspect.signature(detect_sfx_cues)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("scene_data", sig.parameters)
        self.assertIn("keywords", sig.parameters)
        self.assertIn("on_progress", sig.parameters)
        self.assertIsNone(sig.parameters["scene_data"].default)
        self.assertIsNone(sig.parameters["on_progress"].default)

    def test_generate_sfx_signature(self):
        from opencut.core.ai_sfx import generate_sfx
        sig = inspect.signature(generate_sfx)
        self.assertIn("cue", sig.parameters)
        self.assertIn("sfx_output_path", sig.parameters)
        self.assertIn("use_ai", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_apply_sfx_to_video_signature(self):
        from opencut.core.ai_sfx import apply_sfx_to_video
        sig = inspect.signature(apply_sfx_to_video)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("sfx_list", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_sfx_categories_defined(self):
        from opencut.core.ai_sfx import SFX_CATEGORIES
        self.assertIn("impact", SFX_CATEGORIES)
        self.assertIn("whoosh", SFX_CATEGORIES)
        self.assertIn("click", SFX_CATEGORIES)
        self.assertIn("explosion", SFX_CATEGORIES)
        for cat in SFX_CATEGORIES.values():
            self.assertIn("freq", cat)
            self.assertIn("type", cat)
            self.assertIn("duration", cat)

    def test_action_to_sfx_mapping(self):
        from opencut.core.ai_sfx import ACTION_TO_SFX, SFX_CATEGORIES
        for action, category in ACTION_TO_SFX.items():
            self.assertIn(category, SFX_CATEGORIES,
                          f"ACTION_TO_SFX maps '{action}' to unknown category '{category}'")

    def test_sfx_cue_dataclass(self):
        from opencut.core.ai_sfx import SFXCue
        cue = SFXCue(timestamp=1.5, duration=0.3, category="impact",
                     description="test", confidence=0.9, volume=0.8)
        self.assertEqual(cue.timestamp, 1.5)
        self.assertEqual(cue.category, "impact")
        self.assertEqual(cue.confidence, 0.9)

    def test_sfx_result_dataclass(self):
        from opencut.core.ai_sfx import SFXResult
        r = SFXResult(output_path="/tmp/sfx.wav", category="whoosh",
                      duration=0.5, method="ffmpeg_synth")
        self.assertEqual(r.method, "ffmpeg_synth")

    @patch("opencut.core.ai_sfx.get_video_info", return_value={"duration": 60.0})
    def test_detect_sfx_cues_with_scene_data(self, mock_info):
        from opencut.core.ai_sfx import detect_sfx_cues
        scenes = [
            {"start": 2.0, "end": 5.0, "description": "A man throws a ball"},
            {"start": 10.0, "end": 12.0, "description": "Glass shatters on floor"},
        ]
        cues = detect_sfx_cues("/fake/video.mp4", scene_data=scenes)
        self.assertGreater(len(cues), 0)
        categories = [c.category for c in cues]
        self.assertTrue(any(c in ("whoosh", "glass") for c in categories))

    @patch("opencut.core.ai_sfx.get_video_info", return_value={"duration": 30.0})
    def test_detect_sfx_cues_with_keywords(self, mock_info):
        from opencut.core.ai_sfx import detect_sfx_cues
        cues = detect_sfx_cues("/fake/video.mp4", keywords=["hit", "splash"])
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].category, "impact")
        self.assertEqual(cues[1].category, "water")

    @patch("opencut.core.ai_sfx.get_video_info", return_value={"duration": 30.0})
    def test_detect_sfx_cues_auto_generates_when_no_data(self, mock_info):
        from opencut.core.ai_sfx import detect_sfx_cues
        cues = detect_sfx_cues("/fake/video.mp4")
        self.assertGreater(len(cues), 0)
        for cue in cues:
            self.assertGreater(cue.timestamp, 0)

    @patch("opencut.core.ai_sfx.run_ffmpeg")
    def test_generate_sfx_calls_ffmpeg(self, mock_run):
        from opencut.core.ai_sfx import SFXCue, generate_sfx
        cue = SFXCue(timestamp=1.0, duration=0.5, category="impact",
                     description="test")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            result = generate_sfx(cue, sfx_output_path=out)
            mock_run.assert_called_once()
            self.assertEqual(result.category, "impact")
            self.assertEqual(result.method, "ffmpeg_synth")
        finally:
            if os.path.isfile(out):
                os.unlink(out)

    def test_build_synth_filter_all_types(self):
        from opencut.core.ai_sfx import SFX_CATEGORIES, _build_synth_filter
        for cat_name, cat_info in SFX_CATEGORIES.items():
            filt = _build_synth_filter(cat_name, cat_info["duration"])
            self.assertIsInstance(filt, str)
            self.assertGreater(len(filt), 0)

    def test_apply_sfx_raises_on_empty_list(self):
        from opencut.core.ai_sfx import apply_sfx_to_video
        with self.assertRaises(ValueError):
            apply_sfx_to_video("/fake/video.mp4", sfx_list=[])


# ============================================================
# Spatial Audio (2.5) Tests
# ============================================================
class TestSpatialAudio(unittest.TestCase):
    """Tests for opencut.core.spatial_audio module."""

    def test_detect_channel_layout_signature(self):
        from opencut.core.spatial_audio import detect_channel_layout
        sig = inspect.signature(detect_channel_layout)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_to_binaural_signature(self):
        from opencut.core.spatial_audio import to_binaural
        sig = inspect.signature(to_binaural)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("sofa_path", sig.parameters)
        self.assertIn("gain", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_to_surround_signature(self):
        from opencut.core.spatial_audio import to_surround
        sig = inspect.signature(to_surround)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("channels", sig.parameters)
        self.assertIn("lfe_cutoff", sig.parameters)
        self.assertEqual(sig.parameters["channels"].default, 6)
        self.assertEqual(sig.parameters["lfe_cutoff"].default, 120)

    def test_spatial_result_dataclass(self):
        from opencut.core.spatial_audio import SpatialResult
        r = SpatialResult(output_path="/tmp/out.wav", input_layout="stereo",
                          output_layout="binaural", channels=2, method="sofalizer")
        self.assertEqual(r.output_layout, "binaural")
        self.assertEqual(r.channels, 2)

    @patch("opencut.core.spatial_audio.subprocess.run")
    def test_detect_channel_layout_returns_dict(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [{"channel_layout": "5.1", "channels": 6,
                             "sample_rate": "48000", "codec_name": "aac"}]
            }).encode()
        )
        from opencut.core.spatial_audio import detect_channel_layout
        result = detect_channel_layout("/fake/audio.wav")
        self.assertEqual(result["channel_layout"], "5.1")
        self.assertEqual(result["channels"], 6)
        self.assertEqual(result["sample_rate"], 48000)

    @patch("opencut.core.spatial_audio.subprocess.run")
    def test_detect_channel_layout_fallback_on_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        from opencut.core.spatial_audio import detect_channel_layout
        result = detect_channel_layout("/fake/audio.wav")
        self.assertEqual(result["channel_layout"], "stereo")
        self.assertEqual(result["channels"], 2)

    @patch("opencut.core.spatial_audio.run_ffmpeg")
    @patch("opencut.core.spatial_audio.detect_channel_layout",
           return_value={"channel_layout": "stereo", "channels": 2,
                         "sample_rate": 44100, "codec": "pcm_s16le"})
    def test_to_binaural_crossfeed_fallback(self, mock_detect, mock_run):
        from opencut.core.spatial_audio import to_binaural
        result = to_binaural("/fake/audio.wav", output="/fake/out.wav")
        mock_run.assert_called_once()
        self.assertEqual(result.method, "crossfeed_fallback")
        self.assertEqual(result.output_layout, "binaural")

    def test_to_surround_rejects_invalid_channels(self):
        from opencut.core.spatial_audio import to_surround
        with self.assertRaises(ValueError):
            to_surround("/fake/audio.wav", channels=4)

    @patch("opencut.core.spatial_audio.run_ffmpeg")
    @patch("opencut.core.spatial_audio.detect_channel_layout",
           return_value={"channel_layout": "stereo", "channels": 2,
                         "sample_rate": 44100, "codec": "pcm_s16le"})
    def test_to_surround_51(self, mock_detect, mock_run):
        from opencut.core.spatial_audio import to_surround
        result = to_surround("/fake/audio.wav", channels=6, output="/fake/out.wav")
        mock_run.assert_called_once()
        self.assertEqual(result.output_layout, "5.1")
        self.assertEqual(result.channels, 6)

    @patch("opencut.core.spatial_audio.run_ffmpeg")
    @patch("opencut.core.spatial_audio.detect_channel_layout",
           return_value={"channel_layout": "stereo", "channels": 2,
                         "sample_rate": 44100, "codec": "pcm_s16le"})
    def test_to_surround_71(self, mock_detect, mock_run):
        from opencut.core.spatial_audio import to_surround
        result = to_surround("/fake/audio.wav", channels=8, output="/fake/out.wav")
        mock_run.assert_called_once()
        self.assertEqual(result.output_layout, "7.1")
        self.assertEqual(result.channels, 8)


# ============================================================
# Stem Remix (2.7) Tests
# ============================================================
class TestStemRemix(unittest.TestCase):
    """Tests for opencut.core.stem_remix module."""

    def test_remix_stems_signature(self):
        from opencut.core.stem_remix import remix_stems
        sig = inspect.signature(remix_stems)
        self.assertIn("stem_paths", sig.parameters)
        self.assertIn("effects_config", sig.parameters)
        self.assertIn("output", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_apply_stem_effect_signature(self):
        from opencut.core.stem_remix import apply_stem_effect
        sig = inspect.signature(apply_stem_effect)
        self.assertIn("stem_path", sig.parameters)
        self.assertIn("effect", sig.parameters)
        self.assertIn("output", sig.parameters)

    def test_mix_stems_signature(self):
        from opencut.core.stem_remix import mix_stems
        sig = inspect.signature(mix_stems)
        self.assertIn("stem_paths", sig.parameters)
        self.assertIn("mix_config", sig.parameters)
        self.assertIn("output_file", sig.parameters)

    def test_supported_effects_defined(self):
        from opencut.core.stem_remix import SUPPORTED_EFFECTS
        expected = ["reverb", "compress", "eq_bass_boost", "eq_treble_boost",
                    "eq_mid_cut", "highpass", "lowpass", "normalize",
                    "chorus", "flanger"]
        for eff in expected:
            self.assertIn(eff, SUPPORTED_EFFECTS)
            self.assertIn("params", SUPPORTED_EFFECTS[eff])

    def test_stem_effect_dataclass(self):
        from opencut.core.stem_remix import StemEffect
        e = StemEffect(name="reverb", params={"delay": 80})
        self.assertEqual(e.name, "reverb")
        self.assertEqual(e.params["delay"], 80)

    def test_remix_result_dataclass(self):
        from opencut.core.stem_remix import RemixResult
        r = RemixResult(output_path="/tmp/out.wav", stems_processed=3,
                        effects_applied=5, duration=120.0)
        self.assertEqual(r.stems_processed, 3)

    def test_effect_to_filter_all_effects(self):
        from opencut.core.stem_remix import SUPPORTED_EFFECTS, StemEffect, _effect_to_filter
        for name in SUPPORTED_EFFECTS:
            eff = StemEffect(name=name)
            filt = _effect_to_filter(eff)
            self.assertIsInstance(filt, str)
            self.assertGreater(len(filt), 0)

    def test_effect_to_filter_unknown_raises(self):
        from opencut.core.stem_remix import StemEffect, _effect_to_filter
        with self.assertRaises(ValueError):
            _effect_to_filter(StemEffect(name="nonexistent_effect"))

    def test_apply_stem_effect_missing_file_raises(self):
        from opencut.core.stem_remix import StemEffect, apply_stem_effect
        with self.assertRaises(FileNotFoundError):
            apply_stem_effect("/nonexistent/stem.wav", StemEffect(name="reverb"))

    def test_mix_stems_empty_raises(self):
        from opencut.core.stem_remix import mix_stems
        with self.assertRaises(ValueError):
            mix_stems(stem_paths=[])

    def test_remix_stems_empty_raises(self):
        from opencut.core.stem_remix import remix_stems
        with self.assertRaises(ValueError):
            remix_stems(stem_paths=[], effects_config=[])

    def test_build_pan_filter(self):
        from opencut.core.stem_remix import _build_pan_filter
        # Center pan should have equal gains
        filt = _build_pan_filter(0.0)
        self.assertIn("pan=stereo", filt)
        # Left pan
        filt_left = _build_pan_filter(-1.0)
        self.assertIn("pan=stereo", filt_left)
        # Right pan
        filt_right = _build_pan_filter(1.0)
        self.assertIn("pan=stereo", filt_right)


# ============================================================
# SFX Library (19.2) Tests
# ============================================================
class TestSFXLibrary(unittest.TestCase):
    """Tests for opencut.core.sfx_library module."""

    def test_search_sfx_signature(self):
        from opencut.core.sfx_library import search_sfx
        sig = inspect.signature(search_sfx)
        self.assertIn("query", sig.parameters)
        self.assertIn("filters", sig.parameters)
        self.assertIn("api_key", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_download_sfx_signature(self):
        from opencut.core.sfx_library import download_sfx
        sig = inspect.signature(download_sfx)
        self.assertIn("sfx_id", sig.parameters)
        self.assertIn("output_dir", sig.parameters)
        self.assertIn("api_key", sig.parameters)

    def test_list_cached_sfx_signature(self):
        from opencut.core.sfx_library import list_cached_sfx
        sig = inspect.signature(list_cached_sfx)
        self.assertIn("on_progress", sig.parameters)

    def test_builtin_sfx_catalog(self):
        from opencut.core.sfx_library import BUILTIN_SFX
        self.assertGreaterEqual(len(BUILTIN_SFX), 15)
        for sfx in BUILTIN_SFX:
            self.assertIn("id", sfx)
            self.assertIn("name", sfx)
            self.assertIn("tags", sfx)
            self.assertTrue(sfx["id"].startswith("builtin_"))

    def test_search_sfx_empty_query_raises(self):
        from opencut.core.sfx_library import search_sfx
        with self.assertRaises(ValueError):
            search_sfx("")

    def test_search_sfx_builtin_click(self):
        from opencut.core.sfx_library import search_sfx
        results = search_sfx("click", use_freesound=False)
        self.assertGreater(len(results), 0)
        self.assertTrue(any("click" in r.name.lower() for r in results))

    def test_search_sfx_builtin_with_category_filter(self):
        from opencut.core.sfx_library import search_sfx
        results = search_sfx("water", filters={"category": "nature"},
                             use_freesound=False)
        for r in results:
            self.assertEqual(r.category, "nature")

    def test_search_sfx_builtin_with_duration_filter(self):
        from opencut.core.sfx_library import search_sfx
        results = search_sfx("sound", filters={"max_duration": 0.5},
                             use_freesound=False)
        for r in results:
            self.assertLessEqual(r.duration, 0.5)

    def test_sfx_search_result_dataclass(self):
        from opencut.core.sfx_library import SFXSearchResult
        r = SFXSearchResult(id="test_001", name="Test", tags=["a", "b"],
                            duration=1.0, source="builtin")
        self.assertEqual(r.source, "builtin")
        self.assertEqual(r.id, "test_001")

    def test_download_sfx_unknown_builtin_raises(self):
        from opencut.core.sfx_library import download_sfx
        with self.assertRaises(ValueError):
            download_sfx("builtin_999")

    def test_download_sfx_freesound_without_key_raises(self):
        from opencut.core.sfx_library import download_sfx
        with self.assertRaises(ValueError):
            download_sfx("12345", api_key="")

    def test_list_cached_sfx_returns_list(self):
        from opencut.core.sfx_library import list_cached_sfx
        result = list_cached_sfx()
        self.assertIsInstance(result, list)


# ============================================================
# Transition SFX (18.7) Tests
# ============================================================
class TestTransitionSFX(unittest.TestCase):
    """Tests for opencut.core.transition_sfx module."""

    def test_get_transition_sfx_signature(self):
        from opencut.core.transition_sfx import get_transition_sfx
        sig = inspect.signature(get_transition_sfx)
        self.assertIn("transition_type", sig.parameters)
        self.assertIn("style", sig.parameters)
        self.assertIn("duration", sig.parameters)
        self.assertIn("output", sig.parameters)

    def test_apply_transition_sfx_signature(self):
        from opencut.core.transition_sfx import apply_transition_sfx
        sig = inspect.signature(apply_transition_sfx)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("transitions", sig.parameters)
        self.assertIn("master_volume", sig.parameters)

    def test_list_available_sfx_signature(self):
        from opencut.core.transition_sfx import list_available_sfx
        sig = inspect.signature(list_available_sfx)
        self.assertIn("on_progress", sig.parameters)

    def test_transition_sfx_map_populated(self):
        from opencut.core.transition_sfx import TRANSITION_SFX_MAP
        self.assertGreaterEqual(len(TRANSITION_SFX_MAP), 20)
        expected = ["cut", "wipe", "dissolve", "fade_in", "zoom_in", "glitch",
                    "impact", "spin"]
        for t in expected:
            self.assertIn(t, TRANSITION_SFX_MAP)

    def test_sfx_synthesis_functions_callable(self):
        from opencut.core.transition_sfx import SFX_SYNTHESIS
        for name, fn in SFX_SYNTHESIS.items():
            result = fn(0.5)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_transition_sfx_result_dataclass(self):
        from opencut.core.transition_sfx import TransitionSFXResult
        r = TransitionSFXResult(output_path="/tmp/t.wav", transition_type="wipe",
                                sfx_category="whoosh", style="smooth", duration=0.5)
        self.assertEqual(r.transition_type, "wipe")

    @patch("opencut.core.transition_sfx.run_ffmpeg")
    def test_get_transition_sfx_generates_file(self, mock_run):
        from opencut.core.transition_sfx import get_transition_sfx
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            result = get_transition_sfx("wipe", output=out)
            mock_run.assert_called_once()
            self.assertEqual(result.sfx_category, "whoosh")
        finally:
            if os.path.isfile(out):
                os.unlink(out)

    def test_list_available_sfx_structure(self):
        from opencut.core.transition_sfx import list_available_sfx
        result = list_available_sfx()
        self.assertIn("transition_types", result)
        self.assertIn("sfx_categories", result)
        self.assertIn("total_transitions", result)
        self.assertIn("total_categories", result)
        self.assertGreater(result["total_transitions"], 0)
        self.assertGreater(result["total_categories"], 0)

    def test_apply_transition_sfx_empty_raises(self):
        from opencut.core.transition_sfx import apply_transition_sfx
        with self.assertRaises(ValueError):
            apply_transition_sfx("/fake/video.mp4", transitions=[])

    @patch("opencut.core.transition_sfx.run_ffmpeg")
    def test_get_transition_sfx_unknown_type_defaults(self, mock_run):
        from opencut.core.transition_sfx import get_transition_sfx
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            result = get_transition_sfx("totally_unknown", output=out)
            mock_run.assert_called_once()
            self.assertEqual(result.sfx_category, "whoosh")
        finally:
            if os.path.isfile(out):
                os.unlink(out)


# ============================================================
# Podcast Audio Extract (40.3) Tests
# ============================================================
class TestPodcastExtract(unittest.TestCase):
    """Tests for opencut.core.podcast_extract module."""

    def test_extract_podcast_audio_signature(self):
        from opencut.core.podcast_extract import extract_podcast_audio
        sig = inspect.signature(extract_podcast_audio)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("audio_format", sig.parameters)
        self.assertIn("bitrate", sig.parameters)
        self.assertIn("mono", sig.parameters)
        self.assertIn("target_lufs", sig.parameters)
        self.assertIn("noise_reduce", sig.parameters)
        self.assertIn("trim_silence", sig.parameters)
        self.assertEqual(sig.parameters["audio_format"].default, "mp3")
        self.assertTrue(sig.parameters["mono"].default)
        self.assertEqual(sig.parameters["target_lufs"].default, -16.0)

    def test_add_id3_metadata_signature(self):
        from opencut.core.podcast_extract import add_id3_metadata
        sig = inspect.signature(add_id3_metadata)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("metadata", sig.parameters)
        self.assertIn("artwork_path", sig.parameters)
        self.assertIn("output", sig.parameters)

    def test_supported_formats(self):
        from opencut.core.podcast_extract import SUPPORTED_FORMATS
        expected = {"mp3", "aac", "flac", "wav", "ogg", "opus", "m4a"}
        self.assertEqual(SUPPORTED_FORMATS, expected)

    def test_podcast_audio_result_dataclass(self):
        from opencut.core.podcast_extract import PodcastAudioResult
        r = PodcastAudioResult(output_path="/tmp/pod.mp3", format="mp3",
                               duration=3600.0, sample_rate=44100, channels=1,
                               loudness_lufs=-16.0, noise_reduced=False,
                               file_size_bytes=12345678)
        self.assertEqual(r.format, "mp3")
        self.assertEqual(r.channels, 1)
        self.assertFalse(r.noise_reduced)

    def test_metadata_result_dataclass(self):
        from opencut.core.podcast_extract import MetadataResult
        r = MetadataResult(output_path="/tmp/pod.mp3", metadata_fields=5,
                           has_artwork=True)
        self.assertTrue(r.has_artwork)
        self.assertEqual(r.metadata_fields, 5)

    def test_extract_unsupported_format_raises(self):
        from opencut.core.podcast_extract import extract_podcast_audio
        with self.assertRaises(ValueError):
            extract_podcast_audio("/fake/video.mp4", audio_format="xyz")

    def test_add_id3_metadata_missing_file_raises(self):
        from opencut.core.podcast_extract import add_id3_metadata
        with self.assertRaises(FileNotFoundError):
            add_id3_metadata("/nonexistent/audio.mp3", metadata={"title": "Test"})

    @patch("opencut.core.podcast_extract.run_ffmpeg")
    @patch("opencut.core.podcast_extract.subprocess.run")
    @patch("opencut.core.podcast_extract.get_video_info", return_value={"duration": 600.0})
    @patch("opencut.core.podcast_extract.os.path.getsize", return_value=5000000)
    @patch("opencut.core.podcast_extract.os.path.isfile", return_value=True)
    def test_extract_podcast_audio_runs_pipeline(self, mock_isfile, mock_size,
                                                  mock_info, mock_subprocess,
                                                  mock_run):
        from opencut.core.podcast_extract import extract_podcast_audio
        mock_subprocess.return_value = MagicMock(returncode=0, stderr=b"{}")
        result = extract_podcast_audio("/fake/video.mp4", output="/fake/out.mp3",
                                       audio_format="mp3")
        mock_run.assert_called_once()
        self.assertEqual(result.format, "mp3")
        self.assertEqual(result.channels, 1)
        self.assertEqual(result.loudness_lufs, -16.0)

    def test_default_metadata_keys(self):
        from opencut.core.podcast_extract import DEFAULT_METADATA
        self.assertIn("title", DEFAULT_METADATA)
        self.assertIn("artist", DEFAULT_METADATA)
        self.assertIn("genre", DEFAULT_METADATA)
        self.assertEqual(DEFAULT_METADATA["genre"], "Podcast")


# ============================================================
# Google Fonts Browser (19.3) Tests
# ============================================================
class TestGoogleFonts(unittest.TestCase):
    """Tests for opencut.core.google_fonts module."""

    def test_list_fonts_signature(self):
        from opencut.core.google_fonts import list_fonts
        sig = inspect.signature(list_fonts)
        self.assertIn("category", sig.parameters)
        self.assertIn("api_key", sig.parameters)
        self.assertIn("refresh", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_search_fonts_signature(self):
        from opencut.core.google_fonts import search_fonts
        sig = inspect.signature(search_fonts)
        self.assertIn("query", sig.parameters)
        self.assertIn("category", sig.parameters)
        self.assertIn("api_key", sig.parameters)

    def test_download_font_signature(self):
        from opencut.core.google_fonts import download_font
        sig = inspect.signature(download_font)
        self.assertIn("font_name", sig.parameters)
        self.assertIn("variant", sig.parameters)
        self.assertIn("api_key", sig.parameters)

    def test_font_categories_defined(self):
        from opencut.core.google_fonts import FONT_CATEGORIES
        self.assertIn("serif", FONT_CATEGORIES)
        self.assertIn("sans-serif", FONT_CATEGORIES)
        self.assertIn("monospace", FONT_CATEGORIES)
        self.assertIn("display", FONT_CATEGORIES)
        self.assertIn("handwriting", FONT_CATEGORIES)

    def test_bundled_fonts_populated(self):
        from opencut.core.google_fonts import BUNDLED_FONTS
        self.assertGreaterEqual(len(BUNDLED_FONTS), 30)
        for font in BUNDLED_FONTS:
            self.assertIn("family", font)
            self.assertIn("category", font)
            self.assertIn("variants", font)

    def test_font_info_dataclass(self):
        from opencut.core.google_fonts import FontInfo
        f = FontInfo(family="Roboto", category="sans-serif",
                     variants=["regular", "bold"], is_downloaded=False)
        self.assertEqual(f.family, "Roboto")
        self.assertFalse(f.is_downloaded)

    def test_font_download_result_dataclass(self):
        from opencut.core.google_fonts import FontDownloadResult
        r = FontDownloadResult(family="Roboto", local_dir="/tmp/fonts/Roboto",
                               files=["/tmp/fonts/Roboto/Roboto.ttf"],
                               total_size_bytes=50000)
        self.assertEqual(r.family, "Roboto")
        self.assertEqual(r.total_size_bytes, 50000)

    def test_list_fonts_returns_bundled(self):
        from opencut.core.google_fonts import list_fonts
        fonts = list_fonts()
        self.assertGreater(len(fonts), 0)
        families = [f.family for f in fonts]
        self.assertIn("Roboto", families)

    def test_list_fonts_filter_by_category(self):
        from opencut.core.google_fonts import list_fonts
        monospace = list_fonts(category="monospace")
        self.assertGreater(len(monospace), 0)
        for f in monospace:
            self.assertEqual(f.category, "monospace")

    def test_search_fonts_empty_query_raises(self):
        from opencut.core.google_fonts import search_fonts
        with self.assertRaises(ValueError):
            search_fonts("")

    def test_search_fonts_finds_roboto(self):
        from opencut.core.google_fonts import search_fonts
        results = search_fonts("roboto")
        self.assertGreater(len(results), 0)
        self.assertTrue(any("Roboto" in f.family for f in results))

    def test_search_fonts_partial_match(self):
        from opencut.core.google_fonts import search_fonts
        results = search_fonts("mono")
        families = [f.family for f in results]
        self.assertTrue(any("Mono" in fam for fam in families))

    def test_download_font_empty_name_raises(self):
        from opencut.core.google_fonts import download_font
        with self.assertRaises(ValueError):
            download_font("")


# ============================================================
# Audio Expansion Routes (smoke tests)
# ============================================================
class TestAudioExpansionRoutes(unittest.TestCase):
    """Smoke tests for audio expansion route registration."""

    def test_blueprint_exists(self):
        from opencut.routes.audio_expansion_routes import audio_expand_bp
        self.assertEqual(audio_expand_bp.name, "audio_expand")

    def test_blueprint_registered(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/api/audio/generate-sfx", rules)
        self.assertIn("/api/audio/spatial", rules)
        self.assertIn("/api/audio/stem-remix", rules)
        self.assertIn("/api/sfx/search", rules)
        self.assertIn("/api/sfx/download", rules)
        self.assertIn("/api/audio/transition-sfx", rules)
        self.assertIn("/api/podcast/extract-audio", rules)
        self.assertIn("/api/fonts/list", rules)
        self.assertIn("/api/fonts/download", rules)

    def test_fonts_list_get_route(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/api/fonts/list")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("fonts", data)
        self.assertIn("total", data)
        self.assertIn("categories", data)
        self.assertGreater(data["total"], 0)

    def test_fonts_list_with_category_filter(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/api/fonts/list?category=serif")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for font in data["fonts"]:
            self.assertEqual(font["category"], "serif")

    def test_fonts_list_with_search_query(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/api/fonts/list?query=roboto")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(data["total"], 0)

    def test_sfx_search_no_query_returns_cache(self):
        """POST to /api/sfx/search with no query should return cached SFX."""
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        app = create_app(config=OpenCutConfig())
        app.config["TESTING"] = True
        client = app.test_client()
        # Get CSRF token
        health = client.get("/health").get_json()
        token = health.get("csrf_token", "")
        resp = client.post("/api/sfx/search",
                           json={"query": ""},
                           headers={"X-OpenCut-Token": token,
                                    "Content-Type": "application/json"})
        self.assertIn(resp.status_code, [200, 202])


if __name__ == "__main__":
    unittest.main()
