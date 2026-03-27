"""
Tests for remaining untested OpenCut core modules (batch 2).

Covers: animated_captions, audio_enhance, audio_pro, audio_suite,
        broll_generate, broll_insert, caption_burnin, captions_enhanced,
        color_management, emotion_highlights, face_swap, face_tools,
        lut_library, motion_graphics, multimodal_diarize, music_ai,
        music_gen, object_removal, particles, shorts_pipeline,
        social_post, style_transfer, styled_captions, transitions_3d,
        upscale_pro, video_ai, voice_gen, zoom.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# TestAnimatedCaptions
# ============================================================
class TestAnimatedCaptions(unittest.TestCase):
    """Tests for animated_captions.py — pure Python helpers."""

    def test_get_animation_presets_returns_list(self):
        from opencut.core.animated_captions import get_animation_presets
        presets = get_animation_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)
        for p in presets:
            self.assertIn("name", p)
            self.assertIn("label", p)

    def test_group_words_into_lines_basic(self):
        from opencut.core.animated_captions import _group_words_into_lines
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
            {"word": "foo", "start": 1.0, "end": 1.5},
            {"word": "bar", "start": 1.5, "end": 2.0},
            {"word": "baz", "start": 2.0, "end": 2.5},
        ]
        lines = _group_words_into_lines(words, max_per_line=3)
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0]["words"]), 3)
        self.assertEqual(len(lines[1]["words"]), 2)

    def test_group_words_into_lines_empty(self):
        from opencut.core.animated_captions import _group_words_into_lines
        lines = _group_words_into_lines([], max_per_line=5)
        self.assertEqual(lines, [])

    def test_animation_presets_constant_has_pop(self):
        from opencut.core.animated_captions import ANIMATION_PRESETS
        self.assertIn("pop", ANIMATION_PRESETS)
        self.assertIn("fade", ANIMATION_PRESETS)


# ============================================================
# TestAudioEnhance
# ============================================================
class TestAudioEnhance(unittest.TestCase):
    """Tests for audio_enhance.py."""

    def test_is_video_detects_mp4(self):
        from opencut.core.audio_enhance import _is_video
        self.assertTrue(_is_video("/path/to/file.mp4"))

    def test_is_video_detects_wav_as_audio(self):
        from opencut.core.audio_enhance import _is_video
        self.assertFalse(_is_video("/path/to/file.wav"))

    @patch("subprocess.run")
    def test_extract_audio_calls_ffmpeg(self, mock_run):
        from opencut.core.audio_enhance import _extract_audio
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a non-empty file so the size check passes
            f.write(b"RIFF" + b"\x00" * 100)
            out_path = f.name
        try:
            _extract_audio("/input/video.mp4", out_path)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], "ffmpeg")
            self.assertIn("-vn", cmd)
        finally:
            os.unlink(out_path)

    @patch("subprocess.run")
    def test_extract_audio_raises_on_failure(self, mock_run):
        from opencut.core.audio_enhance import _extract_audio
        mock_run.return_value = MagicMock(returncode=1, stderr="error info")
        with self.assertRaises(RuntimeError):
            _extract_audio("/input/video.mp4", "/tmp/out.wav")


# ============================================================
# TestAudioPro
# ============================================================
class TestAudioPro(unittest.TestCase):
    """Tests for audio_pro.py — pedalboard/deepfilter effects."""

    def test_get_pedalboard_effects_returns_list(self):
        from opencut.core.audio_pro import get_pedalboard_effects
        effects = get_pedalboard_effects()
        self.assertIsInstance(effects, list)
        self.assertGreater(len(effects), 0)
        for e in effects:
            self.assertIn("name", e)
            self.assertIn("label", e)

    def test_pedalboard_effects_constant(self):
        from opencut.core.audio_pro import PEDALBOARD_EFFECTS
        self.assertIn("compressor", PEDALBOARD_EFFECTS)
        self.assertIsInstance(PEDALBOARD_EFFECTS["compressor"], dict)

    def test_check_pedalboard_returns_bool(self):
        from opencut.core.audio_pro import check_pedalboard_available
        result = check_pedalboard_available()
        self.assertIsInstance(result, bool)

    def test_check_deepfilter_returns_bool(self):
        from opencut.core.audio_pro import check_deepfilter_available
        result = check_deepfilter_available()
        self.assertIsInstance(result, bool)


# ============================================================
# TestAudioSuite
# ============================================================
class TestAudioSuite(unittest.TestCase):
    """Tests for audio_suite.py — FFmpeg audio processing."""

    def test_loudness_presets_has_youtube(self):
        from opencut.core.audio_suite import LOUDNESS_PRESETS
        self.assertIn("youtube", LOUDNESS_PRESETS)
        self.assertIn("podcast", LOUDNESS_PRESETS)
        yt = LOUDNESS_PRESETS["youtube"]
        self.assertIn("i", yt)
        self.assertIn("tp", yt)

    def test_loudness_info_dataclass_defaults(self):
        from opencut.core.audio_suite import LoudnessInfo
        info = LoudnessInfo()
        self.assertEqual(info.input_i, -24.0)
        self.assertEqual(info.input_tp, -1.0)

    def test_beat_info_dataclass_defaults(self):
        from opencut.core.audio_suite import BeatInfo
        info = BeatInfo()
        self.assertEqual(info.bpm, 120.0)
        self.assertEqual(info.beat_times, [])
        self.assertEqual(info.confidence, 0.0)

    @patch("subprocess.run")
    def test_denoise_audio_unknown_method_raises(self, mock_run):
        from opencut.core.audio_suite import denoise_audio
        with self.assertRaises(ValueError) as ctx:
            denoise_audio("/input.wav", method="nonexistent")
        self.assertIn("Unknown denoise method", str(ctx.exception))

    @patch("subprocess.run")
    def test_denoise_audio_calls_ffmpeg(self, mock_run):
        from opencut.core.audio_suite import denoise_audio
        mock_run.return_value = MagicMock(returncode=0)
        result = denoise_audio("/input.wav", output_path="/output.wav", method="afftdn")
        self.assertEqual(result, "/output.wav")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("afftdn", cmd[cmd.index("-af") + 1])

    @patch("subprocess.run")
    def test_denoise_audio_raises_on_ffmpeg_failure(self, mock_run):
        from opencut.core.audio_suite import denoise_audio
        mock_run.return_value = MagicMock(
            returncode=1, stderr=b"ffmpeg error"
        )
        with self.assertRaises(RuntimeError):
            denoise_audio("/input.wav", output_path="/output.wav")

    @patch("subprocess.run")
    def test_measure_loudness_parses_json(self, mock_run):
        from opencut.core.audio_suite import measure_loudness
        json_block = json.dumps({
            "input_i": "-18.5",
            "input_tp": "-0.8",
            "input_lra": "9.2",
            "input_thresh": "-28.5",
            "target_offset": "4.5",
        })
        mock_run.return_value = MagicMock(
            returncode=0, stderr=f"some output\n{json_block}\n"
        )
        info = measure_loudness("/input.wav")
        self.assertAlmostEqual(info.input_i, -18.5)
        self.assertAlmostEqual(info.input_tp, -0.8)

    def test_get_available_effects_returns_list(self):
        from opencut.core.audio_suite import get_available_effects
        effects = get_available_effects()
        self.assertIsInstance(effects, list)
        self.assertGreater(len(effects), 0)

    def test_audio_effects_constant(self):
        from opencut.core.audio_suite import AUDIO_EFFECTS
        self.assertIn("reverb", AUDIO_EFFECTS)


# ============================================================
# TestCaptionBurnin
# ============================================================
class TestCaptionBurnin(unittest.TestCase):
    """Tests for caption_burnin.py — subtitle burn-in."""

    def test_format_ass_time(self):
        from opencut.core.caption_burnin import _format_ass_time
        self.assertEqual(_format_ass_time(0.0), "0:00:00.00")
        self.assertEqual(_format_ass_time(65.5), "0:01:05.50")
        self.assertEqual(_format_ass_time(3661.23), "1:01:01.23")

    def test_format_ass_time_negative_clamped(self):
        from opencut.core.caption_burnin import _format_ass_time
        result = _format_ass_time(-5.0)
        self.assertEqual(result, "0:00:00.00")

    def test_get_burnin_styles_returns_list(self):
        from opencut.core.caption_burnin import get_burnin_styles
        styles = get_burnin_styles()
        self.assertIsInstance(styles, list)
        self.assertGreater(len(styles), 0)
        for s in styles:
            self.assertIn("name", s)
            self.assertIn("label", s)

    def test_burnin_styles_constant(self):
        from opencut.core.caption_burnin import BURNIN_STYLES
        self.assertIn("default", BURNIN_STYLES)


# ============================================================
# TestCaptionsEnhanced
# ============================================================
class TestCaptionsEnhanced(unittest.TestCase):
    """Tests for captions_enhanced.py — WhisperX, NLLB, pysubs2."""

    def test_check_whisperx_returns_bool(self):
        from opencut.core.captions_enhanced import check_whisperx_available
        self.assertIsInstance(check_whisperx_available(), bool)

    def test_check_nllb_returns_bool(self):
        from opencut.core.captions_enhanced import check_nllb_available
        self.assertIsInstance(check_nllb_available(), bool)

    def test_check_pysubs2_returns_bool(self):
        from opencut.core.captions_enhanced import check_pysubs2_available
        self.assertIsInstance(check_pysubs2_available(), bool)

    def test_nllb_languages_mapping(self):
        from opencut.core.captions_enhanced import NLLB_LANGUAGES
        self.assertIn("en", NLLB_LANGUAGES)
        self.assertEqual(NLLB_LANGUAGES["en"], "eng_Latn")

    def test_translation_languages_list(self):
        from opencut.core.captions_enhanced import TRANSLATION_LANGUAGES
        self.assertIsInstance(TRANSLATION_LANGUAGES, list)
        codes = [lang["code"] for lang in TRANSLATION_LANGUAGES]
        self.assertIn("en", codes)
        self.assertIn("es", codes)

    def test_parse_ass_color(self):
        from opencut.core.captions_enhanced import _parse_ass_color
        # ASS color format: &HAABBGGRR
        r, g, b, a = _parse_ass_color("&H00FFFFFF")
        self.assertEqual(r, 255)
        self.assertEqual(g, 255)
        self.assertEqual(b, 255)


# ============================================================
# TestColorManagement
# ============================================================
class TestColorManagement(unittest.TestCase):
    """Tests for color_management.py — FFmpeg colorspace tools."""

    def test_color_spaces_constant(self):
        from opencut.core.color_management import COLOR_SPACES
        self.assertIn("srgb", COLOR_SPACES)
        self.assertIn("rec709", COLOR_SPACES)
        for cs in COLOR_SPACES.values():
            self.assertIn("label", cs)
            self.assertIn("matrix", cs)

    def test_check_ocio_returns_bool(self):
        from opencut.core.color_management import check_ocio_available
        self.assertIsInstance(check_ocio_available(), bool)

    @patch("opencut.core.color_management.run_ffmpeg")
    def test_convert_colorspace_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.color_management import convert_colorspace
        result = convert_colorspace("/input.mp4", target="rec709", output_path="/output.mp4")
        self.assertEqual(result, "/output.mp4")
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        self.assertIn("colorspace", " ".join(cmd))

    def test_get_color_capabilities_returns_dict(self):
        from opencut.core.color_management import get_color_capabilities
        caps = get_color_capabilities()
        self.assertIsInstance(caps, dict)


# ============================================================
# TestEmotionHighlights
# ============================================================
class TestEmotionHighlights(unittest.TestCase):
    """Tests for emotion_highlights.py — emotion peak detection."""

    def test_emotion_peak_dataclass(self):
        from opencut.core.emotion_highlights import EmotionPeak
        peak = EmotionPeak(time=5.0, duration=2.0, emotion="happy", intensity=0.9)
        self.assertEqual(peak.time, 5.0)
        self.assertEqual(peak.emotion, "happy")

    def test_emotion_curve_dataclass_defaults(self):
        from opencut.core.emotion_highlights import EmotionCurve
        curve = EmotionCurve()
        self.assertEqual(curve.samples, [])
        self.assertEqual(curve.peaks, [])
        self.assertEqual(curve.avg_intensity, 0.0)

    def test_check_deepface_returns_bool(self):
        from opencut.core.emotion_highlights import check_deepface_available
        self.assertIsInstance(check_deepface_available(), bool)


# ============================================================
# TestFaceSwap
# ============================================================
class TestFaceSwap(unittest.TestCase):
    """Tests for face_swap.py — insightface/GFPGAN face operations."""

    def test_check_insightface_returns_bool(self):
        from opencut.core.face_swap import check_insightface_available
        self.assertIsInstance(check_insightface_available(), bool)

    def test_check_gfpgan_returns_bool(self):
        from opencut.core.face_swap import check_gfpgan_available
        self.assertIsInstance(check_gfpgan_available(), bool)

    def test_get_face_capabilities_returns_dict(self):
        from opencut.core.face_swap import get_face_capabilities
        caps = get_face_capabilities()
        self.assertIsInstance(caps, dict)
        self.assertIn("insightface", caps)
        self.assertIn("gfpgan", caps)


# ============================================================
# TestFaceTools
# ============================================================
class TestFaceTools(unittest.TestCase):
    """Tests for face_tools.py — face detection/blur."""

    def test_check_mediapipe_returns_bool(self):
        from opencut.core.face_tools import check_mediapipe_available
        self.assertIsInstance(check_mediapipe_available(), bool)

    def test_check_face_tools_returns_dict(self):
        from opencut.core.face_tools import check_face_tools_available
        caps = check_face_tools_available()
        self.assertIsInstance(caps, dict)

    def test_check_insightface_returns_bool(self):
        from opencut.core.face_tools import check_insightface_available
        self.assertIsInstance(check_insightface_available(), bool)


# ============================================================
# TestLutLibrary
# ============================================================
class TestLutLibrary(unittest.TestCase):
    """Tests for lut_library.py — LUT generation and application."""

    def test_clamp(self):
        from opencut.core.lut_library import _clamp
        self.assertEqual(_clamp(0.5), 0.5)
        self.assertEqual(_clamp(-0.1), 0.0)
        self.assertEqual(_clamp(1.5), 1.0)
        self.assertEqual(_clamp(0.5, 0.2, 0.8), 0.5)
        self.assertEqual(_clamp(0.1, 0.2, 0.8), 0.2)

    def test_builtin_luts_constant(self):
        from opencut.core.lut_library import BUILTIN_LUTS
        self.assertIn("teal_orange", BUILTIN_LUTS)
        self.assertIn("vintage_warm", BUILTIN_LUTS)
        for name, lut in BUILTIN_LUTS.items():
            self.assertIn("label", lut)
            self.assertIn("description", lut)

    def test_get_lut_list_returns_list(self):
        from opencut.core.lut_library import get_lut_list
        luts = get_lut_list()
        self.assertIsInstance(luts, list)
        self.assertGreater(len(luts), 0)

    def test_teal_orange_transform(self):
        from opencut.core.lut_library import _teal_orange
        r, g, b = _teal_orange(0.5, 0.5, 0.5)
        self.assertIsInstance(r, float)
        self.assertIsInstance(g, float)
        self.assertIsInstance(b, float)

    def test_sepia_transform(self):
        from opencut.core.lut_library import _sepia
        r, g, b = _sepia(0.5, 0.5, 0.5)
        self.assertTrue(0.0 <= r <= 2.0)  # May exceed 1.0 before clamp
        self.assertTrue(0.0 <= g <= 2.0)

    def test_high_contrast_bw_transform(self):
        from opencut.core.lut_library import _high_contrast_bw
        r, g, b = _high_contrast_bw(0.5, 0.5, 0.5)
        # B&W means all channels should be equal
        self.assertAlmostEqual(r, g, places=5)
        self.assertAlmostEqual(g, b, places=5)


# ============================================================
# TestMotionGraphics
# ============================================================
class TestMotionGraphics(unittest.TestCase):
    """Tests for motion_graphics.py — title cards and overlays."""

    def test_title_presets_constant(self):
        from opencut.core.motion_graphics import TITLE_PRESETS
        self.assertIn("fade_center", TITLE_PRESETS)
        for name, p in TITLE_PRESETS.items():
            self.assertIn("label", p)

    def test_get_title_presets_returns_list(self):
        from opencut.core.motion_graphics import get_title_presets
        presets = get_title_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)

    def test_check_manim_returns_bool(self):
        from opencut.core.motion_graphics import check_manim_available
        self.assertIsInstance(check_manim_available(), bool)

    @patch("opencut.core.motion_graphics.run_ffmpeg")
    def test_render_title_card_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.motion_graphics import render_title_card
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "title.mp4")
            result = render_title_card("Hello World", output_path=out)
            self.assertEqual(result, out)
            mock_ffmpeg.assert_called_once()


# ============================================================
# TestMusicAi
# ============================================================
class TestMusicAi(unittest.TestCase):
    """Tests for music_ai.py — MusicGen/AudioCraft."""

    def test_check_audiocraft_returns_bool(self):
        from opencut.core.music_ai import check_audiocraft_available
        self.assertIsInstance(check_audiocraft_available(), bool)

    def test_check_torch_cuda_returns_bool(self):
        from opencut.core.music_ai import check_torch_cuda
        self.assertIsInstance(check_torch_cuda(), bool)

    def test_musicgen_models_constant(self):
        from opencut.core.music_ai import MUSICGEN_MODELS
        self.assertIsInstance(MUSICGEN_MODELS, list)
        self.assertGreater(len(MUSICGEN_MODELS), 0)
        for m in MUSICGEN_MODELS:
            self.assertIn("name", m)
            self.assertIn("label", m)

    def test_get_music_ai_capabilities_returns_dict(self):
        from opencut.core.music_ai import get_music_ai_capabilities
        caps = get_music_ai_capabilities()
        self.assertIsInstance(caps, dict)


# ============================================================
# TestMusicGen
# ============================================================
class TestMusicGen(unittest.TestCase):
    """Tests for music_gen.py — FFmpeg tone/SFX generation."""

    def test_waveforms_constant(self):
        from opencut.core.music_gen import WAVEFORMS
        self.assertIn("sine", WAVEFORMS)
        self.assertIn("square", WAVEFORMS)
        self.assertIn("triangle", WAVEFORMS)

    def test_sfx_presets_constant(self):
        from opencut.core.music_gen import SFX_PRESETS
        self.assertIn("swoosh", SFX_PRESETS)
        self.assertIn("impact", SFX_PRESETS)

    @patch("opencut.core.music_gen.run_ffmpeg")
    def test_generate_tone_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.music_gen import generate_tone
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "tone.wav")
            result = generate_tone(output_path=out, frequency=440.0, duration=1.0)
            self.assertEqual(result, out)
            mock_ffmpeg.assert_called_once()

    def test_generate_tone_invalid_waveform_raises(self):
        from opencut.core.music_gen import generate_tone
        with self.assertRaises(ValueError) as ctx:
            generate_tone(waveform="nonexistent")
        self.assertIn("Invalid waveform", str(ctx.exception))

    @patch("opencut.core.music_gen.run_ffmpeg")
    def test_generate_sfx_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.music_gen import generate_sfx
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sfx.wav")
            result = generate_sfx(preset="swoosh", output_path=out)
            self.assertEqual(result, out)
            mock_ffmpeg.assert_called_once()

    def test_generate_sfx_invalid_preset_raises(self):
        from opencut.core.music_gen import generate_sfx
        with self.assertRaises(ValueError) as ctx:
            generate_sfx(preset="nonexistent")
        self.assertIn("Unknown SFX preset", str(ctx.exception))

    @patch("opencut.core.music_gen.run_ffmpeg")
    def test_generate_silence_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.music_gen import generate_silence
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "silence.wav")
            result = generate_silence(duration=2.0, output_path=out)
            self.assertEqual(result, out)
            mock_ffmpeg.assert_called_once()

    def test_concatenate_audio_empty_raises(self):
        from opencut.core.music_gen import concatenate_audio
        with self.assertRaises(ValueError):
            concatenate_audio([])

    def test_get_audio_generators_returns_dict(self):
        from opencut.core.music_gen import get_audio_generators
        gens = get_audio_generators()
        self.assertIsInstance(gens, dict)


# ============================================================
# TestObjectRemoval
# ============================================================
class TestObjectRemoval(unittest.TestCase):
    """Tests for object_removal.py — SAM2/ProPainter/delogo."""

    def test_check_sam2_returns_bool(self):
        from opencut.core.object_removal import check_sam2_available
        self.assertIsInstance(check_sam2_available(), bool)

    def test_check_propainter_returns_bool(self):
        from opencut.core.object_removal import check_propainter_available
        self.assertIsInstance(check_propainter_available(), bool)

    def test_check_lama_returns_bool(self):
        from opencut.core.object_removal import check_lama_available
        self.assertIsInstance(check_lama_available(), bool)

    def test_get_removal_capabilities(self):
        from opencut.core.object_removal import get_removal_capabilities
        caps = get_removal_capabilities()
        self.assertIn("sam2", caps)
        self.assertIn("delogo", caps)
        # delogo is always True (FFmpeg built-in)
        self.assertTrue(caps["delogo"])

    @patch("opencut.core.object_removal.run_ffmpeg")
    def test_remove_watermark_delogo_calls_ffmpeg(self, mock_ffmpeg):
        from opencut.core.object_removal import remove_watermark_delogo
        region = {"x": 10, "y": 20, "width": 100, "height": 50}
        result = remove_watermark_delogo(
            "/input.mp4", region, output_path="/output.mp4"
        )
        self.assertEqual(result, "/output.mp4")
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        self.assertIn("delogo", cmd[vf_idx + 1])


# ============================================================
# TestParticles
# ============================================================
class TestParticles(unittest.TestCase):
    """Tests for particles.py — particle overlay effects."""

    def test_particle_presets_constant(self):
        from opencut.core.particles import PARTICLE_PRESETS
        self.assertIn("confetti", PARTICLE_PRESETS)
        for name, p in PARTICLE_PRESETS.items():
            self.assertIn("label", p)

    def test_get_particle_presets_returns_list(self):
        from opencut.core.particles import get_particle_presets
        presets = get_particle_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)

    def test_particle_class_update(self):
        from opencut.core.particles import Particle
        preset_cfg = {"gravity": 0.1, "fade": True, "streak": False}
        p = Particle(x=100, y=100, vx=1.0, vy=-2.0, size=5,
                     color=(255, 0, 0), lifetime=10, preset_cfg=preset_cfg)
        self.assertTrue(p.alive)
        p.update()
        self.assertEqual(p.age, 1)
        self.assertAlmostEqual(p.y, 100 + (-2.0) + 0.1, places=0)

    def test_particle_get_alpha_fading(self):
        from opencut.core.particles import Particle
        preset_cfg = {"gravity": 0, "fade": True, "streak": False}
        p = Particle(x=0, y=0, vx=0, vy=0, size=5,
                     color=(255, 255, 255), lifetime=10, preset_cfg=preset_cfg)
        alpha_start = p.get_alpha()
        p.age = 5
        alpha_mid = p.get_alpha()
        self.assertGreater(alpha_start, alpha_mid)

    def test_particle_dies_at_lifetime(self):
        from opencut.core.particles import Particle
        preset_cfg = {"gravity": 0, "fade": False, "streak": False}
        p = Particle(x=0, y=0, vx=0, vy=0, size=5,
                     color=(255, 255, 255), lifetime=3, preset_cfg=preset_cfg)
        for _ in range(3):
            p.update()
        self.assertFalse(p.alive)


# ============================================================
# TestShortsPipeline
# ============================================================
class TestShortsPipeline(unittest.TestCase):
    """Tests for shorts_pipeline.py — automated shorts workflow."""

    def test_shorts_pipeline_config_defaults(self):
        from opencut.core.shorts_pipeline import ShortsPipelineConfig
        cfg = ShortsPipelineConfig()
        self.assertEqual(cfg.whisper_model, "base")
        self.assertEqual(cfg.max_shorts, 3)
        self.assertEqual(cfg.target_w, 1080)
        self.assertEqual(cfg.target_h, 1920)
        self.assertTrue(cfg.burn_captions)

    def test_short_clip_dataclass(self):
        from opencut.core.shorts_pipeline import ShortClip
        clip = ShortClip(
            index=0, output_path="/out.mp4",
            start=10.0, end=30.0, duration=20.0, title="Test", score=0.8,
        )
        self.assertEqual(clip.duration, 20.0)
        self.assertEqual(clip.title, "Test")

    @patch("subprocess.run")
    def test_probe_duration_returns_float(self, mock_run):
        from opencut.core.shorts_pipeline import _probe_duration
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "125.5"}}),
        )
        dur = _probe_duration("/test.mp4")
        self.assertAlmostEqual(dur, 125.5)

    @patch("subprocess.run")
    def test_probe_duration_returns_zero_on_failure(self, mock_run):
        from opencut.core.shorts_pipeline import _probe_duration
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        dur = _probe_duration("/nonexistent.mp4")
        self.assertEqual(dur, 0.0)


# ============================================================
# TestStyleTransfer
# ============================================================
class TestStyleTransfer(unittest.TestCase):
    """Tests for style_transfer.py — neural style transfer."""

    def test_style_models_constant(self):
        from opencut.core.style_transfer import STYLE_MODELS
        self.assertIn("candy", STYLE_MODELS)
        for name, model in STYLE_MODELS.items():
            self.assertIn("label", model)
            self.assertIn("description", model)
            self.assertIn("filename", model)

    def test_get_available_styles_returns_list(self):
        from opencut.core.style_transfer import get_available_styles
        styles = get_available_styles()
        self.assertIsInstance(styles, list)
        self.assertGreater(len(styles), 0)
        for s in styles:
            self.assertIn("name", s)
            self.assertIn("label", s)
            self.assertIn("downloaded", s)

    def test_models_dir_set(self):
        from opencut.core.style_transfer import MODELS_DIR
        self.assertIn("style_transfer", MODELS_DIR)


# ============================================================
# TestStyledCaptions
# ============================================================
class TestStyledCaptions(unittest.TestCase):
    """Tests for styled_captions.py — YouTube-style captions."""

    def test_styles_constant(self):
        from opencut.core.styled_captions import DEFAULT_STYLE, STYLES
        self.assertIn(DEFAULT_STYLE, STYLES)
        self.assertIn("youtube_bold", STYLES)

    def test_caption_style_dataclass(self):
        from opencut.core.styled_captions import CaptionStyle
        style = CaptionStyle(name="test", label="Test Style",
                             font_file="Arial.ttf", font_size=48,
                             text_color=(255, 255, 255, 255),
                             stroke_color=(0, 0, 0, 255),
                             stroke_width=3,
                             highlight_color=(255, 230, 0, 255),
                             line_spacing=1.2)
        self.assertEqual(style.name, "test")
        self.assertEqual(style.font_size, 48)
        self.assertEqual(style.label, "Test Style")

    def test_get_style_info_returns_list(self):
        from opencut.core.styled_captions import get_style_info
        info = get_style_info()
        self.assertIsInstance(info, list)
        self.assertGreater(len(info), 0)

    def test_action_keywords_set(self):
        from opencut.core.styled_captions import _ACTION_KEYWORDS
        self.assertIn("amazing", _ACTION_KEYWORDS)
        self.assertIn("awesome", _ACTION_KEYWORDS)


# ============================================================
# TestTransitions3d
# ============================================================
class TestTransitions3d(unittest.TestCase):
    """Tests for transitions_3d.py — xfade transitions."""

    def test_xfade_transitions_constant(self):
        from opencut.core.transitions_3d import XFADE_TRANSITIONS
        self.assertIn("fade", XFADE_TRANSITIONS)
        for name, t in XFADE_TRANSITIONS.items():
            self.assertIn("label", t)

    def test_get_transition_list_returns_list(self):
        from opencut.core.transitions_3d import get_transition_list
        tl = get_transition_list()
        self.assertIsInstance(tl, list)
        self.assertGreater(len(tl), 0)

    def test_check_moderngl_returns_bool(self):
        from opencut.core.transitions_3d import check_moderngl_available
        self.assertIsInstance(check_moderngl_available(), bool)

    @patch("opencut.core.transitions_3d._has_audio_stream", return_value=False)
    @patch("opencut.core.transitions_3d.get_video_info", return_value={"duration": 10.0})
    @patch("opencut.core.transitions_3d.run_ffmpeg")
    def test_apply_transition_calls_ffmpeg(self, mock_ffmpeg, mock_info, mock_audio):
        from opencut.core.transitions_3d import apply_transition
        result = apply_transition(
            "/clip_a.mp4", "/clip_b.mp4",
            output_path="/output.mp4", transition="fade",
        )
        self.assertEqual(result, "/output.mp4")
        mock_ffmpeg.assert_called_once()

    @patch("opencut.core.transitions_3d._has_audio_stream", return_value=False)
    @patch("opencut.core.transitions_3d.get_video_info", return_value={"duration": 10.0})
    @patch("opencut.core.transitions_3d.run_ffmpeg")
    def test_apply_transition_unknown_falls_back_to_fade(self, mock_ffmpeg, mock_info, mock_audio):
        from opencut.core.transitions_3d import apply_transition
        apply_transition(
            "/clip_a.mp4", "/clip_b.mp4",
            output_path="/output.mp4", transition="nonexistent_transition",
        )
        cmd = mock_ffmpeg.call_args[0][0]
        fc_idx = cmd.index("-filter_complex")
        self.assertIn("fade", cmd[fc_idx + 1])


# ============================================================
# TestUpscalePro
# ============================================================
class TestUpscalePro(unittest.TestCase):
    """Tests for upscale_pro.py — video upscaling."""

    def test_check_realesrgan_returns_bool(self):
        from opencut.core.upscale_pro import check_realesrgan_available
        self.assertIsInstance(check_realesrgan_available(), bool)

    def test_check_video2x_returns_bool(self):
        from opencut.core.upscale_pro import check_video2x_available
        self.assertIsInstance(check_video2x_available(), bool)

    def test_upscale_presets_constant(self):
        from opencut.core.upscale_pro import UPSCALE_PRESETS
        self.assertIn("fast", UPSCALE_PRESETS)
        self.assertIn("label", UPSCALE_PRESETS["fast"])

    def test_get_upscale_capabilities_returns_dict(self):
        from opencut.core.upscale_pro import get_upscale_capabilities
        caps = get_upscale_capabilities()
        self.assertIsInstance(caps, dict)

    @patch("opencut.core.upscale_pro.get_video_info",
           return_value={"width": 640, "height": 480})
    @patch("opencut.core.upscale_pro.run_ffmpeg")
    def test_upscale_lanczos_calls_ffmpeg(self, mock_ffmpeg, mock_info):
        from opencut.core.upscale_pro import upscale_lanczos
        result = upscale_lanczos("/input.mp4", scale=2, output_path="/output.mp4")
        self.assertEqual(result, "/output.mp4")
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        self.assertIn("1280", cmd[vf_idx + 1])  # 640*2
        self.assertIn("960", cmd[vf_idx + 1])    # 480*2


# ============================================================
# TestVideoAi
# ============================================================
class TestVideoAi(unittest.TestCase):
    """Tests for video_ai.py — AI upscale, bg removal, frame interp."""

    def test_check_upscale_returns_bool(self):
        from opencut.core.video_ai import check_upscale_available
        self.assertIsInstance(check_upscale_available(), bool)

    def test_check_rembg_returns_bool(self):
        from opencut.core.video_ai import check_rembg_available
        self.assertIsInstance(check_rembg_available(), bool)

    def test_check_rife_returns_bool(self):
        from opencut.core.video_ai import check_rife_available
        self.assertIsInstance(check_rife_available(), bool)

    @patch("subprocess.run")
    def test_count_frames_returns_int(self, mock_run):
        from opencut.core.video_ai import _count_frames
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"1500\n"
        )
        count = _count_frames("/test.mp4")
        self.assertEqual(count, 1500)

    @patch("subprocess.run")
    def test_count_frames_returns_zero_on_failure(self, mock_run):
        from opencut.core.video_ai import _count_frames
        mock_run.return_value = MagicMock(
            returncode=1, stdout=b""
        )
        count = _count_frames("/test.mp4")
        self.assertEqual(count, 0)

    def test_get_ai_capabilities_returns_dict(self):
        from opencut.core.video_ai import get_ai_capabilities
        caps = get_ai_capabilities()
        self.assertIsInstance(caps, dict)


# ============================================================
# TestVoiceGen
# ============================================================
class TestVoiceGen(unittest.TestCase):
    """Tests for voice_gen.py — TTS voice generation."""

    def test_check_edge_tts_returns_bool(self):
        from opencut.core.voice_gen import check_edge_tts_available
        self.assertIsInstance(check_edge_tts_available(), bool)

    def test_check_kokoro_returns_bool(self):
        from opencut.core.voice_gen import check_kokoro_available
        self.assertIsInstance(check_kokoro_available(), bool)

    def test_edge_voices_constant(self):
        from opencut.core.voice_gen import EDGE_VOICES
        self.assertIsInstance(EDGE_VOICES, list)
        self.assertGreater(len(EDGE_VOICES), 0)
        for v in EDGE_VOICES:
            self.assertIn("id", v)
            self.assertIn("label", v)

    def test_get_voice_list_edge(self):
        from opencut.core.voice_gen import get_voice_list
        voices = get_voice_list(engine="edge")
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)

    def test_get_voice_list_unknown_engine_returns_empty(self):
        from opencut.core.voice_gen import get_voice_list
        voices = get_voice_list(engine="nonexistent")
        self.assertEqual(voices, [])

    def test_check_chatterbox_returns_bool(self):
        from opencut.core.voice_gen import check_chatterbox_available
        self.assertIsInstance(check_chatterbox_available(), bool)


# ============================================================
# TestZoom
# ============================================================
class TestZoom(unittest.TestCase):
    """Tests for zoom.py — auto-zoom keyframe generation."""

    def test_zoom_keyframe_dataclass(self):
        from opencut.core.zoom import ZoomKeyframe
        kf = ZoomKeyframe(time=1.0, scale=1.3)
        self.assertEqual(kf.time, 1.0)
        self.assertEqual(kf.scale, 1.3)
        self.assertEqual(kf.anchor_x, 0.5)

    def test_zoom_event_to_keyframes(self):
        from opencut.core.zoom import ZoomEvent
        event = ZoomEvent(start=1.0, peak=2.0, end=3.0, max_scale=1.3)
        kfs = event.to_keyframes()
        self.assertEqual(len(kfs), 3)
        self.assertEqual(kfs[0].scale, 1.0)  # Start at 1x
        self.assertEqual(kfs[1].scale, 1.3)  # Peak
        self.assertEqual(kfs[2].scale, 1.0)  # Back to 1x

    def test_zoom_event_duration(self):
        from opencut.core.zoom import ZoomEvent
        event = ZoomEvent(start=1.0, peak=2.0, end=4.0, max_scale=1.2)
        self.assertAlmostEqual(event.duration, 3.0)

    def test_zoom_events_to_keyframes_sorted(self):
        from opencut.core.zoom import ZoomEvent, zoom_events_to_keyframes
        events = [
            ZoomEvent(start=5.0, peak=6.0, end=7.0, max_scale=1.2),
            ZoomEvent(start=1.0, peak=2.0, end=3.0, max_scale=1.3),
        ]
        kfs = zoom_events_to_keyframes(events)
        times = [kf.time for kf in kfs]
        self.assertEqual(times, sorted(times))

    @patch("opencut.core.zoom.find_emphasis_points", side_effect=RuntimeError("No audio"))
    def test_generate_zoom_events_handles_error(self, mock_emphasis):
        from opencut.core.zoom import generate_zoom_events
        events = generate_zoom_events("/test.mp4")
        self.assertEqual(events, [])

    def test_filter_to_speech(self):
        from opencut.core.silence import TimeSegment
        from opencut.core.zoom import _filter_to_speech
        emphasis = [
            TimeSegment(start=1.0, end=1.5),
            TimeSegment(start=5.0, end=5.5),
            TimeSegment(start=10.0, end=10.5),
        ]
        speech = [
            TimeSegment(start=0.0, end=3.0),
            TimeSegment(start=9.0, end=11.0),
        ]
        filtered = _filter_to_speech(emphasis, speech)
        # Only emphasis at 1.0 and 10.0 fall within speech
        self.assertEqual(len(filtered), 2)


# ============================================================
# TestBRollInsert
# ============================================================
class TestBRollInsert(unittest.TestCase):
    """Tests for broll_insert.py — B-roll insertion point analysis."""

    def test_broll_window_is_valid(self):
        from opencut.core.broll_insert import BRollWindow
        valid = BRollWindow(start=1.0, end=3.0, duration=2.0,
                            reason="pause", keywords=["camera"])
        self.assertTrue(valid.is_valid)
        invalid = BRollWindow(start=1.0, end=1.2, duration=0.2,
                              reason="pause", keywords=[])
        self.assertFalse(invalid.is_valid)

    def test_analyze_broll_empty_transcript(self):
        from opencut.core.broll_insert import analyze_broll_opportunities
        plan = analyze_broll_opportunities([])
        self.assertEqual(plan.windows, [])
        self.assertEqual(plan.total_windows, 0)

    def test_analyze_broll_finds_gap(self):
        from opencut.core.broll_insert import analyze_broll_opportunities
        segments = [
            {"text": "We are talking about cameras and lenses", "start": 0, "end": 5},
            {"text": "Now let us discuss the results", "start": 8, "end": 12},
        ]
        plan = analyze_broll_opportunities(segments, min_gap=1.0)
        # There's a 3-second gap between segments — should be detected
        gap_windows = [w for w in plan.windows if w.reason == "dialogue_pause"]
        self.assertGreater(len(gap_windows), 0)

    def test_extract_keywords(self):
        from opencut.core.broll_insert import _extract_keywords
        keywords = _extract_keywords("We are talking about cameras and photography today")
        self.assertIn("cameras", keywords)
        self.assertIn("photography", keywords)
        # Stopwords should be filtered
        self.assertNotIn("are", keywords)
        self.assertNotIn("the", keywords)

    def test_find_visual_references(self):
        from opencut.core.broll_insert import _find_visual_references
        refs = _find_visual_references("Look at this beautiful sunset over the ocean")
        self.assertIn("sunset", refs)
        self.assertIn("ocean", refs)

    def test_deduplicate_windows(self):
        from opencut.core.broll_insert import BRollWindow, _deduplicate_windows
        windows = [
            BRollWindow(start=1.0, end=3.0, duration=2.0,
                        reason="a", keywords=[], score=0.5),
            BRollWindow(start=2.0, end=4.0, duration=2.0,
                        reason="b", keywords=[], score=0.8),
        ]
        deduped = _deduplicate_windows(windows)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].score, 0.8)  # Higher score kept


# ============================================================
# TestBRollGenerate
# ============================================================
class TestBRollGenerate(unittest.TestCase):
    """Tests for broll_generate.py — AI video generation."""

    def test_generated_clip_dataclass(self):
        from opencut.core.broll_generate import GeneratedClip
        clip = GeneratedClip(
            output_path="/out.mp4", prompt="sunset", duration=4.0,
            resolution="720x480", backend="cogvideox",
            generation_time=30.0, seed=42,
        )
        self.assertEqual(clip.prompt, "sunset")
        self.assertEqual(clip.backend, "cogvideox")

    def test_backends_constant(self):
        from opencut.core.broll_generate import BACKENDS
        self.assertIn("cogvideox", BACKENDS)
        self.assertIn("wan", BACKENDS)

    def test_check_functions_return_bool(self):
        from opencut.core.broll_generate import (
            check_broll_generate_available,
            check_cogvideox_available,
            check_hunyuan_available,
            check_svd_available,
            check_wan_available,
        )
        for fn in [check_cogvideox_available, check_wan_available,
                    check_hunyuan_available, check_svd_available,
                    check_broll_generate_available]:
            self.assertIsInstance(fn(), bool)

    def test_get_available_backends_returns_list(self):
        from opencut.core.broll_generate import get_available_backends
        backends = get_available_backends()
        self.assertIsInstance(backends, list)


# ============================================================
# TestMultimodalDiarize
# ============================================================
class TestMultimodalDiarize(unittest.TestCase):
    """Tests for multimodal_diarize.py — audio+face diarization."""

    def test_face_segment_dataclass(self):
        from opencut.core.multimodal_diarize import FaceSegment
        seg = FaceSegment(face_id=0, start=1.0, end=5.0, confidence=0.95)
        self.assertAlmostEqual(seg.duration, 4.0)

    def test_speaker_face_mapping_dataclass(self):
        from opencut.core.multimodal_diarize import SpeakerFaceMapping
        m = SpeakerFaceMapping(speaker="SPEAKER_00", face_id=1,
                               confidence=0.85, overlap_seconds=12.0)
        self.assertEqual(m.speaker, "SPEAKER_00")
        self.assertEqual(m.face_id, 1)

    def test_multimodal_result_get_speaker_face(self):
        from opencut.core.multimodal_diarize import (
            MultimodalDiarizationResult,
            SpeakerFaceMapping,
        )
        result = MultimodalDiarizationResult(
            speaker_segments=[],
            face_segments=[],
            mappings=[
                SpeakerFaceMapping("SPEAKER_00", face_id=2,
                                   confidence=0.9, overlap_seconds=10.0),
                SpeakerFaceMapping("SPEAKER_01", face_id=3,
                                   confidence=0.7, overlap_seconds=5.0),
            ],
        )
        self.assertEqual(result.get_speaker_face("SPEAKER_00"), 2)
        self.assertEqual(result.get_speaker_face("SPEAKER_01"), 3)
        self.assertIsNone(result.get_speaker_face("SPEAKER_99"))


# ============================================================
# TestSocialPost
# ============================================================
class TestSocialPost(unittest.TestCase):
    """Tests for social_post.py — social media upload."""

    def test_platform_limits_constant(self):
        from opencut.core.social_post import PLATFORM_LIMITS
        self.assertIn("youtube", PLATFORM_LIMITS)
        self.assertIn("tiktok", PLATFORM_LIMITS)
        self.assertIn("instagram", PLATFORM_LIMITS)
        for p, limits in PLATFORM_LIMITS.items():
            self.assertIn("max_file_size_mb", limits)
            self.assertIn("supported_formats", limits)

    def test_upload_result_dataclass(self):
        from opencut.core.social_post import UploadResult
        r = UploadResult(platform="youtube", success=True,
                         video_id="abc123", url="https://youtube.com/watch?v=abc123")
        self.assertTrue(r.success)
        self.assertEqual(r.platform, "youtube")

    def test_platform_auth_is_expired(self):
        import time

        from opencut.core.social_post import PlatformAuth
        # Expired token
        auth = PlatformAuth(platform="youtube", access_token="tok",
                            expires_at=time.time() - 100)
        self.assertTrue(auth.is_expired)
        # Fresh token
        auth2 = PlatformAuth(platform="youtube", access_token="tok",
                             expires_at=time.time() + 3600)
        self.assertFalse(auth2.is_expired)
        # No expiry
        auth3 = PlatformAuth(platform="youtube", access_token="tok",
                             expires_at=0)
        self.assertFalse(auth3.is_expired)

    @patch("opencut.core.social_post._load_credentials", return_value={})
    def test_get_connected_platforms_empty(self, mock_load):
        from opencut.core.social_post import get_connected_platforms
        result = get_connected_platforms()
        self.assertEqual(result, [])

    def test_validate_upload_file_not_found(self):
        from opencut.core.social_post import _validate_upload
        err = _validate_upload("/nonexistent/file.mp4", "youtube")
        self.assertIsNotNone(err)
        self.assertIn("not found", err)

    def test_validate_upload_unknown_platform(self):
        from opencut.core.social_post import _validate_upload
        # Create a temp file so file-not-found check passes
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * 100)
            path = f.name
        try:
            err = _validate_upload(path, "unknown_platform")
            self.assertIsNotNone(err)
            self.assertIn("Unknown platform", err)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
