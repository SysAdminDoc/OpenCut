"""
Tests for OpenCut AI Sound Design & Music features (Category 75).

Covers:
  - Sound Design AI (event detection, SFX mapping, PCM synthesis)
  - Ambient Generator (all presets, intensity scaling, duration)
  - Music Mood Morph (all transforms, keyframe interpolation)
  - Beat Sync Edit (all modes, energy matching)
  - Stem Remix (all presets, per-stem effects, mixing)
  - Sound Music Routes (smoke tests)
"""

import inspect
import os
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Sound Design AI Tests
# ============================================================
class TestSoundDesignConstants(unittest.TestCase):
    """Tests for sound_design_ai constants and categories."""

    def test_sfx_categories_exist(self):
        from opencut.core.sound_design_ai import SFX_CATEGORIES
        self.assertGreaterEqual(len(SFX_CATEGORIES), 12)

    def test_sfx_categories_required_keys(self):
        from opencut.core.sound_design_ai import SFX_CATEGORIES
        for name, info in SFX_CATEGORIES.items():
            self.assertIn("description", info, f"Missing description for {name}")
            self.assertIn("base_freq", info, f"Missing base_freq for {name}")
            self.assertIn("duration", info, f"Missing duration for {name}")
            self.assertIn("synthesis", info, f"Missing synthesis for {name}")

    def test_all_12_categories_present(self):
        from opencut.core.sound_design_ai import SFX_CATEGORIES
        expected = {
            "impact", "whoosh", "ambient", "mechanical", "nature",
            "musical_hit", "riser", "drop", "glitch", "sweep",
            "texture", "stinger",
        }
        self.assertEqual(expected, set(SFX_CATEGORIES.keys()))

    def test_motion_thresholds(self):
        from opencut.core.sound_design_ai import MOTION_THRESHOLDS
        self.assertIn("high", MOTION_THRESHOLDS)
        self.assertIn("medium", MOTION_THRESHOLDS)
        self.assertIn("low", MOTION_THRESHOLDS)
        self.assertGreater(MOTION_THRESHOLDS["high"], MOTION_THRESHOLDS["medium"])
        self.assertGreater(MOTION_THRESHOLDS["medium"], MOTION_THRESHOLDS["low"])

    def test_motion_to_sfx_mapping(self):
        from opencut.core.sound_design_ai import MOTION_TO_SFX
        for intensity in ("high", "medium", "low"):
            self.assertIn(intensity, MOTION_TO_SFX)
            self.assertIsInstance(MOTION_TO_SFX[intensity], list)
            self.assertTrue(len(MOTION_TO_SFX[intensity]) > 0)

    def test_synth_map_covers_all_types(self):
        from opencut.core.sound_design_ai import SFX_CATEGORIES, _SYNTH_MAP
        synthesis_types = set(info["synthesis"] for info in SFX_CATEGORIES.values())
        for st in synthesis_types:
            self.assertIn(st, _SYNTH_MAP, f"Missing synth function for type: {st}")


class TestSoundDesignDataclasses(unittest.TestCase):
    """Tests for sound_design_ai dataclasses."""

    def test_motion_event_fields(self):
        from opencut.core.sound_design_ai import MotionEvent
        evt = MotionEvent(timestamp=1.5, duration=0.3, magnitude=75.0, intensity="high")
        self.assertEqual(evt.timestamp, 1.5)
        self.assertEqual(evt.intensity, "high")

    def test_motion_event_to_dict(self):
        from opencut.core.sound_design_ai import MotionEvent
        evt = MotionEvent(timestamp=1.5, duration=0.3, magnitude=75.0, intensity="high", frame_index=45)
        d = evt.to_dict()
        self.assertEqual(d["timestamp"], 1.5)
        self.assertEqual(d["frame_index"], 45)

    def test_sfx_event_fields(self):
        from opencut.core.sound_design_ai import SFXEvent
        evt = SFXEvent(timestamp=2.0, duration=0.5, category="impact", volume=0.8)
        self.assertEqual(evt.category, "impact")
        self.assertEqual(evt.volume, 0.8)

    def test_sfx_event_to_dict(self):
        from opencut.core.sound_design_ai import SFXEvent
        evt = SFXEvent(timestamp=2.0, duration=0.5, category="whoosh", description="Fast swipe")
        d = evt.to_dict()
        self.assertEqual(d["category"], "whoosh")
        self.assertEqual(d["description"], "Fast swipe")

    def test_sound_design_result_defaults(self):
        from opencut.core.sound_design_ai import SoundDesignResult
        r = SoundDesignResult()
        self.assertEqual(r.events, [])
        self.assertEqual(r.audio_path, "")
        self.assertEqual(r.event_count, 0)

    def test_sound_design_result_to_dict(self):
        from opencut.core.sound_design_ai import SFXEvent, SoundDesignResult
        r = SoundDesignResult(
            events=[SFXEvent(timestamp=1.0, duration=0.3, category="impact")],
            audio_path="/tmp/out.wav",
            event_count=1,
            duration=10.0,
        )
        d = r.to_dict()
        self.assertEqual(d["event_count"], 1)
        self.assertEqual(len(d["events"]), 1)
        self.assertEqual(d["duration"], 10.0)


class TestSoundDesignSynthesis(unittest.TestCase):
    """Tests for PCM synthesis functions."""

    def test_clamp_sample(self):
        from opencut.core.sound_design_ai import _clamp_sample
        self.assertEqual(_clamp_sample(0), 0)
        self.assertEqual(_clamp_sample(40000), 32767)
        self.assertEqual(_clamp_sample(-40000), -32768)

    def test_fade_envelope_attack(self):
        from opencut.core.sound_design_ai import _fade_envelope
        # Beginning of envelope should be near 0
        self.assertAlmostEqual(_fade_envelope(0.0, 1.0, attack=0.1), 0.0, places=2)
        # Middle should be 1.0
        self.assertAlmostEqual(_fade_envelope(0.5, 1.0, attack=0.1, release=0.1), 1.0)

    def test_fade_envelope_release(self):
        from opencut.core.sound_design_ai import _fade_envelope
        # End should be near 0
        self.assertAlmostEqual(_fade_envelope(0.99, 1.0, release=0.05), 0.2, places=1)

    def test_synthesize_noise_burst_length(self):
        from opencut.core.sound_design_ai import _synthesize_noise_burst
        duration = 0.5
        data = _synthesize_noise_burst(duration, 80.0, 0.7)
        expected_samples = int(44100 * duration)
        # Each sample is 2 bytes (16-bit)
        self.assertEqual(len(data), expected_samples * 2)

    def test_synthesize_sine_sweep_length(self):
        from opencut.core.sound_design_ai import _synthesize_sine_sweep
        duration = 0.4
        data = _synthesize_sine_sweep(duration, 300.0, 0.5)
        expected_samples = int(44100 * duration)
        self.assertEqual(len(data), expected_samples * 2)

    def test_synthesize_filtered_noise_length(self):
        from opencut.core.sound_design_ai import _synthesize_filtered_noise
        duration = 1.0
        data = _synthesize_filtered_noise(duration, 200.0, 0.6)
        expected_samples = int(44100 * duration)
        self.assertEqual(len(data), expected_samples * 2)

    def test_synthesize_pulse_train_length(self):
        from opencut.core.sound_design_ai import _synthesize_pulse_train
        duration = 0.5
        data = _synthesize_pulse_train(duration, 120.0, 0.7)
        expected_samples = int(44100 * duration)
        self.assertEqual(len(data), expected_samples * 2)

    def test_synthesize_harmonic_decay_length(self):
        from opencut.core.sound_design_ai import _synthesize_harmonic_decay
        duration = 0.8
        data = _synthesize_harmonic_decay(duration, 440.0, 0.6)
        expected_samples = int(44100 * duration)
        self.assertEqual(len(data), expected_samples * 2)

    def test_synthesize_glitch_burst_length(self):
        from opencut.core.sound_design_ai import _synthesize_glitch_burst
        duration = 0.15
        data = _synthesize_glitch_burst(duration, 800.0, 0.5)
        expected_samples = int(44100 * duration)
        self.assertEqual(len(data), expected_samples * 2)

    def test_synthesize_sfx_creates_wav(self):
        from opencut.core.sound_design_ai import synthesize_sfx
        with tempfile.TemporaryDirectory() as td:
            path = synthesize_sfx("impact", duration=0.2, volume=0.5, output_dir=td, seed=42)
            self.assertTrue(os.path.isfile(path))
            with wave.open(path, "rb") as wf:
                self.assertEqual(wf.getframerate(), 44100)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getnchannels(), 1)

    def test_synthesize_sfx_fallback_category(self):
        from opencut.core.sound_design_ai import synthesize_sfx
        # Unknown category falls back to ambient
        path = synthesize_sfx("nonexistent_category", duration=0.1)
        self.assertTrue(os.path.isfile(path))
        os.unlink(path)


class TestSoundDesignMapping(unittest.TestCase):
    """Tests for event-to-SFX mapping."""

    def test_map_events_empty(self):
        from opencut.core.sound_design_ai import map_events_to_sfx
        result = map_events_to_sfx([])
        self.assertEqual(result, [])

    def test_map_events_produces_sfx(self):
        from opencut.core.sound_design_ai import MotionEvent, map_events_to_sfx
        events = [
            MotionEvent(timestamp=1.0, duration=0.3, magnitude=90.0, intensity="high"),
            MotionEvent(timestamp=3.0, duration=0.5, magnitude=50.0, intensity="medium"),
            MotionEvent(timestamp=5.0, duration=1.0, magnitude=20.0, intensity="low"),
        ]
        sfx = map_events_to_sfx(events, seed=42)
        self.assertEqual(len(sfx), 3)
        for e in sfx:
            self.assertIn(e.category, [
                "impact", "whoosh", "ambient", "mechanical", "nature",
                "musical_hit", "riser", "drop", "glitch", "sweep",
                "texture", "stinger",
            ])

    def test_map_events_seed_reproducibility(self):
        from opencut.core.sound_design_ai import MotionEvent, map_events_to_sfx
        events = [MotionEvent(timestamp=i, duration=0.3, magnitude=60.0, intensity="medium") for i in range(5)]
        sfx1 = map_events_to_sfx(events, seed=123)
        sfx2 = map_events_to_sfx(events, seed=123)
        for a, b in zip(sfx1, sfx2):
            self.assertEqual(a.category, b.category)

    def test_map_events_volume_from_magnitude(self):
        from opencut.core.sound_design_ai import MotionEvent, map_events_to_sfx
        low_event = MotionEvent(timestamp=1.0, duration=0.3, magnitude=10.0, intensity="low")
        high_event = MotionEvent(timestamp=2.0, duration=0.3, magnitude=95.0, intensity="high")
        sfx = map_events_to_sfx([low_event, high_event], seed=1)
        self.assertLessEqual(sfx[0].volume, sfx[1].volume)

    def test_list_sfx_categories(self):
        from opencut.core.sound_design_ai import list_sfx_categories
        cats = list_sfx_categories()
        self.assertEqual(len(cats), 12)
        names = [c["name"] for c in cats]
        self.assertIn("impact", names)
        self.assertIn("whoosh", names)


class TestSoundDesignSignatures(unittest.TestCase):
    """Tests for function signatures."""

    def test_detect_motion_events_signature(self):
        from opencut.core.sound_design_ai import detect_motion_events
        sig = inspect.signature(detect_motion_events)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("sensitivity", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_generate_sound_design_signature(self):
        from opencut.core.sound_design_ai import generate_sound_design
        sig = inspect.signature(generate_sound_design)
        self.assertIn("video_path", sig.parameters)
        self.assertIn("sensitivity", sig.parameters)
        self.assertIn("categories", sig.parameters)
        self.assertIn("seed", sig.parameters)
        self.assertIn("on_progress", sig.parameters)


# ============================================================
# Ambient Generator Tests
# ============================================================
class TestAmbientPresets(unittest.TestCase):
    """Tests for ambient_generator presets."""

    def test_all_presets_exist(self):
        from opencut.core.ambient_generator import AMBIENT_PRESETS
        expected = {"forest", "ocean", "city", "rain", "office", "space", "cafe"}
        self.assertEqual(expected, set(AMBIENT_PRESETS.keys()))

    def test_presets_have_layers(self):
        from opencut.core.ambient_generator import AMBIENT_PRESETS
        for name, preset in AMBIENT_PRESETS.items():
            self.assertIn("description", preset, f"Missing description for {name}")
            self.assertIn("layers", preset, f"Missing layers for {name}")
            self.assertGreater(len(preset["layers"]), 0, f"Empty layers for {name}")

    def test_layer_required_keys(self):
        from opencut.core.ambient_generator import AMBIENT_PRESETS
        for preset_name, preset in AMBIENT_PRESETS.items():
            for layer in preset["layers"]:
                self.assertIn("name", layer, f"Missing name in {preset_name}")
                self.assertIn("type", layer, f"Missing type in {preset_name}")
                self.assertIn("freq_low", layer, f"Missing freq_low in {preset_name}")
                self.assertIn("freq_high", layer, f"Missing freq_high in {preset_name}")
                self.assertIn("volume", layer, f"Missing volume in {preset_name}")

    def test_layer_generators_cover_all_types(self):
        from opencut.core.ambient_generator import AMBIENT_PRESETS, _LAYER_GENERATORS
        all_types = set()
        for preset in AMBIENT_PRESETS.values():
            for layer in preset["layers"]:
                all_types.add(layer["type"])
        for lt in all_types:
            self.assertIn(lt, _LAYER_GENERATORS, f"Missing generator for type: {lt}")


class TestAmbientDataclasses(unittest.TestCase):
    """Tests for ambient_generator dataclasses."""

    def test_ambient_result_defaults(self):
        from opencut.core.ambient_generator import AmbientResult
        r = AmbientResult()
        self.assertEqual(r.audio_path, "")
        self.assertEqual(r.preset, "")
        self.assertEqual(r.layers_used, [])
        self.assertEqual(r.intensity, 0.5)

    def test_ambient_result_to_dict(self):
        from opencut.core.ambient_generator import AmbientResult
        r = AmbientResult(audio_path="/tmp/a.wav", preset="forest", duration=30.0,
                          layers_used=["wind", "birds"], intensity=0.7, seed=42)
        d = r.to_dict()
        self.assertEqual(d["preset"], "forest")
        self.assertEqual(d["seed"], 42)
        self.assertEqual(len(d["layers_used"]), 2)


class TestAmbientGeneration(unittest.TestCase):
    """Tests for ambient generation output."""

    def test_generate_forest_creates_wav(self):
        from opencut.core.ambient_generator import generate_ambient
        with tempfile.TemporaryDirectory() as td:
            result = generate_ambient(preset="forest", duration=2.0, intensity=0.5,
                                      seed=42, output_dir=td)
            self.assertTrue(os.path.isfile(result.audio_path))
            self.assertEqual(result.preset, "forest")
            self.assertGreater(len(result.layers_used), 0)
            with wave.open(result.audio_path, "rb") as wf:
                self.assertEqual(wf.getframerate(), 44100)

    def test_generate_ocean(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="ocean", duration=2.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        self.assertEqual(result.preset, "ocean")
        os.unlink(result.audio_path)

    def test_generate_city(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="city", duration=2.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        os.unlink(result.audio_path)

    def test_generate_rain(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="rain", duration=2.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        os.unlink(result.audio_path)

    def test_generate_office(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="office", duration=2.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        os.unlink(result.audio_path)

    def test_generate_space(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="space", duration=2.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        os.unlink(result.audio_path)

    def test_generate_cafe(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="cafe", duration=2.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        os.unlink(result.audio_path)

    def test_unknown_preset_falls_back(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="nonexistent", duration=2.0, seed=1)
        self.assertEqual(result.preset, "forest")
        os.unlink(result.audio_path)

    def test_intensity_zero(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="forest", duration=1.0, intensity=0.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        self.assertEqual(result.intensity, 0.0)
        os.unlink(result.audio_path)

    def test_intensity_max(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="forest", duration=1.0, intensity=1.0, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        self.assertEqual(result.intensity, 1.0)
        os.unlink(result.audio_path)

    def test_seed_reproducibility(self):
        from opencut.core.ambient_generator import generate_ambient
        r1 = generate_ambient(preset="forest", duration=1.0, seed=999)
        r2 = generate_ambient(preset="forest", duration=1.0, seed=999)
        # Same seed should produce identical audio
        with open(r1.audio_path, "rb") as f1, open(r2.audio_path, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())
        os.unlink(r1.audio_path)
        os.unlink(r2.audio_path)

    def test_duration_clamping(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="forest", duration=0.1, seed=1)
        # Should be clamped to minimum 1.0
        self.assertGreaterEqual(result.duration, 1.0)
        os.unlink(result.audio_path)

    def test_crossfade_disabled(self):
        from opencut.core.ambient_generator import generate_ambient
        result = generate_ambient(preset="forest", duration=3.0, crossfade=False, seed=1)
        self.assertTrue(os.path.isfile(result.audio_path))
        os.unlink(result.audio_path)

    def test_progress_callback(self):
        from opencut.core.ambient_generator import generate_ambient
        progress_values = []
        result = generate_ambient(
            preset="forest", duration=1.0, seed=1,
            on_progress=lambda pct: progress_values.append(pct),
        )
        self.assertTrue(len(progress_values) >= 3)
        os.unlink(result.audio_path)

    def test_list_presets(self):
        from opencut.core.ambient_generator import list_presets
        presets = list_presets()
        self.assertEqual(len(presets), 7)
        names = [p["name"] for p in presets]
        self.assertIn("forest", names)
        self.assertIn("ocean", names)
        for p in presets:
            self.assertIn("description", p)
            self.assertIn("layers", p)
            self.assertIn("layer_count", p)


class TestAmbientCrossfade(unittest.TestCase):
    """Tests for crossfade loop helper."""

    def test_crossfade_short_audio(self):
        from opencut.core.ambient_generator import _apply_crossfade_loop
        samples = [float(i) for i in range(10)]
        # Crossfade longer than half -> no change
        result = _apply_crossfade_loop(samples, 8)
        self.assertEqual(result, samples)

    def test_crossfade_normal(self):
        from opencut.core.ambient_generator import _apply_crossfade_loop
        samples = [100.0] * 100
        result = _apply_crossfade_loop(samples, 10)
        self.assertEqual(len(result), 100)


# ============================================================
# Music Mood Morph Tests
# ============================================================
class TestMoodMorphConstants(unittest.TestCase):
    """Tests for mood morph constants."""

    def test_all_transforms_exist(self):
        from opencut.core.music_mood_morph import MOOD_TRANSFORMS
        expected = {"brighten", "darken", "energize", "calm", "build", "drop"}
        self.assertEqual(expected, set(MOOD_TRANSFORMS.keys()))

    def test_transform_required_keys(self):
        from opencut.core.music_mood_morph import MOOD_TRANSFORMS
        for name, tf in MOOD_TRANSFORMS.items():
            self.assertIn("description", tf, f"Missing description for {name}")
            self.assertIn("eq_treble", tf, f"Missing eq_treble for {name}")
            self.assertIn("eq_bass", tf, f"Missing eq_bass for {name}")
            self.assertIn("tempo_factor", tf, f"Missing tempo_factor for {name}")


class TestMoodMorphDataclasses(unittest.TestCase):
    """Tests for mood morph dataclasses."""

    def test_audio_properties_defaults(self):
        from opencut.core.music_mood_morph import AudioProperties
        p = AudioProperties()
        self.assertEqual(p.duration, 0.0)
        self.assertEqual(p.sample_rate, 44100)
        self.assertEqual(p.estimated_tempo, 120.0)

    def test_audio_properties_to_dict(self):
        from opencut.core.music_mood_morph import AudioProperties
        p = AudioProperties(duration=120.5, estimated_tempo=128.0, estimated_energy=0.7)
        d = p.to_dict()
        self.assertEqual(d["duration"], 120.5)
        self.assertEqual(d["estimated_tempo"], 128.0)

    def test_mood_keyframe(self):
        from opencut.core.music_mood_morph import MoodKeyframe
        kf = MoodKeyframe(time=5.0, mood="brighten", intensity=0.8)
        d = kf.to_dict()
        self.assertEqual(d["time"], 5.0)
        self.assertEqual(d["mood"], "brighten")

    def test_mood_morph_result_defaults(self):
        from opencut.core.music_mood_morph import MoodMorphResult
        r = MoodMorphResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.applied_transforms, [])
        self.assertEqual(r.duration, 0.0)

    def test_mood_morph_result_to_dict(self):
        from opencut.core.music_mood_morph import MoodMorphResult
        r = MoodMorphResult(output_path="/tmp/out.wav", applied_transforms=["brighten"],
                            duration=60.0, keyframes=[{"time": 0, "mood": "brighten"}])
        d = r.to_dict()
        self.assertEqual(d["applied_transforms"], ["brighten"])


class TestMoodMorphFilterChain(unittest.TestCase):
    """Tests for filter chain building."""

    def test_brighten_has_treble_boost(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        filters, tempo = _build_filter_chain("brighten", 1.0)
        filter_str = ",".join(filters)
        self.assertIn("treble", filter_str)

    def test_darken_has_bass_boost(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        filters, tempo = _build_filter_chain("darken", 1.0)
        filter_str = ",".join(filters)
        self.assertIn("bass", filter_str)

    def test_energize_has_compand(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        filters, tempo = _build_filter_chain("energize", 1.0)
        filter_str = ",".join(filters)
        self.assertIn("compand", filter_str)

    def test_calm_tempo_slows(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        _, tempo = _build_filter_chain("calm", 1.0)
        self.assertLess(tempo, 1.0)

    def test_build_tempo_increases(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        _, tempo = _build_filter_chain("build", 1.0)
        self.assertGreater(tempo, 1.0)

    def test_drop_tempo_slows(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        _, tempo = _build_filter_chain("drop", 1.0)
        self.assertLess(tempo, 1.0)

    def test_intensity_zero_gives_neutral(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        filters, tempo = _build_filter_chain("energize", 0.0)
        self.assertAlmostEqual(tempo, 1.0, places=2)

    def test_unknown_mood_fallback(self):
        from opencut.core.music_mood_morph import _build_filter_chain
        filters, tempo = _build_filter_chain("nonexistent", 1.0)
        # Should fallback to brighten
        self.assertIsInstance(filters, list)

    def test_atempo_chain_normal(self):
        from opencut.core.music_mood_morph import _build_atempo_chain
        parts = _build_atempo_chain(1.5)
        self.assertTrue(len(parts) >= 1)
        self.assertTrue(any("atempo" in p for p in parts))

    def test_atempo_chain_unity(self):
        from opencut.core.music_mood_morph import _build_atempo_chain
        parts = _build_atempo_chain(1.0)
        self.assertEqual(parts, [])

    def test_atempo_chain_extreme(self):
        from opencut.core.music_mood_morph import _build_atempo_chain
        parts = _build_atempo_chain(3.0)
        # Should chain multiple atempo filters
        self.assertTrue(len(parts) >= 2)


class TestMoodMorphSignatures(unittest.TestCase):
    """Tests for mood morph function signatures."""

    def test_analyze_audio_properties_signature(self):
        from opencut.core.music_mood_morph import analyze_audio_properties
        sig = inspect.signature(analyze_audio_properties)
        self.assertIn("filepath", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_apply_mood_morph_signature(self):
        from opencut.core.music_mood_morph import apply_mood_morph
        sig = inspect.signature(apply_mood_morph)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("mood", sig.parameters)
        self.assertIn("intensity", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_apply_keyframed_morph_signature(self):
        from opencut.core.music_mood_morph import apply_keyframed_morph
        sig = inspect.signature(apply_keyframed_morph)
        self.assertIn("input_path", sig.parameters)
        self.assertIn("keyframes", sig.parameters)
        self.assertIn("segment_duration", sig.parameters)

    def test_list_mood_transforms(self):
        from opencut.core.music_mood_morph import list_mood_transforms
        transforms = list_mood_transforms()
        self.assertEqual(len(transforms), 6)
        names = [t["name"] for t in transforms]
        self.assertIn("brighten", names)
        self.assertIn("darken", names)


class TestMoodMorphKeyframeInterpolation(unittest.TestCase):
    """Tests for keyframe interpolation."""

    def test_empty_keyframes(self):
        from opencut.core.music_mood_morph import _interpolate_keyframes
        segments = _interpolate_keyframes([], 30.0, 5.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][2], "brighten")

    def test_single_keyframe(self):
        from opencut.core.music_mood_morph import MoodKeyframe, _interpolate_keyframes
        kfs = [MoodKeyframe(time=0.0, mood="darken", intensity=0.8)]
        segments = _interpolate_keyframes(kfs, 10.0, 5.0)
        self.assertGreater(len(segments), 0)
        # All segments should use darken
        for seg in segments:
            self.assertEqual(seg[2], "darken")

    def test_two_keyframes(self):
        from opencut.core.music_mood_morph import MoodKeyframe, _interpolate_keyframes
        kfs = [
            MoodKeyframe(time=0.0, mood="calm", intensity=0.2),
            MoodKeyframe(time=10.0, mood="energize", intensity=1.0),
        ]
        segments = _interpolate_keyframes(kfs, 10.0, 2.5)
        self.assertGreater(len(segments), 2)


# ============================================================
# Beat Sync Edit Tests
# ============================================================
class TestBeatSyncConstants(unittest.TestCase):
    """Tests for beat sync constants."""

    def test_beat_sync_modes(self):
        from opencut.core.beat_sync_edit import BEAT_SYNC_MODES
        expected = {"every_beat", "every_bar", "accent_only", "every_2_beats",
                    "every_8_beats", "custom"}
        self.assertEqual(expected, set(BEAT_SYNC_MODES.keys()))

    def test_modes_have_required_keys(self):
        from opencut.core.beat_sync_edit import BEAT_SYNC_MODES
        for name, info in BEAT_SYNC_MODES.items():
            self.assertIn("description", info)
            self.assertIn("beat_divisor", info)


class TestBeatSyncDataclasses(unittest.TestCase):
    """Tests for beat sync dataclasses."""

    def test_beat_fields(self):
        from opencut.core.beat_sync_edit import Beat
        b = Beat(timestamp=1.0, strength=0.8, beat_number=4, is_downbeat=True)
        d = b.to_dict()
        self.assertEqual(d["timestamp"], 1.0)
        self.assertTrue(d["is_downbeat"])

    def test_cut_point_fields(self):
        from opencut.core.beat_sync_edit import CutPoint
        cp = CutPoint(clip_index=0, clip_path="/v.mp4", in_point=1.0,
                      out_point=2.0, beat_number=4, beat_timestamp=1.0)
        d = cp.to_dict()
        self.assertEqual(d["clip_index"], 0)
        self.assertEqual(d["in_point"], 1.0)

    def test_beat_detect_result_defaults(self):
        from opencut.core.beat_sync_edit import BeatDetectResult
        r = BeatDetectResult()
        self.assertEqual(r.beats, [])
        self.assertEqual(r.tempo_bpm, 0.0)

    def test_beat_detect_result_to_dict(self):
        from opencut.core.beat_sync_edit import Beat, BeatDetectResult
        r = BeatDetectResult(
            beats=[Beat(timestamp=0.5, beat_number=0)],
            tempo_bpm=120.0, total_beats=1, duration=10.0,
        )
        d = r.to_dict()
        self.assertEqual(d["tempo_bpm"], 120.0)
        self.assertEqual(len(d["beats"]), 1)

    def test_beat_sync_result_defaults(self):
        from opencut.core.beat_sync_edit import BeatSyncResult
        r = BeatSyncResult()
        self.assertEqual(r.cuts, [])
        self.assertEqual(r.output_path, "")

    def test_beat_sync_result_to_dict(self):
        from opencut.core.beat_sync_edit import BeatSyncResult
        r = BeatSyncResult(tempo_bpm=128.0, total_beats=64, mode="every_bar", duration=30.0)
        d = r.to_dict()
        self.assertEqual(d["mode"], "every_bar")


class TestBeatSyncLogic(unittest.TestCase):
    """Tests for beat sync planning logic."""

    def _make_beats(self, count=16, bpm=120.0):
        from opencut.core.beat_sync_edit import Beat
        interval = 60.0 / bpm
        return [
            Beat(timestamp=i * interval, strength=0.8 if i % 4 == 0 else 0.5,
                 beat_number=i, is_downbeat=(i % 4 == 0))
            for i in range(count)
        ]

    def test_select_every_beat(self):
        from opencut.core.beat_sync_edit import _select_cut_beats
        beats = self._make_beats(16)
        selected = _select_cut_beats(beats, "every_beat")
        self.assertEqual(len(selected), 16)

    def test_select_every_bar(self):
        from opencut.core.beat_sync_edit import _select_cut_beats
        beats = self._make_beats(16)
        selected = _select_cut_beats(beats, "every_bar")
        self.assertEqual(len(selected), 4)  # beats 0, 4, 8, 12

    def test_select_accent_only(self):
        from opencut.core.beat_sync_edit import _select_cut_beats
        beats = self._make_beats(16)
        selected = _select_cut_beats(beats, "accent_only")
        for b in selected:
            self.assertTrue(b.is_downbeat)

    def test_select_every_2_beats(self):
        from opencut.core.beat_sync_edit import _select_cut_beats
        beats = self._make_beats(16)
        selected = _select_cut_beats(beats, "every_2_beats")
        self.assertEqual(len(selected), 8)

    def test_select_custom_n(self):
        from opencut.core.beat_sync_edit import _select_cut_beats
        beats = self._make_beats(16)
        selected = _select_cut_beats(beats, "custom", custom_n=3)
        expected_count = sum(1 for b in beats if b.beat_number % 3 == 0)
        self.assertEqual(len(selected), expected_count)

    def test_plan_cuts_empty_inputs(self):
        from opencut.core.beat_sync_edit import plan_beat_sync_cuts
        self.assertEqual(plan_beat_sync_cuts([], [], "every_beat"), [])

    @patch("opencut.core.beat_sync_edit.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0})
    def test_plan_cuts_single_clip(self, mock_info):
        from opencut.core.beat_sync_edit import plan_beat_sync_cuts
        beats = self._make_beats(8)
        cuts = plan_beat_sync_cuts(["/clip1.mp4"], beats, "every_bar")
        self.assertGreater(len(cuts), 0)
        for c in cuts:
            self.assertEqual(c.clip_path, "/clip1.mp4")

    @patch("opencut.core.beat_sync_edit.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0})
    def test_plan_cuts_multiple_clips_rotate(self, mock_info):
        from opencut.core.beat_sync_edit import plan_beat_sync_cuts
        beats = self._make_beats(16)
        clips = ["/a.mp4", "/b.mp4", "/c.mp4"]
        cuts = plan_beat_sync_cuts(clips, beats, "every_beat")
        # Clips should rotate
        indices = [c.clip_index for c in cuts]
        self.assertTrue(any(i == 0 for i in indices))
        self.assertTrue(any(i == 1 for i in indices))
        self.assertTrue(any(i == 2 for i in indices))

    @patch("opencut.core.beat_sync_edit.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0})
    def test_plan_cuts_energy_match(self, mock_info):
        from opencut.core.beat_sync_edit import plan_beat_sync_cuts
        beats = self._make_beats(8)
        clips = ["/short.mp4", "/long.mp4"]
        cuts = plan_beat_sync_cuts(clips, beats, "every_beat", energy_match=True)
        self.assertGreater(len(cuts), 0)

    def test_estimate_tempo(self):
        from opencut.core.beat_sync_edit import _estimate_tempo
        # 120 BPM = 0.5s intervals
        onsets = [(i * 0.5, 0.8) for i in range(10)]
        tempo = _estimate_tempo(onsets)
        self.assertAlmostEqual(tempo, 120.0, delta=5.0)

    def test_estimate_tempo_few_onsets(self):
        from opencut.core.beat_sync_edit import _estimate_tempo
        self.assertEqual(_estimate_tempo([(0.0, 1.0)]), 120.0)

    def test_quantize_to_grid(self):
        from opencut.core.beat_sync_edit import _quantize_to_grid
        onsets = [(0.5, 0.9), (1.0, 0.7), (1.5, 0.8)]
        beats = _quantize_to_grid(onsets, 120.0, 5.0)
        self.assertGreater(len(beats), 0)
        # First beat should be downbeat
        self.assertTrue(beats[0].is_downbeat)

    def test_list_beat_sync_modes(self):
        from opencut.core.beat_sync_edit import list_beat_sync_modes
        modes = list_beat_sync_modes()
        self.assertEqual(len(modes), 6)
        names = [m["name"] for m in modes]
        self.assertIn("every_beat", names)
        self.assertIn("every_bar", names)

    def test_detect_beats_signature(self):
        from opencut.core.beat_sync_edit import detect_beats
        sig = inspect.signature(detect_beats)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("sensitivity", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_assemble_beat_sync_signature(self):
        from opencut.core.beat_sync_edit import assemble_beat_sync
        sig = inspect.signature(assemble_beat_sync)
        self.assertIn("audio_path", sig.parameters)
        self.assertIn("clip_paths", sig.parameters)
        self.assertIn("mode", sig.parameters)
        self.assertIn("energy_match", sig.parameters)
        self.assertIn("on_progress", sig.parameters)


# ============================================================
# Stem Remix Tests
# ============================================================
class TestStemRemixConstants(unittest.TestCase):
    """Tests for stem remix constants."""

    def test_stem_names(self):
        from opencut.core.stem_remix import STEM_NAMES
        self.assertEqual(set(STEM_NAMES), {"vocals", "drums", "bass", "other"})

    def test_all_presets_exist(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        expected = {"acapella", "instrumental", "karaoke", "lo_fi",
                    "nightcore", "slowed_reverb", "drum_emphasis"}
        self.assertEqual(expected, set(REMIX_PRESETS.keys()))

    def test_presets_have_stems(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        for name, preset in REMIX_PRESETS.items():
            self.assertIn("description", preset, f"Missing description for {name}")
            self.assertIn("stems", preset, f"Missing stems for {name}")

    def test_default_stem_settings(self):
        from opencut.core.stem_remix import DEFAULT_STEM_SETTINGS
        self.assertEqual(DEFAULT_STEM_SETTINGS["volume"], 1.0)
        self.assertEqual(DEFAULT_STEM_SETTINGS["pan"], 0.0)
        self.assertFalse(DEFAULT_STEM_SETTINGS["mute"])


class TestStemRemixDataclasses(unittest.TestCase):
    """Tests for stem remix dataclasses."""

    def test_stem_settings_defaults(self):
        from opencut.core.stem_remix import StemSettings
        s = StemSettings()
        self.assertEqual(s.volume, 1.0)
        self.assertEqual(s.pan, 0.0)
        self.assertFalse(s.reverse)
        self.assertFalse(s.mute)

    def test_stem_settings_to_dict(self):
        from opencut.core.stem_remix import StemSettings
        s = StemSettings(volume=0.8, pan=-0.5, reverb_amount=0.3)
        d = s.to_dict()
        self.assertEqual(d["volume"], 0.8)
        self.assertEqual(d["pan"], -0.5)
        self.assertEqual(d["reverb_amount"], 0.3)

    def test_stem_settings_from_dict(self):
        from opencut.core.stem_remix import StemSettings
        s = StemSettings.from_dict({"volume": 0.6, "reverse": True, "mute": False})
        self.assertEqual(s.volume, 0.6)
        self.assertTrue(s.reverse)
        self.assertFalse(s.mute)

    def test_stem_settings_from_dict_defaults(self):
        from opencut.core.stem_remix import StemSettings
        s = StemSettings.from_dict({})
        self.assertEqual(s.volume, 1.0)

    def test_remix_result_defaults(self):
        from opencut.core.stem_remix import RemixResult
        r = RemixResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.preset_name, "")
        self.assertEqual(r.stem_settings, {})

    def test_remix_result_to_dict(self):
        from opencut.core.stem_remix import RemixResult
        r = RemixResult(output_path="/tmp/out.wav", preset_name="karaoke",
                        stem_settings={"vocals": {"volume": 0.15}}, duration=120.0)
        d = r.to_dict()
        self.assertEqual(d["preset_name"], "karaoke")
        self.assertEqual(d["duration"], 120.0)


class TestStemRemixFilterBuilding(unittest.TestCase):
    """Tests for per-stem filter chain building."""

    def test_muted_stem_only_volume_zero(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(mute=True)
        filters = _build_stem_filter(s)
        self.assertEqual(filters, ["volume=0"])

    def test_volume_change(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(volume=0.5)
        filters = _build_stem_filter(s)
        filter_str = ",".join(filters)
        self.assertIn("volume=0.50", filter_str)

    def test_unity_volume_no_filter(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(volume=1.0)
        filters = _build_stem_filter(s)
        # No volume filter needed at unity
        self.assertFalse(any("volume" in f for f in filters))

    def test_reverse_filter(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(reverse=True)
        filters = _build_stem_filter(s)
        self.assertIn("areverse", filters)

    def test_pan_filter(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(pan=0.5)
        filters = _build_stem_filter(s)
        filter_str = ",".join(filters)
        self.assertIn("pan=", filter_str)

    def test_delay_filter(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(delay_ms=200)
        filters = _build_stem_filter(s)
        filter_str = ",".join(filters)
        self.assertIn("adelay=200", filter_str)

    def test_reverb_filter(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(reverb_amount=0.5)
        filters = _build_stem_filter(s)
        filter_str = ",".join(filters)
        self.assertIn("aecho", filter_str)

    def test_pitch_shift_filter(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(pitch_shift_semitones=3)
        filters = _build_stem_filter(s)
        filter_str = ",".join(filters)
        self.assertIn("asetrate", filter_str)
        self.assertIn("atempo", filter_str)
        self.assertIn("aresample", filter_str)

    def test_no_pitch_shift_when_zero(self):
        from opencut.core.stem_remix import StemSettings, _build_stem_filter
        s = StemSettings(pitch_shift_semitones=0.0)
        filters = _build_stem_filter(s)
        self.assertFalse(any("asetrate" in f for f in filters))


class TestStemRemixPresets(unittest.TestCase):
    """Tests for remix preset configurations."""

    def test_acapella_mutes_non_vocals(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["acapella"]
        self.assertTrue(preset["stems"]["drums"]["mute"])
        self.assertTrue(preset["stems"]["bass"]["mute"])
        self.assertTrue(preset["stems"]["other"]["mute"])
        self.assertFalse(preset["stems"]["vocals"].get("mute", False))

    def test_instrumental_mutes_vocals(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["instrumental"]
        self.assertTrue(preset["stems"]["vocals"]["mute"])
        self.assertFalse(preset["stems"]["drums"].get("mute", False))

    def test_karaoke_reduces_vocals(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["karaoke"]
        self.assertLess(preset["stems"]["vocals"]["volume"], 0.5)
        self.assertGreater(preset["stems"]["vocals"]["reverb_amount"], 0)

    def test_nightcore_has_global_tempo(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["nightcore"]
        self.assertGreater(preset["global_tempo"], 1.0)
        self.assertGreater(preset["stems"]["vocals"]["pitch_shift_semitones"], 0)

    def test_slowed_reverb_slows_down(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["slowed_reverb"]
        self.assertLess(preset["global_tempo"], 1.0)

    def test_lo_fi_has_lowpass(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["lo_fi"]
        self.assertIn("global_lowpass", preset)
        self.assertGreater(preset["global_lowpass"], 0)

    def test_drum_emphasis_boosts_drums(self):
        from opencut.core.stem_remix import REMIX_PRESETS
        preset = REMIX_PRESETS["drum_emphasis"]
        self.assertGreater(preset["stems"]["drums"]["volume"], 1.0)
        self.assertLess(preset["stems"]["vocals"]["volume"], 1.0)


class TestStemRemixSignatures(unittest.TestCase):
    """Tests for stem remix function signatures."""

    def test_remix_stems_signature(self):
        from opencut.core.stem_remix import remix_stems
        sig = inspect.signature(remix_stems)
        self.assertIn("stem_dir", sig.parameters)
        self.assertIn("stem_paths", sig.parameters)
        self.assertIn("preset", sig.parameters)
        self.assertIn("custom_settings", sig.parameters)
        self.assertIn("on_progress", sig.parameters)

    def test_preview_remix_signature(self):
        from opencut.core.stem_remix import preview_remix
        sig = inspect.signature(preview_remix)
        self.assertIn("preview_duration", sig.parameters)
        self.assertIn("preview_start", sig.parameters)

    def test_apply_stem_effects_signature(self):
        from opencut.core.stem_remix import apply_stem_effects
        sig = inspect.signature(apply_stem_effects)
        self.assertIn("stem_path", sig.parameters)
        self.assertIn("settings", sig.parameters)

    def test_list_remix_presets(self):
        from opencut.core.stem_remix import list_remix_presets
        presets = list_remix_presets()
        self.assertEqual(len(presets), 7)
        names = [p["name"] for p in presets]
        self.assertIn("acapella", names)
        self.assertIn("nightcore", names)
        for p in presets:
            self.assertIn("description", p)
            self.assertIn("stem_settings", p)


class TestStemRemixResolution(unittest.TestCase):
    """Tests for stem path resolution."""

    def test_resolve_with_explicit_paths(self):
        from opencut.core.stem_remix import _resolve_stem_paths
        with tempfile.TemporaryDirectory() as td:
            # Create fake stem files
            vocals_path = os.path.join(td, "my_vocals.wav")
            drums_path = os.path.join(td, "my_drums.wav")
            for p in (vocals_path, drums_path):
                with open(p, "w") as f:
                    f.write("fake")

            resolved = _resolve_stem_paths("", {"vocals": vocals_path, "drums": drums_path})
            self.assertIn("vocals", resolved)
            self.assertIn("drums", resolved)
            self.assertEqual(resolved["vocals"], vocals_path)

    def test_resolve_from_directory(self):
        from opencut.core.stem_remix import _resolve_stem_paths
        with tempfile.TemporaryDirectory() as td:
            for name in ("vocals", "drums", "bass", "other"):
                with open(os.path.join(td, f"{name}.wav"), "w") as f:
                    f.write("fake")

            resolved = _resolve_stem_paths(td)
            self.assertEqual(len(resolved), 4)

    def test_resolve_empty(self):
        from opencut.core.stem_remix import _resolve_stem_paths
        resolved = _resolve_stem_paths("", None)
        self.assertEqual(resolved, {})

    def test_global_filter_tempo(self):
        from opencut.core.stem_remix import _build_global_filter
        filters = _build_global_filter({"global_tempo": 0.85})
        filter_str = ",".join(filters)
        self.assertIn("atempo=0.8500", filter_str)

    def test_global_filter_lowpass(self):
        from opencut.core.stem_remix import _build_global_filter
        filters = _build_global_filter({"global_lowpass": 6000})
        filter_str = ",".join(filters)
        self.assertIn("lowpass=f=6000", filter_str)

    def test_global_filter_empty(self):
        from opencut.core.stem_remix import _build_global_filter
        filters = _build_global_filter({})
        self.assertEqual(filters, [])


# ============================================================
# Route Smoke Tests
# ============================================================
class TestSoundMusicRoutes(unittest.TestCase):
    """Smoke tests for sound_music_routes blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.sound_music_routes import sound_music_bp
        self.assertEqual(sound_music_bp.name, "sound_music")

    def test_route_functions_exist(self):
        from opencut.routes import sound_music_routes as mod
        self.assertTrue(callable(getattr(mod, "route_sound_design", None)))
        self.assertTrue(callable(getattr(mod, "route_sfx_categories", None)))
        self.assertTrue(callable(getattr(mod, "route_ambient_generate", None)))
        self.assertTrue(callable(getattr(mod, "route_ambient_presets", None)))
        self.assertTrue(callable(getattr(mod, "route_mood_morph", None)))
        self.assertTrue(callable(getattr(mod, "route_beat_sync", None)))
        self.assertTrue(callable(getattr(mod, "route_beat_detect", None)))
        self.assertTrue(callable(getattr(mod, "route_stem_remix", None)))
        self.assertTrue(callable(getattr(mod, "route_remix_presets", None)))
        self.assertTrue(callable(getattr(mod, "route_stem_remix_preview", None)))

    def test_blueprint_has_rules(self):
        from opencut.routes.sound_music_routes import sound_music_bp
        # Blueprint may not expose rules until registered; just verify it has
        # deferred functions (route registrations)
        self.assertGreater(len(sound_music_bp.deferred_functions), 0)

    def test_imports_correct_modules(self):
        """Verify routes import from correct core modules."""
        import importlib
        mod = importlib.import_module("opencut.routes.sound_music_routes")
        # Module should have loaded without errors
        self.assertIsNotNone(mod)


if __name__ == "__main__":
    unittest.main()
