"""
OpenCut Procedural Ambient Soundscape Generator (Category 75)

Generate continuous ambient audio from environment presets using layered
PCM synthesis.  Each preset combines multiple synthesized layers with
randomized timing for organic, natural-sounding results.

Functions:
    generate_ambient  - Generate ambient soundscape from a preset
    list_presets       - List available ambient presets
"""

import logging
import math
import os
import random
import struct
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit

# ---------------------------------------------------------------------------
# Ambient Presets
# ---------------------------------------------------------------------------
AMBIENT_PRESETS = {
    "forest": {
        "description": "Birds, gentle wind, rustling leaves",
        "layers": [
            {"name": "wind", "type": "filtered_noise", "freq_low": 100, "freq_high": 600, "volume": 0.25},
            {"name": "rustling", "type": "modulated_noise", "freq_low": 2000, "freq_high": 6000, "volume": 0.12, "mod_rate": 0.3},
            {"name": "birds", "type": "chirps", "freq_low": 2000, "freq_high": 5000, "volume": 0.18, "interval": 2.5},
        ],
    },
    "ocean": {
        "description": "Waves, seagulls, ocean breeze",
        "layers": [
            {"name": "waves", "type": "wave_cycle", "freq_low": 80, "freq_high": 400, "volume": 0.35, "cycle_period": 6.0},
            {"name": "surf_foam", "type": "filtered_noise", "freq_low": 1000, "freq_high": 8000, "volume": 0.10},
            {"name": "seagulls", "type": "chirps", "freq_low": 1500, "freq_high": 3500, "volume": 0.08, "interval": 5.0},
            {"name": "breeze", "type": "filtered_noise", "freq_low": 200, "freq_high": 800, "volume": 0.15},
        ],
    },
    "city": {
        "description": "Traffic, distant voices, urban hum",
        "layers": [
            {"name": "traffic_hum", "type": "filtered_noise", "freq_low": 60, "freq_high": 300, "volume": 0.30},
            {"name": "distant_traffic", "type": "modulated_noise", "freq_low": 200, "freq_high": 1500, "volume": 0.15, "mod_rate": 0.1},
            {"name": "voices", "type": "modulated_noise", "freq_low": 300, "freq_high": 3000, "volume": 0.08, "mod_rate": 0.5},
            {"name": "sirens", "type": "chirps", "freq_low": 800, "freq_high": 1200, "volume": 0.04, "interval": 15.0},
        ],
    },
    "rain": {
        "description": "Rainfall, varying intensity, optional thunder",
        "layers": [
            {"name": "rain_steady", "type": "filtered_noise", "freq_low": 1000, "freq_high": 12000, "volume": 0.30},
            {"name": "rain_drops", "type": "modulated_noise", "freq_low": 3000, "freq_high": 10000, "volume": 0.15, "mod_rate": 2.0},
            {"name": "puddle_drips", "type": "chirps", "freq_low": 500, "freq_high": 2000, "volume": 0.10, "interval": 0.8},
            {"name": "thunder", "type": "rumble", "freq_low": 30, "freq_high": 120, "volume": 0.20, "interval": 20.0},
        ],
    },
    "office": {
        "description": "Typing, HVAC hum, muffled speech",
        "layers": [
            {"name": "hvac", "type": "filtered_noise", "freq_low": 80, "freq_high": 250, "volume": 0.20},
            {"name": "typing", "type": "clicks", "freq_low": 2000, "freq_high": 5000, "volume": 0.08, "interval": 0.15},
            {"name": "muffled_speech", "type": "modulated_noise", "freq_low": 200, "freq_high": 1500, "volume": 0.06, "mod_rate": 0.4},
            {"name": "fluorescent", "type": "tone", "freq_low": 120, "freq_high": 120, "volume": 0.05},
        ],
    },
    "space": {
        "description": "Deep hum, digital bleeps, void ambiance",
        "layers": [
            {"name": "deep_hum", "type": "tone", "freq_low": 40, "freq_high": 60, "volume": 0.25},
            {"name": "void", "type": "filtered_noise", "freq_low": 20, "freq_high": 150, "volume": 0.15},
            {"name": "digital_bleeps", "type": "chirps", "freq_low": 800, "freq_high": 2000, "volume": 0.06, "interval": 3.0},
            {"name": "static", "type": "modulated_noise", "freq_low": 4000, "freq_high": 10000, "volume": 0.03, "mod_rate": 0.8},
        ],
    },
    "cafe": {
        "description": "Chatter, clinking, espresso machine",
        "layers": [
            {"name": "chatter", "type": "modulated_noise", "freq_low": 200, "freq_high": 3000, "volume": 0.20, "mod_rate": 0.6},
            {"name": "clinking", "type": "chirps", "freq_low": 3000, "freq_high": 6000, "volume": 0.08, "interval": 1.5},
            {"name": "espresso", "type": "filtered_noise", "freq_low": 500, "freq_high": 4000, "volume": 0.05},
            {"name": "music_bg", "type": "modulated_noise", "freq_low": 100, "freq_high": 800, "volume": 0.10, "mod_rate": 0.2},
        ],
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class AmbientResult:
    """Result of ambient generation."""
    audio_path: str = ""
    preset: str = ""
    duration: float = 0.0
    layers_used: List[str] = field(default_factory=list)
    intensity: float = 0.5
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "preset": self.preset,
            "duration": round(self.duration, 3),
            "layers_used": self.layers_used,
            "intensity": round(self.intensity, 2),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# PCM Synthesis Helpers
# ---------------------------------------------------------------------------
def _clamp16(val: float) -> int:
    """Clamp to 16-bit signed int."""
    return max(-32768, min(32767, int(val)))


def _gen_filtered_noise(
    n_samples: int, freq_low: float, freq_high: float, volume: float, rng: random.Random,
) -> List[float]:
    """Bandpass-filtered white noise layer."""
    lp_cutoff = freq_high / SAMPLE_RATE
    hp_cutoff = freq_low / SAMPLE_RATE
    lp_alpha = min(1.0, 2.0 * math.pi * lp_cutoff)
    hp_alpha = min(1.0, 2.0 * math.pi * hp_cutoff)
    lp_prev = 0.0
    hp_prev = 0.0
    hp_out = 0.0
    samples = []
    for _ in range(n_samples):
        noise = rng.uniform(-1.0, 1.0)
        lp_prev = lp_prev + lp_alpha * (noise - lp_prev)
        hp_out = hp_alpha * (hp_out + lp_prev - hp_prev)
        hp_prev = lp_prev
        samples.append(hp_out * volume * 32767)
    return samples


def _gen_modulated_noise(
    n_samples: int, freq_low: float, freq_high: float,
    volume: float, mod_rate: float, rng: random.Random,
) -> List[float]:
    """Filtered noise with amplitude modulation for organic variation."""
    base = _gen_filtered_noise(n_samples, freq_low, freq_high, 1.0, rng)
    samples = []
    for i, s in enumerate(base):
        t = i / SAMPLE_RATE
        mod = 0.5 + 0.5 * math.sin(2.0 * math.pi * mod_rate * t + rng.uniform(0, math.pi))
        samples.append(s * mod * volume)
    return samples


def _gen_chirps(
    n_samples: int, freq_low: float, freq_high: float,
    volume: float, interval: float, rng: random.Random,
) -> List[float]:
    """Sporadic chirp/blip events at randomized intervals."""
    samples = [0.0] * n_samples
    duration = n_samples / SAMPLE_RATE
    t = rng.uniform(0.5, interval)

    while t < duration - 0.2:
        chirp_dur = rng.uniform(0.05, 0.3)
        chirp_freq = rng.uniform(freq_low, freq_high)
        start = int(t * SAMPLE_RATE)
        chirp_samples = int(chirp_dur * SAMPLE_RATE)

        for i in range(chirp_samples):
            pos = start + i
            if pos >= n_samples:
                break
            ct = i / SAMPLE_RATE
            # Frequency glide
            freq = chirp_freq * (1.0 + 0.3 * ct / chirp_dur)
            env = math.sin(math.pi * ct / chirp_dur)  # Hann envelope
            val = math.sin(2.0 * math.pi * freq * ct) * env * volume * 32767
            samples[pos] += val

        t += interval + rng.uniform(-interval * 0.3, interval * 0.5)

    return samples


def _gen_wave_cycle(
    n_samples: int, freq_low: float, freq_high: float,
    volume: float, cycle_period: float, rng: random.Random,
) -> List[float]:
    """Cyclic wave-like sound (ocean waves) with surge and retreat."""
    samples = []
    lp_prev = 0.0
    alpha = min(1.0, 2.0 * math.pi * (freq_high / SAMPLE_RATE))
    phase_offset = rng.uniform(0, 2 * math.pi)

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Wave surge envelope
        surge = 0.3 + 0.7 * max(0, math.sin(2.0 * math.pi * t / cycle_period + phase_offset))
        # Noise with frequency content that shifts with surge
        noise = rng.uniform(-1.0, 1.0)
        lp_prev = lp_prev + alpha * (noise - lp_prev)
        val = lp_prev * surge * volume * 32767
        samples.append(val)

    return samples


def _gen_rumble(
    n_samples: int, freq_low: float, freq_high: float,
    volume: float, interval: float, rng: random.Random,
) -> List[float]:
    """Low rumble events (thunder) at sporadic intervals."""
    samples = [0.0] * n_samples
    duration = n_samples / SAMPLE_RATE
    t = rng.uniform(interval * 0.5, interval)

    while t < duration - 3.0:
        rumble_dur = rng.uniform(1.5, 4.0)
        start = int(t * SAMPLE_RATE)
        rumble_samples = int(rumble_dur * SAMPLE_RATE)
        rumble_freq = rng.uniform(freq_low, freq_high)
        lp_prev = 0.0
        alpha = min(1.0, 2.0 * math.pi * (rumble_freq / SAMPLE_RATE))

        for i in range(rumble_samples):
            pos = start + i
            if pos >= n_samples:
                break
            ct = i / SAMPLE_RATE
            noise = rng.uniform(-1.0, 1.0)
            lp_prev = lp_prev + alpha * (noise - lp_prev)
            # Attack-sustain-decay envelope
            if ct < 0.1:
                env = ct / 0.1
            elif ct < rumble_dur * 0.4:
                env = 1.0
            else:
                env = max(0.0, 1.0 - (ct - rumble_dur * 0.4) / (rumble_dur * 0.6))
            samples[pos] += lp_prev * env * volume * 32767

        t += interval + rng.uniform(-interval * 0.2, interval * 0.5)

    return samples


def _gen_clicks(
    n_samples: int, freq_low: float, freq_high: float,
    volume: float, interval: float, rng: random.Random,
) -> List[float]:
    """Rapid small click events (typing, clinking)."""
    samples = [0.0] * n_samples
    duration = n_samples / SAMPLE_RATE
    t = rng.uniform(0.0, interval)

    while t < duration:
        click_dur = rng.uniform(0.005, 0.025)
        click_freq = rng.uniform(freq_low, freq_high)
        start = int(t * SAMPLE_RATE)
        click_samples = int(click_dur * SAMPLE_RATE)

        for i in range(click_samples):
            pos = start + i
            if pos >= n_samples:
                break
            ct = i / SAMPLE_RATE
            env = math.exp(-ct * 200)  # Fast decay
            val = math.sin(2.0 * math.pi * click_freq * ct) * env * volume * 32767
            samples[pos] += val

        t += interval + rng.uniform(-interval * 0.4, interval * 0.6)

    return samples


def _gen_tone(
    n_samples: int, freq_low: float, freq_high: float, volume: float, rng: random.Random,
) -> List[float]:
    """Continuous low-frequency tone (hum, drone)."""
    samples = []
    freq = (freq_low + freq_high) / 2.0
    # Add subtle modulation for realism
    mod_freq = rng.uniform(0.05, 0.2)
    mod_depth = 0.03

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        mod = 1.0 + mod_depth * math.sin(2.0 * math.pi * mod_freq * t)
        val = math.sin(2.0 * math.pi * freq * mod * t) * volume * 32767
        samples.append(val)

    return samples


# Layer type dispatcher
_LAYER_GENERATORS = {
    "filtered_noise": lambda n, layer, rng: _gen_filtered_noise(
        n, layer["freq_low"], layer["freq_high"], layer["volume"], rng),
    "modulated_noise": lambda n, layer, rng: _gen_modulated_noise(
        n, layer["freq_low"], layer["freq_high"], layer["volume"],
        layer.get("mod_rate", 0.3), rng),
    "chirps": lambda n, layer, rng: _gen_chirps(
        n, layer["freq_low"], layer["freq_high"], layer["volume"],
        layer.get("interval", 2.0), rng),
    "wave_cycle": lambda n, layer, rng: _gen_wave_cycle(
        n, layer["freq_low"], layer["freq_high"], layer["volume"],
        layer.get("cycle_period", 6.0), rng),
    "rumble": lambda n, layer, rng: _gen_rumble(
        n, layer["freq_low"], layer["freq_high"], layer["volume"],
        layer.get("interval", 20.0), rng),
    "clicks": lambda n, layer, rng: _gen_clicks(
        n, layer["freq_low"], layer["freq_high"], layer["volume"],
        layer.get("interval", 0.15), rng),
    "tone": lambda n, layer, rng: _gen_tone(
        n, layer["freq_low"], layer["freq_high"], layer["volume"], rng),
}


# ---------------------------------------------------------------------------
# Crossfade Loop Helper
# ---------------------------------------------------------------------------
def _apply_crossfade_loop(samples: List[float], crossfade_samples: int) -> List[float]:
    """Apply crossfade at the end to make the audio loop seamlessly."""
    n = len(samples)
    if crossfade_samples >= n // 2:
        return samples  # Too short to crossfade

    for i in range(crossfade_samples):
        blend = i / crossfade_samples
        tail_idx = n - crossfade_samples + i
        # Fade out the tail, fade in the head
        samples[tail_idx] = samples[tail_idx] * (1.0 - blend) + samples[i] * blend

    return samples


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------
def generate_ambient(
    preset: str = "forest",
    duration: float = 30.0,
    intensity: float = 0.5,
    seed: Optional[int] = None,
    output_dir: str = "",
    crossfade: bool = True,
    on_progress: Optional[Callable] = None,
) -> AmbientResult:
    """Generate an ambient soundscape WAV from a preset.

    Args:
        preset: Preset name from AMBIENT_PRESETS.
        duration: Output duration in seconds.
        intensity: Volume intensity 0.0-1.0.
        seed: Random seed for reproducibility.
        output_dir: Output directory for WAV file.
        crossfade: Apply crossfade for seamless looping.
        on_progress: Progress callback (int percentage).

    Returns:
        AmbientResult with audio path, preset name, duration, layers used.
    """
    if on_progress:
        on_progress(2)

    preset_data = AMBIENT_PRESETS.get(preset)
    if preset_data is None:
        logger.warning("Unknown preset '%s', falling back to 'forest'", preset)
        preset = "forest"
        preset_data = AMBIENT_PRESETS["forest"]

    duration = max(1.0, min(600.0, duration))
    intensity = max(0.0, min(1.0, intensity))
    n_samples = int(SAMPLE_RATE * duration)
    rng = random.Random(seed)

    layers = preset_data["layers"]
    layer_names = []
    mix_buffer = [0.0] * n_samples

    if on_progress:
        on_progress(5)

    for idx, layer in enumerate(layers):
        layer_type = layer["type"]
        gen_fn = _LAYER_GENERATORS.get(layer_type)
        if gen_fn is None:
            logger.warning("Unknown layer type '%s' in preset '%s'", layer_type, preset)
            continue

        if on_progress:
            pct = 5 + int(((idx + 1) / len(layers)) * 75)
            on_progress(pct)

        # Scale layer volume by global intensity
        scaled_layer = dict(layer)
        scaled_layer["volume"] = layer["volume"] * intensity

        layer_samples = gen_fn(n_samples, scaled_layer, rng)
        for i in range(min(n_samples, len(layer_samples))):
            mix_buffer[i] += layer_samples[i]

        layer_names.append(layer["name"])

    if on_progress:
        on_progress(82)

    # Apply crossfade for looping
    if crossfade and duration > 2.0:
        crossfade_len = int(SAMPLE_RATE * min(1.0, duration * 0.05))
        mix_buffer = _apply_crossfade_loop(mix_buffer, crossfade_len)

    if on_progress:
        on_progress(88)

    # Normalize
    peak = max(abs(s) for s in mix_buffer) if mix_buffer else 1.0
    if peak > 32767:
        scale = 32767.0 / peak
        mix_buffer = [s * scale for s in mix_buffer]

    # Write WAV
    samples_int = [_clamp16(s) for s in mix_buffer]

    if output_dir and os.path.isdir(output_dir):
        fd, wav_path = tempfile.mkstemp(suffix=f"_ambient_{preset}.wav", dir=output_dir)
    else:
        fd, wav_path = tempfile.mkstemp(suffix=f"_ambient_{preset}.wav")
    os.close(fd)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f"<{len(samples_int)}h", *samples_int))

    if on_progress:
        on_progress(95)

    result = AmbientResult(
        audio_path=wav_path,
        preset=preset,
        duration=duration,
        layers_used=layer_names,
        intensity=intensity,
        seed=seed,
    )
    logger.info(
        "Ambient generation complete: preset='%s', %.1fs, %d layers -> %s",
        preset, duration, len(layer_names), wav_path,
    )
    return result


def list_presets() -> List[Dict]:
    """Return list of available ambient presets with descriptions."""
    return [
        {
            "name": name,
            "description": info["description"],
            "layers": [layer["name"] for layer in info["layers"]],
            "layer_count": len(info["layers"]),
        }
        for name, info in AMBIENT_PRESETS.items()
    ]
