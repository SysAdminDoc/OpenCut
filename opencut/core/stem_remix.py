"""
OpenCut Stem Remix Module (Category 75)

Creative stem manipulation and remix.  Given separated stems (vocals, drums,
bass, other from existing Demucs integration), apply per-stem effects and
remix with preset configurations.

Functions:
    apply_stem_effects  - Apply effects to individual stems
    remix_stems         - Mix stems back together with effects
    preview_remix       - Preview remix on a short segment
"""

import json
import logging
import os
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEM_NAMES = ["vocals", "drums", "bass", "other"]

SAMPLE_RATE = 44100

# Per-stem effect defaults
DEFAULT_STEM_SETTINGS = {
    "volume": 1.0,         # 0.0-2.0
    "pan": 0.0,            # -1.0 (left) to 1.0 (right)
    "reverb_amount": 0.0,  # 0.0-1.0
    "delay_ms": 0.0,       # 0-1000
    "pitch_shift_semitones": 0.0,  # -12 to 12
    "reverse": False,
    "mute": False,
}

# ---------------------------------------------------------------------------
# Remix Presets
# ---------------------------------------------------------------------------
REMIX_PRESETS = {
    "acapella": {
        "description": "Vocals only — isolated vocal track",
        "stems": {
            "vocals": {"volume": 1.0, "mute": False},
            "drums": {"mute": True},
            "bass": {"mute": True},
            "other": {"mute": True},
        },
    },
    "instrumental": {
        "description": "Mute vocals — instrumental backing track",
        "stems": {
            "vocals": {"mute": True},
            "drums": {"volume": 1.0, "mute": False},
            "bass": {"volume": 1.0, "mute": False},
            "other": {"volume": 1.0, "mute": False},
        },
    },
    "karaoke": {
        "description": "Vocals reduced with reverb for karaoke backing",
        "stems": {
            "vocals": {"volume": 0.15, "reverb_amount": 0.6, "mute": False},
            "drums": {"volume": 1.0, "mute": False},
            "bass": {"volume": 1.0, "mute": False},
            "other": {"volume": 0.9, "mute": False},
        },
    },
    "lo_fi": {
        "description": "Slowed down with vinyl warmth and low-pass filter",
        "stems": {
            "vocals": {"volume": 0.7, "reverb_amount": 0.3, "mute": False},
            "drums": {"volume": 0.8, "mute": False},
            "bass": {"volume": 1.1, "mute": False},
            "other": {"volume": 0.6, "reverb_amount": 0.2, "mute": False},
        },
        "global_tempo": 0.92,
        "global_lowpass": 8000,
    },
    "nightcore": {
        "description": "Sped up with pitch shift for nightcore style",
        "stems": {
            "vocals": {"volume": 1.0, "pitch_shift_semitones": 3, "mute": False},
            "drums": {"volume": 1.1, "mute": False},
            "bass": {"volume": 0.9, "mute": False},
            "other": {"volume": 1.0, "pitch_shift_semitones": 3, "mute": False},
        },
        "global_tempo": 1.25,
    },
    "slowed_reverb": {
        "description": "Slowed down with heavy reverb for dreamy atmosphere",
        "stems": {
            "vocals": {"volume": 0.9, "reverb_amount": 0.7, "mute": False},
            "drums": {"volume": 0.6, "reverb_amount": 0.4, "mute": False},
            "bass": {"volume": 1.0, "reverb_amount": 0.3, "mute": False},
            "other": {"volume": 0.8, "reverb_amount": 0.6, "mute": False},
        },
        "global_tempo": 0.82,
    },
    "drum_emphasis": {
        "description": "Boosted drums with ducked melodic elements",
        "stems": {
            "vocals": {"volume": 0.5, "mute": False},
            "drums": {"volume": 1.5, "mute": False},
            "bass": {"volume": 1.2, "mute": False},
            "other": {"volume": 0.4, "mute": False},
        },
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class StemSettings:
    """Effect settings for a single stem."""
    volume: float = 1.0
    pan: float = 0.0
    reverb_amount: float = 0.0
    delay_ms: float = 0.0
    pitch_shift_semitones: float = 0.0
    reverse: bool = False
    mute: bool = False

    def to_dict(self) -> dict:
        return {
            "volume": round(self.volume, 2),
            "pan": round(self.pan, 2),
            "reverb_amount": round(self.reverb_amount, 2),
            "delay_ms": round(self.delay_ms, 1),
            "pitch_shift_semitones": round(self.pitch_shift_semitones, 1),
            "reverse": self.reverse,
            "mute": self.mute,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StemSettings":
        return cls(
            volume=float(d.get("volume", 1.0)),
            pan=float(d.get("pan", 0.0)),
            reverb_amount=float(d.get("reverb_amount", 0.0)),
            delay_ms=float(d.get("delay_ms", 0.0)),
            pitch_shift_semitones=float(d.get("pitch_shift_semitones", 0.0)),
            reverse=bool(d.get("reverse", False)),
            mute=bool(d.get("mute", False)),
        )


@dataclass
class RemixResult:
    """Result of stem remix operation."""
    output_path: str = ""
    preset_name: str = ""
    stem_settings: Dict[str, dict] = field(default_factory=dict)
    duration: float = 0.0
    # New (effect-config API) — populated when remix_stems is called with
    # the list-based stem_paths/effects_config signature used by
    # routes/audio_expansion_routes.py::stem_remix and the v1.15.0 tests.
    stems_processed: int = 0
    effects_applied: int = 0

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "preset_name": self.preset_name,
            "stem_settings": self.stem_settings,
            "duration": round(self.duration, 3),
            "stems_processed": self.stems_processed,
            "effects_applied": self.effects_applied,
        }


# ---------------------------------------------------------------------------
# Effect Catalog (effect-config API)
# ---------------------------------------------------------------------------
# Each entry maps a named effect to its default parameter set. The
# routes/audio_expansion_routes.py stem-remix endpoint and the v1.15.0
# audio expansion tests both reference this dict by name; do not rename.
SUPPORTED_EFFECTS: Dict[str, dict] = {
    "reverb": {
        "params": {"delay": 60, "decay": 0.5, "wet": 0.6},
        "filter": "aecho",
    },
    "compress": {
        "params": {"threshold": 0.125, "ratio": 4.0, "attack": 20, "release": 250},
        "filter": "acompressor",
    },
    "eq_bass_boost": {
        "params": {"gain_db": 6.0, "freq": 100},
        "filter": "bass",
    },
    "eq_treble_boost": {
        "params": {"gain_db": 4.0, "freq": 6000},
        "filter": "treble",
    },
    "eq_mid_cut": {
        "params": {"gain_db": -4.0, "freq": 1500, "width": 200},
        "filter": "equalizer",
    },
    "highpass": {
        "params": {"freq": 80},
        "filter": "highpass",
    },
    "lowpass": {
        "params": {"freq": 8000},
        "filter": "lowpass",
    },
    "normalize": {
        "params": {"target_lufs": -16.0, "true_peak": -1.5},
        "filter": "loudnorm",
    },
    "chorus": {
        "params": {"in_gain": 0.5, "out_gain": 0.9, "delays": "50", "decays": "0.4",
                   "speeds": "0.25", "depths": "2"},
        "filter": "chorus",
    },
    "flanger": {
        "params": {"delay": 10, "depth": 2, "regen": 0, "width": 71, "speed": 0.5},
        "filter": "flanger",
    },
}


@dataclass
class StemEffect:
    """A single named effect to apply to one stem.

    Parameters override the matching ``SUPPORTED_EFFECTS[name]["params"]``
    defaults; unknown params are passed through to the FFmpeg filter as-is.
    """
    name: str
    params: Dict = field(default_factory=dict)
    stem_index: int = 0  # which stem in the input list this effect targets

    def to_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "stem_index": self.stem_index}

    @classmethod
    def from_dict(cls, d: dict) -> "StemEffect":
        if not isinstance(d, dict):
            raise ValueError("StemEffect.from_dict requires a dict")
        # Accept nested {"effect": {...}} shape too
        inner = d.get("effect") if isinstance(d.get("effect"), dict) else d
        name = str(inner.get("name", "")).strip()
        if not name:
            raise ValueError("StemEffect requires a non-empty 'name'")
        params = inner.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        try:
            stem_index = int(d.get("stem_index", inner.get("stem_index", 0)))
        except (TypeError, ValueError):
            stem_index = 0
        return cls(name=name, params=params, stem_index=stem_index)


# ---------------------------------------------------------------------------
# Stem Duration Helper
# ---------------------------------------------------------------------------
def _get_audio_duration(filepath: str) -> float:
    """Get audio file duration via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-Stem Effect Application
# ---------------------------------------------------------------------------
def _build_stem_filter(settings: StemSettings) -> List[str]:
    """Build FFmpeg audio filter chain for a single stem's effects."""
    filters = []

    if settings.mute:
        filters.append("volume=0")
        return filters

    # Volume
    if abs(settings.volume - 1.0) > 0.01:
        vol = max(0.0, min(3.0, settings.volume))
        filters.append(f"volume={vol:.2f}")

    # Reverse
    if settings.reverse:
        filters.append("areverse")

    # Pan (stereo balance using pan filter)
    if abs(settings.pan) > 0.01:
        # Pan: -1 = left, 0 = center, 1 = right
        left_gain = max(0.0, 1.0 - settings.pan) if settings.pan >= 0 else 1.0
        right_gain = max(0.0, 1.0 + settings.pan) if settings.pan <= 0 else 1.0
        filters.append(f"pan=stereo|c0={left_gain:.2f}*c0|c1={right_gain:.2f}*c0")

    # Delay
    if settings.delay_ms > 0:
        delay_ms = min(1000.0, settings.delay_ms)
        filters.append(f"adelay={delay_ms:.0f}|{delay_ms:.0f}")

    # Reverb (simulated via aecho)
    if settings.reverb_amount > 0.01:
        rev = max(0.0, min(1.0, settings.reverb_amount))
        # aecho: in_gain|out_gain|delays|decays
        delays = "60|120|180"
        decays = f"{rev * 0.5:.2f}|{rev * 0.35:.2f}|{rev * 0.2:.2f}"
        filters.append(f"aecho=0.8:0.88:{delays}:{decays}")

    # Pitch shift (via asetrate + atempo combo)
    if abs(settings.pitch_shift_semitones) > 0.01:
        semitones = max(-12.0, min(12.0, settings.pitch_shift_semitones))
        rate_factor = 2.0 ** (semitones / 12.0)
        new_rate = int(SAMPLE_RATE * rate_factor)
        filters.append(f"asetrate={new_rate}")
        # Compensate speed change
        tempo_comp = 1.0 / rate_factor
        tempo_comp = max(0.5, min(2.0, tempo_comp))
        filters.append(f"atempo={tempo_comp:.4f}")
        filters.append(f"aresample={SAMPLE_RATE}")

    return filters


def _build_pan_filter(pan_value: float) -> str:
    """Build a stereo ``pan`` FFmpeg filter for *pan_value* in [-1, 1].

    -1 = full left, 0 = centered, 1 = full right. Out-of-range values are
    clamped silently (callers may receive unsanitised user input).
    """
    try:
        p = float(pan_value)
    except (TypeError, ValueError):
        p = 0.0
    if p != p:  # NaN guard
        p = 0.0
    p = max(-1.0, min(1.0, p))
    # Standard equal-power constant-power pan rules:
    #   left  = sqrt((1 - p) / 2)   when p in [-1, 1]
    #   right = sqrt((1 + p) / 2)
    # This avoids unity gain mixing of the same source into both channels.
    import math
    left = math.sqrt((1.0 - p) / 2.0)
    right = math.sqrt((1.0 + p) / 2.0)
    return f"pan=stereo|c0={left:.4f}*c0|c1={right:.4f}*c0"


def _effect_to_filter(effect: StemEffect) -> str:
    """Convert a :class:`StemEffect` into a single FFmpeg filter string.

    Raises ``ValueError`` for unknown effect names so callers can validate
    user-supplied effect lists upfront instead of waiting for FFmpeg to
    fail with a cryptic filter-graph error.
    """
    if not isinstance(effect, StemEffect):
        raise ValueError("effect must be a StemEffect instance")
    spec = SUPPORTED_EFFECTS.get(effect.name)
    if spec is None:
        raise ValueError(f"Unknown stem effect: {effect.name!r}")
    base_params = dict(spec.get("params", {}))
    base_params.update(effect.params or {})
    filter_name = spec["filter"]

    # Per-effect parameter formatting — keep the FFmpeg-side keys explicit
    # so we never silently swallow malformed user input.
    if effect.name == "reverb":
        delay = max(0, min(int(base_params.get("delay", 60)), 1000))
        decay = max(0.0, min(float(base_params.get("decay", 0.5)), 1.0))
        wet = max(0.0, min(float(base_params.get("wet", 0.6)), 1.0))
        return f"aecho=0.8:{wet:.2f}:{delay}:{decay:.2f}"
    if effect.name == "compress":
        threshold = max(0.001, min(float(base_params.get("threshold", 0.125)), 1.0))
        ratio = max(1.0, min(float(base_params.get("ratio", 4.0)), 20.0))
        attack = max(0.01, min(float(base_params.get("attack", 20)), 2000.0))
        release = max(0.01, min(float(base_params.get("release", 250)), 9000.0))
        return f"acompressor=threshold={threshold:.3f}:ratio={ratio:.2f}:attack={attack:.2f}:release={release:.2f}"
    if effect.name in {"eq_bass_boost", "eq_treble_boost"}:
        gain = max(-30.0, min(float(base_params.get("gain_db", 6.0)), 30.0))
        freq = max(20, min(int(base_params.get("freq", 100 if effect.name == "eq_bass_boost" else 6000)), 20000))
        return f"{filter_name}=g={gain:.2f}:f={freq}"
    if effect.name == "eq_mid_cut":
        gain = max(-30.0, min(float(base_params.get("gain_db", -4.0)), 30.0))
        freq = max(20, min(int(base_params.get("freq", 1500)), 20000))
        width = max(1, min(int(base_params.get("width", 200)), 20000))
        return f"equalizer=f={freq}:width_type=h:width={width}:g={gain:.2f}"
    if effect.name == "highpass":
        freq = max(20, min(int(base_params.get("freq", 80)), 20000))
        return f"highpass=f={freq}"
    if effect.name == "lowpass":
        freq = max(20, min(int(base_params.get("freq", 8000)), 20000))
        return f"lowpass=f={freq}"
    if effect.name == "normalize":
        target = max(-70.0, min(float(base_params.get("target_lufs", -16.0)), 0.0))
        peak = max(-9.0, min(float(base_params.get("true_peak", -1.5)), 0.0))
        return f"loudnorm=I={target:.1f}:TP={peak:.1f}:LRA=11"
    if effect.name == "chorus":
        in_gain = max(0.0, min(float(base_params.get("in_gain", 0.5)), 1.0))
        out_gain = max(0.0, min(float(base_params.get("out_gain", 0.9)), 1.0))
        return f"chorus={in_gain:.2f}:{out_gain:.2f}:50:0.4:0.25:2"
    if effect.name == "flanger":
        delay = max(0, min(int(base_params.get("delay", 10)), 30))
        depth = max(0, min(int(base_params.get("depth", 2)), 10))
        speed = max(0.1, min(float(base_params.get("speed", 0.5)), 10.0))
        return f"flanger=delay={delay}:depth={depth}:speed={speed:.2f}"
    # Unreachable — SUPPORTED_EFFECTS membership was already checked.
    raise ValueError(f"Unhandled stem effect: {effect.name!r}")


def apply_stem_effect(
    stem_path: str,
    effect: StemEffect,
    output: Optional[str] = None,
) -> str:
    """Apply a single named effect to one stem file.

    Returns the path to the rendered file. The output file is created in
    the system temp directory if *output* is not supplied. Raises
    ``FileNotFoundError`` upfront if the stem path does not exist.
    """
    if not isinstance(effect, StemEffect):
        if isinstance(effect, dict):
            effect = StemEffect.from_dict(effect)
        else:
            raise ValueError("effect must be a StemEffect or dict")

    if not stem_path or not os.path.isfile(stem_path):
        raise FileNotFoundError(f"Stem file not found: {stem_path}")

    filt = _effect_to_filter(effect)

    if output:
        out_path = output
    else:
        fd, out_path = tempfile.mkstemp(suffix=f"_stem_{effect.name}.wav")
        os.close(fd)

    cmd = [
        get_ffmpeg_path(), "-y", "-i", stem_path,
        "-af", filt,
        "-ar", str(SAMPLE_RATE),
        out_path,
    ]
    run_ffmpeg(cmd)
    return out_path


def apply_stem_effects(
    stem_path: str,
    settings: StemSettings,
    output_dir: str = "",
) -> str:
    """Apply effects to a single stem file.

    Args:
        stem_path: Path to stem audio file.
        settings: Effect settings.
        output_dir: Output directory (uses temp if not set).

    Returns:
        Path to processed stem file.
    """
    filters = _build_stem_filter(settings)

    if output_dir and os.path.isdir(output_dir):
        fd, out_path = tempfile.mkstemp(suffix="_stem_fx.wav", dir=output_dir)
    else:
        fd, out_path = tempfile.mkstemp(suffix="_stem_fx.wav")
    os.close(fd)

    cmd = [get_ffmpeg_path(), "-y", "-i", stem_path]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-ar", str(SAMPLE_RATE), out_path])

    run_ffmpeg(cmd)
    return out_path


# ---------------------------------------------------------------------------
# Stem Mixing
# ---------------------------------------------------------------------------
def _resolve_stem_paths(
    stem_dir: str,
    stem_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Resolve actual stem file paths from directory or explicit paths.

    Looks for vocals.wav, drums.wav, bass.wav, other.wav in stem_dir,
    or uses explicitly provided paths.
    """
    resolved = {}

    for stem_name in STEM_NAMES:
        # Check explicit path first
        if stem_paths and stem_name in stem_paths:
            path = stem_paths[stem_name]
            if os.path.isfile(path):
                resolved[stem_name] = path
                continue

        # Search in stem directory
        if stem_dir and os.path.isdir(stem_dir):
            for ext in (".wav", ".mp3", ".flac", ".m4a"):
                candidate = os.path.join(stem_dir, f"{stem_name}{ext}")
                if os.path.isfile(candidate):
                    resolved[stem_name] = candidate
                    break

    return resolved


def _build_global_filter(preset_data: dict) -> List[str]:
    """Build global filters (tempo, lowpass) from preset data."""
    filters = []

    tempo = preset_data.get("global_tempo", 1.0)
    if abs(tempo - 1.0) > 0.01:
        tempo = max(0.5, min(2.0, tempo))
        filters.append(f"atempo={tempo:.4f}")

    lowpass = preset_data.get("global_lowpass", 0)
    if lowpass > 0:
        filters.append(f"lowpass=f={lowpass}")

    return filters


def mix_stems(
    stem_paths: Optional[List[str]] = None,
    mix_config: Optional[List[dict]] = None,
    output_file: str = "",
    on_progress: Optional[Callable] = None,
) -> RemixResult:
    """Mix a list of stem audio files together with optional per-stem volume/pan.

    *mix_config* is a list of ``{"volume": float, "pan": float}`` entries
    indexed by position in *stem_paths*. Missing entries fall back to
    unity volume / centred pan. Raises ``ValueError`` for an empty
    *stem_paths* list — clients almost always reach this path because of
    a UI bug, and a silent ``amix=inputs=0`` would crash inside FFmpeg
    with a less actionable error.
    """
    if not stem_paths:
        raise ValueError("mix_stems requires at least one stem path")
    if not isinstance(stem_paths, list):
        raise ValueError("stem_paths must be a list of file paths")

    cfg = list(mix_config) if isinstance(mix_config, list) else []

    if on_progress:
        on_progress(5)

    # Validate paths upfront so errors surface before we spawn ffmpeg.
    for sp in stem_paths:
        if not isinstance(sp, str) or not sp.strip():
            raise ValueError("Each stem path must be a non-empty string")
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"Stem file not found: {sp}")

    # Apply per-stem volume/pan up front by chaining filters per input.
    pre_processed: List[str] = []
    temp_files: List[str] = []
    try:
        for idx, sp in enumerate(stem_paths):
            entry = cfg[idx] if idx < len(cfg) and isinstance(cfg[idx], dict) else {}
            try:
                vol = float(entry.get("volume", 1.0))
            except (TypeError, ValueError):
                vol = 1.0
            try:
                pan = float(entry.get("pan", 0.0))
            except (TypeError, ValueError):
                pan = 0.0

            vol = max(0.0, min(3.0, vol))
            filters: List[str] = []
            if abs(vol - 1.0) > 0.01:
                filters.append(f"volume={vol:.2f}")
            if abs(pan) > 0.01:
                filters.append(_build_pan_filter(pan))

            if not filters:
                pre_processed.append(sp)
                continue

            fd, processed = tempfile.mkstemp(suffix=f"_premix_{idx}.wav")
            os.close(fd)
            cmd = [
                get_ffmpeg_path(), "-y", "-i", sp,
                "-af", ",".join(filters),
                "-ar", str(SAMPLE_RATE),
                processed,
            ]
            run_ffmpeg(cmd)
            pre_processed.append(processed)
            temp_files.append(processed)

        if on_progress:
            on_progress(55)

        if output_file:
            out = output_file
        else:
            fd, out = tempfile.mkstemp(suffix="_mix.wav")
            os.close(fd)

        cmd = [get_ffmpeg_path(), "-y"]
        for p in pre_processed:
            cmd.extend(["-i", p])
        amix = f"amix=inputs={len(pre_processed)}:duration=longest:dropout_transition=2"
        cmd.extend(["-filter_complex", amix, "-ar", str(SAMPLE_RATE), out])
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(95)

        dur = _get_audio_duration(out)
        return RemixResult(
            output_path=out,
            preset_name="",
            stem_settings={},
            duration=dur,
            stems_processed=len(pre_processed),
            effects_applied=0,
        )
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


def _remix_stems_with_effects(
    stem_paths: List[str],
    effects_config: List,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> RemixResult:
    """Apply a list of named effects to one or more stems, then mix.

    *effects_config* may be a list of dicts shaped as ``{"stem_index": N,
    "name": "reverb", "params": {...}}`` (or the same payload nested
    under an ``effect`` key). Effects targeted at the same stem are
    chained in submission order.
    """
    if not stem_paths:
        raise ValueError("remix_stems requires at least one stem path")

    # Normalise effect entries upfront so a malformed item fails fast.
    parsed_effects: List[StemEffect] = []
    for raw in effects_config or []:
        if isinstance(raw, StemEffect):
            parsed_effects.append(raw)
        elif isinstance(raw, dict):
            parsed_effects.append(StemEffect.from_dict(raw))
        else:
            raise ValueError(f"effects_config items must be StemEffect or dict, got {type(raw).__name__}")

    # Validate stem files exist upfront.
    for sp in stem_paths:
        if not isinstance(sp, str) or not sp.strip():
            raise ValueError("Each stem path must be a non-empty string")
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"Stem file not found: {sp}")

    if on_progress:
        on_progress(5)

    # Group effects by stem index.
    effects_by_stem: Dict[int, List[StemEffect]] = {}
    for eff in parsed_effects:
        idx = max(0, min(len(stem_paths) - 1, eff.stem_index))
        effects_by_stem.setdefault(idx, []).append(eff)

    processed_stems: List[str] = []
    temp_files: List[str] = []
    effects_applied = 0

    try:
        for idx, sp in enumerate(stem_paths):
            stem_effects = effects_by_stem.get(idx, [])
            if not stem_effects:
                processed_stems.append(sp)
                continue

            filter_chain = ",".join(_effect_to_filter(e) for e in stem_effects)
            fd, processed = tempfile.mkstemp(suffix=f"_fxstem_{idx}.wav")
            os.close(fd)

            cmd = [
                get_ffmpeg_path(), "-y", "-i", sp,
                "-af", filter_chain,
                "-ar", str(SAMPLE_RATE),
                processed,
            ]
            run_ffmpeg(cmd)
            processed_stems.append(processed)
            temp_files.append(processed)
            effects_applied += len(stem_effects)

            if on_progress:
                on_progress(10 + int(40 * (idx + 1) / max(1, len(stem_paths))))

        if output:
            out = output
        else:
            fd, out = tempfile.mkstemp(suffix="_remix_fx.wav")
            os.close(fd)

        cmd = [get_ffmpeg_path(), "-y"]
        for p in processed_stems:
            cmd.extend(["-i", p])
        amix = f"amix=inputs={len(processed_stems)}:duration=longest:dropout_transition=2"
        cmd.extend(["-filter_complex", amix, "-ar", str(SAMPLE_RATE), out])
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(95)

        return RemixResult(
            output_path=out,
            preset_name="",
            stem_settings={},
            duration=_get_audio_duration(out),
            stems_processed=len(processed_stems),
            effects_applied=effects_applied,
        )
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


def remix_stems(
    stem_dir: str = "",
    stem_paths=None,
    preset: str = "",
    custom_settings: Optional[Dict[str, dict]] = None,
    output_path_val: str = "",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
    effects_config: Optional[List] = None,
    output: Optional[str] = None,
) -> RemixResult:
    """Mix stems together with per-stem effects.

    Args:
        stem_dir: Directory containing stem files (vocals.wav, etc.).
        stem_paths: Explicit stem paths dict {"vocals": "/path/to/vocals.wav", ...}.
        preset: Preset name from REMIX_PRESETS (empty for custom settings).
        custom_settings: Custom per-stem settings dict.
        output_path_val: Explicit output file path.
        output_dir: Output directory.
        on_progress: Progress callback (int percentage).

    Returns:
        RemixResult with output path, preset name, settings, duration.
    """
    # Effect-config (list-based) signature: routes/audio_expansion_routes.py
    # and the v1.15.0 audio expansion tests pass stem_paths as a *list* of
    # file paths plus an effects_config list. Detect that shape upfront so
    # the rest of the function can keep its existing dict/preset semantics
    # without an explosion of conditionals.
    if isinstance(stem_paths, list) or effects_config is not None:
        if not isinstance(stem_paths, list):
            # `effects_config` was provided but stem_paths wasn't a list — treat
            # as an empty input so the test path raises ValueError consistently.
            raise ValueError("remix_stems requires stem_paths as a list when effects_config is given")
        return _remix_stems_with_effects(
            stem_paths=stem_paths,
            effects_config=effects_config or [],
            output=output or output_path_val or None,
            on_progress=on_progress,
        )

    if on_progress:
        on_progress(2)

    # Resolve stem file paths
    resolved = _resolve_stem_paths(stem_dir, stem_paths)
    if not resolved:
        raise ValueError(
            "No stem files found. Provide stem_dir with vocals/drums/bass/other "
            "files, or explicit stem_paths dict."
        )

    if on_progress:
        on_progress(8)

    # Determine settings from preset or custom
    preset_data = {}
    all_settings = {}

    if preset and preset in REMIX_PRESETS:
        preset_data = REMIX_PRESETS[preset]
        preset_stems = preset_data.get("stems", {})
        for stem_name in STEM_NAMES:
            stem_cfg = preset_stems.get(stem_name, {})
            merged = dict(DEFAULT_STEM_SETTINGS)
            merged.update(stem_cfg)
            all_settings[stem_name] = StemSettings.from_dict(merged)
    elif custom_settings:
        for stem_name in STEM_NAMES:
            stem_cfg = custom_settings.get(stem_name, {})
            merged = dict(DEFAULT_STEM_SETTINGS)
            merged.update(stem_cfg)
            all_settings[stem_name] = StemSettings.from_dict(merged)
    else:
        # Default: all stems at unity
        for stem_name in STEM_NAMES:
            all_settings[stem_name] = StemSettings()

    if on_progress:
        on_progress(12)

    # Process each stem
    processed_stems = []
    temp_files = []

    try:
        for idx, stem_name in enumerate(STEM_NAMES):
            if stem_name not in resolved:
                continue

            settings = all_settings.get(stem_name, StemSettings())
            if settings.mute:
                logger.debug("Skipping muted stem: %s", stem_name)
                continue

            if on_progress:
                pct = 12 + int(((idx + 1) / len(STEM_NAMES)) * 50)
                on_progress(pct)

            stem_path = resolved[stem_name]
            processed = apply_stem_effects(stem_path, settings, output_dir=output_dir)
            processed_stems.append(processed)
            temp_files.append(processed)

        if not processed_stems:
            logger.warning("All stems are muted — producing silence")
            # Find any stem to get duration
            any_stem = next(iter(resolved.values()))
            dur = _get_audio_duration(any_stem)

            if output_path_val:
                out = output_path_val
            elif output_dir:
                out = os.path.join(output_dir, "remix_silence.wav")
            else:
                fd, out = tempfile.mkstemp(suffix="_remix_silence.wav")
                os.close(fd)

            # Write silent WAV
            n_samples = int(SAMPLE_RATE * max(1.0, dur))
            silence = struct.pack(f"<{n_samples}h", *([0] * n_samples))
            with wave.open(out, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(silence)

            return RemixResult(
                output_path=out,
                preset_name=preset,
                stem_settings={n: all_settings[n].to_dict() for n in STEM_NAMES if n in all_settings},
                duration=dur,
            )

        if on_progress:
            on_progress(65)

        # Mix stems together using FFmpeg amix
        if output_path_val:
            out = output_path_val
        elif output_dir:
            base = f"remix_{preset}" if preset else "remix_custom"
            out = os.path.join(output_dir, f"{base}.wav")
        else:
            any_stem = next(iter(resolved.values()))
            out = output_path(any_stem, f"remix_{preset}" if preset else "remix_custom")
            if not out.lower().endswith((".wav", ".mp3", ".flac")):
                out = os.path.splitext(out)[0] + ".wav"

        n_inputs = len(processed_stems)
        cmd = [get_ffmpeg_path(), "-y"]
        for p in processed_stems:
            cmd.extend(["-i", p])

        # Build amix filter
        amix_filter = f"amix=inputs={n_inputs}:duration=longest:dropout_transition=2"

        # Add global filters from preset
        global_filters = _build_global_filter(preset_data)
        if global_filters:
            filter_chain = amix_filter + "," + ",".join(global_filters)
        else:
            filter_chain = amix_filter

        cmd.extend(["-filter_complex", filter_chain, "-ar", str(SAMPLE_RATE), out])

        if on_progress:
            on_progress(75)

        run_ffmpeg(cmd)

        if on_progress:
            on_progress(92)

        # Get output duration
        dur = _get_audio_duration(out) or _get_audio_duration(next(iter(resolved.values())))

        result = RemixResult(
            output_path=out,
            preset_name=preset,
            stem_settings={n: all_settings[n].to_dict() for n in STEM_NAMES if n in all_settings},
            duration=dur,
        )
        logger.info("Stem remix complete: preset='%s', %d stems -> %s", preset, n_inputs, out)
        return result

    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
def preview_remix(
    stem_dir: str = "",
    stem_paths: Optional[Dict[str, str]] = None,
    preset: str = "",
    custom_settings: Optional[Dict[str, dict]] = None,
    preview_duration: float = 15.0,
    preview_start: float = 0.0,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> RemixResult:
    """Preview remix settings on a short segment.

    Extracts a short segment from each stem before applying effects,
    making the preview much faster than processing entire tracks.

    Args:
        stem_dir: Directory containing stem files.
        stem_paths: Explicit stem paths.
        preset: Preset name.
        custom_settings: Custom per-stem settings.
        preview_duration: Duration of preview in seconds.
        preview_start: Start time in seconds.
        output_dir: Output directory.
        on_progress: Progress callback (int percentage).

    Returns:
        RemixResult for the preview segment.
    """
    if on_progress:
        on_progress(5)

    resolved = _resolve_stem_paths(stem_dir, stem_paths)
    if not resolved:
        raise ValueError("No stem files found for preview")

    preview_duration = max(3.0, min(30.0, preview_duration))
    preview_start = max(0.0, preview_start)

    # Extract preview segments from each stem
    preview_stems = {}
    temp_previews = []
    try:
        for stem_name, path in resolved.items():
            fd, preview_path = tempfile.mkstemp(suffix=f"_preview_{stem_name}.wav")
            os.close(fd)

            cmd = [
                get_ffmpeg_path(), "-y",
                "-ss", f"{preview_start:.3f}",
                "-i", path,
                "-t", f"{preview_duration:.3f}",
                "-vn", "-ar", str(SAMPLE_RATE),
                preview_path,
            ]
            try:
                run_ffmpeg(cmd)
                preview_stems[stem_name] = preview_path
                temp_previews.append(preview_path)
            except RuntimeError as e:
                logger.warning("Preview extraction failed for %s: %s", stem_name, e)
                try:
                    os.unlink(preview_path)
                except OSError:
                    pass

        if on_progress:
            on_progress(30)

        if not preview_stems:
            raise RuntimeError("Failed to extract preview segments from any stem")

        # Run remix on previews
        result = remix_stems(
            stem_paths=preview_stems,
            preset=preset,
            custom_settings=custom_settings,
            output_dir=output_dir,
            on_progress=lambda pct: on_progress(30 + int(pct * 0.65)) if on_progress else None,
        )

        if on_progress:
            on_progress(95)

        return result

    finally:
        for p in temp_previews:
            try:
                os.unlink(p)
            except OSError:
                pass


def list_remix_presets() -> List[Dict]:
    """Return available remix presets with descriptions."""
    return [
        {
            "name": name,
            "description": info["description"],
            "stem_settings": {
                stem: cfg for stem, cfg in info.get("stems", {}).items()
            },
            "has_global_tempo": "global_tempo" in info,
            "has_global_lowpass": "global_lowpass" in info,
        }
        for name, info in REMIX_PRESETS.items()
    ]
