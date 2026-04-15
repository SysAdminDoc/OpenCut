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

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "preset_name": self.preset_name,
            "stem_settings": self.stem_settings,
            "duration": round(self.duration, 3),
        }


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


def remix_stems(
    stem_dir: str = "",
    stem_paths: Optional[Dict[str, str]] = None,
    preset: str = "",
    custom_settings: Optional[Dict[str, dict]] = None,
    output_path_val: str = "",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
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
