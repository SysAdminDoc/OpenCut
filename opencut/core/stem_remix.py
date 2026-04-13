"""
OpenCut Stem Remix Module (Feature 2.7)

After stem separation, apply per-stem effects (reverb, compress, EQ)
and recombine. Supports per-stem volume and pan controls.

Functions:
    remix_stems      - Apply effects config to multiple stems and recombine
    apply_stem_effect - Apply a single effect to a single stem
    mix_stems        - Mix multiple stems with volume/pan controls
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Supported Effects & Their FFmpeg Filter Mappings
# ---------------------------------------------------------------------------
SUPPORTED_EFFECTS = {
    "reverb": {
        "description": "Add reverb/echo effect",
        "params": {"delay": 60, "decay": 0.4, "mix": 0.3},
    },
    "compress": {
        "description": "Dynamic range compression",
        "params": {"threshold": 0.1, "ratio": 4, "attack": 20, "release": 250},
    },
    "eq_bass_boost": {
        "description": "Boost bass frequencies",
        "params": {"frequency": 100, "gain": 6, "width": 200},
    },
    "eq_treble_boost": {
        "description": "Boost treble frequencies",
        "params": {"frequency": 8000, "gain": 4, "width": 2000},
    },
    "eq_mid_cut": {
        "description": "Cut mid frequencies (scoop)",
        "params": {"frequency": 1000, "gain": -4, "width": 800},
    },
    "highpass": {
        "description": "High-pass filter",
        "params": {"frequency": 80},
    },
    "lowpass": {
        "description": "Low-pass filter",
        "params": {"frequency": 12000},
    },
    "normalize": {
        "description": "Loudness normalization",
        "params": {"target_lufs": -16},
    },
    "chorus": {
        "description": "Chorus effect for widening",
        "params": {"depth": 0.5, "speed": 0.4},
    },
    "flanger": {
        "description": "Flanger effect",
        "params": {"delay": 5, "depth": 2, "speed": 0.5},
    },
}


@dataclass
class StemEffect:
    """Configuration for a single effect on a stem."""
    name: str
    params: Dict = field(default_factory=dict)


@dataclass
class StemMixConfig:
    """Mix configuration for a single stem."""
    path: str
    volume: float = 1.0       # 0.0 to 2.0
    pan: float = 0.0          # -1.0 (left) to 1.0 (right)
    mute: bool = False
    solo: bool = False
    effects: List[StemEffect] = field(default_factory=list)


@dataclass
class RemixResult:
    """Result of a stem remix operation."""
    output_path: str
    stems_processed: int
    effects_applied: int
    duration: float


def _effect_to_filter(effect: StemEffect) -> str:
    """Convert a StemEffect to an FFmpeg audio filter string.

    Args:
        effect: StemEffect with name and optional params.

    Returns:
        FFmpeg filter string.

    Raises:
        ValueError: If the effect name is not supported.
    """
    name = effect.name.lower()
    params = effect.params or {}
    defaults = SUPPORTED_EFFECTS.get(name, {}).get("params", {})

    # Merge defaults with user params
    p = {**defaults, **params}

    if name == "reverb":
        delay = int(p.get("delay", 60))
        decay = float(p.get("decay", 0.4))
        return f"aecho=0.8:0.88:{delay}:{decay}"

    elif name == "compress":
        threshold = float(p.get("threshold", 0.1))
        ratio = int(p.get("ratio", 4))
        attack = int(p.get("attack", 20))
        release = int(p.get("release", 250))
        return f"acompressor=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}"

    elif name == "eq_bass_boost":
        freq = int(p.get("frequency", 100))
        gain = float(p.get("gain", 6))
        width = int(p.get("width", 200))
        return f"equalizer=f={freq}:t=h:w={width}:g={gain}"

    elif name == "eq_treble_boost":
        freq = int(p.get("frequency", 8000))
        gain = float(p.get("gain", 4))
        width = int(p.get("width", 2000))
        return f"equalizer=f={freq}:t=h:w={width}:g={gain}"

    elif name == "eq_mid_cut":
        freq = int(p.get("frequency", 1000))
        gain = float(p.get("gain", -4))
        width = int(p.get("width", 800))
        return f"equalizer=f={freq}:t=h:w={width}:g={gain}"

    elif name == "highpass":
        freq = int(p.get("frequency", 80))
        return f"highpass=f={freq}"

    elif name == "lowpass":
        freq = int(p.get("frequency", 12000))
        return f"lowpass=f={freq}"

    elif name == "normalize":
        target = float(p.get("target_lufs", -16))
        return f"loudnorm=I={target}:TP=-1.5:LRA=11"

    elif name == "chorus":
        depth = float(p.get("depth", 0.5))
        speed = float(p.get("speed", 0.4))
        return f"chorus=0.5:0.9:{int(depth * 40)}:{depth}:{speed}:2"

    elif name == "flanger":
        delay = float(p.get("delay", 5))
        depth = float(p.get("depth", 2))
        speed = float(p.get("speed", 0.5))
        return f"flanger=delay={delay}:depth={depth}:speed={speed}"

    else:
        raise ValueError(f"Unsupported effect: {name}. Available: {', '.join(SUPPORTED_EFFECTS.keys())}")


def apply_stem_effect(
    stem_path: str,
    effect: StemEffect,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply a single audio effect to a stem file.

    Args:
        stem_path: Path to the input stem audio file.
        effect: StemEffect to apply.
        output: Output file path. Auto-generated if None.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        Path to the processed stem file.
    """
    if not os.path.isfile(stem_path):
        raise FileNotFoundError(f"Stem file not found: {stem_path}")

    if on_progress:
        on_progress(10, f"Applying {effect.name} to stem...")

    filter_str = _effect_to_filter(effect)

    if output is None:
        fd, output = tempfile.mkstemp(
            suffix=".wav", prefix=f"stem_{effect.name}_"
        )
        os.close(fd)

    cmd = (
        FFmpegCmd()
        .input(stem_path)
        .audio_filter(filter_str)
        .audio_codec("pcm_s16le")
        .output(output)
        .build()
    )

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90, f"{effect.name} applied")

    return output


def _build_pan_filter(pan: float) -> str:
    """Build a pan filter for stereo output.

    Args:
        pan: Pan value from -1.0 (left) to 1.0 (right). 0.0 is center.

    Returns:
        FFmpeg filter string for panning.
    """
    pan = max(-1.0, min(1.0, pan))
    # Convert -1..1 to left/right gains
    left_gain = min(1.0, 1.0 - pan)
    right_gain = min(1.0, 1.0 + pan)
    return f"pan=stereo|FL={left_gain}*c0+{left_gain}*c1|FR={right_gain}*c0+{right_gain}*c1"


def mix_stems(
    stem_paths: List[str],
    mix_config: Optional[List[Dict]] = None,
    output_file: str = "",
    on_progress: Optional[Callable] = None,
) -> RemixResult:
    """Mix multiple audio stems together with volume and pan controls.

    Args:
        stem_paths: List of paths to stem audio files.
        mix_config: Optional list of dicts with 'volume' (0-2), 'pan' (-1 to 1),
                    'mute' (bool) per stem. Index-matched to stem_paths.
        output_file: Output file path. Auto-generated if empty.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        RemixResult with output path and stem count.
    """
    if not stem_paths:
        raise ValueError("No stem paths provided")

    valid_stems = [p for p in stem_paths if os.path.isfile(p)]
    if not valid_stems:
        raise ValueError("No valid stem files found")

    if output_file == "":
        fd, output_file = tempfile.mkstemp(suffix=".wav", prefix="mix_")
        os.close(fd)

    if on_progress:
        on_progress(5, f"Mixing {len(valid_stems)} stems...")

    configs = mix_config or []

    # Build filter_complex
    cmd_builder = FFmpegCmd()
    filter_parts = []
    mix_labels = []
    stem_idx = 0

    for i, stem in enumerate(valid_stems):
        cfg = configs[i] if i < len(configs) else {}

        if cfg.get("mute", False):
            continue

        cmd_builder.input(stem)

        volume = float(cfg.get("volume", 1.0))
        volume = max(0.0, min(2.0, volume))
        pan = float(cfg.get("pan", 0.0))

        # Build per-stem filter chain
        chain = f"[{stem_idx}:a]volume={volume}"
        if abs(pan) > 0.01:
            left_gain = min(1.0, 1.0 - pan)
            right_gain = min(1.0, 1.0 + pan)
            chain += f",pan=stereo|FL={left_gain}*c0|FR={right_gain}*c0"

        label = f"s{stem_idx}"
        chain += f"[{label}]"
        filter_parts.append(chain)
        mix_labels.append(f"[{label}]")
        stem_idx += 1

    if not mix_labels:
        raise ValueError("All stems are muted")

    # Solo mode: if any stem has solo=True, only include solo stems
    solo_indices = [i for i, cfg in enumerate(configs) if cfg.get("solo", False)]
    if solo_indices:
        # Rebuild with only solo stems
        solo_labels = [f"[s{i}]" for i in solo_indices if i < stem_idx]
        if solo_labels:
            mix_labels = solo_labels

    # Amix
    n_mix = len(mix_labels)
    labels_str = "".join(mix_labels)
    if n_mix > 1:
        filter_parts.append(
            f"{labels_str}amix=inputs={n_mix}:duration=longest"
            f":dropout_transition=2:normalize=0[out]"
        )
    else:
        # Single stem: just rename the label
        existing = mix_labels[0]  # e.g. [s0]
        filter_parts.append(f"{existing}acopy[out]")

    fc = ";".join(filter_parts)
    cmd_builder.filter_complex(fc, maps=["[out]"])
    cmd_builder.audio_codec("pcm_s16le")
    cmd_builder.option("ar", "44100")
    cmd_builder.output(output_file)

    cmd = cmd_builder.build()

    if on_progress:
        on_progress(40, "Running mix...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "Mix complete")

    return RemixResult(
        output_path=output_file,
        stems_processed=n_mix,
        effects_applied=0,
        duration=0,
    )


def remix_stems(
    stem_paths: List[str],
    effects_config: List[Dict],
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> RemixResult:
    """Apply per-stem effects and recombine into a final mix.

    Args:
        stem_paths: List of paths to stem audio files.
        effects_config: List of dicts per stem with:
            - 'effects': list of {'name': str, 'params': dict}
            - 'volume': float (0-2, default 1.0)
            - 'pan': float (-1 to 1, default 0.0)
            - 'mute': bool (default False)
            - 'solo': bool (default False)
        output: Output file path. Auto-generated if None.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        RemixResult with output path and processing summary.
    """
    if not stem_paths:
        raise ValueError("No stem paths provided")

    valid_stems = [p for p in stem_paths if os.path.isfile(p)]
    if not valid_stems:
        raise ValueError("No valid stem files found")

    if on_progress:
        on_progress(5, f"Processing {len(valid_stems)} stems with effects...")

    temp_files = []
    total_effects = 0

    try:
        processed_paths = []

        for i, stem in enumerate(valid_stems):
            cfg = effects_config[i] if i < len(effects_config) else {}
            effects_list = cfg.get("effects", [])

            if effects_list:
                # Chain effects sequentially
                current_path = stem
                for j, eff_dict in enumerate(effects_list):
                    eff = StemEffect(
                        name=eff_dict.get("name", ""),
                        params=eff_dict.get("params", {}),
                    )
                    pct = 10 + int((i * len(effects_list) + j) / max(1, len(valid_stems) * max(1, len(effects_list))) * 50)
                    if on_progress:
                        on_progress(pct, f"Applying {eff.name} to stem {i + 1}...")

                    out = apply_stem_effect(current_path, eff)
                    if current_path != stem:
                        temp_files.append(current_path)
                    current_path = out
                    total_effects += 1

                processed_paths.append(current_path)
                if current_path != stem:
                    temp_files.append(current_path)
            else:
                processed_paths.append(stem)

        if on_progress:
            on_progress(70, "Mixing processed stems...")

        # Mix with volume/pan/mute/solo from effects_config
        mix_config = []
        for i in range(len(processed_paths)):
            cfg = effects_config[i] if i < len(effects_config) else {}
            mix_config.append({
                "volume": cfg.get("volume", 1.0),
                "pan": cfg.get("pan", 0.0),
                "mute": cfg.get("mute", False),
                "solo": cfg.get("solo", False),
            })

        result = mix_stems(
            processed_paths,
            mix_config=mix_config,
            output_file=output or "",
            on_progress=on_progress,
        )
        result.effects_applied = total_effects

        return result

    finally:
        # Clean up intermediate temp files
        for tmp in temp_files:
            try:
                if os.path.isfile(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
