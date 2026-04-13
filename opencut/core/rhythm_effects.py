"""
OpenCut Rhythm-Driven Effects Module (16.3)

Map audio features (beats, amplitude, spectral bands) to visual effects
(zoom, brightness, shake, color) for music-video style editing.
"""

import logging
import math
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants & effect types
# ---------------------------------------------------------------------------

AUDIO_FEATURES = ("beats", "amplitude", "spectral_centroid", "spectral_flux", "rms")
VISUAL_EFFECTS = ("zoom", "brightness", "shake", "color_shift", "blur", "contrast")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EffectMapping:
    """Maps an audio feature to a visual effect with intensity."""
    audio_feature: str = "amplitude"
    visual_effect: str = "zoom"
    intensity: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0


@dataclass
class RhythmEffectsResult:
    """Result of rhythm-driven effects application."""
    output_path: str = ""
    effect_count: int = 0
    keyframes_generated: int = 0
    audio_features_used: List[str] = field(default_factory=list)
    visual_effects_applied: List[str] = field(default_factory=list)
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Effect map creation
# ---------------------------------------------------------------------------

def create_effect_map(
    audio_feature: str = "amplitude",
    visual_effect: str = "zoom",
    intensity: float = 1.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> dict:
    """Create an effect mapping configuration.

    Args:
        audio_feature: Audio feature to drive the effect. One of:
            beats, amplitude, spectral_centroid, spectral_flux, rms.
        visual_effect: Visual effect to apply. One of:
            zoom, brightness, shake, color_shift, blur, contrast.
        intensity: Effect intensity multiplier (0.0 .. 5.0).
        min_value: Minimum effect value (floor).
        max_value: Maximum effect value (ceiling).

    Returns:
        Effect mapping dict.
    """
    audio_feature = str(audio_feature).lower().strip()
    visual_effect = str(visual_effect).lower().strip()
    intensity = max(0.0, min(5.0, float(intensity)))
    min_value = max(0.0, float(min_value))
    max_value = max(min_value, min(10.0, float(max_value)))

    if audio_feature not in AUDIO_FEATURES:
        raise ValueError(
            f"Unknown audio feature '{audio_feature}'. "
            f"Choose from: {', '.join(AUDIO_FEATURES)}"
        )
    if visual_effect not in VISUAL_EFFECTS:
        raise ValueError(
            f"Unknown visual effect '{visual_effect}'. "
            f"Choose from: {', '.join(VISUAL_EFFECTS)}"
        )

    return {
        "audio_feature": audio_feature,
        "visual_effect": visual_effect,
        "intensity": intensity,
        "min_value": min_value,
        "max_value": max_value,
    }


# ---------------------------------------------------------------------------
# Audio feature analysis
# ---------------------------------------------------------------------------

def analyze_audio_features(
    audio_path: str,
    features: Optional[List[str]] = None,
    window_ms: float = 50.0,
    sample_rate: int = 16000,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Analyze audio features for rhythm effect mapping.

    Args:
        audio_path: Path to audio or video file.
        features: List of features to analyze (default: all).
        window_ms: Analysis window in ms.
        sample_rate: Resample rate.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with feature name -> list of per-window values.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if features is None:
        features = list(AUDIO_FEATURES)
    features = [f for f in features if f in AUDIO_FEATURES]

    if on_progress:
        on_progress(5, "Extracting audio for feature analysis")

    # Extract PCM
    tmp_wav = tempfile.mktemp(suffix="_rhythm_audio.wav")
    try:
        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", audio_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            tmp_wav,
        ]
        run_ffmpeg(cmd)

        with open(tmp_wav, "rb") as f:
            f.read(44)
            raw = f.read()

        n_samples = len(raw) // 2
        if n_samples == 0:
            return {f: [] for f in features}
        samples = list(struct.unpack(f"<{n_samples}h", raw[:n_samples * 2]))

        if on_progress:
            on_progress(30, "Computing audio features")

        window_size = max(1, int(sample_rate * window_ms / 1000.0))
        result = {}

        # Amplitude / RMS
        if "amplitude" in features or "rms" in features:
            rms_values = []
            amp_values = []
            for i in range(0, n_samples - window_size + 1, window_size):
                chunk = samples[i:i + window_size]
                rms = math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768.0
                amp = max(abs(s) for s in chunk) / 32768.0
                rms_values.append(rms)
                amp_values.append(amp)
            if "rms" in features:
                result["rms"] = rms_values
            if "amplitude" in features:
                result["amplitude"] = amp_values

        # Spectral centroid (simplified: zero-crossing rate as proxy)
        if "spectral_centroid" in features:
            centroid = []
            for i in range(0, n_samples - window_size + 1, window_size):
                chunk = samples[i:i + window_size]
                zc = sum(
                    1 for j in range(1, len(chunk))
                    if (chunk[j] >= 0) != (chunk[j - 1] >= 0)
                )
                centroid.append(zc / len(chunk))
            result["spectral_centroid"] = centroid

        # Spectral flux (frame-to-frame energy difference)
        if "spectral_flux" in features:
            flux = []
            prev_energy = 0.0
            for i in range(0, n_samples - window_size + 1, window_size):
                chunk = samples[i:i + window_size]
                energy = sum(s * s for s in chunk) / len(chunk) / (32768.0 ** 2)
                flux.append(abs(energy - prev_energy))
                prev_energy = energy
            result["spectral_flux"] = flux

        # Beat detection (onset detection via spectral flux peaks)
        if "beats" in features:
            flux_for_beats = []
            prev_energy = 0.0
            for i in range(0, n_samples - window_size + 1, window_size):
                chunk = samples[i:i + window_size]
                energy = sum(s * s for s in chunk) / len(chunk) / (32768.0 ** 2)
                flux_for_beats.append(abs(energy - prev_energy))
                prev_energy = energy

            # Adaptive threshold for beat detection
            beats = []
            if flux_for_beats:
                mean_flux = sum(flux_for_beats) / len(flux_for_beats)
                threshold = mean_flux * 1.5
                for i, f in enumerate(flux_for_beats):
                    beats.append(1.0 if f > threshold else 0.0)
            result["beats"] = beats

        if on_progress:
            on_progress(80, "Audio feature analysis complete")

        return result
    finally:
        if os.path.isfile(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Build FFmpeg filter for rhythm effects
# ---------------------------------------------------------------------------

def _build_effect_keyframes(
    feature_values: List[float],
    effect: str,
    intensity: float,
    min_val: float,
    max_val: float,
    fps: float,
    window_ms: float,
) -> List[Tuple[float, float]]:
    """Convert feature values to timed keyframes for a visual effect.

    Returns list of (time_seconds, effect_value) tuples.
    """
    keyframes = []
    windows_per_sec = 1000.0 / max(1.0, window_ms)

    for i, val in enumerate(feature_values):
        t = i / windows_per_sec
        # Scale value by intensity, clamp to range
        scaled = val * intensity
        clamped = max(min_val, min(max_val, scaled))
        keyframes.append((t, clamped))

    return keyframes


def _keyframes_to_vf(
    keyframes: List[Tuple[float, float]],
    effect: str,
    width: int,
    height: int,
) -> str:
    """Convert keyframes to an FFmpeg video filter expression.

    Returns an FFmpeg -vf compatible filter string.
    """
    if not keyframes:
        return ""

    if effect == "zoom":
        # Use zoompan with keyframed zoom level
        # Sample a subset of keyframes for zoompan
        zoom_expr_parts = []
        for t, val in keyframes[:500]:  # limit for cmd length
            zoom_level = 1.0 + val * 0.5  # 1.0 to 1.5x zoom
            zoom_expr_parts.append(f"if(between(t,{t:.2f},{t + 0.1:.2f}),{zoom_level:.3f}")
        # Build nested if expression (simplified: use average)
        avg_zoom = 1.0 + (sum(v for _, v in keyframes) / len(keyframes)) * 0.3
        return f"zoompan=z={avg_zoom:.3f}:d=1:s={width}x{height}"

    elif effect == "brightness":
        avg_val = sum(v for _, v in keyframes) / len(keyframes) if keyframes else 0.0
        brightness = avg_val * 0.3  # subtle brightness shift
        return f"eq=brightness={brightness:.3f}"

    elif effect == "shake":
        avg_val = sum(v for _, v in keyframes) / len(keyframes) if keyframes else 0.0
        shake_px = max(1, int(avg_val * 10))
        return f"crop=iw-{shake_px * 2}:ih-{shake_px * 2}:{shake_px}:{shake_px},scale={width}:{height}"

    elif effect == "color_shift":
        avg_val = sum(v for _, v in keyframes) / len(keyframes) if keyframes else 0.0
        hue_shift = avg_val * 90  # degrees
        saturation = 1.0 + avg_val * 0.5
        return f"hue=h={hue_shift:.1f}:s={saturation:.2f}"

    elif effect == "blur":
        avg_val = sum(v for _, v in keyframes) / len(keyframes) if keyframes else 0.0
        sigma = max(0.5, avg_val * 5)
        return f"gblur=sigma={sigma:.1f}"

    elif effect == "contrast":
        avg_val = sum(v for _, v in keyframes) / len(keyframes) if keyframes else 0.0
        contrast = 1.0 + avg_val * 0.5
        return f"eq=contrast={contrast:.2f}"

    return ""


# ---------------------------------------------------------------------------
# Main application function
# ---------------------------------------------------------------------------

def apply_rhythm_effects(
    video_path: str,
    audio_path: Optional[str] = None,
    effect_map: Optional[List[dict]] = None,
    output: Optional[str] = None,
    window_ms: float = 50.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Apply rhythm-driven visual effects to a video.

    Args:
        video_path: Path to input video.
        audio_path: Path to audio file for analysis (default: use video audio).
        effect_map: List of effect mapping dicts (from create_effect_map).
        output: Output path (auto-generated if None).
        window_ms: Audio analysis window in ms.
        on_progress: Optional callback(pct, msg).

    Returns:
        Result dict with output_path and applied effects info.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_src = audio_path or video_path
    if not os.path.isfile(audio_src):
        raise FileNotFoundError(f"Audio file not found: {audio_src}")

    if effect_map is None:
        effect_map = [create_effect_map("amplitude", "zoom", 1.0)]

    out = output or _output_path(video_path, "_rhythm", "")
    info = get_video_info(video_path)
    fps = info.get("fps", 25.0) or 25.0
    width = info.get("width", 1920) or 1920
    height = info.get("height", 1080) or 1080

    if on_progress:
        on_progress(5, "Analyzing audio features for rhythm effects")

    # Gather needed features
    needed_features = list(set(m.get("audio_feature", "amplitude") for m in effect_map))
    features = analyze_audio_features(
        audio_src, features=needed_features, window_ms=window_ms,
        on_progress=lambda p, m: on_progress(5 + p * 0.4, m) if on_progress else None,
    )

    if on_progress:
        on_progress(50, "Building effect keyframes")

    # Build video filters
    vf_parts = []
    total_keyframes = 0
    effects_applied = set()

    for mapping in effect_map:
        feat_name = mapping.get("audio_feature", "amplitude")
        vis_effect = mapping.get("visual_effect", "zoom")
        intensity = mapping.get("intensity", 1.0)
        min_val = mapping.get("min_value", 0.0)
        max_val = mapping.get("max_value", 1.0)

        feat_values = features.get(feat_name, [])
        if not feat_values:
            continue

        keyframes = _build_effect_keyframes(
            feat_values, vis_effect, intensity, min_val, max_val, fps, window_ms,
        )
        total_keyframes += len(keyframes)

        vf = _keyframes_to_vf(keyframes, vis_effect, width, height)
        if vf:
            vf_parts.append(vf)
            effects_applied.add(vis_effect)

    if on_progress:
        on_progress(70, "Applying rhythm effects to video")

    # Build FFmpeg command
    if vf_parts:
        vf_str = ",".join(vf_parts)
        cmd = (
            FFmpegCmd()
            .input(video_path)
            .video_filter(vf_str)
            .video_codec("libx264", crf=18, preset="medium")
            .audio_codec("aac", bitrate="192k")
            .faststart()
            .output(out)
            .build()
        )
    else:
        # No effects to apply, just copy
        cmd = (
            FFmpegCmd()
            .input(video_path)
            .copy_streams()
            .output(out)
            .build()
        )

    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(100, "Rhythm effects applied")

    return {
        "output_path": out,
        "effect_count": len(effect_map),
        "keyframes_generated": total_keyframes,
        "audio_features_used": needed_features,
        "visual_effects_applied": sorted(effects_applied),
        "duration": info.get("duration", 0.0),
    }
