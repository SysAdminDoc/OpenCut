"""
OpenCut Low-Light Enhancement Module

Beyond simple denoising -- actual light recovery for dark footage.
Uses FFmpeg curves for shadow lift + midtone boost + highlight protection,
followed by detail recovery via unsharp mask and optional denoise.

Topaz Nyx model does this at the AI level; this is FFmpeg-native.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class LowLightResult:
    """Result of low-light enhancement."""
    output_path: str = ""
    original_avg_luminance: float = 0.0
    enhanced_avg_luminance: float = 0.0
    denoise_applied: bool = False


# ---------------------------------------------------------------------------
# Luminance analysis
# ---------------------------------------------------------------------------
def _measure_avg_luminance(video_path: str, sample_seconds: float = 5.0) -> float:
    """Measure average luminance (Y channel) from video histogram.

    Returns average luminance as 0-255 value.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-t", str(sample_seconds),
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr

        yavg_vals = re.findall(r"YAVG=(\d+\.?\d*)", stderr)
        if yavg_vals:
            avg = sum(float(v) for v in yavg_vals) / len(yavg_vals)
            return round(avg, 2)
    except Exception as exc:
        logger.debug("Luminance measurement failed: %s", exc)

    # Fallback: use histogram with lavfi
    cmd2 = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-t", str(sample_seconds),
        "-vf", "histogram=level_height=200:scale_height=12,metadata=mode=print",
        "-f", "null", "-",
    ]
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
        # If signalstats failed, try to extract from histogram metadata
        yavg_vals2 = re.findall(r"YAVG=(\d+\.?\d*)", result2.stderr)
        if yavg_vals2:
            return round(sum(float(v) for v in yavg_vals2) / len(yavg_vals2), 2)
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Filter builders
# ---------------------------------------------------------------------------
def _build_curves_filter(strength: float) -> str:
    """Build FFmpeg curves filter for low-light lift.

    Shadow lift + midtone boost + highlight protection.
    Strength 0.0 = no change, 1.0 = standard lift, 2.0 = aggressive.
    """
    # Clamp strength
    strength = max(0.0, min(2.0, strength))

    # Scale control points by strength
    # Shadows: lift dark values significantly
    shadow_out = min(0.35, 0.15 * strength)
    # Midtones: gentle boost
    mid_in = 0.5
    mid_out = min(0.75, 0.5 + 0.12 * strength)
    # Highlights: protect from clipping
    hi_in = 0.85
    hi_out = min(0.95, 0.85 + 0.05 * strength)

    curve = (
        f"curves=all='"
        f"0/0 "
        f"0.08/{shadow_out:.3f} "
        f"{mid_in}/{mid_out:.3f} "
        f"{hi_in}/{hi_out:.3f} "
        f"1/1'"
    )
    return curve


def _build_unsharp_filter(strength: float) -> str:
    """Build unsharp mask filter to recover detail after brightness lift."""
    # Moderate sharpening -- too aggressive amplifies noise
    amount = min(2.0, 0.5 + 0.5 * strength)
    return f"unsharp=5:5:{amount:.1f}:5:5:0.0"


def _build_denoise_filter(strength: float) -> str:
    """Build denoise filter -- noise becomes visible after brightness boost.

    Uses nlmeans for quality, falls back conceptually to hqdn3d.
    """
    # nlmeans strength scales with how much we boosted
    s = min(8.0, 3.0 + 2.0 * strength)
    return f"nlmeans=s={s:.1f}:p=7:pc=5:r=15:rc=9"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def enhance_low_light(
    video_path: str,
    output_path: Optional[str] = None,
    strength: float = 1.0,
    denoise: bool = True,
    on_progress: Optional[Callable] = None,
) -> LowLightResult:
    """
    Low-Light Enhancement -- recover detail from dark footage.

    Uses FFmpeg curves for shadow/midtone lift, unsharp for detail recovery,
    and optional nlmeans denoise (noise amplifies with brightness boost).

    Args:
        video_path: Path to input video.
        output_path: Explicit output path. Auto-generated if None.
        strength: Enhancement strength 0.0-2.0 (1.0 = standard).
        denoise: Apply denoise after brightness boost (recommended).
        on_progress: Progress callback(pct, msg).

    Returns:
        LowLightResult with output path and luminance stats.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    strength = max(0.0, min(2.0, float(strength)))

    if on_progress:
        on_progress(5, "Measuring exposure level...")

    # Auto-detect exposure
    original_lum = _measure_avg_luminance(video_path)

    if on_progress:
        on_progress(15, f"Average luminance: {original_lum:.1f}/255")

    # If not actually dark, skip enhancement
    # 40% of 255 = 102
    luminance_pct = (original_lum / 255.0) * 100.0 if original_lum > 0 else 0.0
    if luminance_pct > 40.0:
        logger.info("Video not low-light (avg luminance %.1f%%), skipping enhancement",
                     luminance_pct)
        if on_progress:
            on_progress(100, f"Not low-light ({luminance_pct:.0f}% luminance), skipped")

        if output_path is None:
            output_path = _output_path(video_path, "lowlight", "")

        # Copy original -- no enhancement needed
        import shutil
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copy2(video_path, output_path)

        return LowLightResult(
            output_path=output_path,
            original_avg_luminance=original_lum,
            enhanced_avg_luminance=original_lum,
            denoise_applied=False,
        )

    if on_progress:
        on_progress(20, "Building enhancement filters...")

    # Build filter chain
    filters = []

    # 1. Curves: shadow lift + midtone boost
    filters.append(_build_curves_filter(strength))

    # 2. Unsharp: recover detail after brightness boost
    filters.append(_build_unsharp_filter(strength))

    # 3. Denoise: noise amplifies with brightness
    denoise_applied = False
    if denoise:
        filters.append(_build_denoise_filter(strength))
        denoise_applied = True

    vf_chain = ",".join(filters)

    # Build output path
    if output_path is None:
        output_path = _output_path(video_path, "lowlight", "")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if on_progress:
        on_progress(30, "Enhancing low-light footage...")

    # Run FFmpeg
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", video_path,
        "-vf", vf_chain,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(85, "Measuring enhanced exposure...")

    # Measure enhanced luminance
    enhanced_lum = _measure_avg_luminance(output_path)

    if on_progress:
        on_progress(100, f"Low-light enhanced: {original_lum:.0f} -> {enhanced_lum:.0f} avg luminance")

    return LowLightResult(
        output_path=output_path,
        original_avg_luminance=original_lum,
        enhanced_avg_luminance=enhanced_lum,
        denoise_applied=denoise_applied,
    )
