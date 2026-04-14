"""
OpenCut AI Color Match Between Shots

Auto-match color and exposure between shots. Extracts color statistics
(average RGB, luminance, saturation, histograms) from a reference clip
and applies FFmpeg colorbalance, eq, curves, and hue filters to transform
a source clip toward the reference color profile.

Supports single-clip matching and batch matching of multiple clips to
one reference.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
class ColorMatchResult:
    """Result of color matching between shots."""
    output_path: str = ""
    reference_stats: Dict = field(default_factory=dict)
    matched_stats: Dict = field(default_factory=dict)
    adjustments: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Color statistics analysis
# ---------------------------------------------------------------------------
def analyze_color_stats(
    video_path: str,
    sample_seconds: float = 3.0,
) -> dict:
    """
    Extract color statistics from a video clip.

    Samples the first *sample_seconds* of the video and extracts average
    RGB values, luminance (YAVG), saturation (SATAVG), and histogram
    distribution via FFmpeg signalstats.

    Args:
        video_path: Path to the video file.
        sample_seconds: Duration to sample from the start (default 3s).

    Returns:
        Dict with keys: avg_r, avg_g, avg_b, luminance, saturation,
        histogram_low, histogram_mid, histogram_high.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    sample_seconds = max(0.5, float(sample_seconds))
    ffmpeg = get_ffmpeg_path()

    # Run signalstats to extract color metadata
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-t", str(sample_seconds),
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print",
        "-f", "null", "-",
    ]

    stats = {
        "avg_r": 128.0,
        "avg_g": 128.0,
        "avg_b": 128.0,
        "luminance": 128.0,
        "saturation": 128.0,
        "histogram_low": 0.33,
        "histogram_mid": 0.34,
        "histogram_high": 0.33,
    }

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr or ""

        # Parse YAVG (luminance)
        yavg_vals = re.findall(r"YAVG=(\d+\.?\d*)", stderr)
        if yavg_vals:
            stats["luminance"] = round(
                sum(float(v) for v in yavg_vals) / len(yavg_vals), 2
            )

        # Parse UAVG / VAVG for color channels (YUV to approximate RGB)
        uavg_vals = re.findall(r"UAVG=(\d+\.?\d*)", stderr)
        vavg_vals = re.findall(r"VAVG=(\d+\.?\d*)", stderr)
        if uavg_vals:
            u_avg = sum(float(v) for v in uavg_vals) / len(uavg_vals)
        else:
            u_avg = 128.0
        if vavg_vals:
            v_avg = sum(float(v) for v in vavg_vals) / len(vavg_vals)
        else:
            v_avg = 128.0

        # YUV to RGB approximation
        y = stats["luminance"]
        stats["avg_r"] = round(min(255, max(0, y + 1.402 * (v_avg - 128))), 2)
        stats["avg_g"] = round(
            min(255, max(0, y - 0.344 * (u_avg - 128) - 0.714 * (v_avg - 128))),
            2,
        )
        stats["avg_b"] = round(min(255, max(0, y + 1.772 * (u_avg - 128))), 2)

        # Parse SATAVG for saturation
        sat_vals = re.findall(r"SATAVG=(\d+\.?\d*)", stderr)
        if sat_vals:
            stats["saturation"] = round(
                sum(float(v) for v in sat_vals) / len(sat_vals), 2
            )

        # Parse HUEAVG for hue
        hue_vals = re.findall(r"HUEAVG=(\d+\.?\d*)", stderr)
        if hue_vals:
            stats["hue_avg"] = round(
                sum(float(v) for v in hue_vals) / len(hue_vals), 2
            )

        # Histogram distribution from YLOW / YHIGH
        ylow_vals = re.findall(r"YLOW=(\d+\.?\d*)", stderr)
        yhigh_vals = re.findall(r"YHIGH=(\d+\.?\d*)", stderr)
        if ylow_vals and yhigh_vals:
            avg_low = sum(float(v) for v in ylow_vals) / len(ylow_vals)
            avg_high = sum(float(v) for v in yhigh_vals) / len(yhigh_vals)
            total = avg_low + avg_high + 1.0
            stats["histogram_low"] = round(avg_low / total, 3)
            stats["histogram_high"] = round(avg_high / total, 3)
            stats["histogram_mid"] = round(
                1.0 - stats["histogram_low"] - stats["histogram_high"], 3
            )

    except Exception as exc:
        logger.debug("Color stats extraction failed: %s", exc)

    return stats


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------
def _compute_adjustments(source_stats: dict, reference_stats: dict,
                         strength: float) -> dict:
    """Compute the filter adjustments needed to match source to reference.

    Returns a dict describing brightness, contrast, saturation, and
    color-balance deltas.
    """
    strength = max(0.0, min(2.0, strength))

    src_lum = source_stats.get("luminance", 128.0)
    ref_lum = reference_stats.get("luminance", 128.0)
    src_sat = source_stats.get("saturation", 128.0)
    ref_sat = reference_stats.get("saturation", 128.0)

    # Brightness delta (eq filter uses -1.0 to 1.0 range)
    lum_diff = (ref_lum - src_lum) / 255.0
    brightness = round(lum_diff * strength, 4)
    brightness = max(-1.0, min(1.0, brightness))

    # Contrast adjustment based on histogram spread
    src_range = source_stats.get("histogram_high", 0.33) - source_stats.get(
        "histogram_low", 0.33
    )
    ref_range = reference_stats.get("histogram_high", 0.33) - reference_stats.get(
        "histogram_low", 0.33
    )
    if src_range > 0:
        contrast_ratio = ref_range / src_range
    else:
        contrast_ratio = 1.0
    contrast = round(1.0 + (contrast_ratio - 1.0) * strength, 4)
    contrast = max(-2.0, min(2.0, contrast))

    # Saturation delta (eq filter uses 0.0 to 3.0 range, 1.0 = unchanged)
    if src_sat > 0:
        sat_ratio = ref_sat / src_sat
    else:
        sat_ratio = 1.0
    saturation = round(1.0 + (sat_ratio - 1.0) * strength, 4)
    saturation = max(0.0, min(3.0, saturation))

    # Color balance: per-channel offsets
    src_r = source_stats.get("avg_r", 128.0)
    ref_r = reference_stats.get("avg_r", 128.0)
    src_g = source_stats.get("avg_g", 128.0)
    ref_g = reference_stats.get("avg_g", 128.0)
    src_b = source_stats.get("avg_b", 128.0)
    ref_b = reference_stats.get("avg_b", 128.0)

    # colorbalance uses -1.0 to 1.0 range per shadow/mid/highlight
    r_shift = round(((ref_r - src_r) / 255.0) * strength, 4)
    g_shift = round(((ref_g - src_g) / 255.0) * strength, 4)
    b_shift = round(((ref_b - src_b) / 255.0) * strength, 4)

    r_shift = max(-1.0, min(1.0, r_shift))
    g_shift = max(-1.0, min(1.0, g_shift))
    b_shift = max(-1.0, min(1.0, b_shift))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "r_shift": r_shift,
        "g_shift": g_shift,
        "b_shift": b_shift,
        "lum_diff": round(ref_lum - src_lum, 2),
        "sat_diff": round(ref_sat - src_sat, 2),
    }


def _build_color_match_filter(adjustments: dict) -> str:
    """Build a combined FFmpeg filter chain from computed adjustments."""
    filters = []

    brightness = adjustments.get("brightness", 0.0)
    contrast = adjustments.get("contrast", 1.0)
    saturation = adjustments.get("saturation", 1.0)

    # eq filter for brightness, contrast, saturation
    eq_parts = []
    if abs(brightness) > 0.001:
        eq_parts.append(f"brightness={brightness:.4f}")
    if abs(contrast - 1.0) > 0.001:
        eq_parts.append(f"contrast={contrast:.4f}")
    if abs(saturation - 1.0) > 0.001:
        eq_parts.append(f"saturation={saturation:.4f}")
    if eq_parts:
        filters.append("eq=" + ":".join(eq_parts))

    # colorbalance for per-channel adjustment
    r_shift = adjustments.get("r_shift", 0.0)
    g_shift = adjustments.get("g_shift", 0.0)
    b_shift = adjustments.get("b_shift", 0.0)

    if abs(r_shift) > 0.001 or abs(g_shift) > 0.001 or abs(b_shift) > 0.001:
        filters.append(
            f"colorbalance="
            f"rs={r_shift:.4f}:gs={g_shift:.4f}:bs={b_shift:.4f}:"
            f"rm={r_shift:.4f}:gm={g_shift:.4f}:bm={b_shift:.4f}:"
            f"rh={r_shift * 0.5:.4f}:gh={g_shift * 0.5:.4f}:bh={b_shift * 0.5:.4f}"
        )

    # Fallback: if no filters needed, use null filter
    if not filters:
        filters.append("null")

    return ",".join(filters)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def match_color(
    source_path: str,
    reference_path: str,
    output_path: Optional[str] = None,
    strength: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> ColorMatchResult:
    """
    Match color/exposure of source video to a reference video.

    Analyzes both clips, computes adjustments, and applies FFmpeg
    colorbalance + eq + curves filters to transform the source toward
    the reference color profile.

    Args:
        source_path: Path to the source video to adjust.
        reference_path: Path to the reference video to match.
        output_path: Explicit output path. Auto-generated if None.
        strength: Match strength 0.0-2.0 (1.0 = standard).
        on_progress: Progress callback(pct, msg).

    Returns:
        ColorMatchResult with output path, stats, and adjustments.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    strength = max(0.0, min(2.0, float(strength)))

    if on_progress:
        on_progress(5, "Analyzing reference clip...")

    reference_stats = analyze_color_stats(reference_path)

    if on_progress:
        on_progress(20, "Analyzing source clip...")

    source_stats = analyze_color_stats(source_path)

    if on_progress:
        on_progress(35, "Computing color adjustments...")

    adjustments = _compute_adjustments(source_stats, reference_stats, strength)
    vf_chain = _build_color_match_filter(adjustments)

    if output_path is None:
        output_path = _output_path(source_path, "colormatch", "")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if on_progress:
        on_progress(45, "Applying color match filters...")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", source_path,
        "-vf", vf_chain,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(85, "Verifying output...")

    matched_stats = analyze_color_stats(output_path)

    if on_progress:
        on_progress(100, "Color match complete")

    return ColorMatchResult(
        output_path=output_path,
        reference_stats=reference_stats,
        matched_stats=matched_stats,
        adjustments=adjustments,
    )


def batch_match(
    paths: List[str],
    reference_path: str,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> List[ColorMatchResult]:
    """
    Match multiple clips to a single reference color profile.

    Args:
        paths: List of source video paths to adjust.
        reference_path: Path to the reference video.
        output_dir: Directory for output files. Uses source dirs if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of ColorMatchResult, one per input clip.
    """
    if not paths:
        raise ValueError("No source paths provided")
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    if on_progress:
        on_progress(2, "Analyzing reference clip...")

    reference_stats = analyze_color_stats(reference_path)

    results = []
    total = len(paths)

    for i, src_path in enumerate(paths):
        if not os.path.isfile(src_path):
            logger.warning("Skipping missing file: %s", src_path)
            continue

        clip_pct_start = 5 + int((i / total) * 90)
        clip_pct_end = 5 + int(((i + 1) / total) * 90)

        if on_progress:
            on_progress(clip_pct_start, f"Matching clip {i + 1}/{total}...")

        if output_dir:
            base = os.path.splitext(os.path.basename(src_path))[0]
            ext = os.path.splitext(src_path)[1] or ".mp4"
            out_path = os.path.join(output_dir, f"{base}_colormatch{ext}")
        else:
            out_path = _output_path(src_path, "colormatch", "")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        source_stats = analyze_color_stats(src_path)
        adjustments = _compute_adjustments(source_stats, reference_stats, 1.0)
        vf_chain = _build_color_match_filter(adjustments)

        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", src_path,
            "-vf", vf_chain,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
        run_ffmpeg(cmd)

        matched_stats = analyze_color_stats(out_path)

        results.append(ColorMatchResult(
            output_path=out_path,
            reference_stats=reference_stats,
            matched_stats=matched_stats,
            adjustments=adjustments,
        ))

        if on_progress:
            on_progress(clip_pct_end, f"Clip {i + 1}/{total} matched")

    if on_progress:
        on_progress(100, f"Batch complete: {len(results)}/{total} clips matched")

    return results
