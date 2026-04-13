"""
OpenCut AI Auto-Color Grading

Mood-based color grading, reference image color matching, and LUT application.
Maps mood names to predefined FFmpeg color correction filter chains.

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import os
import re
import subprocess as _sp
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Mood Presets  -- each maps to an FFmpeg filter chain string
# ---------------------------------------------------------------------------
MOOD_PRESETS: Dict[str, Dict] = {
    "warm sunset": {
        "description": "Golden-hour warmth with lifted shadows and amber highlights",
        "filters": (
            "colorbalance=rs=0.15:gs=0.05:bs=-0.1:rh=0.12:gh=0.04:bh=-0.08,"
            "eq=brightness=0.04:saturation=1.25:contrast=1.05,"
            "curves=master='0/0 0.25/0.30 0.5/0.55 0.75/0.80 1/1'"
        ),
    },
    "teal orange": {
        "description": "Hollywood blockbuster teal-and-orange split-tone",
        "filters": (
            "colorbalance=rs=-0.05:gs=-0.1:bs=0.15:rm=0.0:gm=-0.05:bm=0.05:rh=0.15:gh=0.05:bh=-0.12,"
            "eq=saturation=1.3:contrast=1.1,"
            "curves=master='0/0 0.15/0.08 0.5/0.52 0.85/0.92 1/1'"
        ),
    },
    "horror": {
        "description": "Desaturated cold tones with crushed blacks and high contrast",
        "filters": (
            "colorbalance=rs=-0.05:gs=-0.08:bs=0.03:rm=-0.03:gm=-0.05:bm=0.02,"
            "eq=brightness=-0.05:saturation=0.5:contrast=1.3:gamma=0.85,"
            "curves=master='0/0 0.2/0.05 0.5/0.42 0.8/0.85 1/1'"
        ),
    },
    "noir": {
        "description": "Classic film noir -- near-monochrome with deep blacks",
        "filters": (
            "colorbalance=rs=0.02:gs=0.0:bs=0.0,"
            "eq=saturation=0.15:contrast=1.4:brightness=-0.03:gamma=0.9,"
            "curves=master='0/0 0.15/0.03 0.5/0.45 0.85/0.95 1/1'"
        ),
    },
    "vintage": {
        "description": "Faded retro film with lifted blacks and warm tone",
        "filters": (
            "colorbalance=rs=0.1:gs=0.05:bs=-0.05:rm=0.05:gm=0.02:bm=-0.02,"
            "eq=saturation=0.85:contrast=0.9:brightness=0.03,"
            "curves=master='0/0.06 0.25/0.28 0.5/0.52 0.75/0.76 1/0.95'"
        ),
    },
    "cyberpunk": {
        "description": "Neon-saturated magenta and cyan push",
        "filters": (
            "colorbalance=rs=0.1:gs=-0.1:bs=0.15:rm=0.08:gm=-0.08:bm=0.1,"
            "eq=saturation=1.6:contrast=1.2:brightness=0.02,"
            "curves=master='0/0 0.2/0.15 0.5/0.55 0.8/0.9 1/1'"
        ),
    },
    "cool blue": {
        "description": "Clean cool blue tones for corporate or tech content",
        "filters": (
            "colorbalance=rs=-0.08:gs=-0.03:bs=0.12:rh=-0.05:gh=0.0:bh=0.08,"
            "eq=saturation=1.1:contrast=1.05:brightness=0.01"
        ),
    },
    "bleach bypass": {
        "description": "Desaturated high-contrast with metallic highlights",
        "filters": (
            "eq=saturation=0.4:contrast=1.5:brightness=-0.02:gamma=0.95,"
            "curves=master='0/0 0.2/0.08 0.5/0.5 0.8/0.92 1/1'"
        ),
    },
}


def list_mood_presets() -> List[str]:
    """Return list of available mood preset names."""
    return sorted(MOOD_PRESETS.keys())


def auto_grade(
    input_path: str,
    mood: Optional[str] = None,
    reference_image: Optional[str] = None,
    lut_name: Optional[str] = None,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply AI auto-color grading to a video.

    Supports three modes (in priority order):
    1. mood: Apply a predefined mood color grade
    2. reference_image: Match colors to a reference image
    3. lut_name: Apply a LUT from the library

    Args:
        input_path: Source video file.
        mood: Mood preset name (e.g. "warm sunset", "noir").
        reference_image: Path to reference image for color matching.
        lut_name: Name of LUT from lut_library to apply.
        output_path: Explicit output path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and grading_method.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not mood and not reference_image and not lut_name:
        raise ValueError("At least one of mood, reference_image, or lut_name is required")

    # Determine grading method
    if mood:
        return _grade_by_mood(input_path, mood, output_path, on_progress)
    elif reference_image:
        return _grade_by_reference(input_path, reference_image, output_path, on_progress)
    else:
        return _grade_by_lut(input_path, lut_name, output_path, on_progress)


def _grade_by_mood(
    input_path: str,
    mood: str,
    out_path: Optional[str],
    on_progress: Optional[Callable],
) -> dict:
    """Apply a mood preset via FFmpeg filter chain."""
    mood_lower = mood.lower().strip()
    if mood_lower not in MOOD_PRESETS:
        available = ", ".join(list_mood_presets())
        raise ValueError(f"Unknown mood '{mood}'. Available: {available}")

    preset = MOOD_PRESETS[mood_lower]
    if on_progress:
        on_progress(10, f"Applying mood: {mood} -- {preset['description']}")

    safe_mood = mood_lower.replace(" ", "_")
    if out_path is None:
        out_path = _output_path(input_path, f"grade_{safe_mood}")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(preset["filters"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out_path)
        .build()
    )

    if on_progress:
        on_progress(30, "Rendering color grade...")

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, f"Color grade complete: {mood}")

    return {
        "output_path": out_path,
        "grading_method": "mood",
        "mood": mood_lower,
        "description": preset["description"],
    }


def _extract_color_stats(filepath: str) -> dict:
    """Extract average RGB color statistics from an image or video frame."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-i", filepath,
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print",
        "-vframes", "1",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
    except _sp.TimeoutExpired:
        return {"hueavg": 128, "satavg": 128, "yavg": 128}

    stats = {"hueavg": 128, "satavg": 128, "yavg": 128}
    for line in result.stderr.splitlines():
        for key in ("HUEAVG", "SATAVG", "YAVG"):
            m = re.search(rf"lavfi\.signalstats\.{key}=(\d+\.?\d*)", line)
            if m:
                stats[key.lower()] = float(m.group(1))
    return stats


def _grade_by_reference(
    input_path: str,
    reference_image: str,
    out_path: Optional[str],
    on_progress: Optional[Callable],
) -> dict:
    """Match video colors to a reference image using colorbalance adjustment."""
    if not os.path.isfile(reference_image):
        raise FileNotFoundError(f"Reference image not found: {reference_image}")

    if on_progress:
        on_progress(10, "Analyzing reference image colors...")

    ref_stats = _extract_color_stats(reference_image)
    src_stats = _extract_color_stats(input_path)

    if on_progress:
        on_progress(30, "Computing color adjustment...")

    # Compute adjustment deltas (normalized to -1..1 range for colorbalance)
    # YAVG is 0-255, we normalize difference to a small adjustment factor
    y_diff = (ref_stats["yavg"] - src_stats["yavg"]) / 255.0
    sat_diff = (ref_stats["satavg"] - src_stats["satavg"]) / 255.0

    # Clamp adjustments to reasonable range
    brightness_adj = max(-0.2, min(0.2, y_diff * 0.5))
    sat_factor = max(0.5, min(1.8, 1.0 + sat_diff * 0.8))

    # Build filter chain for color matching
    filters = (
        f"eq=brightness={brightness_adj:.3f}:saturation={sat_factor:.3f},"
        f"colorbalance=rs={y_diff * 0.1:.3f}:gs={y_diff * 0.05:.3f}:bs={-y_diff * 0.05:.3f}"
    )

    if out_path is None:
        out_path = _output_path(input_path, "grade_ref")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(filters)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out_path)
        .build()
    )

    if on_progress:
        on_progress(50, "Rendering reference-matched grade...")

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Reference color match complete")

    return {
        "output_path": out_path,
        "grading_method": "reference",
        "adjustments": {
            "brightness": round(brightness_adj, 3),
            "saturation": round(sat_factor, 3),
        },
    }


def _grade_by_lut(
    input_path: str,
    lut_name: str,
    out_path: Optional[str],
    on_progress: Optional[Callable],
) -> dict:
    """Apply a LUT from the lut_library."""
    from opencut.core.lut_library import apply_lut

    if on_progress:
        on_progress(10, f"Applying LUT: {lut_name}...")

    if out_path is None:
        out_path = _output_path(input_path, f"grade_lut_{lut_name.replace('/', '_')}")

    result_path = apply_lut(
        input_path=input_path,
        lut_name=lut_name,
        intensity=1.0,
        output_path=out_path,
        on_progress=on_progress,
    )

    if on_progress:
        on_progress(100, f"LUT grade complete: {lut_name}")

    return {
        "output_path": result_path if isinstance(result_path, str) else out_path,
        "grading_method": "lut",
        "lut_name": lut_name,
    }
