"""
OpenCut Lens Distortion Correction

Corrects barrel/pincushion lens distortion and levels horizons
using FFmpeg's lenscorrection and rotate filters.

Uses FFmpeg only — no additional dependencies required.
"""

import logging
import math
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Lens correction presets
# ---------------------------------------------------------------------------
LENS_PRESETS: Dict[str, dict] = {
    "gopro_wide": {"k1": -0.3, "k2": 0.0, "name": "GoPro Wide"},
    "gopro_linear": {"k1": -0.1, "k2": 0.0, "name": "GoPro Linear"},
    "dji_mini": {"k1": -0.2, "k2": 0.0, "name": "DJI Mini"},
    "dji_mavic": {"k1": -0.15, "k2": 0.0, "name": "DJI Mavic"},
    "iphone_wide": {"k1": -0.05, "k2": 0.0, "name": "iPhone Wide"},
    "fisheye_moderate": {"k1": -0.5, "k2": 0.1, "name": "Moderate Fisheye"},
    "fisheye_strong": {"k1": -0.7, "k2": 0.2, "name": "Strong Fisheye"},
}


# ---------------------------------------------------------------------------
# Lens distortion correction
# ---------------------------------------------------------------------------
def correct_lens_distortion(
    input_path: str,
    output_path_override: str = None,
    k1: float = None,
    k2: float = None,
    preset: str = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Correct barrel or pincushion lens distortion.

    Uses FFmpeg's lenscorrection filter with radial distortion
    coefficients k1 and k2.  A preset name can be given instead
    of explicit coefficients.

    Args:
        input_path: Source video file.
        output_path_override: Optional output path. Auto-generated if None.
        k1: Primary radial distortion coefficient (negative = barrel,
            positive = pincushion).
        k2: Secondary radial distortion coefficient.
        preset: Name of a camera lens preset (overrides k1/k2).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path of the corrected file.

    Raises:
        ValueError: If neither preset nor k1 is specified, or preset
            is unknown.
    """
    # Resolve coefficients
    if preset:
        preset_lower = preset.lower().strip()
        if preset_lower not in LENS_PRESETS:
            raise ValueError(
                f"Unknown lens preset '{preset}'. "
                f"Available: {', '.join(sorted(LENS_PRESETS))}"
            )
        p = LENS_PRESETS[preset_lower]
        k1 = p["k1"]
        k2 = p["k2"]
    else:
        if k1 is None:
            raise ValueError("Either preset or k1 must be specified")
        if k2 is None:
            k2 = 0.0

    if on_progress:
        label = f"preset '{preset}'" if preset else f"k1={k1}, k2={k2}"
        on_progress(10, f"Correcting lens distortion ({label})...")

    vf = f"lenscorrection=k1={k1}:k2={k2}"

    out = output_path_override or output_path(input_path, "lens_corrected")

    if on_progress:
        on_progress(20, "Applying lenscorrection filter...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Lens correction complete")

    return {"output_path": out}


# ---------------------------------------------------------------------------
# Horizon levelling
# ---------------------------------------------------------------------------
def level_horizon(
    input_path: str,
    output_path_override: str = None,
    angle: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Level a tilted horizon by rotating the video.

    Uses FFmpeg's rotate filter.  Positive angle rotates clockwise,
    negative rotates counter-clockwise (degrees).

    Args:
        input_path: Source video file.
        output_path_override: Optional output path.
        angle: Rotation angle in degrees.  Positive = clockwise.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path.
    """
    if on_progress:
        on_progress(10, f"Levelling horizon ({angle:+.1f} deg)...")

    # FFmpeg rotate filter takes radians and uses bilinear fill
    angle_rad = angle * (math.pi / 180.0)

    # Get video dimensions so we can set the output size to avoid black bars
    info = get_video_info(input_path)
    w = info["width"]
    h = info["height"]

    # The rotate filter expression:
    #   rotate=angle:ow=rotw(angle):oh=roth(angle):fillcolor=black
    # We keep original dimensions and let the edges fill with black
    vf = (
        f"rotate={angle_rad}:"
        f"ow={w}:oh={h}:"
        f"fillcolor=black"
    )

    out = output_path_override or output_path(input_path, "levelled")

    if on_progress:
        on_progress(20, "Applying rotation filter...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Horizon levelling complete")

    return {"output_path": out}


# ---------------------------------------------------------------------------
# List presets
# ---------------------------------------------------------------------------
def list_lens_presets() -> List[dict]:
    """
    Return available lens correction presets.

    Returns:
        List of dicts, each with id, name, k1, k2.
    """
    presets = []
    for preset_id, data in sorted(LENS_PRESETS.items()):
        presets.append({
            "id": preset_id,
            "name": data["name"],
            "k1": data["k1"],
            "k2": data["k2"],
        })
    return presets
