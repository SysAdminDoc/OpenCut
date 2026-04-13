"""
OpenCut Rolling Shutter Correction Module

Correct rolling shutter (jello) artifacts using FFmpeg's dejudder filter
and vidstab-based stabilization with rolling shutter compensation.
"""

import logging
import os
import tempfile
from typing import Callable, Optional

from opencut.helpers import get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


def correct_rolling_shutter(
    input_path: str,
    strength: float = 0.5,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Correct rolling shutter (jello/wobble) artifacts in a video.

    Uses a two-pass approach:
    1. FFmpeg vidstabdetect to analyse motion
    2. FFmpeg vidstabtransform with rolling shutter compensation

    For light correction, the dejudder filter is also applied.

    Args:
        input_path: Path to the input video.
        strength: Correction intensity 0.0-1.0 (0=none, 1=max).
        output_path_override: Custom output path (auto-generated if None).
        on_progress: Progress callback (percent, message).

    Returns:
        dict with output_path and correction parameters.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    strength = max(0.0, min(1.0, float(strength)))
    out = output_path_override or output_path(input_path, "rs_corrected")

    if on_progress:
        on_progress(5, "Analysing video for rolling shutter...")

    info = get_video_info(input_path)
    fps = info.get("fps", 30.0)

    # Map strength to vidstab parameters
    # shakiness: 1-10 (how much to detect)
    shakiness = max(1, int(strength * 10))
    # smoothing: higher = more aggressive correction
    smoothing = max(1, int(strength * 30))
    # stepsize: analysis step size (smaller = more accurate but slower)
    stepsize = max(1, 6 - int(strength * 4))

    # Create temp file for vidstab transforms data
    transforms_fd = tempfile.NamedTemporaryFile(
        suffix=".trf", delete=False, prefix="opencut_rs_"
    )
    transforms_path = transforms_fd.name
    transforms_fd.close()

    try:
        if on_progress:
            on_progress(10, "Pass 1: Detecting motion vectors...")

        # Pass 1: Detect motion
        detect_filter = (
            f"vidstabdetect=shakiness={shakiness}:accuracy=15"
            f":stepsize={stepsize}:result='{transforms_path}'"
        )
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vf", detect_filter,
            "-f", "null", "-",
        ], timeout=7200)

        if on_progress:
            on_progress(50, "Pass 2: Applying rolling shutter correction...")

        # Pass 2: Apply transform with rolling shutter compensation
        # optzoom=0 keeps original zoom, tripod=0 allows camera movement
        transform_filter = (
            f"vidstabtransform=input='{transforms_path}'"
            f":smoothing={smoothing}:optzoom=0:interpol=linear"
        )

        # Add dejudder for additional temporal smoothing based on strength
        if strength > 0.3:
            # dejudder helps with temporal wobble
            cycle = max(2, int(fps / 6))
            vf = f"{transform_filter},dejudder=cycle={cycle}"
        else:
            vf = transform_filter

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            out,
        ], timeout=7200)

    finally:
        try:
            os.unlink(transforms_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Rolling shutter correction complete!")

    return {
        "output_path": out,
        "strength": strength,
        "shakiness": shakiness,
        "smoothing": smoothing,
    }
