"""
OpenCut Advanced Stabilization Modes Module

Three stabilization modes via FFmpeg vidstab (two-pass):
- smooth: standard stabilization with configurable smoothing
- lockdown: zero smoothing for a tripod-lock effect
- perspective: vidstab with optzoom=2 for perspective correction
"""

import logging
import os
import tempfile
from typing import Callable, Optional

from opencut.helpers import output_path, run_ffmpeg

logger = logging.getLogger("opencut")

VALID_MODES = ("smooth", "lockdown", "perspective")


def stabilize_advanced(
    input_path: str,
    mode: str = "smooth",
    smoothing: int = 30,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Advanced video stabilization using two-pass vidstab.

    Modes:
        smooth: Standard stabilization with configurable smoothing value.
        lockdown: Zero smoothing for a tripod-lock effect (static framing).
        perspective: vidstab with optzoom=2 for perspective correction.

    Args:
        input_path: Path to the input video.
        mode: Stabilization mode - "smooth", "lockdown", or "perspective".
        smoothing: Smoothing radius for "smooth" mode (1-100, default 30).
        output_path_override: Custom output path (auto-generated if None).
        on_progress: Progress callback (percent, message).

    Returns:
        dict with output_path plus stabilization stats.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(VALID_MODES)}")

    smoothing = max(1, min(100, int(smoothing)))
    out = output_path_override or output_path(input_path, f"stabilized_{mode}")

    if on_progress:
        on_progress(5, f"Starting {mode} stabilization...")

    # Create temp file for transforms data
    transforms_fd = tempfile.NamedTemporaryFile(
        suffix=".trf", delete=False, prefix="opencut_stab_"
    )
    transforms_path = transforms_fd.name
    transforms_fd.close()

    try:
        # -------------------------------------------------------------------
        # Pass 1: Motion detection (common to all modes)
        # -------------------------------------------------------------------
        if on_progress:
            on_progress(10, "Pass 1: Analysing camera motion...")

        detect_filter = (
            f"vidstabdetect=shakiness=8:accuracy=15"
            f":stepsize=4:result='{transforms_path}'"
        )
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vf", detect_filter,
            "-f", "null", "-",
        ], timeout=7200)

        if on_progress:
            on_progress(50, "Pass 2: Applying stabilization...")

        # -------------------------------------------------------------------
        # Pass 2: Apply transform (mode-specific)
        # -------------------------------------------------------------------
        if mode == "lockdown":
            # Zero smoothing = tripod lock effect
            transform_filter = (
                f"vidstabtransform=input='{transforms_path}'"
                f":smoothing=0:optzoom=0:interpol=linear"
            )
        elif mode == "perspective":
            # optzoom=2 for perspective/adaptive zoom correction
            transform_filter = (
                f"vidstabtransform=input='{transforms_path}'"
                f":smoothing={smoothing}:optzoom=2:interpol=bicubic"
            )
        else:
            # smooth mode (default)
            transform_filter = (
                f"vidstabtransform=input='{transforms_path}'"
                f":smoothing={smoothing}:optzoom=1:interpol=bilinear"
            )

        # Apply with unsharp mask to recover sharpness lost during transform
        vf = f"{transform_filter},unsharp=5:5:0.8:3:3:0.4"

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            out,
        ], timeout=7200)

        # -------------------------------------------------------------------
        # Extract stabilization stats from transforms file
        # -------------------------------------------------------------------
        max_shift = 0.0
        avg_rotation = 0.0
        line_count = 0

        try:
            with open(transforms_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # vidstab format: frame dx dy da (and possibly more)
                    if len(parts) >= 4:
                        try:
                            dx = abs(float(parts[1]))
                            dy = abs(float(parts[2]))
                            da = abs(float(parts[3]))
                            shift = (dx ** 2 + dy ** 2) ** 0.5
                            if shift > max_shift:
                                max_shift = shift
                            avg_rotation += da
                            line_count += 1
                        except (ValueError, IndexError):
                            pass
        except (OSError, IOError):
            logger.debug("Could not read transforms file for stats")

        if line_count > 0:
            avg_rotation /= line_count

    finally:
        try:
            os.unlink(transforms_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, f"{mode.capitalize()} stabilization complete!")

    return {
        "output_path": out,
        "mode": mode,
        "smoothing": smoothing if mode == "smooth" else (0 if mode == "lockdown" else smoothing),
        "max_shift": round(max_shift, 2),
        "avg_rotation": round(avg_rotation, 4),
    }
