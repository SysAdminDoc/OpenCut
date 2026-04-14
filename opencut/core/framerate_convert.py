"""
OpenCut Frame Rate Conversion with Optical Flow

Convert video frame rates using FFmpeg minterpolate for smooth optical
flow interpolation.

Presets:
  - Smooth:    High-quality interpolation for fluid motion
  - Cinematic: Pulldown removal / cadence detection (e.g. 29.97 -> 24)
  - Sport:     High frame rate output (60/120 fps)
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

FRAMERATE_PRESETS = {
    "smooth": {
        "description": "Smooth interpolation for fluid motion (e.g. 30->60)",
        "mi_mode": "mci",
        "mc_mode": "aobmc",
        "me_mode": "bidir",
        "vsbmc": 1,
        "default_fps": 60,
    },
    "cinematic": {
        "description": "Pulldown removal / telecine reversal (e.g. 29.97->24)",
        "mi_mode": "mci",
        "mc_mode": "aobmc",
        "me_mode": "bidir",
        "vsbmc": 1,
        "default_fps": 24,
    },
    "sport": {
        "description": "High frame rate for fast action (e.g. 30->120)",
        "mi_mode": "mci",
        "mc_mode": "aobmc",
        "me_mode": "bidir",
        "vsbmc": 1,
        "default_fps": 120,
    },
}


@dataclass
class FrameRateResult:
    """Result of frame rate conversion."""
    output_path: str = ""
    original_fps: float = 0.0
    target_fps: float = 0.0
    preset: str = ""
    method: str = ""
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Frame rate conversion
# ---------------------------------------------------------------------------

def convert_framerate(
    video_path: str,
    target_fps: Optional[float] = None,
    preset: str = "smooth",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Convert video frame rate using optical flow interpolation.

    Uses FFmpeg's ``minterpolate`` filter for motion-compensated
    frame interpolation.  The preset controls the algorithm parameters
    and default target FPS.

    Args:
        video_path:  Input video file.
        target_fps:  Target frame rate.  If *None*, uses the preset
            default.
        preset:  ``"smooth"``, ``"cinematic"``, or ``"sport"``.
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *original_fps*, *target_fps*,
        *preset*, *method*, *duration*.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    if preset not in FRAMERATE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(FRAMERATE_PRESETS.keys())}")

    cfg = FRAMERATE_PRESETS[preset]
    info = get_video_info(video_path)
    original_fps = info.get("fps", 30.0)
    duration = info.get("duration", 0.0)

    if target_fps is None:
        target_fps = float(cfg["default_fps"])
    else:
        target_fps = float(target_fps)

    if target_fps <= 0:
        raise ValueError(f"Invalid target fps: {target_fps}")

    out = output_path_override or output_path(video_path, f"fps_{int(target_fps)}")

    if on_progress:
        on_progress(5, f"Converting {original_fps:.2f} fps -> {target_fps:.2f} fps ({preset})...")

    # Use different strategies based on conversion direction
    method = "minterpolate"

    if preset == "cinematic" and abs(original_fps - 29.97) < 1.0:
        # Telecine removal: fieldmatch + decimate for true inverse telecine
        if on_progress:
            on_progress(10, "Applying inverse telecine (fieldmatch + decimate)...")
        method = "inverse_telecine"
        vf = "fieldmatch,yadif,decimate"
    elif target_fps > original_fps:
        # Upsampling: full minterpolate
        if on_progress:
            on_progress(10, "Interpolating frames with optical flow...")
        vf = (
            f"minterpolate=fps={target_fps}:"
            f"mi_mode={cfg['mi_mode']}:"
            f"mc_mode={cfg['mc_mode']}:"
            f"me_mode={cfg['me_mode']}:"
            f"vsbmc={cfg['vsbmc']}"
        )
    else:
        # Downsampling: fps filter (drop frames cleanly)
        if on_progress:
            on_progress(10, f"Reducing frame rate to {target_fps} fps...")
        method = "fps_filter"
        vf = f"fps={target_fps}"

    if on_progress:
        on_progress(15, f"Processing with {method}...")

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(95, "Verifying output...")

    out_info = get_video_info(out)

    if on_progress:
        on_progress(100, "Frame rate conversion complete!")

    return {
        "output_path": out,
        "original_fps": round(original_fps, 3),
        "target_fps": round(target_fps, 3),
        "actual_fps": round(out_info.get("fps", target_fps), 3),
        "preset": preset,
        "method": method,
        "duration": round(duration, 2),
    }


def list_presets() -> list:
    """Return available frame rate conversion presets."""
    return [
        {"name": name, "description": cfg["description"], "default_fps": cfg["default_fps"]}
        for name, cfg in FRAMERATE_PRESETS.items()
    ]
