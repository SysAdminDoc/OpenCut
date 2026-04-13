"""
OpenCut ND Filter Simulation

Simulates the motion blur effect of a neutral-density (ND) filter by
adding per-frame directional blur based on inter-frame motion.

Uses FFmpeg minterpolate with blend mode for simple ND-style motion
blur.  Configurable blur amount based on target shutter angle vs actual.

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import os
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Standard shutter angle reference (cinema norm = 180 degrees)
# Higher angle = more motion blur; lower = sharper/less blur
_STANDARD_SHUTTER = 180.0


def simulate_nd_filter(
    input_path: str,
    shutter_angle: float = 180.0,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Simulate ND filter motion blur on a video.

    Uses FFmpeg minterpolate with blend mode to create inter-frame
    motion blur similar to what a physical ND filter produces when
    shooting with a wider shutter angle.

    Args:
        input_path: Source video file.
        shutter_angle: Target shutter angle in degrees (1-360).
                       180 = cinema standard, 360 = maximum blur.
        output_path_override: Explicit output; auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with "output_path", "shutter_angle", "blur_strength".
    """
    # Clamp shutter angle
    shutter_angle = max(1.0, min(float(shutter_angle), 360.0))

    if on_progress:
        on_progress(5, "Analyzing source video...")

    info = get_video_info(input_path)
    fps = info.get("fps", 30.0)
    info.get("duration", 0)

    # Calculate blur strength from shutter angle
    # At 360 degrees, each frame exposes for the full frame interval
    # At 180 degrees (standard), half the interval
    # blur_factor: 0.0 (no extra blur) to 1.0 (maximum blend)
    blur_factor = shutter_angle / 360.0

    # minterpolate blend factor: how much to blend adjacent frames
    # mi_mode=blend blends frames together to simulate motion blur
    # The number of interpolated sub-frames controls blur intensity
    # More sub-frames = smoother/heavier blur
    sub_frames = max(2, min(int(blur_factor * 8), 8))

    if on_progress:
        on_progress(10, f"Applying ND filter (shutter {shutter_angle}deg)...")

    # Build filter: minterpolate for motion blur, then fps reset
    vf = (
        f"minterpolate=fps={int(fps * sub_frames)}:mi_mode=blend,"
        f"fps={fps}"
    )

    # Build output path
    if output_path_override:
        out = output_path_override
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        out = os.path.join(
            os.path.dirname(input_path),
            f"{base}_nd{int(shutter_angle)}{ext}",
        )

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, "ND filter simulation complete")

    return {
        "output_path": out,
        "shutter_angle": shutter_angle,
        "blur_strength": round(blur_factor, 3),
        "sub_frames": sub_frames,
        "source_fps": fps,
    }
