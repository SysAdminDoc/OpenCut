"""
OpenCut Hyperlapse Stabilization Module

Create smooth hyperlapses from handheld or action-camera footage:
1. Sample every Nth frame for the target speed factor.
2. Apply aggressive multi-pass stabilization (vidstab) with high smoothing.
3. Optionally fill edge borders caused by stabilization warping.

Uses FFmpeg vidstabdetect/vidstabtransform for stabilization.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class HyperlapseResult:
    """Result from hyperlapse creation."""
    output_path: str = ""
    speed_factor: float = 1.0
    original_duration: float = 0.0
    output_duration: float = 0.0
    frames_sampled: int = 0
    stabilization_smoothing: int = 0
    edge_fill: str = "none"


@dataclass
class StabilizeResult:
    """Result from stabilization pass."""
    output_path: str = ""
    smoothing: int = 30
    passes: int = 1
    edge_fill: str = "none"


# ---------------------------------------------------------------------------
# Hyperlapse Creation
# ---------------------------------------------------------------------------
def create_hyperlapse(
    video_path: str,
    speed_factor: float = 10.0,
    smoothing: int = 45,
    edge_fill: str = "mirror",
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    stabilize: bool = True,
    passes: int = 2,
    on_progress: Optional[Callable] = None,
) -> HyperlapseResult:
    """
    Create a hyperlapse from video footage.

    Pipeline:
      1. Speed up the video by sampling every Nth frame (using setpts filter).
      2. Optionally apply multi-pass vidstab stabilization.
      3. Fill edges caused by stabilization transforms.

    Args:
        video_path: Input video file.
        speed_factor: Speed multiplier (e.g. 10.0 = 10x faster).
        smoothing: Vidstab smoothing value (higher = smoother, more crop).
        edge_fill: Edge fill method: "mirror", "black", "crop", or "none".
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        stabilize: Whether to apply stabilization after speed-up.
        passes: Number of stabilization passes (1 or 2).
        on_progress: Progress callback(pct, msg).

    Returns:
        HyperlapseResult with output path and processing metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    speed_factor = max(1.0, min(100.0, float(speed_factor)))
    smoothing = max(1, min(200, int(smoothing)))
    passes = max(1, min(3, int(passes)))
    edge_fill = edge_fill if edge_fill in ("mirror", "black", "crop", "none") else "mirror"

    info = get_video_info(video_path)
    original_duration = info["duration"]
    expected_duration = original_duration / speed_factor

    if on_progress:
        on_progress(5, f"Creating {speed_factor}x hyperlapse...")

    # Step 1: Speed up via setpts
    tmp_sped = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_sped_path = tmp_sped.name
    tmp_sped.close()

    # PTS factor: 1/speed_factor (e.g. 10x = PTS*0.1)
    pts_factor = 1.0 / speed_factor
    speed_vf = f"setpts={pts_factor:.6f}*PTS"

    if on_progress:
        on_progress(10, f"Speeding up video ({speed_factor}x)...")

    cmd_speed = (FFmpegCmd()
                 .input(video_path)
                 .video_filter(speed_vf)
                 .video_codec("libx264", crf=16, preset="medium")
                 .option("an")  # Drop audio for hyperlapse
                 .faststart()
                 .output(tmp_sped_path)
                 .build())
    run_ffmpeg(cmd_speed, timeout=3600)

    if not stabilize:
        # No stabilization requested -- just rename
        if output_path_str is None:
            directory = output_dir or os.path.dirname(video_path)
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_path_str = os.path.join(directory, f"{base}_hyperlapse.mp4")
        os.replace(tmp_sped_path, output_path_str)

        if on_progress:
            on_progress(100, "Hyperlapse created (no stabilization)")

        return HyperlapseResult(
            output_path=output_path_str,
            speed_factor=speed_factor,
            original_duration=original_duration,
            output_duration=expected_duration,
            frames_sampled=int(info.get("duration", 0) * info.get("fps", 30) / speed_factor),
            stabilization_smoothing=0,
            edge_fill="none",
        )

    if on_progress:
        on_progress(30, f"Stabilizing (pass 1/{passes})...")

    # Step 2: Multi-pass stabilization
    try:
        stab_result = stabilize_hyperlapse(
            video_path=tmp_sped_path,
            smoothing=smoothing,
            edge_fill=edge_fill,
            passes=passes,
            output_path_str=output_path_str,
            output_dir=output_dir or os.path.dirname(video_path),
            _base_name=os.path.splitext(os.path.basename(video_path))[0],
            on_progress=lambda pct, msg: on_progress(30 + int(pct * 0.65), msg) if on_progress else None,
        )
    finally:
        try:
            os.unlink(tmp_sped_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Hyperlapse stabilized!")

    sped_info = get_video_info(stab_result.output_path)

    return HyperlapseResult(
        output_path=stab_result.output_path,
        speed_factor=speed_factor,
        original_duration=original_duration,
        output_duration=sped_info.get("duration", expected_duration),
        frames_sampled=int(original_duration * info.get("fps", 30) / speed_factor),
        stabilization_smoothing=smoothing,
        edge_fill=edge_fill,
    )


# ---------------------------------------------------------------------------
# Stabilization
# ---------------------------------------------------------------------------
def stabilize_hyperlapse(
    video_path: str,
    smoothing: int = 45,
    edge_fill: str = "mirror",
    passes: int = 2,
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    _base_name: str = "",
    on_progress: Optional[Callable] = None,
) -> StabilizeResult:
    """
    Apply aggressive multi-pass stabilization to a video.

    Uses FFmpeg's vidstabdetect + vidstabtransform filters with high
    smoothing values appropriate for hyperlapse footage.

    Args:
        video_path: Input video (already sped up, or raw footage).
        smoothing: Vidstab smoothing parameter (higher = smoother).
        edge_fill: Edge handling: "mirror", "black", "crop", or "none".
        passes: Number of stabilization passes.
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        on_progress: Progress callback(pct, msg).

    Returns:
        StabilizeResult.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    smoothing = max(1, min(200, int(smoothing)))
    passes = max(1, min(3, int(passes)))

    # Map edge_fill to vidstab border parameter
    border_map = {
        "mirror": "mirror",
        "black": "black",
        "crop": "black",  # We'll crop after
        "none": "keep",
    }
    border_mode = border_map.get(edge_fill, "mirror")

    current_input = video_path

    for pass_num in range(1, passes + 1):
        if on_progress:
            pct_base = int((pass_num - 1) / passes * 100)
            on_progress(pct_base + 5, f"Stabilization pass {pass_num}/{passes}: analyzing motion...")

        # Detect transforms
        transforms_file = os.path.join(
            tempfile.gettempdir(), f"vidstab_pass{pass_num}_{os.getpid()}.trf"
        )

        # Pass 1: Detect
        detect_vf = (
            f"vidstabdetect=stepsize=6:shakiness=10:accuracy=15:"
            f"result='{transforms_file}'"
        )
        cmd_detect = (FFmpegCmd()
                      .input(current_input)
                      .video_filter(detect_vf)
                      .format("null")
                      .output("-")
                      .build())
        # Note: output is /dev/null equivalent -- we just need the transforms file
        run_ffmpeg(cmd_detect, timeout=3600)

        if on_progress:
            on_progress(pct_base + 50, f"Pass {pass_num}/{passes}: applying transforms...")

        # Pass 2: Apply transforms
        is_last_pass = (pass_num == passes)
        if is_last_pass and output_path_str:
            pass_output = output_path_str
        elif is_last_pass:
            directory = output_dir or os.path.dirname(video_path)
            base = _base_name or os.path.splitext(os.path.basename(video_path))[0]
            pass_output = os.path.join(directory, f"{base}_hyperlapse.mp4")
        else:
            tmp_pass = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            pass_output = tmp_pass.name
            tmp_pass.close()

        transform_vf = (
            f"vidstabtransform=input='{transforms_file}':"
            f"smoothing={smoothing}:interpol=bicubic:"
            f"optzoom=2:zoomspeed=0.25:border={border_mode}"
        )

        # If cropping, add a crop to remove unstable edges
        if edge_fill == "crop" and is_last_pass:
            info = get_video_info(current_input)
            crop_margin = max(20, int(min(info["width"], info["height"]) * 0.05))
            crop_w = info["width"] - 2 * crop_margin
            crop_h = info["height"] - 2 * crop_margin
            transform_vf += f",crop={crop_w}:{crop_h}:{crop_margin}:{crop_margin}"

        cmd_transform = (FFmpegCmd()
                         .input(current_input)
                         .video_filter(transform_vf)
                         .video_codec("libx264", crf=16, preset="medium")
                         .faststart()
                         .output(pass_output)
                         .build())
        run_ffmpeg(cmd_transform, timeout=3600)

        # Cleanup
        try:
            os.unlink(transforms_file)
        except OSError:
            pass

        # For multi-pass: the output of this pass becomes input for next
        if not is_last_pass:
            if current_input != video_path:
                try:
                    os.unlink(current_input)
                except OSError:
                    pass
            current_input = pass_output

    if on_progress:
        on_progress(100, "Stabilization complete")

    return StabilizeResult(
        output_path=pass_output,
        smoothing=smoothing,
        passes=passes,
        edge_fill=edge_fill,
    )
