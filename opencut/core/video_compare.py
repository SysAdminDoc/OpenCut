"""
OpenCut Video Comparison / Diff Tool

Compare two video sources side-by-side, as an overlay blend, wipe transition,
or absolute pixel difference.  Also supports single-frame comparison for
quick visual QA.

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import os
import tempfile
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

VALID_MODES = ("sidebyside", "overlay", "wipe", "difference")


def compare_videos(
    input_a: str,
    input_b: str,
    mode: str = "sidebyside",
    output_path: Optional[str] = None,
    timestamp: float = 0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Compare two videos using the specified visual mode.

    Args:
        input_a: Path to the first (reference) video.
        input_b: Path to the second (comparison) video.
        mode: Comparison mode -- "sidebyside", "overlay", "wipe", or "difference".
        output_path: Explicit output path. Auto-generated if None.
        timestamp: Start time in seconds (both inputs seek to this point).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with ``output_path`` key.
    """
    if not os.path.isfile(input_a):
        raise FileNotFoundError(f"Input A not found: {input_a}")
    if not os.path.isfile(input_b):
        raise FileNotFoundError(f"Input B not found: {input_b}")
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {VALID_MODES}")

    info_a = get_video_info(input_a)
    get_video_info(input_b)

    if output_path is None:
        output_path = _output_path(input_a, f"compare_{mode}")

    if on_progress:
        on_progress(10, f"Preparing {mode} comparison...")

    # Build filter_complex based on mode.  Both inputs are scaled to the
    # same resolution (use input_a as reference) so filter chains work.
    w, h = info_a["width"], info_a["height"]
    scale_b = f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2[b];"

    if mode == "sidebyside":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]hstack=inputs=2[outv]"
        )
    elif mode == "overlay":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]blend=all_mode=average:all_opacity=0.5[outv]"
        )
    elif mode == "wipe":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]xfade=transition=wipeleft:duration=0.001:offset=0[outv]"
        )
    elif mode == "difference":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]blend=all_mode=difference[outv]"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cmd = (
        FFmpegCmd()
        .input(input_a, ss=str(timestamp))
        .input(input_b, ss=str(timestamp))
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="192k")
        .option("shortest")
        .faststart()
        .output(output_path)
        .build()
    )

    if on_progress:
        on_progress(30, f"Rendering {mode} comparison...")

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Comparison complete")

    return {"output_path": output_path}


def compare_frames(
    input_a: str,
    input_b: str,
    timestamp: float = 0,
    mode: str = "sidebyside",
) -> dict:
    """
    Compare a single frame from two videos.

    Extracts a frame at *timestamp* from each input and composites them
    using the given mode.  Returns the path to a PNG image.

    Args:
        input_a: Path to the first video.
        input_b: Path to the second video.
        timestamp: Time in seconds to extract the frame.
        mode: Comparison mode (same as ``compare_videos``).

    Returns:
        dict with ``output_path`` pointing to the comparison image.
    """
    if not os.path.isfile(input_a):
        raise FileNotFoundError(f"Input A not found: {input_a}")
    if not os.path.isfile(input_b):
        raise FileNotFoundError(f"Input B not found: {input_b}")
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {VALID_MODES}")

    info_a = get_video_info(input_a)
    w, h = info_a["width"], info_a["height"]

    out_dir = tempfile.gettempdir()
    out_name = f"opencut_compare_frame_{os.getpid()}.png"
    out_path = os.path.join(out_dir, out_name)

    scale_b = f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2[b];"

    if mode == "sidebyside":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]hstack=inputs=2[outv]"
        )
    elif mode == "overlay":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]blend=all_mode=average:all_opacity=0.5[outv]"
        )
    elif mode == "wipe":
        # For a single frame, split the image left/right
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]xfade=transition=wipeleft:duration=0.001:offset=0[outv]"
        )
    elif mode == "difference":
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]blend=all_mode=difference[outv]"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cmd = (
        FFmpegCmd()
        .input(input_a, ss=str(timestamp))
        .input(input_b, ss=str(timestamp))
        .filter_complex(fc, maps=["[outv]"])
        .frames(1)
        .option("update", "1")
        .output(out_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=60)

    return {"output_path": out_path}
