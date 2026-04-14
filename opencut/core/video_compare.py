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
EXPORT_MODES = ("vertical_wipe", "horizontal_wipe", "side_by_side", "alternating")


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


# ---------------------------------------------------------------------------
# Before/After Comparison Video Export (57.3)
# ---------------------------------------------------------------------------
def export_comparison_video(
    original: str,
    processed: str,
    mode: str = "vertical_wipe",
    out_path: Optional[str] = None,
    label_original: str = "Original",
    label_processed: str = "Processed",
    wipe_speed: float = 1.0,
    alternating_interval: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Export a before/after comparison video with animated wipe and labels.

    Args:
        original: Path to the original (before) video.
        processed: Path to the processed (after) video.
        mode: Export mode -- "vertical_wipe", "horizontal_wipe",
              "side_by_side", or "alternating".
        out_path: Explicit output path. Auto-generated if None.
        label_original: Text label for the original side.
        label_processed: Text label for the processed side.
        wipe_speed: Wipe animation speed multiplier (affects cycle rate).
        alternating_interval: Seconds between alternating cuts.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with ``output_path`` and ``mode``.
    """
    if not os.path.isfile(original):
        raise FileNotFoundError(f"Original not found: {original}")
    if not os.path.isfile(processed):
        raise FileNotFoundError(f"Processed not found: {processed}")
    if mode not in EXPORT_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {EXPORT_MODES}")

    info = get_video_info(original)
    w, h = info["width"], info["height"]
    duration = info.get("duration", 0)
    fps = info.get("fps", 30)

    if out_path is None:
        out_path = _output_path(original, f"comparison_{mode}")

    if on_progress:
        on_progress(10, f"Building {mode} comparison export...")

    # Label font size
    font_size = max(16, min(w, h) // 25)

    # Escape labels for FFmpeg drawtext
    def _esc(t):
        return t.replace("'", "\\'").replace(":", "\\:")

    label_orig_esc = _esc(label_original[:50])
    label_proc_esc = _esc(label_processed[:50])

    scale_b = (
        f"[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2[b];"
    )

    if mode == "side_by_side":
        hw = w // 2
        hw = hw - (hw % 2)
        fc = (
            f"[0:v]scale={hw}:{h}[a];"
            f"[1:v]scale={hw}:{h}[b];"
            f"[a][b]hstack=inputs=2[stacked];"
            f"[stacked]drawtext=text='{label_orig_esc}':"
            f"fontsize={font_size}:fontcolor=white:x=10:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2,"
            f"drawtext=text='{label_proc_esc}':"
            f"fontsize={font_size}:fontcolor=white:x={hw + 10}:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2[outv]"
        )

    elif mode == "vertical_wipe":
        # Animated vertical wipe: crop position oscillates left<->right
        cycle = max(1.0, 4.0 / wipe_speed)
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]overlay=x=0:y=0[base];"
            f"[base]drawbox=x=0:y=0:"
            f"w='(W/2)+(W/2)*sin(2*PI*t/{cycle:.2f})':"
            f"h=H:color=black@0:t=fill,"
            f"drawtext=text='{label_orig_esc}':"
            f"fontsize={font_size}:fontcolor=white:x=10:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2,"
            f"drawtext=text='{label_proc_esc}':"
            f"fontsize={font_size}:fontcolor=white:x={w - font_size * 8}:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2[outv]"
        )

    elif mode == "horizontal_wipe":
        cycle = max(1.0, 4.0 / wipe_speed)
        fc = (
            f"[0:v]scale={w}:{h}[a];"
            f"{scale_b}"
            f"[a][b]overlay=x=0:y=0[base];"
            f"[base]drawbox=x=0:y=0:w=W:"
            f"h='(H/2)+(H/2)*sin(2*PI*t/{cycle:.2f})':"
            f"color=black@0:t=fill,"
            f"drawtext=text='{label_orig_esc}':"
            f"fontsize={font_size}:fontcolor=white:x=10:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2,"
            f"drawtext=text='{label_proc_esc}':"
            f"fontsize={font_size}:fontcolor=white:x=10:y={h - font_size - 10}:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2[outv]"
        )

    elif mode == "alternating":
        # Alternate between original and processed every N seconds
        interval = max(0.5, alternating_interval)
        fc = (
            f"[0:v]scale={w}:{h},"
            f"drawtext=text='{label_orig_esc}':"
            f"fontsize={font_size}:fontcolor=white:x=10:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2[a];"
            f"[1:v]scale={w}:{h},"
            f"drawtext=text='{label_proc_esc}':"
            f"fontsize={font_size}:fontcolor=white:x=10:y=10:"
            f"shadowcolor=black@0.7:shadowx=2:shadowy=2[b];"
            f"[a][b]xfade=transition=fade:duration=0.3:offset={interval}[outv]"
        )

    else:
        raise ValueError(f"Unsupported export mode: {mode}")

    if on_progress:
        on_progress(30, f"Rendering {mode} comparison...")

    cmd = (
        FFmpegCmd()
        .input(original)
        .input(processed)
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="192k")
        .option("shortest")
        .faststart()
        .output(out_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Comparison export complete")

    return {"output_path": out_path, "mode": mode}
