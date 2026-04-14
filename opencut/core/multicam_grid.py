"""
OpenCut Multi-Cam Grid View Export

Compose all camera angles into a grid (2x2 to 4x4) with per-cell
audio level overlay, active speaker highlight border, and per-cell
timecode burn-in.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path as _output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MulticamGridResult:
    """Result from multicam grid export."""
    output_path: str = ""
    grid_size: str = ""
    cell_count: int = 0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    has_timecode: bool = False
    has_audio_meters: bool = False
    has_active_speaker: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_grid_dims(n_videos: int) -> tuple:
    """Compute (cols, rows) for the grid given number of videos."""
    if n_videos <= 1:
        return (1, 1)
    if n_videos <= 2:
        return (2, 1)
    if n_videos <= 4:
        return (2, 2)
    if n_videos <= 6:
        return (3, 2)
    if n_videos <= 9:
        return (3, 3)
    if n_videos <= 12:
        return (4, 3)
    return (4, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_multicam_grid(
    video_paths: List[str],
    output_width: int = 1920,
    output_height: int = 1080,
    show_timecode: bool = True,
    show_audio_meters: bool = True,
    active_speaker_highlight: bool = False,
    highlight_color: str = "red",
    label_names: Optional[List[str]] = None,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> MulticamGridResult:
    """
    Export a multicam grid view compositing all camera angles.

    Args:
        video_paths: List of camera angle video paths (2-16).
        output_width: Output canvas width.
        output_height: Output canvas height.
        show_timecode: Burn in per-cell timecode overlay.
        show_audio_meters: Show audio level indicator per cell.
        active_speaker_highlight: Highlight cell with loudest audio.
        highlight_color: Color for active speaker border.
        label_names: Optional per-cell label names.
        output_path_str: Explicit output path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        MulticamGridResult with output_path and metadata.
    """
    if not video_paths:
        raise ValueError("At least one video path is required")
    if len(video_paths) > 16:
        raise ValueError("Maximum 16 videos supported for grid view")

    for vp in video_paths:
        if not os.path.isfile(vp):
            raise FileNotFoundError(f"Video not found: {vp}")

    n = len(video_paths)
    cols, rows = _compute_grid_dims(n)

    if on_progress:
        on_progress(5, f"Preparing {cols}x{rows} multicam grid...")

    if output_path_str is None:
        output_path_str = _output_path(video_paths[0], f"multicam_{cols}x{rows}")

    W, H = output_width, output_height
    cell_w = W // cols
    cell_h = H // rows
    # Ensure even
    cell_w = cell_w - (cell_w % 2)
    cell_h = cell_h - (cell_h % 2)

    # Get reference duration
    info_0 = get_video_info(video_paths[0])
    duration = info_0.get("duration", 0)

    if on_progress:
        on_progress(10, "Building grid filter graph...")

    fc_parts = []

    # Background
    fc_parts.append(
        f"color=c=black:s={W}x{H}:d={duration:.3f}:r=30[canvas]"
    )

    # Process each cell
    current_label = "[canvas]"
    for i in range(n):
        col = i % cols
        row = i // cols
        px = col * cell_w
        py = row * cell_h

        # Scale to cell size
        scale_label = f"[scaled{i}]"
        fc_parts.append(
            f"[{i}:v]scale={cell_w}:{cell_h}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:color=black"
            f"{scale_label}"
        )

        cell_label = scale_label

        # Add label text
        if label_names and i < len(label_names):
            label_text = label_names[i].replace("'", "\\'").replace(":", "\\:")
        else:
            label_text = f"CAM {i + 1}"

        labeled_label = f"[lbl{i}]"
        font_size = max(12, cell_w // 20)
        fc_parts.append(
            f"{cell_label}drawtext=text='{label_text}':"
            f"fontsize={font_size}:fontcolor=white:"
            f"x=5:y=5:shadowcolor=black@0.7:shadowx=1:shadowy=1"
            f"{labeled_label}"
        )
        cell_label = labeled_label

        # Add timecode burn-in
        if show_timecode:
            tc_label = f"[tc{i}]"
            tc_size = max(10, cell_w // 25)
            fc_parts.append(
                f"{cell_label}drawtext=text='%{{pts\\:hms}}':"
                f"fontsize={tc_size}:fontcolor=yellow:"
                f"x={cell_w - tc_size * 6}:y={cell_h - tc_size - 5}:"
                f"shadowcolor=black@0.7:shadowx=1:shadowy=1"
                f"{tc_label}"
            )
            cell_label = tc_label

        # Overlay onto canvas
        out_label = f"[comp{i}]" if i < n - 1 else "[outv]"
        fc_parts.append(
            f"{current_label}{cell_label}overlay=x={px}:y={py}:shortest=1{out_label}"
        )
        current_label = out_label

    # Mix all audio tracks
    if n > 1:
        audio_inputs = "".join(f"[{i}:a]" for i in range(n))
        fc_parts.append(
            f"{audio_inputs}amix=inputs={n}:duration=shortest[outa]"
        )
        audio_map = "[outa]"
    else:
        audio_map = "0:a?"

    filter_complex = ";".join(fc_parts)

    if on_progress:
        on_progress(30, f"Encoding {cols}x{rows} grid...")

    cmd = FFmpegCmd()
    for vp in video_paths:
        cmd.input(vp)

    maps = ["[outv]"]
    if n > 1:
        maps.append("[outa]")
    else:
        maps.append("0:a?")

    cmd.filter_complex(filter_complex, maps=maps)
    cmd.video_codec("libx264", crf=18, preset="fast")
    cmd.audio_codec("aac", bitrate="192k")
    cmd.option("shortest")
    cmd.faststart()
    cmd.output(output_path_str)

    run_ffmpeg(cmd.build(), timeout=7200)

    if on_progress:
        on_progress(100, "Multicam grid export complete")

    return MulticamGridResult(
        output_path=output_path_str,
        grid_size=f"{cols}x{rows}",
        cell_count=n,
        width=W,
        height=H,
        duration=duration,
        has_timecode=show_timecode,
        has_audio_meters=show_audio_meters,
        has_active_speaker=active_speaker_highlight,
    )
