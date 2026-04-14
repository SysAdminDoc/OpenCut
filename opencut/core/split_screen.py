"""
OpenCut Split-Screen Layout Templates

Define composite layouts as JSON (array of cells with x/y/w/h percentages).
Preset layouts: side-by-side, 2x2 grid, 3x3, PiP variants, diagonal, L-shaped.
Composite via FFmpeg overlay chain with optional border/gap between cells.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
class LayoutCell:
    """A single cell within a split-screen layout (percentages 0-100)."""
    x: float = 0.0
    y: float = 0.0
    w: float = 100.0
    h: float = 100.0
    label: str = ""


@dataclass
class SplitScreenLayout:
    """Named layout template with cell definitions."""
    name: str = ""
    cells: List[LayoutCell] = field(default_factory=list)
    description: str = ""


@dataclass
class SplitScreenResult:
    """Result from a split-screen composite operation."""
    output_path: str = ""
    layout_name: str = ""
    cell_count: int = 0
    width: int = 0
    height: int = 0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Preset layouts
# ---------------------------------------------------------------------------
_PRESET_LAYOUTS: Dict[str, SplitScreenLayout] = {
    "side_by_side": SplitScreenLayout(
        name="side_by_side",
        description="Two videos side by side (50/50 split)",
        cells=[
            LayoutCell(x=0, y=0, w=50, h=100, label="Left"),
            LayoutCell(x=50, y=0, w=50, h=100, label="Right"),
        ],
    ),
    "2x2_grid": SplitScreenLayout(
        name="2x2_grid",
        description="Four videos in a 2x2 grid",
        cells=[
            LayoutCell(x=0, y=0, w=50, h=50, label="Top-Left"),
            LayoutCell(x=50, y=0, w=50, h=50, label="Top-Right"),
            LayoutCell(x=0, y=50, w=50, h=50, label="Bottom-Left"),
            LayoutCell(x=50, y=50, w=50, h=50, label="Bottom-Right"),
        ],
    ),
    "3x3_grid": SplitScreenLayout(
        name="3x3_grid",
        description="Nine videos in a 3x3 grid",
        cells=[
            LayoutCell(x=i * 33.33, y=j * 33.33, w=33.33, h=33.33,
                       label=f"Cell-{j * 3 + i + 1}")
            for j in range(3) for i in range(3)
        ],
    ),
    "pip_top_right": SplitScreenLayout(
        name="pip_top_right",
        description="Picture-in-picture with small video in top-right corner",
        cells=[
            LayoutCell(x=0, y=0, w=100, h=100, label="Main"),
            LayoutCell(x=70, y=5, w=25, h=25, label="PiP"),
        ],
    ),
    "pip_bottom_right": SplitScreenLayout(
        name="pip_bottom_right",
        description="Picture-in-picture with small video in bottom-right corner",
        cells=[
            LayoutCell(x=0, y=0, w=100, h=100, label="Main"),
            LayoutCell(x=70, y=70, w=25, h=25, label="PiP"),
        ],
    ),
    "pip_bottom_left": SplitScreenLayout(
        name="pip_bottom_left",
        description="Picture-in-picture with small video in bottom-left corner",
        cells=[
            LayoutCell(x=0, y=0, w=100, h=100, label="Main"),
            LayoutCell(x=5, y=70, w=25, h=25, label="PiP"),
        ],
    ),
    "pip_top_left": SplitScreenLayout(
        name="pip_top_left",
        description="Picture-in-picture with small video in top-left corner",
        cells=[
            LayoutCell(x=0, y=0, w=100, h=100, label="Main"),
            LayoutCell(x=5, y=5, w=25, h=25, label="PiP"),
        ],
    ),
    "diagonal": SplitScreenLayout(
        name="diagonal",
        description="Two videos placed diagonally",
        cells=[
            LayoutCell(x=0, y=0, w=55, h=55, label="Top-Left"),
            LayoutCell(x=45, y=45, w=55, h=55, label="Bottom-Right"),
        ],
    ),
    "l_shaped": SplitScreenLayout(
        name="l_shaped",
        description="L-shaped layout: one large + two small stacked",
        cells=[
            LayoutCell(x=0, y=0, w=66, h=100, label="Main"),
            LayoutCell(x=66, y=0, w=34, h=50, label="Top-Right"),
            LayoutCell(x=66, y=50, w=34, h=50, label="Bottom-Right"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_preset_layouts() -> Dict[str, dict]:
    """Return all preset layout definitions as serializable dicts."""
    result = {}
    for name, layout in _PRESET_LAYOUTS.items():
        result[name] = {
            "name": layout.name,
            "description": layout.description,
            "cell_count": len(layout.cells),
            "cells": [
                {"x": c.x, "y": c.y, "w": c.w, "h": c.h, "label": c.label}
                for c in layout.cells
            ],
        }
    return result


def parse_layout(layout_data: dict) -> SplitScreenLayout:
    """Parse a layout dict (from JSON) into a SplitScreenLayout."""
    cells = []
    for cell_data in layout_data.get("cells", []):
        cells.append(LayoutCell(
            x=float(cell_data.get("x", 0)),
            y=float(cell_data.get("y", 0)),
            w=float(cell_data.get("w", 100)),
            h=float(cell_data.get("h", 100)),
            label=str(cell_data.get("label", "")),
        ))
    return SplitScreenLayout(
        name=str(layout_data.get("name", "custom")),
        description=str(layout_data.get("description", "")),
        cells=cells,
    )


def create_split_screen(
    video_paths: List[str],
    layout_name: str = "side_by_side",
    custom_layout: Optional[dict] = None,
    output_width: int = 1920,
    output_height: int = 1080,
    border_width: int = 0,
    border_color: str = "black",
    gap: int = 0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> SplitScreenResult:
    """
    Create a split-screen composite from multiple videos.

    Args:
        video_paths: List of input video file paths.
        layout_name: Name of a preset layout, or 'custom' with custom_layout.
        custom_layout: Custom layout dict with cells array.
        output_width: Canvas width in pixels.
        output_height: Canvas height in pixels.
        border_width: Border thickness around each cell in pixels.
        border_color: Border color name or hex.
        gap: Gap between cells in pixels.
        output_path_str: Explicit output path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        SplitScreenResult with output_path and metadata.
    """
    if not video_paths:
        raise ValueError("At least one video path is required")

    for vp in video_paths:
        if not os.path.isfile(vp):
            raise FileNotFoundError(f"Video not found: {vp}")

    # Resolve layout
    if custom_layout:
        layout = parse_layout(custom_layout)
    elif layout_name in _PRESET_LAYOUTS:
        layout = _PRESET_LAYOUTS[layout_name]
    else:
        raise ValueError(
            f"Unknown layout '{layout_name}'. "
            f"Available: {list(_PRESET_LAYOUTS.keys())}"
        )

    cells = layout.cells
    if not cells:
        raise ValueError("Layout has no cells defined")

    # Pad video_paths to match cell count (loop last video)
    while len(video_paths) < len(cells):
        video_paths.append(video_paths[-1])

    if on_progress:
        on_progress(5, f"Preparing {layout.name} split-screen...")

    # Output path
    if output_path_str is None:
        output_path_str = _output_path(video_paths[0], f"split_{layout.name}")

    # Get reference duration from first video
    info_0 = get_video_info(video_paths[0])
    duration = info_0.get("duration", 0)

    if on_progress:
        on_progress(10, "Building filter graph...")

    # Build FFmpeg filter_complex
    W, H = output_width, output_height
    fc_parts = []

    # Create background canvas
    fc_parts.append(
        f"color=c={border_color}:s={W}x{H}:d={duration:.3f}:r=30[canvas]"
    )

    current_label = "[canvas]"

    for i, cell in enumerate(cells):
        if i >= len(video_paths):
            break

        # Calculate pixel positions with gap
        cx = int(cell.x / 100.0 * W) + gap
        cy = int(cell.y / 100.0 * H) + gap
        cw = int(cell.w / 100.0 * W) - 2 * gap
        ch = int(cell.h / 100.0 * H) - 2 * gap

        # Apply border
        if border_width > 0:
            cx += border_width
            cy += border_width
            cw -= 2 * border_width
            ch -= 2 * border_width

        cw = max(cw, 16)
        ch = max(ch, 16)
        # Ensure even dimensions
        cw = cw - (cw % 2)
        ch = ch - (ch % 2)

        # Scale input to cell size
        fc_parts.append(
            f"[{i}:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black[cell{i}]"
        )

        # Overlay cell on canvas
        out_label = f"[comp{i}]" if i < len(cells) - 1 else "[outv]"
        fc_parts.append(
            f"{current_label}[cell{i}]overlay=x={cx}:y={cy}:shortest=1{out_label}"
        )
        current_label = out_label

    filter_complex = ";".join(fc_parts)

    if on_progress:
        on_progress(20, "Encoding split-screen composite...")

    # Build command
    cmd = FFmpegCmd()
    for vp in video_paths[:len(cells)]:
        cmd.input(vp)
    cmd.filter_complex(filter_complex, maps=["[outv]", "0:a?"])
    cmd.video_codec("libx264", crf=18, preset="fast")
    cmd.audio_codec("aac", bitrate="192k")
    cmd.option("shortest")
    cmd.faststart()
    cmd.output(output_path_str)

    run_ffmpeg(cmd.build(), timeout=3600)

    if on_progress:
        on_progress(100, "Split-screen complete")

    return SplitScreenResult(
        output_path=output_path_str,
        layout_name=layout.name,
        cell_count=min(len(cells), len(video_paths)),
        width=W,
        height=H,
        duration=duration,
    )
