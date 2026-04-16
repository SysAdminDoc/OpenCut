"""
OpenCut Composition Guide Overlay

Draw composition guide overlays on video frames via Pillow:
rule-of-thirds grid, golden ratio spiral, diagonal lines,
center cross, and broadcast safe areas (title 80%, action 90%).

Display-only overlays -- not burned into output video.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
)

logger = logging.getLogger("opencut")

# Golden ratio constant
PHI = 1.6180339887

# Guide types
GUIDE_TYPES = [
    "rule_of_thirds",
    "golden_ratio",
    "diagonal",
    "center_cross",
    "safe_areas",
    "fibonacci_spiral",
    "triangle",
    "grid_4x4",
]

# Default colors (RGBA)
DEFAULT_COLORS = {
    "rule_of_thirds": (255, 255, 255, 128),
    "golden_ratio": (255, 215, 0, 128),
    "diagonal": (0, 200, 255, 100),
    "center_cross": (255, 0, 0, 120),
    "safe_areas": (255, 255, 0, 100),
    "fibonacci_spiral": (0, 255, 128, 110),
    "triangle": (200, 100, 255, 100),
    "grid_4x4": (180, 180, 180, 80),
    "title_safe": (255, 100, 100, 90),
    "action_safe": (100, 255, 100, 90),
}


@dataclass
class GuideOverlayResult:
    """Result of composition guide overlay generation."""
    output_path: str = ""
    width: int = 0
    height: int = 0
    guides_applied: List[str] = field(default_factory=list)
    timestamp: float = 0.0


def _extract_frame(video_path: str, timestamp: float, output_frame: str) -> bool:
    """Extract a single frame from video at the given timestamp."""
    import subprocess

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y", output_frame,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.isfile(output_frame)
    except Exception as exc:
        logger.debug("Frame extraction failed: %s", exc)
        return False


def _draw_rule_of_thirds(draw, width: int, height: int, color: Tuple):
    """Draw rule-of-thirds grid (2 vertical + 2 horizontal lines)."""
    third_w = width / 3
    third_h = height / 3
    line_width = max(1, min(width, height) // 500)

    for i in range(1, 3):
        x = int(third_w * i)
        draw.line([(x, 0), (x, height)], fill=color, width=line_width)
    for i in range(1, 3):
        y = int(third_h * i)
        draw.line([(0, y), (width, y)], fill=color, width=line_width)

    # Power points (intersections)
    dot_r = max(3, min(width, height) // 200)
    for ix in range(1, 3):
        for iy in range(1, 3):
            cx = int(third_w * ix)
            cy = int(third_h * iy)
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                fill=color,
            )


def _draw_golden_ratio(draw, width: int, height: int, color: Tuple):
    """Draw golden ratio lines (phi-based divisions)."""
    line_width = max(1, min(width, height) // 500)

    gw = width / PHI
    gh = height / PHI

    # Vertical lines at golden ratio positions
    draw.line([(int(gw), 0), (int(gw), height)], fill=color, width=line_width)
    draw.line([(int(width - gw), 0), (int(width - gw), height)],
              fill=color, width=line_width)

    # Horizontal lines at golden ratio positions
    draw.line([(0, int(gh)), (width, int(gh))], fill=color, width=line_width)
    draw.line([(0, int(height - gh)), (width, int(height - gh))],
              fill=color, width=line_width)


def _draw_diagonal(draw, width: int, height: int, color: Tuple):
    """Draw diagonal composition lines."""
    line_width = max(1, min(width, height) // 600)

    # Main diagonals
    draw.line([(0, 0), (width, height)], fill=color, width=line_width)
    draw.line([(width, 0), (0, height)], fill=color, width=line_width)

    # Bisecting diagonals (corner to midpoint of opposite side)
    draw.line([(0, 0), (width, height // 2)], fill=color, width=line_width)
    draw.line([(0, 0), (width // 2, height)], fill=color, width=line_width)
    draw.line([(width, 0), (0, height // 2)], fill=color, width=line_width)
    draw.line([(width, 0), (width // 2, height)], fill=color, width=line_width)


def _draw_center_cross(draw, width: int, height: int, color: Tuple):
    """Draw center crosshair."""
    cx = width // 2
    cy = height // 2
    line_width = max(1, min(width, height) // 400)
    arm_len = min(width, height) // 15

    # Cross
    draw.line([(cx - arm_len, cy), (cx + arm_len, cy)],
              fill=color, width=line_width)
    draw.line([(cx, cy - arm_len), (cx, cy + arm_len)],
              fill=color, width=line_width)

    # Circle around center
    r = arm_len // 2
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=color, width=line_width,
    )


def _draw_safe_areas(draw, width: int, height: int,
                     title_color: Tuple, action_color: Tuple):
    """Draw broadcast safe areas: title safe (80%) and action safe (90%)."""
    line_width = max(1, min(width, height) // 400)

    # Title safe area (80% of frame)
    title_margin_x = int(width * 0.10)
    title_margin_y = int(height * 0.10)
    draw.rectangle(
        [title_margin_x, title_margin_y,
         width - title_margin_x, height - title_margin_y],
        outline=title_color, width=line_width,
    )

    # Action safe area (90% of frame)
    action_margin_x = int(width * 0.05)
    action_margin_y = int(height * 0.05)
    draw.rectangle(
        [action_margin_x, action_margin_y,
         width - action_margin_x, height - action_margin_y],
        outline=action_color, width=line_width,
    )


def _draw_fibonacci_spiral(draw, width: int, height: int, color: Tuple):
    """Draw an approximation of the Fibonacci (golden) spiral."""
    line_width = max(1, min(width, height) // 500)

    # Draw arcs in successive golden-ratio rectangles
    x, y = 0, 0
    w, h = width, height

    for i in range(7):
        if w < 4 or h < 4:
            break
        # Draw arc in current rectangle quadrant
        if i % 4 == 0:
            draw.arc([x, y, x + 2 * w, y + 2 * h], 180, 270,
                     fill=color, width=line_width)
            split = int(w / PHI)
            x += split
            w -= split
        elif i % 4 == 1:
            draw.arc([x - w, y, x + w, y + 2 * h], 270, 360,
                     fill=color, width=line_width)
            split = int(h / PHI)
            h -= split
        elif i % 4 == 2:
            draw.arc([x - 2 * w, y - h, x, y + h], 0, 90,
                     fill=color, width=line_width)
            split = int(w / PHI)
            w -= split
        else:
            draw.arc([x, y - h, x + 2 * w, y + h], 90, 180,
                     fill=color, width=line_width)
            split = int(h / PHI)
            y += split
            h -= split


def _draw_triangle(draw, width: int, height: int, color: Tuple):
    """Draw triangular composition guides (diagonals with bisectors)."""
    line_width = max(1, min(width, height) // 500)

    # Main diagonal
    draw.line([(0, height), (width, 0)], fill=color, width=line_width)
    # Perpendicular from top-left corner to diagonal
    draw.line([(0, 0), (width // 2, height // 2)], fill=color, width=line_width)
    # Perpendicular from bottom-right corner to diagonal
    draw.line([(width, height), (width // 2, height // 2)],
              fill=color, width=line_width)


def _draw_grid_4x4(draw, width: int, height: int, color: Tuple):
    """Draw 4x4 grid overlay."""
    line_width = max(1, min(width, height) // 600)

    for i in range(1, 4):
        x = int(width * i / 4)
        draw.line([(x, 0), (x, height)], fill=color, width=line_width)
    for i in range(1, 4):
        y = int(height * i / 4)
        draw.line([(0, y), (width, y)], fill=color, width=line_width)


def generate_guide_overlay(
    input_path: str,
    guides: Optional[List[str]] = None,
    timestamp: float = 0.0,
    output_dir: str = "",
    opacity: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate composition guide overlay on a video frame.

    Args:
        input_path: Source video file path.
        guides: List of guide types to draw. Defaults to ["rule_of_thirds"].
        timestamp: Time in seconds to extract frame from.
        output_dir: Output directory. Uses input dir if empty.
        opacity: Global opacity modifier (0.0-1.0).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, dimensions, guides applied.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if guides is None:
        guides = ["rule_of_thirds"]

    # Validate guide types
    valid_guides = []
    for g in guides:
        g = g.strip().lower()
        if g in GUIDE_TYPES:
            valid_guides.append(g)
    if not valid_guides:
        valid_guides = ["rule_of_thirds"]

    if on_progress:
        on_progress(5, "Ensuring Pillow is available...")

    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow is required for composition guides")

    from PIL import Image, ImageDraw

    if on_progress:
        on_progress(10, "Extracting frame...")

    # Get video dimensions
    info = get_video_info(input_path)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    # Extract frame
    tmp_dir = tempfile.mkdtemp(prefix="opencut_guide_")
    frame_path = os.path.join(tmp_dir, "frame.png")

    try:
        extracted = _extract_frame(input_path, timestamp, frame_path)

        if on_progress:
            on_progress(30, "Drawing composition guides...")

        if extracted and os.path.isfile(frame_path):
            base_img = Image.open(frame_path).convert("RGBA")
            width, height = base_img.size
        else:
            # Create a transparent overlay sized to video dimensions
            base_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Create overlay layer
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Apply opacity modifier to all colors
        def _apply_opacity(color: Tuple, opacity_mod: float) -> Tuple:
            r, g, b, a = color
            return (r, g, b, int(a * opacity_mod))

        for guide in valid_guides:
            if on_progress:
                idx = valid_guides.index(guide)
                pct = 30 + int(50 * (idx + 1) / len(valid_guides))
                on_progress(pct, f"Drawing {guide}...")

            if guide == "rule_of_thirds":
                color = _apply_opacity(DEFAULT_COLORS["rule_of_thirds"], opacity)
                _draw_rule_of_thirds(draw, width, height, color)
            elif guide == "golden_ratio":
                color = _apply_opacity(DEFAULT_COLORS["golden_ratio"], opacity)
                _draw_golden_ratio(draw, width, height, color)
            elif guide == "diagonal":
                color = _apply_opacity(DEFAULT_COLORS["diagonal"], opacity)
                _draw_diagonal(draw, width, height, color)
            elif guide == "center_cross":
                color = _apply_opacity(DEFAULT_COLORS["center_cross"], opacity)
                _draw_center_cross(draw, width, height, color)
            elif guide == "safe_areas":
                title_c = _apply_opacity(DEFAULT_COLORS["title_safe"], opacity)
                action_c = _apply_opacity(DEFAULT_COLORS["action_safe"], opacity)
                _draw_safe_areas(draw, width, height, title_c, action_c)
            elif guide == "fibonacci_spiral":
                color = _apply_opacity(DEFAULT_COLORS["fibonacci_spiral"], opacity)
                _draw_fibonacci_spiral(draw, width, height, color)
            elif guide == "triangle":
                color = _apply_opacity(DEFAULT_COLORS["triangle"], opacity)
                _draw_triangle(draw, width, height, color)
            elif guide == "grid_4x4":
                color = _apply_opacity(DEFAULT_COLORS["grid_4x4"], opacity)
                _draw_grid_4x4(draw, width, height, color)

        # Composite overlay onto frame
        result_img = Image.alpha_composite(base_img, overlay)

        if on_progress:
            on_progress(85, "Saving output...")

        # Determine output path
        out_dir = output_dir or os.path.dirname(input_path)
        guide_suffix = "_".join(valid_guides[:3])
        out_name = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(out_dir, f"{out_name}_guide_{guide_suffix}.png")

        result_img.save(out_path, "PNG")

        if on_progress:
            on_progress(100, "Composition guide complete")

        return {
            "output_path": out_path,
            "width": width,
            "height": height,
            "guides_applied": valid_guides,
            "timestamp": timestamp,
        }

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def list_guide_types() -> List[Dict]:
    """Return available guide types with descriptions."""
    descriptions = {
        "rule_of_thirds": "Classic rule-of-thirds grid with power points",
        "golden_ratio": "Golden ratio (phi) division lines",
        "diagonal": "Diagonal composition lines with bisectors",
        "center_cross": "Center crosshair with circle",
        "safe_areas": "Broadcast safe areas (title 80%, action 90%)",
        "fibonacci_spiral": "Fibonacci/golden spiral approximation",
        "triangle": "Triangular composition guides",
        "grid_4x4": "4x4 grid overlay",
    }
    return [
        {"type": g, "description": descriptions.get(g, "")}
        for g in GUIDE_TYPES
    ]
