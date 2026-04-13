"""
OpenCut Scrolling Credits Generator

Renders scrolling movie-style credits as a video using Pillow + FFmpeg.
Supports section headers with two-column name layout.
"""

import logging
import os
import tempfile
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")


def parse_credits_file(filepath: str) -> List[Dict]:
    """
    Parse a plain-text credits file into structured credit dicts.

    Expected format:
        SECTION HEADER
        Name One
        Name Two

        NEXT SECTION
        Another Name
        ...

    Section headers are detected as lines in ALL CAPS (or lines followed by
    a blank line then indented/normal-case names). A blank line separates
    sections.

    Returns:
        List of dicts: [{"section": "Directed By", "names": ["Jane Smith"]}, ...]
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Credits file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    sections: List[Dict] = []
    current_section: Optional[str] = None
    current_names: List[str] = []

    for line in raw.splitlines():
        stripped = line.strip()

        if not stripped:
            # Blank line: flush current section if we have one
            if current_section is not None and current_names:
                sections.append({
                    "section": current_section,
                    "names": list(current_names),
                })
                current_section = None
                current_names = []
            elif current_section is not None and not current_names:
                # Header with no names yet -- keep waiting
                pass
            continue

        # Detect section header: all-uppercase line (at least 2 alpha chars)
        alpha_chars = [c for c in stripped if c.isalpha()]
        is_upper = (
            len(alpha_chars) >= 2
            and stripped == stripped.upper()
            and not stripped.startswith("-")
        )

        if is_upper and not current_names:
            # Flush previous section if any
            if current_section is not None:
                sections.append({
                    "section": current_section,
                    "names": list(current_names),
                })
                current_names = []
            # Title-case the header for display
            current_section = stripped.title()
        elif current_section is None:
            # First non-blank line without a header -- create unnamed section
            current_section = ""
            current_names.append(stripped)
        else:
            current_names.append(stripped)

    # Flush final section
    if current_section is not None:
        sections.append({
            "section": current_section,
            "names": list(current_names),
        })

    return sections


def generate_credits(
    credits_data: List[Dict],
    output_path_str: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    scroll_speed: int = 60,
    font_size: int = 36,
    font_color: str = "white",
    bg_color: str = "black",
    font_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate a scrolling credits video.

    Args:
        credits_data: List of dicts with "section" and "names" keys.
        output_path_str: Destination video file path.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        scroll_speed: Pixels per second of scroll.
        font_size: Base font size in pixels.
        font_color: Text color name or hex.
        bg_color: Background color name or hex.
        font_path: Optional path to a .ttf/.otf font file.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and duration.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError(
            "Pillow is required for credits generation. Install with: pip install Pillow"
        )

    if on_progress:
        on_progress(5, "Preparing credits layout...")

    # --- Resolve colors ---
    _COLOR_MAP = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }

    def _parse_color(c):
        if isinstance(c, tuple):
            return c
        c_lower = c.lower().strip()
        if c_lower in _COLOR_MAP:
            return _COLOR_MAP[c_lower]
        # Hex color
        c_clean = c_lower.lstrip("#")
        if len(c_clean) == 6:
            return tuple(int(c_clean[i:i+2], 16) for i in (0, 2, 4))
        return (255, 255, 255)

    text_color = _parse_color(font_color)
    background_color = _parse_color(bg_color)

    # --- Load font ---
    header_size = int(font_size * 1.4)
    try:
        if font_path and os.path.isfile(font_path):
            body_font = ImageFont.truetype(font_path, font_size)
            header_font = ImageFont.truetype(font_path, header_size)
        else:
            body_font = ImageFont.truetype("arial.ttf", font_size)
            header_font = ImageFont.truetype("arial.ttf", header_size)
    except (OSError, IOError):
        body_font = ImageFont.load_default()
        header_font = body_font

    if on_progress:
        on_progress(10, "Calculating credits dimensions...")

    # --- Calculate total image height ---
    line_spacing = int(font_size * 0.6)
    section_spacing = int(font_size * 2.0)
    top_padding = height  # Start off-screen (one full screen of padding)
    bottom_padding = height  # End off-screen

    # Pre-calculate all lines and their heights
    content_height = 0
    layout_lines = []  # list of (type, text, y_offset)

    y_cursor = 0
    for i, section in enumerate(credits_data):
        if i > 0:
            y_cursor += section_spacing

        section_name = section.get("section", "")
        names = section.get("names", [])

        # Section header (centered)
        if section_name:
            layout_lines.append(("header", section_name, y_cursor))
            bbox = header_font.getbbox(section_name) if hasattr(header_font, 'getbbox') else (0, 0, len(section_name) * header_size, header_size)
            y_cursor += (bbox[3] - bbox[1]) + line_spacing

        # Names
        for name in names:
            layout_lines.append(("name", name, y_cursor))
            bbox = body_font.getbbox(name) if hasattr(body_font, 'getbbox') else (0, 0, len(name) * font_size, font_size)
            y_cursor += (bbox[3] - bbox[1]) + line_spacing

    content_height = y_cursor
    total_image_height = top_padding + content_height + bottom_padding

    if on_progress:
        on_progress(20, "Rendering credits image...")

    # --- Render tall credits image ---
    img = Image.new("RGB", (width, total_image_height), background_color)
    draw = ImageDraw.Draw(img)

    for line_type, text, y_offset in layout_lines:
        actual_y = top_padding + y_offset

        if line_type == "header":
            font = header_font
            # Center the header
            bbox = font.getbbox(text) if hasattr(font, 'getbbox') else (0, 0, len(text) * header_size, header_size)
            text_w = bbox[2] - bbox[0]
            x = (width - text_w) // 2
            draw.text((x, actual_y), text, fill=text_color, font=font)
        else:
            font = body_font
            # Check for role/name pair separated by " as " or " - "
            parts = None
            for sep in [" as ", " - ", "\t"]:
                if sep in text:
                    parts = text.split(sep, 1)
                    break

            if parts and len(parts) == 2:
                # Two-column layout: role left-aligned, name right-aligned
                left_text = parts[0].strip()
                right_text = parts[1].strip()
                width // 6
                col_center = width // 2

                # Left side: right-aligned to center
                lbbox = font.getbbox(left_text) if hasattr(font, 'getbbox') else (0, 0, len(left_text) * font_size, font_size)
                lw = lbbox[2] - lbbox[0]
                draw.text((col_center - lw - 20, actual_y), left_text, fill=text_color, font=font)

                # Right side: left-aligned from center
                draw.text((col_center + 20, actual_y), right_text, fill=text_color, font=font)
            else:
                # Single name, centered
                bbox = font.getbbox(text) if hasattr(font, 'getbbox') else (0, 0, len(text) * font_size, font_size)
                text_w = bbox[2] - bbox[0]
                x = (width - text_w) // 2
                draw.text((x, actual_y), text, fill=text_color, font=font)

    if on_progress:
        on_progress(50, "Saving credits image...")

    # --- Save image to temp file ---
    tmp_dir = tempfile.gettempdir()
    img_path = os.path.join(tmp_dir, f"opencut_credits_{os.getpid()}.png")
    img.save(img_path, "PNG")

    # --- Calculate duration ---
    # The scroll distance is the entire image height (start off-screen, end off-screen)
    scroll_distance = total_image_height
    duration = scroll_distance / max(1, scroll_speed)

    if on_progress:
        on_progress(60, f"Encoding credits video ({duration:.1f}s)...")

    # --- FFmpeg: scroll the image over a solid background ---
    # Use overlay filter to scroll the credits image upward
    ffmpeg = get_ffmpeg_path()
    int(duration * fps)

    # The scroll expression: overlay y position moves from 0 to -image_height over time
    # y starts at 0 (image top at video top, credits off-screen below)
    # y ends at -(total_image_height - height) so credits scroll up past
    scroll_expr = f"y='H-t*{scroll_speed}'"

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d={duration:.3f}:r={fps}",
        "-i", img_path,
        "-filter_complex",
        f"[1:v]format=rgba[credits];[0:v][credits]overlay=x=0:{scroll_expr}:shortest=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path_str,
    ]

    if on_progress:
        on_progress(70, "Encoding with FFmpeg...")

    run_ffmpeg(cmd, timeout=max(300, int(duration * 5)))

    if on_progress:
        on_progress(95, "Cleaning up...")

    # Cleanup temp image
    try:
        os.unlink(img_path)
    except OSError:
        pass

    if on_progress:
        on_progress(100, "Credits video complete")

    return {
        "output_path": output_path_str,
        "duration": round(duration, 2),
        "width": width,
        "height": height,
        "total_lines": len(layout_lines),
    }
