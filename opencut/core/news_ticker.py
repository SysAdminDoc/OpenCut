"""
OpenCut News Ticker / Crawl Overlay v1.0.0

Scrolling text ticker overlay for video:
  - Text from string, list, JSON, or text file
  - Configurable speed, font, colors, direction, position
  - Standalone ticker video or overlay on existing video
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKER_POSITIONS = {
    "bottom": "main_h-th-{margin}",
    "top": "{margin}",
    "center": "(main_h-th)/2",
}

TICKER_DIRECTIONS = {
    "left": "scroll_left",
    "right": "scroll_right",
}

DEFAULT_FONT_SIZE = 48
DEFAULT_SPEED = 100  # pixels per second
DEFAULT_BG_COLOR = "black@0.7"
DEFAULT_TEXT_COLOR = "white"
DEFAULT_MARGIN = 20

# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class TickerConfig:
    """Ticker configuration."""
    text: str = ""
    speed: int = DEFAULT_SPEED
    font_size: int = DEFAULT_FONT_SIZE
    font_color: str = DEFAULT_TEXT_COLOR
    bg_color: str = DEFAULT_BG_COLOR
    direction: str = "left"
    position: str = "bottom"
    margin: int = DEFAULT_MARGIN
    separator: str = "   +++   "
    loop_count: int = 1


# ---------------------------------------------------------------------------
# Text Loading
# ---------------------------------------------------------------------------

def _load_ticker_text(
    text_content: Union[str, list, dict],
    separator: str = "   +++   ",
) -> str:
    """Load and normalize ticker text from various sources.

    Args:
        text_content: String, list of strings, dict with 'items' key,
                      or path to a text/JSON file.
        separator: Separator between items when joining a list.

    Returns:
        Single string of ticker text ready for display.
    """
    if isinstance(text_content, list):
        return separator.join(str(item) for item in text_content if item)

    if isinstance(text_content, dict):
        items = text_content.get("items", text_content.get("headlines", []))
        if isinstance(items, list):
            return separator.join(str(item) for item in items if item)
        return str(items)

    text = str(text_content).strip()

    # Check if it's a file path
    if os.path.isfile(text):
        try:
            with open(text, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Try parsing as JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return separator.join(str(item) for item in data if item)
                elif isinstance(data, dict):
                    items = data.get("items", data.get("headlines", []))
                    if isinstance(items, list):
                        return separator.join(str(item) for item in items if item)
                    return str(items)
            except json.JSONDecodeError:
                pass

            # Plain text: join lines
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if len(lines) > 1:
                return separator.join(lines)
            return content

        except (IOError, UnicodeDecodeError) as e:
            logger.warning("Failed to read ticker text file: %s", e)
            return text

    return text


def _escape_drawtext(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    # FFmpeg drawtext special characters
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    text = text.replace("\n", " ")
    return text


# ---------------------------------------------------------------------------
# Ticker Creation
# ---------------------------------------------------------------------------

def create_ticker(
    text_content: Union[str, list, dict],
    video_path: str,
    output_path_str: str = "",
    speed: int = DEFAULT_SPEED,
    font_size: int = DEFAULT_FONT_SIZE,
    font_color: str = DEFAULT_TEXT_COLOR,
    bg_color: str = DEFAULT_BG_COLOR,
    direction: str = "left",
    position: str = "bottom",
    margin: int = DEFAULT_MARGIN,
    separator: str = "   +++   ",
    bg_height: int = 0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Overlay a scrolling news ticker on an existing video.

    Args:
        text_content: Ticker text (string, list, dict, or file path).
        video_path: Path to the input video.
        output_path_str: Output path. Auto-generated if empty.
        speed: Scroll speed in pixels per second.
        font_size: Font size in pixels.
        font_color: Text color (FFmpeg color name or hex).
        bg_color: Background bar color (with optional alpha, e.g. "black@0.7").
        direction: Scroll direction — "left" or "right".
        position: Vertical position — "bottom", "top", or "center".
        margin: Margin from edge in pixels.
        separator: Separator between text items.
        bg_height: Background bar height in pixels (0 = auto from font size).
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, duration, ticker_text, config.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If text_content is empty.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    ticker_text = _load_ticker_text(text_content, separator)
    if not ticker_text.strip():
        raise ValueError("Ticker text is empty.")

    safe_text = _escape_drawtext(ticker_text)
    info = get_video_info(video_path)
    duration = info["duration"]
    info["width"]

    speed = max(10, min(1000, int(speed)))
    font_size = max(12, min(200, int(font_size)))
    margin = max(0, min(200, int(margin)))

    direction = direction.lower().strip()
    if direction not in TICKER_DIRECTIONS:
        direction = "left"

    position = position.lower().strip()
    if position not in TICKER_POSITIONS:
        position = "bottom"

    if not output_path_str:
        output_path_str = output_path(video_path, "ticker")

    if on_progress:
        on_progress(10, "Building ticker overlay...")

    # Calculate bar height
    bar_h = bg_height if bg_height > 0 else font_size + margin
    TICKER_POSITIONS[position].format(margin=margin)

    # Build drawtext filter for scrolling text
    if direction == "left":
        # Text scrolls from right to left
        x_expr = f"w-mod(t*{speed}\\,w+tw)-tw+w"
    else:
        # Text scrolls from left to right
        x_expr = f"-tw+mod(t*{speed}\\,w+tw)"

    # Y position for the text within the bar
    if position == "bottom":
        bar_y = f"h-{bar_h}-{margin}"
        text_y = f"h-{bar_h}-{margin}+({bar_h}-th)/2"
    elif position == "top":
        bar_y = str(margin)
        text_y = f"{margin}+({bar_h}-th)/2"
    else:
        bar_y = f"(h-{bar_h})/2"
        text_y = f"(h-{bar_h})/2+({bar_h}-th)/2"

    # Background bar + scrolling text
    vf = (
        f"drawbox=x=0:y={bar_y}:w=iw:h={bar_h}:color={bg_color}:t=fill,"
        f"drawtext=text='{safe_text}'"
        f":fontsize={font_size}:fontcolor={font_color}"
        f":x='{x_expr}':y={text_y}"
    )

    cmd = (FFmpegCmd()
           .input(video_path)
           .video_filter(vf)
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("copy")
           .faststart()
           .output(output_path_str)
           .build())

    if on_progress:
        on_progress(30, "Rendering ticker overlay...")

    try:
        run_ffmpeg(cmd, timeout=3600)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to create ticker overlay: {e}")

    if on_progress:
        on_progress(100, "Ticker overlay complete.")

    return {
        "output_path": output_path_str,
        "duration": duration,
        "ticker_text": ticker_text[:200] + ("..." if len(ticker_text) > 200 else ""),
        "config": {
            "speed": speed,
            "font_size": font_size,
            "direction": direction,
            "position": position,
            "font_color": font_color,
            "bg_color": bg_color,
        },
    }


def create_ticker_overlay(
    text_content: Union[str, list, dict],
    duration: float = 10.0,
    output_path_str: str = "",
    width: int = 1920,
    height: int = 80,
    speed: int = DEFAULT_SPEED,
    font_size: int = DEFAULT_FONT_SIZE,
    font_color: str = DEFAULT_TEXT_COLOR,
    bg_color: str = "000000",
    direction: str = "left",
    separator: str = "   +++   ",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a standalone ticker overlay video (no source video needed).

    This creates a video with alpha channel that can be composited over
    other videos in an editor.

    Args:
        text_content: Ticker text (string, list, dict, or file path).
        duration: Duration in seconds.
        output_path_str: Output path. Auto-generated if empty.
        width: Output width.
        height: Output height (ticker bar height).
        speed: Scroll speed in pixels per second.
        font_size: Font size.
        font_color: Text color.
        bg_color: Background color (hex without alpha).
        direction: Scroll direction.
        separator: Separator between text items.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, duration, dimensions.
    """
    ticker_text = _load_ticker_text(text_content, separator)
    if not ticker_text.strip():
        raise ValueError("Ticker text is empty.")

    safe_text = _escape_drawtext(ticker_text)

    duration = max(1.0, min(300.0, float(duration)))
    speed = max(10, min(1000, int(speed)))
    font_size = max(12, min(200, int(font_size)))
    height = max(font_size + 10, int(height))

    if not output_path_str:
        output_path_str = os.path.join(
            tempfile.gettempdir(),
            f"opencut_ticker_{int(time.time())}.mov"
        )

    if on_progress:
        on_progress(10, "Building standalone ticker...")

    if direction == "right":
        x_expr = f"-tw+mod(t*{speed}\\,{width}+tw)"
    else:
        x_expr = f"{width}-mod(t*{speed}\\,{width}+tw)-tw+{width}"

    text_y = f"({height}-th)/2"

    # Generate with color source + drawtext
    fc = (
        f"color=c=0x{bg_color}:s={width}x{height}:d={duration},"
        f"drawtext=text='{safe_text}'"
        f":fontsize={font_size}:fontcolor={font_color}"
        f":x='{x_expr}':y={text_y}"
        f"[out]"
    )

    cmd = (FFmpegCmd()
           .filter_complex(fc, maps=["[out]"])
           .video_codec("libx264", crf=18, preset="fast")
           .option("t", str(duration))
           .output(output_path_str)
           .build())

    if on_progress:
        on_progress(30, "Rendering ticker overlay...")

    try:
        run_ffmpeg(cmd, timeout=300)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to create ticker overlay: {e}")

    if on_progress:
        on_progress(100, "Ticker overlay created.")

    return {
        "output_path": output_path_str,
        "duration": duration,
        "width": width,
        "height": height,
        "ticker_text": ticker_text[:200] + ("..." if len(ticker_text) > 200 else ""),
    }
