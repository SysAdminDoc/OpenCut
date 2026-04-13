"""
OpenCut Click & Keystroke Overlay (Feature 11.2)

Render click ripple animations and keystroke badges (e.g. "Ctrl+S") on video.
Parse click/keystroke log files or accept manual event lists.

Uses FFmpeg drawtext/drawbox filters - no additional dependencies required.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ClickEvent:
    """A mouse click event at a specific time and position."""
    timestamp: float  # seconds into the video
    x: int            # pixel x coordinate
    y: int            # pixel y coordinate
    button: str = "left"    # left, right, middle
    duration: float = 0.4   # ripple animation duration in seconds


@dataclass
class KeystrokeEvent:
    """A keystroke event at a specific time."""
    timestamp: float        # seconds into the video
    keys: str               # e.g. "Ctrl+S", "Enter", "Alt+Tab"
    duration: float = 1.5   # how long the badge stays visible


@dataclass
class ClickOverlayResult:
    """Result of rendering click overlay."""
    output_path: str
    click_count: int
    duration: float

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "click_count": self.click_count,
            "duration": self.duration,
        }


@dataclass
class KeystrokeOverlayResult:
    """Result of rendering keystroke overlay."""
    output_path: str
    keystroke_count: int
    duration: float

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "keystroke_count": self.keystroke_count,
            "duration": self.duration,
        }


# ---------------------------------------------------------------------------
# Ripple color by button type
# ---------------------------------------------------------------------------
_BUTTON_COLORS = {
    "left": ("FFD700", "FFFF00"),     # gold/yellow
    "right": ("FF4500", "FF6347"),    # red-orange
    "middle": ("00BFFF", "87CEEB"),   # blue
}

# ---------------------------------------------------------------------------
# Click log parsing
# ---------------------------------------------------------------------------

def parse_click_log(log_path: str) -> Dict[str, list]:
    """Parse a click/keystroke log file.

    Supports two formats:
      1. JSON: ``{"clicks": [...], "keystrokes": [...]}``
      2. Text lines: ``click <timestamp> <x> <y> [button]``
                     ``key <timestamp> <keys> [duration]``

    Args:
        log_path: Path to the log file.

    Returns:
        Dict with ``clicks`` (list of ClickEvent) and
        ``keystrokes`` (list of KeystrokeEvent).

    Raises:
        FileNotFoundError: If the log file doesn't exist.
        ValueError: If the file cannot be parsed.
    """
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("Log file is empty")

    # Try JSON format first
    try:
        data = json.loads(content)
        clicks = []
        keystrokes = []
        for c in data.get("clicks", []):
            clicks.append(ClickEvent(
                timestamp=float(c["timestamp"]),
                x=int(c["x"]),
                y=int(c["y"]),
                button=str(c.get("button", "left")),
                duration=float(c.get("duration", 0.4)),
            ))
        for k in data.get("keystrokes", []):
            keystrokes.append(KeystrokeEvent(
                timestamp=float(k["timestamp"]),
                keys=str(k["keys"]),
                duration=float(k.get("duration", 1.5)),
            ))
        return {"clicks": clicks, "keystrokes": keystrokes}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Text format: "click 1.5 320 240 left" or "key 2.0 Ctrl+S 1.5"
    clicks = []
    keystrokes = []
    for line_num, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue

        kind = parts[0].lower()
        if kind == "click":
            if len(parts) < 4:
                raise ValueError(
                    f"Line {line_num}: click requires at least timestamp, x, y"
                )
            try:
                ts = float(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                button = parts[4] if len(parts) > 4 else "left"
                dur = float(parts[5]) if len(parts) > 5 else 0.4
                clicks.append(ClickEvent(
                    timestamp=ts, x=x, y=y, button=button, duration=dur
                ))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Line {line_num}: invalid click data: {e}")

        elif kind in ("key", "keystroke"):
            if len(parts) < 3:
                raise ValueError(
                    f"Line {line_num}: keystroke requires at least timestamp and keys"
                )
            try:
                ts = float(parts[1])
                keys = parts[2]
                dur = float(parts[3]) if len(parts) > 3 else 1.5
                keystrokes.append(KeystrokeEvent(
                    timestamp=ts, keys=keys, duration=dur
                ))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Line {line_num}: invalid keystroke data: {e}")

    if not clicks and not keystrokes:
        raise ValueError("No click or keystroke events found in log file")

    return {"clicks": clicks, "keystrokes": keystrokes}


# ---------------------------------------------------------------------------
# Click ripple rendering
# ---------------------------------------------------------------------------

def _build_click_filter(
    click: ClickEvent,
    video_width: int,
    video_height: int,
    max_radius: int = 30,
) -> str:
    """Build FFmpeg drawbox filter for a click ripple animation.

    Creates an expanding circle effect by using multiple concentric
    drawbox commands that fade in/out around the click timestamp.
    """
    filters = []
    t_start = click.timestamp
    t_end = t_start + click.duration
    cx, cy = click.x, click.y

    # Clamp to video bounds
    cx = max(0, min(cx, video_width - 1))
    cy = max(0, min(cy, video_height - 1))

    colors = _BUTTON_COLORS.get(click.button, _BUTTON_COLORS["left"])
    inner_color = colors[0]
    outer_color = colors[1]

    # Create 3 expanding rings for ripple effect
    for ring_idx, (radius_frac, alpha_start) in enumerate([
        (0.4, 0.9), (0.7, 0.6), (1.0, 0.3)
    ]):
        ring_delay = ring_idx * (click.duration * 0.15)
        r_start = t_start + ring_delay
        r_end = t_end

        if r_start >= r_end:
            continue

        radius = int(max_radius * radius_frac)
        half = radius // 2

        # Draw a box centered on the click position with time-based enable
        bx = max(0, cx - half)
        by = max(0, cy - half)
        color = outer_color if ring_idx > 0 else inner_color

        # Alpha fades from alpha_start to 0 over the ring's lifetime
        f"{alpha_start}*(1-((t-{r_start})/{max(r_end - r_start, 0.01)}))"
        enable_expr = f"between(t,{r_start:.3f},{r_end:.3f})"

        filters.append(
            f"drawbox=x={bx}:y={by}:w={radius}:h={radius}"
            f":color={color}@0.5:t=3:enable='{enable_expr}'"
        )

    # Center dot (solid small box)
    dot_size = 6
    dx = max(0, cx - dot_size // 2)
    dy = max(0, cy - dot_size // 2)
    enable_dot = f"between(t,{t_start:.3f},{t_end:.3f})"
    filters.append(
        f"drawbox=x={dx}:y={dy}:w={dot_size}:h={dot_size}"
        f":color={inner_color}@0.9:t=fill:enable='{enable_dot}'"
    )

    return filters


def render_click_overlay(
    video_path: str,
    click_events: List[ClickEvent],
    output_path_str: str = None,
    max_radius: int = 30,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Render click ripple animations on a video.

    Args:
        video_path: Source video file path.
        click_events: List of ClickEvent objects.
        output_path_str: Output file path (auto-generated if None).
        max_radius: Maximum ripple radius in pixels.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, click_count, duration.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If click_events is empty.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not click_events:
        raise ValueError("No click events provided")

    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(video_path)
    w, h = info["width"], info["height"]
    duration = info["duration"]

    if on_progress:
        on_progress(10, f"Building overlay for {len(click_events)} clicks...")

    # Sort events by timestamp
    sorted_events = sorted(click_events, key=lambda e: e.timestamp)

    # Build filter chain
    all_filters = []
    for click in sorted_events:
        if click.timestamp < 0 or (duration > 0 and click.timestamp > duration):
            logger.warning(
                "Click at t=%.2f outside video duration (%.2f), skipping",
                click.timestamp, duration,
            )
            continue
        click_filters = _build_click_filter(click, w, h, max_radius)
        all_filters.extend(click_filters)

    if not all_filters:
        raise ValueError("No valid click events within video duration")

    # Join all filters with commas for the -vf chain
    vf_chain = ",".join(all_filters)

    if on_progress:
        on_progress(20, "Encoding video with click overlays...")

    out = output_path_str or output_path(video_path, "clicks")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf_chain)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    if on_progress:
        on_progress(30, "Running FFmpeg...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Done")

    result = ClickOverlayResult(
        output_path=out,
        click_count=len(sorted_events),
        duration=duration,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Keystroke badge rendering
# ---------------------------------------------------------------------------

_BADGE_POSITIONS = {
    "top-left": ("20", "20"),
    "top-right": ("w-tw-20", "20"),
    "bottom-left": ("20", "h-th-20"),
    "bottom-right": ("w-tw-20", "h-th-20"),
    "center": ("(w-tw)/2", "(h-th)/2"),
}


def _escape_drawtext(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # FFmpeg drawtext requires escaping of : ; ' \ and %
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "%%")
    return text


def render_keystroke_overlay(
    video_path: str,
    keystroke_events: List[KeystrokeEvent],
    output_path_str: str = None,
    position: str = "bottom-left",
    font_size: int = 28,
    font_color: str = "white",
    bg_color: str = "black",
    bg_opacity: float = 0.7,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Render keystroke badge overlays on a video.

    Displays keyboard shortcut badges (e.g. "Ctrl+S") at the specified
    position with a background box.

    Args:
        video_path: Source video file path.
        keystroke_events: List of KeystrokeEvent objects.
        output_path_str: Output file path (auto-generated if None).
        position: Badge position (top-left, top-right, bottom-left,
                  bottom-right, center).
        font_size: Font size in pixels.
        font_color: Text color.
        bg_color: Background color.
        bg_opacity: Background opacity (0.0 - 1.0).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, keystroke_count, duration.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If keystroke_events is empty or position is invalid.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not keystroke_events:
        raise ValueError("No keystroke events provided")

    pos_key = position.lower().strip()
    if pos_key not in _BADGE_POSITIONS:
        raise ValueError(
            f"Invalid position: {position!r}. "
            f"Valid: {', '.join(_BADGE_POSITIONS.keys())}"
        )

    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(video_path)
    duration = info["duration"]

    if on_progress:
        on_progress(10, f"Building overlay for {len(keystroke_events)} keystrokes...")

    x_expr, y_expr = _BADGE_POSITIONS[pos_key]

    # Sort events by timestamp
    sorted_events = sorted(keystroke_events, key=lambda e: e.timestamp)

    # Build drawtext filters for each keystroke
    filters = []
    valid_count = 0
    for ks in sorted_events:
        if ks.timestamp < 0 or (duration > 0 and ks.timestamp > duration):
            logger.warning(
                "Keystroke at t=%.2f outside video duration, skipping",
                ks.timestamp,
            )
            continue

        t_start = ks.timestamp
        t_end = t_start + ks.duration
        escaped_keys = _escape_drawtext(ks.keys)
        enable_expr = f"between(t,{t_start:.3f},{t_end:.3f})"

        # Background box + text
        dt_filter = (
            f"drawtext=text='{escaped_keys}'"
            f":fontsize={font_size}"
            f":fontcolor={font_color}"
            f":x={x_expr}:y={y_expr}"
            f":box=1:boxcolor={bg_color}@{bg_opacity:.2f}"
            f":boxborderw=8"
            f":enable='{enable_expr}'"
        )
        filters.append(dt_filter)
        valid_count += 1

    if not filters:
        raise ValueError("No valid keystroke events within video duration")

    vf_chain = ",".join(filters)

    if on_progress:
        on_progress(20, "Encoding video with keystroke overlays...")

    out = output_path_str or output_path(video_path, "keystrokes")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf_chain)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    if on_progress:
        on_progress(30, "Running FFmpeg...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Done")

    result = KeystrokeOverlayResult(
        output_path=out,
        keystroke_count=valid_count,
        duration=duration,
    )
    return result.to_dict()
