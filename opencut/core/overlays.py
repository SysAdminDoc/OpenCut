"""
OpenCut Overlay Effects

Timecode burn-in and countdown/elapsed timer overlays using FFmpeg drawtext.

Uses FFmpeg only - no additional dependencies required.
"""

import logging
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Position presets for drawtext x:y expressions
# ---------------------------------------------------------------------------
_POSITION_MAP = {
    "top-left": ("20", "20"),
    "top-right": ("w-tw-20", "20"),
    "bottom-left": ("20", "h-th-20"),
    "bottom-right": ("w-tw-20", "h-th-20"),
    "center": ("(w-tw)/2", "(h-th)/2"),
}

_VALID_POSITIONS = list(_POSITION_MAP.keys())


def _resolve_position(position: str):
    """Return (x_expr, y_expr) for a named position."""
    key = position.lower().strip()
    if key not in _POSITION_MAP:
        raise ValueError(
            f"Invalid position: {position!r}. Valid: {', '.join(_VALID_POSITIONS)}"
        )
    return _POSITION_MAP[key]


# =========================================================================
# Feature 44.1 - Timecode Burn-In Overlay
# =========================================================================

def burn_timecode(
    input_path: str,
    output_path_override: str = None,
    position: str = "top-left",
    font_size: int = 24,
    color: str = "white",
    bg_color: str = "black@0.5",
    start_tc: str = "00:00:00:00",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Burn timecode text (HH:MM:SS:FF) onto a video.

    Args:
        input_path: Source video file path.
        output_path_override: Output file path. Auto-generated if None.
        position: Text position (top-left, top-right, bottom-left,
                  bottom-right, center).
        font_size: Font size in pixels.
        color: Text color (FFmpeg color name or hex).
        bg_color: Background box color with optional opacity
                  (e.g. "black@0.5").
        start_tc: Starting timecode in HH:MM:SS:FF format.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path key.
    """
    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(input_path)
    fps = info["fps"]
    duration = info["duration"]

    if on_progress:
        on_progress(10, "Building timecode filter...")

    x_expr, y_expr = _resolve_position(position)

    # Parse start timecode to compute frame offset
    tc_parts = start_tc.replace(";", ":").split(":")
    if len(tc_parts) != 4:
        raise ValueError(
            f"Invalid timecode format: {start_tc!r}. Expected HH:MM:SS:FF"
        )
    try:
        tc_h, tc_m, tc_s, tc_f = [int(p) for p in tc_parts]
    except ValueError:
        raise ValueError(
            f"Invalid timecode values: {start_tc!r}. All parts must be integers."
        )

    tc_h * 3600 + tc_m * 60 + tc_s + tc_f / fps

    # Build drawtext filter with timecode expression
    # Use pts-based timecode: hours, minutes, seconds, frame number
    # The expression calculates HH:MM:SS:FF from PTS + offset
    #
    # We use the FFmpeg timecode option for clean HH:MM:SS:FF display.
    # drawtext timecode option uses the format HH\:MM\:SS\:FF
    escaped_tc = start_tc.replace(":", "\\:")

    # Escape the bg_color colons for filter syntax
    escaped_bg = bg_color.replace(":", "\\:")

    drawtext = (
        f"drawtext="
        f"timecode='{escaped_tc}'"
        f":rate={fps:.6f}"
        f":fontsize={int(font_size)}"
        f":fontcolor={color}"
        f":box=1:boxcolor={escaped_bg}:boxborderw=8"
        f":x={x_expr}:y={y_expr}"
    )

    out = output_path_override or output_path(input_path, "timecode")

    if on_progress:
        on_progress(15, "Rendering timecode overlay...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(drawtext)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=max(300, int(duration * 3)))

    if on_progress:
        on_progress(100, "Timecode burn-in complete")

    return {"output_path": out}


# =========================================================================
# Feature 34.4 - Countdown Timer Overlay
# =========================================================================

def burn_countdown(
    input_path: str,
    output_path_override: str = None,
    duration_seconds: float = None,
    position: str = "center",
    font_size: int = 48,
    color: str = "white",
    bg_color: str = "black@0.7",
    timer_format: str = "MM:SS",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Burn a countdown timer overlay onto a video.

    The timer counts down from duration_seconds (or video duration) to zero.

    Args:
        input_path: Source video file path.
        output_path_override: Output file path. Auto-generated if None.
        duration_seconds: Countdown starting value in seconds.
                         Defaults to video duration.
        position: Text position preset.
        font_size: Font size in pixels.
        color: Text color.
        bg_color: Background color with opacity.
        timer_format: Display format - "MM:SS" or "HH:MM:SS".
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path key.
    """
    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(input_path)
    vid_duration = info["duration"]

    if duration_seconds is None:
        duration_seconds = vid_duration
    duration_seconds = max(0.0, float(duration_seconds))

    if on_progress:
        on_progress(10, "Building countdown filter...")

    x_expr, y_expr = _resolve_position(position)

    # Escape the bg_color colons for filter syntax
    escaped_bg = bg_color.replace(":", "\\:")

    # Build drawtext with countdown expression
    # remaining = duration_seconds - pts  (clamped to >= 0)
    # We use FFmpeg text expansion with expression evaluation
    if timer_format.upper() == "HH:MM:SS":
        # HH:MM:SS countdown
        text_expr = (
            f"%{{eif\\:max(0,floor(({duration_seconds}-t)/3600))\\:d\\:2}}"
            f"\\:"
            f"%{{eif\\:max(0,mod(floor(({duration_seconds}-t)/60),60))\\:d\\:2}}"
            f"\\:"
            f"%{{eif\\:max(0,mod(floor({duration_seconds}-t),60))\\:d\\:2}}"
        )
    else:
        # MM:SS countdown (default)
        text_expr = (
            f"%{{eif\\:max(0,floor(({duration_seconds}-t)/60))\\:d\\:2}}"
            f"\\:"
            f"%{{eif\\:max(0,mod(floor({duration_seconds}-t),60))\\:d\\:2}}"
        )

    drawtext = (
        f"drawtext="
        f"text='{text_expr}'"
        f":fontsize={int(font_size)}"
        f":fontcolor={color}"
        f":box=1:boxcolor={escaped_bg}:boxborderw=10"
        f":x={x_expr}:y={y_expr}"
    )

    out = output_path_override or output_path(input_path, "countdown")

    if on_progress:
        on_progress(15, "Rendering countdown overlay...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(drawtext)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=max(300, int(vid_duration * 3)))

    if on_progress:
        on_progress(100, "Countdown overlay complete")

    return {"output_path": out}


# =========================================================================
# Feature 34.4 - Elapsed Timer Overlay
# =========================================================================

def burn_elapsed_timer(
    input_path: str,
    output_path_override: str = None,
    start_seconds: float = 0,
    position: str = "bottom-right",
    font_size: int = 24,
    color: str = "white",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Burn an elapsed (count-up) timer overlay onto a video.

    Timer shows time elapsed from start_seconds, counting upward.

    Args:
        input_path: Source video file path.
        output_path_override: Output file path. Auto-generated if None.
        start_seconds: Starting value for the timer (default 0).
        position: Text position preset.
        font_size: Font size in pixels.
        color: Text color.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path key.
    """
    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(input_path)
    vid_duration = info["duration"]

    start_seconds = max(0.0, float(start_seconds))

    if on_progress:
        on_progress(10, "Building elapsed timer filter...")

    x_expr, y_expr = _resolve_position(position)

    # Elapsed timer: HH:MM:SS counting up from start_seconds
    text_expr = (
        f"%{{eif\\:floor(({start_seconds}+t)/3600)\\:d\\:2}}"
        f"\\:"
        f"%{{eif\\:mod(floor(({start_seconds}+t)/60),60)\\:d\\:2}}"
        f"\\:"
        f"%{{eif\\:mod(floor({start_seconds}+t),60)\\:d\\:2}}"
    )

    drawtext = (
        f"drawtext="
        f"text='{text_expr}'"
        f":fontsize={int(font_size)}"
        f":fontcolor={color}"
        f":box=1:boxcolor=black@0.5:boxborderw=6"
        f":x={x_expr}:y={y_expr}"
    )

    out = output_path_override or output_path(input_path, "elapsed")

    if on_progress:
        on_progress(15, "Rendering elapsed timer overlay...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(drawtext)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=max(300, int(vid_duration * 3)))

    if on_progress:
        on_progress(100, "Elapsed timer overlay complete")

    return {"output_path": out}
