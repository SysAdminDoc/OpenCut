"""
OpenCut Timecode Utilities

Drop-frame / non-drop-frame timecode conversion implementing the
SMPTE 12M standard.

Supports:
- Detecting timecode format from media files via ffprobe
- Converting between frame numbers and timecodes
- Converting timecodes between different frame rates
- Proper SMPTE 12M drop-frame algorithm for 29.97 / 59.94 fps

Uses FFmpeg/ffprobe only -- no additional dependencies required.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from opencut.helpers import get_ffprobe_path

logger = logging.getLogger("opencut")

# Frame rates that support drop-frame timecode
_DROP_FRAME_RATES = {29.97, 59.94}

# Tolerance for matching frame rates (handles float imprecision)
_FPS_TOLERANCE = 0.02


@dataclass
class TimecodeInfo:
    """Detected timecode format from a media file."""
    fps: float
    is_drop_frame: bool
    detected_tc: str


def _is_drop_frame_rate(fps: float) -> bool:
    """Check if a frame rate supports drop-frame timecode."""
    for df_fps in _DROP_FRAME_RATES:
        if abs(fps - df_fps) < _FPS_TOLERANCE:
            return True
    return False


def _round_fps(fps: float) -> float:
    """Round to nearest standard frame rate."""
    standards = [23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0]
    closest = min(standards, key=lambda s: abs(s - fps))
    if abs(fps - closest) < _FPS_TOLERANCE:
        return closest
    return fps


def detect_timecode_format(input_path: str) -> TimecodeInfo:
    """
    Detect the timecode format of a media file.

    Probes for embedded timecode and frame rate to determine if
    the file uses drop-frame or non-drop-frame timecode.

    Args:
        input_path: Path to media file.

    Returns:
        TimecodeInfo with fps, is_drop_frame, and detected_tc.
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-show_entries", "stream_tags=timecode",
        "-show_entries", "format_tags=timecode",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)

    fps = 30.0
    detected_tc = ""
    is_df = False

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout.decode())
            streams = data.get("streams", [])
            if streams:
                # Parse frame rate
                fps_str = streams[0].get("r_frame_rate", "30/1")
                parts = fps_str.split("/")
                if len(parts) == 2 and float(parts[1]) > 0:
                    fps = float(parts[0]) / float(parts[1])
                else:
                    fps = float(parts[0])
                fps = _round_fps(fps)

                # Check for embedded timecode in stream tags
                tags = streams[0].get("tags", {})
                detected_tc = tags.get("timecode", "")

            # Also check format-level tags
            if not detected_tc:
                fmt_tags = data.get("format", {}).get("tags", {})
                detected_tc = fmt_tags.get("timecode", "")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug("Failed to parse ffprobe timecode output: %s", e)

    # Determine drop-frame from timecode separator or frame rate
    if detected_tc:
        # Semicolons in timecode indicate drop-frame
        is_df = ";" in detected_tc
    else:
        # No embedded timecode -- infer from frame rate
        is_df = _is_drop_frame_rate(fps)

    return TimecodeInfo(fps=fps, is_drop_frame=is_df, detected_tc=detected_tc)


def frames_to_timecode(
    frame_num: int,
    fps: float,
    drop_frame: bool = False,
) -> str:
    """
    Convert a frame number to a timecode string.

    Implements SMPTE 12M drop-frame algorithm for 29.97 and 59.94 fps.

    Args:
        frame_num: Zero-based frame number.
        fps: Frame rate.
        drop_frame: Use drop-frame timecode format.

    Returns:
        Timecode string "HH:MM:SS:FF" (NDF) or "HH:MM:SS;FF" (DF).
    """
    frame_num = max(0, int(frame_num))
    fps = _round_fps(float(fps))

    if drop_frame and _is_drop_frame_rate(fps):
        return _frames_to_df_timecode(frame_num, fps)
    return _frames_to_ndf_timecode(frame_num, fps)


def _frames_to_ndf_timecode(frame_num: int, fps: float) -> str:
    """Non-drop-frame: simple division."""
    fps_int = round(fps)
    frames = frame_num % fps_int
    seconds = (frame_num // fps_int) % 60
    minutes = (frame_num // (fps_int * 60)) % 60
    hours = (frame_num // (fps_int * 3600)) % 24
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _frames_to_df_timecode(frame_num: int, fps: float) -> str:
    """SMPTE 12M drop-frame timecode.

    For 29.97 fps: drop frames 0 and 1 at the start of each minute,
    EXCEPT every 10th minute.

    For 59.94 fps: drop frames 0,1,2,3 at the start of each minute,
    EXCEPT every 10th minute.
    """
    fps_int = round(fps)  # 30 for 29.97, 60 for 59.94

    if abs(fps - 29.97) < _FPS_TOLERANCE:
        drop_count = 2  # Drop 2 frame numbers per minute
    elif abs(fps - 59.94) < _FPS_TOLERANCE:
        drop_count = 4  # Drop 4 frame numbers per minute
    else:
        # Not a standard DF rate -- fall back to NDF
        return _frames_to_ndf_timecode(frame_num, fps)

    # Total frames in various time periods
    frames_per_min = fps_int * 60 - drop_count
    frames_per_10min = frames_per_min * 10 + drop_count * 9  # Restore drops for 10th min
    # Actually: 10 minutes = 10 * (fps_int*60) - 9*drop_count
    frames_per_10min = fps_int * 60 * 10 - 9 * drop_count

    # Decompose frame number
    d = frame_num
    ten_min_blocks = d // frames_per_10min
    d = d % frames_per_10min

    # First minute of 10-min block has no drops
    if d < fps_int * 60:
        unit_minutes = 0
        remaining = d
    else:
        d -= fps_int * 60
        unit_minutes = 1 + d // frames_per_min
        remaining = d % frames_per_min
        remaining += drop_count  # Add back the dropped frames

    minutes = ten_min_blocks * 10 + unit_minutes
    hours = minutes // 60
    minutes = minutes % 60

    frames = remaining % fps_int
    seconds = (remaining // fps_int) % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d};{frames:02d}"


def timecode_to_frames(
    timecode: str,
    fps: float,
    drop_frame: Optional[bool] = None,
) -> int:
    """
    Convert a timecode string to a frame number.

    Args:
        timecode: Timecode string "HH:MM:SS:FF" or "HH:MM:SS;FF".
        fps: Frame rate.
        drop_frame: Force drop-frame interpretation. If None, auto-detect
                    from separator (";" = drop-frame, ":" = non-drop-frame).

    Returns:
        Zero-based frame number.
    """
    fps = _round_fps(float(fps))

    # Parse timecode -- accept both : and ; as separators
    tc = timecode.strip()
    parts = re.split(r"[:;]", tc)
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode format: {timecode}")

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        frames = int(parts[3])
    except ValueError:
        raise ValueError(f"Invalid timecode values: {timecode}")

    # Auto-detect drop-frame from separator
    if drop_frame is None:
        drop_frame = ";" in tc

    fps_int = round(fps)

    if drop_frame and _is_drop_frame_rate(fps):
        return _df_timecode_to_frames(hours, minutes, seconds, frames, fps)
    return _ndf_timecode_to_frames(hours, minutes, seconds, frames, fps_int)


def _ndf_timecode_to_frames(h: int, m: int, s: int, f: int, fps_int: int) -> int:
    """Non-drop-frame to frames: simple multiplication."""
    return (h * 3600 + m * 60 + s) * fps_int + f


def _df_timecode_to_frames(
    h: int, m: int, s: int, f: int, fps: float,
) -> int:
    """SMPTE 12M drop-frame timecode to frames."""
    fps_int = round(fps)

    if abs(fps - 29.97) < _FPS_TOLERANCE:
        drop_count = 2
    elif abs(fps - 59.94) < _FPS_TOLERANCE:
        drop_count = 4
    else:
        return _ndf_timecode_to_frames(h, m, s, f, fps_int)

    total_minutes = h * 60 + m

    # Total frames without drops
    total_frames = (
        (h * 3600 + m * 60 + s) * fps_int + f
    )

    # Subtract dropped frames:
    # drop_count frames are dropped each minute except every 10th minute
    # Total drops = drop_count * (total_minutes - total_minutes // 10)
    drops = drop_count * (total_minutes - total_minutes // 10)
    total_frames -= drops

    return total_frames


def convert_timecode(
    timecode: str,
    source_fps: float,
    target_fps: float,
    source_df: bool = False,
    target_df: bool = False,
) -> str:
    """
    Convert a timecode from one frame rate / format to another.

    Args:
        timecode: Source timecode string.
        source_fps: Source frame rate.
        target_fps: Target frame rate.
        source_df: Source uses drop-frame.
        target_df: Target uses drop-frame.

    Returns:
        Converted timecode string.
    """
    # Convert source timecode to absolute time (seconds)
    source_frames = timecode_to_frames(timecode, source_fps, drop_frame=source_df)
    source_fps_rounded = _round_fps(float(source_fps))

    # Absolute time in seconds
    if source_df and _is_drop_frame_rate(source_fps_rounded):
        # For DF, use the nominal fps for time calculation
        seconds = source_frames / source_fps_rounded
    else:
        seconds = source_frames / round(source_fps_rounded)

    # Convert absolute time to target frame number
    target_fps_rounded = _round_fps(float(target_fps))
    if target_df and _is_drop_frame_rate(target_fps_rounded):
        target_frame = round(seconds * target_fps_rounded)
    else:
        target_frame = round(seconds * round(target_fps_rounded))

    return frames_to_timecode(target_frame, target_fps, drop_frame=target_df)
