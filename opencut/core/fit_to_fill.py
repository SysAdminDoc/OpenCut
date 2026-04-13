"""
OpenCut Smart Fit-to-Fill Module v1.0.0

Adjust video duration to match a target length:
- Uniform speed change (constant setpts)
- Eased speed ramp (ease-in/ease-out for natural feel)
- Auto mode: uniform for small changes, eased for larger adjustments
- Audio pitch-corrected via atempo chain

All FFmpeg-based, zero external dependencies.
"""

import logging
import os
import tempfile
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_to_fill(
    input_path: str,
    target_duration: float,
    method: str = "auto",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Fit video to a target duration by adjusting playback speed.

    Args:
        input_path: Path to source video.
        target_duration: Desired output duration in seconds.
        method: Speed adjustment method:
            - "uniform": constant speed change.
            - "eased": ease-in/ease-out speed ramp for natural feel.
            - "auto": uniform if <15% change, eased otherwise.
        output_path_override: Custom output path. Auto-generated if None.
        on_progress: Callback(percent, message).

    Returns:
        dict with output_path, actual_duration, speed_factor, method.
    """
    if target_duration <= 0:
        raise ValueError("target_duration must be positive")

    if method not in ("uniform", "eased", "auto"):
        raise ValueError(f"Unknown method '{method}'. Use: uniform, eased, auto")

    if on_progress:
        on_progress(5, "Analyzing source video...")

    info = get_video_info(input_path)
    source_duration = info["duration"]

    if source_duration <= 0:
        raise RuntimeError("Could not determine source video duration")

    # Calculate speed factor: how much faster we need to play
    speed_factor = source_duration / target_duration
    speed_factor = max(0.25, min(8.0, speed_factor))

    # Determine actual method
    change_pct = abs(speed_factor - 1.0)
    if method == "auto":
        actual_method = "uniform" if change_pct < 0.15 else "eased"
    else:
        actual_method = method

    out = output_path_override or output_path(input_path, "fit_to_fill")

    if on_progress:
        direction = "speed up" if speed_factor > 1.0 else "slow down"
        on_progress(10, f"Will {direction} by {speed_factor:.2f}x ({actual_method})...")

    if actual_method == "uniform":
        _apply_uniform(input_path, out, speed_factor, on_progress)
    else:
        _apply_eased(input_path, out, speed_factor, source_duration, on_progress)

    # Verify output duration
    out_info = get_video_info(out)
    actual_duration = out_info["duration"] if out_info["duration"] > 0 else target_duration

    if on_progress:
        on_progress(100, f"Fit to {target_duration:.1f}s (actual: {actual_duration:.1f}s)")

    return {
        "output_path": out,
        "actual_duration": round(actual_duration, 3),
        "target_duration": round(target_duration, 3),
        "speed_factor": round(speed_factor, 4),
        "method": actual_method,
        "source_duration": round(source_duration, 3),
    }


# ---------------------------------------------------------------------------
# Internal: Uniform Speed
# ---------------------------------------------------------------------------

def _apply_uniform(
    input_path: str, out: str, speed_factor: float,
    on_progress: Optional[Callable],
) -> None:
    """Apply constant speed change using setpts + atempo."""
    if on_progress:
        on_progress(20, f"Applying uniform {speed_factor:.2f}x speed...")

    vf = f"setpts={1.0/speed_factor:.6f}*PTS"
    af = _build_atempo_chain(speed_factor)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
    ]
    if af:
        cmd += ["-af", af]
    cmd += [
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        out,
    ]
    run_ffmpeg(cmd, timeout=7200)


# ---------------------------------------------------------------------------
# Internal: Eased Speed
# ---------------------------------------------------------------------------

def _apply_eased(
    input_path: str, out: str, speed_factor: float,
    source_duration: float, on_progress: Optional[Callable],
) -> None:
    """
    Apply ease-in/ease-out speed change by splitting into 3 segments:
    - First 15%: ramp from 1.0 to target speed
    - Middle 70%: target speed
    - Last 15%: ramp from target speed to 1.0

    Each segment is processed separately and concatenated.
    """
    if on_progress:
        on_progress(20, "Building eased speed ramp...")

    ramp_pct = 0.15
    ramp_dur = source_duration * ramp_pct
    mid_dur = source_duration * (1.0 - 2 * ramp_pct)

    # For ramp segments, use average of 1.0 and target speed
    ramp_speed = (1.0 + speed_factor) / 2.0
    ramp_speed = max(0.25, min(8.0, ramp_speed))
    mid_speed = max(0.25, min(8.0, speed_factor))

    tmp_dir = tempfile.mkdtemp(prefix="opencut_fit_")
    try:
        segments = []

        # Segment 1: ease-in (0 to ramp_dur)
        if ramp_dur > 0.1:
            seg1 = os.path.join(tmp_dir, "seg_ramp_in.mp4")
            _speed_segment(input_path, seg1, 0, ramp_dur, ramp_speed)
            segments.append(seg1)
            if on_progress:
                on_progress(35, "Ease-in segment done...")

        # Segment 2: middle (ramp_dur to ramp_dur + mid_dur)
        if mid_dur > 0.1:
            seg2 = os.path.join(tmp_dir, "seg_mid.mp4")
            _speed_segment(input_path, seg2, ramp_dur, ramp_dur + mid_dur, mid_speed)
            segments.append(seg2)
            if on_progress:
                on_progress(60, "Middle segment done...")

        # Segment 3: ease-out (ramp_dur + mid_dur to end)
        ease_out_start = ramp_dur + mid_dur
        if source_duration - ease_out_start > 0.1:
            seg3 = os.path.join(tmp_dir, "seg_ramp_out.mp4")
            _speed_segment(input_path, seg3, ease_out_start, source_duration, ramp_speed)
            segments.append(seg3)
            if on_progress:
                on_progress(80, "Ease-out segment done...")

        if not segments:
            raise RuntimeError("No segments produced for eased speed")

        # Concatenate
        if on_progress:
            on_progress(85, "Concatenating segments...")

        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy", out,
        ], timeout=7200)

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _speed_segment(
    input_path: str, output: str,
    start: float, end: float, speed: float,
) -> None:
    """Extract a time range and apply speed change."""
    vf = f"setpts={1.0/speed:.6f}*PTS"
    af = _build_atempo_chain(speed)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", input_path,
        "-vf", vf,
    ]
    if af:
        cmd += ["-af", af]
    cmd += [
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output,
    ]
    run_ffmpeg(cmd, timeout=7200)


# ---------------------------------------------------------------------------
# Internal: atempo chain
# ---------------------------------------------------------------------------

def _build_atempo_chain(speed: float) -> str:
    """Build FFmpeg atempo filter chain. atempo supports 0.5-100.0 per stage."""
    if abs(speed - 1.0) < 0.001:
        return ""

    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.001:
        parts.append(f"atempo={remaining:.6f}")

    return ",".join(parts) if parts else ""
