"""
OpenCut Dead-Time Detection & Speed Ramp

Detects "dead time" segments where BOTH visual motion and audio are
inactive, then optionally speed-ramps those segments to compress
dead time while preserving normal-paced content.

Uses FFmpeg mpdecimate (motion) + silencedetect (audio) -- no
additional dependencies required.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class DeadSegment:
    """A single dead-time segment."""
    start: float
    end: float
    duration: float
    motion_score: float = 0.0


@dataclass
class DeadTimeResult:
    """Result of dead-time detection."""
    segments: List[DeadSegment] = field(default_factory=list)
    total_dead_time: float = 0.0
    total_duration: float = 0.0
    dead_percentage: float = 0.0


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _detect_low_motion_segments(
    input_path: str,
    threshold: float = 0.001,
    min_duration: float = 3.0,
    fps: float = 30.0,
) -> List[tuple]:
    """Return list of (start, end) for low-motion windows via mpdecimate."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", input_path,
        "-vf", f"mpdecimate=hi={int(threshold * 64000)}:lo={int(threshold * 32000)}:frac=0.5",
        "-loglevel", "debug",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, timeout=3600, text=True,
    )
    stderr = result.stderr

    # Parse "drop pts:XXXXX" lines to find dropped (static) frames
    drop_pattern = re.compile(r"drop.*pts:\s*(\d+).*pts_time:([\d.]+)")
    dropped_times: List[float] = []
    for m in drop_pattern.finditer(stderr):
        dropped_times.append(float(m.group(2)))

    if not dropped_times:
        return []

    # Merge consecutive dropped frames into contiguous segments
    frame_gap = 2.0 / fps  # allow small gaps between frames
    segments = []
    seg_start = dropped_times[0]
    seg_end = dropped_times[0]

    for t in dropped_times[1:]:
        if t - seg_end <= frame_gap:
            seg_end = t
        else:
            dur = seg_end - seg_start
            if dur >= min_duration:
                segments.append((seg_start, seg_end))
            seg_start = t
            seg_end = t

    # Final segment
    dur = seg_end - seg_start
    if dur >= min_duration:
        segments.append((seg_start, seg_end))

    return segments


def _detect_silence_segments(
    input_path: str,
    threshold_db: float = -30.0,
    min_duration: float = 3.0,
) -> List[tuple]:
    """Return list of (start, end) for silent segments via silencedetect."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", input_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-vn", "-f", "null", "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, timeout=3600, text=True,
    )
    stderr = result.stderr

    start_pat = re.compile(r"silence_start:\s*([\d.]+)")
    end_pat = re.compile(r"silence_end:\s*([\d.]+)")

    starts = [float(m.group(1)) for m in start_pat.finditer(stderr)]
    ends = [float(m.group(1)) for m in end_pat.finditer(stderr)]

    segments = []
    for s, e in zip(starts, ends):
        if e - s >= min_duration:
            segments.append((s, e))
    return segments


def _intersect_segments(
    motion_segs: List[tuple],
    silence_segs: List[tuple],
    min_duration: float = 3.0,
) -> List[tuple]:
    """Return the intersection of two sorted segment lists."""
    result = []
    j = 0
    for ms, me in motion_segs:
        while j < len(silence_segs) and silence_segs[j][1] <= ms:
            j += 1
        k = j
        while k < len(silence_segs) and silence_segs[k][0] < me:
            ss, se = silence_segs[k]
            overlap_start = max(ms, ss)
            overlap_end = min(me, se)
            if overlap_end - overlap_start >= min_duration:
                result.append((overlap_start, overlap_end))
            k += 1
    return result


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def detect_dead_time(
    input_path: str,
    motion_threshold: float = 0.001,
    min_duration: float = 3.0,
    on_progress: Optional[Callable] = None,
) -> DeadTimeResult:
    """
    Detect dead-time segments where both motion and audio are inactive.

    Args:
        input_path: Source video file.
        motion_threshold: Sensitivity for motion detection (lower = stricter).
        min_duration: Minimum dead segment duration in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        DeadTimeResult with list of DeadSegment items.
    """
    if on_progress:
        on_progress(5, "Analyzing motion...")

    info = get_video_info(input_path)
    total_dur = info.get("duration", 0)
    fps = info.get("fps", 30.0)

    motion_segs = _detect_low_motion_segments(
        input_path, threshold=motion_threshold, min_duration=min_duration, fps=fps,
    )

    if on_progress:
        on_progress(40, "Analyzing audio silence...")

    silence_segs = _detect_silence_segments(
        input_path, min_duration=min_duration,
    )

    if on_progress:
        on_progress(70, "Intersecting motion and silence segments...")

    dead_segs = _intersect_segments(motion_segs, silence_segs, min_duration)

    segments = []
    total_dead = 0.0
    for s, e in dead_segs:
        dur = e - s
        total_dead += dur
        segments.append(DeadSegment(start=s, end=e, duration=dur, motion_score=0.0))

    pct = (total_dead / total_dur * 100) if total_dur > 0 else 0.0

    if on_progress:
        on_progress(100, f"Found {len(segments)} dead segments ({pct:.1f}%)")

    return DeadTimeResult(
        segments=segments,
        total_dead_time=round(total_dead, 3),
        total_duration=round(total_dur, 3),
        dead_percentage=round(pct, 2),
    )


def speed_ramp_dead_time(
    input_path: str,
    dead_segments: List[dict],
    speed_factor: float = 8.0,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Speed up dead-time segments while keeping normal segments at 1x.

    Args:
        input_path: Source video file.
        dead_segments: List of dicts with "start" and "end" keys (seconds).
        speed_factor: Speed multiplier for dead segments (e.g. 8.0 = 8x).
        output_path_override: Explicit output; auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with "output_path", "segments_ramped", "time_saved".
    """
    if not dead_segments:
        raise ValueError("No dead segments provided")

    speed_factor = max(1.0, min(speed_factor, 100.0))

    if on_progress:
        on_progress(5, "Building speed ramp filter...")

    info = get_video_info(input_path)
    total_dur = info.get("duration", 0)

    # Sort segments by start time
    segs = sorted(dead_segments, key=lambda s: s["start"])

    # Build segment list for ffmpeg concat approach
    # We split the video into chunks, speed up dead ones, and concat
    out_ext = os.path.splitext(input_path)[1] or ".mp4"
    if output_path_override:
        out = output_path_override
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        out = os.path.join(
            os.path.dirname(input_path),
            f"{base}_speedramped{out_ext}",
        )

    # Build filter_complex with setpts for speed changes
    # Strategy: use trim + setpts + atempo for each segment, then concat
    parts = []
    current = 0.0
    part_idx = 0

    for seg in segs:
        start = seg["start"]
        end = seg["end"]

        # Normal segment before dead time
        if start > current:
            parts.append({"start": current, "end": start, "speed": 1.0, "idx": part_idx})
            part_idx += 1

        # Dead segment (sped up)
        parts.append({"start": start, "end": end, "speed": speed_factor, "idx": part_idx})
        part_idx += 1
        current = end

    # Trailing normal segment
    if current < total_dur:
        parts.append({"start": current, "end": total_dur, "speed": 1.0, "idx": part_idx})
        part_idx += 1

    if not parts:
        raise ValueError("No segments to process")

    if on_progress:
        on_progress(20, f"Processing {len(parts)} segments...")

    # Build filter_complex string
    fc_parts = []
    concat_inputs_v = []
    concat_inputs_a = []

    for p in parts:
        i = p["idx"]
        s = p["speed"]
        fc_parts.append(
            f"[0:v]trim=start={p['start']}:end={p['end']},"
            f"setpts=(PTS-STARTPTS)/{s}[v{i}]"
        )
        # Audio: atempo only supports 0.5-100.0
        # For speeds > 2.0, chain multiple atempo filters
        atempo_chain = _build_atempo_chain(s)
        fc_parts.append(
            f"[0:a]atrim=start={p['start']}:end={p['end']},"
            f"asetpts=PTS-STARTPTS,{atempo_chain}[a{i}]"
        )
        concat_inputs_v.append(f"[v{i}]")
        concat_inputs_a.append(f"[a{i}]")

    n = len(parts)
    concat_in = "".join(f"[v{p['idx']}][a{p['idx']}]" for p in parts)
    fc_parts.append(f"{concat_in}concat=n={n}:v=1:a=1[outv][outa]")

    filter_complex = ";\n".join(fc_parts)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out,
    ]

    if on_progress:
        on_progress(30, "Encoding speed-ramped video...")

    run_ffmpeg(cmd, timeout=7200)

    # Calculate time saved
    time_saved = sum(
        (s["end"] - s["start"]) * (1.0 - 1.0 / speed_factor)
        for s in segs
    )

    if on_progress:
        on_progress(100, "Speed ramp complete")

    return {
        "output_path": out,
        "segments_ramped": len(segs),
        "time_saved": round(time_saved, 2),
        "speed_factor": speed_factor,
    }


def _build_atempo_chain(speed: float) -> str:
    """Build chained atempo filters for arbitrary speed values.

    FFmpeg atempo supports 0.5 to 100.0, but for accuracy we chain
    multiple atempo filters when speed > 2.0.
    """
    if speed <= 1.0:
        return "atempo=1.0"

    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)
