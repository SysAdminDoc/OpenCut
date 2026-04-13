"""
Smart Render / Partial Re-Encode.

Re-encode only changed segments and stream-copy unchanged GOPs
for dramatically faster exports on incremental edits.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class ChangedSegment:
    """A segment of video that needs re-encoding."""
    start: float
    end: float
    change_type: str = ""     # modified | inserted | deleted | effect
    description: str = ""


@dataclass
class SmartRenderPlan:
    """Plan for smart rendering a video."""
    total_duration: float
    changed_segments: List[ChangedSegment] = field(default_factory=list)
    copy_segments: List[Tuple[float, float]] = field(default_factory=list)
    encode_duration: float = 0.0
    copy_duration: float = 0.0
    estimated_speedup: float = 1.0


def _get_keyframes(video_path: str) -> List[float]:
    """Extract keyframe timestamps from a video."""
    cmd = [
        get_ffprobe_path(),
        "-loglevel", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    keyframes = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                keyframes.append(float(parts[0]))
            except (ValueError, IndexError):
                pass
    return sorted(keyframes)


def _snap_to_keyframe(timestamp: float, keyframes: List[float], direction: str = "before") -> float:
    """Snap a timestamp to the nearest keyframe."""
    if not keyframes:
        return timestamp
    if direction == "before":
        candidates = [k for k in keyframes if k <= timestamp]
        return max(candidates) if candidates else keyframes[0]
    else:
        candidates = [k for k in keyframes if k >= timestamp]
        return min(candidates) if candidates else keyframes[-1]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def detect_changed_segments(
    video_path: str,
    edit_history: List[Dict],
    on_progress: Optional[Callable] = None,
) -> SmartRenderPlan:
    """Analyze edit history to determine which segments need re-encoding.

    Args:
        video_path: Path to the source video.
        edit_history: List of edit operations, each with 'type', 'start', 'end'.

    Returns:
        SmartRenderPlan with changed and copy segments.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not edit_history:
        raise ValueError("Edit history cannot be empty")

    if on_progress:
        on_progress(10, "Analyzing video structure")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        raise ValueError("Could not determine video duration")

    if on_progress:
        on_progress(30, "Extracting keyframes")

    keyframes = _get_keyframes(video_path)

    if on_progress:
        on_progress(50, "Computing changed segments")

    # Parse edit history into changed segments
    changed: List[ChangedSegment] = []
    for edit in edit_history:
        start = float(edit.get("start", 0))
        end = float(edit.get("end", start + 1))
        change_type = edit.get("type", "modified")
        description = edit.get("description", "")

        # Snap to keyframe boundaries for clean cuts
        snap_start = _snap_to_keyframe(start, keyframes, "before")
        snap_end = _snap_to_keyframe(end, keyframes, "after")

        changed.append(ChangedSegment(
            start=snap_start,
            end=snap_end,
            change_type=change_type,
            description=description,
        ))

    # Merge overlapping segments
    changed.sort(key=lambda s: s.start)
    merged = []
    for seg in changed:
        if merged and seg.start <= merged[-1].end:
            merged[-1].end = max(merged[-1].end, seg.end)
        else:
            merged.append(seg)

    if on_progress:
        on_progress(70, "Computing copy segments")

    # Determine copy segments (gaps between changed segments)
    copy_segments = []
    prev_end = 0.0
    for seg in merged:
        if seg.start > prev_end:
            copy_segments.append((prev_end, seg.start))
        prev_end = seg.end
    if prev_end < duration:
        copy_segments.append((prev_end, duration))

    encode_dur = sum(s.end - s.start for s in merged)
    copy_dur = sum(e - s for s, e in copy_segments)
    speedup = duration / encode_dur if encode_dur > 0 else float("inf")

    if on_progress:
        on_progress(100, "Analysis complete")

    return SmartRenderPlan(
        total_duration=duration,
        changed_segments=merged,
        copy_segments=copy_segments,
        encode_duration=encode_dur,
        copy_duration=copy_dur,
        estimated_speedup=round(speedup, 2),
    )


def smart_render(
    video_path: str,
    changes: List[Dict],
    output_path_str: Optional[str] = None,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Smart render a video, re-encoding only changed segments.

    Args:
        video_path: Path to the source video.
        changes: List of change dicts with 'start', 'end', 'type'.
        output_path_str: Output file path (auto-generated if None).
        codec: Video codec for re-encoded segments.
        crf: Quality factor for re-encoded segments.
        preset: Encoding speed preset.

    Returns:
        Dict with output_path, encode_time, copy_time, total_time.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Planning smart render")

    plan = detect_changed_segments(video_path, changes)

    if output_path_str is None:
        output_path_str = output_path(video_path, "_smart")

    if on_progress:
        on_progress(15, f"Rendering {len(plan.changed_segments)} segments")

    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="opencut_smart_")
    segment_files = []

    total_segs = len(plan.copy_segments) + len(plan.changed_segments)
    all_segments = []

    # Create ordered list of all segments
    for seg in plan.copy_segments:
        all_segments.append(("copy", seg[0], seg[1]))
    for seg in plan.changed_segments:
        all_segments.append(("encode", seg.start, seg.end))
    all_segments.sort(key=lambda s: s[1])

    for i, (mode, start, end) in enumerate(all_segments):
        seg_file = os.path.join(temp_dir, f"seg_{i:04d}.ts")

        if mode == "copy":
            # Stream copy unchanged segment
            cmd = (
                FFmpegCmd()
                .input(video_path)
                .seek(start=str(start), end=str(end))
                .copy_streams()
                .option("avoid_negative_ts", "make_zero")
                .format("mpegts")
                .output(seg_file)
                .build()
            )
        else:
            # Re-encode changed segment
            cmd = (
                FFmpegCmd()
                .input(video_path)
                .seek(start=str(start), end=str(end))
                .video_codec(codec, crf=crf, preset=preset)
                .audio_codec("aac", bitrate="192k")
                .format("mpegts")
                .output(seg_file)
                .build()
            )

        run_ffmpeg(cmd, timeout=600)
        segment_files.append(seg_file)

        if on_progress:
            pct = 15 + int(70 * (i + 1) / max(total_segs, 1))
            on_progress(pct, f"Segment {i+1}/{total_segs}")

    if on_progress:
        on_progress(90, "Concatenating segments")

    # Build concat file
    concat_file = os.path.join(temp_dir, "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as fh:
        for sf in segment_files:
            fh.write(f"file '{sf}'\n")

    # Concatenate
    cmd = (
        FFmpegCmd()
        .pre_input("-f", "concat")
        .pre_input("-safe", "0")
        .input(concat_file)
        .copy_streams()
        .faststart()
        .output(output_path_str)
        .build()
    )
    run_ffmpeg(cmd, timeout=300)

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    if on_progress:
        on_progress(100, "Smart render complete")

    file_size = os.path.getsize(output_path_str) if os.path.exists(output_path_str) else 0

    logger.info(
        "Smart render: %d segments, %.1fx speedup, output=%s",
        len(all_segments), plan.estimated_speedup, output_path_str,
    )

    return {
        "output_path": output_path_str,
        "file_size": file_size,
        "total_segments": len(all_segments),
        "encoded_segments": len(plan.changed_segments),
        "copied_segments": len(plan.copy_segments),
        "encode_duration": plan.encode_duration,
        "copy_duration": plan.copy_duration,
        "estimated_speedup": plan.estimated_speedup,
    }


def estimate_smart_render_savings(
    video_path: str,
    changes: List[Dict],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Estimate time and space savings from smart rendering.

    Args:
        video_path: Path to the source video.
        changes: List of change dicts.

    Returns:
        Dict with estimated savings percentages and durations.
    """
    plan = detect_changed_segments(video_path, changes, on_progress=on_progress)

    pct_encode = (plan.encode_duration / plan.total_duration * 100) if plan.total_duration > 0 else 100
    pct_copy = 100 - pct_encode

    return {
        "total_duration": plan.total_duration,
        "encode_duration": plan.encode_duration,
        "copy_duration": plan.copy_duration,
        "percent_reencoded": round(pct_encode, 1),
        "percent_copied": round(pct_copy, 1),
        "estimated_speedup": plan.estimated_speedup,
        "changed_segment_count": len(plan.changed_segments),
        "copy_segment_count": len(plan.copy_segments),
    }
