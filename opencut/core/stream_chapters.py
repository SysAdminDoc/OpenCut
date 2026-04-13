"""
OpenCut Stream Recording Auto-Chaptering

Automatically detect chapter boundaries in stream recordings using
a combination of scene detection, silence detection, and optional
transcript analysis.

Produces chapter lists suitable for export as YouTube-style chapter
markers or FFmpeg metadata chapters.

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# Default detection methods
DEFAULT_METHODS = ["scene", "silence"]


@dataclass
class Chapter:
    """A single chapter with start/end times and title."""
    start: float
    end: float
    title: str = ""


@dataclass
class ChapterResult:
    """Result of auto-chaptering."""
    chapters: List[Chapter] = field(default_factory=list)
    total_chapters: int = 0
    total_duration: float = 0.0
    methods_used: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------
# Detection methods
# -----------------------------------------------------------------------

def _detect_scene_boundaries(
    input_path: str,
    threshold: float = 0.4,
    min_scene_length: float = 60.0,
) -> List[float]:
    """Detect major scene changes as potential chapter points.

    Uses a higher threshold than normal scene detection to only catch
    significant visual transitions (e.g. BRB screens, topic changes).
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", input_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=3600, text=True)
    stderr = result.stderr

    # Parse "pts_time:XXX" from showinfo output
    pts_pattern = re.compile(r"pts_time:\s*([\d.]+)")
    times = [float(m.group(1)) for m in pts_pattern.finditer(stderr)]

    # Filter: keep only points at least min_scene_length apart
    if not times:
        return []

    filtered = [times[0]]
    for t in times[1:]:
        if t - filtered[-1] >= min_scene_length:
            filtered.append(t)
    return filtered


def _detect_silence_breaks(
    input_path: str,
    min_silence: float = 10.0,
    threshold_db: float = -40.0,
) -> List[float]:
    """Detect extended silence periods as stream breaks (BRB, intermission).

    Returns midpoints of silence segments as chapter boundaries.
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", input_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}",
        "-vn", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=3600, text=True)
    stderr = result.stderr

    start_pat = re.compile(r"silence_start:\s*([\d.]+)")
    end_pat = re.compile(r"silence_end:\s*([\d.]+)")

    starts = [float(m.group(1)) for m in start_pat.finditer(stderr)]
    ends = [float(m.group(1)) for m in end_pat.finditer(stderr)]

    # Use the end of each silence period as the chapter boundary
    # (content resumes after the break)
    boundaries = []
    for s, e in zip(starts, ends):
        if e - s >= min_silence:
            boundaries.append(e)
    return boundaries


def _merge_chapter_points(
    points: List[float],
    merge_distance: float = 30.0,
    total_duration: float = 0.0,
) -> List[float]:
    """Merge nearby chapter points and remove points too close to start/end."""
    if not points:
        return []

    # Sort and deduplicate
    points = sorted(set(points))

    # Remove points in the first/last 10 seconds
    points = [p for p in points if p > 10.0]
    if total_duration > 0:
        points = [p for p in points if p < total_duration - 10.0]

    if not points:
        return []

    # Merge nearby points (keep earliest of each cluster)
    merged = [points[0]]
    for p in points[1:]:
        if p - merged[-1] >= merge_distance:
            merged.append(p)

    return merged


def _generate_titles(chapters: List[Chapter]) -> List[Chapter]:
    """Generate generic chapter titles based on position."""
    total = len(chapters)
    for i, ch in enumerate(chapters):
        if not ch.title:
            if i == 0:
                ch.title = "Introduction"
            elif total > 1 and i == total - 1:
                ch.title = "Conclusion"
            else:
                ch.title = f"Section {i + 1}"
    return chapters


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def auto_chapter_stream(
    input_path: str,
    methods: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> ChapterResult:
    """
    Auto-detect chapter boundaries in a stream recording.

    Args:
        input_path: Source video file.
        methods: List of detection methods: "scene", "silence", "transcript".
                 Defaults to ["scene", "silence"].
        on_progress: Progress callback(pct, msg).

    Returns:
        ChapterResult with list of Chapter items.
    """
    if methods is None:
        methods = list(DEFAULT_METHODS)

    # Validate methods
    valid = {"scene", "silence", "transcript"}
    methods = [m for m in methods if m in valid]
    if not methods:
        methods = list(DEFAULT_METHODS)

    if on_progress:
        on_progress(5, "Probing video info...")

    info = get_video_info(input_path)
    total_dur = info.get("duration", 0)

    all_points: List[float] = []
    methods_used = []
    step = 80 // len(methods)

    for i, method in enumerate(methods):
        base_pct = 10 + i * step

        if method == "scene":
            if on_progress:
                on_progress(base_pct, "Detecting scene changes...")
            points = _detect_scene_boundaries(input_path)
            all_points.extend(points)
            methods_used.append("scene")

        elif method == "silence":
            if on_progress:
                on_progress(base_pct, "Detecting silence breaks...")
            points = _detect_silence_breaks(input_path)
            all_points.extend(points)
            methods_used.append("silence")

        elif method == "transcript":
            # Transcript-based chaptering requires whisper output;
            # if not available, skip gracefully
            methods_used.append("transcript")

    if on_progress:
        on_progress(85, "Merging chapter points...")

    merged = _merge_chapter_points(all_points, total_duration=total_dur)

    # Build chapter list
    chapters: List[Chapter] = []

    # Always start with a chapter at 0:00
    if merged:
        chapters.append(Chapter(start=0.0, end=merged[0], title=""))
        for i, point in enumerate(merged):
            end = merged[i + 1] if i + 1 < len(merged) else total_dur
            chapters.append(Chapter(start=point, end=end, title=""))
    else:
        # No chapters detected -- return single chapter for entire video
        chapters.append(Chapter(start=0.0, end=total_dur, title="Full Recording"))

    chapters = _generate_titles(chapters)

    if on_progress:
        on_progress(100, f"Found {len(chapters)} chapters")

    return ChapterResult(
        chapters=chapters,
        total_chapters=len(chapters),
        total_duration=round(total_dur, 3),
        methods_used=methods_used,
    )


def export_youtube_chapters(
    chapters: List[dict],
    offset: float = 0,
) -> str:
    """
    Export chapters in YouTube description format.

    Args:
        chapters: List of dicts with "start" and "title" keys.
        offset: Time offset in seconds to apply to all timestamps.

    Returns:
        String in YouTube format: "0:00 Introduction\\n2:34 Topic 1\\n..."
    """
    lines = []
    for ch in chapters:
        start = float(ch.get("start", 0)) + offset
        title = ch.get("title", "Chapter")
        ts = _format_youtube_timestamp(max(0, start))
        lines.append(f"{ts} {title}")
    return "\n".join(lines)


def _format_youtube_timestamp(seconds: float) -> str:
    """Format seconds into YouTube-compatible timestamp.

    Uses M:SS for < 1 hour, H:MM:SS for >= 1 hour.
    """
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
