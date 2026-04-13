"""
OpenCut Slide Change Detection (Feature 33.1)

Detect slide transitions in screen recordings, extract slide images,
and generate chapter markers.

Uses FFmpeg scene detection and frame extraction.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SlideChange:
    """A detected slide transition."""
    timestamp: float       # seconds into the video
    frame_number: int      # frame index
    score: float           # scene change score (0.0 - 1.0)
    slide_index: int = 0   # sequential slide number

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_number": self.frame_number,
            "score": round(self.score, 4),
            "slide_index": self.slide_index,
        }


@dataclass
class SlideChapter:
    """A chapter marker derived from slide changes."""
    title: str
    start_time: float
    end_time: float
    slide_index: int

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "slide_index": self.slide_index,
        }


@dataclass
class SlideDetectionResult:
    """Result of slide change detection."""
    changes: List[dict]
    slide_count: int
    duration: float
    threshold_used: float

    def to_dict(self) -> dict:
        return {
            "changes": self.changes,
            "slide_count": self.slide_count,
            "duration": self.duration,
            "threshold_used": self.threshold_used,
        }


@dataclass
class SlideExtractionResult:
    """Result of extracting slide images."""
    output_dir: str
    image_paths: List[str]
    slide_count: int

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "image_paths": self.image_paths,
            "slide_count": self.slide_count,
        }


# ---------------------------------------------------------------------------
# Scene change detection via FFprobe
# ---------------------------------------------------------------------------

def _detect_scenes_ffprobe(
    video_path: str,
    threshold: float = 0.3,
) -> List[dict]:
    """Use FFprobe with scene detection to find frame-level changes.

    FFprobe lavfi input with select filter detects scene changes and
    outputs frame timestamps where the scene score exceeds threshold.
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-f", "lavfi",
        f"movie={video_path.replace(os.sep, '/')},select=gt(scene\\,{threshold})",
        "-show_entries", "frame=pts_time,pkt_pts_time",
        "-show_entries", "frame_tags=lavfi.scene_score",
        "-of", "json",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        # Fallback: use simpler approach
        return _detect_scenes_ffmpeg_log(video_path, threshold)

    try:
        data = json.loads(result.stdout.decode())
        frames = data.get("frames", [])
        scenes = []
        for f in frames:
            ts = float(f.get("pts_time", f.get("pkt_pts_time", 0)))
            tags = f.get("tags", {})
            score = float(tags.get("lavfi.scene_score", 0))
            scenes.append({"timestamp": ts, "score": score})
        return scenes
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse FFprobe scene output: %s", e)
        return _detect_scenes_ffmpeg_log(video_path, threshold)


def _detect_scenes_ffmpeg_log(
    video_path: str,
    threshold: float = 0.3,
) -> List[dict]:
    """Fallback scene detection using FFmpeg select filter with metadata logging.

    Runs FFmpeg with the select filter and parses stdout metadata lines
    to find scene change timestamps.
    """
    # Use FFmpeg to write scene scores to a temp metadata file
    cmd = [
        get_ffmpeg_path(),
        "-hide_banner", "-y",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',metadata=print",
        "-an", "-f", "null",
        os.devnull,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    stderr = result.stderr

    scenes = []
    current_ts = None
    for line in stderr.splitlines():
        line = line.strip()
        if "pts_time:" in line:
            try:
                current_ts = float(line.split("pts_time:")[1].strip())
            except (ValueError, IndexError):
                current_ts = None
        elif "lavfi.scene_score=" in line and current_ts is not None:
            try:
                score = float(line.split("=")[1].strip())
                scenes.append({"timestamp": current_ts, "score": score})
            except (ValueError, IndexError):
                pass
            current_ts = None

    return scenes


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_slide_changes(
    video_path: str,
    threshold: float = 0.3,
    min_interval: float = 1.0,
    max_slides: int = 200,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Detect slide transitions in a screen recording.

    Uses FFmpeg/FFprobe scene change detection to find frames where
    content changes significantly, indicating a slide transition.

    Args:
        video_path: Source video file path.
        threshold: Scene change threshold (0.0 - 1.0). Lower = more
                   sensitive. Default 0.3 works well for presentations.
        min_interval: Minimum seconds between detected changes (debounce).
        max_slides: Maximum number of slides to detect.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with changes, slide_count, duration, threshold_used.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If threshold is out of range.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(video_path)
    duration = info["duration"]
    fps = info["fps"]

    if on_progress:
        on_progress(10, "Detecting scene changes...")

    raw_scenes = _detect_scenes_ffprobe(video_path, threshold)

    if on_progress:
        on_progress(70, f"Found {len(raw_scenes)} raw scene changes, filtering...")

    # Filter: enforce minimum interval and max count
    filtered = []
    last_ts = -min_interval  # allow first detection at t=0

    # Always include t=0 as the first slide
    filtered.append(SlideChange(
        timestamp=0.0,
        frame_number=0,
        score=1.0,
        slide_index=0,
    ))

    for scene in sorted(raw_scenes, key=lambda s: s["timestamp"]):
        ts = scene["timestamp"]
        score = scene.get("score", threshold)

        if ts - last_ts < min_interval:
            continue
        if ts <= 0:
            continue

        slide_idx = len(filtered)
        if slide_idx >= max_slides:
            break

        filtered.append(SlideChange(
            timestamp=ts,
            frame_number=int(ts * fps),
            score=score,
            slide_index=slide_idx,
        ))
        last_ts = ts

    if on_progress:
        on_progress(90, f"Detected {len(filtered)} slides")

    result = SlideDetectionResult(
        changes=[c.to_dict() for c in filtered],
        slide_count=len(filtered),
        duration=duration,
        threshold_used=threshold,
    )

    if on_progress:
        on_progress(100, "Done")

    return result.to_dict()


# ---------------------------------------------------------------------------
# Slide image extraction
# ---------------------------------------------------------------------------

def extract_slide_images(
    video_path: str,
    timestamps: List[float],
    output_dir: str,
    image_format: str = "png",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract slide images at specific timestamps.

    Args:
        video_path: Source video file path.
        timestamps: List of timestamps (seconds) to extract frames at.
        output_dir: Directory to save extracted images.
        image_format: Output image format ("png" or "jpg").
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_dir, image_paths, slide_count.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If timestamps is empty.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not timestamps:
        raise ValueError("No timestamps provided")
    if image_format not in ("png", "jpg", "jpeg"):
        raise ValueError(f"Unsupported image format: {image_format}")

    os.makedirs(output_dir, exist_ok=True)

    sorted_ts = sorted(timestamps)
    n = len(sorted_ts)
    image_paths = []

    for i, ts in enumerate(sorted_ts):
        if on_progress:
            pct = 10 + int((i / n) * 85)
            on_progress(pct, f"Extracting slide {i + 1}/{n}...")

        ext = "jpg" if image_format in ("jpg", "jpeg") else "png"
        out_file = os.path.join(output_dir, f"slide_{i:04d}.{ext}")

        cmd = (
            FFmpegCmd()
            .pre_input("ss", f"{ts:.3f}")
            .input(video_path)
            .frames(1)
            .option("q:v", "2")
            .output(out_file)
            .build()
        )

        try:
            run_ffmpeg(cmd, timeout=30)
            if os.path.isfile(out_file):
                image_paths.append(out_file)
        except RuntimeError as e:
            logger.warning("Failed to extract frame at t=%.3f: %s", ts, e)

    if on_progress:
        on_progress(100, "Done")

    result = SlideExtractionResult(
        output_dir=output_dir,
        image_paths=image_paths,
        slide_count=len(image_paths),
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Chapter marker generation
# ---------------------------------------------------------------------------

def generate_slide_chapters(
    timestamps: List[float],
    video_duration: float = 0,
    title_prefix: str = "Slide",
    custom_titles: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Generate chapter markers from slide change timestamps.

    Args:
        timestamps: List of slide change timestamps (seconds).
        video_duration: Total video duration (used for last chapter end time).
        title_prefix: Prefix for auto-generated chapter titles.
        custom_titles: Optional list of custom titles (same length as timestamps).
        on_progress: Progress callback(pct, msg).

    Returns:
        List of chapter dicts with title, start_time, end_time, slide_index.

    Raises:
        ValueError: If timestamps is empty.
    """
    if not timestamps:
        raise ValueError("No timestamps provided")

    sorted_ts = sorted(timestamps)

    if custom_titles and len(custom_titles) != len(sorted_ts):
        raise ValueError(
            f"custom_titles length ({len(custom_titles)}) must match "
            f"timestamps length ({len(sorted_ts)})"
        )

    chapters = []
    n = len(sorted_ts)

    for i, ts in enumerate(sorted_ts):
        if on_progress:
            pct = int((i / n) * 100)
            on_progress(pct, f"Creating chapter {i + 1}/{n}...")

        # Title
        if custom_titles:
            title = custom_titles[i]
        else:
            title = f"{title_prefix} {i + 1}"

        # End time: start of next slide, or video duration
        if i + 1 < n:
            end_time = sorted_ts[i + 1]
        elif video_duration > 0:
            end_time = video_duration
        else:
            end_time = ts + 60.0  # default 60s for last chapter

        chapters.append(SlideChapter(
            title=title,
            start_time=ts,
            end_time=end_time,
            slide_index=i,
        ).to_dict())

    if on_progress:
        on_progress(100, "Done")

    return chapters
