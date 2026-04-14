"""
OpenCut AI Scene Edit Detection Module

Detect cuts in pre-edited footage -- hard cuts, dissolves, fades.
Adobe added this to Premiere natively. Useful for conforming received footage.

Uses FFmpeg scdet filter for scene change detection and frame-to-frame
difference analysis for cut type classification.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class EditPoint:
    """A detected edit point in the footage."""
    timestamp: float = 0.0
    frame_number: int = 0
    confidence: float = 0.0
    type: str = "hard_cut"  # "hard_cut", "dissolve", "fade"

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_number": self.frame_number,
            "confidence": round(self.confidence, 4),
            "type": self.type,
        }


@dataclass
class EditDetectionResult:
    """Complete edit detection results."""
    cuts: List[EditPoint] = field(default_factory=list)
    total_scenes: int = 0
    avg_scene_duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cuts": [c.to_dict() for c in self.cuts],
            "total_scenes": self.total_scenes,
            "avg_scene_duration": round(self.avg_scene_duration, 3),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception as exc:
        logger.debug("Duration probe failed: %s", exc)
    return 0.0


def _get_fps(video_path: str) -> float:
    """Get video FPS."""
    info = get_video_info(video_path)
    return info.get("fps", 30.0)


# ---------------------------------------------------------------------------
# Frame difference scoring
# ---------------------------------------------------------------------------
def _compute_frame_diffs(video_path: str, on_progress: Optional[Callable] = None) -> List[dict]:
    """Compute frame-to-frame difference scores using FFmpeg.

    Returns list of {time, score} dicts for each frame transition.
    """
    ffmpeg = get_ffmpeg_path()

    # Use scdet filter which outputs scene change scores
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vf", "scdet=s=1:t=0",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=max(300, int(_get_duration(video_path) * 3)),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Frame analysis timed out")

    diffs = []
    # Parse scdet output: lavfi.scd.score=X, lavfi.scd.time=Y
    score_pattern = re.compile(r"lavfi\.scd\.score=(\d+\.?\d*)")
    time_pattern = re.compile(r"lavfi\.scd\.time=(\d+\.?\d*)")

    lines = result.stderr.split("\n")
    for line in lines:
        score_match = score_pattern.search(line)
        time_match = time_pattern.search(line)

        if score_match and time_match:
            score = float(score_match.group(1))
            timestamp = float(time_match.group(1))
            diffs.append({"time": timestamp, "score": score / 100.0})

    return diffs


def _detect_with_scdet(video_path: str, threshold: float,
                       on_progress: Optional[Callable] = None) -> List[dict]:
    """Detect scene changes using FFmpeg scdet filter.

    Returns list of {time, score} dicts for detected changes.
    """
    ffmpeg = get_ffmpeg_path()
    threshold_pct = max(1.0, min(99.0, threshold * 100.0))

    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vf", f"scdet=s=1:t={threshold_pct:.1f}",
        "-f", "null", "-",
    ]

    duration = _get_duration(video_path)
    timeout = max(300, int(duration * 3) + 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Scene detection timed out")

    detections = []
    score_pattern = re.compile(r"lavfi\.scd\.score=(\d+\.?\d*)")
    time_pattern = re.compile(r"lavfi\.scd\.time=(\d+\.?\d*)")

    for line in result.stderr.split("\n"):
        score_match = score_pattern.search(line)
        time_match = time_pattern.search(line)

        if score_match and time_match:
            score = float(score_match.group(1)) / 100.0
            timestamp = float(time_match.group(1))
            detections.append({"time": timestamp, "score": score})

    return detections


# ---------------------------------------------------------------------------
# Cut type classification
# ---------------------------------------------------------------------------
def _classify_cut_type(score: float, nearby_scores: List[float],
                       prev_black: bool = False, next_black: bool = False) -> str:
    """Classify the type of edit based on score characteristics.

    - hard_cut: instant high delta (score > 0.7 with no gradual ramp)
    - dissolve: gradual ramp in scores around the cut point
    - fade: transition to/from black frames
    """
    # Fade detection: if frames near the cut are very dark (to/from black)
    if prev_black or next_black:
        return "fade"

    # Hard cut: single high-score frame with low neighbors
    if score > 0.7:
        # Check if nearby scores are low (not a gradual transition)
        if nearby_scores:
            avg_nearby = sum(nearby_scores) / len(nearby_scores)
            if avg_nearby < 0.15:
                return "hard_cut"
            else:
                return "dissolve"
        return "hard_cut"

    # Dissolve: medium score with elevated neighbors
    if 0.2 <= score <= 0.7:
        if nearby_scores:
            elevated = sum(1 for s in nearby_scores if s > 0.1)
            if elevated >= 2:
                return "dissolve"
        return "hard_cut"

    return "hard_cut"


def _detect_black_frames(video_path: str, timestamps: List[float],
                         window: float = 0.2) -> dict:
    """Detect near-black frames around given timestamps.

    Returns dict mapping timestamp -> {"before": bool, "after": bool}.
    """
    if not timestamps:
        return {}

    ffmpeg = get_ffmpeg_path()
    black_info = {}

    # Use blackdetect filter on the full video
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vf", "blackdetect=d=0.05:pix_th=0.10",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return {t: {"before": False, "after": False} for t in timestamps}

    # Parse black_start/black_end from stderr
    black_ranges = []
    for line in result.stderr.split("\n"):
        start_match = re.search(r"black_start:\s*(-?[\d.]+)", line)
        end_match = re.search(r"black_end:\s*(-?[\d.]+)", line)
        if start_match and end_match:
            black_ranges.append((
                float(start_match.group(1)),
                float(end_match.group(1)),
            ))

    for ts in timestamps:
        before_black = any(
            start <= ts <= end + window or (end >= ts - window and end <= ts)
            for start, end in black_ranges
        )
        after_black = any(
            start <= ts + window and end >= ts
            for start, end in black_ranges
        )
        black_info[ts] = {"before": before_black, "after": after_black}

    return black_info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect_edits(
    video_path: str,
    threshold: float = 0.3,
    min_scene_duration: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> EditDetectionResult:
    """
    Detect edit points (cuts) in pre-edited footage.

    Uses FFmpeg scdet filter to detect scene changes, then classifies
    each cut as hard_cut, dissolve, or fade.

    Args:
        video_path: Path to video file.
        threshold: Detection threshold 0.0-1.0 (lower = more sensitive).
        min_scene_duration: Minimum duration between cuts in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        EditDetectionResult with cuts, total scenes, and avg scene duration.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    threshold = max(0.01, min(1.0, float(threshold)))
    min_scene_duration = max(0.0, float(min_scene_duration))

    if on_progress:
        on_progress(5, "Probing video...")

    duration = _get_duration(video_path)
    fps = _get_fps(video_path)

    if on_progress:
        on_progress(10, "Running scene change detection...")

    # Detect scene changes
    detections = _detect_with_scdet(video_path, threshold, on_progress=on_progress)

    if on_progress:
        on_progress(50, f"Found {len(detections)} potential cuts, analyzing types...")

    # Filter by minimum scene duration
    filtered = []
    last_time = -min_scene_duration  # Allow first detection
    for det in detections:
        if det["time"] - last_time >= min_scene_duration:
            filtered.append(det)
            last_time = det["time"]

    if on_progress:
        on_progress(60, f"{len(filtered)} cuts after min-duration filter")

    # Build score lookup for nearby-score analysis
    all_scores = {round(d["time"], 3): d["score"] for d in detections}

    # Detect black frames for fade classification
    cut_timestamps = [d["time"] for d in filtered]
    if on_progress:
        on_progress(65, "Detecting black frames for fade classification...")

    black_info = _detect_black_frames(video_path, cut_timestamps)

    if on_progress:
        on_progress(80, "Classifying cut types...")

    # Classify each cut
    cuts = []
    for det in filtered:
        ts = det["time"]
        score = det["score"]

        # Gather nearby scores (within 0.5s window)
        nearby = []
        for other_time, other_score in all_scores.items():
            if other_time != round(ts, 3) and abs(other_time - ts) < 0.5:
                nearby.append(other_score)

        # Check black frame info
        bi = black_info.get(ts, {"before": False, "after": False})

        cut_type = _classify_cut_type(
            score=score,
            nearby_scores=nearby,
            prev_black=bi["before"],
            next_black=bi["after"],
        )

        frame_num = int(ts * fps) if fps > 0 else 0

        cuts.append(EditPoint(
            timestamp=ts,
            frame_number=frame_num,
            confidence=score,
            type=cut_type,
        ))

    # Compute stats
    total_scenes = len(cuts) + 1  # N cuts = N+1 scenes
    avg_scene_dur = duration / total_scenes if total_scenes > 0 and duration > 0 else 0.0

    if on_progress:
        hard = sum(1 for c in cuts if c.type == "hard_cut")
        dissolves = sum(1 for c in cuts if c.type == "dissolve")
        fades = sum(1 for c in cuts if c.type == "fade")
        on_progress(
            100,
            f"Detected {len(cuts)} edits: {hard} hard cuts, "
            f"{dissolves} dissolves, {fades} fades",
        )

    return EditDetectionResult(
        cuts=cuts,
        total_scenes=total_scenes,
        avg_scene_duration=avg_scene_dur,
    )
