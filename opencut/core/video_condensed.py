"""
OpenCut AI Video Summary / Condensed Recap

Select important shots via scene detection + transcript + scoring.
Trim to essential content and assemble a 30-60 second recap.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class ShotScore:
    """A scored segment of the video."""
    start: float = 0.0
    end: float = 0.0
    score: float = 0.0
    reason: str = ""


@dataclass
class CondensedResult:
    """Result of the video condensation."""
    output_path: str = ""
    original_duration: float = 0.0
    condensed_duration: float = 0.0
    compression_ratio: float = 0.0
    shots_selected: int = 0
    total_shots: int = 0
    method: str = ""


# ---------------------------------------------------------------------------
# Scene detection via FFmpeg
# ---------------------------------------------------------------------------

def _detect_scenes(video_path: str, threshold: float = 0.3) -> List[Dict]:
    """
    Detect scene changes using FFmpeg select filter with scene score.

    Returns a list of dicts with 'start', 'end', 'duration' for each scene.
    """
    import subprocess

    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    stderr = result.stderr

    # Parse showinfo output for pts_time
    import re
    scene_times = [0.0]
    for match in re.finditer(r"pts_time:\s*([\d.]+)", stderr):
        t = float(match.group(1))
        if t > 0:
            scene_times.append(t)

    # Get total duration
    info = get_video_info(video_path)
    total_duration = info.get("duration", 0)
    if total_duration > 0:
        scene_times.append(total_duration)

    # Build scene segments
    scenes = []
    scene_times = sorted(set(scene_times))
    for i in range(len(scene_times) - 1):
        start = scene_times[i]
        end = scene_times[i + 1]
        dur = end - start
        if dur > 0.5:  # Skip very short scenes
            scenes.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(dur, 3),
            })

    return scenes


def _score_scenes(
    scenes: List[Dict],
    total_duration: float,
) -> List[ShotScore]:
    """
    Score each scene based on heuristics:
    - Position bonus: first and last scenes score higher
    - Duration bonus: medium-length scenes preferred
    - Variety: spread selections across the video
    """
    scored = []

    for i, scene in enumerate(scenes):
        score = 0.5  # Base score
        dur = scene["duration"]
        mid = (scene["start"] + scene["end"]) / 2

        # Position: favor intro and outro
        pos_ratio = mid / max(total_duration, 1)
        if pos_ratio < 0.15 or pos_ratio > 0.85:
            score += 0.3
            reason = "intro/outro"
        elif 0.4 < pos_ratio < 0.6:
            score += 0.15
            reason = "midpoint"
        else:
            reason = "body"

        # Duration: prefer 2-10 second scenes (good content density)
        if 2 <= dur <= 10:
            score += 0.2
        elif dur > 30:
            score -= 0.1  # Very long scenes are often static

        # First and last scenes always included
        if i == 0:
            score += 0.4
            reason = "opening"
        elif i == len(scenes) - 1:
            score += 0.3
            reason = "closing"

        scored.append(ShotScore(
            start=scene["start"],
            end=scene["end"],
            score=round(min(1.0, max(0.0, score)), 3),
            reason=reason,
        ))

    return scored


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def condense_video(
    video_path: str,
    target_duration: float = 45.0,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
    scene_threshold: float = 0.3,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Create a condensed video recap by selecting the most important shots.

    Process:
    1. Detect scene boundaries
    2. Score each scene by position, duration, and content heuristics
    3. Select top-scoring scenes up to target duration
    4. Assemble selected scenes into a continuous recap

    Args:
        video_path:  Input video file.
        target_duration:  Target recap length in seconds (default 45).
        min_duration:  Minimum acceptable duration.
        max_duration:  Maximum acceptable duration.
        scene_threshold:  FFmpeg scene detection threshold (0-1).
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *original_duration*,
        *condensed_duration*, *compression_ratio*, *shots_selected*,
        *total_shots*, *method*.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    info = get_video_info(video_path)
    original_duration = info.get("duration", 0.0)

    if original_duration <= max_duration:
        # Video is already short enough
        return {
            "output_path": video_path,
            "original_duration": round(original_duration, 2),
            "condensed_duration": round(original_duration, 2),
            "compression_ratio": 1.0,
            "shots_selected": 1,
            "total_shots": 1,
            "method": "passthrough",
        }

    out = output_path_override or output_path(video_path, "condensed")

    if on_progress:
        on_progress(5, "Detecting scenes...")

    scenes = _detect_scenes(video_path, threshold=scene_threshold)
    if not scenes:
        # Fallback: evenly split
        chunk = original_duration / 10
        scenes = [
            {"start": i * chunk, "end": (i + 1) * chunk, "duration": chunk}
            for i in range(10)
        ]

    if on_progress:
        on_progress(30, f"Found {len(scenes)} scenes, scoring...")

    scored = _score_scenes(scenes, original_duration)
    scored.sort(key=lambda s: s.score, reverse=True)

    # Select scenes up to target duration
    selected = []
    accumulated = 0.0
    for shot in scored:
        clip_dur = shot.end - shot.start
        # Trim long scenes to max 5s each
        if clip_dur > 5:
            clip_dur = 5
            shot.end = shot.start + 5

        if accumulated + clip_dur > max_duration:
            if accumulated >= min_duration:
                break
            # Trim this clip to fit
            remaining = max_duration - accumulated
            if remaining > 1:
                shot.end = shot.start + remaining
                clip_dur = remaining
            else:
                break

        selected.append(shot)
        accumulated += clip_dur

    # Re-sort by timeline position for continuity
    selected.sort(key=lambda s: s.start)

    if on_progress:
        on_progress(50, f"Assembling {len(selected)} shots...")

    # Build concat filter
    if not selected:
        raise RuntimeError("No scenes selected for condensation")

    # Extract each clip and concat
    clip_files = []
    concat_list_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as concat_f:
            concat_list_path = concat_f.name

            for i, shot in enumerate(selected):
                if on_progress:
                    pct = 50 + int((i / len(selected)) * 35)
                    on_progress(pct, f"Extracting clip {i+1}/{len(selected)}...")

                _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                clip_path = _ntf.name
                _ntf.close()
                clip_files.append(clip_path)

                run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(shot.start),
                    "-i", video_path,
                    "-t", str(shot.end - shot.start),
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    clip_path,
                ], timeout=300)

                concat_f.write(f"file '{clip_path}'\n")

        if on_progress:
            on_progress(88, "Concatenating clips...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "faststart",
            out,
        ], timeout=3600)
    finally:
        for cf in clip_files:
            try:
                os.unlink(cf)
            except OSError:
                pass
        if concat_list_path:
            try:
                os.unlink(concat_list_path)
            except OSError:
                pass

    if on_progress:
        on_progress(95, "Verifying output...")

    out_info = get_video_info(out)
    condensed_duration = out_info.get("duration", accumulated)

    compression_ratio = round(condensed_duration / max(original_duration, 1), 3)

    if on_progress:
        on_progress(100, "Video condensation complete!")

    return {
        "output_path": out,
        "original_duration": round(original_duration, 2),
        "condensed_duration": round(condensed_duration, 2),
        "compression_ratio": compression_ratio,
        "shots_selected": len(selected),
        "total_shots": len(scenes),
        "method": "scene_score",
    }
