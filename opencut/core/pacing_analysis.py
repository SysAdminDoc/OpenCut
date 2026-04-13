"""
OpenCut AI Pacing & Rhythm Analysis

Analyzes video edit pacing by detecting cuts and comparing against
genre-specific profiles. Provides actionable suggestions for re-editing.

Uses FFmpeg scene detection -- no additional dependencies required.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Genre Profiles -- target pacing metrics for common video genres
# ---------------------------------------------------------------------------
GENRE_PROFILES: Dict[str, Dict] = {
    "general": {
        "target_cpm": 10.0,
        "target_avg": 6.0,
        "min_avg": 2.0,
        "max_avg": 15.0,
        "description": "General content",
    },
    "trailer": {
        "target_cpm": 30.0,
        "target_avg": 2.0,
        "min_avg": 0.5,
        "max_avg": 5.0,
        "description": "Movie/game trailer (fast-paced)",
    },
    "interview": {
        "target_cpm": 5.0,
        "target_avg": 12.0,
        "min_avg": 6.0,
        "max_avg": 30.0,
        "description": "Interview or talking head",
    },
    "documentary": {
        "target_cpm": 8.0,
        "target_avg": 7.5,
        "min_avg": 3.0,
        "max_avg": 20.0,
        "description": "Documentary or educational",
    },
    "music_video": {
        "target_cpm": 25.0,
        "target_avg": 2.4,
        "min_avg": 0.5,
        "max_avg": 6.0,
        "description": "Music video (beat-driven editing)",
    },
    "vlog": {
        "target_cpm": 12.0,
        "target_avg": 5.0,
        "min_avg": 2.0,
        "max_avg": 12.0,
        "description": "Vlog or personal content",
    },
    "commercial": {
        "target_cpm": 20.0,
        "target_avg": 3.0,
        "min_avg": 1.0,
        "max_avg": 8.0,
        "description": "Advertisement or commercial",
    },
    "cinematic": {
        "target_cpm": 6.0,
        "target_avg": 10.0,
        "min_avg": 4.0,
        "max_avg": 30.0,
        "description": "Cinematic film or short",
    },
}


@dataclass
class ShotInfo:
    """A single detected shot with timing."""
    index: int
    start: float
    end: float
    duration: float


@dataclass
class PacingResult:
    """Complete pacing analysis results."""
    cuts_per_minute: float = 0.0
    avg_shot_duration: float = 0.0
    min_shot_duration: float = 0.0
    max_shot_duration: float = 0.0
    total_shots: int = 0
    total_duration: float = 0.0
    shot_duration_distribution: Dict[str, int] = field(default_factory=dict)
    pacing_curve: List[Dict] = field(default_factory=list)
    genre: str = "general"
    genre_profile: Dict = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    shots: List[ShotInfo] = field(default_factory=list)


def analyze_pacing(
    input_path: str,
    genre: str = "general",
    threshold: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Analyze video edit pacing and rhythm.

    Detects cuts via scene detection, calculates pacing metrics,
    and compares against genre-specific profiles.

    Args:
        input_path: Source video file.
        genre: Genre profile to compare against.
        threshold: Scene detection threshold (0.0-1.0).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with pacing metrics, distribution, pacing curve, and suggestions.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    genre = genre.lower().strip()
    if genre not in GENRE_PROFILES:
        genre = "general"

    profile = GENRE_PROFILES[genre]

    if on_progress:
        on_progress(5, "Detecting scene cuts...")

    # Use existing scene detection
    from opencut.core.scene_detect import detect_scenes
    scene_info = detect_scenes(
        input_path,
        threshold=threshold,
        min_scene_length=0.3,
        on_progress=None,
    )

    if on_progress:
        on_progress(50, "Calculating pacing metrics...")

    boundaries = scene_info.boundaries
    duration = scene_info.duration
    if duration <= 0:
        info = get_video_info(input_path)
        duration = info.get("duration", 0.0)

    # Build shot list from boundaries
    shots: List[ShotInfo] = []
    for i in range(len(boundaries)):
        start = boundaries[i].time
        end = boundaries[i + 1].time if i + 1 < len(boundaries) else duration
        shot_dur = max(0.0, end - start)
        if shot_dur > 0:
            shots.append(ShotInfo(index=i + 1, start=start, end=end, duration=shot_dur))

    total_shots = len(shots)

    # Basic metrics
    shot_durations = [s.duration for s in shots]
    avg_shot = sum(shot_durations) / total_shots if total_shots > 0 else duration
    min_shot = min(shot_durations) if shot_durations else 0.0
    max_shot = max(shot_durations) if shot_durations else 0.0
    cpm = (total_shots / (duration / 60.0)) if duration > 0 else 0.0

    if on_progress:
        on_progress(65, "Building pacing distribution...")

    # Shot duration distribution (buckets)
    distribution = {
        "under_1s": 0,
        "1s_to_3s": 0,
        "3s_to_5s": 0,
        "5s_to_10s": 0,
        "10s_to_20s": 0,
        "over_20s": 0,
    }
    for d in shot_durations:
        if d < 1.0:
            distribution["under_1s"] += 1
        elif d < 3.0:
            distribution["1s_to_3s"] += 1
        elif d < 5.0:
            distribution["3s_to_5s"] += 1
        elif d < 10.0:
            distribution["5s_to_10s"] += 1
        elif d < 20.0:
            distribution["10s_to_20s"] += 1
        else:
            distribution["over_20s"] += 1

    if on_progress:
        on_progress(75, "Computing pacing curve...")

    # Pacing curve: rolling average of shot duration over 30s windows
    pacing_curve = _compute_pacing_curve(shots, duration, window_seconds=30.0)

    if on_progress:
        on_progress(85, "Generating suggestions...")

    # Generate suggestions by comparing against genre profile
    suggestions = _generate_suggestions(shots, cpm, avg_shot, profile, genre, duration)

    if on_progress:
        on_progress(100, f"Pacing analysis complete: {cpm:.1f} cuts/min")

    result = PacingResult(
        cuts_per_minute=round(cpm, 2),
        avg_shot_duration=round(avg_shot, 2),
        min_shot_duration=round(min_shot, 2),
        max_shot_duration=round(max_shot, 2),
        total_shots=total_shots,
        total_duration=round(duration, 2),
        shot_duration_distribution=distribution,
        pacing_curve=pacing_curve,
        genre=genre,
        genre_profile=profile,
        suggestions=suggestions,
        shots=shots,
    )
    return _result_to_dict(result)


def _compute_pacing_curve(
    shots: List[ShotInfo],
    total_duration: float,
    window_seconds: float = 30.0,
) -> List[Dict]:
    """Compute rolling average shot duration over time windows."""
    if not shots or total_duration <= 0:
        return []

    curve = []
    step = max(5.0, window_seconds / 3.0)
    t = 0.0

    while t < total_duration:
        window_start = max(0.0, t - window_seconds / 2.0)
        window_end = min(total_duration, t + window_seconds / 2.0)

        # Find shots overlapping this window
        window_shots = [
            s for s in shots
            if s.start < window_end and s.end > window_start
        ]

        if window_shots:
            avg_dur = sum(s.duration for s in window_shots) / len(window_shots)
            local_cpm = len(window_shots) / ((window_end - window_start) / 60.0) if window_end > window_start else 0
        else:
            avg_dur = 0.0
            local_cpm = 0.0

        curve.append({
            "time": round(t, 1),
            "avg_shot_duration": round(avg_dur, 2),
            "local_cpm": round(local_cpm, 1),
        })
        t += step

    return curve


def _generate_suggestions(
    shots: List[ShotInfo],
    cpm: float,
    avg_shot: float,
    profile: Dict,
    genre: str,
    duration: float,
) -> List[str]:
    """Generate actionable pacing suggestions based on genre comparison."""
    suggestions = []

    target_cpm = profile["target_cpm"]
    target_avg = profile["target_avg"]
    min_avg = profile.get("min_avg", 1.0)
    max_avg = profile.get("max_avg", 30.0)

    # Overall pacing comparison
    if cpm < target_cpm * 0.5:
        suggestions.append(
            f"Pacing is very slow for {genre} content ({cpm:.1f} cuts/min vs target {target_cpm:.0f}). "
            f"Consider adding more cuts or tightening shots."
        )
    elif cpm < target_cpm * 0.75:
        suggestions.append(
            f"Pacing is slightly slow for {genre} ({cpm:.1f} cuts/min vs target {target_cpm:.0f})."
        )
    elif cpm > target_cpm * 2.0:
        suggestions.append(
            f"Pacing is very fast for {genre} content ({cpm:.1f} cuts/min vs target {target_cpm:.0f}). "
            f"Consider letting some shots breathe longer."
        )
    elif cpm > target_cpm * 1.5:
        suggestions.append(
            f"Pacing is slightly fast for {genre} ({cpm:.1f} cuts/min vs target {target_cpm:.0f})."
        )

    # Find slow and fast sections
    if len(shots) >= 3:
        # Check for consecutive slow shots
        slow_runs = _find_runs(shots, max_avg, "slow")
        for run_start, run_end, run_avg in slow_runs:
            suggestions.append(
                f"Shots {run_start}-{run_end} average {run_avg:.1f}s, "
                f"consider tightening to {target_avg:.0f}-{target_avg * 1.2:.0f}s for {genre}."
            )

        # Check for consecutive fast shots
        fast_runs = _find_runs(shots, min_avg, "fast")
        for run_start, run_end, run_avg in fast_runs:
            suggestions.append(
                f"Shots {run_start}-{run_end} average {run_avg:.1f}s, "
                f"very rapid for {genre} -- consider adding breathing room."
            )

    # Individual outliers
    for shot in shots:
        if shot.duration > max_avg * 1.5 and shot.duration > 20.0:
            suggestions.append(
                f"Shot {shot.index} is {shot.duration:.1f}s -- very long for {genre}. "
                f"Consider splitting or trimming."
            )

    # Limit to top 8 suggestions
    return suggestions[:8]


def _find_runs(
    shots: List[ShotInfo],
    threshold: float,
    direction: str,
) -> List[tuple]:
    """Find runs of consecutive shots that are above/below threshold."""
    runs = []
    i = 0
    while i < len(shots):
        if direction == "slow" and shots[i].duration > threshold:
            run_start = i
            while i < len(shots) and shots[i].duration > threshold:
                i += 1
            run_end = i - 1
            if run_end - run_start >= 2:  # At least 3 consecutive shots
                avg = sum(shots[j].duration for j in range(run_start, run_end + 1)) / (run_end - run_start + 1)
                runs.append((shots[run_start].index, shots[run_end].index, avg))
        elif direction == "fast" and shots[i].duration < threshold:
            run_start = i
            while i < len(shots) and shots[i].duration < threshold:
                i += 1
            run_end = i - 1
            if run_end - run_start >= 4:  # At least 5 consecutive fast shots
                avg = sum(shots[j].duration for j in range(run_start, run_end + 1)) / (run_end - run_start + 1)
                runs.append((shots[run_start].index, shots[run_end].index, avg))
        else:
            i += 1
    return runs


def _result_to_dict(result: PacingResult) -> dict:
    """Convert PacingResult to a JSON-serializable dict."""
    return {
        "cuts_per_minute": result.cuts_per_minute,
        "avg_shot_duration": result.avg_shot_duration,
        "min_shot_duration": result.min_shot_duration,
        "max_shot_duration": result.max_shot_duration,
        "total_shots": result.total_shots,
        "total_duration": result.total_duration,
        "shot_duration_distribution": result.shot_duration_distribution,
        "pacing_curve": result.pacing_curve,
        "genre": result.genre,
        "genre_profile": result.genre_profile,
        "suggestions": result.suggestions,
        "shots": [
            {
                "index": s.index,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "duration": round(s.duration, 3),
            }
            for s in result.shots
        ],
    }
