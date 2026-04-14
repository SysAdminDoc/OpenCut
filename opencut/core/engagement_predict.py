"""
OpenCut Engagement & Retention Prediction

Predict where viewers will drop off using heuristic analysis of:
- Hook strength (first 3 seconds energy)
- Pacing (cut frequency, shot durations)
- Audio energy curve (loud = engaging)
- Speech rate variations
- Silence ratio (dead air = dropoff)

Inspired by Opus Clip's virality scoring. Uses FFmpeg/FFprobe only.
"""

import logging
import math
import os
import re
import subprocess as _sp
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
HOOK_WEIGHT = 0.30
PACING_WEIGHT = 0.25
AUDIO_ENERGY_WEIGHT = 0.25
VARIETY_WEIGHT = 0.20

# Thresholds
SILENCE_THRESHOLD_DB = -40.0
DEAD_AIR_SECONDS = 2.0
OPTIMAL_CUT_INTERVAL = (2.0, 8.0)  # sweet spot for social content pacing
WINDOW_SIZE = 5.0  # seconds per retention window
_HOOK_WINDOW = 3.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EngagementResult:
    """Complete engagement prediction result."""
    overall_score: int = 0
    hook_score: int = 0
    pacing_score: int = 0
    audio_energy_score: int = 0
    variety_score: int = 0
    retention_curve: List[Tuple[float, float]] = field(default_factory=list)
    drop_off_points: List[float] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    duration: float = 0.0
    silence_ratio: float = 0.0
    avg_cut_interval: float = 0.0
    total_cuts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------
def _extract_audio_levels(video_path: str) -> List[float]:
    """Extract per-second RMS audio levels (dB) using FFmpeg astats."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-i", video_path,
        "-af", "astats=metadata=1:reset=44100,"
               "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    except _sp.TimeoutExpired:
        return [-30.0]

    rms_values = re.findall(
        r"lavfi\.astats\.Overall\.RMS_level=([-\d.]+)", result.stderr
    )
    if not rms_values:
        return [-30.0]

    levels: List[float] = []
    for v in rms_values:
        try:
            val = float(v)
            if not math.isinf(val):
                levels.append(val)
        except ValueError:
            continue
    return levels if levels else [-30.0]


def _detect_silence_segments(
    video_path: str,
    threshold_db: float = SILENCE_THRESHOLD_DB,
    min_duration: float = 0.5,
) -> List[Tuple[float, float]]:
    """Detect silence segments via FFmpeg silencedetect."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", video_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    segments: List[Tuple[float, float]] = []
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    except _sp.TimeoutExpired:
        return segments

    start: Optional[float] = None
    for line in result.stderr.split("\n"):
        if "silence_start:" in line:
            try:
                start = float(
                    line.split("silence_start:")[1].strip().split()[0]
                )
            except (ValueError, IndexError):
                start = None
        elif "silence_end:" in line and start is not None:
            try:
                parts = line.split("silence_end:")[1].strip().split()
                end = float(parts[0])
                segments.append((start, end))
            except (ValueError, IndexError):
                pass
            start = None
    return segments


def _detect_scene_changes(
    video_path: str, threshold: float = 0.3
) -> List[float]:
    """Detect scene-change timestamps via FFmpeg select filter."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    cuts: List[float] = []
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=180)
    except _sp.TimeoutExpired:
        return cuts

    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            try:
                pts = line.split("pts_time:")[1].strip().split()[0]
                cuts.append(float(pts))
            except (ValueError, IndexError):
                continue
    return sorted(cuts)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def _score_hook(
    audio_levels: List[float], duration: float, cuts: List[float]
) -> Tuple[int, List[str]]:
    """Score the first 3 seconds for hook strength (0-100)."""
    suggestions: List[str] = []
    if duration < 1.0:
        return 50, suggestions

    hook_window = min(_HOOK_WINDOW, duration)
    hook_levels = audio_levels[: max(1, int(hook_window))]
    avg_hook_energy = sum(hook_levels) / len(hook_levels) if hook_levels else -60.0

    # Normalize: -60 dB -> 0, -10 dB -> 100
    energy_score = max(0, min(100, int((avg_hook_energy + 60) * 2)))

    early_cuts = sum(1 for c in cuts if c <= _HOOK_WINDOW)
    cut_bonus = min(20, early_cuts * 10)

    score = min(100, energy_score + cut_bonus)

    if avg_hook_energy < -35:
        suggestions.append(
            "Hook is too quiet -- add energy or music in the first 3 seconds"
        )
    if early_cuts == 0 and duration > 5:
        suggestions.append(
            "No visual changes in first 3s -- add a cut or dynamic element"
        )
    if score < 40:
        suggestions.append(
            "Consider starting with your most compelling moment (teaser hook)"
        )

    return score, suggestions


def _score_pacing(
    cuts: List[float], duration: float
) -> Tuple[int, List[str]]:
    """Score pacing based on cut frequency and consistency (0-100)."""
    suggestions: List[str] = []
    if duration < 2.0:
        return 50, suggestions

    if not cuts:
        if duration > 15:
            suggestions.append(
                "No cuts detected -- consider adding visual variety"
            )
            return 20, suggestions
        return 50, suggestions

    all_points = [0.0] + cuts + [duration]
    intervals = [
        all_points[i + 1] - all_points[i] for i in range(len(all_points) - 1)
    ]
    avg_interval = sum(intervals) / len(intervals) if intervals else duration

    lo, hi = OPTIMAL_CUT_INTERVAL
    if lo <= avg_interval <= hi:
        base_score = 90
    elif avg_interval < lo:
        base_score = max(40, 90 - int((lo - avg_interval) * 20))
        suggestions.append(
            "Cuts are very rapid -- ensure viewers can follow the content"
        )
    else:
        base_score = max(20, 90 - int((avg_interval - hi) * 5))
        suggestions.append(
            f"Average shot length is {avg_interval:.1f}s -- "
            "consider tighter editing for better retention"
        )

    if len(intervals) > 1:
        mean = avg_interval
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance)
        consistency = max(0, 10 - int(std_dev * 2))
        base_score = min(100, base_score + consistency)

    return base_score, suggestions


def _score_audio_energy(audio_levels: List[float]) -> Tuple[int, List[str]]:
    """Score overall audio energy and dynamics (0-100)."""
    suggestions: List[str] = []
    if not audio_levels:
        return 30, ["No audio detected -- consider adding music or narration"]

    avg_level = sum(audio_levels) / len(audio_levels)
    base_score = max(0, min(100, int((avg_level + 60) * 2)))

    if len(audio_levels) > 1:
        level_range = max(audio_levels) - min(audio_levels)
        if level_range > 15:
            base_score = min(100, base_score + 10)
        elif level_range < 5:
            suggestions.append(
                "Audio is very flat -- add dynamic variation to maintain attention"
            )

    if avg_level < -35:
        suggestions.append(
            "Overall audio is too quiet -- boost levels or add background music"
        )

    return base_score, suggestions


def _score_variety(
    cuts: List[float],
    silence_segments: List[Tuple[float, float]],
    duration: float,
) -> Tuple[int, List[str]]:
    """Score content variety (0-100)."""
    suggestions: List[str] = []
    if duration < 2.0:
        return 50, suggestions

    total_silence = sum(end - start for start, end in silence_segments)
    silence_ratio = total_silence / duration if duration > 0 else 0

    cut_density = len(cuts) / (duration / 60.0) if duration > 0 else 0

    if 6 <= cut_density <= 15:
        base_score = 85
    elif cut_density > 15:
        base_score = 70
    elif cut_density > 0:
        base_score = max(30, int(cut_density * 10))
    else:
        base_score = 30

    silence_penalty = int(silence_ratio * 80)
    base_score = max(0, base_score - silence_penalty)

    if silence_ratio > 0.15:
        suggestions.append(
            f"Dead air covers {silence_ratio:.0%} of the video -- "
            "fill gaps with music or cut them out"
        )
    if cut_density < 3 and duration > 30:
        suggestions.append(
            "Very low visual variety -- viewers may lose interest"
        )

    return base_score, suggestions


# ---------------------------------------------------------------------------
# Retention curve
# ---------------------------------------------------------------------------
def _generate_retention_curve(
    audio_levels: List[float],
    cuts: List[float],
    silence_segments: List[Tuple[float, float]],
    duration: float,
    window: float = WINDOW_SIZE,
) -> List[Tuple[float, float]]:
    """Generate predicted retention curve as ``(timestamp, pct)`` pairs."""
    if duration <= 0:
        return [(0.0, 100.0)]

    curve: List[Tuple[float, float]] = []
    num_windows = max(1, int(math.ceil(duration / window)))
    base_retention = 100.0

    for i in range(num_windows):
        t_start = i * window
        t_end = min((i + 1) * window, duration)
        t_mid = (t_start + t_end) / 2

        natural_decay = 2.5

        idx_s = int(t_start)
        idx_e = min(len(audio_levels), int(t_end) + 1)
        window_levels = audio_levels[idx_s:idx_e]
        if window_levels:
            avg_energy = sum(window_levels) / len(window_levels)
            energy_factor = max(0, (avg_energy + 60) / 50)
            natural_decay -= energy_factor * 1.5

        window_cuts = sum(1 for c in cuts if t_start <= c < t_end)
        natural_decay -= min(1.5, window_cuts * 0.5)

        silence_in_window = sum(
            min(end, t_end) - max(start, t_start)
            for start, end in silence_segments
            if start < t_end and end > t_start
        )
        silence_frac = silence_in_window / window if window > 0 else 0
        natural_decay += silence_frac * 5

        base_retention = max(5.0, base_retention - max(0, natural_decay))
        curve.append((round(t_mid, 1), round(base_retention, 1)))

    return curve


def _find_drop_off_points(
    curve: List[Tuple[float, float]], threshold: float = 10.0
) -> List[float]:
    """Return timestamps where retention drops by more than *threshold* %."""
    drops: List[float] = []
    for i in range(1, len(curve)):
        if curve[i - 1][1] - curve[i][1] > threshold:
            drops.append(curve[i][0])
    return drops


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def predict_engagement(
    video_path: str,
    transcript: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Predict viewer engagement and retention for a video.

    Args:
        video_path: Path to input video file.
        transcript: Optional transcript text (reserved for speech-rate analysis).
        on_progress: Callback ``(pct, msg)`` for progress reporting.

    Returns:
        dict serialised from :class:`EngagementResult`.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Analyzing video metadata...")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        raise ValueError("Could not determine video duration")

    result = EngagementResult(duration=duration)

    # Step 1 -- audio levels
    if on_progress:
        on_progress(15, "Analyzing audio energy...")
    audio_levels = _extract_audio_levels(video_path)

    # Step 2 -- silence
    if on_progress:
        on_progress(30, "Detecting silence segments...")
    silence_segments = _detect_silence_segments(video_path)
    total_silence = sum(end - start for start, end in silence_segments)
    result.silence_ratio = round(total_silence / duration, 3) if duration > 0 else 0

    # Step 3 -- scene changes
    if on_progress:
        on_progress(45, "Detecting scene changes...")
    cuts = _detect_scene_changes(video_path)
    result.total_cuts = len(cuts)
    result.avg_cut_interval = round(duration / (len(cuts) + 1), 2)

    # Step 4 -- score each dimension
    if on_progress:
        on_progress(60, "Scoring engagement factors...")
    all_suggestions: List[str] = []

    hook_score, hook_tips = _score_hook(audio_levels, duration, cuts)
    pacing_score, pacing_tips = _score_pacing(cuts, duration)
    energy_score, energy_tips = _score_audio_energy(audio_levels)
    variety_score, variety_tips = _score_variety(cuts, silence_segments, duration)

    all_suggestions.extend(hook_tips)
    all_suggestions.extend(pacing_tips)
    all_suggestions.extend(energy_tips)
    all_suggestions.extend(variety_tips)

    result.hook_score = hook_score
    result.pacing_score = pacing_score
    result.audio_energy_score = energy_score
    result.variety_score = variety_score

    result.overall_score = int(
        hook_score * HOOK_WEIGHT
        + pacing_score * PACING_WEIGHT
        + energy_score * AUDIO_ENERGY_WEIGHT
        + variety_score * VARIETY_WEIGHT
    )

    # Step 5 -- retention curve
    if on_progress:
        on_progress(80, "Generating retention curve...")
    result.retention_curve = _generate_retention_curve(
        audio_levels, cuts, silence_segments, duration
    )

    # Step 6 -- drop-off points
    result.drop_off_points = _find_drop_off_points(result.retention_curve)

    # Deduplicate suggestions
    seen: set = set()
    unique: List[str] = []
    for tip in all_suggestions:
        if tip not in seen:
            seen.add(tip)
            unique.append(tip)
    result.suggestions = unique

    if on_progress:
        on_progress(100, "Engagement analysis complete")

    return result.to_dict()
