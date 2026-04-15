"""
OpenCut Timeline Engagement Scorer

Per-segment engagement scoring across an entire video timeline.
Divides video into equal segments and scores each for:
- Visual interest (motion, color variety, face presence)
- Audio engagement (speech, music, energy)
- Pacing effectiveness
- Content diversity

Uses FFmpeg/FFprobe only — no additional dependencies required.
"""

import logging
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Scoring sub-weights (within each segment)
# ---------------------------------------------------------------------------
VISUAL_WEIGHT = 0.30
AUDIO_WEIGHT = 0.30
PACING_WEIGHT = 0.20
DIVERSITY_WEIGHT = 0.20

# Motion thresholds
HIGH_MOTION_THRESHOLD = 0.15   # scene change score indicating high motion
LOW_MOTION_THRESHOLD = 0.02    # below this is static/boring
SPEECH_ENERGY_FLOOR = -50.0    # dBFS below which audio is considered silent


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SegmentScore:
    """Engagement score for a single timeline segment."""
    index: int = 0
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    visual_score: float = 0.0
    audio_score: float = 0.0
    pacing_score: float = 0.0
    diversity_score: float = 0.0
    overall_score: float = 0.0
    motion_level: float = 0.0
    audio_energy: float = -70.0
    has_speech: bool = False
    scene_changes: int = 0
    label: str = ""  # "high", "medium", "low"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimelineScoreResult:
    """Complete timeline scoring result."""
    total_duration: float = 0.0
    segment_count: int = 0
    segment_duration: float = 10.0
    average_score: float = 0.0
    peak_score: float = 0.0
    lowest_score: float = 0.0
    peak_segment_index: int = 0
    lowest_segment_index: int = 0
    engagement_curve: List[float] = field(default_factory=list)
    segments: List[Dict] = field(default_factory=list)
    high_segments: int = 0
    medium_segments: int = 0
    low_segments: int = 0
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Motion analysis
# ---------------------------------------------------------------------------
def _measure_motion(video_path: str, start: float, duration: float) -> Tuple[float, int]:
    """Measure motion level in a segment via FFmpeg scene detection.

    Returns (motion_level_0_1, scene_change_count).
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-vf", "select='gte(scene,0.01)',metadata=print",
        "-vsync", "vfr", "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        stderr = result.stderr.decode(errors="replace")
    except Exception as exc:
        logger.debug("Motion measurement failed at %.1f: %s", start, exc)
        return 0.0, 0

    # Count scene changes and extract scores
    scores = []
    for line in stderr.split("\n"):
        match = re.search(r"scene_score=([0-9.]+)", line)
        if match:
            scores.append(float(match.group(1)))

    scene_count = sum(1 for s in scores if s > 0.1)
    avg_motion = sum(scores) / len(scores) if scores else 0.0

    return min(1.0, avg_motion), scene_count


def _measure_color_variety(video_path: str, start: float, duration: float) -> float:
    """Estimate color variety by measuring standard deviation of hue via signalstats.

    Higher hue stddev = more color variety (0..1 normalized).
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(start), "-t", str(min(duration, 5.0)),
        "-i", video_path,
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=print",
        "-vframes", "10",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        stderr = result.stderr.decode(errors="replace")
        # Parse HUEAVG values
        hues = []
        for line in stderr.split("\n"):
            match = re.search(r"HUEAVG=\s*([0-9.]+)", line)
            if match:
                hues.append(float(match.group(1)))
        if len(hues) < 2:
            return 0.5
        mean_h = sum(hues) / len(hues)
        variance = sum((h - mean_h) ** 2 for h in hues) / len(hues)
        stddev = math.sqrt(variance)
        # Normalize: stddev of 50+ = high variety
        return min(1.0, stddev / 50.0)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------
def _measure_audio_energy(video_path: str, start: float, duration: float) -> Tuple[float, bool]:
    """Measure audio RMS energy and detect speech presence.

    Returns (energy_db, has_speech).
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-af", "astats=metadata=1:reset=1,ametadata=print",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        stderr = result.stderr.decode(errors="replace")
    except Exception:
        return -70.0, False

    # Parse RMS levels
    rms_values = []
    for line in stderr.split("\n"):
        match = re.search(r"RMS_level=(-?[0-9.]+)", line)
        if match:
            val = float(match.group(1))
            if val > -100:
                rms_values.append(val)

    if not rms_values:
        return -70.0, False

    avg_rms = sum(rms_values) / len(rms_values)

    # Speech detection heuristic: audio above floor with moderate variance
    # indicates speech (vs constant music or silence)
    has_speech = False
    if avg_rms > SPEECH_ENERGY_FLOOR and len(rms_values) >= 3:
        rms_var = sum((r - avg_rms) ** 2 for r in rms_values) / len(rms_values)
        rms_stddev = math.sqrt(rms_var)
        # Speech typically has 3-15 dB variation
        if 2.0 < rms_stddev < 20.0:
            has_speech = True

    return avg_rms, has_speech


def _score_audio(energy_db: float, has_speech: bool) -> float:
    """Convert audio metrics to engagement score (0..100)."""
    # Normalize energy: -70 = 0, -10 = 100
    energy_norm = max(0, min(1.0, (energy_db + 70) / 60.0))
    base = energy_norm * 70

    # Speech bonus
    if has_speech:
        base += 25

    # Music detection would go here with more advanced analysis
    return min(100.0, base)


# ---------------------------------------------------------------------------
# Visual scoring
# ---------------------------------------------------------------------------
def _score_visual(motion: float, color_variety: float, scene_changes: int,
                  segment_duration: float) -> float:
    """Combine visual metrics into engagement score (0..100)."""
    # Motion contribution (0..40)
    if motion > HIGH_MOTION_THRESHOLD:
        motion_score = 35 + min(5, (motion - HIGH_MOTION_THRESHOLD) * 50)
    elif motion > LOW_MOTION_THRESHOLD:
        motion_range = HIGH_MOTION_THRESHOLD - LOW_MOTION_THRESHOLD
        motion_score = 15 + 20 * ((motion - LOW_MOTION_THRESHOLD) / motion_range)
    else:
        motion_score = motion / max(LOW_MOTION_THRESHOLD, 0.001) * 15

    # Color variety contribution (0..30)
    color_score = color_variety * 30

    # Scene changes contribution (0..30)
    changes_per_sec = scene_changes / max(segment_duration, 1.0)
    if 0.1 <= changes_per_sec <= 1.0:
        change_score = 30  # ideal range
    elif changes_per_sec < 0.1:
        change_score = changes_per_sec / 0.1 * 20
    else:
        change_score = max(10, 30 - (changes_per_sec - 1.0) * 10)

    return min(100.0, motion_score + color_score + change_score)


# ---------------------------------------------------------------------------
# Pacing scoring
# ---------------------------------------------------------------------------
def _score_pacing(scene_changes: int, segment_duration: float, segment_index: int,
                  total_segments: int) -> float:
    """Score pacing based on cut rate relative to position in the timeline."""
    cuts_per_sec = scene_changes / max(segment_duration, 1.0)

    # Ideal pacing varies by position: faster in intro/outro, moderate in middle
    position_ratio = segment_index / max(total_segments - 1, 1)

    if position_ratio < 0.15:
        # Intro — faster pacing expected
        ideal_rate = 0.5
    elif position_ratio > 0.85:
        # Outro — can be slower
        ideal_rate = 0.3
    else:
        # Body — moderate
        ideal_rate = 0.2

    deviation = abs(cuts_per_sec - ideal_rate)
    score = max(0, 100 - deviation * 150)
    return min(100.0, score)


# ---------------------------------------------------------------------------
# Diversity scoring
# ---------------------------------------------------------------------------
def _score_diversity(
    segments_so_far: List["SegmentScore"],
    current_motion: float,
    current_energy: float,
    current_speech: bool,
) -> float:
    """Score how different this segment is from recent segments (variety = engagement)."""
    if not segments_so_far:
        return 50.0  # neutral for first segment

    # Compare against last 3 segments
    recent = segments_so_far[-3:]

    motion_diffs = [abs(current_motion - s.motion_level) for s in recent]
    energy_diffs = [abs(current_energy - s.audio_energy) for s in recent]

    avg_motion_diff = sum(motion_diffs) / len(motion_diffs)
    avg_energy_diff = sum(energy_diffs) / len(energy_diffs)

    # More difference = more engaging
    motion_diversity = min(1.0, avg_motion_diff / 0.1) * 40
    energy_diversity = min(1.0, avg_energy_diff / 20) * 30

    # Speech toggle is good for variety
    recent_speech = [s.has_speech for s in recent]
    speech_variety = 30 if current_speech != (sum(recent_speech) > len(recent) / 2) else 10

    return min(100.0, motion_diversity + energy_diversity + speech_variety)


# ---------------------------------------------------------------------------
# Segment label assignment
# ---------------------------------------------------------------------------
def _classify_score(score: float) -> str:
    """Classify engagement score into label."""
    if score >= 70:
        return "high"
    elif score >= 40:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
def _generate_summary(result: "TimelineScoreResult") -> str:
    """Generate a human-readable engagement summary."""
    parts = []
    parts.append(f"Analyzed {result.segment_count} segments over {result.total_duration:.1f}s")
    parts.append(f"Average engagement: {result.average_score:.0f}/100")

    if result.high_segments > result.segment_count * 0.6:
        parts.append("Strong overall engagement throughout the timeline")
    elif result.low_segments > result.segment_count * 0.4:
        parts.append("Significant portions have low engagement — consider re-editing")
    else:
        parts.append("Mixed engagement levels — focus on strengthening weak segments")

    if result.peak_segment_index >= 0:
        peak_seg = result.segments[result.peak_segment_index] if result.segments else {}
        parts.append(f"Peak engagement at segment {result.peak_segment_index + 1} "
                     f"({peak_seg.get('start', 0):.1f}s)")

    if result.lowest_segment_index >= 0 and result.segment_count > 1:
        low_seg = result.segments[result.lowest_segment_index] if result.segments else {}
        parts.append(f"Lowest engagement at segment {result.lowest_segment_index + 1} "
                     f"({low_seg.get('start', 0):.1f}s)")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------
def score_timeline(
    video_path: str,
    segment_duration: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> TimelineScoreResult:
    """
    Score engagement across a video timeline.

    Divides the video into equal segments and scores each for visual interest,
    audio engagement, pacing, and content diversity.

    Args:
        video_path: Path to the video file.
        segment_duration: Duration of each analysis segment in seconds.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        TimelineScoreResult with per-segment scores and overall metrics.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If segment_duration is invalid.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if segment_duration <= 0:
        raise ValueError(f"segment_duration must be positive, got {segment_duration}")

    if on_progress:
        on_progress(1, "Starting timeline scoring...")

    info = get_video_info(video_path)
    total_duration = info.get("duration", 0)
    if total_duration <= 0:
        raise ValueError("Video has zero or negative duration")

    # Calculate segment boundaries
    segment_count = max(1, int(math.ceil(total_duration / segment_duration)))
    actual_seg_dur = total_duration / segment_count

    scored_segments: List[SegmentScore] = []

    for i in range(segment_count):
        seg_start = i * actual_seg_dur
        seg_end = min((i + 1) * actual_seg_dur, total_duration)
        seg_dur = seg_end - seg_start

        if on_progress:
            pct = 5 + int(85 * i / segment_count)
            on_progress(pct, f"Scoring segment {i + 1}/{segment_count}...")

        # Measure visual metrics
        motion, scene_changes = _measure_motion(video_path, seg_start, seg_dur)
        color_variety = _measure_color_variety(video_path, seg_start, seg_dur)

        # Measure audio metrics
        audio_energy, has_speech = _measure_audio_energy(video_path, seg_start, seg_dur)

        # Compute sub-scores
        visual = _score_visual(motion, color_variety, scene_changes, seg_dur)
        audio = _score_audio(audio_energy, has_speech)
        pacing = _score_pacing(scene_changes, seg_dur, i, segment_count)
        diversity = _score_diversity(scored_segments, motion, audio_energy, has_speech)

        # Weighted overall
        overall = (
            visual * VISUAL_WEIGHT
            + audio * AUDIO_WEIGHT
            + pacing * PACING_WEIGHT
            + diversity * DIVERSITY_WEIGHT
        )

        seg = SegmentScore(
            index=i,
            start=round(seg_start, 3),
            end=round(seg_end, 3),
            duration=round(seg_dur, 3),
            visual_score=round(visual, 1),
            audio_score=round(audio, 1),
            pacing_score=round(pacing, 1),
            diversity_score=round(diversity, 1),
            overall_score=round(overall, 1),
            motion_level=round(motion, 4),
            audio_energy=round(audio_energy, 1),
            has_speech=has_speech,
            scene_changes=scene_changes,
            label=_classify_score(overall),
        )
        scored_segments.append(seg)

    if on_progress:
        on_progress(92, "Computing summary...")

    # Aggregate metrics
    scores = [s.overall_score for s in scored_segments]
    avg_score = sum(scores) / len(scores) if scores else 0
    peak_idx = scores.index(max(scores)) if scores else 0
    low_idx = scores.index(min(scores)) if scores else 0

    high_count = sum(1 for s in scored_segments if s.label == "high")
    med_count = sum(1 for s in scored_segments if s.label == "medium")
    low_count = sum(1 for s in scored_segments if s.label == "low")

    result = TimelineScoreResult(
        total_duration=round(total_duration, 3),
        segment_count=segment_count,
        segment_duration=round(actual_seg_dur, 3),
        average_score=round(avg_score, 1),
        peak_score=round(max(scores) if scores else 0, 1),
        lowest_score=round(min(scores) if scores else 0, 1),
        peak_segment_index=peak_idx,
        lowest_segment_index=low_idx,
        engagement_curve=[round(s, 1) for s in scores],
        segments=[s.to_dict() for s in scored_segments],
        high_segments=high_count,
        medium_segments=med_count,
        low_segments=low_count,
    )
    result.summary = _generate_summary(result)

    if on_progress:
        on_progress(100, "Timeline scoring complete")

    return result
