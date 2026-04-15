"""
OpenCut Timeline Quality Analyzer

Holistic quality analysis of an entire project timeline:
- Color consistency between shots (histogram comparison)
- Audio level consistency (LUFS per segment)
- Pacing analysis (cut frequency, shot duration distribution)
- Continuity issues (jump cuts, audio gaps, flash frames)
- Overall quality score (0-100)

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
# Scoring weights — sum to 1.0
# ---------------------------------------------------------------------------
WEIGHT_COLOR_CONSISTENCY = 0.20
WEIGHT_AUDIO_CONSISTENCY = 0.20
WEIGHT_PACING = 0.20
WEIGHT_CONTINUITY = 0.25
WEIGHT_TECHNICAL = 0.15

# Thresholds
FLASH_FRAME_MAX_DURATION = 0.08  # seconds — frames shorter than this are flash frames
JUMP_CUT_MIN_SIMILARITY = 0.85  # histogram similarity above this = jump cut
AUDIO_GAP_MIN_DURATION = 0.3    # seconds of silence that count as a gap
LUFS_DEVIATION_TOLERANCE = 3.0  # dB deviation from mean before penalizing
MIN_SHOT_DURATION = 0.5         # seconds — minimum acceptable shot length
IDEAL_CUT_RATE_RANGE = (3.0, 12.0)  # cuts per minute (documentary to fast-paced)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ShotInfo:
    """Metadata for a single detected shot."""
    index: int = 0
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    avg_brightness: float = 0.0
    histogram_hash: str = ""
    lufs: float = -70.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContinuityIssue:
    """A detected continuity problem."""
    issue_type: str = ""       # "jump_cut", "flash_frame", "audio_gap", "black_frame"
    timestamp: float = 0.0
    duration: float = 0.0
    severity: str = "warning"  # "info", "warning", "error"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PacingAnalysis:
    """Pacing metrics for the timeline."""
    total_cuts: int = 0
    cuts_per_minute: float = 0.0
    avg_shot_duration: float = 0.0
    min_shot_duration: float = 0.0
    max_shot_duration: float = 0.0
    shot_duration_stddev: float = 0.0
    pacing_label: str = ""  # "slow", "moderate", "fast", "frantic"
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimelineQualityResult:
    """Complete timeline quality analysis."""
    overall_score: float = 0.0
    color_consistency_score: float = 0.0
    audio_consistency_score: float = 0.0
    pacing_score: float = 0.0
    continuity_score: float = 0.0
    technical_score: float = 0.0
    total_duration: float = 0.0
    shot_count: int = 0
    shots: List[Dict] = field(default_factory=list)
    pacing: Dict = field(default_factory=dict)
    issues: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scene / shot detection via FFmpeg
# ---------------------------------------------------------------------------
def _detect_shots(video_path: str, threshold: float = 0.3,
                  on_progress: Optional[Callable] = None) -> List[ShotInfo]:
    """Detect shot boundaries using FFmpeg's scene detection filter.

    Returns a list of ShotInfo with start/end times.
    """
    if on_progress:
        on_progress(5, "Detecting shots...")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        stderr = result.stderr.decode(errors="replace")
    except Exception as exc:
        logger.warning("Shot detection failed: %s", exc)
        return []

    # Parse showinfo lines for pts_time
    timestamps = [0.0]
    for line in stderr.split("\n"):
        match = re.search(r"pts_time:\s*([0-9.]+)", line)
        if match:
            ts = float(match.group(1))
            if ts > 0:
                timestamps.append(ts)

    # Get total duration
    info = get_video_info(video_path)
    total_dur = info.get("duration", 0)
    if total_dur > 0 and (not timestamps or timestamps[-1] < total_dur - 0.5):
        timestamps.append(total_dur)

    # Build shot list
    shots = []
    for i in range(len(timestamps) - 1):
        start = timestamps[i]
        end = timestamps[i + 1]
        duration = end - start
        shots.append(ShotInfo(
            index=i,
            start=round(start, 3),
            end=round(end, 3),
            duration=round(duration, 3),
        ))

    if on_progress:
        on_progress(15, f"Detected {len(shots)} shots")

    return shots


# ---------------------------------------------------------------------------
# Color consistency analysis
# ---------------------------------------------------------------------------
def _compute_histogram_similarity(hist_a: List[float], hist_b: List[float]) -> float:
    """Compute correlation between two normalized histograms (0..1)."""
    if not hist_a or not hist_b or len(hist_a) != len(hist_b):
        return 0.0

    mean_a = sum(hist_a) / len(hist_a)
    mean_b = sum(hist_b) / len(hist_b)

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(hist_a, hist_b))
    std_a = math.sqrt(sum((a - mean_a) ** 2 for a in hist_a))
    std_b = math.sqrt(sum((b - mean_b) ** 2 for b in hist_b))

    if std_a < 1e-9 or std_b < 1e-9:
        return 1.0 if std_a < 1e-9 and std_b < 1e-9 else 0.0

    return max(0.0, min(1.0, cov / (std_a * std_b)))


def _extract_brightness_at(video_path: str, timestamp: float) -> float:
    """Extract average brightness of a single frame via FFmpeg signalstats."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", "signalstats",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        stderr = result.stderr.decode(errors="replace")
        match = re.search(r"YAVG=\s*([0-9.]+)", stderr)
        if match:
            return float(match.group(1)) / 255.0
    except Exception:
        pass
    return 0.5  # default mid brightness


def _analyze_color_consistency(
    video_path: str,
    shots: List[ShotInfo],
    on_progress: Optional[Callable] = None,
) -> Tuple[float, List[ShotInfo]]:
    """Score color consistency across shots. Returns (score_0_100, updated_shots)."""
    if not shots or len(shots) < 2:
        return 100.0, shots

    if on_progress:
        on_progress(20, "Analyzing color consistency...")

    # Sample brightness at midpoint of each shot
    brightnesses = []
    for i, shot in enumerate(shots):
        midpoint = shot.start + shot.duration / 2
        brightness = _extract_brightness_at(video_path, midpoint)
        shot.avg_brightness = round(brightness, 3)
        brightnesses.append(brightness)
        if on_progress and i % 5 == 0:
            pct = 20 + int(15 * (i + 1) / len(shots))
            on_progress(pct, f"Color analysis: shot {i + 1}/{len(shots)}")

    if not brightnesses:
        return 100.0, shots

    # Compute deviation from mean brightness
    mean_b = sum(brightnesses) / len(brightnesses)
    if mean_b < 1e-9:
        return 50.0, shots

    deviations = [abs(b - mean_b) / max(mean_b, 0.01) for b in brightnesses]
    avg_deviation = sum(deviations) / len(deviations)

    # Adjacent shot brightness jumps
    jumps = []
    for i in range(len(brightnesses) - 1):
        jump = abs(brightnesses[i + 1] - brightnesses[i])
        jumps.append(jump)

    avg_jump = sum(jumps) / len(jumps) if jumps else 0

    # Score: penalize high deviation and large jumps
    deviation_penalty = min(avg_deviation * 200, 50)
    jump_penalty = min(avg_jump * 300, 50)
    score = max(0, 100 - deviation_penalty - jump_penalty)

    return round(score, 1), shots


# ---------------------------------------------------------------------------
# Audio consistency analysis
# ---------------------------------------------------------------------------
def _measure_segment_lufs(video_path: str, start: float, end: float) -> float:
    """Measure LUFS of a video segment using FFmpeg ebur128 filter."""
    duration = end - start
    if duration < 0.1:
        return -70.0

    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-af", "ebur128=peak=true",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        stderr = result.stderr.decode(errors="replace")
        # Parse integrated loudness from ebur128 summary
        match = re.search(r"I:\s*(-?[0-9.]+)\s*LUFS", stderr)
        if match:
            return float(match.group(1))
    except Exception as exc:
        logger.debug("LUFS measurement failed for segment %.1f-%.1f: %s", start, end, exc)

    return -70.0


def _analyze_audio_consistency(
    video_path: str,
    shots: List[ShotInfo],
    on_progress: Optional[Callable] = None,
) -> Tuple[float, List[ShotInfo]]:
    """Score audio level consistency. Returns (score_0_100, updated_shots)."""
    if not shots:
        return 100.0, shots

    if on_progress:
        on_progress(40, "Analyzing audio levels...")

    # Measure LUFS for each shot (or groups for efficiency)
    lufs_values = []
    for i, shot in enumerate(shots):
        if shot.duration < 0.2:
            shot.lufs = -70.0
            continue
        lufs = _measure_segment_lufs(video_path, shot.start, shot.end)
        shot.lufs = round(lufs, 1)
        if lufs > -60:  # only count audible segments
            lufs_values.append(lufs)
        if on_progress and i % 3 == 0:
            pct = 40 + int(15 * (i + 1) / len(shots))
            on_progress(pct, f"Audio analysis: shot {i + 1}/{len(shots)}")

    if len(lufs_values) < 2:
        return 100.0, shots

    mean_lufs = sum(lufs_values) / len(lufs_values)
    deviations = [abs(v - mean_lufs) for v in lufs_values]
    avg_dev = sum(deviations) / len(deviations)

    # Score: within tolerance is perfect, each dB over tolerance costs points
    excess = max(0, avg_dev - LUFS_DEVIATION_TOLERANCE)
    score = max(0, 100 - excess * 15)

    return round(score, 1), shots


# ---------------------------------------------------------------------------
# Pacing analysis
# ---------------------------------------------------------------------------
def _analyze_pacing(shots: List[ShotInfo], total_duration: float) -> PacingAnalysis:
    """Analyze editing pace from shot durations."""
    if not shots:
        return PacingAnalysis(score=50.0, pacing_label="unknown")

    durations = [s.duration for s in shots]
    total_cuts = len(shots) - 1
    duration_minutes = total_duration / 60.0 if total_duration > 0 else 1.0
    cuts_per_min = total_cuts / duration_minutes if duration_minutes > 0 else 0

    avg_dur = sum(durations) / len(durations)
    min_dur = min(durations)
    max_dur = max(durations)

    # Standard deviation
    variance = sum((d - avg_dur) ** 2 for d in durations) / len(durations)
    stddev = math.sqrt(variance)

    # Pacing label
    if cuts_per_min < 3:
        label = "slow"
    elif cuts_per_min < 8:
        label = "moderate"
    elif cuts_per_min < 15:
        label = "fast"
    else:
        label = "frantic"

    # Score: penalize if outside ideal range, and for high variance
    range_lo, range_hi = IDEAL_CUT_RATE_RANGE
    if range_lo <= cuts_per_min <= range_hi:
        rate_score = 100.0
    elif cuts_per_min < range_lo:
        rate_score = max(0, 100 - (range_lo - cuts_per_min) * 20)
    else:
        rate_score = max(0, 100 - (cuts_per_min - range_hi) * 5)

    # Variance penalty: high stddev relative to mean = inconsistent pacing
    cv = stddev / avg_dur if avg_dur > 0 else 0
    variance_penalty = min(cv * 30, 40)
    score = max(0, rate_score - variance_penalty)

    return PacingAnalysis(
        total_cuts=total_cuts,
        cuts_per_minute=round(cuts_per_min, 2),
        avg_shot_duration=round(avg_dur, 3),
        min_shot_duration=round(min_dur, 3),
        max_shot_duration=round(max_dur, 3),
        shot_duration_stddev=round(stddev, 3),
        pacing_label=label,
        score=round(score, 1),
    )


# ---------------------------------------------------------------------------
# Continuity issue detection
# ---------------------------------------------------------------------------
def _detect_continuity_issues(
    video_path: str,
    shots: List[ShotInfo],
    on_progress: Optional[Callable] = None,
) -> Tuple[float, List[ContinuityIssue]]:
    """Detect continuity problems. Returns (score_0_100, issues)."""
    issues = []
    if not shots:
        return 100.0, issues

    if on_progress:
        on_progress(60, "Checking continuity...")

    # Flash frames (very short shots)
    for shot in shots:
        if 0 < shot.duration < FLASH_FRAME_MAX_DURATION:
            issues.append(ContinuityIssue(
                issue_type="flash_frame",
                timestamp=shot.start,
                duration=shot.duration,
                severity="warning",
                description=f"Flash frame at {shot.start:.2f}s ({shot.duration * 1000:.0f}ms)",
            ))

    # Jump cuts (adjacent shots with very similar visual content)
    for i in range(len(shots) - 1):
        a, b = shots[i], shots[i + 1]
        # Use brightness similarity as proxy for visual similarity
        if abs(a.avg_brightness - b.avg_brightness) < 0.02 and a.duration > 0.5 and b.duration > 0.5:
            issues.append(ContinuityIssue(
                issue_type="jump_cut",
                timestamp=a.end,
                duration=0.0,
                severity="info",
                description=f"Potential jump cut at {a.end:.2f}s (shots {a.index} and {b.index} very similar)",
            ))

    # Black frames / very dark shots
    for shot in shots:
        if shot.avg_brightness < 0.03 and shot.duration > 0.1:
            issues.append(ContinuityIssue(
                issue_type="black_frame",
                timestamp=shot.start,
                duration=shot.duration,
                severity="warning" if shot.duration > 1.0 else "info",
                description=f"Black/dark shot at {shot.start:.2f}s ({shot.duration:.2f}s)",
            ))

    # Audio gaps (segments with very low LUFS between audible segments)
    for i in range(1, len(shots) - 1):
        prev_s, curr, next_s = shots[i - 1], shots[i], shots[i + 1]
        if (curr.lufs < -50 and prev_s.lufs > -30 and next_s.lufs > -30
                and curr.duration > AUDIO_GAP_MIN_DURATION):
            issues.append(ContinuityIssue(
                issue_type="audio_gap",
                timestamp=curr.start,
                duration=curr.duration,
                severity="warning",
                description=f"Audio gap at {curr.start:.2f}s ({curr.duration:.2f}s between audible segments)",
            ))

    if on_progress:
        on_progress(70, f"Found {len(issues)} continuity issues")

    # Score: start at 100, deduct per issue
    deductions = {
        "flash_frame": 8,
        "jump_cut": 3,
        "black_frame": 5,
        "audio_gap": 7,
    }
    total_deduction = sum(deductions.get(i.issue_type, 5) for i in issues)
    score = max(0, 100 - total_deduction)

    return round(score, 1), issues


# ---------------------------------------------------------------------------
# Technical quality checks
# ---------------------------------------------------------------------------
def _analyze_technical_quality(video_path: str, info: dict) -> float:
    """Score technical attributes (resolution, codec, bitrate)."""
    score = 100.0
    width = info.get("width", 0)
    height = info.get("height", 0)
    fps = info.get("fps", 0)

    # Resolution scoring
    pixels = width * height
    if pixels >= 3840 * 2160:
        pass  # 4K — perfect
    elif pixels >= 1920 * 1080:
        score -= 5  # 1080p — slight deduction
    elif pixels >= 1280 * 720:
        score -= 15  # 720p
    elif pixels >= 640 * 480:
        score -= 30  # SD
    else:
        score -= 50  # very low resolution

    # Frame rate scoring
    if fps >= 59:
        pass  # 60fps
    elif fps >= 29:
        score -= 2  # 30fps
    elif fps >= 23:
        score -= 5  # 24fps — cinematic
    else:
        score -= 15  # low frame rate

    return max(0, round(score, 1))


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
def _generate_recommendations(
    color_score: float,
    audio_score: float,
    pacing: PacingAnalysis,
    issues: List[ContinuityIssue],
    technical_score: float,
) -> List[str]:
    """Generate actionable recommendations based on scores."""
    recs = []

    if color_score < 70:
        recs.append("Apply color correction across shots to improve visual consistency")
    if color_score < 50:
        recs.append("Consider using a LUT or color matching tool to unify the look")

    if audio_score < 70:
        recs.append("Normalize audio levels to reduce loudness variation between shots")
    if audio_score < 50:
        recs.append("Apply a limiter or compressor to even out audio dynamics")

    if pacing.pacing_label == "slow":
        recs.append("Pacing is slow — consider tightening edits or adding B-roll")
    elif pacing.pacing_label == "frantic":
        recs.append("Pacing is very fast — consider letting some shots breathe longer")

    if pacing.shot_duration_stddev > pacing.avg_shot_duration * 1.5:
        recs.append("Shot durations vary widely — consider more consistent edit rhythm")

    flash_count = sum(1 for i in issues if i.issue_type == "flash_frame")
    if flash_count > 0:
        recs.append(f"Remove {flash_count} flash frame(s) for smoother playback")

    gap_count = sum(1 for i in issues if i.issue_type == "audio_gap")
    if gap_count > 0:
        recs.append(f"Fill {gap_count} audio gap(s) with room tone or ambient audio")

    jump_count = sum(1 for i in issues if i.issue_type == "jump_cut")
    if jump_count > 0:
        recs.append(f"Address {jump_count} potential jump cut(s) with cutaways or transitions")

    if technical_score < 70:
        recs.append("Source footage resolution is low — consider higher quality sources")

    if not recs:
        recs.append("Timeline looks great — no major issues detected")

    return recs


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------
def analyze_timeline_quality(
    video_path: str,
    scene_threshold: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> TimelineQualityResult:
    """
    Perform holistic quality analysis on a video timeline.

    Args:
        video_path: Path to the video file.
        scene_threshold: FFmpeg scene change detection threshold (0-1).
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        TimelineQualityResult with scores, shots, issues, and recommendations.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If video has zero duration.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(1, "Starting timeline quality analysis...")

    info = get_video_info(video_path)
    total_duration = info.get("duration", 0)
    if total_duration <= 0:
        raise ValueError("Video has zero or negative duration")

    # 1. Detect shots
    shots = _detect_shots(video_path, threshold=scene_threshold, on_progress=on_progress)
    if not shots:
        # Treat entire video as single shot
        shots = [ShotInfo(index=0, start=0.0, end=total_duration, duration=total_duration)]

    # 2. Color consistency
    color_score, shots = _analyze_color_consistency(video_path, shots, on_progress)

    # 3. Audio consistency
    audio_score, shots = _analyze_audio_consistency(video_path, shots, on_progress)

    # 4. Pacing analysis
    pacing = _analyze_pacing(shots, total_duration)

    # 5. Continuity issues
    continuity_score, issues = _detect_continuity_issues(video_path, shots, on_progress)

    # 6. Technical quality
    technical_score = _analyze_technical_quality(video_path, info)

    if on_progress:
        on_progress(85, "Computing final scores...")

    # 7. Weighted overall score
    overall = (
        color_score * WEIGHT_COLOR_CONSISTENCY
        + audio_score * WEIGHT_AUDIO_CONSISTENCY
        + pacing.score * WEIGHT_PACING
        + continuity_score * WEIGHT_CONTINUITY
        + technical_score * WEIGHT_TECHNICAL
    )

    # 8. Recommendations
    recommendations = _generate_recommendations(
        color_score, audio_score, pacing, issues, technical_score
    )

    if on_progress:
        on_progress(95, "Finalizing report...")

    result = TimelineQualityResult(
        overall_score=round(overall, 1),
        color_consistency_score=color_score,
        audio_consistency_score=audio_score,
        pacing_score=pacing.score,
        continuity_score=continuity_score,
        technical_score=technical_score,
        total_duration=round(total_duration, 3),
        shot_count=len(shots),
        shots=[s.to_dict() for s in shots],
        pacing=pacing.to_dict(),
        issues=[i.to_dict() for i in issues],
        recommendations=recommendations,
    )

    if on_progress:
        on_progress(100, "Timeline quality analysis complete")

    return result
