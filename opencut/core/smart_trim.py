"""
Intelligent Trim Point Detection (Category 74)

Analyze clips to find optimal in/out points: skip pre-roll silence/slate,
find first speech onset, detect natural ending (last word + room tone tail),
identify chapter boundaries. Uses FFmpeg audio analysis for silence detection,
scene detection for visual cues.

Modes: tight (minimal padding), broadcast (standard handles), social (hook-first).
Batch mode: process multiple clips.
"""

import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRIM_MODES = {
    "tight": {
        "pre_pad_s": 0.1,
        "post_pad_s": 0.3,
        "silence_threshold_db": -35.0,
        "min_speech_duration_s": 0.3,
        "scene_detect_threshold": 0.4,
        "description": "Minimal padding, cuts close to content",
    },
    "broadcast": {
        "pre_pad_s": 1.0,
        "post_pad_s": 1.5,
        "silence_threshold_db": -40.0,
        "min_speech_duration_s": 0.5,
        "scene_detect_threshold": 0.3,
        "description": "Standard handles for broadcast editing",
    },
    "social": {
        "pre_pad_s": 0.0,
        "post_pad_s": 0.2,
        "silence_threshold_db": -30.0,
        "min_speech_duration_s": 0.2,
        "scene_detect_threshold": 0.5,
        "description": "Hook-first, minimal padding for social media",
    },
}

MAX_BATCH_SIZE = 200
MIN_CLIP_DURATION = 0.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrimPoint:
    """A potential trim point in a clip."""
    time: float = 0.0
    point_type: str = "speech_onset"  # speech_onset, speech_end, scene_change, silence_start, silence_end
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 3),
            "point_type": self.point_type,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


@dataclass
class SmartTrimResult:
    """Result of smart trim analysis for a single clip."""
    file_path: str = ""
    original_duration: float = 0.0
    suggested_in: float = 0.0
    suggested_out: float = 0.0
    trim_reason: str = ""
    confidence: float = 0.0
    alternative_points: List[TrimPoint] = field(default_factory=list)
    silence_regions: List[Dict] = field(default_factory=list)
    scene_changes: List[float] = field(default_factory=list)
    speech_regions: List[Dict] = field(default_factory=list)
    trimmed_duration: float = 0.0
    mode: str = "tight"

    @property
    def time_saved(self) -> float:
        return max(0.0, self.original_duration - self.trimmed_duration)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "original_duration": round(self.original_duration, 3),
            "suggested_in": round(self.suggested_in, 3),
            "suggested_out": round(self.suggested_out, 3),
            "trimmed_duration": round(self.trimmed_duration, 3),
            "time_saved": round(self.time_saved, 3),
            "trim_reason": self.trim_reason,
            "confidence": round(self.confidence, 3),
            "mode": self.mode,
            "alternative_points": [p.to_dict() for p in self.alternative_points],
            "silence_region_count": len(self.silence_regions),
            "scene_change_count": len(self.scene_changes),
            "speech_region_count": len(self.speech_regions),
        }


@dataclass
class BatchTrimResult:
    """Result of batch smart trim processing."""
    results: List[SmartTrimResult] = field(default_factory=list)
    total_original_duration: float = 0.0
    total_trimmed_duration: float = 0.0
    total_time_saved: float = 0.0
    clips_processed: int = 0
    clips_failed: int = 0
    errors: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_original_duration": round(self.total_original_duration, 3),
            "total_trimmed_duration": round(self.total_trimmed_duration, 3),
            "total_time_saved": round(self.total_time_saved, 3),
            "clips_processed": self.clips_processed,
            "clips_failed": self.clips_failed,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------
def _detect_silence_regions(
    file_path: str,
    threshold_db: float = -35.0,
    min_duration: float = 0.3,
) -> List[Dict]:
    """Detect silence regions in audio using FFmpeg silencedetect."""
    cmd = [
        get_ffmpeg_path(), "-i", file_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    segments = []
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")

        starts = re.findall(r"silence_start:\s*([\d.]+)", stderr)
        ends = re.findall(r"silence_end:\s*([\d.]+)", stderr)

        for i in range(min(len(starts), len(ends))):
            start = float(starts[i])
            end = float(ends[i])
            segments.append({
                "start": start,
                "end": end,
                "duration": end - start,
            })
    except Exception as exc:
        logger.debug("Silence detection failed for %s: %s", file_path, exc)

    return segments


def _detect_speech_regions(
    silence_regions: List[Dict],
    total_duration: float,
) -> List[Dict]:
    """Derive speech regions from silence regions."""
    if total_duration <= 0:
        return []

    speech = []
    prev_end = 0.0

    for seg in silence_regions:
        if seg["start"] > prev_end + 0.05:
            speech.append({
                "start": prev_end,
                "end": seg["start"],
                "duration": seg["start"] - prev_end,
            })
        prev_end = seg["end"]

    if total_duration > prev_end + 0.05:
        speech.append({
            "start": prev_end,
            "end": total_duration,
            "duration": total_duration - prev_end,
        })

    return speech


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------
def _detect_scene_changes(
    file_path: str,
    threshold: float = 0.4,
    max_scenes: int = 100,
) -> List[float]:
    """Detect scene changes using FFmpeg scene filter."""
    cmd = [
        get_ffmpeg_path(), "-i", file_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    timestamps = []
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")

        # Parse pts_time from showinfo output
        times = re.findall(r"pts_time:\s*([\d.]+)", stderr)
        timestamps = [float(t) for t in times[:max_scenes]]
    except Exception as exc:
        logger.debug("Scene detection failed for %s: %s", file_path, exc)

    return timestamps


# ---------------------------------------------------------------------------
# Trim point detection
# ---------------------------------------------------------------------------
def _find_first_speech_onset(
    speech_regions: List[Dict],
    min_speech_duration: float = 0.3,
) -> Optional[TrimPoint]:
    """Find the first substantial speech onset."""
    for region in speech_regions:
        if region["duration"] >= min_speech_duration:
            return TrimPoint(
                time=region["start"],
                point_type="speech_onset",
                confidence=min(1.0, region["duration"] / 2.0),
                reason=f"First speech at {region['start']:.2f}s "
                       f"(duration {region['duration']:.2f}s)",
            )
    return None


def _find_last_speech_end(
    speech_regions: List[Dict],
    min_speech_duration: float = 0.3,
) -> Optional[TrimPoint]:
    """Find the end of the last substantial speech segment."""
    for region in reversed(speech_regions):
        if region["duration"] >= min_speech_duration:
            return TrimPoint(
                time=region["end"],
                point_type="speech_end",
                confidence=min(1.0, region["duration"] / 2.0),
                reason=f"Last speech ends at {region['end']:.2f}s "
                       f"(duration {region['duration']:.2f}s)",
            )
    return None


def _find_hook_point(
    speech_regions: List[Dict],
    scene_changes: List[float],
    total_duration: float,
) -> Optional[TrimPoint]:
    """Find the best hook point for social media (first engaging moment)."""
    # Prefer first scene change near speech
    for sc_time in scene_changes:
        if sc_time < total_duration * 0.3:
            # Check if there's speech near this scene change
            for region in speech_regions:
                if abs(region["start"] - sc_time) < 1.0:
                    return TrimPoint(
                        time=min(sc_time, region["start"]),
                        point_type="hook",
                        confidence=0.8,
                        reason=f"Hook: scene change + speech at {sc_time:.2f}s",
                    )

    # Fallback: first speech onset
    return _find_first_speech_onset(speech_regions, min_speech_duration=0.2)


def _generate_alternative_points(
    speech_regions: List[Dict],
    scene_changes: List[float],
    silence_regions: List[Dict],
    total_duration: float,
) -> List[TrimPoint]:
    """Generate alternative trim points for user selection."""
    alternatives = []

    # Scene change points
    for sc_time in scene_changes[:10]:
        alternatives.append(TrimPoint(
            time=sc_time,
            point_type="scene_change",
            confidence=0.7,
            reason=f"Scene change at {sc_time:.2f}s",
        ))

    # Speech onset points (first 5)
    for i, region in enumerate(speech_regions[:5]):
        if region["duration"] >= 0.3:
            alternatives.append(TrimPoint(
                time=region["start"],
                point_type="speech_onset",
                confidence=min(1.0, region["duration"] / 3.0),
                reason=f"Speech onset #{i + 1} at {region['start']:.2f}s",
            ))

    # Speech end points (last 5)
    for i, region in enumerate(reversed(speech_regions[-5:])):
        if region["duration"] >= 0.3:
            alternatives.append(TrimPoint(
                time=region["end"],
                point_type="speech_end",
                confidence=min(1.0, region["duration"] / 3.0),
                reason=f"Speech end at {region['end']:.2f}s",
            ))

    # Silence boundaries (good cut points)
    for seg in silence_regions[:10]:
        if seg["duration"] >= 0.5:
            alternatives.append(TrimPoint(
                time=seg["start"],
                point_type="silence_start",
                confidence=0.6,
                reason=f"Silence starts at {seg['start']:.2f}s "
                       f"({seg['duration']:.2f}s long)",
            ))

    # Sort by time
    alternatives.sort(key=lambda p: p.time)
    return alternatives


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def smart_trim(
    file_path: str,
    mode: str = "tight",
    on_progress: Optional[Callable] = None,
) -> SmartTrimResult:
    """Analyze a clip and find optimal trim points.

    Args:
        file_path: Path to the media file.
        mode: Trim mode — 'tight', 'broadcast', or 'social'.
        on_progress: Callback(pct) for progress updates.

    Returns:
        SmartTrimResult with suggested in/out points and alternatives.
    """
    if mode not in TRIM_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use: {', '.join(TRIM_MODES.keys())}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mode_settings = TRIM_MODES[mode]

    if on_progress:
        on_progress(5)

    # Get clip duration
    info = get_video_info(file_path)
    total_duration = info.get("duration", 0.0)

    if total_duration < MIN_CLIP_DURATION:
        return SmartTrimResult(
            file_path=file_path,
            original_duration=total_duration,
            suggested_in=0.0,
            suggested_out=total_duration,
            trimmed_duration=total_duration,
            trim_reason="Clip too short to trim",
            confidence=1.0,
            mode=mode,
        )

    if on_progress:
        on_progress(15)

    # Detect silence
    silence_regions = _detect_silence_regions(
        file_path,
        threshold_db=mode_settings["silence_threshold_db"],
    )

    if on_progress:
        on_progress(40)

    # Derive speech regions
    speech_regions = _detect_speech_regions(silence_regions, total_duration)

    if on_progress:
        on_progress(55)

    # Detect scene changes
    scene_changes = _detect_scene_changes(
        file_path,
        threshold=mode_settings["scene_detect_threshold"],
    )

    if on_progress:
        on_progress(70)

    # Find optimal in point
    pre_pad = mode_settings["pre_pad_s"]
    post_pad = mode_settings["post_pad_s"]
    min_speech = mode_settings["min_speech_duration_s"]

    if mode == "social":
        in_point_tp = _find_hook_point(speech_regions, scene_changes, total_duration)
    else:
        in_point_tp = _find_first_speech_onset(speech_regions, min_speech)

    # Find optimal out point
    out_point_tp = _find_last_speech_end(speech_regions, min_speech)

    # Apply padding
    if in_point_tp:
        suggested_in = max(0.0, in_point_tp.time - pre_pad)
        in_reason = in_point_tp.reason
        in_confidence = in_point_tp.confidence
    else:
        suggested_in = 0.0
        in_reason = "No speech detected, using start"
        in_confidence = 0.3

    if out_point_tp:
        suggested_out = min(total_duration, out_point_tp.time + post_pad)
        out_reason = out_point_tp.reason
        out_confidence = out_point_tp.confidence
    else:
        suggested_out = total_duration
        out_reason = "No speech detected, using end"
        out_confidence = 0.3

    # Ensure in < out
    if suggested_in >= suggested_out:
        suggested_in = 0.0
        suggested_out = total_duration
        in_reason = "Fallback to full clip"
        in_confidence = 0.2

    trimmed_duration = suggested_out - suggested_in
    avg_confidence = (in_confidence + out_confidence) / 2.0

    # Generate alternatives
    alternatives = _generate_alternative_points(
        speech_regions, scene_changes, silence_regions, total_duration,
    )

    if on_progress:
        on_progress(90)

    trim_reason = f"In: {in_reason} | Out: {out_reason}"

    result = SmartTrimResult(
        file_path=file_path,
        original_duration=total_duration,
        suggested_in=suggested_in,
        suggested_out=suggested_out,
        trim_reason=trim_reason,
        confidence=avg_confidence,
        alternative_points=alternatives,
        silence_regions=silence_regions,
        scene_changes=scene_changes,
        speech_regions=speech_regions,
        trimmed_duration=trimmed_duration,
        mode=mode,
    )

    if on_progress:
        on_progress(100)

    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def batch_smart_trim(
    file_paths: List[str],
    mode: str = "tight",
    on_progress: Optional[Callable] = None,
) -> BatchTrimResult:
    """Process multiple clips with smart trim.

    Args:
        file_paths: List of media file paths.
        mode: Trim mode for all clips.
        on_progress: Callback(pct) for progress updates.

    Returns:
        BatchTrimResult with per-clip results and totals.
    """
    if not file_paths:
        raise ValueError("No files provided")

    if len(file_paths) > MAX_BATCH_SIZE:
        raise ValueError(f"Too many files ({len(file_paths)}). Max: {MAX_BATCH_SIZE}")

    if mode not in TRIM_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Use: {', '.join(TRIM_MODES.keys())}")

    results = []
    errors = []
    total = len(file_paths)

    for i, fp in enumerate(file_paths):
        if on_progress:
            pct = int((i / max(total, 1)) * 95)
            on_progress(pct)

        try:
            result = smart_trim(fp, mode=mode)
            results.append(result)
        except Exception as exc:
            logger.warning("Smart trim failed for %s: %s", fp, exc)
            errors.append({
                "file_path": fp,
                "error": str(exc),
            })

    total_original = sum(r.original_duration for r in results)
    total_trimmed = sum(r.trimmed_duration for r in results)

    if on_progress:
        on_progress(100)

    return BatchTrimResult(
        results=results,
        total_original_duration=total_original,
        total_trimmed_duration=total_trimmed,
        total_time_saved=total_original - total_trimmed,
        clips_processed=len(results),
        clips_failed=len(errors),
        errors=errors,
    )
