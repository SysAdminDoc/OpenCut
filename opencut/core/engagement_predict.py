"""
OpenCut AI Engagement Prediction

Heuristic-based engagement scoring for video content. Analyzes hook strength,
audio energy, speech pace, and visual change rate to predict retention
and virality without requiring ML models.

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import math
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Scoring weights and thresholds
# ---------------------------------------------------------------------------
_HOOK_WINDOW = 3.0          # First N seconds for hook analysis
_ENERGY_WINDOW = 5.0        # Window size for energy curve sampling
_RETENTION_INTERVAL = 10.0  # Seconds between retention curve points


@dataclass
class EngagementResult:
    """Complete engagement prediction results."""
    hook_score: int = 50           # 0-100 score for first 3 seconds
    retention_curve: List[Dict] = field(default_factory=list)
    virality_score: int = 50       # 0-100 overall virality prediction
    overall_score: int = 50        # 0-100 combined engagement score
    suggestions: List[str] = field(default_factory=list)
    analysis: Dict = field(default_factory=dict)


def predict_engagement(
    input_path: str,
    transcript_text: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Predict video engagement using heuristic analysis.

    Analyzes first 3 seconds (hook), audio energy curve, speech rate,
    and visual change rate to produce engagement scores.

    Args:
        input_path: Source video file.
        transcript_text: Optional transcript for speech rate analysis.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with hook_score, retention_curve, virality_score, suggestions.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    info = get_video_info(input_path)
    duration = info.get("duration", 0.0)
    info.get("fps", 30.0)

    if on_progress:
        on_progress(5, "Analyzing hook (first 3 seconds)...")

    # --- Hook analysis (first 3 seconds) ---
    hook_audio_energy = _measure_audio_energy(input_path, 0.0, min(_HOOK_WINDOW, duration))
    hook_visual_change = _measure_visual_change(input_path, 0.0, min(_HOOK_WINDOW, duration))

    if on_progress:
        on_progress(25, "Measuring audio energy curve...")

    # --- Full audio energy curve ---
    energy_curve = _build_energy_curve(input_path, duration)

    if on_progress:
        on_progress(45, "Analyzing visual change rate...")

    # --- Scene change rate ---
    scene_change_rate = _measure_scene_change_rate(input_path)

    if on_progress:
        on_progress(60, "Analyzing speech pace...")

    # --- Speech rate from transcript ---
    speech_pace = _analyze_speech_pace(transcript_text, duration) if transcript_text else None

    if on_progress:
        on_progress(75, "Computing scores...")

    # --- Compute hook score ---
    hook_score = _compute_hook_score(hook_audio_energy, hook_visual_change, duration)

    # --- Build retention curve ---
    retention_curve = _build_retention_curve(energy_curve, scene_change_rate, duration)

    # --- Compute virality score ---
    virality_score = _compute_virality_score(
        hook_score, energy_curve, scene_change_rate, speech_pace, duration
    )

    # --- Overall score ---
    overall_score = int(hook_score * 0.35 + virality_score * 0.35 +
                        (sum(p["predicted_retention_pct"] for p in retention_curve) /
                         max(len(retention_curve), 1)) * 0.30)
    overall_score = max(0, min(100, overall_score))

    if on_progress:
        on_progress(90, "Generating suggestions...")

    # --- Suggestions ---
    suggestions = _generate_suggestions(
        hook_score, hook_audio_energy, hook_visual_change,
        energy_curve, scene_change_rate, speech_pace, duration
    )

    analysis = {
        "hook_audio_energy": round(hook_audio_energy, 2),
        "hook_visual_change": round(hook_visual_change, 2),
        "avg_audio_energy": round(sum(e["energy"] for e in energy_curve) / max(len(energy_curve), 1), 2),
        "scene_changes_per_minute": round(scene_change_rate, 2),
        "duration": round(duration, 2),
    }
    if speech_pace is not None:
        analysis["speech_words_per_minute"] = round(speech_pace, 1)

    if on_progress:
        on_progress(100, f"Engagement prediction complete: {overall_score}/100")

    result = EngagementResult(
        hook_score=hook_score,
        retention_curve=retention_curve,
        virality_score=virality_score,
        overall_score=overall_score,
        suggestions=suggestions,
        analysis=analysis,
    )
    return _result_to_dict(result)


def _measure_audio_energy(input_path: str, start: float, end: float) -> float:
    """Measure RMS audio energy for a time range. Returns 0-100 normalized."""
    duration = end - start
    if duration <= 0:
        return 0.0

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-ss", str(start), "-t", str(duration),
        "-i", input_path,
        "-af", "astats=metadata=1:reset=0,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
    except _sp.TimeoutExpired:
        return 30.0  # Default moderate energy

    # Parse RMS values
    rms_values = re.findall(r"lavfi\.astats\.Overall\.RMS_level=([-\d.]+)", result.stderr)
    if not rms_values:
        return 30.0

    rms_avg = sum(float(v) for v in rms_values) / len(rms_values)
    # RMS is typically -70 to 0 dB. Normalize to 0-100.
    # -40 dB = quiet, -20 dB = moderate, -10 dB = loud
    energy = max(0.0, min(100.0, (rms_avg + 50.0) * 2.0))
    return energy


def _measure_visual_change(input_path: str, start: float, end: float) -> float:
    """Measure visual change rate in a time window. Returns 0-100 normalized."""
    duration = end - start
    if duration <= 0:
        return 0.0

    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-ss", str(start), "-t", str(duration),
        "-i", input_path,
        "-vf", "select='gt(scene,0.1)',showinfo",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
    except _sp.TimeoutExpired:
        return 30.0

    # Count scene changes
    changes = len(re.findall(r"pts_time:", result.stderr))
    # Normalize: 0 changes in 3s = low (0), 3+ changes = high (100)
    rate = changes / max(duration, 0.1)
    visual_score = max(0.0, min(100.0, rate * 50.0))
    return visual_score


def _build_energy_curve(input_path: str, duration: float) -> List[Dict]:
    """Build an energy curve by sampling audio energy across the video."""
    if duration <= 0:
        return [{"time": 0.0, "energy": 30.0}]

    curve = []
    step = max(_ENERGY_WINDOW, duration / 20.0)  # At most 20 samples
    t = 0.0

    # Measure full-file RMS in one pass for efficiency
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-i", input_path,
        "-af", f"astats=metadata=1:reset={int(step * 44100)},"
               f"ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    except _sp.TimeoutExpired:
        # Return flat curve
        while t < duration:
            curve.append({"time": round(t, 1), "energy": 30.0})
            t += step
        return curve

    rms_values = re.findall(r"lavfi\.astats\.Overall\.RMS_level=([-\d.]+)", result.stderr)

    # Distribute parsed values across time
    if rms_values:
        samples_per_point = max(1, len(rms_values) // max(int(duration / step), 1))
        idx = 0
        while t < duration and idx < len(rms_values):
            chunk = rms_values[idx:idx + samples_per_point]
            if chunk:
                avg_rms = sum(float(v) for v in chunk) / len(chunk)
                energy = max(0.0, min(100.0, (avg_rms + 50.0) * 2.0))
            else:
                energy = 30.0
            curve.append({"time": round(t, 1), "energy": round(energy, 1)})
            t += step
            idx += samples_per_point
    else:
        while t < duration:
            curve.append({"time": round(t, 1), "energy": 30.0})
            t += step

    return curve if curve else [{"time": 0.0, "energy": 30.0}]


def _measure_scene_change_rate(input_path: str) -> float:
    """Measure overall scene change rate (changes per minute)."""
    try:
        from opencut.core.scene_detect import detect_scenes
        result = detect_scenes(input_path, threshold=0.3, min_scene_length=0.5)
        if result.duration > 0:
            return (result.total_scenes / (result.duration / 60.0))
        return 0.0
    except Exception as e:
        logger.debug("Scene change measurement failed: %s", e)
        return 5.0  # Default moderate rate


def _analyze_speech_pace(transcript_text: Optional[str], duration: float) -> Optional[float]:
    """Calculate speech rate in words per minute from transcript."""
    if not transcript_text or duration <= 0:
        return None

    # Strip SRT formatting
    cleaned = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", transcript_text)
    cleaned = re.sub(r"^\d+\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    words = re.findall(r"\b\w+\b", cleaned)
    word_count = len(words)

    if word_count == 0:
        return None

    wpm = word_count / (duration / 60.0)
    return wpm


def _compute_hook_score(audio_energy: float, visual_change: float, duration: float) -> int:
    """Compute hook score (0-100) from first 3 seconds analysis."""
    if duration < 1.0:
        return 30  # Very short clips get a low base score

    # Weight: audio energy 40%, visual change 40%, instant start bonus 20%
    audio_component = audio_energy * 0.4
    visual_component = visual_change * 0.4

    # Bonus for immediate activity (energy > 40 means something is happening)
    instant_start = 20.0 if (audio_energy > 40 or visual_change > 40) else 5.0

    score = int(audio_component + visual_component + instant_start)
    return max(0, min(100, score))


def _build_retention_curve(
    energy_curve: List[Dict],
    scene_change_rate: float,
    duration: float,
) -> List[Dict]:
    """Build predicted retention curve based on energy and pacing."""
    if duration <= 0:
        return [{"timestamp": 0.0, "predicted_retention_pct": 50.0}]

    curve = []
    step = max(_RETENTION_INTERVAL, duration / 20.0)
    t = 0.0

    # Base retention decays over time (natural drop-off)
    while t <= duration:
        progress = t / max(duration, 1.0)

        # Natural decay: starts at ~95%, drops to ~30% by end
        base_retention = 95.0 * math.exp(-1.2 * progress)

        # Energy boost: high energy sections retain better
        energy_at_t = _interpolate_energy(energy_curve, t)
        energy_boost = (energy_at_t - 30.0) / 70.0 * 15.0  # -15 to +15

        # Pacing boost: good pacing retains
        pacing_boost = 0.0
        if 5 <= scene_change_rate <= 25:
            pacing_boost = 5.0
        elif scene_change_rate > 25:
            pacing_boost = -3.0  # Too fast can fatigue

        retention = max(5.0, min(100.0, base_retention + energy_boost + pacing_boost))

        curve.append({
            "timestamp": round(t, 1),
            "predicted_retention_pct": round(retention, 1),
        })
        t += step

    return curve


def _interpolate_energy(energy_curve: List[Dict], t: float) -> float:
    """Interpolate energy value at time t from energy curve."""
    if not energy_curve:
        return 30.0
    if t <= energy_curve[0]["time"]:
        return energy_curve[0]["energy"]
    if t >= energy_curve[-1]["time"]:
        return energy_curve[-1]["energy"]

    for i in range(len(energy_curve) - 1):
        if energy_curve[i]["time"] <= t <= energy_curve[i + 1]["time"]:
            span = energy_curve[i + 1]["time"] - energy_curve[i]["time"]
            if span <= 0:
                return energy_curve[i]["energy"]
            frac = (t - energy_curve[i]["time"]) / span
            return energy_curve[i]["energy"] + frac * (energy_curve[i + 1]["energy"] - energy_curve[i]["energy"])

    return energy_curve[-1]["energy"]


def _compute_virality_score(
    hook_score: int,
    energy_curve: List[Dict],
    scene_change_rate: float,
    speech_pace: Optional[float],
    duration: float,
) -> int:
    """Compute virality score (0-100) based on multiple signals."""
    score = 0.0

    # Hook is critical for virality (40% weight)
    score += hook_score * 0.4

    # Energy consistency (20% weight) -- high average energy is good
    if energy_curve:
        avg_energy = sum(e["energy"] for e in energy_curve) / len(energy_curve)
        score += (avg_energy / 100.0) * 20.0

    # Pacing (20% weight) -- moderate-fast pacing is viral
    if scene_change_rate >= 10 and scene_change_rate <= 30:
        score += 20.0
    elif scene_change_rate >= 5:
        score += 12.0
    elif scene_change_rate >= 2:
        score += 6.0

    # Speech pace (10% weight) -- 120-180 WPM is engaging
    if speech_pace is not None:
        if 120 <= speech_pace <= 180:
            score += 10.0
        elif 100 <= speech_pace <= 200:
            score += 6.0
        elif speech_pace > 0:
            score += 3.0

    # Duration (10% weight) -- short-form (<60s) is more viral
    if 15 <= duration <= 60:
        score += 10.0
    elif 60 < duration <= 180:
        score += 7.0
    elif 180 < duration <= 600:
        score += 4.0
    else:
        score += 2.0

    return max(0, min(100, int(score)))


def _generate_suggestions(
    hook_score: int,
    hook_audio: float,
    hook_visual: float,
    energy_curve: List[Dict],
    scene_change_rate: float,
    speech_pace: Optional[float],
    duration: float,
) -> List[str]:
    """Generate engagement improvement suggestions."""
    suggestions = []

    # Hook suggestions
    if hook_score < 40:
        suggestions.append(
            "Weak hook: The first 3 seconds lack energy. Start with a compelling "
            "visual, question, or audio cue to grab attention immediately."
        )
    if hook_audio < 20:
        suggestions.append(
            "Low audio energy at start. Consider adding music, a sound effect, "
            "or starting with speech to engage viewers instantly."
        )
    if hook_visual < 20:
        suggestions.append(
            "Low visual activity at start. Open with motion, a close-up, "
            "or a visually striking frame."
        )

    # Energy suggestions
    if energy_curve:
        avg_energy = sum(e["energy"] for e in energy_curve) / len(energy_curve)
        if avg_energy < 25:
            suggestions.append(
                "Overall audio energy is low. Consider adding background music "
                "or increasing audio levels to maintain engagement."
            )

        # Check for energy drops
        for i in range(1, len(energy_curve)):
            if energy_curve[i]["energy"] < energy_curve[i - 1]["energy"] * 0.4:
                t = energy_curve[i]["time"]
                suggestions.append(
                    f"Energy drop at {t:.0f}s -- viewers may lose interest. "
                    f"Consider adding a visual change or audio cue here."
                )
                break  # Only flag the first major drop

    # Pacing suggestions
    if scene_change_rate < 3:
        suggestions.append(
            "Very few cuts detected. Add more visual variety with B-roll, "
            "cutaways, or angle changes to maintain viewer attention."
        )
    elif scene_change_rate > 40:
        suggestions.append(
            "Very rapid cutting may fatigue viewers. Consider letting key "
            "shots breathe for 2-3 seconds longer."
        )

    # Speech pace
    if speech_pace is not None:
        if speech_pace < 100:
            suggestions.append(
                f"Speech pace is slow ({speech_pace:.0f} WPM). "
                f"Consider speeding up delivery or editing out pauses."
            )
        elif speech_pace > 200:
            suggestions.append(
                f"Speech pace is very fast ({speech_pace:.0f} WPM). "
                f"Viewers may struggle to follow -- consider slowing down "
                f"or adding visual reinforcement."
            )

    # Duration
    if duration > 600:
        suggestions.append(
            "Video is over 10 minutes. For better engagement, consider "
            "splitting into shorter segments or trimming less essential content."
        )

    return suggestions[:8]


def _result_to_dict(result: EngagementResult) -> dict:
    """Convert EngagementResult to a JSON-serializable dict."""
    return {
        "hook_score": result.hook_score,
        "retention_curve": result.retention_curve,
        "virality_score": result.virality_score,
        "overall_score": result.overall_score,
        "suggestions": result.suggestions,
        "analysis": result.analysis,
    }
