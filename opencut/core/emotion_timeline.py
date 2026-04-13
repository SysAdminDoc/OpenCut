"""
OpenCut Emotion Timeline

Builds a composite emotion/energy timeline by combining multiple signals:
- Audio RMS energy levels
- Speech rate from word timestamps
- Basic sentiment from transcript text

Normalizes and composites signals into a unified energy curve with
configurable peak detection.

Uses FFmpeg for audio analysis.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
)

logger = logging.getLogger("opencut")

# Sentiment word lists for basic analysis
_POSITIVE_WORDS = frozenset({
    "good", "great", "best", "love", "amazing", "awesome", "excellent",
    "wonderful", "fantastic", "beautiful", "happy", "perfect", "brilliant",
    "incredible", "outstanding", "superb", "terrific", "magnificent",
    "delightful", "exciting", "fun", "enjoy", "thank", "thanks", "wow",
    "yes", "absolutely", "success", "win", "winner", "celebrate",
    "inspiring", "inspired", "proud", "joy", "laugh", "smile",
    "impressive", "remarkable", "positive", "hope", "dream",
})

_NEGATIVE_WORDS = frozenset({
    "bad", "worst", "hate", "terrible", "horrible", "awful", "poor",
    "ugly", "sad", "wrong", "fail", "failure", "never", "no", "not",
    "problem", "issue", "difficult", "hard", "pain", "hurt", "damage",
    "destroy", "destroy", "kill", "death", "die", "scary", "fear",
    "afraid", "angry", "anger", "rage", "frustrate", "frustrating",
    "disappoint", "disappointing", "boring", "dull", "waste", "worse",
    "crisis", "disaster", "tragedy", "suffer", "struggle", "lost",
})

_EMPHASIS_WORDS = frozenset({
    "very", "really", "extremely", "incredibly", "absolutely", "totally",
    "completely", "literally", "seriously", "honestly", "actually",
    "basically", "exactly", "definitely", "certainly", "obviously",
    "especially", "particularly", "truly", "deeply", "highly",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TimelinePoint:
    """A single point on the emotion/energy timeline."""
    time: float
    energy: float = 0.0        # Composite energy 0.0-1.0
    audio_rms: float = 0.0     # Audio RMS level 0.0-1.0
    speech_rate: float = 0.0   # Words per second 0.0-1.0 (normalized)
    sentiment: float = 0.5     # Sentiment 0.0 (negative) - 1.0 (positive)
    label: str = ""


@dataclass
class EmotionPeakInfo:
    """A detected peak in the emotion timeline."""
    time: float
    energy: float = 0.0
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    dominant_signal: str = ""  # "audio", "speech", "sentiment"
    label: str = ""


@dataclass
class EmotionTimelineResult:
    """Complete emotion timeline analysis."""
    timeline: List[TimelinePoint] = field(default_factory=list)
    peaks: List[EmotionPeakInfo] = field(default_factory=list)
    duration: float = 0.0
    avg_energy: float = 0.0
    max_energy: float = 0.0
    sample_interval: float = 1.0
    signals_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audio RMS extraction
# ---------------------------------------------------------------------------
def _extract_audio_rms(video_path: str, interval: float = 1.0) -> List[Tuple[float, float]]:
    """
    Extract audio RMS energy levels at regular intervals using FFmpeg.

    Returns list of (timestamp, rms_normalized) tuples.
    """
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", video_path,
            "-af", f"astats=metadata=1:reset={int(1.0 / interval)},ametadata=print:file=-",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        rms_values: List[Tuple[float, float]] = []
        current_time = 0.0
        time_pattern = re.compile(r"pts_time:(\d+\.?\d*)")
        rms_pattern = re.compile(r"RMS_level=(-?\d+\.?\d*)")

        for line in result.stderr.splitlines():
            time_match = time_pattern.search(line)
            if time_match:
                current_time = float(time_match.group(1))

            rms_match = rms_pattern.search(line)
            if rms_match:
                rms_db = float(rms_match.group(1))
                # Convert dB to linear (0.0 - 1.0 range)
                # RMS in dB is typically -60 to 0 range
                if rms_db <= -60:
                    rms_linear = 0.0
                else:
                    rms_linear = 10 ** (rms_db / 20.0)
                    rms_linear = min(1.0, rms_linear)
                rms_values.append((current_time, rms_linear))

        return rms_values

    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Audio RMS extraction failed: %s", exc)
        return []


def _extract_audio_volumedetect(video_path: str, duration: float, interval: float = 1.0) -> List[Tuple[float, float]]:
    """
    Fallback: extract audio energy using volumedetect segmented approach.

    Splits audio into chunks and measures volume of each.
    """
    rms_values: List[Tuple[float, float]] = []

    # Use ffmpeg with the ebur128 loudness meter for segment-level analysis
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
            "-i", video_path,
            "-af", "ebur128=peak=true:framelog=verbose",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse momentary loudness values from ebur128 output
        # Format: "t: X.XXX    M: -XX.X S: -XX.X ..."
        loudness_pattern = re.compile(r"t:\s*(\d+\.?\d*)\s+M:\s*(-?\d+\.?\d*)")
        for line in result.stderr.splitlines():
            match = loudness_pattern.search(line)
            if match:
                t = float(match.group(1))
                lufs = float(match.group(2))
                # Convert LUFS to 0-1 range (-70 to 0 LUFS)
                normalized = max(0.0, min(1.0, (lufs + 70) / 70.0))
                rms_values.append((t, normalized))

    except (subprocess.TimeoutExpired, Exception) as exc:
        logger.debug("Volume detect extraction failed: %s", exc)

    # If no results, generate synthetic flat curve
    if not rms_values and duration > 0:
        t = 0.0
        while t < duration:
            rms_values.append((t, 0.3))
            t += interval

    return rms_values


# ---------------------------------------------------------------------------
# Speech rate analysis
# ---------------------------------------------------------------------------
def _compute_speech_rate(
    transcript: Optional[dict],
    duration: float,
    interval: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Compute speech rate (words per second) from transcript word timestamps.

    Transcript format: {"words": [{"word": str, "start": float, "end": float}, ...]}
    Or simple format: {"text": str, "segments": [{"start": float, "end": float, "text": str}]}

    Returns list of (timestamp, normalized_rate) tuples.
    """
    speech_rates: List[Tuple[float, float]] = []
    if not transcript:
        return speech_rates

    words = transcript.get("words", [])
    segments = transcript.get("segments", [])

    if words:
        # Word-level timestamps available
        window = interval
        t = 0.0
        while t < duration:
            # Count words in this time window
            count = sum(
                1 for w in words
                if w.get("start", 0) >= t and w.get("start", 0) < t + window
            )
            rate = count / window  # words per second
            speech_rates.append((t, rate))
            t += interval

    elif segments:
        # Segment-level timestamps
        t = 0.0
        while t < duration:
            # Find segments overlapping this window
            window_text = ""
            for seg in segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                if seg_end > t and seg_start < t + interval:
                    window_text += " " + seg.get("text", "")
            word_count = len(window_text.split()) if window_text.strip() else 0
            rate = word_count / interval
            speech_rates.append((t, rate))
            t += interval

    elif "text" in transcript:
        # Plain text only - assume uniform speech rate
        text = transcript["text"]
        total_words = len(text.split())
        avg_rate = total_words / max(duration, 1)
        t = 0.0
        while t < duration:
            speech_rates.append((t, avg_rate))
            t += interval

    # Normalize rates to 0-1 range
    if speech_rates:
        max_rate = max(r[1] for r in speech_rates) if speech_rates else 1.0
        if max_rate > 0:
            speech_rates = [(t, min(1.0, r / max_rate)) for t, r in speech_rates]

    return speech_rates


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------
def _compute_sentiment(
    transcript: Optional[dict],
    duration: float,
    interval: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Compute basic sentiment from transcript text using keyword matching.

    Returns list of (timestamp, sentiment) where sentiment 0.0=negative, 0.5=neutral, 1.0=positive.
    """
    sentiments: List[Tuple[float, float]] = []
    if not transcript:
        return sentiments

    segments = transcript.get("segments", [])
    text = transcript.get("text", "")

    if segments:
        t = 0.0
        while t < duration:
            # Find text in this time window
            window_text = ""
            for seg in segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                if seg_end > t and seg_start < t + interval:
                    window_text += " " + seg.get("text", "")

            sentiment = _score_text_sentiment(window_text)
            sentiments.append((t, sentiment))
            t += interval

    elif text:
        # Split text into equal chunks across duration
        words = text.split()
        words_per_interval = max(1, len(words) * interval / max(duration, 1))
        t = 0.0
        idx = 0
        while t < duration:
            end_idx = min(len(words), int(idx + words_per_interval))
            chunk = " ".join(words[int(idx):end_idx])
            sentiment = _score_text_sentiment(chunk)
            sentiments.append((t, sentiment))
            idx = end_idx
            t += interval

    return sentiments


def _score_text_sentiment(text: str) -> float:
    """Score text sentiment. Returns 0.0 (negative) to 1.0 (positive), 0.5 neutral."""
    if not text or not text.strip():
        return 0.5

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 0.5

    positive_count = sum(1 for w in words if w in _POSITIVE_WORDS)
    negative_count = sum(1 for w in words if w in _NEGATIVE_WORDS)
    emphasis_count = sum(1 for w in words if w in _EMPHASIS_WORDS)

    # Emphasis words amplify the dominant sentiment direction
    if positive_count > negative_count:
        positive_count += emphasis_count * 0.5
    elif negative_count > positive_count:
        negative_count += emphasis_count * 0.5

    total_signal = positive_count + negative_count
    if total_signal == 0:
        return 0.5

    # Map to 0-1 scale
    score = (positive_count - negative_count) / total_signal
    # Scale from [-1, 1] to [0, 1]
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


# ---------------------------------------------------------------------------
# Signal interpolation and compositing
# ---------------------------------------------------------------------------
def _interpolate_signal(
    signal: List[Tuple[float, float]],
    timestamps: List[float],
) -> List[float]:
    """Interpolate a signal to match the given timestamps."""
    if not signal:
        return [0.0] * len(timestamps)

    result = []
    for t in timestamps:
        # Find bracketing samples
        before = None
        after = None
        for st, sv in signal:
            if st <= t:
                before = (st, sv)
            if st >= t and after is None:
                after = (st, sv)

        if before is None and after is None:
            result.append(0.0)
        elif before is None:
            result.append(after[1])
        elif after is None:
            result.append(before[1])
        elif before[0] == after[0]:
            result.append(before[1])
        else:
            # Linear interpolation
            frac = (t - before[0]) / (after[0] - before[0])
            val = before[1] + frac * (after[1] - before[1])
            result.append(val)

    return result


def _normalize_signal(values: List[float]) -> List[float]:
    """Normalize a signal to 0.0-1.0 range."""
    if not values:
        return values
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v
    if span <= 0:
        return [0.5] * len(values)
    return [(v - min_v) / span for v in values]


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------
def find_peaks(
    timeline: List[TimelinePoint],
    threshold: float = 0.7,
    min_distance: float = 5.0,
) -> List[EmotionPeakInfo]:
    """
    Find peaks in the emotion timeline.

    A peak is a local maximum above the threshold with minimum
    distance from other peaks.

    Args:
        timeline: List of TimelinePoint entries.
        threshold: Minimum energy to qualify as a peak (0.0-1.0).
        min_distance: Minimum seconds between peaks.

    Returns:
        List of EmotionPeakInfo for detected peaks.
    """
    if not timeline:
        return []

    peaks: List[EmotionPeakInfo] = []
    energies = [p.energy for p in timeline]
    times = [p.time for p in timeline]

    for i in range(1, len(energies) - 1):
        # Check if local maximum
        if energies[i] <= energies[i - 1] or energies[i] <= energies[i + 1]:
            continue
        if energies[i] < threshold:
            continue

        # Check minimum distance from previous peak
        if peaks and (times[i] - peaks[-1].time) < min_distance:
            # Keep the higher peak
            if energies[i] > peaks[-1].energy:
                peaks[-1] = _make_peak(timeline, i)
            continue

        peaks.append(_make_peak(timeline, i))

    return peaks


def _make_peak(timeline: List[TimelinePoint], index: int) -> EmotionPeakInfo:
    """Create an EmotionPeakInfo from a timeline index."""
    point = timeline[index]

    # Find the dominant contributing signal
    signals = {
        "audio": point.audio_rms,
        "speech": point.speech_rate,
        "sentiment": abs(point.sentiment - 0.5) * 2,  # Distance from neutral
    }
    dominant = max(signals, key=signals.get)

    # Estimate peak region (contiguous points above 80% of peak energy)
    threshold_80 = point.energy * 0.8
    start_idx = index
    end_idx = index
    while start_idx > 0 and timeline[start_idx - 1].energy >= threshold_80:
        start_idx -= 1
    while end_idx < len(timeline) - 1 and timeline[end_idx + 1].energy >= threshold_80:
        end_idx += 1

    start_time = timeline[start_idx].time
    end_time = timeline[end_idx].time
    interval = timeline[1].time - timeline[0].time if len(timeline) > 1 else 1.0
    end_time += interval  # Include the last sample duration

    return EmotionPeakInfo(
        time=point.time,
        energy=round(point.energy, 3),
        start=round(start_time, 3),
        end=round(end_time, 3),
        duration=round(end_time - start_time, 3),
        dominant_signal=dominant,
        label=f"Peak at {point.time:.1f}s ({dominant})",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_emotion_timeline(
    video_path: str,
    transcript: Optional[dict] = None,
    interval: float = 1.0,
    audio_weight: float = 0.4,
    speech_weight: float = 0.3,
    sentiment_weight: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> EmotionTimelineResult:
    """
    Build a composite emotion/energy timeline from multiple signals.

    Combines audio energy, speech rate, and text sentiment into a
    single normalized energy curve.

    Args:
        video_path: Path to video file.
        transcript: Optional transcript dict with words/segments/text.
        interval: Sampling interval in seconds.
        audio_weight: Weight for audio RMS signal (0.0-1.0).
        speech_weight: Weight for speech rate signal (0.0-1.0).
        sentiment_weight: Weight for text sentiment signal (0.0-1.0).
        on_progress: Progress callback(pct, msg).

    Returns:
        EmotionTimelineResult with timeline and peaks.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    interval = max(0.25, float(interval))

    if on_progress:
        on_progress(5, "Getting video info...")

    video_info = get_video_info(video_path)
    duration = video_info.get("duration", 0)

    if duration <= 0:
        raise ValueError("Could not determine video duration")

    # Normalize weights
    total_weight = audio_weight + speech_weight + sentiment_weight
    if total_weight <= 0:
        total_weight = 1.0
    audio_weight = audio_weight / total_weight
    speech_weight = speech_weight / total_weight
    sentiment_weight = sentiment_weight / total_weight

    # Generate common timestamp grid
    timestamps = []
    t = 0.0
    while t < duration:
        timestamps.append(round(t, 3))
        t += interval

    if not timestamps:
        timestamps = [0.0]

    signals_used = []

    # --- Audio RMS signal ---
    if on_progress:
        on_progress(15, "Extracting audio energy...")

    audio_rms_raw = _extract_audio_rms(video_path, interval)
    if not audio_rms_raw:
        audio_rms_raw = _extract_audio_volumedetect(video_path, duration, interval)
    audio_rms_interp = _interpolate_signal(audio_rms_raw, timestamps)
    audio_rms_norm = _normalize_signal(audio_rms_interp)
    if any(v > 0 for v in audio_rms_norm):
        signals_used.append("audio_rms")

    # --- Speech rate signal ---
    if on_progress:
        on_progress(45, "Analyzing speech rate...")

    speech_rate_raw = _compute_speech_rate(transcript, duration, interval)
    speech_rate_interp = _interpolate_signal(speech_rate_raw, timestamps)
    speech_rate_norm = _normalize_signal(speech_rate_interp)
    if any(v > 0 for v in speech_rate_norm):
        signals_used.append("speech_rate")

    # --- Sentiment signal ---
    if on_progress:
        on_progress(65, "Analyzing text sentiment...")

    sentiment_raw = _compute_sentiment(transcript, duration, interval)
    sentiment_interp = _interpolate_signal(sentiment_raw, timestamps)
    # Sentiment doesn't need full normalization - keep 0-1 scale
    if any(v != 0.5 for v in sentiment_interp):
        signals_used.append("sentiment")

    # --- Composite energy ---
    if on_progress:
        on_progress(80, "Computing composite energy curve...")

    timeline_points: List[TimelinePoint] = []
    for i, t in enumerate(timestamps):
        audio_val = audio_rms_norm[i] if i < len(audio_rms_norm) else 0.0
        speech_val = speech_rate_norm[i] if i < len(speech_rate_norm) else 0.0
        sentiment_val = sentiment_interp[i] if i < len(sentiment_interp) else 0.5

        # Composite energy: weighted sum
        # For sentiment, use distance from neutral (0.5) as energy contribution
        sentiment_energy = abs(sentiment_val - 0.5) * 2.0
        energy = (
            audio_val * audio_weight +
            speech_val * speech_weight +
            sentiment_energy * sentiment_weight
        )
        energy = max(0.0, min(1.0, energy))

        timeline_points.append(TimelinePoint(
            time=t,
            energy=round(energy, 3),
            audio_rms=round(audio_val, 3),
            speech_rate=round(speech_val, 3),
            sentiment=round(sentiment_val, 3),
        ))

    # --- Peak detection ---
    if on_progress:
        on_progress(90, "Detecting energy peaks...")

    peaks = find_peaks(timeline_points, threshold=0.7, min_distance=5.0)

    # Stats
    energies = [p.energy for p in timeline_points]
    avg_energy = sum(energies) / len(energies) if energies else 0.0
    max_energy = max(energies) if energies else 0.0

    if on_progress:
        on_progress(100, f"Timeline complete: {len(peaks)} peaks detected")

    return EmotionTimelineResult(
        timeline=timeline_points,
        peaks=peaks,
        duration=duration,
        avg_energy=round(avg_energy, 3),
        max_energy=round(max_energy, 3),
        sample_interval=interval,
        signals_used=signals_used,
    )
