"""
Emotion-Based Highlight Detection

Analyzes faces in video frames for emotional expression, building an
emotion curve over time. Peaks in the curve identify emotionally
significant moments — ideal for highlight extraction.

Uses deepface for lightweight facial emotion recognition.

Requires: pip install deepface opencv-python-headless
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class EmotionPeak:
    """A detected emotional peak in the video."""
    time: float          # Timestamp in seconds
    duration: float      # Duration of the emotional peak region
    emotion: str         # Dominant emotion at peak (happy, surprise, angry, etc.)
    intensity: float     # 0-1 intensity
    start: float = 0.0   # Region start
    end: float = 0.0     # Region end


@dataclass
class EmotionCurve:
    """Emotion analysis results for a video."""
    samples: List[Dict] = field(default_factory=list)   # {time, emotions: {happy, sad, ...}, dominant, intensity}
    peaks: List[EmotionPeak] = field(default_factory=list)
    avg_intensity: float = 0.0
    dominant_emotion: str = ""


def check_deepface_available() -> bool:
    """Check if deepface is installed."""
    try:
        import deepface  # noqa: F401
        return True
    except ImportError:
        return False


def analyze_video_emotions(
    filepath: str,
    sample_interval: float = 1.0,
    min_peak_intensity: float = 0.6,
    min_peak_duration: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> EmotionCurve:
    """
    Analyze facial emotions across a video to build an emotion curve.

    Samples frames at regular intervals, detects faces and their emotions,
    and identifies emotional peaks for highlight extraction.

    Args:
        filepath: Path to video file.
        sample_interval: Seconds between sampled frames (1.0 = 1 fps analysis).
        min_peak_intensity: Minimum emotion intensity to count as a peak (0-1).
        min_peak_duration: Minimum duration (seconds) for a peak to be reported.
        on_progress: Progress callback(pct, msg).

    Returns:
        EmotionCurve with samples and detected peaks.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise ImportError("OpenCV required. Install: pip install opencv-python-headless")

    try:
        from deepface import DeepFace
    except ImportError:
        raise ImportError("deepface required. Install: pip install deepface")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Video not found: {filepath}")

    if on_progress:
        on_progress(5, "Opening video for emotion analysis...")

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {filepath}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_skip = max(1, int(fps * sample_interval))

    samples = []
    frame_idx = 0

    if on_progress:
        on_progress(10, f"Analyzing {int(duration)}s of video ({int(duration / sample_interval)} samples)...")

    try:
      while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / fps

            try:
                # DeepFace returns a list of face analysis results
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )

                if results and isinstance(results, list):
                    # Take the most prominent face
                    face = results[0]
                    emotions = face.get("emotion", {})
                    dominant = face.get("dominant_emotion", "neutral")
                    # Intensity: the confidence of the dominant emotion (0-100 → 0-1)
                    intensity = emotions.get(dominant, 0) / 100.0

                    samples.append({
                        "time": round(timestamp, 2),
                        "emotions": {k: round(v / 100.0, 3) for k, v in emotions.items()},
                        "dominant": dominant,
                        "intensity": round(intensity, 3),
                    })
                else:
                    samples.append({
                        "time": round(timestamp, 2),
                        "emotions": {},
                        "dominant": "none",
                        "intensity": 0.0,
                    })

            except Exception:
                # No face detected or analysis failed — record neutral
                samples.append({
                    "time": round(timestamp, 2),
                    "emotions": {},
                    "dominant": "none",
                    "intensity": 0.0,
                })

            if on_progress and len(samples) % 10 == 0:
                pct = 10 + int((frame_idx / max(1, total_frames)) * 80)
                on_progress(min(90, pct), f"Analyzed {len(samples)} frames...")

        frame_idx += 1
    finally:
      cap.release()

    if not samples:
        return EmotionCurve()

    if on_progress:
        on_progress(92, "Detecting emotion peaks...")

    # Find peaks — contiguous regions where intensity > threshold
    peaks = _find_peaks(samples, min_peak_intensity, min_peak_duration)

    # Compute overall stats
    intensities = [s["intensity"] for s in samples if s["intensity"] > 0]
    avg_intensity = sum(intensities) / len(intensities) if intensities else 0

    # Most common dominant emotion (excluding "none" and "neutral")
    emotion_counts = {}
    for s in samples:
        e = s["dominant"]
        if e not in ("none", "neutral"):
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
    dominant_overall = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"

    if on_progress:
        on_progress(100, f"Found {len(peaks)} emotion peak(s)")

    return EmotionCurve(
        samples=samples,
        peaks=peaks,
        avg_intensity=round(avg_intensity, 3),
        dominant_emotion=dominant_overall,
    )


def _find_peaks(
    samples: List[Dict],
    min_intensity: float,
    min_duration: float,
) -> List[EmotionPeak]:
    """Find contiguous emotion peaks above the intensity threshold."""
    peaks = []
    in_peak = False
    peak_start = 0.0
    peak_samples = []

    for s in samples:
        if s["intensity"] >= min_intensity and s["dominant"] not in ("none", "neutral"):
            if not in_peak:
                in_peak = True
                peak_start = s["time"]
                peak_samples = []
            peak_samples.append(s)
        else:
            if in_peak:
                # End of peak
                peak_end = peak_samples[-1]["time"] if peak_samples else peak_start
                peak_duration = peak_end - peak_start
                if peak_duration >= min_duration and peak_samples:
                    # Find the peak moment
                    best = max(peak_samples, key=lambda x: x["intensity"])
                    peaks.append(EmotionPeak(
                        time=best["time"],
                        duration=round(peak_duration, 2),
                        emotion=best["dominant"],
                        intensity=best["intensity"],
                        start=round(peak_start, 2),
                        end=round(peak_end, 2),
                    ))
                in_peak = False
                peak_samples = []

    # Handle peak at end of video
    if in_peak and peak_samples:
        peak_end = peak_samples[-1]["time"]
        peak_duration = peak_end - peak_start
        if peak_duration >= min_duration:
            best = max(peak_samples, key=lambda x: x["intensity"])
            peaks.append(EmotionPeak(
                time=best["time"],
                duration=round(peak_duration, 2),
                emotion=best["dominant"],
                intensity=best["intensity"],
                start=round(peak_start, 2),
                end=round(peak_end, 2),
            ))

    # Sort by intensity descending
    peaks.sort(key=lambda p: p.intensity, reverse=True)
    return peaks


def emotion_peaks_to_highlights(peaks: List[EmotionPeak], padding: float = 3.0) -> List[Dict]:
    """
    Convert emotion peaks to highlight-compatible cut regions.

    Adds padding around each peak to capture context.

    Args:
        peaks: List of EmotionPeak objects.
        padding: Seconds to add before/after each peak.

    Returns:
        List of dicts with start, end, title, score keys.
    """
    highlights = []
    for peak in peaks:
        start = max(0, peak.start - padding)
        end = peak.end + padding
        highlights.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "title": f"{peak.emotion.title()} moment ({peak.intensity:.0%})",
            "score": peak.intensity,
            "reason": f"Emotional peak: {peak.emotion} at {peak.time:.1f}s",
        })
    return highlights
