"""
OpenCut Audio Category Tagging

Auto-classify timeline audio as speech, music, SFX, ambience, or silence.
Uses FFmpeg audio analysis (astats, ebur128) plus numpy spectral analysis
for heuristic classification without requiring ML models.

Optional: Silero VAD for improved speech detection, librosa for spectral features.
"""

import json
import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_ffprobe_path,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = ("speech", "music", "sound_effect", "ambience", "silence")
DEFAULT_SEGMENT_DURATION = 2.0
SILENCE_RMS_THRESHOLD = -50.0  # dBFS
SPEECH_SPECTRAL_CENTROID_LOW = 300  # Hz
SPEECH_SPECTRAL_CENTROID_HIGH = 4000  # Hz


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AudioSegment:
    """A classified segment of audio."""
    start_time: float = 0.0
    end_time: float = 0.0
    category: str = "silence"
    confidence: float = 0.0


@dataclass
class AudioClassificationResult:
    """Full classification result for an audio file."""
    segments: List[AudioSegment] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------
def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", audio_path,
    ]
    result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}")
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


# ---------------------------------------------------------------------------
# FFmpeg-based audio stats extraction
# ---------------------------------------------------------------------------
def _extract_segment_stats(
    audio_path: str,
    start: float,
    duration: float,
) -> dict:
    """Extract audio statistics for a segment using FFmpeg astats filter.

    Returns dict with rms_level, peak_level, crest_factor, flat_factor, etc.
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-ss", str(start),
        "-t", str(duration),
        "-i", audio_path,
        "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level"
               ":key=lavfi.astats.Overall.Peak_level"
               ":key=lavfi.astats.Overall.Flat_factor"
               ":key=lavfi.astats.Overall.Crest_factor"
               ":key=lavfi.astats.Overall.Zero_crossings_rate",
        "-f", "null", "-",
    ]
    result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
    stderr = result.stderr if result.stderr else ""
    stdout = result.stdout if result.stdout else ""
    combined = stderr + stdout

    stats = {
        "rms_level": -100.0,
        "peak_level": -100.0,
        "flat_factor": 0.0,
        "crest_factor": 0.0,
        "zero_crossings_rate": 0.0,
    }

    for line in combined.split("\n"):
        line = line.strip()
        if "RMS_level" in line:
            match = re.search(r"[-+]?\d+\.?\d*", line.split("RMS_level")[-1])
            if match:
                stats["rms_level"] = float(match.group())
        elif "Peak_level" in line:
            match = re.search(r"[-+]?\d+\.?\d*", line.split("Peak_level")[-1])
            if match:
                stats["peak_level"] = float(match.group())
        elif "Flat_factor" in line:
            match = re.search(r"[-+]?\d+\.?\d*", line.split("Flat_factor")[-1])
            if match:
                stats["flat_factor"] = float(match.group())
        elif "Crest_factor" in line:
            match = re.search(r"[-+]?\d+\.?\d*", line.split("Crest_factor")[-1])
            if match:
                stats["crest_factor"] = float(match.group())
        elif "Zero_crossings_rate" in line:
            match = re.search(r"[-+]?\d+\.?\d*", line.split("Zero_crossings_rate")[-1])
            if match:
                stats["zero_crossings_rate"] = float(match.group())

    return stats


# ---------------------------------------------------------------------------
# Numpy spectral analysis (optional, improves accuracy)
# ---------------------------------------------------------------------------
def _extract_raw_audio(audio_path: str, start: float, duration: float) -> bytes:
    """Extract raw PCM audio segment via FFmpeg."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", audio_path,
        "-ac", "1", "-ar", "16000",
        "-f", "s16le",
        "pipe:1",
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        return b""
    return result.stdout


def _analyze_spectral_numpy(pcm_data: bytes, sample_rate: int = 16000) -> dict:
    """Compute spectral features using numpy FFT."""
    import numpy as np

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return {"spectral_centroid": 0, "spectral_flatness": 0, "energy_variance": 0}

    samples /= 32768.0  # Normalize

    # FFT
    n = len(samples)
    fft = np.fft.rfft(samples)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Spectral centroid
    total_mag = np.sum(magnitude)
    if total_mag > 0:
        centroid = float(np.sum(freqs * magnitude) / total_mag)
    else:
        centroid = 0.0

    # Spectral flatness (geometric mean / arithmetic mean of magnitude spectrum)
    mag_nonzero = magnitude[magnitude > 0]
    if len(mag_nonzero) > 0:
        log_mean = np.mean(np.log(mag_nonzero + 1e-10))
        geo_mean = np.exp(log_mean)
        arith_mean = np.mean(mag_nonzero)
        flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0
    else:
        flatness = 0.0

    # Energy variance (windowed)
    window_size = sample_rate // 10  # 100ms windows
    if window_size > 0 and len(samples) >= window_size:
        n_windows = len(samples) // window_size
        energies = []
        for i in range(n_windows):
            chunk = samples[i * window_size:(i + 1) * window_size]
            energies.append(float(np.mean(chunk ** 2)))
        energy_var = float(np.var(energies)) if energies else 0.0
    else:
        energy_var = 0.0

    return {
        "spectral_centroid": centroid,
        "spectral_flatness": flatness,
        "energy_variance": energy_var,
    }


# ---------------------------------------------------------------------------
# Classification heuristics
# ---------------------------------------------------------------------------
def _classify_segment(stats: dict, spectral: Optional[dict] = None) -> tuple:
    """Classify a single segment based on audio stats and spectral features.

    Returns (category, confidence).
    """
    rms = stats.get("rms_level", -100.0)
    peak = stats.get("peak_level", -100.0)
    crest = stats.get("crest_factor", 0.0)

    # 1. Silence detection
    if rms <= SILENCE_RMS_THRESHOLD or peak <= SILENCE_RMS_THRESHOLD:
        return "silence", 0.95

    # Use spectral features if available
    if spectral:
        centroid = spectral.get("spectral_centroid", 0)
        flatness = spectral.get("spectral_flatness", 0)
        energy_var = spectral.get("energy_variance", 0)

        # Speech: mid-range centroid, moderate flatness, moderate energy variance
        if (SPEECH_SPECTRAL_CENTROID_LOW <= centroid <= SPEECH_SPECTRAL_CENTROID_HIGH
                and flatness < 0.3 and energy_var > 1e-6):
            return "speech", 0.75

        # Music: lower flatness (tonal), lower energy variance (steady)
        if flatness < 0.2 and energy_var < 1e-5:
            return "music", 0.70

        # SFX: high energy variance (transients), high crest factor
        if energy_var > 1e-4 or crest > 15:
            return "sound_effect", 0.65

        # Ambience: high flatness (noise-like), low variance
        if flatness > 0.5 and energy_var < 1e-6:
            return "ambience", 0.70

    # Fallback heuristics using FFmpeg stats only
    # Speech: moderate RMS, moderate crest factor
    if -35 < rms < -10 and 3 < crest < 20:
        return "speech", 0.55

    # Music: steady energy, low crest factor
    if -30 < rms < -5 and crest < 10:
        return "music", 0.50

    # SFX: high crest factor (transient)
    if crest > 15:
        return "sound_effect", 0.50

    # Ambience: low-level continuous signal
    if -50 < rms < -30:
        return "ambience", 0.50

    # Default: ambience for anything else
    return "ambience", 0.35


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_audio(
    audio_path: str,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    on_progress: Optional[Callable] = None,
) -> AudioClassificationResult:
    """
    Classify audio into categories: speech, music, SFX, ambience, silence.

    Analyzes audio in segments (default 2 seconds each) using FFmpeg
    audio stats and optional numpy spectral analysis.

    Args:
        audio_path: Path to audio or video file.
        segment_duration: Length of each analysis segment in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        AudioClassificationResult with per-segment categories and summary.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segment_duration = max(0.5, min(30.0, segment_duration))

    if on_progress:
        on_progress(5, "Analyzing audio file...")

    total_duration = _get_audio_duration(audio_path)
    if total_duration <= 0:
        raise RuntimeError("Could not determine audio duration")

    has_numpy = ensure_package("numpy", "numpy", on_progress)

    num_segments = max(1, int(total_duration / segment_duration))
    segments = []

    if on_progress:
        on_progress(10, f"Classifying {num_segments} segments...")

    for i in range(num_segments):
        start = i * segment_duration
        seg_dur = min(segment_duration, total_duration - start)
        if seg_dur <= 0:
            break

        stats = _extract_segment_stats(audio_path, start, seg_dur)

        spectral = None
        if has_numpy:
            pcm = _extract_raw_audio(audio_path, start, seg_dur)
            if pcm:
                spectral = _analyze_spectral_numpy(pcm)

        category, confidence = _classify_segment(stats, spectral)

        segments.append(AudioSegment(
            start_time=round(start, 3),
            end_time=round(start + seg_dur, 3),
            category=category,
            confidence=round(confidence, 3),
        ))

        if on_progress and num_segments > 0:
            pct = 10 + int((i + 1) / num_segments * 85)
            on_progress(pct, f"Classified segment {i + 1}/{num_segments}")

    # Build summary
    summary: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    for seg in segments:
        dur = seg.end_time - seg.start_time
        if seg.category in summary:
            summary[seg.category] += dur
        else:
            summary[seg.category] = dur
    # Round summary values
    summary = {k: round(v, 3) for k, v in summary.items()}

    if on_progress:
        on_progress(100, "Classification complete")

    return AudioClassificationResult(segments=segments, summary=summary)
