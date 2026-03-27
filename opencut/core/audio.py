"""
Audio analysis utilities.

Provides audio energy analysis for zoom detection and other features
that don't require ML models.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path

from .silence import TimeSegment


@dataclass
class AudioEnergy:
    """Audio energy measurement at a point in time."""
    time: float
    rms: float        # Root mean square energy (0.0 - 1.0)
    peak: float       # Peak amplitude (0.0 - 1.0)


def extract_audio_pcm(
    filepath: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[bytes, int]:
    """
    Extract raw PCM audio from a media file using FFmpeg.

    Args:
        filepath: Path to the media file.
        sample_rate: Output sample rate.
        mono: Downmix to mono.

    Returns:
        Tuple of (raw PCM bytes as 16-bit signed LE, sample_rate).
    """
    channels = "1" if mono else "2"

    cmd = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", filepath,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", channels,
        "-f", "s16le",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr.decode(errors='replace')}")

    return result.stdout, sample_rate


def extract_audio_wav(filepath: str, output_path: Optional[str] = None, sample_rate: int = 16000) -> str:
    """
    Extract audio from a media file as a WAV file.

    Args:
        filepath: Path to the media file.
        output_path: Path for the output WAV. Auto-generated if None.
        sample_rate: Output sample rate.

    Returns:
        Path to the output WAV file.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", filepath,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr.decode(errors='replace')}")

    return output_path


def analyze_energy(
    filepath: str,
    window_size: float = 0.05,
    hop_size: float = 0.025,
) -> List[AudioEnergy]:
    """
    Analyze audio energy using windowed RMS.

    Uses array module for memory-efficient sample storage (~4x less RAM
    than a Python list of floats) and processes in-place to avoid creating
    a second normalized copy.

    Args:
        filepath: Path to the media file.
        window_size: Analysis window in seconds.
        hop_size: Hop between windows in seconds.

    Returns:
        List of AudioEnergy measurements.
    """
    import array as _array
    import math

    sample_rate = 16000
    pcm_data, sr = extract_audio_pcm(filepath, sample_rate=sample_rate)

    # Convert bytes to signed 16-bit array (2 bytes/sample, ~8x less than list of floats)
    samples = _array.array("h")
    samples.frombytes(pcm_data)
    num_samples = len(samples)

    # Work directly with int16 values — normalize at the point of use
    max_val = 32768.0

    # Windowed analysis
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    energies = []
    pos = 0
    while pos + window_samples <= num_samples:
        # Compute RMS and peak directly from int16 samples
        sum_sq = 0.0
        pk = 0
        for i in range(pos, pos + window_samples):
            v = samples[i]
            av = v if v >= 0 else -v
            sum_sq += v * v
            if av > pk:
                pk = av

        rms = math.sqrt(sum_sq / window_samples) / max_val
        peak = pk / max_val

        time_sec = pos / sample_rate
        energies.append(AudioEnergy(time=time_sec, rms=rms, peak=peak))

        pos += hop_samples

    return energies


def find_emphasis_points(
    filepath: str,
    threshold: float = 0.7,
    min_interval: float = 3.0,
) -> List[TimeSegment]:
    """
    Find points of vocal emphasis (loud moments) suitable for zoom effects.

    Args:
        filepath: Path to the media file.
        threshold: Energy threshold (0.0-1.0 relative to max energy).
        min_interval: Minimum seconds between emphasis points.

    Returns:
        List of TimeSegment objects at emphasis points.
    """
    energies = analyze_energy(filepath)

    if not energies:
        return []

    # Find max RMS for normalization
    max_rms = max(e.rms for e in energies)
    if max_rms <= 0:
        return []

    # Find points above threshold
    emphasis_points = []
    last_time = -min_interval

    for e in energies:
        normalized_rms = e.rms / max_rms
        if normalized_rms >= threshold and (e.time - last_time) >= min_interval:
            emphasis_points.append(TimeSegment(
                start=e.time,
                end=e.time + 0.5,  # Emphasis duration
                label="emphasis",
            ))
            last_time = e.time

    return emphasis_points
