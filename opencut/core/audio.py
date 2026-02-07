"""
Audio analysis utilities.

Provides audio energy analysis for zoom detection and other features
that don't require ML models.
"""

import subprocess
import struct
import tempfile
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

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
        "ffmpeg",
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
        raise RuntimeError(f"Audio extraction failed: {result.stderr.decode()}")

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
        "ffmpeg",
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
        raise RuntimeError(f"Audio extraction failed: {result.stderr.decode()}")

    return output_path


def analyze_energy(
    filepath: str,
    window_size: float = 0.05,
    hop_size: float = 0.025,
) -> List[AudioEnergy]:
    """
    Analyze audio energy using windowed RMS.

    Args:
        filepath: Path to the media file.
        window_size: Analysis window in seconds.
        hop_size: Hop between windows in seconds.

    Returns:
        List of AudioEnergy measurements.
    """
    sample_rate = 16000
    pcm_data, sr = extract_audio_pcm(filepath, sample_rate=sample_rate)

    # Convert bytes to samples
    num_samples = len(pcm_data) // 2
    samples = struct.unpack(f"<{num_samples}h", pcm_data)

    # Normalize to -1.0 to 1.0
    max_val = 32768.0
    normalized = [s / max_val for s in samples]

    # Windowed analysis
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    energies = []
    pos = 0
    while pos + window_samples <= len(normalized):
        window = normalized[pos:pos + window_samples]

        # RMS
        rms = (sum(s * s for s in window) / len(window)) ** 0.5

        # Peak
        peak = max(abs(s) for s in window)

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
