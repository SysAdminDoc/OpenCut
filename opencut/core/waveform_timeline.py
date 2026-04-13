"""
Interactive Waveform Timeline Backend.

Generate waveform data arrays and images from audio for frontend
timeline rendering and visualization.
"""

import logging
import os
import struct
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import FFmpegCmd, get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")


def _extract_pcm(audio_path: str, sample_rate: int = 8000) -> Tuple[bytes, int]:
    """Extract raw PCM data from audio/video at a given sample rate."""
    cmd = [
        get_ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg PCM extraction failed: {result.stderr[:200].decode()}")
    return result.stdout, sample_rate


def _pcm_to_samples(pcm_data: bytes) -> List[int]:
    """Convert raw PCM bytes (16-bit signed LE) to sample list."""
    n_samples = len(pcm_data) // 2
    return list(struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2]))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_waveform_data(
    audio_path: str,
    samples_per_second: int = 100,
    normalize: bool = True,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Generate waveform amplitude data for frontend rendering.

    Produces a downsampled array of min/max amplitude pairs suitable
    for drawing waveform visualizations.

    Args:
        audio_path: Path to audio or video file.
        samples_per_second: Output resolution (samples per second of audio).
        normalize: Normalize amplitudes to 0.0-1.0 range.

    Returns:
        Dict with 'waveform' (list of [min, max] pairs), 'duration',
        'sample_rate', and 'sample_count'.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if on_progress:
        on_progress(10, "Extracting audio data")

    # Use a higher internal sample rate, then downsample
    internal_sr = max(samples_per_second * 10, 8000)
    pcm_data, sr = _extract_pcm(audio_path, sample_rate=internal_sr)
    samples = _pcm_to_samples(pcm_data)

    if not samples:
        return {"waveform": [], "duration": 0, "sample_rate": samples_per_second, "sample_count": 0}

    if on_progress:
        on_progress(50, "Computing waveform")

    duration = len(samples) / sr
    # How many raw samples per output sample
    window = max(int(sr / samples_per_second), 1)
    waveform = []

    for i in range(0, len(samples), window):
        chunk = samples[i:i + window]
        if chunk:
            waveform.append([min(chunk), max(chunk)])

    if normalize and waveform:
        peak = max(max(abs(p[0]), abs(p[1])) for p in waveform)
        if peak > 0:
            waveform = [[p[0] / peak, p[1] / peak] for p in waveform]

    if on_progress:
        on_progress(100, "Waveform data ready")

    logger.info("Generated waveform: %d samples, %.2fs duration", len(waveform), duration)
    return {
        "waveform": waveform,
        "duration": round(duration, 3),
        "sample_rate": samples_per_second,
        "sample_count": len(waveform),
    }


def generate_waveform_image(
    audio_path: str,
    width: int = 1920,
    height: int = 200,
    output_path: Optional[str] = None,
    color: str = "0x00FF88",
    bg_color: str = "0x1a1a2e",
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate a waveform image using FFmpeg's showwavespic filter.

    Args:
        audio_path: Path to audio or video file.
        width: Image width in pixels.
        height: Image height in pixels.
        output_path: Output image path (auto-generated if None).
        color: Waveform color in hex.
        bg_color: Background color in hex.

    Returns:
        Path to the generated waveform image.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"waveform_{os.getpid()}.png"
        )

    if on_progress:
        on_progress(20, "Generating waveform image")

    cmd = (
        FFmpegCmd()
        .input(audio_path)
        .option("filter_complex",
                f"showwavespic=s={width}x{height}"
                f":colors={color}")
        .frames(1)
        .output(output_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=60)

    if on_progress:
        on_progress(100, "Waveform image generated")

    logger.info("Generated waveform image: %s (%dx%d)", output_path, width, height)
    return output_path


def get_waveform_region(
    audio_path: str,
    start: float,
    end: float,
    samples: int = 500,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Get waveform data for a specific time region.

    Args:
        audio_path: Path to audio or video file.
        start: Start time in seconds.
        end: End time in seconds.
        samples: Number of output samples for the region.

    Returns:
        Dict with 'waveform' data for the region.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if end <= start:
        raise ValueError("End time must be greater than start time")
    if samples < 1:
        raise ValueError("Samples must be positive")

    if on_progress:
        on_progress(10, "Extracting audio region")

    region_duration = end - start
    internal_sr = max(int(samples * 10 / region_duration), 8000)

    # Extract only the requested region
    cmd = [
        get_ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-t", str(region_duration),
        "-i", audio_path,
        "-vn", "-ac", "1",
        "-ar", str(internal_sr),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError("Failed to extract audio region")

    raw_samples = _pcm_to_samples(result.stdout)
    if not raw_samples:
        return {"waveform": [], "start": start, "end": end, "sample_count": 0}

    if on_progress:
        on_progress(60, "Computing region waveform")

    window = max(len(raw_samples) // samples, 1)
    waveform = []
    for i in range(0, len(raw_samples), window):
        chunk = raw_samples[i:i + window]
        if chunk:
            waveform.append([min(chunk), max(chunk)])

    # Normalize
    peak = max((max(abs(p[0]), abs(p[1])) for p in waveform), default=1)
    if peak > 0:
        waveform = [[p[0] / peak, p[1] / peak] for p in waveform]

    # Trim to requested sample count
    waveform = waveform[:samples]

    if on_progress:
        on_progress(100, "Region waveform ready")

    return {
        "waveform": waveform,
        "start": start,
        "end": end,
        "duration": region_duration,
        "sample_count": len(waveform),
    }
