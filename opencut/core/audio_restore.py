"""
OpenCut Audio Restoration Toolkit

Professional audio restoration tools using FFmpeg filters:
- Declip: repair clipped audio using adeclip
- Dehum: remove electrical hum at fundamental + harmonics
- Decrackle: reduce vinyl/tape crackle via afftdn
- Dewind: high-pass filter to remove wind noise
- Dereverb: reduce room reverb via gating approach

All features use FFmpeg only - no additional model downloads required.
"""

import logging
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Declip
# ---------------------------------------------------------------------------
def declip(
    input_path: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Repair clipped/distorted audio using FFmpeg adeclip filter.

    Args:
        input_path: Source audio/video file.
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, filter_used.
    """
    if output_path is None:
        output_path = _output_path(input_path, "declipped")

    if on_progress:
        on_progress(10, "Applying declip filter...")

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", "adeclip=window=55:overlap=75",
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Declip complete")

    logger.info("Declipped audio: %s -> %s", input_path, output_path)
    return {"output_path": output_path, "filter_used": "adeclip"}


# ---------------------------------------------------------------------------
# Dehum
# ---------------------------------------------------------------------------
def dehum(
    input_path: str,
    frequency: float = 60.0,
    harmonics: int = 4,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove electrical hum at a fundamental frequency and its harmonics.

    Uses FFmpeg bandreject filters stacked at the fundamental and each
    harmonic (e.g. 60Hz -> 120, 180, 240Hz for harmonics=4).

    Args:
        input_path: Source audio/video file.
        frequency: Fundamental hum frequency in Hz (50 or 60 typical).
        harmonics: Number of harmonics to remove (including fundamental).
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, filter_used, frequencies_removed.
    """
    if output_path is None:
        output_path = _output_path(input_path, "dehummed")

    frequency = max(20.0, min(500.0, float(frequency)))
    harmonics = max(1, min(8, int(harmonics)))

    if on_progress:
        on_progress(10, f"Removing hum at {frequency}Hz + {harmonics - 1} harmonics...")

    # Build bandreject filter chain for fundamental + harmonics
    filters = []
    freqs_removed = []
    for i in range(1, harmonics + 1):
        freq = frequency * i
        # Q-factor of ~5 gives a narrow notch
        filters.append(f"bandreject=f={freq}:width_type=q:w=5")
        freqs_removed.append(freq)

    af_chain = ",".join(filters)

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af_chain,
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Dehum complete")

    logger.info("Dehummed audio at %s: %s -> %s", freqs_removed, input_path, output_path)
    return {
        "output_path": output_path,
        "filter_used": "bandreject",
        "frequencies_removed": freqs_removed,
    }


# ---------------------------------------------------------------------------
# Decrackle
# ---------------------------------------------------------------------------
def decrackle(
    input_path: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Reduce crackle/pop noise from audio using FFmpeg afftdn filter.

    Uses adaptive FFT-based denoising tuned for impulsive noise (crackle).

    Args:
        input_path: Source audio/video file.
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, filter_used.
    """
    if output_path is None:
        output_path = _output_path(input_path, "decrackled")

    if on_progress:
        on_progress(10, "Applying decrackle filter...")

    # afftdn with noise type set to white noise profile works well for crackle.
    # nt=w = white noise, nr=20dB reduction, om=o = output mode cleaned
    af = "afftdn=nt=w:nr=20:om=o"

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af,
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Decrackle complete")

    logger.info("Decrackled audio: %s -> %s", input_path, output_path)
    return {"output_path": output_path, "filter_used": "afftdn"}


# ---------------------------------------------------------------------------
# Dewind
# ---------------------------------------------------------------------------
def dewind(
    input_path: str,
    cutoff_hz: float = 80.0,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove wind noise using a high-pass filter.

    Wind noise is predominantly low-frequency; a highpass filter at the
    specified cutoff removes it effectively.

    Args:
        input_path: Source audio/video file.
        cutoff_hz: High-pass cutoff frequency in Hz.
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, filter_used, cutoff_hz.
    """
    if output_path is None:
        output_path = _output_path(input_path, "dewinded")

    cutoff_hz = max(20.0, min(500.0, float(cutoff_hz)))

    if on_progress:
        on_progress(10, f"Applying highpass filter at {cutoff_hz}Hz...")

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", f"highpass=f={cutoff_hz}:poles=2",
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Dewind complete")

    logger.info("Dewinded audio (cutoff=%sHz): %s -> %s", cutoff_hz, input_path, output_path)
    return {
        "output_path": output_path,
        "filter_used": "highpass",
        "cutoff_hz": cutoff_hz,
    }


# ---------------------------------------------------------------------------
# Dereverb
# ---------------------------------------------------------------------------
def dereverb(
    input_path: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Reduce room reverb from audio using a gate-based approach.

    Applies a noise gate combined with a dry/wet mix to suppress reverb tails
    while preserving direct sound. Uses FFmpeg agate + highpass combination.

    Args:
        input_path: Source audio/video file.
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, filter_used.
    """
    if output_path is None:
        output_path = _output_path(input_path, "dereverbed")

    if on_progress:
        on_progress(10, "Applying dereverb processing...")

    # Strategy: noise gate with fast attack to clip reverb tails,
    # combined with a gentle high-pass to reduce low-frequency rumble
    # that often accompanies reverb.
    af = (
        "highpass=f=60:poles=2,"
        "agate=threshold=0.01:ratio=3:attack=0.5:release=50:makeup=1,"
        "acompressor=threshold=-20dB:ratio=4:attack=5:release=100"
    )

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af,
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Dereverb complete")

    logger.info("Dereverbed audio: %s -> %s", input_path, output_path)
    return {"output_path": output_path, "filter_used": "agate+acompressor"}
