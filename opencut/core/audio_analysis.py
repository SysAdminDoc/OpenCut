"""
OpenCut Audio Spectrum Analyzer & Loudness Meter

Measures EBU R128 loudness (integrated LUFS, true peak, LRA),
analyses frequency spectrum bands, and checks compliance against
platform loudness targets (YouTube, Spotify, Apple Podcasts, etc.).

Uses FFmpeg only — no additional dependencies required.
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Platform loudness targets (LUFS)
# ---------------------------------------------------------------------------
PLATFORM_TARGETS: Dict[str, float] = {
    "youtube": -14.0,
    "spotify": -14.0,
    "apple_podcasts": -16.0,
    "broadcast": -24.0,
    "tiktok": -14.0,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class LoudnessResult:
    """EBU R128 loudness measurement results."""
    integrated_lufs: float = -70.0
    true_peak_dbtp: float = -70.0
    lra: float = 0.0
    momentary_max: float = -70.0
    short_term_max: float = -70.0


@dataclass
class SpectrumResult:
    """Average dB levels across standard frequency bands."""
    sub_bass: float = -70.0      # 20-60 Hz
    bass: float = -70.0          # 60-250 Hz
    low_mid: float = -70.0       # 250-500 Hz
    mid: float = -70.0           # 500-2000 Hz
    upper_mid: float = -70.0     # 2000-4000 Hz
    presence: float = -70.0      # 4000-6000 Hz
    brilliance: float = -70.0    # 6000-20000 Hz


@dataclass
class PlatformLoudnessResult:
    """Platform loudness compliance check."""
    platform: str = ""
    target_lufs: float = -14.0
    actual_lufs: float = -70.0
    passes: bool = False
    adjustment_needed_db: float = 0.0


# ---------------------------------------------------------------------------
# Loudness measurement (EBU R128)
# ---------------------------------------------------------------------------
def measure_loudness(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> LoudnessResult:
    """
    Measure EBU R128 loudness of an audio/video file.

    Uses FFmpeg's ebur128 filter to compute integrated loudness (LUFS),
    true peak (dBTP), loudness range (LRA), and momentary/short-term max.

    Args:
        input_path: Source audio or video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        LoudnessResult with all measurements.
    """
    if on_progress:
        on_progress(10, "Measuring loudness (EBU R128)...")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-i", input_path,
        "-af", "ebur128=peak=true",
        "-f", "null", "-",
    ]

    if on_progress:
        on_progress(20, "Running ebur128 analysis...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if on_progress:
        on_progress(70, "Parsing loudness data...")

    stderr = result.stderr

    # Parse the summary block from ebur128 output
    integrated = -70.0
    true_peak = -70.0
    lra = 0.0
    momentary_max = -70.0
    short_term_max = -70.0

    # Integrated loudness: "I: -14.0 LUFS"
    m = re.search(r"I:\s*([-\d.]+)\s*LUFS", stderr)
    if m:
        try:
            integrated = float(m.group(1))
        except ValueError:
            pass

    # True peak: "Peak:\s+X.X dBFS" or "True peak:" line
    m = re.search(r"True\s+peak:\s*([-\d.]+)\s*dBFS", stderr)
    if not m:
        m = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", stderr)
    if m:
        try:
            true_peak = float(m.group(1))
        except ValueError:
            pass

    # LRA: "LRA: 7.0 LU"
    m = re.search(r"LRA:\s*([-\d.]+)\s*LU", stderr)
    if m:
        try:
            lra = float(m.group(1))
        except ValueError:
            pass

    # Momentary max: track the maximum of all momentary readings
    # ebur128 outputs lines like "    M: -14.2  S: -15.1  ..."
    momentary_values = re.findall(r"M:\s*([-\d.]+)", stderr)
    if momentary_values:
        try:
            momentary_max = max(float(v) for v in momentary_values)
        except ValueError:
            pass

    # Short-term max
    short_term_values = re.findall(r"S:\s*([-\d.]+)", stderr)
    if short_term_values:
        try:
            short_term_max = max(float(v) for v in short_term_values)
        except ValueError:
            pass

    if on_progress:
        on_progress(100, f"Loudness: {integrated:.1f} LUFS")

    return LoudnessResult(
        integrated_lufs=round(integrated, 1),
        true_peak_dbtp=round(true_peak, 1),
        lra=round(lra, 1),
        momentary_max=round(momentary_max, 1),
        short_term_max=round(short_term_max, 1),
    )


# ---------------------------------------------------------------------------
# Spectrum analysis
# ---------------------------------------------------------------------------

# Frequency band definitions (Hz ranges)
_BANDS = [
    ("sub_bass",   20,    60),
    ("bass",       60,   250),
    ("low_mid",   250,   500),
    ("mid",       500,  2000),
    ("upper_mid", 2000,  4000),
    ("presence",  4000,  6000),
    ("brilliance", 6000, 20000),
]


def analyze_spectrum(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> SpectrumResult:
    """
    Analyse audio frequency spectrum and return average dB per band.

    Uses FFmpeg astats filter on band-pass filtered copies of the audio
    to measure average RMS level per frequency band.

    Args:
        input_path: Source audio or video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        SpectrumResult with per-band average dB levels.
    """
    if on_progress:
        on_progress(5, "Analysing frequency spectrum...")

    band_levels: Dict[str, float] = {}
    total_bands = len(_BANDS)

    for i, (name, lo, hi) in enumerate(_BANDS):
        if on_progress:
            pct = 10 + int(80 * i / total_bands)
            on_progress(pct, f"Measuring {name} ({lo}-{hi} Hz)...")

        # Use bandpass filter + astats to get RMS level for each band
        af_chain = (
            f"highpass=f={lo},lowpass=f={hi},"
            f"astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
        )

        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-nostats",
            "-i", input_path,
            "-af", af_chain,
            "-f", "null", "-",
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            logger.warning("Spectrum band %s timed out", name)
            band_levels[name] = -70.0
            continue

        # Parse RMS values from stderr (astats outputs them there)
        combined = proc.stdout + proc.stderr
        rms_values = re.findall(
            r"lavfi\.astats\.Overall\.RMS_level=([-\d.]+|inf|-inf)",
            combined,
        )

        if rms_values:
            valid = []
            for v in rms_values:
                try:
                    fv = float(v)
                    if fv > -200:  # filter out -inf / nonsense
                        valid.append(fv)
                except ValueError:
                    continue
            if valid:
                band_levels[name] = round(sum(valid) / len(valid), 1)
            else:
                band_levels[name] = -70.0
        else:
            band_levels[name] = -70.0

    if on_progress:
        on_progress(100, "Spectrum analysis complete")

    return SpectrumResult(
        sub_bass=band_levels.get("sub_bass", -70.0),
        bass=band_levels.get("bass", -70.0),
        low_mid=band_levels.get("low_mid", -70.0),
        mid=band_levels.get("mid", -70.0),
        upper_mid=band_levels.get("upper_mid", -70.0),
        presence=band_levels.get("presence", -70.0),
        brilliance=band_levels.get("brilliance", -70.0),
    )


# ---------------------------------------------------------------------------
# Platform loudness compliance check
# ---------------------------------------------------------------------------
def check_platform_loudness(
    input_path: str,
    platform: str = "youtube",
    on_progress: Optional[Callable] = None,
) -> PlatformLoudnessResult:
    """
    Check if audio meets a streaming platform's loudness standard.

    Measures actual loudness via EBU R128 and compares against
    the platform's target LUFS.  A file "passes" if its integrated
    loudness is within 1.0 LU of the target (standard tolerance).

    Args:
        input_path: Source audio or video file.
        platform: Target platform name (youtube, spotify, apple_podcasts,
                  broadcast, tiktok).
        on_progress: Progress callback(pct, msg).

    Returns:
        PlatformLoudnessResult with pass/fail and needed adjustment.
    """
    platform = platform.lower().strip()
    target = PLATFORM_TARGETS.get(platform, -14.0)

    if on_progress:
        on_progress(5, f"Checking loudness for {platform} (target {target} LUFS)...")

    loudness = measure_loudness(input_path, on_progress=on_progress)

    actual = loudness.integrated_lufs
    adjustment = target - actual
    # Passes if within 1.0 LU tolerance
    passes = abs(adjustment) <= 1.0

    return PlatformLoudnessResult(
        platform=platform,
        target_lufs=target,
        actual_lufs=actual,
        passes=passes,
        adjustment_needed_db=round(adjustment, 1),
    )
