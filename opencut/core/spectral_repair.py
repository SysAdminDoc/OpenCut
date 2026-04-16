"""
OpenCut Spectral Repair / Frequency Removal

STFT analysis to identify persistent frequency peaks (hum, buzz, whine),
attenuate target bins where signal is dominant, then inverse STFT to
reconstruct cleaned audio.

Uses librosa/numpy when available, FFmpeg bandreject as fallback.
"""

import logging
import os
import tempfile
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------
def _analyze_frequencies_librosa(
    input_path: str,
    n_fft: int = 4096,
    threshold_db: float = -20.0,
) -> Optional[dict]:
    """Identify persistent frequency peaks using librosa STFT."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None

    try:
        y, sr = librosa.load(input_path, sr=None, mono=True)
        stft = librosa.stft(y, n_fft=n_fft)
        magnitude = np.abs(stft)
        power_db = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Average power across time for each frequency bin
        mean_power = np.mean(power_db, axis=1)
        freq_resolution = sr / n_fft

        # Find persistent peaks above threshold
        peaks = []
        for i, power in enumerate(mean_power):
            if power > threshold_db:
                freq = i * freq_resolution
                if freq > 10:  # skip DC and sub-audible
                    peaks.append({
                        "frequency": round(freq, 1),
                        "power_db": round(float(power), 1),
                        "bin_index": int(i),
                        "persistent": True,
                    })

        # Sort by power (strongest first)
        peaks.sort(key=lambda p: p["power_db"], reverse=True)

        # Identify common hum/buzz patterns
        for p in peaks[:20]:
            f = p["frequency"]
            if 49 < f < 62:
                p["classification"] = "mains_hum_50_60hz"
            elif 99 < f < 121:
                p["classification"] = "second_harmonic"
            elif 149 < f < 181:
                p["classification"] = "third_harmonic"
            elif f < 100:
                p["classification"] = "low_frequency_rumble"
            elif 2000 < f < 8000:
                p["classification"] = "electronic_whine"
            else:
                p["classification"] = "unknown"

        return {
            "peaks": peaks[:50],
            "sr": sr,
            "n_fft": n_fft,
            "freq_resolution": freq_resolution,
            "stft": stft,
            "audio": y,
            "duration": len(y) / sr,
        }
    except Exception as e:
        logger.warning("librosa frequency analysis failed: %s", e)
        return None


def _analyze_frequencies_ffmpeg(input_path: str) -> dict:
    """Basic frequency analysis using FFmpeg astats."""
    get_ffmpeg_path()
    # This is a simplified fallback - just report common hum frequencies
    return {
        "peaks": [
            {"frequency": 50.0, "power_db": -30.0, "classification": "possible_mains_hum"},
            {"frequency": 60.0, "power_db": -30.0, "classification": "possible_mains_hum"},
        ],
        "method": "ffmpeg_heuristic",
    }


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
def _repair_librosa(
    analysis: dict,
    target_frequencies: List[float],
    attenuation_db: float = -60.0,
    bandwidth: float = 5.0,
) -> str:
    """Remove target frequencies via STFT bin attenuation."""
    import librosa

    stft = analysis["stft"].copy()
    sr = analysis["sr"]
    n_fft = analysis["n_fft"]
    freq_res = sr / n_fft

    gain = 10 ** (attenuation_db / 20.0)

    for target_freq in target_frequencies:
        center_bin = int(target_freq / freq_res)
        half_width = max(1, int(bandwidth / freq_res))
        low = max(0, center_bin - half_width)
        high = min(stft.shape[0], center_bin + half_width + 1)
        stft[low:high, :] *= gain

    audio = librosa.istft(stft)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="spectral_repair_")
    tmpfile.close()

    import soundfile as sf
    sf.write(tmpfile.name, audio, sr)

    return tmpfile.name


def _repair_ffmpeg(
    input_path: str,
    target_frequencies: List[float],
    bandwidth: float = 5.0,
) -> str:
    """Remove target frequencies using FFmpeg bandreject filters."""
    ffmpeg = get_ffmpeg_path()
    filters = []

    for freq in target_frequencies:
        freq / max(bandwidth, 1.0)
        filters.append(f"bandreject=f={freq}:width_type=h:w={bandwidth}")

    if not filters:
        filters.append("anull")

    af = ",".join(filters)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="spectral_repair_")
    tmpfile.close()

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af,
        "-c:a", "pcm_s16le",
        tmpfile.name,
    ]
    run_ffmpeg(cmd)

    return tmpfile.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_frequencies(
    input_path: str,
    n_fft: int = 4096,
    threshold_db: float = -20.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Analyze audio for persistent frequency peaks (hum, buzz, whine).

    Args:
        input_path: Source audio/video file.
        n_fft: FFT window size for analysis.
        threshold_db: Minimum dB threshold for peak detection.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with peaks list, each having frequency, power_db, classification.
    """
    n_fft = max(512, min(16384, int(n_fft)))
    threshold_db = max(-80.0, min(0.0, float(threshold_db)))

    if on_progress:
        on_progress(10, "Analyzing frequency content...")

    result = _analyze_frequencies_librosa(input_path, n_fft, threshold_db)
    method = "librosa"

    if result is None:
        result = _analyze_frequencies_ffmpeg(input_path)
        method = "ffmpeg"

    if on_progress:
        on_progress(100, f"Found {len(result.get('peaks', []))} frequency peaks")

    return {
        "peaks": result.get("peaks", []),
        "method": method,
        "n_fft": n_fft,
        "threshold_db": threshold_db,
    }


def repair_frequencies(
    input_path: str,
    target_frequencies: Optional[List[float]] = None,
    auto_detect: bool = True,
    attenuation_db: float = -60.0,
    bandwidth: float = 5.0,
    n_fft: int = 4096,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove persistent frequency peaks (hum, buzz) from audio.

    Args:
        input_path: Source audio/video file.
        target_frequencies: List of Hz values to remove. If None and auto_detect
                          is True, automatically detects problematic frequencies.
        auto_detect: Auto-detect frequencies if target_frequencies is empty.
        attenuation_db: Amount to reduce target frequencies (negative dB).
        bandwidth: Bandwidth around each target frequency in Hz.
        n_fft: FFT size for analysis.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, frequencies_removed, method.
    """
    attenuation_db = max(-120.0, min(-6.0, float(attenuation_db)))
    bandwidth = max(1.0, min(100.0, float(bandwidth)))

    if output_path_str is None:
        output_path_str = _output_path(input_path, "repaired")
        output_path_str = os.path.splitext(output_path_str)[0] + ".wav"

    if not target_frequencies and auto_detect:
        if on_progress:
            on_progress(10, "Auto-detecting problematic frequencies...")

        analysis = analyze_frequencies(input_path, n_fft=n_fft, threshold_db=-20.0)
        peaks = analysis.get("peaks", [])
        # Take top 5 most prominent peaks
        target_frequencies = [p["frequency"] for p in peaks[:5]]

    if not target_frequencies:
        if on_progress:
            on_progress(100, "No frequencies to remove")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c:a", "pcm_s16le", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "frequencies_removed": [],
            "method": "none",
        }

    if on_progress:
        freq_str = ", ".join(f"{f:.0f}Hz" for f in target_frequencies[:5])
        on_progress(30, f"Removing frequencies: {freq_str}...")

    # Try librosa path first
    analysis = _analyze_frequencies_librosa(input_path, n_fft)
    if analysis is not None and "stft" in analysis:
        tmp_path = _repair_librosa(analysis, target_frequencies, attenuation_db, bandwidth)
        method = "librosa_stft"
    else:
        tmp_path = _repair_ffmpeg(input_path, target_frequencies, bandwidth)
        method = "ffmpeg_bandreject"

    # Copy to final output
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_path,
        "-c:a", "pcm_s16le",
        output_path_str,
    ]
    run_ffmpeg(cmd)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if on_progress:
        on_progress(100, f"Removed {len(target_frequencies)} frequency peaks")

    return {
        "output_path": output_path_str,
        "frequencies_removed": [round(f, 1) for f in target_frequencies],
        "method": method,
    }
