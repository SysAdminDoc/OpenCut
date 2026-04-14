"""
OpenCut Visual Spectrogram Editor

Generate a spectrogram via STFT, accept a time-frequency mask, apply the
mask via inverse STFT, preview and export cleaned audio.

Uses librosa/numpy when available, falls back to FFmpeg showspectrumpic
for visualization.
"""

import json
import logging
import math
import os
import struct
import tempfile
import wave
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, output_path as _output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Spectrogram generation
# ---------------------------------------------------------------------------
def _generate_spectrogram_librosa(
    input_path: str, n_fft: int = 2048, hop_length: int = 512,
) -> Optional[dict]:
    """Generate spectrogram data using librosa."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None

    try:
        y, sr = librosa.load(input_path, sr=None, mono=True)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        power_db = librosa.amplitude_to_db(magnitude, ref=np.max)

        return {
            "stft": stft,
            "magnitude": magnitude,
            "phase": phase,
            "power_db": power_db,
            "sr": sr,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "audio": y,
            "duration": len(y) / sr,
            "freq_bins": magnitude.shape[0],
            "time_frames": magnitude.shape[1],
        }
    except Exception as e:
        logger.warning("librosa spectrogram failed: %s", e)
        return None


def _generate_spectrogram_image(input_path: str, output_image: str, width: int = 1920, height: int = 512) -> None:
    """Generate spectrogram image using FFmpeg showspectrumpic."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-lavfi", f"showspectrumpic=s={width}x{height}:mode=combined:color=intensity",
        output_image,
    ]
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------
def _apply_mask_librosa(
    stft_data: dict,
    mask: List[dict],
) -> str:
    """
    Apply a time-frequency mask to STFT data and reconstruct audio.

    Args:
        stft_data: Dict from _generate_spectrogram_librosa.
        mask: List of mask regions {time_start, time_end, freq_low, freq_high, gain}.

    Returns:
        Path to reconstructed WAV file.
    """
    import librosa
    import numpy as np

    stft = stft_data["stft"].copy()
    sr = stft_data["sr"]
    hop_length = stft_data["hop_length"]
    n_fft = stft_data["n_fft"]
    freq_bins = stft.shape[0]
    time_frames = stft.shape[1]

    freq_resolution = sr / n_fft  # Hz per bin
    time_resolution = hop_length / sr  # seconds per frame

    for m in mask:
        t_start = float(m.get("time_start", 0))
        t_end = float(m.get("time_end", stft_data["duration"]))
        f_low = float(m.get("freq_low", 0))
        f_high = float(m.get("freq_high", sr / 2))
        gain = float(m.get("gain", 0.0))  # 0.0 = remove, 1.0 = keep

        frame_start = max(0, int(t_start / time_resolution))
        frame_end = min(time_frames, int(t_end / time_resolution))
        bin_low = max(0, int(f_low / freq_resolution))
        bin_high = min(freq_bins, int(f_high / freq_resolution))

        stft[bin_low:bin_high, frame_start:frame_end] *= gain

    # Inverse STFT
    audio = librosa.istft(stft, hop_length=hop_length)

    # Write to WAV
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="spec_edit_")
    tmpfile.close()

    import soundfile as sf
    sf.write(tmpfile.name, audio, sr)

    return tmpfile.name


def _apply_mask_ffmpeg(
    input_path: str,
    mask: List[dict],
    duration: float,
) -> str:
    """
    Apply frequency mask using FFmpeg bandreject/bandpass filters.

    Fallback when librosa is not available.
    """
    ffmpeg = get_ffmpeg_path()
    filters = []

    for m in mask:
        f_low = float(m.get("freq_low", 0))
        f_high = float(m.get("freq_high", 20000))
        gain = float(m.get("gain", 0.0))
        t_start = float(m.get("time_start", 0))
        t_end = float(m.get("time_end", duration))

        if gain < 0.1:
            # Remove this frequency range
            center = (f_low + f_high) / 2
            width = max(1, f_high - f_low)
            q = center / max(width, 1)
            enable = f"between(t,{t_start},{t_end})"
            filters.append(f"bandreject=f={center}:width_type=h:w={width}:enable='{enable}'")

    if not filters:
        filters.append("anull")

    af = ",".join(filters)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="spec_edit_")
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
def generate_spectrogram(
    input_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    image_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate a spectrogram from audio.

    Args:
        input_path: Source audio/video file.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        image_path: Path for spectrogram image (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with image_path, duration, freq_bins, time_frames, method.
    """
    n_fft = max(256, min(8192, int(n_fft)))
    hop_length = max(64, min(n_fft, int(hop_length)))

    if on_progress:
        on_progress(10, "Generating spectrogram...")

    if image_path is None:
        image_path = os.path.splitext(_output_path(input_path, "spectrogram"))[0] + ".png"

    # Try librosa first for data + image
    spec_data = _generate_spectrogram_librosa(input_path, n_fft, hop_length)
    method = "librosa"

    if spec_data is None:
        method = "ffmpeg"
        _generate_spectrogram_image(input_path, image_path)

        if on_progress:
            on_progress(100, "Spectrogram generated (FFmpeg)")

        return {
            "image_path": image_path,
            "duration": 0.0,
            "freq_bins": 0,
            "time_frames": 0,
            "method": method,
        }

    # Save spectrogram image via librosa
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import librosa.display

        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        librosa.display.specshow(spec_data["power_db"], sr=spec_data["sr"],
                                 hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title("Spectrogram")
        plt.colorbar(ax.images[0], ax=ax, format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(image_path, dpi=150)
        plt.close(fig)
    except Exception:
        # Fallback to FFmpeg image
        _generate_spectrogram_image(input_path, image_path)

    if on_progress:
        on_progress(100, "Spectrogram generated")

    return {
        "image_path": image_path,
        "duration": spec_data["duration"],
        "freq_bins": spec_data["freq_bins"],
        "time_frames": spec_data["time_frames"],
        "method": method,
    }


def apply_spectrogram_mask(
    input_path: str,
    mask: List[dict],
    n_fft: int = 2048,
    hop_length: int = 512,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a time-frequency mask to audio via spectrogram editing.

    Args:
        input_path: Source audio file.
        mask: List of mask regions with keys:
              time_start, time_end (seconds),
              freq_low, freq_high (Hz),
              gain (0.0=remove, 1.0=keep).
        n_fft: FFT window size.
        hop_length: STFT hop length.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, regions_applied, method.
    """
    if output_path_str is None:
        output_path_str = _output_path(input_path, "spec_edited")
        output_path_str = os.path.splitext(output_path_str)[0] + ".wav"

    if not mask:
        if on_progress:
            on_progress(100, "No mask regions, copying original")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {"output_path": output_path_str, "regions_applied": 0, "method": "none"}

    if on_progress:
        on_progress(10, "Loading audio and computing STFT...")

    # Try librosa path
    spec_data = _generate_spectrogram_librosa(input_path, n_fft, hop_length)

    if spec_data is not None:
        if on_progress:
            on_progress(50, "Applying mask via inverse STFT...")

        tmp_path = _apply_mask_librosa(spec_data, mask)
        method = "librosa_istft"
    else:
        if on_progress:
            on_progress(50, "Applying mask via FFmpeg filters...")

        from opencut.helpers import get_video_info
        info = get_video_info(input_path)
        tmp_path = _apply_mask_ffmpeg(input_path, mask, info.get("duration", 0))
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
        on_progress(100, "Spectrogram edit applied")

    return {
        "output_path": output_path_str,
        "regions_applied": len(mask),
        "method": method,
    }
