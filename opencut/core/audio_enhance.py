"""
OpenCut Audio Enhancement Module

Speech super-resolution using Resemble Enhance.
Upsamples low-quality speech audio to studio quality.

Requires: pip install resemble-enhance
"""

import logging
import os
import subprocess
import tempfile
from typing import Callable, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"})
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"})


def _is_video(filepath):
    """Check if file is a video (vs pure audio) by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _VIDEO_EXTENSIONS:
        return True
    if ext in _AUDIO_EXTENSIONS:
        return False
    # Unknown extension — probe for video stream
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=10,
        )
        return "video" in result.stdout.lower()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def _extract_audio(input_path, output_wav):
    """
    Extract audio from video file as WAV using FFmpeg.

    Args:
        input_path: Path to the video file.
        output_wav: Path to write the extracted WAV.

    Raises:
        RuntimeError: If FFmpeg extraction fails.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        output_wav,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Audio extraction timed out for '{os.path.basename(input_path)}'")

    if result.returncode != 0:
        stderr = result.stderr.strip()[-500:] if result.stderr else "unknown error"
        raise RuntimeError(f"Audio extraction failed: {stderr}")

    if not os.path.isfile(output_wav) or os.path.getsize(output_wav) == 0:
        raise RuntimeError(f"Audio extraction produced empty output for '{os.path.basename(input_path)}'")

    logger.info("Extracted audio: %s -> %s", input_path, output_wav)


# ---------------------------------------------------------------------------
# Main enhancement function
# ---------------------------------------------------------------------------
def enhance_speech(
    input_path,
    output_path=None,
    output_dir="",
    denoise=True,
    enhance=True,
    solver="midpoint",
    nfe=64,
    chunk_seconds=30.0,
    on_progress=None,
):
    """
    Enhance speech audio using Resemble Enhance.

    Applies denoising and/or super-resolution to produce studio-quality
    speech from low-quality recordings. Works on both audio and video files
    (audio is extracted from video automatically).

    Args:
        input_path: Path to input audio/video file.
        output_path: Optional explicit output path.
        output_dir: Directory for output (auto-names if output_path not given).
        denoise: Apply denoising pass.
        enhance: Apply enhancement/super-resolution pass.
        solver: ODE solver ("midpoint", "rk4", "euler").
        nfe: Number of function evaluations (higher = better quality, slower).
        chunk_seconds: Process audio in chunks of this length.
        on_progress: Progress callback(pct, msg).

    Returns:
        Output file path string.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not denoise and not enhance:
        raise ValueError("At least one of 'denoise' or 'enhance' must be True")

    if on_progress:
        on_progress(5, "Loading audio...")

    # Lazy imports — these are heavy
    try:
        import torch
        import torchaudio
    except ImportError:
        raise RuntimeError(
            "torch and torchaudio are required. Install with: "
            "pip install torch torchaudio"
        )

    try:
        from resemble_enhance.enhancer.inference import denoise as _denoise_fn, enhance as _enhance_fn
    except ImportError:
        raise RuntimeError(
            "resemble-enhance is required. Install with: "
            "pip install resemble-enhance"
        )

    # If input is video, extract audio to temp WAV
    temp_wav = None
    audio_path = input_path

    if _is_video(input_path):
        if on_progress:
            on_progress(10, "Extracting audio from video...")

        temp_wav = tempfile.mktemp(suffix=".wav", prefix="opencut_enhance_")
        _extract_audio(input_path, temp_wav)
        audio_path = temp_wav

    try:
        # Load audio
        if on_progress:
            on_progress(15, "Loading audio data...")

        audio, sr = torchaudio.load(audio_path)
        logger.info("Loaded audio: %s (sr=%d, shape=%s)", audio_path, sr, audio.shape)

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info("Using GPU for audio enhancement")

        # Denoising pass
        if denoise:
            if on_progress:
                on_progress(30, "Denoising speech...")

            logger.info("Running denoiser (device=%s)...", device)
            audio, sr = _denoise_fn(audio.squeeze(0), sr, device)
            # _denoise returns (tensor, sr) — ensure 2D for torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            logger.info("Denoising complete (sr=%d)", sr)

        # Enhancement/super-resolution pass
        if enhance:
            if on_progress:
                on_progress(70, "Enhancing speech quality...")

            logger.info("Running enhancer (solver=%s, nfe=%d, device=%s)...", solver, nfe, device)
            audio, sr = _enhance_fn(
                audio.squeeze(0), sr,
                device,
                nfe=nfe,
                solver=solver,
                chunk_seconds=chunk_seconds,
            )
            # _enhance returns (tensor, sr) — ensure 2D for torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            logger.info("Enhancement complete (sr=%d)", sr)

        # Build output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            suffix = "_enhanced.wav"

            if output_dir and os.path.isdir(output_dir):
                output_path = os.path.join(output_dir, base_name + suffix)
            else:
                output_path = os.path.join(
                    os.path.dirname(input_path), base_name + suffix
                )

        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if on_progress:
            on_progress(90, "Saving enhanced audio...")

        torchaudio.save(output_path, audio, sr)
        logger.info("Saved enhanced audio: %s", output_path)

        if on_progress:
            on_progress(100, "Audio enhanced!")

        return output_path

    finally:
        # Clean up temp file
        if temp_wav and os.path.isfile(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass
