"""
OpenCut Audio Enhancement Module

Speech denoising and super-resolution:
- ClearerVoice-Studio (recommended): MossFormer2/FRCRN, 16kHz/48kHz, denoise+enhance+separation
- Resemble Enhance (legacy): ODE-based super-resolution

Requires: pip install clearvoice (preferred) or pip install resemble-enhance (legacy)
"""

import logging
import os
import subprocess
import tempfile
from contextlib import suppress

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"})
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"})


def _binary_env(resolved_path):
    """Prefer invoking ffmpeg/ffprobe by name while honoring bundled binaries."""
    if not resolved_path:
        return None
    binary_dir = os.path.dirname(resolved_path)
    if not binary_dir:
        return None
    env = os.environ.copy()
    current_path = env.get("PATH", "")
    env["PATH"] = binary_dir if not current_path else binary_dir + os.pathsep + current_path
    return env


def _is_video(filepath):
    """Check if file is a video (vs pure audio) by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _VIDEO_EXTENSIONS:
        return True
    if ext in _AUDIO_EXTENSIONS:
        return False
    # Unknown extension — probe for video stream
    try:
        ffprobe_path = get_ffprobe_path()
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=10, env=_binary_env(ffprobe_path), check=False,
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
    ffmpeg_path = get_ffmpeg_path()
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        output_wav,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=_binary_env(ffmpeg_path),
            check=False,
        )
    except FileNotFoundError as err:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html") from err
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"Audio extraction timed out for '{os.path.basename(input_path)}'") from err

    if result.returncode != 0:
        stderr = result.stderr.strip()[-500:] if result.stderr else "unknown error"
        raise RuntimeError(f"Audio extraction failed: {stderr}")

    if not os.path.isfile(output_wav) or os.path.getsize(output_wav) == 0:
        raise RuntimeError(f"Audio extraction produced empty output for '{os.path.basename(input_path)}'")

    logger.debug("Extracted audio: %s -> %s", input_path, output_wav)


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
    except ImportError as err:
        raise RuntimeError(
            "torch and torchaudio are required. Install with: "
            "pip install torch torchaudio"
        ) from err

    try:
        from resemble_enhance.enhancer.inference import denoise as _denoise_fn
        from resemble_enhance.enhancer.inference import enhance as _enhance_fn
    except ImportError as err:
        raise RuntimeError(
            "resemble-enhance is required. Install with: "
            "pip install resemble-enhance"
        ) from err

    # If input is video, extract audio to temp WAV
    temp_wav = None
    audio_path = input_path

    if _is_video(input_path):
        if on_progress:
            on_progress(10, "Extracting audio from video...")

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="opencut_enhance_",
            delete=False,
        ) as _tmp:
            temp_wav = _tmp.name
        _extract_audio(input_path, temp_wav)
        audio_path = temp_wav

    try:
        # Load audio
        if on_progress:
            on_progress(15, "Loading audio data...")

        audio, sr = torchaudio.load(audio_path)
        logger.debug("Loaded audio: %s (sr=%d, shape=%s)", audio_path, sr, audio.shape)

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.debug("Using GPU for audio enhancement")

        # Denoising pass
        if denoise:
            if on_progress:
                on_progress(30, "Denoising speech...")

            logger.debug("Running denoiser (device=%s)...", device)
            # squeeze(0) is a no-op on stereo (2,N) — explicitly mix to mono
            if audio.shape[0] > 1:
                mono = audio.mean(dim=0)
            else:
                mono = audio.squeeze(0)
            audio, sr = _denoise_fn(mono, sr, device)
            # _denoise returns (tensor, sr) — ensure 2D for torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            logger.debug("Denoising complete (sr=%d)", sr)

        # Enhancement/super-resolution pass
        if enhance:
            if on_progress:
                on_progress(70, "Enhancing speech quality...")

            logger.debug("Running enhancer (solver=%s, nfe=%d, device=%s)...", solver, nfe, device)
            # squeeze(0) is a no-op on stereo (2,N) — explicitly mix to mono
            if audio.shape[0] > 1:
                mono = audio.mean(dim=0)
            else:
                mono = audio.squeeze(0)
            audio, sr = _enhance_fn(
                mono, sr,
                device,
                nfe=nfe,
                solver=solver,
                chunk_seconds=chunk_seconds,
            )
            # _enhance returns (tensor, sr) — ensure 2D for torchaudio
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            logger.debug("Enhancement complete (sr=%d)", sr)

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
            with suppress(OSError):
                os.remove(temp_wav)
        # Free GPU memory — delete all tensor references before clearing cache
        for _var in ("audio", "mono", "sr"):
            with suppress(Exception):
                if _var in locals():
                    obj = locals()[_var]
                    if hasattr(obj, "cpu"):
                        obj.cpu()  # move off GPU before delete
                    del obj
        with suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# ClearerVoice-Studio enhancement (recommended alternative)
# ---------------------------------------------------------------------------
def enhance_speech_clearvoice(
    input_path,
    output_path=None,
    output_dir="",
    task="speech_enhancement",
    model="MossFormer2_SE_48K",
    on_progress=None,
):
    """
    Enhance speech audio using ClearerVoice-Studio (Alibaba).

    Superior to Resemble Enhance: single library handles denoising,
    super-resolution, and separation. Supports 16kHz and 48kHz models.

    Args:
        input_path: Path to input audio/video file.
        output_path: Optional explicit output path.
        output_dir: Directory for output.
        task: "speech_enhancement" (denoise+enhance) or "speech_separation".
        model: ClearerVoice model name. Options:
            - "MossFormer2_SE_48K" (best quality, 48kHz)
            - "FRCRN_SE_16K" (fast, 16kHz, 3M+ uses on ModelScope)
            - "MossFormerGAN_SE_16K" (balanced, 16kHz)
        on_progress: Progress callback(pct, msg).

    Returns:
        Output file path string.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if on_progress:
        on_progress(5, "Loading ClearerVoice model...")

    try:
        from clearvoice import ClearVoice
    except ImportError as err:
        raise RuntimeError(
            "clearvoice is required. Install with: pip install clearvoice"
        ) from err

    # If input is video, extract audio to temp WAV
    temp_wav = None
    audio_path = input_path

    if _is_video(input_path):
        if on_progress:
            on_progress(10, "Extracting audio from video...")

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="opencut_cv_",
            delete=False,
        ) as _tmp:
            temp_wav = _tmp.name
        _extract_audio(input_path, temp_wav)
        audio_path = temp_wav

    try:
        if on_progress:
            on_progress(20, f"Running {model}...")

        cv = ClearVoice(task=task, model_names=[model])
        result = cv(input_path=audio_path, online_write=False)

        if on_progress:
            on_progress(80, "Saving enhanced audio...")

        # Build output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            suffix = "_enhanced.wav"
            if output_dir and os.path.isdir(output_dir):
                output_path = os.path.join(output_dir, base_name + suffix)
            else:
                output_path = os.path.join(os.path.dirname(input_path), base_name + suffix)

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # ClearVoice returns dict of {model: output_array} or writes to file
        # Write result using the library's write method
        cv.write(result, output_path=output_path)

        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("ClearerVoice produced empty or missing output file")

        if on_progress:
            on_progress(100, "Audio enhanced!")

        logger.info("ClearerVoice enhanced: %s -> %s", input_path, output_path)
        return output_path

    finally:
        if temp_wav and os.path.isfile(temp_wav):
            with suppress(OSError):
                os.remove(temp_wav)
        # Release model and result tensors before clearing GPU cache
        for _var in ("cv", "result"):
            with suppress(Exception):
                if _var in locals():
                    del locals()[_var]
        with suppress(Exception):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
