"""
OpenCut AI Voice Conversion / RVC Module v0.1.0

Convert speaker voice to match a target voice model:
- Load RVC (Retrieval-based Voice Conversion) models
- Extract pitch (f0) from source audio
- Convert voice timbre while preserving pitch/intonation
- Falls back to FFmpeg pitch-shift + formant approximation

Supports .pth voice model files in ~/.opencut/models/rvc/
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Default model directory
RVC_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models", "rvc")
RVC_MODEL_EXTENSIONS = (".pth",)
RVC_INDEX_EXTENSIONS = (".index",)
DEFAULT_RVC_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Configuration / Result
# ---------------------------------------------------------------------------
@dataclass
class VoiceConversionConfig:
    """Configuration for voice conversion."""
    pitch_shift: int = 0           # Semitones to shift (-12 to +12)
    index_rate: float = 0.75       # How much to use the voice index (0.0-1.0)
    filter_radius: int = 3         # Median filter radius for pitch
    rms_mix_rate: float = 0.25     # Volume envelope mixing rate
    protect: float = 0.33          # Protect voiceless consonants (0.0-0.5)
    f0_method: str = "harvest"     # Pitch extraction: "harvest", "crepe", "pm"
    sample_rate: int = 44100       # Output sample rate


@dataclass
class VoiceConversionResult:
    """Result of voice conversion."""
    output_path: str = ""
    model_used: str = ""
    duration: float = 0.0
    method: str = ""               # "rvc" or "ffmpeg_fallback"


# ---------------------------------------------------------------------------
# Voice Model Management
# ---------------------------------------------------------------------------
def list_voice_models() -> List[Dict]:
    """
    List available RVC voice models in the models directory.

    Scans ~/.opencut/models/rvc/ for .pth files.

    Returns:
        List of dicts with 'name', 'path', 'size_mb' for each model.
    """
    models = []
    os.makedirs(RVC_MODELS_DIR, exist_ok=True)

    for entry in os.listdir(RVC_MODELS_DIR):
        if entry.endswith(RVC_MODEL_EXTENSIONS):
            full_path = os.path.join(RVC_MODELS_DIR, entry)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            name = os.path.splitext(entry)[0]
            models.append({
                "name": name,
                "path": full_path,
                "size_mb": round(size_mb, 1),
            })

    # Also scan for index files (.index)
    index_files = {
        os.path.splitext(f)[0]: os.path.join(RVC_MODELS_DIR, f)
        for f in os.listdir(RVC_MODELS_DIR) if f.endswith(RVC_INDEX_EXTENSIONS)
    }

    for model in models:
        model["has_index"] = model["name"] in index_files
        if model["has_index"]:
            model["index_path"] = index_files[model["name"]]

    models.sort(key=lambda m: m["name"])
    return models


def _find_model(model_path: str) -> str:
    """Resolve a model path or name to a full path."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise FileNotFoundError("Voice model not found: empty model path")

    model_path = model_path.strip()
    looks_like_path = (
        os.path.isabs(model_path)
        or os.path.sep in model_path
        or (os.path.altsep and os.path.altsep in model_path)
    )

    if looks_like_path:
        resolved = os.path.realpath(os.path.normpath(model_path))
        if os.path.isfile(resolved) and resolved.lower().endswith(RVC_MODEL_EXTENSIONS):
            return resolved
        raise FileNotFoundError(f"Voice model not found: {model_path}")

    if ".." in model_path.replace("\\", "/").split("/"):
        raise FileNotFoundError(f"Voice model not found: {model_path}")

    # Try as a name in the models directory
    candidate = os.path.join(RVC_MODELS_DIR, model_path)
    if os.path.isfile(candidate) and candidate.lower().endswith(RVC_MODEL_EXTENSIONS):
        return candidate

    if not model_path.endswith(".pth"):
        candidate = os.path.join(RVC_MODELS_DIR, f"{model_path}.pth")
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Voice model not found: {model_path}. "
        f"Place .pth models in {RVC_MODELS_DIR}"
    )


def _find_index_for_model(model_path: str) -> str:
    """Return the sibling RVC index path for a model if one exists."""
    base, _ext = os.path.splitext(model_path)
    for ext in RVC_INDEX_EXTENSIONS:
        candidate = f"{base}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return ""


def _discover_rvc_backend() -> str:
    """Find an external RVC inference executable or script."""
    env_candidates = [
        os.environ.get("OPENCUT_RVC_CLI", ""),
        os.environ.get("OPENCUT_RVC_INFER_CLI", ""),
    ]
    for candidate in env_candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.realpath(candidate)

    for exe_name in ("rvc_cli", "rvc"):
        resolved = shutil.which(exe_name)
        if resolved:
            return resolved

    script_candidates = [
        os.path.join(RVC_MODELS_DIR, "RVC", "infer_cli.py"),
        os.path.expanduser("~/.opencut/models/RVC/infer_cli.py"),
        os.path.expanduser("~/RVC/infer_cli.py"),
        os.path.expanduser("~/Retrieval-based-Voice-Conversion-WebUI/infer_cli.py"),
    ]
    for candidate in script_candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return ""


def _build_rvc_command(
    backend_path: str,
    input_audio: str,
    model_path: str,
    output_audio: str,
    config: VoiceConversionConfig,
    index_path: str = "",
) -> List[str]:
    """Build a best-effort RVC inference command for common CLI variants."""
    if backend_path.lower().endswith(".py"):
        python_exe = os.environ.get("OPENCUT_RVC_PYTHON", sys.executable)
        cmd = [
            python_exe,
            backend_path,
            "--input_path", input_audio,
            "--output_path", output_audio,
            "--model_path", model_path,
            "--f0_method", config.f0_method,
            "--transpose", str(config.pitch_shift),
            "--index_rate", f"{config.index_rate:.3f}",
            "--filter_radius", str(config.filter_radius),
            "--rms_mix_rate", f"{config.rms_mix_rate:.3f}",
            "--protect", f"{config.protect:.3f}",
        ]
        if index_path:
            cmd.extend(["--index_path", index_path])
        return cmd

    cmd = [
        backend_path,
        "infer",
        "--input", input_audio,
        "--output", output_audio,
        "--model", model_path,
        "--f0-method", config.f0_method,
        "--transpose", str(config.pitch_shift),
        "--index-rate", f"{config.index_rate:.3f}",
        "--filter-radius", str(config.filter_radius),
        "--rms-mix-rate", f"{config.rms_mix_rate:.3f}",
        "--protect", f"{config.protect:.3f}",
    ]
    if index_path:
        cmd.extend(["--index", index_path])
    return cmd


def _run_rvc_conversion(
    input_audio: str,
    model_path: str,
    output_audio: str,
    config: VoiceConversionConfig,
    on_progress: Optional[Callable] = None,
) -> str:
    """Run an external RVC backend and return the converted audio path."""
    backend = _discover_rvc_backend()
    if not backend:
        raise RuntimeError(
            "RVC backend not found. Set OPENCUT_RVC_CLI/OPENCUT_RVC_INFER_CLI "
            "or install rvc_cli."
        )

    if on_progress:
        on_progress(35, "Running RVC voice conversion...")

    index_path = _find_index_for_model(model_path)
    cmd = _build_rvc_command(
        backend_path=backend,
        input_audio=input_audio,
        model_path=model_path,
        output_audio=output_audio,
        config=config,
        index_path=index_path,
    )
    timeout = int(os.environ.get("OPENCUT_RVC_TIMEOUT", DEFAULT_RVC_TIMEOUT))
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-800:]
        raise RuntimeError(f"RVC conversion failed: {stderr}")
    if not os.path.isfile(output_audio):
        raise RuntimeError("RVC conversion finished without creating output audio")

    if on_progress:
        on_progress(75, "RVC voice conversion complete")
    return output_audio


# ---------------------------------------------------------------------------
# FFmpeg Fallback Voice Conversion
# ---------------------------------------------------------------------------
def _convert_voice_ffmpeg(
    audio_path: str,
    output_path: str,
    config: VoiceConversionConfig,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Fallback voice conversion using FFmpeg audio filters.

    Applies pitch shifting and formant-like manipulation using
    rubberband or asetrate/atempo filters.
    """
    if on_progress:
        on_progress(30, "Applying FFmpeg voice transformation...")

    # Calculate pitch shift as frequency ratio
    pitch_semitones = config.pitch_shift
    if pitch_semitones == 0:
        pitch_semitones = 2  # Default subtle shift if no model available

    # Use asetrate + aresample for pitch shift (changes speed),
    # then atempo to correct speed back
    ratio = 2.0 ** (pitch_semitones / 12.0)
    original_rate = config.sample_rate
    new_rate = int(original_rate * ratio)

    # atempo only accepts 0.5-2.0 range, chain if needed
    tempo = 1.0 / ratio
    tempo_filters = []
    remaining = tempo
    while remaining < 0.5 or remaining > 2.0:
        if remaining < 0.5:
            tempo_filters.append("atempo=0.5")
            remaining /= 0.5
        else:
            tempo_filters.append("atempo=2.0")
            remaining /= 2.0
    tempo_filters.append(f"atempo={remaining:.6f}")

    af_chain = f"asetrate={new_rate}," + ",".join(tempo_filters) + f",aresample={original_rate}"

    # Add subtle reverb/EQ for more natural sound
    af_chain += ",equalizer=f=200:t=q:w=1:g=2,equalizer=f=3000:t=q:w=1:g=-1"

    output_ext = os.path.splitext(output_path)[1].lower()
    audio_codec = "pcm_s16le" if output_ext == ".wav" else "aac"
    cmd = (
        FFmpegCmd()
        .input(audio_path)
        .audio_filter(af_chain)
        .audio_codec(audio_codec, bitrate=None if audio_codec == "pcm_s16le" else "192k")
        .option("-ar", str(original_rate))
        .output(output_path)
    )

    run_ffmpeg(cmd.build(), timeout=300)

    if on_progress:
        on_progress(80, "FFmpeg voice conversion complete")

    return output_path


# ---------------------------------------------------------------------------
# Main Voice Conversion
# ---------------------------------------------------------------------------
def convert_voice(
    audio_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[VoiceConversionConfig] = None,
    on_progress: Optional[Callable] = None,
) -> VoiceConversionResult:
    """
    Convert the speaker's voice in an audio/video file to a target voice.

    Attempts to use RVC (Retrieval-based Voice Conversion) if the
    required libraries are available. Falls back to FFmpeg-based
    pitch/formant shifting.

    Args:
        audio_path: Input audio or video file path.
        model_path: Path to .pth voice model or model name.
        output_path: Optional explicit output path.
        output_dir: Output directory (defaults to input dir).
        config: VoiceConversionConfig with conversion parameters.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        VoiceConversionResult with output path and metadata.
    """
    if config is None:
        config = VoiceConversionConfig()

    result = VoiceConversionResult()

    # Resolve model path
    try:
        resolved_model = _find_model(model_path)
        result.model_used = os.path.basename(resolved_model)
    except FileNotFoundError:
        logger.warning("Voice model not found: %s, will use FFmpeg fallback", model_path)
        resolved_model = None
        result.model_used = "ffmpeg_fallback"

    if output_path is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        directory = output_dir or os.path.dirname(audio_path)
        ext = ".mp4" if audio_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov")) else ".wav"
        output_path = os.path.join(directory, f"{base}_voice_converted{ext}")

    if on_progress:
        on_progress(5, "Preparing voice conversion...")

    # Determine if input is video (needs audio extraction + remux)
    is_video = audio_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm"))
    tmp_files = []

    try:
        # Extract audio if video
        if is_video:
            _ntf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            src_audio = _ntf.name
            _ntf.close()
            tmp_files.append(src_audio)

            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", audio_path, "-vn",
                "-acodec", "pcm_s16le", "-ar", str(config.sample_rate), "-ac", "1",
                src_audio,
            ], timeout=120)

            if on_progress:
                on_progress(15, "Audio extracted from video...")
        else:
            src_audio = audio_path

        info = get_video_info(src_audio)
        result.duration = info.get("duration", 0)

        # Try RVC conversion
        rvc_success = False
        if resolved_model is not None:
            try:
                if on_progress:
                    on_progress(20, "Attempting RVC voice conversion...")

                _ntf2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                converted_audio = _ntf2.name
                _ntf2.close()
                tmp_files.append(converted_audio)

                _run_rvc_conversion(
                    input_audio=src_audio,
                    model_path=resolved_model,
                    output_audio=converted_audio,
                    config=config,
                    on_progress=on_progress,
                )
                result.method = "rvc"
                rvc_success = True

            except Exception as e:
                logger.info("RVC conversion unavailable (%s), using FFmpeg fallback", e)
                rvc_success = False

        # Fallback to FFmpeg conversion
        if not rvc_success:
            _ntf3 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            converted_audio = _ntf3.name
            _ntf3.close()
            tmp_files.append(converted_audio)

            _convert_voice_ffmpeg(src_audio, converted_audio, config, on_progress)
            result.method = "ffmpeg_fallback"

        if on_progress:
            on_progress(85, "Finalizing output...")

        # Mux back with video if needed
        if is_video:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", audio_path, "-i", converted_audio,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest", output_path,
            ], timeout=600)
        else:
            # Just encode the converted audio
            output_ext = os.path.splitext(output_path)[1].lower()
            if output_ext == ".wav":
                run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", converted_audio,
                    "-acodec", "pcm_s16le", "-ar", str(config.sample_rate),
                    output_path,
                ], timeout=120)
            else:
                run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", converted_audio,
                    "-acodec", "aac", "-b:a", "192k",
                    output_path,
                ], timeout=120)

        result.output_path = output_path

    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if on_progress:
        on_progress(100, "Voice conversion complete!")

    return result
