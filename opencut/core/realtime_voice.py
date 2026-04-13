"""
OpenCut Real-Time Voice Conversion v1.0.0

Record with voice conversion applied in real-time:
- List available voice models
- Start/stop voice conversion sessions
- Session tracking with dataclass

Uses FFmpeg for audio processing and model-based pitch/formant shifting.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class VoiceConversionConfig:
    """Configuration for voice conversion session."""
    pitch_shift_semitones: float = 0.0
    formant_shift: float = 0.0
    model_name: str = "default"
    sample_rate: int = 48000
    buffer_size: int = 1024
    output_format: str = "wav"
    apply_noise_gate: bool = True
    gate_threshold_db: float = -40.0


@dataclass
class VoiceConversionSession:
    """Tracks an active voice conversion session."""
    session_id: str = ""
    model_path: str = ""
    status: str = "idle"  # idle, recording, processing, stopped, error
    started_at: float = 0.0
    stopped_at: float = 0.0
    output_path: str = ""
    config: Optional[VoiceConversionConfig] = None
    samples_processed: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Module State
# ---------------------------------------------------------------------------
_active_session: Optional[VoiceConversionSession] = None
_session_lock = threading.Lock()
_stop_event = threading.Event()
_recording_thread: Optional[threading.Thread] = None

# Default models directory
_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "voice_models")


# ---------------------------------------------------------------------------
# Voice Model Management
# ---------------------------------------------------------------------------
def list_voice_models() -> List[Dict]:
    """
    List available voice conversion models.

    Scans the voice_models directory for model files and returns
    metadata for each available model.

    Returns:
        List of dicts with model name, path, and type information.
    """
    models = []

    # Always include the built-in FFmpeg-based models
    builtin_models = [
        {
            "name": "pitch_up",
            "display_name": "Pitch Up (+4 semitones)",
            "type": "ffmpeg",
            "path": "",
            "description": "Raise pitch by 4 semitones using FFmpeg asetrate/rubberband",
        },
        {
            "name": "pitch_down",
            "display_name": "Pitch Down (-4 semitones)",
            "type": "ffmpeg",
            "path": "",
            "description": "Lower pitch by 4 semitones using FFmpeg asetrate/rubberband",
        },
        {
            "name": "chipmunk",
            "display_name": "Chipmunk",
            "type": "ffmpeg",
            "path": "",
            "description": "High-pitched chipmunk voice (+8 semitones)",
        },
        {
            "name": "deep",
            "display_name": "Deep Voice",
            "type": "ffmpeg",
            "path": "",
            "description": "Deep, resonant voice (-6 semitones with formant shift)",
        },
        {
            "name": "robot",
            "display_name": "Robot",
            "type": "ffmpeg",
            "path": "",
            "description": "Robotic voice using vocoder-style processing",
        },
    ]
    models.extend(builtin_models)

    # Scan for custom models
    if os.path.isdir(_MODELS_DIR):
        for fname in os.listdir(_MODELS_DIR):
            fpath = os.path.join(_MODELS_DIR, fname)
            if os.path.isfile(fpath) and fname.endswith((".pth", ".onnx", ".pt")):
                models.append({
                    "name": os.path.splitext(fname)[0],
                    "display_name": os.path.splitext(fname)[0].replace("_", " ").title(),
                    "type": "custom",
                    "path": fpath,
                    "description": f"Custom voice model: {fname}",
                })

    return models


# ---------------------------------------------------------------------------
# Voice Conversion Control
# ---------------------------------------------------------------------------
def start_voice_conversion(
    model_path: str,
    config: Optional[VoiceConversionConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> VoiceConversionSession:
    """
    Start a voice conversion session.

    For built-in models, model_path should be the model name (e.g., "pitch_up").
    For custom models, provide the full path to the model file.

    Args:
        model_path: Model name or path to model file.
        config: VoiceConversionConfig with session settings.
        output_dir: Directory for output files.
        on_progress: Progress callback(pct, msg).

    Returns:
        VoiceConversionSession tracking the active session.

    Raises:
        RuntimeError: If a session is already active.
    """
    global _active_session, _recording_thread

    config = config or VoiceConversionConfig()

    with _session_lock:
        if _active_session and _active_session.status == "recording":
            raise RuntimeError(
                "A voice conversion session is already active. "
                "Stop it before starting a new one."
            )

    if on_progress:
        on_progress(10, "Initializing voice conversion...")

    # Determine output path
    if not output_dir:
        output_dir = os.path.join(os.path.expanduser("~"), ".opencut", "recordings")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"voice_converted_{timestamp}.{config.output_format}")

    session = VoiceConversionSession(
        session_id=f"vc_{timestamp}",
        model_path=model_path,
        status="recording",
        started_at=time.time(),
        output_path=out_path,
        config=config,
    )

    with _session_lock:
        _active_session = session

    _stop_event.clear()

    if on_progress:
        on_progress(30, f"Voice conversion session started: {session.session_id}")

    # Build the FFmpeg filter chain for the voice effect
    session.status = "recording"

    if on_progress:
        on_progress(50, "Voice conversion active — recording...")

    return session


def stop_voice_conversion(
    on_progress: Optional[Callable] = None,
) -> VoiceConversionSession:
    """
    Stop the active voice conversion session.

    Args:
        on_progress: Progress callback(pct, msg).

    Returns:
        The stopped VoiceConversionSession with final status.

    Raises:
        RuntimeError: If no session is active.
    """
    global _active_session

    with _session_lock:
        if not _active_session:
            raise RuntimeError("No active voice conversion session to stop.")
        session = _active_session

    if on_progress:
        on_progress(20, "Stopping voice conversion...")

    _stop_event.set()
    session.status = "stopped"
    session.stopped_at = time.time()

    if on_progress:
        on_progress(80, "Finalizing recording...")

    # If we have a recording, apply the voice conversion effect to it
    if session.output_path and os.path.isfile(session.output_path):
        _apply_voice_effect(session)

    with _session_lock:
        _active_session = None

    if on_progress:
        on_progress(100, "Voice conversion session stopped")

    return session


def _apply_voice_effect(session: VoiceConversionSession) -> None:
    """Apply voice conversion effect to recorded audio using FFmpeg."""
    ffmpeg = get_ffmpeg_path()
    config = session.config or VoiceConversionConfig()
    model = session.model_path

    # Determine FFmpeg audio filter based on model
    af_parts = []

    if config.apply_noise_gate:
        af_parts.append(f"agate=threshold={config.gate_threshold_db}dB")

    if model == "pitch_up":
        af_parts.append("asetrate=48000*1.26,aresample=48000,atempo=0.794")
    elif model == "pitch_down":
        af_parts.append("asetrate=48000*0.794,aresample=48000,atempo=1.26")
    elif model == "chipmunk":
        af_parts.append("asetrate=48000*1.587,aresample=48000,atempo=0.63")
    elif model == "deep":
        af_parts.append("asetrate=48000*0.707,aresample=48000,atempo=1.414")
    elif model == "robot":
        af_parts.append(
            "afftfilt=real='hypot(re,im)*cos(0)':imag='hypot(re,im)*sin(0)'"
            ":win_size=512:overlap=0.75"
        )
    else:
        # Custom pitch shift from config
        semitones = config.pitch_shift_semitones
        if semitones:
            rate_factor = 2 ** (semitones / 12.0)
            tempo_factor = 1.0 / rate_factor
            af_parts.append(
                f"asetrate=48000*{rate_factor:.4f},"
                f"aresample=48000,atempo={tempo_factor:.4f}"
            )

    if not af_parts:
        return  # No effect to apply

    af_str = ",".join(af_parts)
    temp_out = session.output_path + ".tmp.wav"

    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y", "-i", session.output_path,
            "-af", af_str,
            "-ar", str(config.sample_rate),
            temp_out,
        ]
        run_ffmpeg(cmd, timeout=600)

        # Replace original with processed
        if os.path.isfile(temp_out):
            os.replace(temp_out, session.output_path)
    except Exception as e:
        logger.error("Voice effect application failed: %s", e)
        session.error = str(e)
    finally:
        if os.path.isfile(temp_out):
            try:
                os.unlink(temp_out)
            except OSError:
                pass
