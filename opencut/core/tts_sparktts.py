"""
OpenCut Spark-TTS Backend

Zero-shot TTS with natural prosody and voice cloning from a 3-second
reference clip.  ONNX-compatible, runs on CPU without CUDA.

Licence: Apache-2.0
Repository: https://github.com/SparkAudio/Spark-TTS
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install sparktts  # Apache-2.0, CPU-native zero-shot TTS"

# Supported Spark-TTS voice presets (built-in speakers)
SPARK_VOICE_PRESETS = {
    "default": "Default — balanced English speaker",
    "warm": "Warm — conversational tone, lower pitch",
    "bright": "Bright — energetic, upbeat delivery",
    "calm": "Calm — slow, measured narration pace",
    "deep": "Deep — authoritative, low register",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SparkTTSResult:
    output: str = ""
    voice: str = ""
    model: str = "spark-tts"
    duration_seconds: float = 0.0
    sample_rate: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "model", "duration_seconds",
                "sample_rate", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_sparktts_available() -> bool:
    """Return True when the sparktts package is importable."""
    return _try_import("sparktts") is not None


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------

def list_voices() -> List[dict]:
    """Return available Spark-TTS voices (built-in presets)."""
    voices = []
    for vid, desc in SPARK_VOICE_PRESETS.items():
        voices.append({
            "voice_id": vid,
            "name": vid.replace("_", " ").title(),
            "description": desc,
            "type": "preset",
        })
    return voices


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    voice: str = "default",
    reference_audio: str = "",
    speed: float = 1.0,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> SparkTTSResult:
    """Synthesize speech from text using Spark-TTS.

    Args:
        text: Text to synthesize.
        voice: Built-in preset name or ignored when *reference_audio* is set.
        reference_audio: Path to a 3-10 second reference clip for voice cloning.
        speed: Playback speed multiplier (0.5 to 2.0).
        output_path: Where to write the WAV file. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        SparkTTSResult with the output path and metadata.

    Raises:
        RuntimeError: When sparktts is not installed or synthesis fails.
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty")

    speed = max(0.5, min(2.0, float(speed)))

    sparktts = _try_import("sparktts")
    if sparktts is None:
        raise RuntimeError(
            f"sparktts is not installed. {INSTALL_HINT}"
        )

    if on_progress:
        on_progress(10, "Loading Spark-TTS model...")

    notes: List[str] = []

    # Resolve output path
    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_sparktts_",
        )
        os.close(fd)

    try:
        # Import and initialize the model
        # Spark-TTS exposes a simple generate() API
        from sparktts import SparkTTS as _SparkTTS

        model = _SparkTTS()

        if on_progress:
            on_progress(30, "Synthesizing speech...")

        kwargs = {"text": text.strip(), "output_path": output_path}

        if reference_audio and os.path.isfile(reference_audio):
            kwargs["reference_audio"] = reference_audio
            notes.append(f"Voice cloned from: {os.path.basename(reference_audio)}")
        elif voice in SPARK_VOICE_PRESETS:
            kwargs["speaker"] = voice
            notes.append(f"Using preset voice: {voice}")
        else:
            kwargs["speaker"] = "default"
            notes.append("Using default voice (unknown preset requested)")

        if speed != 1.0:
            kwargs["speed"] = speed
            notes.append(f"Speed: {speed:.1f}x")

        model.generate(**kwargs)

        if on_progress:
            on_progress(90, "Finalizing...")

        # Get output file info
        duration = 0.0
        sample_rate = 0
        try:
            import wave
            with wave.open(output_path, "rb") as wf:
                sample_rate = wf.getframerate()
                duration = wf.getnframes() / sample_rate if sample_rate > 0 else 0.0
        except Exception:
            notes.append("Could not read WAV metadata")

        if on_progress:
            on_progress(100, "Done")

        return SparkTTSResult(
            output=output_path,
            voice=voice if not reference_audio else "cloned",
            model="spark-tts",
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"sparktts import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        # Clean up partial output on failure
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Spark-TTS synthesis failed: {exc}") from exc
