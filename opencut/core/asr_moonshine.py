"""
OpenCut Moonshine ASR Backend

10x faster-than-realtime CPU speech recognition. Whisper-compatible API
with streaming support. English models are MIT-licensed; multilingual
models use a community (non-commercial) licence and are gated separately.

Licence: MIT (English models)
Repository: https://github.com/usefulsensors/moonshine
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install moonshine  # MIT, CPU-optimized STT"

MOONSHINE_MODELS = {
    "moonshine-tiny": "Tiny — fastest, lowest accuracy (~39M params)",
    "moonshine-base": "Base — balanced speed/accuracy (~60M params)",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MoonshineResult:
    text: str = ""
    segments: List[dict] = field(default_factory=list)
    language: str = "en"
    model: str = "moonshine-base"
    duration_seconds: float = 0.0
    processing_seconds: float = 0.0
    realtime_factor: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("text", "segments", "language", "model",
                "duration_seconds", "processing_seconds",
                "realtime_factor", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_moonshine_available() -> bool:
    """Return True when the moonshine package is importable."""
    return _try_import("moonshine") is not None


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    model: str = "moonshine-base",
    on_progress: Optional[Callable] = None,
) -> MoonshineResult:
    """Transcribe audio using Moonshine ASR (CPU-optimized).

    Args:
        audio_path: Path to audio/video file.
        model: Model name — moonshine-tiny or moonshine-base.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        MoonshineResult with text, segments, and timing info.

    Raises:
        RuntimeError: When moonshine is not installed or transcription fails.
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")

    if model not in MOONSHINE_MODELS:
        model = "moonshine-base"

    moonshine_mod = _try_import("moonshine")
    if moonshine_mod is None:
        raise RuntimeError(f"moonshine is not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(10, f"Loading Moonshine model ({model})...")

    import time
    notes: List[str] = []

    try:
        # Moonshine exposes a simple transcribe() function
        from moonshine import transcribe as _moonshine_transcribe

        if on_progress:
            on_progress(30, "Transcribing audio...")

        start_time = time.monotonic()
        result = _moonshine_transcribe(audio_path, model=model)
        elapsed = time.monotonic() - start_time

        if on_progress:
            on_progress(90, "Processing results...")

        # Parse result — Moonshine returns text or a dict with segments
        text = ""
        segments = []

        if isinstance(result, str):
            text = result.strip()
        elif isinstance(result, dict):
            text = result.get("text", "").strip()
            raw_segments = result.get("segments", [])
            for seg in raw_segments:
                if isinstance(seg, dict):
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "text": str(seg.get("text", "")).strip(),
                    })
        elif isinstance(result, (list, tuple)):
            # Some Moonshine versions return a list of segment dicts
            for seg in result:
                if isinstance(seg, dict):
                    seg_text = str(seg.get("text", "")).strip()
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "text": seg_text,
                    })
            text = " ".join(s["text"] for s in segments if s.get("text"))

        # Estimate audio duration from file if not in result
        duration = 0.0
        try:
            from opencut.helpers import _get_file_duration
            duration = _get_file_duration(audio_path)
        except Exception:
            pass

        rtf = elapsed / duration if duration > 0 else 0.0
        notes.append(f"Processed in {elapsed:.1f}s ({rtf:.2f}x realtime)")

        if on_progress:
            on_progress(100, "Done")

        return MoonshineResult(
            text=text,
            segments=segments,
            language="en",
            model=model,
            duration_seconds=round(duration, 2),
            processing_seconds=round(elapsed, 2),
            realtime_factor=round(rtf, 3),
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"moonshine import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Moonshine transcription failed: {exc}") from exc
