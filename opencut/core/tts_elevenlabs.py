"""
OpenCut ElevenLabs TTS Backend v1.29.0

Cloud TTS via ElevenLabs API — 3,000+ voices, 32 languages.
Requires OPENCUT_ELEVENLABS_API_KEY environment variable.

Licence: ElevenLabs Developer Agreement (cloud API).
Python SDK: https://github.com/elevenlabs/elevenlabs-python (Apache 2.0)
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install elevenlabs  # then set OPENCUT_ELEVENLABS_API_KEY"

# Supported ElevenLabs models as of 2025-06
ELEVENLABS_MODELS = {
    "eleven_multilingual_v2": "Multilingual v2 — best quality, 29 languages",
    "eleven_turbo_v2_5": "Turbo v2.5 — low latency, 32 languages",
    "eleven_turbo_v2": "Turbo v2 — low latency, English optimised",
    "eleven_monolingual_v1": "Monolingual v1 — English only, legacy",
    "eleven_multilingual_v1": "Multilingual v1 — legacy",
}

DEFAULT_MODEL = "eleven_multilingual_v2"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ElevenLabsResult:
    output: str = ""
    voice_id: str = ""
    voice_name: str = ""
    model: str = ""
    characters: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice_id", "voice_name", "model", "characters", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_elevenlabs_available() -> bool:
    """True only if both the SDK *and* an API key are present."""
    if _try_import("elevenlabs") is None:
        return False
    return bool(os.environ.get("OPENCUT_ELEVENLABS_API_KEY", "").strip())


def _get_api_key() -> str:
    key = os.environ.get("OPENCUT_ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENCUT_ELEVENLABS_API_KEY environment variable is not set. "
            "Get a free key at https://elevenlabs.io"
        )
    return key


def _get_client():
    from elevenlabs import ElevenLabs
    return ElevenLabs(api_key=_get_api_key())


# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

def list_voices() -> List[dict]:
    """Return all available voices from ElevenLabs (API call)."""
    if not check_elevenlabs_available():
        return []
    try:
        client = _get_client()
        voices_response = client.voices.get_all()
        voices = []
        for v in voices_response.voices:
            voices.append({
                "voice_id": v.voice_id,
                "name": v.name,
                "category": getattr(v, "category", "premade"),
                "description": getattr(v, "description", "") or "",
                "labels": dict(v.labels) if v.labels else {},
            })
        return sorted(voices, key=lambda x: x["name"])
    except Exception as exc:
        logger.warning("ElevenLabs voice list failed: %s", exc)
        return []


def list_models() -> List[dict]:
    """Return available synthesis models."""
    return [
        {"model_id": k, "description": v}
        for k, v in ELEVENLABS_MODELS.items()
    ]


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    voice: str = "Rachel",
    model: str = DEFAULT_MODEL,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> ElevenLabsResult:
    """
    Synthesize speech via ElevenLabs.

    Args:
        text:               Input text (max 5 000 chars per call).
        voice:              Voice name *or* voice_id (matched case-insensitively).
        model:              ElevenLabs model ID.  Defaults to multilingual v2.
        stability:          Voice stability 0–1 (higher = less expressive).
        similarity_boost:   Voice clarity 0–1 (higher = closer to source voice).
        style:              Style exaggeration 0–1 (multilingual v2+ only).
        use_speaker_boost:  Speaker boost (recommended True).
        output:             Output wav/mp3 path.  Auto-generated if None.
        on_progress:        Progress callback ``(percent, message)``.
    """
    if not text or not text.strip():
        raise ValueError("text is required")
    if len(text) > 5000:
        raise ValueError("text must be ≤5000 characters (ElevenLabs limit per call)")

    if model not in ELEVENLABS_MODELS:
        model = DEFAULT_MODEL

    from elevenlabs import VoiceSettings

    if on_progress:
        on_progress(5, "Connecting to ElevenLabs API...")

    client = _get_client()

    # Resolve voice: treat `voice` as name first, then as voice_id
    voice_id = voice
    voice_name = voice
    all_voices = list_voices()
    for v in all_voices:
        if v["name"].lower() == voice.lower() or v["voice_id"] == voice:
            voice_id = v["voice_id"]
            voice_name = v["name"]
            break

    if on_progress:
        on_progress(20, f"Synthesizing {len(text)} chars with voice '{voice_name}'...")

    settings = VoiceSettings(
        stability=float(max(0.0, min(1.0, stability))),
        similarity_boost=float(max(0.0, min(1.0, similarity_boost))),
        style=float(max(0.0, min(1.0, style))),
        use_speaker_boost=bool(use_speaker_boost),
    )

    audio_stream: Iterator[bytes] = client.generate(
        text=text,
        voice=voice_id,
        model=model,
        voice_settings=settings,
        stream=True,
    )

    if output is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in voice_name[:20])
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            prefix=f"opencut_elevenlabs_{safe_name}_",
            dir=os.path.join(os.path.expanduser("~"), ".opencut"),
            delete=False,
        )
        output = tmp.name
        tmp.close()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    if on_progress:
        on_progress(40, "Downloading audio stream...")

    total_bytes = 0
    with open(output, "wb") as f:
        for chunk in audio_stream:
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    if on_progress:
        on_progress(95, f"Wrote {total_bytes // 1024} KB")

    logger.info(
        "ElevenLabs synthesis: voice=%s model=%s chars=%d output=%s",
        voice_name, model, len(text), output,
    )

    if on_progress:
        on_progress(100, "ElevenLabs TTS complete")

    return ElevenLabsResult(
        output=output,
        voice_id=voice_id,
        voice_name=voice_name,
        model=model,
        characters=len(text),
        notes=[f"Synthesized {len(text)} chars via {model}"],
    )


__all__ = [
    "ElevenLabsResult",
    "ELEVENLABS_MODELS",
    "DEFAULT_MODEL",
    "INSTALL_HINT",
    "check_elevenlabs_available",
    "list_voices",
    "list_models",
    "synthesize",
]
