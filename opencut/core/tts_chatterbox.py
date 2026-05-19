"""
OpenCut Chatterbox TTS Backend (M1.1)

Chatterbox-Turbo (350M) — fastest emotional open-source TTS with native
paralinguistic tags [laugh], [chuckle], [cough]. Zero-shot voice cloning
from a 10-second clip. MIT licensed.

Chatterbox-Multilingual (500M) — 23 languages, benchmarks ahead of
ElevenLabs Turbo v2.5 in independent evaluation. Built-in Perth
perceptual watermarking.

Repository: https://github.com/resemble-ai/chatterbox
Licence: MIT
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install chatterbox-tts  # MIT, emotional TTS + voice cloning"

CHATTERBOX_MODELS = {
    "turbo": "Chatterbox-Turbo (350M) — fastest, English, emotion tags",
    "multilingual": "Chatterbox-Multilingual (500M) — 23 languages",
}

CHATTERBOX_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
    "ja", "ko", "zh", "ar", "hi", "tr", "sv", "da", "fi",
    "no", "el", "cs", "hu", "ro",
]

# Emotion tags supported natively by the model
EMOTION_TAGS = [
    "[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]",
    "[groan]", "[hmm]", "[uh]", "[um]",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ChatterboxResult:
    output: str = ""
    voice: str = ""
    model: str = "turbo"
    language: str = "en"
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    has_emotion_tags: bool = False
    voice_cloned: bool = False
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "model", "language",
                "duration_seconds", "sample_rate",
                "has_emotion_tags", "voice_cloned", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_chatterbox_available() -> bool:
    """Return True when chatterbox-tts is importable."""
    return _try_import("chatterbox") is not None


def check_chatterbox_multilingual_available() -> bool:
    """Return True when multilingual variant is available."""
    if not check_chatterbox_available():
        return False
    try:
        from chatterbox.tts import ChatterboxTTS
        return hasattr(ChatterboxTTS, "from_pretrained_multilingual")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    reference_audio: str = "",
    model: str = "turbo",
    language: str = "en",
    exaggeration: float = 0.5,
    speed: float = 1.0,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> ChatterboxResult:
    """Synthesize speech using Chatterbox TTS.

    Args:
        text: Text to synthesize. May include emotion tags like [laugh].
        reference_audio: Path to 10-second reference clip for voice cloning.
        model: "turbo" (English, fast) or "multilingual" (23 languages).
        language: Language code for multilingual model.
        exaggeration: Emotion/expressiveness intensity (0.0 to 1.0).
        speed: Playback speed (0.5 to 2.0).
        output_path: Output WAV path. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        ChatterboxResult with output path and metadata.
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty")

    chatterbox_mod = _try_import("chatterbox")
    if chatterbox_mod is None:
        raise RuntimeError(f"chatterbox-tts is not installed. {INSTALL_HINT}")

    if model not in CHATTERBOX_MODELS:
        model = "turbo"
    exaggeration = max(0.0, min(1.0, float(exaggeration)))
    speed = max(0.5, min(2.0, float(speed)))

    if on_progress:
        on_progress(10, f"Loading Chatterbox ({model})...")

    notes: List[str] = []
    has_emotions = any(tag in text for tag in EMOTION_TAGS)
    voice_cloned = bool(reference_audio and os.path.isfile(reference_audio))

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_chatterbox_",
        )
        os.close(fd)

    try:
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model == "multilingual" and hasattr(ChatterboxTTS, "from_pretrained_multilingual"):
            tts = ChatterboxTTS.from_pretrained_multilingual(device=device)
            notes.append(f"Language: {language}")
        else:
            tts = ChatterboxTTS.from_pretrained(device=device)
            model = "turbo"

        if on_progress:
            on_progress(40, "Generating speech...")

        start_time = time.monotonic()

        kwargs = {"text": text.strip()}
        if voice_cloned:
            kwargs["audio_prompt"] = reference_audio
            notes.append(f"Voice cloned from: {os.path.basename(reference_audio)}")
        if exaggeration != 0.5:
            kwargs["exaggeration"] = exaggeration
            notes.append(f"Exaggeration: {exaggeration:.1f}")

        wav = tts.generate(**kwargs)

        # Save output
        import torchaudio
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(output_path, wav.cpu(), tts.sr)

        gen_time = time.monotonic() - start_time
        duration = wav.shape[-1] / tts.sr if tts.sr > 0 else 0.0
        sample_rate = tts.sr

        notes.append(f"Model: {model}")
        notes.append(f"Generated in {gen_time:.1f}s")
        if has_emotions:
            notes.append("Contains emotion tags")

        del tts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return ChatterboxResult(
            output=output_path,
            voice="cloned" if voice_cloned else "default",
            model=model,
            language=language if model == "multilingual" else "en",
            duration_seconds=round(duration, 2),
            sample_rate=sample_rate,
            has_emotion_tags=has_emotions,
            voice_cloned=voice_cloned,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"chatterbox import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Chatterbox synthesis failed: {exc}") from exc


__all__ = [
    "ChatterboxResult",
    "check_chatterbox_available",
    "check_chatterbox_multilingual_available",
    "INSTALL_HINT",
    "CHATTERBOX_MODELS",
    "CHATTERBOX_LANGUAGES",
    "EMOTION_TAGS",
    "synthesize",
]
