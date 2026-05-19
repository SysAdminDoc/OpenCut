"""
OpenCut Kokoro TTS Backend (M1.2)

82M parameter ultralight TTS. CPU-only, no CUDA required.
9 languages: US/UK English, Spanish, French, Hindi, Italian,
Japanese, Portuguese, Mandarin. 24 kHz output.

Positioned as the last-resort TTS fallback for machines without GPU.

Licence: Apache-2.0
Repository: https://github.com/hexgrad/kokoro
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

INSTALL_HINT = (
    "pip install kokoro  # Apache-2.0, 82M params, CPU-only TTS\n"
    "Also requires espeak-ng system package for G2P fallback:\n"
    "  Windows: choco install espeak-ng  OR  download from github.com/espeak-ng/espeak-ng/releases\n"
    "  macOS:   brew install espeak-ng\n"
    "  Linux:   apt install espeak-ng"
)

# Built-in voice presets per language
KOKORO_VOICES = {
    "en-us": {
        "af_heart": "Heart — warm American female",
        "af_alloy": "Alloy — neutral American female",
        "am_adam": "Adam — natural American male",
        "am_michael": "Michael — clear American male",
    },
    "en-gb": {
        "bf_emma": "Emma — British female",
        "bm_george": "George — British male",
    },
    "es": {"ef_dora": "Dora — Spanish female"},
    "fr": {"ff_siwis": "Siwis — French female"},
    "hi": {"hf_alpha": "Alpha — Hindi female"},
    "it": {"if_sara": "Sara — Italian female"},
    "ja": {"jf_alpha": "Alpha — Japanese female"},
    "pt-br": {"pf_dora": "Dora — Portuguese female"},
    "zh": {"zf_xiaobei": "Xiaobei — Mandarin female"},
}

KOKORO_LANGUAGES = {
    "en-us": "English (US)",
    "en-gb": "English (UK)",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "pt-br": "Portuguese (Brazil)",
    "zh": "Mandarin Chinese",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class KokoroResult:
    output: str = ""
    voice: str = ""
    language: str = ""
    model: str = "kokoro"
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "language", "model",
                "duration_seconds", "sample_rate", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_kokoro_available() -> bool:
    """Return True when the kokoro package is importable."""
    return _try_import("kokoro") is not None


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------

def list_voices(language: str = "") -> List[dict]:
    """Return available Kokoro voices, optionally filtered by language."""
    voices = []
    for lang, lang_voices in KOKORO_VOICES.items():
        if language and lang != language:
            continue
        for vid, desc in lang_voices.items():
            voices.append({
                "voice_id": vid,
                "name": vid.split("_", 1)[-1].title() if "_" in vid else vid,
                "description": desc,
                "language": lang,
                "language_name": KOKORO_LANGUAGES.get(lang, lang),
            })
    return voices


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    voice: str = "af_heart",
    language: str = "en-us",
    speed: float = 1.0,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> KokoroResult:
    """Synthesize speech using Kokoro TTS (82M, CPU-only).

    Args:
        text: Text to synthesize.
        voice: Voice preset ID (e.g., af_heart, am_adam).
        language: Language code (en-us, en-gb, es, fr, hi, it, ja, pt-br, zh).
        speed: Speed multiplier (0.5 to 2.0).
        output_path: Where to write WAV. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        KokoroResult with output path and metadata.

    Raises:
        RuntimeError: When kokoro is not installed or synthesis fails.
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty")

    speed = max(0.5, min(2.0, float(speed)))

    kokoro_mod = _try_import("kokoro")
    if kokoro_mod is None:
        raise RuntimeError(f"kokoro is not installed. {INSTALL_HINT}")

    if language not in KOKORO_LANGUAGES:
        language = "en-us"

    if on_progress:
        on_progress(10, "Loading Kokoro model...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".wav", prefix="opencut_kokoro_",
        )
        os.close(fd)

    try:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=language)

        if on_progress:
            on_progress(30, "Synthesizing speech...")

        start_time = time.monotonic()

        # Kokoro's generate returns audio samples
        generator = pipeline(
            text.strip(),
            voice=voice,
            speed=speed,
        )

        # Collect all audio segments
        import soundfile as sf
        all_audio = []
        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)

        if not all_audio:
            raise RuntimeError("Kokoro produced no audio output")

        import numpy as np
        combined = np.concatenate(all_audio)

        # Write output
        sf.write(output_path, combined, 24000)

        gen_time = time.monotonic() - start_time
        duration = len(combined) / 24000.0

        notes.append(f"Voice: {voice}")
        notes.append(f"Language: {KOKORO_LANGUAGES.get(language, language)}")
        notes.append(f"Generated in {gen_time:.1f}s")
        if speed != 1.0:
            notes.append(f"Speed: {speed:.1f}x")

        if on_progress:
            on_progress(100, "Done")

        return KokoroResult(
            output=output_path,
            voice=voice,
            language=language,
            model="kokoro",
            duration_seconds=round(duration, 2),
            sample_rate=24000,
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"kokoro dependency missing: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Kokoro synthesis failed: {exc}") from exc


__all__ = [
    "KokoroResult",
    "check_kokoro_available",
    "INSTALL_HINT",
    "KOKORO_VOICES",
    "KOKORO_LANGUAGES",
    "list_voices",
    "synthesize",
]
