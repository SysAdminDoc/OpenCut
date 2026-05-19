"""
OpenCut YuE Lyrics-to-Full-Song (O3.1)

Open-source lyrics2song: genre tags + structured lyrics ([verse]/[chorus])
-> complete song with vocal track + backing accompaniment. Multilingual:
English, Mandarin, Cantonese, Japanese, Korean. ICL style transfer.

S1: YuE-s1-7B-anneal-en-cot (7B LM), S2: YuE-s2-1B-general (1B decoder).
24 GB VRAM for 2 sessions.

Licence: Apache-2.0
Repository: https://github.com/multimodal-art-projection/YuE
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "pip install yue-inference  # Apache-2.0\n"
    "YuE weights (~28 GB total) auto-download from HuggingFace."
)

YUE_GENRES = [
    "pop", "rock", "hip-hop", "electronic", "r&b", "folk",
    "country", "jazz", "classical", "metal", "indie",
    "cinematic", "lo-fi", "blues", "reggae",
]

YUE_LANGUAGES = ["en", "zh", "yue", "ja", "ko"]


@dataclass
class YuEResult:
    output: str = ""
    vocal_path: str = ""
    backing_path: str = ""
    duration_seconds: float = 0.0
    genre: str = ""
    language: str = "en"
    has_vocals: bool = True
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "vocal_path", "backing_path", "duration_seconds",
                "genre", "language", "has_vocals", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_yue_available() -> bool:
    return _try_import("yue") is not None or _try_import("yue_inference") is not None


def generate(
    lyrics: str,
    genre: str = "pop",
    language: str = "en",
    instrumental_only: bool = False,
    reference_vocal: str = "",
    reference_backing: str = "",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> YuEResult:
    """Generate a complete song from structured lyrics.

    Args:
        lyrics: Structured lyrics with [verse], [chorus], [bridge] tags.
        genre: Genre tag for style guidance.
        language: Language code (en/zh/yue/ja/ko).
        instrumental_only: Generate backing track only (no vocals).
        reference_vocal: Reference vocal for ICL style transfer.
        reference_backing: Reference instrumental for ICL style transfer.
        output_path: Output WAV path.
        on_progress: Callback.
    """
    if not lyrics or not lyrics.strip():
        raise ValueError("Lyrics are required")
    if not check_yue_available():
        raise RuntimeError(f"YuE not installed. {INSTALL_HINT}")

    if genre not in YUE_GENRES:
        genre = "pop"
    if language not in YUE_LANGUAGES:
        language = "en"

    if on_progress:
        on_progress(5, "Loading YuE models (s1 + s2)...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_yue_")
        os.close(fd)

    try:
        from yue_inference import YuEPipeline

        pipe = YuEPipeline()

        genre_tag = f"[{genre}]"
        notes.append(f"Genre: {genre}")
        notes.append(f"Language: {language}")

        if on_progress:
            on_progress(20, "Generating song (this may take several minutes)...")

        start = time.monotonic()

        kwargs: Dict[str, Any] = {
            "lyrics": lyrics.strip(),
            "genre_tags": genre_tag,
            "language": language,
            "output_path": output_path,
        }
        if instrumental_only:
            kwargs["instrumental_only"] = True
            notes.append("Instrumental only (no vocals)")
        if reference_vocal and os.path.isfile(reference_vocal):
            kwargs["reference_vocal"] = reference_vocal
            notes.append("ICL style: vocal reference provided")
        if reference_backing and os.path.isfile(reference_backing):
            kwargs["reference_backing"] = reference_backing
            notes.append("ICL style: backing reference provided")

        pipe.generate(**kwargs)
        elapsed = time.monotonic() - start

        notes.append(f"Generated in {elapsed:.1f}s")

        # Check for separate stems
        vocal_path = ""
        backing_path = ""
        base = os.path.splitext(output_path)[0]
        if os.path.isfile(f"{base}_vocal.wav"):
            vocal_path = f"{base}_vocal.wav"
        if os.path.isfile(f"{base}_backing.wav"):
            backing_path = f"{base}_backing.wav"

        # Get duration
        duration = 0.0
        try:
            import wave
            with wave.open(output_path, "rb") as wf:
                sr = wf.getframerate()
                duration = wf.getnframes() / sr if sr > 0 else 0
        except Exception:
            pass

        if on_progress:
            on_progress(100, "Done")

        return YuEResult(
            output=output_path, vocal_path=vocal_path,
            backing_path=backing_path,
            duration_seconds=round(duration, 2),
            genre=genre, language=language,
            has_vocals=not instrumental_only,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"YuE import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"YuE generation failed: {exc}") from exc


__all__ = ["YuEResult", "check_yue_available", "INSTALL_HINT",
           "YUE_GENRES", "YUE_LANGUAGES", "generate"]
