"""
OpenCut Parler-TTS Natural Language Voice Description (O1.2)

TTS from free-text voice description: "A female speaker with animated
speech at moderate pace with clear audio." Mini (880M) and Large (2.3B).
34 named speaker presets.

Licence: Apache-2.0
Repository: https://huggingface.co/parler-tts
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

INSTALL_HINT = "pip install parler-tts  # Apache-2.0, voice description TTS"

PARLER_MODELS = {
    "mini": "Mini (880M, ~2 GB) — fast, good for previews",
    "large": "Large (2.3B, ~5 GB) — highest quality",
}

PARLER_SPEAKERS = [
    "Laura", "Gary", "Jon", "Lea", "Karen", "Rick", "Brenda", "David",
    "Eileen", "Jordan", "Mike", "Yolanda", "Patrick", "Ruby", "Thomas",
    "Alisa", "Jerry", "Tina", "Jenna", "Bill", "Will", "Barbara",
    "Eric", "Emily", "Anna", "Bruce", "Rose", "Daniel", "Jenny",
    "Naomi", "Talia", "Peter", "Semaj", "Nathan",
]


@dataclass
class ParlerResult:
    output: str = ""
    description: str = ""
    speaker: str = ""
    model: str = "mini"
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "description", "speaker", "model",
                "duration_seconds", "sample_rate", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_parler_available() -> bool:
    return _try_import("parler_tts") is not None


def list_speakers() -> List[dict]:
    return [{"name": s, "id": s.lower()} for s in PARLER_SPEAKERS]


def synthesize(
    text: str,
    description: str = "",
    speaker: str = "",
    model: str = "mini",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> ParlerResult:
    """Synthesize speech matching a natural language voice description.

    Args:
        text: Text to speak.
        description: Free-text voice description (e.g., "A warm male narrator").
        speaker: Named speaker preset (overrides description if set).
        model: mini or large.
        output_path: Output WAV path.
        on_progress: Callback.
    """
    if not text or not text.strip():
        raise ValueError("Text required")
    if not check_parler_available():
        raise RuntimeError(f"parler-tts not installed. {INSTALL_HINT}")
    if model not in PARLER_MODELS:
        model = "mini"

    if on_progress:
        on_progress(10, f"Loading Parler-TTS ({model})...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_parler_")
        os.close(fd)

    try:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"parler-tts/parler-tts-{model}-v1"
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Build description
        if speaker and speaker in PARLER_SPEAKERS:
            desc = f"{speaker} speaks clearly with natural intonation."
            notes.append(f"Speaker: {speaker}")
        elif description:
            desc = description
            notes.append(f"Description: {description[:80]}")
        else:
            desc = "A clear female voice with moderate pace and natural delivery."

        if on_progress:
            on_progress(40, "Synthesizing...")

        start = time.monotonic()

        input_ids = tokenizer(desc, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text.strip(), return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            audio = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)

        elapsed = time.monotonic() - start
        audio_np = audio.cpu().numpy().squeeze()

        import soundfile as sf
        sr = tts_model.config.sampling_rate
        sf.write(output_path, audio_np, sr)
        duration = len(audio_np) / sr

        notes.append(f"Model: {model}")
        notes.append(f"Generated in {elapsed:.1f}s")

        del tts_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return ParlerResult(
            output=output_path, description=desc[:200],
            speaker=speaker, model=model,
            duration_seconds=round(duration, 2), sample_rate=sr,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"parler_tts import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Parler-TTS failed: {exc}") from exc


__all__ = ["ParlerResult", "check_parler_available", "INSTALL_HINT",
           "PARLER_MODELS", "PARLER_SPEAKERS", "list_speakers", "synthesize"]
