"""
OpenCut Dia 1.6B Dialogue TTS (O1.1)

Multi-speaker scripted dialogue with nonverbal sounds.
Accepts [S1]/[S2] speaker tags + (laughs), (coughs), (sighs).
Voice cloning from 5-10s reference. 4.4 GB VRAM on bfloat16.

Licence: Apache-2.0
Repository: https://github.com/nari-labs/dia
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

INSTALL_HINT = "pip install dia  # Apache-2.0, 1.6B dialogue TTS"

NONVERBALS = ["(laughs)", "(coughs)", "(sighs)", "(applause)", "(gasps)",
              "(clears throat)", "(sniffs)", "(chuckles)"]


@dataclass
class DiaResult:
    output: str = ""
    speakers: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 44100
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "speakers", "duration_seconds", "sample_rate",
                "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_dia_available() -> bool:
    return _try_import("dia") is not None


def generate_dialogue(
    turns: List[Dict],
    reference_audio: str = "",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> DiaResult:
    """Generate multi-speaker dialogue audio.

    Args:
        turns: List of {speaker: "S1"|"S2", text: str, nonverbals: [str]}.
        reference_audio: Optional voice clone reference (5-10s).
        output_path: Output WAV path.
        on_progress: Callback.
    """
    if not turns:
        raise ValueError("At least one dialogue turn required")
    if not check_dia_available():
        raise RuntimeError(f"Dia not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(10, "Loading Dia model...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_dia_")
        os.close(fd)

    try:
        import torch
        from dia.model import Dia

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", device=device)

        # Build transcript with speaker tags
        transcript = ""
        speakers_seen = set()
        for turn in turns:
            speaker = turn.get("speaker", "S1")
            text = turn.get("text", "")
            nonverbals = turn.get("nonverbals", [])
            speakers_seen.add(speaker)
            parts = []
            for nv in nonverbals:
                if nv in NONVERBALS:
                    parts.append(nv)
            parts.append(text)
            transcript += f"[{speaker}] {' '.join(parts)}\n"

        notes.append(f"Speakers: {len(speakers_seen)}")
        notes.append(f"Turns: {len(turns)}")

        if on_progress:
            on_progress(40, "Generating dialogue...")

        start = time.monotonic()
        kwargs = {"transcript": transcript.strip()}
        if reference_audio and os.path.isfile(reference_audio):
            kwargs["audio_prompt"] = reference_audio
            notes.append("Voice cloned from reference")

        output = model.generate(**kwargs)
        elapsed = time.monotonic() - start

        # Save
        import soundfile as sf
        sf.write(output_path, output, model.sample_rate)
        duration = len(output) / model.sample_rate
        sr = model.sample_rate

        notes.append(f"Generated in {elapsed:.1f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return DiaResult(
            output=output_path, speakers=len(speakers_seen),
            duration_seconds=round(duration, 2), sample_rate=sr,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"Dia import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Dia generation failed: {exc}") from exc


__all__ = ["DiaResult", "check_dia_available", "INSTALL_HINT", "NONVERBALS",
           "generate_dialogue"]
