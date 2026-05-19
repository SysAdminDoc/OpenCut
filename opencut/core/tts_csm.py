"""
OpenCut CSM-1B Conversational Speech (N3.4)

1B-parameter speech generation model producing contextually-aware
conversation audio. Unlike Chatterbox (single utterance) and Kokoro
(text-only), CSM accepts conversation context and generates the next
utterance with consistent speaker identity.

Licence: Apache-2.0 (code + CSM weights);
         Meta Llama Community License (Llama-3.2-1B backbone, gated)
Repository: https://huggingface.co/sesame/csm-1b
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
    "pip install transformers>=4.52.1  # CSM-1B via Transformers\n"
    "Requires accepting Meta Llama Community License for Llama-3.2-1B backbone."
)


@dataclass
class CSMResult:
    output: str = ""
    speaker_id: int = 0
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    context_turns: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "speaker_id", "duration_seconds", "sample_rate",
                "context_turns", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_csm_available() -> bool:
    """Return True when transformers >= 4.52.1 with CSM support."""
    transformers = _try_import("transformers")
    if transformers is None:
        return False
    try:
        from transformers import CsmForConditionalGeneration  # noqa: F401
        return True
    except ImportError:
        return False


def _check_llama_ack() -> bool:
    """Check if user has acknowledged the Meta Llama license."""
    import json
    config_path = os.path.join(os.path.expanduser("~"), ".opencut", "config.json")
    try:
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return bool(config.get("csm_llama_ack"))
    except Exception:
        pass
    return False


def generate(
    text: str,
    context: Optional[List[Dict]] = None,
    speaker_id: int = 0,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> CSMResult:
    """Generate conversational speech using CSM-1B.

    Args:
        text: Text for the next utterance.
        context: Prior conversation context — list of
            {"speaker": int, "text": str, "audio_path": str} dicts.
        speaker_id: Speaker identity (0 or 1).
        output_path: Output WAV path. Auto-generated if empty.
        on_progress: Optional callback.

    Returns:
        CSMResult with output audio path.
    """
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    if not check_csm_available():
        raise RuntimeError(f"CSM-1B not available. {INSTALL_HINT}")
    if not _check_llama_ack():
        raise RuntimeError(
            "CSM-1B requires accepting the Meta Llama Community License. "
            "Set csm_llama_ack=true in ~/.opencut/config.json."
        )

    context = context or []
    speaker_id = max(0, min(1, int(speaker_id)))

    if on_progress:
        on_progress(10, "Loading CSM-1B model...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_csm_")
        os.close(fd)

    try:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CsmForConditionalGeneration.from_pretrained(
            "sesame/csm-1b", torch_dtype=torch.float16
        ).to(device)
        processor = AutoProcessor.from_pretrained("sesame/csm-1b")

        if on_progress:
            on_progress(40, "Generating conversational speech...")

        start = time.monotonic()

        # Build conversation input
        conversation = []
        for ctx in context[-5:]:  # Limit context window
            conversation.append({
                "role": f"speaker_{ctx.get('speaker', 0)}",
                "content": [{"type": "text", "text": ctx.get("text", "")}],
            })
        conversation.append({
            "role": f"speaker_{speaker_id}",
            "content": [{"type": "text", "text": text.strip()}],
        })

        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True
        ).to(device)

        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True)

        # Save audio
        import torchaudio
        audio_tensor = audio.squeeze()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        torchaudio.save(output_path, audio_tensor.cpu(), model.config.sampling_rate)

        elapsed = time.monotonic() - start
        duration = audio_tensor.shape[-1] / model.config.sampling_rate
        sr = model.config.sampling_rate

        notes.append(f"Speaker: {speaker_id}")
        notes.append(f"Context: {len(context)} prior turns")
        notes.append(f"Generated in {elapsed:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return CSMResult(
            output=output_path, speaker_id=speaker_id,
            duration_seconds=round(duration, 2), sample_rate=sr,
            context_turns=len(context), generation_seconds=round(elapsed, 2),
            notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"CSM import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"CSM generation failed: {exc}") from exc


__all__ = ["CSMResult", "check_csm_available", "INSTALL_HINT", "generate"]
