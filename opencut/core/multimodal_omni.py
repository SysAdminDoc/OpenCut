"""
OpenCut Qwen2.5-Omni Multimodal Video Narrator (P3.1)

Accepts video + audio, generates written commentary + spoken narration.
First local model that watches AND narrates video.

Licence: Apache-2.0
Repository: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")
INSTALL_HINT = "pip install transformers  # Qwen2.5-Omni-7B weights ~14 GB auto-download"

NARRATION_STYLES = ["documentary", "sports_commentary", "educational",
                    "news_report", "casual", "dramatic"]


@dataclass
class OmniResult:
    text_response: str = ""
    audio_path: str = ""
    narration_style: str = ""
    duration_seconds: float = 0.0
    processing_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("text_response", "audio_path", "narration_style",
                "duration_seconds", "processing_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_omni_available() -> bool:
    return _try_import("transformers") is not None and _try_import("torch") is not None


def narrate_video(video_path: str, style: str = "documentary",
                  output_audio: str = "", on_progress=None) -> OmniResult:
    """Watch a video and generate written + spoken narration."""
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")
    if not check_omni_available():
        raise RuntimeError(f"Qwen2.5-Omni not installed. {INSTALL_HINT}")
    if style not in NARRATION_STYLES:
        style = "documentary"

    notes: List[str] = []
    if not output_audio:
        fd, output_audio = tempfile.mkstemp(suffix=".wav", prefix="opencut_omni_")
        os.close(fd)

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

        if on_progress:
            on_progress(10, "Loading Qwen2.5-Omni...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16
        ).to(device)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        prompt = (f"Watch this video and provide a {style} style narration. "
                  "Describe what happens scene by scene, including key visual "
                  "elements, actions, and mood.")

        if on_progress:
            on_progress(30, "Analyzing video and generating narration...")

        start = time.monotonic()

        # Build multimodal input
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt},
        ]}]
        text_input = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = processor(text=[text_input], videos=[video_path],
                           return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2048,
                                     output_audio=True)

        text_response = processor.batch_decode(
            outputs.sequences[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Save audio output if model generated it
        if hasattr(outputs, "audio") and outputs.audio is not None:
            import torchaudio
            audio = outputs.audio.squeeze()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(output_audio, audio.cpu(), model.config.audio_sampling_rate)
            notes.append("Audio narration generated")
        else:
            notes.append("Text narration only (no audio output)")

        elapsed = time.monotonic() - start
        notes.append(f"Style: {style}")
        notes.append(f"Processed in {elapsed:.1f}s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return OmniResult(text_response=text_response.strip(), audio_path=output_audio,
                          narration_style=style, processing_seconds=round(elapsed, 2),
                          notes=notes)
    except ImportError as exc:
        raise RuntimeError(f"Omni import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        raise RuntimeError(f"Omni narration failed: {exc}") from exc


def video_qa(video_path: str, question: str, on_progress=None) -> OmniResult:
    """Ask a question about a video and get a text+audio answer."""
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")
    if not question:
        raise ValueError("Question required")
    if not check_omni_available():
        raise RuntimeError(f"Qwen2.5-Omni not installed. {INSTALL_HINT}")

    notes: List[str] = [f"Q: {question[:100]}"]
    fd, output_audio = tempfile.mkstemp(suffix=".wav", prefix="opencut_omni_qa_")
    os.close(fd)

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

        if on_progress:
            on_progress(15, "Loading Qwen2.5-Omni for QA...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", torch_dtype=torch.bfloat16
        ).to(device)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]}]
        text_input = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = processor(text=[text_input], videos=[video_path],
                           return_tensors="pt").to(device)

        if on_progress:
            on_progress(50, "Analyzing...")

        start = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024)

        answer = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        elapsed = time.monotonic() - start

        notes.append(f"Answered in {elapsed:.1f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return OmniResult(text_response=answer.strip(), audio_path="",
                          processing_seconds=round(elapsed, 2), notes=notes)
    except Exception as exc:
        raise RuntimeError(f"Omni QA failed: {exc}") from exc


__all__ = ["OmniResult", "check_omni_available", "INSTALL_HINT",
           "NARRATION_STYLES", "narrate_video", "video_qa"]
