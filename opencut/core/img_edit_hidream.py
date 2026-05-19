"""
OpenCut HiDream-E1 Instruction Image Editing (P2.2)

Natural language instruction-based image editing companion to HiDream-I1.
Style transfer, object add/remove, color change, attribute modification.

Licence: MIT
Repository: https://huggingface.co/HiDream-ai/HiDream-E1
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
INSTALL_HINT = "pip install diffusers>=0.32  # HiDream-E1 weights ~17 GB"


@dataclass
class HiDreamEditResult:
    output: str = ""
    instruction: str = ""
    width: int = 0
    height: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "instruction", "width", "height",
                "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_hidream_edit_available() -> bool:
    return (_try_import("diffusers") is not None
            and _try_import("torch") is not None)


def edit(image_path: str, instruction: str, seed: int = -1,
         output_path: str = "", on_progress=None) -> HiDreamEditResult:
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not instruction:
        raise ValueError("Instruction required")
    if not check_hidream_edit_available():
        raise RuntimeError(f"HiDream-E1 deps not installed. {INSTALL_HINT}")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix="opencut_hidream_edit_")
        os.close(fd)

    try:
        import torch
        from diffusers import HiDreamImageEditPipeline
        from PIL import Image

        if on_progress:
            on_progress(10, "Loading HiDream-E1...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = HiDreamImageEditPipeline.from_pretrained(
            "HiDream-ai/HiDream-E1", torch_dtype=torch.bfloat16
        ).to(device)

        source = Image.open(image_path).convert("RGB")
        if on_progress:
            on_progress(30, "Editing image...")

        start = time.monotonic()
        gen_kwargs = {"image": source, "prompt": instruction.strip()}
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        result_img = pipe(**gen_kwargs).images[0]
        elapsed = time.monotonic() - start
        result_img.save(output_path)

        notes.append(f"Instruction: {instruction[:80]}")
        notes.append(f"Edited in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return HiDreamEditResult(output=output_path, instruction=instruction[:200],
                                 width=result_img.width, height=result_img.height,
                                 generation_seconds=round(elapsed, 2), notes=notes)
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"HiDream-E1 edit failed: {exc}") from exc

__all__ = ["HiDreamEditResult", "check_hidream_edit_available", "INSTALL_HINT", "edit"]
