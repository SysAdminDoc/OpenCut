"""
OpenCut HiDream-I1 SOTA Text-to-Image (P2.1)

17B Sparse DiT — highest DPG-Bench / GenEval / HPSv2.1 scores of any
open model. Three variants: Full (50 steps), Dev (28), Fast (16).
Requires Meta Llama 3.1-8B backbone (gated, user opt-in).

Licence: MIT (code + HiDream weights); Meta Community Licence (Llama backbone)
Repository: https://huggingface.co/HiDream-ai/HiDream-I1-Full
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
INSTALL_HINT = "pip install diffusers>=0.32  # HiDream-I1 + Llama-3.1-8B weights ~33 GB total"

HIDREAM_VARIANTS = {
    "fast": "Fast (16 steps) — quickest, default",
    "dev": "Dev (28 steps) — balanced",
    "full": "Full (50 steps) — highest quality",
}


@dataclass
class HiDreamResult:
    output: str = ""
    width: int = 0
    height: int = 0
    variant: str = "fast"
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "width", "height", "variant",
                "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_hidream_available() -> bool:
    return (_try_import("diffusers") is not None
            and _try_import("torch") is not None
            and _try_import("transformers") is not None)


def generate(prompt: str, variant: str = "fast", width: int = 1024, height: int = 1024,
             seed: int = -1, output_path: str = "", on_progress=None) -> HiDreamResult:
    if not prompt:
        raise ValueError("Prompt required")
    if not check_hidream_available():
        raise RuntimeError(f"HiDream deps not installed. {INSTALL_HINT}")
    if variant not in HIDREAM_VARIANTS:
        variant = "fast"

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix="opencut_hidream_")
        os.close(fd)

    try:
        import torch
        from diffusers import HiDreamImagePipeline

        if on_progress:
            on_progress(10, f"Loading HiDream-I1 ({variant})...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = f"HiDream-ai/HiDream-I1-{variant.capitalize()}"
        pipe = HiDreamImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

        steps_map = {"fast": 16, "dev": 28, "full": 50}
        if on_progress:
            on_progress(30, f"Generating image ({steps_map[variant]} steps)...")

        start = time.monotonic()
        gen_kwargs = {"prompt": prompt.strip(), "height": height, "width": width,
                      "num_inference_steps": steps_map[variant]}
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        image = pipe(**gen_kwargs).images[0]
        elapsed = time.monotonic() - start
        image.save(output_path)

        notes.append(f"Variant: {variant} ({steps_map[variant]} steps)")
        notes.append(f"Generated in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return HiDreamResult(output=output_path, width=image.width, height=image.height,
                             variant=variant, generation_seconds=round(elapsed, 2), notes=notes)
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"HiDream generation failed: {exc}") from exc

__all__ = ["HiDreamResult", "check_hidream_available", "INSTALL_HINT",
           "HIDREAM_VARIANTS", "generate"]
