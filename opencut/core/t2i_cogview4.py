"""
OpenCut CogView4-6B Bilingual T2I (P2.3)

Native English + Chinese text-to-image. 13 GB VRAM with int4 text encoder.
Competitive with FLUX.1-dev on DPG-Bench. No gated deps.

Licence: Apache-2.0
Repository: https://huggingface.co/THUDM/CogView4-6B
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
INSTALL_HINT = "pip install diffusers>=0.32  # CogView4 weights ~12 GB auto-download"


@dataclass
class CogView4Result:
    output: str = ""
    width: int = 0
    height: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "width", "height", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_cogview4_available() -> bool:
    return _try_import("diffusers") is not None and _try_import("torch") is not None


def generate(prompt: str, width: int = 1024, height: int = 1024,
             guidance_scale: float = 3.5, seed: int = -1,
             output_path: str = "", on_progress=None) -> CogView4Result:
    if not prompt:
        raise ValueError("Prompt required")
    if not check_cogview4_available():
        raise RuntimeError(f"CogView4 deps not installed. {INSTALL_HINT}")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".png", prefix="opencut_cogview4_")
        os.close(fd)

    try:
        import torch
        from diffusers import CogView4Pipeline

        if on_progress:
            on_progress(10, "Loading CogView4-6B...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = CogView4Pipeline.from_pretrained(
            "THUDM/CogView4-6B", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        if on_progress:
            on_progress(30, "Generating image...")

        start = time.monotonic()
        gen_kwargs = {"prompt": prompt.strip(), "height": height, "width": width,
                      "guidance_scale": guidance_scale, "num_inference_steps": 50}
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        image = pipe(**gen_kwargs).images[0]
        elapsed = time.monotonic() - start
        image.save(output_path)

        notes.append(f"Generated in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return CogView4Result(output=output_path, width=image.width, height=image.height,
                              generation_seconds=round(elapsed, 2), notes=notes)
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"CogView4 failed: {exc}") from exc

__all__ = ["CogView4Result", "check_cogview4_available", "INSTALL_HINT", "generate"]
