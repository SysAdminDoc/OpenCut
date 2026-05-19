"""
OpenCut Allegro Lightweight T2V + TI2V (P1.2)

Lowest-VRAM T2V in the stack: 9.3 GB with CPU offload. 2.8B DiT.
6-second 720x1280 at 15 FPS. First-and-last-frame interpolation.

Licence: Apache-2.0
Repository: https://github.com/rhymes-ai/Allegro
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
INSTALL_HINT = "pip install diffusers>=0.32  # Allegro weights ~5 GB auto-download"

ALLEGRO_MODELS = {
    "Allegro-T2V": "T2V — text-to-video, 9.3 GB VRAM",
    "Allegro-TI2V": "TI2V — first+last frame interpolation",
}


@dataclass
class AllegroResult:
    output: str = ""
    mode: str = "t2v"
    duration_seconds: float = 6.0
    fps: float = 15.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "duration_seconds", "fps",
                "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_allegro_available() -> bool:
    return _try_import("diffusers") is not None and _try_import("torch") is not None


def generate_t2v(prompt: str, seed: int = -1, output_path: str = "",
                 on_progress=None) -> AllegroResult:
    if not prompt:
        raise ValueError("Prompt required")
    if not check_allegro_available():
        raise RuntimeError(f"Allegro deps not installed. {INSTALL_HINT}")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_allegro_")
        os.close(fd)

    try:
        import torch
        from diffusers import AllegroPipeline
        from diffusers.utils import export_to_video

        if on_progress:
            on_progress(10, "Loading Allegro T2V...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AllegroPipeline.from_pretrained(
            "rhymes-ai/Allegro", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        if on_progress:
            on_progress(30, "Generating video (9.3 GB VRAM)...")

        start = time.monotonic()
        gen_kwargs = {"prompt": prompt.strip(), "num_frames": 88}
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)
        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start
        export_to_video(video, output_path, fps=15)

        notes.append(f"Generated in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return AllegroResult(output=output_path, mode="t2v", duration_seconds=6.0,
                             fps=15.0, generation_seconds=round(elapsed, 2), notes=notes)
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Allegro T2V failed: {exc}") from exc


def generate_ti2v(first_frame: str, last_frame: str = "", prompt: str = "",
                  seed: int = -1, output_path: str = "", on_progress=None) -> AllegroResult:
    if not first_frame or not os.path.isfile(first_frame):
        raise ValueError(f"First frame not found: {first_frame}")
    if not check_allegro_available():
        raise RuntimeError(f"Allegro deps not installed. {INSTALL_HINT}")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_allegro_ti2v_")
        os.close(fd)

    try:
        import torch
        from diffusers import AllegroTI2VPipeline
        from diffusers.utils import export_to_video, load_image

        if on_progress:
            on_progress(10, "Loading Allegro TI2V...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AllegroTI2VPipeline.from_pretrained(
            "rhymes-ai/Allegro-TI2V", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        first = load_image(first_frame)
        gen_kwargs = {"first_frame": first, "num_frames": 88}
        if last_frame and os.path.isfile(last_frame):
            gen_kwargs["last_frame"] = load_image(last_frame)
            notes.append("First+last frame interpolation")
        if prompt:
            gen_kwargs["prompt"] = prompt.strip()
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        if on_progress:
            on_progress(30, "Generating interpolation...")

        start = time.monotonic()
        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start
        export_to_video(video, output_path, fps=15)

        notes.append(f"Generated in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return AllegroResult(output=output_path, mode="ti2v", duration_seconds=6.0,
                             fps=15.0, generation_seconds=round(elapsed, 2), notes=notes)
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Allegro TI2V failed: {exc}") from exc

__all__ = ["AllegroResult", "check_allegro_available", "INSTALL_HINT",
           "ALLEGRO_MODELS", "generate_t2v", "generate_ti2v"]
