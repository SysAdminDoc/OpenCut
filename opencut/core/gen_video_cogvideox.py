"""
OpenCut CogVideoX T2V + I2V (N3.1)

Second T2V family alongside Wan2.2. Runs on RTX 3060 (12 GB VRAM).
CogVideoX-2B even runs on GTX 1080Ti. 10-second videos at higher
resolution. DDIM inverse support for non-destructive video editing.

Licence: Apache-2.0
Repository: https://github.com/THUDM/CogVideo
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
    "pip install diffusers>=0.32  # CogVideoX weights auto-download from HuggingFace (~18 GB)"
)

COGVIDEOX_MODELS = {
    "CogVideoX-2B": "2B — runs on GTX 1080Ti (8 GB), lower quality",
    "CogVideoX-5B": "5B — runs on RTX 3060 (12 GB), recommended",
    "CogVideoX1.5-5B": "1.5-5B — 10s videos, any resolution I2V",
}


@dataclass
class CogVideoXResult:
    output: str = ""
    mode: str = "t2v"
    model: str = ""
    duration_seconds: float = 0.0
    fps: float = 8.0
    width: int = 0
    height: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "model", "duration_seconds", "fps",
                "width", "height", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_cogvideox_available() -> bool:
    return (_try_import("diffusers") is not None
            and _try_import("torch") is not None)


def generate_t2v(
    prompt: str,
    model: str = "CogVideoX-5B",
    num_frames: int = 49,
    guidance_scale: float = 6.0,
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> CogVideoXResult:
    """Generate video from text using CogVideoX."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt required")
    if not check_cogvideox_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")
    if model not in COGVIDEOX_MODELS:
        model = "CogVideoX-5B"

    if on_progress:
        on_progress(5, f"Loading CogVideoX ({model})...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_cogvx_")
        os.close(fd)

    try:
        import torch
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = CogVideoXPipeline.from_pretrained(
            f"THUDM/{model}", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        if on_progress:
            on_progress(30, "Generating video...")

        start = time.monotonic()
        gen_kwargs: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "num_frames": num_frames,
            "guidance_scale": guidance_scale,
        }
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start

        export_to_video(video, output_path, fps=8)

        notes.append(f"Model: {model}")
        notes.append(f"Prompt: {prompt[:100]}")
        notes.append(f"Generated in {elapsed:.1f}s")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        duration = num_frames / 8.0
        return CogVideoXResult(
            output=output_path, mode="t2v", model=model,
            duration_seconds=round(duration, 2), fps=8.0,
            width=720, height=480, generation_seconds=round(elapsed, 2),
            notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"CogVideoX import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"CogVideoX generation failed: {exc}") from exc


def generate_i2v(
    image_path: str,
    prompt: str = "",
    model: str = "CogVideoX1.5-5B",
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> CogVideoXResult:
    """Generate video from image using CogVideoX I2V."""
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not check_cogvideox_available():
        raise RuntimeError(f"Dependencies not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, f"Loading CogVideoX I2V ({model})...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_cogvx_i2v_")
        os.close(fd)

    try:
        import torch
        from diffusers import CogVideoXImageToVideoPipeline
        from diffusers.utils import export_to_video, load_image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            f"THUDM/{model}", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        image = load_image(image_path)
        if on_progress:
            on_progress(30, "Generating I2V...")

        start = time.monotonic()
        gen_kwargs: Dict[str, Any] = {"image": image, "num_frames": 49}
        if prompt:
            gen_kwargs["prompt"] = prompt.strip()
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start

        export_to_video(video, output_path, fps=8)
        notes.append(f"Model: {model}")
        notes.append(f"Generated in {elapsed:.1f}s")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return CogVideoXResult(
            output=output_path, mode="i2v", model=model,
            duration_seconds=6.125, fps=8.0, generation_seconds=round(elapsed, 2),
            notes=notes,
        )
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"CogVideoX I2V failed: {exc}") from exc


__all__ = ["CogVideoXResult", "check_cogvideox_available", "INSTALL_HINT",
           "COGVIDEOX_MODELS", "generate_t2v", "generate_i2v"]
