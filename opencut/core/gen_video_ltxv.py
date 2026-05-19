"""
OpenCut LTX-Video Generation (O2.1-O2.4)

Fastest Apache-2 DiT video model. Up to 60s video. T2V, I2V,
multi-keyframe conditioning, forward/backward extension.

Licence: Apache-2.0
Repository: https://github.com/Lightricks/LTX-Video
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

INSTALL_HINT = "pip install diffusers>=0.32  # LTX-Video weights auto-download"

LTXV_MODELS = {
    "LTXV-2B": "2B — fast, lower quality, ~6 GB",
    "LTXV-13B": "13B — highest quality, up to 60s, ~25 GB",
}


@dataclass
class LTXVResult:
    output: str = ""
    mode: str = "t2v"
    model: str = "LTXV-2B"
    duration_seconds: float = 0.0
    fps: float = 24.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "model", "duration_seconds", "fps",
                "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_ltxv_available() -> bool:
    return (_try_import("diffusers") is not None
            and _try_import("torch") is not None)


def generate_t2v(
    prompt: str,
    model: str = "LTXV-2B",
    duration: float = 5.0,
    negative_prompt: str = "",
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> LTXVResult:
    """Generate video from text using LTX-Video."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt required")
    if not check_ltxv_available():
        raise RuntimeError(f"LTX-Video deps not installed. {INSTALL_HINT}")
    if model not in LTXV_MODELS:
        model = "LTXV-2B"

    duration = max(1.0, min(60.0, float(duration)))
    if on_progress:
        on_progress(5, f"Loading LTX-Video ({model})...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_ltxv_")
        os.close(fd)

    try:
        import torch
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = LTXPipeline.from_pretrained(
            f"Lightricks/{model}", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        if on_progress:
            on_progress(30, "Generating video...")

        start = time.monotonic()
        gen_kwargs: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "num_frames": int(duration * 24),
            "num_inference_steps": 50,
        }
        if negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start

        export_to_video(video, output_path, fps=24)
        notes.append(f"Model: {model}")
        notes.append(f"Generated in {elapsed:.1f}s")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return LTXVResult(
            output=output_path, mode="t2v", model=model,
            duration_seconds=round(duration, 2), fps=24.0,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"LTX-Video import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"LTX-Video generation failed: {exc}") from exc


def generate_i2v(
    image_path: str,
    prompt: str = "",
    model: str = "LTXV-2B",
    duration: float = 5.0,
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> LTXVResult:
    """Generate video from image using LTX-Video I2V."""
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not check_ltxv_available():
        raise RuntimeError(f"LTX-Video deps not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, "Loading LTX-Video I2V...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_ltxv_i2v_")
        os.close(fd)

    try:
        import torch
        from diffusers import LTXImageToVideoPipeline
        from diffusers.utils import export_to_video, load_image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = LTXImageToVideoPipeline.from_pretrained(
            f"Lightricks/{model}", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        image = load_image(image_path)
        if on_progress:
            on_progress(30, "Generating I2V...")

        start = time.monotonic()
        gen_kwargs: Dict[str, Any] = {
            "image": image,
            "num_frames": int(duration * 24),
        }
        if prompt:
            gen_kwargs["prompt"] = prompt.strip()
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start
        export_to_video(video, output_path, fps=24)

        notes.append(f"Model: {model}")
        notes.append(f"Generated in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return LTXVResult(
            output=output_path, mode="i2v", model=model,
            duration_seconds=round(duration, 2), fps=24.0,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"LTX-Video I2V failed: {exc}") from exc


def extend_video(
    video_path: str,
    direction: str = "forward",
    duration_sec: float = 3.0,
    model: str = "LTXV-2B",
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> LTXVResult:
    """Extend a video forward or backward in time."""
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"Video not found: {video_path}")
    if direction not in ("forward", "backward"):
        direction = "forward"
    if not check_ltxv_available():
        raise RuntimeError(f"LTX-Video deps not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(10, f"Extending video ({direction})...")

    notes: List[str] = [f"Direction: {direction}", f"Extension: {duration_sec}s"]
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_ltxv_ext_")
        os.close(fd)

    try:
        import torch
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = LTXPipeline.from_pretrained(
            f"Lightricks/{model}", torch_dtype=torch.bfloat16
        ).to(device)

        start = time.monotonic()
        # Extension uses the video conditioning path
        video = pipe.extend(
            video_path=video_path,
            direction=direction,
            num_frames=int(duration_sec * 24),
        ).frames[0]
        elapsed = time.monotonic() - start

        export_to_video(video, output_path, fps=24)
        notes.append(f"Generated in {elapsed:.1f}s")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return LTXVResult(
            output=output_path, mode="extend", model=model,
            duration_seconds=round(duration_sec, 2), fps=24.0,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"LTX-Video extension failed: {exc}") from exc


__all__ = ["LTXVResult", "check_ltxv_available", "INSTALL_HINT", "LTXV_MODELS",
           "generate_t2v", "generate_i2v", "extend_video"]
