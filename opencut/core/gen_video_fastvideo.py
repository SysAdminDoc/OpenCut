"""
OpenCut FastVideo Inference Acceleration (N1.1)

>50x denoising speedup for Wan2.2 T2V/TI2V via sparse-attention
distillation. 5-second 1080P video in 4.5s on a single 4090.

Licence: Apache-2.0
Repository: https://github.com/hao-ai-lab/FastVideo
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

INSTALL_HINT = "pip install fastvideo  # Apache-2.0, >50x Wan2.2 speedup"

FAST_MODELS = {
    "FastWan2.2-TI2V-5B": "Distilled TI2V-5B — consumer GPU, real-time 1080P",
    "FastWan2.1-T2V-1.3B": "Distilled T2V-1.3B — ultra-fast, lower quality",
}


@dataclass
class FastVideoResult:
    output: str = ""
    duration_seconds: float = 0.0
    fps: float = 24.0
    model: str = ""
    speedup_factor: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "fps", "model",
                "speedup_factor", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_fastvideo_available() -> bool:
    return _try_import("fastvideo") is not None


def generate(
    prompt: str,
    duration: float = 5.0,
    model: str = "FastWan2.2-TI2V-5B",
    image_path: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> FastVideoResult:
    """Generate video using FastVideo distilled models."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if not check_fastvideo_available():
        raise RuntimeError(f"fastvideo not installed. {INSTALL_HINT}")

    duration = max(1.0, min(10.0, float(duration)))
    if model not in FAST_MODELS:
        model = "FastWan2.2-TI2V-5B"

    if on_progress:
        on_progress(5, f"Loading FastVideo ({model})...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_fast_")
        os.close(fd)

    try:
        from fastvideo import FastVideoGenerator

        gen = FastVideoGenerator(model_name=model)
        if on_progress:
            on_progress(20, "Generating video (accelerated)...")

        start = time.monotonic()
        kwargs: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "num_frames": int(duration * 24),
            "output_path": output_path,
        }
        if image_path and os.path.isfile(image_path):
            kwargs["image_path"] = image_path
            notes.append(f"Image: {os.path.basename(image_path)}")
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed >= 0:
            kwargs["seed"] = seed

        gen.generate(**kwargs)
        elapsed = time.monotonic() - start

        # Estimate speedup vs baseline
        baseline_estimate = duration * 10  # rough: baseline ~10s per 1s video
        speedup = baseline_estimate / max(elapsed, 0.01)

        notes.append(f"Model: {model}")
        notes.append(f"Generated in {elapsed:.1f}s (~{speedup:.0f}x vs baseline)")

        if on_progress:
            on_progress(100, "Done")

        return FastVideoResult(
            output=output_path, duration_seconds=round(duration, 2),
            fps=24.0, model=model, speedup_factor=round(speedup, 1),
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"fastvideo import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"FastVideo generation failed: {exc}") from exc


__all__ = ["FastVideoResult", "check_fastvideo_available", "INSTALL_HINT", "FAST_MODELS", "generate"]
