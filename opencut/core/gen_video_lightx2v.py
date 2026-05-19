"""
OpenCut LightX2V Quantization + Step Distillation (N1.2)

FP8/INT8 quantization + 4-step distilled Wan2.2 I2V A14B.
Reduces 80 GB VRAM to 24 GB cards. Up to 42x acceleration.

Licence: Apache-2.0
Repository: https://github.com/ModelTC/lightx2v
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

INSTALL_HINT = "pip install lightx2v  # Apache-2.0, FP8/4-step Wan2.2 I2V"

QUANT_MODES = {
    "fp8": "FP8 quantization — ~2x speedup, minimal quality loss",
    "int8": "INT8 quantization — ~2.5x speedup, slight quality loss",
    "fp16": "FP16 baseline — no quantization",
}

STEP_PRESETS = {
    4: "4-step distilled — fastest, good quality for previews",
    8: "8-step — balanced speed/quality",
    20: "20-step — near-baseline quality",
    50: "50-step — full baseline (no distillation)",
}


@dataclass
class LightX2VResult:
    output: str = ""
    duration_seconds: float = 0.0
    quant_mode: str = ""
    steps: int = 0
    vram_used_mb: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "quant_mode", "steps",
                "vram_used_mb", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_lightx2v_available() -> bool:
    return _try_import("lightx2v") is not None


def generate_i2v(
    image_path: str,
    prompt: str = "",
    duration: float = 5.0,
    quant: str = "fp8",
    steps: int = 4,
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> LightX2VResult:
    """Generate I2V video with quantized/distilled Wan2.2 A14B."""
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not check_lightx2v_available():
        raise RuntimeError(f"lightx2v not installed. {INSTALL_HINT}")

    if quant not in QUANT_MODES:
        quant = "fp8"
    if steps not in STEP_PRESETS:
        steps = 4

    if on_progress:
        on_progress(5, f"Loading LightX2V ({quant}, {steps} steps)...")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_lx2v_")
        os.close(fd)

    try:
        from lightx2v import LightX2VPipeline

        pipe = LightX2VPipeline(quant_mode=quant, num_steps=steps)
        if on_progress:
            on_progress(25, "Generating quantized I2V...")

        start = time.monotonic()
        kwargs: Dict[str, Any] = {
            "image_path": image_path,
            "num_frames": int(duration * 24),
            "output_path": output_path,
        }
        if prompt:
            kwargs["prompt"] = prompt.strip()
        if seed >= 0:
            kwargs["seed"] = seed

        pipe.generate(**kwargs)
        elapsed = time.monotonic() - start

        notes.append(f"Quant: {quant}, Steps: {steps}")
        notes.append(f"Generated in {elapsed:.1f}s")

        if on_progress:
            on_progress(100, "Done")

        return LightX2VResult(
            output=output_path, duration_seconds=round(duration, 2),
            quant_mode=quant, steps=steps, vram_used_mb=0,
            generation_seconds=round(elapsed, 2), notes=notes,
        )
    except ImportError as exc:
        raise RuntimeError(f"lightx2v import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"LightX2V generation failed: {exc}") from exc


__all__ = ["LightX2VResult", "check_lightx2v_available", "INSTALL_HINT",
           "QUANT_MODES", "STEP_PRESETS", "generate_i2v"]
