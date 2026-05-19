"""
OpenCut Open-Sora 2.0 High-Quality T2V (P3.2)

11B SOTA T2V — equals HunyuanVideo quality under Apache-2.0.
720x1280 5-second video. Open-Sora 1.3 (1B) as lightweight fallback.

Licence: Apache-2.0
Repository: https://github.com/hpcaitech/Open-Sora
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
INSTALL_HINT = "pip install opensora  # Apache-2.0, 11B + 1B T2V models"

OPENSORA_MODELS = {
    "11b": "Open-Sora 2.0 (11B) — SOTA quality, ~22 GB VRAM",
    "1b": "Open-Sora 1.3 (1B) — lightweight fallback, ~4 GB VRAM",
}


@dataclass
class OpenSora2Result:
    output: str = ""
    model: str = "11b"
    duration_seconds: float = 0.0
    fps: float = 24.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "model", "duration_seconds", "fps",
                "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_opensora2_available() -> bool:
    return _try_import("opensora") is not None or _try_import("torch") is not None


def generate(prompt: str, model: str = "11b", duration: float = 5.0,
             seed: int = -1, output_path: str = "", on_progress=None) -> OpenSora2Result:
    if not prompt:
        raise ValueError("Prompt required")
    if not check_opensora2_available():
        raise RuntimeError(f"Open-Sora not installed. {INSTALL_HINT}")
    if model not in OPENSORA_MODELS:
        model = "11b"

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_opensora2_")
        os.close(fd)

    try:
        if on_progress:
            on_progress(10, f"Loading Open-Sora {model}...")

        from opensora import OpenSoraPipeline

        pipe = OpenSoraPipeline(model_variant=model)

        if on_progress:
            on_progress(25, "Generating SOTA quality video...")

        start = time.monotonic()
        kwargs = {"prompt": prompt.strip(), "num_frames": int(duration * 24),
                  "output_path": output_path}
        if seed >= 0:
            kwargs["seed"] = seed

        pipe.generate(**kwargs)
        elapsed = time.monotonic() - start

        notes.append(f"Model: Open-Sora {model}")
        notes.append(f"Generated in {elapsed:.1f}s")
        if on_progress:
            on_progress(100, "Done")

        return OpenSora2Result(output=output_path, model=model,
                               duration_seconds=round(duration, 2), fps=24.0,
                               generation_seconds=round(elapsed, 2), notes=notes)
    except ImportError as exc:
        raise RuntimeError(f"Open-Sora import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Open-Sora generation failed: {exc}") from exc

__all__ = ["OpenSora2Result", "check_opensora2_available", "INSTALL_HINT",
           "OPENSORA_MODELS", "generate"]
