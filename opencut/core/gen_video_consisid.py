"""
OpenCut ConsisID Identity-Preserving T2V (P1.1)

Face reference -> consistent identity throughout generated video.
Built on CogVideoX-5B, ~18 GB VRAM. CVPR 2025 Highlight.

Licence: Apache-2.0
Repository: https://github.com/PKU-YuanGroup/ConsisID
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
INSTALL_HINT = "pip install diffusers>=0.33  # ConsisID weights ~18 GB auto-download"


@dataclass
class ConsisIDResult:
    output: str = ""
    duration_seconds: float = 0.0
    fps: float = 8.0
    generation_seconds: float = 0.0
    identity_preserved: bool = True
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "fps", "generation_seconds",
                "identity_preserved", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_consisid_available() -> bool:
    return _try_import("diffusers") is not None and _try_import("torch") is not None


def generate(face_image: str, prompt: str, duration: float = 6.0, seed: int = -1,
             output_path: str = "", on_progress=None) -> ConsisIDResult:
    if not face_image or not os.path.isfile(face_image):
        raise ValueError(f"Face image not found: {face_image}")
    if not prompt:
        raise ValueError("Prompt required")
    if not check_consisid_available():
        raise RuntimeError(f"ConsisID deps not installed. {INSTALL_HINT}")

    notes: List[str] = []
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_consisid_")
        os.close(fd)

    try:
        import torch
        from diffusers import ConsisIDPipeline
        from diffusers.utils import export_to_video, load_image

        if on_progress:
            on_progress(10, "Loading ConsisID model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = ConsisIDPipeline.from_pretrained(
            "BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16
        ).to(device)
        pipe.enable_model_cpu_offload()

        face = load_image(face_image)
        if on_progress:
            on_progress(30, "Generating identity-preserving video...")

        start = time.monotonic()
        gen_kwargs = {"prompt": prompt.strip(), "image": face,
                      "num_frames": int(duration * 8)}
        if seed >= 0:
            gen_kwargs["generator"] = torch.Generator(device).manual_seed(seed)

        video = pipe(**gen_kwargs).frames[0]
        elapsed = time.monotonic() - start
        export_to_video(video, output_path, fps=8)

        notes.append(f"Generated in {elapsed:.1f}s")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_progress:
            on_progress(100, "Done")

        return ConsisIDResult(output=output_path, duration_seconds=round(duration, 2),
                              fps=8.0, generation_seconds=round(elapsed, 2),
                              identity_preserved=True, notes=notes)
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"ConsisID failed: {exc}") from exc

__all__ = ["ConsisIDResult", "check_consisid_available", "INSTALL_HINT", "generate"]
