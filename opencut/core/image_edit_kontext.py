"""
OpenCut FLUX.1 Kontext Image Editing (M2.4)

Context-aware image-to-image editing: accepts an image + natural language
instruction and returns the edited result. Primary workflow: apply per-frame
AI edits (object removal, style transfer, subject replacement, background
swap) and propagate changes across a video clip using TokenFlow.

Uses the Apache-2.0 dev variant (FLUX.1 Kontext-dev). The pro variant
is commercial and is NOT used.

Licence: Apache-2.0 (dev variant)
Repository: Black Forest Labs — https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = (
    "pip install diffusers>=0.29 torch transformers accelerate  "
    "# FLUX Kontext-dev weights (~24 GB) downloaded on first use"
)

KONTEXT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class KontextResult:
    output: str = ""
    instruction: str = ""
    width: int = 0
    height: int = 0
    model: str = "flux-kontext-dev"
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "instruction", "width", "height",
                "model", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_kontext_available() -> bool:
    """Return True when diffusers + torch are available for FLUX Kontext."""
    return (_try_import("diffusers") is not None
            and _try_import("torch") is not None
            and _try_import("transformers") is not None)


def get_model_info() -> dict:
    """Return model download status and size info."""
    info = {
        "available": check_kontext_available(),
        "model_id": KONTEXT_MODEL_ID,
        "size_gb": 24,
        "downloaded": False,
        "licence": "Apache-2.0 (dev variant)",
    }
    # Check if model is cached locally
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if "FLUX.1-Kontext-dev" in str(repo.repo_id):
                info["downloaded"] = True
                break
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Image editing
# ---------------------------------------------------------------------------

def edit_image(
    image_path: str,
    instruction: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.5,
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> KontextResult:
    """Edit an image using FLUX.1 Kontext natural language instructions.

    Args:
        image_path: Path to source image (JPEG/PNG).
        instruction: Natural language edit instruction
            (e.g., "remove the person in the background",
             "change the sky to sunset", "make it look like a painting").
        num_inference_steps: Diffusion steps (higher = better quality, slower).
        guidance_scale: How closely to follow the instruction (1.0 to 15.0).
        seed: Random seed (-1 for random).
        output_path: Where to write result. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        KontextResult with output path and metadata.
    """
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not instruction or not instruction.strip():
        raise ValueError("Instruction must not be empty")

    if not check_kontext_available():
        raise RuntimeError(f"FLUX Kontext dependencies not installed. {INSTALL_HINT}")

    num_inference_steps = max(4, min(100, int(num_inference_steps)))
    guidance_scale = max(1.0, min(15.0, float(guidance_scale)))

    if on_progress:
        on_progress(5, "Loading FLUX Kontext model...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".png", prefix="opencut_kontext_",
        )
        os.close(fd)

    try:
        import torch
        from diffusers import FluxKontextPipeline
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = FluxKontextPipeline.from_pretrained(
            KONTEXT_MODEL_ID,
            torch_dtype=dtype,
        ).to(device)

        if on_progress:
            on_progress(30, "Editing image...")

        source_img = Image.open(image_path).convert("RGB")
        notes.append(f"Source: {source_img.width}x{source_img.height}")
        notes.append(f"Instruction: {instruction[:100]}")

        start_time = time.monotonic()

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)
            notes.append(f"Seed: {seed}")

        result_img = pipe(
            image=source_img,
            prompt=instruction.strip(),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        gen_time = time.monotonic() - start_time

        result_img.save(output_path)
        notes.append(f"Steps: {num_inference_steps}, CFG: {guidance_scale}")
        notes.append(f"Generated in {gen_time:.1f}s")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if on_progress:
            on_progress(100, "Done")

        return KontextResult(
            output=output_path,
            instruction=instruction[:200],
            width=result_img.width,
            height=result_img.height,
            model="flux-kontext-dev",
            generation_seconds=round(gen_time, 2),
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"FLUX Kontext dependency missing: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"FLUX Kontext editing failed: {exc}") from exc


__all__ = [
    "KontextResult",
    "check_kontext_available",
    "get_model_info",
    "INSTALL_HINT",
    "KONTEXT_MODEL_ID",
    "edit_image",
]
