"""
OpenCut FramePack Image-to-Video Backend

Next-frame-prediction video diffusion that generates up to 60-second
video from a single image + prompt on 6 GB VRAM.

Licence: Apache-2.0
Repository: https://github.com/lllyasviel/FramePack
Paper: NeurIPS 2025
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

INSTALL_HINT = "pip install framepack  # Apache-2.0, 6 GB VRAM I2V diffusion"

FRAMEPACK_MODELS = {
    "framepack-standard": "Standard — balanced quality/speed (13B)",
    "framepack-fast": "Fast — lower quality, 2x faster inference",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FramePackResult:
    output: str = ""
    duration_seconds: float = 0.0
    fps: float = 24.0
    width: int = 0
    height: int = 0
    model: str = "framepack-standard"
    generation_seconds: float = 0.0
    prompt: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "fps", "width", "height",
                "model", "generation_seconds", "prompt", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_framepack_available() -> bool:
    """Return True when the framepack package is importable."""
    return _try_import("framepack") is not None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    image_path: str,
    prompt: str = "",
    duration: float = 5.0,
    fps: float = 24.0,
    model: str = "framepack-standard",
    negative_prompt: str = "",
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> FramePackResult:
    """Generate video from a single image using FramePack.

    Args:
        image_path: Path to source image (JPEG/PNG).
        prompt: Text description of desired motion/action.
        duration: Target duration in seconds (max 60).
        fps: Output frame rate (default 24).
        model: Model variant — framepack-standard or framepack-fast.
        negative_prompt: Things to avoid in generation.
        seed: Random seed (-1 for random).
        output_path: Where to write MP4. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        FramePackResult with output path and metadata.

    Raises:
        RuntimeError: When framepack is not installed or generation fails.
    """
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    framepack_mod = _try_import("framepack")
    if framepack_mod is None:
        raise RuntimeError(f"framepack is not installed. {INSTALL_HINT}")

    duration = max(1.0, min(60.0, float(duration)))
    fps = max(8.0, min(60.0, float(fps)))
    if model not in FRAMEPACK_MODELS:
        model = "framepack-standard"

    if on_progress:
        on_progress(5, f"Loading FramePack model ({model})...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(
            suffix=".mp4", prefix="opencut_framepack_",
        )
        os.close(fd)

    try:
        from framepack import FramePack

        fp_model = FramePack(model_name=model)

        if on_progress:
            on_progress(20, f"Generating {duration:.0f}s video from image...")

        start_time = time.monotonic()

        kwargs = {
            "image_path": image_path,
            "output_path": output_path,
            "num_frames": int(duration * fps),
            "fps": int(fps),
        }
        if prompt:
            kwargs["prompt"] = prompt
            notes.append(f"Prompt: {prompt[:100]}")
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed >= 0:
            kwargs["seed"] = seed
            notes.append(f"Seed: {seed}")

        fp_model.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        if on_progress:
            on_progress(90, "Finalizing...")

        notes.append(f"Generated in {gen_time:.1f}s")

        # Probe output dimensions
        width, height = 0, 0
        actual_duration = duration
        try:
            from opencut.helpers import get_video_info
            info = get_video_info(output_path)
            width = info.get("width", 0)
            height = info.get("height", 0)
            if info.get("duration", 0) > 0:
                actual_duration = info["duration"]
        except Exception:
            pass

        if on_progress:
            on_progress(100, "Done")

        return FramePackResult(
            output=output_path,
            duration_seconds=round(actual_duration, 2),
            fps=fps,
            width=width,
            height=height,
            model=model,
            generation_seconds=round(gen_time, 2),
            prompt=prompt[:200] if prompt else "",
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(
            f"framepack import failed: {exc}. {INSTALL_HINT}"
        ) from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"FramePack generation failed: {exc}") from exc
