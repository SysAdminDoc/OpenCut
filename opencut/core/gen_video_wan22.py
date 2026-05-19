"""
OpenCut Wan2.2 Video Generation Backend (M2.1)

Full Wan2.2 model family: T2V (text-to-video), I2V (image-to-video),
and TI2V (text+image-to-video). MoE architecture with cinematic
aesthetics labels. TI2V-5B runs on consumer GPUs (4090 at 720p@24fps).

Replaces the earlier Wan2.1 VACE stub.

Licence: Apache-2.0
Repository: https://github.com/Wan-Video/Wan2.2
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
    "git clone https://github.com/Wan-Video/Wan2.2 && "
    "pip install -r Wan2.2/requirements.txt  # Apache-2.0"
)

WAN22_MODELS = {
    "ti2v-5b": "TI2V-5B — text+image-to-video, consumer GPU (720p@24fps on 4090)",
    "t2v-14b": "T2V-14B — text-to-video, multi-GPU or offload",
    "i2v-14b": "I2V-14B — image-to-video, multi-GPU or offload",
}

WAN22_MODES = {
    "t2v": "Text-to-video generation",
    "i2v": "Image-to-video generation",
    "ti2v": "Text+image-to-video (recommended for consumer GPUs)",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class Wan22Result:
    output: str = ""
    mode: str = ""
    model: str = "ti2v-5b"
    duration_seconds: float = 0.0
    fps: float = 24.0
    width: int = 0
    height: int = 0
    generation_seconds: float = 0.0
    prompt: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "model", "duration_seconds", "fps",
                "width", "height", "generation_seconds", "prompt", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_wan22_available() -> bool:
    """Return True when Wan2.2 inference is available."""
    if _try_import("wan") is not None:
        return True
    # Check env-configured path
    wan_path = os.environ.get("OPENCUT_WAN22_PATH", "")
    if wan_path and os.path.isdir(wan_path):
        return True
    return False


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_t2v(
    prompt: str,
    duration: float = 5.0,
    model: str = "ti2v-5b",
    negative_prompt: str = "",
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    seed: int = -1,
    offload_model: bool = True,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> Wan22Result:
    """Generate video from text using Wan2.2.

    Args:
        prompt: Text description of desired video.
        duration: Target duration in seconds (max 16 for T2V).
        model: Model variant — ti2v-5b, t2v-14b, i2v-14b.
        negative_prompt: What to avoid.
        fps: Frame rate (default 24).
        width: Output width (default 1280).
        height: Output height (default 720).
        seed: Random seed (-1 for random).
        offload_model: Use CPU offload for VRAM-limited GPUs.
        output_path: Output path. Auto-generated if empty.
        on_progress: Optional ``(pct, msg="")`` callback.

    Returns:
        Wan22Result with output path and metadata.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if not check_wan22_available():
        raise RuntimeError(f"Wan2.2 not installed. {INSTALL_HINT}")

    duration = max(1.0, min(16.0, float(duration)))
    if model not in WAN22_MODELS:
        model = "ti2v-5b"

    if on_progress:
        on_progress(5, f"Loading Wan2.2 {model}...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_wan22_")
        os.close(fd)

    try:
        from wan import Wan22Pipeline

        pipe = Wan22Pipeline(model_name=model, offload=offload_model)

        if on_progress:
            on_progress(20, f"Generating {duration:.0f}s video...")

        start_time = time.monotonic()

        kwargs: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "num_frames": int(duration * fps),
            "fps": int(fps),
            "width": width,
            "height": height,
            "output_path": output_path,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed >= 0:
            kwargs["seed"] = seed
            notes.append(f"Seed: {seed}")

        pipe.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        notes.append(f"Model: {model}")
        notes.append(f"Prompt: {prompt[:100]}")
        notes.append(f"Generated in {gen_time:.1f}s")
        if offload_model:
            notes.append("CPU offload enabled")

        # Probe output
        actual_duration = duration
        try:
            from opencut.helpers import get_video_info
            info = get_video_info(output_path)
            actual_duration = info.get("duration", duration)
            width = info.get("width", width)
            height = info.get("height", height)
        except Exception:
            pass

        if on_progress:
            on_progress(100, "Done")

        return Wan22Result(
            output=output_path,
            mode="t2v",
            model=model,
            duration_seconds=round(actual_duration, 2),
            fps=fps,
            width=width,
            height=height,
            generation_seconds=round(gen_time, 2),
            prompt=prompt[:200],
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(f"Wan2.2 import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Wan2.2 generation failed: {exc}") from exc


def generate_i2v(
    image_path: str,
    prompt: str = "",
    duration: float = 5.0,
    model: str = "ti2v-5b",
    seed: int = -1,
    offload_model: bool = True,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> Wan22Result:
    """Generate video from image + optional text using Wan2.2.

    Args:
        image_path: Source image path.
        prompt: Optional motion/action description.
        duration: Target duration (max 16s).
        model: Model variant.
        seed: Random seed.
        offload_model: CPU offload for VRAM-limited GPUs.
        output_path: Output path.
        on_progress: Callback.

    Returns:
        Wan22Result with output path and metadata.
    """
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Image not found: {image_path}")
    if not check_wan22_available():
        raise RuntimeError(f"Wan2.2 not installed. {INSTALL_HINT}")

    duration = max(1.0, min(16.0, float(duration)))
    if model not in WAN22_MODELS:
        model = "ti2v-5b"

    if on_progress:
        on_progress(5, f"Loading Wan2.2 {model} for I2V...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_wan22_i2v_")
        os.close(fd)

    try:
        from wan import Wan22Pipeline

        pipe = Wan22Pipeline(model_name=model, offload=offload_model)

        if on_progress:
            on_progress(20, f"Generating {duration:.0f}s video from image...")

        start_time = time.monotonic()

        kwargs: Dict[str, Any] = {
            "image_path": image_path,
            "num_frames": int(duration * 24),
            "fps": 24,
            "output_path": output_path,
        }
        if prompt:
            kwargs["prompt"] = prompt.strip()
            notes.append(f"Prompt: {prompt[:100]}")
        if seed >= 0:
            kwargs["seed"] = seed

        pipe.generate_i2v(**kwargs)
        gen_time = time.monotonic() - start_time

        notes.append(f"Model: {model}")
        notes.append(f"Generated in {gen_time:.1f}s")

        if on_progress:
            on_progress(100, "Done")

        return Wan22Result(
            output=output_path,
            mode="i2v",
            model=model,
            duration_seconds=round(duration, 2),
            fps=24.0,
            generation_seconds=round(gen_time, 2),
            prompt=prompt[:200] if prompt else "",
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(f"Wan2.2 import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Wan2.2 I2V failed: {exc}") from exc


__all__ = [
    "Wan22Result",
    "check_wan22_available",
    "INSTALL_HINT",
    "WAN22_MODELS",
    "WAN22_MODES",
    "generate_t2v",
    "generate_i2v",
]
