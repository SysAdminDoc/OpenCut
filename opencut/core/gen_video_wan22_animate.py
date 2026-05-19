"""
OpenCut Wan2.2-Animate Character Animation (M2.3)

Unified model for two workflows:
(a) Animate a still character photo to match motions from a reference video.
(b) Replace the character in a video with a different appearance while
    preserving all movements and expressions.

Complements EchoMimic V3 (portrait-only lip-sync from audio) — Animate
excels at full-body motion transfer and character swap.

Licence: Apache-2.0
Repository: https://github.com/Wan-Video/Wan2.2 (Animate variant)
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
    "git clone https://github.com/Wan-Video/Wan2.2 && "
    "pip install -r Wan2.2/requirements.txt  # Apache-2.0, Animate-14B weights"
)

ANIMATE_MODES = {
    "motion_transfer": "Animate a still photo using motions from a reference video",
    "character_replace": "Replace the character in a video with a new appearance",
}


@dataclass
class AnimateResult:
    output: str = ""
    mode: str = ""
    duration_seconds: float = 0.0
    fps: float = 24.0
    width: int = 0
    height: int = 0
    model: str = "wan2.2-animate-14b"
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "duration_seconds", "fps", "width",
                "height", "model", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_animate_available() -> bool:
    """Return True when Wan2.2-Animate is available."""
    if _try_import("wan") is not None:
        return True
    anim_path = os.environ.get("OPENCUT_WAN22_ANIMATE_PATH", "")
    return bool(anim_path and os.path.isdir(anim_path))


def animate(
    character_image: str,
    motion_video: str,
    mode: str = "motion_transfer",
    offload_model: bool = True,
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> AnimateResult:
    """Animate a character or replace a character in video.

    Args:
        character_image: Path to the character/appearance image.
        motion_video: Path to the motion reference video.
        mode: motion_transfer or character_replace.
        offload_model: CPU offload for VRAM savings.
        seed: Random seed (-1 for random).
        output_path: Output MP4 path. Auto-generated if empty.
        on_progress: Optional callback.

    Returns:
        AnimateResult with output path and metadata.
    """
    if not character_image or not os.path.isfile(character_image):
        raise ValueError(f"Character image not found: {character_image}")
    if not motion_video or not os.path.isfile(motion_video):
        raise ValueError(f"Motion video not found: {motion_video}")
    if mode not in ANIMATE_MODES:
        mode = "motion_transfer"
    if not check_animate_available():
        raise RuntimeError(f"Wan2.2-Animate not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, f"Loading Wan2.2-Animate ({mode})...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_animate_")
        os.close(fd)

    try:
        from wan import Wan22AnimatePipeline

        pipe = Wan22AnimatePipeline(offload=offload_model)

        if on_progress:
            on_progress(20, f"Running {mode}...")

        start_time = time.monotonic()

        kwargs = {
            "character_image": character_image,
            "motion_video": motion_video,
            "mode": mode,
            "output_path": output_path,
        }
        if seed >= 0:
            kwargs["seed"] = seed
        if offload_model:
            notes.append("CPU offload enabled")

        pipe.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        notes.append(f"Mode: {mode}")
        notes.append(f"Character: {os.path.basename(character_image)}")
        notes.append(f"Motion: {os.path.basename(motion_video)}")
        notes.append(f"Generated in {gen_time:.1f}s")

        duration = 0.0
        w, h = 0, 0
        try:
            from opencut.helpers import get_video_info
            info = get_video_info(output_path)
            duration = info.get("duration", 0)
            w = info.get("width", 0)
            h = info.get("height", 0)
        except Exception:
            pass

        if on_progress:
            on_progress(100, "Done")

        return AnimateResult(
            output=output_path,
            mode=mode,
            duration_seconds=round(duration, 2),
            fps=24.0,
            width=w,
            height=h,
            model="wan2.2-animate-14b",
            generation_seconds=round(gen_time, 2),
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(f"Wan2.2-Animate import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Wan2.2-Animate failed: {exc}") from exc


__all__ = ["AnimateResult", "check_animate_available", "INSTALL_HINT", "ANIMATE_MODES", "animate"]
