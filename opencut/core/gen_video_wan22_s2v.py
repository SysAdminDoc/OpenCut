"""
OpenCut Wan2.2-S2V Speech-to-Video (M2.2)

14B model that generates a talking-head video from an audio clip +
reference portrait image. Accepts real voice files — no TTS needed.
Optional CosyVoice2 integration enables full text->speech->video mode.

Requires 80 GB VRAM single-GPU, multi-GPU via FSDP, or
--offload_model True for consumer cards.

Licence: Apache-2.0
Repository: https://github.com/Wan-Video/Wan2.2 (S2V variant)
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
    "pip install -r Wan2.2/requirements_s2v.txt  # Apache-2.0, 14B S2V model"
)


@dataclass
class S2VResult:
    output: str = ""
    duration_seconds: float = 0.0
    fps: float = 24.0
    width: int = 0
    height: int = 0
    model: str = "wan2.2-s2v-14b"
    generation_seconds: float = 0.0
    audio_source: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "fps", "width", "height",
                "model", "generation_seconds", "audio_source", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_s2v_available() -> bool:
    """Return True when Wan2.2-S2V inference is available."""
    if _try_import("wan") is not None:
        return True
    s2v_path = os.environ.get("OPENCUT_WAN22_S2V_PATH", "")
    return bool(s2v_path and os.path.isdir(s2v_path))


def generate(
    audio_path: str,
    portrait_path: str,
    prompt: str = "",
    offload_model: bool = True,
    half_body: bool = False,
    seed: int = -1,
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> S2VResult:
    """Generate talking-head video from audio + portrait.

    Args:
        audio_path: Path to audio/speech file.
        portrait_path: Path to reference portrait image.
        prompt: Optional text prompt for scene context.
        offload_model: CPU offload for VRAM savings.
        half_body: Generate upper-body (True) or head-only (False).
        seed: Random seed (-1 for random).
        output_path: Output MP4 path. Auto-generated if empty.
        on_progress: Optional callback.

    Returns:
        S2VResult with output path and metadata.
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError(f"Audio not found: {audio_path}")
    if not portrait_path or not os.path.isfile(portrait_path):
        raise ValueError(f"Portrait not found: {portrait_path}")
    if not check_s2v_available():
        raise RuntimeError(f"Wan2.2-S2V not installed. {INSTALL_HINT}")

    if on_progress:
        on_progress(5, "Loading Wan2.2-S2V model...")

    notes: List[str] = []

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_s2v_")
        os.close(fd)

    try:
        from wan import Wan22S2VPipeline

        pipe = Wan22S2VPipeline(offload=offload_model)

        if on_progress:
            on_progress(20, "Generating talking-head video...")

        start_time = time.monotonic()

        kwargs = {
            "audio_path": audio_path,
            "portrait_path": portrait_path,
            "output_path": output_path,
        }
        if prompt:
            kwargs["prompt"] = prompt.strip()
            notes.append(f"Prompt: {prompt[:80]}")
        if half_body:
            kwargs["half_body"] = True
            notes.append("Half-body mode")
        if seed >= 0:
            kwargs["seed"] = seed
        if offload_model:
            notes.append("CPU offload enabled")

        pipe.generate(**kwargs)
        gen_time = time.monotonic() - start_time

        notes.append(f"Audio: {os.path.basename(audio_path)}")
        notes.append(f"Portrait: {os.path.basename(portrait_path)}")
        notes.append(f"Generated in {gen_time:.1f}s")

        # Probe output
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

        return S2VResult(
            output=output_path,
            duration_seconds=round(duration, 2),
            fps=24.0,
            width=w,
            height=h,
            model="wan2.2-s2v-14b",
            generation_seconds=round(gen_time, 2),
            audio_source=os.path.basename(audio_path),
            notes=notes,
        )

    except ImportError as exc:
        raise RuntimeError(f"Wan2.2-S2V import failed: {exc}. {INSTALL_HINT}") from exc
    except Exception as exc:
        if output_path and os.path.isfile(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise RuntimeError(f"Wan2.2-S2V generation failed: {exc}") from exc


__all__ = ["S2VResult", "check_s2v_available", "INSTALL_HINT", "generate"]
