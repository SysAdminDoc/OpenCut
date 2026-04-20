"""
OpenCut ROSE Video Inpainting (Wave H2.2, v1.25.0) — **STUB**

Video inpainting that preserves shadows / reflections / translucency.
Solves the "remove object but keep shadow" problem that current
ProPainter workflows can't.  Source: https://rose2025-inpaint.github.io/

Stubbed behind ``check_rose_available()`` in v1.25.0; 503 with install
hint until the weights ship under a permissive licence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class RoseResult:
    output: str = ""
    masked_frames: int = 0
    duration: float = 0.0
    preserves_shadows: bool = True
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "masked_frames", "duration",
                "preserves_shadows", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_rose_available() -> bool:
    """Requires torch + diffusers + a ROSE checkpoint env var."""
    import os
    if _try_import("torch") is None:
        return False
    if _try_import("diffusers") is None:
        return False
    ckpt = os.environ.get("OPENCUT_ROSE_CHECKPOINT", "").strip()
    return bool(ckpt) and os.path.isfile(ckpt)


INSTALL_HINT = (
    "pip install torch diffusers && "
    "set OPENCUT_ROSE_CHECKPOINT=<path to rose.safetensors> "
    "(track https://rose2025-inpaint.github.io/ for weights)"
)


def inpaint(
    video_path: str,
    mask_path: str,
    output: Optional[str] = None,
    preserve_shadows: bool = True,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> RoseResult:
    """Entry-point stub. Raises RuntimeError until backend is installed."""
    if not check_rose_available():
        raise RuntimeError(
            "ROSE inpainting is not installed. " + INSTALL_HINT
        )
    raise NotImplementedError(
        "ROSE wiring ships in a future release. Track "
        "https://rose2025-inpaint.github.io/ for the weights + pip "
        "package."
    )


__all__ = [
    "RoseResult",
    "check_rose_available",
    "INSTALL_HINT",
    "inpaint",
]
