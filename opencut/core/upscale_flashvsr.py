"""
OpenCut FlashVSR Upscaling (Wave H2.1, v1.25.0) — **STUB**

Real-time streaming video super-resolution via locality-constrained
sparse attention. Source: https://github.com/OpenImagingLab/FlashVSR

Ships as a ``check_X_available()``-gated stub in v1.25.0 — the route
returns 503 ``MISSING_DEPENDENCY`` with an install hint. Full
integration lands in v1.26.0 once the Python package is pinned.

Design plan for the full implementation:
- Lazy-import ``flashvsr`` (pip) + ``torch``.
- Subscriptable ``FlashVSRResult`` dataclass ``{output, input_fps,
  output_fps, duration, tiles, notes}``.
- ``upscale(video_path, scale=2.0, tile_size=256, output=None,
  on_progress=None)`` entry point. Progress callback defaults
  ``msg=""``. ``@gpu_exclusive`` inside the route's worker body.
- Falls back to Real-ESRGAN (already present) when FlashVSR is absent
  AND the caller passes ``allow_fallback=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class FlashVSRResult:
    output: str = ""
    scale: float = 2.0
    input_fps: float = 0.0
    output_fps: float = 0.0
    duration: float = 0.0
    tiles: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "scale", "input_fps", "output_fps",
                "duration", "tiles", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_flashvsr_available() -> bool:
    """Requires flashvsr pip package + torch."""
    return _try_import("flashvsr") is not None and _try_import("torch") is not None


INSTALL_HINT = (
    "pip install flashvsr torch --index-url https://download.pytorch.org/whl/cu121"
)


def upscale(
    video_path: str,
    scale: float = 2.0,
    tile_size: int = 256,
    output: Optional[str] = None,
    allow_fallback: bool = False,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> FlashVSRResult:
    """Entry-point stub. Raises RuntimeError unless the backend is installed."""
    if not check_flashvsr_available():
        raise RuntimeError(
            "FlashVSR is not installed. Install with:\n"
            f"    {INSTALL_HINT}"
        )
    raise NotImplementedError(
        "FlashVSR wiring ships in v1.26.0. Track "
        "https://github.com/OpenImagingLab/FlashVSR for the pip release."
    )


__all__ = [
    "FlashVSRResult",
    "check_flashvsr_available",
    "INSTALL_HINT",
    "upscale",
]
