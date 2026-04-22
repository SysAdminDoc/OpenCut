"""
OpenCut CineFocus v1.28.0 — STUB

AI rack-focus from depth map using Depth Pro or Depth Anything V2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


def check_cinefocus_available() -> bool:
    return _try_import("torch") is not None and _try_import("transformers") is not None


INSTALL_HINT = (
    "pip install torch transformers "
    "(uses Depth Pro or Depth Anything V2 depth backend)"
)


@dataclass
class CineFocusResult:
    output: str = ""
    focal_z: float = 0.5
    aperture_f: float = 2.8
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "focal_z", "aperture_f", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def preview(
    video_path: str,
    focal_z: float = 0.5,
    aperture_f: float = 2.8,
    frame: int = 0,
) -> Dict:
    if not check_cinefocus_available():
        raise RuntimeError(f"torch/transformers not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("CineFocus wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


def render(
    video_path: str,
    focal_z_start: float = 0.5,
    focal_z_end: float = 0.5,
    focal_frame_start: int = 0,
    focal_frame_end: int = 0,
    aperture_f: float = 2.8,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> CineFocusResult:
    if not check_cinefocus_available():
        raise RuntimeError(f"torch/transformers not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("CineFocus wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["check_cinefocus_available", "INSTALL_HINT", "CineFocusResult", "preview", "render"]
