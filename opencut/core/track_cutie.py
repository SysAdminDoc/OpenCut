"""
OpenCut Cutie Object Tracking v1.28.0 — STUB

Memory-efficient long-term object tracking via Cutie.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class CutieResult:
    output_mask_path: str = ""
    tracked_frames: int = 0
    object_id: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output_mask_path", "tracked_frames", "object_id", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_cutie_available() -> bool:
    return _try_import("cutie") is not None or (
        _try_import("torch") is not None and _try_import("torchvision") is not None
    )


INSTALL_HINT = "pip install cutie torch torchvision"


def track(
    video_path: str,
    mask_frame0_path: str,
    object_id: int = 1,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> CutieResult:
    if not check_cutie_available():
        raise RuntimeError(f"Cutie is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("Cutie tracking wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["CutieResult", "check_cutie_available", "INSTALL_HINT", "track"]
