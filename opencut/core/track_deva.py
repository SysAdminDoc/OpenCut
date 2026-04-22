"""
OpenCut DEVA Tracking v1.28.0 — STUB

Decoupled video segmentation with any detector via DEVA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class DEVAResult:
    output_mask_path: str = ""
    tracked_objects: List[str] = field(default_factory=list)
    frames: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output_mask_path", "tracked_objects", "frames", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_deva_available() -> bool:
    return _try_import("deva") is not None or (
        _try_import("torch") is not None and _try_import("transformers") is not None
    )


INSTALL_HINT = "pip install DEVA torch torchvision transformers groundingdino"


def track(
    video_path: str,
    text_prompt: str,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> DEVAResult:
    if not check_deva_available():
        raise RuntimeError(f"DEVA is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("DEVA tracking wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["DEVAResult", "check_deva_available", "INSTALL_HINT", "track"]
