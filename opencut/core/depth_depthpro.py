"""
OpenCut Depth Pro v1.28.0 — STUB

Metric-depth estimation from single image/frame via Depth Pro (Apple) or Depth Anything V2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


@dataclass
class DepthProResult:
    output_depth_path: str = ""
    scale_factor: float = 1.0
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output_depth_path", "scale_factor", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_depthpro_available() -> bool:
    return _try_import("depth_pro") is not None


INSTALL_HINT = "pip install depth-pro torch"


def list_backends() -> List[Dict]:
    return [
        {"name": "depthpro", "available": check_depthpro_available()},
        {"name": "depth_anything", "available": (
            _try_import("transformers") is not None and _try_import("torch") is not None
        )},
    ]


def estimate(
    video_path: str,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> DepthProResult:
    if not check_depthpro_available():
        raise RuntimeError(f"Depth Pro is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("Depth Pro wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["DepthProResult", "check_depthpro_available", "INSTALL_HINT", "list_backends", "estimate"]
