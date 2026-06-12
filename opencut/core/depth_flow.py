"""
OpenCut DepthFlow v1.28.0 — STUB

2.5D depth-of-field parallax motion from a still image via DepthFlow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class DepthFlowResult:
    output: str = ""
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_depthflow_available() -> bool:
    return _try_import("DepthFlow") is not None


INSTALL_HINT = "pip install DepthFlow (requires ModernGL; on headless Linux: apt install xvfb)"


def generate(
    image_path: str,
    depth_path: Optional[str] = None,
    duration: float = 5.0,
    parallax_intensity: float = 0.5,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> DepthFlowResult:
    if not check_depthflow_available():
        raise RuntimeError(f"DepthFlow is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("DepthFlow wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["DepthFlowResult", "check_depthflow_available", "INSTALL_HINT", "generate"]
