"""
OpenCut Wan2.1 VACE v1.28.0 — Tier 3 strategic stub

Wan2.1 VACE video editing (background change, re-light, modify action).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import

INSTALL_HINT = (
    "pip install wan torch; confirm model-weight license terms before enabling"
)


def check_wan_vace_available() -> bool:
    return _try_import("wan") is not None or _try_import("torch") is not None


@dataclass
class WanVACEResult:
    output: str = ""
    edit_type: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "edit_type", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def edit(
    video_path: str,
    prompt: str,
    edit_type: str = "background",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> WanVACEResult:
    raise NotImplementedError(
        "Wan2.1 VACE wiring is not implemented yet. Track the live ROADMAP.md entry."
    )


__all__ = ["check_wan_vace_available", "INSTALL_HINT", "WanVACEResult", "edit"]
