"""
OpenCut DiffBIR Restoration v1.28.0 — STUB

Blind image/video restoration via DiffBIR diffusion model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class DiffBIRResult:
    output: str = ""
    tile_size: int = 512
    fast_mode: bool = False
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "tile_size", "fast_mode", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_diffbir_available() -> bool:
    return _try_import("diffbir") is not None or (
        _try_import("diffusers") is not None and _try_import("torch") is not None
    )


INSTALL_HINT = "pip install diffbir diffusers torch"


def restore(
    video_path: str,
    tile_size: int = 512,
    fast_mode: bool = False,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> DiffBIRResult:
    if not check_diffbir_available():
        raise RuntimeError(f"DiffBIR is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("DiffBIR wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["DiffBIRResult", "check_diffbir_available", "INSTALL_HINT", "restore"]
