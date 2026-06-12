"""
OpenCut SEA-RAFT Optical Flow v1.28.0 — STUB

State-of-the-art optical flow estimation via SEA-RAFT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class SEARaftResult:
    output_flow_path: str = ""
    backend: str = "searaft"
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output_flow_path", "backend", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_searaft_available() -> bool:
    return _try_import("searaft") is not None or _try_import("torch") is not None


INSTALL_HINT = "pip install sea-raft torch"


def compute_flow(
    video_path: str,
    output: Optional[str] = None,
    max_resolution: int = 1080,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> SEARaftResult:
    if not check_searaft_available():
        raise RuntimeError(f"SEA-RAFT is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("SEA-RAFT wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["SEARaftResult", "check_searaft_available", "INSTALL_HINT", "compute_flow"]
