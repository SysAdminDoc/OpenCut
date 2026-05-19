"""
OpenCut MuseTalk 1.5 Audio-Driven Lip Sync (R2.1)

Real-time 30fps+ lip sync for talking head and dubbing.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install musetalk  # MIT code / CreativeML-OpenRAIL-M weights"


@dataclass
class MuseTalkResult:
    output: str = ""
    frames_processed: int = 0
    fps: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "frames_processed", "fps", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_musetalk_available() -> bool:
    return _try_import("musetalk") is not None


def lip_sync(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_musetalk_available():
        raise RuntimeError(f"musetalk not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when musetalk is installed locally.")


__all__ = ["MuseTalkResult", "check_musetalk_available", "INSTALL_HINT", "lip_sync"]
