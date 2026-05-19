"""
OpenCut Mochi-1 Consumer T2V (R3.1)

10B model with best open motion fidelity. CPU offload for 16 GB VRAM.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install diffusers>=0.31  # MochiPipeline, Apache-2.0"


@dataclass
class MochiResult:
    output: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    fps: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "model", "duration_seconds", "fps", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_diffusers_available() -> bool:
    return _try_import("diffusers") is not None


def generate(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_diffusers_available():
        raise RuntimeError(f"diffusers not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when diffusers is installed locally.")


__all__ = ["MochiResult", "check_diffusers_available", "INSTALL_HINT", "generate"]
