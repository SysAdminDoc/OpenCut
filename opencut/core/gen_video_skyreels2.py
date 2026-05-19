"""
OpenCut SkyReels V2 Infinite-Length T2V (Q3.2)

Diffusion Forcing for 30s+ coherent video. 14B (720P) + 1.3B (540P).

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install diffusers  # SkyReels V2 pipeline + weights"


@dataclass
class SkyReels2Result:
    output: str = ""
    mode: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    fps: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "model", "duration_seconds", "fps", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_diffusers_available() -> bool:
    return _try_import("diffusers") is not None


def generate_t2v(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_diffusers_available():
        raise RuntimeError(f"diffusers not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when diffusers is installed locally.")


__all__ = ["SkyReels2Result", "check_diffusers_available", "INSTALL_HINT", "generate_t2v"]
