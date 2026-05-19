"""
OpenCut SeedVR2 One-Step Diffusion VSR (S2.1)

Single-step VSR at 10x FlashVSR throughput. 3B + 7B models.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install diffusers  # SeedVR2 3B (~6 GB) + 7B (~14 GB)"


@dataclass
class SeedVR2Result:
    output: str = ""
    model: str = ""
    scale: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "model", "scale", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_diffusers_available() -> bool:
    return _try_import("diffusers") is not None


def upscale(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_diffusers_available():
        raise RuntimeError(f"diffusers not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when diffusers is installed locally.")


__all__ = ["SeedVR2Result", "check_diffusers_available", "INSTALL_HINT", "upscale"]
