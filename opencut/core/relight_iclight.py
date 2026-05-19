"""
OpenCut IC-Light V2 Per-Frame Relight (S1.1)

Per-frame relighting: text-conditioned or background-conditioned.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install diffusers>=0.32  # IC-Light V2 LoRA weights ~3 GB"


@dataclass
class ICLightResult:
    output: str = ""
    mode: str = ""
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_diffusers_available() -> bool:
    return _try_import("diffusers") is not None


def relight(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_diffusers_available():
        raise RuntimeError(f"diffusers not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when diffusers is installed locally.")


__all__ = ["ICLightResult", "check_diffusers_available", "INSTALL_HINT", "relight"]
