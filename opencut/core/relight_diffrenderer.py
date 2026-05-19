"""
OpenCut DiffusionRenderer Physically Grounded Relight (S1.3)

NVIDIA inverse+forward rendering: G-buffers + HDR env-map relight.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install diffusionrenderer  # Apache-2.0, ~12 GB weights"


@dataclass
class DiffRendererResult:
    output: str = ""
    gbuffer_path: str = ""
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "gbuffer_path", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_diffusionrenderer_available() -> bool:
    return _try_import("diffusionrenderer") is not None


def relight(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_diffusionrenderer_available():
        raise RuntimeError(f"diffusionrenderer not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when diffusionrenderer is installed locally.")


__all__ = ["DiffRendererResult", "check_diffusionrenderer_available", "INSTALL_HINT", "relight"]
