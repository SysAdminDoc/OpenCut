"""
OpenCut MaskGCT Zero-Shot Parallel TTS (Q2.2)

Fully non-autoregressive TTS from Amphion. Fastest inference, 100K hrs training.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install amphion  # MIT, MaskGCT weights ~5 GB"


@dataclass
class MaskGCTResult:
    output: str = ""
    voice: str = ""
    duration_seconds: float = 0.0
    sample_rate: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "duration_seconds", "sample_rate", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_amphion_available() -> bool:
    return _try_import("amphion") is not None


def synthesize(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_amphion_available():
        raise RuntimeError(f"amphion not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when amphion is installed locally.")


__all__ = ["MaskGCTResult", "check_amphion_available", "INSTALL_HINT", "synthesize"]
