"""
OpenCut VACE All-in-One Video Compositing (Q1.1)

Wan2.1-VACE compositing: V2V, MV2V, R2V (mask+prompt editing).

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install wan@git+https://github.com/Wan-Video/Wan2.1  # Apache-2.0"


@dataclass
class VACEResult:
    output: str = ""
    mode: str = ""
    task: str = ""
    duration_seconds: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "task", "duration_seconds", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_wan_available() -> bool:
    return _try_import("wan") is not None


def compose(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_wan_available():
        raise RuntimeError(f"wan not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when wan is installed locally.")


__all__ = ["VACEResult", "check_wan_available", "INSTALL_HINT", "compose"]
