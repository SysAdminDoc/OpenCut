"""
OpenCut OmniGen2 Multi-Reference Image Generation (Q3.1)

Combine 2-4 reference images into one coherent output.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install omnigen2  # Apache-2.0, ~13 GB weights"


@dataclass
class OmniGen2Result:
    output: str = ""
    width: int = 0
    height: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "width", "height", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_omnigen2_available() -> bool:
    return _try_import("omnigen2") is not None


def generate(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_omnigen2_available():
        raise RuntimeError(f"omnigen2 not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when omnigen2 is installed locally.")


__all__ = ["OmniGen2Result", "check_omnigen2_available", "INSTALL_HINT", "generate"]
