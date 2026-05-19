"""
OpenCut SkyReels V3 Talking Avatar (Q3.3)

19B A2V model: portrait + audio -> lifelike talking avatar, up to 200s.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install skyreels  # Skywork Community Licence"


@dataclass
class SkyReels3Result:
    output: str = ""
    mode: str = ""
    duration_seconds: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "duration_seconds", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_skyreels_available() -> bool:
    return _try_import("skyreels") is not None


def generate_avatar(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_skyreels_available():
        raise RuntimeError(f"skyreels not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when skyreels is installed locally.")


__all__ = ["SkyReels3Result", "check_skyreels_available", "INSTALL_HINT", "generate_avatar"]
