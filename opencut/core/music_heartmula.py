"""
OpenCut HeartMuLa Music Generation (S3.4)

Lyric-aligned generation with word-level timing for music-video sync.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install transformers>=4.45  # HeartMuLa weights ~5 GB"


@dataclass
class HeartMuLaResult:
    output: str = ""
    duration_seconds: float = 0.0
    genre: str = ""
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "duration_seconds", "genre", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_transformers_available() -> bool:
    return _try_import("transformers") is not None


def generate(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_transformers_available():
        raise RuntimeError(f"transformers not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when transformers is installed locally.")


__all__ = ["HeartMuLaResult", "check_transformers_available", "INSTALL_HINT", "generate"]
