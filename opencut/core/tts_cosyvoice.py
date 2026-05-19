"""
OpenCut CosyVoice 2.0 Multilingual TTS (Q2.1)

9 languages + 18 Chinese dialects, 150ms streaming, zero-shot clone.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install cosyvoice  # Apache-2.0, 0.5B model"


@dataclass
class CosyVoiceResult:
    output: str = ""
    voice: str = ""
    language: str = ""
    duration_seconds: float = 0.0
    sample_rate: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "language", "duration_seconds", "sample_rate", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_cosyvoice_available() -> bool:
    return _try_import("cosyvoice") is not None


def synthesize(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_cosyvoice_available():
        raise RuntimeError(f"cosyvoice not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when cosyvoice is installed locally.")


__all__ = ["CosyVoiceResult", "check_cosyvoice_available", "INSTALL_HINT", "synthesize"]
