"""
OpenCut EzAudio Text-to-Audio/Foley (R1.1-R1.3)

DiT-based text-to-sound-effects generation, audio inpainting, ControlNet.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install ezaudio  # MIT, 44.1 kHz output"


@dataclass
class EzAudioResult:
    output: str = ""
    prompt: str = ""
    duration_seconds: float = 0.0
    sample_rate: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "prompt", "duration_seconds", "sample_rate", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_ezaudio_available() -> bool:
    return _try_import("ezaudio") is not None


def generate(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_ezaudio_available():
        raise RuntimeError(f"ezaudio not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when ezaudio is installed locally.")


__all__ = ["EzAudioResult", "check_ezaudio_available", "INSTALL_HINT", "generate"]
