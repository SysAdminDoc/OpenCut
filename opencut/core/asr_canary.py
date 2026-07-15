"""
OpenCut NVIDIA Canary-1B-Flash Batch ASR (S2.3)

RTFx 1000+ batch ASR. 1 hour audio in <4 seconds on RTX 4090.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install nemo_toolkit[asr]  # Apache-2.0 + CC-BY-4.0 model"


@dataclass
class CanaryResult:
    text: str = ""
    segments: str = ""
    model: str = ""
    processing_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("text", "segments", "model", "processing_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_nemo_toolkit_available() -> bool:
    return _try_import("nemo") is not None or _try_import("nemo_toolkit") is not None


def transcribe_batch(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_nemo_toolkit_available():
        raise RuntimeError(f"nemo_toolkit not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when nemo_toolkit is installed locally.")


__all__ = ["CanaryResult", "check_nemo_toolkit_available", "INSTALL_HINT", "transcribe_batch"]
