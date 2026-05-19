"""
OpenCut InternVL3 Alternative VLM (S3.2)

Parallel VLM from OpenGVLab. Variable Visual Position Encoding.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install transformers>=4.45  # InternVL3-8B ~16 GB"


@dataclass
class InternVL3Result:
    query: str = ""
    response: str = ""
    structured_data: List = field(default_factory=list)
    model: str = ""
    processing_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("query", "response", "structured_data", "model", "processing_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_transformers_available() -> bool:
    return _try_import("transformers") is not None


def analyze(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_transformers_available():
        raise RuntimeError(f"transformers not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when transformers is installed locally.")


__all__ = ["InternVL3Result", "check_transformers_available", "INSTALL_HINT", "analyze"]
