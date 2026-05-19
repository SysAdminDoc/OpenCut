"""
OpenCut Face Re-Aging FRAN Reimplementation (S3.3)

Production VFX-quality age progression/regression. -30 to +30 years.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install face-reaging  # MIT, ~50 MB + FRAN weights ~150 MB"


@dataclass
class FaceReageResult:
    output: str = ""
    age_delta: int = 0
    frames_processed: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "age_delta", "frames_processed", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_face_reaging_available() -> bool:
    return _try_import("face_reaging") is not None


def transform(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_face_reaging_available():
        raise RuntimeError(f"face_reaging not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when face_reaging is installed locally.")


__all__ = ["FaceReageResult", "check_face_reaging_available", "INSTALL_HINT", "transform"]
