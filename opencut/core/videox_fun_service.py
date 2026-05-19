"""
OpenCut VideoX-Fun Camera Control + Structural Control I2V (R2.2-R2.3)

Pan/zoom/orbit trajectories + Canny/Depth/Pose/MLSD structural control.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install videox-fun  # Apache-2.0"


@dataclass
class VideoXFunResult:
    output: str = ""
    control_type: str = ""
    duration_seconds: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "control_type", "duration_seconds", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_videox_fun_available() -> bool:
    return _try_import("videox_fun") is not None


def generate(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_videox_fun_available():
        raise RuntimeError(f"videox_fun not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when videox_fun is installed locally.")


__all__ = ["VideoXFunResult", "check_videox_fun_available", "INSTALL_HINT", "generate"]
