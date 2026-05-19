"""
OpenCut Step-Video T2V + Ti2V HPC Video Generation (R3.2-R3.3)

30B bilingual EN+ZH T2V. 204 frames at 30fps. Linux-only CUDA kernels.

Licence: See ROADMAP.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install stepvideo  # MIT, requires Linux + sm_80+ GPU"


@dataclass
class StepVideoResult:
    output: str = ""
    mode: str = ""
    model: str = ""
    duration_seconds: float = 0.0
    fps: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "model", "duration_seconds", "fps", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_stepvideo_available() -> bool:
    return _try_import("stepvideo") is not None


def generate_t2v(**kwargs):
    """Entry point. Raises RuntimeError when deps not installed."""
    if not check_stepvideo_available():
        raise RuntimeError(f"stepvideo not installed. {INSTALL_HINT}")
    raise NotImplementedError("Full implementation ships when stepvideo is installed locally.")


__all__ = ["StepVideoResult", "check_stepvideo_available", "INSTALL_HINT", "generate_t2v"]
