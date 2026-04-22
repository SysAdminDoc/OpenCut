"""
OpenCut Motion Deblur v1.28.0 — STUB

Motion deblur via NAFNet or MIMO-UNet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


@dataclass
class DeblurResult:
    output: str = ""
    backend: str = "nafnet"
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "backend", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_nafnet_available() -> bool:
    return _try_import("nafnet") is not None or _try_import("basicsr") is not None


def check_mimounet_available() -> bool:
    return _try_import("mimounet") is not None or _try_import("torch") is not None


def check_deblur_motion_available() -> bool:
    return check_nafnet_available() or check_mimounet_available()


INSTALL_HINT = "pip install basicsr torch (NAFNet) or pip install torch (MIMO-UNet)"


def list_backends() -> List[Dict]:
    return [
        {"name": "nafnet", "available": check_nafnet_available()},
        {"name": "mimounet", "available": check_mimounet_available()},
    ]


def deblur(
    video_path: str,
    backend: str = "auto",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> DeblurResult:
    if not check_deblur_motion_available():
        raise RuntimeError(f"No deblur backend available. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("Motion deblur wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["DeblurResult", "check_deblur_motion_available", "INSTALL_HINT",
           "list_backends", "deblur"]
