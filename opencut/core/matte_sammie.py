"""
OpenCut Sammie-Roto-2 Rotoscoping (Wave H2.3, v1.25.0) — **STUB**

AI-assisted temporal rotoscoping via VideoMaMa segmentation + user
in/out point markers. Complements the existing BiRefNet still-frame
matte by propagating masks through a clip.
Source: https://github.com/Zarxrax/Sammie-Roto-2

Stubbed behind ``check_sammie_available()`` in v1.25.0. Full wiring
lands when Sammie-Roto-2 ships a headless Python entry point (the
current CLI is GUI-centric).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


@dataclass
class SammieResult:
    output: str = ""
    mask_path: str = ""
    masked_frames: int = 0
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mask_path", "masked_frames", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_sammie_available() -> bool:
    """Requires torch + sammie (headless entry point not yet shipped)."""
    if _try_import("torch") is None:
        return False
    if _try_import("sammie_roto") is None:
        return False
    return True


INSTALL_HINT = (
    "pip install torch sammie-roto "
    "(track https://github.com/Zarxrax/Sammie-Roto-2 for the headless "
    "Python entry point)"
)


def rotoscope(
    video_path: str,
    markers: List[Dict[str, Any]],
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> SammieResult:
    """Entry-point stub. Raises RuntimeError until backend is installed.

    ``markers`` is a list of ``{frame, point, label}`` dicts. Fewer
    than 2 markers → raises ``ValueError`` (no way to disambiguate).
    """
    if not isinstance(markers, list) or len(markers) < 2:
        raise ValueError("at least 2 markers (in + out) required")
    if not check_sammie_available():
        raise RuntimeError(
            "Sammie-Roto-2 is not installed. " + INSTALL_HINT
        )
    raise NotImplementedError(
        "Sammie-Roto-2 wiring ships in a future release."
    )


__all__ = [
    "SammieResult",
    "check_sammie_available",
    "INSTALL_HINT",
    "rotoscope",
]
