"""
OpenCut VidMuse Video-to-Music (Wave H2.6, v1.25.0) — **STUB**

Video-to-music generation using long-short-term modelling (CVPR'25).
Complements the existing MusicGen text-to-music pipeline — VidMuse
takes the video itself as conditioning signal.
Source: https://vidmuse.github.io/

Stubbed behind ``check_vidmuse_available()`` in v1.25.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class VidMuseResult:
    output: str = ""
    bpm: float = 0.0
    duration: float = 0.0
    mood: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "bpm", "duration", "mood", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_vidmuse_available() -> bool:
    return _try_import("vidmuse") is not None and _try_import("torch") is not None


INSTALL_HINT = "pip install vidmuse torch (track https://vidmuse.github.io/)"


def generate(
    video_path: str,
    duration: float = 30.0,
    output: Optional[str] = None,
    style_hint: str = "",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> VidMuseResult:
    """Entry-point stub. Raises RuntimeError until backend is installed."""
    if not check_vidmuse_available():
        raise RuntimeError("VidMuse is not installed. " + INSTALL_HINT)
    raise NotImplementedError("VidMuse wiring ships in a future release.")


__all__ = [
    "VidMuseResult",
    "check_vidmuse_available",
    "INSTALL_HINT",
    "generate",
]
