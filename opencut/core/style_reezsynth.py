"""
OpenCut ReEzSynth Style Transfer (Wave H2.5, v1.25.0) — **STUB**

Flicker-free Ebsynth successor using bidirectional nearest-neighbour-
field blending + temporal NNF propagation. Drop-in replacement for
per-frame style transfer with much higher temporal stability.
Source: https://github.com/FuouM/ReEzSynth

Stubbed behind ``check_reezsynth_available()`` in v1.25.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


@dataclass
class ReEzSynthResult:
    output: str = ""
    keyframe_count: int = 0
    propagation_cost: float = 0.0
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "keyframe_count", "propagation_cost",
                "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_reezsynth_available() -> bool:
    return _try_import("reezsynth") is not None or _try_import("ezsynth") is not None


INSTALL_HINT = "pip install reezsynth (track https://github.com/FuouM/ReEzSynth)"


def stylize(
    video_path: str,
    style_images: List[Dict[str, Any]],
    output: Optional[str] = None,
    blend_window: int = 5,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> ReEzSynthResult:
    """Entry-point stub. Raises RuntimeError until backend is installed.

    ``style_images`` is ``[{frame: int, path: str}, ...]``.  At least
    one keyframe is required.
    """
    if not isinstance(style_images, list) or not style_images:
        raise ValueError("at least one style keyframe is required")
    if not check_reezsynth_available():
        raise RuntimeError(
            "ReEzSynth is not installed. " + INSTALL_HINT
        )
    raise NotImplementedError("ReEzSynth wiring ships in a future release.")


__all__ = [
    "ReEzSynthResult",
    "check_reezsynth_available",
    "INSTALL_HINT",
    "stylize",
]
