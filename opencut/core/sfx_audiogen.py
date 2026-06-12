"""
OpenCut AudioGen SFX v1.28.0 — STUB

Text-to-sound-effects via Meta AudioGen (audiocraft).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class AudioGenResult:
    output: str = ""
    prompt: str = ""
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "prompt", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_audiogen_available() -> bool:
    return _try_import("audiocraft") is not None


INSTALL_HINT = (
    "pip install audiocraft\n"
    "Weights (CC-BY-NC): https://github.com/facebookresearch/audiocraft "
    "-- run model.download() after pip install"
)


def generate(
    prompt: str,
    duration: float = 3.0,
    model_size: str = "medium",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> AudioGenResult:
    if not check_audiogen_available():
        raise RuntimeError(f"AudioGen (audiocraft) is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("AudioGen wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["AudioGenResult", "check_audiogen_available", "INSTALL_HINT", "generate"]
