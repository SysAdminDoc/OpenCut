"""
OpenCut EchoMimic Lipsync v1.28.0 — STUB

Portrait and half-body audio-driven lipsync via EchoMimic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class EchoMimicResult:
    output: str = ""
    mode: str = "portrait"
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_echomimic_available() -> bool:
    return _try_import("echomimic") is not None or (
        _try_import("torch") is not None and _try_import("diffusers") is not None
    )


INSTALL_HINT = "pip install echomimic torch diffusers accelerate"


def animate(
    image_path: str,
    audio_path: str,
    mode: str = "portrait",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> EchoMimicResult:
    if not check_echomimic_available():
        raise RuntimeError(f"EchoMimic is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("EchoMimic wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["EchoMimicResult", "check_echomimic_available", "INSTALL_HINT", "animate"]
