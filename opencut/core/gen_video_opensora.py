"""
OpenCut OpenSora Video Generation v1.28.0 — STUB

Text-to-video generation via Open-Sora (open-source Sora replica).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class OpenSoraResult:
    output: str = ""
    prompt: str = ""
    duration: float = 0.0
    resolution: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "prompt", "duration", "resolution", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_opensora_available() -> bool:
    return _try_import("opensora") is not None or (
        _try_import("torch") is not None and _try_import("transformers") is not None
    )


INSTALL_HINT = "pip install opensora torch transformers diffusers"


def generate(
    prompt: str,
    duration: float = 4.0,
    resolution: str = "720p",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> OpenSoraResult:
    if not check_opensora_available():
        raise RuntimeError(f"OpenSora is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("OpenSora wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["OpenSoraResult", "check_opensora_available", "INSTALL_HINT", "generate"]
