"""
OpenCut LTX Video Generation v1.28.0 — STUB

Text-to-video and image-to-video via LTX-Video (LTX-0.9.8 and LTX-2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import


@dataclass
class LTXResult:
    output: str = ""
    audio_output: str = ""
    prompt: str = ""
    duration: float = 0.0
    version: str = "ltx-0.9.8"
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "audio_output", "prompt", "duration", "version", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_ltx_available() -> bool:
    return _try_import("ltx_video") is not None or (
        _try_import("diffusers") is not None and _try_import("torch") is not None
    )


def check_ltx2_available() -> bool:
    return check_ltx_available()


INSTALL_HINT = "pip install ltx-video torch diffusers transformers"


def list_backends() -> List[Dict]:
    return [
        {"name": "ltx-0.9.8", "available": check_ltx_available()},
        {"name": "ltx-2", "available": check_ltx2_available()},
    ]


def generate(
    prompt: str,
    duration: float = 5.0,
    resolution: str = "720p",
    audio_prompt: Optional[str] = None,
    version: str = "ltx-0.9.8",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> LTXResult:
    if not check_ltx_available():
        raise RuntimeError(f"LTX-Video is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("LTX wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["LTXResult", "check_ltx_available", "check_ltx2_available", "INSTALL_HINT",
           "list_backends", "generate"]
