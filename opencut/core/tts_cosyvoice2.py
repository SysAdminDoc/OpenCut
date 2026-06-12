"""
OpenCut CosyVoice2 TTS v1.28.0 — STUB

Streaming zero-latency TTS with voice cloning via CosyVoice2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class CosyVoice2Result:
    output: str = ""
    voice: str = ""
    latency_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "voice", "latency_ms", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_cosyvoice2_available() -> bool:
    return _try_import("cosyvoice") is not None


INSTALL_HINT = "pip install cosyvoice2 torch"


def list_voices() -> List[str]:
    return []


def synthesize(
    text: str,
    reference_audio: Optional[str] = None,
    voice: Optional[str] = None,
    streaming: bool = False,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> CosyVoice2Result:
    if not check_cosyvoice2_available():
        raise RuntimeError(f"CosyVoice2 is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("CosyVoice2 wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["CosyVoice2Result", "check_cosyvoice2_available", "INSTALL_HINT", "list_voices", "synthesize"]
