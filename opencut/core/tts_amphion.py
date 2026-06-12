"""
OpenCut Amphion TTS v1.28.0 — STUB

Zero-shot TTS via MaskGCT (Amphion). Includes Vevo2 singing voice conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class AmphionResult:
    output: str = ""
    model: str = "maskgct"
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "model", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_amphion_available() -> bool:
    return _try_import("amphion") is not None or (
        _try_import("transformers") is not None and _try_import("torch") is not None
    )


INSTALL_HINT = "pip install amphion torch torchaudio"


def list_models() -> List[str]:
    return ["maskgct", "vevo2"] if check_amphion_available() else []


def synthesize(
    text: str,
    reference_audio: Optional[str] = None,
    model: str = "maskgct",
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> AmphionResult:
    if not check_amphion_available():
        raise RuntimeError(f"Amphion is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("Amphion TTS wiring is not implemented yet. Track the live ROADMAP.md entry.")


__all__ = ["AmphionResult", "check_amphion_available", "INSTALL_HINT", "list_models", "synthesize"]
