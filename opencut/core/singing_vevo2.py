"""
OpenCut Vevo2 Singing Voice Conversion v1.28.0 — STUB

AI singing voice conversion via Vevo2 (part of Amphion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class Vevo2Result:
    output: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_vevo2_available() -> bool:
    return _try_import("amphion") is not None or (
        _try_import("transformers") is not None and _try_import("torch") is not None
    )


INSTALL_HINT = "pip install amphion torch torchaudio (Vevo2 is part of Amphion)"


def convert(
    audio_path: str,
    reference_path: str,
    pitch_shift: int = 0,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> Vevo2Result:
    if not check_vevo2_available():
        raise RuntimeError(f"Vevo2/Amphion is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("Vevo2 wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["Vevo2Result", "check_vevo2_available", "INSTALL_HINT", "convert"]
