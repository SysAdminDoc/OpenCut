"""
OpenCut TokenFlow Style Transfer v1.28.0 — STUB

Temporally consistent video style transfer via TokenFlow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import _try_import


@dataclass
class TokenFlowResult:
    output: str = ""
    style_prompt: str = ""
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "style_prompt", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_tokenflow_available() -> bool:
    return _try_import("tokenflow") is not None or (
        _try_import("diffusers") is not None and _try_import("torch") is not None
    )


INSTALL_HINT = "pip install tokenflow diffusers torch transformers"


def restyle(
    video_path: str,
    style_prompt: str,
    strength: float = 0.7,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> TokenFlowResult:
    if not check_tokenflow_available():
        raise RuntimeError(f"TokenFlow is not installed. Install with:\n    {INSTALL_HINT}")
    raise NotImplementedError("TokenFlow wiring ships in v1.28.x. Track ROADMAP-NEXT.md Wave K.")


__all__ = ["TokenFlowResult", "check_tokenflow_available", "INSTALL_HINT", "restyle"]
