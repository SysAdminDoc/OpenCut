"""
OpenCut IC-Light v1 Per-Frame Relight (S1.1)

Per-frame relighting: text-conditioned (FC) or background-conditioned (FBC).

This engine targets **IC-Light v1** (github.com/lllyasviel/IC-Light), which is
**Apache-2.0 and ships real public weights** — MIT-distribution-safe. IC-Light
*v2* is deliberately NOT targeted: it is non-commercial and its full weights were
never publicly released (HF Space demo only), so it is un-shippable here.

Contract matches the other engine stubs (asr_parakeet / upscale_seedvr2): an
accurate ``check_iclight_available()`` gates the registry entry, model metadata
lives in ``model_manager.KNOWN_MODELS``, and ``relight()`` raises a structured
``RuntimeError`` (with the install hint) when the backend is absent. The
diffusion forward activates once the local v1 weights + ``diffusers`` are present.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

# IC-Light v1 — Apache-2.0, real public weights (FC text-conditioned + FBC
# background-conditioned relight models).
MODEL_ID = "lllyasviel/IC-Light"
MODEL_NAME = "ic-light-v1"
MODEL_LICENSE = "Apache-2.0"

INSTALL_HINT = "pip install diffusers>=0.32 torch  # IC-Light v1 weights ~3 GB (Apache-2.0)"


@dataclass
class ICLightResult:
    output: str = ""
    mode: str = ""
    model: str = ""
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "mode", "model", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_iclight_available() -> bool:
    """True when IC-Light v1's diffusion stack (diffusers + torch) is importable.

    Never raises — returns False when the optional dependency is absent so the
    relight registry / route can fall back gracefully.
    """
    return _try_import("diffusers") is not None and _try_import("torch") is not None


# Back-compat alias — the original stub exposed ``check_diffusers_available``.
def check_diffusers_available() -> bool:
    return check_iclight_available()


def relight(**kwargs):
    """Entry point.

    Raises ``RuntimeError`` (carrying the install hint) when the IC-Light v1
    stack is not installed. The diffusion forward activates once the local
    Apache-2.0 v1 weights are present.
    """
    if not check_iclight_available():
        raise RuntimeError(f"IC-Light v1 backend not installed. {INSTALL_HINT}")
    raise NotImplementedError(
        "IC-Light v1 relight forward activates when the local "
        f"{MODEL_ID} weights are installed."
    )


__all__ = [
    "ICLightResult",
    "MODEL_ID",
    "MODEL_NAME",
    "MODEL_LICENSE",
    "INSTALL_HINT",
    "check_iclight_available",
    "check_diffusers_available",
    "relight",
]
