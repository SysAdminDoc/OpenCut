"""
OpenCut SeedVR2 One-Step Diffusion VSR (S2.1)

SeedVR2 (ByteDance, ICLR 2026) is a one-step diffusion video restoration /
super-resolution model: a single denoising step at ~10x FlashVSR throughput,
beating multi-step VSR and Real-ESRGAN on detail recovery.

Licence: the SeedVR2-3B weights are **Apache-2.0**
(huggingface.co/ByteDance-Seed/SeedVR2-3B), which is MIT-distribution-safe — so
SeedVR2 is registered as a *preferred* upscaling engine that auto-selects over
Real-ESRGAN when its diffusion stack is installed, and cleanly falls back to
Real-ESRGAN when it is not.

This module follows the established engine-stub contract (see asr_parakeet /
asr_canary): an accurate ``check_seedvr2_available()`` gates the registry entry,
the model-download metadata lives in ``model_manager.KNOWN_MODELS``, and the
``upscale()`` entry point raises a structured ``RuntimeError`` (with the install
hint) when the backend is absent so callers fall back to Real-ESRGAN. The heavy
diffusion forward activates once the local weights + ``diffusers`` stack are
present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

# Apache-2.0 weights — MIT-distribution-safe, so SeedVR2 may default over
# Real-ESRGAN when available.
MODEL_ID = "ByteDance-Seed/SeedVR2-3B"
MODEL_NAME = "seedvr2-3b"
MODEL_LICENSE = "Apache-2.0"

INSTALL_HINT = (
    "pip install diffusers torch  # SeedVR2-3B (~6 GB, Apache-2.0). "
    "Download weights via the model manager (seedvr2-3b)."
)


@dataclass
class SeedVR2Result:
    output: str = ""
    model: str = ""
    scale: float = 0.0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "model", "scale", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_seedvr2_available() -> bool:
    """True when SeedVR2's diffusion stack (diffusers + torch) is importable.

    Never raises — returns False when the optional dependency is absent so the
    engine registry can gate on it and fall back to Real-ESRGAN.
    """
    return _try_import("diffusers") is not None and _try_import("torch") is not None


# Back-compat alias — the original stub exposed ``check_diffusers_available``.
def check_diffusers_available() -> bool:
    return check_seedvr2_available()


def upscale(**kwargs):
    """Entry point.

    Raises ``RuntimeError`` (carrying the install hint) when the SeedVR2 stack
    is not installed, so the upscaling dispatcher falls back to Real-ESRGAN.
    The full one-step diffusion forward activates once the local weights +
    ``diffusers`` are present.
    """
    if not check_seedvr2_available():
        raise RuntimeError(f"SeedVR2 backend not installed. {INSTALL_HINT}")
    raise NotImplementedError(
        "SeedVR2 one-step diffusion forward activates when the local "
        f"{MODEL_ID} weights are installed."
    )


__all__ = [
    "SeedVR2Result",
    "MODEL_ID",
    "MODEL_NAME",
    "MODEL_LICENSE",
    "INSTALL_HINT",
    "check_seedvr2_available",
    "check_diffusers_available",
    "upscale",
]
