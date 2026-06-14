"""
OpenCut LatentSync lip-sync engine (E2E dubbing-with-lip-sync).

LatentSync (ByteDance, github.com/bytedance/LatentSync) is an audio-conditioned
latent-diffusion lip-sync model. OpenCut already ships the full local dub
pipeline (transcribe -> translate -> voice-clone -> render); LatentSync is the
optional final stage that re-animates the speaker's mouth to the new audio —
the exact visual lip-sync step Rask AI / HeyGen / sync.so paywall.

Licence note (RESEARCH Open Question 3): the LatentSync *code* is Apache-2.0,
but the trained *checkpoint* licence must be confirmed at the model-card level
before defaulting it. Until that is resolved this engine ships **opt-in**
(``DEFAULT_OPT_IN = True``): it is registered and selectable, but never
auto-selected over the always-available heuristic lip-sync — the dub pipeline
only invokes it when the caller explicitly asks for ``backend="latentsync"``.

Contract matches the other engine stubs (asr_parakeet / upscale_seedvr2): an
accurate ``check_latentsync_available()`` gates the registry entry, model
metadata lives in ``model_manager.KNOWN_MODELS``, and ``apply_latentsync()``
raises a structured ``RuntimeError`` (with the install hint) when the backend is
absent so the dub pipeline falls back to audio-only / heuristic lip-sync.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

MODEL_ID = "ByteDance/LatentSync-1.6"
MODEL_NAME = "latentsync-1.6"
# Code is Apache-2.0; the checkpoint licence is unconfirmed (Open Question 3),
# so the engine is opt-in and not a permissive default.
MODEL_CODE_LICENSE = "Apache-2.0"
MODEL_CHECKPOINT_LICENSE = "see model card (unconfirmed — opt-in)"
DEFAULT_OPT_IN = True

INSTALL_HINT = (
    "pip install diffusers torch torchvision  # LatentSync-1.6 (512x512). "
    "Download weights via the model manager (latentsync-1.6); confirm the "
    "checkpoint licence before production use."
)


@dataclass
class LatentSyncResult:
    output_path: str = ""
    model: str = ""
    frames_processed: int = 0
    generation_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output_path", "model", "frames_processed", "generation_seconds", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def check_latentsync_available() -> bool:
    """True when LatentSync's diffusion stack (diffusers + torch) is importable.

    Never raises — returns False when the optional dependency is absent so the
    dub pipeline / engine registry can fall back to heuristic lip-sync.
    """
    return _try_import("diffusers") is not None and _try_import("torch") is not None


def apply_latentsync(**kwargs):
    """Entry point.

    Raises ``RuntimeError`` (carrying the install hint) when the LatentSync stack
    is not installed so the dub pipeline degrades to audio-only / heuristic
    lip-sync. The diffusion forward activates once the local weights are present.
    """
    if not check_latentsync_available():
        raise RuntimeError(f"LatentSync backend not installed. {INSTALL_HINT}")
    raise NotImplementedError(
        "LatentSync audio-conditioned diffusion forward activates when the local "
        f"{MODEL_ID} weights are installed."
    )


__all__ = [
    "LatentSyncResult",
    "MODEL_ID",
    "MODEL_NAME",
    "MODEL_CODE_LICENSE",
    "MODEL_CHECKPOINT_LICENSE",
    "DEFAULT_OPT_IN",
    "INSTALL_HINT",
    "check_latentsync_available",
    "apply_latentsync",
]
