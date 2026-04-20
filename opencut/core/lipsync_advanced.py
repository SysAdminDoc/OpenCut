"""
OpenCut Advanced Lip-Sync (Wave H3.3, v1.25.0) — **STUB**

GaussianHeadTalk (WACV'26) + FantasyTalking2 (AAAI'26) alternatives to
the planned LatentSync / MuseTalk backends. Positioned at the higher-
end of the lip-sync quality ladder — Gaussian-splatting-based
representations avoid the temporal wobble that diffusion lip-sync
introduces.

Routes return 501 ``ROUTE_STUBBED`` in v1.25.0. Promoted once either
upstream ships a pip package with weights under a permissive licence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

SUPPORTED_BACKENDS = ("gaussian_headtalk", "fantasy_talking2")


def check_lipsync_advanced_available() -> bool:
    """Always False in v1.25.0 — Tier 3 stub."""
    return False


INSTALL_HINT = (
    "GaussianHeadTalk / FantasyTalking2 are Tier 3 stubs in v1.25.0. "
    "No pip packages yet — track the WACV'26 / AAAI'26 supplementary "
    "material pages for weight releases."
)


def animate(
    portrait_image: str,
    audio_path: str,
    backend: str = "gaussian_headtalk",
    output: Optional[str] = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Stub. Raises NotImplementedError."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}"
        )
    raise NotImplementedError(
        "Advanced lip-sync is a Tier 3 stub. Expected in v1.27.0+."
    )


def list_backends() -> Dict[str, Any]:
    """Return the Tier 3 backend catalogue (all advertised as unavailable)."""
    return {
        "backends": list(SUPPORTED_BACKENDS),
        "available": False,
        "install_hint": INSTALL_HINT,
    }


__all__ = [
    "SUPPORTED_BACKENDS",
    "check_lipsync_advanced_available",
    "INSTALL_HINT",
    "animate",
    "list_backends",
]
