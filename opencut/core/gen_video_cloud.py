"""
OpenCut Cloud Video Generation (Wave H3.2, v1.25.0) — **STUB**

Hailuo 2.3 (MiniMax) and Seedance 2.0 (ByteDance) HTTP API wrappers.
Closed-weights; requires user-supplied API keys.  Complements the
existing OSS gen-video modules (LTX-Video, Wan 2.1, CogVideoX) for
users who need higher quality at the cost of cloud dependency.

Routes return 501 ``ROUTE_STUBBED`` in v1.25.0 — promoted once the
vendor APIs stabilise enough to pin request / response shapes.
"""

from __future__ import annotations

from typing import Any, Dict

SUPPORTED_BACKENDS = ("hailuo", "seedance")


def check_gen_video_cloud_available() -> bool:
    """Always False in v1.25.0 — Tier 3 stub."""
    return False


INSTALL_HINT = (
    "Hailuo / Seedance cloud generation is a Tier 3 stub in v1.25.0. "
    "Set HAILUO_API_KEY or SEEDANCE_API_KEY env vars and track the "
    "vendor docs for a stable endpoint contract."
)


def submit(prompt: str, backend: str = "hailuo", **_kwargs: Any) -> Dict[str, Any]:
    """Stub. Raises NotImplementedError."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}"
        )
    raise NotImplementedError(
        "Cloud gen-video submit is a Tier 3 stub. Expected in v1.27.0+."
    )


def status(job_id: str, backend: str = "hailuo") -> Dict[str, Any]:
    """Stub. Raises NotImplementedError."""
    raise NotImplementedError(
        "Cloud gen-video status is a Tier 3 stub. Expected in v1.27.0+."
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
    "check_gen_video_cloud_available",
    "INSTALL_HINT",
    "submit",
    "status",
    "list_backends",
]
