"""
OpenCut Face Age Transform v1.28.0 — Tier 3 strategic stub

AI face age slider via IP-Adapter + Cutie tracking.
"""
from __future__ import annotations

INSTALL_HINT = "Confirm model-weight license terms before enabling this capability."


def check_face_age_available() -> bool:
    return False


def transform(video_path, target_age=30, output=None, on_progress=None):
    raise NotImplementedError(
        "Face age transform is not implemented yet. Track the live ROADMAP.md entry."
    )


__all__ = ["check_face_age_available", "INSTALL_HINT", "transform"]
