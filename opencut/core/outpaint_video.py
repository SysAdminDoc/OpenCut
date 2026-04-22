"""
OpenCut Video Outpainting v1.28.0 — Tier 3 strategic stub

Expand video borders with diffusion.
"""
from __future__ import annotations

INSTALL_HINT = "See ROADMAP-NEXT.md Wave K3.6 — depends on K2.17 (LTX-2) or K3.7 (Wan2.1 VACE)."


def check_outpaint_available() -> bool:
    return False


def outpaint(video_path, target_width, target_height, output=None, on_progress=None):
    raise NotImplementedError(
        "Video outpainting ships in v1.29.0. Track ROADMAP-NEXT.md Wave K3.6."
    )


__all__ = ["check_outpaint_available", "INSTALL_HINT", "outpaint"]
