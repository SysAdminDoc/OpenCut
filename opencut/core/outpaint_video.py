"""
OpenCut Video Outpainting v1.28.0 — Tier 3 strategic stub

Expand video borders with diffusion.
"""
from __future__ import annotations

INSTALL_HINT = "Requires a supported diffusion video-editing backend such as LTX or Wan VACE."


def check_outpaint_available() -> bool:
    return False


def outpaint(video_path, target_width, target_height, output=None, on_progress=None):
    raise NotImplementedError(
        "Video outpainting is not implemented yet. Track the live ROADMAP.md entry."
    )


__all__ = ["check_outpaint_available", "INSTALL_HINT", "outpaint"]
