"""
OpenCut Trailer Generator v1.28.0 — Tier 3 strategic stub

Auto trailer/promo generator.
"""
from __future__ import annotations

INSTALL_HINT = "See ROADMAP-NEXT.md Wave K3.2 — requires LLM + MusicGen + declarative_compose."


def check_trailer_gen_available() -> bool:
    return False


def generate(video_path, style="trailer", duration=60.0, output=None, on_progress=None):
    raise NotImplementedError(
        "Trailer generator ships in v1.29.0. Track ROADMAP-NEXT.md Wave K3.2."
    )


__all__ = ["check_trailer_gen_available", "INSTALL_HINT", "generate"]
