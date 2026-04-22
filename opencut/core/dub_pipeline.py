"""
OpenCut Dubbing Pipeline v1.28.0 — Tier 3 strategic stub

Full local dubbing: STT -> translate -> TTS -> lipsync -> composite.
"""
from __future__ import annotations

INSTALL_HINT = "See ROADMAP-NEXT.md Wave K3.1 — requires K2.4 + K2.5 to be installed first."


def check_dub_pipeline_available() -> bool:
    return False


def dub(video_path, target_language, voice=None, output=None, on_progress=None):
    raise NotImplementedError(
        "Dubbing pipeline ships in v1.29.0. Track ROADMAP-NEXT.md Wave K3.1."
    )


__all__ = ["check_dub_pipeline_available", "INSTALL_HINT", "dub"]
