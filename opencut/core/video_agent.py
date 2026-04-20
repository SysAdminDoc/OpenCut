"""
OpenCut VideoAgent + ViMax (Wave H3.1, v1.25.0) — **STUB**

Agentic LLM-routed search across indexed footage + auto-storyboard
from a script. Sources:
- https://github.com/HKUDS/VideoAgent  (search / understanding)
- https://github.com/HKUDS/ViMax       (script-to-storyboard)

All routes return 501 ``ROUTE_STUBBED`` in v1.25.0. Promoted to Tier 2
once a user files a feature request or the upstream ships a stable
Python entry point.
"""

from __future__ import annotations

from typing import Any, Dict


def check_video_agent_available() -> bool:
    """Always False — Tier 3 stub."""
    return False


INSTALL_HINT = (
    "VideoAgent / ViMax are Tier 3 strategic stubs in v1.25.0. No pip "
    "package is pinned yet. Track the HKUDS repos for a stable entry "
    "point."
)


def search_footage(query: str, **_kwargs: Any) -> Dict[str, Any]:
    """Stub. Raises NotImplementedError."""
    raise NotImplementedError(
        "VideoAgent search is a Tier 3 stub. Expected in v1.27.0+."
    )


def storyboard(script: str, **_kwargs: Any) -> Dict[str, Any]:
    """Stub. Raises NotImplementedError."""
    raise NotImplementedError(
        "ViMax storyboard is a Tier 3 stub. Expected in v1.27.0+."
    )


__all__ = [
    "check_video_agent_available",
    "INSTALL_HINT",
    "search_footage",
    "storyboard",
]
