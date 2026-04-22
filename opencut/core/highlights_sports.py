"""
OpenCut Sports/Genre Highlights v1.28.0 — Tier 3 strategic stub

Highlights: optical flow velocity + YAMNet crowd energy + face peaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import _try_import

GENRES = ["sports", "concert", "event", "reaction", "game"]
INSTALL_HINT = "pip install torch torchaudio tensorflow (YAMNet uses TensorFlow Hub)"


def check_sports_highlights_available() -> bool:
    return _try_import("torch") is not None


@dataclass
class HighlightSegment:
    start: float = 0.0
    end: float = 0.0
    score: float = 0.0
    signals: Dict = field(default_factory=dict)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("start", "end", "score", "signals")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def list_genres() -> List[str]:
    return GENRES


def extract(
    video_path: str,
    genre: str = "sports",
    top_n: int = 5,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[HighlightSegment]:
    raise NotImplementedError(
        "Sports highlights extraction ships in v1.29.0. Track ROADMAP-NEXT.md Wave K3.8."
    )


__all__ = ["check_sports_highlights_available", "INSTALL_HINT", "GENRES",
           "HighlightSegment", "list_genres", "extract"]
