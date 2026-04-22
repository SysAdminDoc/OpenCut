"""
OpenCut Batch Reframe v1.28.0

Multi-ratio batch reframe. Calls smart_reframe per ratio.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

RATIO_PRESETS: Dict[str, Tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4:3": (1440, 1080),
}


def check_batch_reframe_available() -> bool:
    """Always True — uses FFmpeg via smart_reframe."""
    return True


@dataclass
class BatchReframeResult:
    outputs: Dict[str, str] = field(default_factory=dict)
    ratios: List[str] = field(default_factory=list)
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("outputs", "ratios", "duration", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def batch_reframe(
    input_path: str,
    ratios: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> BatchReframeResult:
    """Reframe input video to multiple aspect ratios."""
    from opencut.core import smart_reframe

    if ratios is None:
        ratios = list(RATIO_PRESETS.keys())

    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    outputs: Dict[str, str] = {}
    total = len(ratios)

    for i, ratio in enumerate(ratios):
        if ratio not in RATIO_PRESETS:
            logger.warning("Unknown ratio %s — skipping", ratio)
            continue
        w, h = RATIO_PRESETS[ratio]
        safe_ratio = ratio.replace(":", "x")
        out_path = os.path.join(output_dir, f"{base}_{safe_ratio}.mp4")

        if on_progress:
            on_progress(int(i / total * 90), f"Reframing {ratio}")

        result = smart_reframe.reframe(input_path, width=w, height=h, output=out_path)
        out = result.get("output") if isinstance(result, dict) else getattr(result, "output", out_path)
        outputs[ratio] = str(out)

    if on_progress:
        on_progress(100, "Done")

    return BatchReframeResult(outputs=outputs, ratios=list(outputs.keys()), duration=0.0, notes=[])


__all__ = ["RATIO_PRESETS", "check_batch_reframe_available", "BatchReframeResult", "batch_reframe"]
