"""
OpenCut Depth-Anything-3 (DA3) Depth Estimation

DA3 (Nov 2025) is a single-transformer monocular depth model.
**DA3-Small is Apache-2.0**, so it is
MIT-distribution-safe and is the preferred default depth backend for the
CineFocus bokeh / parallax / depth-guided effects, with Depth-Anything-V2-Small
retained as an automatic fallback.

Mirrors :mod:`opencut.core.depth_anything_v2`: an accurate
``check_depth_anything_3_available()`` gates the engine and the HF model ids are
centralised so depth-loading call sites resolve DA3 first and degrade to DA2 on
any failure. DA3 uses its official ``depth_anything_3.api`` package rather than
the Transformers Depth Anything V2 adapter.

Licence: Apache-2.0 (DA3-Small).
Repository: https://github.com/ByteDance-Seed/depth-anything-3
Model: https://huggingface.co/depth-anything/DA3-SMALL
"""

from __future__ import annotations

import logging

from opencut.helpers import _try_import

logger = logging.getLogger("opencut")

MODEL_ID = "depth-anything/DA3-SMALL"
MODEL_NAME = "da3-small"
MODEL_LICENSE = "Apache-2.0"

INSTALL_HINT = "pip install depth-anything-3==0.1.1  # Apache-2.0; optional"

# HF model ids by size. DA3-Small is the default; the others are available for
# callers that want more geometry at higher cost.
DA3_MODEL_IDS = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3-LARGE",
}


def check_depth_anything_3_available() -> bool:
    """True when the official DA3 inference stack is importable.

    Never raises — returns False when the optional dependency is absent so the
    depth backends fall back to Depth-Anything-V2.
    """
    return _try_import("torch") is not None and _try_import("depth_anything_3") is not None


def model_id(size: str = "small") -> str:
    """Resolve a DA3 HF model id by size, defaulting to Small."""
    return DA3_MODEL_IDS.get((size or "small").lower(), DA3_MODEL_IDS["small"])


def resolve_depth_model(size: str = "small") -> tuple[str, str]:
    """Return ``(backend, hf_model_id)`` preferring DA3, falling back to DA2.

    Pure resolver (no heavy imports beyond the availability probe) so call sites
    can pick the model id deterministically and keep DA2 as the documented
    fallback.
    """
    if check_depth_anything_3_available():
        return ("da3", model_id(size))
    da2_size = (size or "small").capitalize()
    return ("da2", f"depth-anything/Depth-Anything-V2-{da2_size}-hf")


__all__ = [
    "MODEL_ID",
    "MODEL_NAME",
    "MODEL_LICENSE",
    "INSTALL_HINT",
    "DA3_MODEL_IDS",
    "check_depth_anything_3_available",
    "model_id",
    "resolve_depth_model",
]
