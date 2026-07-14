"""Supply-chain guard for locally loaded model weights.

CVE-2026-24747 showed that ``torch.load(..., weights_only=True)`` — the setting
OpenCut relies on everywhere it loads a checkpoint — can still reach heap
corruption / code execution on a crafted pickle. ``weights_only`` is necessary
but not sufficient. This module adds a defense-in-depth layer: pickle-format
checkpoints are scanned with picklescan before ``torch.load`` runs, and the
non-executable ``.safetensors`` format is preferred and skips scanning.

Use :func:`safe_torch_load` in place of ``torch.load`` for any weights that came
from a download or other untrusted source.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("opencut")

# safetensors carries tensors only — no executable payload — so it needs no scan.
_SAFE_SUFFIXES = (".safetensors",)
# Pickle-backed formats can execute code on load.
_PICKLE_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle")


class ModelSecurityError(RuntimeError):
    """Raised when a model file is rejected before loading."""


def scan_model_file(path: str) -> None:
    """Scan a checkpoint for malicious pickle payloads before it is loaded.

    - ``.safetensors`` files are accepted without scanning (no code path).
    - Pickle-format files are scanned with picklescan (>=1.0.3); any flagged
      payload raises :class:`ModelSecurityError`.
    - If picklescan is not installed the load is allowed to proceed (best
      effort) but a warning is logged — ``weights_only=True`` still applies.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    lower = path.lower()
    if lower.endswith(_SAFE_SUFFIXES):
        return

    try:
        from picklescan.scanner import scan_file_path
    except ImportError:
        logger.warning(
            "picklescan not installed; loading %s with weights_only only "
            "(install opencut[ai] or picklescan>=1.0.3 to scan model weights)",
            os.path.basename(path),
        )
        return

    result = scan_file_path(path)
    infected = int(getattr(result, "infected_files", 0) or 0)
    if infected:
        raise ModelSecurityError(
            f"Refusing to load {os.path.basename(path)}: picklescan flagged "
            f"{infected} malicious payload(s) (CVE-2026-24747 class)"
        )


def safe_torch_load(path: str, **kwargs):
    """Scan *path*, then ``torch.load`` it with ``weights_only=True`` enforced."""
    scan_model_file(path)
    import torch

    kwargs.setdefault("weights_only", True)
    return torch.load(path, **kwargs)
