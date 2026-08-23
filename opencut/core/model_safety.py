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
import re
from importlib import metadata
from pathlib import PurePosixPath, PureWindowsPath

logger = logging.getLogger("opencut")

HUGGINGFACE_HUB_MIN_VERSION = "1.26.0"
HUGGINGFACE_HUB_MAX_VERSION = "2.0.0"
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")

# Full commit hashes captured from the Hub API on 2026-08-23. Only repositories
# in this table may execute Python supplied by a model repository.
REVIEWED_REMOTE_CODE_MODELS = {
    "microsoft/Florence-2-base": "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac",
    "ZhengPeng7/BiRefNet": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    "BadToBest/EchoMimic": "987c98d1d00a5fd062c5f087719af663fa90bc53",
}

PINNED_HF_MODELS = {
    **REVIEWED_REMOTE_CODE_MODELS,
    "JustFrederik/nllb-200-distilled-600M-ct2-float16": (
        "9a293c97317609fe6a1782c672a7f51443086549"
    ),
    "facebook/seamless-m4t-v2-large": "5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
}

# safetensors carries tensors only — no executable payload — so it needs no scan.
_SAFE_SUFFIXES = (".safetensors",)
# Pickle-backed formats can execute code on load. NeMo ``.nemo`` archives are
# tarballs that wrap a pickle-format checkpoint, so they belong to this class.
_PICKLE_SUFFIXES = (".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle", ".nemo")


def is_pickle_format(path: str) -> bool:
    """True when *path* is a pickle-backed (code-executing) checkpoint format."""
    return path.lower().endswith(_PICKLE_SUFFIXES)


class ModelSecurityError(RuntimeError):
    """Raised when a model file is rejected before loading."""


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?", str(value))
    return tuple(int(part) for part in match.groups()) if match else ()


def require_safe_huggingface_hub() -> str:
    """Return the installed Hub version or reject an unreviewed release."""
    try:
        version = metadata.version("huggingface-hub")
    except metadata.PackageNotFoundError as exc:
        raise ModelSecurityError(
            "huggingface-hub>=1.26,<2 is required for model downloads"
        ) from exc
    parsed = _version_tuple(version)
    if not (
        _version_tuple(HUGGINGFACE_HUB_MIN_VERSION)
        <= parsed
        < _version_tuple(HUGGINGFACE_HUB_MAX_VERSION)
    ):
        raise ModelSecurityError(
            f"huggingface-hub {version} is outside the reviewed "
            f">={HUGGINGFACE_HUB_MIN_VERSION},<2 security lane for CVE-2026-15717"
        )
    return version


def _load_huggingface_hub():
    require_safe_huggingface_hub()
    import huggingface_hub

    return huggingface_hub


def require_immutable_hf_revision(revision: str) -> str:
    """Require a full Hub commit hash rather than a mutable branch or tag."""
    normalized = str(revision or "").strip()
    if not _IMMUTABLE_REVISION_RE.fullmatch(normalized):
        raise ValueError("Hugging Face revision must be a full 40-character commit hash")
    return normalized.lower()


def validate_hf_filename(filename: str) -> str:
    """Reject repository names that escape a local download directory."""
    normalized = str(filename or "")
    segments = normalized.replace("\\", "/").split("/")
    pure_paths = [
        PurePosixPath(normalized),
        *(PureWindowsPath(segment) for segment in normalized.split("/")),
    ]
    unsafe = (
        not normalized
        or "\x00" in normalized
        or ".." in segments
        or any(path.drive or path.root for path in pure_paths)
    )
    if unsafe:
        raise ValueError(f"unsafe Hugging Face filename: {normalized!r}")
    return normalized


def safe_list_repo_files(repo_id: str, *, revision: str, **kwargs) -> list[str]:
    """List and validate every repository filename at one immutable commit."""
    immutable_revision = require_immutable_hf_revision(revision)
    hub = _load_huggingface_hub()
    list_kwargs = {
        key: kwargs[key]
        for key in ("repo_type", "token")
        if key in kwargs and kwargs[key] is not None
    }
    files = hub.list_repo_files(
        repo_id=repo_id,
        revision=immutable_revision,
        **list_kwargs,
    )
    return [validate_hf_filename(filename) for filename in files]


def safe_snapshot_download(repo_id: str, *, revision: str, **kwargs) -> str:
    """Preflight repository paths, then download the same immutable commit."""
    immutable_revision = require_immutable_hf_revision(revision)
    safe_list_repo_files(repo_id, revision=immutable_revision, **kwargs)
    hub = _load_huggingface_hub()
    return str(
        hub.snapshot_download(
            repo_id=repo_id,
            revision=immutable_revision,
            **kwargs,
        )
    )


def safe_hf_hub_download(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    **kwargs,
) -> str:
    """Download one safe repository path from one immutable commit."""
    immutable_revision = require_immutable_hf_revision(revision)
    safe_filename = validate_hf_filename(filename)
    hub = _load_huggingface_hub()
    return str(
        hub.hf_hub_download(
            repo_id=repo_id,
            filename=safe_filename,
            revision=immutable_revision,
            **kwargs,
        )
    )


def reviewed_remote_code_kwargs(model_id: str, *, revision: str | None = None) -> dict:
    """Return loader arguments only for a reviewed repository and commit."""
    canonical = str(model_id)
    if not canonical:
        raise ModelSecurityError(f"remote model code is not reviewed for {model_id!r}")
    if canonical not in REVIEWED_REMOTE_CODE_MODELS:
        raise ModelSecurityError(f"remote model code is not reviewed for {model_id!r}")
    expected = REVIEWED_REMOTE_CODE_MODELS[canonical]
    requested = require_immutable_hf_revision(revision or expected)
    if requested != expected:
        raise ModelSecurityError(
            f"remote model code revision mismatch for {canonical}: expected {expected}"
        )
    return {"trust_remote_code": True, "revision": expected}


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
