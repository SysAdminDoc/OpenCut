"""Disk preflight helpers for heavyweight async routes."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, Optional

from opencut.core import disk_monitor
from opencut.security import validate_output_path, validate_path

MIN_REQUIRED_MB = 500

_OPERATION_RATIOS = {
    "transcribe": 1.5,
    "captions": 1.5,
    "transcript": 1.5,
    "whisperx": 1.5,
    "full_pipeline": 2.5,
    "demucs": 4.0,
    "separate": 4.0,
    "deepfilter": 2.5,
    "video_export": 1.2,
    "export": 1.2,
    "export_preset": 1.2,
    "video_ai_heavy": 6.0,
    "video_ai_depth": 6.0,
}


def estimate_required_mb(
    operation: str,
    source_path: str = "",
    *,
    minimum_mb: int = MIN_REQUIRED_MB,
) -> int:
    """Estimate required free disk space for *operation*."""
    required_mb = int(max(1, minimum_mb))
    try:
        if source_path and os.path.isfile(source_path):
            size_mb = os.path.getsize(source_path) / (1024 * 1024)
            ratio = _OPERATION_RATIOS.get(operation, 1.0)
            required_mb = max(required_mb, int(math.ceil(size_mb * ratio)))
    except OSError:
        pass
    return required_mb


def resolve_output_dir(source_path: str = "", data: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the directory that the route is expected to write into."""
    data = data or {}
    output_path = data.get("output_path") or data.get("output")
    if isinstance(output_path, str) and output_path.strip():
        return os.path.dirname(validate_output_path(output_path.strip()))

    output_dir = data.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        return validate_path(output_dir.strip())

    if source_path:
        source_dir = os.path.dirname(os.path.abspath(source_path))
        if source_dir:
            return source_dir

    return tempfile.gettempdir()


def ensure_disk_for(
    operation: str,
    source_path: str = "",
    data: Optional[Dict[str, Any]] = None,
    *,
    required_mb: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a structured disk preflight result for a planned operation."""
    output_dir = resolve_output_dir(source_path, data)
    required = int(required_mb) if required_mb is not None else estimate_required_mb(
        operation,
        source_path,
    )
    result = dict(disk_monitor.preflight(output_dir, required_mb=required))
    result.setdefault("ok", True)
    result["operation"] = operation
    result["output_dir"] = output_dir
    result["required_mb"] = int(result.get("required_mb") or required)
    result["free_mb"] = int(result.get("free_mb") or 0)
    result["note"] = str(result.get("note") or "")
    return result
