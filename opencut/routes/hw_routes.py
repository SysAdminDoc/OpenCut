"""
OpenCut Hardware-Accelerated Encoding Routes

Provides endpoints for detecting HW encoders, running HW-accelerated
encodes, and querying available HW presets.
"""

import logging

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir, _unique_output_path
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
)

logger = logging.getLogger("opencut")

hw_bp = Blueprint("hw", __name__)


# ---------------------------------------------------------------------------
# GET /hw/encoders - Detect available HW encoders (cached)
# ---------------------------------------------------------------------------
@hw_bp.route("/hw/encoders", methods=["GET"])
def hw_encoders():
    """Return detected hardware encoders (cached result)."""
    try:
        from opencut.core.hw_accel import detect_hw_encoders
        info = detect_hw_encoders()
        return jsonify(info.to_dict())
    except Exception as exc:
        return safe_error(exc, "hw_encoders")


# ---------------------------------------------------------------------------
# POST /hw/encoders/refresh - Re-detect hardware encoders
# ---------------------------------------------------------------------------
@hw_bp.route("/hw/encoders/refresh", methods=["POST"])
@require_csrf
def hw_encoders_refresh():
    """Force re-detection of hardware encoders."""
    try:
        from opencut.core.hw_accel import detect_hw_encoders
        info = detect_hw_encoders(force_refresh=True)
        return jsonify(info.to_dict())
    except Exception as exc:
        return safe_error(exc, "hw_encoders_refresh")


# ---------------------------------------------------------------------------
# POST /hw/encode - Hardware-accelerated encode (async job)
# ---------------------------------------------------------------------------
@hw_bp.route("/hw/encode", methods=["POST"])
@require_csrf
@async_job("hw_encode")
def hw_encode_route(job_id, filepath, data):
    """Encode a video using hardware acceleration with automatic fallback."""
    codec = data.get("codec", "h264").strip().lower()
    quality = data.get("quality", "balanced").strip().lower()
    hw_type = data.get("hw_type", "auto").strip().lower()
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    if codec not in ("h264", "hevc", "av1"):
        raise ValueError(f"Unsupported codec: {codec}. Use h264, hevc, or av1.")
    if quality not in ("speed", "balanced", "quality"):
        raise ValueError(f"Unsupported quality: {quality}. Use speed, balanced, or quality.")
    if hw_type not in ("auto", "nvenc", "qsv", "amf", "videotoolbox", "software"):
        raise ValueError(f"Unsupported hw_type: {hw_type}. Use auto, nvenc, qsv, amf, videotoolbox, or software.")

    # Resolve output path
    import os
    resolved_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(resolved_dir, f"{base}_hw_{codec}_{quality}.mp4")
    out_path = _unique_output_path(out_path)

    # Collect extra args if provided
    extra_args = data.get("extra_args", None)
    if extra_args is not None and not isinstance(extra_args, list):
        extra_args = None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.hw_accel import hw_encode

    result = hw_encode(
        input_path=filepath,
        output_path_override=out_path,
        codec=codec,
        quality=quality,
        hw_type=hw_type,
        extra_args=extra_args,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /hw/presets - Available HW encoding presets
# ---------------------------------------------------------------------------
@hw_bp.route("/hw/presets", methods=["GET"])
def hw_presets():
    """Return available hardware encoding presets with availability info."""
    try:
        from opencut.core.hw_accel import get_hw_presets
        presets = get_hw_presets()
        return jsonify({"presets": presets})
    except Exception as exc:
        return safe_error(exc, "hw_presets")
