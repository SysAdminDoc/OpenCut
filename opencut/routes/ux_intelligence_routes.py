"""
OpenCut UX Intelligence Routes

Endpoints for the UX intelligence layer:
  - Feature search (Cmd+K command palette)
  - Feature usage tracking (recents)
  - Preview frame extraction with before/after comparison
  - Smart defaults for operations
  - Contextual suggestions based on clip analysis
  - Full feature index
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

logger = logging.getLogger("opencut")

ux_intel_bp = Blueprint("ux_intel", __name__)


# ===================================================================
# POST /ux/search -- Fuzzy search features (synchronous)
# ===================================================================
@ux_intel_bp.route("/ux/search", methods=["POST"])
@require_csrf
def ux_search():
    """Fuzzy search the feature index.

    Expects JSON: {"query": "...", "limit": 10}
    Returns: {"results": [...], "total": N}
    """
    try:
        from opencut.core.command_palette import fuzzy_search

        data = request.get_json(force=True, silent=True) or {}
        query = str(data.get("query", "")).strip()
        limit = safe_int(data.get("limit", 10), default=10, min_val=1, max_val=50)

        if not query:
            return jsonify({"results": [], "total": 0})

        results = fuzzy_search(query, limit=limit)
        return jsonify({
            "results": results,
            "total": len(results),
        })
    except Exception as exc:
        return safe_error(exc, context="ux-search")


# ===================================================================
# POST /ux/search/record -- Record feature usage (synchronous)
# ===================================================================
@ux_intel_bp.route("/ux/search/record", methods=["POST"])
@require_csrf
def ux_record_feature():
    """Record a feature usage event.

    Expects JSON: {"feature_id": "..."}
    Returns: {"feature_id": "...", "name": "...", "use_count": N, "timestamp": T}
    """
    try:
        from opencut.core.command_palette import record_feature_use

        data = request.get_json(force=True, silent=True) or {}
        feature_id = str(data.get("feature_id", "")).strip()

        if not feature_id:
            return jsonify({"error": "feature_id is required"}), 400

        result = record_feature_use(feature_id)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, context="ux-record-feature")


# ===================================================================
# GET /ux/recents -- Get recent features (synchronous)
# ===================================================================
@ux_intel_bp.route("/ux/recents", methods=["GET"])
def ux_recents():
    """Return recently used features.

    Query params: limit (int, default 5)
    Returns: {"recents": [...]}
    """
    try:
        from opencut.core.command_palette import get_recent_features

        limit = safe_int(request.args.get("limit", 5), default=5, min_val=1, max_val=50)
        recents = get_recent_features(limit=limit)
        return jsonify({"recents": recents})
    except Exception as exc:
        return safe_error(exc, context="ux-recents")


# ===================================================================
# POST /ux/preview -- Preview operation on single frame (async)
# ===================================================================
@ux_intel_bp.route("/ux/preview", methods=["POST"])
@require_csrf
@async_job("ux_preview")
def ux_preview(job_id, filepath, data):
    """Extract a frame, apply an operation, return before/after.

    Expects JSON: {
        "filepath": "...",
        "operation": "denoise",
        "params": {},
        "timestamp": 0.0
    }
    """
    from opencut.core.preview_frame import preview_operation

    operation = str(data.get("operation", "")).strip()
    if not operation:
        raise ValueError("operation is required")

    params = data.get("params") or {}
    timestamp = safe_float(data.get("timestamp", 0.0), min_val=0.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = preview_operation(
        filepath,
        operation=operation,
        params=params,
        timestamp=timestamp,
        on_progress=_progress,
    )

    return {
        "original_b64": result.original_b64,
        "processed_b64": result.processed_b64,
        "width": result.width,
        "height": result.height,
        "timestamp": result.timestamp,
        "operation": operation,
    }


# ===================================================================
# POST /ux/smart-defaults -- Get smart defaults (synchronous)
# ===================================================================
@ux_intel_bp.route("/ux/smart-defaults", methods=["POST"])
@require_csrf
def ux_smart_defaults():
    """Get smart defaults for an operation based on clip properties.

    Expects JSON: {
        "operation": "normalize",
        "clip_profile": {
            "avg_loudness_lufs": -20.0,
            "resolution": 1920,
            "fps": 30.0,
            "codec": "h264",
            "duration_s": 120.0,
            "has_audio": true,
            "has_video": true,
            "is_static_camera": false,
            "detected_content_type": "interview",
            ...
        }
    }
    """
    try:
        from opencut.core.smart_defaults import ClipProfile, get_smart_defaults

        data = request.get_json(force=True, silent=True) or {}
        operation = str(data.get("operation", "")).strip()

        if not operation:
            return jsonify({"error": "operation is required"}), 400

        profile_data = data.get("clip_profile", {})
        if not isinstance(profile_data, dict):
            return jsonify({"error": "clip_profile must be a dict"}), 400

        # Build ClipProfile from provided data
        profile = ClipProfile(
            avg_loudness_lufs=profile_data.get("avg_loudness_lufs"),
            peak_db=profile_data.get("peak_db"),
            resolution=safe_int(profile_data.get("resolution", 0), min_val=0),
            fps=safe_float(profile_data.get("fps", 0.0), min_val=0.0),
            codec=str(profile_data.get("codec", "")),
            duration_s=safe_float(profile_data.get("duration_s", 0.0), min_val=0.0),
            has_audio=bool(profile_data.get("has_audio", False)),
            has_video=bool(profile_data.get("has_video", True)),
            is_static_camera=bool(profile_data.get("is_static_camera", False)),
            detected_content_type=str(profile_data.get("detected_content_type", "unknown")),
            width=safe_int(profile_data.get("width", 0), min_val=0),
            height=safe_int(profile_data.get("height", 0), min_val=0),
            audio_channels=safe_int(profile_data.get("audio_channels", 0), min_val=0),
            bitrate_kbps=safe_int(profile_data.get("bitrate_kbps", 0), min_val=0),
            pixel_format=str(profile_data.get("pixel_format", "")),
            has_alpha=bool(profile_data.get("has_alpha", False)),
            rotation=safe_int(profile_data.get("rotation", 0), min_val=0),
            sample_rate=safe_int(profile_data.get("sample_rate", 0), min_val=0),
        )

        defaults = get_smart_defaults(operation, profile)
        return jsonify({
            "operation": operation,
            "defaults": defaults,
        })
    except Exception as exc:
        return safe_error(exc, context="ux-smart-defaults")


# ===================================================================
# POST /ux/suggest -- Contextual suggestions (async, probes video)
# ===================================================================
@ux_intel_bp.route("/ux/suggest", methods=["POST"])
@require_csrf
@async_job("ux_suggest")
def ux_suggest(job_id, filepath, data):
    """Analyze clip and suggest relevant operations.

    Expects JSON: {
        "filepath": "...",
        "recent_ops": ["normalize_audio", "add_captions"],
        "max_suggestions": 3
    }
    """
    from opencut.core.contextual_suggest import suggest_operations

    recent_ops = data.get("recent_ops") or []
    if not isinstance(recent_ops, list):
        recent_ops = []
    max_suggestions = safe_int(data.get("max_suggestions", 3), default=3, min_val=1, max_val=10)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    suggestions = suggest_operations(
        filepath,
        recent_ops=recent_ops,
        max_suggestions=max_suggestions,
        on_progress=_progress,
    )

    return {
        "suggestions": suggestions,
        "total": len(suggestions),
    }


# ===================================================================
# GET /ux/feature-index -- Full feature index (synchronous)
# ===================================================================
@ux_intel_bp.route("/ux/feature-index", methods=["GET"])
def ux_feature_index():
    """Return the full feature index.

    Returns: {"features": [...], "total": N}
    """
    try:
        from opencut.core.command_palette import build_feature_index

        index = build_feature_index()
        return jsonify({
            "features": index,
            "total": len(index),
        })
    except Exception as exc:
        return safe_error(exc, context="ux-feature-index")
