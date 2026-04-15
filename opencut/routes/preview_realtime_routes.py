"""
OpenCut Preview & Real-Time AI Routes

Endpoints for live effect previews, GPU pipeline rendering, A/B
comparison, video scopes, and preview cache management.

Blueprint: preview_realtime_bp  (url_prefix /api)
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

preview_realtime_bp = Blueprint("preview_realtime", __name__, url_prefix="/api")


# ===================================================================
# POST /api/preview/live — Live effect preview on a single frame
# ===================================================================
@preview_realtime_bp.route("/preview/live", methods=["POST"])
@require_csrf
def live_preview():
    """Generate a live preview of an effect applied to a video frame."""
    try:
        from opencut.core.live_preview import generate_preview_base64

        data = request.get_json(silent=True) or {}
        filepath = (data.get("filepath") or "").strip()
        if not filepath:
            return jsonify({"error": "filepath is required"}), 400
        filepath = validate_filepath(filepath)

        effect = (data.get("effect") or "").strip()
        if not effect:
            return jsonify({"error": "effect is required"}), 400

        params = data.get("params", {})
        timestamp = safe_float(data.get("timestamp", 0), min_val=0.0)
        width = safe_int(data.get("width", 854), min_val=64, max_val=3840)
        height = safe_int(data.get("height", 480), min_val=64, max_val=2160)

        result = generate_preview_base64(
            video_path=filepath,
            effect=effect,
            params=params,
            timestamp=timestamp,
            width=width,
            height=height,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, "live_preview")


# ===================================================================
# POST /api/preview/scrub — Batch preview frames at multiple timestamps
# ===================================================================
@preview_realtime_bp.route("/preview/scrub", methods=["POST"])
@require_csrf
@async_job("preview_scrub")
def scrub_preview(job_id, filepath, data):
    """Generate preview frames at evenly-spaced timestamps for scrubbing."""
    from opencut.core.gpu_preview_pipeline import render_scrub_previews

    num_frames = safe_int(data.get("num_frames", 10), min_val=1, max_val=100)
    effects = data.get("effects")  # optional effect chain
    width = safe_int(data.get("width", 854), min_val=64, max_val=3840)
    height = safe_int(data.get("height", 480), min_val=64, max_val=2160)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Generating scrub previews... {pct}%")

    result = render_scrub_previews(
        video_path=filepath,
        num_frames=num_frames,
        effects=effects,
        width=width,
        height=height,
        on_progress=_progress,
    )
    return result


# ===================================================================
# POST /api/preview/compare — A/B comparison frame generation
# ===================================================================
@preview_realtime_bp.route("/preview/compare", methods=["POST"])
@require_csrf
def compare_preview():
    """Generate A/B comparison frames between original and processed video."""
    try:
        from opencut.core.ab_compare import generate_comparison

        data = request.get_json(silent=True) or {}
        original = (data.get("original") or "").strip()
        processed = (data.get("processed") or "").strip()
        if not original or not processed:
            return jsonify({"error": "original and processed paths are required"}), 400

        original = validate_filepath(original)
        processed = validate_filepath(processed)

        mode = data.get("mode", "side_by_side")
        timestamps = data.get("timestamps")
        num_frames = safe_int(data.get("num_frames", 5), min_val=1, max_val=50)
        width = safe_int(data.get("width", 854), min_val=64, max_val=3840)
        height = safe_int(data.get("height", 480), min_val=64, max_val=2160)
        wipe_position = safe_float(data.get("wipe_position", 0.5),
                                   min_val=0.0, max_val=1.0)
        compute_metrics = data.get("compute_metrics", True)

        result = generate_comparison(
            original_path=original,
            processed_path=processed,
            mode=mode,
            timestamps=timestamps,
            num_frames=num_frames,
            width=width,
            height=height,
            wipe_position=wipe_position,
            compute_metrics=compute_metrics,
        )
        return jsonify(result.to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, "compare_preview")


# ===================================================================
# GET /api/preview/compare/metrics — Quality metrics for comparison
# ===================================================================
@preview_realtime_bp.route("/preview/compare/metrics", methods=["GET"])
def compare_metrics():
    """Get quality metrics (SSIM, PSNR, colour delta) for two videos."""
    try:
        from opencut.core.ab_compare import get_compare_metrics

        original = request.args.get("original", "").strip()
        processed = request.args.get("processed", "").strip()
        if not original or not processed:
            return jsonify({"error": "original and processed query params required"}), 400

        original = validate_filepath(original)
        processed = validate_filepath(processed)

        num_frames = safe_int(request.args.get("num_frames", 10),
                              min_val=1, max_val=100)

        result = get_compare_metrics(
            original_path=original,
            processed_path=processed,
            num_frames=num_frames,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, "compare_metrics")


# ===================================================================
# POST /api/preview/scopes — Generate scope data for a frame
# ===================================================================
@preview_realtime_bp.route("/preview/scopes", methods=["POST"])
@require_csrf
def generate_scopes():
    """Generate video scope data for a frame."""
    try:
        data = request.get_json(silent=True) or {}
        filepath = (data.get("filepath") or "").strip()
        if not filepath:
            return jsonify({"error": "filepath is required"}), 400
        filepath = validate_filepath(filepath)

        scope_type = data.get("scope_type", "")
        preset = data.get("preset", "")
        timestamp = safe_float(data.get("timestamp", 0), min_val=0.0)
        check_legal = bool(data.get("check_legal", False))

        if preset:
            from opencut.core.realtime_scopes import generate_scopes_preset
            result = generate_scopes_preset(
                video_path=filepath,
                preset=preset,
                timestamp=timestamp,
            )
            return jsonify(result.to_dict())
        elif scope_type:
            from opencut.core.realtime_scopes import generate_scope
            result = generate_scope(
                video_path=filepath,
                scope_type=scope_type,
                timestamp=timestamp,
                check_legal=check_legal,
            )
            return jsonify(result.to_dict())
        else:
            return jsonify({"error": "scope_type or preset is required"}), 400

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, "generate_scopes")


# ===================================================================
# GET /api/preview/scopes/presets — List scope presets
# ===================================================================
@preview_realtime_bp.route("/preview/scopes/presets", methods=["GET"])
def scope_presets():
    """List available scope presets."""
    try:
        from opencut.core.realtime_scopes import list_presets
        return jsonify({"presets": list_presets()})
    except Exception as exc:
        return safe_error(exc, "scope_presets")


# ===================================================================
# GET /api/preview/cache/stats — Cache statistics
# ===================================================================
@preview_realtime_bp.route("/preview/cache/stats", methods=["GET"])
def cache_stats():
    """Return preview cache statistics."""
    try:
        from opencut.core.preview_cache import cache_stats as get_stats
        return jsonify(get_stats())
    except Exception as exc:
        return safe_error(exc, "cache_stats")


# ===================================================================
# POST /api/preview/cache/warm — Pre-warm cache for a clip
# ===================================================================
@preview_realtime_bp.route("/preview/cache/warm", methods=["POST"])
@require_csrf
@async_job("cache_warm")
def warm_cache(job_id, filepath, data):
    """Pre-warm the preview cache for a video clip."""
    from opencut.core.preview_cache import get_cache_manager

    effect = (data.get("effect") or "").strip()
    if not effect:
        raise ValueError("effect is required")

    params = data.get("params", {})
    num_frames = safe_int(data.get("num_frames", 10), min_val=1, max_val=100)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Warming cache... {pct}%")

    mgr = get_cache_manager()
    cached = mgr.warm_cache(
        video_path=filepath,
        effect_name=effect,
        params=params,
        num_frames=num_frames,
        on_progress=_progress,
    )
    return {"cached_frames": cached, "effect": effect}


# ===================================================================
# DELETE /api/preview/cache — Clear preview cache
# ===================================================================
@preview_realtime_bp.route("/preview/cache", methods=["DELETE"])
@require_csrf
def clear_cache():
    """Clear all preview cache entries."""
    try:
        from opencut.core.preview_cache import cache_flush

        source = request.args.get("source", "").strip()
        effect = request.args.get("effect", "").strip()

        if source:
            from opencut.core.preview_cache import cache_invalidate_file
            count = cache_invalidate_file(source)
            return jsonify({"cleared": count, "scope": "file", "path": source})
        elif effect:
            from opencut.core.preview_cache import cache_invalidate_effect
            count = cache_invalidate_effect(effect)
            return jsonify({"cleared": count, "scope": "effect", "name": effect})
        else:
            count = cache_flush()
            return jsonify({"cleared": count, "scope": "all"})
    except Exception as exc:
        return safe_error(exc, "clear_cache")


# ===================================================================
# POST /api/preview/pipeline — Run GPU pipeline on frame sequence
# ===================================================================
@preview_realtime_bp.route("/preview/pipeline", methods=["POST"])
@require_csrf
@async_job("preview_pipeline")
def run_pipeline(job_id, filepath, data):
    """Run the GPU-accelerated preview pipeline on a frame sequence."""
    from opencut.core.gpu_preview_pipeline import get_pipeline

    timestamps = data.get("timestamps")
    num_frames = safe_int(data.get("num_frames", 10), min_val=1, max_val=100)
    effects = data.get("effects")
    width = safe_int(data.get("width", 854), min_val=64, max_val=3840)
    height = safe_int(data.get("height", 480), min_val=64, max_val=2160)

    def _progress(pct):
        _update_job(job_id, progress=pct, message=f"Pipeline rendering... {pct}%")

    pipeline = get_pipeline()
    result = pipeline.render_batch(
        video_path=filepath,
        timestamps=timestamps,
        num_frames=num_frames,
        effects=effects,
        width=width,
        height=height,
        on_progress=_progress,
    )
    return result.to_dict()


# ===================================================================
# GET /api/preview/effects — List available preview effects
# ===================================================================
@preview_realtime_bp.route("/preview/effects", methods=["GET"])
def list_effects():
    """List all available preview effects."""
    try:
        from opencut.core.live_preview import list_effects as get_effects
        return jsonify({"effects": get_effects()})
    except Exception as exc:
        return safe_error(exc, "list_effects")


# ===================================================================
# GET /api/preview/pipeline/status — Pipeline status
# ===================================================================
@preview_realtime_bp.route("/preview/pipeline/status", methods=["GET"])
def pipeline_status():
    """Return GPU pipeline status."""
    try:
        from opencut.core.gpu_preview_pipeline import pipeline_status as get_status
        return jsonify(get_status())
    except Exception as exc:
        return safe_error(exc, "pipeline_status")
