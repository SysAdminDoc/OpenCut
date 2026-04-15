"""
OpenCut VFX Advanced Routes

Pika-style object effects and planar tracking endpoints.
"""

import logging

from flask import Blueprint, jsonify

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
)

logger = logging.getLogger("opencut")

vfx_advanced_bp = Blueprint("vfx_advanced", __name__)


# ===================================================================
# Object Effects
# ===================================================================

# ---------------------------------------------------------------------------
# POST /object-effects/apply -- apply effect to object in video (async)
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/object-effects/apply", methods=["POST"])
@require_csrf
@async_job("object_effect")
def object_effects_apply(job_id, filepath, data):
    """Apply a physics-simulated effect to a selected object."""
    from opencut.core.object_effects import (
        EffectConfig,
        apply_object_effect,
        generate_effect_mask,
    )

    effect_type = data.get("effect_type", "squish").strip()
    intensity = safe_float(data.get("intensity", 0.7), 0.7, min_val=0.0, max_val=1.0)
    duration = safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=30.0)
    seed = safe_int(data.get("seed", 42), 42)
    click_x = safe_int(data.get("click_x", 0), 0, min_val=0)
    click_y = safe_int(data.get("click_y", 0), 0, min_val=0)
    num_frames = safe_int(data.get("num_frames", 60), 60, min_val=1, max_val=9999)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"fx_{effect_type}", effective_dir)

    config = EffectConfig(
        effect_type=effect_type,
        intensity=intensity,
        duration=duration,
        seed=seed,
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Generating object mask...")
    mask = generate_effect_mask(
        filepath, (click_x, click_y), num_frames=num_frames,
        on_progress=_progress,
    )

    _progress(30, f"Applying {effect_type} effect...")
    result = apply_object_effect(
        video_path=filepath,
        mask_or_points=mask,
        effect_config=config,
        out_path=output,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "frames_processed": result.frames_processed,
        "effect_applied": result.effect_applied,
    }


# ---------------------------------------------------------------------------
# POST /object-effects/preview -- preview effect on single frame (async)
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/object-effects/preview", methods=["POST"])
@require_csrf
@async_job("object_effect_preview")
def object_effects_preview(job_id, filepath, data):
    """Preview an object effect on a single frame."""
    from opencut.core.object_effects import (
        EffectConfig,
        generate_effect_mask,
        preview_effect_frame,
    )

    effect_type = data.get("effect_type", "squish").strip()
    intensity = safe_float(data.get("intensity", 0.7), 0.7, min_val=0.0, max_val=1.0)
    duration = safe_float(data.get("duration", 1.0), 1.0, min_val=0.1, max_val=30.0)
    seed = safe_int(data.get("seed", 42), 42)
    click_x = safe_int(data.get("click_x", 0), 0, min_val=0)
    click_y = safe_int(data.get("click_y", 0), 0, min_val=0)
    timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    config = EffectConfig(
        effect_type=effect_type,
        intensity=intensity,
        duration=duration,
        seed=seed,
    )

    _progress(10, "Generating mask for preview...")
    mask = generate_effect_mask(
        filepath, (click_x, click_y), num_frames=1,
        on_progress=_progress,
    )

    _progress(60, "Rendering preview frame...")
    result = preview_effect_frame(
        video_path=filepath,
        mask=mask,
        effect_config=config,
        timestamp=timestamp,
    )

    return result


# ---------------------------------------------------------------------------
# POST /object-effects/generate-mask -- generate object mask from click
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/object-effects/generate-mask", methods=["POST"])
@require_csrf
@async_job("generate_mask")
def object_effects_generate_mask(job_id, filepath, data):
    """Generate object mask from a click point on the video."""
    from opencut.core.object_effects import generate_effect_mask

    click_x = safe_int(data.get("click_x", 0), 0, min_val=0)
    click_y = safe_int(data.get("click_y", 0), 0, min_val=0)
    num_frames = safe_int(data.get("num_frames", 30), 30, min_val=1, max_val=9999)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    mask = generate_effect_mask(
        filepath, (click_x, click_y), num_frames=num_frames,
        on_progress=_progress,
    )

    return {
        "num_frames": len(mask.mask_frames),
        "bbox_first_frame": list(mask.bbox_per_frame[0]) if mask.bbox_per_frame else [],
    }


# ---------------------------------------------------------------------------
# GET /object-effects/types -- list available effect types
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/object-effects/types", methods=["GET"])
def object_effects_types():
    """List all available object effect types."""
    from opencut.core.object_effects import get_available_effects
    return jsonify({"effects": get_available_effects()})


# ===================================================================
# Planar Tracking
# ===================================================================

# ---------------------------------------------------------------------------
# POST /planar-track/track -- track planar surface (async)
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/planar-track/track", methods=["POST"])
@require_csrf
@async_job("planar_track")
def planar_track_track(job_id, filepath, data):
    """Track a planar surface through video using corner points."""
    from opencut.core.planar_track import track_planar_surface

    corners = data.get("corners", [])
    if not corners or len(corners) != 4:
        raise ValueError(
            "Exactly 4 corner points required as [[x,y], ...]. "
            f"Got {len(corners) if corners else 0}."
        )

    initial_corners = [(float(c[0]), float(c[1])) for c in corners]
    start_frame = safe_int(data.get("start_frame", 0), 0, min_val=0)
    end_frame_val = data.get("end_frame", None)
    end_frame = safe_int(end_frame_val, None) if end_frame_val is not None else None

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = track_planar_surface(
        video_path=filepath,
        initial_corners=initial_corners,
        start_frame=start_frame,
        end_frame=end_frame,
        on_progress=_progress,
    )

    return {
        "total_frames": result.total_frames,
        "fps": result.fps,
        "avg_confidence": round(result.avg_confidence, 4),
        "duration": round(result.duration, 2),
        "first_frame_corners": result.frames[0].as_list() if result.frames else [],
        "last_frame_corners": result.frames[-1].as_list() if result.frames else [],
    }


# ---------------------------------------------------------------------------
# POST /planar-track/insert -- insert replacement content (async)
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/planar-track/insert", methods=["POST"])
@require_csrf
@async_job("planar_insert")
def planar_track_insert(job_id, filepath, data):
    """Insert replacement content onto a tracked planar surface."""
    from opencut.core.planar_track import (
        insert_replacement,
    )

    replacement_path = data.get("replacement_path", "").strip()
    if not replacement_path:
        raise ValueError("No replacement_path provided")
    replacement_path = validate_filepath(replacement_path)

    corners_data = data.get("corners", [])
    if not corners_data or len(corners_data) != 4:
        raise ValueError("Exactly 4 corner points required for initial track region")

    initial_corners = [(float(c[0]), float(c[1])) for c in corners_data]
    start_frame = safe_int(data.get("start_frame", 0), 0, min_val=0)
    end_frame_val = data.get("end_frame", None)
    end_frame = safe_int(end_frame_val, None) if end_frame_val is not None else None
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "planar_insert", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    # First track the surface
    from opencut.core.planar_track import track_planar_surface

    _progress(5, "Tracking planar surface...")
    track_result = track_planar_surface(
        video_path=filepath,
        initial_corners=initial_corners,
        start_frame=start_frame,
        end_frame=end_frame,
        on_progress=lambda pct, msg: _progress(int(pct * 0.5), msg),
    )

    _progress(50, "Inserting replacement content...")
    result_path = insert_replacement(
        video_path=filepath,
        track_result=track_result,
        replacement_image_or_video=replacement_path,
        out_path=output,
        on_progress=lambda pct, msg: _progress(50 + int(pct * 0.5), msg),
    )

    return {
        "output_path": result_path,
        "frames_tracked": track_result.total_frames,
        "avg_confidence": round(track_result.avg_confidence, 4),
    }


# ---------------------------------------------------------------------------
# POST /planar-track/export -- export track data
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/planar-track/export", methods=["POST"])
@require_csrf
@async_job("planar_export")
def planar_track_export(job_id, filepath, data):
    """Export planar tracking data in various formats."""
    from opencut.core.planar_track import (
        export_track_data,
        track_planar_surface,
    )

    corners = data.get("corners", [])
    if not corners or len(corners) != 4:
        raise ValueError("Exactly 4 corner points required")

    initial_corners = [(float(c[0]), float(c[1])) for c in corners]
    start_frame = safe_int(data.get("start_frame", 0), 0, min_val=0)
    end_frame_val = data.get("end_frame", None)
    end_frame = safe_int(end_frame_val, None) if end_frame_val is not None else None
    export_format = data.get("format", "json").strip()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Tracking surface for export...")
    track_result = track_planar_surface(
        video_path=filepath,
        initial_corners=initial_corners,
        start_frame=start_frame,
        end_frame=end_frame,
        on_progress=_progress,
    )

    _progress(90, f"Exporting as {export_format}...")
    exported = export_track_data(track_result, format=export_format)

    return {
        "format": export_format,
        "data": exported,
        "total_frames": track_result.total_frames,
    }


# ---------------------------------------------------------------------------
# POST /planar-track/preview -- preview tracking on frame (async)
# ---------------------------------------------------------------------------
@vfx_advanced_bp.route("/planar-track/preview", methods=["POST"])
@require_csrf
@async_job("planar_preview")
def planar_track_preview(job_id, filepath, data):
    """Preview tracking visualization on a single frame."""
    from opencut.core.planar_track import (
        preview_track_frame,
        track_planar_surface,
    )

    corners = data.get("corners", [])
    if not corners or len(corners) != 4:
        raise ValueError("Exactly 4 corner points required")

    initial_corners = [(float(c[0]), float(c[1])) for c in corners]
    frame_number = safe_int(data.get("frame_number", 0), 0, min_val=0)
    start_frame = safe_int(data.get("start_frame", 0), 0, min_val=0)
    # Track at least to the requested frame
    end_frame = max(frame_number + 2, safe_int(data.get("end_frame", frame_number + 2), frame_number + 2))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Tracking for preview...")
    track_result = track_planar_surface(
        video_path=filepath,
        initial_corners=initial_corners,
        start_frame=start_frame,
        end_frame=end_frame,
        on_progress=_progress,
    )

    _progress(80, "Rendering preview...")
    # Clamp frame_number to tracked range
    preview_frame = min(frame_number - start_frame, len(track_result.frames) - 1)
    preview_frame = max(0, preview_frame)

    result = preview_track_frame(
        video_path=filepath,
        track_result=track_result,
        frame_number=preview_frame,
    )

    return result
