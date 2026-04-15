"""
OpenCut Object Intelligence Routes (Category 69)

Endpoints for advanced object AI features:
- Text-based video segmentation (CLIP + SAM2)
- Physics-aware object removal (shadow + reflection)
- Object tracking with graphic overlays
- Semantic video search (CLIP embeddings)
- Multi-subject intelligent reframe
"""

import logging

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

object_intel_bp = Blueprint("object_intel", __name__)


# ---------------------------------------------------------------------------
# POST /video/text-segment
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/text-segment", methods=["POST"])
@require_csrf
@async_job("text_segment")
def text_segment(job_id, filepath, data):
    """Segment an object from video using a natural language description."""
    from opencut.core.text_segment import segment_by_text

    query = (data.get("query") or "").strip()
    if not query:
        raise ValueError("query is required (e.g., 'the red car')")

    use_sam2 = bool(data.get("use_sam2", True))
    sam2_model = (data.get("sam2_model") or "tiny").strip()
    grid_cols = safe_int(data.get("grid_cols"), 8, 2, 20)
    grid_rows = safe_int(data.get("grid_rows"), 6, 2, 16)
    sample_frames = safe_int(data.get("sample_frames"), 8, 1, 50)
    output_dir = (data.get("output_dir") or "").strip()

    if output_dir:
        output_dir = validate_filepath(output_dir) if output_dir else ""

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = segment_by_text(
        video_path=filepath,
        query=query,
        use_sam2=use_sam2,
        sam2_model=sam2_model,
        output_dir=output_dir,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        sample_frames=sample_frames,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /video/text-segment/preview
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/text-segment/preview", methods=["POST"])
@require_csrf
@async_job("text_segment_preview")
def text_segment_preview(job_id, filepath, data):
    """Preview text segmentation on a single frame."""
    from opencut.core.text_segment import find_target_region

    query = (data.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")

    frame_path = (data.get("frame_path") or "").strip()
    grid_cols = safe_int(data.get("grid_cols"), 8, 2, 20)
    grid_rows = safe_int(data.get("grid_rows"), 6, 2, 16)

    # If frame_path provided, use it; otherwise extract first frame from video
    if frame_path:
        frame_path = validate_filepath(frame_path)
    else:
        import os
        import tempfile
        from opencut.helpers import get_ffmpeg_path, run_ffmpeg

        tmp_dir = tempfile.mkdtemp(prefix="textseg_preview_")
        frame_path = os.path.join(tmp_dir, "preview_frame.png")
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", filepath, "-frames:v", "1", "-q:v", "2", frame_path,
        ])

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    region = find_target_region(
        frame_path=frame_path,
        query=query,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        on_progress=_progress,
    )

    return region.to_dict()


# ---------------------------------------------------------------------------
# POST /video/physics-remove
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/physics-remove", methods=["POST"])
@require_csrf
@async_job("physics_remove")
def physics_remove(job_id, filepath, data):
    """Remove an object plus its shadow and reflection from video."""
    from opencut.core.physics_remove import remove_with_physics

    mask_points = data.get("mask_points")
    object_bbox = data.get("object_bbox")

    if not mask_points and not object_bbox:
        raise ValueError("mask_points or object_bbox is required")

    # Validate mask_points format
    if mask_points:
        if not isinstance(mask_points, list):
            raise ValueError("mask_points must be a list")
        for pt in mask_points:
            if "x" not in pt or "y" not in pt:
                raise ValueError("Each mask_point must have 'x' and 'y'")

    # Validate object_bbox format
    if object_bbox:
        if not isinstance(object_bbox, (list, tuple)) or len(object_bbox) != 4:
            raise ValueError("object_bbox must be [x, y, w, h]")
        object_bbox = tuple(int(v) for v in object_bbox)

    detect_shadows = bool(data.get("detect_shadows", True))
    detect_reflections = bool(data.get("detect_reflections", True))
    inpaint_method = (data.get("inpaint_method") or "lama").strip()
    sam2_model = (data.get("sam2_model") or "tiny").strip()
    output_dir = (data.get("output_dir") or "").strip()

    if output_dir:
        output_dir = validate_filepath(output_dir) if output_dir else ""

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = remove_with_physics(
        video_path=filepath,
        mask_points=mask_points,
        object_bbox=object_bbox,
        detect_shadows=detect_shadows,
        detect_reflections=detect_reflections,
        inpaint_method=inpaint_method,
        output_dir=output_dir,
        sam2_model=sam2_model,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /video/physics-remove/detect-shadow
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/physics-remove/detect-shadow", methods=["POST"])
@require_csrf
@async_job("detect_shadow")
def detect_shadow_route(job_id, filepath, data):
    """Detect shadow direction and extent for an object in a frame."""
    from opencut.core.physics_remove import detect_shadow

    frame_path = (data.get("frame_path") or "").strip()
    object_mask_path = (data.get("object_mask_path") or "").strip()
    object_bbox = data.get("object_bbox")
    mask_points = data.get("mask_points")

    # Use frame_path if given, otherwise extract first frame
    if frame_path:
        frame_path = validate_filepath(frame_path)
    else:
        import os
        import tempfile
        from opencut.helpers import get_ffmpeg_path, run_ffmpeg

        tmp_dir = tempfile.mkdtemp(prefix="shadow_detect_")
        frame_path = os.path.join(tmp_dir, "shadow_frame.png")
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", filepath, "-frames:v", "1", "-q:v", "2", frame_path,
        ])

    if object_mask_path:
        object_mask_path = validate_filepath(object_mask_path)

    if object_bbox:
        if not isinstance(object_bbox, (list, tuple)) or len(object_bbox) != 4:
            raise ValueError("object_bbox must be [x, y, w, h]")
        object_bbox = tuple(int(v) for v in object_bbox)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_shadow(
        frame_path=frame_path,
        object_mask_path=object_mask_path,
        object_bbox=object_bbox,
        mask_points=mask_points,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /video/track-overlay
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/track-overlay", methods=["POST"])
@require_csrf
@async_job("track_overlay")
def track_overlay(job_id, filepath, data):
    """Track an object and render a graphic overlay that follows it."""
    from opencut.core.object_track_overlay import track_and_overlay

    track_point = data.get("track_point")
    if not track_point or not isinstance(track_point, (list, tuple)) or len(track_point) != 2:
        raise ValueError("track_point must be [x, y]")
    track_point = (int(track_point[0]), int(track_point[1]))

    overlay_config = data.get("overlay_config", {})
    if not isinstance(overlay_config, dict):
        overlay_config = {}

    max_frames = safe_int(data.get("max_frames"), 0, 0, 100000)
    smooth = bool(data.get("smooth", True))
    export_track_json = bool(data.get("export_track_json", True))
    output_dir = (data.get("output_dir") or "").strip()

    if output_dir:
        output_dir = validate_filepath(output_dir) if output_dir else ""

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = track_and_overlay(
        video_path=filepath,
        track_point=track_point,
        overlay_config=overlay_config,
        max_frames=max_frames,
        output_dir=output_dir,
        smooth=smooth,
        export_track_json=export_track_json,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /video/overlay-types
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/overlay-types", methods=["GET"])
def list_overlay_types():
    """Return the list of available overlay types for track-overlay."""
    from opencut.core.object_track_overlay import OVERLAY_TYPES

    return jsonify({
        "overlay_types": OVERLAY_TYPES,
        "count": len(OVERLAY_TYPES),
    })


# ---------------------------------------------------------------------------
# POST /search/semantic
# ---------------------------------------------------------------------------
@object_intel_bp.route("/search/semantic", methods=["POST"])
@require_csrf
@async_job("semantic_search", filepath_required=False)
def semantic_search_route(job_id, filepath, data):
    """Search video clips by text query or image similarity."""
    from opencut.core.semantic_video_search import semantic_search

    clip_paths = data.get("clip_paths", [])
    if not isinstance(clip_paths, list) or not clip_paths:
        raise ValueError("clip_paths must be a non-empty list of file paths")

    # Validate each clip path
    validated_paths = []
    for cp in clip_paths:
        if isinstance(cp, str) and cp.strip():
            validated_paths.append(validate_filepath(cp.strip()))

    if not validated_paths:
        raise ValueError("No valid clip paths provided")

    query = (data.get("query") or "").strip()
    query_image = (data.get("query_image") or "").strip()

    if not query and not query_image:
        raise ValueError("Must provide query text or query_image path")

    if query_image:
        query_image = validate_filepath(query_image)

    max_results = safe_int(data.get("max_results"), 20, 1, 200)
    min_score = safe_float(data.get("min_score"), 0.15, 0.0, 1.0)
    auto_index = bool(data.get("auto_index", True))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = semantic_search(
        clip_paths=validated_paths,
        query=query,
        query_image=query_image,
        max_results=max_results,
        min_score=min_score,
        auto_index=auto_index,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /search/semantic/index
# ---------------------------------------------------------------------------
@object_intel_bp.route("/search/semantic/index", methods=["POST"])
@require_csrf
@async_job("semantic_index", filepath_required=False)
def semantic_index_route(job_id, filepath, data):
    """Pre-build a CLIP embedding index for a set of clips."""
    from opencut.core.semantic_video_search import build_clip_index

    clip_paths = data.get("clip_paths", [])
    if not isinstance(clip_paths, list) or not clip_paths:
        raise ValueError("clip_paths must be a non-empty list of file paths")

    validated_paths = []
    for cp in clip_paths:
        if isinstance(cp, str) and cp.strip():
            validated_paths.append(validate_filepath(cp.strip()))

    if not validated_paths:
        raise ValueError("No valid clip paths provided")

    frames_per_clip = safe_int(data.get("frames_per_clip"), 12, 1, 100)
    force_rebuild = bool(data.get("force_rebuild", False))

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = build_clip_index(
        clip_paths=validated_paths,
        frames_per_clip=frames_per_clip,
        force_rebuild=force_rebuild,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /video/reframe-multi
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/reframe-multi", methods=["POST"])
@require_csrf
@async_job("reframe_multi")
def reframe_multi(job_id, filepath, data):
    """Reframe video for a different aspect ratio with multi-subject awareness."""
    from opencut.core.ai_reframe_multi import reframe_multi_subject

    target_ratio = (data.get("target_ratio") or "9:16").strip()
    enable_split_screen = bool(data.get("enable_split_screen", True))
    analysis_interval = safe_int(data.get("analysis_interval"), 5, 1, 30)
    smooth_strength = safe_int(data.get("smooth_strength"), 15, 1, 60)
    output_dir = (data.get("output_dir") or "").strip()

    if output_dir:
        output_dir = validate_filepath(output_dir) if output_dir else ""

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = reframe_multi_subject(
        video_path=filepath,
        target_ratio=target_ratio,
        output_dir=output_dir,
        enable_split_screen=enable_split_screen,
        analysis_interval=analysis_interval,
        smooth_strength=smooth_strength,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /video/aspect-ratios
# ---------------------------------------------------------------------------
@object_intel_bp.route("/video/aspect-ratios", methods=["GET"])
def list_aspect_ratios():
    """Return the list of supported aspect ratios for multi-subject reframe."""
    from opencut.core.ai_reframe_multi import ASPECT_RATIOS

    ratios = []
    for name, (w, h) in ASPECT_RATIOS.items():
        ratios.append({
            "name": name,
            "width": w,
            "height": h,
            "decimal": round(w / h, 4),
        })

    return jsonify({
        "aspect_ratios": ratios,
        "count": len(ratios),
    })
