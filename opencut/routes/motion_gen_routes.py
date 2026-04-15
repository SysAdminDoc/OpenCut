"""
OpenCut Motion & Generation Routes

Endpoints for competitive-parity features vs Runway, Descript, and CapCut:
- Generative Extend (Adobe Firefly-style clip extension)
- Green-Screen-Free Background Replacement (Descript/CapCut)
- Consistent Character Generation (IP-Adapter conditioning)
- Motion Brush (Runway-style painted motion control)
"""

import logging
import os
import tempfile

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath, validate_output_path

logger = logging.getLogger("opencut")

motion_gen_bp = Blueprint("motion_gen", __name__)


# ---------------------------------------------------------------------------
# POST /video/generative-extend
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/video/generative-extend", methods=["POST"])
@require_csrf
@async_job("generative_extend")
def generative_extend(job_id, filepath, data):
    """Extend a video clip beyond its recorded length using AI generation."""
    from opencut.core.generative_extend import extend_clip

    extend_seconds = safe_float(data.get("extend_seconds"), 2.0, 0.1, 30.0)
    direction = data.get("direction", "forward").strip().lower()
    model = data.get("model", "auto").strip().lower()
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extend_clip(
        video_path=filepath,
        extend_seconds=extend_seconds,
        direction=direction,
        output=out_path,
        model=model,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /video/replace-background
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/video/replace-background", methods=["POST"])
@require_csrf
@async_job("replace_background")
def replace_background(job_id, filepath, data):
    """Replace video background without a green screen."""
    from opencut.core.greenscreen_free import replace_background as bg_replace

    background = data.get("background", "blur")
    method = data.get("method", "auto").strip().lower()
    edge_blur = safe_int(data.get("edge_blur"), 2, 0, 20)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    # Validate background if it's a file path
    if isinstance(background, str) and background not in ("blur", "none"):
        if not background.startswith("#") and os.path.sep in background or "/" in background:
            background = validate_filepath(background)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = bg_replace(
        video_path=filepath,
        background=background,
        output=out_path,
        method=method,
        edge_blur=edge_blur,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /video/replace-background/preview
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/video/replace-background/preview", methods=["POST"])
@require_csrf
@async_job("bg_replace_preview")
def replace_background_preview(job_id, filepath, data):
    """Preview background replacement on a single frame."""
    from opencut.core.greenscreen_free import replace_background_frame

    background = data.get("background", "blur")
    method = data.get("method", "auto").strip().lower()
    edge_blur = safe_int(data.get("edge_blur"), 2, 0, 20)
    frame_index = safe_int(data.get("frame_index"), 0, 0, 999999)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Extracting frame...")

    # Extract the target frame from the video
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("OpenCV is required for frame preview")

    import cv2

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {filepath}")

    try:
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Could not read frame {frame_index} from video")
    finally:
        cap.release()

    _progress(30, "Processing frame...")

    result = replace_background_frame(
        frame_data=frame,
        background=background,
        method=method,
        edge_blur=edge_blur,
    )

    # Save the result frame to a temp file
    fd, preview_path = tempfile.mkstemp(suffix=".png", prefix="bg_preview_")
    os.close(fd)
    cv2.imwrite(preview_path, result["frame"])

    _progress(100, "Preview complete")

    return {
        "preview_path": preview_path,
        "method_used": method,
        "confidence": round(result["confidence"], 4),
    }


# ---------------------------------------------------------------------------
# POST /ai/character/create
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/ai/character/create", methods=["POST"])
@require_csrf
@async_job("character_create", filepath_required=False)
def character_create(job_id, filepath, data):
    """Create a character identity profile from reference images."""
    from opencut.core.character_consistency import create_character_profile

    reference_images = data.get("reference_images", [])
    if not reference_images or not isinstance(reference_images, list):
        raise ValueError("reference_images must be a non-empty list of image file paths")

    # Validate each reference image path
    validated_images = []
    for img_path in reference_images:
        if isinstance(img_path, str) and img_path.strip():
            validated_images.append(validate_filepath(img_path.strip()))

    if not validated_images:
        raise ValueError("No valid reference image paths provided")

    name = data.get("name", "character").strip() or "character"

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_character_profile(
        reference_images=validated_images,
        name=name,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# GET /ai/character/list
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/ai/character/list", methods=["GET"])
def character_list():
    """List all saved character profiles."""
    try:
        from opencut.core.character_consistency import list_character_profiles

        profiles = list_character_profiles()
        return jsonify({"profiles": profiles, "count": len(profiles)})
    except Exception as e:
        return safe_error(e, "character_list")


# ---------------------------------------------------------------------------
# POST /ai/character/generate
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/ai/character/generate", methods=["POST"])
@require_csrf
@async_job("character_generate", filepath_required=False)
def character_generate(job_id, filepath, data):
    """Generate a scene with a consistent character."""
    from opencut.core.character_consistency import (
        generate_consistent_scene,
        load_character_profile,
    )

    profile_id = data.get("profile_id", "").strip()
    if not profile_id:
        raise ValueError("profile_id is required")

    prompt = data.get("prompt", "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    duration = safe_float(data.get("duration"), 4.0, 1.0, 30.0)
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    # Load the character profile
    profile = load_character_profile(profile_id)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_consistent_scene(
        prompt=prompt,
        character_profile=profile,
        output=out_path,
        duration=duration,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /video/motion-brush
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/video/motion-brush", methods=["POST"])
@require_csrf
@async_job("motion_brush")
def motion_brush(job_id, filepath, data):
    """Apply motion brush effects to an image or video."""
    from opencut.core.motion_brush import apply_motion_brush

    motion_mask = data.get("motion_mask")
    if not motion_mask:
        raise ValueError("motion_mask is required (mask image path, region dict, or list of regions)")

    # Validate mask path if it's a file reference
    if isinstance(motion_mask, str):
        motion_mask = validate_filepath(motion_mask)

    motion_params = data.get("motion_params", {})
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_motion_brush(
        input_path=filepath,
        motion_mask=motion_mask,
        motion_params=motion_params,
        output=out_path,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /video/motion-brush/preview
# ---------------------------------------------------------------------------
@motion_gen_bp.route("/video/motion-brush/preview", methods=["POST"])
@require_csrf
@async_job("motion_brush_preview")
def motion_brush_preview(job_id, filepath, data):
    """Preview motion brush effect on a single frame."""
    from opencut.core.motion_brush import preview_motion_brush

    motion_mask = data.get("motion_mask")
    if not motion_mask:
        raise ValueError("motion_mask is required")

    # Validate mask path if it's a file reference
    if isinstance(motion_mask, str):
        motion_mask = validate_filepath(motion_mask)

    motion_params = data.get("motion_params", {})
    frame_index = safe_int(data.get("frame_index"), 0, 0, 999999)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Generating motion preview...")

    result = preview_motion_brush(
        input_path=filepath,
        motion_mask=motion_mask,
        motion_params=motion_params,
        frame_index=frame_index,
    )

    _progress(100, "Preview complete")

    return result


# ---------------------------------------------------------------------------
# Helper: ensure_package shortcut for this module
# ---------------------------------------------------------------------------
def ensure_package(pkg, pip_name=None):
    """Convenience wrapper around helpers.ensure_package."""
    from opencut.helpers import ensure_package as _ep
    return _ep(pkg, pip_name)
