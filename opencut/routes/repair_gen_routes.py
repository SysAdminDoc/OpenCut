"""
OpenCut Repair & AI Generation Routes

Blueprint ``repair_gen_bp`` provides endpoints for:
  - Corrupted file recovery
  - Adaptive deinterlacing
  - Old footage restoration pipeline
  - SDR-to-HDR upconversion
  - Frame rate conversion with optical flow
  - AI outpainting / frame extension
  - Image-to-video animation
  - AI scene extension
  - AI video summary / condensed recap
  - AI background replacement
"""

import logging

from flask import Blueprint

from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

repair_gen_bp = Blueprint("repair_gen", __name__)


# ===================================================================
# REPAIR ROUTES
# ===================================================================


# ---------------------------------------------------------------------------
# POST /repair/recover — MOOV atom / corrupted file recovery
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/repair/recover", methods=["POST"])
@require_csrf
@async_job("repair_recover")
def repair_recover(job_id, filepath, data):
    """Recover a corrupted video file (MOOV atom or frame salvage)."""
    reference_path = data.get("reference_path", "").strip()
    if reference_path:
        reference_path = validate_filepath(reference_path)

    mode = data.get("mode", "moov").strip().lower()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if mode == "salvage":
        from opencut.core.video_repair import salvage_frames
        result = salvage_frames(filepath, on_progress=_p)
    else:
        from opencut.core.video_repair import recover_moov_atom
        result = recover_moov_atom(
            filepath,
            reference_path=reference_path or None,
            on_progress=_p,
        )
    return result


# ---------------------------------------------------------------------------
# POST /repair/deinterlace — Adaptive deinterlace
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/repair/deinterlace", methods=["POST"])
@require_csrf
@async_job("adaptive_deinterlace")
def repair_deinterlace(job_id, filepath, data):
    """Adaptive deinterlace with auto-detection."""
    method = data.get("method", "auto").strip().lower()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.deinterlace import adaptive_deinterlace
    result = adaptive_deinterlace(
        filepath,
        method=method,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /repair/restore — Full old-footage restoration pipeline
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/repair/restore", methods=["POST"])
@require_csrf
@async_job("old_restoration")
def repair_restore(job_id, filepath, data):
    """Run the old-footage restoration pipeline."""
    preset = data.get("preset", "VHS").strip()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.old_restoration import restore_old_footage
    result = restore_old_footage(
        filepath,
        preset=preset,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /repair/sdr-to-hdr — SDR to HDR upconversion
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/repair/sdr-to-hdr", methods=["POST"])
@require_csrf
@async_job("sdr_to_hdr")
def repair_sdr_to_hdr(job_id, filepath, data):
    """Convert SDR video to HDR."""
    transfer = data.get("transfer", "pq").strip().lower()
    max_lum = safe_int(data.get("max_luminance"), 1000, min_val=100, max_val=10000)
    max_cll = safe_int(data.get("max_cll"), 1000, min_val=100, max_val=10000)
    max_fall = safe_int(data.get("max_fall"), 400, min_val=50, max_val=4000)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.sdr_to_hdr import sdr_to_hdr
    result = sdr_to_hdr(
        filepath,
        transfer=transfer,
        max_luminance=max_lum,
        max_cll=max_cll,
        max_fall=max_fall,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /repair/framerate — Frame rate conversion with optical flow
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/repair/framerate", methods=["POST"])
@require_csrf
@async_job("framerate_convert")
def repair_framerate(job_id, filepath, data):
    """Convert video frame rate with optical flow interpolation."""
    preset = data.get("preset", "smooth").strip().lower()
    target_fps = data.get("target_fps")
    if target_fps is not None:
        target_fps = safe_float(target_fps, 0, min_val=1, max_val=240)
        if target_fps == 0:
            target_fps = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.framerate_convert import convert_framerate
    result = convert_framerate(
        filepath,
        target_fps=target_fps,
        preset=preset,
        on_progress=_p,
    )
    return result


# ===================================================================
# AI GENERATION ROUTES
# ===================================================================


# ---------------------------------------------------------------------------
# POST /ai-gen/outpaint — Frame outpainting / aspect ratio extension
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/ai-gen/outpaint", methods=["POST"])
@require_csrf
@async_job("ai_outpaint")
def aigen_outpaint(job_id, filepath, data):
    """Outpaint video frames to a new aspect ratio."""
    target_ratio = data.get("target_ratio", "16:9").strip()
    fill_method = data.get("fill_method", "auto").strip().lower()
    ai_enhance = safe_bool(data.get("ai_enhance"), True)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.frame_extension import outpaint_aspect_ratio
    result = outpaint_aspect_ratio(
        filepath,
        target_ratio=target_ratio,
        fill_method=fill_method,
        ai_enhance=ai_enhance,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /ai-gen/img-to-video — Image animation
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/ai-gen/img-to-video", methods=["POST"])
@require_csrf
@async_job("ai_img_to_video")
def aigen_img_to_video(job_id, filepath, data):
    """Animate a still image into a video clip."""
    duration = safe_float(data.get("duration"), 5.0, min_val=1, max_val=30)
    motion_preset = data.get("motion_preset", "zoom_in").strip()
    motion_prompt = data.get("motion_prompt", "").strip() or None
    fps = safe_float(data.get("fps"), 30.0, min_val=10, max_val=60)
    width = safe_int(data.get("width"), 1920, min_val=320, max_val=3840)
    height = safe_int(data.get("height"), 1080, min_val=240, max_val=2160)
    method = data.get("method", "auto").strip().lower()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.img_to_video import animate_image
    result = animate_image(
        filepath,
        duration=duration,
        motion_preset=motion_preset,
        motion_prompt=motion_prompt,
        fps=fps,
        width=width,
        height=height,
        method=method,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /ai-gen/extend-scene — AI scene extension
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/ai-gen/extend-scene", methods=["POST"])
@require_csrf
@async_job("ai_extend_scene")
def aigen_extend_scene(job_id, filepath, data):
    """Extend a video clip by generating continuation frames."""
    extra_seconds = safe_float(data.get("extra_seconds"), 3.0, min_val=0.5, max_val=30)
    blend_frames = safe_int(data.get("blend_frames"), 10, min_val=0, max_val=60)
    method = data.get("method", "auto").strip().lower()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.scene_extend import extend_scene
    result = extend_scene(
        filepath,
        extra_seconds=extra_seconds,
        blend_frames=blend_frames,
        method=method,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /ai-gen/summarize — Video summary / condensed recap
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/ai-gen/summarize", methods=["POST"])
@require_csrf
@async_job("ai_summarize")
def aigen_summarize(job_id, filepath, data):
    """Create a condensed video recap from the most important shots."""
    target_duration = safe_float(data.get("target_duration"), 45.0, min_val=10, max_val=300)
    min_duration = safe_float(data.get("min_duration"), 30.0, min_val=5, max_val=300)
    max_duration = safe_float(data.get("max_duration"), 60.0, min_val=10, max_val=600)
    scene_threshold = safe_float(data.get("scene_threshold"), 0.3, min_val=0.05, max_val=0.95)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.video_condensed import condense_video
    result = condense_video(
        filepath,
        target_duration=target_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        scene_threshold=scene_threshold,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# POST /ai-gen/replace-bg — AI background replacement
# ---------------------------------------------------------------------------
@repair_gen_bp.route("/ai-gen/replace-bg", methods=["POST"])
@require_csrf
@async_job("ai_replace_bg")
def aigen_replace_bg(job_id, filepath, data):
    """Replace video background with AI-generated or custom background."""
    bg_type = data.get("bg_type", "blur").strip().lower()
    prompt = data.get("prompt", "").strip()
    bg_image_path = data.get("bg_image_path", "").strip()
    bg_video_path = data.get("bg_video_path", "").strip()
    bg_color = data.get("bg_color", "#004400").strip()
    removal_method = data.get("removal_method", "auto").strip().lower()

    if bg_image_path:
        bg_image_path = validate_filepath(bg_image_path)
    if bg_video_path:
        bg_video_path = validate_filepath(bg_video_path)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.bg_replace_ai import replace_background
    result = replace_background(
        filepath,
        bg_type=bg_type,
        prompt=prompt,
        bg_image_path=bg_image_path or None,
        bg_video_path=bg_video_path or None,
        bg_color=bg_color,
        removal_method=removal_method,
        on_progress=_p,
    )
    return result
