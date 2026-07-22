"""Upscaling, face, segmentation, and depth routes formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

from .wave_l_contract import wave_l_bp

# =============================================================
# L1.2 — Smart Upscaling Hub
# =============================================================

@wave_l_bp.route("/video/upscale/smart", methods=["POST"])
@require_csrf
@async_job("upscale_smart")
def route_upscale_smart(job_id, filepath, data):
    """
    Upscale video using the best available backend (AI or lanczos fallback).

    Body params:
      scale   int     Scale factor (default 2).
      hint    str     Content hint: auto|fast|quality|anime|face|film.
      backend str     Force backend: auto|lanczos|realesrgan|video2x|flashvsr.
      output  str     Output path (optional).
    """
    from opencut.core import upscale_hub

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = upscale_hub.upscale(
        video_path=filepath,
        scale=safe_int(data.get("scale"), 2),
        hint=str(data.get("hint") or "auto"),
        backend=str(data.get("backend") or "auto"),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "backend": result.backend,
        "scale": result.scale,
        "width_out": result.width_out,
        "height_out": result.height_out,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/upscale/smart/info", methods=["GET"])
def route_upscale_smart_info():
    """Return available upscaling backends and auto-selection logic."""
    try:
        from opencut.core import upscale_hub
        return jsonify(upscale_hub.get_hub_info())
    except Exception as exc:
        return safe_error(exc, "upscale_smart_info")

# =============================================================
# L1.3 — AI Face Reshape
# =============================================================

@wave_l_bp.route("/video/face/reshape", methods=["POST"])
@require_csrf
@async_job("face_reshape")
def route_face_reshape(job_id, filepath, data):
    """
    Liquify-style face warp via MediaPipe FaceMesh.

    Body params:
      operation  str    slim_face|enlarge_eyes|shrink_nose|raise_cheekbones|smooth_jaw
      strength   float  0–1 warp intensity (default 0.5).
      output     str    Output path (optional).
    """
    from opencut.core import face_reshape
    if not face_reshape.check_face_reshape_available():
        raise RuntimeError(
            "face_reshape dependencies not installed. " + face_reshape.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = face_reshape.reshape(
        video_path=filepath,
        operation=str(data.get("operation") or "slim_face"),
        strength=safe_float(data.get("strength"), 0.5),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "operation": result.operation,
        "frames_processed": result.frames_processed,
        "faces_found": result.faces_found,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/face/reshape/info", methods=["GET"])
def route_face_reshape_info():
    """Return face reshape capabilities."""
    try:
        from opencut.core import face_reshape
        return jsonify({
            "available": face_reshape.check_face_reshape_available(),
            "operations": list(face_reshape.OPERATIONS),
            "install_hint": face_reshape.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "face_reshape_info")

# =============================================================
# L1.4 — AI Skin Retouch
# =============================================================

@wave_l_bp.route("/video/face/retouch", methods=["POST"])
@require_csrf
@async_job("skin_retouch")
def route_skin_retouch(job_id, filepath, data):
    """
    Skin retouching and blemish removal for video.

    Body params:
      intensity  float  0–1 retouching strength (default 0.6).
      mode       str    bilateral (fast) or gfpgan (deep, GPU).
      radiance   float  0–1 brightness boost on face (default 0).
      output     str    Output path (optional).
    """
    from opencut.core import skin_retouch
    if not skin_retouch.check_skin_retouch_available():
        raise RuntimeError(
            "skin_retouch dependencies not installed. " + skin_retouch.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = skin_retouch.retouch(
        video_path=filepath,
        intensity=safe_float(data.get("intensity"), 0.6),
        mode=str(data.get("mode") or "bilateral"),
        radiance=safe_float(data.get("radiance"), 0.0),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "frames_processed": result.frames_processed,
        "faces_found": result.faces_found,
        "mode": result.mode,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/face/retouch/info", methods=["GET"])
def route_skin_retouch_info():
    """Return skin retouch capabilities."""
    try:
        from opencut.core import skin_retouch
        return jsonify({
            "available": skin_retouch.check_skin_retouch_available(),
            "gfpgan_available": skin_retouch.check_gfpgan_available(),
            "modes": ["bilateral", "gfpgan"],
            "install_hint": skin_retouch.INSTALL_HINT,
            "gfpgan_hint": skin_retouch.GFPGAN_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "skin_retouch_info")

# =============================================================
# N2.1 — SAM 2.1 Video Object Segmentation
# =============================================================

@wave_l_bp.route("/video/segment/sam2", methods=["POST"])
@require_csrf
@async_job("sam2_segment")
def route_sam2_segment(job_id, filepath, data):
    """Segment objects in video using SAM 2.1 with prompt propagation.

    Body params:
      prompts         list   Prompt objects with type/frame/coordinates.
      model           str    tiny / small / base_plus / large.
      output_format   str    alpha_video / matted_video / mask_frames / coco_json.
      propagate       bool   Propagate across all frames (default true).
      output          str    Output path (optional).
    """
    from opencut.core import segment_sam2
    if not segment_sam2.check_sam2_available():
        raise RuntimeError(
            "SAM 2.1 not installed. " + segment_sam2.INSTALL_HINT
        )

    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty list")

    model = str(data.get("model") or "small").strip()
    if model not in segment_sam2.SAM2_MODELS:
        model = "small"

    output_format = str(data.get("output_format") or "alpha_video").strip()
    if output_format not in segment_sam2.SAM2_OUTPUT_FORMATS:
        output_format = "alpha_video"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = segment_sam2.segment_video(
        video_path=filepath,
        prompts=prompts,
        model=model,
        output_format=output_format,
        propagate=safe_bool(data.get("propagate"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "output_format": result.output_format,
        "frames_processed": result.frames_processed,
        "objects_tracked": result.objects_tracked,
        "model": result.model,
        "processing_seconds": result.processing_seconds,
        "mask_count": result.mask_count,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/segment/sam2/info", methods=["GET"])
def route_sam2_info():
    """Return SAM 2.1 availability and model catalogue."""
    try:
        from opencut.core import segment_sam2
        return jsonify({
            "available": segment_sam2.check_sam2_available(),
            "models": segment_sam2.SAM2_MODELS,
            "prompt_types": segment_sam2.SAM2_PROMPT_TYPES,
            "output_formats": segment_sam2.SAM2_OUTPUT_FORMATS,
            "licence": "Apache-2.0",
            "install_hint": segment_sam2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "sam2_info")

# =============================================================
# N2.2 — Depth-Anything-V2 Depth Estimation
# =============================================================

@wave_l_bp.route("/video/depth/estimate-v2", methods=["POST"])
@require_csrf
@async_job("depth_estimate_v2", resumable=True)
def route_depth_estimate(job_id, filepath, data):
    """Estimate per-frame depth maps using Depth-Anything-V2."""
    from opencut.core import depth_anything_v2
    if not depth_anything_v2.check_depth_anything_v2_available():
        raise RuntimeError("Depth deps not installed. " + depth_anything_v2.INSTALL_HINT)

    model = str(data.get("model") or "small").strip()
    if model not in depth_anything_v2.DA2_MODELS:
        model = "small"

    fmt = str(data.get("output_format") or "video").strip()
    if fmt not in ("video", "numpy"):
        fmt = "video"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = depth_anything_v2.estimate_depth(
        video_path=filepath, model=model, output_format=fmt,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/video/depth/parallax-v2", methods=["POST"])
@require_csrf
@async_job("depth_parallax_v2")
def route_depth_parallax(job_id, filepath, data):
    """Generate 2.5D parallax via Depth-Anything-V2 layer separation."""
    from opencut.core import depth_anything_v2
    if not depth_anything_v2.check_depth_anything_v2_available():
        raise RuntimeError("Depth deps not installed. " + depth_anything_v2.INSTALL_HINT)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = depth_anything_v2.generate_parallax(
        video_path=filepath,
        shift_x=safe_float(data.get("shift_x"), 20.0, min_val=-100.0, max_val=100.0),
        shift_y=safe_float(data.get("shift_y"), 0.0, min_val=-100.0, max_val=100.0),
        model=str(data.get("model") or "small"),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/video/depth/info", methods=["GET"])
def route_depth_info():
    try:
        from opencut.core import depth_anything_v2
        return jsonify({
            "available": depth_anything_v2.check_depth_anything_v2_available(),
            "models": depth_anything_v2.DA2_MODELS,
            "licence": "Apache-2.0",
            "install_hint": depth_anything_v2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "depth_info")

# =============================================================
# N2.3 — SAM2 + Depth Compositor Pipeline
# =============================================================

@wave_l_bp.route("/video/compose/depth_segment", methods=["POST"])
@require_csrf
@async_job("depth_segment_compose")
def route_depth_segment_compose(job_id, filepath, data):
    """SAM2 segmentation + Depth estimation + layered compositing."""
    from opencut.core import compose_depth_segment
    if not compose_depth_segment.check_composite_available():
        raise RuntimeError("Pipeline deps not installed. " + compose_depth_segment.INSTALL_HINT)

    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty list")

    effects = data.get("effects", ["blur_background"])
    if not isinstance(effects, list):
        effects = ["blur_background"]

    bg_img = str(data.get("background_image") or "").strip()
    if bg_img:
        from opencut.security import validate_filepath
        bg_img = validate_filepath(bg_img)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = compose_depth_segment.compose(
        video_path=filepath, prompts=prompts, effects=effects,
        background_color=str(data.get("background_color") or ""),
        background_image=bg_img,
        blur_strength=safe_float(data.get("blur_strength"), 15.0, min_val=1.0, max_val=99.0),
        parallax_shift=safe_float(data.get("parallax_shift"), 20.0, min_val=-100.0, max_val=100.0),
        sam_model=str(data.get("sam_model") or "small"),
        depth_model=str(data.get("depth_model") or "small"),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/video/compose/depth_segment/info", methods=["GET"])
def route_compose_info():
    try:
        from opencut.core import compose_depth_segment
        return jsonify({
            "available": compose_depth_segment.check_composite_available(),
            "effects": compose_depth_segment.COMPOSITE_EFFECTS,
            "install_hint": compose_depth_segment.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "compose_info")
