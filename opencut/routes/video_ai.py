"""
OpenCut Video AI Routes

AI upscale, background removal, interpolation, denoise, face tools,
style transfer.
"""

import logging
import re

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import (
    _resolve_output_dir,
)
from opencut.jobs import (
    _update_job,
    async_job,
)
from opencut.security import (
    rate_limit,
    rate_limit_release,
    require_csrf,
    safe_float,
    safe_int,
    safe_pip_install,
    validate_filepath,
)

logger = logging.getLogger("opencut")

video_ai_bp = Blueprint("video_ai", __name__)


# ---------------------------------------------------------------------------
# Video AI (upscale, bg removal, interpolation, denoise)
# ---------------------------------------------------------------------------
@video_ai_bp.route("/video/ai/capabilities", methods=["GET"])
def video_ai_capabilities():
    """Return available AI video capabilities."""
    try:
        from opencut.core.video_ai import get_ai_capabilities
        return jsonify(get_ai_capabilities())
    except Exception as e:
        return safe_error(e, "video_ai_capabilities")


@video_ai_bp.route("/video/ai/upscale", methods=["POST"])
@require_csrf
@async_job("upscale")
def video_ai_upscale(job_id, filepath, data):
    """AI upscale video using Real-ESRGAN."""
    output_dir = data.get("output_dir", "")
    scale = safe_int(data.get("scale", 2), 2, min_val=1, max_val=4)
    model = data.get("model", "realesrgan-x4plus")
    if model not in ("realesrgan-x4plus", "realesrgan-x4plus-anime", "realesrgan-x2plus"):
        model = "realesrgan-x4plus"

    if not rate_limit("ai_gpu"):
        raise ValueError("A ai_gpu operation is already running. Please wait.")
    try:
        from opencut.core.video_ai import upscale_video

        def _on_progress(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = upscale_video(
            filepath, output_dir=effective_dir,
            scale=scale, model=model,
            on_progress=_on_progress,
        )
        return {"output_path": out}
    finally:
        rate_limit_release("ai_gpu")


@video_ai_bp.route("/video/ai/rembg", methods=["POST"])
@require_csrf
@async_job("rembg")
def video_ai_rembg(job_id, filepath, data):
    """AI background removal using rembg."""
    output_dir = data.get("output_dir", "")
    backend = data.get("backend", "rembg")
    if backend not in ("rembg", "rvm"):
        backend = "rembg"
    model = data.get("model", "birefnet-general" if backend == "rembg" else "mobilenetv3")
    allowed_rembg = ("u2net", "u2net_human_seg", "isnet-general-use", "birefnet-general", "birefnet-massive")
    allowed_rvm = ("mobilenetv3", "resnet50")
    if backend == "rembg" and model not in allowed_rembg:
        model = "birefnet-general"
    elif backend == "rvm" and model not in allowed_rvm:
        model = "mobilenetv3"
    bg_color = data.get("bg_color", "")
    if bg_color and not re.match(r'^[a-zA-Z0-9#]+$', bg_color):
        bg_color = ""
    alpha_only = data.get("alpha_only", False)

    if not rate_limit("ai_gpu"):
        raise ValueError("A ai_gpu operation is already running. Please wait.")
    try:
        from opencut.core.video_ai import remove_background

        def _on_progress(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = remove_background(
            filepath, output_dir=effective_dir,
            model=model, bg_color=bg_color,
            alpha_only=alpha_only,
            backend=backend,
            on_progress=_on_progress,
        )
        return {"output_path": out}
    finally:
        rate_limit_release("ai_gpu")


@video_ai_bp.route("/video/ai/interpolate", methods=["POST"])
@require_csrf
@async_job("interpolate")
def video_ai_interpolate(job_id, filepath, data):
    """AI frame interpolation."""
    output_dir = data.get("output_dir", "")
    multiplier = safe_int(data.get("multiplier", 2), 2, min_val=2, max_val=8)

    if not rate_limit("ai_gpu"):
        raise ValueError("A ai_gpu operation is already running. Please wait.")
    try:
        from opencut.core.video_ai import frame_interpolate

        def _on_progress(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = frame_interpolate(
            filepath, output_dir=effective_dir,
            multiplier=multiplier,
            on_progress=_on_progress,
        )
        return {"output_path": out}
    finally:
        rate_limit_release("ai_gpu")


@video_ai_bp.route("/video/ai/denoise", methods=["POST"])
@require_csrf
@async_job("denoise")
def video_ai_denoise(job_id, filepath, data):
    """AI video noise reduction."""
    output_dir = data.get("output_dir", "")
    method = data.get("method", "nlmeans")
    if method not in ("nlmeans", "hqdn3d", "basicvsr"):
        method = "nlmeans"
    strength = safe_float(data.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0)

    _needs_gpu = method == "basicvsr"
    if _needs_gpu:
        if not rate_limit("ai_gpu"):
            raise ValueError("A ai_gpu operation is already running. Please wait.")
    try:
        from opencut.core.video_ai import video_denoise

        def _on_progress(pct, msg=""):
            _update_job(job_id, progress=pct, message=msg)

        effective_dir = _resolve_output_dir(filepath, output_dir)
        out = video_denoise(
            filepath, output_dir=effective_dir,
            method=method, strength=strength,
            on_progress=_on_progress,
        )
        return {"output_path": out}
    finally:
        if _needs_gpu:
            rate_limit_release("ai_gpu")


@video_ai_bp.route("/video/ai/install", methods=["POST"])
@require_csrf
@async_job("install", filepath_required=False)
def video_ai_install(job_id, filepath, data):
    """Install AI video dependencies."""
    component = data.get("component", "upscale")
    packages = {
        "upscale": ["realesrgan", "basicsr", "opencv-python-headless"],
        "rembg": ["rembg[gpu]", "opencv-python-headless"],
        "rembg_cpu": ["rembg", "opencv-python-headless"],
    }
    pkgs = packages.get(component, [])
    if not pkgs:
        raise ValueError(f"Unknown component: {component}")

    if not rate_limit("model_install"):
        raise ValueError("A model_install operation is already running. Please wait.")
    try:
        for i, pkg in enumerate(pkgs):
            pct = int((i / len(pkgs)) * 90)
            _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
            safe_pip_install(pkg, timeout=600)
        return {"component": component}
    finally:
        rate_limit_release("model_install")


# ---------------------------------------------------------------------------
# Face Tools (detection, blur, auto-censor)
# ---------------------------------------------------------------------------
@video_ai_bp.route("/video/face/capabilities", methods=["GET"])
def face_capabilities():
    """Return face tool capabilities."""
    try:
        from opencut.core.face_tools import check_face_tools_available
        return jsonify(check_face_tools_available())
    except Exception as e:
        return safe_error(e, "face_capabilities")


@video_ai_bp.route("/video/face/detect", methods=["POST"])
@require_csrf
def face_detect():
    """Detect faces in image or first frame of video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    detector = data.get("detector", "mediapipe")
    if detector not in ("insightface", "mediapipe", "haar"):
        detector = "mediapipe"

    if not filepath:
        return jsonify({"error": "File not found"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.face_tools import detect_faces_in_frame
        result = detect_faces_in_frame(filepath, detector=detector)
        return jsonify(result)
    except Exception as e:
        return safe_error(e, "face_detect")


@video_ai_bp.route("/video/face/blur", methods=["POST"])
@require_csrf
@async_job("face_blur")
def face_blur(job_id, filepath, data):
    """Auto-detect and blur faces in video."""
    output_dir = data.get("output_dir", "")
    method = data.get("method", "gaussian")
    if method not in ("gaussian", "pixelate", "black"):
        method = "gaussian"
    strength = safe_int(data.get("strength", 51), 51, min_val=1, max_val=99)
    detector = data.get("detector", "mediapipe")
    if detector not in ("insightface", "mediapipe", "haar"):
        detector = "mediapipe"

    from opencut.core.face_tools import blur_faces

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = blur_faces(
        filepath, output_dir=effective_dir,
        method=method, strength=strength,
        detector=detector, on_progress=_on_progress,
    )
    return {"output_path": out}


from opencut.jobs import make_install_route

make_install_route(video_ai_bp, "/video/face/install", "mediapipe",
                   ["mediapipe", "opencv-python-headless"],
                   doc="Install MediaPipe for face detection.")


# ---------------------------------------------------------------------------
# Style Transfer
# ---------------------------------------------------------------------------
@video_ai_bp.route("/video/style/list", methods=["GET"])
def style_list():
    """Return available style transfer models."""
    try:
        from opencut.core.style_transfer import get_available_styles
        return jsonify({"styles": get_available_styles()})
    except Exception as e:
        return safe_error(e, "style_list")


@video_ai_bp.route("/video/style/apply", methods=["POST"])
@require_csrf
@async_job("style_transfer")
def style_apply(job_id, filepath, data):
    """Apply neural style transfer to video."""
    output_dir = data.get("output_dir", "")
    style_name = data.get("style", "candy")
    if style_name not in ("candy", "mosaic", "rain_princess", "udnie", "starry_night",
                          "la_muse", "the_scream", "pointilism"):
        style_name = "candy"
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=1.0)

    from opencut.core.style_transfer import style_transfer_video

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    out = style_transfer_video(
        filepath, style_name=style_name,
        output_dir=effective_dir,
        intensity=intensity,
        on_progress=_on_progress,
    )
    return {"output_path": out, "style": style_name}


@video_ai_bp.route("/video/style/arbitrary", methods=["POST"])
@require_csrf
@async_job("style_arbitrary")
def style_arbitrary(job_id, filepath, data):
    """Apply arbitrary style transfer using any reference image."""
    style_image = data.get("style_image", "").strip()
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=1.0)
    if not style_image:
        raise ValueError("No style image provided")
    style_image = validate_filepath(style_image)

    from opencut.core.style_transfer import arbitrary_style_transfer

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out = arbitrary_style_transfer(
        filepath, style_image,
        output_dir=effective_dir,
        intensity=intensity,
        on_progress=_on_progress,
    )
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Face Swap & Enhancement
# ---------------------------------------------------------------------------
@video_ai_bp.route("/video/face/swap/capabilities", methods=["GET"])
def face_swap_capabilities():
    try:
        from opencut.core.face_swap import get_face_capabilities
        return jsonify(get_face_capabilities())
    except Exception as e:
        return safe_error(e, "face_swap_capabilities")


@video_ai_bp.route("/video/face/enhance", methods=["POST"])
@require_csrf
@async_job("face_enhance")
def face_enhance_route(job_id, filepath, data):
    """Enhance/restore faces in video using GFPGAN or CodeFormer."""
    fp = data.get("filepath", "").strip()
    _enhance_model = data.get("model", "gfpgan")
    if _enhance_model not in ("gfpgan", "codeformer"):
        _enhance_model = "gfpgan"
    _fidelity = safe_float(data.get("fidelity", 0.5), 0.5, min_val=0.0, max_val=1.0)
    if not fp:
        raise ValueError("File not found")
    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        raise ValueError(str(e))

    from opencut.core.face_swap import enhance_faces

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(fp, data.get("output_dir", ""))
    out = enhance_faces(fp, output_dir=d, model=_enhance_model, upscale=safe_int(data.get("upscale", 2), 2, min_val=1, max_val=4), fidelity=_fidelity, on_progress=_p)
    return {"output_path": out}


@video_ai_bp.route("/video/face/swap", methods=["POST"])
@require_csrf
@async_job("face_swap")
def face_swap_route(job_id, filepath, data):
    """Swap faces in video with a reference image."""
    fp = data.get("filepath", "").strip()
    ref = data.get("reference_face", "").strip()
    if not fp:
        raise ValueError("Video not found")
    if not ref:
        raise ValueError("Reference face not found")
    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        raise ValueError(f"Video: {e}")
    try:
        ref = validate_filepath(ref)
    except ValueError as e:
        raise ValueError(f"Reference face: {e}")

    from opencut.core.face_swap import swap_face

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(fp, data.get("output_dir", ""))
    out = swap_face(fp, ref, output_dir=d, on_progress=_p)
    return {"output_path": out}


# ---------------------------------------------------------------------------
# Pro Upscaling
# ---------------------------------------------------------------------------
@video_ai_bp.route("/video/upscale/capabilities", methods=["GET"])
def upscale_capabilities():
    try:
        from opencut.core.upscale_pro import get_upscale_capabilities
        return jsonify(get_upscale_capabilities())
    except Exception as e:
        return safe_error(e, "upscale_capabilities")


@video_ai_bp.route("/video/upscale/run", methods=["POST"])
@require_csrf
@async_job("upscale")
def upscale_run(job_id, filepath, data):
    """Upscale video with quality preset."""
    fp = data.get("filepath", "").strip()
    if not fp:
        raise ValueError("File not found")
    try:
        fp = validate_filepath(fp)
    except ValueError as e:
        raise ValueError(str(e))

    from opencut.core.upscale_pro import upscale_with_preset

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)
    d = _resolve_output_dir(fp, data.get("output_dir", ""))
    out = upscale_with_preset(fp, preset=data.get("preset", "fast"),
                               scale=safe_int(data.get("scale", 2), 2, min_val=1, max_val=4),
                               output_dir=d, on_progress=_p)
    return {"output_path": out}
