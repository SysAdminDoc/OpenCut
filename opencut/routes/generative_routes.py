"""
OpenCut Generative Routes

Endpoints for AI talking head generation and 3D Gaussian splat rendering:
- Talking head: generate lip-synced video from photo + audio
- Gaussian splat: load PLY files, define camera paths, render to video
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
    validate_path,
)

logger = logging.getLogger("opencut")

generative_bp = Blueprint("generative", __name__)


# ===========================================================================
# Talking Head Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /talking-head/generate — generate talking head video (async)
# ---------------------------------------------------------------------------
@generative_bp.route("/talking-head/generate", methods=["POST"])
@require_csrf
@async_job("talking_head_generate", filepath_param="image_path")
def talking_head_generate(job_id, filepath, data):
    """Generate a talking head video from a still image + audio."""
    from opencut.core.talking_head import TalkingHeadConfig, generate_talking_head

    audio_path = data.get("audio_path", "").strip()
    if not audio_path:
        raise ValueError("No audio_path provided")
    audio_path = validate_filepath(audio_path)

    backend = data.get("backend", "simple").strip()
    fps = safe_int(data.get("fps", 25), 25, min_val=1, max_val=60)
    res_w = safe_int(data.get("width", 512), 512, min_val=64, max_val=4096)
    res_h = safe_int(data.get("height", 512), 512, min_val=64, max_val=4096)
    expression_scale = safe_float(data.get("expression_scale", 1.0), 1.0,
                                  min_val=0.0, max_val=3.0)
    pose_style = safe_int(data.get("pose_style", 0), 0, min_val=0, max_val=46)
    still_mode = bool(data.get("still_mode", False))
    enhancer = data.get("enhancer", "").strip()
    preprocess = data.get("preprocess", "crop").strip()

    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    if out_path is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        out_path = _op(filepath, f"talking_{backend}", effective_dir)

    config = TalkingHeadConfig(
        image_path=filepath,
        audio_path=audio_path,
        backend=backend,
        fps=fps,
        resolution=(res_w, res_h),
        expression_scale=expression_scale,
        pose_style=pose_style,
        still_mode=still_mode,
        enhancer=enhancer,
        preprocess=preprocess,
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_talking_head(
        config=config,
        output=out_path,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "duration": result.duration,
        "frames": result.frames,
        "backend_used": result.backend_used,
        "face_detected": result.face_detected,
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# POST /talking-head/simple — simple FFmpeg-based image+audio → video (async)
# ---------------------------------------------------------------------------
@generative_bp.route("/talking-head/simple", methods=["POST"])
@require_csrf
@async_job("talking_head_simple", filepath_param="image_path")
def talking_head_simple(job_id, filepath, data):
    """Generate a simple talking head video (static image + audio + Ken Burns)."""
    from opencut.core.talking_head import generate_simple_talking_head

    audio_path = data.get("audio_path", "").strip()
    if not audio_path:
        raise ValueError("No audio_path provided")
    audio_path = validate_filepath(audio_path)

    fps = safe_int(data.get("fps", 25), 25, min_val=1, max_val=60)
    res_w = safe_int(data.get("width", 512), 512, min_val=64, max_val=4096)
    res_h = safe_int(data.get("height", 512), 512, min_val=64, max_val=4096)
    zoom_speed = safe_float(data.get("zoom_speed", 0.0003), 0.0003,
                            min_val=0.0, max_val=0.01)
    fade_duration = safe_float(data.get("fade_duration", 1.0), 1.0,
                               min_val=0.0, max_val=10.0)

    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    if out_path is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        out_path = _op(filepath, "talking_simple", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result_path = generate_simple_talking_head(
        image_path=filepath,
        audio_path=audio_path,
        output=out_path,
        fps=fps,
        resolution=(res_w, res_h),
        zoom_speed=zoom_speed,
        fade_duration=fade_duration,
        on_progress=_progress,
    )

    return {"output_path": result_path}


# ---------------------------------------------------------------------------
# POST /talking-head/detect-face — detect face in image (sync)
# ---------------------------------------------------------------------------
@generative_bp.route("/talking-head/detect-face", methods=["POST"])
@require_csrf
def talking_head_detect_face():
    """Detect faces in an image."""
    try:
        from opencut.core.talking_head import detect_face_in_image

        data = request.get_json(force=True) or {}
        image_path = data.get("image_path", "").strip()
        if not image_path:
            return jsonify({"error": "No image_path provided"}), 400

        image_path = validate_filepath(image_path)
        result = detect_face_in_image(image_path)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "talking_head_detect_face")


# ---------------------------------------------------------------------------
# GET /talking-head/backends — list available backends (sync)
# ---------------------------------------------------------------------------
@generative_bp.route("/talking-head/backends", methods=["GET"])
def talking_head_backends():
    """List available talking head backends."""
    try:
        from opencut.core.talking_head import list_available_backends
        backends = list_available_backends()
        return jsonify({"backends": backends})
    except Exception as e:
        return safe_error(e, "talking_head_backends")


# ===========================================================================
# Gaussian Splat Routes
# ===========================================================================


# ---------------------------------------------------------------------------
# POST /gaussian-splat/load — load and validate splat file (sync)
# ---------------------------------------------------------------------------
@generative_bp.route("/gaussian-splat/load", methods=["POST"])
@require_csrf
def gaussian_splat_load():
    """Load a Gaussian splat PLY file and return scene metadata."""
    try:
        from opencut.core.gaussian_splat import load_splat, validate_splat

        data = request.get_json(force=True) or {}
        ply_path = data.get("ply_path", "").strip()
        if not ply_path:
            return jsonify({"error": "No ply_path provided"}), 400

        ply_path = validate_filepath(ply_path)

        # Validate first
        validation = validate_splat(ply_path)
        if not validation["valid"]:
            return jsonify({
                "error": "Invalid PLY file",
                "validation": validation,
            }), 400

        # Load scene
        scene = load_splat(ply_path)
        return jsonify({
            "ply_path": scene.ply_path,
            "point_count": scene.point_count,
            "bounds": scene.bounds,
            "has_colors": scene.has_colors,
            "has_normals": scene.has_normals,
            "has_sh_coeffs": scene.has_sh_coeffs,
            "has_opacity": scene.has_opacity,
            "properties": scene.properties,
            "message": scene.message,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "gaussian_splat_load")


# ---------------------------------------------------------------------------
# POST /gaussian-splat/render — render splat to video (async)
# ---------------------------------------------------------------------------
@generative_bp.route("/gaussian-splat/render", methods=["POST"])
@require_csrf
@async_job("splat_render", filepath_param="ply_path")
def gaussian_splat_render(job_id, filepath, data):
    """Render a Gaussian splat scene to video along a camera path."""
    from opencut.core.gaussian_splat import (
        CameraKeyframe,
        define_camera_path,
        render_splat_to_video,
    )

    res_w = safe_int(data.get("width", 1280), 1280, min_val=64, max_val=4096)
    res_h = safe_int(data.get("height", 720), 720, min_val=64, max_val=4096)
    fps = safe_int(data.get("fps", 30), 30, min_val=1, max_val=60)
    interpolation = data.get("interpolation", "linear").strip()

    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    if out_path is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        out_path = _op(filepath, "splat_render", effective_dir)

    # Parse keyframes from request
    raw_keyframes = data.get("keyframes", [])
    if not raw_keyframes:
        raise ValueError("No keyframes provided for camera path")

    keyframes = []
    for kf in raw_keyframes:
        pos = tuple(float(v) for v in kf.get("position", [0, 0, 0]))
        rot = tuple(float(v) for v in kf.get("rotation", [0, 0, 0]))
        fov = safe_float(kf.get("fov", 60.0), 60.0, min_val=1.0, max_val=179.0)
        time = safe_float(kf.get("time", 0.0), 0.0, min_val=0.0)
        keyframes.append(CameraKeyframe(
            position=pos, rotation=rot, fov=fov, time=time,
        ))

    camera_path = define_camera_path(keyframes, interpolation, fps)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = render_splat_to_video(
        ply_path=filepath,
        camera_path=camera_path,
        output_path=out_path,
        resolution=(res_w, res_h),
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "duration": result.duration,
        "frames": result.frames,
        "resolution": list(result.resolution),
        "renderer": result.renderer,
        "message": result.message,
    }


# ---------------------------------------------------------------------------
# POST /gaussian-splat/orbit — generate orbit camera path (sync)
# ---------------------------------------------------------------------------
@generative_bp.route("/gaussian-splat/orbit", methods=["POST"])
@require_csrf
def gaussian_splat_orbit():
    """Generate a circular orbit camera path for a splat scene."""
    try:
        from opencut.core.gaussian_splat import create_orbit_path

        data = request.get_json(force=True) or {}

        center = data.get("center", [0, 0, 0])
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            return jsonify({"error": "center must be a list of 3 floats"}), 400
        center = tuple(float(v) for v in center)

        radius = safe_float(data.get("radius", 3.0), 3.0, min_val=0.01, max_val=1000.0)
        height = safe_float(data.get("height", 1.0), 1.0, min_val=-1000.0, max_val=1000.0)
        frames = safe_int(data.get("frames", 120), 120, min_val=2, max_val=10000)
        fps = safe_int(data.get("fps", 30), 30, min_val=1, max_val=60)
        fov = safe_float(data.get("fov", 60.0), 60.0, min_val=1.0, max_val=179.0)
        loops = safe_int(data.get("loops", 1), 1, min_val=1, max_val=10)

        path = create_orbit_path(
            center=center,
            radius=radius,
            height=height,
            frames=frames,
            fps=fps,
            fov=fov,
            loops=loops,
        )

        keyframes_json = [
            {
                "position": list(kf.position),
                "rotation": list(kf.rotation),
                "fov": kf.fov,
                "time": round(kf.time, 4),
            }
            for kf in path.keyframes
        ]

        return jsonify({
            "keyframes": keyframes_json,
            "duration": round(path.duration, 4),
            "total_frames": path.total_frames,
            "fps": path.fps,
            "interpolation": path.interpolation,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "gaussian_splat_orbit")


# ---------------------------------------------------------------------------
# POST /gaussian-splat/preview-frame — render single frame preview (sync)
# ---------------------------------------------------------------------------
@generative_bp.route("/gaussian-splat/preview-frame", methods=["POST"])
@require_csrf
def gaussian_splat_preview_frame():
    """Render a single preview frame of a Gaussian splat scene."""
    try:
        from opencut.core.gaussian_splat import CameraKeyframe, render_splat_frame

        data = request.get_json(force=True) or {}
        ply_path = data.get("ply_path", "").strip()
        if not ply_path:
            return jsonify({"error": "No ply_path provided"}), 400

        ply_path = validate_filepath(ply_path)

        pos = tuple(float(v) for v in data.get("position", [0, 0, 5]))
        rot = tuple(float(v) for v in data.get("rotation", [0, 0, 0]))
        fov = safe_float(data.get("fov", 60.0), 60.0, min_val=1.0, max_val=179.0)
        res_w = safe_int(data.get("width", 1280), 1280, min_val=64, max_val=4096)
        res_h = safe_int(data.get("height", 720), 720, min_val=64, max_val=4096)

        camera = CameraKeyframe(position=pos, rotation=rot, fov=fov, time=0.0)

        frame_path = render_splat_frame(
            ply_path=ply_path,
            camera=camera,
            resolution=(res_w, res_h),
        )

        from flask import send_file
        return send_file(frame_path, mimetype="image/png")
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "gaussian_splat_preview_frame")
