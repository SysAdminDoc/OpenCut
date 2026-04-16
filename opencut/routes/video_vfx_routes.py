"""
OpenCut Video VFX Routes

Routes for motion tracking, AI relighting, 360 video, clean plate,
gaming highlights, deepfake detection, face tagging, and holy grail timelapse.
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

video_vfx_bp = Blueprint("video_vfx", __name__)


# ===================================================================
# 1. Motion Tracking & Object Annotation
# ===================================================================
@video_vfx_bp.route("/video/track-object", methods=["POST"])
@require_csrf
@async_job("motion_track")
def track_object_route(job_id, filepath, data):
    """Track an object from a click point through video frames."""
    from opencut.core.motion_tracking import track_object

    point_x = safe_int(data.get("x", 0), 0, min_val=0)
    point_y = safe_int(data.get("y", 0), 0, min_val=0)
    if point_x == 0 and point_y == 0:
        raise ValueError("Initial point (x, y) is required")

    max_frames = safe_int(data.get("max_frames", 0), 0, min_val=0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "track", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = track_object(
        video_path=filepath,
        initial_point=(point_x, point_y),
        output_path=output,
        max_frames=max_frames,
        on_progress=_progress,
    )

    return result


@video_vfx_bp.route("/video/annotate-tracked", methods=["POST"])
@require_csrf
@async_job("annotate_track")
def annotate_tracked_route(job_id, filepath, data):
    """Annotate tracked objects on a video."""
    from opencut.core.motion_tracking import annotate_tracked

    track_data = data.get("track_data", [])
    if not track_data:
        raise ValueError("No track_data provided")

    annotation = data.get("annotation", {"type": "box"})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "annotated", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = annotate_tracked(
        video_path=filepath,
        track_data=track_data,
        annotation=annotation,
        output_path=output,
        on_progress=_progress,
    )

    return {"output_path": result}


@video_vfx_bp.route("/video/export-track", methods=["POST"])
@require_csrf
def export_track_route():
    """Export track data to JSON or CSV."""
    try:
        from opencut.core.motion_tracking import export_track_data

        data = request.get_json(force=True) or {}
        track_data = data.get("track_data", [])
        if not track_data:
            return jsonify({"error": "No track_data provided"}), 400

        output_path = data.get("output_path", "").strip()
        if output_path:
            output_path = validate_output_path(output_path)
        if not output_path:
            return jsonify({"error": "No output_path provided"}), 400

        fmt = data.get("format", "json").strip()

        result = export_track_data(track_data, output_path, format=fmt)

        return jsonify({"output_path": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "export_track")


# ===================================================================
# 2. AI Relighting
# ===================================================================
@video_vfx_bp.route("/video/relight", methods=["POST"])
@require_csrf
@async_job("relight")
def relight_route(job_id, filepath, data):
    """Relight a video with virtual lighting."""
    from opencut.core.relighting import relight_video

    light_config = data.get("light_config", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)
    depth_source = data.get("depth_source", "auto")

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "relit", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = relight_video(
        video_path=filepath,
        light_config=light_config,
        output_path=output,
        depth_source=depth_source,
        on_progress=_progress,
    )

    return {"output_path": result}


# ===================================================================
# 3. 360 Video Support
# ===================================================================
@video_vfx_bp.route("/video/360/detect", methods=["POST"])
@require_csrf
def detect_360_route():
    """Detect if a video is 360 and its projection type."""
    try:
        from opencut.core.video_360 import detect_360_format

        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        result = detect_360_format(filepath)

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "detect_360")


@video_vfx_bp.route("/video/360/convert", methods=["POST"])
@require_csrf
@async_job("360_convert")
def convert_360_route(job_id, filepath, data):
    """Convert 360 video projection."""
    from opencut.core.video_360 import convert_360_projection

    projection = data.get("projection", "cubemap").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"360_{projection}", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = convert_360_projection(
        video_path=filepath,
        projection=projection,
        output_path=output,
        on_progress=_progress,
    )

    return {"output_path": result}


@video_vfx_bp.route("/video/360/crop", methods=["POST"])
@require_csrf
@async_job("360_crop")
def crop_360_route(job_id, filepath, data):
    """Extract a flat crop from 360 video."""
    from opencut.core.video_360 import extract_360_crop

    yaw = safe_float(data.get("yaw", 0.0), 0.0, min_val=-180.0, max_val=180.0)
    pitch = safe_float(data.get("pitch", 0.0), 0.0, min_val=-90.0, max_val=90.0)
    fov = safe_float(data.get("fov", 90.0), 90.0, min_val=30.0, max_val=160.0)
    out_w = safe_int(data.get("output_width", 1920), 1920, min_val=320, max_val=7680)
    out_h = safe_int(data.get("output_height", 1080), 1080, min_val=240, max_val=4320)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "360_crop", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_360_crop(
        video_path=filepath,
        yaw=yaw, pitch=pitch, fov=fov,
        output_path=output,
        output_width=out_w, output_height=out_h,
        on_progress=_progress,
    )

    return {"output_path": result}


@video_vfx_bp.route("/video/360/stabilize", methods=["POST"])
@require_csrf
@async_job("360_stabilize")
def stabilize_360_route(job_id, filepath, data):
    """Stabilise a 360 video."""
    from opencut.core.video_360 import stabilize_360

    smoothing = safe_int(data.get("smoothing", 10), 10, min_val=1, max_val=30)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "stabilized_360", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = stabilize_360(
        video_path=filepath,
        output_path=output,
        smoothing=smoothing,
        on_progress=_progress,
    )

    return {"output_path": result}


# ===================================================================
# 4. Clean Plate Generation
# ===================================================================
@video_vfx_bp.route("/video/clean-plate", methods=["POST"])
@require_csrf
@async_job("clean_plate")
def clean_plate_route(job_id, filepath, data):
    """Generate a clean background plate from video."""
    from opencut.core.clean_plate import generate_clean_plate

    num_samples = safe_int(data.get("num_samples", 30), 30, min_val=3, max_val=300)
    do_inpaint = data.get("inpaint", True)
    if isinstance(do_inpaint, str):
        do_inpaint = do_inpaint.lower() in ("true", "1", "yes")
    inpaint_method = data.get("inpaint_method", "telea").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "clean_plate", effective_dir)
        # Change extension to .png for image output
        if output.endswith(".mp4"):
            output = output.rsplit(".", 1)[0] + ".png"

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_clean_plate(
        video_path=filepath,
        output_path=output,
        num_samples=num_samples,
        inpaint=do_inpaint,
        inpaint_method=inpaint_method,
        on_progress=_progress,
    )

    return result


# ===================================================================
# 5. Gaming Highlight Detection
# ===================================================================
@video_vfx_bp.route("/video/detect-highlights", methods=["POST"])
@require_csrf
@async_job("highlight_detect")
def detect_highlights_route(job_id, filepath, data):
    """Detect gaming highlight moments in a video."""
    from opencut.core.highlight_detect import detect_highlights

    top_n = safe_int(data.get("top_n", 5), 5, min_val=1, max_val=100)
    segment_duration = safe_float(data.get("segment_duration", 10.0), 10.0, min_val=2.0, max_val=120.0)
    min_score = safe_float(data.get("min_score", 0.3), 0.3, min_val=0.0, max_val=1.0)
    chat_log = data.get("chat_log_path", None)
    if chat_log:
        chat_log = validate_filepath(chat_log)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_highlights(
        video_path=filepath,
        top_n=top_n,
        segment_duration=segment_duration,
        min_score=min_score,
        chat_log_path=chat_log,
        on_progress=_progress,
    )

    return result


@video_vfx_bp.route("/video/extract-highlights", methods=["POST"])
@require_csrf
@async_job("highlight_extract")
def extract_highlights_route(job_id, filepath, data):
    """Extract highlight clips from a video."""
    from opencut.core.highlight_detect import extract_highlight_clips

    highlights = data.get("highlights", [])
    if not highlights:
        raise ValueError("No highlights provided")

    padding = safe_float(data.get("padding", 2.0), 2.0, min_val=0.0, max_val=30.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    if not output_dir:
        output_dir = ""

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    clips = extract_highlight_clips(
        video_path=filepath,
        highlights=highlights,
        output_dir=output_dir,
        padding=padding,
        on_progress=_progress,
    )

    return {"clips": clips, "count": len(clips)}


# ===================================================================
# 6. Deepfake Detection
# ===================================================================
@video_vfx_bp.route("/video/detect-deepfake", methods=["POST"])
@require_csrf
@async_job("deepfake_detect")
def detect_deepfake_route(job_id, filepath, data):
    """Run deepfake detection analysis on a video."""
    from opencut.core.deepfake_detect import detect_deepfake

    segment_duration = safe_float(data.get("segment_duration", 5.0), 5.0, min_val=1.0, max_val=60.0)
    threshold = safe_float(data.get("threshold", 0.5), 0.5, min_val=0.0, max_val=1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_deepfake(
        video_path=filepath,
        segment_duration=segment_duration,
        threshold=threshold,
        on_progress=_progress,
    )

    return result


@video_vfx_bp.route("/video/authenticity-report", methods=["POST"])
@require_csrf
@async_job("authenticity_report")
def authenticity_report_route(job_id, filepath, data):
    """Generate an authenticity report from deepfake detection."""
    from opencut.core.deepfake_detect import detect_deepfake, generate_authenticity_report

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None:
        effective_dir = _resolve_output_dir(filepath, output_dir) if output_dir else ""
        from opencut.helpers import output_path as _op
        output = _op(filepath, "authenticity_report", effective_dir)
        if output.endswith(".mp4"):
            output = output.rsplit(".", 1)[0] + ".json"

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    # First run detection
    _progress(5, "Running deepfake detection...")
    results = detect_deepfake(
        video_path=filepath,
        on_progress=lambda p, m="": _progress(5 + int(p * 0.8), m),
    )

    # Then generate report
    _progress(90, "Generating report...")
    report_path = generate_authenticity_report(results, output)

    return {"output_path": report_path, "verdict": results.get("verdict", "unknown")}


# ===================================================================
# 7. Face Tagging & Recognition
# ===================================================================
@video_vfx_bp.route("/video/detect-faces", methods=["POST"])
@require_csrf
@async_job("face_detect")
def detect_faces_route(job_id, filepath, data):
    """Detect faces in a video."""
    from opencut.core.face_tagging import detect_faces

    sample_rate = safe_float(data.get("sample_rate", 1.0), 1.0, min_val=0.1, max_val=30.0)
    min_face_size = safe_int(data.get("min_face_size", 60), 60, min_val=20, max_val=500)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_faces(
        video_path=filepath,
        sample_rate=sample_rate,
        min_face_size=min_face_size,
        on_progress=_progress,
    )

    return result


@video_vfx_bp.route("/video/cluster-faces", methods=["POST"])
@require_csrf
def cluster_faces_route():
    """Cluster face detections by identity."""
    try:
        from dataclasses import asdict

        from opencut.core.face_tagging import cluster_faces

        data = request.get_json(force=True) or {}
        embeddings = data.get("embeddings", [])
        if not embeddings:
            return jsonify({"error": "No embeddings provided"}), 400

        threshold = safe_float(data.get("distance_threshold", 0.6), 0.6, min_val=0.1, max_val=2.0)
        min_size = safe_int(data.get("min_cluster_size", 2), 2, min_val=1, max_val=100)

        clusters = cluster_faces(
            embeddings=embeddings,
            distance_threshold=threshold,
            min_cluster_size=min_size,
        )

        return jsonify({
            "clusters": [asdict(c) for c in clusters],
            "count": len(clusters),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "cluster_faces")


@video_vfx_bp.route("/video/tag-face", methods=["POST"])
@require_csrf
def tag_face_route():
    """Tag a face cluster with a name."""
    try:
        from opencut.core.face_tagging import tag_face_cluster

        data = request.get_json(force=True) or {}
        cluster_id = safe_int(data.get("cluster_id", -1), -1)
        name = data.get("name", "").strip()

        if cluster_id < 0:
            return jsonify({"error": "Valid cluster_id is required"}), 400
        if not name:
            return jsonify({"error": "Name is required"}), 400

        result = tag_face_cluster(cluster_id, name)

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "tag_face")


@video_vfx_bp.route("/video/search-face", methods=["POST"])
@require_csrf
def search_face_route():
    """Search for faces by name."""
    try:
        from opencut.core.face_tagging import search_by_face

        data = request.get_json(force=True) or {}
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Search name is required"}), 400

        results = search_by_face(name)

        return jsonify({"results": results, "count": len(results)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "search_face")


# ===================================================================
# 8. Holy Grail Timelapse
# ===================================================================
@video_vfx_bp.route("/video/holy-grail", methods=["POST"])
@require_csrf
@async_job("holy_grail", filepath_required=False)
def holy_grail_route(job_id, filepath, data):
    """Process a holy grail timelapse image sequence."""
    from opencut.core.holy_grail_timelapse import process_holy_grail

    image_paths = data.get("image_paths", [])
    if not image_paths:
        raise ValueError("No image_paths provided")

    # Validate all image paths
    validated = []
    for p in image_paths:
        validated.append(validate_filepath(p))

    config = data.get("config", {})
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = process_holy_grail(
        image_paths=validated,
        output_path=output,
        output_dir=output_dir,
        config=config,
        on_progress=_progress,
    )

    return result


@video_vfx_bp.route("/video/analyze-exposure", methods=["POST"])
@require_csrf
def analyze_exposure_route():
    """Analyse exposure ramp of an image sequence."""
    try:
        from opencut.core.holy_grail_timelapse import analyze_exposure_ramp

        data = request.get_json(force=True) or {}
        image_paths = data.get("image_paths", [])
        if not image_paths:
            return jsonify({"error": "No image_paths provided"}), 400

        validated = []
        for p in image_paths:
            validated.append(validate_filepath(p))

        result = analyze_exposure_ramp(validated)

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "analyze_exposure")
