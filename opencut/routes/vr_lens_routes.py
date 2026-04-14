"""
OpenCut VR & Lens Correction Routes

Endpoints for:
- VR/360 video stabilization (gyro + visual)
- Equirectangular reframing with keyframes
- Automatic FOV region extraction from 360 video
- Spatial audio encoding for VR
- Camera auto-detection and lens profile correction
- Chromatic aberration removal
- Lens profile listing
"""

import logging

from flask import Blueprint, jsonify

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

vr_lens_bp = Blueprint("vr_lens", __name__)


# ---------------------------------------------------------------------------
# POST /vr/stabilize — Stabilize 360 video
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/vr/stabilize", methods=["POST"])
@require_csrf
@async_job("vr_stabilize")
def vr_stabilize(job_id, filepath, data):
    """Stabilize a 360/VR video using gyro metadata or visual tracking."""
    from opencut.core.vr_stabilize import stabilize_vr

    smoothing = safe_int(data.get("smoothing", 10), 10, min_val=1, max_val=30)
    force_visual = safe_bool(data.get("force_visual", False), False)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "vr_stabilized", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = stabilize_vr(
        video_path=filepath,
        output_path_override=output,
        output_dir=output_dir,
        smoothing=smoothing,
        force_visual=force_visual,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "method": result.method,
        "gyro_samples": result.gyro_samples,
        "smoothing": result.smoothing,
        "frames_processed": result.frames_processed,
        "camera_model": result.camera_model,
    }


# ---------------------------------------------------------------------------
# POST /vr/reframe — Equirectangular to flat with keyframes
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/vr/reframe", methods=["POST"])
@require_csrf
@async_job("vr_reframe")
def vr_reframe(job_id, filepath, data):
    """Reframe 360 video to flat with optional keyframed camera motion."""
    from opencut.core.video_360 import equirect_to_flat, keyframed_reframe

    keyframes = data.get("keyframes", [])
    yaw = safe_float(data.get("yaw", 0.0), 0.0, min_val=-180.0, max_val=180.0)
    pitch = safe_float(data.get("pitch", 0.0), 0.0, min_val=-90.0, max_val=90.0)
    roll = safe_float(data.get("roll", 0.0), 0.0, min_val=-180.0, max_val=180.0)
    h_fov = safe_float(data.get("h_fov", 90.0), 90.0, min_val=30.0, max_val=160.0)
    w_fov = safe_float(data.get("w_fov", 0.0), 0.0, min_val=0.0, max_val=160.0)
    output_width = safe_int(data.get("output_width", 1920), 1920, min_val=320, max_val=7680)
    output_height = safe_int(data.get("output_height", 1080), 1080, min_val=240, max_val=4320)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "reframed", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if keyframes and isinstance(keyframes, list) and len(keyframes) > 0:
        result = keyframed_reframe(
            video_path=filepath,
            keyframes=keyframes,
            output_width=output_width,
            output_height=output_height,
            output_path_override=output,
            output_dir=output_dir,
            on_progress=_progress,
        )
    else:
        result = equirect_to_flat(
            video_path=filepath,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            h_fov=h_fov,
            w_fov=w_fov,
            output_width=output_width,
            output_height=output_height,
            output_path_override=output,
            output_dir=output_dir,
            on_progress=_progress,
        )

    return {
        "output_path": result.output_path,
        "width": result.width,
        "height": result.height,
        "keyframes_used": result.keyframes_used,
        "duration": result.duration,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# POST /vr/extract-fov — Auto-extract FOV regions
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/vr/extract-fov", methods=["POST"])
@require_csrf
@async_job("vr_extract_fov")
def vr_extract_fov(job_id, filepath, data):
    """Auto-detect and extract FOV regions from 360 video."""
    from opencut.core.video_360 import extract_fov_regions

    num_regions = safe_int(data.get("num_regions", 4), 4, min_val=1, max_val=8)
    h_fov = safe_float(data.get("h_fov", 90.0), 90.0, min_val=30.0, max_val=160.0)
    output_width = safe_int(data.get("output_width", 1920), 1920, min_val=320, max_val=7680)
    output_height = safe_int(data.get("output_height", 1080), 1080, min_val=240, max_val=4320)
    generate_xml = safe_bool(data.get("generate_xml", True), True)
    output_dir = data.get("output_dir", "")

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_fov_regions(
        video_path=filepath,
        output_dir=output_dir,
        num_regions=num_regions,
        h_fov=h_fov,
        output_width=output_width,
        output_height=output_height,
        generate_xml=generate_xml,
        on_progress=_progress,
    )

    return {
        "regions": [
            {
                "label": r.label,
                "yaw": r.yaw,
                "pitch": r.pitch,
                "h_fov": r.h_fov,
                "confidence": r.confidence,
                "output_path": r.output_path,
            }
            for r in result.regions
        ],
        "xml_path": result.xml_path,
        "output_dir": result.output_dir,
        "total_regions": result.total_regions,
        "duration": result.duration,
    }


# ---------------------------------------------------------------------------
# POST /vr/spatial-audio — Spatialize audio
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/vr/spatial-audio", methods=["POST"])
@require_csrf
@async_job("vr_spatial_audio")
def vr_spatial_audio(job_id, filepath, data):
    """Convert audio to spatial ambisonics for VR video."""
    from opencut.core.spatial_audio_vr import spatialize_audio

    speakers = data.get("speakers", None)
    auto_detect = safe_bool(data.get("auto_detect", True), True)
    sample_rate = safe_int(data.get("sample_rate", 48000), 48000,
                           min_val=22050, max_val=96000)
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "spatial_audio", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = spatialize_audio(
        video_path=filepath,
        speakers=speakers,
        auto_detect=auto_detect,
        output_path_override=output,
        output_dir=output_dir,
        sample_rate=sample_rate,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "speakers": result.speakers,
        "channels": result.channels,
        "format": result.format,
        "sample_rate": result.sample_rate,
        "muxed": result.muxed,
    }


# ---------------------------------------------------------------------------
# POST /lens/auto-detect — Detect camera + suggest corrections
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/lens/auto-detect", methods=["POST"])
@require_csrf
@async_job("lens_auto_detect")
def lens_auto_detect(job_id, filepath, data):
    """Auto-detect camera model and suggest lens corrections."""
    from opencut.core.lens_correction import auto_detect_camera
    from opencut.core.lens_profile import get_lens_info

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Reading camera metadata...")
    lens_info = get_lens_info(filepath)

    _progress(50, "Detecting camera profile...")
    detection = auto_detect_camera(filepath)

    _progress(100, "Detection complete")

    profile_data = None
    if detection.profile:
        p = detection.profile
        profile_data = {
            "camera_model": p.camera_model,
            "lens_model": p.lens_model,
            "k1": p.k1,
            "k2": p.k2,
            "focal_length_mm": p.focal_length_mm,
            "fov_degrees": p.fov_degrees,
            "category": p.category,
        }

    return {
        "detected": detection.detected,
        "camera_model": detection.camera_model,
        "lens_model": detection.lens_model,
        "profile": profile_data,
        "confidence": detection.confidence,
        "suggested_k1": detection.suggested_k1,
        "suggested_k2": detection.suggested_k2,
        "lens_info": {
            "camera_make": lens_info.camera_make,
            "camera_model": lens_info.camera_model,
            "lens_model": lens_info.lens_model,
            "focal_length_mm": lens_info.focal_length_mm,
            "aperture": lens_info.aperture,
            "iso": lens_info.iso,
            "resolution": lens_info.resolution,
            "frame_rate": lens_info.frame_rate,
            "codec": lens_info.codec,
            "creation_date": lens_info.creation_date,
        },
    }


# ---------------------------------------------------------------------------
# POST /lens/correct-distortion — Apply lens profile correction
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/lens/correct-distortion", methods=["POST"])
@require_csrf
@async_job("lens_correct_distortion")
def lens_correct_distortion(job_id, filepath, data):
    """Apply lens distortion correction using a profile or manual k1/k2."""
    from opencut.core.lens_correction import (
        correct_lens_distortion,
        correct_with_profile,
    )

    profile_id = data.get("profile_id", "").strip() or None
    auto_detect = safe_bool(data.get("auto_detect", False), False)
    k1 = data.get("k1", None)
    k2 = data.get("k2", None)
    preset = data.get("preset", "").strip() or None
    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "lens_corrected", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    # Use profile-based correction if profile_id or auto_detect
    if profile_id or auto_detect:
        result = correct_with_profile(
            input_path=filepath,
            profile_id=profile_id,
            auto_detect=auto_detect,
            output_path_override=output,
            on_progress=_progress,
        )
        return result

    # Manual k1/k2 or preset
    if k1 is not None:
        k1 = safe_float(k1, -0.1, min_val=-1.0, max_val=1.0)
    if k2 is not None:
        k2 = safe_float(k2, 0.0, min_val=-1.0, max_val=1.0)

    result = correct_lens_distortion(
        input_path=filepath,
        output_path_override=output,
        k1=k1,
        k2=k2,
        preset=preset,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /lens/chromatic-aberration — Remove CA
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/lens/chromatic-aberration", methods=["POST"])
@require_csrf
@async_job("lens_chromatic_aberration")
def lens_chromatic_aberration(job_id, filepath, data):
    """Detect and remove chromatic aberration."""
    from opencut.core.chromatic_aberration import correct_chromatic_aberration

    auto_detect = safe_bool(data.get("auto_detect", True), True)
    red_shift_x = data.get("red_shift_x", None)
    red_shift_y = data.get("red_shift_y", None)
    blue_shift_x = data.get("blue_shift_x", None)
    blue_shift_y = data.get("blue_shift_y", None)

    if red_shift_x is not None:
        red_shift_x = safe_int(red_shift_x, 0, min_val=-10, max_val=10)
    if red_shift_y is not None:
        red_shift_y = safe_int(red_shift_y, 0, min_val=-10, max_val=10)
    if blue_shift_x is not None:
        blue_shift_x = safe_int(blue_shift_x, 0, min_val=-10, max_val=10)
    if blue_shift_y is not None:
        blue_shift_y = safe_int(blue_shift_y, 0, min_val=-10, max_val=10)

    output_dir = data.get("output_dir", "")
    output = data.get("output_path", None) or None

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "ca_corrected", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = correct_chromatic_aberration(
        video_path=filepath,
        red_shift_x=red_shift_x,
        red_shift_y=red_shift_y,
        blue_shift_x=blue_shift_x,
        blue_shift_y=blue_shift_y,
        auto_detect=auto_detect,
        output_path_override=output,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "red_shift_x": result.red_shift_x,
        "red_shift_y": result.red_shift_y,
        "blue_shift_x": result.blue_shift_x,
        "blue_shift_y": result.blue_shift_y,
        "method": result.method,
        "severity": result.severity,
    }


# ---------------------------------------------------------------------------
# GET /lens/profiles — List available camera profiles
# ---------------------------------------------------------------------------
@vr_lens_bp.route("/api/lens/profiles", methods=["GET"])
def lens_profiles():
    """List all available camera lens profiles."""
    from flask import request
    from opencut.core.lens_correction import list_camera_profiles, list_lens_presets

    category = request.args.get("category", None)
    include_presets = request.args.get("include_presets", "true").lower() == "true"

    profiles = list_camera_profiles(category=category)

    result = {
        "profiles": profiles,
        "total": len(profiles),
        "categories": list(set(p["category"] for p in profiles)),
    }

    if include_presets:
        result["presets"] = list_lens_presets()

    return jsonify(result)
