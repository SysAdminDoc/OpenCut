"""
OpenCut Video Effects & Color Routes

Endpoints for sky replacement, LOG profile detection/application,
LUT stacking, display calibration, cinemagraph creation, hyperlapse
stabilization, and lossless intermediate codec pipeline.
"""

import logging
import os
import tempfile

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
    validate_path,
)

logger = logging.getLogger("opencut")

video_effects_bp = Blueprint("video_effects", __name__)


# ---------------------------------------------------------------------------
# Sky Replacement
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/video/sky-replace", methods=["POST"])
@require_csrf
@async_job("sky_replace")
def sky_replace(job_id, filepath, data):
    """Replace sky in a video with a provided sky image/video."""
    from opencut.core.sky_replace import replace_sky

    sky_source = data.get("sky_source", "").strip()
    if not sky_source:
        raise ValueError("No sky_source path provided")
    sky_source = validate_filepath(sky_source)

    method = data.get("method", "brightness").strip()
    threshold = safe_float(data.get("threshold", 0.55), 0.55, min_val=0.1, max_val=0.95)
    blue_weight = safe_float(data.get("blue_weight", 0.3), 0.3, min_val=0.0, max_val=1.0)
    feather = safe_int(data.get("feather", 15), 15, min_val=0, max_val=100)
    adjust_lighting = safe_bool(data.get("adjust_lighting", True), True)
    lighting_strength = safe_float(data.get("lighting_strength", 0.6), 0.6,
                                   min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "sky_replaced", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = replace_sky(
        video_path=filepath,
        sky_source=sky_source,
        output_path_str=output,
        output_dir=output_dir,
        method=method,
        threshold=threshold,
        blue_weight=blue_weight,
        feather=feather,
        adjust_lighting=adjust_lighting,
        lighting_strength=lighting_strength,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "frames_processed": result.frames_processed,
        "avg_sky_fraction": result.avg_sky_fraction,
        "foreground_adjusted": result.foreground_adjusted,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# LOG Profile Detection
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/video/log-detect", methods=["POST"])
@require_csrf
@async_job("log_detect")
def log_detect(job_id, filepath, data):
    """Detect LOG camera profile from video metadata."""
    from opencut.core.log_profiles import detect_log_profile

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_log_profile(
        video_path=filepath,
        on_progress=_progress,
    )

    return {
        "detected_profile": result.detected_profile,
        "profile_name": result.profile_name,
        "camera": result.camera,
        "confidence": result.confidence,
        "color_trc": result.color_trc,
        "color_primaries": result.color_primaries,
        "metadata": result.metadata,
    }


# ---------------------------------------------------------------------------
# Apply IDT (Input Display Transform)
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/video/log-apply", methods=["POST"])
@require_csrf
@async_job("log_apply_idt")
def log_apply_idt(job_id, filepath, data):
    """Apply an Input Display Transform for a LOG profile."""
    from opencut.core.log_profiles import apply_idt

    profile = data.get("profile", "").strip()
    if not profile:
        raise ValueError("No LOG profile specified")

    lut_size = safe_int(data.get("lut_size", 33), 33, min_val=17, max_val=65)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"idt_{profile}", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_idt(
        video_path=filepath,
        profile=profile,
        output_path_str=output,
        output_dir=output_dir,
        lut_size=lut_size,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "profile": result.profile,
        "lut_path": result.lut_path,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# LUT Stacking
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/video/lut-stack", methods=["POST"])
@require_csrf
@async_job("lut_stack")
def lut_stack(job_id, filepath, data):
    """Apply multiple LUTs sequentially (technical + creative)."""
    from opencut.core.log_profiles import stack_luts

    lut_paths = data.get("lut_paths", [])
    if not lut_paths or not isinstance(lut_paths, list):
        raise ValueError("No lut_paths array provided")

    # Validate each LUT path
    validated_paths = []
    for lp in lut_paths:
        if isinstance(lp, str) and lp.strip():
            validated_paths.append(validate_filepath(lp.strip()))

    if not validated_paths:
        raise ValueError("No valid LUT file paths provided")

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "graded", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = stack_luts(
        video_path=filepath,
        lut_paths=validated_paths,
        output_path_str=output,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "luts_applied": result.luts_applied,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# Display Test Pattern Generation
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/display/test-pattern", methods=["POST"])
@require_csrf
@async_job("test_pattern", filepath_required=False)
def display_test_pattern(job_id, filepath, data):
    """Generate a display calibration test pattern."""
    from opencut.core.display_calibration import (
        generate_gamut_test,
        generate_grayscale_ramp,
        generate_smpte_bars,
        get_verification_guide,
    )

    pattern_type = data.get("pattern", "smpte_bars").strip()
    width = safe_int(data.get("width", 1920), 1920, min_val=160, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=120, max_val=4320)
    resolution = (width, height)
    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=1.0, max_val=60.0)
    output_format = data.get("format", "png").strip().lower()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    # Determine output directory
    if output_dir:
        out_dir = _resolve_output_dir(output_dir, output_dir)
    else:
        out_dir = os.path.join(tempfile.gettempdir(), "opencut_calibration")
    os.makedirs(out_dir, exist_ok=True)

    ext = ".mp4" if output_format in ("mp4", "video") else ".png"
    output_path = os.path.join(out_dir, f"test_{pattern_type}{ext}")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if pattern_type == "smpte_bars":
        result = generate_smpte_bars(
            output_path_str=output_path,
            resolution=resolution,
            duration=duration,
            on_progress=_progress,
        )
    elif pattern_type == "grayscale_ramp":
        steps = safe_int(data.get("steps", 32), 32, min_val=8, max_val=256)
        result = generate_grayscale_ramp(
            output_path_str=output_path,
            resolution=resolution,
            steps=steps,
            duration=duration,
            on_progress=_progress,
        )
    elif pattern_type == "gamut_test":
        include_skin = safe_bool(data.get("include_skin_tones", True), True)
        result = generate_gamut_test(
            output_path_str=output_path,
            resolution=resolution,
            include_skin_tones=include_skin,
            duration=duration,
            on_progress=_progress,
        )
    elif pattern_type == "guide":
        guide = get_verification_guide()
        return {"guide": guide}
    else:
        raise ValueError(
            f"Unknown pattern type: {pattern_type}. "
            f"Supported: smpte_bars, grayscale_ramp, gamut_test, guide"
        )

    return {
        "output_path": result.output_path,
        "pattern_type": result.pattern_type,
        "resolution": list(result.resolution),
        "description": result.description,
    }


# ---------------------------------------------------------------------------
# Cinemagraph
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/video/cinemagraph", methods=["POST"])
@require_csrf
@async_job("cinemagraph")
def cinemagraph(job_id, filepath, data):
    """Create a cinemagraph from a video clip."""
    from opencut.core.cinemagraph import create_cinemagraph

    mask_data = data.get("mask", data.get("mask_data", {}))
    if not mask_data or not isinstance(mask_data, dict):
        raise ValueError("No mask/mask_data provided. "
                         "Use {\"type\": \"rect\", \"x\": 0, \"y\": 0, \"w\": 100, \"h\": 100}")

    loop_duration = safe_float(data.get("loop_duration", 3.0), 3.0,
                               min_val=0.5, max_val=30.0)
    start_time = safe_float(data.get("start_time", 0.0), 0.0, min_val=0.0)
    crossfade = safe_float(data.get("crossfade", 0.5), 0.5, min_val=0.0, max_val=5.0)
    output_format = data.get("output_format", "mp4").strip().lower()
    ref_timestamp = data.get("ref_timestamp", None)
    if ref_timestamp is not None:
        ref_timestamp = safe_float(ref_timestamp, 0.0, min_val=0.0)

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_cinemagraph(
        video_path=filepath,
        mask_data=mask_data,
        loop_duration=loop_duration,
        start_time=start_time,
        crossfade=crossfade,
        output_path_str=output,
        output_dir=output_dir,
        output_format=output_format,
        ref_timestamp=ref_timestamp,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "frames_written": result.frames_written,
        "loop_duration": result.loop_duration,
        "crossfade_frames": result.crossfade_frames,
        "resolution": list(result.resolution),
    }


# ---------------------------------------------------------------------------
# Hyperlapse
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/video/hyperlapse", methods=["POST"])
@require_csrf
@async_job("hyperlapse")
def hyperlapse(job_id, filepath, data):
    """Create a stabilized hyperlapse from video footage."""
    from opencut.core.hyperlapse import create_hyperlapse

    speed_factor = safe_float(data.get("speed_factor", 10.0), 10.0,
                              min_val=1.0, max_val=100.0)
    smoothing = safe_int(data.get("smoothing", 45), 45, min_val=1, max_val=200)
    edge_fill = data.get("edge_fill", "mirror").strip()
    stabilize = safe_bool(data.get("stabilize", True), True)
    passes = safe_int(data.get("passes", 2), 2, min_val=1, max_val=3)

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "hyperlapse", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = create_hyperlapse(
        video_path=filepath,
        speed_factor=speed_factor,
        smoothing=smoothing,
        edge_fill=edge_fill,
        output_path_str=output,
        output_dir=output_dir,
        stabilize=stabilize,
        passes=passes,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "speed_factor": result.speed_factor,
        "original_duration": result.original_duration,
        "output_duration": result.output_duration,
        "frames_sampled": result.frames_sampled,
        "stabilization_smoothing": result.stabilization_smoothing,
        "edge_fill": result.edge_fill,
    }


# ---------------------------------------------------------------------------
# Lossless Intermediate Codec Pipeline
# ---------------------------------------------------------------------------
@video_effects_bp.route("/api/encoding/intermediate", methods=["POST"])
@require_csrf
@async_job("lossless_intermediate")
def lossless_intermediate(job_id, filepath, data):
    """Convert to/from lossless intermediate codec."""
    from opencut.core.lossless_intermediate import (
        from_intermediate,
        get_recommended_codec,
        list_intermediate_codecs,
        to_intermediate,
    )

    direction = data.get("direction", "to").strip().lower()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if direction == "to":
        codec = data.get("codec", "ffv1").strip().lower()
        pix_fmt = data.get("pix_fmt", None)
        if isinstance(pix_fmt, str):
            pix_fmt = pix_fmt.strip() or None

        result = to_intermediate(
            video_path=filepath,
            codec=codec,
            pix_fmt=pix_fmt,
            output_path_str=output,
            output_dir=output_dir,
            on_progress=_progress,
        )
        return {
            "output_path": result.output_path,
            "codec": result.codec,
            "pix_fmt": result.pix_fmt,
            "file_size_mb": result.file_size_mb,
            "duration": result.duration,
            "lossless": result.lossless,
        }

    elif direction == "from":
        delivery_codec = data.get("delivery_codec", "h264_web").strip().lower()

        result = from_intermediate(
            intermediate_path=filepath,
            delivery_codec=delivery_codec,
            output_path_str=output,
            output_dir=output_dir,
            on_progress=_progress,
        )
        return {
            "output_path": result.output_path,
            "source_codec": result.source_codec,
            "delivery_codec": result.delivery_codec,
            "delivery_preset": result.delivery_preset,
            "file_size_mb": result.file_size_mb,
        }

    elif direction == "list":
        return {"codecs": list_intermediate_codecs()}

    elif direction == "recommend":
        use_case = data.get("use_case", "general").strip()
        return get_recommended_codec(use_case)

    else:
        raise ValueError(
            f"Unknown direction: {direction}. "
            f"Supported: to, from, list, recommend"
        )
