"""
OpenCut Video Processing Routes

Blueprint for HDR tone mapping, multi-camera audio sync, video repair,
rolling shutter correction, advanced stabilization, and color space
detection/conversion.
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
)

logger = logging.getLogger("opencut")

video_proc_bp = Blueprint("video_proc", __name__)


# ---------------------------------------------------------------------------
# HDR Detection (sync)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/hdr/detect", methods=["POST"])
@require_csrf
def hdr_detect():
    """Detect HDR metadata in a video file (synchronous)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.hdr_tools import detect_and_suggest
        result = detect_and_suggest(filepath)
        return jsonify(result)
    except Exception as e:
        return safe_error(e, "hdr_detect")


# ---------------------------------------------------------------------------
# HDR Tone Mapping (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/hdr/tonemap", methods=["POST"])
@require_csrf
@async_job("hdr_tonemap")
def hdr_tonemap(job_id, filepath, data):
    """Tone map an HDR video to SDR."""
    algorithm = data.get("algorithm", "hable").strip().lower()
    _resolve_output_dir(filepath, data.get("output_dir", ""))

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.hdr_tools import tonemap_hdr_to_sdr
    result = tonemap_hdr_to_sdr(
        filepath,
        algorithm=algorithm,
        output_path_override=None,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Multi-Camera Audio Sync (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/audio-sync", methods=["POST"])
@require_csrf
@async_job("audio_sync")
def audio_sync(job_id, filepath, data):
    """Compute sync offset between reference and target audio."""
    target_path = data.get("target_path", "").strip()
    if not target_path:
        raise ValueError("No target_path provided")
    target_path = validate_filepath(target_path)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.audio_sync import compute_sync_offset
    result = compute_sync_offset(filepath, target_path, on_progress=_p)
    return {
        "offset_seconds": result.offset_seconds,
        "confidence": result.confidence,
        "method": result.method,
    }


@video_proc_bp.route("/video/audio-sync/multi", methods=["POST"])
@require_csrf
@async_job("audio_sync_multi")
def audio_sync_multi(job_id, filepath, data):
    """Compute sync offsets for multiple targets against one reference."""
    target_paths = data.get("target_paths", [])
    if not target_paths or not isinstance(target_paths, list):
        raise ValueError("No target_paths provided (expected a list)")
    validated = [validate_filepath(p.strip()) for p in target_paths if isinstance(p, str) and p.strip()]
    if not validated:
        raise ValueError("No valid target paths after validation")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.audio_sync import sync_multiple
    results = sync_multiple(filepath, validated, on_progress=_p)
    return {
        "results": [
            {
                "offset_seconds": r.offset_seconds,
                "confidence": r.confidence,
                "method": r.method,
            }
            for r in results
        ]
    }


# ---------------------------------------------------------------------------
# Corrupted Video Repair - Diagnose (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/repair/diagnose", methods=["POST"])
@require_csrf
@async_job("repair_diagnose")
def repair_diagnose(job_id, filepath, data):
    """Diagnose video corruption."""
    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.video_repair import diagnose_corruption
    diag = diagnose_corruption(filepath, on_progress=_p)
    return {
        "corruption_type": diag.corruption_type,
        "severity": diag.severity,
        "description": diag.description,
        "recoverable": diag.recoverable,
        "suggested_action": diag.suggested_action,
    }


# ---------------------------------------------------------------------------
# Corrupted Video Repair (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/repair", methods=["POST"])
@require_csrf
@async_job("repair_video")
def repair_video_route(job_id, filepath, data):
    """Repair a corrupted video file."""
    reference_path = data.get("reference_path", "").strip()
    if reference_path:
        reference_path = validate_filepath(reference_path)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.video_repair import repair_video
    result = repair_video(
        filepath,
        reference_path=reference_path or None,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Rolling Shutter Correction (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/rolling-shutter", methods=["POST"])
@require_csrf
@async_job("rolling_shutter")
def rolling_shutter(job_id, filepath, data):
    """Correct rolling shutter artifacts."""
    strength = safe_float(data.get("strength"), 0.5)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.rolling_shutter import correct_rolling_shutter
    result = correct_rolling_shutter(filepath, strength=strength, on_progress=_p)
    return result


# ---------------------------------------------------------------------------
# Advanced Stabilization (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/stabilize-advanced", methods=["POST"])
@require_csrf
@async_job("stabilize_advanced")
def stabilize_advanced_route(job_id, filepath, data):
    """Advanced two-pass video stabilization."""
    mode = data.get("mode", "smooth").strip().lower()
    smoothing = safe_int(data.get("smoothing"), 30)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.advanced_stabilize import stabilize_advanced
    result = stabilize_advanced(
        filepath,
        mode=mode,
        smoothing=smoothing,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Color Space Detection (sync)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/colorspace/detect", methods=["POST"])
@require_csrf
def colorspace_detect():
    """Detect color space metadata in a video file (synchronous)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        from opencut.core.colorspace import detect_colorspace
        info = detect_colorspace(filepath)
        return jsonify({
            "primaries": info.primaries,
            "transfer": info.transfer,
            "matrix": info.matrix,
            "bit_depth": info.bit_depth,
            "is_hdr": info.is_hdr,
            "is_wide_gamut": info.is_wide_gamut,
            "profile_name": info.profile_name,
        })
    except Exception as e:
        return safe_error(e, "colorspace_detect")


# ---------------------------------------------------------------------------
# Color Space Conversion (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/colorspace/convert", methods=["POST"])
@require_csrf
@async_job("colorspace_convert")
def colorspace_convert(job_id, filepath, data):
    """Convert video color space."""
    target_primaries = data.get("target_primaries", "bt709").strip()
    target_transfer = data.get("target_transfer", "bt709").strip()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.colorspace import convert_colorspace
    result = convert_colorspace(
        filepath,
        target_primaries=target_primaries,
        target_transfer=target_transfer,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Batch Color Space Detection (async)
# ---------------------------------------------------------------------------
@video_proc_bp.route("/video/colorspace/batch-detect", methods=["POST"])
@require_csrf
@async_job("colorspace_batch_detect", filepath_required=False)
def colorspace_batch_detect(job_id, filepath, data):
    """Detect color space for multiple files."""
    file_paths = data.get("file_paths", [])
    if not file_paths or not isinstance(file_paths, list):
        raise ValueError("No file_paths provided (expected a list)")
    validated = [validate_filepath(p.strip()) for p in file_paths if isinstance(p, str) and p.strip()]
    if not validated:
        raise ValueError("No valid file paths after validation")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.colorspace import batch_detect_colorspace
    results = batch_detect_colorspace(validated, on_progress=_p)
    return {
        "results": [
            {
                "file_path": fp,
                "primaries": info.primaries,
                "transfer": info.transfer,
                "matrix": info.matrix,
                "bit_depth": info.bit_depth,
                "is_hdr": info.is_hdr,
                "is_wide_gamut": info.is_wide_gamut,
                "profile_name": info.profile_name,
            }
            for fp, info in zip(validated, results)
        ]
    }
