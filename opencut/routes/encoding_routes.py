"""
OpenCut Encoding & Export Routes

Blueprint ``encoding_bp`` provides endpoints for ProRes, AV1, DNxHR
exports and batch transcode operations.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir, _unique_output_path
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

encoding_bp = Blueprint("encoding", __name__)


# ===================================================================
# ProRes
# ===================================================================

@encoding_bp.route("/export/prores", methods=["POST"])
@require_csrf
@async_job("prores_export")
def prores_export_route(job_id, filepath, data):
    """Export a video in Apple ProRes format."""
    profile = data.get("profile", "422hq").strip().lower()
    include_alpha = safe_bool(data.get("include_alpha"), default=False)
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    resolved_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(resolved_dir, f"{base}_prores_{profile}.mov")
    out_path = _unique_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.prores_export import export_prores

    result = export_prores(
        input_path=filepath,
        profile=profile,
        output_path_override=out_path,
        include_alpha=include_alpha,
        on_progress=_on_progress,
    )

    return result


@encoding_bp.route("/export/prores/profiles", methods=["GET"])
def prores_profiles_route():
    """Return available ProRes profiles with encoder availability."""
    try:
        from opencut.core.prores_export import get_prores_profiles
        profiles = get_prores_profiles()
        return jsonify({"profiles": profiles})
    except Exception as exc:
        return safe_error(exc, "prores_profiles")


# ===================================================================
# AV1
# ===================================================================

@encoding_bp.route("/export/av1", methods=["POST"])
@require_csrf
@async_job("av1_export")
def av1_export_route(job_id, filepath, data):
    """Export a video using AV1 encoding."""
    encoder = data.get("encoder", "auto").strip().lower()
    quality = data.get("quality", "balanced").strip().lower()
    crf = int(data.get("crf", 28))
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    resolved_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(resolved_dir, f"{base}_av1_{quality}.mp4")
    out_path = _unique_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.av1_export import export_av1

    result = export_av1(
        input_path=filepath,
        encoder=encoder,
        quality=quality,
        crf=crf,
        output_path_override=out_path,
        on_progress=_on_progress,
    )

    return result


@encoding_bp.route("/export/av1/encoders", methods=["GET"])
def av1_encoders_route():
    """Return available AV1 encoders."""
    try:
        from opencut.core.av1_export import detect_av1_encoders
        encoders = detect_av1_encoders()
        return jsonify({"encoders": encoders})
    except Exception as exc:
        return safe_error(exc, "av1_encoders")


# ===================================================================
# DNxHR
# ===================================================================

@encoding_bp.route("/export/dnxhr", methods=["POST"])
@require_csrf
@async_job("dnxhr_export")
def dnxhr_export_route(job_id, filepath, data):
    """Export a video in Avid DNxHR format."""
    profile = data.get("profile", "dnxhr_hq").strip().lower()
    container = data.get("container", "mov").strip().lower()
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    ext = f".{container}" if container in ("mov", "mxf") else ".mov"
    resolved_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(resolved_dir, f"{base}_{profile}{ext}")
    out_path = _unique_output_path(out_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.dnx_export import export_dnxhr

    result = export_dnxhr(
        input_path=filepath,
        profile=profile,
        container=container,
        output_path_override=out_path,
        on_progress=_on_progress,
    )

    return result


@encoding_bp.route("/export/dnxhr/profiles", methods=["GET"])
def dnxhr_profiles_route():
    """Return available DNxHR profiles with encoder availability."""
    try:
        from opencut.core.dnx_export import get_dnxhr_profiles
        profiles = get_dnxhr_profiles()
        return jsonify({"profiles": profiles})
    except Exception as exc:
        return safe_error(exc, "dnxhr_profiles")


# ===================================================================
# Batch Transcode
# ===================================================================

@encoding_bp.route("/batch/transcode", methods=["POST"])
@require_csrf
@async_job("batch_transcode", filepath_required=False)
def batch_transcode_route(job_id, filepath, data):
    """Batch transcode files across multiple presets."""
    file_paths = data.get("file_paths", [])
    preset_names = data.get("preset_names", [])
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    parallel = int(data.get("parallel", 1))

    if not isinstance(file_paths, list) or not file_paths:
        raise ValueError("file_paths must be a non-empty list of file paths.")
    if not isinstance(preset_names, list) or not preset_names:
        raise ValueError("preset_names must be a non-empty list of preset names.")

    # Validate each file path
    validated_paths = []
    for fp in file_paths:
        if isinstance(fp, str) and fp.strip():
            validated_paths.append(validate_filepath(fp.strip()))

    if not validated_paths:
        raise ValueError("No valid file paths provided.")

    # Resolve output directory from first file if not specified
    if output_dir:
        resolved_dir = _resolve_output_dir(validated_paths[0], output_dir)
    else:
        resolved_dir = None  # batch_transcode will use per-file dirs

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.batch_transcode import batch_transcode

    result = batch_transcode(
        file_paths=validated_paths,
        preset_names=preset_names,
        output_base_dir=resolved_dir,
        parallel=parallel,
        on_progress=_on_progress,
    )

    return result.to_dict()


@encoding_bp.route("/batch/transcode/estimate", methods=["POST"])
@require_csrf
def batch_transcode_estimate_route():
    """Estimate output size and time for a batch transcode job."""
    try:
        data = request.get_json(force=True) or {}
        file_paths = data.get("file_paths", [])
        preset_names = data.get("preset_names", [])

        if not isinstance(file_paths, list) or not file_paths:
            return jsonify({"error": "file_paths must be a non-empty list"}), 400
        if not isinstance(preset_names, list) or not preset_names:
            return jsonify({"error": "preset_names must be a non-empty list"}), 400

        from opencut.core.batch_transcode import estimate_batch_size

        estimate = estimate_batch_size(
            file_paths=file_paths,
            preset_names=preset_names,
        )

        return jsonify(estimate)
    except Exception as exc:
        return safe_error(exc, "batch_transcode_estimate")
