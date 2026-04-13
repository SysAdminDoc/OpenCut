"""
OpenCut Professional / Broadcast Routes

Film stock emulation, glitch effects, duplicate detection,
fit-to-fill, wide gamut workflow, MXF container support.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    validate_filepath,
)

logger = logging.getLogger("opencut")

professional_bp = Blueprint("professional", __name__)


# ===================================================================
# Film Stock Emulation
# ===================================================================

@professional_bp.route("/effects/film-stocks", methods=["GET"])
def list_film_stocks():
    """Return available film stock presets."""
    try:
        from opencut.core.film_emulation import list_film_stocks
        return jsonify({"stocks": list_film_stocks()})
    except Exception as e:
        return safe_error(e, "list_film_stocks")


@professional_bp.route("/effects/film-stock", methods=["POST"])
@require_csrf
@async_job("film_stock")
def apply_film_stock_route(job_id, filepath, data):
    """Apply a film stock emulation to a video."""
    stock = data.get("stock", "kodak_portra_400").strip()
    grain_amount = safe_float(data.get("grain_amount", 0.5), 0.5, min_val=0.0, max_val=1.0)
    halation = safe_float(data.get("halation", 0.3), 0.3, min_val=0.0, max_val=1.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.film_emulation import apply_film_stock

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_film_{stock}.mp4")

    result = apply_film_stock(
        filepath,
        stock=stock,
        grain_amount=grain_amount,
        halation=halation,
        output_path_override=out_path,
        on_progress=_p,
    )
    return result


# ===================================================================
# Glitch Effects
# ===================================================================

@professional_bp.route("/effects/glitch", methods=["POST"])
@require_csrf
@async_job("glitch")
def apply_glitch_route(job_id, filepath, data):
    """Apply a glitch effect to a video."""
    effect = data.get("effect", "rgb_split").strip()
    intensity = safe_float(data.get("intensity", 0.5), 0.5, min_val=0.0, max_val=1.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.glitch_effects import apply_glitch

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_glitch_{effect}.mp4")

    result = apply_glitch(
        filepath,
        effect=effect,
        intensity=intensity,
        output_path_override=out_path,
        on_progress=_p,
    )
    return result


@professional_bp.route("/effects/glitch-sequence", methods=["POST"])
@require_csrf
@async_job("glitch_sequence")
def apply_glitch_sequence_route(job_id, filepath, data):
    """Apply a sequence of timed glitch effects."""
    effects_timeline = data.get("effects_timeline", [])
    if not effects_timeline:
        raise ValueError("No effects_timeline provided")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.glitch_effects import apply_glitch_sequence

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_glitch_seq.mp4")

    result = apply_glitch_sequence(
        filepath,
        effects_timeline=effects_timeline,
        output_path_override=out_path,
        on_progress=_p,
    )
    return result


# ===================================================================
# Duplicate Detection
# ===================================================================

@professional_bp.route("/media/find-duplicates", methods=["POST"])
@require_csrf
@async_job("find_duplicates", filepath_required=False)
def find_duplicates_route(job_id, filepath, data):
    """Find duplicate/near-duplicate videos from a list of paths."""
    file_paths = data.get("file_paths", [])
    if not file_paths or not isinstance(file_paths, list):
        raise ValueError("Provide 'file_paths' as a list of video paths")
    if len(file_paths) < 2:
        raise ValueError("Need at least 2 files to compare")
    if len(file_paths) > 500:
        raise ValueError("Maximum 500 files per scan")

    # Validate each path
    validated = []
    for fp in file_paths:
        if isinstance(fp, str) and fp.strip():
            validated.append(validate_filepath(fp.strip()))

    if len(validated) < 2:
        raise ValueError("Need at least 2 valid file paths")

    threshold = safe_float(data.get("threshold", 0.85), 0.85, min_val=0.5, max_val=1.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.duplicate_detect import find_duplicates

    groups = find_duplicates(
        validated,
        threshold=threshold,
        on_progress=_p,
    )
    return {"duplicate_groups": groups, "total_groups": len(groups)}


# ===================================================================
# Fit-to-Fill
# ===================================================================

@professional_bp.route("/video/fit-to-fill", methods=["POST"])
@require_csrf
@async_job("fit_to_fill")
def fit_to_fill_route(job_id, filepath, data):
    """Fit video to a target duration by adjusting speed."""
    target_duration = safe_float(data.get("target_duration", 0), 0, min_val=0.1, max_val=86400)
    if target_duration <= 0:
        raise ValueError("target_duration must be a positive number (seconds)")

    method = data.get("method", "auto").strip()
    if method not in ("uniform", "eased", "auto"):
        method = "auto"

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.fit_to_fill import fit_to_fill

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_fit_to_fill.mp4")

    result = fit_to_fill(
        filepath,
        target_duration=target_duration,
        method=method,
        output_path_override=out_path,
        on_progress=_p,
    )
    return result


# ===================================================================
# Wide Gamut Workflow
# ===================================================================

@professional_bp.route("/video/gamut/detect", methods=["POST"])
@require_csrf
def detect_gamut_route():
    """Detect color gamut of a video file (sync)."""
    try:
        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        from opencut.core.wide_gamut import detect_gamut
        result = detect_gamut(filepath)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "detect_gamut")


@professional_bp.route("/video/gamut/convert", methods=["POST"])
@require_csrf
@async_job("gamut_convert")
def convert_gamut_route(job_id, filepath, data):
    """Convert video to a different color gamut."""
    target_gamut = data.get("target_gamut", "bt709").strip()
    intent = data.get("intent", "perceptual").strip()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.wide_gamut import convert_gamut

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_gamut_{target_gamut}.mp4")

    result = convert_gamut(
        filepath,
        target_gamut=target_gamut,
        intent=intent,
        output_path_override=out_path,
        on_progress=_p,
    )
    return result


@professional_bp.route("/video/gamut/check-clipping", methods=["POST"])
@require_csrf
@async_job("gamut_clipping")
def check_gamut_clipping_route(job_id, filepath, data):
    """Check for gamut clipping when converting to a target gamut."""
    target_gamut = data.get("target_gamut", "bt709").strip()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.wide_gamut import check_gamut_clipping

    result = check_gamut_clipping(
        filepath,
        target_gamut=target_gamut,
        on_progress=_p,
    )
    return result


# ===================================================================
# MXF Container Support
# ===================================================================

@professional_bp.route("/mxf/probe", methods=["POST"])
@require_csrf
def probe_mxf_route():
    """Probe an MXF file for metadata (sync)."""
    try:
        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        filepath = validate_filepath(filepath)

        from opencut.core.mxf_support import probe_mxf
        result = probe_mxf(filepath)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "probe_mxf")


@professional_bp.route("/mxf/export", methods=["POST"])
@require_csrf
@async_job("mxf_export")
def export_mxf_route(job_id, filepath, data):
    """Export/rewrap a video into MXF container."""
    op_pattern = data.get("op_pattern", "op1a").strip()
    if op_pattern not in ("op1a", "opatom"):
        op_pattern = "op1a"
    audio_tracks = data.get("audio_tracks")
    if audio_tracks is not None and not isinstance(audio_tracks, list):
        audio_tracks = None
    timecode = data.get("timecode")
    if timecode and not isinstance(timecode, str):
        timecode = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.mxf_support import export_mxf

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_{op_pattern}.mxf")

    result = export_mxf(
        filepath,
        output_path_override=out_path,
        op_pattern=op_pattern,
        audio_tracks=audio_tracks,
        timecode=timecode,
        on_progress=_p,
    )
    return result


@professional_bp.route("/mxf/convert", methods=["POST"])
@require_csrf
@async_job("mxf_convert")
def convert_to_mxf_route(job_id, filepath, data):
    """Transcode video to DNxHR/XDCAM in MXF container."""
    codec = data.get("codec", "dnxhd").strip()
    profile = data.get("profile", "dnxhr_hq").strip()

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.mxf_support import convert_to_mxf

    d = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = None
    if d:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(d, f"{base}_{profile}.mxf")

    result = convert_to_mxf(
        filepath,
        codec=codec,
        profile=profile,
        output_path_override=out_path,
        on_progress=_p,
    )
    return result
