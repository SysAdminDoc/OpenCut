"""
OpenCut Format Routes - GIF/WebP/APNG Export & Metadata Tools

Blueprint: format_bp
Routes:
  POST /export/gif   -> async GIF export with optimization
  POST /export/webp  -> async animated WebP export
  POST /export/apng  -> async animated APNG export
  POST /metadata/read  -> sync metadata reading
  POST /metadata/strip -> async metadata stripping
  POST /metadata/copy  -> async copy with metadata overrides
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
)

logger = logging.getLogger("opencut")

format_bp = Blueprint("format", __name__)


# ---------------------------------------------------------------------------
# GIF Export
# ---------------------------------------------------------------------------
@format_bp.route("/export/gif", methods=["POST"])
@require_csrf
@async_job("export_gif")
def export_gif_route(job_id, filepath, data):
    """Export video as optimized GIF with two-pass palette generation."""
    from opencut.core.gif_export import export_gif

    output_dir = data.get("output_dir", "")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    out_path = data.get("output_path", "")
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path and effective_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(effective_dir, f"{base}_optimized.gif")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = export_gif(
        input_path=filepath,
        output_path=out_path or None,
        max_width=safe_int(data.get("max_width", 480), 480, min_val=16, max_val=3840),
        fps=safe_int(data.get("fps", 15), 15, min_val=1, max_val=60),
        max_colors=safe_int(data.get("max_colors", 256), 256, min_val=2, max_val=256),
        dither=str(data.get("dither", "sierra2_4a"))[:30],
        loop=safe_int(data.get("loop", 0), 0, min_val=0),
        max_file_size_mb=safe_float(data.get("max_file_size_mb"), None) if data.get("max_file_size_mb") is not None else None,
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Animated WebP Export
# ---------------------------------------------------------------------------
@format_bp.route("/export/webp", methods=["POST"])
@require_csrf
@async_job("export_webp")
def export_webp_route(job_id, filepath, data):
    """Export video as animated WebP."""
    from opencut.core.gif_export import export_webp

    output_dir = data.get("output_dir", "")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    out_path = data.get("output_path", "")
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path and effective_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(effective_dir, f"{base}_animated.webp")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = export_webp(
        input_path=filepath,
        output_path=out_path or None,
        max_width=safe_int(data.get("max_width", 480), 480, min_val=16, max_val=3840),
        fps=safe_int(data.get("fps", 15), 15, min_val=1, max_val=60),
        quality=safe_int(data.get("quality", 75), 75, min_val=0, max_val=100),
        loop=safe_int(data.get("loop", 0), 0, min_val=0),
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Animated APNG Export
# ---------------------------------------------------------------------------
@format_bp.route("/export/apng", methods=["POST"])
@require_csrf
@async_job("export_apng")
def export_apng_route(job_id, filepath, data):
    """Export video as animated PNG (APNG)."""
    from opencut.core.gif_export import export_apng

    output_dir = data.get("output_dir", "")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    out_path = data.get("output_path", "")
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path and effective_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(effective_dir, f"{base}_animated.apng")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = export_apng(
        input_path=filepath,
        output_path=out_path or None,
        max_width=safe_int(data.get("max_width", 480), 480, min_val=16, max_val=3840),
        fps=safe_int(data.get("fps", 15), 15, min_val=1, max_val=60),
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Metadata Read (SYNC - no async_job)
# ---------------------------------------------------------------------------
@format_bp.route("/metadata/read", methods=["POST"])
@require_csrf
def metadata_read_route():
    """Read all metadata from a media file. Synchronous."""
    from opencut.core.metadata_tools import get_metadata
    from opencut.errors import safe_error

    data = request.get_json(force=True) or {}
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = get_metadata(filepath)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as exc:
        return safe_error(exc, "metadata_read")


# ---------------------------------------------------------------------------
# Metadata Strip (ASYNC)
# ---------------------------------------------------------------------------
@format_bp.route("/metadata/strip", methods=["POST"])
@require_csrf
@async_job("metadata_strip")
def metadata_strip_route(job_id, filepath, data):
    """Strip or selectively preserve metadata from a media file."""
    from opencut.core.metadata_tools import strip_metadata

    output_dir = data.get("output_dir", "")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    out_path = data.get("output_path", "")
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path and effective_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1] or ".mp4"
        out_path = os.path.join(effective_dir, f"{base}_stripped{ext}")

    mode = str(data.get("mode", "strip_all"))[:20]
    preserve_fields = data.get("preserve_fields")
    strip_fields = data.get("strip_fields")

    # Validate list types
    if preserve_fields is not None and not isinstance(preserve_fields, list):
        preserve_fields = None
    if strip_fields is not None and not isinstance(strip_fields, list):
        strip_fields = None

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = strip_metadata(
        input_path=filepath,
        output_path=out_path or None,
        preserve_fields=preserve_fields,
        strip_fields=strip_fields,
        mode=mode,
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Metadata Copy with Overrides (ASYNC)
# ---------------------------------------------------------------------------
@format_bp.route("/metadata/copy", methods=["POST"])
@require_csrf
@async_job("metadata_copy")
def metadata_copy_route(job_id, filepath, data):
    """Copy a media file with custom metadata overrides."""
    from opencut.core.metadata_tools import copy_with_metadata

    output_dir = data.get("output_dir", "")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    out_path = data.get("output_path", "")
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path and effective_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1] or ".mp4"
        out_path = os.path.join(effective_dir, f"{base}_meta{ext}")

    if not out_path:
        raise ValueError("output_path is required (or provide output_dir)")

    metadata_overrides = data.get("metadata_overrides")
    if metadata_overrides is not None and not isinstance(metadata_overrides, dict):
        raise ValueError("metadata_overrides must be a JSON object")

    def _progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = copy_with_metadata(
        input_path=filepath,
        output_path=out_path,
        metadata_overrides=metadata_overrides,
        on_progress=_progress,
    )
    return result
