"""
OpenCut Batch Data Routes

Routes for structured ingest, storage tiering, batch metadata,
batch conforming, star trail compositing, construction timelapse,
expression scripting, and subtitle positioning.
"""

import logging
import os
import tempfile

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import MAX_BATCH_FILES, _update_job, async_job
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

batch_data_bp = Blueprint("batch_data", __name__)


# ---------------------------------------------------------------------------
# Helper: validate list of file paths
# ---------------------------------------------------------------------------
def _validate_file_list(data, key="file_paths", required=True, max_files=None):
    """Extract and validate a list of file paths from request data."""
    max_files = max_files or MAX_BATCH_FILES
    paths = data.get(key, [])
    if not isinstance(paths, list):
        raise ValueError(f"{key} must be a list")
    if required and not paths:
        raise ValueError(f"{key} must be a non-empty list")
    if len(paths) > max_files:
        raise ValueError(f"Too many files: {len(paths)} (max {max_files})")
    validated = []
    for p in paths:
        if isinstance(p, str) and p.strip():
            try:
                validated.append(validate_filepath(p.strip()))
            except ValueError:
                validated.append(p.strip())  # include for error reporting
    return validated


# =========================================================================
# 1. Structured Ingest (23.4)
# =========================================================================

@batch_data_bp.route("/ingest/run", methods=["POST"])
@require_csrf
@async_job("structured_ingest", filepath_required=False)
def ingest_run(job_id, filepath, data):
    """Run structured ingest from a source directory."""
    source_dir = data.get("source_dir", "").strip()
    if not source_dir:
        raise ValueError("source_dir is required")
    try:
        source_dir = validate_path(source_dir)
    except ValueError as e:
        raise ValueError(f"Invalid source_dir: {e}")

    dest_dir = data.get("dest_dir", "").strip()
    if dest_dir:
        try:
            dest_dir = validate_path(dest_dir)
        except ValueError:
            dest_dir = ""

    config = data.get("config", {})
    if not isinstance(config, dict):
        config = {}

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.structured_ingest import run_ingest
    result = run_ingest(
        source_dir=source_dir,
        config=config,
        dest_dir=dest_dir or None,
        on_progress=_p,
    )
    return result.to_dict()


@batch_data_bp.route("/ingest/verify-checksum", methods=["POST"])
@require_csrf
def ingest_verify_checksum():
    """Verify checksum of a single file (sync)."""
    try:
        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No filepath provided"}), 400
        try:
            filepath = validate_filepath(filepath)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        algorithm = str(data.get("algorithm", "sha256"))[:10]
        expected = str(data.get("expected", ""))[:256]

        from opencut.core.structured_ingest import verify_checksum
        result = verify_checksum(filepath, algorithm, expected)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "ingest_verify_checksum")


@batch_data_bp.route("/ingest/rename-preview", methods=["POST"])
@require_csrf
def ingest_rename_preview():
    """Preview rename-by-pattern result (sync)."""
    try:
        data = request.get_json(force=True) or {}
        filename = str(data.get("filename", ""))[:500]
        pattern = str(data.get("pattern", ""))[:500]
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        if not filename:
            return jsonify({"error": "filename is required"}), 400
        if not pattern:
            return jsonify({"error": "pattern is required"}), 400

        from opencut.core.structured_ingest import rename_by_pattern
        result = rename_by_pattern(filename, pattern, metadata)
        return jsonify({"original": filename, "renamed": result, "pattern": pattern})
    except Exception as e:
        return safe_error(e, "ingest_rename_preview")


# =========================================================================
# 2. Storage Tiering (23.5)
# =========================================================================

@batch_data_bp.route("/storage/scan", methods=["POST"])
@require_csrf
@async_job("storage_scan", filepath_required=False)
def storage_scan(job_id, filepath, data):
    """Scan a project directory for files eligible for archival."""
    project_dir = data.get("project_dir", "").strip()
    if not project_dir:
        raise ValueError("project_dir is required")
    try:
        project_dir = validate_path(project_dir)
    except ValueError as e:
        raise ValueError(f"Invalid project_dir: {e}")

    idle_days = safe_float(data.get("idle_days", 30), 30, min_val=1, max_val=3650)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.storage_tiering import scan_for_archival
    result = scan_for_archival(project_dir, idle_days=idle_days, on_progress=_p)
    return result.to_dict()


@batch_data_bp.route("/storage/archive", methods=["POST"])
@require_csrf
@async_job("storage_archive", filepath_required=False)
def storage_archive(job_id, filepath, data):
    """Archive a list of files to an archive directory."""
    file_list = data.get("file_list", [])
    if not isinstance(file_list, list) or not file_list:
        raise ValueError("file_list must be a non-empty list")

    archive_path = data.get("archive_path", "").strip()
    if not archive_path:
        raise ValueError("archive_path is required")
    try:
        archive_path = validate_path(archive_path)
    except ValueError as e:
        raise ValueError(f"Invalid archive_path: {e}")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.storage_tiering import archive_files
    result = archive_files(file_list, archive_path, on_progress=_p)
    return result.to_dict()


@batch_data_bp.route("/storage/restore", methods=["POST"])
@require_csrf
@async_job("storage_restore", filepath_required=False)
def storage_restore(job_id, filepath, data):
    """Restore a file from archive using its stub."""
    stub_path = data.get("stub_path", "").strip()
    if not stub_path:
        raise ValueError("stub_path is required")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.storage_tiering import restore_file
    return restore_file(stub_path, on_progress=_p)


@batch_data_bp.route("/storage/manifest", methods=["POST"])
@require_csrf
def storage_manifest():
    """Get the archive manifest (sync)."""
    try:
        data = request.get_json(force=True) or {}
        archive_path = data.get("archive_path", "").strip()

        from opencut.core.storage_tiering import get_archive_manifest
        manifest = get_archive_manifest(archive_path)
        return jsonify(manifest)
    except Exception as e:
        return safe_error(e, "storage_manifest")


# =========================================================================
# 3. Batch Metadata Editor (47.2)
# =========================================================================

@batch_data_bp.route("/batch-metadata/read", methods=["POST"])
@require_csrf
def metadata_read():
    """Read metadata from multiple files (sync)."""
    try:
        data = request.get_json(force=True) or {}
        file_paths = _validate_file_list(data)

        from opencut.core.batch_metadata import read_batch_metadata
        results = read_batch_metadata(file_paths)
        return jsonify({"results": results, "count": len(results)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "metadata_read")


@batch_data_bp.route("/batch-metadata/write", methods=["POST"])
@require_csrf
@async_job("metadata_write", filepath_required=False)
def metadata_write(job_id, filepath, data):
    """Write metadata to multiple files."""
    file_paths = _validate_file_list(data)
    metadata_updates = data.get("metadata_updates", {})
    if not isinstance(metadata_updates, dict):
        raise ValueError("metadata_updates must be a dict")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.batch_metadata import write_batch_metadata
    results = write_batch_metadata(file_paths, metadata_updates, on_progress=_p)
    updated = sum(1 for r in results if r["status"] == "updated")
    return {"results": results, "updated": updated, "total": len(results)}


@batch_data_bp.route("/batch-metadata/template", methods=["POST"])
@require_csrf
@async_job("metadata_template", filepath_required=False)
def metadata_template(job_id, filepath, data):
    """Apply a metadata template to multiple files."""
    file_paths = _validate_file_list(data)
    template = data.get("template", {})
    if not isinstance(template, dict) or not template:
        raise ValueError("template must be a non-empty dict")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.batch_metadata import apply_metadata_template
    results = apply_metadata_template(file_paths, template, on_progress=_p)
    updated = sum(1 for r in results if r["status"] == "updated")
    return {"results": results, "updated": updated, "total": len(results)}


@batch_data_bp.route("/batch-metadata/export-csv", methods=["POST"])
@require_csrf
def metadata_export_csv():
    """Export metadata from multiple files to CSV (sync)."""
    try:
        data = request.get_json(force=True) or {}
        file_paths = _validate_file_list(data)

        output_path = data.get("output_path", "").strip()
        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), "opencut_metadata.csv")
        else:
            output_path = validate_output_path(output_path)

        from opencut.core.batch_metadata import export_metadata_csv
        csv_path = export_metadata_csv(file_paths, output_path)
        return jsonify({"output_path": csv_path, "count": len(file_paths)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "metadata_export_csv")


# =========================================================================
# 4. Batch Conforming (47.5)
# =========================================================================

@batch_data_bp.route("/batch-conform/analyze", methods=["POST"])
@require_csrf
def conform_analyze():
    """Analyze conformance of multiple files against target spec (sync)."""
    try:
        data = request.get_json(force=True) or {}
        file_paths = _validate_file_list(data)
        target_spec = data.get("target_spec", {})
        if not isinstance(target_spec, dict):
            raise ValueError("target_spec must be a dict")

        from opencut.core.batch_conform import analyze_conformance_batch
        results = analyze_conformance_batch(file_paths, target_spec)
        needs_conform = sum(1 for r in results if r.get("needs_conform"))
        return jsonify({
            "results": results,
            "total": len(results),
            "needs_conform": needs_conform,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "conform_analyze")


@batch_data_bp.route("/batch-conform/run", methods=["POST"])
@require_csrf
@async_job("batch_conform", filepath_required=False)
def conform_run(job_id, filepath, data):
    """Conform multiple files to a target specification."""
    file_paths = _validate_file_list(data)
    target_spec = data.get("target_spec", {})
    if not isinstance(target_spec, dict):
        raise ValueError("target_spec must be a dict")

    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError:
            output_dir = ""

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.batch_conform import conform_batch
    result = conform_batch(
        file_paths, target_spec,
        output_dir=output_dir or None,
        on_progress=_p,
    )
    return result.to_dict()


# =========================================================================
# 5. Star Trail Compositing (42.4)
# =========================================================================

@batch_data_bp.route("/star-trail/composite", methods=["POST"])
@require_csrf
@async_job("star_trail_composite", filepath_required=False)
def star_trail_composite(job_id, filepath, data):
    """Composite star trails from a series of images."""
    image_paths = _validate_file_list(data, key="image_paths", max_files=5000)
    output_path = data.get("output_path", "").strip()
    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "star_trail_composite.jpg")
    else:
        output_path = validate_output_path(output_path)

    mode = str(data.get("mode", "lighten"))[:20]
    gap_fill = safe_bool(data.get("gap_fill", True), True)
    skip_streaks = safe_bool(data.get("skip_streaks", False), False)
    streak_threshold = safe_float(data.get("streak_threshold", 50), 50,
                                  min_val=10, max_val=250)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.star_trail import composite_star_trails
    result = composite_star_trails(
        image_paths, output_path,
        mode=mode, gap_fill=gap_fill,
        skip_streaks=skip_streaks,
        streak_threshold=streak_threshold,
        on_progress=_p,
    )
    return result.to_dict()


@batch_data_bp.route("/star-trail/remove-streaks", methods=["POST"])
@require_csrf
@async_job("star_trail_streaks", filepath_required=False)
def star_trail_remove_streaks(job_id, filepath, data):
    """Analyze frames for airplane/satellite streaks."""
    frames = _validate_file_list(data, key="frames", max_files=5000)
    threshold = safe_float(data.get("threshold", 50), 50, min_val=10, max_val=250)
    min_length = safe_int(data.get("min_length", 20), 20, min_val=5, max_val=500)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.star_trail import remove_streaks
    return remove_streaks(frames, threshold=threshold, min_length=min_length,
                          on_progress=_p)


@batch_data_bp.route("/star-trail/animation", methods=["POST"])
@require_csrf
@async_job("star_trail_animation", filepath_required=False)
def star_trail_animation(job_id, filepath, data):
    """Create a progressive star trail animation video."""
    image_paths = _validate_file_list(data, key="image_paths", max_files=5000)
    output_path = data.get("output_path", "").strip()
    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "star_trail_animation.mp4")
    else:
        output_path = validate_output_path(output_path)

    fps = safe_float(data.get("fps", 24), 24, min_val=1, max_val=120)
    trail_length = safe_int(data.get("trail_length", 0), 0, min_val=0, max_val=1000)
    mode = str(data.get("mode", "lighten"))[:20]

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.star_trail import create_trail_animation
    return create_trail_animation(
        image_paths, output_path,
        fps=fps, trail_length=trail_length,
        mode=mode, on_progress=_p,
    )


# =========================================================================
# 6. Construction Timelapse (42.5)
# =========================================================================

@batch_data_bp.route("/construction-timelapse/align", methods=["POST"])
@require_csrf
@async_job("timelapse_align", filepath_required=False)
def construction_timelapse_align(job_id, filepath, data):
    """Align construction timelapse frames."""
    image_paths = _validate_file_list(data, key="image_paths", max_files=5000)
    reference_index = safe_int(data.get("reference_index", 0), 0, min_val=0)
    max_shift = safe_int(data.get("max_shift", 100), 100, min_val=10, max_val=1000)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.construction_timelapse import align_frames
    results = align_frames(
        image_paths, reference_index=reference_index,
        max_shift=max_shift, on_progress=_p,
    )
    return {
        "results": [
            {
                "frame_path": r.frame_path,
                "aligned": r.aligned,
                "shift_x": r.shift_x,
                "shift_y": r.shift_y,
                "confidence": r.confidence,
                "error": r.error,
            }
            for r in results
        ],
        "aligned_count": sum(1 for r in results if r.aligned),
        "total": len(results),
    }


@batch_data_bp.route("/construction-timelapse/build", methods=["POST"])
@require_csrf
@async_job("timelapse_build", filepath_required=False)
def construction_timelapse_build(job_id, filepath, data):
    """Build a construction timelapse video."""
    image_paths = _validate_file_list(data, key="image_paths", max_files=5000)
    output_path = data.get("output_path", "").strip()
    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "construction_timelapse.mp4")
    else:
        output_path = validate_output_path(output_path)

    fps = safe_float(data.get("fps", 24), 24, min_val=1, max_val=120)
    do_align = safe_bool(data.get("align", True), True)
    do_deflicker = safe_bool(data.get("deflicker", True), True)
    auto_crop = safe_bool(data.get("auto_crop", True), True)
    fill_gaps = safe_bool(data.get("fill_gaps", True), True)
    max_shift = safe_int(data.get("max_shift", 100), 100, min_val=10, max_val=1000)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.construction_timelapse import build_construction_timelapse
    result = build_construction_timelapse(
        image_paths, output_path,
        fps=fps, align=do_align, deflicker=do_deflicker,
        auto_crop=auto_crop, fill_gaps=fill_gaps,
        max_shift=max_shift, on_progress=_p,
    )
    return result.to_dict()


@batch_data_bp.route("/construction-timelapse/fill-frames", methods=["POST"])
@require_csrf
@async_job("timelapse_fill", filepath_required=False)
def construction_timelapse_fill(job_id, filepath, data):
    """Fill missing frames in a timelapse sequence."""
    frames_raw = data.get("frames", [])
    if not isinstance(frames_raw, list):
        raise ValueError("frames must be a list (use null for missing)")

    # Convert to list of Optional[str]
    frames = []
    for f in frames_raw:
        if f is None or f == "":
            frames.append(None)
        elif isinstance(f, str):
            frames.append(f.strip())
        else:
            frames.append(None)

    timestamps = data.get("timestamps")
    if timestamps is not None and not isinstance(timestamps, list):
        timestamps = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.construction_timelapse import fill_missing_frames
    results = fill_missing_frames(frames, timestamps=timestamps, on_progress=_p)
    filled = sum(1 for r in results if r.get("interpolated"))
    return {"results": results, "filled": filled, "total": len(results)}


# =========================================================================
# 7. Expression Scripting (26.4)
# =========================================================================

@batch_data_bp.route("/expression/validate", methods=["POST"])
@require_csrf
def expression_validate():
    """Validate an expression (sync)."""
    try:
        data = request.get_json(force=True) or {}
        expr = str(data.get("expression", ""))[:2000]
        if not expr:
            return jsonify({"error": "expression is required"}), 400

        from opencut.core.expression_engine import validate_expression
        result = validate_expression(expr)
        return jsonify(result)
    except Exception as e:
        return safe_error(e, "expression_validate")


@batch_data_bp.route("/expression/evaluate", methods=["POST"])
@require_csrf
def expression_evaluate():
    """Evaluate an expression with context (sync)."""
    try:
        data = request.get_json(force=True) or {}
        expr = str(data.get("expression", ""))[:2000]
        if not expr:
            return jsonify({"error": "expression is required"}), 400

        context_params = data.get("context", {})
        if not isinstance(context_params, dict):
            context_params = {}

        from opencut.core.expression_engine import (
            create_expression_context,
            evaluate_expression,
        )

        ctx = create_expression_context(
            frame=safe_int(context_params.get("frame", 0), 0),
            time=safe_float(context_params.get("time", 0.0), 0.0),
            audio_amp=safe_float(context_params.get("audio_amp", 0.0), 0.0,
                                 min_val=0.0, max_val=1.0),
            fps=safe_float(context_params.get("fps", 24.0), 24.0),
            duration=safe_float(context_params.get("duration", 0.0), 0.0),
            width=safe_int(context_params.get("width", 1920), 1920),
            height=safe_int(context_params.get("height", 1080), 1080),
            custom_vars=context_params.get("custom_vars"),
        )

        result = evaluate_expression(expr, ctx)

        # Ensure result is JSON serializable
        if isinstance(result, (int, float, bool, str)):
            serialized = result
        elif isinstance(result, (list, tuple)):
            serialized = list(result)
        else:
            serialized = str(result)

        return jsonify({"result": serialized, "type": type(result).__name__})
    except Exception as e:
        return safe_error(e, "expression_evaluate")


@batch_data_bp.route("/expression/evaluate-batch", methods=["POST"])
@require_csrf
def expression_evaluate_batch():
    """Evaluate an expression across multiple frames (sync)."""
    try:
        data = request.get_json(force=True) or {}
        expr = str(data.get("expression", ""))[:2000]
        if not expr:
            return jsonify({"error": "expression is required"}), 400

        fps = safe_float(data.get("fps", 24.0), 24.0)
        start_frame = safe_int(data.get("start_frame", 0), 0, min_val=0)
        end_frame = safe_int(data.get("end_frame", 100), 100, min_val=1, max_val=10000)
        step = safe_int(data.get("step", 1), 1, min_val=1, max_val=100)
        duration = safe_float(data.get("duration", 0.0), 0.0)

        from opencut.core.expression_engine import (
            compile_expression,
            create_expression_context,
            evaluate_expression,
        )

        compiled = compile_expression(expr)
        if not compiled.valid:
            return jsonify({"error": compiled.error}), 400

        results = []
        for frame in range(start_frame, end_frame, step):
            t = frame / max(fps, 0.001)
            ctx = create_expression_context(
                frame=frame, time=t, fps=fps, duration=duration,
            )
            try:
                value = evaluate_expression(compiled, ctx)
                if isinstance(value, (int, float, bool)):
                    results.append({"frame": frame, "time": round(t, 4), "value": value})
                else:
                    results.append({"frame": frame, "time": round(t, 4), "value": str(value)})
            except Exception as e:
                results.append({"frame": frame, "time": round(t, 4), "error": str(e)})

        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return safe_error(e, "expression_evaluate_batch")


# =========================================================================
# 8. Subtitle Positioning (24.5)
# =========================================================================

@batch_data_bp.route("/subtitle-position/analyze-frame", methods=["POST"])
@require_csrf
def subtitle_position_analyze_frame():
    """Analyze a frame for obstructions (sync)."""
    try:
        data = request.get_json(force=True) or {}
        frame_path = data.get("frame_path", "").strip()
        if not frame_path:
            return jsonify({"error": "frame_path is required"}), 400
        try:
            frame_path = validate_filepath(frame_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        detect_faces = safe_bool(data.get("detect_faces", True), True)
        detect_text = safe_bool(data.get("detect_text", True), True)
        detect_logos = safe_bool(data.get("detect_logos", True), True)

        from opencut.core.subtitle_position import analyze_frame_obstructions
        obstructions = analyze_frame_obstructions(
            frame_path,
            detect_faces=detect_faces,
            detect_text=detect_text,
            detect_logos=detect_logos,
        )
        return jsonify({
            "obstructions": [
                {
                    "x": o.x, "y": o.y,
                    "width": o.width, "height": o.height,
                    "label": o.label, "confidence": o.confidence,
                }
                for o in obstructions
            ],
            "count": len(obstructions),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "subtitle_position_analyze_frame")


@batch_data_bp.route("/subtitle-position/compute", methods=["POST"])
@require_csrf
def subtitle_position_compute():
    """Compute subtitle position from obstructions (sync)."""
    try:
        data = request.get_json(force=True) or {}
        obstructions_raw = data.get("obstructions", [])
        frame_width = safe_int(data.get("frame_width", 1920), 1920, min_val=1)
        frame_height = safe_int(data.get("frame_height", 1080), 1080, min_val=1)
        alignment = safe_int(data.get("alignment", 2), 2, min_val=1, max_val=9)
        margin = safe_int(data.get("margin", 50), 50, min_val=0, max_val=500)

        from opencut.core.subtitle_position import Obstruction, compute_subtitle_position

        obstructions = []
        for o in obstructions_raw:
            if isinstance(o, dict):
                obstructions.append(Obstruction(
                    x=int(o.get("x", 0)),
                    y=int(o.get("y", 0)),
                    width=int(o.get("width", 0)),
                    height=int(o.get("height", 0)),
                    label=str(o.get("label", "")),
                    confidence=float(o.get("confidence", 0)),
                ))

        pos = compute_subtitle_position(
            obstructions, (frame_width, frame_height),
            preferred_alignment=alignment, margin=margin,
        )
        return jsonify({
            "x": pos.x, "y": pos.y,
            "alignment": pos.alignment,
            "margin_bottom": pos.margin_bottom,
            "safe": pos.safe,
            "reason": pos.reason,
        })
    except Exception as e:
        return safe_error(e, "subtitle_position_compute")


@batch_data_bp.route("/subtitle-position/apply", methods=["POST"])
@require_csrf
@async_job("subtitle_positioning", filepath_required=False)
def subtitle_position_apply(job_id, filepath, data):
    """Apply dynamic subtitle positioning to a video."""
    subtitle_path = data.get("subtitle_path", "").strip()
    video_path = data.get("video_path", "").strip()

    if not subtitle_path:
        raise ValueError("subtitle_path is required")
    if not video_path:
        raise ValueError("video_path is required")

    try:
        subtitle_path = validate_filepath(subtitle_path)
    except ValueError as e:
        raise ValueError(f"Invalid subtitle_path: {e}")
    try:
        video_path = validate_filepath(video_path)
    except ValueError as e:
        raise ValueError(f"Invalid video_path: {e}")

    output_path = data.get("output_path", "").strip()
    if not output_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            _resolve_output_dir(video_path, data.get("output_dir", "")),
            f"{base}_positioned.mp4",
        )
    else:
        output_path = validate_output_path(output_path)

    sample_interval = safe_float(data.get("sample_interval", 2.0), 2.0,
                                 min_val=0.5, max_val=30.0)

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.subtitle_position import apply_dynamic_positioning
    result = apply_dynamic_positioning(
        subtitle_path, video_path, output_path,
        sample_interval=sample_interval, on_progress=_p,
    )
    return result.to_dict()
