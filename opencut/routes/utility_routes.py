"""
OpenCut Utility Routes

Blueprint for watermark, webhook, team presets, project notes,
license tracking, batch thumbnails, and change annotations.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_output_path

logger = logging.getLogger("opencut")

utility_bp = Blueprint("utility", __name__)


# ===================================================================
# Watermark
# ===================================================================

@utility_bp.route("/watermark/apply", methods=["POST"])
@require_csrf
@async_job("watermark_apply")
def watermark_apply(job_id, filepath, data):
    """Apply a watermark to a single video file."""
    from opencut.core.watermark import WATERMARK_PRESETS, apply_watermark

    preset_name = data.get("preset", "").strip()
    if preset_name and preset_name in WATERMARK_PRESETS:
        config = dict(WATERMARK_PRESETS[preset_name])
    else:
        config = {}

    # Explicit params override preset values
    wtype = data.get("watermark_type") or config.get("watermark_type", "text")
    content = data.get("content") or config.get("content", "DRAFT")
    position = data.get("position") or config.get("position", "center")
    opacity = safe_float(data.get("opacity", config.get("opacity", 0.4)), 0.4, min_val=0.0, max_val=1.0)
    font_size = safe_int(data.get("font_size", config.get("font_size", 48)), 48, min_val=8, max_val=200)
    angle = safe_int(data.get("angle", config.get("angle", 0)), 0, min_val=0, max_val=360)
    output_dir = data.get("output_dir", "")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    out_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1] or ".mp4"
        out_path = os.path.join(output_dir, f"{base}_watermarked{ext}")

    result = apply_watermark(
        input_path=filepath,
        watermark_type=wtype,
        content=content,
        position=position,
        opacity=opacity,
        font_size=font_size,
        angle=angle,
        output_path_str=out_path,
        on_progress=_on_progress,
    )
    return result


@utility_bp.route("/watermark/batch", methods=["POST"])
@require_csrf
@async_job("watermark_batch", filepath_required=False)
def watermark_batch(job_id, filepath, data):
    """Apply watermark to multiple video files."""
    from opencut.core.watermark import batch_apply_watermark

    file_paths = data.get("file_paths", [])
    if not file_paths or not isinstance(file_paths, list):
        raise ValueError("file_paths must be a non-empty list")

    watermark_config = data.get("watermark_config", {})
    output_dir = data.get("output_dir", "")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = batch_apply_watermark(
        file_paths=file_paths,
        watermark_config=watermark_config,
        output_dir=output_dir or None,
        on_progress=_on_progress,
    )
    return result


# ===================================================================
# Webhooks
# ===================================================================

@utility_bp.route("/webhook/test", methods=["POST"])
@require_csrf
def webhook_test():
    """Send a test webhook to a given URL (sync)."""
    from opencut.core.webhooks import send_webhook

    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    payload = {
        "event": "test",
        "message": "OpenCut webhook test",
    }
    success = send_webhook(url, "test", payload, timeout=data.get("timeout", 10))
    return jsonify({"success": success, "url": url})


@utility_bp.route("/webhook/config", methods=["GET"])
@require_csrf
def webhook_config_get():
    """Return current webhook configuration."""
    from opencut.core.webhooks import load_webhook_config

    configs = load_webhook_config()
    return jsonify({"webhooks": configs})


@utility_bp.route("/webhook/config", methods=["POST"])
@require_csrf
def webhook_config_save():
    """Save webhook configuration."""
    from opencut.core.webhooks import save_webhook_config

    data = request.get_json(force=True)
    configs = data.get("webhooks", [])
    if not isinstance(configs, list):
        return jsonify({"error": "webhooks must be a list"}), 400

    save_webhook_config(configs)
    return jsonify({"saved": len(configs)})


# ===================================================================
# Team Presets
# ===================================================================

@utility_bp.route("/team/sync", methods=["POST"])
@require_csrf
@async_job("team_sync", filepath_required=False)
def team_sync(job_id, filepath, data):
    """Sync team presets from a shared folder."""
    from opencut.core.team_presets import get_shared_folder_path, sync_team_presets

    shared_folder = data.get("shared_folder", "").strip()
    if not shared_folder:
        shared_folder = get_shared_folder_path()
    if not shared_folder:
        raise ValueError("No shared folder configured. Set one via POST /team/sync or settings.")

    local_folder = data.get("local_folder", "").strip() or None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _on_progress(10, "Scanning shared folder...")
    result = sync_team_presets(shared_folder, local_folder=local_folder)
    _on_progress(100, "Sync complete")
    return result


@utility_bp.route("/team/status", methods=["GET"])
@require_csrf
def team_status():
    """Return current shared folder path and scan results."""
    from opencut.core.team_presets import get_shared_folder_path, scan_shared_folder

    path = get_shared_folder_path()
    if not path:
        return jsonify({"configured": False, "shared_folder": None})

    try:
        scan = scan_shared_folder(path)
        return jsonify({
            "configured": True,
            "shared_folder": path,
            "total": scan["total"],
            "presets": len(scan["presets"]),
            "workflows": len(scan["workflows"]),
            "luts": len(scan["luts"]),
        })
    except FileNotFoundError:
        return jsonify({
            "configured": True,
            "shared_folder": path,
            "error": "Shared folder not found",
        })


# ===================================================================
# Project Notes
# ===================================================================

@utility_bp.route("/notes/add", methods=["POST"])
@require_csrf
def notes_add():
    """Add a timestamped note to a project."""
    from opencut.core.project_notes import add_note

    data = request.get_json(force=True)
    project_id = data.get("project_id", "").strip()
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    priority = data.get("priority", "normal")
    author = data.get("author", "")

    note = add_note(project_id, timestamp, text, priority=priority, author=author)
    return jsonify(note)


@utility_bp.route("/notes/list", methods=["GET"])
@require_csrf
def notes_list():
    """List notes for a project. Query param: project_id, status (optional)."""
    from opencut.core.project_notes import get_notes

    project_id = request.args.get("project_id", "").strip()
    if not project_id:
        return jsonify({"error": "project_id query parameter is required"}), 400

    status = request.args.get("status", "").strip() or None
    notes = get_notes(project_id, status=status)
    return jsonify({"notes": notes, "count": len(notes)})


@utility_bp.route("/notes/update", methods=["POST"])
@require_csrf
def notes_update():
    """Update an existing note."""
    from opencut.core.project_notes import update_note

    data = request.get_json(force=True)
    note_id = data.get("note_id", "").strip()
    if not note_id:
        return jsonify({"error": "note_id is required"}), 400

    try:
        updated = update_note(
            note_id,
            text=data.get("text"),
            status=data.get("status"),
            priority=data.get("priority"),
        )
        return jsonify(updated)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@utility_bp.route("/notes/export", methods=["POST"])
@require_csrf
def notes_export():
    """Export project notes as text, csv, or markdown."""
    from opencut.core.project_notes import export_notes

    data = request.get_json(force=True)
    project_id = data.get("project_id", "").strip()
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    fmt = data.get("format", "text")
    result = export_notes(project_id, format=fmt)
    return jsonify({"content": result, "format": fmt})


# ===================================================================
# License Tracking
# ===================================================================

@utility_bp.route("/license/record", methods=["POST"])
@require_csrf
def license_record():
    """Record usage of a third-party asset with license info."""
    from opencut.core.license_tracker import record_asset_usage

    data = request.get_json(force=True)
    source_url = data.get("source_url", "").strip()
    filename = data.get("filename", "").strip()
    license_type = data.get("license_type", "").strip()

    if not source_url or not filename or not license_type:
        return jsonify({"error": "source_url, filename, and license_type are required"}), 400

    record = record_asset_usage(
        source_url=source_url,
        filename=filename,
        license_type=license_type,
        attribution_text=data.get("attribution_text", ""),
        project_id=data.get("project_id", ""),
    )
    return jsonify(record)


@utility_bp.route("/license/list", methods=["GET"])
@require_csrf
def license_list():
    """List license records for a project. Query param: project_id."""
    from opencut.core.license_tracker import get_project_licenses

    project_id = request.args.get("project_id", "").strip()
    if not project_id:
        return jsonify({"error": "project_id query parameter is required"}), 400

    records = get_project_licenses(project_id)
    return jsonify({"licenses": records, "count": len(records)})


@utility_bp.route("/license/export", methods=["POST"])
@require_csrf
def license_export():
    """Export attribution document for a project."""
    from opencut.core.license_tracker import export_attribution

    data = request.get_json(force=True)
    project_id = data.get("project_id", "").strip()
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    fmt = data.get("format", "text")
    result = export_attribution(project_id, format=fmt)
    return jsonify({"content": result, "format": fmt})


# ===================================================================
# Batch Thumbnails
# ===================================================================

@utility_bp.route("/batch/thumbnails", methods=["POST"])
@require_csrf
@async_job("batch_thumbnails", filepath_required=False)
def batch_thumbnails(job_id, filepath, data):
    """Extract thumbnails from multiple video files."""
    from opencut.core.batch_thumbnails import extract_thumbnails

    file_paths = data.get("file_paths", [])
    if not file_paths or not isinstance(file_paths, list):
        raise ValueError("file_paths must be a non-empty list")

    mode = data.get("mode", "auto")
    timestamp_pct = safe_float(data.get("timestamp_pct", 0.1), 0.1, min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")
    width = safe_int(data.get("width", 640), 640, min_val=32, max_val=3840)
    fmt = data.get("format", "jpg")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = extract_thumbnails(
        file_paths=file_paths,
        mode=mode,
        timestamp_pct=timestamp_pct,
        output_dir=output_dir or None,
        width=width,
        format=fmt,
        on_progress=_on_progress,
    )
    return result


@utility_bp.route("/batch/contact-sheet", methods=["POST"])
@require_csrf
@async_job("batch_contact_sheet", filepath_required=False)
def batch_contact_sheet(job_id, filepath, data):
    """Generate a contact sheet from thumbnail images."""
    from opencut.core.batch_thumbnails import generate_contact_sheet

    thumbnails = data.get("thumbnails", [])
    if not thumbnails or not isinstance(thumbnails, list):
        raise ValueError("thumbnails must be a non-empty list of image paths")

    columns = safe_int(data.get("columns", 4), 4, min_val=1, max_val=20)
    output_path = data.get("output_path", "")
    if output_path:
        output_path = validate_output_path(output_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_contact_sheet(
        thumbnails=thumbnails,
        columns=columns,
        output_path=output_path or None,
        on_progress=_on_progress,
    )
    return result


# ===================================================================
# Annotations
# ===================================================================

@utility_bp.route("/annotations/add", methods=["POST"])
@require_csrf
def annotations_add():
    """Add an annotation to a project snapshot."""
    from opencut.core.annotations import add_annotation

    data = request.get_json(force=True)
    snapshot_id = data.get("snapshot_id", "").strip()
    if not snapshot_id:
        return jsonify({"error": "snapshot_id is required"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    annotation = add_annotation(
        snapshot_id=snapshot_id,
        text=text,
        change_ref=data.get("change_ref", ""),
        author=data.get("author", ""),
    )
    return jsonify(annotation)


@utility_bp.route("/annotations/list", methods=["GET"])
@require_csrf
def annotations_list():
    """List annotations for a snapshot. Query param: snapshot_id."""
    from opencut.core.annotations import get_annotations

    snapshot_id = request.args.get("snapshot_id", "").strip()
    if not snapshot_id:
        return jsonify({"error": "snapshot_id query parameter is required"}), 400

    annotations = get_annotations(snapshot_id)
    return jsonify({"annotations": annotations, "count": len(annotations)})


@utility_bp.route("/annotations/export", methods=["POST"])
@require_csrf
def annotations_export():
    """Export revision history for one or more snapshots."""
    from opencut.core.annotations import export_revision_history

    data = request.get_json(force=True)
    snapshot_ids = data.get("snapshot_ids", [])
    if not snapshot_ids or not isinstance(snapshot_ids, list):
        return jsonify({"error": "snapshot_ids must be a non-empty list"}), 400

    fmt = data.get("format", "markdown")
    result = export_revision_history(snapshot_ids, format=fmt)
    return jsonify({"content": result, "format": fmt})
