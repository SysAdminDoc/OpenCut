"""
OpenCut Workflow & Developer Routes

Routes for undo stack, EDL/AAF import/export, project archival,
scripting console, macro recording, edit snapshots, through-edit
cleanup, and ripple edit automation.

Blueprint: workflow_dev_bp
15 routes covering all 8 features.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import get_json_dict, require_csrf, safe_bool, safe_float, validate_path

logger = logging.getLogger("opencut")

workflow_dev_bp = Blueprint("workflow_dev", __name__)


def _json_object_or_400():
    try:
        return get_json_dict(), None
    except ValueError as exc:
        return None, (
            jsonify({
                "error": str(exc),
                "code": "INVALID_INPUT",
                "suggestion": "Send a top-level JSON object in the request body.",
            }),
            400,
        )


def _clean_string(value, field_name, *, allow_empty=True, default=""):
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


# ===========================================================================
# 1. Undo Stack / Operation History  (3 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/undo/push", methods=["POST"])
@require_csrf
def undo_push():
    """Push an operation onto the undo stack.

    Expects JSON::

        {
            "operation": "silence_remove",
            "input_file": "/path/to/input.mp4",
            "output_file": "/path/to/output.mp4",
            "parameters": {"threshold": -30},
            "session_id": "default"
        }
    """
    from opencut.core.undo_stack import push_operation

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        session_id = _clean_string(data.get("session_id", "default"), "session_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    op_data = {
        "operation": data.get("operation", ""),
        "input_file": data.get("input_file", ""),
        "output_file": data.get("output_file", ""),
        "parameters": data.get("parameters", {}),
    }

    try:
        op_data["operation"] = _clean_string(op_data["operation"], "operation", allow_empty=False)
        op_data["input_file"] = _clean_string(op_data["input_file"], "input_file")
        op_data["output_file"] = _clean_string(op_data["output_file"], "output_file")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if op_data["parameters"] is None:
        op_data["parameters"] = {}
    if not isinstance(op_data["parameters"], dict):
        return jsonify({"error": "parameters must be an object"}), 400

    try:
        record = push_operation(op_data, session_id=session_id)
        from dataclasses import asdict
        return jsonify({"success": True, "record": asdict(record)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@workflow_dev_bp.route("/api/undo/history", methods=["GET"])
def undo_history():
    """Get the operation history for a session.

    Query params:
        session_id: Session identifier (default: "default").
    """
    from opencut.core.undo_stack import get_history

    session_id = request.args.get("session_id", "default")
    history = get_history(session_id=session_id)
    return jsonify({"history": history, "count": len(history)})


@workflow_dev_bp.route("/api/undo/undo", methods=["POST"])
@require_csrf
def undo_last():
    """Undo the last operation.

    Expects JSON::

        {"session_id": "default"}
    """
    from opencut.core.undo_stack import undo_last as _undo_last

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        session_id = _clean_string(data.get("session_id", "default"), "session_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = _undo_last(session_id=session_id)
    if result is None:
        return jsonify({"error": "Nothing to undo"}), 404
    return jsonify(result)


@workflow_dev_bp.route("/api/undo/clear", methods=["POST"])
@require_csrf
def undo_clear():
    """Clear the operation history for a session.

    Expects JSON::

        {"session_id": "default"}
    """
    from opencut.core.undo_stack import clear_history

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        session_id = _clean_string(data.get("session_id", "default"), "session_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    count = clear_history(session_id=session_id)
    return jsonify({"success": True, "cleared": count})


# ===========================================================================
# 2. EDL / AAF Import & Export  (3 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/edl/export", methods=["POST"])
@require_csrf
def edl_export():
    """Export a cut list as CMX3600 EDL.

    Expects JSON::

        {
            "cuts": [...],
            "output_path": "/path/to/output.edl",
            "title": "My Edit",
            "fps": 30.0
        }
    """
    from opencut.core.edl_aaf import export_edl

    data, error = _json_object_or_400()
    if error:
        return error

    cuts = data.get("cuts", [])
    try:
        output_path = _clean_string(data.get("output_path", ""), "output_path", allow_empty=False)
        title = _clean_string(data.get("title", "OpenCut Export"), "title", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        output_path = validate_path(output_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fps = safe_float(data.get("fps", 30.0), default=30.0, min_val=0.001, max_val=240.0)

    if not isinstance(cuts, list):
        return jsonify({"error": "cuts must be a list"}), 400
    if not cuts:
        return jsonify({"error": "No cuts provided"}), 400

    try:
        result = export_edl(cuts, output_path, title=title, fps=fps)
        return jsonify({
            "success": True,
            "output_path": result.output_path,
            "event_count": result.event_count,
        })
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


@workflow_dev_bp.route("/api/edl/import", methods=["POST"])
@require_csrf
def edl_import():
    """Import a CMX3600 EDL file.

    Expects JSON::

        {
            "edl_path": "/path/to/input.edl",
            "fps": 30.0
        }
    """
    from opencut.core.edl_aaf import import_edl

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        edl_path = _clean_string(data.get("edl_path", ""), "edl_path", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        edl_path = validate_path(edl_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fps = safe_float(data.get("fps", 30.0), default=30.0, min_val=0.001, max_val=240.0)

    try:
        result = import_edl(edl_path, fps=fps)
        return jsonify({
            "success": True,
            "cuts": result.cuts,
            "title": result.title,
            "event_count": result.event_count,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


@workflow_dev_bp.route("/api/aaf/export", methods=["POST"])
@require_csrf
def aaf_export():
    """Export a cut list as an AAF stub (JSON structure).

    Expects JSON::

        {
            "cuts": [...],
            "output_path": "/path/to/output.json",
            "title": "My Edit",
            "fps": 30.0
        }
    """
    from opencut.core.edl_aaf import export_aaf_stub

    data, error = _json_object_or_400()
    if error:
        return error

    cuts = data.get("cuts", [])
    try:
        output_path = _clean_string(data.get("output_path", ""), "output_path", allow_empty=False)
        title = _clean_string(data.get("title", "OpenCut Export"), "title", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        output_path = validate_path(output_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    fps = safe_float(data.get("fps", 30.0), default=30.0, min_val=0.001, max_val=240.0)

    if not isinstance(cuts, list):
        return jsonify({"error": "cuts must be a list"}), 400
    if not cuts:
        return jsonify({"error": "No cuts provided"}), 400

    try:
        result = export_aaf_stub(cuts, output_path, title=title, fps=fps)
        return jsonify({"success": True, **result})
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


# ===========================================================================
# 3. Project Archival / Package  (3 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/project/archive", methods=["POST"])
@require_csrf
@async_job("project_archive", filepath_required=False)
def project_archive(job_id, filepath, data):
    """Create a project archive.

    Expects JSON::

        {
            "project_data": {
                "name": "My Project",
                "source_files": [...],
                "output_files": [...],
                "workflows": [...],
                "presets": [...]
            },
            "output_path": "/path/to/archive.zip"
        }
    """
    from opencut.core.project_archive import create_archive

    project_data = data.get("project_data", {})
    if project_data is None:
        project_data = {}
    if not isinstance(project_data, dict):
        raise ValueError("project_data must be an object")

    output_path = _clean_string(data.get("output_path", ""), "output_path", allow_empty=False)

    try:
        output_path = validate_path(output_path)
    except ValueError as exc:
        raise ValueError(str(exc))

    def on_progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = create_archive(project_data, output_path, on_progress=on_progress)
    return {
        "archive_path": result.archive_path,
        "total_files": result.total_files,
        "total_bytes": result.total_bytes,
    }


@workflow_dev_bp.route("/api/project/restore", methods=["POST"])
@require_csrf
@async_job("project_restore", filepath_required=False)
def project_restore(job_id, filepath, data):
    """Restore a project from an archive.

    Expects JSON::

        {
            "archive_path": "/path/to/archive.zip",
            "dest_path": "/path/to/restore/dir"
        }
    """
    from opencut.core.project_archive import restore_archive

    archive_path = _clean_string(data.get("archive_path", ""), "archive_path", allow_empty=False)
    dest_path = _clean_string(data.get("dest_path", ""), "dest_path", allow_empty=False)

    try:
        archive_path = validate_path(archive_path)
        dest_path = validate_path(dest_path)
    except ValueError as exc:
        raise ValueError(str(exc))

    def on_progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = restore_archive(archive_path, dest_path, on_progress=on_progress)
    return {
        "dest_path": result.dest_path,
        "files_restored": result.files_restored,
        "manifest": result.manifest,
    }


@workflow_dev_bp.route("/api/project/archive/contents", methods=["POST"])
@require_csrf
def project_archive_contents():
    """List the contents of a project archive without extracting.

    Expects JSON::

        {"archive_path": "/path/to/archive.zip"}
    """
    from opencut.core.project_archive import list_archive_contents

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        archive_path = _clean_string(data.get("archive_path", ""), "archive_path", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        archive_path = validate_path(archive_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = list_archive_contents(archive_path)
        return jsonify({"success": True, **result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# ===========================================================================
# 4. Python Scripting Console  (2 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/scripting/execute", methods=["POST"])
@require_csrf
def scripting_execute():
    """Execute Python code in a sandboxed environment.

    Expects JSON::

        {
            "code": "print('hello')",
            "context": {"my_var": 42}
        }
    """
    from opencut.core.scripting_console import execute_script

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        code = _clean_string(data.get("code", ""), "code", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    context = data.get("context", {})

    if context is None:
        context = {}
    if not isinstance(context, dict):
        return jsonify({"error": "context must be an object"}), 400

    result = execute_script(code, context=context)
    return jsonify(result)


@workflow_dev_bp.route("/api/scripting/modules", methods=["GET"])
def scripting_modules():
    """List available modules in the scripting sandbox."""
    from opencut.core.scripting_console import get_available_modules

    modules = get_available_modules()
    return jsonify({"modules": modules, "count": len(modules)})


# ===========================================================================
# 5. Macro Recording & Playback  (5 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/macro/start", methods=["POST"])
@require_csrf
def macro_start():
    """Start macro recording.

    Expects JSON::

        {"session_id": "default"}
    """
    from opencut.core.macro_recorder import start_recording

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        session_id = _clean_string(data.get("session_id", "default"), "session_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = start_recording(session_id=session_id)
    return jsonify(result)


@workflow_dev_bp.route("/api/macro/stop", methods=["POST"])
@require_csrf
def macro_stop():
    """Stop macro recording and return the captured macro.

    Expects JSON::

        {
            "session_id": "default",
            "name": "My Macro",
            "description": "Does things"
        }
    """
    from opencut.core.macro_recorder import stop_recording

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        session_id = _clean_string(data.get("session_id", "default"), "session_id")
        name = _clean_string(data.get("name", "Untitled Macro"), "name", allow_empty=False)
        description = _clean_string(data.get("description", ""), "description")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        macro = stop_recording(
            session_id=session_id,
            name=name,
            description=description,
        )
        return jsonify({
            "success": True,
            "macro": macro.to_dict(),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@workflow_dev_bp.route("/api/macro/play", methods=["POST"])
@require_csrf
def macro_play():
    """Play back a macro (dry run).

    Expects JSON::

        {
            "macro": {... macro dict ...},
            "target_file": "/path/to/file.mp4"
        }
    """
    from opencut.core.macro_recorder import Macro, play_macro

    data, error = _json_object_or_400()
    if error:
        return error

    macro_data = data.get("macro", {})
    try:
        target_file = _clean_string(data.get("target_file", ""), "target_file", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not isinstance(macro_data, dict) or not macro_data:
        return jsonify({"error": "macro must be an object"}), 400

    macro = Macro.from_dict(macro_data)
    results = play_macro(macro, target_file)
    return jsonify({"results": results, "step_count": len(results)})


@workflow_dev_bp.route("/api/macro/save", methods=["POST"])
@require_csrf
def macro_save():
    """Save a macro to a JSON file.

    Expects JSON::

        {
            "macro": {... macro dict ...},
            "path": "/path/to/macro.json"
        }
    """
    from opencut.core.macro_recorder import Macro, save_macro

    data, error = _json_object_or_400()
    if error:
        return error

    macro_data = data.get("macro", {})
    try:
        path = _clean_string(data.get("path", ""), "path", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not isinstance(macro_data, dict) or not macro_data:
        return jsonify({"error": "macro must be an object"}), 400

    macro = Macro.from_dict(macro_data)

    try:
        saved_path = save_macro(macro, path)
        return jsonify({"success": True, "path": saved_path})
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


@workflow_dev_bp.route("/api/macro/load", methods=["POST"])
@require_csrf
def macro_load():
    """Load a macro from a JSON file.

    Expects JSON::

        {"path": "/path/to/macro.json"}
    """
    from opencut.core.macro_recorder import load_macro

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        path = _clean_string(data.get("path", ""), "path", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        macro = load_macro(path)
        return jsonify({"success": True, "macro": macro.to_dict()})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


# ===========================================================================
# 6. Edit Decision Snapshots  (4 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/snapshots/create", methods=["POST"])
@require_csrf
def snapshots_create():
    """Create a named snapshot.

    Expects JSON::

        {
            "name": "Before color grade",
            "project_id": "default",
            "project_state": {
                "job_history": [...],
                "output_files": [...],
                "parameters": {...}
            }
        }
    """
    from opencut.core.edit_snapshots import create_snapshot

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        name = _clean_string(data.get("name", ""), "name", allow_empty=False)
        project_id = _clean_string(data.get("project_id", "default"), "project_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    project_state = data.get("project_state", {})

    if project_state is None:
        project_state = {}
    if not isinstance(project_state, dict):
        return jsonify({"error": "project_state must be an object"}), 400

    try:
        result = create_snapshot(name, project_state, project_id=project_id)
        return jsonify({"success": True, **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@workflow_dev_bp.route("/api/snapshots/restore", methods=["POST"])
@require_csrf
def snapshots_restore():
    """Restore a named snapshot.

    Expects JSON::

        {
            "name": "Before color grade",
            "project_id": "default"
        }
    """
    from opencut.core.edit_snapshots import restore_snapshot

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        name = _clean_string(data.get("name", ""), "name", allow_empty=False)
        project_id = _clean_string(data.get("project_id", "default"), "project_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = restore_snapshot(name, project_id=project_id)
        return jsonify({"success": True, **result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@workflow_dev_bp.route("/api/snapshots/list", methods=["GET"])
def snapshots_list():
    """List all snapshots for a project.

    Query params:
        project_id: Project identifier (default: "default").
    """
    from opencut.core.edit_snapshots import list_snapshots

    project_id = request.args.get("project_id", "default")
    snapshots = list_snapshots(project_id=project_id)
    return jsonify({"snapshots": snapshots, "count": len(snapshots)})


@workflow_dev_bp.route("/api/snapshots/compare", methods=["POST"])
@require_csrf
def snapshots_compare():
    """Compare two snapshots to find differences.

    Expects JSON::

        {
            "name_a": "Before color grade",
            "name_b": "After color grade",
            "project_id": "default"
        }
    """
    from opencut.core.edit_snapshots import compare_snapshots

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        name_a = _clean_string(data.get("name_a", ""), "name_a", allow_empty=False)
        name_b = _clean_string(data.get("name_b", ""), "name_b", allow_empty=False)
        project_id = _clean_string(data.get("project_id", "default"), "project_id")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = compare_snapshots(name_a, name_b, project_id=project_id)
        return jsonify({"success": True, **result})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


# ===========================================================================
# 7. Through-Edit Cleanup  (1 route)
# ===========================================================================

@workflow_dev_bp.route("/api/timeline/through-edits", methods=["POST"])
@require_csrf
def timeline_through_edits():
    """Detect through-edits in a cut list.

    Expects JSON::

        {
            "cut_list": [
                {"source_file": "a.mp4", "source_in": 0, "source_out": 5},
                {"source_file": "a.mp4", "source_in": 5, "source_out": 10}
            ],
            "tolerance": 0.05,
            "auto_merge": false
        }
    """
    from opencut.core.through_edit import detect_through_edits, merge_through_edits

    data, error = _json_object_or_400()
    if error:
        return error

    cut_list = data.get("cut_list", [])
    tolerance = safe_float(data.get("tolerance", 0.05), default=0.05, min_val=0.0, max_val=10.0)
    auto_merge = safe_bool(data.get("auto_merge", False), False)

    if not isinstance(cut_list, list):
        return jsonify({"error": "cut_list must be a list"}), 400
    if not cut_list:
        return jsonify({"error": "No cut_list provided"}), 400

    result = detect_through_edits(cut_list, tolerance=tolerance)

    response = {
        "through_edits": [
            {
                "index_a": te.index_a,
                "index_b": te.index_b,
                "source_file": te.source_file,
                "gap_seconds": te.gap_seconds,
                "merged_in": te.merged_in,
                "merged_out": te.merged_out,
            }
            for te in result.through_edits
        ],
        "total_cuts": result.total_cuts,
        "mergeable_count": result.mergeable_count,
    }

    if auto_merge and result.through_edits:
        merged = merge_through_edits(cut_list, tolerance=tolerance)
        response["merged_cut_list"] = merged
        response["merged_count"] = len(merged)

    return jsonify(response)


# ===========================================================================
# 8. Ripple Edit Automation  (2 routes)
# ===========================================================================

@workflow_dev_bp.route("/api/timeline/detect-gaps", methods=["POST"])
@require_csrf
def timeline_detect_gaps():
    """Detect gaps in timeline without closing them.

    Expects JSON::

        {
            "timeline_items": [
                {"start": 0, "end": 5, "track": "V1"},
                {"start": 7, "end": 12, "track": "V1"}
            ],
            "min_gap": 0.001
        }
    """
    from opencut.core.ripple_edit import detect_gaps

    data, error = _json_object_or_400()
    if error:
        return error

    timeline_items = data.get("timeline_items", [])
    min_gap = safe_float(data.get("min_gap", 0.001), default=0.001, min_val=0.0, max_val=3600.0)

    if not isinstance(timeline_items, list):
        return jsonify({"error": "timeline_items must be a list"}), 400
    if not timeline_items:
        return jsonify({"error": "No timeline_items provided"}), 400

    result = detect_gaps(timeline_items, min_gap=min_gap)

    return jsonify({
        "gaps": [
            {
                "index": g.index,
                "start": g.start,
                "end": g.end,
                "duration": g.duration,
                "track": g.track,
            }
            for g in result.gaps
        ],
        "total_gap_duration": result.total_gap_duration,
        "timeline_duration": result.timeline_duration,
        "item_count": result.item_count,
    })


@workflow_dev_bp.route("/api/timeline/ripple-close", methods=["POST"])
@require_csrf
def timeline_ripple_close():
    """Close gaps in timeline via ripple edit.

    Expects JSON::

        {
            "timeline_items": [
                {"start": 0, "end": 5, "track": "V1"},
                {"start": 7, "end": 12, "track": "V1"}
            ],
            "locked_tracks": ["A1"],
            "min_gap": 0.001
        }
    """
    from opencut.core.ripple_edit import ripple_close_gaps

    data, error = _json_object_or_400()
    if error:
        return error

    timeline_items = data.get("timeline_items", [])
    locked_tracks = data.get("locked_tracks", [])
    min_gap = safe_float(data.get("min_gap", 0.001), default=0.001, min_val=0.0, max_val=3600.0)

    if not isinstance(timeline_items, list):
        return jsonify({"error": "timeline_items must be a list"}), 400
    if locked_tracks is None:
        locked_tracks = []
    if not isinstance(locked_tracks, list):
        return jsonify({"error": "locked_tracks must be a list"}), 400
    if not timeline_items:
        return jsonify({"error": "No timeline_items provided"}), 400

    result = ripple_close_gaps(
        timeline_items,
        locked_tracks=locked_tracks,
        min_gap=min_gap,
    )

    return jsonify({
        "items": result.items,
        "gaps_closed": result.gaps_closed,
        "total_shift": result.total_shift,
        "locked_tracks_skipped": result.locked_tracks_skipped,
    })
