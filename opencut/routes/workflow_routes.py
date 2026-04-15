"""
OpenCut Workflow Automation Routes

Watch folders, render queue, conditional workflows, and best-take selection.
"""

import logging
import os

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import get_json_dict, require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

workflow_auto_bp = Blueprint("workflow_auto", __name__)


def _json_object_or_400():
    try:
        return get_json_dict(), None
    except ValueError as e:
        return None, (
            jsonify({
                "error": str(e),
                "code": "INVALID_INPUT",
                "suggestion": "Send a top-level JSON object in the request body.",
            }),
            400,
        )


def _string_field(data: dict, key: str) -> str:
    value = data.get(key, "")
    if isinstance(value, str):
        return value.strip()
    return ""


# ===========================================================================
# Watch Folder Routes
# ===========================================================================

@workflow_auto_bp.route("/watch/start", methods=["POST"])
@require_csrf
def watch_start():
    """Start a watch folder monitor.

    Expects JSON::

        {
            "folder_path": "/path/to/watch",
            "workflow_name": "Clean Interview",
            "output_dir": "/path/to/output",
            "file_extensions": [".mp4", ".mov"],
            "poll_interval_sec": 5
        }
    """
    from opencut.core.watch_folder import WatchFolderConfig, start_watch

    data, error = _json_object_or_400()
    if error:
        return error
    folder_path = _string_field(data, "folder_path")
    if not folder_path:
        return jsonify({"error": "folder_path is required"}), 400
    # Validate paths against traversal/null byte attacks
    try:
        from opencut.security import validate_path
        folder_path = validate_path(folder_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not os.path.isdir(folder_path):
        return jsonify({"error": f"Folder does not exist: {folder_path}"}), 400
    output_dir = _string_field(data, "output_dir")
    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not os.path.isdir(output_dir):
            return jsonify({"error": f"Output folder does not exist: {output_dir}"}), 400
    file_extensions = data.get("file_extensions")
    if file_extensions is not None and (
        not isinstance(file_extensions, list)
        or not all(isinstance(ext, str) and ext.strip() for ext in file_extensions)
    ):
        return jsonify({"error": "file_extensions must be a list of strings"}), 400

    config = WatchFolderConfig(
        folder_path=folder_path,
        workflow_name=_string_field(data, "workflow_name"),
        output_dir=output_dir,
        file_extensions=[ext.strip() for ext in file_extensions] if file_extensions else None,
        poll_interval_sec=safe_float(data.get("poll_interval_sec", 5), default=5.0, min_val=1.0),
    )

    try:
        handle = start_watch(config)
        return jsonify({"success": True, "watcher_id": handle.id, "config": {
            "folder_path": config.folder_path,
            "workflow_name": config.workflow_name,
            "poll_interval_sec": config.poll_interval_sec,
        }})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@workflow_auto_bp.route("/watch/stop", methods=["POST"])
@require_csrf
def watch_stop():
    """Stop a running watch folder monitor.

    Expects JSON::

        {"watcher_id": "abc123def456"}
    """
    from opencut.core.watch_folder import _active_watches, _watches_lock, stop_watch

    data, error = _json_object_or_400()
    if error:
        return error
    watcher_id = _string_field(data, "watcher_id")
    if not watcher_id:
        return jsonify({"error": "watcher_id is required"}), 400

    with _watches_lock:
        handle = _active_watches.get(watcher_id)

    if handle is None:
        return jsonify({"error": "Watcher not found"}), 404

    stop_watch(handle)
    return jsonify({"success": True, "watcher_id": watcher_id})


@workflow_auto_bp.route("/watch/list", methods=["GET"])
def watch_list():
    """List all active watch folder monitors."""
    from opencut.core.watch_folder import list_active_watches
    watches = list_active_watches()
    return jsonify({"watches": watches, "count": len(watches)})


@workflow_auto_bp.route("/watch/config", methods=["POST"])
@require_csrf
def watch_config_save():
    """Save watch folder configurations for persistence.

    Expects JSON::

        {
            "configs": [
                {"folder_path": "/path", "workflow_name": "...", ...}
            ]
        }
    """
    from opencut.core.watch_folder import WatchFolderConfig, save_watch_configs

    data, error = _json_object_or_400()
    if error:
        return error
    raw_configs = data.get("configs", [])
    if not isinstance(raw_configs, list):
        return jsonify({"error": "configs must be a list"}), 400
    if len(raw_configs) > 100:
        return jsonify({"error": "Too many watch configs (max 100)"}), 400

    configs = []
    for item in raw_configs:
        if not isinstance(item, dict):
            continue
        folder_path = _string_field(item, "folder_path")
        if not folder_path:
            continue
        file_extensions = item.get("file_extensions")
        if file_extensions is not None and (
            not isinstance(file_extensions, list)
            or not all(isinstance(ext, str) and ext.strip() for ext in file_extensions)
        ):
            continue
        configs.append(WatchFolderConfig(
            folder_path=folder_path,
            workflow_name=_string_field(item, "workflow_name"),
            output_dir=_string_field(item, "output_dir"),
            file_extensions=[ext.strip() for ext in file_extensions] if file_extensions else None,
            poll_interval_sec=safe_float(item.get("poll_interval_sec", 5), default=5.0, min_val=1.0),
            id=_string_field(item, "id"),
        ))

    save_watch_configs(configs)
    return jsonify({"success": True, "saved": len(configs)})


# ===========================================================================
# Render Queue Routes
# ===========================================================================

@workflow_auto_bp.route("/render-queue/add", methods=["POST"])
@require_csrf
def render_queue_add():
    """Add an item to the render queue.

    Expects JSON::

        {
            "input_path": "/path/to/file.mp4",
            "preset_name": "youtube_1080p",
            "params": {},
            "priority": 3
        }
    """
    from opencut.core.render_queue import add_to_queue

    data, error = _json_object_or_400()
    if error:
        return error
    input_path = _string_field(data, "input_path")
    preset_name = _string_field(data, "preset_name")

    if not input_path:
        return jsonify({"error": "input_path is required"}), 400
    if not preset_name:
        return jsonify({"error": "preset_name is required"}), 400

    try:
        input_path = validate_filepath(input_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    priority = safe_int(data.get("priority", 3), default=3, min_val=1, max_val=5)
    params = data.get("params")
    if params is not None and not isinstance(params, dict):
        return jsonify({"error": "params must be an object"}), 400

    item_id = add_to_queue(input_path, preset_name, params=params, priority=priority)
    return jsonify({"success": True, "item_id": item_id})


@workflow_auto_bp.route("/render-queue/remove", methods=["DELETE"])
@require_csrf
def render_queue_remove():
    """Remove an item from the render queue.

    Expects JSON::

        {"item_id": "abc123"}
    """
    from opencut.core.render_queue import remove_from_queue

    data, error = _json_object_or_400()
    if error:
        return error
    item_id = _string_field(data, "item_id")
    if not item_id:
        return jsonify({"error": "item_id is required"}), 400

    removed = remove_from_queue(item_id)
    if not removed:
        return jsonify({"error": "Item not found or currently rendering"}), 404
    return jsonify({"success": True, "item_id": item_id})


@workflow_auto_bp.route("/render-queue/list", methods=["GET"])
def render_queue_list():
    """List all items in the render queue."""
    from dataclasses import asdict

    from opencut.core.render_queue import get_queue, is_queue_paused, is_queue_running

    items = get_queue()
    return jsonify({
        "items": [asdict(item) for item in items],
        "count": len(items),
        "running": is_queue_running(),
        "paused": is_queue_paused(),
    })


@workflow_auto_bp.route("/render-queue/start", methods=["POST"])
@require_csrf
def render_queue_start():
    """Start processing the render queue."""
    from opencut.core.render_queue import start_queue_processing
    start_queue_processing()
    return jsonify({"success": True, "message": "Render queue started"})


@workflow_auto_bp.route("/render-queue/pause", methods=["POST"])
@require_csrf
def render_queue_pause():
    """Pause the render queue."""
    from opencut.core.render_queue import pause_queue
    pause_queue()
    return jsonify({"success": True, "message": "Render queue paused"})


@workflow_auto_bp.route("/render-queue/resume", methods=["POST"])
@require_csrf
def render_queue_resume():
    """Resume the render queue."""
    from opencut.core.render_queue import resume_queue
    resume_queue()
    return jsonify({"success": True, "message": "Render queue resumed"})


# ===========================================================================
# Conditional Workflow Routes
# ===========================================================================

@workflow_auto_bp.route("/workflow/conditional", methods=["POST"])
@require_csrf
@async_job("conditional_workflow")
def workflow_conditional(job_id, filepath, data):
    """Run a conditional workflow on a media file.

    Expects JSON::

        {
            "filepath": "/path/to/file.mp4",
            "steps": [
                {"action": "normalize", "params": {}, "condition": "loudness_lufs < -20"},
                {"action": "export", "params": {"preset_name": "youtube_1080p"}}
            ]
        }
    """
    from opencut.core.conditional_workflow import run_conditional_workflow

    steps = data.get("steps", [])
    if not steps:
        raise ValueError("No workflow steps provided")
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")
    if len(steps) > 50:
        raise ValueError("Too many workflow steps (max 50)")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = run_conditional_workflow(
        input_path=filepath,
        steps=steps,
        on_progress=_on_progress,
    )
    return result


# ===========================================================================
# Best Take Routes
# ===========================================================================

@workflow_auto_bp.route("/takes/score", methods=["POST"])
@require_csrf
@async_job("take_scoring", filepath_required=False)
def takes_score(job_id, filepath, data):
    """Score multiple takes and recommend the best one.

    Expects JSON::

        {
            "takes": ["/path/to/take1.mp4", "/path/to/take2.mp4"]
        }
    """
    from dataclasses import asdict

    from opencut.core.best_take import score_takes

    takes = data.get("takes", [])
    if not takes:
        raise ValueError("No takes provided")
    if not isinstance(takes, list):
        raise ValueError("takes must be a list")
    if len(takes) > 50:
        raise ValueError("Too many takes (max 50)")

    # Validate all paths
    valid_takes = []
    for t in takes:
        t = t.strip() if isinstance(t, str) else ""
        if t:
            try:
                t = validate_filepath(t)
                valid_takes.append(t)
            except ValueError:
                logger.debug("Skipping invalid take path: %s", t)

    if not valid_takes:
        raise ValueError("No valid take paths provided")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = score_takes(valid_takes, on_progress=_on_progress)

    return {
        "takes": [asdict(t) for t in result.takes],
        "best_take": result.best_take,
        "total_scored": result.total_scored,
    }


@workflow_auto_bp.route("/takes/find-repeats", methods=["POST"])
@require_csrf
@async_job("find_repeats", filepath_required=False)
def takes_find_repeats(job_id, filepath, data):
    """Find repeated takes among a list of files.

    Expects JSON::

        {
            "file_paths": ["/path/to/file1.mp4", "/path/to/file2.mp4", ...],
            "similarity_threshold": 0.8
        }
    """
    from opencut.core.best_take import find_repeated_takes

    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise ValueError("No file paths provided")
    if not isinstance(file_paths, list):
        raise ValueError("file_paths must be a list")
    if len(file_paths) > 200:
        raise ValueError("Too many files (max 200)")

    threshold = safe_float(data.get("similarity_threshold", 0.8), default=0.8, min_val=0.1, max_val=1.0)

    # Validate paths
    valid_paths = []
    for fp in file_paths:
        fp = fp.strip() if isinstance(fp, str) else ""
        if fp:
            try:
                fp = validate_filepath(fp)
                valid_paths.append(fp)
            except ValueError:
                logger.debug("Skipping invalid path: %s", fp)

    if not valid_paths:
        raise ValueError("No valid file paths provided")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    groups = find_repeated_takes(
        valid_paths,
        similarity_threshold=threshold,
        on_progress=_on_progress,
    )

    return {
        "groups": groups,
        "group_count": len(groups),
        "total_files": len(valid_paths),
    }
