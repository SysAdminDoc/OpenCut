"""
OpenCut Developer & Scripting Platform Routes

Blueprint providing scripting console, macro recorder, filter chain
builder, webhook management, and batch scripting endpoints.

Blueprint: dev_scripting_bp (url_prefix /api)
16 routes covering all 5 developer features.
"""

import logging
import os
import time

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import get_json_dict, require_csrf, safe_float, safe_int, validate_path

logger = logging.getLogger("opencut")

dev_scripting_bp = Blueprint("dev_scripting", __name__)


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
# 1. Scripting Console  (3 routes)
# ===========================================================================

@dev_scripting_bp.route("/api/dev/scripting/execute", methods=["POST"])
@dev_scripting_bp.route("/api/scripting/execute", methods=["POST"])
@require_csrf
def scripting_execute():
    """Execute Python code in the sandboxed scripting console.

    Expects JSON::

        {
            "code": "print('hello')",
            "timeout": 30,
            "context": {}
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

    timeout = safe_int(data.get("timeout", 30), default=30, min_val=1, max_val=60)
    context = data.get("context", {})

    if context is None:
        context = {}
    if not isinstance(context, dict):
        return jsonify({"error": "context must be an object"}), 400

    try:
        result = execute_script(
            code=code,
            context=context,
            timeout=timeout,
        )
        return jsonify({
            "output": result.output,
            "error": result.error or None,
            "success": result.success,
            "execution_time_ms": round(result.execution_time_ms, 2),
        })
    except Exception as exc:
        return safe_error(exc, context="scripting_execute")


@dev_scripting_bp.route("/api/scripting/history", methods=["GET"])
def scripting_history():
    """Get the scripting console execution history.

    Query params:
        limit: Max entries to return (default 50).
    """
    from opencut.core.scripting_console import get_history

    limit = safe_int(request.args.get("limit", 50), default=50,
                     min_val=1, max_val=200)
    history = get_history(limit=limit)
    return jsonify({"history": history, "count": len(history)})


@dev_scripting_bp.route("/api/scripting/namespace", methods=["GET"])
def scripting_namespace():
    """List available functions and modules in the sandbox."""
    from opencut.core.scripting_console import get_namespace_info

    info = get_namespace_info()
    return jsonify(info)


# ===========================================================================
# 2. Macro Recorder  (5 routes)
# ===========================================================================

@dev_scripting_bp.route("/api/macro/record/start", methods=["POST"])
@require_csrf
def macro_record_start():
    """Start a macro recording session.

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


@dev_scripting_bp.route("/api/macro/record/stop", methods=["POST"])
@require_csrf
def macro_record_stop():
    """Stop recording and save the macro.

    Expects JSON::

        {
            "session_id": "default",
            "name": "My Macro",
            "description": ""
        }
    """
    from opencut.core.macro_recorder import save_macro, stop_recording

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
        path = save_macro(macro)
        return jsonify({
            "success": True,
            "name": macro.name,
            "step_count": len(macro.steps),
            "path": path,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@dev_scripting_bp.route("/api/dev/macro/play", methods=["POST"])
@dev_scripting_bp.route("/api/macro/play", methods=["POST"])
@require_csrf
@async_job("macro_play", filepath_required=False)
def macro_play(job_id, filepath, data):
    """Play a saved macro.

    Expects JSON::

        {
            "name": "My Macro",
            "target_file": "/path/to/file.mp4",
            "output_dir": ""
        }
    """
    from opencut.core.macro_recorder import (
        _macro_path,
        load_macro,
        play_macro,
    )

    name = data.get("name", "")
    target_file = data.get("target_file", filepath)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    name = _clean_string(name, "name", allow_empty=False)
    target_file = _clean_string(target_file, "target_file", default=filepath)
    output_dir = _clean_string(output_dir, "output_dir", default="")

    macro_path = _macro_path(name)
    if not os.path.isfile(macro_path):
        raise FileNotFoundError(f"Macro '{name}' not found")

    macro = load_macro(macro_path)

    def _progress(pct):
        _update_job(job_id, progress=pct,
                    message=f"Playing macro step ({pct}%)")

    results = play_macro(
        macro=macro,
        target_file=target_file,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return {
        "name": macro.name,
        "steps_played": len(results),
        "results": results,
    }


@dev_scripting_bp.route("/api/macro/list", methods=["GET"])
def macro_list():
    """List all saved macros."""
    from opencut.core.macro_recorder import list_macros

    macros = list_macros()
    return jsonify({"macros": macros, "count": len(macros)})


@dev_scripting_bp.route("/api/macro/<name>", methods=["DELETE"])
@require_csrf
def macro_delete(name):
    """Delete a saved macro by name."""
    from opencut.core.macro_recorder import delete_macro

    deleted = delete_macro(name)
    if deleted:
        return jsonify({"success": True, "message": f"Macro '{name}' deleted"})
    return jsonify({"error": f"Macro '{name}' not found"}), 404


# ===========================================================================
# 3. Filter Chain Builder  (2 routes)
# ===========================================================================

@dev_scripting_bp.route("/api/filter-chain/build", methods=["POST"])
@require_csrf
def filter_chain_build():
    """Build and validate a filter chain from a node graph.

    Expects JSON::

        {
            "nodes": [
                {"node_id": "n0", "filter_name": "scale", "params": {"w": 1280, "h": 720}},
                ...
            ],
            "connections": [
                {"from_node": "n0", "from_pad": "default", "to_node": "n1", "to_pad": "default"}
            ]
        }
    """
    from opencut.core.filter_chain_builder import (
        FilterChain,
        build_filter_string,
        validate_chain,
    )

    data, error = _json_object_or_400()
    if error:
        return error

    nodes = data.get("nodes", [])

    if not isinstance(nodes, list):
        return jsonify({"error": "nodes must be a list"}), 400
    if not nodes:
        return jsonify({"error": "At least one filter node is required"}), 400

    chain = FilterChain.from_dict(data)

    # Validate first
    validation = validate_chain(chain)

    if not validation["valid"]:
        return jsonify({
            "valid": False,
            "errors": validation["errors"],
            "warnings": validation.get("warnings", []),
        }), 400

    try:
        filter_string = build_filter_string(chain)
        return jsonify({
            "valid": True,
            "filter_complex": filter_string,
            "node_count": validation["node_count"],
            "warnings": validation.get("warnings", []),
        })
    except Exception as exc:
        return safe_error(exc, context="filter_chain_build")


@dev_scripting_bp.route("/api/filter-chain/preview", methods=["POST"])
@require_csrf
@async_job("filter_preview")
def filter_chain_preview(job_id, filepath, data):
    """Preview a filter chain applied to a single frame.

    Expects JSON::

        {
            "filepath": "/path/to/video.mp4",
            "filter_complex": "scale=1280:720",
            "timestamp": 0.0
        }
    """
    from opencut.core.filter_chain_builder import preview_filter

    filter_chain = data.get("filter_complex", "")
    timestamp = safe_float(data.get("timestamp", 0), default=0.0, min_val=0.0)

    if not filter_chain:
        raise ValueError("filter_complex string is required")

    def _progress(pct):
        _update_job(job_id, progress=pct, message="Generating preview...")

    output_path = preview_filter(
        video_path=filepath,
        filter_chain=filter_chain,
        timestamp=timestamp,
        on_progress=_progress,
    )

    return {"preview_path": output_path}


# ===========================================================================
# 4. Webhooks  (4 routes)
# ===========================================================================

@dev_scripting_bp.route("/api/webhooks", methods=["POST"])
@require_csrf
def webhook_register():
    """Register or update a webhook.

    Expects JSON::

        {
            "url": "https://example.com/hook",
            "events": ["job_complete", "job_failed"],
            "description": "My webhook",
            "id": "optional-existing-id"
        }
    """
    from opencut.core.webhook_system import register_webhook

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        url = _clean_string(data.get("url", ""), "url", allow_empty=False)
        description = _clean_string(data.get("description", ""), "description")
        webhook_id = _clean_string(data.get("id", ""), "id", default="") or None
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    events = data.get("events", [])
    if events is None:
        events = []
    if not isinstance(events, list):
        return jsonify({"error": "events must be a list"}), 400

    try:
        webhook = register_webhook(
            url=url,
            events=events,
            description=description,
            webhook_id=webhook_id,
        )
        return jsonify({
            "success": True,
            "webhook": webhook.to_dict(),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@dev_scripting_bp.route("/api/webhooks", methods=["GET"])
def webhook_list():
    """List all registered webhooks."""
    from opencut.core.webhook_system import list_webhooks

    webhooks = list_webhooks()
    return jsonify({"webhooks": webhooks, "count": len(webhooks)})


@dev_scripting_bp.route("/api/webhooks/<webhook_id>", methods=["DELETE"])
@require_csrf
def webhook_delete(webhook_id):
    """Remove a registered webhook by ID."""
    from opencut.core.webhook_system import unregister_webhook

    deleted = unregister_webhook(webhook_id)
    if deleted:
        return jsonify({
            "success": True,
            "message": f"Webhook '{webhook_id}' removed",
        })
    return jsonify({"error": f"Webhook '{webhook_id}' not found"}), 404


@dev_scripting_bp.route("/api/webhooks/test", methods=["POST"])
@require_csrf
def webhook_test():
    """Send a test event to a webhook.

    Expects JSON::

        {"id": "webhook-id"}
    """
    from opencut.core.webhook_system import test_webhook

    data, error = _json_object_or_400()
    if error:
        return error

    try:
        webhook_id = _clean_string(data.get("id", ""), "id", allow_empty=False)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        delivery = test_webhook(webhook_id)
        return jsonify({
            "success": delivery.success,
            "delivery": delivery.to_dict(),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


# ===========================================================================
# 5. Batch Scripting  (2 routes)
# ===========================================================================

@dev_scripting_bp.route("/api/batch/execute", methods=["POST"])
@require_csrf
@async_job("batch_script", filepath_required=False)
def batch_execute(job_id, filepath, data):
    """Execute a batch script.

    Expects JSON::

        {
            "name": "My Batch",
            "operations": [
                {
                    "operation": "silence",
                    "file_pattern": "/videos/*.mp4",
                    "parameters": {"threshold": -30},
                    "output_pattern": "${dir}/${basename}_clean${ext}",
                    "continue_on_error": true,
                    "skip_existing": false
                }
            ]
        }
    """
    from opencut.core.batch_script import BatchScript, execute_script

    operations = data.get("operations", [])
    if operations is None:
        operations = []
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")

    script = BatchScript.from_dict(data)
    if not script.name:
        script.name = "batch_" + str(int(time.time()))

    def _progress(pct):
        _update_job(job_id, progress=pct,
                    message=f"Batch processing ({pct}%)")

    result = execute_script(
        script=script,
        on_progress=_progress,
    )

    return result.to_dict()


@dev_scripting_bp.route("/api/batch/validate", methods=["POST"])
@require_csrf
def batch_validate():
    """Validate a batch script (dry run).

    Expects JSON with same format as /api/batch/execute.
    """
    from opencut.core.batch_script import BatchScript, validate_script

    data, error = _json_object_or_400()
    if error:
        return error

    operations = data.get("operations", [])
    if operations is None:
        operations = []
    if not isinstance(operations, list):
        return jsonify({"error": "operations must be a list"}), 400

    script = BatchScript.from_dict(data)

    result = validate_script(script)
    return jsonify(result.to_dict())

