"""
OpenCut Workflow Routes

Run, list, save, and delete multi-step processing workflows.
"""

import logging
import time

from flask import Blueprint, current_app, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf
from opencut.user_data import load_workflows, save_workflows

logger = logging.getLogger("opencut")

workflow_bp = Blueprint("workflow", __name__)

# ---------------------------------------------------------------------------
# Built-in Workflow Presets
# ---------------------------------------------------------------------------
BUILTIN_WORKFLOWS = [
    {
        "name": "Clean Interview",
        "builtin": True,
        "description": "Detect silence, remove it, then normalize audio levels.",
        "steps": [
            {"endpoint": "/silence", "params": {}},
            {"endpoint": "/audio/normalize", "params": {}},
        ],
    },
    {
        "name": "Podcast Polish",
        "builtin": True,
        "description": "Denoise audio, normalize, and match loudness to -16 LUFS.",
        "steps": [
            {"endpoint": "/audio/denoise", "params": {}},
            {"endpoint": "/audio/normalize", "params": {}},
            {"endpoint": "/audio/loudness-match", "params": {"target_lufs": -16}},
        ],
    },
    {
        "name": "Social Media Clip",
        "builtin": True,
        "description": "Auto-edit, reframe to 9:16 portrait, and export.",
        "steps": [
            {"endpoint": "/video/auto-edit", "params": {}},
            {"endpoint": "/video/reframe", "params": {"aspect": "9:16"}},
            {"endpoint": "/export-video", "params": {}},
        ],
    },
    {
        "name": "YouTube Upload",
        "builtin": True,
        "description": "Detect and remove silence, normalize audio, match loudness to -14 LUFS.",
        "steps": [
            {"endpoint": "/silence", "params": {}},
            {"endpoint": "/audio/normalize", "params": {}},
            {"endpoint": "/audio/loudness-match", "params": {"target_lufs": -14}},
        ],
    },
    {
        "name": "Documentary Rough Cut",
        "builtin": True,
        "description": "Detect scenes, auto-edit, then normalize audio.",
        "steps": [
            {"endpoint": "/video/scenes", "params": {}},
            {"endpoint": "/video/auto-edit", "params": {}},
            {"endpoint": "/audio/normalize", "params": {}},
        ],
    },
    {
        "name": "Studio Audio",
        "builtin": True,
        "description": "Denoise, normalize, and match loudness to -14 LUFS.",
        "steps": [
            {"endpoint": "/audio/denoise", "params": {}},
            {"endpoint": "/audio/normalize", "params": {}},
            {"endpoint": "/audio/loudness-match", "params": {"target_lufs": -14}},
        ],
    },
]


# ---------------------------------------------------------------------------
# Run a Workflow (async job)
# ---------------------------------------------------------------------------
@workflow_bp.route("/workflow/run", methods=["POST"])
@require_csrf
@async_job("workflow")
def run_workflow_route(job_id, filepath, data):
    """Execute a multi-step workflow on a file.

    Expects JSON body::

        {
            "filepath": "/path/to/input.mp4",
            "workflow": [
                {"endpoint": "/silence", "params": {}},
                {"endpoint": "/audio/normalize", "params": {}}
            ]
        }
    """
    from opencut.core.workflow import run_workflow, validate_workflow_steps
    from opencut.security import get_csrf_token

    steps = data.get("workflow", [])

    # Validate
    valid, error = validate_workflow_steps(steps)
    if not valid:
        raise ValueError(error)

    # Grab a CSRF token for internal requests
    csrf_token = get_csrf_token()

    def _on_progress(pct, msg):
        _update_job(job_id, progress=pct, message=msg)

    result = run_workflow(
        app=current_app._get_current_object(),
        filepath=filepath,
        steps=steps,
        csrf_token=csrf_token,
        on_progress=_on_progress,
        parent_job_id=job_id,
    )

    if not result.get("success"):
        raise RuntimeError(result.get("error", "Workflow failed"))

    return result


# ---------------------------------------------------------------------------
# Workflow Presets (built-in + user-saved)
# ---------------------------------------------------------------------------
@workflow_bp.route("/workflow/presets", methods=["GET"])
def list_workflow_presets():
    """Return built-in workflow presets plus any user-saved workflows."""
    user_workflows = load_workflows()
    # Tag user workflows so the frontend can distinguish
    tagged_user = []
    for wf in user_workflows:
        entry = dict(wf)
        entry["builtin"] = False
        tagged_user.append(entry)
    return jsonify({"builtins": BUILTIN_WORKFLOWS, "custom": tagged_user})


# ---------------------------------------------------------------------------
# Save a Custom Workflow
# ---------------------------------------------------------------------------
@workflow_bp.route("/workflow/save", methods=["POST"])
@require_csrf
def save_custom_workflow():
    """Save a named custom workflow.

    Expects JSON::

        {"name": "My Workflow", "steps": [...], "description": "optional"}
    """
    from opencut.core.workflow import validate_workflow_steps

    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    steps = data.get("steps", [])
    description = data.get("description", "").strip()

    if not name:
        return jsonify({"error": "Workflow name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Workflow name too long"}), 400

    # Check for collision with built-in names
    builtin_names = {wf["name"] for wf in BUILTIN_WORKFLOWS}
    if name in builtin_names:
        return jsonify({"error": "Cannot overwrite a built-in workflow"}), 400

    if len(steps) > 50:
        return jsonify({"error": "Too many workflow steps (max 50)"}), 400

    valid, error = validate_workflow_steps(steps)
    if not valid:
        return jsonify({"error": error}), 400

    workflows = load_workflows()

    # Update existing or append
    found = False
    for wf in workflows:
        if wf.get("name") == name:
            wf["steps"] = steps
            wf["description"] = description
            wf["updated"] = time.time()
            found = True
            break

    if not found:
        if len(workflows) >= 100:
            return jsonify({"error": "Too many custom workflows (max 100)"}), 400
        workflows.append({
            "name": name,
            "steps": steps,
            "description": description,
            "created": time.time(),
        })

    save_workflows(workflows)
    return jsonify({"success": True, "name": name})


# ---------------------------------------------------------------------------
# Delete a Custom Workflow
# ---------------------------------------------------------------------------
@workflow_bp.route("/workflow/delete", methods=["DELETE"])
@require_csrf
def delete_custom_workflow():
    """Delete a saved custom workflow by name.

    Expects JSON::

        {"name": "My Workflow"}
    """
    data = request.get_json(force=True)
    name = data.get("name", "").strip()

    if not name:
        return jsonify({"error": "Workflow name required"}), 400

    # Prevent deletion of built-in workflows
    builtin_names = {wf["name"] for wf in BUILTIN_WORKFLOWS}
    if name in builtin_names:
        return jsonify({"error": "Cannot delete a built-in workflow"}), 400

    workflows = load_workflows()
    original_len = len(workflows)
    workflows = [wf for wf in workflows if wf.get("name") != name]

    if len(workflows) == original_len:
        return jsonify({"error": "Workflow not found"}), 404

    save_workflows(workflows)
    return jsonify({"success": True})
