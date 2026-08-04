"""
OpenCut Workflow Routes

Run, list, save, and delete multi-step processing workflows.
"""

import logging
import copy
import inspect
import time

from flask import Blueprint, current_app, has_app_context, jsonify

from opencut.jobs import _update_job, async_job
from opencut.routes._common import _json_object_or_400
from opencut.security import (
    build_destructive_plan,
    destructive_confirmation_required_response,
    require_csrf,
    safe_bool,
    verify_destructive_confirm_token,
)
from opencut.user_data import (
    build_user_data_destructive_record,
    create_user_tombstone,
    load_workflows,
    save_workflows,
    summarize_user_tombstone,
    user_file_lock,
)

logger = logging.getLogger("opencut")

workflow_bp = Blueprint("workflow", __name__)
_WORKFLOW_CAPABILITY_CACHE = {"ts": 0.0, "value": None}


def _extract_workflow_steps(data):
    """Accept both ``workflow: [...]`` and ``workflow: {steps: [...]}`` payloads."""
    workflow = data.get("workflow")
    if isinstance(workflow, dict):
        return workflow.get("steps", [])
    if workflow is not None:
        return workflow
    return data.get("steps", [])


def _extract_workflow_plan(data):
    """Accept a compiled plan at the top level or inside ``workflow``."""
    plan = data.get("plan")
    if isinstance(plan, dict):
        return plan
    workflow = data.get("workflow")
    if isinstance(workflow, dict) and isinstance(workflow.get("plan"), dict):
        return workflow["plan"]
    return None


def _workflow_steps_with_output_dir(data):
    """Normalize the route-level output directory into the immutable steps."""
    steps = _extract_workflow_steps(data)
    output_dir = data.get("output_dir", "")
    if not output_dir or not isinstance(steps, list):
        return steps
    normalized = copy.deepcopy(steps)
    for step in normalized:
        if isinstance(step, dict):
            params = step.setdefault("params", {})
            if isinstance(params, dict) and "output_dir" not in params:
                params["output_dir"] = output_dir
    return normalized


def _workflow_runtime_capabilities():
    """Build the small capability snapshot used by workflow preflight."""
    now = time.time()
    cached = _WORKFLOW_CAPABILITY_CACHE.get("value")
    if isinstance(cached, dict) and now - float(_WORKFLOW_CAPABILITY_CACHE.get("ts") or 0.0) < 30:
        return copy.deepcopy(cached)
    capabilities = {}
    try:
        from opencut.routes.system import _build_capabilities

        capabilities.update(_build_capabilities())
    except Exception as exc:  # pragma: no cover - defensive in minimal installs
        logger.debug("Workflow capability registry unavailable: %s", exc)
    try:
        from opencut.core.capability_profile import build_profile

        profile = build_profile()
        capabilities["ffmpeg"] = profile.get("ffmpeg", {})
        capabilities["ffprobe"] = profile.get("ffprobe", {})
        capabilities["gpu"] = profile.get("gpu", {})
    except Exception as exc:  # pragma: no cover - capability probe is advisory
        logger.debug("Workflow capability profile unavailable: %s", exc)
    _WORKFLOW_CAPABILITY_CACHE["ts"] = now
    _WORKFLOW_CAPABILITY_CACHE["value"] = copy.deepcopy(capabilities)
    return capabilities


def _compile_workflow(filepath, steps, output_dir=""):
    from opencut.core.workflow import compile_workflow_plan

    return compile_workflow_plan(
        filepath,
        steps,
        output_dir=output_dir,
        capabilities=_workflow_runtime_capabilities(),
    )


def _workflow_resume_has_approval(plan, resume_state):
    """Preserve approval across a restart without restoring the secret token."""
    if not isinstance(resume_state, dict):
        return False
    plan_id = str(plan.get("plan_id") or "")
    if not plan_id:
        return False

    prior_result = resume_state.get("result")
    if isinstance(prior_result, dict) and str(prior_result.get("plan_id") or "") == plan_id:
        return True

    prior_payload = resume_state.get("payload")
    if not isinstance(prior_payload, dict):
        prior_payload = resume_state.get("_payload")
    prior_plan = _extract_workflow_plan(prior_payload) if isinstance(prior_payload, dict) else None
    prior_approval = prior_plan.get("approval") if isinstance(prior_plan, dict) else None
    return bool(
        isinstance(prior_plan, dict)
        and str(prior_plan.get("plan_id") or "") == plan_id
        and isinstance(prior_approval, dict)
        and prior_approval.get("approved") is True
        and str(prior_approval.get("plan_id") or "") == plan_id
    )


def _workflow_plan_from_request(data, filepath, *, require_ready=True, resume_state=None):
    """Return the exact client plan or compile a new legacy request."""
    from opencut.core.workflow import (
        compile_workflow_plan,
        validate_workflow_plan,
        workflow_plan_requires_approval,
    )

    steps = _workflow_steps_with_output_dir(data)
    plan = _extract_workflow_plan(data)
    if plan is None:
        plan = compile_workflow_plan(
            filepath,
            steps,
            output_dir=data.get("output_dir", ""),
            capabilities=_workflow_runtime_capabilities(),
        )
    else:
        valid, error = validate_workflow_plan(
            plan,
            filepath=filepath,
            steps=steps if steps else None,
        )
        if not valid:
            raise ValueError(error)

    preflight = plan.get("preflight") or {}
    if require_ready and preflight.get("status") == "blocked":
        reasons = "; ".join(str(item) for item in preflight.get("blocked_reasons", []))
        raise ValueError("Workflow preflight blocked: " + (reasons or "review the plan"))
    if require_ready and workflow_plan_requires_approval(plan):
        from opencut.core.workflow import validate_workflow_approval
        from opencut.security import get_csrf_token

        approval = plan.get("approval") or {}
        approved = validate_workflow_approval(
            plan,
            data.get("approval_token") or approval.get("token", ""),
            get_csrf_token(),
        )
        if not approved and _workflow_resume_has_approval(plan, resume_state):
            approved = True
        if not approved:
            raise ValueError(
                "Workflow requires explicit approval. Compile the plan, review destructive or cloud steps, then approve it."
            )
    return plan


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
# Compile and approve a workflow plan
# ---------------------------------------------------------------------------
@workflow_bp.route("/workflow/compile", methods=["POST"])
@require_csrf
def compile_workflow_route():
    """Compile a workflow without starting any processing job.

    The response always includes the reviewable plan, including blocked
    checks, so CEP/UXP can explain a dependency, media, disk, or output issue
    before the user commits to a long-running run.
    """
    from opencut.core.workflow import compile_workflow_plan
    from opencut.security import validate_filepath

    data, error = _json_object_or_400()
    if error:
        return error
    filepath = data.get("filepath", "")
    if not isinstance(filepath, str) or not filepath.strip():
        return jsonify({"error": "No file path provided", "code": "INVALID_INPUT"}), 400
    try:
        filepath = validate_filepath(filepath.strip())
        plan = compile_workflow_plan(
            filepath,
            _workflow_steps_with_output_dir(data),
            output_dir=data.get("output_dir", ""),
            capabilities=_workflow_runtime_capabilities(),
        )
    except (ValueError, OSError) as exc:
        return jsonify({
            "error": str(exc),
            "code": "WORKFLOW_COMPILE_FAILED",
            "suggestion": "Review the workflow steps and selected media before compiling again.",
        }), 400
    return jsonify({
        "success": plan.get("preflight", {}).get("status") == "ready",
        "plan": plan,
        "requires_approval": safe_bool(
            (plan.get("approval") or {}).get("required"),
            False,
        ),
    })


@workflow_bp.route("/workflow/approve", methods=["POST"])
@require_csrf
def approve_workflow_route():
    """Bind explicit user approval to a compiled workflow plan."""
    from opencut.core.workflow import (
        validate_workflow_plan,
        workflow_approval_token,
    )
    from opencut.security import get_csrf_token

    data, error = _json_object_or_400()
    if error:
        return error
    plan = _extract_workflow_plan(data)
    if not isinstance(plan, dict):
        return jsonify({"error": "Compiled workflow plan required", "code": "PLAN_REQUIRED"}), 400
    valid, reason = validate_workflow_plan(plan, filepath=str((plan.get("source") or {}).get("filepath") or ""))
    if not valid:
        return jsonify({"error": reason, "code": "INVALID_WORKFLOW_PLAN"}), 400
    if (plan.get("preflight") or {}).get("status") == "blocked":
        return jsonify({"error": "Workflow plan is blocked", "code": "WORKFLOW_PREFLIGHT_BLOCKED", "plan": plan}), 409

    approved_plan = copy.deepcopy(plan)
    token = workflow_approval_token(str(plan.get("plan_id") or ""), get_csrf_token())
    approved_plan["approval"] = {
        "required": safe_bool(
            (plan.get("approval") or {}).get("required"),
            False,
        ),
        "approved": True,
        "plan_id": plan.get("plan_id"),
        "token": token,
    }
    return jsonify({
        "success": True,
        "plan": approved_plan,
        "approval": approved_plan["approval"],
    })


# ---------------------------------------------------------------------------
# Run a Workflow (async job)
# ---------------------------------------------------------------------------
@workflow_bp.route("/workflow/run", methods=["POST"])
@require_csrf
@async_job("workflow", resumable=True, partial_output_param="partial_output_path")
def run_workflow_route(job_id, filepath, data):
    """Execute a multi-step workflow on a file.

    Expects JSON body::

        {
            "filepath": "/path/to/input.mp4",
            "workflow": {
                "steps": [
                    {"endpoint": "/silence", "params": {}},
                    {"endpoint": "/audio/normalize", "params": {}}
                ]
            }
        }
    """
    from opencut.core.workflow import run_workflow
    from opencut.security import get_csrf_token
    from opencut.server import app as server_app

    resume_state = None
    resume_source_job_id = str(
        data.get("resume_source_job_id") or data.get("resume_from_job_id") or ""
    ).strip()
    if resume_source_job_id:
        try:
            from opencut.job_store import get_job as get_persisted_job

            resume_state = get_persisted_job(resume_source_job_id)
        except Exception as exc:
            raise ValueError(f"Workflow resume state unavailable: {exc}") from exc
        if not resume_state:
            raise ValueError("Workflow resume source job was not found")

    steps = _workflow_steps_with_output_dir(data)
    plan = _workflow_plan_from_request(data, filepath, resume_state=resume_state)

    if resume_source_job_id:
        prior_result = resume_state.get("result") if isinstance(resume_state, dict) else None
        prior_plan_id = prior_result.get("plan_id") if isinstance(prior_result, dict) else ""
        if prior_plan_id and prior_plan_id != plan.get("plan_id"):
            raise ValueError("Workflow resume source belongs to a different plan")

    # Grab a CSRF token for internal requests
    csrf_token = get_csrf_token()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    def _on_checkpoint(checkpoint):
        _update_job(
            job_id,
            result=checkpoint,
            partial_output_path=checkpoint.get("output", ""),
            message="Checkpointed step %d/%d" % (
                checkpoint.get("steps_completed", 0),
                checkpoint.get("total_steps", len(steps)),
            ),
        )

    app_obj = current_app._get_current_object() if has_app_context() else server_app

    run_kwargs = {
        "app": app_obj,
        "filepath": filepath,
        "steps": steps,
        "csrf_token": csrf_token,
        "on_progress": _on_progress,
        "parent_job_id": job_id,
    }
    # Keep compatibility with downstream embedders that monkeypatch the old
    # six-argument engine while the built-in engine receives plan checkpoints.
    signature = inspect.signature(run_workflow)
    if "plan" in signature.parameters:
        run_kwargs.update({
            "plan": plan,
            "resume_state": resume_state,
            "on_checkpoint": _on_checkpoint,
        })
    result = run_workflow(**run_kwargs)

    if not result.get("success"):
        _update_job(
            job_id,
            result=result,
            partial_output_path=result.get("output", ""),
        )
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
    from opencut.core.workflow import compile_workflow_template

    data, error = _json_object_or_400()
    if error:
        return error
    steps = data.get("steps", [])
    name = data.get("name", "")
    description = data.get("description", "")

    if not isinstance(name, str):
        return jsonify({"error": "Workflow name must be a string"}), 400
    if description and not isinstance(description, str):
        return jsonify({"error": "Workflow description must be a string"}), 400

    name = name.strip()
    description = description.strip()

    if not name:
        return jsonify({"error": "Workflow name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Workflow name too long"}), 400
    if not isinstance(steps, list):
        return jsonify({"error": "Workflow steps must be a list"}), 400

    # Check for collision with built-in names
    builtin_names = {wf["name"] for wf in BUILTIN_WORKFLOWS}
    if name in builtin_names:
        return jsonify({"error": "Cannot overwrite a built-in workflow"}), 400

    if len(steps) > 50:
        return jsonify({"error": "Too many workflow steps (max 50)"}), 400

    try:
        template = compile_workflow_template(steps)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_WORKFLOW"}), 400
    if (template.get("preflight") or {}).get("status") == "blocked":
        return jsonify({
            "error": "Workflow cannot be saved until preflight blockers are resolved",
            "code": "WORKFLOW_PREFLIGHT_BLOCKED",
            "plan": template,
        }), 400

    with user_file_lock("workflows.json"):
        workflows = load_workflows()

        # Update existing or append
        found = False
        for wf in workflows:
            if wf.get("name") == name:
                wf["steps"] = steps
                wf["description"] = description
                wf["definition_id"] = template.get("definition_id", "")
                wf["plan_template"] = template
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
                "definition_id": template.get("definition_id", ""),
                "plan_template": template,
                "created": time.time(),
            })

        save_workflows(workflows)
    return jsonify({
        "success": True,
        "name": name,
        "definition_id": template.get("definition_id", ""),
        "plan_template": template,
    })


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
    data, error = _json_object_or_400()
    if error:
        return error
    name = data.get("name", "")
    if not isinstance(name, str):
        return jsonify({"error": "Workflow name must be a string"}), 400
    name = name.strip()

    if not name:
        return jsonify({"error": "Workflow name required"}), 400

    # Prevent deletion of built-in workflows
    builtin_names = {wf["name"] for wf in BUILTIN_WORKFLOWS}
    if name in builtin_names:
        return jsonify({"error": "Cannot delete a built-in workflow"}), 400

    with user_file_lock("workflows.json"):
        workflows = load_workflows()
        original_len = len(workflows)
        removed = next((wf for wf in workflows if wf.get("name") == name), None)
        workflows = [wf for wf in workflows if wf.get("name") != name]

        if len(workflows) == original_len:
            return jsonify({"error": "Workflow not found"}), 404

        record = build_user_data_destructive_record(
            "workflow",
            name,
            removed or {"name": name},
            source_file="workflows.json",
            route="/workflow/delete",
        )
        plan = build_destructive_plan(
            "user_data.workflow.delete",
            records=[record],
            metadata={"route": "/workflow/delete", "name": name, "tombstone": True},
            reversible=True,
        )
        dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
        if dry_run:
            return jsonify({
                "success": True,
                "dry_run": True,
                "deleted": None,
                "would_delete": name,
                "destructive_plan": plan,
                "confirm_token": plan["confirm_token"],
            })
        if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409

        tombstone = create_user_tombstone(
            "workflow",
            name,
            removed or {"name": name},
            source_file="workflows.json",
            metadata={"route": "/workflow/delete"},
        )
        save_workflows(workflows)
    return jsonify({
        "success": True,
        "tombstone": summarize_user_tombstone(tombstone),
    })
