"""
OpenCut Collaboration & Review Routes

Blueprint ``collab_review_bp`` with url_prefix ``/api``.

Routes cover review comments, version comparison, approval workflow,
shared presets, and edit history export.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath, validate_path

logger = logging.getLogger("opencut")

collab_review_bp = Blueprint("collab_review", __name__)


# ===========================================================================
# Review Comments (5 routes)
# ===========================================================================

@collab_review_bp.route("/api/review/comments", methods=["POST"])
@require_csrf
def add_review_comment():
    """Add a review comment to a project.

    Expects JSON::

        {
            "project_path": "/path/to/project.prproj",
            "text": "Fix the color grading at this frame",
            "author": "editor1",
            "timestamp_sec": 42.5,
            "frame_number": 1275,
            "parent_id": "",
            "annotation_type": "text",
            "annotation_data": {},
            "tags": ["color", "fix"]
        }
    """
    from opencut.core.review_comments import add_comment

    data = request.get_json(force=True) or {}
    project_path = data.get("project_path", "")
    text = data.get("text", "")

    if not project_path:
        return jsonify({"error": "project_path is required"}), 400
    try:
        project_path = validate_path(project_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not text or not text.strip():
        return jsonify({"error": "Comment text is required"}), 400

    try:
        result = add_comment(
            project_path=project_path,
            text=text,
            author=data.get("author", "anonymous"),
            timestamp_sec=safe_float(data.get("timestamp_sec"), 0.0, min_val=0.0),
            frame_number=safe_int(data.get("frame_number"), 0, min_val=0),
            parent_id=data.get("parent_id", ""),
            annotation_type=data.get("annotation_type", "text"),
            annotation_data=data.get("annotation_data", {}),
            tags=data.get("tags", []),
        )
        return jsonify({"success": True, "comment": result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="add_review_comment")


@collab_review_bp.route("/api/review/comments", methods=["GET"])
def list_review_comments():
    """List/filter review comments for a project.

    Query params:
        project_path: Required. Path to the project.
        status: Filter by status (open/resolved/wontfix).
        author: Filter by author name.
        start_sec: Filter by time range start.
        end_sec: Filter by time range end.
    """
    from opencut.core.review_comments import get_stats, list_comments

    project_path = request.args.get("project_path", "")
    if not project_path:
        return jsonify({"error": "project_path query param is required"}), 400
    try:
        project_path = validate_path(project_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        comments = list_comments(
            project_path=project_path,
            status=request.args.get("status"),
            start_sec=(safe_float(request.args.get("start_sec"))
                        if request.args.get("start_sec") else None),
            end_sec=(safe_float(request.args.get("end_sec"))
                      if request.args.get("end_sec") else None),
            author=request.args.get("author"),
        )
        stats = get_stats(project_path)
        return jsonify({
            "comments": comments,
            "count": len(comments),
            "stats": stats,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="list_review_comments")


@collab_review_bp.route("/api/review/comments/<comment_id>/resolve", methods=["PUT"])
@require_csrf
def resolve_review_comment(comment_id):
    """Resolve a review comment.

    Expects JSON::

        {
            "project_path": "/path/to/project.prproj",
            "status": "resolved"
        }
    """
    from opencut.core.review_comments import resolve_comment

    data = request.get_json(force=True) or {}
    project_path = data.get("project_path", "")
    if not project_path:
        return jsonify({"error": "project_path is required"}), 400
    try:
        project_path = validate_path(project_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = resolve_comment(
            project_path=project_path,
            comment_id=comment_id,
            status=data.get("status", "resolved"),
        )
        return jsonify({"success": True, "comment": result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="resolve_review_comment")


@collab_review_bp.route("/api/review/comments/<comment_id>", methods=["DELETE"])
@require_csrf
def delete_review_comment(comment_id):
    """Delete a review comment.

    Expects JSON::

        {"project_path": "/path/to/project.prproj"}
    """
    from opencut.core.review_comments import delete_comment

    data = request.get_json(force=True) or {}
    project_path = data.get("project_path", "")
    if not project_path:
        return jsonify({"error": "project_path is required"}), 400
    try:
        project_path = validate_path(project_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        deleted = delete_comment(project_path=project_path,
                                 comment_id=comment_id)
        if not deleted:
            return jsonify({"error": f"Comment not found: {comment_id}"}), 404
        return jsonify({"success": True, "deleted": comment_id})
    except Exception as exc:
        return safe_error(exc, context="delete_review_comment")


@collab_review_bp.route("/api/review/comments/export", methods=["POST"])
@require_csrf
def export_review_comments():
    """Export review comments as JSON or CSV.

    Expects JSON::

        {
            "project_path": "/path/to/project.prproj",
            "format": "json"
        }
    """
    from opencut.core.review_comments import export_comments

    data = request.get_json(force=True) or {}
    project_path = data.get("project_path", "")
    if not project_path:
        return jsonify({"error": "project_path is required"}), 400
    try:
        project_path = validate_path(project_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        fmt = data.get("format", "json")
        content = export_comments(project_path=project_path, fmt=fmt)
        return jsonify({"success": True, "format": fmt, "content": content})
    except Exception as exc:
        return safe_error(exc, context="export_review_comments")


# ===========================================================================
# Version Compare (1 async route)
# ===========================================================================

@collab_review_bp.route("/api/version/compare", methods=["POST"])
@require_csrf
@async_job("version_compare", filepath_required=False)
def version_compare(job_id, filepath, data):
    """Compare two video files frame-by-frame.

    Expects JSON::

        {
            "file_a": "/path/to/render_v1.mp4",
            "file_b": "/path/to/render_v2.mp4",
            "mode": "side_by_side",
            "frame_interval": 1.0,
            "include_audio": true,
            "generate_video": true
        }
    """
    from opencut.core.version_compare import compare_versions

    file_a = data.get("file_a", "")
    file_b = data.get("file_b", "")
    if not file_a or not file_b:
        raise ValueError("Both file_a and file_b are required")

    file_a = validate_filepath(file_a)
    file_b = validate_filepath(file_b)

    mode = data.get("mode", "side_by_side")
    frame_interval = safe_float(data.get("frame_interval"), 1.0,
                                min_val=0.1, max_val=60.0)
    max_frames = safe_int(data.get("max_frames"), 300, min_val=1, max_val=1000)
    include_audio = data.get("include_audio", True)
    generate_video = data.get("generate_video", True)

    def on_progress(pct):
        _update_job(job_id, progress=pct,
                    message=f"Comparing... {pct}%")

    result = compare_versions(
        file_a=file_a,
        file_b=file_b,
        mode=mode,
        frame_interval=frame_interval,
        max_frames=max_frames,
        include_audio=include_audio,
        generate_video=generate_video,
        on_progress=on_progress,
    )
    return result


# ===========================================================================
# Approval Workflow (3 routes)
# ===========================================================================

@collab_review_bp.route("/api/approval/status", methods=["GET"])
def approval_status():
    """Get approval workflow status.

    Query params:
        workflow_id: Get status for a specific workflow.
        project_id: Get status for a project's latest workflow.
        (neither): Return the dashboard of all workflows.
    """
    from opencut.core.approval_workflow import get_status

    try:
        result = get_status(
            workflow_id=request.args.get("workflow_id"),
            project_id=request.args.get("project_id"),
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return safe_error(exc, context="approval_status")


@collab_review_bp.route("/api/approval/advance", methods=["POST"])
@require_csrf
def approval_advance():
    """Approve, reject, or request changes on a workflow.

    Expects JSON::

        {
            "workflow_id": "abc123",
            "action": "approve",
            "actor": "editor1",
            "notes": "Looks good!"
        }
    """
    from opencut.core.approval_workflow import (
        approve_workflow,
        reject_workflow,
        request_changes_workflow,
    )

    data = request.get_json(force=True) or {}
    workflow_id = data.get("workflow_id", "")
    action = data.get("action", "")
    actor = data.get("actor", "")

    if not workflow_id:
        return jsonify({"error": "workflow_id is required"}), 400
    if not action:
        return jsonify({"error": "action is required (approve/reject/request_changes)"}), 400
    if not actor:
        return jsonify({"error": "actor is required"}), 400

    try:
        notes = data.get("notes", "")
        if action == "approve":
            result = approve_workflow(workflow_id, actor, notes)
        elif action == "reject":
            reason = data.get("reason", notes)
            result = reject_workflow(workflow_id, actor, reason)
        elif action == "request_changes":
            result = request_changes_workflow(workflow_id, actor, notes)
        else:
            return jsonify({"error": f"Invalid action: {action}"}), 400

        return jsonify({"success": True, **result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="approval_advance")


@collab_review_bp.route("/api/approval/create", methods=["POST"])
@require_csrf
def approval_create():
    """Create a new approval workflow.

    Expects JSON::

        {
            "project_id": "my_project",
            "project_name": "My Video Project",
            "required_approvers": {
                "internal_review": ["editor1", "supervisor"],
                "client_review": ["client_lead"]
            },
            "deadline": 1700000000.0
        }
    """
    from opencut.core.approval_workflow import create_workflow

    data = request.get_json(force=True) or {}
    project_id = data.get("project_id", "")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    try:
        result = create_workflow(
            project_id=project_id,
            project_name=data.get("project_name", ""),
            required_approvers=data.get("required_approvers"),
            deadline=safe_float(data.get("deadline"), 0.0, min_val=0.0),
            metadata=data.get("metadata"),
        )
        return jsonify({"success": True, "workflow": result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="approval_create")


# ===========================================================================
# Shared Presets (2 routes)
# ===========================================================================

@collab_review_bp.route("/api/presets/shared", methods=["POST"])
@require_csrf
def upload_shared_preset():
    """Upload a shared preset.

    Expects JSON::

        {
            "name": "Cinematic Color",
            "category": "color_grades",
            "parameters": {"contrast": 1.2, "saturation": 0.9},
            "author": "colorist1",
            "tags": ["cinema", "film"],
            "description": "Warm cinematic look"
        }
    """
    from opencut.core.shared_presets import add_preset

    data = request.get_json(force=True) or {}
    name = data.get("name", "")
    if not name:
        return jsonify({"error": "Preset name is required"}), 400

    category = data.get("category", "export_profiles")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict):
        return jsonify({"error": "parameters must be a JSON object"}), 400

    try:
        result = add_preset(
            name=name,
            category=category,
            parameters=parameters,
            author=data.get("author", ""),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )
        return jsonify({"success": True, "preset": result})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="upload_shared_preset")


@collab_review_bp.route("/api/presets/shared", methods=["GET"])
def list_shared_presets():
    """List shared presets with optional filtering.

    Query params:
        category: Filter by category.
        author: Filter by author.
        search: Search in name/description.
        tags: Comma-separated tag filter.
        sort_by: Sort field (name/rating/created_at/updated_at/author).
    """
    from opencut.core.shared_presets import get_library_stats, list_presets

    tags_raw = request.args.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None

    try:
        presets = list_presets(
            category=request.args.get("category"),
            author=request.args.get("author"),
            tags=tags,
            search=request.args.get("search"),
            sort_by=request.args.get("sort_by", "name"),
        )
        stats = get_library_stats()
        return jsonify({
            "presets": presets,
            "count": len(presets),
            "stats": stats,
        })
    except Exception as exc:
        return safe_error(exc, context="list_shared_presets")


# ===========================================================================
# Edit History Export (1 route)
# ===========================================================================

@collab_review_bp.route("/api/edit-history/export", methods=["POST"])
@require_csrf
def export_edit_history():
    """Export edit history for a project.

    Expects JSON::

        {
            "project_id": "my_project",
            "format": "json",
            "include_undone": false
        }

    Supported formats: json, timeline, replay.
    """
    from opencut.core.edit_history import export_history, get_statistics

    data = request.get_json(force=True) or {}
    project_id = data.get("project_id", "")
    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    try:
        fmt = data.get("format", "json")
        include_undone = bool(data.get("include_undone", False))
        content = export_history(project_id, fmt=fmt,
                                 include_undone=include_undone)
        stats = get_statistics(project_id)
        return jsonify({
            "success": True,
            "format": fmt,
            "content": content,
            "statistics": stats,
        })
    except Exception as exc:
        return safe_error(exc, context="export_edit_history")
