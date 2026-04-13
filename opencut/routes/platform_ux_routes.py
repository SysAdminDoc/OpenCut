"""
OpenCut Platform UX Routes

Routes for:
  - Standalone Web UI backend (9.2)
  - After Effects Extension backend (9.3)
  - Panel UX features (6.1, 6.2, 6.6, 6.7, 6.8, 37.1, 37.2, 37.5)
"""

import logging

from flask import Blueprint, Response, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf

logger = logging.getLogger("opencut")

platform_ux_bp = Blueprint("platform_ux", __name__)


# ===================================================================
# Web UI (9.2)
# ===================================================================

@platform_ux_bp.route("/web-ui/session/create", methods=["POST"])
@require_csrf
def web_ui_create_session():
    """Create a new web UI session."""
    try:
        from opencut.core.web_ui import create_session
        session = create_session()
        return jsonify({
            "session_id": session.session_id,
            "created_at": session.created_at,
        })
    except Exception as exc:
        return safe_error(exc, context="web-ui-session-create")


@platform_ux_bp.route("/web-ui/upload", methods=["POST"])
@require_csrf
def web_ui_upload():
    """Upload a file to a web UI session."""
    try:
        session_id = request.form.get("session_id") or (
            request.get_json(force=True, silent=True) or {}
        ).get("session_id", "")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        # Handle multipart file upload
        if "file" in request.files:
            f = request.files["file"]
            filename = f.filename or "upload"
            file_data = f.read()
        else:
            # Fallback: JSON body with base64 data
            data = request.get_json(force=True, silent=True) or {}
            filename = data.get("filename", "upload")
            import base64
            file_data = base64.b64decode(data.get("data", ""))

        from opencut.core.web_ui import upload_file
        uploaded = upload_file(session_id, filename, file_data)
        return jsonify({
            "filename": uploaded.filename,
            "path": uploaded.path,
            "size": uploaded.size,
            "mime_type": uploaded.mime_type,
            "uploaded_at": uploaded.uploaded_at,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return safe_error(exc, context="web-ui-upload")


@platform_ux_bp.route("/web-ui/session/<session_id>/files", methods=["GET"])
def web_ui_list_files(session_id):
    """List uploaded files for a session."""
    try:
        from opencut.core.web_ui import list_uploads
        uploads = list_uploads(session_id)
        return jsonify({
            "session_id": session_id,
            "files": [
                {
                    "filename": u.filename,
                    "path": u.path,
                    "size": u.size,
                    "mime_type": u.mime_type,
                    "uploaded_at": u.uploaded_at,
                }
                for u in uploads
            ],
        })
    except Exception as exc:
        return safe_error(exc, context="web-ui-list-files")


@platform_ux_bp.route("/web-ui/operations", methods=["GET"])
def web_ui_operations():
    """Return the operation catalog grouped by category."""
    try:
        from opencut.core.web_ui import get_operation_catalog
        catalog = get_operation_catalog()
        return jsonify({"catalog": catalog})
    except Exception as exc:
        return safe_error(exc, context="web-ui-operations")


@platform_ux_bp.route("/web-ui/session/<session_id>", methods=["DELETE"])
@require_csrf
def web_ui_cleanup_session(session_id):
    """Delete a web UI session and its files."""
    try:
        from opencut.core.web_ui import cleanup_session
        found = cleanup_session(session_id)
        if not found:
            return jsonify({"error": "Session not found"}), 404
        return jsonify({"status": "cleaned_up", "session_id": session_id})
    except Exception as exc:
        return safe_error(exc, context="web-ui-cleanup")


# ===================================================================
# After Effects Extension (9.3)
# ===================================================================

@platform_ux_bp.route("/ae/supported-operations", methods=["GET"])
def ae_supported_operations():
    """Return AE-relevant operations."""
    try:
        from opencut.core.ae_extension import ae_supported_operations
        ops = ae_supported_operations()
        return jsonify({"operations": ops})
    except Exception as exc:
        return safe_error(exc, context="ae-supported-ops")


@platform_ux_bp.route("/ae/manifest", methods=["GET"])
def ae_manifest():
    """Generate the AE CEP manifest XML."""
    try:
        from opencut.core.ae_extension import generate_ae_manifest
        xml_str = generate_ae_manifest()
        return Response(xml_str, mimetype="application/xml")
    except Exception as exc:
        return safe_error(exc, context="ae-manifest")


@platform_ux_bp.route("/ae/project-info", methods=["GET"])
def ae_project_info():
    """Parse AE project info from query parameters or JSON body."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        from dataclasses import asdict

        from opencut.core.ae_extension import get_ae_project_info
        project = get_ae_project_info(data)
        result = asdict(project)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, context="ae-project-info")


# ===================================================================
# Panel UX — Workspace Layouts (6.2)
# ===================================================================

@platform_ux_bp.route("/panel/layout/save", methods=["POST"])
@require_csrf
def panel_save_layout():
    """Save a workspace layout."""
    try:
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        state = data.get("state", {})
        if not name:
            return jsonify({"error": "Layout name is required"}), 400
        from opencut.core.panel_ux import save_layout
        layout = save_layout(name, state)
        return jsonify(layout)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="panel-layout-save")


@platform_ux_bp.route("/panel/layout/<name>", methods=["GET"])
def panel_load_layout(name):
    """Load a workspace layout by name."""
    try:
        from opencut.core.panel_ux import load_layout
        layout = load_layout(name)
        if layout is None:
            return jsonify({"error": f"Layout not found: {name}"}), 404
        return jsonify(layout)
    except Exception as exc:
        return safe_error(exc, context="panel-layout-load")


@platform_ux_bp.route("/panel/layouts", methods=["GET"])
def panel_list_layouts():
    """List all available layouts."""
    try:
        from opencut.core.panel_ux import list_layouts
        layouts = list_layouts()
        return jsonify({"layouts": layouts})
    except Exception as exc:
        return safe_error(exc, context="panel-layouts-list")


# ===================================================================
# Panel UX — Drag-and-Drop (6.1)
# ===================================================================

@platform_ux_bp.route("/panel/drop-handler", methods=["POST"])
@require_csrf
def panel_drop_handler():
    """Register a drag-and-drop file-to-operation mapping."""
    try:
        data = request.get_json(force=True)
        file_path = data.get("file_path", "")
        operation = data.get("operation", "")
        if not file_path or not operation:
            return jsonify({"error": "file_path and operation are required"}), 400
        from opencut.core.panel_ux import register_drop_handler
        action = register_drop_handler(file_path, operation)
        return jsonify({
            "file_path": action.file_path,
            "operation": action.operation,
            "timestamp": action.timestamp,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, context="panel-drop-handler")


# ===================================================================
# Panel UX — Context Menu (6.6)
# ===================================================================

@platform_ux_bp.route("/panel/context-menu", methods=["GET"])
def panel_context_menu():
    """Get context menu actions for a clip type."""
    try:
        clip_type = request.args.get("clip_type", "video")
        from opencut.core.panel_ux import get_context_menu_actions
        actions = get_context_menu_actions(clip_type)
        return jsonify({"clip_type": clip_type, "actions": actions})
    except Exception as exc:
        return safe_error(exc, context="panel-context-menu")


# ===================================================================
# Panel UX — Themes (6.8)
# ===================================================================

@platform_ux_bp.route("/panel/themes", methods=["GET"])
def panel_list_themes():
    """List available themes."""
    try:
        from opencut.core.panel_ux import list_themes
        themes = list_themes()
        return jsonify({"themes": themes})
    except Exception as exc:
        return safe_error(exc, context="panel-themes-list")


@platform_ux_bp.route("/panel/theme/<name>", methods=["GET"])
def panel_get_theme(name):
    """Get CSS custom properties for a specific theme."""
    try:
        from opencut.core.panel_ux import get_theme
        theme = get_theme(name)
        if theme is None:
            return jsonify({"error": f"Theme not found: {name}"}), 404
        return jsonify(theme)
    except Exception as exc:
        return safe_error(exc, context="panel-theme-get")


# ===================================================================
# Panel UX — Walkthroughs (37.1)
# ===================================================================

@platform_ux_bp.route("/panel/walkthrough/<feature_id>", methods=["GET"])
def panel_get_walkthrough(feature_id):
    """Get walkthrough steps for a feature."""
    try:
        from opencut.core.panel_ux import get_walkthrough
        wt = get_walkthrough(feature_id)
        if wt is None:
            return jsonify({"error": f"Walkthrough not found: {feature_id}"}), 404
        return jsonify(wt)
    except Exception as exc:
        return safe_error(exc, context="panel-walkthrough-get")


@platform_ux_bp.route("/panel/walkthrough/<feature_id>/complete", methods=["POST"])
@require_csrf
def panel_complete_walkthrough(feature_id):
    """Mark a walkthrough as completed."""
    try:
        from opencut.core.panel_ux import mark_walkthrough_completed
        success = mark_walkthrough_completed(feature_id)
        if not success:
            return jsonify({"error": f"Walkthrough not found: {feature_id}"}), 404
        return jsonify({"feature_id": feature_id, "completed": True})
    except Exception as exc:
        return safe_error(exc, context="panel-walkthrough-complete")


# ===================================================================
# Panel UX — Session State (37.2)
# ===================================================================

@platform_ux_bp.route("/panel/state/save", methods=["POST"])
@require_csrf
def panel_save_state():
    """Save panel session state."""
    try:
        data = request.get_json(force=True)
        state = data.get("state", data)
        from opencut.core.panel_ux import save_session_state
        result = save_session_state(state)
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, context="panel-state-save")


@platform_ux_bp.route("/panel/state/restore", methods=["GET"])
def panel_restore_state():
    """Restore the last saved panel session state."""
    try:
        from opencut.core.panel_ux import restore_session_state
        state = restore_session_state()
        if state is None:
            return jsonify({"state": None, "message": "No saved state found"})
        return jsonify(state)
    except Exception as exc:
        return safe_error(exc, context="panel-state-restore")


# ===================================================================
# Panel UX — Offline Docs (37.5)
# ===================================================================

@platform_ux_bp.route("/panel/docs/<topic>", methods=["GET"])
def panel_get_doc(topic):
    """Get documentation for a topic."""
    try:
        from opencut.core.panel_ux import get_documentation
        doc = get_documentation(topic)
        if doc is None:
            return jsonify({"error": f"Topic not found: {topic}"}), 404
        return jsonify(doc)
    except Exception as exc:
        return safe_error(exc, context="panel-doc-get")


@platform_ux_bp.route("/panel/docs/search", methods=["GET"])
def panel_search_docs():
    """Search offline documentation."""
    try:
        query = request.args.get("q", "")
        from opencut.core.panel_ux import search_docs
        results = search_docs(query)
        return jsonify({"query": query, "results": results})
    except Exception as exc:
        return safe_error(exc, context="panel-doc-search")
