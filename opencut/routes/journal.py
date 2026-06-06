"""OpenCut Journal Routes (v1.9.28).

Endpoints for the "Operation Journal" feature — frontend-driven record +
query, with the actual ExtendScript inverse dispatched by the panel.
The backend's job is just durable storage and the revert-state flag.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut import journal
from opencut.security import get_json_dict, require_csrf, safe_bool, safe_int, validate_path

logger = logging.getLogger("opencut")

journal_bp = Blueprint("journal", __name__)


def _destructive_flags() -> tuple[bool, bool]:
    payload = request.get_json(silent=True) or {}
    dry_run = safe_bool(request.args.get("dry_run", payload.get("dry_run", False)), False)
    backup = safe_bool(request.args.get("backup", payload.get("backup", False)), False)
    return dry_run, backup


@journal_bp.route("/journal/list", methods=["GET"])
def journal_list():
    """List recent journal entries (newest first).

    Query params:
        limit: max rows (default 50, max 200)
        include_reverted: "0"/"false" to exclude already-reverted entries
    """
    limit = safe_int(request.args.get("limit", 50), 50, min_val=1, max_val=200)
    include_reverted = safe_bool(
        request.args.get("include_reverted", "1"), True
    )
    try:
        entries = journal.list_entries(limit=limit, include_reverted=include_reverted)
        return jsonify(entries)
    except Exception as e:
        logger.exception("journal_list failed")
        return jsonify({"error": f"Could not read journal: {e}"}), 500


@journal_bp.route("/journal/db-diagnostics", methods=["GET"])
def journal_db_diagnostics():
    """Return read-only diagnostics for the operation journal database."""
    try:
        return jsonify(journal.get_db_diagnostics())
    except Exception as e:
        logger.exception("journal_db_diagnostics failed")
        return jsonify({"error": f"Could not read journal diagnostics: {e}"}), 500


@journal_bp.route("/journal/record", methods=["POST"])
@require_csrf
def journal_record():
    """Persist a new entry. Called by the panel immediately after an
    ExtendScript operation that modifies the project succeeds.

    Body::

        {
          "action": "add_markers" | "batch_rename" | ...,
          "label": "3 beat markers on 'Interview_01.mp4'",
          "clip_path": "...",           # optional, for context
          "inverse": {...}              # action-specific payload
        }
    """
    data = get_json_dict() or {}
    action = str(data.get("action", "")).strip()
    label = str(data.get("label", "")).strip()
    clip_path = str(data.get("clip_path", "")).strip()
    if clip_path:
        try:
            clip_path = validate_path(clip_path)
        except ValueError:
            clip_path = ""  # Drop invalid paths silently for journal context
    inverse = data.get("inverse") or {}
    forward = data.get("forward")  # optional; v1.10.3 "Apply to selection"

    if not action:
        return jsonify({"error": "action is required"}), 400
    if action not in journal.VALID_ACTIONS:
        return jsonify({
            "error": f"Unknown action: {action}",
            "valid": sorted(journal.VALID_ACTIONS),
        }), 400
    if not isinstance(inverse, dict):
        return jsonify({"error": "inverse must be an object"}), 400
    if forward is not None and not isinstance(forward, dict):
        return jsonify({"error": "forward must be an object"}), 400

    try:
        entry = journal.record(action, label, inverse, clip_path=clip_path,
                               forward_payload=forward)
        return jsonify(entry), 201
    except Exception as e:
        logger.exception("journal_record failed")
        return jsonify({"error": f"Could not record entry: {e}"}), 500


@journal_bp.route("/journal/mark-reverted/<int:entry_id>", methods=["POST"])
@require_csrf
def journal_mark_reverted(entry_id: int):
    """Mark an entry as reverted. The panel calls this after the
    ExtendScript inverse function succeeds.
    """
    entry = journal.get_entry(entry_id)
    if not entry:
        return jsonify({"error": "Entry not found"}), 404
    if entry["reverted"]:
        return jsonify({"error": "Already reverted", "entry": entry}), 409
    if not entry["revertible"]:
        return jsonify({"error": "This action has no auto-revert"}), 400

    ok = journal.mark_reverted(entry_id)
    if not ok:
        return jsonify({"error": "Failed to mark reverted"}), 500
    return jsonify(journal.get_entry(entry_id))


@journal_bp.route("/journal/<int:entry_id>", methods=["DELETE"])
@require_csrf
def journal_delete(entry_id: int):
    """Delete a single entry (user-initiated cleanup)."""
    dry_run, backup = _destructive_flags()
    result = journal.delete_entry(entry_id, dry_run=dry_run, backup=backup)
    if isinstance(result, dict):
        return jsonify(result)
    ok = result
    if not ok:
        return jsonify({"error": "Entry not found"}), 404
    return jsonify({"ok": True, "id": entry_id})


@journal_bp.route("/journal/clear", methods=["POST"])
@require_csrf
def journal_clear():
    """Delete every journal entry. Requires explicit user action in the UI."""
    dry_run, backup = _destructive_flags()
    result = journal.clear_all(dry_run=dry_run, backup=backup)
    if isinstance(result, dict):
        return jsonify(result)
    removed = result
    return jsonify({"ok": True, "removed": removed})
