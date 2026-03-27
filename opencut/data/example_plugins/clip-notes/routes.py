"""Clip Notes Plugin — save timestamped notes for clips in your project."""

import csv
import io
import json
import logging
import os
import uuid

from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger("opencut")

plugin_bp = Blueprint("opencut_clip_notes", __name__)

# Notes stored as JSON in the plugin's data directory
_NOTES_PATH = os.path.join(
    os.path.expanduser("~"), ".opencut", "plugins", "clip-notes", "notes.json"
)


def _load_notes():
    """Load all notes from the JSON file."""
    if not os.path.isfile(_NOTES_PATH):
        return []
    try:
        with open(_NOTES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load clip notes: %s", e)
        return []


def _save_notes(notes):
    """Persist notes list to the JSON file."""
    os.makedirs(os.path.dirname(_NOTES_PATH), exist_ok=True)
    with open(_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


@plugin_bp.route("/note", methods=["POST"])
def save_note():
    """Save a timestamped note for a clip.

    JSON body:
        filepath (str): path to the clip file (required)
        timestamp (str): timecode or seconds position in the clip (required)
        text (str): the note content (required)
    """
    data = request.get_json(force=True, silent=True) or {}

    filepath = str(data.get("filepath", "")).strip()
    timestamp = str(data.get("timestamp", "")).strip()
    text = str(data.get("text", "")).strip()

    if not filepath:
        return jsonify({"error": "filepath is required"}), 400
    if not timestamp:
        return jsonify({"error": "timestamp is required"}), 400
    if not text:
        return jsonify({"error": "text is required"}), 400

    notes = _load_notes()
    note = {
        "id": uuid.uuid4().hex[:12],
        "filepath": filepath,
        "timestamp": timestamp,
        "text": text,
    }
    notes.append(note)
    _save_notes(notes)

    return jsonify({"success": True, "note": note})


@plugin_bp.route("/notes", methods=["GET"])
def get_notes():
    """Get all notes for a clip.

    Query params:
        filepath (str): path to the clip file (required)
    """
    filepath = request.args.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "filepath query parameter is required"}), 400

    notes = _load_notes()
    clip_notes = [n for n in notes if n["filepath"] == filepath]

    return jsonify({"notes": clip_notes, "count": len(clip_notes)})


@plugin_bp.route("/note", methods=["DELETE"])
def delete_note():
    """Delete a note by ID.

    Query params:
        note_id (str): the unique note identifier (required)
    """
    note_id = request.args.get("note_id", "").strip()
    if not note_id:
        return jsonify({"error": "note_id query parameter is required"}), 400

    notes = _load_notes()
    original_len = len(notes)
    notes = [n for n in notes if n["id"] != note_id]

    if len(notes) == original_len:
        return jsonify({"error": "Note not found"}), 404

    _save_notes(notes)
    return jsonify({"success": True, "deleted_id": note_id})


@plugin_bp.route("/export", methods=["GET"])
def export_notes():
    """Export all notes as plain text or CSV.

    Query params:
        format (str): 'text' (default) or 'csv'
    """
    fmt = request.args.get("format", "text").strip().lower()
    notes = _load_notes()

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "filepath", "timestamp", "text"])
        for n in notes:
            writer.writerow([n["id"], n["filepath"], n["timestamp"], n["text"]])
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=clip_notes.csv"},
        )

    # Plain text format
    lines = []
    for n in notes:
        lines.append(f"[{n['timestamp']}] {n['filepath']}")
        lines.append(f"  {n['text']}")
        lines.append("")
    body = "\n".join(lines) if lines else "No notes."

    return Response(body, mimetype="text/plain")
