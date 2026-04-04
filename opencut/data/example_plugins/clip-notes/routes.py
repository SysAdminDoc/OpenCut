"""Clip Notes example plugin with legacy route compatibility."""

import csv
import io
import json
import logging
import os
import sqlite3
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, request

logger = logging.getLogger("opencut")

plugin_bp = Blueprint("opencut_clip_notes", __name__)

_NOTES_PATH = os.path.join(
    os.path.expanduser("~"), ".opencut", "plugins", "clip-notes", "notes.json"
)
_DB_PATH = None
_thread_local = threading.local()


def _storage_mode():
    """Pick SQLite only when a DB path has been explicitly configured."""
    return "sqlite" if _DB_PATH else "json"


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_tags(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
        return [tag.strip() for tag in stripped.split(",") if tag.strip()]
    return [str(value).strip()]


def _normalize_note(data):
    now = time.time()
    filepath = str(data.get("filepath") or data.get("clip_path") or "").strip()
    clip_path = str(data.get("clip_path") or filepath).strip()
    clip_name = str(data.get("clip_name") or os.path.basename(clip_path or filepath)).strip()
    timestamp = str(data.get("timestamp") or "").strip()
    text = str(data.get("text") or data.get("note") or "").strip()
    note_text = str(data.get("note") or text).strip()
    tags = _parse_tags(data.get("tags"))
    color = str(data.get("color") or "yellow").strip() or "yellow"
    created_at = float(data.get("created_at") or now)
    updated_at = float(data.get("updated_at") or created_at)
    note_id = str(data.get("id") or uuid.uuid4().hex[:12]).strip()

    return {
        "id": note_id,
        "filepath": filepath,
        "clip_path": clip_path,
        "clip_name": clip_name,
        "timestamp": timestamp,
        "text": text,
        "note": note_text,
        "tags": tags,
        "color": color,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _connect_db():
    if not _DB_PATH:
        raise RuntimeError("Clip Notes SQLite backend requested without _DB_PATH")
    _ensure_parent(_DB_PATH)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn=None):
    """Initialize the optional SQLite backend used by legacy tests/routes."""
    if not _DB_PATH:
        return

    own_conn = conn is None
    if conn is None:
        conn = _connect_db()
        _thread_local.conn = conn

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            filepath TEXT NOT NULL,
            clip_path TEXT NOT NULL,
            clip_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            note TEXT NOT NULL,
            tags TEXT NOT NULL,
            color TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()

    if own_conn:
        _thread_local.conn = conn


def _get_db():
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        conn = _connect_db()
        _thread_local.conn = conn
        _init_db(conn)
    return conn


def _load_notes():
    """Load all notes from the active storage backend."""
    if _storage_mode() == "sqlite":
        rows = _get_db().execute("SELECT * FROM notes ORDER BY created_at ASC").fetchall()
        return [_normalize_note(dict(row)) for row in rows]

    if not os.path.isfile(_NOTES_PATH):
        return []
    try:
        with open(_NOTES_PATH, "r", encoding="utf-8") as f:
            raw_notes = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load clip notes: %s", exc)
        return []

    return [_normalize_note(note) for note in raw_notes if isinstance(note, dict)]


def _save_notes(notes):
    """Persist the normalized notes list to the active storage backend."""
    normalized = [_normalize_note(note) for note in notes]

    if _storage_mode() == "sqlite":
        conn = _get_db()
        with conn:
            conn.execute("DELETE FROM notes")
            conn.executemany(
                """
                INSERT INTO notes (
                    id, filepath, clip_path, clip_name, timestamp,
                    text, note, tags, color, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        note["id"],
                        note["filepath"],
                        note["clip_path"],
                        note["clip_name"],
                        note["timestamp"],
                        note["text"],
                        note["note"],
                        json.dumps(note["tags"], ensure_ascii=False),
                        note["color"],
                        note["created_at"],
                        note["updated_at"],
                    )
                    for note in normalized
                ],
            )
        return

    _ensure_parent(_NOTES_PATH)
    with open(_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)


def _append_note(data):
    note = _normalize_note(data)
    notes = _load_notes()
    notes.append(note)
    _save_notes(notes)
    return note


def _filter_by_path(notes, clip_path):
    return [
        note
        for note in notes
        if note["clip_path"] == clip_path or note["filepath"] == clip_path
    ]


def _delete_by_id(note_id):
    notes = _load_notes()
    remaining = [note for note in notes if note["id"] != note_id]
    if len(remaining) == len(notes):
        return False
    _save_notes(remaining)
    return True


@plugin_bp.route("/note", methods=["POST"])
def save_note():
    """Save a timestamped note for a clip."""
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

    note = _append_note(
        {
            "filepath": filepath,
            "clip_path": filepath,
            "clip_name": os.path.basename(filepath),
            "timestamp": timestamp,
            "text": text,
            "note": text,
            "tags": data.get("tags"),
            "color": data.get("color"),
        }
    )
    return jsonify({"success": True, "note": note})


@plugin_bp.route("/notes", methods=["GET"])
def get_notes():
    """Get all notes for a clip."""
    filepath = request.args.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "filepath query parameter is required"}), 400

    clip_notes = _filter_by_path(_load_notes(), filepath)
    return jsonify({"notes": clip_notes, "count": len(clip_notes)})


@plugin_bp.route("/note", methods=["DELETE"])
def delete_note():
    """Delete a note by ID."""
    note_id = request.args.get("note_id", "").strip()
    if not note_id:
        return jsonify({"error": "note_id query parameter is required"}), 400
    if not _delete_by_id(note_id):
        return jsonify({"error": "Note not found"}), 404
    return jsonify({"success": True, "deleted_id": note_id})


@plugin_bp.route("/export", methods=["GET"])
def export_notes():
    """Export all notes as plain text or CSV."""
    fmt = request.args.get("format", "text").strip().lower()
    notes = _load_notes()

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "filepath", "timestamp", "text"])
        for note in notes:
            writer.writerow(
                [note["id"], note["filepath"], note["timestamp"], note["text"]]
            )
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=clip_notes.csv"},
        )

    lines = []
    for note in notes:
        path = note["filepath"] or note["clip_path"]
        lines.append(f"[{note['timestamp']}] {path}")
        lines.append(f"  {note['text'] or note['note']}")
        lines.append("")
    return Response("\n".join(lines) if lines else "No notes.", mimetype="text/plain")


@plugin_bp.route("/list", methods=["GET"])
def list_notes():
    """Legacy route: list all notes or filter by clip path."""
    clip_path = request.args.get("clip_path", "").strip()
    notes = _load_notes()
    if clip_path:
        notes = _filter_by_path(notes, clip_path)
    return jsonify({"notes": notes, "count": len(notes)})


@plugin_bp.route("/add", methods=["POST"])
def add_note():
    """Legacy route: add a clip note."""
    data = request.get_json(force=True, silent=True) or {}
    clip_path = str(data.get("clip_path", "")).strip()
    note_text = str(data.get("note", "")).strip()
    if not clip_path:
        return jsonify({"error": "clip_path is required"}), 400
    if not note_text:
        return jsonify({"error": "note is required"}), 400

    note = _append_note(
        {
            "clip_path": clip_path,
            "filepath": clip_path,
            "clip_name": data.get("clip_name") or os.path.basename(clip_path),
            "timestamp": data.get("timestamp") or "",
            "note": note_text,
            "text": note_text,
            "tags": data.get("tags"),
            "color": data.get("color"),
        }
    )
    return jsonify({"success": True, "note": note})


@plugin_bp.route("/update", methods=["POST"])
def update_note():
    """Legacy route: update an existing note."""
    data = request.get_json(force=True, silent=True) or {}
    note_id = str(data.get("id", "")).strip()
    if not note_id:
        return jsonify({"error": "id is required"}), 400

    notes = _load_notes()
    updated = None
    for idx, note in enumerate(notes):
        if note["id"] != note_id:
            continue
        merged = dict(note)
        if "clip_path" in data:
            merged["clip_path"] = str(data.get("clip_path") or "").strip()
            merged["filepath"] = merged["clip_path"]
        if "clip_name" in data:
            merged["clip_name"] = str(data.get("clip_name") or "").strip()
        if "timestamp" in data:
            merged["timestamp"] = str(data.get("timestamp") or "").strip()
        if "note" in data:
            text = str(data.get("note") or "").strip()
            merged["note"] = text
            merged["text"] = text
        if "text" in data:
            text = str(data.get("text") or "").strip()
            merged["text"] = text
            merged["note"] = text
        if "tags" in data:
            merged["tags"] = _parse_tags(data.get("tags"))
        if "color" in data:
            merged["color"] = str(data.get("color") or "").strip() or note["color"]
        merged["updated_at"] = time.time()
        updated = _normalize_note(merged)
        notes[idx] = updated
        break

    if updated is None:
        return jsonify({"error": "Note not found"}), 404

    _save_notes(notes)
    return jsonify({"success": True, "note": updated})


@plugin_bp.route("/delete", methods=["POST"])
def delete_note_legacy():
    """Legacy route: delete a note by ID via POST body."""
    data = request.get_json(force=True, silent=True) or {}
    note_id = str(data.get("id", "")).strip()
    if not note_id:
        return jsonify({"error": "id is required"}), 400
    if not _delete_by_id(note_id):
        return jsonify({"error": "Note not found"}), 404
    return jsonify({"success": True, "deleted_id": note_id})


@plugin_bp.route("/search", methods=["POST"])
def search_notes():
    """Legacy route: search notes by free-text query."""
    data = request.get_json(force=True, silent=True) or {}
    query = str(data.get("query", "")).strip().lower()
    if not query:
        return jsonify({"notes": [], "count": 0})

    matches = []
    for note in _load_notes():
        haystack = " ".join(
            [
                note["clip_path"],
                note["clip_name"],
                note["note"],
                note["text"],
                " ".join(note["tags"]),
            ]
        ).lower()
        if query in haystack:
            matches.append(note)

    return jsonify({"notes": matches, "count": len(matches)})


@plugin_bp.route("/stats", methods=["GET"])
def stats():
    """Legacy route: basic note statistics."""
    notes = _load_notes()
    clips = {
        note["clip_path"] or note["filepath"]
        for note in notes
        if note["clip_path"] or note["filepath"]
    }
    return jsonify({"total_notes": len(notes), "clips_with_notes": len(clips)})
