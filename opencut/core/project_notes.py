"""
OpenCut Timestamped Project Notes

Store per-project, timestamped notes in a local SQLite database
(``~/.opencut/notes.db``).  Notes carry priority, status, and author
metadata for team review workflows.

Statuses: open, resolved, deferred.
"""

import csv
import io
import logging
import os
import sqlite3
import time
import uuid
from typing import List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_NOTES_DB = os.path.join(_OPENCUT_DIR, "notes.db")

_VALID_STATUSES = {"open", "resolved", "deferred"}
_VALID_PRIORITIES = {"low", "normal", "high", "critical"}


def _get_db() -> sqlite3.Connection:
    """Open (and initialise if needed) the notes database."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    conn = sqlite3.connect(_NOTES_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            note_id   TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            timestamp  REAL NOT NULL,
            text       TEXT NOT NULL,
            priority   TEXT NOT NULL DEFAULT 'normal',
            status     TEXT NOT NULL DEFAULT 'open',
            author     TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_notes_project
        ON notes(project_id)
    """)
    conn.commit()
    return conn


def add_note(
    project_id: str,
    timestamp: float,
    text: str,
    priority: str = "normal",
    author: str = "",
) -> dict:
    """
    Add a timestamped note to a project.

    Args:
        project_id: Project identifier string.
        timestamp: Timeline timestamp in seconds.
        text: Note body text.
        priority: ``"low"``, ``"normal"``, ``"high"``, or ``"critical"``.
        author: Author name.

    Returns:
        dict with the created note fields including ``note_id``.
    """
    if priority not in _VALID_PRIORITIES:
        priority = "normal"

    note_id = uuid.uuid4().hex[:12]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO notes
               (note_id, project_id, timestamp, text, priority, status, author, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
            (note_id, project_id, timestamp, text, priority, author, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "note_id": note_id,
        "project_id": project_id,
        "timestamp": timestamp,
        "text": text,
        "priority": priority,
        "status": "open",
        "author": author,
        "created_at": now,
        "updated_at": now,
    }


def get_notes(project_id: str, status: Optional[str] = None) -> List[dict]:
    """
    Retrieve notes for a project, optionally filtered by status.

    Returns:
        List of note dicts ordered by timestamp ascending.
    """
    conn = _get_db()
    try:
        if status and status in _VALID_STATUSES:
            rows = conn.execute(
                "SELECT * FROM notes WHERE project_id = ? AND status = ? ORDER BY timestamp",
                (project_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notes WHERE project_id = ? ORDER BY timestamp",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_note(
    note_id: str,
    text: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
) -> dict:
    """
    Update fields on an existing note.

    Args:
        note_id: The note to update.
        text: New text (if provided).
        status: New status (if provided, must be valid).
        priority: New priority (if provided, must be valid).

    Returns:
        dict with the updated note fields.

    Raises:
        ValueError: If the note does not exist.
    """
    conn = _get_db()
    try:
        row = conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        if not row:
            raise ValueError(f"Note not found: {note_id}")

        updates = []
        params = []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if text is not None:
            updates.append("text = ?")
            params.append(text)
        if status is not None and status in _VALID_STATUSES:
            updates.append("status = ?")
            params.append(status)
        if priority is not None and priority in _VALID_PRIORITIES:
            updates.append("priority = ?")
            params.append(priority)

        if updates:
            updates.append("updated_at = ?")
            params.append(now)
            params.append(note_id)
            conn.execute(
                f"UPDATE notes SET {', '.join(updates)} WHERE note_id = ?",
                params,
            )
            conn.commit()

        updated = conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        return dict(updated)
    finally:
        conn.close()


def delete_note(note_id: str) -> bool:
    """
    Delete a note by ID.

    Returns:
        True if the note was deleted, False if it did not exist.
    """
    conn = _get_db()
    try:
        cur = conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def export_notes(project_id: str, format: str = "text") -> str:
    """
    Export all notes for a project as a formatted string.

    Args:
        project_id: Project to export.
        format: ``"text"``, ``"csv"``, or ``"markdown"``.

    Returns:
        Formatted string.
    """
    notes = get_notes(project_id)
    if not notes:
        return f"No notes found for project {project_id}."

    if format == "csv":
        return _export_csv(notes)
    elif format == "markdown":
        return _export_markdown(notes, project_id)
    else:
        return _export_text(notes, project_id)


def _export_text(notes: List[dict], project_id: str) -> str:
    lines = [f"Project Notes: {project_id}", "=" * 40]
    for n in notes:
        ts = _fmt_timestamp(n["timestamp"])
        lines.append(f"[{ts}] ({n['priority']}/{n['status']}) {n['text']}")
        if n.get("author"):
            lines.append(f"  — {n['author']}")
    return "\n".join(lines)


def _export_markdown(notes: List[dict], project_id: str) -> str:
    lines = [f"# Project Notes: {project_id}", ""]
    for n in notes:
        ts = _fmt_timestamp(n["timestamp"])
        status_badge = {"open": "[ ]", "resolved": "[x]", "deferred": "[-]"}.get(n["status"], "[ ]")
        lines.append(f"- {status_badge} **{ts}** ({n['priority']}) — {n['text']}")
        if n.get("author"):
            lines.append(f"  *Author: {n['author']}*")
    return "\n".join(lines)


def _export_csv(notes: List[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["note_id", "timestamp", "text", "priority", "status", "author", "created_at"])
    for n in notes:
        writer.writerow([
            n["note_id"], n["timestamp"], n["text"],
            n["priority"], n["status"], n.get("author", ""),
            n["created_at"],
        ])
    return buf.getvalue()


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
