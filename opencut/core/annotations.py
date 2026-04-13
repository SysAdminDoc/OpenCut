"""
OpenCut Change Annotations Module

Attach textual annotations to project snapshots for revision-history
tracking.  Stored in ``~/.opencut/annotations.db`` SQLite database.

Supports export of full revision history in markdown or text format.
"""

import logging
import os
import sqlite3
import time
import uuid
from typing import Dict, List

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_ANNOTATIONS_DB = os.path.join(_OPENCUT_DIR, "annotations.db")


def _get_db() -> sqlite3.Connection:
    """Open (and initialise if needed) the annotations database."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    conn = sqlite3.connect(_ANNOTATIONS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            annotation_id TEXT PRIMARY KEY,
            snapshot_id   TEXT NOT NULL,
            text          TEXT NOT NULL,
            change_ref    TEXT NOT NULL DEFAULT '',
            author        TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_annotations_snapshot
        ON annotations(snapshot_id)
    """)
    conn.commit()
    return conn


def add_annotation(
    snapshot_id: str,
    text: str,
    change_ref: str = "",
    author: str = "",
) -> dict:
    """
    Add an annotation to a project snapshot.

    Args:
        snapshot_id: Identifier of the snapshot/revision being annotated.
        text: Annotation body text.
        change_ref: Optional reference to a specific change (e.g. commit hash,
                    edit operation ID).
        author: Author name.

    Returns:
        dict with the created annotation fields including ``annotation_id``.
    """
    annotation_id = uuid.uuid4().hex[:12]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO annotations
               (annotation_id, snapshot_id, text, change_ref, author, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (annotation_id, snapshot_id, text, change_ref, author, now),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "annotation_id": annotation_id,
        "snapshot_id": snapshot_id,
        "text": text,
        "change_ref": change_ref,
        "author": author,
        "created_at": now,
    }


def get_annotations(snapshot_id: str) -> List[dict]:
    """
    Retrieve all annotations for a snapshot.

    Returns:
        List of annotation dicts ordered by creation date.
    """
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM annotations WHERE snapshot_id = ? ORDER BY created_at",
            (snapshot_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def export_revision_history(
    snapshot_ids: List[str],
    format: str = "markdown",
) -> str:
    """
    Export annotations for multiple snapshots as a revision history.

    Args:
        snapshot_ids: List of snapshot identifiers, in chronological order.
        format: ``"markdown"`` or ``"text"``.

    Returns:
        Formatted revision history string.
    """
    if not snapshot_ids:
        return "No snapshots provided."

    conn = _get_db()
    try:
        all_annotations: Dict[str, List[dict]] = {}
        for sid in snapshot_ids:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE snapshot_id = ? ORDER BY created_at",
                (sid,),
            ).fetchall()
            all_annotations[sid] = [dict(r) for r in rows]
    finally:
        conn.close()

    if format == "text":
        return _export_text(snapshot_ids, all_annotations)
    return _export_markdown(snapshot_ids, all_annotations)


def _export_markdown(
    snapshot_ids: List[str],
    all_annotations: Dict[str, List[dict]],
) -> str:
    lines = ["# Revision History", ""]
    for idx, sid in enumerate(snapshot_ids, 1):
        annotations = all_annotations.get(sid, [])
        lines.append(f"## Revision {idx}: {sid}")
        if not annotations:
            lines.append("*No annotations*")
        else:
            for a in annotations:
                ref = f" (ref: `{a['change_ref']}`)" if a["change_ref"] else ""
                author = f" *— {a['author']}*" if a["author"] else ""
                lines.append(f"- {a['text']}{ref}{author}")
                lines.append(f"  _{a['created_at']}_")
        lines.append("")
    return "\n".join(lines)


def _export_text(
    snapshot_ids: List[str],
    all_annotations: Dict[str, List[dict]],
) -> str:
    lines = ["Revision History", "=" * 40]
    for idx, sid in enumerate(snapshot_ids, 1):
        annotations = all_annotations.get(sid, [])
        lines.append("")
        lines.append(f"--- Revision {idx}: {sid} ---")
        if not annotations:
            lines.append("  (no annotations)")
        else:
            for a in annotations:
                ref = f" [ref: {a['change_ref']}]" if a["change_ref"] else ""
                author = f" -- {a['author']}" if a["author"] else ""
                lines.append(f"  {a['created_at']}: {a['text']}{ref}{author}")
    return "\n".join(lines)
