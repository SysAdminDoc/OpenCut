"""OpenCut Operation Journal.

Persistent record of every OpenCut operation that modifies a Premiere Pro
project, so the panel can offer one-click rollback.

Entries come in via ``POST /journal/record`` after the ExtendScript inverse
info is known (e.g. after a batch rename, the frontend sends back the
``{nodeId: oldName}`` map). They leave via ``POST /journal/mark-reverted/<id>``
once the inverse ExtendScript call succeeds.

Schema is deliberately narrow — this is a journal, not a message queue.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "journal.db")
_thread_local = threading.local()

# Mirror job_store + footage_index_db connection tracking so atexit can
# close every WAL-mode connection on shutdown.
_ALL_CONNECTIONS: "dict[int, sqlite3.Connection]" = {}
_CONN_LOCK = threading.Lock()

# Actions the panel can record + later invert. Adding a new type requires
# (a) a matching ExtendScript inverse function and (b) the frontend revert
# handler to know how to dispatch. Declared as a frozenset so the SQL layer
# can reject unknown types at record time.
VALID_ACTIONS = frozenset({
    "import_sequence",   # frontend imported XML -> created a sequence
    "add_markers",       # frontend added sequence markers
    "batch_rename",      # frontend renamed a set of project items
    "create_smart_bins", # frontend created bins + moved items in
    "import_captions",   # frontend imported SRT as caption track
    "import_overlay",    # frontend imported overlay video
})

# Which actions have a reliable ExtendScript inverse. Other actions are
# still recorded (so the history is complete) but their UI row shows
# "No auto-revert" and the Revert button is disabled.
REVERTIBLE_ACTIONS = frozenset({
    "add_markers",
    "batch_rename",
    "import_sequence",
    "import_overlay",
})


def _get_conn() -> sqlite3.Connection:
    conn = getattr(_thread_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _thread_local.conn = conn
        with _CONN_LOCK:
            _ALL_CONNECTIONS[threading.get_ident()] = conn
    return conn


def close_all_connections() -> None:
    """Close every tracked connection. Called on server shutdown."""
    with _CONN_LOCK:
        for conn in list(_ALL_CONNECTIONS.values()):
            try:
                conn.close()
            except Exception:
                pass
        _ALL_CONNECTIONS.clear()
    try:
        if getattr(_thread_local, "conn", None) is not None:
            _thread_local.conn = None
    except Exception:
        pass


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            action TEXT NOT NULL,
            clip_path TEXT NOT NULL DEFAULT '',
            label TEXT NOT NULL DEFAULT '',
            inverse_json TEXT NOT NULL DEFAULT '{}',
            reverted INTEGER NOT NULL DEFAULT 0,
            reverted_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_journal_created ON journal(created_at DESC);
    """)
    conn.commit()


def record(action: str, label: str, inverse_payload: dict,
           clip_path: str = "") -> dict:
    """Persist a new journal entry. Returns the saved row as a dict.

    Unknown action types are rejected — adding a new type requires an
    ExtendScript inverse and a frontend dispatch branch.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown journal action: {action}")
    init_db()
    conn = _get_conn()
    now = time.time()
    try:
        inverse_json = json.dumps(inverse_payload or {})
    except (TypeError, ValueError):
        inverse_json = "{}"
    cur = conn.execute(
        """INSERT INTO journal (created_at, action, clip_path, label, inverse_json)
           VALUES (?, ?, ?, ?, ?)""",
        (now, action, clip_path or "", label or "", inverse_json),
    )
    conn.commit()
    entry_id = cur.lastrowid
    return _row_to_dict(conn.execute(
        "SELECT * FROM journal WHERE id = ?", (entry_id,)
    ).fetchone())


def list_entries(limit: int = 50, include_reverted: bool = True) -> list:
    init_db()
    conn = _get_conn()
    if include_reverted:
        rows = conn.execute(
            "SELECT * FROM journal ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM journal WHERE reverted = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_entry(entry_id: int) -> "dict | None":
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM journal WHERE id = ?", (entry_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def mark_reverted(entry_id: int) -> bool:
    init_db()
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE journal SET reverted = 1, reverted_at = ? WHERE id = ? AND reverted = 0",
        (time.time(), entry_id),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_entry(entry_id: int) -> bool:
    init_db()
    conn = _get_conn()
    cur = conn.execute("DELETE FROM journal WHERE id = ?", (entry_id,))
    conn.commit()
    return cur.rowcount > 0


def clear_all() -> int:
    """Delete every entry. Returns count removed. For user-initiated wipe."""
    init_db()
    conn = _get_conn()
    cur = conn.execute("DELETE FROM journal")
    conn.commit()
    return cur.rowcount


def _row_to_dict(row) -> dict:
    if row is None:
        return None
    try:
        inverse = json.loads(row["inverse_json"])
    except (json.JSONDecodeError, TypeError):
        inverse = {}
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "action": row["action"],
        "clip_path": row["clip_path"] or "",
        "label": row["label"] or "",
        "inverse": inverse,
        "reverted": bool(row["reverted"]),
        "reverted_at": row["reverted_at"],
        "revertible": row["action"] in REVERTIBLE_ACTIONS,
    }
