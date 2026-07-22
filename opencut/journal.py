"""OpenCut Operation Journal.

Persistent record of every OpenCut operation that modifies a Premiere Pro
project, so the panel can offer one-click rollback.

Checkpoint entries are written before the first host mutation, then completed
atomically after the host returns.  A row left pending is therefore a durable
crash-recovery record rather than an optimistic after-the-fact audit event.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid

from opencut.local_db_diagnostics import build_sqlite_diagnostic
from opencut.local_db_maintenance import count_rows, prepare_destructive_result
from opencut.local_db_migrations import migrate_user_version
from opencut.local_db_payloads import decode_json_or_spill_marker, spill_json_if_needed

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "journal.db")
_thread_local = threading.local()
SCHEMA_VERSION = 3
MAX_JOURNAL_PAYLOAD_JSON_BYTES = 128 * 1024
MAX_JOURNAL_ENTRIES = 1000

# Client-supplied transaction IDs are rendered into a UXP panel HTML
# attribute, so the charset is locked down alongside the length bound.
_TRANSACTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

CHECKPOINT_PENDING = "pending"
CHECKPOINT_COMPLETED = "completed"
CHECKPOINT_RECOVERY_FAILED = "recovery_failed"
CHECKPOINT_RESTORED = "restored"
INCOMPLETE_CHECKPOINT_STATUSES = frozenset({
    CHECKPOINT_PENDING,
    CHECKPOINT_RECOVERY_FAILED,
})

# Mirror job_store + footage_index_db connection tracking so atexit can
# close every WAL-mode connection on shutdown.
_ALL_CONNECTIONS: "dict[int, sqlite3.Connection]" = {}
_CONN_LOCK = threading.Lock()

# Serializes payload-spill writes against the orphan-spill pruner. A spill
# file is written by _encode_payload BEFORE the row that references it is
# committed, so a concurrent prune scan could otherwise delete an in-flight
# spill it cannot yet see in the journal table. Writers acquire this lock
# BEFORE opening their SQLite transaction (never inside one) and hold it
# through commit; the pruner holds it for the whole scan.
_SPILL_IO_LOCK = threading.Lock()

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
    "import_media",      # frontend imported generated media into the project
    "apply_cuts",        # UXP applied destructive timeline cuts
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


def _coerce_limit(value, default=50):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _prune_dead_connections() -> None:
    with _CONN_LOCK:
        alive_ids = {t.ident for t in threading.enumerate() if t.ident is not None}
        dead_ids = [tid for tid in _ALL_CONNECTIONS if tid not in alive_ids]
        for tid in dead_ids:
            try:
                _ALL_CONNECTIONS[tid].close()
            except Exception:
                pass
            del _ALL_CONNECTIONS[tid]


def _connection_is_usable(conn) -> bool:
    """Return True when a cached SQLite connection is still open."""
    if conn is None:
        return False
    try:
        conn.execute("SELECT 1")
    except sqlite3.Error:
        return False
    return True


def _discard_cached_conn(conn) -> None:
    """Drop a stale thread-local connection from the registry and close it."""
    with _CONN_LOCK:
        for tid, tracked in list(_ALL_CONNECTIONS.items()):
            if tracked is conn:
                _ALL_CONNECTIONS.pop(tid, None)
    try:
        conn.close()
    except sqlite3.Error:
        pass
    _thread_local.conn = None
    _thread_local.conn_path = None


def _get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection, path-aware like ``job_store``.

    The cached connection is discarded and reopened when ``_DB_PATH`` changed
    since it was created (tests repoint the path) or when it was closed behind
    our back, so a stale handle never outlives the path it was opened against.
    """
    conn = getattr(_thread_local, "conn", None)
    conn_path = getattr(_thread_local, "conn_path", None)
    if conn is not None and (conn_path != _DB_PATH or not _connection_is_usable(conn)):
        _discard_cached_conn(conn)
        conn = None
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        _thread_local.conn = conn
        _thread_local.conn_path = _DB_PATH
        with _CONN_LOCK:
            _ALL_CONNECTIONS[threading.get_ident()] = conn
    _prune_dead_connections()
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
            _thread_local.conn_path = None
    except Exception:
        pass


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            action TEXT NOT NULL,
            clip_path TEXT NOT NULL DEFAULT '',
            label TEXT NOT NULL DEFAULT '',
            inverse_json TEXT NOT NULL DEFAULT '{}',
            reverted INTEGER NOT NULL DEFAULT 0,
            reverted_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_created ON journal(created_at DESC)")


def _migrate_schema_v2(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(journal)").fetchall()}
    if "forward_json" not in columns:
        conn.execute("ALTER TABLE journal ADD COLUMN forward_json TEXT")


def _migrate_schema_v3(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(journal)").fetchall()}
    additions = {
        "transaction_id": "TEXT",
        "status": "TEXT NOT NULL DEFAULT 'completed'",
        "started_at": "REAL",
        "completed_at": "REAL",
        "preview_json": "TEXT",
        "diagnostics_json": "TEXT",
        "recovery_error": "TEXT NOT NULL DEFAULT ''",
        "recovery_attempted_at": "REAL",
        "updated_at": "REAL",
    }
    for name, declaration in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE journal ADD COLUMN {name} {declaration}")
    conn.execute(
        "UPDATE journal SET started_at = created_at WHERE started_at IS NULL"
    )
    conn.execute(
        "UPDATE journal SET completed_at = created_at WHERE completed_at IS NULL"
    )
    conn.execute(
        "UPDATE journal SET updated_at = created_at WHERE updated_at IS NULL"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_journal_transaction "
        "ON journal(transaction_id) WHERE transaction_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_journal_status_created "
        "ON journal(status, created_at DESC)"
    )


def init_db() -> None:
    conn = _get_conn()
    migrate_user_version(
        conn,
        store_name="journal",
        target_version=SCHEMA_VERSION,
        migrations={
            1: _create_schema_v1,
            2: _migrate_schema_v2,
            3: _migrate_schema_v3,
        },
    )
    conn.commit()


def _encode_payload(payload, *, field_name: str, allow_none: bool = False) -> str | None:
    if payload is None and allow_none:
        return None
    try:
        encoded = json.dumps(payload if payload is not None else {}, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    return spill_json_if_needed(
        encoded,
        base_dir=os.path.dirname(_DB_PATH),
        namespace="journal",
        field_name=field_name,
        max_bytes=MAX_JOURNAL_PAYLOAD_JSON_BYTES,
    )


def _spill_integrity(value) -> dict:
    """Verify a payload spill without trusting the path stored in SQLite."""
    if not isinstance(value, dict) or value.get("_opencut_payload_spill") is not True:
        return {"state": "inline", "ok": True}
    base_dir = os.path.realpath(os.path.dirname(_DB_PATH))
    spill_root = os.path.realpath(os.path.join(base_dir, "payload_spills", "journal"))
    path = os.path.realpath(str(value.get("path") or ""))
    try:
        in_root = os.path.commonpath([spill_root, path]) == spill_root
    except ValueError:
        in_root = False
    if not path or not in_root:
        return {"state": "unsafe_path", "ok": False}
    if not os.path.isfile(path):
        return {"state": "missing", "ok": False, "path": path}
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return {"state": "unreadable", "ok": False, "path": path, "error": str(exc)}
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = str(value.get("sha256") or "")
    expected_bytes = value.get("bytes")
    if expected_digest and actual_digest != expected_digest:
        return {
            "state": "hash_mismatch",
            "ok": False,
            "path": path,
            "expected_sha256": expected_digest,
            "actual_sha256": actual_digest,
        }
    if expected_bytes is not None and len(raw) != int(expected_bytes):
        return {
            "state": "size_mismatch",
            "ok": False,
            "path": path,
            "expected_bytes": int(expected_bytes),
            "actual_bytes": len(raw),
        }
    try:
        json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"state": "invalid_json", "ok": False, "path": path, "error": str(exc)}
    return {
        "state": "verified",
        "ok": True,
        "path": path,
        "bytes": len(raw),
        "sha256": actual_digest,
    }


def _resolve_payload(value):
    integrity = _spill_integrity(value)
    if integrity["state"] == "inline":
        return value, integrity
    if not integrity["ok"]:
        return value, integrity
    with open(integrity["path"], encoding="utf-8") as fh:
        return json.load(fh), integrity


def _prune_history(conn: sqlite3.Connection) -> None:
    """Bound completed history without ever deleting recovery evidence."""
    limit = max(1, int(MAX_JOURNAL_ENTRIES))
    conn.execute(
        """DELETE FROM journal
           WHERE status NOT IN (?, ?)
             AND id NOT IN (
                 SELECT id FROM journal
                 WHERE status NOT IN (?, ?)
                 ORDER BY created_at DESC LIMIT ?
             )""",
        (
            CHECKPOINT_PENDING,
            CHECKPOINT_RECOVERY_FAILED,
            CHECKPOINT_PENDING,
            CHECKPOINT_RECOVERY_FAILED,
            limit,
        ),
    )


def _prune_orphan_payload_spills(conn: sqlite3.Connection) -> int:
    """Delete content-addressed journal payloads no row references anymore."""
    with _SPILL_IO_LOCK:
        return _prune_orphan_payload_spills_locked(conn)


def _prune_orphan_payload_spills_locked(conn: sqlite3.Connection) -> int:
    """Prune scan body; caller must hold ``_SPILL_IO_LOCK``."""
    spill_root = os.path.realpath(os.path.join(
        os.path.dirname(_DB_PATH), "payload_spills", "journal"
    ))
    if not os.path.isdir(spill_root):
        return 0
    referenced: set[str] = set()
    rows = conn.execute(
        "SELECT inverse_json, forward_json, preview_json, diagnostics_json FROM journal"
    ).fetchall()
    for row in rows:
        for field in ("inverse_json", "forward_json", "preview_json", "diagnostics_json"):
            raw = row[field]
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict) or value.get("_opencut_payload_spill") is not True:
                continue
            path = os.path.realpath(str(value.get("path") or ""))
            try:
                if os.path.commonpath([spill_root, path]) == spill_root:
                    referenced.add(path)
            except ValueError:
                continue
    removed = 0
    for root, _dirs, files in os.walk(spill_root, topdown=False):
        for name in files:
            path = os.path.realpath(os.path.join(root, name))
            if path in referenced or not name.endswith(".json"):
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                logger.warning("Could not remove orphan journal payload %s", path)
        try:
            if root != spill_root and not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass
    return removed


def begin_checkpoint(
    action: str,
    label: str,
    *,
    inverse_payload: dict | None = None,
    clip_path: str = "",
    forward_payload: dict | None = None,
    preview_payload: dict | None = None,
    transaction_id: str | None = None,
) -> dict:
    """Durably record a host-write plan before the first mutation."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown journal action: {action}")
    txid = (transaction_id or str(uuid.uuid4())).strip()
    if not _TRANSACTION_ID_RE.fullmatch(txid):
        raise ValueError(
            "transaction_id must be 1-128 characters of letters, digits, "
            "hyphens, or underscores"
        )
    init_db()
    conn = _get_conn()
    now = time.time()
    with _SPILL_IO_LOCK:
        inverse_json = _encode_payload(inverse_payload or {}, field_name="inverse_json")
        forward_json = _encode_payload(
            forward_payload, field_name="forward_json", allow_none=True
        )
        preview_json = _encode_payload(
            preview_payload or {}, field_name="preview_json"
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """INSERT INTO journal (
                       created_at, action, clip_path, label, inverse_json,
                       forward_json, transaction_id, status, started_at,
                       preview_json, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    action,
                    clip_path or "",
                    label or "",
                    inverse_json,
                    forward_json,
                    txid,
                    CHECKPOINT_PENDING,
                    now,
                    preview_json,
                    now,
                ),
            )
            _prune_history(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _prune_orphan_payload_spills(conn)
    return _row_to_dict(conn.execute(
        "SELECT * FROM journal WHERE id = ?", (cur.lastrowid,)
    ).fetchone())


def complete_checkpoint(
    transaction_id: str,
    *,
    inverse_payload: dict | None = None,
    diagnostics_payload: dict | None = None,
) -> dict | None:
    """Atomically attach final inverse data and mark a host write complete."""
    init_db()
    conn = _get_conn()
    now = time.time()
    with _SPILL_IO_LOCK:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM journal WHERE transaction_id = ?", (transaction_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            if row["status"] == CHECKPOINT_COMPLETED:
                conn.commit()
                return _row_to_dict(row)
            if row["status"] not in INCOMPLETE_CHECKPOINT_STATUSES:
                conn.rollback()
                raise ValueError(f"Checkpoint cannot complete from status {row['status']}")
            inverse_json = row["inverse_json"]
            if inverse_payload is not None:
                inverse_json = _encode_payload(inverse_payload, field_name="inverse_json")
            diagnostics_json = row["diagnostics_json"]
            if diagnostics_payload is not None:
                diagnostics_json = _encode_payload(
                    diagnostics_payload, field_name="diagnostics_json"
                )
            conn.execute(
                """UPDATE journal
                   SET inverse_json = ?, diagnostics_json = ?, status = ?,
                       completed_at = ?, updated_at = ?, recovery_error = ''
                   WHERE transaction_id = ?""",
                (
                    inverse_json,
                    diagnostics_json,
                    CHECKPOINT_COMPLETED,
                    now,
                    now,
                    transaction_id,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return get_checkpoint(transaction_id)


def record(action: str, label: str, inverse_payload: dict,
           clip_path: str = "", forward_payload: "dict | None" = None) -> dict:
    """Persist a new journal entry. Returns the saved row as a dict.

    Unknown action types are rejected — adding a new type requires an
    ExtendScript inverse and a frontend dispatch branch.

    *forward_payload* (v1.10.3) stores the original ``{endpoint, payload}``
    so the panel can replay the forward op on a different clip.
    """
    if action not in VALID_ACTIONS:
        logger.warning("Rejected journal action %r (allowed: %s)",
                       action, sorted(VALID_ACTIONS))
        raise ValueError(f"Unknown journal action: {action}")
    init_db()
    conn = _get_conn()
    now = time.time()
    with _SPILL_IO_LOCK:
        inverse_json = _encode_payload(inverse_payload or {}, field_name="inverse_json")
        forward_json = _encode_payload(
            forward_payload, field_name="forward_json", allow_none=True
        )
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """INSERT INTO journal (
                       created_at, action, clip_path, label, inverse_json,
                       forward_json, status, started_at, completed_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    action,
                    clip_path or "",
                    label or "",
                    inverse_json,
                    forward_json,
                    CHECKPOINT_COMPLETED,
                    now,
                    now,
                    now,
                ),
            )
            _prune_history(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _prune_orphan_payload_spills(conn)
    entry_id = cur.lastrowid
    return _row_to_dict(conn.execute(
        "SELECT * FROM journal WHERE id = ?", (entry_id,)
    ).fetchone())


def list_entries(limit: int = 50, include_reverted: bool = True) -> list:
    init_db()
    conn = _get_conn()
    limit = _coerce_limit(limit, 50)
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


def get_checkpoint(transaction_id: str, *, resolve_payloads: bool = False) -> "dict | None":
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM journal WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    return _row_to_dict(row, resolve_payloads=resolve_payloads) if row else None


def list_incomplete_checkpoints(limit: int = 50) -> list[dict]:
    """Return durable host writes that did not reach atomic completion."""
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM journal
           WHERE status IN (?, ?)
           ORDER BY created_at DESC LIMIT ?""",
        (
            CHECKPOINT_PENDING,
            CHECKPOINT_RECOVERY_FAILED,
            _coerce_limit(limit, 50),
        ),
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def mark_recovery_failed(
    transaction_id: str,
    error: str,
    *,
    diagnostics_payload: dict | None = None,
) -> dict | None:
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM journal WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    if row is None:
        return None
    with _SPILL_IO_LOCK:
        diagnostics_json = row["diagnostics_json"]
        if diagnostics_payload is not None:
            diagnostics_json = _encode_payload(
                diagnostics_payload, field_name="diagnostics_json"
            )
        now = time.time()
        conn.execute(
            """UPDATE journal
               SET status = ?, recovery_error = ?, recovery_attempted_at = ?,
                   diagnostics_json = ?, updated_at = ?
               WHERE transaction_id = ? AND status IN (?, ?)""",
            (
                CHECKPOINT_RECOVERY_FAILED,
                str(error or "Recovery failed")[:4000],
                now,
                diagnostics_json,
                now,
                transaction_id,
                CHECKPOINT_PENDING,
                CHECKPOINT_RECOVERY_FAILED,
            ),
        )
        conn.commit()
    return get_checkpoint(transaction_id)


def mark_recovered(transaction_id: str) -> dict | None:
    init_db()
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM journal WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    if row is None:
        return None
    now = time.time()
    cur = conn.execute(
        """UPDATE journal
           SET status = ?, reverted = 1, reverted_at = ?,
               recovery_attempted_at = ?, recovery_error = '', updated_at = ?
           WHERE transaction_id = ? AND status IN (?, ?)""",
        (
            CHECKPOINT_RESTORED,
            now,
            now,
            now,
            transaction_id,
            CHECKPOINT_PENDING,
            CHECKPOINT_RECOVERY_FAILED,
        ),
    )
    conn.commit()
    _prune_orphan_payload_spills(conn)
    if cur.rowcount == 0:
        raise ValueError(f"Checkpoint cannot recover from status {row['status']}")
    return get_checkpoint(transaction_id)


def get_recovery_diagnostics(transaction_id: str) -> dict | None:
    entry = get_checkpoint(transaction_id)
    if entry is None:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "database": get_db_diagnostics(),
        "checkpoint": entry,
        "recovery_contract": {
            "automatic_recovery_available": entry["automatic_recovery_available"],
            "requires_premiere_host": entry["revertible"],
            "manual_recovery_required": not entry["automatic_recovery_available"],
        },
    }


def mark_reverted(entry_id: int) -> bool:
    init_db()
    conn = _get_conn()
    cur = conn.execute(
        """UPDATE journal
           SET reverted = 1, reverted_at = ?, status = ?, updated_at = ?
           WHERE id = ? AND reverted = 0""",
        (time.time(), CHECKPOINT_RESTORED, time.time(), entry_id),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_entry(entry_id: int, *, dry_run: bool = False, backup: bool = False,
                 backup_dir: str | None = None) -> bool | dict:
    init_db()
    conn = _get_conn()
    affected = count_rows(
        conn,
        "journal",
        "id = ? AND status NOT IN (?, ?)",
        (entry_id, CHECKPOINT_PENDING, CHECKPOINT_RECOVERY_FAILED),
    )
    if dry_run or backup:
        result = prepare_destructive_result(
            _DB_PATH,
            store_name="journal",
            operation="delete_entry",
            affected_rows=affected,
            dry_run=dry_run,
            backup=backup,
            backup_dir=backup_dir,
        )
        result["entry_id"] = entry_id
        if dry_run:
            return result
    cur = conn.execute(
        "DELETE FROM journal WHERE id = ? AND status NOT IN (?, ?)",
        (entry_id, CHECKPOINT_PENDING, CHECKPOINT_RECOVERY_FAILED),
    )
    conn.commit()
    _prune_orphan_payload_spills(conn)
    if backup:
        result["affected_rows"] = cur.rowcount
        return result
    return cur.rowcount > 0


def clear_all(*, dry_run: bool = False, backup: bool = False,
              backup_dir: str | None = None) -> int | dict:
    """Delete completed history while preserving incomplete recovery records."""
    init_db()
    conn = _get_conn()
    where = "status NOT IN (?, ?)"
    params = (CHECKPOINT_PENDING, CHECKPOINT_RECOVERY_FAILED)
    affected = count_rows(conn, "journal", where, params)
    if dry_run or backup:
        result = prepare_destructive_result(
            _DB_PATH,
            store_name="journal",
            operation="clear_all",
            affected_rows=affected,
            dry_run=dry_run,
            backup=backup,
            backup_dir=backup_dir,
        )
        if dry_run:
            return result
    cur = conn.execute(f"DELETE FROM journal WHERE {where}", params)
    conn.commit()
    _prune_orphan_payload_spills(conn)
    if backup:
        result["affected_rows"] = cur.rowcount
        return result
    return cur.rowcount


def get_db_diagnostics() -> dict:
    """Return SQLite page, freelist, WAL, and size diagnostics for journal.db."""
    return build_sqlite_diagnostic(_DB_PATH, store_name="journal")


def _row_to_dict(row, *, resolve_payloads: bool = False) -> "dict | None":
    if row is None:
        return None
    try:
        inverse = decode_json_or_spill_marker(row["inverse_json"])
    except (json.JSONDecodeError, TypeError):
        inverse = {}
    forward = None
    try:
        raw_fwd = row["forward_json"]
        if raw_fwd:
            forward = decode_json_or_spill_marker(raw_fwd)
    except (IndexError, KeyError, json.JSONDecodeError, TypeError):
        forward = None
    preview = {}
    diagnostics = None
    try:
        if row["preview_json"]:
            preview = decode_json_or_spill_marker(row["preview_json"])
    except (IndexError, KeyError, json.JSONDecodeError, TypeError):
        preview = {}
    try:
        if row["diagnostics_json"]:
            diagnostics = decode_json_or_spill_marker(row["diagnostics_json"])
    except (IndexError, KeyError, json.JSONDecodeError, TypeError):
        diagnostics = None
    inverse_integrity = _spill_integrity(inverse)
    forward_integrity = _spill_integrity(forward)
    preview_integrity = _spill_integrity(preview)
    diagnostics_integrity = _spill_integrity(diagnostics)
    if resolve_payloads:
        inverse, inverse_integrity = _resolve_payload(inverse)
        forward, forward_integrity = _resolve_payload(forward)
        preview, preview_integrity = _resolve_payload(preview)
        diagnostics, diagnostics_integrity = _resolve_payload(diagnostics)
    try:
        transaction_id = row["transaction_id"]
        status = row["status"] or CHECKPOINT_COMPLETED
        started_at = row["started_at"] or row["created_at"]
        completed_at = row["completed_at"]
        updated_at = row["updated_at"] or row["created_at"]
        recovery_error = row["recovery_error"] or ""
        recovery_attempted_at = row["recovery_attempted_at"]
    except (IndexError, KeyError):
        transaction_id = None
        status = CHECKPOINT_COMPLETED
        started_at = row["created_at"]
        completed_at = row["created_at"]
        updated_at = row["created_at"]
        recovery_error = ""
        recovery_attempted_at = None
    action_revertible = row["action"] in REVERTIBLE_ACTIONS
    inverse_available = inverse_integrity["ok"] and bool(inverse)
    entry_revertible = action_revertible and (transaction_id is None or inverse_available)
    automatic_recovery_available = (
        status in INCOMPLETE_CHECKPOINT_STATUSES
        and entry_revertible
    )
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "action": row["action"],
        "clip_path": row["clip_path"] or "",
        "label": row["label"] or "",
        "inverse": inverse,
        "forward": forward,
        "preview": preview,
        "diagnostics": diagnostics,
        "reverted": bool(row["reverted"]),
        "reverted_at": row["reverted_at"],
        "revertible": entry_revertible,
        "inverse_available": inverse_available,
        "transaction_id": transaction_id,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "updated_at": updated_at,
        "incomplete": status in INCOMPLETE_CHECKPOINT_STATUSES,
        "automatic_recovery_available": automatic_recovery_available,
        "recovery_error": recovery_error,
        "recovery_attempted_at": recovery_attempted_at,
        "artifact_integrity": {
            "inverse": inverse_integrity,
            "forward": forward_integrity,
            "preview": preview_integrity,
            "diagnostics": diagnostics_integrity,
        },
    }
