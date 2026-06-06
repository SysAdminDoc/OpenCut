"""
OpenCut Job Persistence — SQLite Backend

Provides durable job storage so jobs survive server restarts.
The in-memory ``jobs`` dict in ``opencut.jobs`` remains the primary
hot-path store; this module syncs completed/errored jobs to disk
and can restore interrupted jobs on startup.

Schema::

    CREATE TABLE jobs (
        id          TEXT PRIMARY KEY,
        type        TEXT NOT NULL,
        filepath    TEXT DEFAULT '',
        status      TEXT NOT NULL DEFAULT 'running',
        progress    INTEGER DEFAULT 0,
        message     TEXT DEFAULT '',
        result_json TEXT DEFAULT NULL,
        error       TEXT DEFAULT NULL,
        endpoint    TEXT DEFAULT '',
        payload_json TEXT DEFAULT NULL,
        resumable   INTEGER DEFAULT 0,
        partial_output_path TEXT DEFAULT '',
        resume_source_job_id TEXT DEFAULT '',
        resume_attempt INTEGER DEFAULT 0,
        peak_vram_mb INTEGER DEFAULT NULL,
        peak_cpu_pct INTEGER DEFAULT NULL,
        peak_rss_mb INTEGER DEFAULT NULL,
        exit_reason TEXT DEFAULT '',
        created_at  REAL NOT NULL,
        started_at  REAL DEFAULT NULL,
        completed_at REAL DEFAULT NULL
    );
"""

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Optional

from opencut.local_db_diagnostics import build_sqlite_diagnostic
from opencut.local_db_migrations import migrate_user_version
from opencut.local_db_payloads import decode_json_or_spill_marker, spill_json_if_needed

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "jobs.db")
_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_INITIALIZED_PATH = None
_ALL_CONNECTIONS = {}  # thread_id -> Connection; dead thread entries cleaned in close_all
_CONN_LOCK = threading.Lock()
SCHEMA_VERSION = 2

# How long to keep completed jobs in the database
COMPLETED_JOB_TTL = 7 * 24 * 3600  # 7 days

MAX_RESULT_JSON_BYTES = 256 * 1024

_MAX_JOB_LIST_LIMIT = 1000


def _connection_is_usable(conn: Optional[sqlite3.Connection]) -> bool:
    """Return True when a cached SQLite connection is still open."""
    if conn is None:
        return False
    try:
        conn.execute("SELECT 1")
    except sqlite3.Error:
        return False
    return True


def _coerce_limit(value, default):
    """Coerce a user-supplied LIMIT into ``[1, _MAX_JOB_LIST_LIMIT]``.

    Non-numeric or out-of-range inputs fall back to *default*. Capping the
    upper bound protects against callers that pass ``?limit=999999`` and
    pin the job-list endpoint while SQLite streams rows.
    """
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(_MAX_JOB_LIST_LIMIT, coerced))


def _coerce_offset(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_optional_nonnegative_int(value):
    if value is None or value == "":
        return None
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError, OverflowError):
        return None


def _get_conn() -> sqlite3.Connection:
    """Get or create a thread-local SQLite connection."""
    conn = getattr(_LOCAL, "conn", None)
    conn_path = getattr(_LOCAL, "conn_path", None)
    if conn is not None and conn_path != _DB_PATH:
        with _CONN_LOCK:
            for tid, tracked in list(_ALL_CONNECTIONS.items()):
                if tracked is conn:
                    _ALL_CONNECTIONS.pop(tid, None)
        try:
            conn.close()
        except sqlite3.Error:
            pass
        _LOCAL.conn = None
        _LOCAL.conn_path = None
        conn = None
    if conn is not None and not _connection_is_usable(conn):
        with _CONN_LOCK:
            for tid, tracked in list(_ALL_CONNECTIONS.items()):
                if tracked is conn:
                    _ALL_CONNECTIONS.pop(tid, None)
        _LOCAL.conn = None
        _LOCAL.conn_path = None
        conn = None
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _LOCAL.conn = conn
        _LOCAL.conn_path = _DB_PATH
        with _CONN_LOCK:
            _ALL_CONNECTIONS[threading.get_ident()] = conn
            # Prune connections from dead threads to prevent unbounded growth
            alive_ids = {t.ident for t in threading.enumerate() if t.ident is not None}
            dead_ids = [tid for tid in _ALL_CONNECTIONS if tid not in alive_ids]
            for tid in dead_ids:
                try:
                    _ALL_CONNECTIONS[tid].close()
                except Exception:
                    pass
                del _ALL_CONNECTIONS[tid]
    return conn


def close_all_connections():
    """Close all tracked SQLite connections. Call on server shutdown.

    Also prunes connections from threads that are no longer alive.
    """
    with _CONN_LOCK:
        for conn in _ALL_CONNECTIONS.values():
            try:
                conn.close()
            except Exception:
                pass
        _ALL_CONNECTIONS.clear()
    try:
        if getattr(_LOCAL, "conn", None) is not None:
            _LOCAL.conn = None
            _LOCAL.conn_path = None
    except Exception:
        pass
    logger.debug("All job store connections closed")


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id           TEXT PRIMARY KEY,
            type         TEXT NOT NULL,
            filepath     TEXT DEFAULT '',
            status       TEXT NOT NULL DEFAULT 'running',
            progress     INTEGER DEFAULT 0,
            message      TEXT DEFAULT '',
            result_json  TEXT DEFAULT NULL,
            error        TEXT DEFAULT NULL,
            endpoint     TEXT DEFAULT '',
            payload_json TEXT DEFAULT NULL,
            created_at   REAL NOT NULL,
            started_at   REAL DEFAULT NULL,
            completed_at REAL DEFAULT NULL
        )
    """)


def _migrate_schema_v2(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    migrations = {
        "resumable": "ALTER TABLE jobs ADD COLUMN resumable INTEGER DEFAULT 0",
        "partial_output_path": "ALTER TABLE jobs ADD COLUMN partial_output_path TEXT DEFAULT ''",
        "resume_source_job_id": "ALTER TABLE jobs ADD COLUMN resume_source_job_id TEXT DEFAULT ''",
        "resume_attempt": "ALTER TABLE jobs ADD COLUMN resume_attempt INTEGER DEFAULT 0",
        "peak_vram_mb": "ALTER TABLE jobs ADD COLUMN peak_vram_mb INTEGER DEFAULT NULL",
        "peak_cpu_pct": "ALTER TABLE jobs ADD COLUMN peak_cpu_pct INTEGER DEFAULT NULL",
        "peak_rss_mb": "ALTER TABLE jobs ADD COLUMN peak_rss_mb INTEGER DEFAULT NULL",
        "exit_reason": "ALTER TABLE jobs ADD COLUMN exit_reason TEXT DEFAULT ''",
    }
    for column, statement in migrations.items():
        if column not in columns:
            conn.execute(statement)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status
        ON jobs (status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_resume
        ON jobs (status, resumable)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_created
        ON jobs (created_at)
    """)


def init_db():
    """Create the jobs table if it doesn't exist. Safe to call multiple times."""
    global _INITIALIZED, _INITIALIZED_PATH
    if _INITIALIZED and _INITIALIZED_PATH == _DB_PATH:
        return
    with _INIT_LOCK:
        if _INITIALIZED and _INITIALIZED_PATH == _DB_PATH:
            return
        conn = _get_conn()
        migrate_user_version(
            conn,
            store_name="job store",
            target_version=SCHEMA_VERSION,
            migrations={
                1: _create_schema_v1,
                2: _migrate_schema_v2,
            },
        )
        _ensure_indexes(conn)
        conn.commit()
        _INITIALIZED = True
        _INITIALIZED_PATH = _DB_PATH
        logger.debug("Job store initialized at %s", _DB_PATH)


def save_job(job_dict):
    """Insert or update a job record.

    ``job_dict`` should match the in-memory job format from ``opencut.jobs``.
    """
    init_db()
    conn = _get_conn()
    result_json = None
    if job_dict.get("result") is not None:
        try:
            result_json = json.dumps(job_dict["result"])
        except (TypeError, ValueError):
            result_json = str(job_dict["result"])
        result_json = spill_json_if_needed(
            result_json,
            base_dir=os.path.dirname(_DB_PATH),
            namespace="jobs",
            field_name="result_json",
            max_bytes=MAX_RESULT_JSON_BYTES,
        )

    payload_json = None
    if job_dict.get("_payload") is not None:
        try:
            payload_json = json.dumps(job_dict["_payload"])
        except (TypeError, ValueError):
            pass

    resumable = None
    if "resumable" in job_dict:
        resumable = 1 if job_dict.get("resumable") else 0
    partial_output_path = None
    if "partial_output_path" in job_dict:
        partial_output_path = str(job_dict.get("partial_output_path") or "")
    resume_source_job_id = None
    if "resume_source_job_id" in job_dict:
        resume_source_job_id = str(job_dict.get("resume_source_job_id") or "")
    resume_attempt = None
    if "resume_attempt" in job_dict:
        try:
            resume_attempt = max(0, int(job_dict.get("resume_attempt") or 0))
        except (TypeError, ValueError):
            resume_attempt = 0
    peak_vram_mb = (
        _coerce_optional_nonnegative_int(job_dict.get("peak_vram_mb"))
        if "peak_vram_mb" in job_dict else None
    )
    peak_cpu_pct = (
        _coerce_optional_nonnegative_int(job_dict.get("peak_cpu_pct"))
        if "peak_cpu_pct" in job_dict else None
    )
    peak_rss_mb = (
        _coerce_optional_nonnegative_int(job_dict.get("peak_rss_mb"))
        if "peak_rss_mb" in job_dict else None
    )
    exit_reason = ""
    if "exit_reason" in job_dict:
        exit_reason = str(job_dict.get("exit_reason") or "")

    now = time.time()
    completed_at = job_dict.get("completed_at")
    if completed_at is None and job_dict.get("status") in ("complete", "error", "cancelled", "interrupted"):
        completed_at = now
    started_at = job_dict.get("started_at") or job_dict.get("created", now)

    conn.execute("""
        INSERT INTO jobs (id, type, filepath, status, progress, message,
                          result_json, error, endpoint, payload_json,
                          resumable, partial_output_path, resume_source_job_id,
                          resume_attempt, peak_vram_mb, peak_cpu_pct,
                          peak_rss_mb, exit_reason,
                          created_at, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            type = COALESCE(NULLIF(excluded.type, ''), jobs.type),
            filepath = COALESCE(NULLIF(excluded.filepath, ''), jobs.filepath),
            status = excluded.status,
            progress = excluded.progress,
            message = excluded.message,
            result_json = excluded.result_json,
            error = excluded.error,
            endpoint = COALESCE(NULLIF(excluded.endpoint, ''), jobs.endpoint),
            payload_json = COALESCE(excluded.payload_json, jobs.payload_json),
            resumable = COALESCE(excluded.resumable, jobs.resumable),
            partial_output_path = COALESCE(NULLIF(excluded.partial_output_path, ''), jobs.partial_output_path),
            resume_source_job_id = COALESCE(NULLIF(excluded.resume_source_job_id, ''), jobs.resume_source_job_id),
            resume_attempt = COALESCE(excluded.resume_attempt, jobs.resume_attempt),
            peak_vram_mb = COALESCE(excluded.peak_vram_mb, jobs.peak_vram_mb),
            peak_cpu_pct = COALESCE(excluded.peak_cpu_pct, jobs.peak_cpu_pct),
            peak_rss_mb = COALESCE(excluded.peak_rss_mb, jobs.peak_rss_mb),
            exit_reason = COALESCE(NULLIF(excluded.exit_reason, ''), jobs.exit_reason),
            created_at = MIN(jobs.created_at, excluded.created_at),
            started_at = COALESCE(jobs.started_at, excluded.started_at),
            completed_at = COALESCE(excluded.completed_at, jobs.completed_at)
    """, (
        job_dict.get("id", ""),
        job_dict.get("type", ""),
        job_dict.get("filepath", ""),
        job_dict.get("status", "running"),
        job_dict.get("progress", 0),
        job_dict.get("message", ""),
        result_json,
        job_dict.get("error"),
        job_dict.get("_endpoint", ""),
        payload_json,
        resumable,
        partial_output_path,
        resume_source_job_id,
        resume_attempt,
        peak_vram_mb,
        peak_cpu_pct,
        peak_rss_mb,
        exit_reason,
        job_dict.get("created", now),
        started_at,
        completed_at,
    ))
    conn.commit()


def get_job(job_id):
    """Retrieve a single job by ID. Returns dict or None."""
    init_db()
    conn = _get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_jobs(status=None, limit=100, offset=0):
    """List jobs, optionally filtered by status. Newest first."""
    init_db()
    conn = _get_conn()
    limit = _coerce_limit(limit, 100)
    offset = _coerce_offset(offset, 0)
    if status:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (status, limit, offset)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_interrupted_jobs():
    """Find jobs that were 'running' when the server died.

    Call this on startup to offer retry.
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status = 'interrupted' ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def mark_interrupted():
    """Mark all 'running' jobs as 'interrupted'. Call on server startup."""
    init_db()
    conn = _get_conn()
    count = conn.execute(
        "UPDATE jobs SET status = 'interrupted', "
        "message = 'Server restarted during processing', "
        "exit_reason = 'interrupted', "
        "completed_at = ? WHERE status = 'running'",
        (time.time(),)
    ).rowcount
    conn.commit()
    if count:
        logger.info("Marked %d interrupted jobs from previous session", count)
    return count


def cleanup_old_jobs():
    """Delete completed/errored/cancelled jobs older than COMPLETED_JOB_TTL."""
    init_db()
    conn = _get_conn()
    cutoff = time.time() - COMPLETED_JOB_TTL
    count = conn.execute(
        "DELETE FROM jobs WHERE status IN ('complete', 'error', 'cancelled', 'interrupted') "
        "AND COALESCE(completed_at, created_at) < ?",
        (cutoff,)
    ).rowcount
    conn.commit()
    if count:
        logger.debug("Cleaned up %d old jobs from database", count)
    return count


def get_job_stats():
    """Return aggregate job statistics."""
    init_db()
    conn = _get_conn()
    today_start = time.time() - 86400
    stats = {}
    for status in ("running", "complete", "error", "cancelled", "interrupted"):
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
        ).fetchone()
        stats[status] = row[0] if row else 0
    row = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'complete' AND completed_at > ?",
        (today_start,)
    ).fetchone()
    stats["completed_today"] = row[0] if row else 0
    row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
    stats["total"] = row[0] if row else 0
    return stats


def get_db_diagnostics():
    """Return SQLite page, freelist, WAL, and size diagnostics for jobs.db."""
    return build_sqlite_diagnostic(_DB_PATH, store_name="jobs")


def _row_to_dict(row):
    """Convert a sqlite3.Row to a plain dict matching the in-memory format."""
    d = {
        "id": row["id"],
        "type": row["type"],
        "filepath": row["filepath"],
        "status": row["status"],
        "progress": row["progress"],
        "message": row["message"],
        "error": row["error"],
        "created": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "resumable": bool(row["resumable"]),
        "partial_output_path": row["partial_output_path"] or "",
        "resume_source_job_id": row["resume_source_job_id"] or "",
        "resume_attempt": int(row["resume_attempt"] or 0),
        "peak_vram_mb": row["peak_vram_mb"] if row["peak_vram_mb"] is not None else None,
        "peak_cpu_pct": row["peak_cpu_pct"] if row["peak_cpu_pct"] is not None else None,
        "peak_rss_mb": row["peak_rss_mb"] if row["peak_rss_mb"] is not None else None,
        "exit_reason": row["exit_reason"] or "",
    }
    if row["result_json"]:
        try:
            d["result"] = decode_json_or_spill_marker(row["result_json"])
        except (json.JSONDecodeError, TypeError):
            d["result"] = row["result_json"]
    else:
        d["result"] = None
    if row["endpoint"]:
        d["endpoint"] = row["endpoint"]
    if row["payload_json"]:
        try:
            d["payload"] = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError):
            d["payload"] = None
    return d
