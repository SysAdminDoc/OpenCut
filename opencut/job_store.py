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

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "jobs.db")
_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False

# How long to keep completed jobs in the database
COMPLETED_JOB_TTL = 7 * 24 * 3600  # 7 days


def _get_conn() -> sqlite3.Connection:
    """Get or create a thread-local SQLite connection."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _LOCAL.conn = conn
    return conn


def init_db():
    """Create the jobs table if it doesn't exist. Safe to call multiple times."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = _get_conn()
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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs (status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created
            ON jobs (created_at)
        """)
        conn.commit()
        _INITIALIZED = True
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

    payload_json = None
    if job_dict.get("_payload") is not None:
        try:
            payload_json = json.dumps(job_dict["_payload"])
        except (TypeError, ValueError):
            pass

    now = time.time()
    completed_at = None
    if job_dict.get("status") in ("complete", "error", "cancelled"):
        completed_at = now

    conn.execute("""
        INSERT INTO jobs (id, type, filepath, status, progress, message,
                          result_json, error, endpoint, payload_json,
                          created_at, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status,
            progress = excluded.progress,
            message = excluded.message,
            result_json = excluded.result_json,
            error = excluded.error,
            completed_at = excluded.completed_at
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
        job_dict.get("created", now),
        job_dict.get("created", now),
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
        "AND created_at < ?",
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
    }
    if row["result_json"]:
        try:
            d["result"] = json.loads(row["result_json"])
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
