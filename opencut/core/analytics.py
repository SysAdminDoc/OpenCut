"""
OpenCut Feature Usage Analytics

Records endpoint usage to a local SQLite database and provides
aggregated statistics for a usage dashboard.

Database: ``~/.opencut/analytics.db``
"""

import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "analytics.db")
_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


# ---------------------------------------------------------------------------
# Connection / schema helpers
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    """Get or create a thread-local SQLite connection for analytics."""
    conn = getattr(_LOCAL, "analytics_conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _LOCAL.analytics_conn = conn
    return conn


def _init_db():
    """Create the usage_log table if it does not exist."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint    TEXT NOT NULL,
                job_type    TEXT DEFAULT '',
                duration_ms INTEGER DEFAULT 0,
                success     INTEGER DEFAULT 1,
                timestamp   REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_endpoint
            ON usage_log (endpoint)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp
            ON usage_log (timestamp)
        """)
        conn.commit()
        _INITIALIZED = True
        logger.debug("Analytics database initialized at %s", _DB_PATH)


# ---------------------------------------------------------------------------
# Public API — recording
# ---------------------------------------------------------------------------
def record_usage(
    endpoint: str,
    duration_ms: int,
    success: bool,
    job_type: str = "",
) -> None:
    """
    Log a single usage event.

    Args:
        endpoint:    The route/endpoint path (e.g. ``/silence``).
        duration_ms: How long the request took in milliseconds.
        success:     Whether the request succeeded.
        job_type:    Optional job type label.
    """
    _init_db()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO usage_log (endpoint, job_type, duration_ms, success, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (endpoint, job_type, int(duration_ms), 1 if success else 0, time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API — querying
# ---------------------------------------------------------------------------
def get_usage_stats(days: int = 30) -> dict:
    """
    Return aggregated usage statistics for the last *days* days.

    Returns:
        dict with keys:
            top_features  — list of {endpoint, count, avg_duration_ms, error_rate}
            total_jobs    — int
            total_errors  — int
            avg_duration_ms — float
            daily_usage   — list of {date, count} for charting
    """
    _init_db()
    conn = _get_conn()
    cutoff = time.time() - (days * 86400)

    # ---- top features ----
    rows = conn.execute(
        "SELECT endpoint, "
        "  COUNT(*) AS cnt, "
        "  AVG(duration_ms) AS avg_dur, "
        "  SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors "
        "FROM usage_log "
        "WHERE timestamp >= ? "
        "GROUP BY endpoint "
        "ORDER BY cnt DESC",
        (cutoff,),
    ).fetchall()

    top_features = []
    for r in rows:
        count = r["cnt"]
        errors = r["errors"]
        top_features.append({
            "endpoint": r["endpoint"],
            "count": count,
            "avg_duration_ms": round(r["avg_dur"], 1) if r["avg_dur"] else 0.0,
            "error_rate": round(errors / count, 4) if count else 0.0,
        })

    # ---- totals ----
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "  SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors, "
        "  AVG(duration_ms) AS avg_dur "
        "FROM usage_log WHERE timestamp >= ?",
        (cutoff,),
    ).fetchone()

    total_jobs = row["total"] if row["total"] else 0
    total_errors = row["errors"] if row["errors"] else 0
    avg_duration = round(row["avg_dur"], 1) if row["avg_dur"] else 0.0

    # ---- daily usage ----
    daily_rows = conn.execute(
        "SELECT DATE(timestamp, 'unixepoch') AS day, COUNT(*) AS cnt "
        "FROM usage_log "
        "WHERE timestamp >= ? "
        "GROUP BY day "
        "ORDER BY day ASC",
        (cutoff,),
    ).fetchall()

    daily_usage = [{"date": r["day"], "count": r["cnt"]} for r in daily_rows]

    return {
        "top_features": top_features,
        "total_jobs": total_jobs,
        "total_errors": total_errors,
        "avg_duration_ms": avg_duration,
        "daily_usage": daily_usage,
    }


def get_feature_stats(endpoint: str, days: int = 30) -> dict:
    """
    Return detailed statistics for a single endpoint.

    Returns:
        dict with keys:
            endpoint, total_calls, success_count, error_count, error_rate,
            avg_duration_ms, min_duration_ms, max_duration_ms,
            daily_usage (list of {date, count, avg_duration_ms})
    """
    _init_db()
    conn = _get_conn()
    cutoff = time.time() - (days * 86400)

    # ---- aggregate stats ----
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "  SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS ok, "
        "  SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS errors, "
        "  AVG(duration_ms) AS avg_dur, "
        "  MIN(duration_ms) AS min_dur, "
        "  MAX(duration_ms) AS max_dur "
        "FROM usage_log "
        "WHERE endpoint = ? AND timestamp >= ?",
        (endpoint, cutoff),
    ).fetchone()

    total = row["total"] if row["total"] else 0
    ok = row["ok"] if row["ok"] else 0
    errors = row["errors"] if row["errors"] else 0

    # ---- daily breakdown ----
    daily_rows = conn.execute(
        "SELECT DATE(timestamp, 'unixepoch') AS day, "
        "  COUNT(*) AS cnt, "
        "  AVG(duration_ms) AS avg_dur "
        "FROM usage_log "
        "WHERE endpoint = ? AND timestamp >= ? "
        "GROUP BY day ORDER BY day ASC",
        (endpoint, cutoff),
    ).fetchall()

    daily_usage = [
        {
            "date": r["day"],
            "count": r["cnt"],
            "avg_duration_ms": round(r["avg_dur"], 1) if r["avg_dur"] else 0.0,
        }
        for r in daily_rows
    ]

    return {
        "endpoint": endpoint,
        "total_calls": total,
        "success_count": ok,
        "error_count": errors,
        "error_rate": round(errors / total, 4) if total else 0.0,
        "avg_duration_ms": round(row["avg_dur"], 1) if row["avg_dur"] else 0.0,
        "min_duration_ms": row["min_dur"] if row["min_dur"] is not None else 0,
        "max_duration_ms": row["max_dur"] if row["max_dur"] is not None else 0,
        "daily_usage": daily_usage,
    }
