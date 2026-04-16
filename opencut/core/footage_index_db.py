"""
OpenCut Footage Index — SQLite + FTS5

Persistent, searchable index of transcribed media files.
Replaces the JSON-based footage_index.json with a proper database.
"""

import logging
import os
import sqlite3
import threading
import time

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "footage_index.db")
_thread_local = threading.local()

# Track all opened connections so close_all_connections() can close them
# on server shutdown.  Mirrors the pattern in job_store.py — without this
# the ThreadPoolExecutor keeps thread-local connections open for the life
# of the process, which holds open the WAL file and prevents clean
# shutdown on Windows.
_ALL_CONNECTIONS: "dict[int, sqlite3.Connection]" = {}
_CONN_LOCK = threading.Lock()


def _get_conn():
    """Get a thread-local SQLite connection."""
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


def close_all_connections():
    """Close every tracked connection. Called on server shutdown.

    Silently skips connections whose owning thread has already died — those
    will be reaped by the OS. Clears thread-local refs on the current thread
    so subsequent calls re-open fresh connections.
    """
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


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS footage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            transcript TEXT NOT NULL DEFAULT '',
            indexed_at REAL NOT NULL,
            file_mtime REAL NOT NULL DEFAULT 0,
            duration REAL NOT NULL DEFAULT 0,
            file_size INTEGER NOT NULL DEFAULT 0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS footage_fts USING fts5(
            file_path,
            transcript,
            content=footage,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS footage_ai AFTER INSERT ON footage BEGIN
            INSERT INTO footage_fts(rowid, file_path, transcript)
            VALUES (new.id, new.file_path, new.transcript);
        END;

        CREATE TRIGGER IF NOT EXISTS footage_ad AFTER DELETE ON footage BEGIN
            INSERT INTO footage_fts(footage_fts, rowid, file_path, transcript)
            VALUES ('delete', old.id, old.file_path, old.transcript);
        END;

        CREATE TRIGGER IF NOT EXISTS footage_au AFTER UPDATE ON footage BEGIN
            INSERT INTO footage_fts(footage_fts, rowid, file_path, transcript)
            VALUES ('delete', old.id, old.file_path, old.transcript);
            INSERT INTO footage_fts(rowid, file_path, transcript)
            VALUES (new.id, new.file_path, new.transcript);
        END;
    """)
    conn.commit()


def index_file(file_path, transcript, duration=0, file_size=0):
    """Add or update a file in the index.

    Args:
        file_path: absolute path to the media file
        transcript: full transcript text
        duration: media duration in seconds
        file_size: file size in bytes
    """
    # Without init_db the very first call from a fresh thread (no prior
    # search/get_stats) hits "no such table: footage" because the
    # CREATE TABLE IF NOT EXISTS only runs from init_db. Mirror the
    # pattern used by every read function below.
    init_db()
    conn = _get_conn()
    now = time.time()

    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        mtime = 0

    if file_size == 0:
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            pass

    conn.execute("""
        INSERT INTO footage (file_path, transcript, indexed_at, file_mtime, duration, file_size)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            transcript = excluded.transcript,
            indexed_at = excluded.indexed_at,
            file_mtime = excluded.file_mtime,
            duration = excluded.duration,
            file_size = excluded.file_size
    """, (file_path, transcript, now, mtime, duration, file_size))
    conn.commit()

    logger.debug("Indexed: %s (%d chars)", os.path.basename(file_path), len(transcript))


def search(query, limit=50):
    """Full-text search across indexed footage.

    Args:
        query: search query string (FTS5 syntax supported)
        limit: max results

    Returns:
        list of dicts: {file_path, transcript, snippet, rank, indexed_at, duration}
    """
    if not query or not query.strip():
        return []

    conn = _get_conn()
    init_db()  # Ensure tables exist

    # Escape special FTS5 characters for safety
    safe_query = query.strip()
    # Wrap each word in quotes for phrase-like matching
    words = safe_query.split()
    if len(words) > 1:
        fts_query = " ".join(f'"{w}"' for w in words)
    else:
        fts_query = f'"{safe_query}"' if safe_query else safe_query

    try:
        rows = conn.execute("""
            SELECT f.file_path, f.transcript, f.indexed_at, f.duration, f.file_size,
                   snippet(footage_fts, 1, '<mark>', '</mark>', '...', 32) AS snippet,
                   rank
            FROM footage_fts
            JOIN footage f ON f.id = footage_fts.rowid
            WHERE footage_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, limit)).fetchall()

        results = []
        for row in rows:
            results.append({
                "file_path": row["file_path"],
                "transcript": row["transcript"][:500],  # Truncate for response size
                "snippet": row["snippet"] or "",
                "rank": row["rank"],
                "indexed_at": row["indexed_at"],
                "duration": row["duration"],
                "file_size": row["file_size"],
            })
        return results

    except sqlite3.OperationalError as e:
        # FTS5 query syntax error -- fall back to LIKE search.
        # Escape SQL LIKE wildcards (`%`, `_`) and our escape char (`\`) so a
        # search for "test_file" doesn't match "testXfile" via the underscore
        # wildcard. ESCAPE clause tells SQLite which char prefixes a literal.
        logger.warning("FTS5 query failed (%s), falling back to LIKE: %s", e, query)
        like_query = (
            safe_query.replace("\\", "\\\\")
                      .replace("%", "\\%")
                      .replace("_", "\\_")
        )
        rows = conn.execute("""
            SELECT file_path, transcript, indexed_at, duration, file_size
            FROM footage
            WHERE transcript LIKE ? ESCAPE '\\' COLLATE NOCASE
            LIMIT ?
        """, (f"%{like_query}%", limit)).fetchall()

        return [
            {
                "file_path": row["file_path"],
                "transcript": row["transcript"][:500],
                "snippet": "",
                "rank": 0,
                "indexed_at": row["indexed_at"],
                "duration": row["duration"],
                "file_size": row["file_size"],
            }
            for row in rows
        ]


def needs_reindex(file_path):
    """Check if a file needs to be (re-)indexed.

    Returns True if the file is not in the index or its mtime has changed.
    """
    conn = _get_conn()
    init_db()

    row = conn.execute(
        "SELECT file_mtime FROM footage WHERE file_path = ?", (file_path,)
    ).fetchone()

    if row is None:
        return True

    try:
        current_mtime = os.path.getmtime(file_path)
    except OSError:
        return False  # File doesn't exist, don't re-index

    return abs(current_mtime - row["file_mtime"]) > 0.01


def get_stats():
    """Get index statistics.

    Returns:
        dict with total_files, total_size, last_indexed_at, db_size
    """
    conn = _get_conn()
    init_db()

    row = conn.execute("""
        SELECT COUNT(*) as total,
               COALESCE(SUM(file_size), 0) as total_size,
               COALESCE(MAX(indexed_at), 0) as last_indexed
        FROM footage
    """).fetchone()

    db_size = 0
    try:
        db_size = os.path.getsize(_DB_PATH)
    except OSError:
        pass

    return {
        "total_files": row["total"],
        "total_size": row["total_size"],
        "last_indexed_at": row["last_indexed"],
        "db_size": db_size,
    }


def clear_index():
    """Remove all entries from the index."""
    conn = _get_conn()
    conn.execute("DELETE FROM footage")
    conn.execute("INSERT INTO footage_fts(footage_fts) VALUES ('rebuild')")
    conn.commit()
    logger.info("Footage index cleared")


def remove_missing_files():
    """Remove index entries for files that no longer exist on disk.

    Returns:
        int: number of entries removed
    """
    conn = _get_conn()
    init_db()

    rows = conn.execute("SELECT id, file_path FROM footage").fetchall()
    to_remove = []
    for row in rows:
        if not os.path.isfile(row["file_path"]):
            to_remove.append(row["id"])

    if to_remove:
        placeholders = ",".join("?" * len(to_remove))
        conn.execute(f"DELETE FROM footage WHERE id IN ({placeholders})", to_remove)
        conn.commit()
        logger.info("Removed %d missing files from footage index", len(to_remove))

    return len(to_remove)


def get_all_indexed_files():
    """Return list of all indexed file paths."""
    conn = _get_conn()
    init_db()
    rows = conn.execute("SELECT file_path FROM footage ORDER BY indexed_at DESC").fetchall()
    return [row["file_path"] for row in rows]
