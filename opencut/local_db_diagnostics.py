"""Read-only diagnostics for OpenCut's local SQLite stores."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

_VACUUM_FREELIST_RATIO = 0.20
_WAL_SIZE_CHECKPOINT_BYTES = 32 * 1024 * 1024


def _file_info(path: str) -> dict[str, Any]:
    try:
        stat = os.stat(path)
    except OSError:
        return {"path": path, "exists": False, "bytes": 0}
    return {
        "path": path,
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def _pragma_int(conn: sqlite3.Connection, statement: str) -> int:
    row = conn.execute(statement).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _wal_checkpoint_status(*, journal_mode: str, wal_exists: bool, wal_bytes: int) -> dict[str, Any]:
    mode = str(journal_mode or "").lower()
    if mode != "wal":
        return {"ok": True, "status": "not_wal", "checkpoint_needed": False}
    if not wal_exists or wal_bytes <= 0:
        return {"ok": True, "status": "no_wal_file", "checkpoint_needed": False}
    checkpoint_needed = wal_bytes >= _WAL_SIZE_CHECKPOINT_BYTES
    return {
        "ok": True,
        "status": "checkpoint_recommended" if checkpoint_needed else "wal_file_present",
        "checkpoint_needed": checkpoint_needed,
        "wal_bytes": wal_bytes,
        "probe": "file_size_only",
    }


def _recommended_action(*, exists: bool, freelist_count: int, page_count: int, wal_bytes: int) -> str:
    if not exists:
        return "not_initialized"
    if page_count > 0 and freelist_count / page_count >= _VACUUM_FREELIST_RATIO:
        return "vacuum_recommended"
    if wal_bytes >= _WAL_SIZE_CHECKPOINT_BYTES:
        return "checkpoint_recommended"
    return "ok"


def build_sqlite_diagnostic(path: str, *, store_name: str) -> dict[str, Any]:
    """Return page, freelist, WAL, and file-size diagnostics for one SQLite DB."""
    db_path = os.path.abspath(path)
    wal_path = f"{db_path}-wal"
    shm_path = f"{db_path}-shm"
    files = {
        "database": _file_info(db_path),
        "wal": _file_info(wal_path),
        "shm": _file_info(shm_path),
    }
    exists = bool(files["database"]["exists"])
    if not exists:
        return {
            "store": store_name,
            "path": db_path,
            "exists": False,
            "files": files,
            "page_count": 0,
            "page_size": 0,
            "freelist_count": 0,
            "freelist_ratio": 0.0,
            "estimated_free_bytes": 0,
            "journal_mode": "",
            "user_version": 0,
            "wal_checkpoint": {"ok": False, "error": "database_not_initialized"},
            "recommended_action": "not_initialized",
        }

    try:
        conn = sqlite3.connect(db_path, timeout=2)
        try:
            page_count = _pragma_int(conn, "PRAGMA page_count")
            page_size = _pragma_int(conn, "PRAGMA page_size")
            freelist_count = _pragma_int(conn, "PRAGMA freelist_count")
            user_version = _pragma_int(conn, "PRAGMA user_version")
            journal_row = conn.execute("PRAGMA journal_mode").fetchone()
            journal_mode = str(journal_row[0] or "") if journal_row else ""
            wal_info = files["wal"]
            wal_checkpoint = _wal_checkpoint_status(
                journal_mode=journal_mode,
                wal_exists=bool(wal_info["exists"]),
                wal_bytes=int(wal_info["bytes"]),
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {
            "store": store_name,
            "path": db_path,
            "exists": True,
            "files": files,
            "page_count": 0,
            "page_size": 0,
            "freelist_count": 0,
            "freelist_ratio": 0.0,
            "estimated_free_bytes": 0,
            "journal_mode": "",
            "user_version": 0,
            "wal_checkpoint": {"ok": False, "error": str(exc)},
            "recommended_action": "inspect_error",
        }

    freelist_ratio = (freelist_count / page_count) if page_count else 0.0
    return {
        "store": store_name,
        "path": db_path,
        "exists": True,
        "files": files,
        "page_count": page_count,
        "page_size": page_size,
        "freelist_count": freelist_count,
        "freelist_ratio": round(freelist_ratio, 6),
        "estimated_free_bytes": int(freelist_count * page_size),
        "journal_mode": journal_mode,
        "user_version": user_version,
        "wal_checkpoint": wal_checkpoint,
        "recommended_action": _recommended_action(
            exists=True,
            freelist_count=freelist_count,
            page_count=page_count,
            wal_bytes=int(files["wal"]["bytes"]),
        ),
    }


def collect_local_db_diagnostics() -> list[dict[str, Any]]:
    """Collect diagnostics for the local SQLite stores OpenCut owns."""
    from opencut import job_store, journal
    from opencut.core import footage_index_db, pipeline_health

    stores = (
        ("jobs", job_store._DB_PATH),
        ("journal", journal._DB_PATH),
        ("footage_index", footage_index_db._DB_PATH),
        ("pipeline_health", pipeline_health._DB_PATH),
    )
    return [
        build_sqlite_diagnostic(path, store_name=store_name)
        for store_name, path in stores
    ]
