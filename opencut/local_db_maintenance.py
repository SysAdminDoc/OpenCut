"""Backup and audit helpers for destructive local SQLite maintenance."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from typing import Any

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_store_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip() or "store").strip("-")


def _default_backup_dir(db_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "backups")


def _default_audit_path(db_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "local_db_maintenance_audit.jsonl")


def count_rows(conn: sqlite3.Connection, table_name: str, where_clause: str = "",
               params: tuple[Any, ...] = ()) -> int:
    """Count rows in an internal table with an optional fixed WHERE clause."""
    if not _IDENTIFIER_RE.match(table_name):
        raise ValueError("invalid table name")
    statement = f"SELECT COUNT(*) FROM {table_name}"
    if where_clause:
        statement += f" WHERE {where_clause}"
    row = conn.execute(statement, params).fetchone()
    return int(row[0] or 0) if row else 0


def create_sqlite_backup(
    db_path: str,
    *,
    store_name: str,
    operation: str,
    backup_dir: str | None = None,
) -> dict[str, Any] | None:
    """Create a compact SQLite backup with ``VACUUM INTO``.

    Returns ``None`` when the database file does not exist yet.
    """
    source = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.exists(source):
        return None
    target_dir = os.path.abspath(os.path.expanduser(backup_dir or _default_backup_dir(source)))
    os.makedirs(target_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    filename = (
        f"{_safe_store_name(store_name)}-{_safe_store_name(operation)}-"
        f"{stamp}-{uuid.uuid4().hex[:8]}.db"
    )
    target = os.path.join(target_dir, filename)
    with sqlite3.connect(source, timeout=10) as conn:
        conn.execute("VACUUM INTO ?", (target,))
    return {
        "path": target,
        "bytes": os.path.getsize(target),
        "created_at": time.time(),
        "method": "vacuum_into",
    }


def write_maintenance_audit(
    db_path: str,
    *,
    store_name: str,
    operation: str,
    affected_rows: int,
    dry_run: bool,
    backup: dict[str, Any] | None,
    audit_path: str | None = None,
) -> dict[str, Any]:
    """Append one JSONL audit record outside the database being changed."""
    target = os.path.abspath(os.path.expanduser(audit_path or _default_audit_path(db_path)))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    entry = {
        "id": uuid.uuid4().hex,
        "created_at": time.time(),
        "store": store_name,
        "operation": operation,
        "db_path": os.path.abspath(os.path.expanduser(db_path)),
        "affected_rows": int(max(0, affected_rows)),
        "dry_run": bool(dry_run),
        "backup": backup,
    }
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return {"path": target, "entry": entry}


def prepare_destructive_result(
    db_path: str,
    *,
    store_name: str,
    operation: str,
    affected_rows: int,
    dry_run: bool = False,
    backup: bool = False,
    backup_dir: str | None = None,
) -> dict[str, Any]:
    """Build common dry-run/backup/audit metadata for destructive operations."""
    planned_rows = int(max(0, affected_rows))
    backup_info = None
    audit_info = None
    if not dry_run and backup and planned_rows > 0:
        backup_info = create_sqlite_backup(
            db_path,
            store_name=store_name,
            operation=operation,
            backup_dir=backup_dir,
        )
    if not dry_run and planned_rows > 0:
        audit_info = write_maintenance_audit(
            db_path,
            store_name=store_name,
            operation=operation,
            affected_rows=planned_rows,
            dry_run=False,
            backup=backup_info,
        )
    return {
        "ok": True,
        "store": store_name,
        "operation": operation,
        "dry_run": bool(dry_run),
        "affected_rows": planned_rows,
        "backup": backup_info,
        "audit": audit_info,
    }
