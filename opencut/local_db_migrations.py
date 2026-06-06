"""Small helpers for versioned local SQLite databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping

Migration = Callable[[sqlite3.Connection], None]


class LocalDatabaseVersionError(RuntimeError):
    """Raised when a local SQLite database has an unsupported schema version."""


def get_user_version(conn: sqlite3.Connection) -> int:
    """Return SQLite's application-owned user_version value."""
    row = conn.execute("PRAGMA user_version").fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _coerce_schema_version(version: int) -> int:
    coerced = int(version)
    if coerced < 0:
        raise ValueError("schema versions must be non-negative")
    return coerced


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    coerced = _coerce_schema_version(version)
    conn.execute(f"PRAGMA user_version = {coerced}")


def migrate_user_version(
    conn: sqlite3.Connection,
    *,
    store_name: str,
    target_version: int,
    migrations: Mapping[int, Migration],
) -> int:
    """Run ordered idempotent migrations up to *target_version*.

    Each migration owns exactly one user_version step. The helper records the
    step only after the callback succeeds, so a failed migration is retried on
    the next initialization attempt.
    """
    target = _coerce_schema_version(target_version)
    current = get_user_version(conn)

    if current > target:
        raise LocalDatabaseVersionError(
            f"{store_name} database user_version {current} is newer than "
            f"supported schema {target}; refusing to downgrade or open an "
            "unknown local database schema."
        )

    savepoint = "opencut_schema_migration"
    owns_transaction = not conn.in_transaction
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        for next_version in range(current + 1, target + 1):
            migration = migrations.get(next_version)
            if migration is None:
                raise LocalDatabaseVersionError(
                    f"{store_name} database is missing migration for "
                    f"user_version {next_version}."
            )
            migration(conn)
            _set_user_version(conn, next_version)
        conn.execute(f"RELEASE {savepoint}")
        if owns_transaction:
            conn.commit()
    except Exception:
        try:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        except sqlite3.Error:
            if owns_transaction:
                conn.rollback()
        raise

    return target
