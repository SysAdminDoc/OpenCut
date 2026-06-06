"""Regression tests for local SQLite schema versioning."""

import sqlite3
import threading

import pytest

from opencut.local_db_diagnostics import build_sqlite_diagnostic
from opencut.local_db_migrations import (
    LocalDatabaseVersionError,
    get_user_version,
    migrate_user_version,
)


def _read_user_version(path) -> int:
    with sqlite3.connect(path) as conn:
        return get_user_version(conn)


def _set_user_version(path, version: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {int(version)}")


def test_migration_helper_runs_ordered_steps_once():
    conn = sqlite3.connect(":memory:")
    calls = []

    migrate_user_version(
        conn,
        store_name="test store",
        target_version=2,
        migrations={
            1: lambda _conn: calls.append(1),
            2: lambda _conn: calls.append(2),
        },
    )
    migrate_user_version(
        conn,
        store_name="test store",
        target_version=2,
        migrations={
            1: lambda _conn: calls.append("replayed-1"),
            2: lambda _conn: calls.append("replayed-2"),
        },
    )

    assert calls == [1, 2]
    assert get_user_version(conn) == 2


def test_migration_helper_rolls_back_failed_step():
    conn = sqlite3.connect(":memory:")

    def fail_after_table(conn):
        conn.execute("CREATE TABLE partial(id INTEGER PRIMARY KEY)")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        migrate_user_version(
            conn,
            store_name="test store",
            target_version=1,
            migrations={1: fail_after_table},
        )

    assert get_user_version(conn) == 0
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'partial'"
    ).fetchone() is None


def test_migration_helper_rejects_newer_unknown_schema():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA user_version = 9")

    with pytest.raises(LocalDatabaseVersionError, match="refusing to downgrade"):
        migrate_user_version(
            conn,
            store_name="future store",
            target_version=2,
            migrations={1: lambda _conn: None, 2: lambda _conn: None},
        )


def test_job_store_sets_user_version_and_rejects_future_schema(tmp_path, monkeypatch):
    import opencut.job_store as store

    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(store, "_DB_PATH", str(db_path))
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None
    store._LOCAL = threading.local()
    store._ALL_CONNECTIONS = {}

    store.init_db()

    assert _read_user_version(db_path) == store.SCHEMA_VERSION

    store.close_all_connections()
    _set_user_version(db_path, store.SCHEMA_VERSION + 1)
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None
    store._LOCAL = threading.local()

    with pytest.raises(LocalDatabaseVersionError, match="job store.*refusing to downgrade"):
        store.init_db()


def test_journal_sets_user_version_and_rejects_future_schema(tmp_path, monkeypatch):
    from opencut import journal as journal_module

    db_path = tmp_path / "journal.db"
    monkeypatch.setattr(journal_module, "_DB_PATH", str(db_path))
    journal_module.close_all_connections()
    journal_module._thread_local = threading.local()

    journal_module.init_db()

    assert _read_user_version(db_path) == journal_module.SCHEMA_VERSION

    journal_module.close_all_connections()
    _set_user_version(db_path, journal_module.SCHEMA_VERSION + 1)
    journal_module._thread_local = threading.local()

    with pytest.raises(LocalDatabaseVersionError, match="journal.*refusing to downgrade"):
        journal_module.init_db()


def test_footage_index_sets_user_version_and_rejects_future_schema(tmp_path, monkeypatch):
    import opencut.core.footage_index_db as footage_index

    db_path = tmp_path / "footage.db"
    monkeypatch.setattr(footage_index, "_DB_PATH", str(db_path))
    footage_index.close_all_connections()
    footage_index._thread_local = threading.local()

    footage_index.init_db()

    assert _read_user_version(db_path) == footage_index.SCHEMA_VERSION

    footage_index.close_all_connections()
    _set_user_version(db_path, footage_index.SCHEMA_VERSION + 1)
    footage_index._thread_local = threading.local()

    with pytest.raises(LocalDatabaseVersionError, match="footage index.*refusing to downgrade"):
        footage_index.init_db()


def test_pipeline_health_sets_user_version_and_rejects_future_schema(tmp_path, monkeypatch):
    import opencut.core.pipeline_health as pipeline_health

    db_path = tmp_path / "pipeline_health.db"
    monkeypatch.setattr(pipeline_health, "_DB_PATH", str(db_path))
    pipeline_health._INITIALIZED = False
    pipeline_health._LOCAL = threading.local()

    pipeline_health._init_db()

    assert _read_user_version(db_path) == pipeline_health.SCHEMA_VERSION

    conn = pipeline_health._get_conn()
    conn.execute(f"PRAGMA user_version = {pipeline_health.SCHEMA_VERSION + 1}")
    conn.commit()
    pipeline_health._INITIALIZED = False

    with pytest.raises(LocalDatabaseVersionError, match="pipeline health.*refusing to downgrade"):
        pipeline_health._init_db()


def test_sqlite_diagnostic_reports_pages_freelist_and_wal(tmp_path):
    db_path = tmp_path / "diag.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA user_version = 3")
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO items (value) VALUES (?)", [(str(i),) for i in range(5)])
        conn.commit()
    finally:
        conn.close()

    diagnostic = build_sqlite_diagnostic(str(db_path), store_name="test_store")

    assert diagnostic["store"] == "test_store"
    assert diagnostic["exists"] is True
    assert diagnostic["page_count"] >= 1
    assert diagnostic["page_size"] > 0
    assert diagnostic["freelist_count"] >= 0
    assert diagnostic["user_version"] == 3
    assert diagnostic["files"]["database"]["bytes"] > 0
    assert "wal_checkpoint" in diagnostic
    assert diagnostic["recommended_action"] in {
        "ok",
        "checkpoint_recommended",
        "vacuum_recommended",
    }


def test_sqlite_diagnostic_reports_missing_database(tmp_path):
    diagnostic = build_sqlite_diagnostic(str(tmp_path / "missing.db"), store_name="missing")

    assert diagnostic["exists"] is False
    assert diagnostic["recommended_action"] == "not_initialized"
    assert diagnostic["wal_checkpoint"]["error"] == "database_not_initialized"
