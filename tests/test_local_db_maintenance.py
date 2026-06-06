"""Regression tests for local SQLite destructive-maintenance safeguards."""

from __future__ import annotations

import sqlite3
import threading
import time


def _reset_job_store(monkeypatch, db_path):
    import opencut.job_store as store

    store.close_all_connections()
    monkeypatch.setattr(store, "_DB_PATH", str(db_path))
    monkeypatch.setattr(store, "_LOCAL", threading.local())
    store._INITIALIZED = False
    store._INITIALIZED_PATH = None
    return store


def _reset_journal(monkeypatch, db_path):
    from opencut import journal

    journal.close_all_connections()
    monkeypatch.setattr(journal, "_DB_PATH", str(db_path))
    monkeypatch.setattr(journal, "_thread_local", threading.local())
    journal._ALL_CONNECTIONS.clear()
    return journal


def _reset_footage_index(monkeypatch, db_path):
    import opencut.core.footage_index_db as index

    index.close_all_connections()
    monkeypatch.setattr(index, "_DB_PATH", str(db_path))
    monkeypatch.setattr(index, "_thread_local", threading.local())
    index._ALL_CONNECTIONS.clear()
    return index


def _reset_pipeline_health(monkeypatch, db_path):
    import opencut.core.pipeline_health as health

    monkeypatch.setattr(health, "_DB_PATH", str(db_path))
    monkeypatch.setattr(health, "_LOCAL", threading.local())
    health._INITIALIZED = False
    return health


def _csrf_headers(token):
    return {"X-OpenCut-Token": token, "Content-Type": "application/json"}


def test_create_sqlite_backup_uses_vacuum_into(tmp_path):
    from opencut.local_db_maintenance import create_sqlite_backup

    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO items (value) VALUES ('kept')")

    backup = create_sqlite_backup(str(db_path), store_name="test", operation="clear")

    assert backup is not None
    assert backup["method"] == "vacuum_into"
    with sqlite3.connect(backup["path"]) as conn:
        row = conn.execute("SELECT value FROM items").fetchone()
    assert row[0] == "kept"


def test_journal_clear_dry_run_and_backup(monkeypatch, tmp_path):
    journal = _reset_journal(monkeypatch, tmp_path / "journal.db")
    journal.record("add_markers", "one", {})
    journal.record("batch_rename", "two", {})

    dry_run = journal.clear_all(dry_run=True)

    assert dry_run["dry_run"] is True
    assert dry_run["affected_rows"] == 2
    assert len(journal.list_entries()) == 2

    result = journal.clear_all(backup=True)

    assert result["affected_rows"] == 2
    assert result["backup"]["bytes"] > 0
    assert result["audit"]["entry"]["operation"] == "clear_all"
    assert journal.list_entries() == []


def test_job_cleanup_dry_run_preserves_rows(monkeypatch, tmp_path):
    store = _reset_job_store(monkeypatch, tmp_path / "jobs.db")
    old_time = time.time() - store.COMPLETED_JOB_TTL - 60
    store.save_job({
        "id": "old-job",
        "type": "test",
        "status": "complete",
        "created": old_time,
        "completed_at": old_time,
    })

    dry_run = store.cleanup_old_jobs(dry_run=True)

    assert dry_run["affected_rows"] == 1
    assert store.get_job("old-job") is not None


def test_job_cleanup_backup_captures_deleted_rows(monkeypatch, tmp_path):
    store = _reset_job_store(monkeypatch, tmp_path / "jobs_backup.db")
    old_time = time.time() - store.COMPLETED_JOB_TTL - 60
    store.save_job({
        "id": "old-job",
        "type": "test",
        "status": "complete",
        "created": old_time,
        "completed_at": old_time,
    })

    result = store.cleanup_old_jobs(backup=True)

    assert result["affected_rows"] == 1
    assert result["audit"]["entry"]["operation"] == "cleanup_old_jobs"
    assert store.get_job("old-job") is None
    with sqlite3.connect(result["backup"]["path"]) as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = 'old-job'").fetchone()
    assert row[0] == "old-job"


def test_footage_index_clear_dry_run_and_backup(monkeypatch, tmp_path):
    index = _reset_footage_index(monkeypatch, tmp_path / "footage.db")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    index.index_file(str(media), "hello world")

    dry_run = index.clear_index(dry_run=True)

    assert dry_run["affected_rows"] == 1
    assert index.get_stats()["total_files"] == 1

    result = index.clear_index(backup=True)

    assert result["affected_rows"] == 1
    assert result["backup"]["bytes"] > 0
    assert index.get_stats()["total_files"] == 0


def test_pipeline_health_reset_dry_run_and_backup(monkeypatch, tmp_path):
    health = _reset_pipeline_health(monkeypatch, tmp_path / "pipeline.db")
    health.record_metric("trim", 1.0, True)

    dry_run = health.reset_health_db(dry_run=True)

    assert dry_run["affected_rows"] == 1
    assert health.get_pipeline_health(timeframe_hours=1).total_jobs == 1

    result = health.reset_health_db(backup=True)

    assert result["affected_rows"] == 1
    assert result["backup"]["bytes"] > 0
    assert health.get_pipeline_health(timeframe_hours=1).total_jobs == 0


def test_pipeline_health_purge_backup_captures_old_metrics(monkeypatch, tmp_path):
    health = _reset_pipeline_health(monkeypatch, tmp_path / "pipeline_purge.db")
    health.record_metric("transcode", 2.5, True)
    old_timestamp = time.time() - (120 * 86400)
    conn = health._get_conn()
    conn.execute(
        "UPDATE health_metrics SET timestamp = ? WHERE operation = ?",
        (old_timestamp, "transcode"),
    )
    conn.commit()

    result = health.purge_old_metrics(days=90, backup=True)

    assert result["affected_rows"] == 1
    assert result["days"] == 90
    assert result["audit"]["entry"]["operation"] == "purge_old_metrics"
    assert health.get_pipeline_health(timeframe_hours=24 * 365).total_jobs == 0
    with sqlite3.connect(result["backup"]["path"]) as backup_conn:
        row = backup_conn.execute(
            "SELECT operation FROM health_metrics WHERE operation = ?",
            ("transcode",),
        ).fetchone()
    assert row[0] == "transcode"


def test_destructive_maintenance_routes_dry_run(client, csrf_token, monkeypatch, tmp_path):
    journal = _reset_journal(monkeypatch, tmp_path / "route_journal.db")
    _reset_footage_index(monkeypatch, tmp_path / "route_footage.db")
    _reset_pipeline_health(monkeypatch, tmp_path / "route_pipeline.db")
    entry = journal.record("add_markers", "route", {})
    headers = _csrf_headers(csrf_token)

    journal_resp = client.delete(
        f"/journal/{entry['id']}?dry_run=1",
        headers=headers,
    )
    search_resp = client.delete(
        "/search/db-index?dry_run=1",
        headers=headers,
    )
    pipeline_resp = client.post(
        "/api/pipeline/health/reset",
        json={"dry_run": True},
        headers={"X-OpenCut-Token": csrf_token},
    )

    assert journal_resp.status_code == 200
    assert journal_resp.get_json()["dry_run"] is True
    assert search_resp.status_code == 200
    assert search_resp.get_json()["operation"] == "clear_index"
    assert pipeline_resp.status_code == 200
    assert pipeline_resp.get_json()["operation"] == "reset_health_db"
