"""Regression tests for local SQLite maintenance diagnostics."""

from __future__ import annotations


def _isolate_store_paths(monkeypatch, tmp_path):
    import opencut.job_store as job_store
    from opencut import journal
    from opencut.core import footage_index_db, pipeline_health

    monkeypatch.setattr(job_store, "_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(journal, "_DB_PATH", str(tmp_path / "journal.db"))
    monkeypatch.setattr(footage_index_db, "_DB_PATH", str(tmp_path / "footage_index.db"))
    monkeypatch.setattr(pipeline_health, "_DB_PATH", str(tmp_path / "pipeline_health.db"))
    return job_store, journal, footage_index_db, pipeline_health


def test_collect_local_db_diagnostics_reports_all_stores(monkeypatch, tmp_path):
    from opencut.local_db_diagnostics import collect_local_db_diagnostics

    _isolate_store_paths(monkeypatch, tmp_path)

    stores = collect_local_db_diagnostics()

    assert [item["store"] for item in stores] == [
        "jobs",
        "journal",
        "footage_index",
        "pipeline_health",
    ]
    assert all("recommended_action" in item for item in stores)


def test_store_diagnostic_accessors_use_store_names(monkeypatch, tmp_path):
    job_store, journal, footage_index_db, pipeline_health = _isolate_store_paths(
        monkeypatch, tmp_path
    )

    assert job_store.get_db_diagnostics()["store"] == "jobs"
    assert journal.get_db_diagnostics()["store"] == "journal"
    assert footage_index_db.get_db_diagnostics()["store"] == "footage_index"
    assert pipeline_health.get_db_diagnostics()["store"] == "pipeline_health"


def test_local_db_diagnostic_routes(client, monkeypatch, tmp_path):
    _isolate_store_paths(monkeypatch, tmp_path)

    routes = {
        "/jobs/db-diagnostics": "jobs",
        "/journal/db-diagnostics": "journal",
        "/search/db-diagnostics": "footage_index",
        "/api/pipeline/health/db-diagnostics": "pipeline_health",
    }
    for route, store_name in routes.items():
        resp = client.get(route)
        assert resp.status_code == 200, route
        payload = resp.get_json()
        assert payload["store"] == store_name
        assert "page_count" in payload
        assert "freelist_count" in payload
        assert "wal_checkpoint" in payload
        assert "recommended_action" in payload
