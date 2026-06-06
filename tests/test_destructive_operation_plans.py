"""Regression tests for shared destructive-operation dry-run plans."""

from __future__ import annotations


def _headers(token: str) -> dict[str, str]:
    return {"X-OpenCut-Token": token}


def test_queue_clear_requires_confirm_token_after_dry_run(client, csrf_token):
    from opencut.routes import jobs_routes

    with jobs_routes.job_queue_lock:
        jobs_routes.job_queue[:] = [
            {"id": "queued-a", "endpoint": "/silence", "status": "queued", "payload": {}},
            {"id": "running-a", "endpoint": "/captions", "status": "running", "payload": {}},
        ]

    preview = client.post(
        "/queue/clear",
        json={"dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert preview_data["removed"] == 0
    assert preview_data["plan"]["metadata"]["queued_count"] == 1
    assert preview_data["plan"]["confirm_token"]

    missing_token = client.post("/queue/clear", json={}, headers=_headers(csrf_token))
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    with jobs_routes.job_queue_lock:
        assert [entry["id"] for entry in jobs_routes.job_queue] == ["queued-a", "running-a"]

    confirmed = client.post(
        "/queue/clear",
        json={"confirm_token": preview_data["plan"]["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert confirmed.get_json()["removed"] == 1
    with jobs_routes.job_queue_lock:
        assert [entry["id"] for entry in jobs_routes.job_queue] == ["running-a"]


def test_logs_clear_requires_confirm_token_after_dry_run(client, csrf_token, monkeypatch, tmp_path):
    import opencut.routes.settings as settings_routes
    import opencut.server as server

    opencut_dir = tmp_path / "opencut"
    opencut_dir.mkdir()
    crash_log = opencut_dir / "crash.log"
    server_log = tmp_path / "server.log"
    crash_log.write_text("crash", encoding="utf-8")
    server_log.write_text("server-log", encoding="utf-8")
    monkeypatch.setattr(settings_routes, "OPENCUT_DIR", str(opencut_dir))
    monkeypatch.setattr(server, "LOG_FILE", str(server_log))

    preview = client.post(
        "/logs/clear",
        json={"dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert preview_data["total_bytes"] == len("crash") + len("server-log")
    assert preview_data["cleared"] == []
    assert preview_data["plan"]["confirm_token"]
    assert crash_log.read_text(encoding="utf-8") == "crash"

    missing_token = client.post("/logs/clear", json={}, headers=_headers(csrf_token))
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert server_log.read_text(encoding="utf-8") == "server-log"

    confirmed = client.post(
        "/logs/clear",
        json={"confirm_token": preview_data["plan"]["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert set(confirmed.get_json()["cleared"]) == {"crash.log", "server.log"}
    assert crash_log.read_text(encoding="utf-8") == ""
    assert server_log.read_text(encoding="utf-8") == ""
