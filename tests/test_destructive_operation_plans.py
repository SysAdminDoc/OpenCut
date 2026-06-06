"""Regression tests for shared destructive-operation dry-run plans."""

from __future__ import annotations

import json
import os
import time


def _headers(token: str) -> dict[str, str]:
    return {"X-OpenCut-Token": token}


def _isolate_render_cache(monkeypatch, tmp_path, entries):
    import opencut.core.render_cache as render_cache

    cache_dir = tmp_path / "render_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(render_cache, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(render_cache, "CACHE_INDEX", str(cache_dir / "index.json"))
    index = {}
    for key, data in entries.items():
        cache_file = cache_dir / f"{key}.bin"
        cache_file.write_bytes(data.pop("content", b"cache"))
        index[key] = {
            "input_hash": data.get("input_hash", "input"),
            "operation": data.get("operation", "encode"),
            "params_hash": data.get("params_hash", "params"),
            "output_path": str(cache_file),
            "file_size": data.get("file_size", len(cache_file.read_bytes())),
            "created_at": data.get("created_at", 1),
            "last_accessed": data.get("last_accessed", 1),
            "hit_count": data.get("hit_count", 0),
            "dependencies": list(data.get("dependencies", [])),
        }
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return cache_dir


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


def test_render_cache_cleanup_requires_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    cache_dir = _isolate_render_cache(
        monkeypatch,
        tmp_path,
        {
            "cleanupold": {
                "content": b"old-cache",
                "file_size": 200 * 1024 * 1024,
                "last_accessed": 1,
            },
        },
    )
    cache_file = cache_dir / "cleanupold.bin"

    preview = client.post(
        "/cache/cleanup",
        json={"max_size_gb": 0.1, "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert preview_data["would_remove"] == 1
    assert preview_data["confirm_token"]
    assert cache_file.exists()

    missing_token = client.post(
        "/cache/cleanup",
        json={"max_size_gb": 0.1},
        headers=_headers(csrf_token),
    )
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert cache_file.exists()

    confirmed = client.post(
        "/cache/cleanup",
        json={"max_size_gb": 0.1, "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert confirmed.get_json()["removed"] == 1
    assert not cache_file.exists()


def test_render_cache_invalidate_requires_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    cache_dir = _isolate_render_cache(
        monkeypatch,
        tmp_path,
        {
            "seedcache": {
                "content": b"seed",
                "input_hash": "abc",
                "operation": "encode",
                "file_size": 4,
            },
            "childcache": {
                "content": b"child",
                "input_hash": "child",
                "operation": "filter",
                "file_size": 5,
                "dependencies": ["seedcache"],
            },
        },
    )
    seed_file = cache_dir / "seedcache.bin"
    child_file = cache_dir / "childcache.bin"

    preview = client.post(
        "/cache/invalidate",
        json={"input_hash": "abc", "operation": "encode", "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert preview_data["would_invalidate"] == 2
    assert preview_data["confirm_token"]

    missing_token = client.post(
        "/cache/invalidate",
        json={"input_hash": "abc", "operation": "encode"},
        headers=_headers(csrf_token),
    )
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert seed_file.exists()
    assert child_file.exists()

    confirmed = client.post(
        "/cache/invalidate",
        json={
            "input_hash": "abc",
            "operation": "encode",
            "confirm_token": preview_data["confirm_token"],
        },
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert confirmed.get_json()["invalidated"] == 2
    assert not seed_file.exists()
    assert not child_file.exists()


def test_temp_cleanup_sweep_requires_confirm_token(client, csrf_token, monkeypatch, tmp_path):
    from opencut.core import temp_cleanup

    old_file = tmp_path / "opencut_old.txt"
    old_dir = tmp_path / "opencut_old_dir"
    old_file.write_text("old", encoding="utf-8")
    old_dir.mkdir()
    (old_dir / "payload.txt").write_text("old-dir", encoding="utf-8")
    old_time = time.time() - 3600
    os.utime(old_file, (old_time, old_time))
    os.utime(old_dir, (old_time, old_time))
    monkeypatch.setattr(temp_cleanup.tempfile, "gettempdir", lambda: str(tmp_path))

    preview = client.post(
        "/system/temp-cleanup/sweep",
        json={"ttl_seconds": 60, "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert len(preview_data["targets"]) == 2
    assert preview_data["confirm_token"]
    assert old_file.exists()
    assert old_dir.exists()

    missing_token = client.post(
        "/system/temp-cleanup/sweep",
        json={"ttl_seconds": 60},
        headers=_headers(csrf_token),
    )
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert old_file.exists()
    assert old_dir.exists()

    confirmed = client.post(
        "/system/temp-cleanup/sweep",
        json={"ttl_seconds": 60, "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert confirmed.get_json()["bytes_reclaimed"] > 0
    assert not old_file.exists()
    assert not old_dir.exists()


def test_chat_clear_requires_confirm_token_when_session_has_history(client, csrf_token):
    from opencut.core import chat_editor

    session = chat_editor.get_or_create_session("confirm-chat")
    session.history[:] = [chat_editor.ChatMessage(role="user", content="hello")]

    preview = client.post(
        "/chat/clear",
        json={"session_id": "confirm-chat", "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["would_clear"] == 1

    missing_token = client.post(
        "/chat/clear",
        json={"session_id": "confirm-chat"},
        headers=_headers(csrf_token),
    )
    assert missing_token.status_code == 409
    assert any(
        item["session_id"] == "confirm-chat" and item["message_count"] == 1
        for item in chat_editor.list_sessions()
    )

    confirmed = client.post(
        "/chat/clear",
        json={"session_id": "confirm-chat", "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert all(item["session_id"] != "confirm-chat" for item in chat_editor.list_sessions())


def test_search_cleanup_requires_confirm_token_for_missing_files(client, csrf_token, monkeypatch, tmp_path):
    from opencut.core import footage_index_db

    footage_index_db.close_all_connections()
    monkeypatch.setattr(footage_index_db, "_DB_PATH", str(tmp_path / "footage_index.db"))
    footage_index_db.init_db()
    missing_file = tmp_path / "missing.mp4"
    footage_index_db.index_file(str(missing_file), "missing transcript")

    preview = client.post(
        "/search/cleanup",
        json={"dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["would_remove"] == 1
    assert footage_index_db.get_stats()["total_files"] == 1

    missing_token = client.post("/search/cleanup", json={}, headers=_headers(csrf_token))
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

    confirmed = client.post(
        "/search/cleanup",
        json={"confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert confirmed.get_json()["removed"] == 1
    assert footage_index_db.get_stats()["total_files"] == 0


def test_chat_clear_requires_confirm_token(client, csrf_token):
    from opencut.core.chat_editor import get_or_create_session, list_sessions

    session = get_or_create_session("destructive-chat")
    session.add_message("user", "hello")

    preview = client.post(
        "/chat/clear",
        json={"session_id": "destructive-chat", "dry_run": True},
        headers=_headers(csrf_token),
    )
    preview_data = preview.get_json()

    assert preview.status_code == 200
    assert preview_data["dry_run"] is True
    assert preview_data["would_clear"] == 1
    assert preview_data["confirm_token"]

    missing_token = client.post(
        "/chat/clear",
        json={"session_id": "destructive-chat"},
        headers=_headers(csrf_token),
    )
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert any(item["session_id"] == "destructive-chat" for item in list_sessions())

    confirmed = client.post(
        "/chat/clear",
        json={"session_id": "destructive-chat", "confirm_token": preview_data["confirm_token"]},
        headers=_headers(csrf_token),
    )

    assert confirmed.status_code == 200
    assert not any(item["session_id"] == "destructive-chat" for item in list_sessions())
