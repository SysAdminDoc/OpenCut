"""Durable legacy job-queue recovery and interchange tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

from flask import current_app, jsonify

from tests.conftest import csrf_headers


def _entry(queue_id, status="queued", *, endpoint="/silence", payload=None, added=1):
    return {
        "id": queue_id,
        "endpoint": endpoint,
        "payload": payload or {},
        "status": status,
        "added": added,
    }


def _use_temp_queue_store(monkeypatch, tmp_path):
    from opencut import user_data
    from opencut.routes import jobs_routes

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    with jobs_routes.job_queue_lock:
        jobs_routes.job_queue.clear()
        jobs_routes._queue_state["running"] = False
        jobs_routes._queue_persistence_enabled = False
        jobs_routes._queue_app = None
        jobs_routes._queue_storage_error = None
    return jobs_routes


def test_forced_restart_preserves_order_and_marks_active_entries_interrupted(
    app, monkeypatch, tmp_path
):
    from opencut.queue_store import load_queue, save_queue

    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    save_queue([
        _entry("queued-a", "queued", added=10),
        _entry("running-b", "running", added=20),
        _entry("started-c", "started", added=30),
    ])

    # A fresh in-memory list simulates the new process after a forced stop.
    with jobs_routes.job_queue_lock:
        jobs_routes.job_queue.clear()
    result = jobs_routes.initialize_job_queue(app, start_processing=False)

    assert result == {"loaded": 3, "interrupted": 2, "invalid": 0}
    with jobs_routes.job_queue_lock:
        snapshot = [dict(entry) for entry in jobs_routes.job_queue]
    assert [entry["id"] for entry in snapshot] == ["queued-a", "running-b", "started-c"]
    assert [entry["status"] for entry in snapshot] == [
        "queued", "interrupted", "interrupted",
    ]
    assert snapshot[1]["code"] == "SERVER_RESTARTED"
    assert snapshot[2]["code"] == "SERVER_RESTARTED"

    persisted, migrated = load_queue()
    assert migrated is False
    assert [entry["status"] for entry in persisted] == [
        "queued", "interrupted", "interrupted",
    ]


def test_legacy_queue_list_migrates_to_versioned_document(app, monkeypatch, tmp_path):
    from opencut import user_data
    from opencut.queue_store import QUEUE_FILE, QUEUE_SCHEMA_VERSION

    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    user_data.write_user_file(QUEUE_FILE, [_entry("legacy-entry")])

    result = jobs_routes.initialize_job_queue(app, start_processing=False)

    assert result == {"loaded": 1, "interrupted": 0, "invalid": 0}
    migrated = user_data.read_user_file(QUEUE_FILE)
    assert migrated["schema_version"] == QUEUE_SCHEMA_VERSION
    assert [entry["id"] for entry in migrated["entries"]] == ["legacy-entry"]


def test_newer_queue_schema_fails_closed_instead_of_accepting_ephemeral_work(
    app, client, csrf_token, monkeypatch, tmp_path
):
    from opencut import user_data
    from opencut.queue_store import QUEUE_FILE

    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    user_data.write_user_file(QUEUE_FILE, {"schema_version": 999, "entries": []})
    result = jobs_routes.initialize_job_queue(app, start_processing=False)

    assert "Unsupported queue schema version" in result["error"]
    response = client.post(
        "/queue/add",
        json={"endpoint": "/silence", "payload": {}},
        headers=csrf_headers(csrf_token),
    )
    assert response.status_code == 503
    body = response.get_json()
    assert body["code"] == "QUEUE_STORAGE_ERROR"
    assert "Unsupported queue schema version" in body["suggestion"]
    with jobs_routes.job_queue_lock:
        assert jobs_routes.job_queue == []
    # A newer-schema file is never quarantined: it likely holds real work
    # written by a newer OpenCut.
    assert (tmp_path / QUEUE_FILE).is_file()
    assert not (tmp_path / f"{QUEUE_FILE}.corrupt").exists()


def test_structurally_corrupt_queue_file_is_quarantined_and_queue_recovers(
    app, client, csrf_token, monkeypatch, tmp_path
):
    from opencut import user_data
    from opencut.queue_store import QUEUE_FILE

    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    corrupt_document = {"schema_version": 1, "entries": "not-a-list"}
    user_data.write_user_file(QUEUE_FILE, corrupt_document)

    result = jobs_routes.initialize_job_queue(app, start_processing=False)

    assert result == {"loaded": 0, "interrupted": 0, "invalid": 0}
    quarantine = tmp_path / f"{QUEUE_FILE}.corrupt"
    assert quarantine.is_file()
    assert json.loads(quarantine.read_text(encoding="utf-8")) == corrupt_document
    assert not (tmp_path / QUEUE_FILE).exists()

    # Persistence stays enabled: queue mutations work without a restart.
    monkeypatch.setattr(jobs_routes, "_process_queue", lambda app=None: None)
    response = client.post(
        "/queue/add",
        json={"endpoint": "/silence", "payload": {}},
        headers=csrf_headers(csrf_token),
    )
    assert response.status_code == 200
    with jobs_routes.job_queue_lock:
        assert len(jobs_routes.job_queue) == 1
    persisted = user_data.read_user_file(QUEUE_FILE)
    assert len(persisted["entries"]) == 1

    # A pre-existing .corrupt file is overwritten by the next quarantine.
    second_corrupt = {"schema_version": None}
    with jobs_routes.job_queue_lock:
        jobs_routes.job_queue.clear()
    user_data.write_user_file(QUEUE_FILE, second_corrupt)
    jobs_routes.initialize_job_queue(app, start_processing=False)
    assert json.loads(quarantine.read_text(encoding="utf-8")) == second_corrupt


def test_queue_import_export_round_trip_preserves_ids_without_duplicates(
    app, client, csrf_token, monkeypatch, tmp_path
):
    from opencut.queue_store import QUEUE_SCHEMA_VERSION

    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    jobs_routes.initialize_job_queue(app, start_processing=False)
    monkeypatch.setattr(jobs_routes, "_process_queue", lambda app=None: None)
    document = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "entries": [
            _entry("queue-one", added=10),
            _entry("queue-two", "interrupted", added=20),
        ],
    }

    imported = client.post(
        "/queue/import", json=document, headers=csrf_headers(csrf_token)
    )
    assert imported.status_code == 200
    assert imported.get_json()["imported"] == ["queue-one", "queue-two"]

    exported = client.get("/queue/export")
    assert exported.status_code == 200
    exported_document = exported.get_json()
    assert exported_document["schema_version"] == QUEUE_SCHEMA_VERSION
    assert [entry["id"] for entry in exported_document["entries"]] == [
        "queue-one", "queue-two",
    ]

    duplicate = client.post(
        "/queue/import", json=exported_document, headers=csrf_headers(csrf_token)
    )
    assert duplicate.status_code == 200
    assert duplicate.get_json()["imported"] == []
    assert duplicate.get_json()["skipped"] == ["queue-one", "queue-two"]

    with jobs_routes.job_queue_lock:
        jobs_routes.job_queue.clear()
        jobs_routes._persist_queue_locked()
    round_trip = client.post(
        "/queue/import", json=exported_document, headers=csrf_headers(csrf_token)
    )
    assert round_trip.status_code == 200
    with jobs_routes.job_queue_lock:
        assert [entry["id"] for entry in jobs_routes.job_queue] == [
            "queue-one", "queue-two",
        ]


def test_replay_revalidates_output_collision_and_keeps_queue_id(
    app, client, csrf_token, monkeypatch, tmp_path
):
    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    output_path = tmp_path / "already-exists.mp4"
    output_path.write_bytes(b"partial")
    from opencut.queue_store import save_queue

    save_queue([
        _entry(
            "recover-me",
            "interrupted",
            payload={"output_path": str(output_path)},
        )
    ])
    jobs_routes.initialize_job_queue(app, start_processing=False)
    monkeypatch.setattr(jobs_routes, "_process_queue", lambda app=None: None)

    collision = client.post(
        "/queue/replay/recover-me", json={}, headers=csrf_headers(csrf_token)
    )
    assert collision.status_code == 409
    assert collision.get_json()["code"] == "OUTPUT_COLLISION"
    assert collision.get_json()["collisions"] == [str(output_path)]

    output_path.unlink()
    replayed = client.post(
        "/queue/replay/recover-me", json={}, headers=csrf_headers(csrf_token)
    )
    assert replayed.status_code == 200
    assert replayed.get_json() == {"queue_id": "recover-me", "status": "queued"}
    with jobs_routes.job_queue_lock:
        assert len(jobs_routes.job_queue) == 1
        assert jobs_routes.job_queue[0]["id"] == "recover-me"
        assert jobs_routes.job_queue[0]["status"] == "queued"
        assert jobs_routes.job_queue[0]["attempts"] == 1


def test_queue_clear_confirmation_persists_queued_and_interrupted_removal(
    app, client, csrf_token, monkeypatch, tmp_path
):
    from opencut.queue_store import load_queue, save_queue

    jobs_routes = _use_temp_queue_store(monkeypatch, tmp_path)
    save_queue([
        _entry("queued-a", "queued", added=10),
        _entry("interrupted-b", "interrupted", added=20),
    ])
    jobs_routes.initialize_job_queue(app, start_processing=False)

    preview = client.post(
        "/queue/clear", json={"dry_run": True}, headers=csrf_headers(csrf_token)
    )
    assert preview.status_code == 200
    plan = preview.get_json()["plan"]
    assert plan["metadata"] == {"queued_count": 1, "interrupted_count": 1}

    confirmed = client.post(
        "/queue/clear",
        json={"confirm_token": plan["confirm_token"]},
        headers=csrf_headers(csrf_token),
    )
    assert confirmed.status_code == 200
    assert confirmed.get_json()["removed"] == 2
    persisted, _migrated = load_queue()
    assert persisted == []


def test_queue_dispatch_carries_flask_app_into_background_boundary(app):
    from opencut.routes import jobs_routes

    route = f"/test-queue-context-{time.time_ns()}"

    def _queued_handler():
        assert current_app._get_current_object() is app
        return jsonify({"job_id": "context-job"})

    app.add_url_rule(
        route,
        endpoint="test_queue_context",
        view_func=_queued_handler,
        methods=["POST"],
    )
    entry = _entry("context-entry", "running", endpoint=route)

    jobs_routes._dispatch_queue_entry(entry, app)

    assert entry["status"] == "started"
    assert entry["job_id"] == "context-job"


def test_cep_queue_recovery_state_is_wired():
    root = Path(__file__).resolve().parents[1] / "extension" / "com.opencut.panel" / "client"
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "main.js").read_text(encoding="utf-8")

    assert 'id="recoverQueueBtn"' in html
    assert '$("recoverQueueBtn")' in script
    assert '"/queue/replay/"' in script
    assert 't("queue.status_interrupted"' in script
