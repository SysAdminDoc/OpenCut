import json
import time
from unittest.mock import patch

from tests.conftest import csrf_headers


def test_invalid_json_returns_structured_error(client, csrf_token):
    resp = client.post(
        "/presets/save",
        data='{"name":',
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_JSON"
    assert "Fix malformed JSON" in data["suggestion"]


def test_settings_import_rejects_non_object_body(client, csrf_token):
    resp = client.post(
        "/settings/import",
        data=json.dumps(["not", "an", "object"]),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_INPUT"
    assert "top-level JSON object" in data["suggestion"]


def test_direct_request_get_json_routes_reject_non_object_body(client, csrf_token):
    resp = client.post(
        "/chat/clear",
        data=json.dumps(["not", "an", "object"]),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_INPUT"
    assert "top-level JSON object" in data["suggestion"]


def test_queue_sync_failure_does_not_leave_started_entry(client, csrf_token):
    import opencut.routes.jobs_routes as jobs_routes

    with jobs_routes.job_queue_lock:
        jobs_routes.job_queue.clear()
        jobs_routes._queue_state["running"] = False

    try:
        resp = client.post(
            "/queue/add",
            data=json.dumps({"endpoint": "/silence", "payload": {}}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 200
        queue_id = resp.get_json()["queue_id"]

        deadline = time.time() + 5
        while time.time() < deadline:
            with jobs_routes.job_queue_lock:
                snapshot = list(jobs_routes.job_queue)
            if not any(entry["id"] == queue_id for entry in snapshot):
                break
            time.sleep(0.05)

        with jobs_routes.job_queue_lock:
            remaining = list(jobs_routes.job_queue)

        assert not any(entry["id"] == queue_id for entry in remaining)
        assert not any(
            entry.get("status") == "started" and not entry.get("job_id")
            for entry in remaining
        )
    finally:
        with jobs_routes.job_queue_lock:
            jobs_routes.job_queue.clear()
            jobs_routes._queue_state["running"] = False


def test_cancel_route_persists_terminal_state(client, csrf_token):
    from opencut.jobs import _new_job, job_lock, jobs

    job_id = _new_job("test", "cancel-me")
    try:
        with patch("opencut.jobs._persist_job") as persist:
            resp = client.post(
                f"/cancel/{job_id}",
                data=json.dumps({}),
                headers=csrf_headers(csrf_token),
            )

        assert resp.status_code == 200
        persist.assert_called_once()
    finally:
        with job_lock:
            jobs.clear()


def test_logs_tail_filters_structured_json_lines(client, tmp_path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "\n".join([
            json.dumps({"level": "INFO", "job_id": "job-1", "message": "ready"}),
            json.dumps({"level": "ERROR", "job_id": "job-2", "message": "failed"}),
        ]),
        encoding="utf-8",
    )

    with patch("opencut.server.LOG_FILE", str(log_file)):
        resp = client.get("/logs/tail?lines=10&level=ERROR&job_id=job-2")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total"] == 1
    assert len(data["lines"]) == 1
    assert "job-2" in data["lines"][0]


def test_save_llm_settings_clamps_numeric_values(client, csrf_token):
    current = {
        "provider": "ollama",
        "model": "llama3",
        "api_key": "",
        "base_url": "http://localhost:11434",
        "max_tokens": 2000,
        "temperature": 0.3,
    }

    with patch("opencut.user_data.load_llm_settings", return_value=dict(current)), \
            patch("opencut.user_data.save_llm_settings") as save_settings:
        resp = client.post(
            "/settings/llm",
            data=json.dumps({
                "max_tokens": "999999",
                "temperature": "9.5",
                "provider": " openai ",
                "model": " gpt-test ",
            }),
            headers=csrf_headers(csrf_token),
        )

    assert resp.status_code == 200
    saved = save_settings.call_args[0][0]
    assert saved["provider"] == "openai"
    assert saved["model"] == "gpt-test"
    assert saved["max_tokens"] == 32768
    assert saved["temperature"] == 2.0
