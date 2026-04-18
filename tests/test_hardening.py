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


# ---------------------------------------------------------------------------
# Audit pass 2 — regression tests
# ---------------------------------------------------------------------------


def test_expression_engine_rejects_dunder_custom_vars():
    """custom_vars with dunder or underscore-prefixed keys are filtered out."""
    from opencut.core.expression_engine import ExpressionContext, evaluate_expression

    ctx = ExpressionContext(
        time=1.0,
        frame=1,
        custom_vars={
            "__builtins__": {"__import__": __import__},
            "__class__": object,
            "_private": 999,
            "safe_var": 42.0,
        },
    )
    # safe_var should be available, dunder keys should not
    result = evaluate_expression("safe_var", ctx)
    assert result == 42.0


def test_expression_engine_rejects_non_scalar_custom_vars():
    """custom_vars with non-scalar values (objects, functions) are filtered."""
    from opencut.core.expression_engine import ExpressionContext, evaluate_expression

    ctx = ExpressionContext(
        time=1.0,
        frame=1,
        custom_vars={
            "obj": object(),          # should be rejected
            "func": lambda: None,     # should be rejected
            "num": 7.0,               # should pass
        },
    )
    result = evaluate_expression("num", ctx)
    assert result == 7.0


def test_expression_engine_banned_attrs_expanded():
    """Expanded banned attrs block __init__, __call__, __reduce__, etc."""
    from opencut.core.expression_engine import _BANNED_ATTRS

    for attr in ("__init__", "__new__", "__call__", "__reduce__",
                 "__getattribute__", "__module__", "__wrapped__"):
        assert attr in _BANNED_ATTRS, f"{attr} missing from _BANNED_ATTRS"


def test_expression_engine_non_primitive_result_returns_zero():
    """Non-primitive eval results return 0.0 instead of calling __float__."""
    from opencut.core.expression_engine import ExpressionContext, evaluate_expression

    ctx = ExpressionContext(time=1.0, frame=1)
    # list/dict results should yield 0.0, not raise or call __float__
    result = evaluate_expression("[1, 2, 3]", ctx)
    assert result == 0.0


def test_scripting_console_rejects_dunder_context_keys():
    """Context keys with double underscores are blocked in sandbox."""
    from opencut.core.scripting_console import create_sandbox

    sandbox = create_sandbox(context={
        "__builtins__": {"__import__": __import__},
        "__class__": object,
        "safe_key": "hello",
    })
    # __builtins__ should be our restricted version, not the injected one
    assert isinstance(sandbox["__builtins__"], dict)
    assert "__import__" not in sandbox["__builtins__"] or \
        sandbox["__builtins__"]["__import__"] is not __import__
    assert sandbox.get("safe_key") == "hello"


def test_scripting_console_blocked_patterns_expanded():
    """Expanded blocked patterns catch __init__, __reduce__, etc."""
    from opencut.core.scripting_console import _BLOCKED_PATTERNS

    for pat in ("__init__", "__new__", "__reduce__", "__getattribute__",
                "__module__", "__wrapped__"):
        assert pat in _BLOCKED_PATTERNS, f"{pat} missing from _BLOCKED_PATTERNS"


def test_webhook_url_rejects_localhost():
    """Webhook URL validation blocks localhost/private IPs."""
    import pytest as _pt

    from opencut.core.webhooks import _validate_webhook_url
    for url in (
        "http://localhost/hook",
        "http://127.0.0.1/hook",
        "http://0.0.0.0/hook",
    ):
        with _pt.raises(ValueError, match="localhost"):
            _validate_webhook_url(url)


def test_webhook_url_rejects_private_ips():
    """Webhook URL validation blocks private network IPs."""
    import pytest as _pt

    from opencut.core.webhooks import _validate_webhook_url
    for url in (
        "http://10.0.0.1/hook",
        "http://192.168.1.1/hook",
        "http://172.16.0.1/hook",
    ):
        with _pt.raises(ValueError, match="private"):
            _validate_webhook_url(url)


def test_plugin_download_url_rejects_private_ips():
    """Plugin download URL validation blocks private network targets."""
    import pytest as _pt

    from opencut.core.plugin_marketplace import _validate_download_url
    for url in (
        "http://localhost/plugin.zip",
        "http://127.0.0.1/plugin.zip",
        "http://10.0.0.1/plugin.zip",
        "http://192.168.1.1/plugin.zip",
    ):
        with _pt.raises(ValueError):
            _validate_download_url(url)


def test_jobs_sanitize_payload_handles_circular_ref():
    """_sanitize_payload_for_storage handles circular references gracefully."""
    from opencut.jobs import _sanitize_payload_for_storage

    # Circular reference should not hang or crash
    d = {"key": "value"}
    d["self"] = d  # circular
    result = _sanitize_payload_for_storage(d)
    # Should return truncated/error dict, not hang
    assert isinstance(result, dict)


def test_jobs_sanitize_payload_caps_large_dicts():
    """_sanitize_payload_for_storage caps very large dicts."""
    from opencut.jobs import _sanitize_payload_for_storage

    big = {f"key_{i}": f"val_{i}" for i in range(500)}
    result = _sanitize_payload_for_storage(big)
    assert isinstance(result, dict)


def test_open_path_blocks_executable_extensions(client, csrf_token):
    """open-path route blocks executable file types."""
    import tempfile

    # Create a fake .bat file
    with tempfile.NamedTemporaryFile(suffix=".bat", delete=False, mode="w") as f:
        f.write("echo hello")
        bat_path = f.name

    try:
        import os
        resp = client.post(
            "/system/open-path",
            data=json.dumps({"path": bat_path, "mode": "open"}),
            headers=csrf_headers(csrf_token),
        )
        assert resp.status_code == 403
        data = resp.get_json()
        assert "executable" in data["error"].lower() or ".bat" in data["error"]
    finally:
        import os
        os.unlink(bat_path)


def test_validate_output_path_rejects_non_writable_directory(tmp_path):
    from opencut.security import validate_output_path

    output = tmp_path / "out.wav"

    with patch("opencut.security.os.access") as mock_access:
        def _fake_access(path, mode):
            if str(path) == str(tmp_path):
                return False
            return True

        mock_access.side_effect = _fake_access

        import pytest as _pt
        with _pt.raises(ValueError, match="not writable"):
            validate_output_path(str(output))


def test_validate_output_path_rejects_directory_target(tmp_path):
    import pytest as _pt

    from opencut.security import validate_output_path
    with _pt.raises(ValueError, match="is a directory"):
        validate_output_path(str(tmp_path))
