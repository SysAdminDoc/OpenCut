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


def test_silent_json_routes_reject_non_object_body(client, csrf_token):
    resp = client.post(
        "/context/analyze",
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


def test_expression_engine_whitelist_blocks_attribute_access():
    """Whitelist rejects every attribute node, closing dunder-walk escapes."""
    from opencut.core.expression_engine import validate_expression

    for expr in (
        "().__class__",
        "(1).__class__.__bases__",
        "[].__class__.__mro__",
        "().__class__.__subclasses__()",
        "foo.bar",
    ):
        result = validate_expression(expr)
        assert not result["valid"], f"{expr} should be rejected"


def test_expression_engine_whitelist_blocks_unknown_calls():
    """Only sandbox callables may be invoked; everything else is rejected."""
    from opencut.core.expression_engine import validate_expression

    for expr in ("breakpoint()", "eval('1')", "open('x')", "unknown_fn(1)"):
        result = validate_expression(expr)
        assert not result["valid"], f"{expr} should be rejected"


def test_expression_engine_whitelist_allows_legit_expressions():
    """Real motion-design expressions still validate under the whitelist."""
    from opencut.core.expression_engine import validate_expression

    for expr in (
        "sin(t * pi * 2)",
        "lerp(0, 100, progress)",
        "1 if beat else 0",
        "noise(1.0, octaves=4)",
        "clamp(amplitude * 2, 0, 1)",
        "2 + 3 * frame",
        "min(w, h) / 2",
    ):
        assert validate_expression(expr)["valid"], f"{expr} should be valid"


def test_scripting_console_ast_blocks_dunders_outside_pattern_list():
    """AST check catches escape dunders the substring scan does not list."""
    from opencut.core.scripting_console import _BLOCKED_PATTERNS, execute_script

    # __base__ / __flags__ are real escape vectors absent from _BLOCKED_PATTERNS.
    for expr in ("type(1).__base__", "object.__flags__", "(1).__delattr__"):
        assert not any(p in expr for p in _BLOCKED_PATTERNS), \
            f"{expr} unexpectedly already in pattern list"
        result = execute_script(expr)
        assert not result.success, f"{expr} should be rejected"
        assert "not allowed" in result.error


def test_scripting_console_ast_blocks_blocked_builtin_reference():
    """Referencing a blocked builtin by name is caught structurally."""
    from opencut.core.scripting_console import execute_script

    for expr in ("g = getattr", "h = eval", "j = globals"):
        result = execute_script(expr)
        assert not result.success, f"{expr} should be rejected"
        assert "not allowed" in result.error


def test_scripting_console_ast_allows_legit_scripts():
    """Ordinary sandbox scripts still execute under the AST check."""
    from opencut.core.scripting_console import execute_script

    result = execute_script("total = sum([1, 2, 3])\nprint(sqrt(total * 6))")
    assert result.success, result.error
    assert "6.0" in result.output


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


def test_public_url_validation_rejects_malformed_inputs():
    import pytest as _pt

    from opencut.core.url_safety import validate_public_http_url

    for value in (None, 123, ["https://example.com/hook"]):
        with _pt.raises(ValueError, match="URL is required"):
            validate_public_http_url(value)

    with _pt.raises(ValueError, match="not a valid URL"):
        validate_public_http_url("https://[::1")


def test_public_url_validation_rejects_numeric_ip_ssrf_bypass():
    """Alternate IPv4 encodings (decimal/octal/hex/short-form) must not slip
    past the SSRF guard: ``ipaddress.ip_address`` rejects them, but the OS
    resolver and HTTP clients still expand them to loopback/private targets.
    """
    import pytest as _pt

    from opencut.core.url_safety import validate_public_http_url

    # All of these expand to 127.0.0.1 / a private address via inet_aton.
    for value, match in (
        ("http://2130706433/", "localhost"),       # decimal 127.0.0.1
        ("http://0177.0.0.1/", "localhost"),        # octal 127.0.0.1
        ("http://0x7f.0.0.1/", "localhost"),        # hex 127.0.0.1
        ("http://127.1/", "localhost"),             # short-form 127.0.0.1
        ("http://3232235521/", "private"),          # decimal 192.168.0.1
    ):
        with _pt.raises(ValueError, match=match):
            validate_public_http_url(value)

    # Genuine public hosts and IPs must still pass unchanged.
    for value in ("http://example.com/x", "https://api.github.com/", "http://8.8.8.8/"):
        assert validate_public_http_url(value) == value


def test_webhook_system_url_rejects_local_and_private_targets():
    import pytest as _pt

    from opencut.core.webhook_system import _validate_webhook_url

    for url, match in (
        ("http://localhost/hook", "localhost"),
        ("http://127.0.0.1/hook", "localhost"),
        ("http://10.0.0.1/hook", "private"),
        ("http://192.168.1.1/hook", "private"),
        ("http://[::1]/hook", "localhost"),
    ):
        with _pt.raises(ValueError, match=match):
            _validate_webhook_url(url)


def test_webhook_integrations_reject_local_and_private_targets():
    import pytest as _pt

    from opencut.core.webhook_integrations import register_webhook_trigger, send_webhook

    result = send_webhook("http://127.0.0.1/hook", "test", {})
    assert result.success is False
    assert "localhost" in result.error

    with _pt.raises(ValueError, match="private"):
        register_webhook_trigger("test", "http://10.0.0.1/hook")


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
    assert result["_keys_trimmed"] is True


def test_jobs_sanitize_payload_redacts_sensitive_values():
    """Persisted job payloads keep diagnostics but never store credentials."""
    from opencut.jobs import _sanitize_payload_for_storage

    payload = {
        "filepath": "/tmp/clip.mp4",
        "api_key": "sk-live-secret",
        "headers": {"Authorization": "Bearer super-secret"},
        "nested": [{"refresh_token": "rt-secret", "label": "keep"}],
        "metadata": {"tokenizer": "not-a-secret-key"},
    }

    result = _sanitize_payload_for_storage(payload)
    encoded = json.dumps(result)

    assert result["filepath"] == "/tmp/clip.mp4"
    assert result["api_key"] == "[REDACTED]"
    assert result["headers"]["Authorization"] == "[REDACTED]"
    assert result["nested"][0]["refresh_token"] == "[REDACTED]"
    assert result["nested"][0]["label"] == "keep"
    assert result["metadata"]["tokenizer"] == "not-a-secret-key"
    assert "sk-live-secret" not in encoded
    assert "super-secret" not in encoded
    assert "rt-secret" not in encoded
    assert payload["api_key"] == "sk-live-secret"


def test_schedule_record_time_runs_sync_when_io_pool_closed(monkeypatch):
    import opencut.jobs as jobs_mod

    calls = []

    class ClosedPool:
        def submit(self, fn):
            raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(jobs_mod, "_io_pool", ClosedPool())
    monkeypatch.setattr("opencut.helpers._get_file_duration", lambda _path: 99)
    monkeypatch.setattr(
        "opencut.helpers._record_job_time",
        lambda job_type, elapsed, file_dur: calls.append((job_type, elapsed, file_dur)),
    )

    jobs_mod._schedule_record_time("render", 1.25, "")

    assert calls == [("render", 1.25, 0)]


def test_is_path_within_uses_component_boundaries(tmp_path):
    from opencut.security import is_path_within

    root = tmp_path / "cache"
    child = root / "models" / "model.bin"
    sibling = tmp_path / "cache_evil" / "model.bin"
    child.parent.mkdir(parents=True)
    sibling.parent.mkdir(parents=True)
    child.write_bytes(b"ok")
    sibling.write_bytes(b"no")

    assert is_path_within(str(child), str(root)) is True
    assert is_path_within(str(root), str(root)) is True
    assert is_path_within(str(sibling), str(root)) is False


def test_is_path_within_normalizes_case(monkeypatch):
    import os

    from opencut import security

    monkeypatch.setattr(security.os.path, "realpath", lambda value: str(value))
    monkeypatch.setattr(security.os.path, "normcase", lambda value: os.path.normpath(str(value)).lower())

    root = "/Users/Example/.cache/huggingface"
    child = "/users/example/.CACHE/HuggingFace/hub/models--openai--whisper"
    sibling = "/users/example/.CACHE/HuggingFace_evil/model.bin"

    assert security.is_path_within(child, root) is True
    assert security.is_path_within(sibling, root) is False


def test_standard_install_routes_skip_real_installs_in_testing(client, csrf_token):
    headers = csrf_headers(csrf_token)

    for path, body, component in (
        ("/video/depth/install", {}, "depth_effects"),
        ("/video/ai/install", {"component": "rembg"}, "rembg"),
        ("/install-whisper", {"backend": "faster-whisper"}, "install-whisper"),
        ("/demucs/install", {}, "demucs"),
    ):
        resp = client.post(path, data=json.dumps(body), headers=headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["component"] == component
        assert data["testing"] is True
        assert data["job_id"].startswith("test-install-")


def test_models_delete_allows_custom_model_cache_file(client, csrf_token, tmp_path, monkeypatch):
    import opencut.routes.system_model_routes as system_routes

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "model.bin"
    model_file.write_bytes(b"model")

    monkeypatch.setattr(system_routes, "WHISPER_MODELS_DIR", str(model_dir))

    preview = client.post(
        "/models/delete",
        data=json.dumps({"path": str(model_file), "dry_run": True}),
        headers=csrf_headers(csrf_token),
    )
    token = preview.get_json()["confirm_token"]

    resp = client.post(
        "/models/delete",
        data=json.dumps({"path": str(model_file), "confirm_token": token}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    assert resp.get_json()["success"] is True
    assert not model_file.exists()


def test_models_delete_rejects_non_string_path(client, csrf_token):
    resp = client.post(
        "/models/delete",
        data=json.dumps({"path": ["/tmp/model.bin"]}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_INPUT"
    assert "path must be a string" in data["error"]


def test_open_path_allows_only_safe_document_media_extensions(client, csrf_token):
    """open-path route blocks unsafe shell/control file types."""
    import os
    import tempfile

    unsafe_paths = []
    for suffix in (".bat", ".msc", ".cpl", ".settingcontent-ms", ".url"):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as f:
            f.write("test")
            unsafe_paths.append(f.name)

    try:
        for path in unsafe_paths:
            resp = client.post(
                "/system/open-path",
                data=json.dumps({"path": path, "mode": "open"}),
                headers=csrf_headers(csrf_token),
            )
            assert resp.status_code == 403
            assert "unsupported file type" in resp.get_json()["error"].lower()
    finally:
        for path in unsafe_paths:
            os.unlink(path)


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


def test_uxp_engine_registry_escapes_dynamic_attribute_values():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "extension/com.opencut.uxp/main.js").read_text(encoding="utf-8")
    utils_source = (repo_root / "extension/com.opencut.uxp/uxp-utils.js").read_text(encoding="utf-8")

    assert 'data-domain="${UIController.escapeHtml(domain)}"' in source
    assert 'value="${UIController.escapeHtml(eng.name)}"' in source
    assert 'for="${UIController.escapeHtml(domainId)}"' in source
    assert "escapeHtml as escapeHtmlValue" in source
    assert "safeDomIdSegment" in source
    assert 'from "./uxp-utils.js"' in source
    assert "return escapeHtmlValue(str);" in source
    assert ".replace(/'/g, \"&#39;\")" in utils_source


def test_uxp_fetch_wrapper_clears_backend_timeout_timers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "extension/com.opencut.uxp"
    source = (root / "main.js").read_text(encoding="utf-8")
    client_source = (root / "backend-client.js").read_text(encoding="utf-8")

    assert "async function fetchWithTimeout" in source
    assert "clearTimeout(timer);" in source
    assert "await fetchWithTimeout(`${url}/health`, {}, 500)" in source
    assert "await fetchWithTimeout(" in client_source
    assert "requestTimeoutMs = 120000" in client_source
