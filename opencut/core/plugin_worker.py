"""Private JSON-lines worker used by :mod:`opencut.core.plugin_runtime`."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hmac
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_SAFE_REQUEST_HEADERS = {"accept", "content-type", "if-none-match"}
_SAFE_RESPONSE_HEADERS = {"cache-control", "content-type", "etag", "location"}


def _apply_posix_limits(memory_mb: int) -> None:
    if os.name != "posix":
        return
    try:
        import resource

        memory_bytes = max(128, memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        # The parent independently enforces memory and request time limits.
        pass


def _read_message() -> dict[str, Any] | None:
    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 2)
    if not line:
        return None
    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
        return {"_invalid": "request_limit"}
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"_invalid": "invalid_json"}
    return payload if isinstance(payload, dict) else {"_invalid": "invalid_message"}


def _write_message(protocol_out, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
        encoded = json.dumps(
            {
                "id": payload.get("id"),
                "ok": False,
                "code": "response_limit",
            },
            separators=(",", ":"),
        )
    protocol_out.write(encoded + "\n")
    protocol_out.flush()


def _load_plugin(plugin_dir: str, plugin_name: str, context: dict[str, Any]):
    from flask import Flask

    from opencut.core import plugins as plugin_api

    routes_file = Path(plugin_dir) / "routes.py"
    if not routes_file.is_file():
        raise RuntimeError("routes_missing")
    with plugin_api._plugins_lock:
        plugin_api._plugin_contexts[plugin_name] = {
            "capabilities": list(context["capabilities"]),
            "data_dir": context["data_dir"],
            "plugin_dir": plugin_dir,
        }

    inserted = False
    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(
            f"opencut_plugin_worker_{plugin_name}", routes_file
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("module_spec")
        module = importlib.util.module_from_spec(spec)
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                spec.loader.exec_module(module)
    finally:
        if inserted and plugin_dir in sys.path:
            sys.path.remove(plugin_dir)

    blueprint = getattr(module, "plugin_bp", None)
    if blueprint is None:
        raise RuntimeError("blueprint_missing")
    app = Flask(f"opencut-plugin-worker-{plugin_name}")
    app.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)
    app.register_blueprint(blueprint)

    routes = []
    jobs = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        view = app.view_functions[rule.endpoint]
        job = getattr(view, "_opencut_plugin_job", None)
        methods = sorted(set(rule.methods or ()) - {"HEAD", "OPTIONS"})
        entry = {
            "rule": str(rule.rule),
            "endpoint": rule.endpoint,
            "methods": methods,
            "job": dict(job) if isinstance(job, dict) else None,
        }
        routes.append(entry)
        if isinstance(job, dict):
            jobs.append(dict(job))
    catalog = {
        "routes": sorted(routes, key=lambda item: (item["rule"], item["endpoint"])),
        "jobs": sorted(jobs, key=lambda item: str(item.get("id", ""))),
    }
    return app, catalog


def _http_call(app, message: dict[str, Any]) -> dict[str, Any]:
    method = str(message.get("method") or "GET").upper()
    path = str(message.get("path") or "/")
    if not path.startswith("/") or "\x00" in path:
        return {"ok": False, "code": "invalid_path"}
    query = str(message.get("query") or "")
    headers = {
        str(key): str(value)
        for key, value in (message.get("headers") or {}).items()
        if str(key).lower() in _SAFE_REQUEST_HEADERS
    }
    try:
        body = base64.b64decode(str(message.get("body_b64") or ""), validate=True)
    except (ValueError, TypeError):
        return {"ok": False, "code": "invalid_body"}
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            response = app.test_client().open(
                path=f"{path}?{query}" if query else path,
                method=method,
                data=body,
                headers=headers,
            )
    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() in _SAFE_RESPONSE_HEADERS
    }
    return {
        "ok": True,
        "status": response.status_code,
        "headers": response_headers,
        "body_b64": base64.b64encode(response.get_data()).decode("ascii"),
    }


def _job_call(app, message: dict[str, Any]) -> dict[str, Any]:
    endpoint = str(message.get("endpoint") or "")
    view = app.view_functions.get(endpoint)
    meta = getattr(view, "_opencut_plugin_job", None) if view is not None else None
    if view is None or not isinstance(meta, dict):
        return {"ok": False, "code": "unknown_job"}
    data = message.get("data")
    if not isinstance(data, dict):
        return {"ok": False, "code": "invalid_job_data"}
    with app.test_request_context(json=data):
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                result = view(
                    str(message.get("job_id") or ""),
                    message.get("filepath"),
                    data,
                )
    json.dumps(result)
    return {"ok": True, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--plugin-name", required=True)
    parser.add_argument("--memory-mb", type=int, default=512)
    args = parser.parse_args()
    _apply_posix_limits(args.memory_mb)

    protocol_out = sys.stdout
    hello = _read_message()
    if hello is None or hello.get("action") != "hello":
        return 78
    request_id = hello.get("id")
    token = str(hello.get("token") or "")
    if len(token) < 32:
        _write_message(protocol_out, {"id": request_id, "ok": False, "code": "auth"})
        return 78
    raw_context = hello.get("context")
    if not isinstance(raw_context, dict):
        return 78
    capabilities = raw_context.get("capabilities")
    data_dir = raw_context.get("data_dir")
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(item, str) for item in capabilities)
        or not isinstance(data_dir, str)
        or not data_dir
    ):
        return 78
    context = {"capabilities": capabilities, "data_dir": data_dir}
    try:
        app, catalog = _load_plugin(args.plugin_dir, args.plugin_name, context)
    except BaseException:
        _write_message(
            protocol_out,
            {"id": request_id, "ok": False, "code": "startup_failed"},
        )
        return 1
    _write_message(
        protocol_out,
        {"id": request_id, "ok": True, "catalog": catalog},
    )

    while True:
        message = _read_message()
        if message is None:
            return 0
        message_id = message.get("id")
        if message.get("_invalid"):
            _write_message(
                protocol_out,
                {"id": message_id, "ok": False, "code": message["_invalid"]},
            )
            continue
        supplied = str(message.get("token") or "")
        if not hmac.compare_digest(token, supplied):
            _write_message(
                protocol_out,
                {"id": message_id, "ok": False, "code": "auth"},
            )
            continue
        action = message.get("action")
        if action == "shutdown":
            _write_message(protocol_out, {"id": message_id, "ok": True})
            return 0
        try:
            if action == "http":
                result = _http_call(app, message)
            elif action == "job":
                result = _job_call(app, message)
            elif action == "ping":
                result = {"ok": True}
            else:
                result = {"ok": False, "code": "unknown_action"}
        except BaseException:
            result = {"ok": False, "code": "plugin_exception"}
        result["id"] = message_id
        _write_message(protocol_out, result)


if __name__ == "__main__":
    raise SystemExit(main())
