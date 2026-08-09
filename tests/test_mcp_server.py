import http.client
import json
import re
import threading
from types import SimpleNamespace

from opencut import mcp_server

F195_TOOL_ROUTES = {
    "opencut_face_reshape": ("POST", "/video/face/reshape"),
    "opencut_skin_retouch": ("POST", "/video/face/retouch"),
    "opencut_smart_upscale": ("POST", "/video/upscale/smart"),
    "opencut_elevenlabs_tts": ("POST", "/audio/tts/elevenlabs"),
    "opencut_caption_qc": ("POST", "/captions/qc"),
    "opencut_review_bundle": ("POST", "/review/bundle"),
    "opencut_c2pa_provenance": ("POST", "/provenance/c2pa"),
    "opencut_marker_import": ("POST", "/markers/import"),
    "opencut_capability_probe": ("GET", "/system/capabilities"),
    "opencut_brand_kit": ("GET", "/settings/brand-kit"),
    "opencut_semantic_search": ("POST", "/search/ai"),
    "opencut_spectral_match": ("POST", "/audio/spectral-match"),
}

F209_SPECIAL_ACTION_ROUTES = {
    "opencut_generate_music": {("POST", "/audio/music-ai/ace-step")},
    "opencut_style_transfer": {("POST", "/video/style/arbitrary")},
    "opencut_brand_kit": {
        ("GET", "/settings/brand-kit"),
        ("DELETE", "/settings/brand-kit"),
        ("POST", "/settings/brand-kit"),
        ("POST", "/settings/brand-kit/preview"),
    },
    "opencut_semantic_search": {
        ("POST", "/search/ai"),
        ("POST", "/search/ai/index"),
        ("GET", "/search/ai/index/status"),
    },
}


def _mcp_path_to_flask_rule(path):
    return re.sub(r"{([A-Za-z_][A-Za-z0-9_]*)}", r"<\1>", path)


def _live_flask_operations(app):
    operations = set()
    for rule in app.url_map.iter_rules():
        for method in sorted((rule.methods or set()) - {"HEAD", "OPTIONS"}):
            operations.add((method, str(rule.rule)))
    return operations


def _capture_api(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True, "method": method, "path": path}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    return calls


def _mcp_http_post(server, *, body=b"{}", headers=None):
    """Serve one request through the real loopback MCP HTTP handler."""
    server.timeout = 3
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request("POST", "/", body=body, headers=headers or {})
        response = connection.getresponse()
        payload = json.loads(response.read() or b"{}")
    finally:
        connection.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    return response.status, payload


def _new_loopback_mcp_http_server():
    return mcp_server._create_mcp_http_server(
        "127.0.0.1",
        0,
        auth_required=False,
    )


def test_f195_tools_are_registered_and_mapped():
    tools_by_name = {tool["name"]: tool for tool in mcp_server.MCP_TOOLS}

    assert len(mcp_server.MCP_TOOLS) == 87
    assert len(tools_by_name) == len(mcp_server.MCP_TOOLS)
    assert set(F195_TOOL_ROUTES).issubset(tools_by_name)

    for name, route in F195_TOOL_ROUTES.items():
        assert mcp_server._TOOL_ROUTES[name] == route
        assert tools_by_name[name]["inputSchema"]["type"] == "object"


def test_f209_mcp_tools_map_to_live_flask_routes(app):
    tools_by_name = {tool["name"]: tool for tool in mcp_server.MCP_TOOLS}
    live_operations = _live_flask_operations(app)

    assert set(mcp_server._TOOL_ROUTES) == set(tools_by_name)

    missing = []
    for tool_name, (method, path) in sorted(mcp_server._TOOL_ROUTES.items()):
        flask_rule = _mcp_path_to_flask_rule(path)
        if (method, flask_rule) not in live_operations:
            missing.append(f"{tool_name}: {method} {path}")

    for tool_name, routes in sorted(F209_SPECIAL_ACTION_ROUTES.items()):
        assert tool_name in tools_by_name
        for method, path in sorted(routes):
            flask_rule = _mcp_path_to_flask_rule(path)
            if (method, flask_rule) not in live_operations:
                missing.append(f"{tool_name}: {method} {path}")

    assert missing == []


def test_mcp_http_auth_required_only_for_non_loopback_binds():
    assert not mcp_server._mcp_http_bind_requires_auth("127.0.0.1")
    assert not mcp_server._mcp_http_bind_requires_auth("localhost")
    assert not mcp_server._mcp_http_bind_requires_auth("::1")
    assert mcp_server._mcp_http_bind_requires_auth("0.0.0.0")
    assert mcp_server._mcp_http_bind_requires_auth("192.0.2.10")
    assert mcp_server._mcp_http_bind_requires_auth("")


def test_mcp_http_auth_accepts_header_token_only(monkeypatch):
    monkeypatch.setattr(mcp_server._auth, "is_token_valid", lambda token: token == "secret")

    assert mcp_server._mcp_http_request_is_authorized(
        {"X-OpenCut-Auth": "secret"},
        auth_required=True,
    )
    assert not mcp_server._mcp_http_request_is_authorized(
        {},
        auth_required=True,
    )
    assert mcp_server._mcp_http_request_is_authorized(
        {},
        auth_required=False,
    )


def test_mcp_http_rejects_foreign_host(monkeypatch):
    monkeypatch.setattr(mcp_server, "dispatch_jsonrpc", lambda _body: {"ok": True})
    server = _new_loopback_mcp_http_server()
    try:
        status, payload = _mcp_http_post(server, headers={"Host": "evil.example"})
    finally:
        server.server_close()

    assert status == 403
    assert payload["code"] == "HOST_NOT_ALLOWED"


def test_mcp_http_rejects_foreign_origin(monkeypatch):
    monkeypatch.setattr(mcp_server, "dispatch_jsonrpc", lambda _body: {"ok": True})
    server = _new_loopback_mcp_http_server()
    try:
        headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "Origin": "https://evil.example",
        }
        status, payload = _mcp_http_post(server, headers=headers)
    finally:
        server.server_close()

    assert status == 403
    assert payload["code"] == "ORIGIN_NOT_ALLOWED"


def test_mcp_http_requires_token_for_loopback_post(monkeypatch):
    monkeypatch.setattr(mcp_server._auth, "is_token_valid", lambda _token: False)
    monkeypatch.setattr(mcp_server, "dispatch_jsonrpc", lambda _body: {"ok": True})
    server = _new_loopback_mcp_http_server()
    try:
        headers = {"Host": f"127.0.0.1:{server.server_port}"}
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {},
        }).encode()
        status, payload = _mcp_http_post(server, headers=headers, body=body)
    finally:
        server.server_close()

    assert status == 401
    assert payload["code"] == "TOKEN_REQUIRED"


def test_mcp_http_legacy_requests_remain_headerless(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_mcp_http_csrf_token_is_valid",
        lambda token: token == "csrf",
    )
    server = _new_loopback_mcp_http_server()
    try:
        headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "X-OpenCut-Token": "csrf",
        }
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }).encode()
        status, payload = _mcp_http_post(server, headers=headers, body=body)
    finally:
        server.server_close()

    assert status == 200
    assert "resultType" not in payload["result"]


def test_mcp_http_modern_requests_validate_standard_headers(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_mcp_http_csrf_token_is_valid",
        lambda token: token == "csrf",
    )
    server = _new_loopback_mcp_http_server()
    try:
        headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "X-OpenCut-Token": "csrf",
            "MCP-Protocol-Version": mcp_server.LATEST_PROTOCOL_VERSION,
            "Mcp-Method": "server/discover",
        }
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {},
        }).encode()
        status, payload = _mcp_http_post(server, headers=headers, body=body)
    finally:
        server.server_close()

    assert status == 200
    assert payload["result"]["resultType"] == "complete"


def test_mcp_http_modern_requests_reject_missing_or_mismatched_headers(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_mcp_http_csrf_token_is_valid",
        lambda token: token == "csrf",
    )
    server = _new_loopback_mcp_http_server()
    try:
        base_headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "X-OpenCut-Token": "csrf",
            "MCP-Protocol-Version": mcp_server.LATEST_PROTOCOL_VERSION,
        }
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {},
        }).encode()
        status, payload = _mcp_http_post(
            server,
            headers=base_headers,
            body=body,
        )
        assert status == 400
        assert payload["error"]["code"] == mcp_server.ERROR_HEADER_MISMATCH

        mismatched_headers = dict(base_headers)
        mismatched_headers["Mcp-Method"] = "tools/list"
        status, payload = _mcp_http_post(
            server,
            headers=mismatched_headers,
            body=body,
        )
        assert status == 400
        assert payload["error"]["code"] == mcp_server.ERROR_HEADER_MISMATCH
    finally:
        server.server_close()


def test_mcp_http_modern_tools_call_requires_matching_name_header(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_mcp_http_csrf_token_is_valid",
        lambda token: token == "csrf",
    )
    server = _new_loopback_mcp_http_server()
    try:
        headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "X-OpenCut-Token": "csrf",
            "MCP-Protocol-Version": mcp_server.LATEST_PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Mcp-Name": "wrong_tool",
        }
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "actual_tool",
                "arguments": {},
            },
        }).encode()
        status, payload = _mcp_http_post(server, headers=headers, body=body)
    finally:
        server.server_close()

    assert status == 400
    assert payload["error"]["code"] == mcp_server.ERROR_HEADER_MISMATCH


def test_mcp_http_rejects_unsupported_protocol_header(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_mcp_http_csrf_token_is_valid",
        lambda token: token == "csrf",
    )
    server = _new_loopback_mcp_http_server()
    try:
        headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "X-OpenCut-Token": "csrf",
            "MCP-Protocol-Version": "2019-01-01",
            "Mcp-Method": "server/discover",
        }
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {},
        }).encode()
        status, payload = _mcp_http_post(server, headers=headers, body=body)
    finally:
        server.server_close()

    assert status == 400
    assert payload["error"]["code"] == mcp_server.ERROR_UNSUPPORTED_PROTOCOL_VERSION
    assert payload["error"]["data"]["requested"] == "2019-01-01"


def test_mcp_http_rejects_body_over_cap_before_reading(monkeypatch):
    monkeypatch.setattr(mcp_server._auth, "is_token_valid", lambda _token: False)
    monkeypatch.setattr(mcp_server, "_mcp_http_csrf_token_is_valid", lambda token: token == "csrf")
    monkeypatch.setattr(mcp_server, "dispatch_jsonrpc", lambda _body: {"ok": True})
    server = _new_loopback_mcp_http_server()
    try:
        headers = {
            "Host": f"127.0.0.1:{server.server_port}",
            "X-OpenCut-Token": "csrf",
        }
        # Declare an oversized body without sending it. The handler must
        # reject from Content-Length alone, and this keeps the Windows socket
        # test deterministic instead of racing a client write against the
        # intentional early connection close.
        headers["Content-Length"] = str(mcp_server._MCP_HTTP_MAX_BODY_BYTES + 1)
        status, payload = _mcp_http_post(server, headers=headers, body=b"")
    finally:
        server.server_close()

    assert status == 413
    assert payload["code"] == "PAYLOAD_TOO_LARGE"
    assert payload["max_bytes"] == mcp_server._MCP_HTTP_MAX_BODY_BYTES


def test_remote_backend_requests_read_and_forward_current_secret(monkeypatch):
    captured = []
    values = iter(["a" * 64, "b" * 64])

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    monkeypatch.setattr(mcp_server, "BACKEND_URL", "http://opencut-server:5679")
    monkeypatch.setattr(mcp_server, "_csrf_is_fresh", lambda: True)
    monkeypatch.setattr(
        mcp_server._auth,
        "current_token",
        lambda: SimpleNamespace(token=next(values)),
    )
    monkeypatch.setattr(
        mcp_server.urllib.request,
        "urlopen",
        lambda request, timeout: captured.append(request) or Response(),
    )

    assert mcp_server._api("GET", "/system/feature-state") == {"ok": True}
    assert mcp_server._api("GET", "/system/feature-state") == {"ok": True}
    assert captured[0].get_header("X-opencut-auth") == "a" * 64
    assert captured[1].get_header("X-opencut-auth") == "b" * 64


def test_mcp_backend_url_honors_container_service_environment(monkeypatch):
    monkeypatch.setenv("OPENCUT_MCP_BACKEND_URL", "http://opencut-server:5679/")
    monkeypatch.setattr(mcp_server.sys, "argv", ["opencut-mcp-server"])
    monkeypatch.setattr(mcp_server, "run_mcp_stdio", lambda: None)

    mcp_server.main()

    assert mcp_server.BACKEND_URL == "http://opencut-server:5679"


def test_f195_simple_tools_dispatch_to_backend(monkeypatch):
    calls = _capture_api(monkeypatch)
    cases = [
        ("opencut_face_reshape", {"filepath": "media/clip.mp4"}, ("POST", "/video/face/reshape")),
        ("opencut_skin_retouch", {"filepath": "media/clip.mp4"}, ("POST", "/video/face/retouch")),
        ("opencut_smart_upscale", {"filepath": "media/clip.mp4"}, ("POST", "/video/upscale/smart")),
        ("opencut_elevenlabs_tts", {"text": "Read this"}, ("POST", "/audio/tts/elevenlabs")),
        ("opencut_caption_qc", {"srt_text": "1\n00:00:00,000 --> 00:00:01,000\nHi"}, ("POST", "/captions/qc")),
        ("opencut_review_bundle", {"output_path": "out/review.zip"}, ("POST", "/review/bundle")),
        ("opencut_c2pa_provenance", {"asset_path": "renders/final.mp4"}, ("POST", "/provenance/c2pa")),
        ("opencut_marker_import", {"text": "Name,Start,End\nA,0,1", "format": "csv"}, ("POST", "/markers/import")),
        ("opencut_capability_probe", {}, ("GET", "/system/capabilities")),
        ("opencut_spectral_match", {"filepath": "media/clip.mp4", "reference_path": "media/ref.wav"}, ("POST", "/audio/spectral-match")),
    ]

    for tool_name, arguments, expected in cases:
        assert mcp_server.handle_tool_call(tool_name, arguments)["ok"] is True
        assert calls[-1] == (*expected, arguments)


def test_f195_brand_kit_actions_dispatch_to_backend(monkeypatch):
    calls = _capture_api(monkeypatch)

    assert mcp_server.handle_tool_call("opencut_brand_kit", {})["path"] == "/settings/brand-kit"
    assert calls[-1] == ("GET", "/settings/brand-kit", None)

    assert mcp_server.handle_tool_call("opencut_brand_kit", {"action": "delete"})["method"] == "DELETE"
    assert calls[-1] == ("DELETE", "/settings/brand-kit", None)

    brand_kit = {"name": "Launch", "primary_color": "#112233"}
    assert mcp_server.handle_tool_call(
        "opencut_brand_kit",
        {"action": "save", "brand_kit": brand_kit},
    )["path"] == "/settings/brand-kit"
    assert calls[-1] == ("POST", "/settings/brand-kit", brand_kit)

    preview_args = {
        "action": "preview",
        "filepath": "media/clip.mp4",
        "brand_kit": brand_kit,
        "output": "out/preview.mp4",
    }
    assert mcp_server.handle_tool_call("opencut_brand_kit", preview_args)["path"] == "/settings/brand-kit/preview"
    assert calls[-1] == ("POST", "/settings/brand-kit/preview", preview_args)

    before_invalid = len(calls)
    assert mcp_server.handle_tool_call("opencut_brand_kit", {"action": "merge"}) == {
        "error": "Invalid action for opencut_brand_kit"
    }
    assert len(calls) == before_invalid


def test_f195_semantic_search_actions_dispatch_to_backend(monkeypatch):
    calls = _capture_api(monkeypatch)

    search_args = {"query": "speaker at podium"}
    assert mcp_server.handle_tool_call("opencut_semantic_search", search_args)["path"] == "/search/ai"
    assert calls[-1] == ("POST", "/search/ai", search_args)

    index_args = {"action": "index", "media_paths": ["media/a.mp4", "media/b.mp4"]}
    assert mcp_server.handle_tool_call("opencut_semantic_search", index_args)["path"] == "/search/ai/index"
    assert calls[-1] == ("POST", "/search/ai/index", index_args)

    assert mcp_server.handle_tool_call("opencut_semantic_search", {"action": "status"})["path"] == "/search/ai/index/status"
    assert calls[-1] == ("GET", "/search/ai/index/status", None)

    before_invalid = len(calls)
    assert mcp_server.handle_tool_call("opencut_semantic_search", {"action": "purge"}) == {
        "error": "Invalid action for opencut_semantic_search"
    }
    assert len(calls) == before_invalid


def test_f195_path_validation_covers_new_path_keys(monkeypatch):
    calls = _capture_api(monkeypatch)

    assert "Invalid asset_path" in mcp_server.handle_tool_call(
        "opencut_c2pa_provenance",
        {"asset_path": "../renders/final.mp4"},
    )["error"]
    assert "Invalid path in extra_files[0]" in mcp_server.handle_tool_call(
        "opencut_review_bundle",
        {"output_path": "out/review.zip", "extra_files": ["//server/share.mov"]},
    )["error"]
    assert "Invalid path in media_paths[0]" in mcp_server.handle_tool_call(
        "opencut_semantic_search",
        {"action": "index", "media_paths": ["media/../secret.mp4"]},
    )["error"]
    assert "Invalid reference_path" in mcp_server.handle_tool_call(
        "opencut_spectral_match",
        {"filepath": "media/clip.mp4", "reference_path": "\\\\server\\ref.wav"},
    )["error"]
    assert calls == []
