import re

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


def test_f195_tools_are_registered_and_mapped():
    tools_by_name = {tool["name"]: tool for tool in mcp_server.MCP_TOOLS}

    assert len(mcp_server.MCP_TOOLS) == 39
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
