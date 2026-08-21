"""F194 extended MCP route-tool catalogue tests."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from opencut import mcp_extended_tools, mcp_server
from opencut.tools import dump_mcp_extended_tools as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "opencut" / "_generated" / "mcp_extended_tools.json"


def _tool_for(method: str, path: str) -> dict:
    for mcp_tool in mcp_extended_tools.get_extended_tools():
        metadata = mcp_tool["metadata"]
        if metadata["method"] == method and metadata["path"] == path:
            return mcp_tool
    raise AssertionError(f"extended tool for {method} {path} not found")


def test_committed_extended_manifest_matches_live_generator():
    assert MANIFEST.is_file(), f"F194 manifest must exist at {MANIFEST}"

    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live = tool.build_manifest()

    assert committed == live
    assert committed["tool_count"] >= 1000
    # Coarse coverage canary: a healthy build carries dozens of response-schema
    # annotations. The exact figure drifts as routes move between the curated
    # MCP tool set (excluded here) and the extended set, so this is a floor that
    # catches a collapse in schema discovery, not an exact count.
    assert committed["response_schema_count"] >= 80
    assert committed["tool_prefix"] == "opencut_route_"
    assert "POST" in committed["method_counts"]
    assert "GET" in committed["method_counts"]


def test_extended_tools_are_opt_in_and_do_not_change_curated_default(monkeypatch):
    monkeypatch.delenv(mcp_extended_tools.EXTENDED_MCP_ENV, raising=False)

    assert len(mcp_server.MCP_TOOLS) == 98
    assert len(mcp_server.get_mcp_tools()) == 98

    extended_count = len(mcp_server.get_mcp_tools(include_extended=True))
    assert extended_count == 98 + len(mcp_extended_tools.get_extended_tools())
    assert extended_count >= 1000

    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")
    assert len(mcp_server.get_mcp_tools()) == extended_count


def test_recovery_interchange_stays_rest_only():
    route_keys = {
        (entry["metadata"]["method"], entry["metadata"]["path"])
        for entry in mcp_extended_tools.get_extended_tools()
    }

    assert route_keys.isdisjoint({
        ("GET", "/queue/export"),
        ("POST", "/queue/import"),
        ("POST", "/queue/replay/<queue_id>"),
        ("GET", "/journal/recovery"),
        ("POST", "/journal/checkpoints"),
        ("GET", "/journal/checkpoints/<transaction_id>"),
        ("POST", "/journal/checkpoints/<transaction_id>/complete"),
        ("POST", "/journal/checkpoints/<transaction_id>/recovery-failed"),
        ("POST", "/journal/checkpoints/<transaction_id>/recovered"),
        ("GET", "/journal/checkpoints/<transaction_id>/diagnostics"),
    })


def test_mcp_bridge_docs_do_not_pin_stale_tool_counts():
    from opencut.routes import mcp_bridge_routes

    docs = "\n".join(
        str(part or "")
        for part in (
            inspect.getdoc(mcp_bridge_routes),
            inspect.getdoc(mcp_bridge_routes.route_mcp_tools),
        )
    )

    assert "39 curated" not in docs
    assert "1,325" not in docs
    assert "live curated tools" in docs
    assert "auto-generated route tools" in docs


def test_extended_tool_names_are_unique_and_tagged_lower_priority():
    tools = mcp_extended_tools.get_extended_tools()
    names = [tool["name"] for tool in tools]

    assert len(names) == len(set(names))
    assert all(name.startswith("opencut_route_") for name in names)
    for mcp_tool in tools[:25]:
        assert mcp_tool["metadata"]["generated"] is True
        assert mcp_tool["metadata"]["priority"] == "extended"
        assert "lower-priority" in mcp_tool["description"]


def test_extended_tools_include_introspected_core_response_schema():
    mcp_tool = _tool_for("POST", "/delivery/transfer-bundle")

    assert mcp_tool["metadata"]["response_schema"] == "TransferBundleResult"
    assert "Response schema: TransferBundleResult." in mcp_tool["description"]


def test_extended_dispatch_is_disabled_unless_opted_in(monkeypatch):
    monkeypatch.delenv(mcp_extended_tools.EXTENDED_MCP_ENV, raising=False)
    tool_name = _tool_for("GET", "/agent/tools")["name"]

    result = mcp_server.handle_tool_call(tool_name, {})

    assert "disabled" in result["error"]
    assert mcp_extended_tools.EXTENDED_MCP_ENV in result["error"]


def test_extended_dispatch_builds_get_path_and_query(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True, "path": path}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")
    tool_name = _tool_for("GET", "/agent/tools")["name"]

    result = mcp_server.handle_tool_call(
        tool_name,
        {"query": {"compact": "1"}, "source": "mcp-test"},
    )

    assert result["ok"] is True
    assert calls == [("GET", "/agent/tools?compact=1&source=mcp-test", None)]


def test_extended_dispatch_renders_path_params_and_body(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True, "path": path}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")
    tool_name = _tool_for("POST", "/jobs/retry/<job_id>")["name"]

    result = mcp_server.handle_tool_call(
        tool_name,
        {"job_id": "abc-123", "body": {"force": True}, "reason": "retry"},
    )

    assert result["ok"] is True
    assert calls == [("POST", "/jobs/retry/abc-123", {"force": True, "reason": "retry"})]


def test_extended_dispatch_requires_path_params(monkeypatch):
    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")
    tool_name = _tool_for("POST", "/jobs/retry/<job_id>")["name"]

    result = mcp_server.handle_tool_call(tool_name, {"body": {"force": True}})

    assert "Missing path parameter `job_id`" in result["error"]


def test_extended_dispatch_reuses_mcp_path_validation(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")

    get_tool_name = _tool_for("GET", "/agent/tools")["name"]
    assert "Invalid path" in mcp_server.handle_tool_call(
        get_tool_name,
        {"path": "../secret.txt"},
    )["error"]
    assert "Invalid query.path" in mcp_server.handle_tool_call(
        get_tool_name,
        {"query": {"path": "../secret.txt"}},
    )["error"]

    post_tool_name = _tool_for("POST", "/jobs/retry/<job_id>")["name"]
    assert "Invalid body.output_path" in mcp_server.handle_tool_call(
        post_tool_name,
        {"job_id": "abc-123", "body": {"output_path": "../secret.mp4"}},
    )["error"]
    assert "Invalid path in body.media_paths[0]" in mcp_server.handle_tool_call(
        post_tool_name,
        {"job_id": "abc-123", "body": {"media_paths": ["//server/share.mov"]}},
    )["error"]

    assert calls == []


def test_cli_check_passes_in_sync():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_mcp_extended_tools", "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"--check should pass when in sync; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "in sync" in result.stdout


def _representative_get_tools() -> list[dict]:
    """One parameter-free GET tool per top-level path family.

    The catalogue is 1,467 tools; calling all of them here would be a
    different kind of suite. One per family keeps every area of the API in
    the blast radius while staying deterministic.
    """
    chosen: dict[str, dict] = {}
    for mcp_tool in sorted(
        mcp_extended_tools.get_extended_tools(), key=lambda entry: entry["name"]
    ):
        metadata = mcp_tool["metadata"]
        if metadata["method"] != "GET" or metadata.get("path_params"):
            continue
        family = metadata["path"].strip("/").split("/")[0]
        chosen.setdefault(family, mcp_tool)
    return [chosen[name] for name in sorted(chosen)]


def _tools_without_a_handler(tools, view_functions, rules) -> list[str]:
    """Catalogue entries whose endpoint or rule the app does not serve."""
    unbacked = []
    for mcp_tool in tools:
        metadata = mcp_tool["metadata"]
        endpoint = metadata["endpoint"]
        if endpoint not in view_functions:
            unbacked.append(f"{mcp_tool['name']}: no view function `{endpoint}`")
        elif (metadata["path"], metadata["method"]) not in rules:
            unbacked.append(
                f"{mcp_tool['name']}: {endpoint} no longer answers "
                f"{metadata['method']} {metadata['path']}"
            )
    return unbacked


def _app_rules(app) -> set[tuple[str, str]]:
    return {
        (rule.rule, method)
        for rule in app.url_map.iter_rules()
        for method in (rule.methods or ())
    }


def test_every_extended_tool_points_at_a_live_handler(app):
    """The manifest gate diffs names; nothing checked the handlers existed.

    `dump_mcp_extended_tools --check` compares the committed catalogue to a
    regenerated one, so deleting a route handler updates both sides and stays
    green. The dispatch tests in this file all pass a fake `api_call` that
    never reaches Flask.
    """
    unbacked = _tools_without_a_handler(
        mcp_extended_tools.get_extended_tools(), app.view_functions, _app_rules(app)
    )
    assert unbacked == [], "extended tools advertise routes the app does not serve:\n" + "\n".join(unbacked[:20])


def test_the_handler_check_notices_a_deleted_route(app):
    """Worth running only if removing a handler actually fails it."""
    tools = mcp_extended_tools.get_extended_tools()
    victim = tools[0]["metadata"]["endpoint"]
    view_functions = {
        name: view for name, view in app.view_functions.items() if name != victim
    }
    unbacked = _tools_without_a_handler(tools, view_functions, _app_rules(app))
    assert unbacked, "deleting a view function must be visible to the check"
    assert any(victim in entry for entry in unbacked)


def test_representative_extended_tools_run_their_handler(client, monkeypatch):
    """One GET per family, dispatched through the real app rather than a stub."""
    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")
    tools = _representative_get_tools()
    assert len(tools) >= 20, "expected the catalogue to span many route families"

    def api_call(method, path, data=None):
        response = client.open(path, method=method, json=data)
        return {"status": response.status_code}

    for mcp_tool in tools:
        result = mcp_extended_tools.invoke_extended_tool(mcp_tool["name"], {}, api_call)
        assert isinstance(result, dict), mcp_tool["name"]
        # A refusal from the dispatcher means the generated tool cannot be
        # called at all, which no manifest diff would show.
        assert "error" not in result, f"{mcp_tool['name']}: {result.get('error')}"
        # 405 means the rule exists but not for this method, which is the
        # generator disagreeing with the app.
        assert result["status"] != 405, (
            f"{mcp_tool['name']} -> GET {mcp_tool['metadata']['path']} is not a GET route"
        )
