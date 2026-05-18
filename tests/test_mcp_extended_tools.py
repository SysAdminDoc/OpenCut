"""F194 extended MCP route-tool catalogue tests."""

from __future__ import annotations

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
    assert committed["response_schema_count"] >= 70
    assert committed["tool_prefix"] == "opencut_route_"
    assert "POST" in committed["method_counts"]
    assert "GET" in committed["method_counts"]


def test_extended_tools_are_opt_in_and_do_not_change_curated_default(monkeypatch):
    monkeypatch.delenv(mcp_extended_tools.EXTENDED_MCP_ENV, raising=False)

    assert len(mcp_server.MCP_TOOLS) == 39
    assert len(mcp_server.get_mcp_tools()) == 39

    extended_count = len(mcp_server.get_mcp_tools(include_extended=True))
    assert extended_count == 39 + len(mcp_extended_tools.get_extended_tools())
    assert extended_count >= 1000

    monkeypatch.setenv(mcp_extended_tools.EXTENDED_MCP_ENV, "1")
    assert len(mcp_server.get_mcp_tools()) == extended_count


def test_extended_tool_names_are_unique_and_tagged_lower_priority():
    tools = mcp_extended_tools.get_extended_tools()
    names = [tool["name"] for tool in tools]

    assert len(names) == len(set(names))
    assert all(name.startswith("opencut_route_") for name in names)
    for mcp_tool in tools[:25]:
        assert mcp_tool["metadata"]["generated"] is True
        assert mcp_tool["metadata"]["priority"] == "extended"
        assert "lower-priority" in mcp_tool["description"]


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
