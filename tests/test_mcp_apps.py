"""MCP Apps progressive-enhancement and redaction contracts."""

import json

from opencut import mcp_apps, mcp_server


def _apps_meta():
    return {
        mcp_server.META_PROTOCOL_VERSION: mcp_server.LATEST_PROTOCOL_VERSION,
        mcp_server.META_CLIENT_CAPABILITIES: {
            "extensions": {
                mcp_server.MCP_APPS_EXTENSION: {
                    "mimeTypes": [mcp_apps.RESOURCE_MIME_TYPE],
                }
            }
        },
    }


def _rpc(method, params, msg_id=1):
    return mcp_server.dispatch_jsonrpc(
        {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
    )


def test_unsupported_clients_keep_text_tools_and_no_resources():
    params = {"_meta": {mcp_server.META_PROTOCOL_VERSION: mcp_server.LATEST_PROTOCOL_VERSION}}
    tools = _rpc("tools/list", params)["result"]["tools"]
    job_tool = next(tool for tool in tools if tool["name"] == "opencut_job_status")
    assert "_meta" not in job_tool
    assert _rpc("resources/list", params)["result"]["resources"] == []


def test_apps_clients_receive_versioned_resource_and_tool_metadata():
    params = {"_meta": _apps_meta()}
    tools = _rpc("tools/list", params)["result"]["tools"]
    job_tool = next(tool for tool in tools if tool["name"] == "opencut_job_status")
    action_tool = next(tool for tool in tools if tool["name"] == "opencut_review_action")
    assert job_tool["_meta"]["ui"]["resourceUri"] == mcp_apps.RESOURCE_URI
    assert action_tool["_meta"]["ui"]["visibility"] == ["app"]
    resources = _rpc("resources/list", params)["result"]["resources"]
    assert resources[0]["uri"] == mcp_apps.RESOURCE_URI
    assert resources[0]["mimeType"] == mcp_apps.RESOURCE_MIME_TYPE
    assert resources[0]["_meta"]["ui"]["csp"]["connectDomains"] == []

    resource = _rpc(
        "resources/read",
        {"_meta": _apps_meta(), "uri": mcp_apps.RESOURCE_URI},
    )["result"]["contents"][0]
    assert resource["mimeType"] == mcp_apps.RESOURCE_MIME_TYPE
    assert "default-src 'none'" in resource["text"]
    assert "fetch(" not in resource["text"]
    assert "WebSocket" not in resource["text"]

    direct_capabilities = {
        "capabilities": {
            "extensions": {
                mcp_server.MCP_APPS_EXTENSION: {
                    "mimeTypes": [mcp_apps.RESOURCE_MIME_TYPE]
                }
            }
        }
    }
    direct_tools = _rpc("tools/list", direct_capabilities)["result"]["tools"]
    assert next(
        tool for tool in direct_tools if tool["name"] == "opencut_job_status"
    )["_meta"]["ui"]["resourceUri"] == mcp_apps.RESOURCE_URI


def test_resource_read_requires_capability_and_rejects_unknown_uri():
    unsupported = _rpc(
        "resources/read",
        {"_meta": {mcp_server.META_PROTOCOL_VERSION: mcp_server.LATEST_PROTOCOL_VERSION}, "uri": mcp_apps.RESOURCE_URI},
    )
    assert unsupported["error"]["code"] == mcp_server.ERROR_MISSING_REQUIRED_CLIENT_CAPABILITY
    unknown = _rpc(
        "resources/read",
        {"_meta": _apps_meta(), "uri": "ui://opencut/other.html"},
    )
    assert unknown["error"]["code"] == -32602


def test_app_tool_result_is_structured_and_redacted(monkeypatch):
    def fake_api(method, path, data=None):
        assert (method, path) == ("POST", "/review/bundle")
        return {
            "job_id": "abcdef123456",
            "output_path": r"C:\secret\review.zip",
            "manifest": {"path": r"C:\secret\review.zip", "entries": ["media.mp4"]},
            "network": False,
        }

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    response = _rpc(
        "tools/call",
        {
            "_meta": _apps_meta(),
            "name": "opencut_review_bundle",
            "arguments": {"output_path": "out/review.zip"},
        },
    )["result"]
    assert response["structuredContent"]["capabilities"]["local_paths"] is False
    data = response["structuredContent"]["data"]
    assert data["output_path"] == "[redacted local path]"
    assert data["manifest"]["path"] == "[redacted local path]"
    assert r"C:\secret" not in json.dumps(response)


def test_apps_stale_job_keeps_capability_scoped_fallback(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_api",
        lambda method, path, data=None: {"error": "Job not found", "job_id": "abcdef123456"},
    )
    response = _rpc(
        "tools/call",
        {
            "_meta": _apps_meta(),
            "name": "opencut_job_status",
            "arguments": {"job_id": "abcdef123456"},
        },
    )["result"]
    assert response["structuredContent"]["capabilities"]["network"] is False
    assert response["structuredContent"]["data"]["error"] == "Job not found"


def test_app_review_actions_are_allowlisted(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    mcp_server.handle_tool_call(
        "opencut_review_action", {"action": "cancel", "job_id": "abcdef123456"}
    )
    mcp_server.handle_tool_call(
        "opencut_review_action",
        {"action": "approve", "workflow_id": "wf-1", "actor": "editor"},
    )
    assert calls == [
        ("POST", "/cancel/abcdef123456", {}),
        (
            "POST",
            "/api/approval/advance",
            {"workflow_id": "wf-1", "action": "approve", "actor": "editor", "notes": ""},
        ),
    ]
    invalid = mcp_server.handle_tool_call(
        "opencut_review_action", {"action": "delete", "job_id": "abcdef123456"}
    )
    assert invalid == {"error": "Invalid action for opencut_review_action"}
    assert len(calls) == 2
