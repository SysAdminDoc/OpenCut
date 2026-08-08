"""MCP Tasks adapter tests for durable OpenCut jobs."""

import json

from opencut import mcp_server


def _task_meta():
    return {
        mcp_server.META_PROTOCOL_VERSION: mcp_server.LATEST_PROTOCOL_VERSION,
        mcp_server.META_CLIENT_CAPABILITIES: {
            "extensions": {mcp_server.TASKS_EXTENSION: {}},
        },
    }


def _rpc(method, params):
    return mcp_server.dispatch_jsonrpc({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    })


def test_discovery_advertises_tasks_extension():
    capabilities = _rpc("server/discover", {"_meta": _task_meta()})["result"]["capabilities"]
    assert capabilities["extensions"] == {mcp_server.TASKS_EXTENSION: {}}


def test_non_task_client_keeps_legacy_job_result(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        return {"job_id": "abcdef123456"}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    response = _rpc(
        "tools/call",
        {
            "_meta": {
                mcp_server.META_PROTOCOL_VERSION: mcp_server.LATEST_PROTOCOL_VERSION,
            },
            "name": "opencut_silence_remove",
            "arguments": {"filepath": "C:\\clip.mp4"},
        },
    )

    result = response["result"]
    assert result["resultType"] == "complete"
    assert json.loads(result["content"][0]["text"])["job_id"] == "abcdef123456"
    assert calls == [("POST", "/silence", {"filepath": "C:\\clip.mp4"})]


def test_task_client_receives_durable_task_handle(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        if method == "POST":
            return {"job_id": "abcdef123456"}
        return {
            "id": "abcdef123456",
            "status": "running",
            "progress": 25,
            "message": "Working",
            "created": 1_700_000_000,
        }

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    response = _rpc(
        "tools/call",
        {
            "_meta": _task_meta(),
            "name": "opencut_silence_remove",
            "arguments": {"filepath": "C:\\clip.mp4"},
        },
    )

    result = response["result"]
    assert result["resultType"] == "task"
    assert result["taskId"] == "abcdef123456"
    assert result["status"] == "working"
    assert result["pollIntervalMs"] == mcp_server.TASK_POLL_INTERVAL_MS
    assert calls == [
        ("POST", "/silence", {"filepath": "C:\\clip.mp4"}),
        ("GET", "/jobs/abcdef123456", None),
    ]


def test_tasks_get_returns_completed_tool_result(monkeypatch):
    def fake_api(method, path, data=None):
        assert (method, path, data) == ("GET", "/jobs/abcdef123456", None)
        return {
            "id": "abcdef123456",
            "status": "complete",
            "result": {"output": "rendered.mp4"},
            "message": "Done",
            "created": 1_700_000_000,
            "completed_at": 1_700_000_005,
        }

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    result = _rpc(
        "tasks/get",
        {"_meta": _task_meta(), "taskId": "abcdef123456"},
    )["result"]

    assert result["resultType"] == "complete"
    assert result["status"] == "completed"
    assert result["result"]["content"][0]["type"] == "text"
    assert json.loads(result["result"]["content"][0]["text"]) == {
        "output": "rendered.mp4"
    }


def test_tasks_cancel_acknowledges_and_forwards_cooperative_cancel(monkeypatch):
    calls = []

    def fake_api(method, path, data=None):
        calls.append((method, path, data))
        if method == "GET":
            return {"id": "abcdef123456", "status": "running", "created": 1_700_000_000}
        return {"job_id": "abcdef123456", "status": "cancelled"}

    monkeypatch.setattr(mcp_server, "_api", fake_api)
    result = _rpc(
        "tasks/cancel",
        {"_meta": _task_meta(), "taskId": "abcdef123456"},
    )["result"]

    assert result == {
        "resultType": "complete",
        "_meta": {mcp_server.META_SERVER_INFO: mcp_server.server_info()},
    }
    assert calls == [
        ("GET", "/jobs/abcdef123456", None),
        ("POST", "/cancel/abcdef123456", {}),
    ]


def test_tasks_update_is_empty_ack_for_known_job(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_api",
        lambda method, path, data=None: {
            "id": "abcdef123456",
            "status": "running",
            "created": 1_700_000_000,
        },
    )
    result = _rpc(
        "tasks/update",
        {
            "_meta": _task_meta(),
            "taskId": "abcdef123456",
            "inputResponses": {"ignored": {"value": True}},
        },
    )["result"]
    assert result["resultType"] == "complete"
    assert result["_meta"][mcp_server.META_SERVER_INFO] == mcp_server.server_info()


def test_unknown_task_returns_jsonrpc_error(monkeypatch):
    monkeypatch.setattr(mcp_server, "_api", lambda *_args, **_kwargs: {"error": "not found"})
    response = _rpc(
        "tasks/get",
        {"_meta": _task_meta(), "taskId": "abcdef123456"},
    )
    assert response["error"]["code"] == -32004
    assert response["error"]["data"]["taskId"] == "abcdef123456"
