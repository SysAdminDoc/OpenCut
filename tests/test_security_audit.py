"""Regression tests for security rejection audit logging."""

from __future__ import annotations

import json


def _read_entries(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_csrf_rejection_writes_security_audit_without_token_value(client, tmp_path, monkeypatch):
    audit_path = tmp_path / "security_audit.jsonl"
    monkeypatch.setenv("OPENCUT_SECURITY_AUDIT_LOG", str(audit_path))

    response = client.post(
        "/info",
        json={},
        headers={"X-OpenCut-Token": "attacker-token"},
    )

    assert response.status_code == 403
    entries = _read_entries(audit_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["schema"] == "opencut.security-audit.v1"
    assert entry["event"] == "csrf_rejected"
    assert entry["reason"] == "Invalid or missing CSRF token"
    assert entry["method"] == "POST"
    assert entry["path"] == "/info"
    assert entry["request_id"] == response.headers["X-Request-ID"]
    assert entry["metadata"] == {"token_present": True}
    assert "attacker-token" not in audit_path.read_text(encoding="utf-8")


def test_path_traversal_rejection_writes_security_audit(tmp_path, monkeypatch):
    from opencut.security import validate_path

    audit_path = tmp_path / "security_audit.jsonl"
    monkeypatch.setenv("OPENCUT_SECURITY_AUDIT_LOG", str(audit_path))

    try:
        validate_path("../secret.txt")
    except ValueError as exc:
        assert str(exc) == "Path traversal blocked"
    else:
        raise AssertionError("validate_path accepted traversal input")

    entries = _read_entries(audit_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["event"] == "path_validation_rejected"
    assert entry["reason"] == "Path traversal blocked"
    assert entry["metadata"]["path_preview"] == "../secret.txt"
    assert len(entry["metadata"]["path_sha256"]) == 64


def test_rate_limit_rejection_writes_security_audit(tmp_path, monkeypatch):
    from opencut.security import rate_limit, rate_limit_release

    audit_path = tmp_path / "security_audit.jsonl"
    monkeypatch.setenv("OPENCUT_SECURITY_AUDIT_LOG", str(audit_path))
    key = f"security-audit-{tmp_path.name}"

    assert rate_limit(key, max_concurrent=1) is True
    try:
        assert rate_limit(key, max_concurrent=1) is False
    finally:
        rate_limit_release(key)

    entries = _read_entries(audit_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["event"] == "rate_limit_rejected"
    assert entry["reason"] == "Concurrency limit reached"
    assert entry["metadata"] == {
        "key": key,
        "current": 1,
        "max_concurrent": 1,
    }


def test_system_audit_log_endpoint_returns_recent_events(client, tmp_path, monkeypatch):
    from opencut.security_audit import record_security_event

    audit_path = tmp_path / "security_audit.jsonl"
    monkeypatch.setenv("OPENCUT_SECURITY_AUDIT_LOG", str(audit_path))
    record_security_event("first_event", "first")
    record_security_event("second_event", "second")

    response = client.get("/system/audit-log?limit=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["log_path"] == str(audit_path)
    assert payload["events"][0]["event"] == "second_event"


def test_remote_auth_rejection_writes_security_audit(tmp_path, monkeypatch):
    from opencut.config import OpenCutConfig
    from opencut.server import create_app

    audit_path = tmp_path / "security_audit.jsonl"
    monkeypatch.setenv("OPENCUT_SECURITY_AUDIT_LOG", str(audit_path))
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    app = create_app(config=OpenCutConfig())
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get(
        "/system/status",
        environ_base={"REMOTE_ADDR": "192.0.2.10"},
    )

    assert response.status_code == 401
    entries = _read_entries(audit_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["event"] == "auth_token_rejected"
    assert entry["reason"] == "Missing or invalid X-OpenCut-Auth token"
    assert entry["method"] == "GET"
    assert entry["path"] == "/system/status"
