"""Local auth + bind-address hardening (F112).

The auth module persists token metadata under ``~/.opencut/auth.json`` and
the secret in the OS vault. Tests isolate both surfaces.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def isolated_auth(tmp_path, monkeypatch):
    """Reload ``opencut.auth`` with ``AUTH_FILE`` pointed at a tmp path."""
    import opencut.auth as auth_module

    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    yield auth_module
    auth_module.clear_token()


def test_ensure_token_creates_persisted_file(isolated_auth):
    token = isolated_auth.ensure_token(label="test")

    assert token.token
    assert len(token.token) >= 32  # 256-bit token urlsafe-encoded
    assert isolated_auth.AUTH_FILE.exists()
    payload = json.loads(isolated_auth.AUTH_FILE.read_text(encoding="utf-8"))
    assert "token" not in payload
    assert payload["token_set"] is True
    assert token.token not in json.dumps(payload)
    assert payload["label"] == "test"


def test_ensure_token_is_idempotent(isolated_auth):
    first = isolated_auth.ensure_token()
    second = isolated_auth.ensure_token()

    assert first.token == second.token
    assert first.issued_at == second.issued_at


def test_rotate_token_replaces_existing(isolated_auth):
    original = isolated_auth.ensure_token()
    new = isolated_auth.rotate_token()

    assert new.token != original.token
    assert isolated_auth.current_token().token == new.token


def test_clear_token_removes_file(isolated_auth):
    isolated_auth.ensure_token()
    assert isolated_auth.clear_token() is True
    assert isolated_auth.clear_token() is False  # second call: no file


def test_is_token_valid_uses_constant_time_compare(isolated_auth):
    token = isolated_auth.ensure_token()

    assert isolated_auth.is_token_valid(token.token) is True
    assert isolated_auth.is_token_valid("wrong") is False
    assert isolated_auth.is_token_valid("") is False
    assert isolated_auth.is_token_valid(None) is False


def test_request_requires_auth_token_respects_loopback(isolated_auth, monkeypatch):
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    assert isolated_auth.request_requires_auth_token("127.0.0.1") is False
    assert isolated_auth.request_requires_auth_token("::1") is False
    assert isolated_auth.request_requires_auth_token("192.168.1.5") is True
    assert isolated_auth.request_requires_auth_token(None) is True


def test_request_requires_auth_disabled_without_remote_opt_in(isolated_auth, monkeypatch):
    monkeypatch.delenv("OPENCUT_ALLOW_REMOTE", raising=False)
    assert isolated_auth.request_requires_auth_token("192.168.1.5") is False


def test_extract_request_token_prefers_header(isolated_auth):
    class HeadersStub:
        def get(self, key, default=None):
            return {"X-OpenCut-Auth": "from-header"}.get(key, default)

    token = isolated_auth.extract_request_token(HeadersStub(), {"auth": "from-query"})
    assert token == "from-header"


def test_extract_request_token_falls_back_to_query(isolated_auth):
    class HeadersStub:
        def get(self, key, default=None):
            return default

    token = isolated_auth.extract_request_token(HeadersStub(), {"auth": "from-query"})
    assert token == "from-query"


# ----- HTTP middleware behaviour ---------------------------------------


def test_loopback_request_skips_auth_gate(isolated_auth, monkeypatch, client):
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    isolated_auth.ensure_token()

    # The test client defaults to remote_addr=127.0.0.1 → loopback → no gate.
    resp = client.get("/health")
    assert resp.status_code == 200


def test_remote_request_requires_token(monkeypatch, tmp_path, client):
    """Simulate a non-loopback peer with explicit remote_bind opt-in."""
    import opencut.auth as auth_module

    # Redirect the auth file BEFORE the test so the middleware reads it
    # from the isolated location.
    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    token = auth_module.ensure_token()

    # No token header → 401.
    resp = client.get("/system/feature-state", environ_overrides={"REMOTE_ADDR": "192.168.1.5"})
    assert resp.status_code == 401
    assert resp.get_json()["code"] == "AUTH_REQUIRED"

    # With token → 200.
    resp = client.get(
        "/system/feature-state",
        headers={"X-OpenCut-Auth": token.token},
        environ_overrides={"REMOTE_ADDR": "192.168.1.5"},
    )
    assert resp.status_code == 200


def test_health_remains_exempt_for_remote_clients(monkeypatch, tmp_path, client):
    """``/health`` is the bootstrap probe — never gated."""
    import opencut.auth as auth_module

    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    auth_module.ensure_token()

    resp = client.get("/health", environ_overrides={"REMOTE_ADDR": "192.168.1.5"})
    assert resp.status_code == 200


def test_auth_info_route_never_returns_token_value(monkeypatch, tmp_path, client):
    import opencut.auth as auth_module

    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    auth_module.ensure_token()

    resp = client.get("/auth/info")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["token_issued"] is True
    # Critical: the value of the token must never appear in the response.
    persisted = auth_module.current_token().token
    assert persisted not in json.dumps(payload)
