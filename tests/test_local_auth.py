"""Local auth + bind-address hardening (F112).

The auth module persists token metadata under ``~/.opencut/auth.json`` and
the secret in the OS vault. Tests isolate both surfaces.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture()
def isolated_auth(tmp_path, monkeypatch):
    """Reload ``opencut.auth`` with ``AUTH_FILE`` pointed at a tmp path."""
    import opencut.auth as auth_module

    monkeypatch.delenv(auth_module.REMOTE_AUTH_TOKEN_FILE_ENV, raising=False)
    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    yield auth_module
    monkeypatch.delenv(auth_module.REMOTE_AUTH_TOKEN_FILE_ENV, raising=False)
    auth_module.clear_token()


def _configure_secret_file(auth_module, monkeypatch, path, value="a" * 64):
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv(auth_module.REMOTE_AUTH_TOKEN_FILE_ENV, str(path))
    return value


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


def test_extract_request_token_reads_header(isolated_auth):
    class HeadersStub:
        def get(self, key, default=None):
            return {"X-OpenCut-Auth": "from-header"}.get(key, default)

    token = isolated_auth.extract_request_token(HeadersStub())
    assert token == "from-header"


def test_extract_request_token_has_no_query_fallback(isolated_auth):
    """``?auth=`` was removed: query tokens leak into logs/history/Referer."""
    import inspect

    class HeadersStub:
        def get(self, key, default=None):
            return default

    token = isolated_auth.extract_request_token(HeadersStub())
    assert token == ""
    # Guard against the fallback quietly returning via a new signature.
    params = inspect.signature(isolated_auth.extract_request_token).parameters
    assert list(params) == ["headers"]


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

    # A valid token in the query string must NOT authorize (log-leak vector).
    resp = client.get(
        f"/system/feature-state?auth={token.token}",
        environ_overrides={"REMOTE_ADDR": "192.168.1.5"},
    )
    assert resp.status_code == 401


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


def test_secret_file_backend_never_creates_json_and_observes_replacement(isolated_auth, monkeypatch, tmp_path):
    secret_path = tmp_path / "remote-auth-token"
    first_value = _configure_secret_file(isolated_auth, monkeypatch, secret_path)

    first = isolated_auth.ensure_token()
    second_value = "b" * 64
    secret_path.write_text(second_value + "\n", encoding="utf-8")
    secret_path.chmod(0o600)
    second = isolated_auth.current_token()

    assert first.token == first_value
    assert second.token == second_value
    assert first.token != second.token
    assert first.label == "secret-file"
    assert not isolated_auth.AUTH_FILE.exists()


@pytest.mark.parametrize("value", ["", "too-short", "a" * 31, "a" * 32 + " b"])
def test_secret_file_rejects_empty_weak_or_whitespace_tokens(isolated_auth, monkeypatch, tmp_path, value):
    path = tmp_path / "bad-token"
    _configure_secret_file(isolated_auth, monkeypatch, path, value)

    with pytest.raises(isolated_auth.RemoteAuthTokenFileError):
        isolated_auth.ensure_token()
    assert not isolated_auth.AUTH_FILE.exists()


def test_secret_file_rejects_non_regular_and_symlink_sources(isolated_auth, monkeypatch, tmp_path):
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    monkeypatch.setenv(isolated_auth.REMOTE_AUTH_TOKEN_FILE_ENV, str(directory))
    with pytest.raises(isolated_auth.RemoteAuthTokenFileError):
        isolated_auth.current_token()

    target = tmp_path / "target-token"
    _configure_secret_file(isolated_auth, monkeypatch, target)
    link = tmp_path / "linked-token"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    monkeypatch.setenv(isolated_auth.REMOTE_AUTH_TOKEN_FILE_ENV, str(link))
    with pytest.raises(isolated_auth.RemoteAuthTokenFileError):
        isolated_auth.current_token()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not available")
def test_secret_file_rejects_group_or_world_permissions(isolated_auth, monkeypatch, tmp_path):
    path = tmp_path / "permissive-token"
    _configure_secret_file(isolated_auth, monkeypatch, path)
    path.chmod(0o640)

    with pytest.raises(isolated_auth.RemoteAuthTokenFileError):
        isolated_auth.current_token()


def test_secret_file_rotation_is_explicit_for_writable_and_read_only_files(isolated_auth, monkeypatch, tmp_path):
    path = tmp_path / "rotatable-token"
    original = _configure_secret_file(isolated_auth, monkeypatch, path)
    rotated = isolated_auth.rotate_token()

    assert rotated.token != original
    assert path.read_text(encoding="utf-8").strip() == rotated.token
    assert not isolated_auth.AUTH_FILE.exists()

    monkeypatch.setattr(isolated_auth, "_token_file_is_writable", lambda _path: False)
    with pytest.raises(isolated_auth.RemoteAuthTokenFileReadOnlyError):
        isolated_auth.rotate_token()


def test_remote_http_accepts_secret_file_token_without_exposing_source(isolated_auth, monkeypatch, tmp_path, client):
    secret_path = tmp_path / "container-secret"
    value = _configure_secret_file(isolated_auth, monkeypatch, secret_path)
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")

    denied = client.get(
        "/system/feature-state",
        environ_overrides={"REMOTE_ADDR": "192.168.1.5"},
    )
    accepted = client.get(
        "/system/feature-state",
        headers={"X-OpenCut-Auth": value},
        environ_overrides={"REMOTE_ADDR": "192.168.1.5"},
    )
    info = client.get("/auth/info", environ_overrides={"REMOTE_ADDR": "192.168.1.5"})

    assert denied.status_code == 401
    assert accepted.status_code == 200
    payload = info.get_json()
    assert payload["credential_storage"] == "secret_file"
    assert payload["token_file"] is None
    assert payload["metadata_file"] is None
    assert value not in json.dumps(payload)
    assert str(secret_path) not in json.dumps(payload)


def test_read_only_secret_rotation_returns_stable_safe_error(isolated_auth, monkeypatch, tmp_path, client):
    secret_path = tmp_path / "read-only-secret"
    value = _configure_secret_file(isolated_auth, monkeypatch, secret_path)
    monkeypatch.setattr(isolated_auth, "_token_file_is_writable", lambda _path: False)
    csrf = client.get("/health").get_json()["csrf_token"]

    response = client.post(
        "/auth/rotate",
        headers={"X-OpenCut-Token": csrf},
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["code"] == "REMOTE_AUTH_TOKEN_FILE_READ_ONLY"
    assert value not in json.dumps(payload)
    assert str(secret_path) not in json.dumps(payload)


def test_secret_file_cli_never_prints_value_or_path(isolated_auth, monkeypatch, tmp_path, capsys):
    import opencut.server as server

    secret_path = tmp_path / "cli-secret"
    value = _configure_secret_file(isolated_auth, monkeypatch, secret_path)
    monkeypatch.setattr(server.sys, "argv", ["opencut-server", "--print-auth"])

    assert server.main() == 0
    output = capsys.readouterr().out
    assert "value intentionally not displayed" in output
    assert value not in output
    assert str(secret_path) not in output


def test_remote_server_can_boot_from_secret_file_without_desktop_vault(isolated_auth, monkeypatch, tmp_path):
    import opencut.credential_store as credential_store
    import opencut.server as server

    class UnavailableBackend:
        priority = 0

        def get_password(self, *_args):
            raise RuntimeError("vault unavailable")

    secret_path = tmp_path / "server-secret"
    _configure_secret_file(isolated_auth, monkeypatch, secret_path)
    monkeypatch.setattr(credential_store, "_backend_override", UnavailableBackend())
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    monkeypatch.setattr(server.sys, "argv", ["opencut-server", "--host", "0.0.0.0"])
    started = []
    monkeypatch.setattr(server, "run_server", lambda **kwargs: started.append(kwargs))

    assert server.main() == 0
    assert started == [{"host": "0.0.0.0", "port": 5679, "debug": False}]
