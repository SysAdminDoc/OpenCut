"""F471: /health hands the CSRF token only to callers holding this user's secret.

Loopback is shared by every account on the machine, so a missing ``Origin`` or
a copy of the server's own origin proves nothing about who is asking. The
secret file (owner-only, in the data folder of the account that started the
server) is the proof, and every first-party client presents it: the CEP panel,
the UXP panel through a copy in its plugin data folder, the CLI, the MCP server
and the stale-server shutdown in ``opencut.pid``.
"""

from __future__ import annotations

import io
import json
import os
import socket
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from opencut import panel_bootstrap
from opencut.panel_bootstrap import BOOTSTRAP_HEADER, BOOTSTRAP_PATH_ENV, UXP_STORAGE_ENV

MUTATION = ("/settings/loudness-target", {"target_lufs": -14.0})


@pytest.fixture
def secret(tmp_path, monkeypatch):
    monkeypatch.setenv(BOOTSTRAP_PATH_ENV, str(tmp_path / "panel_bootstrap.token"))
    panel_bootstrap.clear_cached_secret()
    yield panel_bootstrap.ensure_bootstrap_secret()
    panel_bootstrap.clear_cached_secret()


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    import opencut.auth as auth_module

    monkeypatch.delenv(auth_module.REMOTE_AUTH_TOKEN_FILE_ENV, raising=False)
    monkeypatch.setattr(auth_module, "AUTH_FILE", tmp_path / "auth.json")
    auth_module.clear_token()
    yield auth_module
    auth_module.clear_token()


@pytest.fixture
def own_port():
    """A loopback port this test process is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]


def _system_port() -> int:
    """A port the Windows System process (PID 4, another account) listens on."""
    if os.name != "nt":
        pytest.skip("needs the Windows System process")
    import psutil

    for conn in psutil.net_connections(kind="tcp"):
        if conn.status == psutil.CONN_LISTEN and conn.pid == 4 and conn.laddr.ip in ("0.0.0.0", "127.0.0.1"):
            return conn.laddr.port
    pytest.skip("nothing on this PC listens as the System account")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _withheld():
    # An empty header keeps the test client from adding the secret itself.
    return {BOOTSTRAP_HEADER: ""}


class TestHealthNeedsTheSecret:
    def test_no_origin_and_no_secret_gets_capabilities_but_no_token(self, client, secret):
        resp = client.get("/health", headers=_withheld())

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        assert isinstance(body["capabilities"], dict) and body["capabilities"]
        assert "csrf_token" not in body
        assert "X-OpenCut-Token" not in resp.headers

    def test_caller_without_the_secret_cannot_mutate(self, client, secret):
        health = client.get("/health", headers=_withheld()).get_json()
        path, payload = MUTATION

        assert client.post(path, json=payload).status_code == 403
        guessed = health.get("csrf_token") or "0" * 64
        assert client.post(path, json=payload, headers={"X-OpenCut-Token": guessed}).status_code == 403

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost", "http://127.0.0.1:5679", "http://localhost:5679"],
    )
    def test_forged_same_origin_gets_no_token(self, client, secret, origin):
        resp = client.get("/health", headers={"Origin": origin})
        assert "csrf_token" not in resp.get_json()

    def test_wrong_secret_without_origin_gets_no_token(self, client, secret):
        resp = client.get("/health", headers={BOOTSTRAP_HEADER: "0" * 64})
        assert "csrf_token" not in resp.get_json()

    def test_secret_without_origin_gets_a_working_token(self, client, secret):
        token = client.get("/health", headers={BOOTSTRAP_HEADER: secret}).get_json()["csrf_token"]
        path, payload = MUTATION

        resp = client.post(path, json=payload, headers={"X-OpenCut-Token": token})
        assert resp.status_code != 403

    def test_remote_auth_token_gets_a_token(self, client, secret, isolated_auth):
        auth_token = isolated_auth.ensure_token().token
        headers = {**_withheld(), isolated_auth.AUTH_HEADER: auth_token}

        assert client.get("/health", headers=headers).get_json().get("csrf_token")

        headers[isolated_auth.AUTH_HEADER] = "x" * len(auth_token)
        assert "csrf_token" not in client.get("/health", headers=headers).get_json()

    def test_configured_cors_origin_still_bootstraps(self, app, client, secret):
        app.config["OPENCUT"].cors_origins = ["https://studio.example"]
        resp = client.get("/health", headers={"Origin": "https://studio.example"})
        assert resp.get_json().get("csrf_token")

    @pytest.mark.parametrize("origin", ["null", "NULL", "null/", "file://", "file:///", "FILE:///", "file://host"])
    def test_null_and_file_origins_stay_refused_even_when_listed(self, app, client, secret, monkeypatch, origin):
        """The blocklist was checked before the trailing slash came off, so
        ``file:///`` and ``null/`` matched a listed ``file://`` or ``null``."""
        monkeypatch.setattr(app.config["OPENCUT"], "cors_origins", ["null", "file://", "file:///"])
        resp = client.get("/health", headers={"Origin": origin})
        assert "csrf_token" not in resp.get_json()


class TestBootstrapHeaders:
    @pytest.mark.parametrize(
        "base_url",
        ["http://127.0.0.1:{port}", "http://localhost:{port}", "http://[::1]:{port}", "http://127.0.0.2:{port}/"],
    )
    def test_loopback_backends_run_by_this_account_get_the_secret(self, secret, own_port, base_url):
        url = base_url.format(port=own_port)
        assert panel_bootstrap.bootstrap_headers(url) == {BOOTSTRAP_HEADER: secret}

    def test_a_listener_in_another_account_never_sees_it(self, secret):
        """A process in another account that took the port first used to be
        handed the secret, and could then replay it to the real server."""
        port = _system_port()
        assert panel_bootstrap.listener_is_ours(f"http://127.0.0.1:{port}") is False
        assert panel_bootstrap.bootstrap_headers(f"http://127.0.0.1:{port}") == {}

    def test_nothing_listening_gets_nothing(self, secret):
        assert panel_bootstrap.bootstrap_headers(f"http://127.0.0.1:{_free_port()}") == {}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://192.168.1.20:5679",
            "http://studio.lan:5679",
            "http://127.0.0.1.evil.example:5679",
            "http://localhost.evil.example",
            "not a url",
            "",
        ],
    )
    def test_other_backends_never_see_it(self, secret, base_url):
        assert panel_bootstrap.bootstrap_headers(base_url) == {}


class TestUxpPanelCopy:
    @pytest.fixture
    def storage(self, tmp_path, monkeypatch):
        root = tmp_path / "PluginsStorage"
        monkeypatch.setenv(UXP_STORAGE_ENV, str(root))
        monkeypatch.setattr(panel_bootstrap, "_last_mirror_at", 0.0)
        return root

    def _data_dir(self, root: Path, host: str, version: str, kind: str, plugin: str) -> Path:
        path = root / host / version / kind / plugin / "PluginData"
        path.mkdir(parents=True)
        return path

    def test_copies_only_into_opencut_premiere_data_folders(self, secret, storage):
        ours = self._data_dir(storage, "PPRO", "25.0.0", "Developer", "com.opencut.uxp")
        external = self._data_dir(storage, "PPRO", "26.0.0", "External", "com.opencut.uxp")
        other_plugin = self._data_dir(storage, "PPRO", "25.0.0", "Developer", "com.example.other")
        photoshop = self._data_dir(storage, "PHSP", "26.0.0", "External", "com.opencut.uxp")
        not_run_yet = storage / "PPRO" / "26.1.0" / "External" / "com.opencut.uxp"
        not_run_yet.mkdir(parents=True)

        written = panel_bootstrap.mirror_secret_to_uxp(force=True)

        assert sorted(written) == sorted(
            [str(ours / "panel_bootstrap.token"), str(external / "panel_bootstrap.token")]
        )
        assert (ours / "panel_bootstrap.token").read_text(encoding="utf-8").strip() == secret
        assert not any(other_plugin.iterdir())
        assert not any(photoshop.iterdir())
        assert not (not_run_yet / "PluginData").exists(), "folders are never created"

        assert panel_bootstrap.mirror_secret_to_uxp(force=True) == [], "an unchanged copy isn't rewritten"

    def test_refused_health_refreshes_the_copy(self, client, secret, storage):
        # Premiere ran the panel for the first time after the server started.
        data_dir = self._data_dir(storage, "PPRO", "25.0.0", "External", "com.opencut.uxp")

        resp = client.get("/health", headers=_withheld())

        assert "csrf_token" not in resp.get_json()
        copy = data_dir / "panel_bootstrap.token"
        assert copy.read_text(encoding="utf-8").strip() == secret
        token = client.get("/health", headers={BOOTSTRAP_HEADER: copy.read_text(encoding="utf-8").strip()})
        assert token.get_json().get("csrf_token")

    def test_refreshes_are_throttled(self, secret, storage):
        data_dir = self._data_dir(storage, "PPRO", "25.0.0", "External", "com.opencut.uxp")
        assert panel_bootstrap.mirror_secret_to_uxp() == [str(data_dir / "panel_bootstrap.token")]
        (data_dir / "panel_bootstrap.token").unlink()
        assert panel_bootstrap.mirror_secret_to_uxp() == []


class _FlaskBridge:
    """Route ``urllib.request.urlopen`` calls to the Flask test client."""

    def __init__(self, client):
        self.client = client
        self.requests = []

    def __call__(self, req, timeout=None):
        if isinstance(req, str):
            req = urllib.request.Request(req)
        self.requests.append(req)
        parts = urllib.parse.urlsplit(req.full_url)
        headers = dict(req.header_items())
        # Only what the caller sent. The conftest default doesn't apply here.
        if BOOTSTRAP_HEADER.lower() not in {key.lower() for key in headers}:
            headers[BOOTSTRAP_HEADER] = ""
        resp = self.client.open(
            parts.path,
            method=req.get_method(),
            headers=headers,
            data=req.data,
        )
        return _BridgeResponse(resp.get_data())


class _BridgeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestFirstPartyClients:
    def test_cli_presents_the_secret(self, client, secret, own_port, monkeypatch):
        from opencut import cli

        bridge = _FlaskBridge(client)
        monkeypatch.setattr(cli.urllib.request, "urlopen", bridge)

        token = cli._fetch_cli_csrf_token(f"http://127.0.0.1:{own_port}", 5)

        assert token
        path, payload = MUTATION
        resp = client.post(path, json=payload, headers={"X-OpenCut-Token": token})
        assert resp.status_code != 403

    def test_mcp_server_presents_the_secret(self, client, secret, own_port, monkeypatch):
        from opencut import mcp_server

        monkeypatch.setattr(mcp_server, "BACKEND_URL", f"http://127.0.0.1:{own_port}")
        monkeypatch.setattr(mcp_server, "_csrf_token", "")
        monkeypatch.setattr(mcp_server, "_csrf_fetched_at", 0.0)
        monkeypatch.setattr(mcp_server.urllib.request, "urlopen", _FlaskBridge(client))

        mcp_server._refresh_csrf()

        assert mcp_server._csrf_token
        assert mcp_server._mcp_http_csrf_token_is_valid(mcp_server._csrf_token)

    def test_mcp_server_keeps_the_secret_from_a_remote_backend(self, client, secret, monkeypatch):
        from opencut import mcp_server

        bridge = _FlaskBridge(client)
        monkeypatch.setattr(mcp_server, "BACKEND_URL", "http://studio.lan:5679")
        monkeypatch.setattr(mcp_server, "_csrf_token", "")
        monkeypatch.setattr(mcp_server, "_csrf_fetched_at", 0.0)
        monkeypatch.setattr(mcp_server.urllib.request, "urlopen", bridge)

        mcp_server._refresh_csrf()

        sent = {key.lower(): value for key, value in bridge.requests[0].header_items()}
        assert BOOTSTRAP_HEADER.lower() not in sent
        assert mcp_server._csrf_token == ""

    def test_stale_server_shutdown_keeps_the_secret_from_another_account(self, secret, monkeypatch):
        """Startup asks whatever holds the port to shut down, and that can be
        a process in another account squatting on it."""
        from opencut import pid

        port = _system_port()
        sent = []

        def urlopen(req, timeout=None):
            sent.append(req)
            raise OSError("refused")

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

        assert pid._kill_via_shutdown_endpoint("127.0.0.1", port) is False
        health = [req for req in sent if req.full_url.endswith("/health")]
        assert health, "the shutdown still asks /health for a token"
        assert BOOTSTRAP_HEADER.lower() not in {key.lower() for key, _ in health[0].header_items()}

    def test_stale_server_shutdown_presents_the_secret(self, client, secret, own_port, monkeypatch):
        from opencut import pid

        bridge = _FlaskBridge(client)
        shutdowns = []

        def urlopen(req, timeout=None):
            # The real /shutdown route calls os._exit, so it never reaches Flask.
            if req.full_url.endswith("/shutdown"):
                shutdowns.append(req)
                return _BridgeResponse(json.dumps({"status": "shutting_down", "pid": 1}).encode())
            return bridge(req, timeout)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

        assert pid._kill_via_shutdown_endpoint("127.0.0.1", own_port) is True

        assert bridge.requests[0].get_header(BOOTSTRAP_HEADER.capitalize()) == secret
        sent_token = shutdowns[0].get_header("X-opencut-token")
        assert sent_token, "the shutdown request must carry the CSRF token it was handed"
        path, payload = MUTATION
        assert client.post(path, json=payload, headers={"X-OpenCut-Token": sent_token}).status_code != 403
