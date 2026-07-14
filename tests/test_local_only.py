"""
Tests for the OPENCUT_LOCAL_ONLY privacy mode.

Verifies that cloud-capable modules are blocked when local-only mode is active.
"""

import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
import requests


@pytest.fixture
def local_only_app():
    """Create an app with local_only=True."""
    from opencut.config import OpenCutConfig
    from opencut.server import create_app
    cfg = OpenCutConfig(local_only=True)
    flask_app = create_app(config=cfg, testing=True)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def local_only_client(local_only_app):
    return local_only_app.test_client()


@pytest.fixture
def local_only_csrf(local_only_client):
    resp = local_only_client.get("/health")
    return resp.get_json().get("csrf_token", "")


# ---- Config layer ----

def test_config_local_only_default():
    from opencut.config import OpenCutConfig
    cfg = OpenCutConfig()
    assert cfg.local_only is False


def test_config_local_only_from_env():
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        from opencut.config import OpenCutConfig
        cfg = OpenCutConfig.from_env()
        assert cfg.local_only is True


def test_is_local_only_env():
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "true"}):
        from opencut.config import is_local_only
        assert is_local_only() is True


def test_is_local_only_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENCUT_LOCAL_ONLY", None)
        from opencut import user_data
        from opencut.config import is_local_only
        with patch.object(user_data, "read_user_file", return_value={}):
            assert is_local_only() is False


def test_require_network_allowed_passes():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENCUT_LOCAL_ONLY", None)
        from opencut import user_data
        from opencut.config import require_network_allowed
        with patch.object(user_data, "read_user_file", return_value={}):
            require_network_allowed("test feature")


def test_require_network_allowed_blocks():
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        from opencut.config import require_network_allowed
        with pytest.raises(RuntimeError, match="Local-only mode"):
            require_network_allowed("Cloud API", "local alternative")


def test_runtime_guard_blocks_dns_before_resolution():
    from opencut.network_policy import LocalOnlyNetworkError

    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        with pytest.raises(LocalOnlyNetworkError) as caught:
            socket.getaddrinfo("example.com", 443)

    assert caught.value.code == "LOCAL_ONLY_NETWORK_BLOCKED"
    assert caught.value.target == "example.com"


def test_runtime_guard_blocks_urllib_before_socket_io():
    from opencut.network_policy import LocalOnlyNetworkError

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        with pytest.raises(LocalOnlyNetworkError):
            opener.open("https://example.com/opencut", timeout=0.1)


def test_runtime_guard_blocks_requests_before_socket_io():
    from opencut.network_policy import LocalOnlyNetworkError

    session = requests.Session()
    session.trust_env = False
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        with pytest.raises(LocalOnlyNetworkError):
            session.get("https://example.com/opencut", timeout=0.1)


def test_runtime_guard_blocks_network_capable_subprocess_before_spawn():
    from opencut.network_policy import LocalOnlyNetworkError

    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        with pytest.raises(LocalOnlyNetworkError) as caught:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "opencut-egress-fixture"],
                check=False,
            )

    assert caught.value.target == "pip"


def test_runtime_guard_allows_explicit_loopback_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
            client = socket.create_connection(server.getsockname(), timeout=1)
            accepted, _address = server.accept()
        client.close()
        accepted.close()
    finally:
        server.close()


def test_runtime_guard_blocks_lan_addresses():
    from opencut.network_policy import LocalOnlyNetworkError

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
            with pytest.raises(LocalOnlyNetworkError):
                client.connect(("192.0.2.1", 9))
    finally:
        client.close()


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("localhost", True),
        ("renderer.localhost", True),
        ("127.0.0.42", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("192.168.1.10", False),
        ("10.0.0.5", False),
        ("example.com", False),
    ],
)
def test_loopback_classification_is_explicit(host, allowed):
    from opencut.network_policy import is_loopback_host

    assert is_loopback_host(host) is allowed


def test_runtime_guard_blocks_url_bearing_media_process_before_spawn():
    from opencut.network_policy import LocalOnlyNetworkError

    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        with pytest.raises(LocalOnlyNetworkError) as caught:
            subprocess.run(
                ["opencut-ffmpeg-fixture", "-i", "clip.mp4", "srt://192.0.2.1:9000"],
                check=False,
            )

    assert caught.value.target == "192.0.2.1"


def test_local_only_route_error_is_stable(local_only_app):
    @local_only_app.get("/_test/local-only-egress")
    def _attempt_egress():
        socket.getaddrinfo("example.com", 443)
        return {"unexpected": True}

    response = local_only_app.test_client().get("/_test/local-only-egress")
    body = response.get_json()

    assert response.status_code == 403
    assert body["code"] == "LOCAL_ONLY_NETWORK_BLOCKED"
    assert "localhost" in body["suggestion"]


def test_external_browser_routes_are_blocked(local_only_client, local_only_csrf):
    issue = local_only_client.get("/system/issue-report/bundle")
    oauth = local_only_client.post(
        "/social/auth-url",
        json={"platform": "youtube"},
        headers={"X-OpenCut-Token": local_only_csrf},
    )

    assert issue.status_code == 403
    assert issue.get_json()["code"] == "LOCAL_ONLY_NETWORK_BLOCKED"
    assert oauth.status_code == 403
    assert oauth.get_json()["code"] == "LOCAL_ONLY_NETWORK_BLOCKED"


def test_direct_egress_inventory_is_complete():
    from opencut.network_policy import (
        EGRESS_INVENTORY,
        stale_egress_inventory,
        unclassified_egress_modules,
    )

    root = Path(__file__).resolve().parents[1] / "opencut"
    assert not unclassified_egress_modules(root), (
        "Classify new direct network clients in EGRESS_INVENTORY: "
        f"{sorted(unclassified_egress_modules(root))}"
    )
    assert not stale_egress_inventory(root), (
        "Remove stale EGRESS_INVENTORY entries: "
        f"{sorted(stale_egress_inventory(root))}"
    )
    assert set(EGRESS_INVENTORY.values()) >= {
        "egress-guard",
        "external-integration",
        "loopback-client",
        "loopback-server",
    }


# ---- LLM guard ----

def test_llm_cloud_blocked_in_local_only():
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        from opencut.core.llm import LLMConfig, query_llm
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        result = query_llm("hello", config=cfg)
        assert result.error
        assert "Local-only" in result.error


def test_llm_ollama_allowed_in_local_only():
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        from opencut.core.llm import LLMConfig, query_llm
        cfg = LLMConfig(provider="ollama")
        # Ollama will fail to connect (no server), but should NOT be blocked
        result = query_llm("hello", config=cfg)
        assert "Local-only" not in (result.error or "")


# ---- Telemetry guard ----

def test_telemetry_blocked_in_local_only():
    with patch.dict(os.environ, {"OPENCUT_LOCAL_ONLY": "1"}):
        from opencut.core.telemetry_aptabase import track
        result = track("test_event", sync=True)
        assert result is False


# ---- Settings route ----

def test_get_local_only_setting(local_only_client):
    resp = local_only_client.get("/settings/local-only")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "active" in data
    assert "enabled" in data


def test_set_local_only_setting(local_only_client, local_only_csrf):
    headers = {"X-OpenCut-Token": local_only_csrf, "Content-Type": "application/json"}
    resp = local_only_client.post(
        "/settings/local-only",
        json={"enabled": True},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["enabled"] is True
