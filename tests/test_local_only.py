"""
Tests for the OPENCUT_LOCAL_ONLY privacy mode.

Verifies that cloud-capable modules are blocked when local-only mode is active.
"""

import os
from unittest.mock import patch

import pytest


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
