"""OS credential-vault persistence, migration, and rollback coverage."""

from __future__ import annotations

import json

import pytest

from opencut import credential_store


class _UnavailableBackend:
    priority = 0


class _FailingBackend(credential_store.MemoryCredentialBackend):
    def set_password(self, service: str, username: str, password: str) -> None:
        raise RuntimeError("vault locked")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_secret_persistence_refuses_unavailable_vault(monkeypatch):
    monkeypatch.setattr(credential_store, "_backend_override", _UnavailableBackend())
    called = False

    def persist(_secure):
        nonlocal called
        called = True

    with pytest.raises(credential_store.CredentialStoreUnavailableError):
        credential_store.persist_secret_changes({"test": "secret"}, persist)
    assert called is False


def test_explicit_insecure_opt_in_is_reported_to_metadata_writer(monkeypatch):
    monkeypatch.setattr(credential_store, "_backend_override", _UnavailableBackend())
    monkeypatch.setenv(credential_store.INSECURE_OPT_IN_ENV, "1")
    modes = []
    result = credential_store.persist_secret_changes(
        {"test": "secret"}, modes.append
    )
    assert result.secure is False
    assert modes == [False]


def test_metadata_failure_rolls_back_verified_vault_write(monkeypatch):
    backend = credential_store.MemoryCredentialBackend()
    backend.set_password(credential_store.SERVICE_NAME, "test", "old")
    monkeypatch.setattr(credential_store, "_backend_override", backend)

    def fail_metadata(_secure):
        raise OSError("disk full")

    with pytest.raises(credential_store.CredentialStoreWriteError):
        credential_store.persist_secret_changes({"test": "new"}, fail_metadata)
    assert backend.get_password(credential_store.SERVICE_NAME, "test") == "old"


def test_legacy_migration_keeps_plaintext_when_vault_write_fails(
    tmp_path, monkeypatch
):
    """Every current credential store retains legacy data on rollback."""
    from opencut import auth, user_data
    from opencut.core import (
        cloud_render,
        notion_sync,
        remote_process,
        social_post,
        webhook_system,
    )

    monkeypatch.setattr(credential_store, "_backend_override", _FailingBackend())
    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))

    llm_path = tmp_path / "llm_settings.json"
    llm_path.write_text(json.dumps({"api_key": "llm-legacy"}), encoding="utf-8")
    assert user_data.load_llm_settings()["api_key"] == "llm-legacy"
    assert _read_json(llm_path)["api_key"] == "llm-legacy"

    telemetry_path = tmp_path / "telemetry_settings.json"
    telemetry_path.write_text(
        json.dumps({"app_key": "telemetry-legacy"}), encoding="utf-8"
    )
    assert user_data.load_telemetry_settings()["app_key"] == "telemetry-legacy"
    assert _read_json(telemetry_path)["app_key"] == "telemetry-legacy"

    notion_path = tmp_path / "notion.json"
    notion_path.write_text(json.dumps({"api_key": "notion-legacy"}), encoding="utf-8")
    monkeypatch.setattr(notion_sync, "_OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(notion_sync, "_CONFIG_FILE", str(notion_path))
    assert notion_sync.load_notion_config()["api_key"] == "notion-legacy"
    assert _read_json(notion_path)["api_key"] == "notion-legacy"

    social_path = tmp_path / "social.json"
    social_path.write_text(
        json.dumps({"youtube": {"access_token": "social-legacy"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(social_post, "CREDENTIALS_PATH", str(social_path))
    assert social_post._load_credentials()["youtube"].access_token == "social-legacy"
    assert _read_json(social_path)["youtube"]["access_token"] == "social-legacy"

    cloud_path = tmp_path / "cloud.json"
    cloud_path.write_text(
        json.dumps([{"name": "gpu", "host": "node", "auth_token": "cloud-legacy"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(cloud_render, "_NODES_FILE", str(cloud_path))
    assert cloud_render.load_nodes()[0].auth_token == "cloud-legacy"
    assert _read_json(cloud_path)[0]["auth_token"] == "cloud-legacy"

    remote_path = tmp_path / "remote.json"
    remote_path.write_text(
        json.dumps(
            {
                "default_node": "https://node.example",
                "nodes": {
                    "https://node.example": {"api_key": "remote-legacy"}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote_process, "REGISTRY_PATH", str(remote_path))
    monkeypatch.setattr(remote_process, "OPENCUT_DIR", str(tmp_path))
    registry = remote_process.reload_registry()
    assert registry.nodes["https://node.example"].api_key == "remote-legacy"
    assert _read_json(remote_path)["nodes"]["https://node.example"]["api_key"] == "remote-legacy"

    webhook_path = tmp_path / "webhooks.json"
    webhook_path.write_text(
        json.dumps(
            [{"id": "hook", "url": "https://hooks.example", "secret": "hook-legacy"}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(webhook_system, "_OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(webhook_system, "_WEBHOOKS_FILE", str(webhook_path))
    assert webhook_system._load_configs()[0].secret == "hook-legacy"
    assert _read_json(webhook_path)[0]["secret"] == "hook-legacy"

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"token": "auth-legacy", "issued_at": 1, "label": "legacy"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "AUTH_FILE", auth_path)
    assert auth.current_token().token == "auth-legacy"
    assert _read_json(auth_path)["token"] == "auth-legacy"


def test_legacy_migration_removes_plaintext_from_every_store(tmp_path, monkeypatch):
    from opencut import auth, user_data
    from opencut.core import (
        cloud_render,
        notion_sync,
        remote_process,
        social_post,
        webhook_system,
    )

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))

    llm_path = tmp_path / "llm_settings.json"
    llm_path.write_text(json.dumps({"api_key": "llm-secret"}), encoding="utf-8")
    assert user_data.load_llm_settings()["api_key"] == "llm-secret"
    assert "llm-secret" not in llm_path.read_text(encoding="utf-8")

    telemetry_path = tmp_path / "telemetry_settings.json"
    telemetry_path.write_text(
        json.dumps({"app_key": "telemetry-secret"}), encoding="utf-8"
    )
    assert user_data.load_telemetry_settings()["app_key"] == "telemetry-secret"
    assert "telemetry-secret" not in telemetry_path.read_text(encoding="utf-8")

    notion_path = tmp_path / "notion.json"
    notion_path.write_text(json.dumps({"api_key": "notion-secret"}), encoding="utf-8")
    monkeypatch.setattr(notion_sync, "_OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(notion_sync, "_CONFIG_FILE", str(notion_path))
    assert notion_sync.load_notion_config()["api_key"] == "notion-secret"
    assert "notion-secret" not in notion_path.read_text(encoding="utf-8")

    social_path = tmp_path / "social.json"
    social_path.write_text(
        json.dumps(
            {
                "youtube": {
                    "access_token": "social-access",
                    "refresh_token": "social-refresh",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(social_post, "CREDENTIALS_PATH", str(social_path))
    social = social_post._load_credentials()["youtube"]
    assert social.access_token == "social-access"
    assert social.refresh_token == "social-refresh"
    assert "social-access" not in social_path.read_text(encoding="utf-8")
    assert "social-refresh" not in social_path.read_text(encoding="utf-8")

    cloud_path = tmp_path / "cloud.json"
    cloud_path.write_text(
        json.dumps([{"name": "gpu", "host": "node", "auth_token": "cloud-secret"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(cloud_render, "_NODES_FILE", str(cloud_path))
    assert cloud_render.load_nodes()[0].auth_token == "cloud-secret"
    assert "cloud-secret" not in cloud_path.read_text(encoding="utf-8")

    remote_path = tmp_path / "remote.json"
    remote_path.write_text(
        json.dumps(
            {
                "default_node": "https://node.example",
                "nodes": {"https://node.example": {"api_key": "remote-secret"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote_process, "REGISTRY_PATH", str(remote_path))
    monkeypatch.setattr(remote_process, "OPENCUT_DIR", str(tmp_path))
    registry = remote_process.reload_registry()
    assert registry.nodes["https://node.example"].api_key == "remote-secret"
    assert "remote-secret" not in remote_path.read_text(encoding="utf-8")

    webhook_path = tmp_path / "webhooks.json"
    webhook_path.write_text(
        json.dumps(
            [{"id": "hook", "url": "https://hooks.example", "secret": "hook-secret"}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(webhook_system, "_OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(webhook_system, "_WEBHOOKS_FILE", str(webhook_path))
    assert webhook_system._load_configs()[0].secret == "hook-secret"
    assert "hook-secret" not in webhook_path.read_text(encoding="utf-8")

    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"token": "auth-secret", "issued_at": 1, "label": "legacy"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "AUTH_FILE", auth_path)
    assert auth.current_token().token == "auth-secret"
    assert "auth-secret" not in auth_path.read_text(encoding="utf-8")


def test_status_route_and_error_contract_never_expose_secret(client, csrf_token, monkeypatch):
    response = client.get("/settings/local-only")
    assert response.status_code == 200
    status = response.get_json()["credential_store"]
    assert status["available"] is True
    assert "values" not in status

    monkeypatch.setattr(credential_store, "_backend_override", _UnavailableBackend())
    response = client.post(
        "/settings/llm",
        json={"api_key": "must-not-leak"},
        headers={"X-OpenCut-Token": csrf_token},
    )
    assert response.status_code == 503
    body = response.get_json()
    assert body["code"] == credential_store.UNAVAILABLE_CODE
    assert "must-not-leak" not in json.dumps(body)


def test_legacy_webhook_config_round_trip_preserves_vault_secret(
    tmp_path, monkeypatch
):
    from opencut.core import webhooks

    path = tmp_path / "webhooks.json"
    monkeypatch.setattr(webhooks, "_OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(webhooks, "_WEBHOOKS_FILE", str(path))
    webhooks.save_webhook_config(
        [
            {
                "id": "hook",
                "url": "https://hooks.example",
                "events": ["job.complete"],
                "secret": "signing-secret",
            }
        ]
    )
    assert "signing-secret" not in path.read_text(encoding="utf-8")
    redacted = webhooks.load_webhook_config()
    assert redacted[0]["has_secret"] is True
    assert "secret" not in redacted[0]

    webhooks.save_webhook_config(redacted)
    identifier = credential_store.secret_id("webhook/signing-secret", "hook")
    assert credential_store.get_secret(identifier) == "signing-secret"


def test_redacted_node_updates_preserve_existing_vault_credentials(
    tmp_path, monkeypatch, client, csrf_token
):
    from opencut.core import cloud_render, remote_process

    cloud_path = tmp_path / "cloud.json"
    monkeypatch.setattr(cloud_render, "_NODES_FILE", str(cloud_path))
    cloud_render.save_nodes(
        [cloud_render.RenderNode(name="gpu", host="old", auth_token="cloud-token")]
    )
    response = client.post(
        "/cloud/nodes",
        json={"name": "gpu", "host": "new"},
        headers={"X-OpenCut-Token": csrf_token},
    )
    assert response.status_code == 200
    assert cloud_render.load_nodes()[0].auth_token == "cloud-token"
    assert "cloud-token" not in json.dumps(response.get_json())

    remote_path = tmp_path / "remote.json"
    monkeypatch.setattr(remote_process, "REGISTRY_PATH", str(remote_path))
    monkeypatch.setattr(remote_process, "OPENCUT_DIR", str(tmp_path))
    monkeypatch.setattr(remote_process, "_registry", remote_process.NodeRegistry())
    monkeypatch.setattr(remote_process, "normalize_node_url", lambda value: value)
    monkeypatch.setattr(
        remote_process,
        "ping_node",
        lambda _url, api_key="": {"name": "node", "capabilities": ["video"]},
    )
    remote_process.register_node("https://node.example", api_key="remote-token")
    updated = remote_process.register_node("https://node.example", api_key="")
    assert updated.api_key == "remote-token"
    assert "remote-token" not in remote_path.read_text(encoding="utf-8")


def test_remote_server_bind_refuses_when_vault_cannot_issue_auth_token(
    monkeypatch
):
    import opencut.server as server

    monkeypatch.setattr(credential_store, "_backend_override", _UnavailableBackend())
    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    monkeypatch.setattr(
        server.sys,
        "argv",
        ["opencut-server", "--host", "0.0.0.0"],
    )
    monkeypatch.setattr(
        server,
        "run_server",
        lambda **_kwargs: pytest.fail("server must not start without auth"),
    )
    assert server.main() == 2


def test_recursive_redaction_covers_current_secret_field_names():
    payload = {
        "api_key": "a",
        "access_token": "b",
        "refresh_token": "c",
        "auth_token": "d",
        "secret": "e",
        "token": "f",
        "nested": [{"app_key": "g"}],
    }
    redacted = credential_store.redact_secret_mapping(payload)
    assert set(json.dumps(redacted))
    for value in "abcdefg":
        assert f'"{value}"' not in json.dumps(redacted)
