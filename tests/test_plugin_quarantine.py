"""Regression tests for plugin uninstall quarantine safeguards."""

from __future__ import annotations

import json


def _csrf_headers(token: str) -> dict[str, str]:
    return {"X-OpenCut-Token": token, "Content-Type": "application/json"}


def _isolate_plugins(monkeypatch, tmp_path):
    import opencut.core.plugins as core_plugins
    import opencut.routes.plugins as plugin_routes

    plugins_dir = tmp_path / "plugins"
    monkeypatch.setattr(core_plugins, "PLUGINS_DIR", str(plugins_dir))
    monkeypatch.setattr(plugin_routes, "PLUGINS_DIR", str(plugins_dir))
    return plugin_routes, plugins_dir


def _write_plugin(plugins_dir, name: str = "demo-plugin"):
    plugin_dir = plugins_dir / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        json.dumps({
            "name": name,
            "version": "1.0.0",
            "description": "Demo plugin",
        }),
        encoding="utf-8",
    )
    (plugin_dir / "payload.txt").write_text("kept", encoding="utf-8")
    return plugin_dir


def test_uninstall_requires_confirm_name(client, csrf_token, monkeypatch, tmp_path):
    _routes, plugins_dir = _isolate_plugins(monkeypatch, tmp_path)
    plugin_dir = _write_plugin(plugins_dir)

    response = client.post(
        "/plugins/uninstall",
        json={"name": "demo-plugin"},
        headers={"X-OpenCut-Token": csrf_token},
    )
    missing_token = client.post(
        "/plugins/uninstall",
        json={"name": "demo-plugin", "confirm_name": "demo-plugin"},
        headers={"X-OpenCut-Token": csrf_token},
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "CONFIRMATION_REQUIRED"
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert plugin_dir.exists()


def test_uninstall_moves_plugin_to_quarantine_and_restores(client, csrf_token, monkeypatch, tmp_path):
    routes, plugins_dir = _isolate_plugins(monkeypatch, tmp_path)
    plugin_dir = _write_plugin(plugins_dir)
    preview_response = client.post(
        "/plugins/uninstall",
        json={"name": "demo-plugin", "dry_run": True},
        headers={"X-OpenCut-Token": csrf_token},
    )
    preview = preview_response.get_json()

    response = client.post(
        "/plugins/uninstall",
        json={
            "name": "demo-plugin",
            "confirm_name": "demo-plugin",
            "confirm_token": preview["confirm_token"],
        },
        headers={"X-OpenCut-Token": csrf_token},
    )

    assert preview_response.status_code == 200
    assert preview["would_quarantine"] is True
    assert response.status_code == 200
    data = response.get_json()
    quarantine_id = data["quarantine_id"]
    quarantine_path = routes._quarantine_path(quarantine_id)
    assert data["quarantined"] is True
    assert not plugin_dir.exists()
    assert (tmp_path / "plugins_quarantine").is_dir()
    assert (plugins_dir.parent / "plugins_quarantine" / quarantine_id / "payload.txt").exists()

    list_response = client.get("/plugins/quarantine/list")
    assert list_response.status_code == 200
    assert list_response.get_json()["total"] == 1

    restore_response = client.post(
        "/plugins/quarantine/restore",
        json={"quarantine_id": quarantine_id},
        headers={"X-OpenCut-Token": csrf_token},
    )

    assert restore_response.status_code == 200
    assert plugin_dir.exists()
    assert not (plugin_dir / ".opencut-quarantine.json").exists()
    assert not (plugins_dir.parent / "plugins_quarantine" / quarantine_id).exists()
    assert quarantine_path.endswith(quarantine_id)


def test_quarantine_permanent_delete_requires_confirm_name(client, csrf_token, monkeypatch, tmp_path):
    _routes, plugins_dir = _isolate_plugins(monkeypatch, tmp_path)
    _write_plugin(plugins_dir)
    uninstall_preview = client.post(
        "/plugins/uninstall",
        json={"name": "demo-plugin", "dry_run": True},
        headers=_csrf_headers(csrf_token),
    )
    uninstall_response = client.post(
        "/plugins/uninstall",
        json={
            "name": "demo-plugin",
            "confirm_name": "demo-plugin",
            "confirm_token": uninstall_preview.get_json()["confirm_token"],
        },
        headers=_csrf_headers(csrf_token),
    )
    quarantine_id = uninstall_response.get_json()["quarantine_id"]

    rejected = client.post(
        "/plugins/quarantine/delete",
        json={"quarantine_id": quarantine_id},
        headers=_csrf_headers(csrf_token),
    )
    preview = client.post(
        "/plugins/quarantine/delete",
        json={"quarantine_id": quarantine_id, "dry_run": True},
        headers=_csrf_headers(csrf_token),
    )
    missing_token = client.post(
        "/plugins/quarantine/delete",
        json={"quarantine_id": quarantine_id, "confirm_name": "demo-plugin"},
        headers=_csrf_headers(csrf_token),
    )
    deleted = client.post(
        "/plugins/quarantine/delete",
        json={
            "quarantine_id": quarantine_id,
            "confirm_name": "demo-plugin",
            "confirm_token": preview.get_json()["confirm_token"],
        },
        headers=_csrf_headers(csrf_token),
    )

    assert rejected.status_code == 400
    assert rejected.get_json()["code"] == "CONFIRMATION_REQUIRED"
    assert preview.status_code == 200
    assert preview.get_json()["would_delete"] is True
    assert missing_token.status_code == 409
    assert missing_token.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"
    assert deleted.status_code == 200
    assert deleted.get_json()["deleted"] is True
    assert client.get("/plugins/quarantine/list").get_json()["total"] == 0
