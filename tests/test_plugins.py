"""Tests for the plugin system."""

import json
import textwrap

import pytest


class TestPluginDiscovery:
    """Test plugin discovery and validation."""

    def test_discover_empty_dir(self, tmp_path):
        import opencut.core.plugins as pm
        from opencut.core.plugins import discover_plugins
        original = pm.PLUGINS_DIR
        pm.PLUGINS_DIR = str(tmp_path)
        try:
            plugins = discover_plugins()
            assert plugins == []
        finally:
            pm.PLUGINS_DIR = original

    def test_discover_valid_plugin(self, tmp_path):
        import opencut.core.plugins as pm
        from opencut.core.plugins import discover_plugins

        # Create a valid plugin
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()
        manifest = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
        }
        (plugin_dir / "plugin.json").write_text(json.dumps(manifest))

        original = pm.PLUGINS_DIR
        pm.PLUGINS_DIR = str(tmp_path)
        try:
            plugins = discover_plugins()
            assert len(plugins) == 1
            assert plugins[0]["name"] == "test-plugin"
            assert plugins[0]["valid"] is True
            assert plugins[0]["enabled"] is True
        finally:
            pm.PLUGINS_DIR = original

    def test_discover_invalid_manifest(self, tmp_path):
        import opencut.core.plugins as pm
        from opencut.core.plugins import discover_plugins

        plugin_dir = tmp_path / "bad-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text("not json")

        original = pm.PLUGINS_DIR
        pm.PLUGINS_DIR = str(tmp_path)
        try:
            plugins = discover_plugins()
            assert len(plugins) == 1
            assert plugins[0]["valid"] is False
        finally:
            pm.PLUGINS_DIR = original

    def test_discover_missing_manifest(self, tmp_path):
        import opencut.core.plugins as pm
        from opencut.core.plugins import discover_plugins

        (tmp_path / "no-manifest").mkdir()

        original = pm.PLUGINS_DIR
        pm.PLUGINS_DIR = str(tmp_path)
        try:
            plugins = discover_plugins()
            assert len(plugins) == 1
            assert plugins[0]["valid"] is False
            assert "Missing" in plugins[0]["error"]
        finally:
            pm.PLUGINS_DIR = original

    def test_validate_manifest_missing_fields(self):
        from opencut.core.plugins import _validate_manifest
        valid, error = _validate_manifest({"name": "x"}, "/tmp")
        assert not valid
        assert "version" in error or "description" in error

    def test_validate_manifest_invalid_name(self):
        from opencut.core.plugins import _validate_manifest
        valid, error = _validate_manifest(
            {"name": "bad plugin!", "version": "1.0", "description": "x"}, "/tmp"
        )
        assert not valid
        assert "invalid characters" in error


class TestPluginRoutes:
    """Test plugin management API routes."""

    @pytest.fixture
    def client(self):
        from opencut.server import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.get("/health")
            token = resp.get_json().get("csrf_token", "")
            c._csrf = token
            yield c

    def test_list_plugins(self, client):
        resp = client.get("/plugins/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "plugins" in data
        assert "total" in data

    def test_loaded_plugins(self, client):
        resp = client.get("/plugins/loaded")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "plugins" in data

    def test_loaded_plugin_mutations_inherit_csrf_guard(self, tmp_path):
        from flask import Flask

        from opencut.core.plugins import _loaded_plugins, _plugins_lock, load_plugin_routes
        from opencut.security import get_csrf_token

        plugin_name = "csrf-guard-test"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                from flask import Blueprint, jsonify

                plugin_bp = Blueprint("csrf_guard_test_plugin", __name__)

                @plugin_bp.route("/mutate", methods=["POST"])
                def mutate():
                    return jsonify({"ok": True})

                @plugin_bp.route("/read", methods=["GET"])
                def read():
                    return jsonify({"ok": True})
                """
            ),
            encoding="utf-8",
        )

        with _plugins_lock:
            _loaded_plugins.pop(plugin_name, None)

        app = Flask(__name__)
        app.config["TESTING"] = True
        assert load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "CSRF guard regression test",
                "author": "tests",
                "path": str(plugin_dir),
                "ui": None,
            },
        )

        client = app.test_client()
        assert client.get(f"/plugins/{plugin_name}/read").status_code == 200

        missing = client.post(f"/plugins/{plugin_name}/mutate", json={})
        assert missing.status_code == 403

        allowed = client.post(
            f"/plugins/{plugin_name}/mutate",
            json={},
            headers={"X-OpenCut-Token": get_csrf_token()},
        )
        assert allowed.status_code == 200
        assert allowed.get_json() == {"ok": True}

        with _plugins_lock:
            _loaded_plugins.pop(plugin_name, None)
