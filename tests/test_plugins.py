"""Tests for the plugin system."""

import json
import os
import textwrap
import threading
import time
from concurrent.futures import Future

import pytest


def _inline_pool():
    class InlinePool:
        def submit(self, job_id, fn):
            future = Future()
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

    return InlinePool()


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

    def test_plugin_trust_dashboard_summarizes_plugin_states(self, client, monkeypatch, tmp_path):
        import opencut.core.plugins as pm
        import opencut.routes.plugins as plugin_routes
        from opencut.core import plugin_marketplace
        from opencut.core.plugin_manifest import write_plugin_lock
        from opencut.core.plugins import _loaded_plugins, _plugins_lock

        plugins_dir = tmp_path / "plugins"
        monkeypatch.setattr(pm, "PLUGINS_DIR", str(plugins_dir))
        monkeypatch.setattr(plugin_routes, "PLUGINS_DIR", str(plugins_dir))
        monkeypatch.setattr(plugin_marketplace, "PLUGINS_DIR", str(tmp_path / "marketplace_plugins"))
        registry_cache = tmp_path / "plugin_registry.json"
        monkeypatch.setattr(plugin_marketplace, "REGISTRY_CACHE", str(registry_cache))

        locked_plugin = plugins_dir / "loaded-plugin"
        locked_plugin.mkdir(parents=True)
        locked_manifest = {
            "name": "loaded-plugin",
            "version": "1.0.0",
            "description": "Loaded plugin",
            "api_version": 1,
            "capabilities": ["http.routes", "jobs.register"],
            "routes": [{"path": "/sync"}],
            "jobs": [{"id": "sync"}],
        }
        (locked_plugin / "plugin.json").write_text(json.dumps(locked_manifest), encoding="utf-8")
        write_plugin_lock(locked_plugin)

        missing_lock_plugin = plugins_dir / "missing-lock"
        missing_lock_plugin.mkdir()
        (missing_lock_plugin / "plugin.json").write_text(
            json.dumps({
                "name": "missing-lock",
                "version": "1.0.0",
                "description": "Missing lock",
                "api_version": 1,
                "capabilities": ["host.network", "ui.panel"],
            }),
            encoding="utf-8",
        )

        disabled_plugin = plugins_dir / "disabled-plugin"
        disabled_plugin.mkdir()
        (disabled_plugin / "plugin.json").write_text(
            json.dumps({
                "name": "disabled-plugin",
                "version": "0.1.0",
                "description": "Disabled",
                "api_version": 1,
                "capabilities": ["ui.panel"],
                "enabled": False,
            }),
            encoding="utf-8",
        )
        write_plugin_lock(disabled_plugin)

        bad_plugin = plugins_dir / "bad-plugin"
        bad_plugin.mkdir()
        (bad_plugin / "plugin.json").write_text("{not json", encoding="utf-8")

        quarantine_id = "old-plugin-20260629-000000-deadbeef"
        quarantine_dir = tmp_path / "plugins_quarantine" / quarantine_id
        quarantine_dir.mkdir(parents=True)
        (quarantine_dir / ".opencut-quarantine.json").write_text(
            json.dumps({
                "name": "old-plugin",
                "original_path": str(plugins_dir / "old-plugin"),
                "created_at": 1782691200,
                "restore_route": "/plugins/quarantine/restore",
                "delete_route": "/plugins/quarantine/delete",
            }),
            encoding="utf-8",
        )

        registry_cache.write_text(
            json.dumps({
                "plugins": [
                    {
                        "id": "market-demo",
                        "name": "Market Demo",
                        "version": "2.0.0",
                        "author": "Tests",
                        "description": "Cached marketplace plugin",
                        "tags": ["captions"],
                    }
                ]
            }),
            encoding="utf-8",
        )

        with _plugins_lock:
            _loaded_plugins["loaded-plugin"] = {
                "info": {
                    **locked_manifest,
                    "author": "",
                    "path": str(locked_plugin),
                    "ui": None,
                },
                "module": None,
                "blueprint": None,
                "jobs": [{"id": "sync"}],
            }
        try:
            resp = client.get("/plugins/trust")
        finally:
            with _plugins_lock:
                _loaded_plugins.pop("loaded-plugin", None)

        assert resp.status_code == 200
        data = resp.get_json()
        plugins = {plugin["name"]: plugin for plugin in data["plugins"]}
        assert data["summary"]["loaded"] == 1
        assert data["summary"]["skipped"] >= 1
        assert data["summary"]["failed"] >= 2
        assert data["summary"]["lock_missing"] == 1
        assert data["summary"]["quarantined"] == 1
        assert data["summary"]["marketplace"] == 1
        assert plugins["loaded-plugin"]["load_status"] == "loaded"
        assert plugins["loaded-plugin"]["trust"]["source"] == "locked"
        assert plugins["missing-lock"]["trust"]["source"] == "lock_missing"
        assert plugins["missing-lock"]["capability_badges"][0]["id"] == "host.network"
        assert plugins["disabled-plugin"]["load_status"] == "skipped"
        assert plugins["bad-plugin"]["load_status"] == "failed"
        assert "Invalid plugin.json" in plugins["bad-plugin"]["error"]
        assert data["quarantine"]["entries"][0]["quarantine_id"] == quarantine_id
        assert data["actions"]["delete_quarantine"]["confirmation_required"] is True
        assert data["marketplace"]["plugins"][0]["plugin_id"] == "market-demo"

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
                "description": "CSRF guard test",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes"],
                "jobs": [],
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

    def test_plugin_job_route_starts_async_job(self, tmp_path, monkeypatch):
        from flask import Flask

        from opencut.core.plugins import _loaded_plugins, _plugins_lock, load_plugin_routes
        from opencut.jobs import jobs, job_lock
        from opencut.security import get_csrf_token

        monkeypatch.setattr("opencut.workers.get_pool", lambda: _inline_pool())
        monkeypatch.setattr("opencut.jobs._persist_job", lambda job_dict, *, sync=False: None)

        plugin_name = "long-job-demo"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                from flask import Blueprint
                from opencut.core.plugins import plugin_job

                plugin_bp = Blueprint("long_job_demo", __name__)

                @plugin_bp.route("/start", methods=["POST"])
                @plugin_job(
                    "long-job-demo",
                    "render_preview",
                    label="Render Preview",
                    filepath_required=False,
                    resumable=True,
                )
                def start(job_id, filepath, data):
                    return {
                        "plugin": "long-job-demo",
                        "job": "render_preview",
                        "payload": data,
                    }
                """
            ),
            encoding="utf-8",
        )

        with _plugins_lock:
            _loaded_plugins.pop(plugin_name, None)
        with job_lock:
            jobs.clear()

        app = Flask(__name__)
        app.config["TESTING"] = True
        assert load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Long job demo",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes", "jobs.register"],
                "jobs": [{"id": "render_preview", "label": "Render Preview"}],
                "ui": None,
            },
        )

        client = app.test_client()
        response = client.post(
            f"/plugins/{plugin_name}/start",
            json={"quality": "draft"},
            headers={"X-OpenCut-Token": get_csrf_token()},
        )
        body = response.get_json()

        assert response.status_code == 200
        assert body["job_id"]
        with job_lock:
            job = jobs[body["job_id"]]
        assert job["type"] == "plugin:long-job-demo:render_preview"
        assert job["status"] == "complete"
        assert job["resumable"] is True
        assert job["result"]["plugin"] == "long-job-demo"

        loaded = app.view_functions
        assert loaded
        with _plugins_lock:
            _loaded_plugins.pop(plugin_name, None)

    def test_plugin_job_must_be_declared_in_manifest(self, tmp_path):
        from flask import Flask

        from opencut.core.plugins import _loaded_plugins, _plugin_contexts, _plugins_lock, load_plugin_routes

        plugin_name = "undeclared-job"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                from flask import Blueprint
                from opencut.core.plugins import plugin_job

                plugin_bp = Blueprint("undeclared_job", __name__)

                @plugin_bp.route("/start", methods=["POST"])
                @plugin_job("undeclared-job", "hidden_task", filepath_required=False)
                def start(job_id, filepath, data):
                    return {"ok": True}
                """
            ),
            encoding="utf-8",
        )

        with _plugins_lock:
            _loaded_plugins.pop(plugin_name, None)
            _plugin_contexts.pop(plugin_name, None)

        app = Flask(__name__)
        app.config["TESTING"] = True
        assert not load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Undeclared job",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes", "jobs.register"],
                "jobs": [{"id": "other_task"}],
                "ui": None,
            },
        )
        with _plugins_lock:
            assert plugin_name not in _loaded_plugins
            assert plugin_name not in _plugin_contexts

    def test_plugin_job_without_filesystem_capability_rejects_host_paths(self, tmp_path, monkeypatch):
        from flask import Flask

        from opencut.core.plugins import _loaded_plugins, _plugins_lock, load_plugin_routes
        from opencut.security import get_csrf_token

        monkeypatch.setattr("opencut.jobs._persist_job", lambda job_dict, *, sync=False: None)

        plugin_name = "path-guard-job"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                from flask import Blueprint
                from opencut.core.plugins import plugin_job

                plugin_bp = Blueprint("path_guard_job", __name__)

                @plugin_bp.route("/start", methods=["POST"])
                @plugin_job("path-guard-job", "scan", filepath_required=False)
                def start(job_id, filepath, data):
                    return {"ok": True}
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
                "description": "Path guard job",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes", "jobs.register"],
                "jobs": [{"id": "scan"}],
                "ui": None,
            },
        )

        outside = tmp_path / "outside.txt"
        outside.write_text("nope", encoding="utf-8")
        response = app.test_client().post(
            f"/plugins/{plugin_name}/start",
            json={"input_path": str(outside)},
            headers={"X-OpenCut-Token": get_csrf_token()},
        )
        body = response.get_json()

        assert response.status_code == 400
        assert body["code"] == "INVALID_INPUT"
        assert "host.filesystem" in body["error"]

        with _plugins_lock:
            _loaded_plugins.pop(plugin_name, None)

    def test_third_party_routes_run_lazily_in_sanitized_worker(self, tmp_path, monkeypatch):
        from flask import Flask

        from opencut.core.plugin_runtime import get_supervisor
        from opencut.core.plugins import get_loaded_plugins, load_plugin_routes, unload_plugin

        plugin_name = "isolated-runtime"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                import os
                from flask import Blueprint, jsonify
                from opencut.core.plugins import _get_plugin_context

                with open(os.path.join(os.path.dirname(__file__), "imports.txt"), "a", encoding="utf-8") as marker:
                    marker.write(str(os.getpid()) + "\\n")

                plugin_bp = Blueprint("isolated_runtime", __name__)

                @plugin_bp.route("/identity", methods=["GET"])
                def identity():
                    context = _get_plugin_context("isolated-runtime")
                    return jsonify({
                        "pid": os.getpid(),
                        "parent_secret_visible": "OPENCUT_PARENT_ONLY_SECRET" in os.environ,
                        "capabilities": context.get("capabilities"),
                        "data_dir_name": os.path.basename(context.get("data_dir", "")),
                    })
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OPENCUT_PARENT_ONLY_SECRET", "must-not-cross-worker-boundary")
        app = Flask(__name__)
        app.config["TESTING"] = True

        assert load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Isolation test",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes"],
                "jobs": [],
                "ui": None,
            },
        )
        supervisor = get_supervisor(plugin_name)
        assert supervisor is not None
        assert supervisor.status()["state"] == "stopped"
        import_pids = [
            int(pid)
            for pid in (plugin_dir / "imports.txt").read_text(encoding="utf-8").splitlines()
        ]
        assert import_pids
        assert os.getpid() not in import_pids

        payload = app.test_client().get(f"/plugins/{plugin_name}/identity").get_json()

        assert payload["pid"] != os.getpid()
        assert payload["parent_secret_visible"] is False
        assert payload["capabilities"] == ["http.routes"]
        assert payload["data_dir_name"] == "data"
        assert supervisor.status()["state"] == "running"
        import_pids = [
            int(pid)
            for pid in (plugin_dir / "imports.txt").read_text(encoding="utf-8").splitlines()
        ]
        assert os.getpid() not in import_pids
        assert get_loaded_plugins()[plugin_name]["runtime"] == "supervised_process"
        assert supervisor._token not in " ".join(supervisor._process.args)
        unload_plugin(plugin_name)

    def test_plugin_crash_loop_is_quarantined_without_harming_other_worker(self, tmp_path):
        from flask import Flask

        from opencut.core.plugin_runtime import get_supervisor
        from opencut.core.plugins import load_plugin_routes, unload_plugin

        app = Flask(__name__)
        app.config["TESTING"] = True

        def install(name, body):
            plugin_dir = tmp_path / name
            plugin_dir.mkdir()
            (plugin_dir / "routes.py").write_text(textwrap.dedent(body), encoding="utf-8")
            assert load_plugin_routes(
                app,
                {
                    "name": name,
                    "version": "1.0.0",
                    "description": name,
                    "author": "tests",
                    "path": str(plugin_dir),
                    "capabilities": ["http.routes"],
                    "jobs": [],
                    "ui": None,
                },
            )

        install(
            "crash-loop",
            """
            import os
            from flask import Blueprint
            plugin_bp = Blueprint("crash_loop", __name__)
            @plugin_bp.route("/crash")
            def crash():
                os._exit(17)
            """,
        )
        install(
            "healthy-worker",
            """
            from flask import Blueprint, jsonify
            plugin_bp = Blueprint("healthy_worker", __name__)
            @plugin_bp.route("/ok")
            def ok():
                return jsonify({"ok": True})
            """,
        )
        client = app.test_client()
        assert client.get("/plugins/healthy-worker/ok").get_json() == {"ok": True}

        for _attempt in range(3):
            response = client.get("/plugins/crash-loop/crash")
            assert response.status_code == 503
            assert response.get_json()["code"] == "PLUGIN_WORKER_UNAVAILABLE"

        crashed = get_supervisor("crash-loop")
        assert crashed is not None
        assert crashed.status()["state"] == "quarantined"
        assert client.get("/plugins/healthy-worker/ok").get_json() == {"ok": True}
        unload_plugin("crash-loop")
        unload_plugin("healthy-worker")

    def test_locked_plugin_is_revalidated_before_production_worker_start(self, tmp_path):
        from flask import Flask

        from opencut.core.plugin_manifest import write_plugin_lock
        from opencut.core.plugin_runtime import get_supervisor
        from opencut.core.plugins import load_plugin_routes, unload_plugin

        plugin_name = "locked-runtime"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        manifest = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": "Locked production runtime test",
            "api_version": 1,
            "capabilities": ["http.routes"],
            "routes": [{"path": "/ready", "method": "GET"}],
        }
        (plugin_dir / "plugin.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                from flask import Blueprint, jsonify
                plugin_bp = Blueprint("locked_runtime", __name__)
                @plugin_bp.route("/ready")
                def ready():
                    return jsonify({"ready": True})
                """
            ),
            encoding="utf-8",
        )
        write_plugin_lock(plugin_dir)
        app = Flask(__name__)

        assert load_plugin_routes(
            app,
            {
                **manifest,
                "author": "tests",
                "path": str(plugin_dir),
                "jobs": [],
                "ui": None,
            },
        )
        supervisor = get_supervisor(plugin_name)
        assert supervisor is not None
        assert supervisor.status()["state"] == "stopped"
        assert app.test_client().get(
            f"/plugins/{plugin_name}/ready"
        ).get_json() == {"ready": True}
        assert supervisor.status()["state"] == "running"
        unload_plugin(plugin_name)

    def test_hung_plugin_times_out_without_blocking_other_requests(
        self, tmp_path, monkeypatch
    ):
        from flask import Flask

        import opencut.core.plugin_runtime as runtime
        from opencut.core.plugins import load_plugin_routes, unload_plugin

        monkeypatch.setattr(runtime, "REQUEST_TIMEOUT_SECONDS", 0.25)
        plugin_name = "hung-worker"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                import time
                from flask import Blueprint
                plugin_bp = Blueprint("hung_worker", __name__)
                @plugin_bp.route("/hang")
                def hang():
                    time.sleep(5)
                    return {"late": True}
                """
            ),
            encoding="utf-8",
        )
        app = Flask(__name__)
        app.config["TESTING"] = True
        assert load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Hang test",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes"],
                "jobs": [],
                "ui": None,
            },
        )

        started = time.monotonic()
        response = app.test_client().get(f"/plugins/{plugin_name}/hang")

        assert response.status_code == 503
        assert time.monotonic() - started < 2.0
        unload_plugin(plugin_name)

    def test_worker_health_stays_live_and_restart_interrupts_hung_request(
        self, tmp_path, monkeypatch
    ):
        from flask import Flask

        import opencut.core.plugin_runtime as runtime
        from opencut.core.plugins import load_plugin_routes, unload_plugin

        monkeypatch.setattr(runtime, "REQUEST_TIMEOUT_SECONDS", 5.0)
        plugin_name = "restart-hung-worker"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                import time
                from flask import Blueprint
                plugin_bp = Blueprint("restart_hung_worker", __name__)
                @plugin_bp.route("/hang")
                def hang():
                    time.sleep(10)
                    return {"late": True}
                """
            ),
            encoding="utf-8",
        )
        app = Flask(__name__)
        app.config["TESTING"] = True
        assert load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Restart hung worker test",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes"],
                "jobs": [],
                "ui": None,
            },
        )
        supervisor = runtime.get_supervisor(plugin_name)
        assert supervisor is not None
        result = {}

        def make_request():
            result["response"] = app.test_client().get(
                f"/plugins/{plugin_name}/hang"
            )

        request_thread = threading.Thread(target=make_request, daemon=True)
        request_thread.start()
        deadline = time.monotonic() + 2.0
        status = supervisor.status()
        while status["state"] != "busy" and time.monotonic() < deadline:
            time.sleep(0.02)
            status = supervisor.status()

        assert status["state"] == "busy"
        started = time.monotonic()
        restarted = supervisor.restart()
        assert time.monotonic() - started < 2.0
        assert restarted["state"] == "running"
        request_thread.join(timeout=2.0)
        assert not request_thread.is_alive()
        assert result["response"].status_code == 503
        unload_plugin(plugin_name)

    def test_only_bundled_plugins_can_select_trusted_in_process_lane(self, tmp_path):
        from opencut.core.plugins import _TRUSTED_BUILTIN_ROOT, _is_trusted_builtin

        bundled = _TRUSTED_BUILTIN_ROOT / "clip-notes"
        assert _is_trusted_builtin(
            {
                "path": str(bundled),
                "runtime": "trusted-in-process",
                "builtin": True,
            }
        )
        assert not _is_trusted_builtin(
            {
                "path": str(tmp_path),
                "runtime": "trusted-in-process",
                "builtin": True,
            }
        )

    def test_worker_health_and_restart_api_are_redacted(self, tmp_path):
        from flask import Flask

        from opencut.core.plugins import load_plugin_routes, unload_plugin
        from opencut.routes.plugins import plugins_bp
        from opencut.security import get_csrf_token

        plugin_name = "health-api-worker"
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "routes.py").write_text(
            textwrap.dedent(
                """
                from flask import Blueprint, jsonify
                plugin_bp = Blueprint("health_api_worker", __name__)
                @plugin_bp.route("/ping")
                def ping():
                    return jsonify({"ok": True})
                """
            ),
            encoding="utf-8",
        )
        app = Flask(__name__)
        app.config["TESTING"] = True
        assert load_plugin_routes(
            app,
            {
                "name": plugin_name,
                "version": "1.0.0",
                "description": "Worker health API test",
                "author": "tests",
                "path": str(plugin_dir),
                "capabilities": ["http.routes"],
                "jobs": [],
                "ui": None,
            },
        )
        app.register_blueprint(plugins_bp)
        client = app.test_client()

        health = client.get("/plugins/workers")
        assert health.status_code == 200
        worker = next(
            entry for entry in health.get_json()["workers"]
            if entry["name"] == plugin_name
        )
        assert worker["state"] == "stopped"
        assert worker["security_boundary"] == "availability isolation; not an OS sandbox"
        serialized = json.dumps(worker)
        assert str(plugin_dir) not in serialized
        assert "token" not in serialized.lower()

        missing_csrf = client.post(
            "/plugins/workers/restart",
            json={"name": plugin_name},
        )
        assert missing_csrf.status_code == 403
        restarted = client.post(
            "/plugins/workers/restart",
            json={"name": plugin_name},
            headers={"X-OpenCut-Token": get_csrf_token()},
        )
        assert restarted.status_code == 200
        assert restarted.get_json()["worker"]["state"] == "running"
        assert client.get(f"/plugins/{plugin_name}/ping").get_json() == {"ok": True}
        unload_plugin(plugin_name)
