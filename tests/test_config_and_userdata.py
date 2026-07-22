import ast
import importlib
import json
import os
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import csrf_headers


def test_config_from_env_parses_limits_and_origins():
    from opencut.config import OpenCutConfig

    with patch.dict(os.environ, {
        "OPENCUT_BUNDLED": "yes",
        "OPENCUT_MAX_CONTENT_LENGTH": "2097152",
        "OPENCUT_CORS_ORIGINS": "null, file://, http://localhost:3000 ",
        "OPENCUT_JOB_MAX_AGE": "120",
        "OPENCUT_MAX_CONCURRENT_JOBS": "4",
        "OPENCUT_MAX_BATCH_FILES": "12",
        "OPENCUT_JOB_STUCK_TIMEOUT": "180",
    }, clear=False):
        config = OpenCutConfig.from_env()

    assert config.bundled_mode is True
    assert config.max_content_length == 2097152
    assert config.cors_origins == ["null", "file://", "http://localhost:3000"]
    assert config.job_max_age == 120
    assert config.max_concurrent_jobs == 4
    assert config.max_batch_files == 12
    assert config.job_stuck_timeout == 180


def test_config_default_cors_is_closed_for_csrf_bootstrap():
    from opencut.config import OpenCutConfig

    with patch.dict(os.environ, {"OPENCUT_CORS_ORIGINS": ""}, clear=False):
        assert OpenCutConfig.from_env().cors_origins == []

    with patch.dict(os.environ, {}, clear=True):
        assert OpenCutConfig.from_env().cors_origins == []
    assert OpenCutConfig().cors_origins == []


def test_jobs_module_uses_env_overrides_for_limits():
    import opencut.jobs as jobs_module

    try:
        with patch.dict(os.environ, {
            "OPENCUT_JOB_MAX_AGE": "90",
            "OPENCUT_MAX_CONCURRENT_JOBS": "3",
            "OPENCUT_MAX_BATCH_FILES": "7",
            "OPENCUT_JOB_STUCK_TIMEOUT": "300",
        }, clear=False):
            reloaded = importlib.reload(jobs_module)
            assert reloaded.JOB_MAX_AGE == 90
            assert reloaded.MAX_CONCURRENT_JOBS == 3
            assert reloaded.MAX_BATCH_FILES == 7
            assert reloaded._JOB_STUCK_TIMEOUT == 300
    finally:
        importlib.reload(jobs_module)


def test_runtime_boot_modules_avoid_pep604_annotations_for_python39():
    """Keep boot modules importable before optional dependencies are installed."""
    repo_root = Path(__file__).resolve().parents[1]
    checked = [
        repo_root / "opencut" / "config.py",
        repo_root / "opencut" / "jobs.py",
        repo_root / "opencut" / "job_store.py",
        repo_root / "opencut" / "security.py",
        repo_root / "opencut" / "workers.py",
    ]
    offenders = []

    class BitOrVisitor(ast.NodeVisitor):
        def __init__(self, path):
            self.path = path

        def visit_BinOp(self, node):
            if isinstance(node.op, ast.BitOr):
                offenders.append(f"{self.path}:{node.lineno}")
            self.generic_visit(node)

    for path in checked:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        BitOrVisitor(path.relative_to(repo_root)).visit(tree)

    assert offenders == []


def test_user_writable_package_dir_is_not_sys_path_priority():
    repo_root = Path(__file__).resolve().parents[1]
    checked = {
        "opencut/helpers.py": "_sys.path.append(_opencut_pkg_dir)",
        "opencut/security.py": "sys.path.append(_target_dir)",
        "opencut/server.py": "sys.path.append(_opencut_packages)",
        "opencut/routes/system_whisper_routes.py": "sys.path.append(_target_dir)",
    }

    for relative_path, expected_append in checked.items():
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert expected_append in text
        assert ".path.insert(0" not in text


def test_create_app_applies_custom_job_config():
    from opencut.config import OpenCutConfig
    from opencut.server import create_app
    import opencut.jobs as jobs_module

    original = (
        jobs_module.JOB_MAX_AGE,
        jobs_module.MAX_CONCURRENT_JOBS,
        jobs_module.MAX_BATCH_FILES,
        jobs_module._JOB_STUCK_TIMEOUT,
    )
    try:
        create_app(OpenCutConfig(
            job_max_age=91,
            max_concurrent_jobs=2,
            max_batch_files=9,
            job_stuck_timeout=301,
        ))

        assert jobs_module.JOB_MAX_AGE == 91
        assert jobs_module.MAX_CONCURRENT_JOBS == 2
        assert jobs_module.MAX_BATCH_FILES == 9
        assert jobs_module._JOB_STUCK_TIMEOUT == 301
    finally:
        jobs_module.JOB_MAX_AGE = original[0]
        jobs_module.MAX_CONCURRENT_JOBS = original[1]
        jobs_module.MAX_BATCH_FILES = original[2]
        jobs_module._JOB_STUCK_TIMEOUT = original[3]


def test_read_user_file_quarantines_corrupt_json(tmp_path):
    import opencut.user_data as user_data

    corrupt_path = tmp_path / "favorites.json"
    corrupt_path.write_text('{"broken":', encoding="utf-8")

    with patch("opencut.user_data.OPENCUT_DIR", str(tmp_path)):
        result = user_data.read_user_file("favorites.json", default=[])

    assert result == []
    assert not corrupt_path.exists()
    quarantined = list(tmp_path.glob("favorites.json.corrupt-*"))
    assert len(quarantined) == 1


def test_json_config_migration_rolls_back_all_steps_and_retries_idempotently(
    tmp_path, monkeypatch, caplog
):
    import opencut.user_data as user_data

    filename = "migration-retry.json"
    source = {"_schema_version": 0, "keep": "original"}
    (tmp_path / filename).write_text(json.dumps(source), encoding="utf-8")
    should_fail = {"value": True}
    calls = []

    def migrate_v1(data):
        calls.append(1)
        data["first_step"] = True
        return data

    def migrate_v2(data):
        calls.append(2)
        data["sensitive_partial"] = "must-not-persist"
        if should_fail["value"]:
            raise RuntimeError("secret-value-must-not-reach-logs")
        data.pop("sensitive_partial")
        data["second_step"] = True
        return data

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    user_data.register_config_schema(
        filename,
        version=2,
        migrations={1: migrate_v1, 2: migrate_v2},
    )

    failed = user_data.read_user_file_versioned(filename)
    assert failed == source
    assert json.loads((tmp_path / filename).read_text(encoding="utf-8")) == source
    assert not (tmp_path / f"{filename}.migration-backup").exists()
    assert "secret-value-must-not-reach-logs" not in caplog.text
    assert "must-not-persist" not in caplog.text

    should_fail["value"] = False
    migrated = user_data.read_user_file_versioned(filename)
    assert migrated == {
        "_schema_version": 2,
        "keep": "original",
        "first_step": True,
        "second_step": True,
    }
    calls_after_success = list(calls)
    assert user_data.read_user_file_versioned(filename) == migrated
    assert calls == calls_after_success


def test_json_config_migration_recovers_interrupted_backup(tmp_path, monkeypatch):
    import opencut.user_data as user_data

    filename = "migration-recovery.json"
    backup = tmp_path / f"{filename}.migration-backup"
    backup.write_text(json.dumps({"_schema_version": 0, "restored": True}), encoding="utf-8")
    (tmp_path / filename).write_text('{"partial":', encoding="utf-8")
    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    user_data.register_config_schema(
        filename,
        version=1,
        migrations={1: lambda data: {**data, "migrated": True}},
    )

    result = user_data.read_user_file_versioned(filename)

    assert result == {
        "_schema_version": 1,
        "restored": True,
        "migrated": True,
    }
    assert json.loads((tmp_path / filename).read_text(encoding="utf-8")) == result
    assert not backup.exists()
    assert len(list(tmp_path.glob(f"{filename}.corrupt-*"))) == 1


def test_json_config_migration_resumes_after_committed_step(tmp_path, monkeypatch):
    import opencut.user_data as user_data

    filename = "migration-resume.json"
    original = {"_schema_version": 0, "value": "original"}
    committed_v1 = {"_schema_version": 1, "value": "original", "v1": True}
    (tmp_path / filename).write_text(json.dumps(committed_v1), encoding="utf-8")
    backup = tmp_path / f"{filename}.migration-backup"
    backup.write_text(json.dumps(original), encoding="utf-8")
    calls = []
    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    user_data.register_config_schema(
        filename,
        version=2,
        migrations={
            1: lambda data: calls.append(1) or {**data, "v1": "replayed"},
            2: lambda data: calls.append(2) or {**data, "v2": True},
        },
    )

    result = user_data.read_user_file_versioned(filename)

    assert calls == [2]
    assert result == {**committed_v1, "_schema_version": 2, "v2": True}
    assert not backup.exists()


def test_json_config_migration_restores_backup_when_atomic_commit_fails(
    tmp_path, monkeypatch
):
    import opencut.user_data as user_data

    filename = "migration-commit-failure.json"
    source = {"_schema_version": 0, "value": "safe"}
    (tmp_path / filename).write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    user_data.register_config_schema(
        filename,
        version=1,
        migrations={1: lambda data: {**data, "new_value": "candidate"}},
    )
    real_write = user_data.write_user_file
    failed_once = {"value": False}

    def fail_commit_once(target, data):
        if (
            target == filename
            and data.get("_schema_version") == 1
            and not failed_once["value"]
        ):
            failed_once["value"] = True
            raise OSError("simulated atomic promotion failure")
        return real_write(target, data)

    monkeypatch.setattr(user_data, "write_user_file", fail_commit_once)

    assert user_data.read_user_file_versioned(filename) == source
    assert json.loads((tmp_path / filename).read_text(encoding="utf-8")) == source
    assert not (tmp_path / f"{filename}.migration-backup").exists()


def test_json_config_migration_rejects_unknown_future_schema_without_writing(
    tmp_path, monkeypatch
):
    import opencut.user_data as user_data

    filename = "migration-future.json"
    future = {"_schema_version": 99, "future_key": "preserve"}
    path = tmp_path / filename
    path.write_text(json.dumps(future, separators=(",", ":")), encoding="utf-8")
    original_bytes = path.read_bytes()
    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path))
    user_data.register_config_schema(filename, version=2, migrations={})

    with pytest.raises(user_data.ConfigSchemaVersionError, match="newer than supported"):
        user_data.read_user_file_versioned(filename)

    assert path.read_bytes() == original_bytes
    assert not (tmp_path / f"{filename}.migration-backup").exists()


def test_workflow_save_rejects_non_object_json_body(client, csrf_token):
    resp = client.post(
        "/workflow/save",
        data=json.dumps(["bad-body"]),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_INPUT"
    assert "top-level JSON object" in data["suggestion"]


def test_plugins_install_rejects_non_object_json_body(client, csrf_token):
    resp = client.post(
        "/plugins/install",
        data=json.dumps(["bad-body"]),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "INVALID_INPUT"
    assert "top-level JSON object" in data["suggestion"]


def test_app_level_csrf_guard_covers_unwrapped_mutating_routes(app):
    @app.route("/_test/unwrapped-mutating", methods=["POST"])
    def _unwrapped_mutating_route():
        return {"ok": True}

    local_client = app.test_client()

    missing_token = local_client.post("/_test/unwrapped-mutating", json={})
    assert missing_token.status_code == 403
    assert missing_token.get_json()["error"] == "Invalid or missing CSRF token"

    token = local_client.get("/health").get_json()["csrf_token"]
    valid_token = local_client.post(
        "/_test/unwrapped-mutating",
        json={},
        headers=csrf_headers(token),
    )
    assert valid_token.status_code == 200
    assert valid_token.get_json() == {"ok": True}


def test_template_save_allows_overwrite_at_limit(client, csrf_token, tmp_path, monkeypatch):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    same_template = templates_dir / "my_template.json"
    same_template.write_text(json.dumps({"id": "user_my_template", "name": "My Template"}), encoding="utf-8")
    for idx in range(49):
        (templates_dir / f"tpl_{idx}.json").write_text(
            json.dumps({"id": f"user_tpl_{idx}", "name": f"Template {idx}"}),
            encoding="utf-8",
        )

    monkeypatch.setattr("opencut.routes.settings.OPENCUT_DIR", str(tmp_path))

    resp = client.post(
        "/templates/save",
        data=json.dumps({"name": "My Template", "description": "Updated"}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["template"]["id"] == "user_my_template"
    saved = json.loads(same_template.read_text(encoding="utf-8"))
    assert saved["description"] == "Updated"


def test_template_save_disambiguates_slug_collisions(client, csrf_token, tmp_path, monkeypatch):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    existing = templates_dir / "my_template.json"
    existing.write_text(json.dumps({"id": "user_my_template", "name": "My/Template"}), encoding="utf-8")

    monkeypatch.setattr("opencut.routes.settings.OPENCUT_DIR", str(tmp_path))

    resp = client.post(
        "/templates/save",
        data=json.dumps({"name": "My Template"}),
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["template"]["id"] == "user_my_template_2"
    assert existing.exists()
    assert (templates_dir / "my_template_2.json").exists()


def test_server_main_fatal_noninteractive_returns_nonzero(monkeypatch, tmp_path):
    import opencut.server as server

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    class _DummyStdin:
        def isatty(self):
            return False

    prompts = []
    original_expanduser = server.os.path.expanduser

    monkeypatch.setattr(server, "run_server", _boom)
    monkeypatch.setattr(server.sys, "argv", ["opencut-server"])
    monkeypatch.setattr(server.sys, "stdin", _DummyStdin())
    monkeypatch.setattr(
        server.os.path,
        "expanduser",
        lambda value: str(tmp_path) if value == "~" else original_expanduser(value),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt))

    assert server.main() == 1
    assert prompts == []


def test_server_main_rejects_remote_bind_without_opt_in(monkeypatch, capsys):
    import opencut.server as server

    calls = []

    monkeypatch.delenv("OPENCUT_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(server.sys, "argv", ["opencut-server", "--host", "0.0.0.0"])
    monkeypatch.setattr(server, "run_server", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert server.main() == 2
    assert calls == []

    out = capsys.readouterr().out
    assert "Refusing to bind OpenCut to a non-loopback host" in out
    assert "OPENCUT_ALLOW_REMOTE=1" in out


def test_create_app_fails_closed_when_remote_auth_gate_cannot_install(monkeypatch):
    from opencut.config import OpenCutConfig
    import opencut.server as server

    def _boom(*args, **kwargs):
        raise RuntimeError("auth import failed")

    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    monkeypatch.setattr(server, "_install_remote_auth_middleware", _boom)

    with pytest.raises(RuntimeError, match="Remote auth middleware install failed"):
        server.create_app(config=OpenCutConfig())


def test_server_main_allows_remote_bind_with_explicit_opt_in(monkeypatch, capsys):
    import opencut.server as server

    calls = []

    monkeypatch.setenv("OPENCUT_ALLOW_REMOTE", "1")
    monkeypatch.setattr(server.sys, "argv", ["opencut-server", "--host", "0.0.0.0", "--port", "5680"])
    monkeypatch.setattr(server, "run_server", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert server.main() == 0
    assert calls == [((), {"host": "0.0.0.0", "port": 5680, "debug": False})]

    out = capsys.readouterr().out
    assert "WARNING: Binding OpenCut to non-loopback host 0.0.0.0" in out


def test_server_loopback_host_detection_accepts_only_local_addresses():
    import opencut.server as server

    assert server._is_loopback_host("127.0.0.1")
    assert server._is_loopback_host("localhost")
    assert server._is_loopback_host("::1")
    assert not server._is_loopback_host("0.0.0.0")
    assert not server._is_loopback_host("192.168.1.25")


def test_server_uses_production_wsgi_for_remote_non_debug():
    import opencut.server as server

    assert server._should_use_production_wsgi(host="0.0.0.0", debug=False)
    assert not server._should_use_production_wsgi(host="0.0.0.0", debug=True)
    assert not server._should_use_production_wsgi(host="127.0.0.1", debug=False)


def test_remote_wsgi_server_invokes_waitress(monkeypatch):
    import opencut.server as server

    calls = []
    waitress = types.ModuleType("waitress")
    waitress.serve = lambda app, **kwargs: calls.append((app, kwargs))
    monkeypatch.setitem(server.sys.modules, "waitress", waitress)
    monkeypatch.setenv("OPENCUT_WAITRESS_THREADS", "12")

    app = object()
    server._serve_wsgi_app(app, host="0.0.0.0", port=5679, debug=False)

    assert calls == [(app, {"host": "0.0.0.0", "port": 5679, "threads": 12})]


def test_loopback_wsgi_server_uses_flask_dev_server():
    import opencut.server as server

    calls = []

    class FakeApp:
        def run(self, **kwargs):
            calls.append(kwargs)

    server._serve_wsgi_app(FakeApp(), host="127.0.0.1", port=5679, debug=False)

    assert calls == [{"host": "127.0.0.1", "port": 5679, "debug": False, "threaded": True}]
