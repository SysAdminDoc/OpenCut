import importlib
import json
import os
from unittest.mock import patch

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
