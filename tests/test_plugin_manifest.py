"""Tests for the plugin manifest validator + lock generator (F116)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencut.core import plugin_manifest as pm


def _write_manifest(plugin_dir: Path, overrides: dict | None = None) -> dict:
    payload = {
        "name": "demo",
        "version": "1.0.0",
        "description": "demo plugin",
        "api_version": 1,
        "capabilities": ["http.routes"],
        "routes": [{"path": "/hello"}],
    }
    if overrides:
        payload.update(overrides)
    (plugin_dir / pm.MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_lock(plugin_dir: Path) -> Path:
    return pm.write_plugin_lock(plugin_dir)


def test_schema_validator_accepts_well_formed_manifest():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 1,
            "capabilities": ["http.routes"],
        }
    )
    assert result.valid
    assert not result.errors


def test_schema_validator_rejects_unknown_capability():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 1,
            "capabilities": ["totally.wrong"],
        }
    )
    assert not result.valid
    assert any("unknown values" in e for e in result.errors)


def test_schema_warns_on_open_network_capability():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 1,
            "capabilities": ["host.network"],
        }
    )
    assert result.valid is True
    assert any("network" in w for w in result.warnings)


def test_schema_accepts_declared_plugin_jobs_with_capability():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 1,
            "capabilities": ["http.routes", "jobs.register"],
            "jobs": [{"id": "long_task", "label": "Long Task"}],
        }
    )
    assert result.valid
    assert not result.errors


def test_schema_rejects_jobs_without_register_capability():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 1,
            "capabilities": ["http.routes"],
            "jobs": [{"id": "long_task"}],
        }
    )
    assert not result.valid
    assert any("jobs.register" in e for e in result.errors)


def test_schema_rejects_invalid_plugin_job_id():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 1,
            "capabilities": ["http.routes", "jobs.register"],
            "jobs": [{"id": "bad job!"}],
        }
    )
    assert not result.valid
    assert any("invalid id" in e for e in result.errors)


def test_schema_requires_api_version_one():
    result = pm.validate_manifest_schema(
        {
            "name": "alpha",
            "version": "0.1",
            "description": "test",
            "api_version": 99,
        }
    )
    assert not result.valid
    assert any("api_version" in e for e in result.errors)


def test_compute_plugin_lock_is_stable(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b')\n", encoding="utf-8")

    payload_a = pm.compute_plugin_lock(tmp_path)
    payload_b = pm.compute_plugin_lock(tmp_path)

    assert payload_a == payload_b
    assert set(payload_a["files"]) == {"a.py", "b.py"}


def test_write_plugin_lock_creates_sibling_file(tmp_path):
    (tmp_path / "main.py").write_text("# demo", encoding="utf-8")

    target = pm.write_plugin_lock(tmp_path)
    assert target.exists()
    body = json.loads(target.read_text(encoding="utf-8"))
    assert "main.py" in body["files"]


def test_verify_lock_detects_drift(tmp_path):
    (tmp_path / "main.py").write_text("# v1\n", encoding="utf-8")
    pm.write_plugin_lock(tmp_path)

    # Drift the file after the lock was generated.
    (tmp_path / "main.py").write_text("# v2 — tampered\n", encoding="utf-8")

    result = pm.verify_plugin_lock(tmp_path)
    assert not result.valid
    assert any("sha-256 mismatch" in e for e in result.errors)


def test_verify_lock_flags_added_files(tmp_path):
    (tmp_path / "main.py").write_text("# v1\n", encoding="utf-8")
    pm.write_plugin_lock(tmp_path)
    (tmp_path / "extra.py").write_text("# sneaky\n", encoding="utf-8")

    result = pm.verify_plugin_lock(tmp_path)
    assert not result.valid
    assert any("absent from lock" in e for e in result.errors)


def test_verify_lock_missing_file_blocks_by_default(tmp_path):
    (tmp_path / "main.py").write_text("# v1\n", encoding="utf-8")
    # Skip lock generation.
    result = pm.verify_plugin_lock(tmp_path)
    assert not result.valid
    assert any("plugin.lock.json missing" in e for e in result.errors)


def test_verify_lock_can_opt_in_to_unsigned(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("# v1\n", encoding="utf-8")
    monkeypatch.setenv("OPENCUT_PLUGIN_ALLOW_UNSIGNED", "1")
    result = pm.verify_plugin_lock(tmp_path)
    assert result.valid
    assert any("OPENCUT_PLUGIN_ALLOW_UNSIGNED" in w for w in result.warnings)


def test_validate_plugin_manifest_happy_path(tmp_path):
    _write_manifest(tmp_path)
    _write_lock(tmp_path)
    result = pm.validate_plugin_manifest(tmp_path)
    assert result.valid


def test_validate_plugin_manifest_fails_when_manifest_missing(tmp_path):
    result = pm.validate_plugin_manifest(tmp_path)
    assert not result.valid
    assert any("plugin.json missing" in e for e in result.errors)


def test_validate_plugin_manifest_fails_when_lock_drifted(tmp_path):
    _write_manifest(tmp_path)
    (tmp_path / "main.py").write_text("# v1\n", encoding="utf-8")
    _write_lock(tmp_path)
    (tmp_path / "main.py").write_text("# v2\n", encoding="utf-8")
    result = pm.validate_plugin_manifest(tmp_path)
    assert not result.valid
    assert any("sha-256 mismatch" in e for e in result.errors)


def test_long_job_demo_example_manifest_schema_is_valid():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "opencut"
        / "data"
        / "example_plugins"
        / "long-job-demo"
        / pm.MANIFEST_FILENAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = pm.validate_manifest_schema(manifest)

    assert result.valid
    assert not result.errors
