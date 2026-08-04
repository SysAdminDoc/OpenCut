"""Installation and health checks for the shipped example plugins."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner
from flask import Flask

EXAMPLE_NAMES = ("clip-notes", "long-job-demo", "timecode-watermark")
EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "opencut" / "data" / "example_plugins"


def _install_examples(destination: Path) -> Path:
    plugins_dir = destination / "plugins"
    plugins_dir.mkdir()
    for name in EXAMPLE_NAMES:
        shutil.copytree(
            EXAMPLE_ROOT / name,
            plugins_dir / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        from opencut.core.plugin_manifest import write_plugin_lock

        write_plugin_lock(plugins_dir / name)
    return plugins_dir


def test_shipped_examples_install_load_and_doctor(monkeypatch, tmp_path):
    from opencut.core import plugin_manifest
    from opencut.core import plugins as plugin_runtime
    from opencut.cli import cli

    plugins_dir = _install_examples(tmp_path)
    monkeypatch.setattr(plugin_runtime, "PLUGINS_DIR", str(plugins_dir))
    app = Flask("example-plugin-test")

    try:
        loaded = plugin_runtime.load_all_plugins(app)
        assert set(loaded["loaded"]) == set(EXAMPLE_NAMES)
        assert loaded["failed"] == []

        real_doctor = plugin_manifest.doctor
        monkeypatch.setattr(
            plugin_manifest,
            "doctor",
            lambda: real_doctor(plugins_dir),
        )
        result = CliRunner().invoke(cli, ["plugins", "doctor", "--json"])
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["healthy"] == len(EXAMPLE_NAMES)
        assert report["incompatible"] == 0
        assert report["invalid"] == 0
    finally:
        for name in EXAMPLE_NAMES:
            plugin_runtime.unload_plugin(name)
