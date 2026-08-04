"""auto-editor must resolve the native binary, not the frozen pip package.

Upstream rewrote auto-editor in Nim and stopped publishing to PyPI at 29.3.1
(2025-11-04), so the pinned `auto-editor>=29.3,<30` package is nine months
stale and every 2026 capability - partial-lossless GOP-copy rendering, linked
dissolve transitions, Parakeet TDT word timestamps, MLT export - is only in
the native binary.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pytest

from opencut.core import auto_edit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_binary(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = "auto-editor.exe" if os.name == "nt" else "auto-editor"
    binary = directory / name
    binary.write_text("#!/bin/sh\necho 30.1.0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


class TestResolution:
    def test_env_override_wins_over_everything(self, tmp_path, monkeypatch):
        override = _make_binary(tmp_path / "override")
        monkeypatch.setenv("OPENCUT_AUTO_EDITOR", str(override))
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: "/bundled/auto-editor")
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: "/on/path/auto-editor")

        assert auto_edit.resolve_auto_editor_binary() == str(override)

    def test_a_bad_override_is_ignored_rather_than_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENCUT_AUTO_EDITOR", str(tmp_path / "nope"))
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: None)
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: "/on/path/auto-editor")

        assert auto_edit.resolve_auto_editor_binary() == "/on/path/auto-editor"

    def test_a_bundled_binary_beats_path(self, monkeypatch):
        monkeypatch.delenv("OPENCUT_AUTO_EDITOR", raising=False)
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: "/bundled/auto-editor")
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: "/on/path/auto-editor")

        assert auto_edit.resolve_auto_editor_binary() == "/bundled/auto-editor"

    def test_path_is_used_when_nothing_is_bundled(self, monkeypatch):
        monkeypatch.delenv("OPENCUT_AUTO_EDITOR", raising=False)
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: None)
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: "/on/path/auto-editor")

        assert auto_edit.resolve_auto_editor_binary() == "/on/path/auto-editor"

    def test_the_native_binary_is_preferred_over_the_pip_package(self, monkeypatch):
        monkeypatch.delenv("OPENCUT_AUTO_EDITOR", raising=False)
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: None)
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: "/on/path/auto-editor")

        assert auto_edit._find_auto_editor() == ["/on/path/auto-editor"]

    def test_the_pip_package_is_the_last_resort(self, monkeypatch):
        monkeypatch.delenv("OPENCUT_AUTO_EDITOR", raising=False)
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: None)
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: None)

        assert auto_edit._find_auto_editor() == [sys.executable, "-m", "auto_editor"]

    def test_bundled_lookup_finds_a_real_file(self, tmp_path, monkeypatch):
        """Guard against a search wired to directories that never match."""
        home = tmp_path / "home"
        binary = _make_binary(home / ".opencut" / "bin")
        monkeypatch.setattr(os.path, "expanduser", lambda _p: str(home))

        assert auto_edit._bundled_auto_editor() == str(binary)


class TestVersionProbeAndMessage:
    def test_version_probe_runs_the_resolved_binary(self, monkeypatch):
        calls = []

        def _fake_run(cmd, **_kwargs):
            calls.append(cmd)

            class _Result:
                returncode = 0
                stdout = "30.1.0"

            return _Result()

        monkeypatch.delenv("OPENCUT_AUTO_EDITOR", raising=False)
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: None)
        monkeypatch.setattr(auto_edit.shutil, "which", lambda _name: "/on/path/auto-editor")
        monkeypatch.setattr(auto_edit.subprocess, "run", _fake_run)

        assert auto_edit.check_auto_editor_version() == "30.1.0"
        assert calls == [["/on/path/auto-editor", "--version"]]

    def test_generation_detection_reports_the_native_path(self, monkeypatch):
        monkeypatch.delenv("OPENCUT_AUTO_EDITOR", raising=False)
        monkeypatch.setattr(auto_edit, "_bundled_auto_editor", lambda: "/bundled/auto-editor")
        monkeypatch.setattr(auto_edit, "check_auto_editor_version", lambda: "30.1.0")

        info = auto_edit.detect_auto_editor_generation()
        assert info == {
            "version": "30.1.0",
            "generation": "v30",
            "native": True,
            "path": "/bundled/auto-editor",
        }

    def test_the_absent_message_names_the_native_download(self):
        assert "github.com/WyattBlue/auto-editor/releases" in auto_edit.INSTALL_HINT
        assert "OPENCUT_AUTO_EDITOR" in auto_edit.INSTALL_HINT
        assert "2025-11-04" in auto_edit.INSTALL_HINT

    def test_the_route_and_the_module_share_one_message(self):
        route = (REPO_ROOT / "opencut" / "routes" / "video_editing.py").read_text(
            encoding="utf-8"
        )
        assert "from opencut.core.auto_edit import INSTALL_HINT" in route
        assert "pip install auto-editor" not in route


def test_the_pip_pin_is_documented_as_legacy_only():
    raw = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "LEGACY ONLY" in raw
    assert "2025-11-04" in raw

    extras = tomllib.loads(raw)["project"]["optional-dependencies"]
    # The pin is kept so existing installs keep working; it must not be the
    # only path the project documents.
    assert any("auto-editor" in entry for entry in extras["auto-edit"])


def test_availability_check_accepts_a_binary_with_no_pip_package(monkeypatch):
    from opencut import checks

    monkeypatch.setattr(auto_edit, "resolve_auto_editor_binary", lambda: "/bundled/auto-editor")
    monkeypatch.setattr(
        "opencut.helpers._try_import", lambda *_a, **_k: pytest.fail("pip package probed first")
    )

    assert checks.check_auto_editor_available() is True
