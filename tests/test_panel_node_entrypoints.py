"""RA-36 guards for UNC/HGFS-safe CEP panel Node entry points."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = REPO_ROOT / "extension" / "com.opencut.panel"
PACKAGE_JSON = PANEL_ROOT / "package.json"
WRAPPER = PANEL_ROOT / "scripts" / "panel-node-gate.ps1"
NODE_ADVISORIES_DOC = REPO_ROOT / "docs" / "NODE_ADVISORIES.md"
README = REPO_ROOT / "README.md"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"

WIN_WRAPPER_PREFIX = (
    'powershell -NoProfile -ExecutionPolicy Bypass -File "%INIT_CWD%\\scripts\\panel-node-gate.ps1" '
)


def _package_scripts() -> dict[str, str]:
    package = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    return package["scripts"]


def test_panel_package_exposes_windows_safe_node_gate_aliases():
    scripts = _package_scripts()

    expected = {
        "audit:check:win": "audit:check",
        "audit:esbuild:win": "audit:esbuild",
        "build:verify:win": "build:verify",
    }
    for script_name, gate_name in expected.items():
        assert scripts[script_name] == f"{WIN_WRAPPER_PREFIX}{gate_name}"


def test_panel_node_gate_wrapper_is_scriptroot_anchored():
    text = WRAPPER.read_text(encoding="utf-8")

    assert "$PSScriptRoot" in text
    assert "%INIT_CWD%" not in text
    assert "Set-Location" not in text
    assert "Get-Command node" in text
    for script_name in (
        "check-advisories.mjs",
        "check-esbuild-pin.mjs",
        "verify-build.mjs",
    ):
        assert script_name in text


def test_windows_shared_folder_commands_are_documented():
    docs = "\n".join(
        [
            NODE_ADVISORIES_DOC.read_text(encoding="utf-8"),
            README.read_text(encoding="utf-8"),
        ]
    )

    assert "Windows UNC/HGFS" in docs
    for command in (
        "npm run audit:check:win -- --json",
        "npm run audit:esbuild:win -- --json",
        "npm run build:verify:win",
    ):
        assert command in docs


def test_release_smoke_runs_panel_node_entrypoint_guard():
    smoke = RELEASE_SMOKE.read_text(encoding="utf-8")

    assert "tests/test_panel_node_entrypoints.py" in smoke
