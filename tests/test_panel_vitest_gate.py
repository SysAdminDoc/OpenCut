from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = REPO_ROOT / "extension" / "com.opencut.panel"


def test_panel_package_exposes_vitest_suite():
    package = json.loads((PANEL_DIR / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["test"] == "vitest run --config vitest.config.mjs"
    assert package["devDependencies"]["vitest"].startswith("^4.")
    assert (PANEL_DIR / "vitest.config.mjs").is_file()
    assert (PANEL_DIR / "tests" / "panel-utils.test.mjs").is_file()
    assert (PANEL_DIR / "tests" / "uxp-utils.test.mjs").is_file()


def test_release_smoke_runs_panel_unit_gate():
    smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")
    assert 'StepDefinition("panel-unit", step_panel_unit' in smoke
    assert "tests/test_panel_vitest_gate.py" in smoke


def test_release_smoke_runs_panel_unit_with_local_npm_test():
    smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")

    assert 'StepDefinition("panel-unit", step_panel_unit' in smoke
    assert '_npm_command("test", cwd=PANEL_DIR)' in smoke
    assert "--skip panel-unit" not in smoke
