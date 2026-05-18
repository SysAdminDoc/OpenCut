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


def test_ci_workflows_run_panel_unit_tests():
    release_workflow = (REPO_ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    pr_workflow = (REPO_ROOT / ".github" / "workflows" / "pr-fast.yml").read_text(encoding="utf-8")

    assert "npm test" in release_workflow
    assert "actions/setup-node@v4" in pr_workflow
    assert "npm ci" in pr_workflow
    assert "npm ci --omit=optional" not in pr_workflow
    assert "--skip panel-unit" not in pr_workflow
