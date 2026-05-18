"""F260 UXP migration risk dashboard guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.core.cep_uxp_parity import build_dashboard_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DASHBOARD = REPO_ROOT / "opencut" / "_generated" / "uxp_migration_dashboard.json"
PANEL_DASHBOARD = REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-migration-dashboard.json"
UXP_HTML = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_MAIN = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f260_dashboard_manifest_derives_from_f198_catalogue():
    manifest = build_dashboard_manifest()

    assert manifest["dashboard_version"] == 1
    assert manifest["source_catalogue_version"] == 1
    assert manifest["summary"]["function_count"] == 18
    assert manifest["summary"]["direct_uxp"] == 14
    assert manifest["summary"]["partial_uxp"] == 1
    assert manifest["summary"]["cep_only"] == 2
    assert manifest["summary"]["hybrid_candidates"] == 2
    assert manifest["cep_only"] == ["ocAddNativeCaptionTrack", "ocQeReflect"]
    assert {row["name"] for row in manifest["priority"]} >= {
        "ocAddNativeCaptionTrack",
        "ocQeReflect",
        "ocApplySequenceCuts",
    }


def test_f260_generated_and_panel_dashboards_are_in_sync():
    live = build_dashboard_manifest()

    assert GENERATED_DASHBOARD.is_file()
    assert PANEL_DASHBOARD.is_file()
    assert json.loads(GENERATED_DASHBOARD.read_text(encoding="utf-8")) == live
    assert json.loads(PANEL_DASHBOARD.read_text(encoding="utf-8")) == live


def test_f260_cli_check_passes_in_sync():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_uxp_migration_dashboard", "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"--check should pass when in sync; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "UXP migration dashboard artifacts in sync" in result.stdout


def test_f260_cli_writes_both_artifacts(tmp_path):
    generated = tmp_path / "generated.json"
    panel = tmp_path / "panel.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencut.tools.dump_uxp_migration_dashboard",
            "--output",
            str(generated),
            "--panel-output",
            str(panel),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert result.returncode == 0
    assert json.loads(generated.read_text(encoding="utf-8")) == build_dashboard_manifest()
    assert json.loads(panel.read_text(encoding="utf-8")) == build_dashboard_manifest()


def test_f260_settings_panel_surfaces_migration_dashboard():
    html = _read(UXP_HTML)

    assert "Migration Risk" in html
    assert "settingsMigrationDirectValue" in html
    assert "settingsMigrationFallbackValue" in html
    assert "settingsMigrationRiskValue" in html
    assert "uxpMigrationRiskGrid" in html
    assert "uxpRefreshMigrationRiskBtn" in html


def test_f260_panel_loads_dashboard_json_and_renders_rows():
    source = _read(UXP_MAIN)

    assert "async function uxpLoadMigrationRisk()" in source
    assert 'fetch(`uxp-migration-dashboard.json?ts=${Date.now()}`, { cache: "no-store" })' in source
    assert "migrationStatusLabel" in source
    assert "migrationStateClass" in source
    assert "settingsMigrationStatus" in source
    assert "uxpLoadMigrationRisk();" in source
    assert 'document.getElementById("uxpRefreshMigrationRiskBtn")?.addEventListener("click", uxpLoadMigrationRisk)' in source


def test_f260_release_smoke_runs_dashboard_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_migration_dashboard.py" in source
