"""F252 UXP UDT result-capture validation guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.core.uxp_udt_harness import build_udt_harness_manifest
from opencut.core.uxp_udt_results import (
    build_udt_result_template,
    validate_udt_result_capture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _passing_capture() -> dict:
    manifest = build_udt_harness_manifest()
    results = [
        {
            "id": scenario["id"],
            "action": scenario["action"],
            "status": "passed",
            "durationMs": 1,
            "result": {"ok": True},
        }
        for scenario in manifest["scenarios"]
    ]
    return {
        "harnessVersion": manifest["harness_version"],
        "scenarioCount": manifest["scenario_count"],
        "includeMutating": True,
        "summary": {"passed": len(results), "failed": 0, "blocked": 0, "skipped": 0},
        "results": results,
    }


def test_f252_result_template_records_capture_contract():
    template = build_udt_result_template()

    assert template["schema_version"] == 1
    assert template["f_number"] == "F252"
    assert template["source_harness_f_number"] == "F267"
    assert "includeMutating: true" in template["capture_command"]
    assert template["capture"]["scenarioCount"] == 14
    assert template["capture"]["includeMutating"] is True


def test_f252_passing_capture_is_ready_for_webview_cutover():
    report = validate_udt_result_capture(_passing_capture())

    assert report["ok"] is True
    assert report["ready_for_webview_cutover"] is True
    assert report["summary"] == {"passed": 14, "failed": 0, "blocked": 0, "skipped": 0}
    assert report["missing_ids"] == []


def test_f252_cutover_validation_rejects_read_only_capture():
    capture = _passing_capture()
    capture["includeMutating"] = False
    capture["results"] = [result for result in capture["results"][:5]]
    capture["summary"] = {"passed": 5, "failed": 0, "blocked": 0, "skipped": 0}

    report = validate_udt_result_capture(capture)

    assert report["ok"] is False
    assert report["ready_for_webview_cutover"] is False
    assert any("includeMutating must be true" in error for error in report["errors"])
    assert any("missing scenario results" in error for error in report["errors"])


def test_f252_blocked_capture_is_diagnostic_only_when_allowed():
    capture = _passing_capture()
    capture["results"][0]["status"] = "blocked"
    capture["results"][0]["reason"] = "No active sequence"
    capture["summary"] = {"passed": 13, "failed": 0, "blocked": 1, "skipped": 0}

    strict = validate_udt_result_capture(capture)
    diagnostic = validate_udt_result_capture(capture, allow_blocked=True)

    assert strict["ok"] is False
    assert strict["ready_for_webview_cutover"] is False
    assert diagnostic["ok"] is True
    assert diagnostic["ready_for_webview_cutover"] is False


def test_f252_validator_detects_summary_drift():
    capture = _passing_capture()
    capture["summary"]["passed"] = 0

    report = validate_udt_result_capture(capture)

    assert report["ok"] is False
    assert any("summary.passed must be 14" in error for error in report["errors"])


def test_f252_cli_template_and_validation(tmp_path):
    capture_path = tmp_path / "udt-result.json"
    capture_path.write_text(json.dumps({"capture": _passing_capture()}), encoding="utf-8")

    template = subprocess.run(
        [sys.executable, "-m", "opencut.tools.validate_uxp_udt_results", "--template"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert template.returncode == 0, template.stderr
    assert '"f_number": "F252"' in template.stdout

    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.validate_uxp_udt_results", str(capture_path), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ready_for_webview_cutover"] is True


def test_f252_release_smoke_runs_udt_result_guardrail():
    source = RELEASE_SMOKE.read_text(encoding="utf-8", errors="replace")

    assert '"tests/test_uxp_udt_results.py"' in source
