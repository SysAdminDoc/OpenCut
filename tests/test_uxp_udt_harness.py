"""F267 UXP Developer Tool smoke-harness guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.core.cep_uxp_parity import build_manifest
from opencut.core.uxp_udt_harness import build_udt_harness_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_MANIFEST = REPO_ROOT / "opencut" / "_generated" / "uxp_udt_harness.json"
PANEL_MANIFEST = REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-udt-harness.json"
UDT_SCRIPT = REPO_ROOT / "extension" / "com.opencut.uxp" / "udt-smoke.js"
INDEX_HTML = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def test_f267_harness_covers_all_direct_uxp_host_actions():
    parity = build_manifest()
    direct_actions = [
        entry["name"]
        for entry in parity["functions"]
        if entry["status"] == "direct_uxp"
    ]
    manifest = build_udt_harness_manifest()
    expected_actions = [
        *direct_actions,
        "ocGetCaptionTrackSnapshot",
        "ocCreateSubsequenceFromRange",
    ]

    assert manifest["f_number"] == "F267"
    assert manifest["scenario_count"] == 16
    assert manifest["actions"] == expected_actions
    assert {scenario["action"] for scenario in manifest["scenarios"]} == set(expected_actions)
    assert "ocAddNativeCaptionTrack" not in manifest["actions"]
    assert "ocQeReflect" not in manifest["actions"]
    assert "ocApplySequenceCuts" not in manifest["actions"]
    assert "ocEmitPingEvent" not in manifest["actions"]


def test_f267_scenarios_have_payloads_and_safety_boundaries():
    manifest = build_udt_harness_manifest()

    assert manifest["safe_default_count"] == 6
    assert manifest["mutating_count"] == 9
    assert manifest["file_write_count"] == 1
    caption_snapshot = next(
        scenario for scenario in manifest["scenarios"]
        if scenario["action"] == "ocGetCaptionTrackSnapshot"
    )
    assert "caption_api_missing" in caption_snapshot["acceptable_blockers"]
    assert caption_snapshot["safe_by_default"] is True
    assert caption_snapshot["mutates_project"] is False
    subsequence = next(
        scenario for scenario in manifest["scenarios"]
        if scenario["action"] == "ocCreateSubsequenceFromRange"
    )
    assert subsequence["fixture"] == "disposable_active_sequence"
    assert subsequence["payload"] == {
        "startSeconds": 0,
        "endSeconds": 1,
        "ignoreTrackTargeting": True,
    }
    assert subsequence["expected_result_keys"] == [
        "ok",
        "sequenceName",
        "rangeVerification",
        "restoration",
    ]
    assert "Premiere did not restore the original sequence range" in subsequence["acceptable_blockers"]
    for scenario in manifest["scenarios"]:
        assert scenario["id"].startswith("f267-")
        assert scenario["status"] == "direct_uxp"
        assert scenario["risk"] == "low"
        assert isinstance(scenario["payload"], dict)
        assert scenario["fixture"]
        assert isinstance(scenario["mutates_project"], bool)
        assert isinstance(scenario["writes_files"], bool)
        assert isinstance(scenario["safe_by_default"], bool)
        assert "ok" in scenario["expected_result_keys"]
        assert "No active sequence" in scenario["acceptable_blockers"]


def test_f267_generated_and_panel_harness_are_in_sync():
    generated = json.loads(GENERATED_MANIFEST.read_text(encoding="utf-8"))
    panel = json.loads(PANEL_MANIFEST.read_text(encoding="utf-8"))
    expected = build_udt_harness_manifest()

    assert generated == expected
    assert panel == expected


def test_f267_cli_check_passes_in_sync():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_uxp_udt_harness", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "16 scenarios" in result.stdout


def test_f267_panel_loads_bundled_udt_harness_runner():
    html = INDEX_HTML.read_text(encoding="utf-8", errors="replace")
    source = UDT_SCRIPT.read_text(encoding="utf-8", errors="replace")

    assert 'localStorage.getItem("opencut_debug") === "1"' in html
    assert 'import("./udt-smoke.js")' in html
    assert "window.OpenCutUXPUdtHarness" in source
    assert "uxp-udt-harness.json" in source
    assert "OpenCutUXPHost.executeHostAction" in source
    assert "includeMutating" in source
    assert "safe_by_default" in source
    assert "Skipped by default" in source
    assert "acceptable_blockers" in source


def test_f267_release_smoke_runs_harness_guardrail():
    source = RELEASE_SMOKE.read_text(encoding="utf-8", errors="replace")

    assert '"tests/test_uxp_udt_harness.py"' in source
