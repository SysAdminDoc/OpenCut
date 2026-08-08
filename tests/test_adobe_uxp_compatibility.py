"""Static Adobe Premiere UXP compatibility and drift-gate tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.tools import adobe_uxp_compatibility as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "opencut" / "_generated" / "adobe_uxp_compatibility.json"


def test_generated_manifest_records_used_api_contracts_and_precise_sources():
    assert MANIFEST.is_file()
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert committed["manifest_version"] == tool.MANIFEST_VERSION
    assert committed["package"] == "@adobe/premierepro"
    assert committed["api_version"] == 2
    assert committed["minimum_host"] == "25.6"
    assert committed["used_capability_count"] >= 60
    assert committed["diagnostics"]["undeclared_capabilities"] == []

    by_id = {row["id"]: row for row in committed["capabilities"]}
    for capability_id in (
        "Sequence.setSelection",
        "Transcript.querySupportedLanguages",
        "Transcript.hasTranscript",
        "Transcript.exportToJSON",
        "Project.lockedAccess",
    ):
        row = by_id[capability_id]
        assert row["package"] == "@adobe/premierepro"
        assert row["package_version"] == "26.3.0"
        assert row["fallback"]
        assert row["host_behavior"]

    assert any(
        ref["file"] == "extension/com.opencut.uxp/main.js"
        and ref["line"] > 0
        for ref in by_id["Transcript.exportToJSON"]["source_refs"]
    )


def test_set_selection_behavior_is_pinned_for_all_supported_host_bands():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert tool.validate_host_fixtures(manifest) == []
    assert tool.capability_for_host(manifest, "Sequence.setSelection", "25.6")["sync_async"] == "async"
    assert tool.capability_for_host(manifest, "Sequence.setSelection", "26.2")["sync_async"] == "async"
    assert tool.capability_for_host(manifest, "Sequence.setSelection", "26.3")["sync_async"] == "sync"


def test_source_scanner_reports_new_capabilities_with_file_and_line(tmp_path):
    source = tmp_path / "extension" / "com.opencut.uxp" / "main.js"
    source.parent.mkdir(parents=True)
    source.write_text(
        'const result = await seq.newFutureApi();\n'
        'const known = await seq.getSelection();\n',
        encoding="utf-8",
    )

    scan = tool.scan_sources(tmp_path)

    assert scan["source_files"] == ["extension/com.opencut.uxp/main.js"]
    assert scan["undeclared_capabilities"] == [
        {
            "file": "extension/com.opencut.uxp/main.js",
            "line": 1,
            "symbol": "Sequence.newFutureApi",
            "message": "undeclared UXP capability Sequence.newFutureApi",
        }
    ]
    known_id = next(
        capability["id"]
        for capability in tool.API_CATALOGUE
        if capability["id"] == "Sequence.getSelection"
    )
    assert scan["uses"][known_id][0]["line"] == 2


def test_manifest_diff_ignores_generation_timestamp():
    manifest = tool.build_manifest()
    changed = dict(manifest)
    changed["generated_at"] = "2099-01-01T00:00:00Z"
    assert tool.diff_manifests(manifest, changed) == {"changed": False, "fields": {}}


def test_cli_check_is_local_and_reports_in_sync():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencut.tools.adobe_uxp_compatibility",
            "--check",
            "--json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["drift"]["changed"] is False
    assert payload["manifest"]["package_drift"]["requires_live_premiere"] is False
    assert payload["diagnostics"] == []
