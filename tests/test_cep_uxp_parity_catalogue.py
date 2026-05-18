"""F198 CEP-only route catalogue guardrails."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from opencut.core.cep_uxp_parity import (
    build_manifest,
    cep_only_names,
    get_parity_entry,
    parity_names,
    validate_catalogue,
)
from opencut.tools import dump_cep_uxp_parity as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_JSX = REPO_ROOT / "extension" / "com.opencut.panel" / "host" / "index.jsx"
MATRIX_DOC = REPO_ROOT / ".ai" / "research" / "2026-05-17" / "CEP_UXP_PARITY_MATRIX.md"
UXP_DOC = REPO_ROOT / "docs" / "UXP_MIGRATION.md"
MANIFEST = REPO_ROOT / "opencut" / "_generated" / "cep_uxp_parity.json"


def _host_oc_functions() -> tuple[str, ...]:
    text = HOST_JSX.read_text(encoding="utf-8")
    names = re.findall(r"\bfunction\s+(oc[A-Za-z0-9_]+)\s*\(", text)
    return tuple(dict.fromkeys(names))


def test_catalogue_matches_host_jsx_surface():
    host_names = _host_oc_functions()

    assert len(host_names) == 18
    assert tuple(parity_names()) == host_names
    assert validate_catalogue(host_names) == []


def test_catalogue_pins_the_true_cep_only_surface():
    assert cep_only_names() == ("ocAddNativeCaptionTrack", "ocQeReflect")

    caption_track = get_parity_entry("ocAddNativeCaptionTrack")
    qe_reflect = get_parity_entry("ocQeReflect")
    apply_cuts = get_parity_entry("ocApplySequenceCuts")

    assert caption_track.risk == "high"
    assert "F253 Hybrid Plugin" in caption_track.replacement_plan
    assert qe_reflect.risk == "high"
    assert "Retire QE reflection" in qe_reflect.replacement_plan
    assert apply_cuts.status == "partial_uxp"
    assert not apply_cuts.cep_only


def test_committed_manifest_matches_live_catalogue():
    assert MANIFEST.is_file(), f"F198 manifest must exist at {MANIFEST}"

    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live = build_manifest()

    assert committed == live
    assert committed["function_count"] == 18
    assert committed["cep_only_count"] == 2
    assert committed["status_counts"]["cep_only"] == 2
    assert committed["status_counts"]["partial_uxp"] == 1


def test_docs_reference_the_catalogued_cep_only_paths():
    docs = [
        MATRIX_DOC.read_text(encoding="utf-8"),
        UXP_DOC.read_text(encoding="utf-8"),
    ]
    for function_name in cep_only_names():
        for doc in docs:
            assert function_name in doc
    assert "No new user-facing feature should depend on QE reflection." in docs[1]


def test_cli_check_passes_in_sync():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_cep_uxp_parity", "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"--check should pass when in sync; stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert "in sync" in result.stdout


def test_cli_writes_to_custom_path(tmp_path):
    target = tmp_path / "cep_uxp_parity.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencut.tools.dump_cep_uxp_parity",
            "--output",
            str(target),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )

    assert result.returncode == 0
    assert json.loads(target.read_text(encoding="utf-8")) == tool.build_manifest()
