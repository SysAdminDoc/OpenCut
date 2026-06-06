"""F254 UXP createSubsequence integration guardrails."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
VERSIONS = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f254_beta_snapshot_still_tracks_create_subsequence_version():
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))

    assert data["package"] == "@adobe/premierepro"
    assert data["dist_tags"]["beta"] == "26.3.0-beta.85"
    assert "26.3.0-beta.85" in data["tracked_versions"]


def test_f254_uxp_bridge_sets_range_before_creating_subsequence():
    source = _read(MAIN_JS)

    assert "async function createSubsequenceFromRange(payload)" in source
    assert "ppro?.TickTime?.createWithSeconds" in source
    assert "seq.createSetInPointAction" in source
    assert "seq.createSetOutPointAction" in source
    assert "seq.createSubsequence(ignoreTrackTargeting)" in source
    assert "OpenCut set subsequence range" in source
    assert "OpenCut restore sequence range" in source


def test_f254_range_restore_uses_project_transaction():
    source = _read(MAIN_JS)

    assert "async function _executeProjectActions(actions, undoString)" in source
    assert "context.proj.executeTransaction" in source
    assert "compoundAction.addAction(action)" in source
    assert "finally" in source
    assert "A valid start/end range in seconds is required." in source


def test_f254_export_dispatch_passes_subsequence_to_encoder():
    source = _read(MAIN_JS)

    assert "const subsequence = await createSubsequenceFromRange(parsed)" in source
    assert "const exportResult = await exportSubsequenceWithEncoder(subsequence.sequence, parsed)" in source
    assert "sequenceName: subsequence.sequenceName" in source
    assert "ignoreTrackTargeting: subsequence.ignoreTrackTargeting" in source
    assert "createSubsequenceFromRange" in source


def test_f254_release_smoke_runs_subsequence_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_create_subsequence_integration.py" in source
