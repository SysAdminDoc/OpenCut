"""F255 UXP EncoderManager integration guardrails."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
VERSIONS = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f255_beta_snapshot_tracks_encoder_manager_version():
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))

    assert data["package"] == "@adobe/premierepro"
    assert data["dist_tags"]["beta"] == "26.3.0-beta.85"
    assert "26.3.0-beta.85" in data["tracked_versions"]


def test_f255_bridge_uses_encoder_manager_export_apis():
    source = _read(MAIN_JS)

    assert "function _encoderManager()" in source
    assert "ppro?.EncoderManager?.getManager?.()" in source
    assert "async function exportSubsequenceWithEncoder(sequence, payload)" in source
    assert "manager.launchEncoder" in source
    assert "manager.exportSequence(sequence, exportType, outputPath, presetFile, exportFull)" in source
    assert "manager.startBatchEncode" in source
    assert "manager.isAMEInstalled === false" in source


def test_f255_export_type_selects_ame_or_immediate():
    source = _read(MAIN_JS)

    assert "function _encoderExportType(queueToAme)" in source
    assert "Constants?.ExportType?.QUEUE_TO_AME" in source
    assert "Constants?.ExportType?.IMMEDIATELY" in source
    assert 'parsed.exportType !== "immediate"' in source
    assert "parsed.queueToAme !== false" in source


def test_f255_export_sequence_range_completes_encoder_handoff():
    source = _read(MAIN_JS)

    assert "const exportResult = await exportSubsequenceWithEncoder(subsequence.sequence, parsed)" in source
    assert "requiresF255" not in source
    assert "An output path is required for UXP encoder export." in source
    assert "Adobe Media Encoder is not installed." in source


def test_f255_release_smoke_runs_encoder_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_encoder_manager_integration.py" in source
