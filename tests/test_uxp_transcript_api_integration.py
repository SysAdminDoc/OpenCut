"""F256 UXP Transcript API integration guardrails."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
VERSIONS = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f256_beta_snapshot_tracks_transcript_api_version():
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))

    assert data["package"] == "@adobe/premierepro"
    assert data["dist_tags"]["beta"] == "26.3.0-beta.85"
    assert "26.3.0-beta.85" in data["tracked_versions"]


def test_f256_bridge_queries_supported_transcript_languages():
    source = _read(MAIN_JS)

    assert "function querySupportedTranscriptLanguages()" in source
    assert "ppro?.Transcript?.querySupportedLanguages" in source
    assert "ppro.Transcript.querySupportedLanguages()" in source
    assert "Premiere Transcript APIs are unavailable." in source


def test_f256_bridge_checks_clip_transcript_state():
    source = _read(MAIN_JS)

    assert "async function _clipProjectItemFromPayload(payload)" in source
    assert "ppro.ClipProjectItem.cast(target.item)" in source
    assert "async function getTranscriptState(payload)" in source
    assert "ppro?.Transcript?.hasTranscript" in source
    assert "ppro.Transcript.hasTranscript(clipProjectItem)" in source
    assert "ppro.Transcript.exportToJSON(clipProjectItem)" in source
    assert "includeJson === true" in source


def test_f256_webview_host_exposes_transcript_helpers():
    source = _read(MAIN_JS)

    assert "window.OpenCutUXPHost" in source
    assert "querySupportedTranscriptLanguages: () => PProBridge.querySupportedTranscriptLanguages()" in source
    assert "getTranscriptState: (payload) => PProBridge.getTranscriptState(payload)" in source


def test_f256_release_smoke_runs_transcript_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_transcript_api_integration.py" in source
