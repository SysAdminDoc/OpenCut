"""F257 UXP ObjectMaskUtils API integration guardrails."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
VERSIONS = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f257_beta_snapshot_tracks_object_mask_api_version():
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))

    assert data["package"] == "@adobe/premierepro"
    assert data["dist_tags"]["beta"] == "26.3.0-beta.85"
    assert "26.3.0-beta.85" in data["tracked_versions"]


def test_f257_bridge_checks_object_mask_state():
    source = _read(MAIN_JS)

    assert "async function getObjectMaskState(payload)" in source
    assert "ppro?.ObjectMaskUtils?.hasObjectMask" in source
    assert "Premiere Object Mask APIs are unavailable." in source
    assert "ppro.ObjectMaskUtils.hasObjectMask(target)" in source
    assert "hasObjectMask" in source


def test_f257_bridge_supports_project_and_sequence_targets():
    source = _read(MAIN_JS)

    assert 'parsed.target ?? parsed.scope ?? "sequence"' in source
    assert 'requestedTarget === "project" ? "project" : "sequence"' in source
    assert "target = context?.proj ?? null" in source
    assert "target = await getActiveSequence()" in source


def test_f257_webview_host_exposes_object_mask_helper():
    source = _read(MAIN_JS)

    assert "window.OpenCutUXPHost" in source
    assert "getObjectMaskState: (payload) => PProBridge.getObjectMaskState(payload)" in source


def test_f257_release_smoke_runs_object_mask_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_object_mask_api_integration.py" in source
