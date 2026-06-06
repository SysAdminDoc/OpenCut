"""F252.2 UXP host-action dispatch guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
PARITY_MANIFEST = REPO_ROOT / "opencut" / "_generated" / "cep_uxp_parity.json"
UXP_ONLY_DIRECT_ACTIONS = {"ocGetCaptionTrackSnapshot"}


def _read_main_js() -> str:
    return MAIN_JS.read_text(encoding="utf-8", errors="replace")


def _direct_action_names(source: str) -> set[str]:
    match = re.search(
        r"const\s+UXP_DIRECT_HOST_ACTIONS\s*=\s*Object\.freeze\(\{(?P<body>.*?)\}\);",
        source,
        re.S,
    )
    assert match, "UXP_DIRECT_HOST_ACTIONS map must exist in PProBridge"
    return set(re.findall(r"\b(oc[A-Za-z0-9_]+)\s*:", match.group("body")))


def test_direct_action_map_matches_parity_manifest():
    source = _read_main_js()
    manifest = json.loads(PARITY_MANIFEST.read_text(encoding="utf-8"))
    direct_manifest_names = {
        entry["name"]
        for entry in manifest["functions"]
        if entry["status"] == "direct_uxp"
    }

    assert _direct_action_names(source) == direct_manifest_names | UXP_ONLY_DIRECT_ACTIONS
    assert len(direct_manifest_names) == 14
    assert len(_direct_action_names(source)) == 15
    assert "ocGetCaptionTrackSnapshot" in _direct_action_names(source)
    assert "ocAddNativeCaptionTrack" not in direct_manifest_names
    assert "ocQeReflect" not in direct_manifest_names


def test_execute_host_action_handles_catalogued_actions():
    source = _read_main_js()
    direct_names = _direct_action_names(source)

    for name in sorted(direct_names):
        assert f'case "{name}"' in source, f"{name} must be handled by executeHostAction"

    assert 'case "ocApplySequenceCuts"' in source
    assert 'case "ocEmitPingEvent"' in source
    assert 'case "ocAddNativeCaptionTrack"' in source
    assert 'case "ocQeReflect"' in source
    assert "cepFallback: true" in source


def test_host_dispatch_exports_for_webview_bridge():
    source = _read_main_js()

    assert "executeHostAction" in source
    assert "hostActionStatus" in source
    assert "window.OpenCutUXPHost" in source
    assert "getHostActionStatus" in source
    assert "PProBridge.executeHostAction(action, payload)" in source


def test_direct_dispatch_stays_off_ceps_evalscript_path():
    source = _read_main_js()
    start = source.index("const UXP_DIRECT_HOST_ACTIONS")
    end = source.index("const BackendClient")
    bridge_block = source[start:end]

    assert "evalScript" not in bridge_block
    assert "CSInterface" not in bridge_block
    assert "import(\"premierepro\")" in bridge_block
    assert "Sequence.createSubsequence is unavailable" in bridge_block
    assert "exportSubsequenceWithEncoder" in bridge_block


def test_caption_track_snapshot_action_is_read_only_and_diff_compatible():
    source = _read_main_js()
    start = source.index("async function _activeSequenceContext")
    end = source.index("async function setSequencePlayhead")
    block = source[start:end]

    assert "reason_code: \"no_open_project\"" in block
    assert "reason_code: \"no_active_sequence\"" in block
    assert "reason_code: \"no_caption_tracks\"" in block
    assert "reason_code: \"caption_api_missing\"" in block
    assert "CaptionTrack.getTrackItems" in block
    assert "caption_track_snapshot" in block
    assert "caption_id" in block
    assert "source_segment_id" in block
    assert "host_locators" in block
    assert "createCaptionTrack" not in block
    assert "addCaptionTrack" not in block
