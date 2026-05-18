"""F258 UXP ProjectConverter.exportAAF integration guardrails."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
VERSIONS = REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_f258_beta_snapshot_tracks_aaf_export_api_version():
    data = json.loads(VERSIONS.read_text(encoding="utf-8"))

    assert data["package"] == "@adobe/premierepro"
    assert data["dist_tags"]["beta"] == "26.3.0-beta.67"
    assert "26.3.0-beta.67" in data["tracked_versions"]


def test_f258_bridge_exports_active_sequence_as_aaf():
    source = _read(MAIN_JS)

    assert "async function exportAafSequence(payload)" in source
    assert "ppro?.ProjectConverter?.exportAAF" in source
    assert "Premiere ProjectConverter.exportAAF is unavailable in this UXP runtime." in source
    assert "An output path is required for UXP AAF export." in source
    assert "ppro.ProjectConverter.exportAAF(seq, outputPath, aafOptions || undefined)" in source


def test_f258_bridge_builds_aaf_export_options():
    source = _read(MAIN_JS)

    assert "function _createAafExportOptions(settings = {})" in source
    assert "new factory()" in source
    assert '["mixdownVideo", "setMixdownVideo"]' in source
    assert '["explodeToMono", "setExplodeToMono"]' in source
    assert '["sampleRate", "setSampleRate"]' in source
    assert '["bitsPerSample", "setBitsPerSample"]' in source
    assert '["handleFrames", "setHandleFrames"]' in source
    assert "options.setAudioFileFormat(_aafAudioFormat(settings.audioFileFormat))" in source
    assert "options.setVideoMixdownPresetPath(presetPath)" in source


def test_f258_bridge_maps_aaf_audio_format_constants():
    source = _read(MAIN_JS)

    assert "function _aafAudioFormat(value)" in source
    assert "ppro?.Constants?.AAFExportAudioFormat?.AIFF" in source
    assert "ppro?.ProjectConverter?.AAF_EXPORT_AUDIO_FORMAT_AIFF" in source
    assert "ppro?.Constants?.AAFExportAudioFormat?.WAV" in source
    assert "ppro?.ProjectConverter?.AAF_EXPORT_AUDIO_FORMAT_WAV" in source


def test_f258_webview_host_exposes_aaf_export_helper():
    source = _read(MAIN_JS)

    assert "window.OpenCutUXPHost" in source
    assert "exportAafSequence: (payload) => PProBridge.exportAafSequence(payload)" in source


def test_f258_release_smoke_runs_aaf_guardrail():
    source = _read(RELEASE_SMOKE)

    assert "tests/test_uxp_aaf_export_integration.py" in source
