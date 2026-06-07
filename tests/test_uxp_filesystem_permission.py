"""RA-11 - UXP least-privilege filesystem permission guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
LIVE_MANIFEST = UXP_ROOT / "manifest.json"
UXP_MAIN = UXP_ROOT / "main.js"
UXP_API_NOTES = UXP_ROOT / "uxp-api-notes.md"
WEBVIEW_CONFIG = UXP_ROOT / "bolt-webview" / "uxp.config.ts"
WEBVIEW_README = UXP_ROOT / "bolt-webview" / "README.md"
UXP_MIGRATION = REPO_ROOT / "docs" / "UXP_MIGRATION.md"


def _manifest() -> dict:
    return json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))


def test_live_manifest_uses_request_scoped_filesystem_permission():
    manifest = _manifest()

    assert manifest["requiredPermissions"]["localFileSystem"] == "request"


def test_live_uxp_file_access_stays_picker_scoped():
    source = UXP_MAIN.read_text(encoding="utf-8")
    unsupported_direct_access = [
        "getFileForSaving",
        "getEntryWithUrl",
        "createEntryWithUrl",
        "getDataFolder",
        "getPluginFolder",
        "getTemporaryFolder",
    ]

    assert "localFileSystem.getFileForOpening" in source
    assert "localFileSystem.getFolder" in source
    for token in unsupported_direct_access:
        assert token not in source


def test_webview_scaffold_matches_request_scoped_filesystem_permission():
    config = WEBVIEW_CONFIG.read_text(encoding="utf-8")

    assert len(re.findall(r"localFileSystem:\s*\"request\"", config)) >= 2
    assert 'localFileSystem: "fullAccess"' not in config


def test_filesystem_permission_docs_explain_request_scope():
    api_notes = UXP_API_NOTES.read_text(encoding="utf-8")
    readme = WEBVIEW_README.read_text(encoding="utf-8")
    migration = UXP_MIGRATION.read_text(encoding="utf-8")

    assert "| `localFileSystem` | `request` |" in api_notes
    assert "picker-scoped" in api_notes
    assert "localFileSystem` permission to `request`" in readme
    assert "RA-11 filesystem permission guard" in migration
