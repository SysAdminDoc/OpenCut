"""RA-17 - UXP manifest schema/version guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
LIVE_MANIFEST = UXP_ROOT / "manifest.json"
WEBVIEW_CONFIG = UXP_ROOT / "bolt-webview" / "uxp.config.ts"
WEBVIEW_README = UXP_ROOT / "bolt-webview" / "README.md"


def _manifest() -> dict:
    return json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))


def test_live_uxp_manifest_declares_supported_manifest_version():
    manifest = _manifest()

    assert manifest["manifestVersion"] == 5
    assert isinstance(manifest["manifestVersion"], int)


def test_live_uxp_manifest_keeps_required_schema_keys():
    manifest = _manifest()
    required = {
        "manifestVersion",
        "id",
        "name",
        "version",
        "main",
        "host",
        "entrypoints",
        "requiredPermissions",
    }

    assert required.issubset(manifest)
    assert manifest["main"] == "index.html"
    assert manifest["host"][0]["app"] == "PPRO"
    assert manifest["host"][0]["minVersion"] == "25.6"
    assert manifest["entrypoints"][0]["type"] == "panel"


def test_webview_scaffold_manifest_version_is_not_live_policy():
    config = WEBVIEW_CONFIG.read_text(encoding="utf-8")
    readme = WEBVIEW_README.read_text(encoding="utf-8")

    assert re.search(r"manifestVersion:\s*6\b", config)
    assert "intentionally dormant" in readme
    assert "../manifest.json" in readme
