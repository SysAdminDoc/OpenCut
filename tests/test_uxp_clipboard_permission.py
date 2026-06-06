"""RA-19 - UXP clipboard permission and fallback guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
LIVE_MANIFEST = UXP_ROOT / "manifest.json"
WEBVIEW_CONFIG = UXP_ROOT / "bolt-webview" / "uxp.config.ts"
UXP_MAIN = UXP_ROOT / "main.js"


def _manifest() -> dict:
    return json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))


def _main_source() -> str:
    return UXP_MAIN.read_text(encoding="utf-8")


def test_live_manifest_declares_clipboard_permission_for_copy_path():
    manifest = _manifest()
    source = _main_source()

    assert "navigator.clipboard.writeText" in source
    assert manifest["manifestVersion"] == 5
    assert manifest["requiredPermissions"]["clipboard"] == "readAndWrite"


def test_webview_scaffold_carries_clipboard_permission_contract():
    config = WEBVIEW_CONFIG.read_text(encoding="utf-8")

    assert len(re.findall(r"clipboard:\s*\"readAndWrite\"", config)) >= 2


def test_copy_button_uses_shared_async_helper():
    source = _main_source()
    copy_handler = source.split(
        'document.getElementById("copySrtBtn")?.addEventListener("click"', 1
    )[1].split('document.getElementById("importSrtBtn")', 1)[0]

    assert "async function copyTextToClipboard" in source
    assert "await navigator.clipboard.writeText(value)" in source
    assert "Clipboard permission is unavailable or denied. Copy the output manually." in source
    assert "copyTextToClipboard(body.value" in copy_handler
    assert "navigator.clipboard.writeText" not in copy_handler
