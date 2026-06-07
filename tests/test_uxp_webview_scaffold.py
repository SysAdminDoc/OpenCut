"""F252.1 — Bolt UXP WebView scaffold guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
SCAFFOLD = UXP_ROOT / "bolt-webview"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_f252_bolt_webview_scaffold_files_exist():
    required = [
        SCAFFOLD / "README.md",
        SCAFFOLD / "uxp.config.ts",
        SCAFFOLD / "src" / "api" / "api.ts",
        SCAFFOLD / "src" / "api" / "uxp.ts",
        SCAFFOLD / "src" / "api" / "premierepro.ts",
        SCAFFOLD / "webview-ui" / "index.html",
        SCAFFOLD / "webview-ui" / "src" / "main.ts",
        SCAFFOLD / "webview-ui" / "src" / "webview-api.ts",
        SCAFFOLD / "webview-ui" / "src" / "webview-setup.ts",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.is_file()]
    assert not missing, f"F252.1 scaffold files missing: {missing}"


def test_f252_config_is_webview_enabled_and_not_live_entrypoint():
    live_manifest = json.loads(_read(UXP_ROOT / "manifest.json"))
    config = _read(SCAFFOLD / "uxp.config.ts")

    assert live_manifest["main"] == "index.html"
    assert "webviewUi: true" in config
    assert 'allow: "yes"' in config
    assert 'allowLocalRendering: "yes"' in config
    assert '"localAndRemote"' in config
    assert 'domains: "all"' not in config
    assert 'import liveManifest from "../manifest.json"' in config
    assert 'app: "PPRO"' in config
    assert 'minVersion: "25.6"' in config


def test_f252_config_preserves_loopback_allowlist():
    config = _read(SCAFFOLD / "uxp.config.ts")
    assert "Array.from({ length: 11 }" in config
    assert "const port = 5679 + index" in config
    assert "`http://127.0.0.1:${port}`" in config
    assert "http://localhost:5679" in config
    assert "ws://127.0.0.1:${HOT_RELOAD_PORT}" in config
    assert "http://127.0.0.1:5173" in config
    assert "http://localhost:5173" in config


def test_f252_host_api_exports_low_risk_premiere_ports():
    source = _read(SCAFFOLD / "src" / "api" / "premierepro.ts")
    for name in (
        "detectBackend",
        "getProjectInfo",
        "getSequenceInfo",
        "addTimelineMarkers",
        "applyTimelineCuts",
        "importFiles",
    ):
        assert re.search(rf"export\s+async\s+function\s+{name}\b", source), (
            f"Premiere host API must export {name}"
        )
    assert 'import("premierepro")' in source
    assert "254016000000" in source


def test_f252_webview_bridge_uses_uxp_host_postmessage():
    setup = _read(SCAFFOLD / "webview-ui" / "src" / "webview-setup.ts")
    api = _read(SCAFFOLD / "webview-ui" / "src" / "webview-api.ts")
    assert "uxpHost" in setup
    assert "postMessage" in setup
    assert "opencut:webview-call" in setup
    assert "opencut:host-response" in setup
    assert "opencut:host-call" in setup
    assert "pingWebview" in api
    assert "updateColorScheme" in api


def test_f252_docs_record_dormant_scaffold_and_cutover():
    readme = _read(SCAFFOLD / "README.md")
    migration = _read(REPO_ROOT / "docs" / "UXP_MIGRATION.md")
    assert "intentionally dormant" in readme
    assert "extension/com.opencut.uxp/bolt-webview" in migration
    assert "Live manifest switch" in migration
