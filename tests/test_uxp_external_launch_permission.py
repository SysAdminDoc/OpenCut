"""RA-13 - UXP external launch permission guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
LIVE_MANIFEST = UXP_ROOT / "manifest.json"
UXP_MAIN = UXP_ROOT / "main.js"
WEBVIEW_CONFIG = UXP_ROOT / "bolt-webview" / "uxp.config.ts"
WEBVIEW_UXP_API = UXP_ROOT / "bolt-webview" / "src" / "api" / "uxp.ts"
SOURCE_SUFFIXES = {".html", ".js", ".mjs", ".ts", ".tsx", ".jsx"}


def _manifest() -> dict:
    return json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))


def _uxp_source_files() -> list[Path]:
    return sorted(
        path
        for path in UXP_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and "node_modules" not in path.parts
        and "dist" not in path.parts
    )


def test_live_manifest_declares_https_only_launch_process_permission():
    manifest = _manifest()
    launch_process = manifest["requiredPermissions"]["launchProcess"]

    assert launch_process["schemes"] == ["https"]
    assert launch_process["extensions"] == []
    assert "http" not in launch_process["schemes"]
    assert "mailto" not in launch_process["schemes"]
    assert "file" not in launch_process["schemes"]


def test_live_uxp_oauth_launch_uses_https_normalizer_and_consent_context():
    source = UXP_MAIN.read_text(encoding="utf-8")

    assert "function normalizeHttpsExternalUrl" in source
    assert 'parsed.protocol === "https:"' in source
    assert "async function openHttpsExternalUrl" in source
    assert "shell.openExternal(" in source
    assert "developerText ||" in source
    assert "Invalid HTTPS authorization URL received from server." in source
    assert "Opening ${platform} authorization page in your browser" in source
    assert "openHttpsExternalUrl(" in source


def test_webview_scaffold_matches_https_only_launch_contract():
    config = WEBVIEW_CONFIG.read_text(encoding="utf-8")
    api = WEBVIEW_UXP_API.read_text(encoding="utf-8")

    assert len(re.findall(r"launchProcess:\s*\{", config)) >= 2
    assert re.search(r"schemes:\s*\[\s*\"https\"\s*\]", config)
    assert re.search(r"extensions:\s*\[\s*\]", config)
    assert 'parsed.protocol !== "https:"' in api
    assert "Only https URLs can be opened from UXP." in api
    assert "Opening a secure web page in your browser" in api


def test_external_launch_calls_stay_in_approved_uxp_boundaries():
    allowed_open_external = {
        "extension/com.opencut.uxp/main.js",
        "extension/com.opencut.uxp/bolt-webview/src/api/uxp.ts",
    }
    open_external_paths: set[str] = set()
    open_path_violations: list[str] = []

    for path in _uxp_source_files():
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        if re.search(r"\bopenExternal\b", source):
            open_external_paths.add(rel)
        if re.search(r"\bopenPath\b", source):
            open_path_violations.append(rel)

    assert allowed_open_external.issubset(open_external_paths)
    assert open_external_paths <= allowed_open_external
    assert not open_path_violations, (
        "UXP openPath usage needs an explicit launchProcess.extensions review:\n"
        + "\n".join(open_path_violations)
    )
