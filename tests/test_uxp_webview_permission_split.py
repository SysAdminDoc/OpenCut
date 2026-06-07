"""RA-14 - UXP WebView development/release permission split guardrails."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
WEBVIEW_CONFIG = UXP_ROOT / "bolt-webview" / "uxp.config.ts"
WEBVIEW_README = UXP_ROOT / "bolt-webview" / "README.md"
UXP_MIGRATION = REPO_ROOT / "docs" / "UXP_MIGRATION.md"


def _config() -> str:
    return WEBVIEW_CONFIG.read_text(encoding="utf-8")


def test_webview_config_exports_dev_and_release_manifest_profiles():
    config = _config()

    assert 'type OpenCutManifestProfile = "development" | "release"' in config
    assert "export function buildManifest" in config
    assert 'export const developmentManifest = buildManifest("development")' in config
    assert 'export const releaseManifest = buildManifest("release")' in config
    assert "export const manifest = developmentManifest" in config
    assert "export default { manifest, developmentManifest, releaseManifest, ...config }" in config


def test_release_webview_profile_omits_dev_domains_and_hot_reload_permissions():
    config = _config()

    assert "const RELEASE_WEBVIEW_DOMAINS: DomainList = []" in config
    assert "const RELEASE_NETWORK_DOMAINS" in config
    assert "const DEV_NETWORK_DOMAINS" in config
    assert re.search(
        r"domains:\s*\[\.\.\.\(releaseProfile \? RELEASE_NETWORK_DOMAINS : DEV_NETWORK_DOMAINS\)\]",
        config,
    )
    assert re.search(
        r"domains:\s*\[\.\.\.\(releaseProfile \? RELEASE_WEBVIEW_DOMAINS : DEV_WEBVIEW_DOMAINS\)\]",
        config,
    )
    assert 'enableMessageBridge: releaseProfile ? "localOnly" : "localAndRemote"' in config


def test_development_webview_profile_keeps_hot_reload_domains_explicit():
    config = _config()

    assert "const DEV_WEBVIEW_DOMAINS" in config
    assert "http://127.0.0.1:5173" in config
    assert "http://localhost:5173" in config
    assert "ws://127.0.0.1:${HOT_RELOAD_PORT}" in config
    assert "ws://localhost:${HOT_RELOAD_PORT}" in config
    assert 'domains: "all"' not in config


def test_docs_record_webview_permission_profiles():
    readme = WEBVIEW_README.read_text(encoding="utf-8")
    migration = UXP_MIGRATION.read_text(encoding="utf-8")

    assert "developmentManifest" in readme
    assert "releaseManifest" in readme
    assert "localOnly" in readme
    assert "RA-14 WebView permission split" in migration
