"""Guardrails for F259 — UXP HTTP on macOS workaround documentation.

F259 closes the documentation portion of the known Premiere Pro 25.6.x
UXP-on-macOS HTTP friction (first-fetch stall, WS retry-on-first-failure,
etc.). These tests pin the four invariants the documentation pass claims
and the loopback-port allowlist the UXP manifest must keep declaring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "UXP_MACOS_HTTP.md"
API_NOTES = REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-api-notes.md"
MANIFEST = REPO_ROOT / "extension" / "com.opencut.uxp" / "manifest.json"
MAIN_JS = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"


def test_doc_exists():
    assert DOC.is_file(), f"F259 doc missing at {DOC}"


def test_doc_covers_required_sections():
    text = DOC.read_text(encoding="utf-8")
    required_headings = [
        "Why this document exists",
        "Permissions the UXP panel must declare",
        "Workarounds already shipped",
        "What the user must do today",
        "Planned auto-HTTPS sidecar",
        "Reproducing the macOS HTTP symptoms",
        "Acceptance criteria for closing F259",
    ]
    for heading in required_headings:
        assert heading in text, f"F259 doc missing section: {heading!r}"


def test_doc_names_each_shipped_workaround():
    text = DOC.read_text(encoding="utf-8")
    for token in (
        "detectBackend",
        "fetchWithTimeout",
        "WebSocket",
        "Exponential backoff",
    ):
        assert token in text, f"F259 doc must reference {token!r}"


def test_doc_marks_https_sidecar_as_deferred():
    """The sidecar plan is captured but not promised for this F-number."""
    text = DOC.read_text(encoding="utf-8")
    assert "deferred" in text.lower() or "F146" in text, (
        "F259 doc must clearly mark the HTTPS sidecar as deferred to F146/F252"
    )


def test_doc_warns_about_macos_incoming_connections_prompt():
    text = DOC.read_text(encoding="utf-8")
    assert "Allow incoming connections" in text
    assert "deny" in text.lower(), (
        "F259 doc must tell macOS users to deny the firewall prompt"
    )


def test_api_notes_cross_links_to_doc():
    text = API_NOTES.read_text(encoding="utf-8")
    assert "UXP_MACOS_HTTP" in text, (
        "extension/com.opencut.uxp/uxp-api-notes.md must cross-link to "
        "docs/UXP_MACOS_HTTP.md"
    )


def _strip_json_comments(blob: str) -> str:
    """Tolerate `// ...` JSONC comments in the UXP manifest."""
    return re.sub(r"^\s*//.*$", "", blob, flags=re.MULTILINE)


@pytest.fixture(scope="module")
def manifest_data():
    raw = MANIFEST.read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(raw))


def test_manifest_declares_canonical_loopback_ports(manifest_data):
    domains = manifest_data["requiredPermissions"]["network"]["domains"]
    domain_set = {d.strip() for d in domains}
    canonical = {
        "http://127.0.0.1:5679",  # HTTP backend
        "ws://127.0.0.1:5680",  # WebSocket bridge
        "http://127.0.0.1:5681",  # MCP
    }
    missing = canonical - domain_set
    assert not missing, f"UXP manifest is missing canonical loopback ports: {missing}"


def test_manifest_declares_full_5679_5689_range(manifest_data):
    """OpenCut autodiscovers across 5679-5689; the manifest must allow each."""
    domains = manifest_data["requiredPermissions"]["network"]["domains"]
    domain_set = {d.strip() for d in domains}
    for port in range(5679, 5690):
        http_url = f"http://127.0.0.1:{port}"
        ws_url = f"ws://127.0.0.1:{port}"
        assert http_url in domain_set, (
            f"UXP manifest must declare {http_url} so port autodiscovery succeeds"
        )
        assert ws_url in domain_set, (
            f"UXP manifest must declare {ws_url} so live-update bridge fallback succeeds"
        )


def test_main_js_uses_fetch_with_timeout_for_health():
    """The first-fetch stall mitigation depends on fetchWithTimeout. Keep it."""
    text = MAIN_JS.read_text(encoding="utf-8")
    assert "fetchWithTimeout" in text, "main.js must keep fetchWithTimeout"
    assert "detectBackend" in text, "main.js must keep detectBackend"
    # detectBackend must use the timeout helper for its probe sweep.
    assert re.search(r"detectBackend\b[\s\S]{0,400}fetchWithTimeout", text), (
        "detectBackend must call fetchWithTimeout so macOS first-fetch stalls "
        "cannot pin the panel"
    )
