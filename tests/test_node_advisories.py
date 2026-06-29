"""Sanity checks for the CEP panel Node advisory policy.

These tests are intentionally text-based: they assert that the documented
allow-list, the npm metadata, and the local release smoke stay in agreement without
requiring a Node toolchain to be installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = REPO_ROOT / "extension" / "com.opencut.panel"
PACKAGE_JSON = PANEL_DIR / "package.json"
ADVISORIES_DOC = REPO_ROOT / "docs" / "NODE_ADVISORIES.md"
CHECK_SCRIPT = PANEL_DIR / "scripts" / "check-advisories.mjs"
VERIFY_SCRIPT = PANEL_DIR / "scripts" / "verify-build.mjs"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"

GHSA_PATTERN = re.compile(r"GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}", re.IGNORECASE)


def _load_package_json() -> dict:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))


def test_package_json_pins_esbuild_override():
    pkg = _load_package_json()

    overrides = pkg.get("overrides", {})
    assert "esbuild" in overrides, "esbuild override missing from package.json"

    spec = overrides["esbuild"]
    assert spec.startswith("^0.25") or spec.startswith("0.25") or spec.startswith(">=0.25"), (
        f"esbuild override must be >=0.25 to close GHSA-67mh-4wv8-2f99 (got {spec!r})"
    )


def test_package_json_pins_js_yaml_override():
    pkg = _load_package_json()

    overrides = pkg.get("overrides", {})
    assert "js-yaml" in overrides, "js-yaml override missing from package.json"

    spec = overrides["js-yaml"]
    assert spec.startswith("^4.2") or spec.startswith("4.2") or spec.startswith(">=4.2"), (
        f"js-yaml override must be >=4.2.0 to close GHSA-h67p-54hq-rp68 (got {spec!r})"
    )


def test_package_json_advertises_audit_and_verify_scripts():
    pkg = _load_package_json()
    scripts = pkg.get("scripts", {})

    assert "audit:check" in scripts, "missing audit:check script entry"
    assert "build:verify" in scripts, "missing build:verify script entry"
    assert scripts["audit:check"].endswith("check-advisories.mjs")
    assert scripts["build:verify"].endswith("verify-build.mjs")


def test_check_advisories_script_present_and_executable_text():
    text = CHECK_SCRIPT.read_text(encoding="utf-8")
    assert "ALLOWED" in text
    assert "new Map()" in text
    assert "npm audit" in text
    assert "--json" in text
    assert "JSON.stringify" in text
    assert "powershell.exe" in text
    assert "Set-Location -LiteralPath" in text


def test_verify_build_script_validates_required_sources():
    text = VERIFY_SCRIPT.read_text(encoding="utf-8")
    for required in ("client/index.html", "client/main.js", "client/style.css", "CSXS/manifest.xml"):
        assert required in text, f"verify-build.mjs must reference {required}"


def test_advisories_doc_documents_every_allowed_entry():
    doc = ADVISORIES_DOC.read_text(encoding="utf-8")
    script = CHECK_SCRIPT.read_text(encoding="utf-8")

    script_ids = set(GHSA_PATTERN.findall(script))
    doc_ids = set(GHSA_PATTERN.findall(doc))

    missing = script_ids - doc_ids
    assert not missing, (
        "Every GHSA waived in check-advisories.mjs must be listed in docs/NODE_ADVISORIES.md "
        f"(missing: {sorted(missing)})"
    )
    if not script_ids:
        assert "No active npm advisory waivers" in doc


def test_release_smoke_runs_panel_advisory_gate_locally():
    smoke = RELEASE_SMOKE.read_text(encoding="utf-8")
    assert "npm-advisory" in smoke, "release smoke must include the npm advisory gate"
    assert "check-advisories.mjs" in smoke, "release smoke must invoke check-advisories.mjs"
    assert "panel-source" in smoke, "release smoke must include the panel source verifier"


def test_node_modules_excluded_from_repo():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "node_modules/" in gitignore, "node_modules/ must be gitignored"
