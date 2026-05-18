"""F131 — esbuild >=0.25 pin verification tests.

The check-esbuild-pin.mjs script is the source of truth at runtime.
These Python tests pin three invariants without needing a Node toolchain:

1. The panel's `package.json` declares the `overrides.esbuild` pin in
   the >=0.25 range (parsed from the npm semver string).
2. The script file is present, parses as valid JS, and is wired into
   the panel's `npm run audit:esbuild` script.
3. The release-smoke step is registered and gracefully skips when
   Node or `node_modules` is missing (the dev-VM and CI matrix
   default state for many legs).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = REPO_ROOT / "extension" / "com.opencut.panel"
PACKAGE_JSON = PANEL_ROOT / "package.json"
SCRIPT = PANEL_ROOT / "scripts" / "check-esbuild-pin.mjs"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _semver_minor(spec: str) -> tuple:
    """Extract (major, minor) from an npm semver range like '^0.25.0'."""
    m = re.search(r"(\d+)\.(\d+)", spec)
    assert m, f"could not parse semver from {spec!r}"
    return int(m.group(1)), int(m.group(2))


def test_package_json_pins_esbuild_at_or_above_minimum():
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    overrides = data.get("overrides") or {}
    spec = overrides.get("esbuild")
    assert spec, "package.json overrides must keep an `esbuild` pin"
    major, minor = _semver_minor(spec)
    assert (major, minor) >= (0, 25), (
        f"esbuild override {spec!r} is below the F131 minimum 0.25"
    )


def test_check_esbuild_pin_script_exists():
    assert SCRIPT.is_file(), f"F131 script missing at {SCRIPT}"


def test_package_json_exposes_audit_esbuild_script():
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    scripts = data.get("scripts") or {}
    assert "audit:esbuild" in scripts, (
        "package.json must expose `npm run audit:esbuild` as the F131 entry point"
    )
    assert "check-esbuild-pin.mjs" in scripts["audit:esbuild"]


def test_release_smoke_registers_esbuild_pin_step():
    text = RELEASE_SMOKE.read_text(encoding="utf-8")
    assert "step_esbuild_pin" in text, "release_smoke.py must define step_esbuild_pin"
    assert '"esbuild-pin"' in text, "release_smoke.py must register the esbuild-pin step"
    # The step must gracefully skip when Node is absent so the gate
    # stays useful on the Linux Python-only legs.
    assert "node executable not on PATH" in text, (
        "release_smoke.py esbuild-pin step must skip when node is missing"
    )


def test_script_invokes_npm_ls_with_correct_filter():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"ls"' in text and '"esbuild"' in text
    assert "--all" in text
    assert "--json" in text


def test_script_carries_documented_minimum():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "MIN_MAJOR" in text
    assert "MIN_MINOR" in text
    assert "MIN_PATCH" in text
    # The minimums must match the F095 / F131 pin.
    assert re.search(r"MIN_MAJOR\s*=\s*0", text)
    assert re.search(r"MIN_MINOR\s*=\s*25", text)


def test_script_rejects_unparseable_versions():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Number.isNaN" in text, (
        "check-esbuild-pin must reject NaN-parsed versions instead of "
        "silently treating them as satisfied"
    )
