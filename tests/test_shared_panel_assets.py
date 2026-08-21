"""F324 — the design-system assets both panels ship must not drift apart.

`studio-workbench-v2.css` and `.js` exist as byte-identical copies under the CEP
and UXP panels, with no generator and nothing asserting they match. Two files
under one name that are meant to be the same thing will diverge the first time
someone patches the panel they happen to be looking at, and the divergence is
invisible until a user reports that one panel looks wrong.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CEP_DIR = REPO_ROOT / "extension" / "com.opencut.panel" / "client"
UXP_DIR = REPO_ROOT / "extension" / "com.opencut.uxp"

#: Assets deliberately shipped to both panels from one source of truth.
SHARED_ASSETS = ("studio-workbench-v2.css", "studio-workbench-v2.js")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", SHARED_ASSETS)
def test_shared_asset_exists_in_both_panels(name):
    for directory in (CEP_DIR, UXP_DIR):
        assert (directory / name).is_file(), f"{name} is missing from {directory}"


@pytest.mark.parametrize("name", SHARED_ASSETS)
def test_shared_asset_is_identical_in_both_panels(name):
    cep = CEP_DIR / name
    uxp = UXP_DIR / name
    if not (cep.is_file() and uxp.is_file()):
        pytest.skip(f"{name} is not present in both panels")

    assert _digest(cep) == _digest(uxp), (
        f"{name} has drifted between the panels. It is shipped to both from one "
        "source of truth, so edit it once and copy it across — or promote it to "
        "a generated asset. Diff:\n"
        f"  CEP: {cep}\n  UXP: {uxp}"
    )


def test_the_gate_covers_every_duplicated_panel_asset():
    """A new same-named pair must be added to SHARED_ASSETS, not left unguarded."""
    cep_names = {p.name for p in CEP_DIR.glob("*") if p.is_file()}
    uxp_names = {p.name for p in UXP_DIR.glob("*") if p.is_file()}
    duplicated = cep_names & uxp_names

    # Files that legitimately differ per panel: each panel has its own entry
    # point, manifest, styling, and locale set.
    panel_specific = {
        "index.html",
        "main.js",
        "style.css",
        "manifest.json",
        "package.json",
        "README.md",
        "CHANGELOG.md",
        ".gitignore",
        # Every remaining name here differs per panel for a structural reason:
        # each panel has its own entry point, manifest, styling, and locale set.
        #
        # Nothing else is co-named any more. The three command-center
        # stylesheets and backend-client.js used to share a name across both
        # panels while being different implementations, which is how an edit
        # lands in the wrong panel — a hazard the drift gate cannot catch,
        # because those files were never copies. The UXP copies are now
        # uxp-prefixed, matching every other file in that directory.
    }
    unguarded = sorted(duplicated - set(SHARED_ASSETS) - panel_specific)

    assert not unguarded, (
        "these files exist under the same name in both panels but no drift gate "
        f"covers them: {unguarded}. Add each to SHARED_ASSETS if it is meant to "
        "be one asset, or to panel_specific if the panels genuinely differ."
    )


def test_studio_workbench_light_theme_flips_clips_with_the_timeline():
    css = (CEP_DIR / "studio-workbench-v2.css").read_text(encoding="utf-8")
    assert "html.theme-light .studio-clip" in css
    assert "html.theme-light .studio-sequence-clip" in css
    assert "html.theme-light .studio-result-thumb" in css
    assert ".studio-wave--slate" not in css
    assert "outline: 2px solid var(--studio-accent)" in css
    assert "box-shadow: 0 0 0 3px var(--studio-accent-soft)" not in css


def test_cep_light_theme_section_no_longer_owns_the_dark_chrome_block():
    css = (CEP_DIR / "style.css").read_text(encoding="utf-8")
    js = (CEP_DIR / "main.js").read_text(encoding="utf-8")
    cc = (CEP_DIR / "command-center.css").read_text(encoding="utf-8")
    uxp_cc = (UXP_DIR / "uxp-command-center.css").read_text(encoding="utf-8")
    assert "Shared layout + dark-default chrome" in css
    assert "html:not(.theme-light) .quick-action-icon" in css
    assert "html.theme-light .quick-action-icon" in css
    assert "html.theme-light .progress-fill" in css
    assert "#466fd3" in css
    assert "html.theme-light .oc-feature-gated::after" in css
    assert "html.theme-light .footage-result-item.is-selected" in css
    assert "--waveform-bg:" in css
    assert 'getPropertyValue("--waveform-bg")' in js
    assert "e.target === document.body" not in js
    assert 'e.target.closest("input, textarea, select, [contenteditable=\'true\']")' in js
    assert "box-shadow: var(--cc-shadow-float)" in cc
    assert "box-shadow: 0 14px 30px rgba(0, 0, 0, 0.34)" not in cc
    assert "html.theme-light body .oc-workspace-guide[data-state=\"ready\"] #workspaceGuideKicker" in uxp_cc
    assert "html.theme-light .oc-toast .oc-toast-msg" in uxp_cc
