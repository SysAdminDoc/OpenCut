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
        # Co-named but genuinely separate implementations — CEP and UXP have
        # different hosts, cascade roots, and transport constraints, so these
        # were never copies and must not be gated as such. The shared name is a
        # readability hazard rather than drift.
        #
        # The three command-center stylesheets carried the same name at the same
        # cascade position while being different files, which is how an edit
        # lands in the wrong panel. The UXP copies are now uxp-prefixed, matching
        # every other file in that directory, so the collision is gone. The same
        # rename for backend-client.js waits on the route manifest, whose
        # surface evidence cites the path.
        "backend-client.js",
    }
    unguarded = sorted(duplicated - set(SHARED_ASSETS) - panel_specific)

    assert not unguarded, (
        "these files exist under the same name in both panels but no drift gate "
        f"covers them: {unguarded}. Add each to SHARED_ASSETS if it is meant to "
        "be one asset, or to panel_specific if the panels genuinely differ."
    )
