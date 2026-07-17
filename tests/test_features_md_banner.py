"""F185 — features.md must carry the aspirational-catalogue banner.

The 402-feature wish-list in `features.md` predates the F-numbered
governance system. Without an explicit "not a ship promise" banner,
contributors and AI tools repeatedly mis-read it as a backlog. F185
fixes that by adding a top-of-file warning and pointing readers at
ROADMAP.md as the authoritative source.

These tests pin the banner so a future refresh of the file cannot
accidentally strip it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_MD = REPO_ROOT / "features.md"

# features.md is a local working artefact excluded from git (the repo
# .gitignore only tracks README/CHANGELOG/ROADMAP/RESEARCH* markdown), so a
# clean checkout legitimately lacks it. Pin the banner only when it exists.
pytestmark = pytest.mark.skipif(
    not FEATURES_MD.is_file(),
    reason="features.md is a gitignored local artefact absent on clean checkouts",
)


def test_features_md_starts_with_banner():
    text = FEATURES_MD.read_text(encoding="utf-8")
    assert text.startswith("# OpenCut — Feature Expansion Plan\n")
    # The first 800 chars must contain the aspirational-marker phrasing.
    header = text[:1200]
    assert "Aspirational catalogue" in header
    assert "not a ship promise" in header


def test_features_md_links_to_roadmap_and_reconciliation():
    text = FEATURES_MD.read_text(encoding="utf-8")
    header = text[:1500]
    assert "ROADMAP.md" in header, (
        "F185 banner must point readers at ROADMAP.md as the authoritative plan"
    )
    assert "FEATURES_RECONCILIATION.md" in header, (
        "F185 banner must reference the F179 reconciliation pass artefact"
    )


def test_features_md_carries_status_line():
    text = FEATURES_MD.read_text(encoding="utf-8")
    header = text[:2000]
    assert "Status (2026-05-17)" in header
    assert "F185" in header


def test_features_md_precedence_rule_is_explicit():
    text = FEATURES_MD.read_text(encoding="utf-8")
    header = text[:1500]
    # The "ROADMAP.md wins" precedence rule must be in the banner so AI
    # tools / contributors know which file to trust.
    assert "ROADMAP.md wins" in header
    assert "the code wins" in header
