"""Sanity check the Windows ARM64 packaging evaluation doc (F101).

The deliverable for F101 is the evaluation document itself — these
tests stop the doc from regressing into a stale stub.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "WINDOWS_ARM64_PACKAGING.md"
SEEDS_PATH = REPO_ROOT / ".github" / "issue-seeds.yml"


def test_doc_exists_and_is_non_empty():
    assert DOC_PATH.exists(), "Windows ARM64 packaging evaluation doc missing"
    assert DOC_PATH.stat().st_size > 1024, "doc must include the full compatibility matrix"


def test_doc_covers_required_sections():
    text = DOC_PATH.read_text(encoding="utf-8")
    for needle in (
        "Component-by-component compatibility",
        "What would have to change",
        "Cost estimate",
        "Acceptance criteria",
        "FFmpeg",
        "PyInstaller spec",
        "Inno",
    ):
        assert needle in text, f"section/keyword {needle!r} missing from doc"


def test_acceptance_criteria_lists_release_smoke_and_signed_installer():
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "release_smoke" in text
    assert "OpenCut-Setup-arm64.exe" in text
    assert "Surface Pro X" in text or "SQ3" in text


def test_seed_manifest_references_F101():
    text = SEEDS_PATH.read_text(encoding="utf-8")
    assert "F101" in text
    assert "WINDOWS_ARM64_PACKAGING" in text or "Windows ARM64" in text
