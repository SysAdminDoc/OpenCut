"""F184 — `docs/ROADMAP*.md` must stay short pointer stubs.

The repo used to maintain two parallel copies of the roadmap: the
canonical one at the repo root and a stale March-2026 mirror under
`docs/`. F184 collapsed the docs/ copies to pointer stubs that
redirect to the canonical files. These tests assert the stubs never
grow back into a parallel ledger.

Why a hard test instead of a soft convention: the previous drift was
discovered only during the Pass-1 audit (six months after it started).
A line-count cap + content-required string catches accidental
re-population on the very next commit.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROADMAP = REPO_ROOT / "docs" / "ROADMAP.md"
DOCS_ROADMAP_COMPLETED = REPO_ROOT / "docs" / "ROADMAP-COMPLETED.md"
ROOT_ROADMAP = REPO_ROOT / "ROADMAP.md"
ROOT_ROADMAP_COMPLETED = REPO_ROOT / "ROADMAP-COMPLETED.md"

POINTER_LINE_CAP = 60


def test_canonical_roadmap_exists():
    assert ROOT_ROADMAP.is_file(), "Canonical roadmap missing at repo root"
    assert ROOT_ROADMAP_COMPLETED.is_file(), "Canonical roadmap-completed missing at repo root"


def test_docs_roadmap_is_pointer_stub():
    if not DOCS_ROADMAP.is_file():
        return  # OK — F184 allows the stub to be removed entirely.
    text = DOCS_ROADMAP.read_text(encoding="utf-8")
    line_count = text.count("\n") + 1
    assert line_count <= POINTER_LINE_CAP, (
        f"docs/ROADMAP.md grew to {line_count} lines. F184 caps the "
        f"pointer stub at {POINTER_LINE_CAP} so it cannot become a "
        f"parallel ledger again."
    )
    assert "Moved" in text, "docs/ROADMAP.md must keep the F184 'Moved' banner"
    assert "ROADMAP.md" in text, "docs/ROADMAP.md must reference the canonical path"


def test_docs_roadmap_completed_is_pointer_stub():
    if not DOCS_ROADMAP_COMPLETED.is_file():
        return
    text = DOCS_ROADMAP_COMPLETED.read_text(encoding="utf-8")
    line_count = text.count("\n") + 1
    assert line_count <= POINTER_LINE_CAP, (
        f"docs/ROADMAP-COMPLETED.md grew to {line_count} lines. "
        f"F184 caps the pointer stub at {POINTER_LINE_CAP}."
    )
    assert "Moved" in text
    assert "ROADMAP-COMPLETED.md" in text


def test_docs_roadmap_does_not_carry_phase_tables():
    """The old stale March-2026 mirror used Phase 1 / Phase 2 / Phase 3
    headings. Make sure no stub ever sprouts those again."""
    if not DOCS_ROADMAP.is_file():
        return
    text = DOCS_ROADMAP.read_text(encoding="utf-8")
    # The stub mentions ROADMAP but should not include any phase markup
    # — those headings only belong in the canonical file.
    for marker in ("## Phase 1", "## Phase 2", "## Phase 3", "## Phase 4"):
        assert marker not in text, (
            f"docs/ROADMAP.md must not re-introduce {marker!r}; that lives "
            f"in the canonical ROADMAP.md"
        )


def test_release_smoke_registers_roadmap_mirror_step():
    smoke = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")
    assert "step_roadmap_mirror" in smoke, (
        "release_smoke.py must define step_roadmap_mirror (F184)"
    )
    assert '"roadmap-mirror"' in smoke, (
        "release_smoke.py must register the roadmap-mirror step"
    )
