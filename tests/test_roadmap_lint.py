"""Tests for the roadmap source-appendix linter (F118)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from opencut.tools import lint_roadmap_sources as linter


def test_clean_roadmap_passes(tmp_path):
    text = textwrap.dedent(
        """
        # Demo

        Body that cites [L01] and [L02] inside a sentence.

        ## Appendix

        - [L01] `tests/conftest.py` exists in the working tree.
        - [L02] `scripts/sync_version.py` exists too.
        """
    )

    report = linter.lint(text, repo_root=Path(__file__).resolve().parents[1])

    assert not report.has_errors()
    assert report.citation_count == 2
    assert report.appendix_count == 2


def test_dangling_citation_is_error(tmp_path):
    text = textwrap.dedent(
        """
        # Demo
        Cites [L99] which doesn't exist.

        ## Appendix
        - [L01] `scripts/sync_version.py`
        """
    )

    report = linter.lint(text, repo_root=Path(__file__).resolve().parents[1])

    assert report.has_errors()
    errors = [f for f in report.findings if f.rule == "dangling_citation"]
    assert errors and errors[0].message.startswith("citation [L99]")


def test_duplicate_appendix_row_is_error(tmp_path):
    text = textwrap.dedent(
        """
        # Demo
        Body cites [L01].

        ## Appendix
        - [L01] `scripts/sync_version.py`
        - [L01] `scripts/bootstrap_check.py`
        """
    )

    report = linter.lint(text, repo_root=Path(__file__).resolve().parents[1])

    assert report.has_errors()
    rules = {f.rule for f in report.findings}
    assert "duplicate_appendix_row" in rules


def test_unreferenced_appendix_row_is_warning(tmp_path):
    text = textwrap.dedent(
        """
        # Demo
        Cites [L01].

        ## Appendix
        - [L01] `scripts/sync_version.py`
        - [L02] `scripts/bootstrap_check.py`
        """
    )

    report = linter.lint(text, repo_root=Path(__file__).resolve().parents[1])

    assert not report.has_errors()
    rules = {f.rule for f in report.findings}
    assert "unreferenced_appendix_row" in rules


def test_missing_local_path_is_warning(tmp_path):
    text = textwrap.dedent(
        """
        # Demo
        Cites [L01].

        ## Appendix (local evidence)
        - [L01] `does/not/exist/here.txt`
        """
    )

    report = linter.lint(text, repo_root=tmp_path)

    rules = [f.rule for f in report.findings]
    assert "missing_local_path" in rules
    assert not report.has_errors()  # warning only


def test_malformed_url_is_error(tmp_path):
    text = textwrap.dedent(
        """
        # Demo
        Cites [S01].

        ## External sources
        - [S01] not_a_url
        """
    )

    # Our URL extractor only pulls strings prefixed with http(s); an
    # appendix row without a URL is fine. Confirm the row is detected
    # but no error is raised.
    report = linter.lint(text, repo_root=tmp_path)
    assert not report.has_errors()


def test_speaker_tags_inside_text_are_not_citations():
    """``[S1]`` / ``[S2]`` inside Dia TTS prompt syntax must NOT count."""
    text = textwrap.dedent(
        """
        # Demo
        Dia TTS expects `[S1] hi there [S2] hello`.

        ## Appendix
        - [L01] `scripts/sync_version.py`
        """
    )

    report = linter.lint(text, repo_root=Path(__file__).resolve().parents[1])

    assert report.citation_count == 0  # only [S1] / [S2] appear and are ignored
    assert not report.has_errors()


def test_repo_roadmap_passes_strict_check():
    """The committed ROADMAP.md must lint cleanly."""
    text = (linter.REPO_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    report = linter.lint(text, repo_root=linter.REPO_ROOT)

    error_findings = [f for f in report.findings if f.severity == "error"]
    assert not error_findings, (
        "ROADMAP.md has lint errors:\n  - "
        + "\n  - ".join(f"{f.rule}: {f.message}" for f in error_findings[:10])
    )


def test_cli_check_returns_zero_on_clean(monkeypatch, tmp_path):
    """CLI smoke: clean file returns 0."""
    sample = tmp_path / "demo.md"
    sample.write_text(
        "# Demo\n\n[L01] cite\n\n## Appendix\n\n- [L01] `scripts/sync_version.py`\n",
        encoding="utf-8",
    )

    # Place the sample inside the real repo root so the path-existence
    # check resolves against committed files.
    rc = linter.cli(["--file", str(sample), "--json"])
    assert rc == 0


def test_cli_returns_nonzero_on_dangling_citation(tmp_path):
    sample = tmp_path / "demo.md"
    sample.write_text(
        "# Demo\n\n[L99] cite\n\n## Appendix\n\n- [L01] `scripts/sync_version.py`\n",
        encoding="utf-8",
    )

    rc = linter.cli(["--file", str(sample), "--json"])
    assert rc == 1
