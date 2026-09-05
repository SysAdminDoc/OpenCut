"""The provenance documents must state what the code enforces.

``docs/RELEASE_PROVENANCE.md`` told a reader that a tagged FFmpeg release
``>= 8.1.1`` was acceptable on an open release lane. By then
``opencut/core/ffmpeg_provenance.py`` had closed the release lane outright,
raised ``RELEASE_FLOOR`` to 8.1.3, and moved ``SNAPSHOT_FLOOR_DATE`` to
2026-07-06, and the installers had moved to a 2026-08-03 git snapshot. Four
statements, all wrong, all pinned by literal assertions in the test that was
supposed to guard the document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opencut.tools.check_provenance_docs import (
    PYTHON_ADVISORIES_DOC,
    RELEASE_PROVENANCE_DOC,
    collect_advisory_facts,
    collect_release_provenance_facts,
    documented_floor_raises,
    find_divergences,
    floor_divergences,
    main,
    undocumented_waivers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_documents_currently_match_the_code():
    divergences = find_divergences()
    assert not divergences, "\n".join(
        f"{item['document']}: {item['field']} -- {item['problem']}" for item in divergences
    )


def test_facts_are_read_from_the_code_not_hardcoded():
    """Each fact must carry the value the module holds right now."""
    from opencut.core import embedded_media_provenance as embedded
    from opencut.core import ffmpeg_provenance as provenance

    facts = {fact.field: fact.value for fact in collect_release_provenance_facts()}
    assert facts["ffmpeg_provenance.SNAPSHOT_FLOOR_DATE"] == provenance.SNAPSHOT_FLOOR_DATE
    assert (
        facts["embedded_media_provenance.FIXED_FFMPEG_VERSION"]
        == embedded.FIXED_FFMPEG_VERSION
    )


def test_a_closed_release_lane_is_not_advertised_as_an_option():
    from opencut.core import ffmpeg_provenance as provenance

    docs = RELEASE_PROVENANCE_DOC.read_text(encoding="utf-8")
    if provenance.RELEASE_LANE_OPEN:
        pytest.skip("release lane is open; the closed-lane wording does not apply")
    assert "Release lane is closed" in docs
    assert "acceptable on **either** lane" not in docs


def test_every_waived_advisory_appears_in_the_public_table():
    from opencut.tools.pip_audit_extras import ALLOWED_ADVISORIES

    docs = PYTHON_ADVISORIES_DOC.read_text(encoding="utf-8")
    for advisory_id, entry in ALLOWED_ADVISORIES.items():
        assert advisory_id in docs, f"{advisory_id} is waived in code but not documented"
        assert f"`{entry.package}`" in docs


def test_a_waiver_the_code_dropped_is_reported():
    """The dangerous direction: the doc says reviewed and accepted, code does not."""
    text = (
        "## Allow-list\n\n"
        "| Advisory | Package |\n|---|---|\n"
        "| CVE-2099-11111 | `ghost` |\n\n"
        "## Floor raises\n"
    )
    assert undocumented_waivers(text) == ["CVE-2099-11111"]


def test_rationale_prose_after_the_table_is_not_treated_as_a_waiver():
    text = (
        "## Allow-list\n\n| Advisory |\n|---|\n\n"
        "## Floor raises\n\n"
        "| pkg | CVE-2099-22222 fixed upstream |\n"
    )
    assert undocumented_waivers(text) == []


# ---------------------------------------------------------------------------
# Floor raises
# ---------------------------------------------------------------------------

def test_floor_rows_are_parsed_from_the_table():
    rows = dict(documented_floor_raises(PYTHON_ADVISORIES_DOC.read_text(encoding="utf-8")))
    assert rows, "no floor-raise rows parsed; the table format changed"
    for package, specifier in rows.items():
        assert specifier.startswith((">", "<", "=", "!", "~")), (
            f"{package} floor {specifier!r} is prose, not a specifier"
        )


def test_prose_cells_are_not_mistaken_for_specifiers():
    text = (
        "## Floor raises\n\n"
        "| Package | Old floor | New floor |\n|---|---|---|\n"
        "| `thing` | transitive | `transitive, lock 1.2.3` |\n"
    )
    assert documented_floor_raises(text) == []


def test_a_documented_floor_missing_from_pyproject_is_reported():
    text = (
        "## Floor raises\n\n"
        "| Package | Old floor | New floor |\n|---|---|---|\n"
        "| `nonexistent-package` | `>=1` | `>=99.0.0,<100` |\n"
    )
    problems = floor_divergences(text)
    assert len(problems) == 1
    assert "nonexistent-package>=99.0.0,<100" in problems[0]["expected"]


def test_every_documented_floor_is_actually_declared():
    """A hardening step recorded but never applied is worse than none."""
    assert floor_divergences(PYTHON_ADVISORIES_DOC.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_cli_passes_on_the_current_tree(capsys):
    assert main(["--check"]) == 0
    assert "match the executable policy" in capsys.readouterr().out


def test_cli_reports_each_divergent_field(tmp_path, monkeypatch, capsys):
    """The acceptance: fail naming the exact divergent field."""
    from opencut.tools import check_provenance_docs

    empty = tmp_path / "RELEASE_PROVENANCE.md"
    empty.write_text("# nothing\n", encoding="utf-8")
    monkeypatch.setattr(check_provenance_docs, "RELEASE_PROVENANCE_DOC", empty)

    assert main(["--check"]) == 1
    output = capsys.readouterr().out
    assert "ffmpeg_provenance.SNAPSHOT_FLOOR_DATE" in output
    assert "AppConstants.BundledFfmpegVersion" in output


def test_without_check_the_cli_reports_but_does_not_fail(tmp_path, monkeypatch):
    from opencut.tools import check_provenance_docs

    empty = tmp_path / "RELEASE_PROVENANCE.md"
    empty.write_text("# nothing\n", encoding="utf-8")
    monkeypatch.setattr(check_provenance_docs, "RELEASE_PROVENANCE_DOC", empty)
    assert main([]) == 0


def test_the_release_gate_runs_this_check():
    import importlib.util
    import sys

    spec_path = REPO_ROOT / "scripts" / "release_smoke.py"
    spec = importlib.util.spec_from_file_location("release_smoke_for_provenance_test", spec_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    commands = [" ".join(str(part) for part in cmd) for _label, cmd in module.GENERATED_DOC_CHECKS]
    assert any("check_provenance_docs" in cmd for cmd in commands)


def test_advisory_facts_cover_aliases():
    fields = {fact.field for fact in collect_advisory_facts()}
    assert any(field.endswith(".alias") for field in fields), (
        "GHSA aliases are how most tools name an advisory; they must be documented too"
    )
