"""F332 — the FFmpeg advisory matrix must declare its own scope.

The matrix graded four CVEs out of a disclosure batch of roughly sixteen and
still reported a clean verdict, which is the same honesty failure the readiness
system exists to prevent: a verdict over a subset read as complete coverage.
These tests pin that every graded advisory is actually checkable, that the
ungraded remainder is enumerated rather than forgotten, and that no report
claims more than it checked.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import ffmpeg_provenance as fp  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class TestGradedEntriesAreCheckable:
    def test_every_graded_advisory_carries_a_full_fix_commit(self):
        """Grading needs a commit whose ancestry can actually be checked."""
        for adv in fp.SECURITY_ADVISORIES:
            assert _COMMIT_RE.match(adv.fix_commit or ""), (
                f"{adv.cve} has no usable fix commit: {adv.fix_commit!r}. "
                "An advisory without one belongs in UNGRADED_ADVISORIES."
            )

    def test_every_graded_advisory_records_when_the_fix_landed(self):
        for adv in fp.SECURITY_ADVISORIES:
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", adv.fix_landed or ""), adv.cve

    def test_every_graded_advisory_names_its_component(self):
        for adv in fp.SECURITY_ADVISORIES:
            assert adv.component.strip(), adv.cve
            assert adv.capability_tokens, (
                f"{adv.cve} has no capability tokens, so 'component absent' can "
                "never be established for it"
            )


class TestCoverageIsHonest:
    def test_graded_and_ungraded_never_overlap(self):
        graded = {a.cve for a in fp.SECURITY_ADVISORIES}
        assert graded.isdisjoint(set(fp.UNGRADED_ADVISORIES)), (
            "an advisory cannot be both graded and ungraded"
        )

    def test_coverage_reports_both_halves(self):
        cov = fp.advisory_coverage()
        assert cov["graded_count"] == len(fp.SECURITY_ADVISORIES)
        assert cov["ungraded_count"] == len(fp.UNGRADED_ADVISORIES)
        assert cov["total_known"] == cov["graded_count"] + cov["ungraded_count"]

    def test_complete_flag_tracks_the_ungraded_list(self):
        cov = fp.advisory_coverage()
        assert cov["complete"] is (not fp.UNGRADED_ADVISORIES)

    def test_ungraded_entries_are_well_formed_cve_ids(self):
        for cve in fp.UNGRADED_ADVISORIES:
            assert re.match(r"^CVE-\d{4}-\d{4,7}$", cve), cve


class TestVerdictStatesItsScope:
    #: A build that clears everything the matrix grades.
    CLEAN_BANNER = (
        "ffmpeg version 2026-08-03-git-01a25f74cc-full_build-www.gyan.dev "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )

    def test_a_passing_verdict_does_not_claim_to_have_checked_everything(self):
        result = fp.check_security_floor(self.CLEAN_BANNER)
        assert result["ok"] is True
        if fp.UNGRADED_ADVISORIES:
            assert "not yet graded" in result["reason"], (
                "a clean verdict over a subset must say what it did not check"
            )
            assert "clears all" in result["reason"]
            assert "graded advisories" in result["reason"]

    def test_the_verdict_carries_machine_readable_coverage(self):
        result = fp.check_security_floor(self.CLEAN_BANNER)
        assert result["coverage"]["graded_count"] == len(fp.SECURITY_ADVISORIES)
        assert result["ungraded_cves"] == list(fp.UNGRADED_ADVISORIES)

    def test_a_vulnerable_build_still_fails(self):
        """Scope honesty must not soften a real failure."""
        result = fp.check_security_floor("ffmpeg version 8.1.2 Copyright (c) 2000-2026")
        assert result["ok"] is False
        assert result["unresolved_cves"]

    def test_grading_never_raises_on_junk(self):
        for junk in ("", "not a banner", "ffmpeg version"):
            assert fp.check_security_floor(junk)["ok"] is False


class TestDocumentedScopeMatchesTheMatrix:
    def _read(self, name):
        path = os.path.join(REPO_ROOT, name)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def test_security_doc_does_not_claim_an_unqualified_clean_bill(self):
        """SECURITY.md must not imply the batch is fully graded when it isn't."""
        if not fp.UNGRADED_ADVISORIES:
            return
        text = self._read("SECURITY.md").lower()
        for overclaim in (
            "all known ffmpeg advisories",
            "every ffmpeg advisory",
            "all ffmpeg cves",
        ):
            assert overclaim not in text, (
                f"SECURITY.md claims {overclaim!r} while "
                f"{len(fp.UNGRADED_ADVISORIES)} advisories are ungraded"
            )
