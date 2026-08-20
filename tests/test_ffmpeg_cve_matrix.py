"""F332 — the FFmpeg advisory matrix must declare its own scope.

The matrix graded four CVEs out of a disclosure batch of roughly sixteen and
still reported a clean verdict, which is the same honesty failure the readiness
system exists to prevent: a verdict over a subset read as complete coverage.
These tests pin that every graded advisory is actually checkable, that the
ungraded remainder is enumerated rather than forgotten, and that no report
claims more than it checked.
"""

from __future__ import annotations

import dataclasses
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


class TestNewerReleaseIsNotAssumedFixed:
    """F333 — "9.0.1 > 8.1.2, therefore it has the 8.1.2 fixes" is false.

    The 9.0 series branched from master on 2026-06-26, before the July fix
    commits landed. Version ordering says nothing about whether a branch cut
    earlier received a backport, and the gate used to accept 9.0.x on ordering
    alone.
    """

    def test_a_newer_series_branched_before_the_fix_is_refused(self):
        result = fp.check_security_floor("ffmpeg version 9.0.1 Copyright (c) 2000-2026")
        assert result["ok"] is False
        assert "branched on" in result["reason"]
        assert "not evidence" in result["reason"]

    def test_a_later_patch_in_the_affected_series_is_accepted(self):
        """8.1.3 would be the advisory's own declared fix."""
        result = fp.check_security_floor("ffmpeg version 8.1.3 Copyright (c) 2000-2026")
        assert result["ok"] is True

    def test_the_affected_release_itself_still_fails(self):
        assert fp.check_security_floor("ffmpeg version 8.1.2 Copyright")["ok"] is False

    def test_recorded_backport_evidence_accepts_the_release(self):
        advisory = fp.SECURITY_ADVISORIES[0]
        target = (9, 0, 1)
        assert fp._grade_newer_release(advisory, target)[0] is False

        backported = dataclasses.replace(advisory, backported_in=(target,))
        accepted, why = fp._grade_newer_release(backported, target)
        assert accepted is True
        assert "backport" in why

    def test_a_series_branched_after_the_fix_is_accepted(self):
        advisory = fp.SECURITY_ADVISORIES[0]
        try:
            fp.RELEASE_BRANCH_POINTS[(10, 0)] = "2027-01-01"
            accepted, why = fp._grade_newer_release(advisory, (10, 0, 0))
            assert accepted is True
            assert "branched on 2027-01-01" in why
        finally:
            fp.RELEASE_BRANCH_POINTS.pop((10, 0), None)

    def test_an_unrecorded_series_is_refused_not_assumed(self):
        advisory = fp.SECURITY_ADVISORIES[0]
        accepted, why = fp._grade_newer_release(advisory, (11, 4, 0))
        assert accepted is False
        assert "branch point is not recorded" in why

    def test_the_pinned_snapshot_still_passes(self):
        """The lane users are actually told to use must keep working."""
        result = fp.check_security_floor(
            "ffmpeg version 2026-08-03-git-01a25f74cc-full_build-www.gyan.dev Copyright"
        )
        assert result["ok"] is True


class TestClosedLaneIsStatedInCode:
    def test_the_closed_lane_carries_a_reason(self):
        if not fp.RELEASE_LANE_OPEN:
            assert "8.1.3 was never published" in fp.RELEASE_LANE_CLOSED_REASON
            assert "branched" in fp.RELEASE_LANE_CLOSED_REASON

    def test_provenance_reports_the_lane_state(self):
        record = fp.provenance_record(banner="ffmpeg version 8.1.2 Copyright")
        assert record["release_lane_open"] is fp.RELEASE_LANE_OPEN
        assert record["release_branch_points"]["9.0"] == "2026-06-26"
        if not fp.RELEASE_LANE_OPEN:
            assert record["release_lane_note"]

    def test_the_error_message_does_not_advertise_a_closed_lane(self):
        grade = fp.check_security_floor("ffmpeg version 8.1.2 Copyright")
        message = str(fp.FfmpegSecurityError("ffmpeg", grade))
        if not fp.RELEASE_LANE_OPEN:
            assert "git-master snapshot" in message
            assert "8.1.3" not in message.split(fp.RELEASE_LANE_CLOSED_REASON)[0]
