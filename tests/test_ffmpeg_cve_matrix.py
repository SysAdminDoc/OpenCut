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
        "ffmpeg version 2026-11-11-git-23c4540053-full_build-www.gyan.dev "
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

    def test_a_later_patch_counts_only_through_a_recorded_backport(self):
        """F451: 8.1.3 used to clear the July advisories on its number alone."""
        rec = fp.parse_version_banner("ffmpeg version 8.1.3 Copyright (c) 2000-2026")
        checked = 0
        for adv in fp.SECURITY_ADVISORIES:
            if adv.affected_max_release not in ((8, 1, 1), (8, 1, 2)):
                continue
            checked += 1
            graded = fp.grade_advisory(adv, rec)
            assert graded["status"] == "fixed", adv.cve
            assert "backport" in graded["reason"], adv.cve

            unrecorded = dataclasses.replace(adv, backported_in=())
            graded = fp.grade_advisory(unrecorded, rec)
            assert graded["status"] == "vulnerable", adv.cve
            assert "not evidence" in graded["reason"], adv.cve
        assert checked == 17

    def test_a_later_tag_inherits_an_earlier_tags_backport(self):
        """Release branches only grow, so 8.1.4 descends from 8.1.3."""
        advisory = fp.SECURITY_ADVISORIES[0]
        accepted, why = fp._grade_newer_release(advisory, (8, 1, 4))
        assert accepted is True
        assert "descends from 8.1.3" in why

    def test_an_earlier_tag_does_not_inherit_a_later_backport(self):
        advisory = fp.SECURITY_ADVISORIES[0]
        assert (9, 0, 2) in advisory.backported_in
        accepted, why = fp._grade_newer_release(advisory, (9, 0, 1))
        assert accepted is False
        assert "not evidence" in why

    def test_a_series_branched_between_fixes_clears_only_the_earlier_ones(self):
        """9.1 branched 2026-10-23: after libssh's fix, before G.726 and EXIF."""
        result = fp.check_security_floor("ffmpeg version 9.1.0 Copyright (c) 2000-2026")
        assert result["ok"] is False
        assert set(result["unresolved_cves"]) == {"CVE-2026-125064", "CVE-2026-125180"}

    def test_8_1_3_is_refused_on_the_advisories_that_reach_9_0_2(self):
        result = fp.check_security_floor("ffmpeg version 8.1.3 Copyright (c) 2000-2026")
        assert result["ok"] is False
        assert set(result["unresolved_cves"]) == set(fp.LATE_2026_CVES)

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
            "ffmpeg version 2026-11-11-git-23c4540053-full_build-www.gyan.dev Copyright"
        )
        assert result["ok"] is True


class TestClosedLaneIsStatedInCode:
    def test_the_lane_is_computed_from_the_published_tags(self):
        assert fp.RELEASE_LANE_OPEN is any(
            not gaps for gaps in fp.RELEASE_LANE_GAPS.values()
        )
        assert set(fp.RELEASE_LANE_GAPS) == set(fp.PUBLISHED_RELEASES)

    def test_both_published_tags_miss_exactly_the_late_2026_fixes(self):
        """Checked upstream 2026-11-15: n8.1.3 and n9.0.2 carry every June and
        July fix and none of the three NVD listed after 9.0.2."""
        for release in ((8, 1, 3), (9, 0, 2)):
            assert release in fp.PUBLISHED_RELEASES
            assert fp.RELEASE_LANE_GAPS[release] == fp.LATE_2026_CVES, release

    def test_the_closed_lane_reason_names_every_missing_fix(self):
        if fp.RELEASE_LANE_OPEN:
            return
        reason = fp.RELEASE_LANE_CLOSED_REASON
        by_cve = {adv.cve: adv for adv in fp.SECURITY_ADVISORIES}
        for release, gaps in fp.RELEASE_LANE_GAPS.items():
            assert ".".join(map(str, release)) in reason
            for cve in gaps:
                assert f"{cve} ({by_cve[cve].fix_commit[:10]})" in reason
        assert fp.SNAPSHOT_FLOOR_DATE in reason
        assert "never published" not in reason

    def test_provenance_reports_the_lane_state(self):
        record = fp.provenance_record(banner="ffmpeg version 8.1.2 Copyright")
        assert record["release_lane_open"] is fp.RELEASE_LANE_OPEN
        assert record["release_branch_points"] == {
            "8.1": "2026-03-08",
            "9.0": "2026-06-26",
            "9.1": "2026-10-23",
        }
        assert record["security_floor"] == fp.SECURITY_FLOOR_SUMMARY
        if not fp.RELEASE_LANE_OPEN:
            assert record["release_lane_note"]

    def test_the_error_message_does_not_advertise_a_closed_lane(self):
        grade = fp.check_security_floor("ffmpeg version 8.1.2 Copyright")
        message = str(fp.FfmpegSecurityError("ffmpeg", grade))
        if not fp.RELEASE_LANE_OPEN:
            assert "git-master snapshot" in message
            assert "8.1.3" not in message.split(fp.RELEASE_LANE_CLOSED_REASON)[0]


def test_every_graded_advisory_is_reported_in_the_public_constants():
    """The matrix and the reported CVE/commit lists were separate sources.

    `verify_ffmpeg_provenance.py` and the release metadata print
    SECURITY_CVES / SECURITY_FIX_COMMITS, so an advisory graded in the matrix
    but missing from those lists is silently absent from every report.
    """
    reported_cves = set(fp.SECURITY_CVES)
    reported_commits = set(fp.SECURITY_FIX_COMMITS)
    for advisory in fp.SECURITY_ADVISORIES:
        assert advisory.cve in reported_cves, (
            f"{advisory.cve} is graded but missing from SECURITY_CVES"
        )
        assert advisory.fix_commit in reported_commits, (
            f"{advisory.cve} fix commit is graded but missing from SECURITY_FIX_COMMITS"
        )


def test_no_advisory_is_both_graded_and_listed_as_ungraded():
    graded = {advisory.cve for advisory in fp.SECURITY_ADVISORIES}
    assert graded.isdisjoint(set(fp.UNGRADED_ADVISORIES))


# ---------------------------------------------------------------------------
# F464: the 2026-11 refresh past three NVD-listed fixes
# ---------------------------------------------------------------------------

LATE_CVES = {"CVE-2026-125064", "CVE-2026-125180", "CVE-2026-107675"}
OLD_PIN_BANNER = "ffmpeg version 2026-08-03-git-01a25f74cc-full_build-www.gyan.dev Copyright"
OLD_PIN_CONFIGURE = "--enable-gpl --enable-gnutls --enable-libssh --enable-whisper"


def test_the_previous_pin_is_refused_on_exactly_the_late_2026_fixes():
    """2026-08-03 predates all three fixes and nothing else in the matrix."""
    result = fp.check_security_floor(OLD_PIN_BANNER, OLD_PIN_CONFIGURE)
    assert result["ok"] is False
    assert set(result["unresolved_cves"]) == LATE_CVES


def test_the_new_pin_clears_the_late_2026_fixes_by_date_too():
    """Even without the known-good commit shortcut, the snapshot date clears them."""
    rec = fp.parse_version_banner(
        "ffmpeg version 2026-11-11-git-bbbbbbbbbb-full_build-www.gyan.dev"
    )
    for adv in fp.SECURITY_ADVISORIES:
        if adv.cve in LATE_CVES:
            assert fp.grade_advisory(adv, rec, OLD_PIN_CONFIGURE)["status"] == "fixed"


def test_snapshot_floor_is_the_latest_graded_fix():
    assert fp.SNAPSHOT_FLOOR_DATE == max(adv.fix_landed for adv in fp.SECURITY_ADVISORIES)
    assert fp.REFERENCE_GIT_DATE >= fp.SNAPSHOT_FLOOR_DATE


def test_late_2026_components_can_be_proven_absent():
    by_cve = {adv.cve: adv for adv in fp.SECURITY_ADVISORIES}
    rec = fp.parse_version_banner(OLD_PIN_BANNER)

    def status(cve, configure):
        return fp.grade_advisory(by_cve[cve], rec, configure)["status"]

    # libssh is opt-in, so leaving it out removes the SFTP protocol.
    assert status("CVE-2026-107675", "--enable-gpl --enable-gnutls") == "not_applicable"
    assert status("CVE-2026-107675", OLD_PIN_CONFIGURE) == "vulnerable"
    # g726 builds by default; only disabling both decoders removes g726.o.
    both = "--disable-decoder=adpcm_g726,adpcm_g726le"
    assert status("CVE-2026-125064", both) == "not_applicable"
    assert status("CVE-2026-125064", "--disable-decoder=adpcm_g726le") == "vulnerable"
    # exif.o has no switch of its own.
    assert status("CVE-2026-125180", "--disable-decoder=mjpeg") == "vulnerable"
    assert status("CVE-2026-125180", "--disable-avcodec") == "not_applicable"


def _load_verifier():
    import importlib.util

    path = os.path.join(REPO_ROOT, "scripts", "verify_ffmpeg_provenance.py")
    spec = importlib.util.spec_from_file_location("verify_ffmpeg_provenance_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_payload(tmp_path, monkeypatch, *, probe_hash=None, configure=None):
    verifier = _load_verifier()
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.write_bytes(b"ffmpeg")
    ffprobe.write_bytes(b"ffprobe")
    configure = configure or "--enable-gpl --enable-gnutls --enable-libssh --enable-whisper"
    banner = (
        f"ffmpeg version {fp.PINNED_INSTALLER_VERSION} Copyright (c) 2000-2026\n"
        f"configuration: {configure}\n"
    )
    hashes = {
        "ffmpeg.exe": fp.PINNED_INSTALLER_BINARY_SHA256["ffmpeg.exe"],
        "ffprobe.exe": probe_hash or fp.PINNED_INSTALLER_BINARY_SHA256["ffprobe.exe"],
    }
    monkeypatch.setattr(verifier, "_banner", lambda _path: banner)
    monkeypatch.setattr(verifier, "_sha256", lambda path: hashes[path.name.lower()])
    return verifier, [str(ffmpeg), "--ffprobe", str(ffprobe), "--pinned-payload"]


def test_pinned_payload_accepts_the_pinned_binaries(tmp_path, monkeypatch, capsys):
    verifier, argv = _fake_payload(tmp_path, monkeypatch)
    assert verifier.main(argv) == 0
    assert "RESULT: OK" in capsys.readouterr().out


def test_pinned_payload_refuses_a_swapped_binary(tmp_path, monkeypatch, capsys):
    verifier, argv = _fake_payload(tmp_path, monkeypatch, probe_hash="0" * 64)
    assert verifier.main(argv) == 3
    assert "ffprobe.exe SHA-256 differs from the pinned archive" in capsys.readouterr().err


def test_pinned_payload_refuses_a_build_without_whisper(tmp_path, monkeypatch, capsys):
    verifier, argv = _fake_payload(
        tmp_path, monkeypatch, configure="--enable-gpl --enable-gnutls --enable-libssh"
    )
    assert verifier.main(argv) == 3
    assert "configuration lacks --enable-whisper" in capsys.readouterr().err
