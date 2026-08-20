"""Tests for opencut.core.ffmpeg_provenance — version parsing + security floor."""

import re

from opencut.core import ffmpeg_provenance as fp

# Real gyan.dev banner shapes (first line of `ffmpeg -version`).
BANNER_801 = (
    "ffmpeg version 8.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2025 "
    "the FFmpeg developers"
)
BANNER_811 = (
    "ffmpeg version 8.1.1-essentials_build-www.gyan.dev Copyright (c) 2000-2026 "
    "the FFmpeg developers"
)
BANNER_812 = (
    "ffmpeg version 8.1.2-essentials_build-www.gyan.dev Copyright (c) 2000-2026 "
    "the FFmpeg developers"
)
BANNER_813 = (
    "ffmpeg version 8.1.3-essentials_build-www.gyan.dev Copyright (c) 2000-2026 "
    "the FFmpeg developers"
)
BANNER_81 = (
    "ffmpeg version 8.1-essentials_build-www.gyan.dev Copyright (c) 2000-2025 "
    "the FFmpeg developers"
)
BANNER_GIT_POSTFIX = (
    "ffmpeg version 2026-08-03-git-01a25f74cc-full_build-www.gyan.dev Copyright "
    "(c) 2000-2026 the FFmpeg developers"
)
BANNER_GIT_PREFIX = (
    "ffmpeg version 2026-05-01-git-aaaaaaaaaa-full_build-www.gyan.dev Copyright "
    "(c) 2000-2026 the FFmpeg developers"
)
BANNER_BTBN_NODATE = "ffmpeg version N-118000-gabcdef1234-20260612 Copyright (c) 2000-2026"
BANNER_DISTRO = "ffmpeg version n8.1.1 Copyright (c) 2000-2026 the FFmpeg developers"


def test_parse_release_banner():
    rec = fp.parse_version_banner(BANNER_801)
    assert rec["release"] == (8, 0, 1)
    assert rec["is_git_snapshot"] is False
    assert rec["flavor"] == "essentials"
    assert rec["builder"] == "gyan.dev"
    assert rec["git_commit"] is None


def test_parse_git_snapshot_banner():
    rec = fp.parse_version_banner(BANNER_GIT_POSTFIX)
    assert rec["is_git_snapshot"] is True
    assert rec["snapshot_date"] == "2026-08-03"
    assert rec["git_commit"] == "01a25f74cc"
    assert rec["flavor"] == "full"
    assert rec["release"] is None


def test_parse_distro_banner_tolerates_n_prefix():
    rec = fp.parse_version_banner(BANNER_DISTRO)
    assert rec["release"] == (8, 1, 1)
    assert rec["is_git_snapshot"] is False


def test_parse_empty_or_garbage_never_raises():
    assert fp.parse_version_banner("")["raw"] == ""
    assert fp.parse_version_banner("not an ffmpeg banner at all")["raw"] == ""


def test_floor_release_801_is_below():
    res = fp.check_security_floor(BANNER_801)
    assert res["ok"] is False
    assert res["lane"] == "release"
    # The matrix names which advisories are unresolved rather than quoting a
    # single global floor, so assert the verdict instead of the prose.
    assert "CVE-2026-64832" in res["unresolved_cves"]
    assert "at/below the last affected release" in res["reason"]


def test_floor_release_811_is_below_cve_floor():
    res = fp.check_security_floor(BANNER_811)
    assert res["ok"] is False
    assert res["lane"] == "release"
    assert set(res["unresolved_cves"]) >= set(fp.JULY_2026_CVES)


def test_floor_release_812_is_rejected_for_july_cves():
    res = fp.check_security_floor(BANNER_812)
    assert res["ok"] is False
    assert res["lane"] == "release"
    assert res["version"].startswith("8.1.2")
    assert "CVE-2026-64832" in res["reason"]


def test_floor_release_813_reenables_release_lane():
    res = fp.check_security_floor(BANNER_813)
    assert res["ok"] is True
    assert res["lane"] == "release"


def test_floor_release_810_is_below():
    # "8.1" parses as 8.1.0 which is below the 8.1.3 floor.
    res = fp.check_security_floor(BANNER_81)
    assert res["ok"] is False
    assert res["lane"] == "release"


def test_floor_git_postfix_snapshot_passes():
    res = fp.check_security_floor(BANNER_GIT_POSTFIX)
    assert res["ok"] is True
    assert res["lane"] == "snapshot"
    assert res["git_commit"] == "01a25f74cc"
    assert res["snapshot_date"] == "2026-08-03"
    assert res["flavor"] == "full"


def test_pinned_snapshot_requires_the_bundled_commit():
    current = fp.check_security_floor(BANNER_GIT_POSTFIX)
    assert fp.is_pinned_snapshot(current) is True
    other = fp.check_security_floor(
        "ffmpeg version 2026-08-03-git-aaaaaaaaaa-full_build-www.gyan.dev"
    )
    assert fp.is_pinned_snapshot(other) is False


def test_floor_git_prefix_snapshot_is_below():
    res = fp.check_security_floor(BANNER_GIT_PREFIX)
    assert res["ok"] is False
    assert res["lane"] == "snapshot"
    assert "predates" in res["reason"]


def test_floor_undated_git_snapshot_is_not_confirmable():
    res = fp.check_security_floor(BANNER_BTBN_NODATE)
    assert res["ok"] is False
    assert res["lane"] == "snapshot"
    assert "carries no date" in res["reason"]
    # An unknown build must fail closed on every advisory, not just the newest.
    assert set(res["unresolved_cves"]) == {a.cve for a in fp.SECURITY_ADVISORIES}


def test_floor_unparseable_banner():
    res = fp.check_security_floor("garbage with no version token")
    assert res["ok"] is False
    assert res["lane"] == "unknown"


def test_cves_listed_in_every_result():
    res = fp.check_security_floor(BANNER_812)
    assert "CVE-2026-8461" in res["cves"]
    assert "CVE-2026-64832" in res["cves"]
    assert "CVE-2026-64833" in res["cves"]
    assert "CVE-2026-64835" in res["cves"]
    assert "CVE-2026-66041" in res["cves"]
    assert len(res["cves"]) == len(fp.SECURITY_CVES)
    assert list(fp.SECURITY_FIX_COMMITS) == res["fix_commits"]
    assert fp.MAGICYUV_FIX_COMMITS == (
        "374b726ffa878ee1cadb987bd1e1e20cc7ed8845",
        "5806e8b9f34f1b0663b3017ef9dd1aa5d08116d1",
    )


def test_provenance_record_documents_floor_without_binary():
    rec = fp.provenance_record(banner="")
    assert rec["required_release_floor"] == "8.1.3"
    assert rec["required_snapshot_floor_date"] == "2026-07-06"
    assert rec["reference_git_commit"] == "01a25f74cc446a683318bab13dfd98a467082ef7"
    assert rec["required_fix_commits"] == list(fp.SECURITY_FIX_COMMITS)
    assert rec["pinned_installer_version"] == fp.PINNED_INSTALLER_VERSION
    assert "CVE-2026-6385" in rec["cves_addressed"]
    assert rec["pinned_source"]["sha256"] == fp.PINNED_INSTALLER_SOURCE_SHA256
    assert rec["bundled"] is None


def test_provenance_record_grades_supplied_banner():
    rec = fp.provenance_record(banner=BANNER_GIT_POSTFIX)
    assert rec["bundled"] is not None
    assert rec["bundled"]["ok"] is True


def test_require_security_floor_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(fp, "probe_binary_security", lambda _binary: fp.check_security_floor(BANNER_811))

    try:
        fp.require_security_floor("unsafe-ffmpeg")
    except fp.FfmpegSecurityError as exc:
        assert exc.code == "FFMPEG_SECURITY_FLOOR"
        assert "8.1.3" in str(exc)
        assert "CVE-2026-64832" in str(exc)
        assert exc.grade["version"].startswith("8.1.1")
    else:  # pragma: no cover - defensive
        raise AssertionError("unsafe FFmpeg should be blocked")


# ---------------------------------------------------------------------------
# F304 — per-CVE advisory matrix
# ---------------------------------------------------------------------------

BANNER_SNAPSHOT_PARTIAL = (
    "ffmpeg version 2026-07-01-git-1111111111-full_build-www.gyan.dev Copyright (c)"
)
CONF_FULL = "--enable-nvdec --enable-cuvid --enable-libquirc --enable-libx264"
CONF_MINIMAL = "--enable-libx264 --enable-gpl"
CONF_DISABLED = (
    "--disable-demuxer=spdif --disable-decoder=adpcm_adx --disable-decoder=magicyuv"
)


def test_every_advisory_names_a_fix_commit_and_landing_date():
    assert fp.SECURITY_ADVISORIES, "the matrix must not be empty"
    for advisory in fp.SECURITY_ADVISORIES:
        assert len(advisory.fix_commit) == 40, advisory.cve
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", advisory.fix_landed), advisory.cve
        assert advisory.capability_tokens, advisory.cve


def test_matrix_covers_every_declared_july_cve():
    """The gate must not certify a CVE it does not actually grade."""
    graded = {advisory.cve for advisory in fp.SECURITY_ADVISORIES}
    assert set(fp.JULY_2026_CVES) <= graded


def test_snapshot_is_graded_per_cve_not_by_one_global_date():
    """A snapshot can carry some fixes and not others."""
    res = fp.check_security_floor(BANNER_SNAPSHOT_PARTIAL, configure_line=CONF_FULL)
    assert res["ok"] is False
    # spdif (2026-06-29) and ADX (2026-06-30) landed before this build.
    assert "CVE-2026-64833" not in res["unresolved_cves"]
    assert "CVE-2026-64835" not in res["unresolved_cves"]
    # NVDEC (2026-07-04) and quirc (2026-07-05) did not.
    assert "CVE-2026-64832" in res["unresolved_cves"]
    assert "CVE-2026-66041" in res["unresolved_cves"]


def _disable_every_default_built_component() -> str:
    """A configure line that explicitly disables each default-built component.

    Derived from the matrix rather than hand-listed so grading a new advisory
    cannot silently turn the waiver test into a date test.
    """
    flags = [
        token[1:]
        for advisory in fp.SECURITY_ADVISORIES
        for token in advisory.capability_tokens
        if token.startswith("!")
    ]
    return " ".join(["--enable-libx264", "--enable-gpl", *sorted(set(flags))])


def test_absent_component_waives_its_advisory():
    """'Prove the component absent' is the acceptance path a date cannot give.

    The snapshot pre-dates the NVDEC and quirc fixes, so nothing about the date
    could clear them; only their absence from the configure line does.
    """
    res = fp.check_security_floor(
        BANNER_SNAPSHOT_PARTIAL,
        configure_line=_disable_every_default_built_component(),
    )
    assert res["ok"] is True
    assert {"CVE-2026-64832", "CVE-2026-66041"} <= set(res["not_applicable_cves"])


def test_the_waiver_test_is_not_secretly_passing_on_the_snapshot_date():
    """Both waived advisories must still be unfixed at the fixture's date."""
    snapshot_date = "2026-07-01"
    for cve in ("CVE-2026-64832", "CVE-2026-66041"):
        advisory = next(a for a in fp.SECURITY_ADVISORIES if a.cve == cve)
        assert advisory.fix_landed > snapshot_date
        assert all(not t.startswith("!") for t in advisory.capability_tokens)


def test_default_built_component_needs_an_explicit_disable_to_be_waived():
    affected = fp.check_security_floor(BANNER_812, configure_line=CONF_MINIMAL)
    assert affected["ok"] is False
    assert "CVE-2026-64833" in affected["unresolved_cves"]

    disabled = fp.check_security_floor(BANNER_812, configure_line=CONF_DISABLED)
    assert "CVE-2026-64833" in disabled["not_applicable_cves"]


def test_missing_configure_line_never_waives_an_advisory():
    """Absence of evidence is not evidence of absence."""
    res = fp.check_security_floor(BANNER_812)
    assert res["ok"] is False
    assert res["not_applicable_cves"] == []


def test_pinned_bundled_snapshot_still_clears_every_advisory():
    res = fp.check_security_floor(
        f"ffmpeg version {fp.PINNED_INSTALLER_VERSION} Copyright (c)"
    )
    assert res["ok"] is True
    assert res["unresolved_cves"] == []
    assert all(entry["status"] == "fixed" for entry in res["advisories"])
