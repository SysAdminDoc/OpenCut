"""Tests for opencut.core.ffmpeg_provenance — version parsing + security floor."""

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
BANNER_81 = (
    "ffmpeg version 8.1-essentials_build-www.gyan.dev Copyright (c) 2000-2025 "
    "the FFmpeg developers"
)
BANNER_GIT_POSTFIX = (
    "ffmpeg version 2026-06-10-git-b29bdd3715-full_build-www.gyan.dev Copyright "
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
    assert rec["snapshot_date"] == "2026-06-10"
    assert rec["git_commit"] == "b29bdd3715"
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
    assert "predates" in res["reason"]


def test_floor_release_811_is_below_cve_floor():
    res = fp.check_security_floor(BANNER_811)
    assert res["ok"] is False
    assert res["lane"] == "release"
    assert "8.1.2" in res["reason"]


def test_floor_release_812_passes():
    res = fp.check_security_floor(BANNER_812)
    assert res["ok"] is True
    assert res["lane"] == "release"
    assert res["version"].startswith("8.1.2")


def test_floor_release_810_is_below():
    # "8.1" parses as 8.1.0 which is < the 8.1.1 floor.
    res = fp.check_security_floor(BANNER_81)
    assert res["ok"] is False
    assert res["lane"] == "release"


def test_floor_git_postfix_snapshot_passes():
    res = fp.check_security_floor(BANNER_GIT_POSTFIX)
    assert res["ok"] is True
    assert res["lane"] == "snapshot"
    assert res["git_commit"] == "b29bdd3715"
    assert res["snapshot_date"] == "2026-06-10"


def test_floor_git_prefix_snapshot_is_below():
    res = fp.check_security_floor(BANNER_GIT_PREFIX)
    assert res["ok"] is False
    assert res["lane"] == "snapshot"
    assert "predates" in res["reason"]


def test_floor_undated_git_snapshot_is_not_confirmable():
    res = fp.check_security_floor(BANNER_BTBN_NODATE)
    assert res["ok"] is False
    assert res["lane"] == "snapshot"
    assert "no embedded date" in res["reason"]


def test_floor_unparseable_banner():
    res = fp.check_security_floor("garbage with no version token")
    assert res["ok"] is False
    assert res["lane"] == "unknown"


def test_cves_listed_in_every_result():
    res = fp.check_security_floor(BANNER_812)
    assert "CVE-2026-8461" in res["cves"]
    assert "CVE-2026-6385" in res["cves"]
    assert len(res["cves"]) == len(fp.JUNE_2026_CVES)
    assert list(fp.MAGICYUV_FIX_COMMITS) == res["fix_commits"]
    assert fp.MAGICYUV_FIX_COMMITS == (
        "374b726ffa878ee1cadb987bd1e1e20cc7ed8845",
        "5806e8b9f34f1b0663b3017ef9dd1aa5d08116d1",
    )


def test_provenance_record_documents_floor_without_binary():
    rec = fp.provenance_record(banner="")
    assert rec["required_release_floor"] == "8.1.2"
    assert rec["required_snapshot_floor_date"] == "2026-06-10"
    assert rec["reference_git_commit"] == "b29bdd3715"
    assert rec["required_fix_commits"] == list(fp.MAGICYUV_FIX_COMMITS)
    assert rec["pinned_installer_version"] == fp.PINNED_INSTALLER_VERSION
    assert "CVE-2026-6385" in rec["cves_addressed"]
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
        assert "8.1.2" in str(exc)
        assert exc.grade["version"].startswith("8.1.1")
    else:  # pragma: no cover - defensive
        raise AssertionError("unsafe FFmpeg should be blocked")
