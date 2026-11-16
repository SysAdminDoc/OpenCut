"""F451: every installer grades FFmpeg the way the Python matrix does.

The Python grader decides the release lane from recorded backports. The C#
installer, the Inno script and Install.ps1 can't import it, so they carry the
same floor as literals. These tests keep the literals in step, which is how the
Inno script kept accepting July snapshots after the floor moved to November.
"""

from __future__ import annotations

import re
from pathlib import Path

from opencut.core import ffmpeg_provenance as fp

REPO = Path(__file__).resolve().parents[1]
APP_CONSTANTS = REPO / "installer/src/OpenCut.Installer/Models/AppConstants.cs"
VERIFIER = REPO / "installer/src/OpenCut.Installer/Services/FfmpegSecurityVerifier.cs"
INNO = REPO / "OpenCut.iss"
INSTALL_PS1 = REPO / "Install.ps1"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _inno_define(name: str) -> str:
    match = re.search(rf'^#define {name} "([^"]*)"$', _read(INNO), re.MULTILINE)
    assert match, f"OpenCut.iss has no {name} define"
    return match.group(1)


def _inno_function(name: str) -> str:
    text = _read(INNO)
    start = text.index(f"function {name}(")
    end = text.index("\nend;\n", start)
    return text[start:end]


def _missing_cves() -> set[str]:
    return {cve for gaps in fp.RELEASE_LANE_GAPS.values() for cve in gaps}


def test_installers_state_the_python_security_floor():
    constants = re.search(
        r'BundledFfmpegSecurityFloor = "([^"]*)";', _read(APP_CONSTANTS)
    )
    assert constants
    assert constants.group(1) == fp.SECURITY_FLOOR_SUMMARY
    assert _inno_define("BundledFfmpegSecurityFloor") == fp.SECURITY_FLOOR_SUMMARY


def test_every_installer_uses_the_python_snapshot_floor():
    year, month, day = (int(part) for part in fp.SNAPSHOT_FLOOR_DATE.split("-"))
    assert f"SnapshotFloor = new({year}, {month}, {day});" in _read(VERIFIER)
    assert _inno_define("BundledFfmpegSnapshotFloor") == fp.SNAPSHOT_FLOOR_DATE
    assert f'$snapshotFloor = "{fp.SNAPSHOT_FLOOR_DATE}"' in _read(INSTALL_PS1)


def test_inno_compares_snapshot_dates_against_the_define():
    """The F464 repin moved every string to 2026-11-06 but left Inno's
    hand-coded year/month/day test at July 6."""
    body = _inno_function("IsPostFixSnapshotDate")
    assert "'{#BundledFfmpegSnapshotFloor}'" in body
    assert "MonthValue" not in body
    assert not re.search(r"\b20\d\d\b", body), "a hard-coded year bypasses the define"


def test_installers_refuse_tagged_releases_while_the_lane_is_closed():
    assert not fp.RELEASE_LANE_OPEN, (
        "A published tag now clears the matrix. Teach FfmpegSecurityVerifier.cs, "
        "OpenCut.iss and Install.ps1 to accept exactly "
        f"{fp.RELEASE_LANE_CLEARED}, then update this test."
    )
    banner_fn = _inno_function("FfmpegBannerClearsFloor")
    assert "MajorValue" not in banner_fn, "Inno still grades releases by number"
    assert "ReleaseFloor" not in _read(VERIFIER)
    assert "[version]::new" not in _read(INSTALL_PS1)


def test_installer_refusals_name_what_the_newest_tags_miss():
    by_cve = {adv.cve: adv for adv in fp.SECURITY_ADVISORIES}
    verifier = _read(VERIFIER)
    install_ps1 = _read(INSTALL_PS1)
    inno = _inno_function("FfmpegBannerClearsFloor")
    for cve in _missing_cves():
        assert f"{cve} ({by_cve[cve].fix_commit[:10]})" in verifier, cve
        assert cve in install_ps1, cve
        assert cve in inno, cve
    for release in fp.PUBLISHED_RELEASES:
        tag = ".".join(map(str, release))
        assert tag in verifier, tag
        assert tag in install_ps1, tag
