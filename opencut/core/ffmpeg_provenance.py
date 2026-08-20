"""Bundled-FFmpeg version + security-patch provenance (RA-FFMPEG-PROV).

The roadmap carries two related FFmpeg items:

* bump the bundled binary from 8.0.x to 8.1.x (for the D3D12VA/Vulkan encoder
  routes that the rest of the codebase already detects), and
* assert a *security patch level*, not just a version string — the June/July
  2026 automated FFmpeg audit disclosed several heap/stack overflows reachable
  via crafted media (the first untrusted-input path a media tool hits). The
  four July advisories landed as post-release master commits, so an ``8.1.x``
  *release tag* can predate them.

This module is the single source of truth for "which bundled FFmpeg build is
acceptable". It parses the human-readable ``ffmpeg -version`` banner into a
structured provenance record, then grades that record against the per-CVE
matrix in :data:`SECURITY_ADVISORIES`. Each advisory carries its own upstream
fix commit, the date that commit landed on master, the last affected release,
and the ``configure`` tokens that decide whether this build even compiles the
vulnerable component.

A build is accepted only when every advisory resolves to one of:

* **fixed** — the build is a recorded post-fix snapshot, a git-master snapshot
  dated on/after that advisory's fix landed, or a release newer than the last
  affected one; or
* **not applicable** — the build's ``configure`` line proves the affected
  component is not compiled in.

Anything else fails closed, including an undated snapshot: absence of evidence
never waives an advisory. :data:`RELEASE_FLOOR` and :data:`SNAPSHOT_FLOOR_DATE`
remain as the documented summary of that matrix, and the exact bundled snapshot
(:data:`REFERENCE_GIT_COMMIT`) is recorded so the Windows release can be
reproduced from a named source archive.

The module is deliberately stdlib-only so it works inside a fresh
``pip install -e .`` and inside the ``scripts/verify_ffmpeg_provenance.py``
build-time gate.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Provenance floor — the bundled binary must clear ONE of these lanes.
# ---------------------------------------------------------------------------

# Lowest release that *could* clear the matrix, used for documentation and the
# install message. Acceptance itself is decided per advisory by
# SECURITY_ADVISORIES, never by this number alone.
#
# It stays on the 8.1 line because that is where a qualifying release would have
# to appear: 8.1.3 would be a later patch in the affected series, i.e. the
# advisories' own declared fix. Moving to 9.x instead of publishing it does not
# make 9.x qualify — see RELEASE_BRANCH_POINTS and RELEASE_LANE_OPEN. No such
# release exists today, which is why the lane is closed rather than merely unmet.
RELEASE_FLOOR: tuple[int, int, int] = (8, 1, 3)

# When each release series was branched from master, ISO ``YYYY-MM-DD``.
#
# This exists because "9.0.1 > 8.1.2, therefore 9.0.1 contains the 8.1.2 fixes"
# is false. The 9.0 branch was cut on 2026-06-26, *before* the July fix commits
# landed on master (2026-06-29 .. 2026-07-05), so a 9.0.x build carries those
# fixes only if they were backported onto the branch. A later release number is
# not evidence; the branch point is. A series absent from this table is treated
# as unknown, which is refused rather than assumed.
RELEASE_BRANCH_POINTS: dict[tuple[int, int], str] = {
    (9, 0): "2026-06-26",
}

# Whether any *published tagged release* currently clears the advisory matrix.
#
# It does not, and this states why in code rather than leaving a comment about a
# version that was never published:
#   * 8.1.2 (2026-06-17) closed the 8.1 line and is affected by the July batch.
#   * 8.1.3 was never released; upstream moved to 9.0.
#   * 9.0 (2026-08-04) and 9.0.1 (2026-08-12) branched on 2026-06-26, before the
#     July fix commits landed on master, and no backport onto that branch is on
#     record here — so they are refused rather than assumed.
# To open the lane: verify the fix commits are ancestors of the 9.0.x tag, add
# each release to the matching advisory's ``backported_in``, and flip this flag.
RELEASE_LANE_OPEN = False
RELEASE_LANE_CLOSED_REASON = (
    "No published FFmpeg release currently clears OpenCut's advisory matrix: "
    "8.1.2 is affected by the July 2026 batch, 8.1.3 was never published, and "
    "the 9.0 series branched on 2026-06-26 before those fixes landed on master "
    "with no backport on record. Use a dated git-master snapshot until a 9.0.x "
    "tag is shown to contain them."
)

# Minimum acceptable *git-master snapshot* date. The four July-2026 fixes
# landed on master by 2026-06-29; the first dated Gyan full build after the
# fixes is 2026-07-06. ISO ``YYYY-MM-DD``.
SNAPSHOT_FLOOR_DATE = "2026-07-06"

# The exact post-fix master snapshot used by the bundled Windows payload. Keep
# the full source SHA here even though Gyan's banner exposes a ten-character
# abbreviated commit.
REFERENCE_GIT_COMMIT = "01a25f74cc446a683318bab13dfd98a467082ef7"
REFERENCE_GIT_DATE = "2026-08-03"

# Upstream MagicYUV fix commits for CVE-2026-8461. They remain part of the
# complete provenance record even though the current snapshot is newer.
MAGICYUV_FIX_COMMITS: tuple[str, ...] = (
    "374b726ffa878ee1cadb987bd1e1e20cc7ed8845",
    "5806e8b9f34f1b0663b3017ef9dd1aa5d08116d1",
)

# The four July-2026 crafted-media fixes that are not present in the 8.1.2
# release branch. These are intentionally named in every failed-release
# message and every generated provenance record.
JULY_2026_CVES: tuple[str, ...] = (
    "CVE-2026-64832",  # NVDEC double-free
    "CVE-2026-64833",  # S/PDIF DTS out-of-bounds read
    "CVE-2026-64835",  # ADX out-of-bounds access
    "CVE-2026-66041",  # vf_quirc heap out-of-bounds write
    # Graded 2026-08-20 (F348). Kept in the same order as SECURITY_ADVISORIES;
    # test_every_graded_advisory_is_reported_in_the_public_constants fails if
    # this list and the matrix drift apart.
    "CVE-2026-64830",  # VobSub demuxer heap overflow
    "CVE-2026-64831",  # Vulkan HEVC hwaccel stack overflow
    "CVE-2026-64834",  # RTP/ASF depacketizer
    "CVE-2026-65703",  # TDSC decoder
    "CVE-2026-65704",  # TY demuxer
    "CVE-2026-65705",  # vf_floodfill
    "CVE-2026-65706",  # vf_swaprect out-of-bounds write
    "CVE-2026-66036",  # vf_hqdn3d out-of-bounds write
    "CVE-2026-66037",  # IAMF demuxer resource consumption
    "CVE-2026-66038",  # LCL/ZLIB decoder heap disclosure
    "CVE-2026-66039",  # MACE6 decoder
    "CVE-2026-66040",  # native PNG/APNG encoder heap OOB write
)
JULY_2026_FIX_COMMITS: tuple[str, ...] = (
    "4c6217477fc64305055b37d9d1d0d76d30e37f97",
    "6f80e2765492700622596af720534cef33dd31b4",
    "1836ef96846937a6cc2443698a693104f5c0b21e",
    "4da9812e25894fb51d62a8875cfa8eb39b5e20f5",
    "dbd495f066a85ba96b17433f4306582aa37c3951",
    "92737390dc133daadce47dd7d2ec8ef3d9ebcbed",
    "11d5f475be95d22d5f0692220cc772b116abc632",
    "fd3ee52fab34d98a95b787d0b5ff45685766200c",
    "de771bd52774a52d45b0e2c82e56995a1ef40df7",
    "f186c50cf53aec20e9a29059cb22ca3f2d59201c",
    "a7e38b617b32f996beaa371bbf04b39907d7a527",
    "5d7112c60e6f0f0742ce47d448e6da0718a70f4c",
    "86708357d126af84c16f80d9c57335d1e8c845c5",
    "e7cbfd1c507b57a806a5825b87d609963e862c8c",
    "aafb5c655edc76a753275c383ebb139feb032718",
    "b506fafec9a19fcbc2be5271875fd4a63d6615bc",
)


@dataclass(frozen=True)
class CveAdvisory:
    """One advisory, the commit that fixed it, and the component it needs.

    ``capability_tokens`` are substrings of the build's ``configure`` line. A
    build that enables none of them cannot reach the vulnerable code at all,
    which is the "prove the component absent" acceptance path — a date alone
    can never establish that.
    """

    cve: str
    component: str
    fix_commit: str
    #: Date the fix landed on master. Master is linear, so a snapshot built
    #: on/after this date contains it.
    fix_landed: str
    #: Releases at or below this are affected; ``None`` means every release.
    affected_max_release: Optional[tuple[int, int, int]]
    capability_tokens: tuple[str, ...]
    #: Releases from a *newer* series that are known to carry this fix as a
    #: backport. Needed because a newer series branched before the fix landed
    #: cannot be assumed to contain it. Empty means no backport is on record.
    backported_in: tuple[tuple[int, int, int], ...] = ()


#: Per-CVE acceptance matrix. Replaces the single global snapshot-date
#: heuristic: each advisory is graded against its own fix commit, its own
#: landing date, and whether this build even compiles the affected component.
SECURITY_ADVISORIES: tuple[CveAdvisory, ...] = (
    CveAdvisory(
        cve="CVE-2026-64832",
        component="NVDEC hardware decoder (libavcodec/nvdec.c)",
        fix_commit="4c6217477fc64305055b37d9d1d0d76d30e37f97",
        fix_landed="2026-07-04",
        affected_max_release=(8, 1, 2),
        capability_tokens=("--enable-nvdec", "--enable-cuvid", "--enable-ffnvcodec"),
    ),
    CveAdvisory(
        cve="CVE-2026-64833",
        component="S/PDIF DTS demuxer",
        fix_commit="6f80e2765492700622596af720534cef33dd31b4",
        fix_landed="2026-06-29",
        affected_max_release=(8, 1, 2),
        # spdif is built unless explicitly disabled, so an explicit disable is
        # the only way to prove absence.
        capability_tokens=("!--disable-demuxer=spdif",),
    ),
    CveAdvisory(
        cve="CVE-2026-64835",
        component="ADX/AAX decoder",
        fix_commit="1836ef96846937a6cc2443698a693104f5c0b21e",
        fix_landed="2026-06-30",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-decoder=adpcm_adx",),
    ),
    CveAdvisory(
        cve="CVE-2026-66041",
        component="quirc QR filter (vf_quirc)",
        fix_commit="4da9812e25894fb51d62a8875cfa8eb39b5e20f5",
        fix_landed="2026-07-05",
        affected_max_release=(8, 1, 2),
        capability_tokens=("--enable-libquirc",),
    ),
    CveAdvisory(
        cve="CVE-2026-8461",
        component="MagicYUV decoder",
        fix_commit="374b726ffa878ee1cadb987bd1e1e20cc7ed8845",
        fix_landed="2026-06-12",
        affected_max_release=(8, 1, 1),
        capability_tokens=("!--disable-decoder=magicyuv",),
    ),
    # Graded 2026-08-20 (F348). Each fix commit below is the one NVD
    # references for that CVE, confirmed against the upstream repository to
    # exist and to touch the source file the advisory names; `fix_landed` is
    # that commit's committer date on master.
    CveAdvisory(
        cve="CVE-2026-64830",
        component="VobSub subtitle demuxer (libavformat/mpeg.c)",
        fix_commit="dbd495f066a85ba96b17433f4306582aa37c3951",
        fix_landed="2026-07-02",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-demuxer=vobsub",),
    ),
    CveAdvisory(
        cve="CVE-2026-64831",
        component="Vulkan HEVC hardware decoder (libavcodec/vulkan_hevc.c)",
        fix_commit="92737390dc133daadce47dd7d2ec8ef3d9ebcbed",
        fix_landed="2026-07-05",
        affected_max_release=(8, 1, 2),
        # Vulkan hwaccel is opt-in, so its absence from the configure line is
        # itself proof the vulnerable path is not compiled in.
        capability_tokens=("--enable-vulkan",),
    ),
    CveAdvisory(
        cve="CVE-2026-64834",
        component="RTP/ASF depacketizer (libavformat/rtpdec_asf.c)",
        fix_commit="11d5f475be95d22d5f0692220cc772b116abc632",
        fix_landed="2026-07-02",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-demuxer=rtsp",),
    ),
    CveAdvisory(
        cve="CVE-2026-65703",
        component="TDSC video decoder (libavcodec/tdsc.c)",
        fix_commit="fd3ee52fab34d98a95b787d0b5ff45685766200c",
        fix_landed="2026-07-13",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-decoder=tdsc",),
    ),
    CveAdvisory(
        cve="CVE-2026-65704",
        component="TY demuxer (libavformat/ty.c)",
        fix_commit="de771bd52774a52d45b0e2c82e56995a1ef40df7",
        fix_landed="2026-07-13",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-demuxer=ty",),
    ),
    CveAdvisory(
        cve="CVE-2026-65705",
        component="floodfill video filter (libavfilter/vf_floodfill.c)",
        fix_commit="f186c50cf53aec20e9a29059cb22ca3f2d59201c",
        fix_landed="2026-07-13",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-filter=floodfill",),
    ),
    CveAdvisory(
        cve="CVE-2026-65706",
        component="swaprect video filter (libavfilter/vf_swaprect.c)",
        fix_commit="a7e38b617b32f996beaa371bbf04b39907d7a527",
        fix_landed="2026-07-13",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-filter=swaprect",),
    ),
    CveAdvisory(
        cve="CVE-2026-66036",
        component="hqdn3d video filter (libavfilter/vf_hqdn3d.c)",
        fix_commit="5d7112c60e6f0f0742ce47d448e6da0718a70f4c",
        fix_landed="2026-07-22",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-filter=hqdn3d",),
    ),
    CveAdvisory(
        cve="CVE-2026-66037",
        component="IAMF demuxer (libavformat/iamf_parse.c)",
        fix_commit="86708357d126af84c16f80d9c57335d1e8c845c5",
        fix_landed="2026-06-28",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-demuxer=iamf",),
    ),
    CveAdvisory(
        cve="CVE-2026-66038",
        component="LCL/ZLIB video decoder (libavcodec/lcldec.c)",
        fix_commit="e7cbfd1c507b57a806a5825b87d609963e862c8c",
        fix_landed="2026-07-05",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-decoder=zlib",),
    ),
    CveAdvisory(
        cve="CVE-2026-66039",
        component="MACE6 audio decoder (libavcodec/mace.c)",
        fix_commit="aafb5c655edc76a753275c383ebb139feb032718",
        fix_landed="2026-07-03",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-decoder=mace6",),
    ),
    CveAdvisory(
        cve="CVE-2026-66040",
        component="native PNG/APNG encoders (libavcodec/pngenc.c)",
        fix_commit="b506fafec9a19fcbc2be5271875fd4a63d6615bc",
        fix_landed="2026-07-21",
        affected_max_release=(8, 1, 2),
        capability_tokens=("!--disable-encoder=png",),
    ),
)

#: Advisories from the same 2026-07-22..24 disclosure batch that are published
#: against "FFmpeg through 8.1.2" but are **not** graded above, because no
#: upstream fix commit has been recorded for them here yet. Grading requires a
#: specific commit whose ancestry can be checked; a plausible-looking hash would
#: make the gate lie in the most dangerous direction. They are enumerated so the
#: report can say what it has *not* checked — a clean verdict over a subset must
#: never read as complete coverage.
#:
#: To graduate one: find the upstream fix commit and the date it landed on
#: master, move it into SECURITY_ADVISORIES with its component and capability
#: tokens, and delete it here. `tests/test_ffmpeg_cve_matrix.py` enforces that
#: the two sets stay disjoint and that every graded entry carries a commit.
#: July 2026 advisories known to exist but not yet graded against a fix commit.
#: Emptied 2026-08-20 (F348) once every entry had an upstream fix commit
#: verified to exist and to touch the component its CVE names. Anything added
#: here must carry the same evidence before it moves into SECURITY_ADVISORIES —
#: a plausible-but-wrong hash makes the gate lie in the dangerous direction.
UNGRADED_ADVISORIES: tuple[str, ...] = ()


def advisory_coverage() -> dict:
    """What the matrix grades, and what it explicitly does not.

    Consumers use this to state scope instead of implying completeness.
    """
    graded = [adv.cve for adv in SECURITY_ADVISORIES]
    return {
        "graded": graded,
        "graded_count": len(graded),
        "ungraded": list(UNGRADED_ADVISORIES),
        "ungraded_count": len(UNGRADED_ADVISORIES),
        "total_known": len(graded) + len(UNGRADED_ADVISORIES),
        "complete": not UNGRADED_ADVISORIES,
    }


# The human-readable version string and exact redistribution inputs the
# installers pin. Kept here so Python, C#, Inno, Docker, and release metadata
# all agree on one full snapshot.
PINNED_INSTALLER_VERSION = "2026-08-03-git-01a25f74cc-full_build-www.gyan.dev"
PINNED_INSTALLER_COMMIT = REFERENCE_GIT_COMMIT
PINNED_INSTALLER_ARCHIVE_URL = (
    "https://www.gyan.dev/ffmpeg/builds/packages/"
    "ffmpeg-2026-08-03-git-01a25f74cc-full_build.7z"
)
PINNED_INSTALLER_ARCHIVE_SHA256 = (
    "8c32ed9800ff421bbcfda96beb0a66783a64a7cd98869b87ec1b494d3c855fcc"
)
PINNED_INSTALLER_SOURCE_URL = (
    "https://github.com/FFmpeg/FFmpeg/archive/01a25f74cc446a683318bab13dfd98a467082ef7.tar.gz"
)
PINNED_INSTALLER_SOURCE_SHA256 = (
    "02f09346860e4b0549eb03003443c66dceb9f355c2db4f01746db33984f1e3cf"
)

# The June-2026 FFmpeg advisories this floor also carries forward.
JUNE_2026_CVES: tuple[str, ...] = (
    "CVE-2026-8461",  # MagicYUV OOB write; releases before 8.1.2
    "CVE-2026-6385",  # GHSA-q22x-99q7-fr6w, CVSS 6.5 (confirmed)
    "CVE-2026-39210",
    "CVE-2026-39211",
    "CVE-2026-39212",
    "CVE-2026-39213",
    "CVE-2026-39214",
    "CVE-2026-39215",
    "CVE-2026-39216",
    "CVE-2026-39217",
    "CVE-2026-39218",
)

# Keep the old name as a compatibility surface for callers that only need the
# original June advisory set. New records use the complete combined tuples.
SECURITY_CVES: tuple[str, ...] = JUNE_2026_CVES + JULY_2026_CVES
SECURITY_FIX_COMMITS: tuple[str, ...] = MAGICYUV_FIX_COMMITS + JULY_2026_FIX_COMMITS

# ``ffmpeg version <token> ...`` — capture the build token (release or git).
_VERSION_RE = re.compile(r"version\s+([^\s]+)")
# gyan.dev git snapshots embed an ISO date and the ``git-<hash>`` commit:
#   2026-08-03-git-01a25f74cc-full_build-www.gyan.dev
# BtbN / native ``git describe`` snapshots use ``N-<rev>-g<hash>`` (no date):
#   N-118000-gabcdef1234
_SNAPSHOT_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_GIT_COMMIT_RE = re.compile(r"git-([0-9a-f]{7,40})", re.IGNORECASE)
_BTBN_DESCRIBE_RE = re.compile(r"^N-\d+-g([0-9a-f]{7,40})", re.IGNORECASE)
# Leading release number, tolerating a ``n`` prefix (distro builds: ``n8.1.1``).
_RELEASE_RE = re.compile(r"^n?(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version_banner(banner: str) -> dict:
    """Parse the first line of ``ffmpeg -version`` into a provenance record.

    Returns a dict with ``raw`` (the build token), ``release`` (``(maj, min,
    patch)`` tuple or ``None``), ``is_git_snapshot``, ``snapshot_date`` (ISO
    string or ``None``), ``git_commit`` (or ``None``), ``flavor`` (``essentials``/
    ``full``/...), and ``builder`` (``gyan.dev``/``BtbN``/...). Never raises —
    a banner it cannot parse returns all-``None`` so callers can grade it as
    "unknown" rather than crashing.
    """
    record: dict = {
        "raw": "",
        "release": None,
        "is_git_snapshot": False,
        "snapshot_date": None,
        "git_commit": None,
        "flavor": "",
        "builder": "",
    }
    if not banner:
        return record

    first_line = banner.splitlines()[0] if banner.splitlines() else banner
    m = _VERSION_RE.search(first_line)
    if not m:
        return record
    token = m.group(1)
    record["raw"] = token

    commit_m = _GIT_COMMIT_RE.search(token)
    if commit_m:
        record["git_commit"] = commit_m.group(1).lower()
        record["is_git_snapshot"] = True
    else:
        btbn_m = _BTBN_DESCRIBE_RE.match(token)
        if btbn_m:
            # BtbN / ``git describe`` ``N-<rev>-g<hash>``: git snapshot, no date.
            record["git_commit"] = btbn_m.group(1).lower()
            record["is_git_snapshot"] = True

    # Source builds may add the provenance token after an ``N-<rev>``
    # fallback version, so search rather than requiring the date at position 0.
    date_m = _SNAPSHOT_DATE_RE.search(token)
    if date_m:
        record["snapshot_date"] = f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}"
        record["is_git_snapshot"] = True

    if not record["is_git_snapshot"]:
        rel_m = _RELEASE_RE.match(token)
        if rel_m:
            record["release"] = (
                int(rel_m.group(1)),
                int(rel_m.group(2)),
                int(rel_m.group(3) or 0),
            )

    low = token.lower()
    if "essentials" in low:
        record["flavor"] = "essentials"
    elif "full" in low:
        record["flavor"] = "full"
    if "gyan" in low:
        record["builder"] = "gyan.dev"
    elif "btbn" in low:
        record["builder"] = "BtbN"

    return record


#: Commits known to descend from every recorded fix. A build reporting one of
#: these is accepted outright — this is how the pinned bundled snapshot passes
#: without re-deriving ancestry offline.
KNOWN_GOOD_COMMITS: tuple[str, ...] = (REFERENCE_GIT_COMMIT,)


def _advisory_applies(advisory: CveAdvisory, configure_line: str) -> Optional[bool]:
    """Is *advisory*'s component compiled into this build?

    Returns ``True``/``False`` when the configure line settles it, and ``None``
    when no configure line is available — an unknown component must be treated
    as present, never as absent.
    """
    if not configure_line:
        return None
    for token in advisory.capability_tokens:
        if token.startswith("!"):
            # Built by default: only an explicit disable proves absence.
            if token[1:] in configure_line:
                return False
        elif token in configure_line:
            return True
    # Every token was an opt-in enable and none appeared -> not compiled in.
    if all(not tok.startswith("!") for tok in advisory.capability_tokens):
        return False
    return True


def _series(version: tuple[int, int, int]) -> tuple[int, int]:
    return (version[0], version[1])


def _grade_newer_release(
    advisory: CveAdvisory,
    release: tuple[int, int, int],
) -> tuple[bool, str]:
    """Decide whether a release *numerically newer* than the affected one is fixed.

    Within the same series a later patch is the advisory's own declared fix, so
    ordering settles it. Across a series boundary it does not: a newer series
    branched from master before the fix landed does not contain it, and shipping
    later says nothing about a backport. That case needs the branch point or
    explicit backport evidence, and is refused without either.
    """
    version = ".".join(map(str, release))
    affected = advisory.affected_max_release
    if affected is None:
        return True, f"release {version} post-dates an advisory with no affected ceiling"

    if _series(release) <= _series(affected):
        return True, (
            f"release {version} is a later patch in the {_series(affected)[0]}."
            f"{_series(affected)[1]} series than the last affected release "
            f"{'.'.join(map(str, affected))}, which is the advisory's declared fix"
        )

    if release in advisory.backported_in:
        return True, (
            f"release {version} is recorded as carrying a backport of "
            f"{advisory.fix_commit[:10]}"
        )

    branch_point = RELEASE_BRANCH_POINTS.get(_series(release))
    if branch_point and branch_point >= advisory.fix_landed:
        return True, (
            f"the {_series(release)[0]}.{_series(release)[1]} series branched on "
            f"{branch_point}, at/after {advisory.fix_landed} when "
            f"{advisory.fix_commit[:10]} landed on master"
        )

    if branch_point:
        return False, (
            f"release {version} is numerically newer, but the "
            f"{_series(release)[0]}.{_series(release)[1]} series branched on "
            f"{branch_point}, before {advisory.fix_landed} when "
            f"{advisory.fix_commit[:10]} landed on master, and no backport is on "
            f"record — a higher version number is not evidence the fix is present"
        )
    return False, (
        f"release {version} belongs to the "
        f"{_series(release)[0]}.{_series(release)[1]} series, whose branch point "
        f"is not recorded, so it cannot be shown to contain {advisory.fix_commit[:10]}"
    )


def grade_advisory(
    advisory: CveAdvisory,
    rec: dict,
    configure_line: str = "",
) -> dict:
    """Grade one advisory against a parsed provenance record.

    Status is ``fixed``, ``not_applicable`` (component absent from this build),
    or ``vulnerable``.
    """
    entry = {
        "cve": advisory.cve,
        "component": advisory.component,
        "fix_commit": advisory.fix_commit,
        "fix_landed": advisory.fix_landed,
        "status": "vulnerable",
        "reason": "",
    }

    commit = str(rec.get("git_commit") or "").lower()
    if commit and any(
        known.lower().startswith(commit) or commit.startswith(known.lower())
        for known in KNOWN_GOOD_COMMITS
    ):
        entry["status"] = "fixed"
        entry["reason"] = f"build commit {commit[:10]} is a recorded post-fix snapshot"
        return entry

    if rec.get("is_git_snapshot"):
        snapshot_date = rec.get("snapshot_date")
        if snapshot_date and snapshot_date >= advisory.fix_landed:
            entry["status"] = "fixed"
            entry["reason"] = (
                f"git-master snapshot {snapshot_date} is at/after {advisory.fix_landed}, "
                f"when {advisory.fix_commit[:10]} landed"
            )
            return entry
        detail = (
            f"git-master snapshot {snapshot_date} predates {advisory.fix_landed}"
            if snapshot_date
            else "git snapshot carries no date, so the fix cannot be confirmed"
        )
    elif rec.get("release"):
        release = rec["release"]
        if advisory.affected_max_release is None or release > advisory.affected_max_release:
            verdict, why = _grade_newer_release(advisory, release)
            if verdict:
                entry["status"] = "fixed"
                entry["reason"] = why
                return entry
            detail = why
        else:
            detail = (
                f"release {'.'.join(map(str, release))} is at/below the last affected "
                f"release {'.'.join(map(str, advisory.affected_max_release))}"
            )
    else:
        detail = "build token could not be classified as a release or a snapshot"

    applies = _advisory_applies(advisory, configure_line)
    if applies is False:
        entry["status"] = "not_applicable"
        entry["reason"] = f"{detail}, but {advisory.component} is not compiled into this build"
        return entry

    entry["reason"] = detail if applies is None else f"{detail}; {advisory.component} is enabled"
    return entry


def grade_advisories(rec: dict, configure_line: str = "") -> list[dict]:
    """Grade every recorded advisory against a parsed provenance record."""
    return [grade_advisory(adv, rec, configure_line) for adv in SECURITY_ADVISORIES]


def probe_build_configuration(ffmpeg_bin: str, timeout: float = 8.0) -> str:
    """Return the build's ``configure`` line, or ``""`` when unavailable."""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-buildconf"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("ffmpeg_provenance: buildconf probe failed: %s", exc)
        return ""
    return (result.stdout or "") + (result.stderr or "")


def _matrix_reason(
    rec: dict,
    advisories: list[dict],
    unresolved: list[str],
    waived: list[str],
) -> str:
    """Human-readable verdict naming the advisories that decided it."""
    if rec.get("is_git_snapshot"):
        build = f"git-master snapshot {rec.get('snapshot_date') or rec.get('git_commit') or '?'}"
    elif rec.get("release"):
        build = f"release {'.'.join(map(str, rec['release']))}"
    else:
        build = f"build {rec.get('raw') or '?'}"

    if unresolved:
        blocking = next(a for a in advisories if a["status"] == "vulnerable")
        extra = f" (+{len(unresolved) - 1} more)" if len(unresolved) > 1 else ""
        return f"{build} does not clear {blocking['cve']}{extra}: {blocking['reason']}"

    parts = [f"{build} clears all {len(advisories)} graded advisories"]
    if waived:
        parts.append(f"{len(waived)} not applicable to this build ({', '.join(waived)})")
    if UNGRADED_ADVISORIES:
        # Say what was not checked. "Clears everything graded" and "is clean"
        # are different claims, and only the first one is supported here.
        parts.append(
            f"{len(UNGRADED_ADVISORIES)} advisories from the same batch are not yet "
            f"graded and were not checked ({', '.join(UNGRADED_ADVISORIES)})"
        )
    return "; ".join(parts)


def check_security_floor(banner: str, configure_line: str = "") -> dict:
    """Grade an ``ffmpeg -version`` banner against the per-CVE advisory matrix.

    ``ok`` is ``True`` only when every advisory in :data:`SECURITY_ADVISORIES`
    is either ``fixed`` (the build demonstrably carries its fix commit) or
    ``not_applicable`` (the affected component is not compiled in). Pass
    *configure_line* to enable the second test. The grading never raises.
    """
    rec = parse_version_banner(banner)
    result: dict = {
        "ok": False,
        "lane": "unknown",
        "version": rec["raw"],
        "snapshot_date": rec["snapshot_date"],
        "git_commit": rec["git_commit"],
        "flavor": rec["flavor"],
        "builder": rec["builder"],
        "reason": "",
        "cves": list(SECURITY_CVES),
        "fix_commits": list(SECURITY_FIX_COMMITS),
        # `ok` means "clears everything this matrix grades", which is not the
        # same as "clean". Coverage travels with the verdict so no consumer has
        # to infer scope.
        "coverage": advisory_coverage(),
        "ungraded_cves": list(UNGRADED_ADVISORIES),
    }

    if not rec["raw"]:
        result["reason"] = "could not parse an ffmpeg version banner"
        result["advisories"] = []
        result["unresolved_cves"] = [adv.cve for adv in SECURITY_ADVISORIES]
        return result

    # The per-CVE matrix decides acceptance. Lane/date reporting below is
    # retained for the human-readable reason and for existing consumers.
    advisories = grade_advisories(rec, configure_line)
    unresolved = [a["cve"] for a in advisories if a["status"] == "vulnerable"]
    waived = [a["cve"] for a in advisories if a["status"] == "not_applicable"]
    result["advisories"] = advisories
    result["unresolved_cves"] = unresolved
    result["not_applicable_cves"] = waived

    if rec["is_git_snapshot"]:
        result["lane"] = "snapshot"
        result["ok"] = not unresolved
        result["reason"] = _matrix_reason(rec, advisories, unresolved, waived)
        return result

    if rec["release"]:
        result["lane"] = "release"
        result["ok"] = not unresolved
        result["reason"] = _matrix_reason(rec, advisories, unresolved, waived)
        return result

    result["reason"] = f"unrecognised ffmpeg build token {rec['raw']!r}"
    return result


def provenance_record(banner: Optional[str] = None) -> dict:
    """Full provenance dict suitable for a release manifest / capability probe.

    When ``banner`` is ``None`` the bundled ffmpeg is resolved and probed. The
    record always carries the declared floor (so a manifest documents the
    requirement even when no binary is present) plus the graded result of any
    binary that *is* present.
    """
    record: dict = {
        "required_release_floor": ".".join(map(str, RELEASE_FLOOR)),
        "required_snapshot_floor_date": SNAPSHOT_FLOOR_DATE,
        "reference_git_commit": REFERENCE_GIT_COMMIT,
        "reference_git_date": REFERENCE_GIT_DATE,
        "required_fix_commits": list(SECURITY_FIX_COMMITS),
        "pinned_installer_version": PINNED_INSTALLER_VERSION,
        "pinned_installer_commit": PINNED_INSTALLER_COMMIT,
        "pinned_installer_archive": {
            "url": PINNED_INSTALLER_ARCHIVE_URL,
            "sha256": PINNED_INSTALLER_ARCHIVE_SHA256,
        },
        "pinned_source": {
            "url": PINNED_INSTALLER_SOURCE_URL,
            "sha256": PINNED_INSTALLER_SOURCE_SHA256,
        },
        "advisory_matrix": [
            {
                "cve": adv.cve,
                "component": adv.component,
                "fix_commit": adv.fix_commit,
                "fix_landed": adv.fix_landed,
                "affected_max_release": (
                    ".".join(map(str, adv.affected_max_release))
                    if adv.affected_max_release
                    else None
                ),
            }
            for adv in SECURITY_ADVISORIES
        ],
        "cves_addressed": list(SECURITY_CVES),
        "release_lane_open": RELEASE_LANE_OPEN,
        "release_lane_note": "" if RELEASE_LANE_OPEN else RELEASE_LANE_CLOSED_REASON,
        "release_branch_points": {
            f"{major}.{minor}": date
            for (major, minor), date in sorted(RELEASE_BRANCH_POINTS.items())
        },
        "bundled": None,
    }
    if banner is None:
        banner = _probe_bundled_banner()
    if banner:
        record["bundled"] = check_security_floor(banner)
    return record


def _resolve_ffmpeg_bin() -> Optional[str]:
    # Do not call helpers.get_ffmpeg_path() here: that resolver invokes this
    # module's fail-closed validator and would recurse.
    return shutil.which("ffmpeg")


class FfmpegSecurityError(RuntimeError):
    """Raised before media processing when FFmpeg does not clear the floor."""

    code = "FFMPEG_SECURITY_FLOOR"

    def __init__(self, binary: str, grade: dict):
        self.binary = binary
        self.grade = grade
        unresolved = grade.get("unresolved_cves") or list(JULY_2026_CVES)
        super().__init__(
            f"FFmpeg is unavailable because {binary!r} does not clear "
            f"{', '.join(unresolved)}: "
            f"{grade.get('reason') or 'version could not be verified'}. "
            + (
                f"Install FFmpeg {'.'.join(map(str, RELEASE_FLOOR))}+, a git-master "
                f"snapshot dated >= {SNAPSHOT_FLOOR_DATE}, or a build compiled "
                f"without the affected components before processing media."
                if RELEASE_LANE_OPEN
                else (
                    f"Install a git-master snapshot dated >= {SNAPSHOT_FLOOR_DATE}, "
                    f"or a build compiled without the affected components, before "
                    f"processing media. {RELEASE_LANE_CLOSED_REASON}"
                )
            )
        )


def is_pinned_snapshot(grade: dict) -> bool:
    """Return whether a passing snapshot matches the bundled source commit."""
    actual = str(grade.get("git_commit") or "").lower()
    expected = PINNED_INSTALLER_COMMIT.lower()
    return bool(
        grade.get("ok")
        and grade.get("lane") == "snapshot"
        and actual
        and (expected.startswith(actual) or actual.startswith(expected))
    )


def probe_binary_security(ffmpeg_bin: str, timeout: float = 8.0) -> dict:
    """Run ``-version`` for one binary and return its security grade.

    When the version alone does not clear every advisory, the build's
    ``configure`` line is probed as well so a build that simply does not
    compile the affected component can still be accepted. That second probe is
    deliberately lazy: the common case costs one subprocess call.
    """
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        grade = check_security_floor("")
        grade["reason"] = f"could not execute the media binary: {exc}"
        return grade

    banner = result.stdout or result.stderr or ""
    grade = check_security_floor(banner)
    if not grade.get("ok") and grade.get("unresolved_cves"):
        configure_line = probe_build_configuration(ffmpeg_bin, timeout=timeout)
        if configure_line:
            grade = check_security_floor(banner, configure_line=configure_line)
    if result.returncode != 0 and grade.get("ok"):
        grade["ok"] = False
        grade["reason"] = f"version probe exited with status {result.returncode}"
    return grade


def require_security_floor(ffmpeg_bin: str) -> dict:
    """Return a verified grade or raise :class:`FfmpegSecurityError`."""
    grade = probe_binary_security(ffmpeg_bin)
    if not grade.get("ok"):
        raise FfmpegSecurityError(ffmpeg_bin, grade)
    return grade


def _probe_bundled_banner() -> str:
    ffmpeg_bin = _resolve_ffmpeg_bin()
    if not ffmpeg_bin:
        return ""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("ffmpeg_provenance: probe failed: %s", exc)
        return ""
    return result.stdout or ""
