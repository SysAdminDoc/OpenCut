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
structured provenance record and grades it against two acceptance lanes:

* **release lane** — a tagged release ``>= 8.1.3``. The 8.1.2 release is
  rejected until upstream publishes a point release containing the four July
  2026 crafted-media fixes.
* **snapshot lane** — a gyan.dev / BtbN git-master snapshot dated on/after
  :data:`SNAPSHOT_FLOOR_DATE`, the guaranteed-clean lane that demonstrably
  carries the July-2026 commits. The exact bundled snapshot
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
from typing import Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Provenance floor — the bundled binary must clear ONE of these lanes.
# ---------------------------------------------------------------------------

# Minimum acceptable *release* version. The four July-2026 advisories affect
# 8.1.2, so the release lane remains closed until upstream publishes 8.1.3.
RELEASE_FLOOR: tuple[int, int, int] = (8, 1, 3)

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
)
JULY_2026_FIX_COMMITS: tuple[str, ...] = (
    "4c6217477fc64305055b37d9d1d0d76d30e37f97",
    "6f80e2765492700622596af720534cef33dd31b4",
    "1836ef96846937a6cc2443698a693104f5c0b21e",
    "4da9812e25894fb51d62a8875cfa8eb39b5e20f5",
)

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


def check_security_floor(banner: str) -> dict:
    """Grade an ``ffmpeg -version`` banner against the security floor.

    Returns ``{ok, lane, version, snapshot_date, git_commit, reason, cves}``.
    ``ok`` is ``True`` only when the build clears the release lane (``>= 8.1.3``)
    or the snapshot lane (git-master dated ``>= SNAPSHOT_FLOOR_DATE``). The
    grading never raises.
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
    }

    if not rec["raw"]:
        result["reason"] = "could not parse an ffmpeg version banner"
        return result

    if rec["is_git_snapshot"]:
        result["lane"] = "snapshot"
        if rec["snapshot_date"]:
            if rec["snapshot_date"] >= SNAPSHOT_FLOOR_DATE:
                result["ok"] = True
                result["reason"] = (
                    f"git-master snapshot {rec['snapshot_date']} is at/after the "
                    f"post-fix floor {SNAPSHOT_FLOOR_DATE} for "
                    f"{', '.join(JULY_2026_CVES)}"
                )
            else:
                result["reason"] = (
                    f"git-master snapshot {rec['snapshot_date']} predates the "
                    f"post-fix floor {SNAPSHOT_FLOOR_DATE} for "
                    f"{', '.join(JULY_2026_CVES)}"
                )
        else:
            result["reason"] = (
                "git snapshot has no embedded date; cannot confirm it carries "
                f"the July-2026 fixes ({', '.join(JULY_2026_CVES)}) — rebuild from "
                f"a snapshot >= {SNAPSHOT_FLOOR_DATE}"
            )
        return result

    if rec["release"]:
        result["lane"] = "release"
        if rec["release"] >= RELEASE_FLOOR:
            result["ok"] = True
            result["reason"] = (
                f"release {'.'.join(map(str, rec['release']))} is at/after the "
                f"{'.'.join(map(str, RELEASE_FLOOR))} security floor; the post-fix "
                f"snapshot fallback is {REFERENCE_GIT_COMMIT[:10]} dated "
                f"{REFERENCE_GIT_DATE}"
            )
        else:
            result["reason"] = (
                f"release {'.'.join(map(str, rec['release']))} predates the "
                f"{'.'.join(map(str, RELEASE_FLOOR))} security floor for "
                f"{', '.join(JULY_2026_CVES)}"
            )
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
        "cves_addressed": list(SECURITY_CVES),
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
        super().__init__(
            f"FFmpeg is unavailable because {binary!r} does not clear the "
            f"{'.'.join(map(str, RELEASE_FLOOR))} security floor for "
            f"{', '.join(JULY_2026_CVES)}: "
            f"{grade.get('reason') or 'version could not be verified'}. "
            f"Install FFmpeg {'.'.join(map(str, RELEASE_FLOOR))}+ or a dated "
            f"post-fix snapshot >= {SNAPSHOT_FLOOR_DATE} before processing media."
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
    """Run ``-version`` for one binary and return its security grade."""
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
