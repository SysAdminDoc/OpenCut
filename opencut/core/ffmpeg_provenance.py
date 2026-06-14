"""Bundled-FFmpeg version + security-patch provenance (RA-FFMPEG-PROV).

The roadmap carries two related FFmpeg items:

* bump the bundled binary from 8.0.x to 8.1.x (for the D3D12VA/Vulkan encoder
  routes that the rest of the codebase already detects), and
* assert a *security patch level*, not just a version string — the June-2026
  automated FFmpeg audit disclosed ~21 zero-days (CVE-2026-6385 confirmed,
  CVE-2026-39210..39218 reserved), several heap/stack overflows reachable via
  crafted media (the first untrusted-input path a media tool hits). Those fixes
  landed as post-release master commits, so an ``8.1.x`` *release tag* can
  predate them.

This module is the single source of truth for "which bundled FFmpeg build is
acceptable". It parses the human-readable ``ffmpeg -version`` banner into a
structured provenance record and grades it against two acceptance lanes:

* **release lane** — a tagged release ``>= 8.1.1`` (gyan.dev's current stable
  point release; point releases carry backported security fixes). Reports
  ``8.1.x`` so the encoder-route bump is satisfied.
* **snapshot lane** — a gyan.dev / BtbN git-master snapshot dated on/after
  :data:`SNAPSHOT_FLOOR_DATE`, the guaranteed-clean lane that demonstrably
  carries the June-2026 commits. The reference snapshot
  (:data:`REFERENCE_GIT_COMMIT`) is recorded so a release that later proves to
  predate a specific fix has a concrete fallback.

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

# Minimum acceptable *release* version. gyan.dev currently ships 8.1.1; its
# point releases backport the FFmpeg security branch. Satisfies the "bundle
# 8.1.x" roadmap item.
RELEASE_FLOOR: tuple[int, int, int] = (8, 1, 1)

# Minimum acceptable *git-master snapshot* date. The June-2026 fixes landed as
# post-release master commits; gyan.dev's git-full snapshot from this date is
# the first build that demonstrably carries them. ISO ``YYYY-MM-DD``.
SNAPSHOT_FLOOR_DATE = "2026-06-10"

# The reference post-fix master snapshot (gyan.dev ``git-full`` build) recorded
# so the provenance pins a concrete commit hash, not merely a version string —
# this is what makes the security claim auditable. If a tagged release is ever
# found to predate a specific June-2026 fix, the build falls back to a snapshot
# at or after this commit.
REFERENCE_GIT_COMMIT = "b29bdd3715"
REFERENCE_GIT_DATE = "2026-06-10"

# The human-readable version string the installers pin (see AppConstants.cs /
# OpenCut.iss). Kept here so the Python side and the C#/Inno side agree.
PINNED_INSTALLER_VERSION = "8.1.1-essentials_build-www.gyan.dev"

# The June-2026 FFmpeg advisories this floor exists to clear.
JUNE_2026_CVES: tuple[str, ...] = (
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

# ``ffmpeg version <token> ...`` — capture the build token (release or git).
_VERSION_RE = re.compile(r"version\s+([^\s]+)")
# gyan.dev git snapshots embed an ISO date and the ``git-<hash>`` commit:
#   2026-06-10-git-b29bdd3715-full_build-www.gyan.dev
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

    date_m = _SNAPSHOT_DATE_RE.match(token)
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
    ``ok`` is ``True`` only when the build clears the release lane (``>= 8.1.1``)
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
        "reason": "",
        "cves": list(JUNE_2026_CVES),
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
                    f"post-fix floor {SNAPSHOT_FLOOR_DATE}"
                )
            else:
                result["reason"] = (
                    f"git-master snapshot {rec['snapshot_date']} predates the "
                    f"post-fix floor {SNAPSHOT_FLOOR_DATE}"
                )
        else:
            result["reason"] = (
                "git snapshot has no embedded date; cannot confirm it carries "
                f"the June-2026 fixes — rebuild from a snapshot >= {SNAPSHOT_FLOOR_DATE}"
            )
        return result

    if rec["release"]:
        result["lane"] = "release"
        if rec["release"] >= RELEASE_FLOOR:
            result["ok"] = True
            result["reason"] = (
                f"release {'.'.join(map(str, rec['release']))} is at/after the "
                f"{'.'.join(map(str, RELEASE_FLOOR))} security floor (point releases "
                "carry the backported June-2026 fixes; the guaranteed-clean fallback "
                f"is git snapshot {REFERENCE_GIT_COMMIT} dated {REFERENCE_GIT_DATE})"
            )
        else:
            result["reason"] = (
                f"release {'.'.join(map(str, rec['release']))} predates the "
                f"{'.'.join(map(str, RELEASE_FLOOR))} security floor"
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
        "pinned_installer_version": PINNED_INSTALLER_VERSION,
        "cves_addressed": list(JUNE_2026_CVES),
        "bundled": None,
    }
    if banner is None:
        banner = _probe_bundled_banner()
    if banner:
        record["bundled"] = check_security_floor(banner)
    return record


def _resolve_ffmpeg_bin() -> Optional[str]:
    try:
        from opencut.helpers import get_ffmpeg_path

        return get_ffmpeg_path()
    except Exception:
        return shutil.which("ffmpeg")


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
