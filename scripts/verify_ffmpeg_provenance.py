#!/usr/bin/env python3
"""Build/CI gate: assert the bundled FFmpeg clears the July-2026 security floor.

Runs ``<ffmpeg> -version`` and grades the banner via
:mod:`opencut.core.ffmpeg_provenance`. Exits non-zero (fails closed) when the
build is below the security floor so a release cannot ship a crafted-media-CVE
binary by accident.

Usage::

    python scripts/verify_ffmpeg_provenance.py                      # bundled ffmpeg/ffmpeg.exe
    python scripts/verify_ffmpeg_provenance.py /path/to/ffmpeg      # explicit binary
    python scripts/verify_ffmpeg_provenance.py --manifest prov.json # also write provenance JSON
    python scripts/verify_ffmpeg_provenance.py --warn-only          # never fail the build

The provenance JSON records the *actual* bundled build (version, git commit /
snapshot date, lane, CVE list) so a release captures ground-truth provenance,
not just the declared floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencut.core import ffmpeg_provenance as fp  # noqa: E402


def _default_ffmpeg() -> str:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = REPO_ROOT / "ffmpeg" / exe
    if bundled.exists():
        return str(bundled)
    # Fall back to the resolver the server uses.
    try:
        from opencut.helpers import get_ffmpeg_path

        return get_ffmpeg_path()
    except Exception:
        import shutil

        return shutil.which("ffmpeg") or exe


def _banner(ffmpeg_bin: str) -> str:
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"ERROR: could not run {ffmpeg_bin!r}: {exc}", file=sys.stderr)
        return ""
    return result.stdout or ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration(banner: str) -> str:
    for line in banner.splitlines():
        if line.strip().lower().startswith("configuration:"):
            return line.split(":", 1)[1].strip()
    return ""


def _banner_token(banner: str) -> str:
    first = banner.splitlines()[0] if banner.strip() else ""
    match = re.search(r"\bversion\s+(\S+)", first)
    return match.group(1) if match else ""


def _pinned_payload_mismatches(record: dict, artifact_paths: list[Path]) -> list[str]:
    """Where the shipped binaries differ from the pinned archive's contents.

    The archive hash alone says nothing about the files actually copied into
    the payload, so each binary's banner, hash and configure switches are
    compared against the pin.
    """
    problems: list[str] = []
    pins = (
        ("package", fp.PINNED_INSTALLER_ARCHIVE_URL, fp.PINNED_INSTALLER_ARCHIVE_SHA256),
        ("source", fp.PINNED_INSTALLER_SOURCE_URL, fp.PINNED_INSTALLER_SOURCE_SHA256),
    )
    for key, url, sha256 in pins:
        if record[key]["url"] != url:
            problems.append(f"{key} URL is not the pinned {url}")
        if record[key]["sha256"] != sha256:
            problems.append(f"{key} SHA-256 differs from the pin")
    hashes = {a["name"].lower(): a["sha256"] for a in record["artifacts"]}
    for path in artifact_paths:
        name = path.name.lower()
        token = _banner_token(_banner(str(path)))
        if token != fp.PINNED_INSTALLER_VERSION:
            problems.append(f"{name} banner {token or '?'} is not {fp.PINNED_INSTALLER_VERSION}")
        expected = fp.PINNED_INSTALLER_BINARY_SHA256.get(name)
        if expected is None:
            problems.append(f"{name} has no pinned hash")
        elif hashes.get(name) != expected:
            problems.append(f"{name} SHA-256 differs from the pinned archive")
    configuration = record["build"]["configuration"]
    for switch in fp.PINNED_INSTALLER_REQUIRED_CONFIGURE:
        if switch not in configuration.split():
            problems.append(f"configuration lacks {switch}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ffmpeg", nargs="?", default=None, help="ffmpeg binary (default: bundled)")
    parser.add_argument("--ffprobe", default=None, help="companion ffprobe binary included in the release")
    parser.add_argument("--manifest", default=None, help="write provenance JSON to this path")
    parser.add_argument("--warn-only", action="store_true", help="exit 0 even when below floor")
    parser.add_argument("--release", action="store_true", help="require complete redistribution evidence")
    parser.add_argument("--source-url", default="", help="exact corresponding-source archive URL")
    parser.add_argument("--source-sha256", default="", help="SHA-256 of the corresponding-source archive")
    parser.add_argument("--package-url", default="", help="URL of the binary package used to obtain the artifact")
    parser.add_argument("--package-sha256", default="", help="SHA-256 of the binary package used to obtain the artifact")
    parser.add_argument("--build-origin", default="", help="builder, release date, and source revision")
    parser.add_argument("--license", default="GPL-3.0-or-later", help="effective binary license")
    parser.add_argument(
        "--corresponding-source",
        default="",
        help="instructions for obtaining and rebuilding the exact corresponding source",
    )
    parser.add_argument(
        "--require-pinned-snapshot",
        action="store_true",
        help="forbid release evidence unless a passing snapshot matches the bundled full-build commit",
    )
    parser.add_argument(
        "--pinned-payload",
        action="store_true",
        help="--require-pinned-snapshot with the pinned package and source filled in (release smoke)",
    )
    args = parser.parse_args(argv)
    if args.pinned_payload:
        args.require_pinned_snapshot = True
        args.package_url = args.package_url or fp.PINNED_INSTALLER_ARCHIVE_URL
        args.package_sha256 = args.package_sha256 or fp.PINNED_INSTALLER_ARCHIVE_SHA256
        args.source_url = args.source_url or fp.PINNED_INSTALLER_SOURCE_URL
        args.source_sha256 = args.source_sha256 or fp.PINNED_INSTALLER_SOURCE_SHA256

    ffmpeg_bin = args.ffmpeg or _default_ffmpeg()
    if Path(ffmpeg_bin).is_file():
        # Windows can't launch a relative path such as ffmpeg/ffmpeg.exe.
        ffmpeg_bin = str(Path(ffmpeg_bin).resolve())
    banner = _banner(ffmpeg_bin)
    record = fp.provenance_record(banner=banner)

    artifact_paths = [Path(ffmpeg_bin).resolve()]
    if args.ffprobe:
        artifact_paths.append(Path(args.ffprobe).resolve())
    artifacts = []
    for artifact_path in artifact_paths:
        if artifact_path.is_file():
            artifacts.append(
                {
                    "name": artifact_path.name,
                    "path": str(artifact_path),
                    "size": artifact_path.stat().st_size,
                    "sha256": _sha256(artifact_path),
                }
            )
    record["artifacts"] = artifacts
    record["source"] = {"url": args.source_url.strip(), "sha256": args.source_sha256.strip().lower()}
    record["package"] = {
        "url": args.package_url.strip(),
        "sha256": args.package_sha256.strip().lower(),
    }
    record["build"] = {
        "origin": args.build_origin.strip(),
        "configuration": _configuration(banner),
        "banner": banner.strip(),
    }
    record["redistribution"] = {
        "license": args.license.strip(),
        "corresponding_source": args.corresponding_source.strip(),
    }
    bundled = record.get("bundled")

    release_missing = []
    if args.release:
        if len(artifacts) != len(artifact_paths):
            release_missing.append("bundled artifact file/hash")
        if not record["source"]["url"]:
            release_missing.append("source URL")
        if not re.fullmatch(r"[0-9a-f]{64}", record["source"]["sha256"]):
            release_missing.append("source SHA-256")
        if not record["build"]["origin"]:
            release_missing.append("build origin")
        if not record["build"]["configuration"]:
            release_missing.append("build configuration")
        if not record["redistribution"]["license"]:
            release_missing.append("license")
        if not record["redistribution"]["corresponding_source"]:
            release_missing.append("corresponding-source instructions")

    if args.require_pinned_snapshot:
        if not bundled or bundled.get("lane") != "snapshot":
            release_missing.append("pinned git-full snapshot")
        else:
            if not fp.is_pinned_snapshot(bundled):
                release_missing.append("pinned snapshot commit")
            if bundled.get("flavor") != "full":
                release_missing.append("full snapshot flavor")
        if not record["package"]["url"]:
            release_missing.append("binary package URL")
        if not re.fullmatch(r"[0-9a-f]{64}", record["package"]["sha256"]):
            release_missing.append("binary package SHA-256")
        release_missing.extend(_pinned_payload_mismatches(record, artifact_paths))

    if args.manifest and not release_missing:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote provenance manifest -> {args.manifest}")

    print(f"ffmpeg binary:   {ffmpeg_bin}")
    print(f"required floor:  release>={record['required_release_floor']} "
          f"OR git-master>={record['required_snapshot_floor_date']} "
          f"(ref commit {record['reference_git_commit']})")
    print(f"CVEs addressed:  {', '.join(record['cves_addressed'])}")
    print(f"required fixes:  {', '.join(record['required_fix_commits'])}")

    # State the scope of the check. "Passed everything we grade" and "is clean"
    # are different claims, and only the first one is supported.
    coverage = fp.advisory_coverage()
    print(f"advisory scope:  {coverage['graded_count']} graded of "
          f"{coverage['total_known']} known")
    if coverage["ungraded"]:
        print(f"NOT GRADED:      {', '.join(coverage['ungraded'])}")
        print("                 (no upstream fix commit recorded yet; "
              "these were not checked)")

    if release_missing:
        print("RESULT: INCOMPLETE RELEASE EVIDENCE — " + ", ".join(release_missing), file=sys.stderr)
        return 0 if args.warn_only else 3

    if not bundled:
        print("RESULT: UNKNOWN — no ffmpeg banner could be read.")
        return 0 if args.warn_only else 2

    status = "OK" if bundled["ok"] else "BELOW FLOOR"
    print(f"bundled version: {bundled.get('version') or 'unknown'} (lane: {bundled.get('lane')})")
    print(f"RESULT: {status} — {bundled.get('reason', '')}")

    if bundled["ok"] or args.warn_only:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
