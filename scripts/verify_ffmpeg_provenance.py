#!/usr/bin/env python3
"""Build/CI gate: assert the bundled FFmpeg clears the June-2026 security floor.

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ffmpeg", nargs="?", default=None, help="ffmpeg binary (default: bundled)")
    parser.add_argument("--ffprobe", default=None, help="companion ffprobe binary included in the release")
    parser.add_argument("--manifest", default=None, help="write provenance JSON to this path")
    parser.add_argument("--warn-only", action="store_true", help="exit 0 even when below floor")
    parser.add_argument("--release", action="store_true", help="require complete redistribution evidence")
    parser.add_argument("--source-url", default="", help="exact corresponding-source archive URL")
    parser.add_argument("--source-sha256", default="", help="SHA-256 of the corresponding-source archive")
    parser.add_argument("--build-origin", default="", help="builder, release date, and source revision")
    parser.add_argument("--license", default="GPL-3.0-or-later", help="effective binary license")
    parser.add_argument(
        "--corresponding-source",
        default="",
        help="instructions for obtaining and rebuilding the exact corresponding source",
    )
    args = parser.parse_args(argv)

    ffmpeg_bin = args.ffmpeg or _default_ffmpeg()
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
    record["build"] = {
        "origin": args.build_origin.strip(),
        "configuration": _configuration(banner),
        "banner": banner.strip(),
    }
    record["redistribution"] = {
        "license": args.license.strip(),
        "corresponding_source": args.corresponding_source.strip(),
    }

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

    if args.manifest and not release_missing:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote provenance manifest -> {args.manifest}")

    bundled = record.get("bundled")
    print(f"ffmpeg binary:   {ffmpeg_bin}")
    print(f"required floor:  release>={record['required_release_floor']} "
          f"OR git-master>={record['required_snapshot_floor_date']} "
          f"(ref commit {record['reference_git_commit']})")
    print(f"CVEs addressed:  {', '.join(record['cves_addressed'])}")
    print(f"required fixes:  {', '.join(record['required_fix_commits'])}")

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
