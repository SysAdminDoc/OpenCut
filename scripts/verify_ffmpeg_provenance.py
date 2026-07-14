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
import json
import os
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ffmpeg", nargs="?", default=None, help="ffmpeg binary (default: bundled)")
    parser.add_argument("--manifest", default=None, help="write provenance JSON to this path")
    parser.add_argument("--warn-only", action="store_true", help="exit 0 even when below floor")
    args = parser.parse_args(argv)

    ffmpeg_bin = args.ffmpeg or _default_ffmpeg()
    banner = _banner(ffmpeg_bin)
    record = fp.provenance_record(banner=banner)

    if args.manifest:
        Path(args.manifest).write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote provenance manifest -> {args.manifest}")

    bundled = record.get("bundled")
    print(f"ffmpeg binary:   {ffmpeg_bin}")
    print(f"required floor:  release>={record['required_release_floor']} "
          f"OR git-master>={record['required_snapshot_floor_date']} "
          f"(ref commit {record['reference_git_commit']})")
    print(f"CVEs addressed:  {', '.join(record['cves_addressed'])}")
    print(f"required fixes:  {', '.join(record['required_fix_commits'])}")

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
