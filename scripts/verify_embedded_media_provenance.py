#!/usr/bin/env python3
"""Fail closed on OpenCV and PyAV FFmpeg copies below the security floor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencut.core.embedded_media_provenance import (  # noqa: E402
    build_release_inventory,
    inspect_runtime,
    write_manifest,
)


def _default_lane() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("windows", "linux", "macos"), default=_default_lane())
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.artifact:
        payload = build_release_inventory(lane=args.lane, artifact_paths=args.artifact)
    else:
        payload = inspect_runtime(required=True, include_hashes=True)
    if args.manifest:
        write_manifest(payload, args.manifest)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif payload.get("ok"):
        opencv = payload.get("opencv") or (payload.get("providers") or {}).get("opencv") or {}
        pyav = payload.get("pyav") or (payload.get("providers") or {}).get("pyav") or {}
        print(
            "embedded media provenance verified: "
            f"OpenCV {opencv.get('version', 'absent')} ({opencv.get('status', 'absent')}), "
            f"PyAV {pyav.get('version', 'absent')} ({pyav.get('status', 'absent')})"
        )
    else:
        for error in payload.get("errors") or ["embedded media provenance failed"]:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
