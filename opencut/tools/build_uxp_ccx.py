"""Build the distributable UXP package.

``docs/UXP_MIGRATION.md`` Phase 4 listed "add a local CCX package build script"
as outstanding, and nothing in the repository could produce one. Run with::

    python -m opencut.tools.build_uxp_ccx

Signing the result for Creative Cloud distribution is a separate, credentialed
step; this produces the unsigned package that step consumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opencut.core.uxp_package import UXPPackageError, build_ccx, read_uxp_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the OpenCut UXP .ccx package")
    parser.add_argument(
        "--output",
        default="",
        help="Destination path (default: dist/OpenCut-UXP-<version>.ccx)",
    )
    args = parser.parse_args(argv)

    try:
        manifest = read_uxp_manifest()
        output = Path(args.output) if args.output else (
            REPO_ROOT / "dist" / f"OpenCut-UXP-{manifest['version']}.ccx"
        )
        result = build_ccx(output)
    except UXPPackageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Built {result['path']}")
    print(f"  plugin:  {result['plugin_id']} {result['version']}")
    print(f"  folder:  {result['folder_name']}")
    print(f"  files:   {result['file_count']}")
    print(f"  bytes:   {result['bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
