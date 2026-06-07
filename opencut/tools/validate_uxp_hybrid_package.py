"""Validate a UXP Hybrid plugin bundle before UDT packaging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from opencut.core.uxp_hybrid_package import validate_uxp_hybrid_package


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin_root", type=Path, help="UXP plugin directory containing manifest.json")
    parser.add_argument(
        "--independent",
        action="store_true",
        help="allow a partial architecture set for independent/internal distribution",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable validation output")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = validate_uxp_hybrid_package(
        args.plugin_root,
        require_marketplace_architectures=not args.independent,
    )
    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report.valid:
        if report.hybrid:
            mode = "Marketplace-ready" if not args.independent else "independent-distribution"
            print(f"UXP Hybrid package is valid for {mode}: {report.addon_name}")
        else:
            print("UXP package is valid and does not declare a Hybrid addon.")
    else:
        print("UXP Hybrid package validation failed:", file=sys.stderr)
        for error in report.errors:
            print(f"  {error}", file=sys.stderr)
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
