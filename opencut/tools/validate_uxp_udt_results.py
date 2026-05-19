"""Validate captured UXP Developer Tool smoke-harness results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from opencut.core.uxp_udt_results import (
    build_udt_result_template,
    validate_udt_result_capture,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="?", type=Path, help="captured UDT JSON result")
    parser.add_argument("--json", action="store_true", help="print machine-readable validation output")
    parser.add_argument("--template", action="store_true", help="print a capture template and exit")
    parser.add_argument("--read-only", action="store_true", help="validate only the safe-by-default scenarios")
    parser.add_argument("--allow-blocked", action="store_true", help="allow accepted environment blockers")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.template:
        print(json.dumps(build_udt_result_template(include_mutating=not args.read_only), indent=2, sort_keys=True))
        return 0

    if args.capture is None:
        parser.error("capture path is required unless --template is used")

    try:
        payload = _load_json(args.capture)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unable to read UDT capture JSON: {exc}", file=sys.stderr)
        return 2

    report = validate_udt_result_capture(
        payload,
        require_mutating=not args.read_only,
        allow_blocked=args.allow_blocked,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        readiness = "ready for WebView cutover" if report["ready_for_webview_cutover"] else "valid diagnostic capture"
        print(
            f"UXP UDT result capture is valid: {readiness} "
            f"({report['summary']['passed']} passed, {report['summary']['blocked']} blocked)."
        )
    else:
        print("UXP UDT result capture is not valid:", file=sys.stderr)
        for error in report["errors"]:
            print(f"  {error}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
