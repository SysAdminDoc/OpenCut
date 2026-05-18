"""Run the F223 caption Unicode validation gate."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Optional

from opencut.core.caption_unicode_validation import (
    build_caption_unicode_report,
    report_to_json,
)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full machine-readable validation report.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any Unicode caption fixture fails.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_caption_unicode_report()
    if args.json:
        print(report_to_json(report))
    else:
        summary = report["summary"]
        print(
            "Caption Unicode validation "
            f"{report['status']} "
            f"({summary['case_count']} cases, "
            f"{summary['warnings']} warnings, "
            f"{summary['failures']} failures)."
        )
        for case in report["cases"]:
            if case["failures"] or case["warnings"]:
                print(
                    f"  {case['case_id']}: "
                    f"{len(case['warnings'])} warnings, "
                    f"{len(case['failures'])} failures"
                )

    if args.check and report["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
