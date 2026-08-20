"""Record and check the direct-surface ratchet baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from opencut.core.surface_ratchet import (
    BASELINE_PATH,
    build_baseline,
    coverage_percent,
    evaluate,
    load_baseline,
    load_manifest,
    render_report,
)


def write_baseline(baseline: dict, path: Path = BASELINE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so Windows does not rewrite the file to CRLF behind the policy
    # in .gitattributes and turn a two-line change into a whole-file diff.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when coverage has fallen or a family grew unjustified.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Only print on failure.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help="Baseline to check against. Exists so the gate's failure paths can be "
             "exercised where they actually run, not only in-process.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = load_manifest()
    if args.check:
        report = evaluate(manifest, load_baseline(args.baseline))
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        elif report["passes"]:
            if not args.quiet:
                print(render_report(report))
        else:
            print(render_report(report))
        return 0 if report["passes"] else 1

    previous = load_baseline()
    baseline = build_baseline(manifest)
    write_baseline(baseline)
    if previous is not None:
        was = float(previous.get("coverage_floor_percent") or 0.0)
        now = coverage_percent(manifest)
        if round(now, 1) < round(was, 1):
            # Re-recording is how the ratchet is released, so say plainly that
            # it happened. A silent regeneration is the failure mode here.
            print(
                f"WARNING: the recorded floor moved down, {was}% -> {now}%. "
                "This lowers the bar for every future change; make sure the commit says why."
            )
    print(
        f"Wrote {BASELINE_PATH} (floor {baseline['coverage_floor_percent']}%, "
        f"{len(baseline['family_ceilings'])} integration-only families)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
