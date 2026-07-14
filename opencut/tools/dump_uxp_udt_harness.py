"""Dump the F267 UXP Developer Tool smoke-harness manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opencut.core.uxp_udt_harness import build_udt_harness_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "uxp_udt_harness.json"
PANEL_PATH = REPO_ROOT / "extension" / "com.opencut.uxp" / "uxp-udt-harness.json"


def _render_manifest() -> str:
    return json.dumps(build_udt_harness_manifest(), indent=2, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _check(path: Path, expected: str) -> list[str]:
    if not path.is_file():
        return [f"{path.relative_to(REPO_ROOT)} is missing"]
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return [f"{path.relative_to(REPO_ROOT)} is out of sync"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed artifacts are stale")
    parser.add_argument("--json", action="store_true", help="print the manifest to stdout")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH, help="repository manifest path")
    parser.add_argument("--panel-output", type=Path, default=PANEL_PATH, help="bundled UXP panel manifest path")
    args = parser.parse_args(argv)

    rendered = _render_manifest()
    if args.json:
        sys.stdout.write(rendered)
        return 0
    if args.check:
        errors = [* _check(args.output, rendered), * _check(args.panel_output, rendered)]
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        manifest = json.loads(rendered)
        print(
            "UXP UDT harness artifacts in sync "
            f"({manifest['scenario_count']} scenarios, "
            f"{manifest['safe_default_count']} safe-by-default)."
        )
        return 0

    _write(args.output, rendered)
    _write(args.panel_output, rendered)
    print(f"Wrote {args.output.relative_to(REPO_ROOT)}")
    print(f"Wrote {args.panel_output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
