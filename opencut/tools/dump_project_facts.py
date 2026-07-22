"""Generate the canonical user-facing project-facts manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opencut.project_facts import build_project_facts, validate_project_facts

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "project_facts.json"


def serialise(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the manifest is stale")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    payload = build_project_facts()
    validate_project_facts(payload)
    expected = serialise(payload)

    if args.check:
        current = MANIFEST_PATH.read_text(encoding="utf-8") if MANIFEST_PATH.exists() else ""
        if current != expected:
            if not args.quiet:
                print(
                    "project facts are stale; run: "
                    "python -m opencut.tools.dump_project_facts"
                )
            return 1
        if not args.quiet:
            print("project facts are in sync")
        return 0

    MANIFEST_PATH.write_text(expected, encoding="utf-8", newline="\n")
    if not args.quiet:
        print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
