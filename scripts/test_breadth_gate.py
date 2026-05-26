#!/usr/bin/env python3
"""
Test breadth ratchet (RESEARCH_FEATURE_PLAN_2026-05-25 Q9).

Q9 in the research file targeted "per-blueprint coverage floor" — but
running coverage.py with per-blueprint groups requires a real pytest
pass, which is too expensive for PR-fast. This script is the cheap
proxy: count how many ``opencut/core/*`` modules are referenced by at
least one ``tests/test_*.py`` file via ``from opencut.core(.X) import``
or ``import opencut.core.X``. Maintain a baseline ratio that ratchets
up; the gate fails if the ratio drops below the baseline.

The original RESEARCH_FEATURE_PLAN claim of "93% untested" was a
sampling artifact — it counted only modules with a *dedicated*
``test_<name>.py`` file and missed indirect imports through shared
fixtures. The real ratio is currently ~78%; this gate locks that in.

Usage:
    python scripts/test_breadth_gate.py            # report
    python scripts/test_breadth_gate.py --check    # exit 1 below baseline
    python scripts/test_breadth_gate.py --json     # JSON for CI
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT / "opencut" / "core"
TESTS_DIR = ROOT / "tests"

# Floor of the tested-module ratio. Ratchet UP after the next coverage push.
# RESEARCH_FEATURE_PLAN_2026-05-25 Q9 measured 78.3% as the live ratio on the
# autonomous-loop branch. Set the floor a little below to absorb refactor
# churn; raise it once new modules ship with tests.
MIN_RATIO = 0.75

_IMPORT_FROM_RE = re.compile(
    r"from\s+opencut\.core(?:\.(\w+))?\s+import\s+([^\n;]+)"
)
_IMPORT_DIRECT_RE = re.compile(r"import\s+opencut\.core\.(\w+)")


def _list_core_modules() -> list[str]:
    return sorted(
        p.stem
        for p in CORE_DIR.glob("*.py")
        if p.name != "__init__.py"
    )


def _tested_module_names() -> set[str]:
    tested: set[str] = set()
    for p in TESTS_DIR.glob("test_*.py"):
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in _IMPORT_FROM_RE.finditer(text):
            sub = m.group(1)
            if sub:
                tested.add(sub)
            else:
                for name in m.group(2).split(","):
                    tested.add(name.strip().split()[0])
        for m in _IMPORT_DIRECT_RE.finditer(text):
            tested.add(m.group(1))
    return tested


def evaluate() -> dict:
    core = _list_core_modules()
    tested = _tested_module_names()
    referenced = [m for m in core if m in tested]
    untested = [m for m in core if m not in tested]
    ratio = (len(referenced) / len(core)) if core else 0.0
    return {
        "core_count": len(core),
        "referenced_count": len(referenced),
        "untested_count": len(untested),
        "ratio": round(ratio, 4),
        "floor": MIN_RATIO,
        "untested_sample": untested[:25],
    }


def cmd_report(check: bool) -> int:
    e = evaluate()
    print(
        f"core modules: {e['core_count']} | "
        f"referenced by tests/: {e['referenced_count']} | "
        f"ratio: {e['ratio']:.1%} (floor: {e['floor']:.0%})"
    )
    if e["untested_count"]:
        print(f"\nFirst {len(e['untested_sample'])} untested modules:")
        for m in e["untested_sample"]:
            print(f"  opencut/core/{m}.py")
    if check and e["ratio"] < e["floor"]:
        print(
            f"\nFAIL: tested-module ratio {e['ratio']:.1%} below floor "
            f"{e['floor']:.0%}. Add a test import or raise the floor "
            "explicitly in scripts/test_breadth_gate.py.",
            file=sys.stderr,
        )
        return 1
    if check:
        print("\nbreadth ratio within floor.")
    return 0


def cmd_json() -> int:
    e = evaluate()
    json.dump(e, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if e["ratio"] < e["floor"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test breadth ratchet gate.")
    parser.add_argument("--check", action="store_true", help="Exit 1 below floor.")
    parser.add_argument("--json", action="store_true", help="JSON for CI.")
    args = parser.parse_args()
    if args.json:
        sys.exit(cmd_json())
    sys.exit(cmd_report(args.check))


if __name__ == "__main__":
    main()
