#!/usr/bin/env python3
"""
Doc-vs-reality size drift check.

Closes RESEARCH_FEATURE_PLAN_2026-05-25.md E1: CLAUDE.md and
PROJECT_CONTEXT.md ship hand-edited file-size and module-count claims that
historically drift 2-3x away from the live filesystem. F252 (CEP→UXP
migration) is sized against those numbers, so the drift directly
mis-prices the migration estimate.

This script enforces a tolerance window (default ±15%) between each
documented size claim and the live ``wc -l`` / file count. Run it
manually before doc edits, and as a release-smoke gate to catch silent
regressions.

Targets are hard-coded here (not configurable) so adding a target is a
visible source edit.

Usage:
    python scripts/check_doc_sizes.py            # human-readable report
    python scripts/check_doc_sizes.py --check    # exit 1 if drift > tol
    python scripts/check_doc_sizes.py --json     # JSON for CI
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"
PROJECT_CONTEXT = ROOT / "PROJECT_CONTEXT.md"

DEFAULT_TOLERANCE_PCT = 15.0  # ±% drift permitted before --check fails


# ---------------------------------------------------------------------------
# Live-value providers
# ---------------------------------------------------------------------------
def _file_lines(rel_path: str) -> int:
    fpath = ROOT / rel_path
    if not fpath.is_file():
        return 0
    return sum(1 for _ in fpath.open("rb"))


def _dir_py_count(rel_path: str) -> int:
    fpath = ROOT / rel_path
    if not fpath.is_dir():
        return 0
    return sum(1 for p in fpath.iterdir() if p.is_file() and p.suffix == ".py")


def _route_module_count() -> int:
    # Route module count *excluding* __init__.py — what humans cite.
    return sum(
        1
        for p in (ROOT / "opencut" / "routes").iterdir()
        if p.is_file() and p.suffix == ".py" and p.name != "__init__.py"
    )


def _test_file_count() -> int:
    return sum(1 for p in (ROOT / "tests").glob("test_*.py"))


# ---------------------------------------------------------------------------
# Targets: (label, regex against doc text, live-value provider, doc paths)
# ---------------------------------------------------------------------------
@dataclass
class DocClaim:
    label: str
    regex: re.Pattern
    live: Callable[[], int]
    docs: tuple[Path, ...]
    unit: str = ""  # "lines", "files", "modules", etc.


# Each regex must capture the documented number as group 1 (or named "n").
TARGETS: list[DocClaim] = [
    DocClaim(
        # Match either the bare form "(~N lines)" or the dated form
        # "(~N lines as of YYYY-MM-DD; was ~M lines through ...)".
        label="CEP client/main.js lines",
        regex=re.compile(r"client/main\.js[^\n]*?\(~?([\d,]+)\s*lines?[^)]*\)", re.IGNORECASE),
        live=lambda: _file_lines("extension/com.opencut.panel/client/main.js"),
        docs=(CLAUDE_MD,),
        unit="lines",
    ),
    DocClaim(
        label="CEP client/main.js inline mention",
        regex=re.compile(r"~([\d,]+)-line\s+(?:vanilla\s+JS\s+)?`?main\.js`?", re.IGNORECASE),
        live=lambda: _file_lines("extension/com.opencut.panel/client/main.js"),
        docs=(PROJECT_CONTEXT,),
        unit="lines",
    ),
    DocClaim(
        label="opencut/core/ python file count",
        regex=re.compile(r"`opencut/core/`\)?\s*\|\s*\*\*([\d,]+)\*\*\s+Python files", re.IGNORECASE),
        live=lambda: _dir_py_count("opencut/core"),
        docs=(PROJECT_CONTEXT,),
        unit="files",
    ),
    DocClaim(
        label="opencut/routes/ blueprint files",
        regex=re.compile(r"`opencut/routes/`\)?\s*\|\s*\*\*([\d,]+)\*\*", re.IGNORECASE),
        live=_route_module_count,
        docs=(PROJECT_CONTEXT,),
        unit="files",
    ),
    DocClaim(
        label="tests file count",
        regex=re.compile(r"Tests\s*\|\s*\*\*([\d,]+)\s*test_\*\.py\s*files", re.IGNORECASE),
        live=_test_file_count,
        docs=(PROJECT_CONTEXT,),
        unit="files",
    ),
]


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
@dataclass
class DriftEntry:
    label: str
    doc: str
    claimed: int | None
    actual: int
    drift_pct: float | None
    over_tolerance: bool

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "doc": self.doc,
            "claimed": self.claimed,
            "actual": self.actual,
            "drift_pct": self.drift_pct,
            "over_tolerance": self.over_tolerance,
        }


def _parse_n(group: str) -> int:
    return int(group.replace(",", "").strip())


def _evaluate(tolerance_pct: float) -> list[DriftEntry]:
    out: list[DriftEntry] = []
    for claim in TARGETS:
        actual = claim.live()
        for doc in claim.docs:
            if not doc.exists():
                continue
            text = doc.read_text(encoding="utf-8", errors="replace")
            m = claim.regex.search(text)
            if not m:
                # No documented number to compare — skip silently
                continue
            claimed = _parse_n(m.group(1))
            if actual == 0:
                # Live value unavailable; surface but never flag drift.
                out.append(DriftEntry(claim.label, str(doc.relative_to(ROOT)),
                                       claimed, actual, None, False))
                continue
            drift = (claimed - actual) / actual * 100.0
            over = abs(drift) > tolerance_pct
            out.append(DriftEntry(claim.label, str(doc.relative_to(ROOT)),
                                   claimed, actual, drift, over))
    return out


def cmd_report(tolerance: float) -> int:
    entries = _evaluate(tolerance)
    if not entries:
        print("No documented claims matched any target. Nothing to check.")
        return 0
    width = max(len(e.label) for e in entries) + 2
    print(f"{'Claim'.ljust(width)} {'Doc':<24} {'Claimed':>10} {'Actual':>10} {'Drift':>10}")
    print("-" * (width + 60))
    any_over = False
    for e in entries:
        drift_str = "n/a" if e.drift_pct is None else f"{e.drift_pct:+.1f}%"
        flag = " !!" if e.over_tolerance else ""
        print(
            f"{e.label.ljust(width)} {e.doc:<24} "
            f"{e.claimed if e.claimed is not None else '-':>10} "
            f"{e.actual:>10} {drift_str:>10}{flag}"
        )
        if e.over_tolerance:
            any_over = True
    print(f"\nTolerance: ±{tolerance:.0f}%.")
    if any_over:
        print("DRIFT DETECTED — update CLAUDE.md / PROJECT_CONTEXT.md.")
    else:
        print("All documented sizes within tolerance.")
    return 1 if any_over else 0


def cmd_json(tolerance: float) -> int:
    entries = _evaluate(tolerance)
    payload = {
        "tolerance_pct": tolerance,
        "entries": [e.to_dict() for e in entries],
        "any_drift": any(e.over_tolerance for e in entries),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if payload["any_drift"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CLAUDE.md / PROJECT_CONTEXT.md size claims against the filesystem.")
    parser.add_argument("--check", action="store_true", help="Exit 1 if drift exceeds tolerance.")
    parser.add_argument("--json", action="store_true", help="Emit JSON for CI.")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PCT,
                        help=f"±%% tolerance (default {DEFAULT_TOLERANCE_PCT}).")
    args = parser.parse_args()
    if args.json:
        sys.exit(cmd_json(args.tolerance))
    if args.check:
        sys.exit(cmd_report(args.tolerance))
    sys.exit(cmd_report(args.tolerance))


if __name__ == "__main__":
    main()
