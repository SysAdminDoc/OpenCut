#!/usr/bin/env python3
"""Compare the installed distributions against the versions OpenCut declares.

`check_dependency_matrix.py` resolves the declared lanes; nothing compared
those lanes to what is actually importable. That gap meant the suite could
report a green baseline on a stack that violates the project's own
constraints, so users installing per `pyproject.toml` would execute code paths
the suite had never run - including across major-version boundaries such as
PySceneDetect 0.6 vs 0.7.

Usage:

* ``python scripts/check_installed_versions.py`` - report every mismatch.
* ``python scripts/check_installed_versions.py --extras ai`` - restrict to
  the named extras (``core`` is always included).
* ``python scripts/check_installed_versions.py --json`` - machine-readable.

Exit status is non-zero when an installed distribution violates its declared
specifier. Distributions that are not installed at all are reported
separately and never fail the check: optional extras are optional.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

CORE_LANE = "core"


@dataclass
class Mismatch:
    distribution: str
    installed: str
    declared: str
    lanes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "distribution": self.distribution,
            "installed": self.installed,
            "declared": self.declared,
            "lanes": sorted(set(self.lanes)),
        }

    def __str__(self) -> str:
        lanes = ", ".join(sorted(set(self.lanes)))
        return f"{self.distribution} {self.installed} violates '{self.declared}' ({lanes})"


def load_declared(pyproject: Path = PYPROJECT) -> Dict[str, List[tuple[str, Requirement]]]:
    """Map distribution name -> [(lane, requirement)] from `pyproject.toml`."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})

    lanes: Dict[str, List[str]] = {CORE_LANE: list(project.get("dependencies", []))}
    for extra, entries in (project.get("optional-dependencies") or {}).items():
        lanes[extra] = list(entries)

    declared: Dict[str, List[tuple[str, Requirement]]] = {}
    for lane, entries in lanes.items():
        for entry in entries:
            try:
                requirement = Requirement(entry)
            except Exception:
                continue
            if requirement.url:
                # A direct reference pins by URL, not by version.
                continue
            key = requirement.name.lower().replace("_", "-")
            declared.setdefault(key, []).append((lane, requirement))
    return declared


def _marker_applies(requirement: Requirement, extra: str) -> bool:
    if requirement.marker is None:
        return True
    try:
        return bool(requirement.marker.evaluate({"extra": extra}))
    except Exception:
        # An un-evaluable marker (unknown variable) is treated as applying, so
        # the check errs toward reporting rather than silently skipping.
        return True


def check_installed(
    extras: Optional[Iterable[str]] = None,
    pyproject: Path = PYPROJECT,
) -> dict:
    declared = load_declared(pyproject)
    requested = {e.strip() for e in (extras or ()) if e.strip()}
    if "all" in requested or not requested:
        # Every declared lane: a distribution installed for one extra is the
        # same import for every other, so a violation anywhere is a violation.
        requested = {lane for entries in declared.values() for lane, _ in entries}
    selected = {CORE_LANE} | requested

    mismatches: List[Mismatch] = []
    absent: List[str] = []
    checked = 0

    for distribution, entries in sorted(declared.items()):
        applicable = [
            (lane, req)
            for lane, req in entries
            if lane in selected and _marker_applies(req, lane)
        ]
        if not applicable:
            continue

        try:
            found = installed_version(distribution)
        except PackageNotFoundError:
            absent.append(distribution)
            continue

        try:
            parsed = Version(found)
        except InvalidVersion:
            absent.append(distribution)
            continue

        checked += 1
        for lane, requirement in applicable:
            if not requirement.specifier:
                continue
            if parsed in requirement.specifier:
                continue
            existing = next(
                (m for m in mismatches if m.distribution == distribution), None
            )
            if existing is None:
                mismatches.append(
                    Mismatch(
                        distribution=distribution,
                        installed=found,
                        declared=str(requirement.specifier),
                        lanes=[lane],
                    )
                )
            else:
                existing.lanes.append(lane)

    return {
        "extras": sorted(selected),
        "checked": checked,
        "absent": sorted(absent),
        "mismatches": [m.as_dict() for m in mismatches],
        "ok": not mismatches,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        default="all",
        help="comma-separated extras to include besides core, or \"all\" (default: all)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    extras = [e for e in args.extras.split(",") if e.strip()]
    report = check_installed(extras)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    lanes = ", ".join(report["extras"])
    if report["ok"]:
        print(
            f"[installed-versions] OK - {report['checked']} distributions match "
            f"their declared specifiers ({lanes})"
        )
        return 0

    print("[installed-versions] FAIL - installed packages violate declared constraints:")
    for entry in report["mismatches"]:
        print(
            f"  - {entry['distribution']} {entry['installed']} violates "
            f"'{entry['declared']}' ({', '.join(entry['lanes'])})"
        )
    print(
        "\nEither correct the environment (pip install -U ...) or correct the "
        "constraint in pyproject.toml/requirements.txt - a suite that runs on "
        "an undeclared stack proves nothing about what users install."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
