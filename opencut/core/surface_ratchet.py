"""Stop the direct-surface ratio from falling any further.

The route manifest measures how many shipped routes are reachable from a
first-party surface — the panels, the command palette, the CLI, or the curated
MCP catalogue. It has been measuring that for a while and the number only ever
goes down, because every wave adds API faster than it adds product. Measuring a
number nothing defends is how it got to 17.9%.

This module turns the measurement into a gate. It records two things at the
moment the ratchet lands: the coverage percentage, and how many integration-only
routes each blueprint had. After that:

* coverage may not fall below the recorded percentage;
* a blueprint may not grow its integration-only count beyond what was recorded;
* a blueprint that had none may not start having them.

Any of those can be satisfied two ways — give the route a surface, or write down
why it is integration-only. The second is deliberately a sentence someone has to
type, not a flag, so "there is no product for this yet" gets said out loud.

The 104 blueprints that already carried integration-only routes are grandfathered
at their recorded counts. Demanding a retroactive justification for 1,313 routes
would produce 104 sentences of boilerplate and defend nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "route_manifest.json"
BASELINE_PATH = REPO_ROOT / "opencut" / "_generated" / "surface_ratchet.json"

BASELINE_VERSION = 1

#: How many integration-only families the report names. Enough to see where the
#: mass is without printing a hundred lines nobody reads.
REPORT_FAMILY_LIMIT = 10

#: Why a blueprint is allowed to carry integration-only routes it did not have
#: when the ratchet landed. Keyed by blueprint name; the text is what a reviewer
#: reads when they ask why a route shipped unreachable. Empty on purpose — the
#: grandfathered families are recorded in the baseline, and everything after
#: this point is a decision someone makes at the time.
JUSTIFICATIONS: Dict[str, str] = {}


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.is_file():
        from opencut._generated import GeneratedManifestMissing

        raise GeneratedManifestMissing(path.name, path.parent)
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(path: Path = BASELINE_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def integration_only_counts(manifest: dict) -> Dict[str, int]:
    """Integration-only route count per blueprint."""
    counts = Counter(
        str(route.get("blueprint") or "")
        for route in manifest.get("routes") or []
        if route.get("surface_class") == "integration-only"
    )
    return dict(sorted(counts.items()))


def largest_families(manifest: dict, limit: int = REPORT_FAMILY_LIMIT) -> List[dict]:
    """The biggest integration-only families, so triage has somewhere to start."""
    counts = integration_only_counts(manifest)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"blueprint": name, "integration_only_routes": count} for name, count in ordered[:limit]]


def coverage_percent(manifest: dict) -> float:
    summary = (manifest.get("surface_coverage") or {}).get("summary") or {}
    return float(summary.get("coverage_percent") or 0.0)


def build_baseline(manifest: dict) -> dict:
    """Record the current ratio and per-family counts as the floor to hold."""
    return {
        "baseline_version": BASELINE_VERSION,
        "coverage_floor_percent": coverage_percent(manifest),
        "shipped_routes": int(manifest.get("shipped_route_count") or 0),
        "family_ceilings": integration_only_counts(manifest),
    }


def evaluate(manifest: dict, baseline: Optional[dict]) -> dict:
    """Check a manifest against the recorded baseline.

    Returns a report rather than raising, so the caller can print every problem
    at once instead of one per run.
    """
    report = {
        "passes": False,
        "errors": [],
        "coverage_percent": coverage_percent(manifest),
        "coverage_floor_percent": None,
        "families_over_ceiling": [],
        "new_unjustified_families": [],
        "justified_families": sorted(JUSTIFICATIONS),
        "largest_integration_only_families": largest_families(manifest),
    }
    if baseline is None:
        report["errors"].append(
            "surface ratchet baseline is missing; regenerate with "
            "`python -m opencut.tools.dump_surface_ratchet`"
        )
        return report

    floor = float(baseline.get("coverage_floor_percent") or 0.0)
    report["coverage_floor_percent"] = floor
    # Rounding in the manifest is one decimal place, so compare at that width
    # rather than letting float noise fail a ratio that did not move.
    if round(report["coverage_percent"], 1) < round(floor, 1):
        report["errors"].append(
            f"direct-surface coverage fell to {report['coverage_percent']}% from the recorded "
            f"floor of {floor}%. Give the new routes a surface, or record why they are "
            f"integration-only in opencut.core.surface_ratchet.JUSTIFICATIONS."
        )

    ceilings = baseline.get("family_ceilings") or {}
    for blueprint, count in integration_only_counts(manifest).items():
        allowed = int(ceilings.get(blueprint, 0))
        if count <= allowed:
            continue
        if blueprint in JUSTIFICATIONS:
            continue
        entry = {
            "blueprint": blueprint,
            "integration_only_routes": count,
            "recorded": allowed,
        }
        if allowed:
            report["families_over_ceiling"].append(entry)
            report["errors"].append(
                f"blueprint '{blueprint}' grew from {allowed} to {count} integration-only "
                f"routes. Give them a surface, or add a justification for '{blueprint}'."
            )
        else:
            report["new_unjustified_families"].append(entry)
            report["errors"].append(
                f"blueprint '{blueprint}' ships {count} route(s) reachable from no first-party "
                f"surface and carries no justification. Give them a surface, or add a "
                f"justification for '{blueprint}'."
            )

    report["passes"] = not report["errors"]
    return report


def render_report(report: dict) -> str:
    lines: List[str] = []
    floor = report.get("coverage_floor_percent")
    lines.append(
        f"direct-surface coverage {report['coverage_percent']}%"
        + (f" against a floor of {floor}%" if floor is not None else "")
    )
    for error in report["errors"]:
        lines.append(f"  FAIL {error}")
    if report["largest_integration_only_families"]:
        lines.append("  largest integration-only families:")
        for family in report["largest_integration_only_families"]:
            lines.append(f"    {family['blueprint']}: {family['integration_only_routes']}")
    return "\n".join(lines)
