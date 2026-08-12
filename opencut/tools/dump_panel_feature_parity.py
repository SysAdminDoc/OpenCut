"""Generate and gate the CEP/UXP backend-route feature parity manifest.

The manifest deliberately measures exact quoted Flask route literals in each
panel's main entrypoint.  Dynamic route construction is reported separately
instead of guessed, keeping the audit deterministic and making its lower-bound
nature explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTE_MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "route_manifest.json"
PARITY_LEDGER_PATH = REPO_ROOT / "extension" / "PANEL_PARITY.json"
CEP_SOURCE_PATH = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "main.js"
UXP_SOURCE_PATH = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "panel_feature_parity.json"

MANIFEST_VERSION = 1
CLASSIFICATIONS = ("uxp-pending", "cep-terminal", "intentional")
_ROUTE_STRING_RE = re.compile(r'''(?P<quote>["'`])(?P<route>/[^"'`\r\n]*)(?P=quote)''')


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def extract_route_literals(path: Path) -> set[str]:
    """Return exact slash-prefixed strings quoted in *path*."""

    source = path.read_text(encoding="utf-8", errors="replace")
    return {match.group("route") for match in _ROUTE_STRING_RE.finditer(source)}


def _source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _real_route_index(route_manifest: dict) -> dict[str, dict]:
    """Index non-stub backend routes by rule for panel matching."""

    index: dict[str, dict] = {}
    for row in route_manifest.get("routes", []):
        if not isinstance(row, dict) or row.get("readiness") == "stub":
            continue
        rule = str(row.get("rule") or "")
        if not rule.startswith("/"):
            continue
        index[rule] = {
            "rule": rule,
            "methods": sorted(str(method) for method in row.get("methods", [])),
            "endpoint": str(row.get("endpoint") or ""),
            "blueprint": str(row.get("blueprint") or ""),
            "readiness": str(row.get("readiness") or "implemented"),
        }
    return index


def _annotation_index(ledger: dict, side: str) -> tuple[dict[str, dict], list[str]]:
    """Expand grouped manual annotations into one record per divergent route."""

    feature_ledger = ledger.get("feature_route_divergences", {})
    groups = feature_ledger.get(side, []) if isinstance(feature_ledger, dict) else []
    errors: list[str] = []
    annotations: dict[str, dict] = {}
    if not isinstance(groups, list):
        return {}, [f"feature_route_divergences.{side} must be an array"]

    for position, group in enumerate(groups):
        label = f"feature_route_divergences.{side}[{position}]"
        if not isinstance(group, dict):
            errors.append(f"{label} must be an object")
            continue
        classification = str(group.get("classification") or "").strip()
        owner = str(group.get("owner") or "").strip()
        justification = str(group.get("justification") or "").strip()
        routes = group.get("routes")
        if classification not in CLASSIFICATIONS:
            errors.append(
                f"{label}.classification must be one of {', '.join(CLASSIFICATIONS)}"
            )
        if not owner:
            errors.append(f"{label}.owner is required")
        if not justification:
            errors.append(f"{label}.justification is required")
        if not isinstance(routes, list) or not routes:
            errors.append(f"{label}.routes must be a non-empty array")
            continue
        for route in routes:
            rule = str(route or "").strip()
            if not rule.startswith("/"):
                errors.append(f"{label} contains invalid route {route!r}")
                continue
            if rule in annotations:
                errors.append(f"{side} route {rule} is annotated more than once")
                continue
            annotations[rule] = {
                "classification": classification,
                "owner": owner,
                "justification": justification,
            }
    return annotations, errors


def validate_feature_annotations(
    cep_only: Iterable[str],
    uxp_only: Iterable[str],
    ledger: dict,
) -> list[str]:
    """Return errors for missing, stale, duplicate, or incomplete annotations."""

    errors: list[str] = []
    live_by_side = {
        "cep_only": set(cep_only),
        "uxp_only": set(uxp_only),
    }
    for side, live in live_by_side.items():
        annotations, annotation_errors = _annotation_index(ledger, side)
        errors.extend(annotation_errors)
        annotated = set(annotations)
        for route in sorted(live - annotated):
            if side == "cep_only":
                errors.append(f"new CEP-only route lacks an owner-assigned annotation: {route}")
            else:
                errors.append(f"UXP-only route lacks an owner-assigned annotation: {route}")
        for route in sorted(annotated - live):
            errors.append(f"stale {side} annotation no longer matches a divergence: {route}")
    return errors


def _annotated_rows(
    rules: Iterable[str],
    route_index: dict[str, dict],
    annotations: dict[str, dict],
) -> list[dict]:
    rows = []
    for rule in sorted(rules):
        row = dict(route_index[rule])
        row.update(annotations.get(rule, {}))
        rows.append(row)
    return rows


def build_manifest(
    *,
    route_manifest_path: Path = ROUTE_MANIFEST_PATH,
    ledger_path: Path = PARITY_LEDGER_PATH,
    cep_source_path: Path = CEP_SOURCE_PATH,
    uxp_source_path: Path = UXP_SOURCE_PATH,
) -> dict:
    """Build the live feature-level parity manifest from repository sources."""

    route_manifest = _load_json(route_manifest_path)
    ledger = _load_json(ledger_path)
    route_index = _real_route_index(route_manifest)
    backend_rules = set(route_index)

    cep_literals = extract_route_literals(cep_source_path)
    uxp_literals = extract_route_literals(uxp_source_path)
    cep_routes = cep_literals & backend_rules
    uxp_routes = uxp_literals & backend_rules
    shared = cep_routes & uxp_routes
    cep_only = cep_routes - uxp_routes
    uxp_only = uxp_routes - cep_routes

    errors = validate_feature_annotations(cep_only, uxp_only, ledger)
    cep_annotations, _ = _annotation_index(ledger, "cep_only")
    uxp_annotations, _ = _annotation_index(ledger, "uxp_only")

    return {
        "manifest_version": MANIFEST_VERSION,
        "source_route_manifest": _source_label(route_manifest_path),
        "source_route_manifest_version": route_manifest.get("version"),
        "panel_sources": {
            "cep": _source_label(cep_source_path),
            "uxp": _source_label(uxp_source_path),
        },
        "scan_contract": {
            "method": "exact quoted slash-prefixed literals intersected with non-stub route-manifest rules",
            "dynamic_routes": "reported as unmatched literals; no dynamic route is inferred",
            "scope": "panel main entrypoints used by the 2026-08-11 F320 research baseline",
        },
        "classifications": list(CLASSIFICATIONS),
        "counts": {
            "cep_route_literals": len(cep_literals),
            "uxp_route_literals": len(uxp_literals),
            "cep_backend_routes": len(cep_routes),
            "uxp_backend_routes": len(uxp_routes),
            "shared_routes": len(shared),
            "cep_only_routes": len(cep_only),
            "uxp_only_routes": len(uxp_only),
        },
        "panels": {
            "cep": {
                "route_count": len(cep_routes),
                "routes": [route_index[rule] for rule in sorted(cep_routes)],
            },
            "uxp": {
                "route_count": len(uxp_routes),
                "routes": [route_index[rule] for rule in sorted(uxp_routes)],
            },
        },
        "shared": [route_index[rule] for rule in sorted(shared)],
        "divergences": {
            "cep_only": _annotated_rows(cep_only, route_index, cep_annotations),
            "uxp_only": _annotated_rows(uxp_only, route_index, uxp_annotations),
        },
        "unmatched_literals": {
            "cep": sorted(cep_literals - backend_rules),
            "uxp": sorted(uxp_literals - backend_rules),
        },
        "gate": {
            "passes": not errors,
            "errors": errors,
            "unannotated_cep_only": sorted(cep_only - set(cep_annotations)),
            "unannotated_uxp_only": sorted(uxp_only - set(uxp_annotations)),
        },
    }


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def diff_manifests(committed: Optional[dict], live: dict) -> list[str]:
    if committed is None:
        return ["committed panel feature-parity manifest is absent"]
    if committed == live:
        return []
    fields = sorted(
        field
        for field in set(committed) | set(live)
        if committed.get(field) != live.get(field)
    )
    return [f"changed fields: {', '.join(fields)}"]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the artifact or annotations drift")
    parser.add_argument("--json", action="store_true", help="emit the live manifest or check result as JSON")
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH, help="artifact path")
    parser.add_argument("--route-manifest", type=Path, default=ROUTE_MANIFEST_PATH)
    parser.add_argument("--ledger", type=Path, default=PARITY_LEDGER_PATH)
    parser.add_argument("--cep-source", type=Path, default=CEP_SOURCE_PATH)
    parser.add_argument("--uxp-source", type=Path, default=UXP_SOURCE_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)

    live = build_manifest(
        route_manifest_path=args.route_manifest,
        ledger_path=args.ledger,
        cep_source_path=args.cep_source,
        uxp_source_path=args.uxp_source,
    )
    gate_errors = live["gate"]["errors"]
    if args.check:
        diff = diff_manifests(load_manifest(args.output), live)
        if args.json:
            print(json.dumps({"diff": diff, "gate_errors": gate_errors, "live": live}, indent=2))
        elif gate_errors or diff:
            print("Panel feature-parity gate failed.")
            for error in gate_errors:
                print(f"  {error}")
            for change in diff:
                print(f"  {change}")
        else:
            counts = live["counts"]
            print(
                "Panel feature-parity manifest in sync "
                f"({counts['shared_routes']} shared, "
                f"{counts['cep_only_routes']} CEP-only, "
                f"{counts['uxp_only_routes']} UXP-only)."
            )
        return 1 if gate_errors or diff else 0

    if gate_errors:
        if args.json:
            print(json.dumps({"gate_errors": gate_errors, "live": live}, indent=2))
        else:
            print("Refusing to write an invalid panel feature-parity manifest:")
            for error in gate_errors:
                print(f"  {error}")
        return 1

    write_manifest(live, args.output)
    if args.json:
        print(json.dumps(live, indent=2, sort_keys=False))
    else:
        counts = live["counts"]
        print(
            f"Wrote {args.output} ({counts['cep_backend_routes']} CEP routes, "
            f"{counts['uxp_backend_routes']} UXP routes)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
