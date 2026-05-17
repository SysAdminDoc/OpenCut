"""Generate the /api route alias manifest (F199).

The route manifest is the source of truth for all Flask routes. This tool
derives a smaller policy file that answers one narrow question: which
``/api/*`` routes are compatibility aliases for an equivalent bare route, and
which ``/api/*`` routes are canonical API routes with no bare counterpart?

Use it three ways:

* ``python -m opencut.tools.dump_api_aliases`` rewrites
  ``opencut/_generated/api_aliases.json``.
* ``python -m opencut.tools.dump_api_aliases --check`` fails when the committed
  alias manifest disagrees with the live Flask app.
* ``build_alias_manifest()`` returns the manifest dict for tests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from opencut.tools.dump_route_manifest import build_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIAS_MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "api_aliases.json"
ALIAS_MANIFEST_VERSION = 1


def _route_key(route: dict) -> Tuple[str, Tuple[str, ...]]:
    return route["rule"], tuple(route["methods"])


def build_alias_manifest(route_manifest: Optional[dict] = None) -> dict:
    """Return the /api alias policy derived from a route manifest."""
    manifest = route_manifest or build_manifest()
    routes = manifest.get("routes", [])
    by_rule_methods = {_route_key(route): route for route in routes}

    aliases: List[dict] = []
    api_only: List[dict] = []
    method_counts: Counter = Counter()

    for route in sorted(routes, key=lambda r: (r["rule"], tuple(r["methods"]))):
        rule = route["rule"]
        if not rule.startswith("/api/"):
            continue

        method_counts.update(route["methods"])
        canonical_rule = rule[4:]
        canonical = by_rule_methods.get((canonical_rule, tuple(route["methods"])))
        if canonical:
            aliases.append(
                {
                    "alias_rule": rule,
                    "canonical_rule": canonical_rule,
                    "methods": route["methods"],
                    "alias_endpoint": route["endpoint"],
                    "canonical_endpoint": canonical["endpoint"],
                    "alias_blueprint": route["blueprint"],
                    "canonical_blueprint": canonical["blueprint"],
                }
            )
        else:
            api_only.append(
                {
                    "rule": rule,
                    "methods": route["methods"],
                    "endpoint": route["endpoint"],
                    "blueprint": route["blueprint"],
                    "canonical_candidate": canonical_rule,
                }
            )

    return {
        "version": ALIAS_MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "opencut/_generated/route_manifest.json",
        "total_api_routes": len(aliases) + len(api_only),
        "alias_count": len(aliases),
        "api_only_count": len(api_only),
        "method_counts": dict(sorted(method_counts.items())),
        "policy": {
            "alias_definition": (
                "A route is an alias only when an /api/* rule has the same methods "
                "as an equivalent bare rule after stripping the /api prefix."
            ),
            "api_only_definition": (
                "An /api/* route without an equivalent bare rule is treated as a "
                "canonical API route, not a compatibility alias."
            ),
        },
        "aliases": aliases,
        "api_only": api_only,
    }


def write_alias_manifest(manifest: dict, path: Path = ALIAS_MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_alias_manifest(path: Path = ALIAS_MANIFEST_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _signature(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "generated_at"}


def diff_alias_manifests(expected: dict, live: dict) -> List[str]:
    expected_sig = _signature(expected)
    live_sig = _signature(live)
    if expected_sig == live_sig:
        return []

    diffs: List[str] = []
    for key in ("version", "total_api_routes", "alias_count", "api_only_count", "method_counts"):
        if expected_sig.get(key) != live_sig.get(key):
            diffs.append(f"{key}: committed={expected_sig.get(key)!r} live={live_sig.get(key)!r}")

    expected_aliases = {
        (item["alias_rule"], item["canonical_rule"], tuple(item["methods"]))
        for item in expected_sig.get("aliases", [])
    }
    live_aliases = {
        (item["alias_rule"], item["canonical_rule"], tuple(item["methods"]))
        for item in live_sig.get("aliases", [])
    }
    for alias_rule, canonical_rule, methods in sorted(expected_aliases - live_aliases):
        diffs.append(f"alias removed: {','.join(methods)} {alias_rule} -> {canonical_rule}")
    for alias_rule, canonical_rule, methods in sorted(live_aliases - expected_aliases):
        diffs.append(f"alias added: {','.join(methods)} {alias_rule} -> {canonical_rule}")

    expected_api_only = {
        (item["rule"], tuple(item["methods"]))
        for item in expected_sig.get("api_only", [])
    }
    live_api_only = {
        (item["rule"], tuple(item["methods"]))
        for item in live_sig.get("api_only", [])
    }
    for rule, methods in sorted(expected_api_only - live_api_only):
        diffs.append(f"api-only removed: {','.join(methods)} {rule}")
    for rule, methods in sorted(live_api_only - expected_api_only):
        diffs.append(f"api-only added: {','.join(methods)} {rule}")

    return diffs


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed alias manifest is out of sync with the live app",
    )
    args = parser.parse_args(argv)

    manifest = build_alias_manifest()

    if args.check:
        existing = load_alias_manifest()
        if existing is None:
            print(
                "[api-aliases] FAIL - no committed manifest at "
                f"{ALIAS_MANIFEST_PATH.relative_to(REPO_ROOT)} (run without --check first)"
            )
            return 1
        diffs = diff_alias_manifests(existing, manifest)
        if diffs:
            print("[api-aliases] FAIL - committed manifest is out of sync with live app:")
            for line in diffs[:25]:
                print(f"  - {line}")
            if len(diffs) > 25:
                print(f"  - ... {len(diffs) - 25} more diff lines")
            return 1
        print(
            "[api-aliases] OK - "
            f"{manifest['alias_count']} aliases, {manifest['api_only_count']} canonical /api routes"
        )
        return 0

    out = write_alias_manifest(manifest)
    print(
        f"[api-aliases] wrote {out.relative_to(REPO_ROOT)} - "
        f"{manifest['alias_count']} aliases, {manifest['api_only_count']} canonical /api routes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
