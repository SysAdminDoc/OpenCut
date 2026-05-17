"""Generate the canonical route/feature manifest (F099).

The README and roadmap quote route counts that drift with every release.
This module is the single source of truth: it walks
:class:`~flask.Flask.url_map`, categorises by blueprint, and writes a
deterministic JSON manifest to ``opencut/_generated/route_manifest.json``.

Use it three ways:

* ``python -m opencut.tools.dump_route_manifest`` — rewrites the manifest.
* ``python -m opencut.tools.dump_route_manifest --check`` — fails when
  the committed manifest disagrees with the live Flask app (used in CI).
* ``opencut.tools.dump_route_manifest.build_manifest()`` — Python API
  used by tests and other generators.

The manifest is keyed by route rule so re-runs are stable across Python
dict orderings, and methods are sorted alphabetically with the standard
``HEAD``/``OPTIONS`` Flask additions stripped.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "route_manifest.json"
MANIFEST_VERSION = 1

_STD_METHODS = {"HEAD", "OPTIONS"}


@dataclass
class RouteEntry:
    rule: str
    methods: List[str]
    endpoint: str
    blueprint: str

    def as_tuple(self):
        return (self.rule, tuple(self.methods), self.endpoint, self.blueprint)


@dataclass
class Manifest:
    version: int
    generated_at: str
    total_routes: int
    blueprint_count: int
    method_counts: Dict[str, int]
    blueprints: Dict[str, Dict[str, object]]
    routes: List[Dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "total_routes": self.total_routes,
            "blueprint_count": self.blueprint_count,
            "method_counts": dict(sorted(self.method_counts.items())),
            "blueprints": {
                name: {
                    "route_count": meta["route_count"],
                    "method_counts": dict(sorted(meta["method_counts"].items())),
                    "sample_rules": meta["sample_rules"],
                }
                for name, meta in sorted(self.blueprints.items())
            },
            "routes": sorted(self.routes, key=lambda r: (r["rule"], tuple(r["methods"]))),
        }


def _collect_routes(app) -> List[RouteEntry]:
    entries: List[RouteEntry] = []
    for rule in app.url_map.iter_rules():
        methods = sorted(m for m in rule.methods if m not in _STD_METHODS)
        if not methods:
            continue
        endpoint = rule.endpoint
        blueprint = endpoint.split(".", 1)[0] if "." in endpoint else "<app>"
        entries.append(
            RouteEntry(
                rule=rule.rule,
                methods=methods,
                endpoint=endpoint,
                blueprint=blueprint,
            )
        )
    return entries


def _summarise(entries: Iterable[RouteEntry]) -> Manifest:
    method_counts: Counter = Counter()
    by_blueprint: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"route_count": 0, "method_counts": Counter(), "sample_rules": []}
    )
    routes_payload: List[Dict[str, object]] = []

    for entry in entries:
        for method in entry.methods:
            method_counts[method] += 1
        bucket = by_blueprint[entry.blueprint]
        bucket["route_count"] += 1
        bucket["method_counts"].update(entry.methods)
        if len(bucket["sample_rules"]) < 3:
            bucket["sample_rules"].append(entry.rule)
        routes_payload.append(
            {
                "rule": entry.rule,
                "methods": list(entry.methods),
                "endpoint": entry.endpoint,
                "blueprint": entry.blueprint,
            }
        )

    blueprints_payload = {
        name: {
            "route_count": meta["route_count"],
            "method_counts": dict(meta["method_counts"]),
            "sample_rules": sorted(meta["sample_rules"]),
        }
        for name, meta in by_blueprint.items()
    }

    return Manifest(
        version=MANIFEST_VERSION,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_routes=len(routes_payload),
        blueprint_count=len(blueprints_payload),
        method_counts=dict(method_counts),
        blueprints=blueprints_payload,
        routes=routes_payload,
    )


def build_manifest(app=None) -> dict:
    """Return the manifest dict computed from a live Flask app.

    ``app`` defaults to a freshly created OpenCut server so call sites
    such as tests can pass in a stripped-down fixture for unit tests.
    """
    if app is None:
        # Import lazily so unit tests can avoid the full server boot.
        from opencut.server import create_app

        app = create_app()

    entries = _collect_routes(app)
    return _summarise(entries).as_dict()


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_signature(manifest: dict) -> dict:
    """Drop the timestamp so diff checks don't rebuild on every run."""
    return {k: v for k, v in manifest.items() if k != "generated_at"}


def diff_manifests(expected: dict, live: dict) -> List[str]:
    """Return human-readable diff lines, empty when identical (ignoring time)."""
    diffs: List[str] = []
    expected_sig = _manifest_signature(expected)
    live_sig = _manifest_signature(live)

    for key in ("version", "total_routes", "blueprint_count"):
        if expected_sig.get(key) != live_sig.get(key):
            diffs.append(
                f"{key}: committed={expected_sig.get(key)!r} live={live_sig.get(key)!r}"
            )

    expected_routes = {(r["rule"], tuple(r["methods"])) for r in expected_sig.get("routes", [])}
    live_routes = {(r["rule"], tuple(r["methods"])) for r in live_sig.get("routes", [])}
    only_in_committed = sorted(expected_routes - live_routes)
    only_in_live = sorted(live_routes - expected_routes)
    for rule, methods in only_in_committed:
        diffs.append(f"removed: {','.join(methods)} {rule}")
    for rule, methods in only_in_live:
        diffs.append(f"new:     {','.join(methods)} {rule}")

    if expected_sig.get("blueprints", {}) != live_sig.get("blueprints", {}):
        expected_bps = set(expected_sig.get("blueprints", {}).keys())
        live_bps = set(live_sig.get("blueprints", {}).keys())
        for name in sorted(expected_bps - live_bps):
            diffs.append(f"blueprint removed: {name}")
        for name in sorted(live_bps - expected_bps):
            diffs.append(f"blueprint added: {name}")

    return diffs


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed manifest is out of sync with the live app",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress info logging",
    )
    args = parser.parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    manifest = build_manifest()

    if args.check:
        existing = load_manifest()
        if existing is None:
            print("[route-manifest] FAIL — no committed manifest at "
                  f"{MANIFEST_PATH.relative_to(REPO_ROOT)} (run without --check first)")
            return 1
        diffs = diff_manifests(existing, manifest)
        if diffs:
            print("[route-manifest] FAIL — committed manifest is out of sync with live app:")
            for line in diffs[:25]:
                print(f"  - {line}")
            if len(diffs) > 25:
                print(f"  - ... {len(diffs) - 25} more diff lines (re-run without --check to rebuild)")
            return 1
        print(
            f"[route-manifest] OK — {manifest['total_routes']} routes across "
            f"{manifest['blueprint_count']} blueprints"
        )
        return 0

    out = write_manifest(manifest)
    print(
        f"[route-manifest] wrote {out.relative_to(REPO_ROOT)} — "
        f"{manifest['total_routes']} routes across {manifest['blueprint_count']} blueprints"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
