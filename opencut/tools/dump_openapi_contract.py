"""Generate the OpenAPI typed-contract manifest and coverage ratchet.

The spec itself is far too large to review in a diff, but the thing that
matters — how many operations carry a typed request, a typed success schema,
typed errors, and documented CSRF — is a handful of numbers. This module
commits those numbers so a refactor that quietly untypes half the surface
fails a test instead of shipping.

Use it three ways:

* ``python -m opencut.tools.dump_openapi_contract`` — rewrite the manifest.
* ``python -m opencut.tools.dump_openapi_contract --check`` — fail when the
  committed manifest disagrees with the live app, or when coverage regressed.
* ``build_manifest()`` — the Python API used by tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "openapi_contract.json"
MANIFEST_VERSION = 1

#: Counters that must never go down. Everything else in the manifest is
#: informational and may move freely.
RATCHET_KEYS = (
    "typed_requests",
    "typed_responses",
    "error_typed_operations",
    "csrf_documented_operations",
    "component_schemas",
)


def _live_app():
    from opencut.server import create_app
    return create_app()


def build_manifest(app=None) -> dict:
    """Build the contract manifest from the live Flask app."""
    from opencut.core import openapi_source

    app = app or _live_app()
    spec = openapi_source.build_spec(app)
    coverage = openapi_source.contract_coverage(spec)

    typed_request_ops: List[str] = []
    typed_response_ops: List[str] = []
    for path, path_item in sorted(spec["paths"].items()):
        for method, operation in sorted(path_item.items()):
            if method == "parameters" or not isinstance(operation, dict):
                continue
            key = f"{method.upper()} {path}"
            if operation.get("requestBody") or any(
                param.get("in") == "query"
                for param in operation.get("parameters", [])
            ):
                typed_request_ops.append(key)
            success = (
                operation.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            if "$ref" in success:
                typed_response_ops.append(key)

    downgraded = openapi_source.downgrade_to_30(spec)

    return {
        "version": MANIFEST_VERSION,
        "canonical_openapi": spec["openapi"],
        "compatibility_openapi": downgraded["openapi"],
        "json_schema_dialect": spec["jsonSchemaDialect"],
        "coverage": {key: coverage[key] for key in sorted(coverage) if key != "openapi"},
        "component_schema_names": sorted(spec["components"]["schemas"]),
        "typed_request_operations": sorted(typed_request_ops),
        "typed_response_operations": sorted(typed_response_ops),
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def diff_manifests(committed: dict, live: dict) -> List[str]:
    """Human-readable differences, ratchet violations first."""
    diffs: List[str] = []
    committed_coverage: Dict[str, int] = committed.get("coverage", {})
    live_coverage: Dict[str, int] = live.get("coverage", {})

    for key in RATCHET_KEYS:
        before = int(committed_coverage.get(key, 0))
        after = int(live_coverage.get(key, 0))
        if after < before:
            diffs.append(f"RATCHET: {key} fell from {before} to {after}")

    for field in ("canonical_openapi", "compatibility_openapi", "json_schema_dialect"):
        if committed.get(field) != live.get(field):
            diffs.append(
                f"{field}: committed={committed.get(field)!r} live={live.get(field)!r}"
            )

    for key in sorted(set(committed_coverage) | set(live_coverage)):
        if committed_coverage.get(key) != live_coverage.get(key):
            diffs.append(
                f"coverage.{key}: committed={committed_coverage.get(key)} "
                f"live={live_coverage.get(key)}"
            )

    for field in ("component_schema_names", "typed_request_operations",
                  "typed_response_operations"):
        before = set(committed.get(field) or [])
        after = set(live.get(field) or [])
        for name in sorted(before - after)[:5]:
            diffs.append(f"{field}: removed {name}")
        for name in sorted(after - before)[:5]:
            diffs.append(f"{field}: added {name}")

    return diffs


def write_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dump_openapi_contract",
        description="Write or verify the OpenAPI typed-contract manifest.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero when the committed manifest is stale or coverage regressed.",
    )
    args = parser.parse_args(argv)

    live = build_manifest()
    if not args.check:
        write_manifest(live)
        coverage = live["coverage"]
        print(
            f"[openapi-contract] wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} — "
            f"{coverage['operations']} operations, "
            f"{coverage['typed_requests']} typed requests, "
            f"{coverage['typed_responses']} typed responses"
        )
        return 0

    if not MANIFEST_PATH.is_file():
        print(f"[openapi-contract] missing manifest at {MANIFEST_PATH}", file=sys.stderr)
        return 1

    diffs = diff_manifests(load_manifest(), live)
    if diffs:
        print("[openapi-contract] manifest is out of sync:", file=sys.stderr)
        for line in diffs[:20]:
            print(f"  - {line}", file=sys.stderr)
        print(
            "Run `python -m opencut.tools.dump_openapi_contract` and commit the result.",
            file=sys.stderr,
        )
        return 1

    print("[openapi-contract] manifest in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
