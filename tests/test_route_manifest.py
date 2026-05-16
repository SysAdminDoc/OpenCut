"""Pin the committed route manifest against the live Flask app (F099)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "opencut" / "_generated" / "route_manifest.json"


@pytest.fixture(scope="module")
def committed_manifest() -> dict:
    assert MANIFEST_PATH.exists(), (
        "opencut/_generated/route_manifest.json missing; run "
        "`python -m opencut.tools.dump_route_manifest` to regenerate"
    )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_manifest() -> dict:
    from opencut.tools.dump_route_manifest import build_manifest

    return build_manifest()


def test_committed_manifest_matches_live_app(committed_manifest, live_manifest):
    from opencut.tools.dump_route_manifest import diff_manifests

    diffs = diff_manifests(committed_manifest, live_manifest)

    assert not diffs, (
        "route_manifest.json is out of sync with the live Flask app.\n"
        "Run `python -m opencut.tools.dump_route_manifest` and commit the result.\n"
        "First diffs:\n  - " + "\n  - ".join(diffs[:10])
    )


def test_committed_manifest_has_sane_totals(committed_manifest):
    assert committed_manifest["version"] >= 1
    assert committed_manifest["total_routes"] >= 1000, "we have thousands of routes; smoke check"
    assert committed_manifest["blueprint_count"] >= 50
    assert committed_manifest["method_counts"].get("GET", 0) > 0
    assert committed_manifest["method_counts"].get("POST", 0) > 0


def test_blueprint_summaries_round_trip(committed_manifest):
    for name, summary in committed_manifest["blueprints"].items():
        assert summary["route_count"] >= 1, f"blueprint {name} reports zero routes"
        assert isinstance(summary["method_counts"], dict)
        assert isinstance(summary["sample_rules"], list)


def test_routes_are_sorted_and_unique(committed_manifest):
    rules_and_methods = [
        (route["rule"], tuple(route["methods"]))
        for route in committed_manifest["routes"]
    ]
    assert rules_and_methods == sorted(rules_and_methods), "routes must be sorted"
    assert len(rules_and_methods) == len(set(rules_and_methods)), "duplicate route entries"


def test_no_route_uses_head_or_options_explicitly(committed_manifest):
    for route in committed_manifest["routes"]:
        for method in route["methods"]:
            assert method not in {"HEAD", "OPTIONS"}, (
                f"manifest should strip HEAD/OPTIONS; saw {method} on {route['rule']}"
            )
