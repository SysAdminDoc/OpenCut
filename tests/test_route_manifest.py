"""Pin the committed route manifest against the live Flask app (F099)."""

from __future__ import annotations

import copy
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


def test_workflow_routes_export_opt_in_metadata(committed_manifest):
    workflow_routes = {
        route["rule"]: route["workflow"]
        for route in committed_manifest["routes"]
        if isinstance(route.get("workflow"), dict)
    }

    assert len(workflow_routes) >= 50
    assert workflow_routes["/silence"]["label"] == "Detecting silence"
    assert workflow_routes["/audio/normalize"]["label"] == "Normalizing audio"
    assert workflow_routes["/export-video"]["label"] == "Exporting video"
    assert "/workflow/run" not in workflow_routes
    assert "/workflow/save" not in workflow_routes


def test_no_route_uses_head_or_options_explicitly(committed_manifest):
    for route in committed_manifest["routes"]:
        for method in route["methods"]:
            assert method not in {"HEAD", "OPTIONS"}, (
                f"manifest should strip HEAD/OPTIONS; saw {method} on {route['rule']}"
            )


def test_manifest_tags_every_route_with_readiness(committed_manifest):
    valid = {"implemented", "dependency-gated", "stub"}
    for route in committed_manifest["routes"]:
        assert route.get("readiness") in valid, (
            f"{route['rule']} has invalid readiness {route.get('readiness')!r}"
        )


def test_manifest_declares_route_surface_classes_and_gate(committed_manifest):
    from opencut.tools.dump_route_manifest import SURFACE_CLASSES, validate_surface_coverage

    surface = committed_manifest["surface_coverage"]
    assert surface["version"] == 1
    assert surface["classes"] == list(SURFACE_CLASSES)
    assert surface["gate"]["passes"] is True
    assert surface["gate"]["unclassified_routes"] == []
    assert validate_surface_coverage(committed_manifest["routes"]) == []

    summary = surface["summary"]
    assert summary["shipped_routes"] == committed_manifest["shipped_route_count"]
    assert summary["direct_surface_routes"] + summary["integration_only_routes"] == summary["shipped_routes"]
    assert summary["integration_only_routes"] > 0
    assert summary["direct_surface_routes"] > 0
    assert all(route.get("surface_class") in SURFACE_CLASSES for route in committed_manifest["routes"])
    assert all(route.get("surface_classes") for route in committed_manifest["routes"])
    assert all(route.get("surface_evidence") for route in committed_manifest["routes"])


def test_surface_gate_rejects_an_unclassified_route():
    from opencut.tools.dump_route_manifest import validate_surface_coverage

    errors = validate_surface_coverage([
        {
            "rule": "/missing-surface",
            "surface_class": "",
            "surface_classes": [],
            "surface_evidence": [],
        }
    ])
    assert errors
    assert any("/missing-surface" in error for error in errors)


def test_readme_explains_backend_route_count_and_reachable_surface(committed_manifest):
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    summary = committed_manifest["surface_coverage"]["summary"]

    assert "literal first-party panel, palette, CLI, and curated MCP references" in readme
    assert f"{summary['direct_surface_routes']:,} shipped routes" in readme
    assert f"{summary['integration_only_routes']:,} integration-only routes" in readme


def test_shipped_count_excludes_stubs(committed_manifest):
    counts = committed_manifest["readiness_counts"]
    stub_count = counts.get("stub", 0)
    assert stub_count >= 1, "expected at least one 501 strategic stub"
    assert (
        committed_manifest["shipped_route_count"]
        == committed_manifest["total_routes"] - stub_count
    )
    assert sum(counts.values()) == committed_manifest["total_routes"]


def test_known_501_stub_routes_are_tagged_stub(committed_manifest):
    by_rule = {r["rule"]: r for r in committed_manifest["routes"]}
    for rule in ("/lipsync/gaussian", "/lipsync/fantasy2",
                 "/generate/cloud/submit", "/video/face-age",
                 "/generate/wan-vace"):
        assert by_rule[rule]["readiness"] == "stub", f"{rule} should be a stub"


def test_real_routes_are_not_tagged_stub(committed_manifest):
    by_rule = {r["rule"]: r for r in committed_manifest["routes"]}
    # Core, fully-implemented routes must never be excluded from the count.
    for rule in ("/silence", "/audio/normalize", "/export-video",
                 "/agent/search-footage", "/agent/storyboard",
                 "/video/trailer/generate", "/video/outpaint"):
        assert by_rule[rule]["readiness"] != "stub", f"{rule} wrongly tagged stub"


def test_route_readiness_endpoint_reports_stubs(client, committed_manifest):
    resp = client.get("/system/route-readiness")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_routes"] == committed_manifest["total_routes"]
    assert data["shipped_route_count"] == committed_manifest["shipped_route_count"]
    assert "/lipsync/gaussian" in data["stub_rules"]
    assert "/silence" not in data["stub_rules"]
    assert "/agent/storyboard" not in data["stub_rules"]
    assert len(data["stub_rules"]) == committed_manifest["readiness_counts"].get("stub", 0)


def test_manifest_diff_catches_route_metadata_changes(committed_manifest):
    from opencut.tools.dump_route_manifest import diff_manifests

    live = copy.deepcopy(committed_manifest)
    for route in live["routes"]:
        if route["rule"] == "/silence":
            route["workflow"]["label"] = "Changed label"
            break
    else:
        raise AssertionError("/silence missing from committed manifest")

    diffs = diff_manifests(committed_manifest, live)

    assert any("changed: POST /silence" in diff and "workflow" in diff for diff in diffs)
