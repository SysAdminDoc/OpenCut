"""F328 — the direct-surface ratio is now defended, not just measured.

The manifest has reported the ratio for a while and it only ever went down:
280 of 1,568 shipped routes reachable from a first-party surface, 1,288
integration-only, and no route whose primary surface is the CLI. The gate
asserted every route was *classified*, never that the ratio held, so each wave
could add API faster than product and nothing said so.

These tests mostly exist to prove the gate fails when it should. A ratchet that
only ever passes is the thing it was supposed to replace.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from opencut.core.surface_ratchet import (
    BASELINE_PATH,
    JUSTIFICATIONS,
    MANIFEST_PATH,
    build_baseline,
    coverage_percent,
    evaluate,
    integration_only_counts,
    largest_families,
    load_baseline,
    load_manifest,
    render_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return load_manifest()


@pytest.fixture(scope="module")
def baseline() -> dict:
    recorded = load_baseline()
    assert recorded is not None, "the ratchet baseline must ship with the repo"
    return recorded


def _route(blueprint: str, surface_class: str = "integration-only", rule: str = "/x") -> dict:
    return {"rule": rule, "blueprint": blueprint, "surface_class": surface_class}


def _manifest(routes, percent: float) -> dict:
    return {
        "shipped_route_count": len(routes),
        "routes": routes,
        "surface_coverage": {"summary": {"coverage_percent": percent}},
    }


# ---------------------------------------------------------------------------
# The recorded baseline
# ---------------------------------------------------------------------------


def test_the_committed_baseline_matches_the_committed_manifest(manifest, baseline):
    assert build_baseline(manifest) == baseline


def test_the_repo_passes_its_own_ratchet(manifest, baseline):
    report = evaluate(manifest, baseline)

    assert report["errors"] == []
    assert report["passes"] is True


def test_the_floor_is_the_ratio_that_was_actually_measured(manifest, baseline):
    assert baseline["coverage_floor_percent"] == coverage_percent(manifest)


def test_every_integration_only_family_is_grandfathered(manifest, baseline):
    """Nothing starts out owing a justification; the ratchet begins from today."""
    assert set(baseline["family_ceilings"]) == set(integration_only_counts(manifest))


def test_baseline_ships_in_the_generated_directory():
    assert BASELINE_PATH.is_file()
    assert BASELINE_PATH.parent.name == "_generated"
    assert MANIFEST_PATH.is_file()


# ---------------------------------------------------------------------------
# The gate has to fail
# ---------------------------------------------------------------------------


def test_a_falling_ratio_fails(manifest, baseline):
    dropped = copy.deepcopy(manifest)
    dropped["surface_coverage"]["summary"]["coverage_percent"] = (
        baseline["coverage_floor_percent"] - 0.1
    )

    report = evaluate(dropped, baseline)

    assert report["passes"] is False
    assert any("fell to" in error for error in report["errors"])


def test_a_rising_ratio_passes(manifest, baseline):
    """The gate is a floor, not a pin. Improving must not be a failure."""
    improved = copy.deepcopy(manifest)
    improved["surface_coverage"]["summary"]["coverage_percent"] = (
        baseline["coverage_floor_percent"] + 5.0
    )

    assert evaluate(improved, baseline)["passes"] is True


def test_float_noise_does_not_fail_an_unchanged_ratio(baseline):
    unchanged = _manifest([], baseline["coverage_floor_percent"] - 0.0000001)

    assert not any("fell to" in error for error in evaluate(unchanged, baseline)["errors"])


def test_a_brand_new_unreachable_family_fails(baseline):
    manifest = _manifest(
        [_route("wave_zzz"), _route("wave_zzz", rule="/y")],
        baseline["coverage_floor_percent"],
    )

    report = evaluate(manifest, baseline)

    assert report["passes"] is False
    assert report["new_unjustified_families"] == [
        {"blueprint": "wave_zzz", "integration_only_routes": 2, "recorded": 0}
    ]
    assert "reachable from no first-party surface" in report["errors"][0]


def test_a_grandfathered_family_that_grows_fails(baseline):
    biggest = max(baseline["family_ceilings"].items(), key=lambda item: item[1])
    name, ceiling = biggest
    routes = [_route(name, rule=f"/r{i}") for i in range(ceiling + 1)]

    report = evaluate(_manifest(routes, baseline["coverage_floor_percent"]), baseline)

    assert report["passes"] is False
    assert report["families_over_ceiling"] == [
        {"blueprint": name, "integration_only_routes": ceiling + 1, "recorded": ceiling}
    ]
    assert "grew from" in report["errors"][0]


def test_a_grandfathered_family_that_shrinks_passes(baseline):
    name, ceiling = max(baseline["family_ceilings"].items(), key=lambda item: item[1])
    routes = [_route(name, rule=f"/r{i}") for i in range(ceiling - 1)]

    assert evaluate(_manifest(routes, baseline["coverage_floor_percent"]), baseline)["passes"] is True


def test_giving_the_route_a_surface_is_the_other_way_out(baseline):
    """The gate must not force a justification when the real fix was applied."""
    reachable = _manifest(
        [_route("wave_zzz", surface_class="panel")],
        baseline["coverage_floor_percent"],
    )

    assert evaluate(reachable, baseline)["passes"] is True


def test_a_justified_family_is_allowed_through(baseline, monkeypatch):
    monkeypatch.setitem(JUSTIFICATIONS, "wave_zzz", "Ships for the REST integration lane only.")
    manifest = _manifest([_route("wave_zzz")], baseline["coverage_floor_percent"])

    report = evaluate(manifest, baseline)

    assert report["passes"] is True
    assert "wave_zzz" in report["justified_families"]


def test_a_missing_baseline_is_a_failure_not_a_pass(manifest):
    """Deleting the baseline must not be a way to silence the gate."""
    report = evaluate(manifest, None)

    assert report["passes"] is False
    assert "baseline is missing" in report["errors"][0]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_names_the_largest_families(manifest):
    families = largest_families(manifest, limit=5)

    assert len(families) == 5
    counts = [family["integration_only_routes"] for family in families]
    assert counts == sorted(counts, reverse=True)
    assert all(family["blueprint"] for family in families)


def test_the_report_is_readable_without_the_json(manifest, baseline):
    rendered = render_report(evaluate(manifest, baseline))

    assert "direct-surface coverage" in rendered
    assert "largest integration-only families" in rendered


def test_a_failure_report_says_what_to_do_about_it(baseline):
    report = evaluate(_manifest([_route("wave_zzz")], baseline["coverage_floor_percent"]), baseline)

    assert "Give them a surface" in render_report(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_surface_ratchet", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_check_passes_in_sync():
    result = _cli("--check")

    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_check_is_quiet_when_it_passes():
    assert _cli("--check", "--quiet").stdout.strip() == ""


def test_cli_json_carries_the_whole_report():
    payload = json.loads(_cli("--check", "--json").stdout)

    assert payload["passes"] is True
    assert payload["largest_integration_only_families"]


def _stale_baseline(tmp_path, manifest, **overrides) -> Path:
    recorded = build_baseline(manifest)
    recorded.update(overrides)
    path = tmp_path / "surface_ratchet.json"
    path.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_cli_check_exits_non_zero_when_the_ratio_has_fallen(tmp_path, manifest, baseline):
    """A gate proven only in-process is not proven where it actually runs."""
    stale = _stale_baseline(
        tmp_path, manifest, coverage_floor_percent=baseline["coverage_floor_percent"] + 10.0
    )

    result = _cli("--check", "--baseline", str(stale))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "fell to" in result.stdout
    # The operator needs to know which families to look at, not just that it failed.
    assert "largest integration-only families" in result.stdout


def test_cli_check_exits_non_zero_when_a_family_has_grown(tmp_path, manifest, baseline):
    ceilings = dict(baseline["family_ceilings"])
    name = max(ceilings, key=lambda key: ceilings[key])
    ceilings[name] = 1
    stale = _stale_baseline(tmp_path, manifest, family_ceilings=ceilings)

    result = _cli("--check", "--baseline", str(stale))

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"blueprint '{name}' grew from 1" in result.stdout


def test_cli_check_exits_non_zero_when_the_baseline_is_deleted(tmp_path):
    """Removing the baseline must not be a way to silence the gate."""
    result = _cli("--check", "--baseline", str(tmp_path / "absent.json"))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "baseline is missing" in result.stdout


def test_the_release_gate_runs_the_ratchet():
    source = (REPO_ROOT / "scripts" / "release_smoke.py").read_text(encoding="utf-8")

    assert "dump_surface_ratchet" in source
    assert 'StepDefinition("surface-ratchet"' in source
