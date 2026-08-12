"""F320 feature-level CEP/UXP backend-route parity guardrails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from opencut.tools import dump_panel_feature_parity as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "opencut" / "_generated" / "panel_feature_parity.json"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _route_rules(rows: list[dict]) -> set[str]:
    return {str(row["rule"]) for row in rows}


def test_generated_feature_manifest_is_in_sync_and_enumerates_the_audit_gap():
    live = tool.build_manifest()
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert committed == live
    assert live["gate"] == {
        "passes": True,
        "errors": [],
        "unannotated_cep_only": [],
        "unannotated_uxp_only": [],
    }
    assert live["counts"] == {
        "cep_route_literals": 209,
        "uxp_route_literals": 90,
        "cep_backend_routes": 185,
        "uxp_backend_routes": 80,
        "shared_routes": 61,
        "cep_only_routes": 124,
        "uxp_only_routes": 19,
    }


def test_manifest_lists_every_matched_panel_route_and_classifies_every_divergence():
    manifest = tool.build_manifest()
    cep_routes = _route_rules(manifest["panels"]["cep"]["routes"])
    uxp_routes = _route_rules(manifest["panels"]["uxp"]["routes"])
    cep_only = manifest["divergences"]["cep_only"]
    uxp_only = manifest["divergences"]["uxp_only"]

    assert _route_rules(cep_only) == cep_routes - uxp_routes
    assert _route_rules(uxp_only) == uxp_routes - cep_routes
    assert _route_rules(manifest["shared"]) == cep_routes & uxp_routes
    for row in [*cep_only, *uxp_only]:
        assert row["classification"] in tool.CLASSIFICATIONS
        assert row["owner"].strip()
        assert row["justification"].strip()


def test_new_cep_only_route_fails_until_an_owner_classifies_it():
    ledger = {
        "feature_route_divergences": {
            "cep_only": [
                {
                    "classification": "uxp-pending",
                    "owner": "timeline",
                    "justification": "Known migration work.",
                    "routes": ["/known"],
                }
            ],
            "uxp_only": [],
        }
    }

    errors = tool.validate_feature_annotations({"/known", "/new"}, set(), ledger)

    assert "new CEP-only route lacks an owner-assigned annotation: /new" in errors


def test_any_unannotated_or_stale_divergence_fails_the_gate():
    ledger = {
        "feature_route_divergences": {
            "cep_only": [],
            "uxp_only": [
                {
                    "classification": "intentional",
                    "owner": "agent",
                    "justification": "UXP-first surface.",
                    "routes": ["/stale"],
                }
            ],
        }
    }

    errors = tool.validate_feature_annotations(set(), {"/unannotated"}, ledger)

    assert "UXP-only route lacks an owner-assigned annotation: /unannotated" in errors
    assert "stale uxp_only annotation no longer matches a divergence: /stale" in errors


def test_feature_parity_cli_check_passes_in_sync():
    result = subprocess.run(
        [sys.executable, "-m", "opencut.tools.dump_panel_feature_parity", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "124 CEP-only" in result.stdout


def test_release_smoke_runs_feature_parity_gate():
    source = RELEASE_SMOKE.read_text(encoding="utf-8")

    assert "step_panel_feature_parity" in source
    assert '"panel-feature-parity"' in source
    assert "opencut.tools.dump_panel_feature_parity" in source
