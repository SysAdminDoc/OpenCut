"""Tests for the generated F191 feature readiness manifest."""

from __future__ import annotations

import json
from pathlib import Path

from opencut.tools import dump_feature_readiness

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = REPO_ROOT / "opencut" / "_generated" / "feature_readiness.json"


def test_generated_feature_readiness_matches_committed_json():
    payload = dump_feature_readiness.build_manifest()
    committed = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    assert dump_feature_readiness.diff_manifests(committed, payload) == []


def test_generated_feature_readiness_has_no_dead_roadmap_anchors():
    payload = dump_feature_readiness.build_manifest()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert "roadmap H" not in encoded
    assert "Roadmap stub" not in encoded


def test_generator_finds_direct_route_probe_bindings():
    payload = dump_feature_readiness.build_manifest()
    records = {record["feature_id"]: record for record in payload["records"]}

    assert payload["total_records"] >= 50
    assert payload["total_routes"] >= 60
    assert records["auto.gptsovits"]["check_name"] == "check_gptsovits"
    assert "/tts/gptsovits" in records["auto.gptsovits"]["routes"]
    assert records["audio.elevenlabs"]["check_name"] == "check_elevenlabs_available"
    assert "/audio/tts/elevenlabs" in records["audio.elevenlabs"]["routes"]


def test_route_probe_scan_resolves_imported_blueprint_contracts(tmp_path):
    route = tmp_path / "purpose_routes.py"
    route.write_text(
        "from .contract import sample_bp\n"
        "@sample_bp.route('/sample')\n"
        "def sample():\n"
        "    return check_sample_available()\n",
        encoding="utf-8",
    )

    bindings = dump_feature_readiness.route_endpoint_probes(
        tmp_path,
        aliases={"check_sample_available": "check_sample"},
    )

    assert bindings == {"sample.sample": ["check_sample"]}


def test_generator_carries_model_card_hardware_requirements():
    payload = dump_feature_readiness.build_manifest()
    records = {record["feature_id"]: record for record in payload["records"]}

    flashvsr = records["video.upscale.flashvsr"]
    assert flashvsr["hardware"] == "gpu (>= 12 GB VRAM)"
    assert flashvsr["requires_gpu"] is True
    assert flashvsr["minimum_vram_mb"] == 12288


def test_generator_suppresses_unsupported_dependency_commands():
    payload = dump_feature_readiness.build_manifest()
    records = {record["feature_id"]: record for record in payload["records"]}

    hint = records["audio.resemble-enhance"]["install_hint"]
    assert "Unavailable in OpenCut's supported dependency matrix" in hint
    assert "pip install" not in hint


def test_dependency_gated_routes_produce_non_available_features():
    payload = dump_feature_readiness.build_manifest()
    records = payload["records"]
    for record in records:
        if record["state"] == "available":
            continue
        assert record["state"] in ("stub", "missing_dependency"), (
            f"{record['feature_id']} has unexpected state {record['state']!r}"
        )

    non_available = [r for r in records if r["state"] != "available"]
    assert len(non_available) > 0, "expected at least one non-available generated record"


def test_derive_feature_state_logic():
    from opencut.tools.dump_feature_readiness import _derive_feature_state

    readiness = {
        "/a": "stub",
        "/b": "dependency-gated",
        "/c": "implemented",
    }
    assert _derive_feature_state(["/a"], readiness) == "stub"
    assert _derive_feature_state(["/a", "/b"], readiness) == "missing_dependency"
    assert _derive_feature_state(["/a", "/c"], readiness) == "available"
    assert _derive_feature_state(["/c"], readiness) == "available"
    assert _derive_feature_state(["/b"], readiness) == "missing_dependency"
    assert _derive_feature_state([], readiness) == "available"


def test_dumper_check_mode_passes_against_committed_artifact():
    rc = dump_feature_readiness.cli(["--check"])
    assert rc == 0
