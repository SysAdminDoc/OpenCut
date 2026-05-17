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


def test_generator_finds_direct_route_probe_bindings():
    payload = dump_feature_readiness.build_manifest()
    records = {record["feature_id"]: record for record in payload["records"]}

    assert payload["total_records"] >= 50
    assert payload["total_routes"] >= 60
    assert records["auto.gptsovits"]["check_name"] == "check_gptsovits"
    assert "/tts/gptsovits" in records["auto.gptsovits"]["routes"]
    assert records["audio.elevenlabs"]["check_name"] == "check_elevenlabs_available"
    assert "/audio/tts/elevenlabs" in records["audio.elevenlabs"]["routes"]


def test_dumper_check_mode_passes_against_committed_artifact():
    rc = dump_feature_readiness.cli(["--check"])
    assert rc == 0
