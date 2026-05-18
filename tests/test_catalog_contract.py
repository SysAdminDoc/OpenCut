"""F196 contract tests for the feature registry/model-card/check surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from opencut import catalog_contract, model_cards, registry


def _format_issues(issues: list[catalog_contract.CatalogContractIssue]) -> str:
    return "\n  - ".join(str(issue) for issue in issues)


def test_catalog_contract_has_no_drift():
    issues = catalog_contract.validate_catalog_contract()
    assert not issues, "F196 catalogue drift:\n  - " + _format_issues(issues)


def test_every_model_card_feature_id_is_registry_primary():
    features = registry.FEATURES
    missing = [
        f"{card.check_name} -> {card.feature_id}"
        for card in model_cards.CARDS
        if card.feature_id not in features
    ]
    assert not missing, (
        "Model-card feature IDs must be registry-owned before merge:\n  - "
        + "\n  - ".join(missing)
    )


def test_registry_model_card_hardware_matches_feature_manifest():
    manifest = registry.feature_manifest()
    features = {record["feature_id"]: record for record in manifest["features"]}
    mismatches = []
    for card in model_cards.CARDS:
        feature = features[card.feature_id]
        if feature["hardware"] != card.hardware:
            mismatches.append(
                f"{card.feature_id}: registry={feature['hardware']!r} card={card.hardware!r}"
            )
    assert not mismatches, (
        "Model-card hardware must flow through /system/feature-state:\n  - "
        + "\n  - ".join(mismatches)
    )


@dataclass
class _FakeCard:
    check_name: str
    feature_id: str
    label: str = "Fake"
    hardware: str = "gpu (>= 8 GB VRAM)"


def test_catalog_contract_reports_missing_registry_rows():
    issues = catalog_contract.validate_catalog_contract(
        cards=[_FakeCard("check_demucs_available", "missing.demucs")],
        features=[registry.FEATURES["audio.demucs"]],
        public_checks=["check_demucs_available"],
        non_ai_checks=[],
    )
    assert any(issue.code == "model_card_missing_registry_feature" for issue in issues)


def test_catalog_contract_reports_hardware_mismatch():
    feature = registry.FeatureRecord(
        feature_id="audio.demucs",
        label="Demucs",
        category="audio",
        state=registry.STATE_AVAILABLE,
        routes=["/audio/separate"],
        probe=lambda: True,
        hardware="cpu",
    )
    feature.probe.__name__ = "check_demucs_available"
    issues = catalog_contract.validate_catalog_contract(
        cards=[_FakeCard("check_demucs_available", "audio.demucs")],
        features=[feature],
        public_checks=["check_demucs_available"],
        non_ai_checks=[],
    )
    codes = {issue.code for issue in issues}
    assert "model_card_registry_hardware_mismatch" in codes
    assert "model_card_registry_gpu_mismatch" in codes
    assert "model_card_registry_vram_mismatch" in codes
