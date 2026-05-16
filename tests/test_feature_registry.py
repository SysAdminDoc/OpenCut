"""Tests for the feature readiness registry + route (F100)."""

from __future__ import annotations

from opencut import registry


def test_states_constant_covers_every_state_used():
    used = {record.state for record in registry.FEATURES.values()}
    assert used.issubset(set(registry.STATES))


def test_no_duplicate_feature_ids():
    ids = [record.feature_id for record in registry.FEATURES.values()]
    assert len(ids) == len(set(ids))


def test_every_record_lists_at_least_one_route():
    for record in registry.FEATURES.values():
        assert record.routes, f"{record.feature_id} has no routes attached"


def test_resolved_state_uses_probe_when_available():
    rec = registry.FeatureRecord(
        feature_id="test.demo",
        label="Demo",
        category="test",
        state=registry.STATE_AVAILABLE,
        probe=lambda: False,
    )
    assert rec.resolved_state() == registry.STATE_MISSING_DEPENDENCY

    rec_ok = registry.FeatureRecord(
        feature_id="test.demo2",
        label="Demo2",
        category="test",
        state=registry.STATE_AVAILABLE,
        probe=lambda: True,
    )
    assert rec_ok.resolved_state() == registry.STATE_AVAILABLE


def test_resolved_state_passes_through_for_non_available_states():
    for state in (
        registry.STATE_STUB,
        registry.STATE_EXPERIMENTAL,
    ):
        rec = registry.FeatureRecord(
            feature_id=f"test.{state}",
            label="x",
            category="test",
            state=state,
        )
        assert rec.resolved_state() == state


def test_manifest_counts_match_features():
    manifest = registry.feature_manifest()

    total = sum(manifest["counts"].values())
    assert total == len(manifest["features"]) == len(registry.FEATURES)
    for state in manifest["counts"]:
        assert state in registry.STATES


def test_manifest_includes_required_fields():
    manifest = registry.feature_manifest()
    sample = manifest["features"][0]

    for field in ("feature_id", "label", "category", "state", "routes"):
        assert field in sample, f"manifest entry missing field {field!r}"


def test_assert_states_valid_rejects_unknown():
    bad = registry.FeatureRecord(
        feature_id="test.bad",
        label="bad",
        category="test",
        state="totally-made-up",
    )
    try:
        registry.assert_states_valid([bad])
    except ValueError as exc:
        assert "totally-made-up" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid state")


def test_route_returns_manifest():
    from opencut.server import create_app

    app = create_app()
    client = app.test_client()
    resp = client.get("/system/feature-state")

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["version"] == 1
    assert "counts" in data and "features" in data
    # `captions.qc` shipped as part of F111 — should resolve to `available`.
    assert any(f["feature_id"] == "captions.qc" and f["state"] == "available" for f in data["features"])
    # Stubs still surface — pick the lipsync rows that remain on the roadmap.
    assert any(f["feature_id"] == "lipsync.latentsync" and f["state"] == "stub" for f in data["features"])


def test_route_single_feature_lookup():
    from opencut.server import create_app

    app = create_app()
    client = app.test_client()
    resp = client.get("/system/feature-state?feature_id=lipsync.latentsync")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["feature_id"] == "lipsync.latentsync"
    assert payload["state"] == "stub"


def test_route_unknown_feature_id_returns_404():
    from opencut.server import create_app

    app = create_app()
    client = app.test_client()
    resp = client.get("/system/feature-state?feature_id=does.not.exist")

    assert resp.status_code == 404
    assert "unknown" in (resp.get_json() or {}).get("error", "").lower()
