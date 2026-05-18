"""F176 — public eval-dataset catalogue tests.

The registry doesn't download anything; it documents the public
datasets OpenCut benchmarks against. These tests pin:

* Schema invariants (unique IDs, http(s) upstream URLs, known
  modalities, every entry carries a license).
* The auto/manual acquisition split.
* The two routes that surface the registry (`/system/eval-datasets`
  list + filter, `/system/eval-datasets/<id>` detail + 404).
* The commercial-use-ok flag and the OPENCUT_DOWNLOAD_EVAL opt-in
  gate.
"""

from __future__ import annotations

import json
from typing import Iterable

import pytest

from opencut.core import eval_datasets as ed


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_has_entries():
    assert len(ed.DATASETS) >= 10, (
        f"F176 catalogue must list at least 10 datasets; got {len(ed.DATASETS)}"
    )


def test_registry_invariants_hold():
    ed.assert_registry_invariants()


def test_every_modality_is_documented():
    for d in ed.DATASETS:
        assert d.modality in ed.MODALITIES, (
            f"{d.dataset_id}: modality {d.modality!r} not in MODALITIES"
        )


def test_every_upstream_is_http_url():
    for d in ed.DATASETS:
        assert d.upstream.startswith(("http://", "https://")), (
            f"{d.dataset_id}: upstream must be an http(s) URL; got {d.upstream!r}"
        )


def test_dataset_ids_are_unique_and_snake_case():
    seen = set()
    for d in ed.DATASETS:
        assert d.dataset_id not in seen, f"duplicate id {d.dataset_id!r}"
        seen.add(d.dataset_id)
        assert d.dataset_id.replace("_", "").isalnum()
        assert d.dataset_id == d.dataset_id.lower(), (
            f"{d.dataset_id!r} must be snake_case"
        )


def test_acquisition_is_auto_or_manual():
    for d in ed.DATASETS:
        assert d.acquisition in {"auto", "manual"}, (
            f"{d.dataset_id}: invalid acquisition {d.acquisition!r}"
        )


def test_non_commercial_datasets_never_auto_download():
    """The download runner must never grab NC-licensed assets without
    explicit operator action."""
    for d in ed.DATASETS:
        if d.acquisition == "auto":
            assert d.commercial_use_ok, (
                f"{d.dataset_id}: auto-acquisition is reserved for "
                f"commercial-safe corpora; flip acquisition='manual' "
                f"or document that the license allows commercial use."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_get_dataset_returns_known_id():
    entry = ed.get_dataset("davis_2017")
    assert entry is not None
    assert entry.modality == "video"


def test_get_dataset_returns_none_for_unknown_id():
    assert ed.get_dataset("does-not-exist") is None


def test_datasets_for_target_finds_t2v():
    rows = ed.datasets_for_target("t2v")
    assert any(d.dataset_id == "vbench" for d in rows)


def test_datasets_for_modality_filters():
    speech = ed.datasets_for_modality("speech")
    assert speech
    assert all(d.modality == "speech" for d in speech)


def test_commercial_safe_subset_excludes_nc():
    commercial = ed.commercial_safe_datasets()
    # MUSDB18-HQ is NC-licensed; must be excluded.
    assert all(d.dataset_id != "musdb18_hq" for d in commercial)
    # DAVIS 2017 is commercial-safe.
    assert any(d.dataset_id == "davis_2017" for d in commercial)


def test_manifest_includes_summary_fields():
    payload = ed.manifest()
    assert payload["version"] == 1
    assert payload["count"] == len(ed.DATASETS)
    assert payload["modalities"] == list(ed.MODALITIES)
    assert isinstance(payload["datasets"], list)
    assert payload["auto_download_count"] >= 1
    assert payload["commercial_safe_count"] >= 1


def test_manifest_compact_strips_verbose_fields():
    compact = ed.manifest(include_metadata=False)
    first = compact["datasets"][0]
    assert "dataset_id" in first
    assert "citation" not in first  # stripped in compact mode
    assert "sha256" not in first


def test_download_opt_in_reads_env(monkeypatch):
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    assert ed.download_opt_in() is True
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "yes")
    assert ed.download_opt_in() is True
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "")
    assert ed.download_opt_in() is False
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "off")
    assert ed.download_opt_in() is False


# ---------------------------------------------------------------------------
# Schema-violation guards (invariant assertions)
# ---------------------------------------------------------------------------


def test_assert_registry_invariants_rejects_duplicate_ids():
    dup = [
        ed.EvalDataset(
            dataset_id="x", label="A", modality="video",
            benchmark_targets=("t",), upstream="https://example.com",
        ),
        ed.EvalDataset(
            dataset_id="x", label="B", modality="video",
            benchmark_targets=("t",), upstream="https://example.com",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        ed.assert_registry_invariants(dup)


def test_assert_registry_invariants_rejects_bad_modality():
    bogus: Iterable = [
        ed.EvalDataset(
            dataset_id="x", label="A", modality="cosmos",
            benchmark_targets=("t",), upstream="https://example.com",
        ),
    ]
    with pytest.raises(ValueError, match="unknown modality"):
        ed.assert_registry_invariants(bogus)


def test_assert_registry_invariants_rejects_non_http_upstream():
    bogus = [
        ed.EvalDataset(
            dataset_id="x", label="A", modality="video",
            benchmark_targets=("t",), upstream="ftp://example.com",
        ),
    ]
    with pytest.raises(ValueError, match="http"):
        ed.assert_registry_invariants(bogus)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    from opencut.server import _get_app

    app = _get_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_route_list_returns_all_datasets(client):
    resp = client.get("/system/eval-datasets")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == len(ed.DATASETS)
    assert isinstance(body["datasets"], list)
    assert body["modalities"] == list(ed.MODALITIES)
    assert isinstance(body["download_opt_in"], bool)


def test_route_filter_by_modality(client):
    resp = client.get("/system/eval-datasets?modality=video")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] >= 1
    assert all(d["modality"] == "video" for d in body["datasets"])


def test_route_filter_by_target(client):
    resp = client.get("/system/eval-datasets?target=lip_sync")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] >= 1
    assert any(d["dataset_id"] == "lrs3" for d in body["datasets"])


def test_route_filter_commercial_only(client):
    resp = client.get("/system/eval-datasets?commercial_only=true")
    assert resp.status_code == 200
    body = resp.get_json()
    assert all(d["commercial_use_ok"] for d in body["datasets"])


def test_route_compact_mode_strips_verbose_fields(client):
    resp = client.get("/system/eval-datasets?compact=true")
    assert resp.status_code == 200
    body = resp.get_json()
    if body["datasets"]:
        first = body["datasets"][0]
        assert "citation" not in first
        assert "sha256" not in first


def test_route_detail_returns_dataset(client):
    resp = client.get("/system/eval-datasets/davis_2017")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["dataset_id"] == "davis_2017"
    assert body["modality"] == "video"


def test_route_detail_404_on_unknown_id(client):
    resp = client.get("/system/eval-datasets/does-not-exist")
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["code"] == "EVAL_DATASET_NOT_FOUND"


def test_route_serializes_full_dataset_shape(client):
    resp = client.get("/system/eval-datasets/davis_2017")
    body = resp.get_json()
    for required_field in (
        "dataset_id", "label", "modality", "benchmark_targets",
        "upstream", "license", "acquisition", "commercial_use_ok",
    ):
        assert required_field in body, f"missing field: {required_field}"
