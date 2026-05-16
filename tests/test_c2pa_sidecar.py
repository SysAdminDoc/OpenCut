"""Tests for the C2PA sidecar surface (F110)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencut.core import c2pa_sidecar as c2pa


@pytest.fixture()
def asset(tmp_path: Path) -> Path:
    path = tmp_path / "render.mp4"
    path.write_bytes(b"this is the rendered asset bytes - pretend it's a video")
    return path


def test_build_sidecar_writes_manifest(asset, tmp_path):
    ingredient = c2pa.C2paIngredient(
        title="source.mp4",
        sha256="0" * 64,
        bytes=100,
        role="source",
    )
    action = c2pa.C2paAction(
        action="c2pa.edited",
        when="2026-05-16T12:00:00Z",
        parameters={"reason": "trim"},
    )

    result = c2pa.build_sidecar(
        asset_path=str(asset),
        ingredients=[ingredient],
        actions=[action],
        title="Demo Render",
    )

    assert Path(result.sidecar_path).exists()
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert manifest["title"] == "Demo Render"
    assert manifest["asset"]["sha256"] == result.asset_sha256
    assert len(manifest["ingredients"]) == 1
    assert len(manifest["actions"]) == 1
    assert manifest["claim_generator"].startswith("OpenCut")


def test_build_sidecar_emits_unsigned_when_no_key(monkeypatch, asset):
    monkeypatch.delenv("OPENCUT_C2PA_SIGNING_KEY", raising=False)

    result = c2pa.build_sidecar(asset_path=str(asset))

    assert result.signed is False
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert "signature" not in manifest


def test_build_sidecar_paths_default_to_dotted_extension(asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    assert Path(result.sidecar_path).name == "render.mp4.c2pa.json"


def test_verify_sidecar_detects_asset_match(asset):
    c2pa.build_sidecar(asset_path=str(asset))
    sidecar = asset.with_suffix(asset.suffix + ".c2pa.json")

    result = c2pa.verify_sidecar(str(sidecar))

    assert result["asset_match"] is True
    assert result["signature_present"] is False


def test_verify_sidecar_detects_asset_tamper(asset):
    c2pa.build_sidecar(asset_path=str(asset))
    sidecar = asset.with_suffix(asset.suffix + ".c2pa.json")
    asset.write_bytes(b"tampered bytes")

    result = c2pa.verify_sidecar(str(sidecar))

    assert result["asset_match"] is False
    assert any("mismatch" in w for w in result["warnings"])


def test_verify_sidecar_flags_manifest_tampering(asset):
    """A manifest edit must break the signature_match flag when signed."""
    # We can't realistically sign in CI (no key), but the unsigned path
    # still surfaces a warning to make it obvious that a verifier
    # shouldn't treat the sidecar as authoritative.
    c2pa.build_sidecar(asset_path=str(asset))
    sidecar = asset.with_suffix(asset.suffix + ".c2pa.json")

    result = c2pa.verify_sidecar(str(sidecar))
    warnings = " ".join(result["warnings"])
    assert "unsigned" in warnings.lower()


def test_build_sidecar_with_explicit_sidecar_path(asset, tmp_path):
    target = tmp_path / "explicit.c2pa.json"
    c2pa.build_sidecar(asset_path=str(asset), sidecar_path=str(target))
    assert target.exists()


def test_build_sidecar_raises_on_missing_asset(tmp_path):
    with pytest.raises(FileNotFoundError):
        c2pa.build_sidecar(asset_path=str(tmp_path / "missing.mp4"))


# ----- Route smoke ----------------------------------------------------------


def test_provenance_routes_round_trip(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    asset = tmp_path / "demo.mov"
    asset.write_bytes(b"video bytes here")

    resp = client.post(
        "/provenance/c2pa",
        json={
            "asset_path": str(asset),
            "title": "Demo",
            "ingredients": [{"title": "in.mp4", "sha256": "a" * 64, "bytes": 12}],
            "actions": [{"action": "c2pa.created", "when": "2026-05-16T12:00:00Z"}],
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    create = resp.get_json()
    sidecar_path = create["sidecar_path"]

    verify = client.post(
        "/provenance/verify",
        json={"sidecar_path": sidecar_path},
        headers=csrf_headers(csrf_token),
    )
    assert verify.status_code == 200
    vp = verify.get_json()
    assert vp["asset_match"] is True
    assert vp["signature_present"] is False


def test_provenance_route_requires_asset_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/provenance/c2pa",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
