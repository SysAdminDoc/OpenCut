"""Tests for the C2PA sidecar surface (F110)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from opencut.core import c2pa_sidecar as c2pa


@pytest.fixture()
def asset(tmp_path: Path) -> Path:
    path = tmp_path / "render.mp4"
    path.write_bytes(b"this is the rendered asset bytes - pretend it's a video")
    return path


def _write_ed25519_key(tmp_path: Path) -> Path:
    ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")

    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "opencut-c2pa-ed25519.pem"
    key_path.write_bytes(pem)
    return key_path


def _write_fake_c2patool(tmp_path: Path) -> Path:
    if os.name == "nt":
        tool = tmp_path / "c2patool.cmd"
        tool.write_text(
            "@echo off\n"
            "set SRC=%~1\n"
            "set OUT=\n"
            ":args\n"
            "if \"%~1\"==\"\" goto done\n"
            "if \"%~1\"==\"-o\" set OUT=%~2\n"
            "shift\n"
            "goto args\n"
            ":done\n"
            "if \"%OUT%\"==\"\" exit /b 2\n"
            "copy /Y \"%SRC%\" \"%OUT%\" >nul\n"
            "exit /b 0\n",
            encoding="utf-8",
        )
    else:
        tool = tmp_path / "c2patool"
        tool.write_text(
            "#!/bin/sh\n"
            "src=\"$1\"\n"
            "out=\"\"\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"-o\" ]; then shift; out=\"$1\"; fi\n"
            "  shift\n"
            "done\n"
            "[ -n \"$out\" ] || exit 2\n"
            "cp \"$src\" \"$out\"\n",
            encoding="utf-8",
        )
        tool.chmod(0o755)
    return tool


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


def test_build_sidecar_signs_and_verifies_with_operator_key(monkeypatch, asset, tmp_path):
    key_path = _write_ed25519_key(tmp_path)
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))

    result = c2pa.build_sidecar(asset_path=str(asset))
    verify = c2pa.verify_sidecar(result.sidecar_path)

    assert result.signed is True
    assert result.signing_algorithm == "ed25519"
    assert verify["status"] == "signed_sidecar"
    assert verify["signature_present"] is True
    assert verify["signature_hash_match"] is True
    assert verify["signature_valid"] is True
    assert verify["signature_match"] is True


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


def test_verify_sidecar_detects_signed_manifest_tampering(monkeypatch, asset, tmp_path):
    key_path = _write_ed25519_key(tmp_path)
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))
    result = c2pa.build_sidecar(asset_path=str(asset))
    sidecar = Path(result.sidecar_path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["title"] = "Tampered title"
    sidecar.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    verify = c2pa.verify_sidecar(str(sidecar))

    assert verify["asset_match"] is True
    assert verify["status"] == "tampered_manifest"
    assert verify["signature_present"] is True
    assert verify["signature_match"] is False
    assert any("signed bytes hash mismatch" in w for w in verify["warnings"])


def test_verify_sidecar_distinguishes_missing_asset(asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    asset.unlink()

    verify = c2pa.verify_sidecar(result.sidecar_path)

    assert verify["asset_found"] is False
    assert verify["status"] == "missing_asset"
    assert any("not found" in w for w in verify["warnings"])


def test_build_sidecar_warns_when_embed_requested_without_key(monkeypatch, asset):
    monkeypatch.delenv("OPENCUT_C2PA_SIGNING_KEY", raising=False)

    result = c2pa.build_sidecar(asset_path=str(asset), embed=True)
    verify = c2pa.verify_sidecar(result.sidecar_path)

    assert result.embedded is False
    assert result.signed is False
    assert verify["status"] == "unsigned_sidecar"
    assert any("requires OPENCUT_C2PA_SIGNING_KEY" in w for w in result.warnings)


def test_build_sidecar_embeds_with_c2patool_when_configured(monkeypatch, asset, tmp_path):
    key_path = _write_ed25519_key(tmp_path)
    fake_tool = _write_fake_c2patool(tmp_path)
    output = tmp_path / "render.content-credentials.mp4"
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))

    result = c2pa.build_sidecar(
        asset_path=str(asset),
        embed=True,
        embedded_output_path=str(output),
        c2patool_path=str(fake_tool),
    )
    verify = c2pa.verify_sidecar(result.sidecar_path)

    assert result.signed is True
    assert result.embedded is True
    assert Path(result.embedded_path) == output
    assert output.exists()
    assert verify["status"] == "embedded_signed"
    assert verify["embedded"] is True


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


def test_provenance_route_embeds_when_tool_and_key_configured(
    monkeypatch,
    client,
    csrf_token,
    tmp_path,
):
    from tests.conftest import csrf_headers

    key_path = _write_ed25519_key(tmp_path)
    fake_tool = _write_fake_c2patool(tmp_path)
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))

    asset = tmp_path / "demo.mp4"
    asset.write_bytes(b"video bytes here")
    embedded = tmp_path / "demo.embedded.mp4"

    resp = client.post(
        "/provenance/c2pa",
        json={
            "asset_path": str(asset),
            "title": "Demo",
            "embed": True,
            "embedded_output_path": str(embedded),
            "c2patool_path": str(fake_tool),
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["signed"] is True
    assert payload["embedded"] is True
    assert Path(payload["embedded_path"]) == embedded
    assert embedded.exists()

    verify = client.post(
        "/provenance/verify",
        json={"sidecar_path": payload["sidecar_path"]},
        headers=csrf_headers(csrf_token),
    )
    assert verify.status_code == 200
    assert verify.get_json()["status"] == "embedded_signed"


def test_provenance_route_requires_asset_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/provenance/c2pa",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# F140 — C2PA 2.3 alignment
# ---------------------------------------------------------------------------


def test_manifest_records_c2pa_2_3_spec_version(asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert manifest["c2pa_spec_version"] == "2.3"
    # Manifest-spec / version fields must both exist so older readers
    # that scanned for `version` keep working.
    assert manifest["manifest_spec"]
    assert manifest["version"] == manifest["manifest_spec"]


def test_manifest_records_extended_action_fields(asset):
    action = c2pa.C2paAction(
        action="c2pa.captioned",
        when="2026-05-17T12:00:00Z",
        parameters={"language": "en"},
        live=True,
        software_agent="OpenCut/1.32.0",
    )
    result = c2pa.build_sidecar(asset_path=str(asset), actions=[action])
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert manifest["actions"][0]["live"] is True
    assert manifest["actions"][0]["software_agent"] == "OpenCut/1.32.0"
    assert manifest["actions"][0]["action"] == "c2pa.captioned"


def test_unknown_action_is_tolerated_but_warned(asset):
    action = c2pa.C2paAction(
        action="opencut.invented_action",
        when="2026-05-17T12:00:00Z",
    )
    result = c2pa.build_sidecar(asset_path=str(asset), actions=[action])
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    # Serialised (forward-compat) ...
    assert manifest["actions"][0]["action"] == "opencut.invented_action"
    # ... and flagged.
    assert "warnings" in manifest
    assert any("vocabulary" in w for w in manifest["warnings"])


def test_known_actions_do_not_produce_warnings(asset):
    actions = [
        c2pa.C2paAction(action="c2pa.created", when="2026-05-17T12:00:00Z"),
        c2pa.C2paAction(action="c2pa.cropped", when="2026-05-17T12:01:00Z"),
        c2pa.C2paAction(action="c2pa.captioned", when="2026-05-17T12:02:00Z"),
    ]
    result = c2pa.build_sidecar(asset_path=str(asset), actions=actions)
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert "warnings" not in manifest


def test_cloud_trust_list_propagates_when_provided(asset):
    result = c2pa.build_sidecar(
        asset_path=str(asset),
        cloud_trust_list="https://example.invalid/c2pa/trust-list",
    )
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert manifest["cloud_trust_list"] == "https://example.invalid/c2pa/trust-list"


def test_cloud_trust_list_omitted_when_blank(asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert "cloud_trust_list" not in manifest


def test_claim_generator_documents_c2pa_2_3():
    """The default claim_generator string must surface the spec version
    we target so verifiers see it without parsing the manifest."""
    from opencut import __version__

    assert f"OpenCut/{__version__}" in c2pa.CLAIM_GENERATOR_DEFAULT
    assert "2.3" in c2pa.CLAIM_GENERATOR_DEFAULT


def test_vocabulary_is_immutable_and_sorted():
    vocab = c2pa.C2PA_ACTION_VOCABULARY
    assert isinstance(vocab, tuple), "vocabulary must be immutable"
    assert list(vocab) == sorted(vocab), "vocabulary tuple must stay sorted"
    # Sanity: the 2.3 working catalogue we promised must be represented.
    for must_have in ("c2pa.created", "c2pa.captioned", "c2pa.transcribed", "c2pa.translated"):
        assert must_have in vocab
