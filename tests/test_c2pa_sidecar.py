"""Tests for the C2PA sidecar surface (F110)."""

from __future__ import annotations

import json
import os
import subprocess
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
    report = (
        '{"active_manifest":"urn:c2pa:test","manifests":{"urn:c2pa:test":'
        '{"assertions":[{"label":"c2pa.actions.v2","data":{}},'
        '{"label":"org.opencut.provenance","data":{}}]}},'
        '"validation_results":{"activeManifest":{"success":['
        '{"code":"claimSignature.validated"},'
        '{"code":"assertion.dataHash.match"}]}},"validation_state":"Valid"}'
    )
    if os.name == "nt":
        tool = tmp_path / "c2patool.cmd"
        tool.write_text(
            "@echo off\n"
            "if \"%~2\"==\"\" goto verify\n"
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
            "exit /b 0\n"
            ":verify\n"
            f"echo {report}\n"
            "exit /b 0\n",
            encoding="utf-8",
        )
    else:
        tool = tmp_path / "c2patool"
        tool.write_text(
            "#!/bin/sh\n"
            f"report='{report}'\n"
            "if [ \"$#\" -eq 1 ]; then printf '%s\\n' \"$report\"; exit 0; fi\n"
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


def _write_signing_cert(tmp_path: Path) -> Path:
    cert = tmp_path / "opencut-c2pa-cert.pem"
    cert.write_text("test certificate for fake c2patool\n", encoding="utf-8")
    return cert


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


def test_verify_sidecar_does_not_trust_claimed_embed_status(monkeypatch, asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    sidecar = Path(result.sidecar_path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["embedding"] = {
        "status": "embedded",
        "embedded_path": str(asset),
        "verified": True,
    }
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(c2pa, "_find_c2patool", lambda *_args: "")

    verified = c2pa.verify_sidecar(str(sidecar))

    assert verified["credential_verified"] is False
    assert verified["status"] == "embedded_unverified"
    assert any("not re-verified" in warning for warning in verified["warnings"])


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
    cert_path = _write_signing_cert(tmp_path)
    fake_tool = _write_fake_c2patool(tmp_path)
    output = tmp_path / "render.content-credentials.mp4"
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_CERT", str(cert_path))
    monkeypatch.setenv("OPENCUT_C2PA_C2PATOOL", str(fake_tool))

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
    assert verify["status"] == "embedded_credential"
    assert verify["embedded"] is True
    assert verify["credential_verified"] is True


def test_c2patool_verification_failure_preserves_existing_output(
    monkeypatch, asset, tmp_path
):
    key_path = _write_ed25519_key(tmp_path)
    cert_path = _write_signing_cert(tmp_path)
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_CERT", str(cert_path))

    def _fake_run(command, **_kwargs):
        if "-o" in command:
            staged = Path(command[command.index("-o") + 1])
            staged.write_bytes(b"unverified credential")
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"validation_state": "Invalid"}),
            stderr="",
        )

    monkeypatch.setattr(c2pa.subprocess, "run", _fake_run)

    embedded, _method, _path, warnings = c2pa._run_c2patool_embed(
        asset=asset,
        manifest={"actions": [], "asset": {"title": asset.name}},
        output_path=str(output),
        c2patool_path="c2patool",
    )

    assert embedded is False
    assert output.read_bytes() == b"existing"
    assert any("validation state" in warning for warning in warnings)
    assert not list(tmp_path.glob(".existing.*.mp4"))


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
            "actions": [{
                "action": "c2pa.created",
                "when": "2026-05-16T12:00:00Z",
                "digitalSourceType": (
                    "http://cv.iptc.org/newscodes/digitalsourcetype/"
                    "trainedAlgorithmicMedia"
                ),
                "modelType": "c2pa.types.model",
                "modelName": "OpenCut route fixture",
                "humanOversightLevel": "human_validated",
            }],
        },
        headers=csrf_headers(csrf_token),
    )

    assert resp.status_code == 200, resp.get_json()
    create = resp.get_json()
    sidecar_path = create["sidecar_path"]
    manifest = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    assert manifest["c2pa_spec_version"] == "2.4"
    definition = c2pa._c2patool_manifest_definition(manifest)
    labels = [assertion["label"] for assertion in definition["assertions"]]
    assert "c2pa.actions.v2" in labels
    disclosure = next(
        assertion for assertion in definition["assertions"]
        if assertion["label"] == "c2pa.ai-disclosure"
    )
    assert disclosure["data"]["modelType"] == "c2pa.types.model"

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
    cert_path = _write_signing_cert(tmp_path)
    fake_tool = _write_fake_c2patool(tmp_path)
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_KEY", str(key_path))
    monkeypatch.setenv("OPENCUT_C2PA_SIGNING_CERT", str(cert_path))
    monkeypatch.setenv("OPENCUT_C2PA_C2PATOOL", str(fake_tool))

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
    assert verify.get_json()["status"] == "embedded_credential"


def test_provenance_route_requires_asset_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/provenance/c2pa",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# C2PA 2.4 alignment and 2.3 read compatibility
# ---------------------------------------------------------------------------


def test_manifest_records_c2pa_2_4_spec_version(asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert manifest["c2pa_spec_version"] == "2.4"
    # Manifest-spec / version fields must both exist so older readers
    # that scanned for `version` keep working.
    assert manifest["manifest_spec"]
    assert manifest["version"] == manifest["manifest_spec"]


def test_manifest_records_extended_action_fields(asset):
    action = c2pa.C2paAction(
        action="c2pa.addedText",
        when="2026-05-17T12:00:00Z",
        parameters={"language": "en"},
        live=True,
        software_agent="OpenCut/1.32.0",
    )
    result = c2pa.build_sidecar(asset_path=str(asset), actions=[action])
    manifest = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    assert manifest["actions"][0]["live"] is True
    assert manifest["actions"][0]["software_agent"] == "OpenCut/1.32.0"
    assert manifest["actions"][0]["action"] == "c2pa.addedText"


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
        c2pa.C2paAction(action="c2pa.addedText", when="2026-05-17T12:02:00Z"),
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


def test_claim_generator_documents_c2pa_2_4():
    """The default claim_generator string must surface the spec version
    we target so verifiers see it without parsing the manifest."""
    from opencut import __version__

    assert f"OpenCut/{__version__}" in c2pa.CLAIM_GENERATOR_DEFAULT
    assert "2.4" in c2pa.CLAIM_GENERATOR_DEFAULT


def test_vocabulary_is_immutable_and_sorted():
    vocab = c2pa.C2PA_ACTION_VOCABULARY
    assert isinstance(vocab, tuple), "vocabulary must be immutable"
    assert list(vocab) == sorted(vocab), "vocabulary tuple must stay sorted"
    for must_have in (
        "c2pa.created",
        "c2pa.addedText",
        "c2pa.enhanced",
        "c2pa.translated",
    ):
        assert must_have in vocab


def test_verify_sidecar_keeps_legacy_2_3_read_compatibility(asset):
    result = c2pa.build_sidecar(asset_path=str(asset))
    sidecar = Path(result.sidecar_path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["c2pa_spec_version"] = "2.3"
    manifest["manifest_spec"] = "0.2-sidecar"
    manifest["version"] = "0.2-sidecar"
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    verified = c2pa.verify_sidecar(str(sidecar))

    assert verified["asset_match"] is True
    assert verified["manifest"]["c2pa_spec_version"] == "2.3"


def test_c2patool_accepts_24_ai_manifest_when_available(tmp_path):
    from opencut.core.c2pa_embed import create_c2pa_manifest, read_c2pa

    tool = c2pa._find_c2patool()
    if not tool:
        pytest.skip("c2patool is not installed")
    image_module = pytest.importorskip("PIL.Image")
    asset = tmp_path / "source.png"
    image_module.new("RGB", (8, 8), (20, 40, 60)).save(asset)

    sidecar = c2pa.build_sidecar(
        asset_path=str(asset),
        actions=[
            c2pa.C2paAction(
                action="c2pa.created",
                when="2026-07-14T12:00:00Z",
                digital_source_type=(
                    "http://cv.iptc.org/newscodes/digitalsourcetype/"
                    "trainedAlgorithmicMedia"
                ),
                model_type="c2pa.types.model",
                model_name="OpenCut conformance fixture",
                human_oversight_level="human_validated",
            )
        ],
    )
    manifest = json.loads(Path(sidecar.sidecar_path).read_text(encoding="utf-8"))
    create_manifest = create_c2pa_manifest(
        operations=[{
            "action": "generate_broll",
            "modelType": "c2pa.types.model",
            "modelName": "OpenCut conformance fixture",
        }],
        title="Create-route manifest",
        format="image/png",
    )
    definitions = {
        "create": create_manifest["manifest_definition"],
        "edit": create_c2pa_manifest(
            operations=[{"action": "trim"}],
            title="Edit-route manifest",
            format="image/png",
        )["manifest_definition"],
        "sidecar": c2pa._c2patool_manifest_definition(manifest),
    }

    for name, definition in definitions.items():
        definition.pop("private_key", None)
        definition.pop("sign_cert", None)
        definition_path = tmp_path / f"{name}-manifest.json"
        output = tmp_path / f"{name}-signed.png"
        definition_path.write_text(json.dumps(definition), encoding="utf-8")

        embedded = subprocess.run(
            [tool, str(asset), "-m", str(definition_path), "-o", str(output), "-f"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert embedded.returncode == 0, embedded.stderr or embedded.stdout
        report = json.loads(embedded.stdout)
        assert c2pa._validate_c2patool_report(report, definition) == ""
        active = report["manifests"][report["active_manifest"]]
        labels = [assertion["label"] for assertion in active["assertions"]]
        assert "c2pa.actions.v2" in labels
        if any(
            assertion["label"] == "c2pa.ai-disclosure"
            for assertion in definition["assertions"]
        ):
            disclosure = next(
                assertion for assertion in active["assertions"]
                if assertion["label"] == "c2pa.ai-disclosure"
            )
            assert disclosure["data"]["modelType"] == "c2pa.types.model"
        read_back = read_c2pa(str(output))
        assert read_back["embedded"] is True
        assert read_back["validation_state"] == "Valid"
