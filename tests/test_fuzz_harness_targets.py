"""Inventory and pure-helper tests for the F215 fuzz target expansion."""

from __future__ import annotations

import json

import pytest

from tests.fuzz import test_parser_fuzz as fuzz

F215_TARGETS = {
    "validate_path",
    "otio_parse",
    "fcpxml_parse",
    "marker_import",
    "c2pa_sidecar",
    "plugin_manifest",
    "webhook_signature",
    "safe_pip_install",
}


def test_f215_fuzz_target_inventory_is_wired():
    assert F215_TARGETS.issubset(fuzz.TARGETS)
    assert fuzz.FUZZ_TARGETS == tuple(fuzz.TARGETS)
    assert len(fuzz.TARGETS) == 13


def test_all_fuzz_targets_are_callable():
    for target in fuzz.TARGETS.values():
        assert callable(target)


def test_safe_pip_package_validation_never_installs():
    from opencut.security import validate_safe_pip_package

    assert validate_safe_pip_package("whisperx==3.8.5") == "whisperx==3.8.5"
    assert validate_safe_pip_package("openai-whisper[all]>=20250625") == "openai-whisper[all]>=20250625"

    for value in ("", "../pkg", "git+https://example.com/pkg", "pkg @ https://example.com/pkg", "pkg name"):
        with pytest.raises(ValueError):
            validate_safe_pip_package(value)


def test_webhook_signature_helpers_round_trip():
    from opencut.core.webhook_signature import (
        normalize_webhook_signature,
        sign_webhook_body,
        verify_webhook_signature,
    )

    body = json.dumps({"event": "job_complete", "ok": True}, sort_keys=True)
    signature = sign_webhook_body("secret", body)

    assert signature.startswith("sha256=")
    assert verify_webhook_signature("secret", body, signature)
    assert verify_webhook_signature("secret", body, signature.removeprefix("sha256="))
    assert normalize_webhook_signature(signature.upper()).startswith("sha256=")
    assert not verify_webhook_signature("secret", body + "!", signature)
    assert not verify_webhook_signature("wrong", body, signature)


def test_c2pa_verify_rejects_non_object_sidecar(tmp_path):
    from opencut.core.c2pa_sidecar import verify_sidecar

    sidecar = tmp_path / "asset.mp4.c2pa.json"
    sidecar.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        verify_sidecar(str(sidecar))


def test_plugin_manifest_rejects_non_object_schema():
    from opencut.core.plugin_manifest import validate_manifest_schema

    result = validate_manifest_schema([])

    assert not result.valid
    assert any("JSON object" in error for error in result.errors)
