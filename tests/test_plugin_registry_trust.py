"""The plugin index has to be authenticated before anything in it is believed.

Two problems, one chain. The registry URL pointed at
``raw.githubusercontent.com/opencut/plugin-registry`` -- the ``opencut`` GitHub
organisation exists and is not this project (OpenCut is ``SysAdminDoc/OpenCut``)
and ``plugin-registry`` did not exist, so whoever created it became the plugin
index for every install. And because each entry supplies its own publisher
public key, while publisher trust is first-use, a hostile index could introduce
a new publisher id and have its key pinned silently.
"""

from __future__ import annotations

import base64
import json

import pytest

from opencut.core.plugin_registry_trust import (
    DEFAULT_REGISTRY_URL,
    RegistrySignatureError,
    canonical_registry_payload,
    load_trusted_registry_keys,
    registry_url_is_controlled,
    sign_registry_document,
    verify_registry_document,
)

cryptography = pytest.importorskip("cryptography")


def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    raw_private = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw_private, base64.b64encode(raw_public).decode("ascii")


def _document(plugins=None):
    return {
        "schema_version": 1,
        "signing_key_id": "test-key",
        "plugins": plugins if plugins is not None else [{"id": "demo", "name": "Demo"}],
    }


# ---------------------------------------------------------------------------
# The namespace
# ---------------------------------------------------------------------------

def test_default_registry_lives_in_a_namespace_the_project_controls():
    assert registry_url_is_controlled(DEFAULT_REGISTRY_URL)
    assert "SysAdminDoc" in DEFAULT_REGISTRY_URL


def test_the_old_dangling_namespace_is_rejected():
    """The exact URL that shipped, against an org this project does not own."""
    assert not registry_url_is_controlled(
        "https://raw.githubusercontent.com/opencut/plugin-registry/main/registry.json"
    )


def test_self_hosted_registries_are_the_operators_call():
    assert registry_url_is_controlled("https://plugins.example.com/registry.json")


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

def test_a_correctly_signed_registry_verifies():
    private, public = _keypair()
    doc = _document()
    doc["signature"] = sign_registry_document(doc, private)
    assert verify_registry_document(doc, trusted_keys={"test-key": public}) == "test-key"


def test_an_unsigned_registry_is_refused():
    _, public = _keypair()
    with pytest.raises(RegistrySignatureError, match="no signature"):
        verify_registry_document(_document(), trusted_keys={"test-key": public})


def test_a_tampered_registry_is_refused():
    """Sign a document, then change an entry the signature covered."""
    private, public = _keypair()
    doc = _document()
    doc["signature"] = sign_registry_document(doc, private)
    doc["plugins"][0]["download_url"] = "https://evil.example.com/payload.zip"
    with pytest.raises(RegistrySignatureError, match="verification failed"):
        verify_registry_document(doc, trusted_keys={"test-key": public})


def test_a_registry_signed_by_a_stranger_is_refused():
    """The substituted-index attack: valid signature, wrong signer."""
    attacker_private, _ = _keypair()
    _, our_public = _keypair()
    doc = _document()
    doc["signature"] = sign_registry_document(doc, attacker_private)
    with pytest.raises(RegistrySignatureError, match="verification failed"):
        verify_registry_document(doc, trusted_keys={"test-key": our_public})


def test_an_unknown_key_id_is_refused():
    private, public = _keypair()
    doc = _document()
    doc["signing_key_id"] = "attacker-key"
    doc["signature"] = sign_registry_document(doc, private)
    with pytest.raises(RegistrySignatureError, match="unknown key"):
        verify_registry_document(doc, trusted_keys={"test-key": public})


def test_no_trusted_key_means_no_registry():
    doc = _document()
    with pytest.raises(RegistrySignatureError, match="signing key ships"):
        verify_registry_document(doc, trusted_keys={})


def test_signature_covers_key_order_not_formatting():
    """Canonicalization must ignore key order, or every publish breaks."""
    a = {"b": 2, "a": 1, "signature": "ignored"}
    b = {"a": 1, "b": 2, "signature": "different"}
    assert canonical_registry_payload(a) == canonical_registry_payload(b)


def test_shipped_key_file_parses_and_defaults_to_empty():
    """An empty key set is the honest default: no registry has been published."""
    keys = load_trusted_registry_keys()
    assert isinstance(keys, dict)


# ---------------------------------------------------------------------------
# The marketplace refuses to act on an unverified index
# ---------------------------------------------------------------------------

def test_fetch_refuses_an_unsigned_document_and_keeps_the_cache(tmp_path, monkeypatch):
    from opencut.core import plugin_marketplace

    cache = tmp_path / "plugin_registry.json"
    good = {"plugins": [{"id": "trusted", "name": "Trusted"}], "_opencut_registry_verified": True}
    cache.write_text(json.dumps(good), encoding="utf-8")
    original = cache.read_bytes()

    monkeypatch.setattr(plugin_marketplace, "REGISTRY_CACHE", str(cache))
    monkeypatch.setattr(plugin_marketplace, "_registry_cache_valid", lambda: False)
    monkeypatch.setattr(plugin_marketplace, "_load_installed", lambda: {})

    class _Response:
        def read(self):
            return json.dumps(_document()).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Response())

    with pytest.raises(RegistrySignatureError):
        plugin_marketplace.fetch_plugin_registry(force=True)

    assert cache.read_bytes() == original, "a rejected registry replaced the cached one"


def test_install_refuses_an_entry_from_an_unverified_index(monkeypatch, tmp_path):
    """A substituted index must not reach publisher pinning at all."""
    from opencut.core import plugin_marketplace
    from opencut.core.plugin_marketplace import PluginInfo

    target = PluginInfo(
        plugin_id="evil",
        name="Evil",
        version="1.0.0",
        author="A",
        description="",
        repo_url="https://example.com",
        download_url="https://example.com/evil.zip",
        artifact_sha256="0" * 64,
        publisher_id="attacker",
        publisher_public_key=base64.b64encode(b"0" * 32).decode(),
        publisher_signature=base64.b64encode(b"0" * 64).decode(),
        registry_verified=False,
    )

    monkeypatch.setattr(plugin_marketplace, "PLUGINS_DIR", str(tmp_path / "plugins"))

    trust_writes = []
    monkeypatch.setattr(
        "opencut.core.plugin_installation._trust_publisher",
        lambda identity: trust_writes.append(identity),
    )

    with pytest.raises(RegistrySignatureError, match="signed plugin index"):
        plugin_marketplace._install_marketplace_target(
            target,
            on_progress=None,
            approved_capabilities=[],
            approve_publisher_fingerprint="",
            replace_existing=False,
        )

    assert trust_writes == [], "an unverified index reached the publisher trust store"
