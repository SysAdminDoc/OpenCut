"""Authenticate the plugin registry document itself.

Per-artifact Ed25519 signatures only prove that whoever holds a publisher key
signed an archive. They say nothing about which publisher key belongs in the
index, and the index is what supplies both the key and the artifact digest. So
an attacker who controls the registry controls the whole chain: they list their
own publisher, their own key, their own digest, and every signature verifies.

That mattered because the registry URL pointed at
``raw.githubusercontent.com/opencut/plugin-registry`` -- an organisation this
project does not own (OpenCut is ``SysAdminDoc/OpenCut``), at a repository that
did not exist. Anyone who could create it became the authoritative plugin index
for every installation, and because publisher trust is first-use, a brand new
publisher id would have been pinned silently.

The registry document now has to be signed by a key shipped with the release,
and it is verified before a single entry is parsed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("opencut")

REGISTRY_KEYS_PATH = Path(__file__).resolve().parents[1] / "data" / "plugin_registry_keys.json"

#: Bump when the signed-payload construction changes.
REGISTRY_SCHEMA_VERSION = 1

SIGNATURE_FIELD = "signature"
KEY_ID_FIELD = "signing_key_id"


class RegistrySignatureError(Exception):
    """The registry document is unsigned, mis-signed, or signed by a stranger."""


def _decode_key(value: str) -> bytes:
    try:
        raw = base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError) as exc:
        raise RegistrySignatureError("registry signing key must be valid base64") from exc
    if len(raw) != 32:
        raise RegistrySignatureError("registry signing key must be a 32-byte Ed25519 public key")
    return raw


def load_trusted_registry_keys(path: Path | str | None = None) -> dict[str, str]:
    """Return ``{key_id: base64 public key}`` shipped with this release.

    An empty map means registry distribution is not enabled for this build.
    That is the honest default: no key has been published yet, so nothing
    should be trusted rather than everything.
    """
    target = Path(path) if path is not None else REGISTRY_KEYS_PATH
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrySignatureError(f"registry key file is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistrySignatureError("registry key file must be a JSON object")
    keys = payload.get("keys")
    if not isinstance(keys, dict):
        raise RegistrySignatureError("registry key file must carry a 'keys' object")
    out: dict[str, str] = {}
    for key_id, value in keys.items():
        if not isinstance(key_id, str) or not isinstance(value, str):
            raise RegistrySignatureError("registry key entries must be strings")
        _decode_key(value)
        out[key_id] = value
    return out


def canonical_registry_payload(document: dict) -> bytes:
    """Return the exact bytes a registry signature covers.

    Everything except the signature itself, serialized deterministically, so
    that reordering keys or reformatting the file cannot change the result.
    """
    if not isinstance(document, dict):
        raise RegistrySignatureError("registry document must be a JSON object")
    body = {key: value for key, value in document.items() if key != SIGNATURE_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_registry_document(document: dict, private_key_bytes: bytes) -> str:
    """Return the base64 signature for ``document``.

    Used by the registry publishing tooling and by the tests; the private key
    never lives in this repository.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return base64.b64encode(key.sign(canonical_registry_payload(document))).decode("ascii")


def verify_registry_document(document: dict, *, trusted_keys: dict[str, str] | None = None) -> str:
    """Verify ``document`` and return the key id that signed it.

    Raises :class:`RegistrySignatureError` for every failure mode, so a caller
    can refuse the document without having to tell them apart.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    keys = load_trusted_registry_keys() if trusted_keys is None else dict(trusted_keys)
    if not keys:
        raise RegistrySignatureError(
            "No plugin registry signing key ships with this build, so the registry "
            "cannot be authenticated and is not used. Install plugins from a local "
            "directory, or add a trusted key to opencut/data/plugin_registry_keys.json."
        )
    if not isinstance(document, dict):
        raise RegistrySignatureError("registry document must be a JSON object")

    key_id = document.get(KEY_ID_FIELD)
    if not isinstance(key_id, str) or not key_id.strip():
        raise RegistrySignatureError("registry document does not name a signing key")
    public_key = keys.get(key_id)
    if public_key is None:
        raise RegistrySignatureError(
            f"registry document is signed by unknown key {key_id!r}; "
            "this build trusts: " + ", ".join(sorted(keys)) or "none"
        )

    signature = document.get(SIGNATURE_FIELD)
    if not isinstance(signature, str) or not signature.strip():
        raise RegistrySignatureError("registry document carries no signature")
    try:
        raw_signature = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError) as exc:
        raise RegistrySignatureError("registry signature must be valid base64") from exc
    if len(raw_signature) != 64:
        raise RegistrySignatureError("registry signature must be 64 bytes")

    try:
        Ed25519PublicKey.from_public_bytes(_decode_key(public_key)).verify(
            raw_signature, canonical_registry_payload(document)
        )
    except InvalidSignature as exc:
        raise RegistrySignatureError("registry signature verification failed") from exc
    return key_id


def registry_distribution_enabled() -> bool:
    """Return True when this build can authenticate a registry at all."""
    try:
        return bool(load_trusted_registry_keys())
    except RegistrySignatureError:
        return False


def registry_url_is_controlled(url: str, *, owner: str = "SysAdminDoc") -> bool:
    """Return True when a githubusercontent URL sits in ``owner``'s namespace.

    A dangling namespace is worse than a missing one: a 404 in a repository
    someone else can create is a landmine, not an outage.
    """
    prefix = "https://raw.githubusercontent.com/"
    if not url.startswith(prefix):
        # Self-hosted registries are the operator's call; this only guards the
        # GitHub default.
        return True
    remainder = url[len(prefix):]
    namespace = remainder.split("/", 1)[0] if "/" in remainder else remainder
    return namespace.lower() == owner.lower()


def default_registry_url() -> str:
    """Return the registry URL, honouring an operator override."""
    override = os.environ.get("OPENCUT_PLUGIN_REGISTRY_URL", "").strip()
    return override or DEFAULT_REGISTRY_URL


#: The GitHub namespace here must be one the maintainer controls.
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/SysAdminDoc/opencut-plugin-registry/main/registry.json"
)
