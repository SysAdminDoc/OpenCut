"""C2PA 2.4 provenance sidecars and verified embedded credentials.

OpenCut's JSON sidecar is a local audit record bound to an asset SHA-256 and
optionally signed with an Ed25519 key. Real Content Credentials are produced
only through ``c2patool`` with operator-provided key/certificate files, then
re-read to prove the claim signature, required assertions, and authoritative
asset hard binding before the staged output is promoted. Version 2.3 sidecars
remain readable for compatibility.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

from opencut.openapi_registry import openapi_response_schema

logger = logging.getLogger("opencut")

SPEC_VERSION = "0.3-sidecar"
MANIFEST_SPEC_VERSION = SPEC_VERSION  # public alias
C2PA_SPEC_VERSION = "2.4"
CLAIM_GENERATOR_DEFAULT = "OpenCut/1.40.0 (sidecar; c2pa-spec 2.4)"
SUPPORTED_EMBED_EXTENSIONS = {".jpg", ".jpeg", ".mp4", ".png"}
DEFAULT_C2PATOOL_TIMEOUT_SECONDS = 120


# C2PA 2.4 action vocabulary. This is the documented set of
# action strings; unknown actions still serialise but get flagged.
# Keep the tuple sorted so test diffs are deterministic.
C2PA_ACTION_VOCABULARY: tuple = (
    "c2pa.addedText",
    "c2pa.adjustedColor",
    "c2pa.changedSpeed",
    "c2pa.converted",
    "c2pa.created",
    "c2pa.cropped",
    "c2pa.deleted",
    "c2pa.drawing",
    "c2pa.dubbed",
    "c2pa.edited",
    "c2pa.edited.metadata",
    "c2pa.enhanced",
    "c2pa.filtered",
    "c2pa.mastered",
    "c2pa.mixed",
    "c2pa.opened",
    "c2pa.orientation",
    "c2pa.placed",
    "c2pa.published",
    "c2pa.redacted",
    "c2pa.remixed",
    "c2pa.removed",
    "c2pa.repackaged",
    "c2pa.resized",
    "c2pa.resized.proportional",
    "c2pa.transcoded",
    "c2pa.translated",
    "c2pa.trimmed",
    "c2pa.unknown",
    "c2pa.watermarked.bound",
    "c2pa.watermarked.unbound",
)


def is_known_c2pa_action(name: str) -> bool:
    """Return True when *name* belongs to the C2PA 2.4 documented vocabulary."""
    return name in C2PA_ACTION_VOCABULARY


@dataclass
class C2paAction:
    action: str           # e.g. "c2pa.created", "c2pa.edited", "c2pa.cropped"
    when: str             # ISO-8601 UTC
    parameters: dict = field(default_factory=dict)
    live: bool = False
    software_agent: Any = ""
    digital_source_type: str = ""
    model_type: str = ""
    model_name: str = ""
    model_identifier: str = ""
    human_oversight_level: str = ""
    scientific_domain: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class C2paIngredient:
    """A source asset that flowed into the rendered output."""

    title: str
    sha256: str
    bytes: int
    role: str = "source"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
@openapi_response_schema("/provenance/c2pa")
class C2paSidecarResult:
    sidecar_path: str
    asset_sha256: str
    signed: bool
    signing_algorithm: str = ""
    embedded: bool = False
    embedding_method: str = ""
    embedded_path: str = ""
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_signing_key() -> Optional[bytes]:
    """Return the configured Ed25519 private key, or ``None``."""
    raw = os.environ.get("OPENCUT_C2PA_SIGNING_KEY", "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.exists():
        try:
            return candidate.read_bytes()
        except OSError as exc:  # pragma: no cover - rare
            logger.warning("c2pa_sidecar: cannot read key file %s: %s", candidate, exc)
            return None
    # Treat the env var as a literal PEM string.
    return raw.encode("utf-8")


def _try_sign(manifest_bytes: bytes) -> Optional[dict]:
    """Attempt to sign the manifest. Returns signature metadata or ``None``."""
    key = _load_signing_key()
    if not key:
        return None
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key
    except Exception:
        logger.info(
            "c2pa_sidecar: cryptography library not installed; manifest will be unsigned. "
            "`pip install cryptography` to enable Ed25519 signing."
        )
        return None
    try:
        priv = load_pem_private_key(key, password=None)
        if not isinstance(priv, Ed25519PrivateKey):
            logger.info(
                "c2pa_sidecar: configured C2PA key is not Ed25519; "
                "the local JSON sidecar will remain unsigned"
            )
            return None
        signature = priv.sign(manifest_bytes)
        public_key = priv.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")
        return {
            "algorithm": "ed25519",
            "value": signature.hex(),
            "signed_bytes_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "public_key": public_key,
        }
    except Exception as exc:
        logger.warning("c2pa_sidecar: signing failed (%s); writing unsigned manifest", exc)
        return None


def _iter_pem_public_key_blocks(text: str) -> List[str]:
    """Extract every ``-----BEGIN PUBLIC KEY-----`` block from *text*."""
    begin_marker = "-----BEGIN PUBLIC KEY-----"
    end_marker = "-----END PUBLIC KEY-----"
    blocks: List[str] = []
    cursor = 0
    while True:
        begin = text.find(begin_marker, cursor)
        if begin < 0:
            break
        end = text.find(end_marker, begin)
        if end < 0:
            break
        end += len(end_marker)
        blocks.append(text[begin:end])
        cursor = end
    return blocks


def _load_trusted_public_keys() -> Optional[List[bytes]]:
    """Return raw Ed25519 public keys pinned as the sidecar trust anchor.

    ``OPENCUT_C2PA_TRUSTED_PUBKEYS`` may be a path to a PEM file or one or
    more literal ``PUBLIC KEY`` PEM blocks. When it is unset but
    ``OPENCUT_C2PA_SIGNING_KEY`` is configured, the local signing key's
    public half becomes the default trust anchor, so sidecars produced by
    this install verify as ``pinned`` out of the box.

    Returns ``None`` when no trust anchor is configured (verification is
    then self-asserted), or a possibly-empty list when one is configured
    (an empty list fails closed: no embedded key can match).
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
            load_pem_private_key,
            load_pem_public_key,
        )
    except Exception:  # pragma: no cover - cryptography absent
        return None

    raw = os.environ.get("OPENCUT_C2PA_TRUSTED_PUBKEYS", "").strip()
    if raw:
        candidate = Path(raw)
        text = raw
        if candidate.exists():
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "c2pa_sidecar: cannot read trusted-key file %s: %s", candidate, exc
                )
                return []  # configured but unreadable: fail closed
        keys: List[bytes] = []
        for block in _iter_pem_public_key_blocks(text):
            try:
                pub = load_pem_public_key(block.encode("utf-8"))
            except Exception as exc:
                logger.warning("c2pa_sidecar: skipping unparseable trusted key: %s", exc)
                continue
            if isinstance(pub, Ed25519PublicKey):
                keys.append(pub.public_bytes(Encoding.Raw, PublicFormat.Raw))
            else:
                logger.warning("c2pa_sidecar: skipping non-Ed25519 trusted key")
        return keys

    # No explicit anchor: derive one from the local signing key when present.
    signing_key = _load_signing_key()
    if signing_key:
        try:
            priv = load_pem_private_key(signing_key, password=None)
            if isinstance(priv, Ed25519PrivateKey):
                return [
                    priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                ]
        except Exception:  # pragma: no cover - malformed signing key
            pass
    return None


def _verify_signature(sig: dict, canonical: bytes, warnings: List[str]) -> tuple[bool, bool, str]:
    """Return ``(hash_match, signature_valid, key_trust)`` for sidecar signature metadata.

    ``key_trust`` is ``"pinned"`` when the embedded public key matches an
    operator trust anchor, ``"untrusted"`` when an anchor is configured but
    the key does not match it, and ``"self_asserted"`` when no anchor exists
    (the signature then proves internal consistency only — anyone can re-sign
    an edited manifest with their own key).
    """
    signed_hash = str(sig.get("signed_bytes_sha256") or "")
    current_hash = hashlib.sha256(canonical).hexdigest()
    hash_match = signed_hash == current_hash
    if not hash_match:
        warnings.append("signed bytes hash mismatch - manifest was edited after signing")

    signature_valid = False
    public_key_pem = str(sig.get("public_key") or "")
    value = str(sig.get("value") or "")
    if not public_key_pem:
        warnings.append("signature public key is missing")
        return hash_match, False, ""
    if not value:
        warnings.append("signature value is missing")
        return hash_match, False, ""
    key_trust = ""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
            load_pem_public_key,
        )

        pub = load_pem_public_key(public_key_pem.encode("utf-8"))
        if not isinstance(pub, Ed25519PublicKey):
            warnings.append("signature public key is not Ed25519")
            return hash_match, False, ""

        trusted = _load_trusted_public_keys()
        if trusted is None:
            key_trust = "self_asserted"
            warnings.append(
                "signature key is self-asserted (no OPENCUT_C2PA_TRUSTED_PUBKEYS "
                "or OPENCUT_C2PA_SIGNING_KEY trust anchor); the signature proves "
                "internal consistency, not authenticity"
            )
        else:
            raw_pub = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
            if any(hmac.compare_digest(raw_pub, anchor) for anchor in trusted):
                key_trust = "pinned"
            else:
                key_trust = "untrusted"
                warnings.append(
                    "signature public key does not match any pinned trust anchor - "
                    "the manifest may have been re-signed by another party"
                )

        pub.verify(bytes.fromhex(value), canonical)
        signature_valid = True
    except Exception as exc:
        warnings.append(f"signature verification failed: {exc}")
    return hash_match, signature_valid, key_trust


def _canonical_manifest(manifest: dict) -> bytes:
    """Return the canonical JSON bytes covered by the sidecar signature."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _find_c2patool(explicit_path: str = "") -> str:
    """Return a configured c2patool executable path, or ``""`` when absent."""
    candidate = (explicit_path or os.environ.get("OPENCUT_C2PA_C2PATOOL") or "").strip()
    if candidate:
        return candidate
    return shutil.which("c2patool") or ""


def _default_embedded_output_path(asset: Path) -> Path:
    return asset.with_name(f"{asset.stem}.content-credentials{asset.suffix}")


def _c2patool_manifest_definition(manifest: dict) -> dict:
    """Convert OpenCut sidecar manifest data into a c2patool manifest definition."""
    from opencut.core.c2pa_embed import (
        build_c2pa_assertions,
        c2patool_manifest_definition,
    )

    operations = []
    for action in manifest.get("actions", []):
        if not isinstance(action, dict):
            continue
        operations.append({
            "action": action.get("action") or "c2pa.unknown",
            "when": action.get("when") or manifest.get("generated_at"),
            "softwareAgent": (
                action.get("software_agent") or manifest.get("claim_generator")
            ),
            "parameters": action.get("parameters") or {},
            "digitalSourceType": action.get("digital_source_type") or "",
            "modelType": action.get("model_type") or "",
            "modelName": action.get("model_name") or "",
            "modelIdentifier": action.get("model_identifier") or "",
            "humanOversightLevel": action.get("human_oversight_level") or "",
            "scientificDomain": action.get("scientific_domain") or [],
        })

    _, assertions = build_c2pa_assertions(
        operations,
        claim_generator=(
            manifest.get("claim_generator") or CLAIM_GENERATOR_DEFAULT
        ),
        created=str(manifest.get("generated_at") or ""),
    )
    assertions.append({"label": "org.opencut.provenance", "data": manifest})
    definition = c2patool_manifest_definition({
        "claim_generator": manifest.get("claim_generator") or CLAIM_GENERATOR_DEFAULT,
        "title": manifest.get("title") or manifest.get("asset", {}).get("title") or "OpenCut export",
        "format": manifest.get("format") or "application/octet-stream",
        "assertions": assertions,
    })

    # c2patool can either use its built-in test certificate or operator-provided
    # signing material. Operators that need a real trust chain can provide the
    # same env vars c2patool documents; the sidecar signer continues to use
    # OPENCUT_C2PA_SIGNING_KEY for OpenCut's local verification path.
    private_key = os.environ.get("OPENCUT_C2PA_SIGNING_KEY", "").strip()
    sign_cert = os.environ.get("OPENCUT_C2PA_SIGNING_CERT", "").strip()
    if private_key and sign_cert and Path(private_key).exists() and Path(sign_cert).exists():
        definition["private_key"] = private_key
        definition["sign_cert"] = sign_cert
    alg = os.environ.get("OPENCUT_C2PA_SIGNING_ALG", "").strip()
    if alg:
        definition["alg"] = alg
    return definition


def _validate_c2patool_report(report: dict, definition: dict) -> str:
    """Return an error when a c2patool report lacks 2.4 or binding evidence."""
    if not isinstance(report, dict):
        return "c2patool verification report is not a JSON object"
    if report.get("validation_state") != "Valid":
        return f"c2patool validation state is {report.get('validation_state') or 'missing'}"
    active_label = str(report.get("active_manifest") or "")
    manifests = report.get("manifests")
    if not active_label or not isinstance(manifests, dict):
        return "c2patool report has no active manifest"
    active = manifests.get(active_label)
    if not isinstance(active, dict):
        return "c2patool active manifest is missing"
    labels = {
        str(assertion.get("label") or "")
        for assertion in active.get("assertions", [])
        if isinstance(assertion, dict)
    }
    if "c2pa.actions.v2" not in labels:
        return "embedded manifest is missing c2pa.actions.v2"
    required_labels = {
        str(assertion.get("label") or "")
        for assertion in definition.get("assertions", [])
        if isinstance(assertion, dict)
        and assertion.get("label") in {"c2pa.ai-disclosure", "org.opencut.provenance"}
    }
    missing = sorted(required_labels - labels)
    if missing:
        return f"embedded manifest is missing assertions: {', '.join(missing)}"

    validation = report.get("validation_results") or {}
    active_results = validation.get("activeManifest") or {}
    success_codes = {
        str(item.get("code") or "")
        for item in active_results.get("success", [])
        if isinstance(item, dict)
    }
    if "claimSignature.validated" not in success_codes:
        return "c2patool did not validate the claim signature"
    if not any(
        code.startswith("assertion.") and code.endswith("Hash.match")
        for code in success_codes
    ):
        return "c2patool did not validate an authoritative hard binding"
    return ""


def _run_c2patool_embed(
    *,
    asset: Path,
    manifest: dict,
    output_path: Optional[str] = None,
    c2patool_path: str = "",
) -> tuple[bool, str, str, List[str]]:
    """Try to embed a signed C2PA manifest with c2patool.

    Returns ``(embedded, method, embedded_path, warnings)``.
    """
    warnings: List[str] = []
    suffix = asset.suffix.lower()
    if suffix not in SUPPORTED_EMBED_EXTENSIONS:
        warnings.append(
            f"embedded C2PA output is limited to {', '.join(sorted(SUPPORTED_EMBED_EXTENSIONS))}; "
            f"got {suffix or '<none>'}"
        )
        return False, "", "", warnings

    tool = _find_c2patool(c2patool_path)
    if not tool:
        warnings.append("c2patool not found; wrote signed sidecar without embedded C2PA manifest")
        return False, "", "", warnings

    private_key = Path(os.environ.get("OPENCUT_C2PA_SIGNING_KEY", "").strip())
    sign_cert = Path(os.environ.get("OPENCUT_C2PA_SIGNING_CERT", "").strip())
    if not private_key.is_file() or not sign_cert.is_file():
        warnings.append(
            "embedded C2PA output requires readable OPENCUT_C2PA_SIGNING_KEY "
            "and OPENCUT_C2PA_SIGNING_CERT files"
        )
        return False, "", "", warnings

    output = Path(output_path) if output_path else _default_embedded_output_path(asset)
    output = output.resolve()
    asset = asset.resolve()
    if output == asset:
        warnings.append("C2PA output must not overwrite the source asset")
        return False, "", "", warnings
    if output.suffix.lower() != suffix:
        warnings.append("C2PA output must keep the source file extension")
        return False, "", "", warnings
    output.parent.mkdir(parents=True, exist_ok=True)
    definition = _c2patool_manifest_definition(manifest)
    definition["private_key"] = str(private_key)
    definition["sign_cert"] = str(sign_cert)

    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=str(output.parent)
    )
    os.close(descriptor)
    staged = Path(staged_name)
    try:
        with tempfile.TemporaryDirectory(prefix="opencut_c2pa_") as root:
            manifest_path = Path(root) / "manifest.json"
            manifest_path.write_text(
                json.dumps(definition, indent=2, sort_keys=True), encoding="utf-8"
            )
            result = subprocess.run(
                [tool, str(asset), "-m", str(manifest_path), "-o", str(staged), "-f"],
                capture_output=True,
                text=True,
                timeout=DEFAULT_C2PATOOL_TIMEOUT_SECONDS,
                env=os.environ.copy(),
                check=False,
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            warnings.append(
                f"c2patool embed failed with exit {result.returncode}: {detail[-300:]}"
            )
            return False, "", "", warnings
        if not staged.is_file() or staged.stat().st_size <= 0:
            warnings.append("c2patool reported success but created no embedded output")
            return False, "", "", warnings

        verified = subprocess.run(
            [tool, str(staged)],
            capture_output=True,
            text=True,
            timeout=DEFAULT_C2PATOOL_TIMEOUT_SECONDS,
            env=os.environ.copy(),
            check=False,
        )
        if verified.returncode != 0:
            detail = (verified.stderr or verified.stdout or "").strip()
            warnings.append(
                f"c2patool verification failed with exit {verified.returncode}: {detail[-300:]}"
            )
            return False, "", "", warnings
        try:
            report = json.loads(verified.stdout)
        except json.JSONDecodeError as exc:
            warnings.append(f"c2patool returned invalid verification JSON: {exc}")
            return False, "", "", warnings
        report_error = _validate_c2patool_report(report, definition)
        if report_error:
            warnings.append(report_error)
            return False, "", "", warnings
        os.replace(staged, output)
        return True, "c2patool", str(output), warnings
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass


def _verify_embedded_credential(path: Path, manifest: dict) -> tuple[bool, str]:
    """Re-read an embedded credential; never trust sidecar status metadata."""
    if not path.is_file():
        return False, f"embedded credential {str(path)!r} was not found"
    tool = _find_c2patool()
    if not tool:
        return False, "c2patool is unavailable; embedded credential was not re-verified"
    try:
        result = subprocess.run(
            [tool, str(path)],
            capture_output=True,
            text=True,
            timeout=DEFAULT_C2PATOOL_TIMEOUT_SECONDS,
            env=os.environ.copy(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"c2patool verification could not run: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"c2patool verification failed: {detail[-300:]}"
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return False, f"c2patool returned invalid verification JSON: {exc}"
    error = _validate_c2patool_report(
        report, _c2patool_manifest_definition(manifest)
    )
    return (not error, error)


def build_sidecar(
    *,
    asset_path: str,
    ingredients: Optional[Sequence[C2paIngredient]] = None,
    actions: Optional[Sequence[C2paAction]] = None,
    title: Optional[str] = None,
    claim_generator: str = CLAIM_GENERATOR_DEFAULT,
    sidecar_path: Optional[str] = None,
    cloud_trust_list: str = "",
    embed: bool = False,
    embedded_output_path: Optional[str] = None,
    c2patool_path: str = "",
) -> C2paSidecarResult:
    """Write a ``<asset>.c2pa.json`` sidecar next to ``asset_path``.

    Emits a C2PA 2.4-aligned manifest. Unknown action strings
    are tolerated but recorded under ``warnings`` so review tooling can
    surface them. ``cloud_trust_list`` is the optional URL of the
    operator's cloud-anchored trust list (the OpenCut signer
    never resolves the URL — it is propagated for downstream
    verifiers).
    """
    asset = Path(asset_path)
    if not asset.exists():
        raise FileNotFoundError(asset_path)

    target = Path(sidecar_path) if sidecar_path else asset.with_suffix(asset.suffix + ".c2pa.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    asset_sha = _sha256(asset)

    warnings: List[str] = []
    for act in (actions or []):
        if not is_known_c2pa_action(act.action):
            warnings.append(
                f"action {act.action!r} is not in the C2PA 2.4 documented vocabulary"
            )

    manifest = {
        "manifest_spec": SPEC_VERSION,
        "c2pa_spec_version": C2PA_SPEC_VERSION,
        "version": SPEC_VERSION,  # legacy alias kept for backward compat
        "claim_generator": claim_generator,
        "claim_id": "urn:uuid:" + str(uuid.uuid4()),
        "title": title or asset.name,
        "instance_id": "xmp:iid:" + str(uuid.uuid4()),
        "format": mimetypes.guess_type(asset.name)[0] or "application/octet-stream",
        "generated_at": _utc_now(),
        "asset": {
            "title": asset.name,
            "sha256": asset_sha,
            "hash_algorithm": "sha256",
            "bytes": asset.stat().st_size,
        },
        "ingredients": [ing.as_dict() for ing in (ingredients or [])],
        "actions": [act.as_dict() for act in (actions or [])],
    }
    if cloud_trust_list:
        manifest["cloud_trust_list"] = cloud_trust_list

    embedded = False
    embedding_method = ""
    embedded_path = ""
    if embed:
        if not _load_signing_key():
            warnings.append(
                "embedded C2PA output requires OPENCUT_C2PA_SIGNING_KEY; wrote unsigned sidecar fallback"
            )
        else:
            embedded, embedding_method, embedded_path, embed_warnings = _run_c2patool_embed(
                asset=asset,
                manifest=manifest,
                output_path=embedded_output_path,
                c2patool_path=c2patool_path,
            )
            warnings.extend(embed_warnings)
    if embed:
        manifest["embedding"] = {
            "requested": True,
            "status": "embedded" if embedded else "sidecar_only",
            "method": embedding_method,
            "embedded_path": embedded_path,
            "verified": embedded,
        }
    if warnings:
        manifest["warnings"] = warnings

    # Sign the canonical JSON (sorted keys, no whitespace) so verifiers
    # can reconstruct the bytes deterministically.
    canonical = _canonical_manifest(manifest)
    signature = _try_sign(canonical)
    if signature is not None:
        manifest["signature"] = signature

    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    staged = Path(staged_name)
    try:
        with staged.open("w", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(staged, target)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass

    return C2paSidecarResult(
        sidecar_path=str(target),
        asset_sha256=asset_sha,
        signed=signature is not None,
        signing_algorithm=str(signature.get("algorithm") or "") if signature else "",
        embedded=embedded,
        embedding_method=embedding_method,
        embedded_path=embedded_path,
        warnings=warnings,
    )


def verify_sidecar(sidecar_path: str) -> dict:
    """Verify a sidecar against the referenced asset.

    Returns ``{"asset_match": True/False, "signature_match": True/False, "warnings": [...]}``.
    """
    sidecar = Path(sidecar_path)
    if not sidecar.exists():
        raise FileNotFoundError(sidecar_path)

    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("C2PA sidecar manifest must be a JSON object")
    warnings: List[str] = []
    asset_match = False
    asset_found = False

    asset_meta = manifest.get("asset") or {}
    if not isinstance(asset_meta, dict):
        asset_meta = {}
        warnings.append("manifest asset metadata is not an object")
    asset_title = str(asset_meta.get("title") or "")
    asset_sha = str(asset_meta.get("sha256") or "")

    # The sidecar lives next to the asset by default; the manifest doesn't
    # carry an absolute path. Walk the same directory for a file matching
    # the recorded basename.
    asset_name = Path(asset_title).name
    if asset_name != asset_title:
        warnings.append("asset title path components were ignored during verification")
    candidate = sidecar.parent / asset_name
    if asset_name and candidate.is_file():
        asset_found = True
        actual_sha = _sha256(candidate)
        asset_match = actual_sha == asset_sha
        if not asset_match:
            warnings.append(
                f"asset hash mismatch (expected {asset_sha[:12]}, got {actual_sha[:12]})"
            )
    else:
        warnings.append(f"asset {asset_title!r} not found alongside sidecar")

    sig = manifest.get("signature")
    if sig and not isinstance(sig, dict):
        warnings.append("signature metadata is not an object")
        sig = {}
    signature_match = False
    signature_hash_match = False
    signature_valid = False
    key_trust = ""
    if sig:
        # Reconstruct the canonical bytes the way build_sidecar wrote them.
        # We must drop the signature field before hashing.
        rebuilt = {k: v for k, v in manifest.items() if k != "signature"}
        canonical = _canonical_manifest(rebuilt)
        signature_hash_match, signature_valid, key_trust = _verify_signature(
            sig, canonical, warnings
        )
        # A cryptographically valid signature from a key outside the operator
        # trust anchor must not report as a match: an attacker can always
        # re-sign an edited manifest with their own key.
        signature_match = (
            signature_hash_match and signature_valid and key_trust != "untrusted"
        )
    else:
        warnings.append("manifest is unsigned (operator did not configure OPENCUT_C2PA_SIGNING_KEY)")

    embedding = manifest.get("embedding") if isinstance(manifest.get("embedding"), dict) else {}
    embedded = bool(embedding.get("status") == "embedded")
    credential_verified = False
    if embedded:
        embedded_path = Path(str(embedding.get("embedded_path") or ""))
        if not embedded_path.is_absolute():
            embedded_path = sidecar.parent / embedded_path
        credential_verified, credential_warning = _verify_embedded_credential(
            embedded_path, manifest
        )
        if credential_warning:
            warnings.append(credential_warning)
    if not asset_found:
        status = "missing_asset"
    elif sig and signature_hash_match and signature_valid and key_trust == "untrusted":
        status = "untrusted_signature"
    elif sig and not signature_match:
        status = "tampered_manifest"
    elif not asset_match:
        status = "tampered_asset"
    elif credential_verified:
        status = "embedded_credential"
    elif embedded:
        status = "embedded_unverified"
    elif signature_match:
        status = "signed_sidecar"
    else:
        status = "unsigned_sidecar"

    return {
        "asset_match": bool(asset_match),
        "asset_found": bool(asset_found),
        "signature_present": bool(sig),
        "signature_hash_match": bool(signature_hash_match),
        "signature_valid": bool(signature_valid),
        "signature_match": bool(signature_match),
        "key_trust": key_trust,
        "embedded": embedded,
        "credential_verified": credential_verified,
        "embedded_path": str(embedding.get("embedded_path") or ""),
        "status": status,
        "warnings": warnings,
        "manifest": manifest,
    }
