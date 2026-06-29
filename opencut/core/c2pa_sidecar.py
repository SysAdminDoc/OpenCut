"""C2PA provenance sidecars and embedded export credentials (F110 + F140).

OpenCut can always write a local sidecar — a single ``<media>.c2pa.json``
file that carries the provenance graph (chain of edits + tool identity +
asset hash). When the operator configures a signing key, the sidecar is
signed and self-verifiable. When ``c2patool`` is also available, OpenCut
can additionally ask the C2PA Tool to write a signed manifest into a
supported export asset while keeping the sidecar as an audit trail.

The fallback sidecar path is deliberately scoped:

* Lives next to the rendered file (no JUMBF embedding).
* Is signed only when the operator provides an Ed25519 private key
  through ``OPENCUT_C2PA_SIGNING_KEY``; otherwise the manifest is
  unsigned and the route response says so.
* Captures the same fields C2PA tooling reads: ``claim_generator``,
  ``title``, ``actions``, ``ingredients`` (input file hashes), and the
  output asset hash.

A real C2PA verifier won't accept the unsigned sidecar as a trust
artifact — but it will validate the *chain* and the asset hash, which
is what review pipelines actually use today. The route surface is
forward-compatible with the embedded variant: when C2PA Tool signing
is configured, ``build_sidecar(embed=True)`` returns the embedded output
path and the verifier distinguishes embedded, signed sidecar, unsigned
sidecar, missing asset, and tampered-manifest cases.

F140 — C2PA 2.3 alignment
=========================

The manifest semantics now target the C2PA 2.3 vocabulary published
in late 2024. Concretely:

* Every emitted manifest records the **C2PA 2.3 specification version**
  it targets in the ``c2pa_spec_version`` field, alongside our own
  sidecar wire-format version (``manifest_spec`` field).
* Action strings are validated against ``C2PA_ACTION_VOCABULARY`` —
  the documented set from the C2PA 2.3 working catalogue
  (``c2pa.created``, ``c2pa.edited``, ``c2pa.opened``, ``c2pa.placed``,
  ``c2pa.removed``, ``c2pa.cropped``, ``c2pa.transcribed``,
  ``c2pa.translated``, ``c2pa.captioned``, ``c2pa.published``, etc.).
  Unknown action strings still serialise (forward compatibility) but
  the ``warnings`` list flags them so downstream review tooling sees
  the mismatch.
* The optional ``cloud_trust_list`` slot is reserved for the
  ``trust_anchor_url`` value emitted by C2PA 2.3's cloud-anchored
  trust lists; OpenCut never resolves the URL itself but propagates
  it from the operator config so the embedded path can pick it up
  later.
* Live-video provenance is supported via the ``live`` boolean on
  ``C2paAction``; verifiers can use it to distinguish a captured
  livestream segment from a regular cut.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from opencut.openapi_registry import openapi_response_schema

logger = logging.getLogger("opencut")

SPEC_VERSION = "0.2-sidecar"  # OpenCut sidecar wire-format; bumped for the F140 C2PA 2.3 alignment
MANIFEST_SPEC_VERSION = SPEC_VERSION  # public alias
C2PA_SPEC_VERSION = "2.3"  # the C2PA specification version our action vocabulary follows
CLAIM_GENERATOR_DEFAULT = "OpenCut/1.33.1 (sidecar; c2pa-spec 2.3)"
SUPPORTED_EMBED_EXTENSIONS = {".jpg", ".jpeg", ".mp4", ".png"}
DEFAULT_C2PATOOL_TIMEOUT_SECONDS = 120


# F140 — C2PA 2.3 action vocabulary. This is the documented set of
# action strings; unknown actions still serialise but get flagged.
# Keep the tuple sorted so test diffs are deterministic.
C2PA_ACTION_VOCABULARY: tuple = (
    "c2pa.captioned",
    "c2pa.color_adjustments",
    "c2pa.converted",
    "c2pa.created",
    "c2pa.cropped",
    "c2pa.dubbed",
    "c2pa.edited",
    "c2pa.filtered",
    "c2pa.opened",
    "c2pa.placed",
    "c2pa.published",
    "c2pa.redacted",
    "c2pa.removed",
    "c2pa.resized",
    "c2pa.transcoded",
    "c2pa.transcribed",
    "c2pa.translated",
    "c2pa.unknown",
)


def is_known_c2pa_action(name: str) -> bool:
    """Return True when *name* belongs to the C2PA 2.3 documented vocabulary."""
    return name in C2PA_ACTION_VOCABULARY


@dataclass
class C2paAction:
    action: str           # e.g. "c2pa.created", "c2pa.edited", "c2pa.cropped"
    when: str             # ISO-8601 UTC
    parameters: dict = field(default_factory=dict)
    # F140 — C2PA 2.3 fields:
    # ``live`` distinguishes a livestream segment from a normal cut.
    # ``software_agent`` carries the tool identity per-action (some
    # workflows do filler-removal in OpenCut + colour grade in a
    # different app within the same render).
    live: bool = False
    software_agent: str = ""

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
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key
    except Exception:
        logger.info(
            "c2pa_sidecar: cryptography library not installed; manifest will be unsigned. "
            "`pip install cryptography` to enable Ed25519 signing."
        )
        return None
    try:
        priv = load_pem_private_key(key, password=None)
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


def _verify_signature(sig: dict, canonical: bytes, warnings: List[str]) -> tuple[bool, bool]:
    """Return ``(hash_match, signature_valid)`` for sidecar signature metadata."""
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
        return hash_match, False
    if not value:
        warnings.append("signature value is missing")
        return hash_match, False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        pub = load_pem_public_key(public_key_pem.encode("utf-8"))
        if not isinstance(pub, Ed25519PublicKey):
            warnings.append("signature public key is not Ed25519")
            return hash_match, False
        pub.verify(bytes.fromhex(value), canonical)
        signature_valid = True
    except Exception as exc:
        warnings.append(f"signature verification failed: {exc}")
    return hash_match, signature_valid


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
    actions = []
    for action in manifest.get("actions", []):
        if not isinstance(action, dict):
            continue
        actions.append({
            "action": action.get("action") or "c2pa.unknown",
            "when": action.get("when") or manifest.get("generated_at"),
            "softwareAgent": action.get("software_agent") or manifest.get("claim_generator"),
            "parameters": action.get("parameters") or {},
        })

    definition: Dict[str, Any] = {
        "claim_generator": manifest.get("claim_generator") or CLAIM_GENERATOR_DEFAULT,
        "title": manifest.get("title") or manifest.get("asset", {}).get("title") or "OpenCut export",
        "format": manifest.get("format") or "application/octet-stream",
        "assertions": [
            {"label": "c2pa.actions", "data": {"actions": actions}},
            {"label": "org.opencut.provenance", "data": manifest},
        ],
    }

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

    output = Path(output_path) if output_path else _default_embedded_output_path(asset)
    output.parent.mkdir(parents=True, exist_ok=True)
    definition = _c2patool_manifest_definition(manifest)

    with tempfile.TemporaryDirectory(prefix="opencut_c2pa_") as root:
        manifest_path = Path(root) / "manifest.json"
        manifest_path.write_text(json.dumps(definition, indent=2, sort_keys=True), encoding="utf-8")
        cmd = [
            tool,
            str(asset),
            "-m",
            str(manifest_path),
            "-o",
            str(output),
            "-f",
        ]
        env = os.environ.copy()
        raw_key = os.environ.get("OPENCUT_C2PA_SIGNING_KEY", "").strip()
        raw_cert = os.environ.get("OPENCUT_C2PA_SIGNING_CERT", "").strip()
        if raw_key and not Path(raw_key).exists():
            env.setdefault("C2PA_PRIVATE_KEY", raw_key)
        if raw_cert and not Path(raw_cert).exists():
            env.setdefault("C2PA_SIGN_CERT", raw_cert)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_C2PATOOL_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        warnings.append(f"c2patool embed failed with exit {result.returncode}: {detail[-300:]}")
        return False, "", "", warnings
    if not output.exists():
        warnings.append("c2patool reported success but did not create the embedded output")
        return False, "", "", warnings
    return True, "c2patool", str(output), warnings


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

    F140 — emits a C2PA 2.3-aligned manifest. Unknown action strings
    are tolerated but recorded under ``warnings`` so review tooling can
    surface them. ``cloud_trust_list`` is the optional URL of the
    operator's C2PA 2.3 cloud-anchored trust list (the OpenCut signer
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
                f"action {act.action!r} is not in the C2PA 2.3 documented vocabulary"
            )

    manifest = {
        "manifest_spec": SPEC_VERSION,
        "c2pa_spec_version": C2PA_SPEC_VERSION,
        "version": SPEC_VERSION,  # legacy alias kept for backward compat
        "claim_generator": claim_generator,
        "claim_id": "urn:uuid:" + str(uuid.uuid4()),
        "title": title or asset.name,
        "instance_id": "xmp:iid:" + str(uuid.uuid4()),
        "format": asset.suffix.lstrip(".") or "application/octet-stream",
        "generated_at": _utc_now(),
        "asset": {
            "title": asset.name,
            "sha256": asset_sha,
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
        }
    if warnings:
        manifest["warnings"] = warnings

    # Sign the canonical JSON (sorted keys, no whitespace) so verifiers
    # can reconstruct the bytes deterministically.
    canonical = _canonical_manifest(manifest)
    signature = _try_sign(canonical)
    if signature is not None:
        manifest["signature"] = signature

    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
    if sig:
        # Reconstruct the canonical bytes the way build_sidecar wrote them.
        # We must drop the signature field before hashing.
        rebuilt = {k: v for k, v in manifest.items() if k != "signature"}
        canonical = _canonical_manifest(rebuilt)
        signature_hash_match, signature_valid = _verify_signature(sig, canonical, warnings)
        signature_match = signature_hash_match and signature_valid
    else:
        warnings.append("manifest is unsigned (operator did not configure OPENCUT_C2PA_SIGNING_KEY)")

    embedding = manifest.get("embedding") if isinstance(manifest.get("embedding"), dict) else {}
    embedded = bool(embedding.get("status") == "embedded")
    if not asset_found:
        status = "missing_asset"
    elif sig and not signature_match:
        status = "tampered_manifest"
    elif not asset_match:
        status = "tampered_asset"
    elif embedded and signature_match:
        status = "embedded_signed"
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
        "embedded": embedded,
        "embedded_path": str(embedding.get("embedded_path") or ""),
        "status": status,
        "warnings": warnings,
        "manifest": manifest,
    }
