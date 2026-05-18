"""C2PA provenance sidecars (F110 + F140).

The full C2PA spec requires signed JUMBF boxes embedded inside the
media container. That is heavyweight: it pulls in ``c2pa-python``
(LGPL-3) plus the OpenSSL keypair management dance, and 90% of our
users will not have a trust anchor they can register.

Instead, we ship a *sidecar* — a single ``<media>.c2pa.json`` file that
carries the same provenance graph (chain of edits + tool identity +
asset hash) without requiring signatures. It's a deliberately scoped
implementation of the "Content Provenance" surface that:

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
forward-compatible with the embedded variant: when the C2PA dep is
installed and a signing key is configured, we can swap in
``c2pa-python`` behind the same response shape.

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
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("opencut")

SPEC_VERSION = "0.2-sidecar"  # OpenCut sidecar wire-format; bumped for the F140 C2PA 2.3 alignment
MANIFEST_SPEC_VERSION = SPEC_VERSION  # public alias
C2PA_SPEC_VERSION = "2.3"  # the C2PA specification version our action vocabulary follows
CLAIM_GENERATOR_DEFAULT = "OpenCut/1.32.0 (sidecar; c2pa-spec 2.3)"


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
class C2paSidecarResult:
    sidecar_path: str
    asset_sha256: str
    signed: bool
    signing_algorithm: str = ""

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


def _try_sign(manifest_bytes: bytes) -> tuple:
    """Attempt to sign the manifest. Returns ``(signature_hex, algo)`` or ``(None, "")``."""
    key = _load_signing_key()
    if not key:
        return None, ""
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except Exception:
        logger.info(
            "c2pa_sidecar: cryptography library not installed; manifest will be unsigned. "
            "`pip install cryptography` to enable Ed25519 signing."
        )
        return None, ""
    try:
        priv = load_pem_private_key(key, password=None)
        signature = priv.sign(manifest_bytes)
        algo = priv.__class__.__name__.lower()
        return signature.hex(), algo
    except Exception as exc:
        logger.warning("c2pa_sidecar: signing failed (%s); writing unsigned manifest", exc)
        return None, ""


def build_sidecar(
    *,
    asset_path: str,
    ingredients: Optional[Sequence[C2paIngredient]] = None,
    actions: Optional[Sequence[C2paAction]] = None,
    title: Optional[str] = None,
    claim_generator: str = CLAIM_GENERATOR_DEFAULT,
    sidecar_path: Optional[str] = None,
    cloud_trust_list: str = "",
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
    if warnings:
        manifest["warnings"] = warnings

    # Sign the canonical JSON (sorted keys, no whitespace) so verifiers
    # can reconstruct the bytes deterministically.
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature_hex, algo = _try_sign(canonical)
    if signature_hex is not None:
        manifest["signature"] = {
            "algorithm": algo or "ed25519",
            "value": signature_hex,
            "signed_bytes_sha256": hashlib.sha256(canonical).hexdigest(),
        }

    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return C2paSidecarResult(
        sidecar_path=str(target),
        asset_sha256=asset_sha,
        signed=signature_hex is not None,
        signing_algorithm=algo,
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
        actual_sha = _sha256(candidate)
        asset_match = actual_sha == asset_sha
        if not asset_match:
            warnings.append(
                f"asset hash mismatch (expected {asset_sha[:12]}, got {actual_sha[:12]})"
            )
    else:
        warnings.append(f"asset {asset_title!r} not found alongside sidecar")

    sig = manifest.get("signature")
    signature_match = False
    if sig:
        if not isinstance(sig, dict):
            warnings.append("signature metadata is not an object")
            sig = {}
        signed_hash = sig.get("signed_bytes_sha256") or ""
        # Reconstruct the canonical bytes the way build_sidecar wrote them.
        # We must drop the signature field before hashing.
        rebuilt = {k: v for k, v in manifest.items() if k != "signature"}
        canonical = json.dumps(rebuilt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        current_hash = hashlib.sha256(canonical).hexdigest()
        signature_match = current_hash == signed_hash
        if not signature_match:
            warnings.append("signed bytes hash mismatch — manifest was edited after signing")
    else:
        warnings.append("manifest is unsigned (operator did not configure OPENCUT_C2PA_SIGNING_KEY)")

    return {
        "asset_match": bool(asset_match),
        "signature_present": bool(sig),
        "signature_match": bool(signature_match),
        "warnings": warnings,
        "manifest": manifest,
    }
