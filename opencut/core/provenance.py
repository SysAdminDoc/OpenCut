"""
OpenCut Provenance Chain & Audit Trail

Generates a signed JSON manifest documenting the full provenance chain
for any processed file:  source hash, every job operation applied, and
an HMAC-SHA256 signature for tamper detection.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("opencut")

_KEY_FILE = os.path.join(os.path.expanduser("~"), ".opencut", "provenance_key.bin")
_key_lock = __import__("threading").Lock()


def _get_provenance_key() -> bytes:
    """Return the HMAC signing key as bytes.

    Priority:
    1. ``OPENCUT_PROVENANCE_KEY`` environment variable (hex or raw string).
    2. Persisted 32-byte random key at ``~/.opencut/provenance_key.bin``.
       Generated on first call and reused on all subsequent calls so
       signatures remain verifiable across server restarts.
    """
    env_key = os.environ.get("OPENCUT_PROVENANCE_KEY", "")
    if env_key:
        return env_key.encode("utf-8")

    with _key_lock:
        # Read existing key
        if os.path.isfile(_KEY_FILE):
            try:
                with open(_KEY_FILE, "rb") as fh:
                    data = fh.read()
                if len(data) >= 32:
                    return data
            except OSError:
                pass

        # Generate and persist a new 32-byte key
        new_key = os.urandom(32)
        try:
            os.makedirs(os.path.dirname(_KEY_FILE), exist_ok=True)
            # Write with owner-read-only permissions
            fd = os.open(_KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(new_key)
            logger.info("Generated new provenance HMAC key at %s", _KEY_FILE)
        except OSError as exc:
            logger.warning("Could not persist provenance key: %s — signatures will not survive restart", exc)
        return new_key


def _sha256_file(filepath: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sign_manifest(manifest_dict: dict) -> str:
    """Produce HMAC-SHA256 hex signature over canonical JSON of *manifest_dict*."""
    # Remove any existing signature before computing
    to_sign = {k: v for k, v in manifest_dict.items() if k != "signature"}
    canonical = json.dumps(to_sign, sort_keys=True, separators=(",", ":"))
    return hmac.new(
        _get_provenance_key(), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def generate_provenance_manifest(
    filepath: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    Build a provenance manifest for *filepath*.

    Queries the job store for every operation applied to this file,
    computes SHA-256 hashes, and signs the manifest with HMAC-SHA256.

    Args:
        filepath: Absolute path to the source/output file.
        output_path: Where to write the sidecar ``.json`` manifest.
                     Defaults to ``<filepath>.provenance.json``.

    Returns:
        The manifest dict (also written to disk).
    """
    from opencut import __version__

    filepath = os.path.abspath(filepath)

    # --- source file metadata ---
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    source_hash = _sha256_file(filepath)
    source_size = os.path.getsize(filepath)

    # --- query job store for operations on this file ---
    operations = []
    try:
        from opencut.job_store import _get_conn, init_db

        init_db()
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, type, status, payload_json, created_at, completed_at "
            "FROM jobs WHERE filepath = ? ORDER BY created_at ASC",
            (filepath,),
        ).fetchall()

        for row in rows:
            op = {
                "job_id": row[0],
                "job_type": row[1],
                "status": row[2],
                "timestamp": row[4],
            }
            # Parse parameters from payload
            if row[3]:
                try:
                    op["parameters"] = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    op["parameters"] = {}
            else:
                op["parameters"] = {}
            if row[5]:
                op["completed_at"] = row[5]
            operations.append(op)
    except Exception as exc:
        logger.warning("Could not query job store for provenance: %s", exc)

    # --- build manifest ---
    manifest = {
        "opencut_version": __version__,
        "generated_at": time.time(),
        "source_file": {
            "path": filepath,
            "hash_sha256": source_hash,
            "size_bytes": source_size,
        },
        "operations": operations,
        "output_file": {},
    }

    # If there's an apparent output sibling (e.g. _clean, _trimmed), hash it
    # Otherwise the caller can extend the manifest later.
    # For now, output_file hash mirrors source since the file IS the output.
    manifest["output_file"]["hash_sha256"] = source_hash

    # --- sign ---
    manifest["signature"] = _sign_manifest(manifest)

    # --- write sidecar JSON ---
    if output_path is None:
        output_path = filepath + ".provenance.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Provenance manifest written to %s", output_path)
    return manifest


def verify_provenance_manifest(manifest: dict) -> bool:
    """
    Verify the HMAC signature on *manifest*.

    Returns True if the signature is valid.
    """
    expected = _sign_manifest(manifest)
    actual = manifest.get("signature", "")
    return hmac.compare_digest(expected, actual)
