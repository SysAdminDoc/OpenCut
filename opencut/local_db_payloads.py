"""Payload spilling helpers for local SQLite JSON fields."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

SPILL_MARKER_KEY = "_opencut_payload_spill"


def _coerce_max_bytes(max_bytes: int) -> int:
    coerced = int(max_bytes)
    if coerced < 1:
        raise ValueError("max_bytes must be positive")
    return coerced


def spill_json_if_needed(
    encoded_json: str,
    *,
    base_dir: str,
    namespace: str,
    field_name: str,
    max_bytes: int,
) -> str:
    """Return inline JSON or a small spill marker for oversized JSON text."""
    limit = _coerce_max_bytes(max_bytes)
    raw = encoded_json.encode("utf-8")
    if len(raw) <= limit:
        return encoded_json

    digest = hashlib.sha256(raw).hexdigest()
    relative_path = os.path.join("payload_spills", namespace, field_name, f"{digest}.json")
    spill_path = os.path.join(base_dir, relative_path)
    os.makedirs(os.path.dirname(spill_path), exist_ok=True)

    if not os.path.exists(spill_path):
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{digest}.",
            suffix=".tmp",
            dir=os.path.dirname(spill_path),
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            os.replace(tmp_path, spill_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    marker = {
        SPILL_MARKER_KEY: True,
        "_spilled": True,
        "_reason": "payload_too_large",
        "field": field_name,
        "bytes": len(raw),
        "sha256": digest,
        "path": spill_path,
        "relative_path": relative_path.replace(os.sep, "/"),
        "max_inline_bytes": limit,
    }
    return json.dumps(marker, sort_keys=True)


def decode_json_or_spill_marker(encoded_json: str):
    """Decode JSON and preserve spill markers as API-visible metadata."""
    value = json.loads(encoded_json)
    if isinstance(value, dict) and value.get(SPILL_MARKER_KEY) is True:
        return value
    return value
