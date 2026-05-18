"""HMAC helpers for webhook payload signatures.

The webhook delivery surfaces are intentionally stdlib-only. These helpers keep
the signature contract small and pure so fuzz tests can exercise verification
without opening sockets or mutating webhook configuration files.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Union

_HEX_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SIGNATURE_PREFIX = "sha256="


def _body_bytes(body: Union[bytes, bytearray, memoryview, str]) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, (bytes, bytearray, memoryview)):
        return bytes(body)
    raise TypeError("webhook body must be bytes-like or str")


def sign_webhook_body(secret: str, body: Union[bytes, bytearray, memoryview, str]) -> str:
    """Return an ``sha256=<hex>`` HMAC signature for *body*.

    ``secret`` must be a non-empty string. The function does not read env vars
    or project state so callers can decide how secrets are scoped.
    """
    if not isinstance(secret, str) or not secret:
        raise ValueError("webhook signing secret is required")
    digest = hmac.new(secret.encode("utf-8"), _body_bytes(body), hashlib.sha256).hexdigest()
    return _SIGNATURE_PREFIX + digest


def normalize_webhook_signature(signature: str) -> str:
    """Normalize a supplied signature header to ``sha256=<hex>`` form."""
    if not isinstance(signature, str):
        return ""
    candidate = signature.strip()
    if _HEX_SHA256_RE.fullmatch(candidate):
        return _SIGNATURE_PREFIX + candidate.lower()
    if candidate.lower().startswith(_SIGNATURE_PREFIX):
        algo, value = candidate.split("=", 1)
        if algo.lower() == "sha256" and _HEX_SHA256_RE.fullmatch(value):
            return _SIGNATURE_PREFIX + value.lower()
    return ""


def verify_webhook_signature(
    secret: str,
    body: Union[bytes, bytearray, memoryview, str],
    signature: str,
) -> bool:
    """Return True when *signature* matches *body* under *secret*."""
    supplied = normalize_webhook_signature(signature)
    if not supplied:
        return False
    try:
        expected = sign_webhook_body(secret, body)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(expected, supplied)
