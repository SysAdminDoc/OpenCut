"""Shared URL validation helpers for outbound network calls."""

import ipaddress
from urllib.parse import urlparse


def validate_public_http_url(url: str, *, label: str = "URL") -> str:
    """Validate that *url* is an HTTP(S) URL outside local/private networks.

    This is a defensive SSRF guard for user-configured outbound calls such as
    webhooks and plugin downloads. It blocks obvious local targets without
    performing DNS lookups, keeping validation deterministic and side-effect
    free.
    """
    if not isinstance(url, str):
        raise ValueError(f"{label} is required")

    cleaned = url.strip()
    if not cleaned:
        raise ValueError(f"{label} is required")
    if any(ch in cleaned for ch in ("\r", "\n", "\x00")):
        raise ValueError(f"{label} contains invalid characters")

    try:
        parsed = urlparse(cleaned)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must use http:// or https:// and include a host")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError(f"{label} has no hostname")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".localdomain"):
        raise ValueError(f"{label} must not target localhost")

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return cleaned

    if addr.is_loopback or addr.is_unspecified:
        raise ValueError(f"{label} must not target localhost")
    if (
        addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    ):
        raise ValueError(f"{label} must not target private/reserved networks")

    return cleaned
