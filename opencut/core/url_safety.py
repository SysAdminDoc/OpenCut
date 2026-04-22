"""Shared URL validation helpers for outbound network calls."""

import ipaddress
from urllib.parse import urlparse


def _blocked_ip_reason(addr: ipaddress._BaseAddress) -> str:
    """Return the rejection reason for an IP literal, or an empty string."""
    if addr.is_loopback or addr.is_unspecified:
        return "localhost"
    if addr.is_private or addr.is_link_local or addr.is_reserved or addr.is_multicast:
        return "private/reserved networks"
    return ""


def validate_public_http_url(url: str, *, label: str = "URL") -> str:
    """Validate that *url* is an HTTP(S) URL outside local/private networks.

    This is a defensive SSRF guard for user-configured outbound calls such as
    webhooks and plugin downloads. It blocks obvious local targets using only
    the URL structure itself: localhost-style names and literal IPs in
    loopback/private/reserved ranges are rejected, while regular hostnames are
    accepted without DNS lookups. That keeps validation deterministic and
    side-effect free in offline and test environments.
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
        # Regular hostnames are allowed here. Resolving them would make
        # validation dependent on network access and still would not fully
        # prevent DNS rebinding between validation and the actual request.
        return cleaned

    reason = _blocked_ip_reason(addr)
    if reason == "localhost":
        raise ValueError(f"{label} must not target localhost")
    if reason:
        raise ValueError(f"{label} must not target {reason}")

    return cleaned
