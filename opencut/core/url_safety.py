"""Shared URL validation helpers for outbound network calls."""

import ipaddress
import socket
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
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not include embedded credentials")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid URL") from exc

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError(f"{label} has no hostname")
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".localdomain"):
        raise ValueError(f"{label} must not target localhost")

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # Not a canonical IP literal. Before treating it as a regular hostname,
        # catch the alternate IPv4 encodings that ``ipaddress`` rejects but the
        # OS resolver and most HTTP clients still expand to a real address:
        # decimal (``2130706433``), octal (``0177.0.0.1``), hex (``0x7f.0.0.1``)
        # and short forms (``127.1``). ``socket.inet_aton`` performs exactly
        # this expansion offline, so the SSRF guard stays deterministic and
        # network-free while no longer being bypassable by a numeric literal.
        try:
            packed = socket.inet_aton(hostname)
        except OSError:
            # Genuine DNS hostname. Resolving it would make validation depend on
            # network access and still would not fully prevent DNS rebinding
            # between validation and the actual request, so accept it as-is.
            return cleaned
        addr = ipaddress.ip_address(packed)

    reason = _blocked_ip_reason(addr)
    if reason == "localhost":
        raise ValueError(f"{label} must not target localhost")
    if reason:
        raise ValueError(f"{label} must not target {reason}")

    return cleaned
