"""Host-header trust policy.

OpenCut binds to loopback and treats loopback peers as trusted, so the only
thing standing between a hostile web page and the local API is the browser's
own origin model. DNS rebinding defeats that model: the attacker's page keeps
its ``evil.example`` origin while the name resolves to ``127.0.0.1``, so the
request arrives from a loopback peer carrying an attacker-chosen ``Host`` and
a matching ``Origin``. Anything derived from ``request.host`` (including
``request.host_url``, which ``/health`` compares the ``Origin`` against) is
then attacker-controlled.

The defence is to decide up front which authorities this server answers for
and reject every other ``Host`` before any authentication, CSRF, or health
processing runs. Literal IP addresses cannot be rebound — a browser only sends
an IP ``Host`` when the user actually navigated to that IP — so loopback
literals are always trusted and other literals are trusted when the operator
opted into a remote bind. Names must be configured explicitly.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Iterable, Optional

#: Hostnames that always resolve to this machine.
LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})

#: Bind addresses that mean "every interface" rather than one authority.
WILDCARD_BIND_HOSTS = frozenset({"", "0.0.0.0", "::", "*"})

_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def split_host_port(raw: str) -> tuple[str, Optional[str]]:
    """Split a ``Host`` header value into ``(hostname, port)``.

    Handles bare names, ``name:port``, bracketed IPv6 (``[::1]``,
    ``[::1]:5679``) and unbracketed IPv6 literals (which carry no port).
    """
    value = str(raw or "").strip()
    if not value:
        return "", None
    if value.startswith("["):
        closing = value.find("]")
        if closing == -1:
            return "", None
        host = value[1:closing]
        remainder = value[closing + 1:]
        if not remainder:
            return host, None
        if not remainder.startswith(":"):
            return "", None
        return host, remainder[1:]
    if value.count(":") > 1:
        # Unbracketed IPv6 literal — no port can be attached.
        return value, None
    if ":" in value:
        host, _, port = value.partition(":")
        return host, port
    return value, None


def normalize_hostname(raw: str) -> str:
    """Lowercase a hostname and strip the FQDN root dot and IPv6 zone id."""
    host = str(raw or "").strip().lower()
    if not host:
        return ""
    host = host.strip("[]")
    if host.endswith(".") and not host.endswith(".."):
        host = host[:-1]
    # ``fe80::1%eth0`` classifies as the underlying interface address.
    return host.split("%", 1)[0]


def _is_ip_literal(hostname: str) -> Optional[ipaddress._BaseAddress]:
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        return None


def is_loopback_hostname(hostname: str) -> bool:
    """Return True when *hostname* can only ever reach this machine."""
    host = normalize_hostname(hostname)
    if not host:
        return False
    if host in LOOPBACK_HOSTNAMES:
        return True
    address = _is_ip_literal(host)
    return bool(address is not None and address.is_loopback)


def _port_is_valid(port: Optional[str]) -> bool:
    if port is None or port == "":
        return True
    if not port.isdigit():
        return False
    return 1 <= int(port) <= 65535


def build_trusted_hosts(
    configured: Iterable[str] = (),
    bind_host: str = "",
) -> frozenset[str]:
    """Return the normalized hostname allowlist for this server.

    Loopback names/literals are implicit. The bind address is added when it
    names a single authority; wildcard binds contribute nothing because they
    do not identify one. Operator-supplied entries are added verbatim, and an
    entry may start with ``.`` to trust a whole subtree (``.example.test``).
    """
    allowed = set(LOOPBACK_HOSTNAMES)
    bind = normalize_hostname(bind_host)
    if bind and bind not in WILDCARD_BIND_HOSTS:
        allowed.add(bind)
    for entry in configured or ():
        host, _ = split_host_port(entry)
        normalized = normalize_hostname(host)
        if normalized:
            allowed.add(normalized)
    return frozenset(allowed)


def bind_is_wildcard(bind_host: str) -> bool:
    return normalize_hostname(bind_host) in WILDCARD_BIND_HOSTS


def remote_bind_enabled() -> bool:
    return os.environ.get("OPENCUT_ALLOW_REMOTE", "").strip().lower() in _TRUE_ENV_VALUES


def is_trusted_host(
    raw_host: str,
    allowed: Iterable[str],
    *,
    allow_ip_literals: bool = False,
) -> bool:
    """Return True when a request's ``Host`` header names a trusted authority.

    *allow_ip_literals* trusts non-loopback IP literals. Those cannot be
    produced by DNS rebinding, so operators who opted into a remote bind keep
    plain ``http://192.168.1.5:5679`` access without having to enumerate every
    interface address.
    """
    host, port = split_host_port(raw_host)
    if not host or not _port_is_valid(port):
        return False
    hostname = normalize_hostname(host)
    if not hostname:
        return False
    if is_loopback_hostname(hostname):
        return True

    address = _is_ip_literal(hostname)
    if address is not None:
        if allow_ip_literals:
            return True
        # A configured literal still matches through the allowlist below.

    for entry in allowed:
        candidate = normalize_hostname(entry)
        if not candidate:
            continue
        if candidate.startswith("."):
            if hostname == candidate[1:] or hostname.endswith(candidate):
                return True
        elif hostname == candidate:
            return True
    return False
