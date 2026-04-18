"""
Route deprecation registry.

Lets maintainers mark a route as deprecated + scheduled for removal
without hunting for every client that still calls it.  Deprecations
surface in three places:

1. **OpenAPI spec** — ``deprecated: true`` on the operation, plus an
   ``x-opencut-deprecation`` extension block carrying the removal
   version + migration hint.
2. **Response headers** — every response from a deprecated route
   carries ``Deprecation: true`` (RFC 8594) + ``Sunset: <date>`` +
   ``Link: </path/to/replacement>; rel="successor-version"`` when a
   replacement route is specified.
3. **Server logs** — the first call per process writes an INFO log
   including the client's ``User-Agent`` and ``X-Request-ID`` so ops
   can track who's still on the legacy surface.

Usage
-----
::

    from opencut.core.deprecation import deprecated_route

    @wave_x_bp.route("/video/old-endpoint", methods=["POST"])
    @require_csrf
    @deprecated_route(
        remove_in="2.0.0",
        replacement="/video/new-endpoint",
        reason="Renamed to match the verb-noun convention.",
    )
    def route_old_endpoint():
        ...

The decorator stores metadata on the route function via
``func.__opencut_deprecation__`` which the OpenAPI generator picks up.
"""

from __future__ import annotations

import functools
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("opencut")


@dataclass
class DeprecationInfo:
    """Metadata attached to a deprecated route."""
    rule: str = ""
    reason: str = ""
    remove_in: str = ""
    replacement: str = ""
    sunset_date: str = ""      # ISO 8601, e.g. "2026-10-01"
    since: str = ""            # version the deprecation landed
    notified: bool = False


# Module-level registry: rule-or-name -> DeprecationInfo
_registry_lock = threading.Lock()
_registry: Dict[str, DeprecationInfo] = {}


def list_deprecations() -> Dict[str, Dict[str, Any]]:
    """Snapshot every registered deprecation — dict form."""
    with _registry_lock:
        return {k: asdict(v) for k, v in _registry.items()}


def get_deprecation(key: str) -> Optional[DeprecationInfo]:
    with _registry_lock:
        return _registry.get(key)


def register(key: str, info: DeprecationInfo) -> None:
    """Register or update a deprecation entry under ``key``."""
    with _registry_lock:
        info.rule = info.rule or key
        _registry[key] = info


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def deprecated_route(
    *,
    reason: str = "",
    remove_in: str = "",
    replacement: str = "",
    sunset_date: str = "",
    since: str = "",
) -> Callable:
    """Mark a Flask route as deprecated.

    Args:
        reason: Human-readable explanation (shown in OpenAPI + logs).
        remove_in: Target version for removal (e.g. ``"2.0.0"``).
        replacement: Absolute path of the replacement endpoint; emitted
            via ``Link: ...; rel="successor-version"``.
        sunset_date: ISO-8601 date for the RFC 8594 ``Sunset`` header.
        since: Version the deprecation landed in (default: current
            ``__version__``).

    The decorator is inert at import time — metadata is stamped on the
    wrapper, but the route still runs normally. Clients get the
    standardised deprecation headers at response time.
    """

    def decorator(func: Callable) -> Callable:
        from flask import make_response, request

        try:
            from opencut import __version__ as _ver
        except Exception:  # noqa: BLE001
            _ver = "unknown"

        info = DeprecationInfo(
            rule=getattr(func, "__name__", "<anon>"),
            reason=str(reason)[:400],
            remove_in=str(remove_in)[:32],
            replacement=str(replacement)[:200],
            sunset_date=str(sunset_date)[:40],
            since=str(since or _ver)[:32],
        )
        setattr(func, "__opencut_deprecation__", info)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # One-time INFO log per route per process so rollback-
            # sensitive operators can see who's still calling.
            if not info.notified:
                ua = request.headers.get("User-Agent", "<unknown>")[:120]
                logger.info(
                    "deprecated route hit: %s (reason=%r, remove_in=%s, ua=%r)",
                    request.path, info.reason, info.remove_in, ua,
                )
                info.notified = True

            resp = make_response(func(*args, **kwargs))
            resp.headers["Deprecation"] = "true"
            if info.sunset_date:
                resp.headers["Sunset"] = info.sunset_date
            link_parts = []
            if info.replacement:
                link_parts.append(
                    f'<{info.replacement}>; rel="successor-version"'
                )
            if link_parts:
                existing = resp.headers.get("Link") or ""
                resp.headers["Link"] = (
                    existing + ", " + ", ".join(link_parts) if existing
                    else ", ".join(link_parts)
                )
            return resp

        # Also keep the info accessible through the wrapper so
        # __wrapped__ walks still find it.
        setattr(wrapper, "__opencut_deprecation__", info)
        # Register under the function's qualified name; the OpenAPI
        # generator will upgrade the key to the actual URL rule at
        # spec-generation time when it has access to the url_map.
        register(func.__qualname__, info)
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# OpenAPI integration helper
# ---------------------------------------------------------------------------

def enrich_openapi_spec(spec: Dict[str, Any], app) -> Dict[str, Any]:
    """Walk ``spec["paths"]`` and mark deprecated operations.

    The function looks up each operation's view-func on ``app`` and, if
    an ``__opencut_deprecation__`` attribute is present, sets
    ``deprecated: true`` + an ``x-opencut-deprecation`` extension
    block.  The generator leaves this optional — callers invoke
    it explicitly when they want deprecation metadata in the spec.

    Returns the mutated spec for chaining convenience.
    """
    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        for method, op in list(path_item.items()):
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            # operationId looks like "post_bp.endpoint"
            operation_id = op.get("operationId", "")
            try:
                endpoint = operation_id.split("_", 1)[1].replace("_", ".")
            except (IndexError, AttributeError):
                endpoint = ""
            view = app.view_functions.get(endpoint) if endpoint else None
            if view is None:
                continue
            info = None
            cur = view
            seen = set()
            while cur is not None and id(cur) not in seen:
                seen.add(id(cur))
                info = getattr(cur, "__opencut_deprecation__", None)
                if info:
                    break
                cur = getattr(cur, "__wrapped__", None)
            if not info:
                continue
            op["deprecated"] = True
            op["x-opencut-deprecation"] = {
                "reason": info.reason,
                "remove_in": info.remove_in,
                "replacement": info.replacement,
                "sunset_date": info.sunset_date,
                "since": info.since,
            }
    return spec


def check_deprecation_available() -> bool:
    return True
