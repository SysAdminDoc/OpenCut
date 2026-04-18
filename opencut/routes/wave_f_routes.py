"""
OpenCut Wave F Routes (v1.23.0) — cross-cutting infrastructure surface.

Endpoints:
- ``GET  /api/openapi.json``              — OpenAPI 3.1 spec (all routes)
- ``GET  /api/docs``                      — Swagger UI hosted at /api/docs
- ``GET  /api/routes``                    — flat list of every registered route
- ``GET  /system/gpu-semaphore``          — GPU-exclusive semaphore state
- ``GET  /system/rate-limits``            — rate-limit category snapshot
- ``POST /system/temp-cleanup/sweep``     — manually trigger a temp sweep
- ``GET  /system/temp-cleanup/status``    — temp-cleanup config + last sweep
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Blueprint, Response, current_app, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf

logger = logging.getLogger("opencut")

wave_f_bp = Blueprint("wave_f", __name__)


# Cached between calls because generating the spec on every hit walks
# ~1,200 routes. Invalidate by process restart — spec drift is a
# deploy-time concern, not a runtime concern.
_OPENAPI_CACHE: Dict[str, Any] = {"spec": None}


# ---------------------------------------------------------------------------
# OpenAPI + Swagger UI
# ---------------------------------------------------------------------------

@wave_f_bp.route("/api/openapi.json", methods=["GET"])
def route_openapi_json():
    """Serve the auto-generated OpenAPI 3.1 spec.

    Query params:
    - ``refresh=1``  — bypass the module-level cache
    - ``server=URL`` — advertise a custom ``servers`` entry
    """
    try:
        from opencut.core import openapi_spec

        server = request.args.get("server", "")
        refresh = request.args.get("refresh", "").lower() in ("1", "true", "yes")

        if _OPENAPI_CACHE["spec"] is None or refresh or server:
            _OPENAPI_CACHE["spec"] = openapi_spec.generate_spec(
                current_app,
                server_url=server,
            )
        return jsonify(_OPENAPI_CACHE["spec"])
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "openapi_json")


@wave_f_bp.route("/api/docs", methods=["GET"])
def route_docs():
    """Serve Swagger UI at ``/api/docs`` pointing at the spec endpoint."""
    from opencut.core import openapi_spec

    spec_url = "/api/openapi.json"
    html = openapi_spec.swagger_ui_html(spec_url)
    return Response(html, status=200, mimetype="text/html")


@wave_f_bp.route("/api/routes", methods=["GET"])
def route_routes():
    """Flat route catalogue — useful for plugin / test tooling.

    Returns one entry per (rule, method) pair with blueprint name,
    queue-allowlist membership, and CSRF + rate-limit metadata.
    """
    from opencut.core.openapi_spec import _route_uses_csrf
    from opencut.core.rate_limit_categories import category_of

    try:
        from opencut.routes.jobs_routes import _ALLOWED_QUEUE_ENDPOINTS
    except Exception:  # noqa: BLE001
        _ALLOWED_QUEUE_ENDPOINTS = frozenset()

    entries = []
    for rule in current_app.url_map.iter_rules():
        methods = sorted((rule.methods or set()) - {"HEAD", "OPTIONS"})
        if not methods:
            continue
        view = current_app.view_functions.get(rule.endpoint)
        bp_name = (rule.endpoint.rsplit(".", 1)[0] if "." in rule.endpoint else "")
        entries.append({
            "rule": str(rule.rule),
            "methods": methods,
            "endpoint": rule.endpoint,
            "blueprint": bp_name,
            "csrf_required": _route_uses_csrf(view) if view else False,
            "queueable": str(rule.rule) in _ALLOWED_QUEUE_ENDPOINTS,
            "rate_category": category_of(view) or None,
        })
    entries.sort(key=lambda e: (e["rule"], e["methods"]))
    return jsonify({
        "count": len(entries),
        "routes": entries,
    })


# ---------------------------------------------------------------------------
# GPU semaphore
# ---------------------------------------------------------------------------

@wave_f_bp.route("/system/gpu-semaphore", methods=["GET"])
def route_gpu_semaphore():
    from opencut.core import gpu_semaphore
    return jsonify(gpu_semaphore.status().to_dict())


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------

@wave_f_bp.route("/system/rate-limits", methods=["GET"])
def route_rate_limits():
    from opencut.core import rate_limit_categories as rlc
    snap = rlc.status()
    return jsonify({
        "categories": [
            {
                "name": s.name,
                "max_concurrent": s.max_concurrent,
                "active": s.active,
                "available": s.available,
                "rejected_total": s.rejected_total,
                "acquired_total": s.acquired_total,
            }
            for s in snap.values()
        ],
    })


# ---------------------------------------------------------------------------
# Temp-file cleanup
# ---------------------------------------------------------------------------

@wave_f_bp.route("/system/temp-cleanup/status", methods=["GET"])
def route_temp_cleanup_status():
    from opencut.core import temp_cleanup
    return jsonify({
        "default_ttl_seconds": temp_cleanup.DEFAULT_TTL,
        "default_interval_seconds": temp_cleanup.DEFAULT_INTERVAL,
        "prefixes": list(temp_cleanup.DEFAULT_PREFIXES),
        "hint": (
            "Tune via OPENCUT_TEMP_CLEANUP_TTL, "
            "OPENCUT_TEMP_CLEANUP_INTERVAL, "
            "OPENCUT_TEMP_CLEANUP_PREFIXES env vars."
        ),
    })


@wave_f_bp.route("/system/temp-cleanup/sweep", methods=["POST"])
@require_csrf
def route_temp_cleanup_sweep():
    """Manually trigger a sweep (same logic as the periodic background pass)."""
    from opencut.core import temp_cleanup
    from opencut.security import safe_int

    data = request.get_json(silent=True) or {}
    ttl = safe_int(
        data.get("ttl_seconds", temp_cleanup.DEFAULT_TTL),
        temp_cleanup.DEFAULT_TTL, min_val=60, max_val=86400 * 7,
    )
    try:
        report = temp_cleanup.sweep(ttl_seconds=ttl)
        return jsonify(report.to_dict())
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "temp_cleanup_sweep")
