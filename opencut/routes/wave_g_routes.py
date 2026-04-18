"""
OpenCut Wave G Routes (v1.24.0) — second wide-net pass.

Endpoints:
- ``GET  /system/disk``                   — disk-space report across tracked mounts
- ``POST /system/disk/preflight``         — check if a path has N MB free
- ``POST /system/disk/track``             — register an additional mount to track
- ``GET  /system/request-correlation``    — current request_id + middleware state
- ``GET  /system/deprecations``           — registered deprecated-route metadata
- ``POST /system/sbom``                   — generate an SBOM on demand
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf, validate_path

logger = logging.getLogger("opencut")

wave_g_bp = Blueprint("wave_g", __name__)


# ---------------------------------------------------------------------------
# Disk monitor
# ---------------------------------------------------------------------------

@wave_g_bp.route("/system/disk", methods=["GET"])
def route_disk_report():
    """Run a one-shot disk probe across every tracked mount."""
    try:
        from opencut.core import disk_monitor
        rep = disk_monitor.report()
        return jsonify(rep.to_dict())
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "disk_report")


@wave_g_bp.route("/system/disk/preflight", methods=["POST"])
@require_csrf
def route_disk_preflight():
    """Check if ``path`` has ``required_mb`` MB free — gate heavy jobs."""
    try:
        from opencut.core import disk_monitor
        from opencut.security import safe_int

        data = request.get_json(force=True) or {}
        path = str(data.get("path") or "").strip()
        if not path:
            return jsonify({
                "error": "'path' is required",
                "code": "INVALID_INPUT",
            }), 400
        path = validate_path(path)
        required_mb = safe_int(
            data.get("required_mb", disk_monitor.DEFAULT_PREFLIGHT_MB),
            disk_monitor.DEFAULT_PREFLIGHT_MB,
            min_val=0, max_val=1_000_000,
        )
        return jsonify(disk_monitor.preflight(path, required_mb=required_mb))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "disk_preflight")


@wave_g_bp.route("/system/disk/track", methods=["POST"])
@require_csrf
def route_disk_track():
    """Register an additional path to track in future reports."""
    try:
        from opencut.core import disk_monitor
        data = request.get_json(force=True) or {}
        path = str(data.get("path") or "").strip()
        if not path:
            return jsonify({
                "error": "'path' is required",
                "code": "INVALID_INPUT",
            }), 400
        path = validate_path(path)
        added = disk_monitor.register_tracked_path(path)
        return jsonify({
            "path": path, "added": added,
            "tracked": disk_monitor.tracked_paths(),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "disk_track")


# ---------------------------------------------------------------------------
# Request correlation
# ---------------------------------------------------------------------------

@wave_g_bp.route("/system/request-correlation", methods=["GET"])
def route_request_correlation():
    """Report the current request's ID + middleware wiring state.

    Useful for clients that want to confirm their request-ID round-trip
    (e.g. sending ``X-Request-ID`` and expecting it echoed back as a
    server-generated fresh ID, per the request_correlation module's
    never-trust-user-input policy).
    """
    from flask import g

    from opencut.core.request_correlation import HEADER_NAME, get_request_id
    client_id = getattr(g, "client_request_id", "") or ""
    return jsonify({
        "request_id": get_request_id(),
        "client_request_id": client_id,
        "header_name": HEADER_NAME,
        "middleware_installed": True,
    })


# ---------------------------------------------------------------------------
# Deprecation registry
# ---------------------------------------------------------------------------

@wave_g_bp.route("/system/deprecations", methods=["GET"])
def route_deprecations():
    """Report every registered deprecated route + its migration hints."""
    from opencut.core import deprecation
    return jsonify({
        "deprecations": deprecation.list_deprecations(),
    })


# ---------------------------------------------------------------------------
# SBOM on demand
# ---------------------------------------------------------------------------

@wave_g_bp.route("/system/sbom", methods=["POST"])
@require_csrf
def route_sbom():
    """Generate a CycloneDX 1.5 SBOM in-process (JSON only).

    Thin wrapper around ``scripts/sbom.py`` so clients don't need to
    shell out.  Returns the BOM document directly rather than writing
    to disk — keeps the Flask layer stateless.
    """
    try:
        # Defer the import so the route blueprint loads even if
        # ``scripts/`` isn't on the sys.path in some deployments.
        import importlib.util
        import os
        sbom_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts", "sbom.py",
        )
        spec = importlib.util.spec_from_file_location("_opencut_sbom", sbom_path)
        if spec is None or spec.loader is None:
            return jsonify({
                "error": "scripts/sbom.py not found",
                "code": "MISSING_DEPENDENCY",
            }), 503
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        bom = mod.build_sbom()
        return jsonify({"bom": bom, "format": "CycloneDX 1.5"})
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "sbom")
