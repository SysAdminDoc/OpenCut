"""
OpenCut Settings Routes

Presets, favorites, workflows, settings import/export, job time estimation.
"""

import time

from flask import Blueprint, request, jsonify

from opencut.jobs import _safe_error
from opencut.helpers import compute_estimate
from opencut.security import require_csrf, safe_float
from opencut.user_data import (
    load_presets, save_presets,
    load_favorites, save_favorites,
    load_workflows, save_workflows,
)

settings_bp = Blueprint("settings", __name__)


# ---------------------------------------------------------------------------
# User Presets
# ---------------------------------------------------------------------------
@settings_bp.route("/presets", methods=["GET"])
def list_presets():
    """List all saved user presets."""
    return jsonify(load_presets())


@settings_bp.route("/presets/save", methods=["POST"])
@require_csrf
def save_preset():
    """Save a named preset."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Preset name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Preset name too long"}), 400
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        return jsonify({"error": "Settings must be an object"}), 400
    presets = load_presets()
    presets[name] = {"settings": settings, "saved": time.time()}
    save_presets(presets)
    return jsonify({"success": True, "name": name})


@settings_bp.route("/presets/delete", methods=["POST"])
@require_csrf
def delete_preset():
    """Delete a named preset."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    presets = load_presets()
    if name in presets:
        del presets[name]
        save_presets(presets)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Favorites / Pinned Operations
# ---------------------------------------------------------------------------
@settings_bp.route("/favorites", methods=["GET"])
def get_favorites():
    """Get user's favorite/pinned operations."""
    return jsonify(load_favorites())


@settings_bp.route("/favorites/save", methods=["POST"])
@require_csrf
def save_favorites_route():
    """Save user's favorite operations list."""
    data = request.get_json(force=True)
    favorites = data.get("favorites", [])
    if not isinstance(favorites, list):
        return jsonify({"error": "favorites must be a list"}), 400
    if len(favorites) > 100:
        return jsonify({"error": "Too many favorites (max 100)"}), 400
    save_favorites(favorites)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Custom Workflow Templates
# ---------------------------------------------------------------------------
@settings_bp.route("/workflows/list", methods=["GET"])
def list_workflows():
    """List saved custom workflow templates."""
    return jsonify(load_workflows())


@settings_bp.route("/workflows/save", methods=["POST"])
@require_csrf
def save_workflow():
    """Save a custom workflow template."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    steps = data.get("steps", [])
    if not name:
        return jsonify({"error": "Name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Workflow name too long"}), 400
    if not steps or not isinstance(steps, list):
        return jsonify({"error": "Steps must be a non-empty list"}), 400
    if len(steps) > 50:
        return jsonify({"error": "Too many workflow steps (max 50)"}), 400
    workflows = load_workflows()
    # Update or add
    found = False
    for wf in workflows:
        if wf.get("name") == name:
            wf["steps"] = steps
            wf["updated"] = time.time()
            found = True
            break
    if not found:
        workflows.append({"name": name, "steps": steps, "created": time.time()})
    save_workflows(workflows)
    return jsonify({"success": True})


@settings_bp.route("/workflows/delete", methods=["POST"])
@require_csrf
def delete_workflow():
    """Delete a custom workflow template."""
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    workflows = load_workflows()
    workflows = [wf for wf in workflows if wf.get("name") != name]
    save_workflows(workflows)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Settings Import/Export
# ---------------------------------------------------------------------------
@settings_bp.route("/settings/export", methods=["GET"])
def export_settings():
    """Export all OpenCut settings (presets, favorites, workflows) as a single JSON bundle."""
    bundle = {"version": "1.3.0", "exported": time.time()}
    try:
        bundle["presets"] = load_presets()
    except Exception:
        bundle["presets"] = {}
    try:
        bundle["favorites"] = load_favorites()
    except Exception:
        bundle["favorites"] = []
    try:
        bundle["workflows"] = load_workflows()
    except Exception:
        bundle["workflows"] = []
    return jsonify(bundle)


@settings_bp.route("/settings/import", methods=["POST"])
@require_csrf
def import_settings():
    """Import settings bundle (presets, favorites, workflows)."""
    data = request.get_json(force=True)
    imported = []
    if "presets" in data and isinstance(data["presets"], dict):
        existing = load_presets()
        existing.update(data["presets"])
        if len(existing) > 500:
            return jsonify({"error": "Too many presets (max 500)"}), 400
        save_presets(existing)
        imported.append("presets")
    if "favorites" in data and isinstance(data["favorites"], list):
        save_favorites(data["favorites"])
        imported.append("favorites")
    if "workflows" in data and isinstance(data["workflows"], list):
        save_workflows(data["workflows"])
        imported.append("workflows")
    return jsonify({"success": True, "imported": imported})


# ---------------------------------------------------------------------------
# Job Time Estimation
# ---------------------------------------------------------------------------
@settings_bp.route("/system/estimate-time", methods=["POST"])
@require_csrf
def estimate_job_time():
    """Estimate processing time based on historical data."""
    data = request.get_json(force=True)
    job_type = data.get("type", "")
    file_duration = safe_float(data.get("file_duration", 0))
    return jsonify(compute_estimate(job_type, file_duration))
