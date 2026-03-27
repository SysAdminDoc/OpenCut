"""
OpenCut Plugin Routes

Plugin discovery, listing, and management endpoints.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf

logger = logging.getLogger("opencut")

plugins_bp = Blueprint("plugins", __name__)

try:
    from ..core.plugins import (
        PLUGINS_DIR,
        discover_plugins,
        get_loaded_plugins,
    )
except ImportError:
    from opencut.core.plugins import (
        PLUGINS_DIR,
        discover_plugins,
        get_loaded_plugins,
    )


@plugins_bp.route("/plugins/list", methods=["GET"])
def list_plugins():
    """List all discovered plugins (loaded and unloaded).

    Returns JSON with discovered plugins and their status.
    """
    discovered = discover_plugins()
    loaded = get_loaded_plugins()

    plugins = []
    for p in discovered:
        info = {
            "name": p["name"],
            "version": p["version"],
            "description": p["description"],
            "author": p["author"],
            "valid": p["valid"],
            "error": p["error"],
            "enabled": p["enabled"],
            "loaded": p["name"] in loaded,
            "has_routes": bool(p["routes"]) or os.path.isfile(os.path.join(p["path"], "routes.py")),
            "ui": p["ui"],
        }
        plugins.append(info)

    return jsonify({
        "plugins": plugins,
        "plugins_dir": PLUGINS_DIR,
        "total": len(plugins),
        "loaded": len(loaded),
    })


@plugins_bp.route("/plugins/loaded", methods=["GET"])
def loaded_plugins():
    """List only currently loaded plugins."""
    loaded = get_loaded_plugins()
    return jsonify({"plugins": loaded})


@plugins_bp.route("/plugins/install", methods=["POST"])
@require_csrf
def install_plugin():
    """Install a plugin from a directory path or archive.

    For now, supports copying a plugin directory into the plugins folder.
    Future: support .zip archives and GitHub URLs.
    """
    data = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "")

    if not source:
        return jsonify({"error": "source path is required"}), 400

    if not os.path.isdir(source):
        return jsonify({"error": "source must be an existing directory"}), 400

    # Check for plugin.json in source
    manifest_path = os.path.join(source, "plugin.json")
    if not os.path.isfile(manifest_path):
        return jsonify({"error": "source directory must contain plugin.json"}), 400

    import json
    import shutil

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Invalid plugin.json: {e}"}), 400

    name = manifest.get("name", "")
    if not name:
        return jsonify({"error": "Plugin manifest missing 'name' field"}), 400

    # Sanitize
    if not all(c.isalnum() or c in "-_" for c in name):
        return jsonify({"error": f"Invalid plugin name: {name}"}), 400

    dest = os.path.join(PLUGINS_DIR, name)
    if os.path.exists(dest):
        return jsonify({"error": f"Plugin '{name}' already installed. Remove it first."}), 409

    os.makedirs(PLUGINS_DIR, exist_ok=True)

    try:
        shutil.copytree(source, dest)
    except OSError as e:
        return safe_error(e, "install_plugin")

    logger.info("Plugin installed: %s → %s", name, dest)
    return jsonify({
        "success": True,
        "name": name,
        "version": manifest.get("version", ""),
        "message": f"Plugin '{name}' installed. Restart server to load routes.",
    })


@plugins_bp.route("/plugins/uninstall", methods=["POST"])
@require_csrf
def uninstall_plugin():
    """Uninstall (delete) a plugin by name."""
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "")

    if not name:
        return jsonify({"error": "Plugin name is required"}), 400

    # Sanitize
    if not all(c.isalnum() or c in "-_" for c in name):
        return jsonify({"error": f"Invalid plugin name: {name}"}), 400

    plugin_dir = os.path.join(PLUGINS_DIR, name)
    if not os.path.isdir(plugin_dir):
        return jsonify({"error": f"Plugin '{name}' not found"}), 404

    import shutil
    try:
        shutil.rmtree(plugin_dir)
    except OSError as e:
        return safe_error(e, "uninstall_plugin")

    # Remove from loaded registry if present
    try:
        from opencut.core.plugins import unload_plugin
        unload_plugin(name)
    except ImportError:
        pass

    logger.info("Plugin uninstalled: %s", name)
    return jsonify({"success": True, "message": f"Plugin '{name}' removed."})
