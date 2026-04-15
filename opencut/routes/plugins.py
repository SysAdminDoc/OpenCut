"""
OpenCut Plugin Routes

Plugin discovery, listing, and management endpoints.
"""

import json
import logging
import os

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.security import get_json_dict, require_csrf, validate_path

logger = logging.getLogger("opencut")

plugins_bp = Blueprint("plugins", __name__)


def _json_object_or_400():
    try:
        return get_json_dict(silent=True), None
    except ValueError as e:
        return None, (
            jsonify({
                "error": str(e),
                "code": "INVALID_INPUT",
                "suggestion": "Send a top-level JSON object in the request body.",
            }),
            400,
        )

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
    data, error = _json_object_or_400()
    if error:
        return error
    source = data.get("source", "")
    if source and not isinstance(source, str):
        return jsonify({"error": "source path must be a string"}), 400

    if not source:
        return jsonify({"error": "source path is required"}), 400

    try:
        source = validate_path(source)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.isdir(source):
        return jsonify({"error": "source must be an existing directory"}), 400

    # Check for plugin.json in source
    manifest_path = os.path.join(source, "plugin.json")
    if not os.path.isfile(manifest_path):
        return jsonify({"error": "source directory must contain plugin.json"}), 400

    import shutil

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return jsonify({"error": f"Invalid plugin.json: {e}"}), 400

    name = manifest.get("name", "")
    if not name:
        return jsonify({"error": "Plugin manifest missing 'name' field"}), 400
    if len(name) > 100:
        return jsonify({"error": "Plugin name is too long"}), 400

    # Sanitize
    if not all(c.isalnum() or c in "-_" for c in name):
        return jsonify({"error": f"Invalid plugin name: {name}"}), 400

    dest = os.path.join(PLUGINS_DIR, name)
    if os.path.realpath(source) == os.path.realpath(dest):
        return jsonify({"error": "Plugin is already installed at that location"}), 409
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
    data, error = _json_object_or_400()
    if error:
        return error
    name = data.get("name", "")
    if name and not isinstance(name, str):
        return jsonify({"error": "Plugin name must be a string"}), 400

    if not name:
        return jsonify({"error": "Plugin name is required"}), 400

    # Sanitize
    if not all(c.isalnum() or c in "-_" for c in name):
        return jsonify({"error": f"Invalid plugin name: {name}"}), 400

    plugin_dir = os.path.join(PLUGINS_DIR, name)
    # Verify resolved path stays within PLUGINS_DIR (prevent symlink escape)
    real_plugin = os.path.realpath(plugin_dir)
    real_plugins = os.path.realpath(PLUGINS_DIR)
    if not real_plugin.startswith(real_plugins + os.sep) and real_plugin != real_plugins:
        return jsonify({"error": "Invalid plugin path"}), 400
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
        logger.warning("Could not unload plugin %s (unload_plugin not available)", name)

    logger.info("Plugin uninstalled: %s", name)
    return jsonify({"success": True, "message": f"Plugin '{name}' removed."})
