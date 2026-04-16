"""
OpenCut Plugin System

Discovers, validates, and loads plugins from ~/.opencut/plugins/.
Each plugin is a directory with a plugin.json manifest and optional Python/JS files.
"""

import importlib
import importlib.util
import json
import logging
import os
import sys
import threading

logger = logging.getLogger("opencut")

PLUGINS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "plugins")

# Required fields in plugin.json
_REQUIRED_MANIFEST_FIELDS = {"name", "version", "description"}

# Loaded plugins registry
_loaded_plugins = {}
_plugins_lock = threading.Lock()


def _validate_manifest(manifest, plugin_dir):
    """Validate a plugin manifest dict.

    Returns (is_valid, error_message).
    """
    missing = _REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    if missing:
        return False, f"Missing required fields: {', '.join(sorted(missing))}"

    name = manifest.get("name", "")
    if not name or not isinstance(name, str):
        return False, "Plugin name must be a non-empty string"

    # Sanitize name — only allow alphanumeric, hyphens, underscores
    if not all(c.isalnum() or c in "-_" for c in name):
        return False, f"Plugin name contains invalid characters: {name}"

    version = manifest.get("version", "")
    if not version or not isinstance(version, str):
        return False, "Plugin version must be a non-empty string"

    return True, ""


def discover_plugins():
    """Scan the plugins directory and return list of discovered plugin manifests.

    Returns:
        list of dicts, each with keys: name, version, description, author,
        path (absolute directory path), valid (bool), error (str or None),
        enabled (bool), routes (list), ui (dict or None)
    """
    if not os.path.isdir(PLUGINS_DIR):
        return []

    plugins = []
    try:
        entries = sorted(os.listdir(PLUGINS_DIR))
    except OSError as e:
        logger.warning("Cannot read plugins directory: %s", e)
        return []

    for entry in entries:
        plugin_dir = os.path.join(PLUGINS_DIR, entry)
        if not os.path.isdir(plugin_dir):
            continue

        manifest_path = os.path.join(plugin_dir, "plugin.json")
        if not os.path.isfile(manifest_path):
            plugins.append({
                "name": entry,
                "version": "",
                "description": "",
                "author": "",
                "path": plugin_dir,
                "valid": False,
                "error": "Missing plugin.json manifest",
                "enabled": False,
                "routes": [],
                "ui": None,
            })
            continue

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            plugins.append({
                "name": entry,
                "version": "",
                "description": "",
                "author": "",
                "path": plugin_dir,
                "valid": False,
                "error": f"Invalid plugin.json: {e}",
                "enabled": False,
                "routes": [],
                "ui": None,
            })
            continue

        valid, error = _validate_manifest(manifest, plugin_dir)

        plugins.append({
            "name": manifest.get("name", entry),
            "version": manifest.get("version", ""),
            "description": manifest.get("description", ""),
            "author": manifest.get("author", ""),
            "path": plugin_dir,
            "valid": valid,
            "error": error if not valid else None,
            "enabled": manifest.get("enabled", True) and valid,
            "routes": manifest.get("routes", []),
            "ui": manifest.get("ui", None),
        })

    return plugins


def load_plugin_routes(app, plugin_info):
    """Load a plugin's Flask routes into the app.

    Looks for a `routes.py` file in the plugin directory that defines
    a Flask Blueprint named `plugin_bp`.

    Args:
        app: Flask application
        plugin_info: dict from discover_plugins()

    Returns:
        bool: True if routes were loaded successfully
    """
    plugin_dir = plugin_info["path"]
    plugin_name = plugin_info["name"]
    routes_file = os.path.join(plugin_dir, "routes.py")

    if not os.path.isfile(routes_file):
        return False

    # Reject duplicate plugin names up front. Flask's ``register_blueprint``
    # raises a confusing ``ValueError: The name 'X' is already registered``
    # when two plugins share a name; catching it here gives the operator a
    # clear "duplicate" signal and prevents a second plugin from silently
    # shadowing the first if Flask's behavior changes.
    with _plugins_lock:
        if plugin_name in _loaded_plugins:
            logger.warning(
                "Plugin name collision: '%s' already loaded — refusing to register",
                plugin_name,
            )
            return False

    try:
        # Add plugin dir to sys.path temporarily for imports
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        spec = importlib.util.spec_from_file_location(
            f"opencut_plugin_{plugin_name}", routes_file
        )
        if spec is None or spec.loader is None:
            logger.warning("Cannot load plugin module: %s", plugin_name)
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for plugin_bp Blueprint
        bp = getattr(module, "plugin_bp", None)
        if bp is None:
            logger.warning("Plugin %s routes.py has no 'plugin_bp' Blueprint", plugin_name)
            return False

        # Register under /plugins/<name>/ prefix
        app.register_blueprint(bp, url_prefix=f"/plugins/{plugin_name}")

        with _plugins_lock:
            _loaded_plugins[plugin_name] = {
                "info": plugin_info,
                "module": module,
                "blueprint": bp,
            }

        logger.info("Loaded plugin: %s v%s", plugin_name, plugin_info["version"])
        return True

    except Exception as e:
        logger.error("Failed to load plugin %s: %s", plugin_name, e)
        return False


def load_all_plugins(app):
    """Discover and load all valid, enabled plugins.

    Args:
        app: Flask application

    Returns:
        dict with keys: loaded (list of names), failed (list of {name, error}), skipped (list of names)
    """
    plugins = discover_plugins()
    result = {"loaded": [], "failed": [], "skipped": []}

    for plugin in plugins:
        if not plugin["valid"] or not plugin["enabled"]:
            result["skipped"].append(plugin["name"])
            continue

        routes_file = os.path.join(plugin["path"], "routes.py")
        if not os.path.isfile(routes_file):
            # Plugin has no routes — might be UI-only or config-only
            with _plugins_lock:
                _loaded_plugins[plugin["name"]] = {
                    "info": plugin,
                    "module": None,
                    "blueprint": None,
                }
            result["loaded"].append(plugin["name"])
            continue

        try:
            success = load_plugin_routes(app, plugin)
            if success:
                result["loaded"].append(plugin["name"])
            else:
                result["failed"].append({"name": plugin["name"], "error": "Blueprint not found"})
        except Exception as e:
            result["failed"].append({"name": plugin["name"], "error": str(e)})

    if result["loaded"]:
        logger.info("Plugins loaded: %s", ", ".join(result["loaded"]))
    if result["failed"]:
        logger.warning("Plugins failed: %s", ", ".join(f["name"] for f in result["failed"]))

    return result


def get_loaded_plugins():
    """Return info about all currently loaded plugins."""
    with _plugins_lock:
        return {
            name: {
                "name": data["info"]["name"],
                "version": data["info"]["version"],
                "description": data["info"]["description"],
                "author": data["info"]["author"],
                "has_routes": data["blueprint"] is not None,
                "ui": data["info"].get("ui"),
            }
            for name, data in _loaded_plugins.items()
        }


def unload_plugin(name):
    """Remove a plugin from the loaded registry.

    Note: Flask doesn't support unregistering blueprints at runtime,
    so the routes remain active until server restart.
    """
    with _plugins_lock:
        if name in _loaded_plugins:
            del _loaded_plugins[name]
            return True
    return False
