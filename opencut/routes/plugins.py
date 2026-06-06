"""
OpenCut Plugin Routes

Plugin discovery, listing, and management endpoints.
"""

import json
import logging
import os
import shutil
import time
import uuid

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.security import get_json_dict, is_path_within, require_csrf, validate_path

logger = logging.getLogger("opencut")

plugins_bp = Blueprint("plugins", __name__)


def _json_object_or_400():
    try:
        return get_json_dict(), None
    except ValueError as e:
        return None, (
            jsonify({
                "error": str(e),
                "code": "INVALID_INPUT",
                "suggestion": "Send a top-level JSON object in the request body.",
            }),
            400,
        )


def _valid_plugin_name(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c in "-_" for c in name)


def _quarantine_root() -> str:
    parent = os.path.dirname(os.path.realpath(PLUGINS_DIR))
    return os.path.join(parent, "plugins_quarantine")


def _quarantine_id_for(name: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{name}-{stamp}-{uuid.uuid4().hex[:8]}"


def _quarantine_path(quarantine_id: str) -> str:
    if not _valid_plugin_name(quarantine_id):
        raise ValueError("Invalid quarantine id")
    root = _quarantine_root()
    path = os.path.join(root, quarantine_id)
    if not is_path_within(path, root):
        raise ValueError("Invalid quarantine path")
    return path


def _metadata_path(quarantine_path: str) -> str:
    return os.path.join(quarantine_path, ".opencut-quarantine.json")


def _write_quarantine_metadata(quarantine_path: str, metadata: dict) -> None:
    with open(_metadata_path(quarantine_path), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def _read_quarantine_metadata(quarantine_id: str) -> tuple[dict | None, str | None]:
    try:
        path = _quarantine_path(quarantine_id)
    except ValueError as exc:
        return None, str(exc)
    if not os.path.isdir(path):
        return None, "Plugin quarantine entry not found"
    try:
        with open(_metadata_path(path), "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"Invalid quarantine metadata: {exc}"
    if not isinstance(metadata, dict):
        return None, "Invalid quarantine metadata"
    metadata["quarantine_id"] = quarantine_id
    metadata["quarantine_path"] = path
    return metadata, None


def _list_quarantine_entries() -> list[dict]:
    root = _quarantine_root()
    if not os.path.isdir(root):
        return []
    entries = []
    for entry in sorted(os.listdir(root)):
        if not _valid_plugin_name(entry):
            continue
        metadata, error = _read_quarantine_metadata(entry)
        entries.append(metadata if metadata is not None else {"quarantine_id": entry, "error": error})
    return entries

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
            "jobs": p.get("jobs", []),
            "loaded_jobs": loaded.get(p["name"], {}).get("jobs", []),
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
    if not _valid_plugin_name(name):
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
    """Move a plugin into quarantine by name."""
    data, error = _json_object_or_400()
    if error:
        return error
    name = data.get("name", "")
    if name and not isinstance(name, str):
        return jsonify({"error": "Plugin name must be a string"}), 400

    if not name:
        return jsonify({"error": "Plugin name is required"}), 400

    # Sanitize
    if not _valid_plugin_name(name):
        return jsonify({"error": f"Invalid plugin name: {name}"}), 400

    plugin_dir = os.path.join(PLUGINS_DIR, name)
    # Verify resolved path stays within PLUGINS_DIR (prevent symlink escape)
    if not is_path_within(plugin_dir, PLUGINS_DIR):
        return jsonify({"error": "Invalid plugin path"}), 400
    if not os.path.isdir(plugin_dir):
        return jsonify({"error": f"Plugin '{name}' not found"}), 404
    confirm_name = data.get("confirm_name", "")
    if confirm_name != name:
        return jsonify({
            "error": "confirm_name must match the plugin name before uninstall",
            "code": "CONFIRMATION_REQUIRED",
        }), 400

    quarantine_id = _quarantine_id_for(name)
    quarantine_root = _quarantine_root()
    quarantine_path = _quarantine_path(quarantine_id)
    try:
        os.makedirs(quarantine_root, exist_ok=True)
        shutil.move(plugin_dir, quarantine_path)
        metadata = {
            "name": name,
            "quarantine_id": quarantine_id,
            "quarantine_path": quarantine_path,
            "original_path": plugin_dir,
            "created_at": time.time(),
            "restore_route": "/plugins/quarantine/restore",
            "delete_route": "/plugins/quarantine/delete",
        }
        _write_quarantine_metadata(quarantine_path, metadata)
    except OSError as e:
        return safe_error(e, "uninstall_plugin")

    # Remove from loaded registry if present
    try:
        from opencut.core.plugins import unload_plugin
        unload_plugin(name)
    except ImportError:
        logger.warning("Could not unload plugin %s (unload_plugin not available)", name)

    logger.info("Plugin quarantined: %s -> %s", name, quarantine_path)
    return jsonify({
        "success": True,
        "name": name,
        "quarantined": True,
        "quarantine_id": quarantine_id,
        "quarantine_path": quarantine_path,
        "restore_route": "/plugins/quarantine/restore",
        "delete_route": "/plugins/quarantine/delete",
        "message": f"Plugin '{name}' moved to quarantine.",
    })


@plugins_bp.route("/plugins/quarantine/list", methods=["GET"])
def list_plugin_quarantine():
    """List quarantined plugin directories."""
    entries = _list_quarantine_entries()
    return jsonify({
        "entries": entries,
        "quarantine_dir": _quarantine_root(),
        "total": len(entries),
    })


@plugins_bp.route("/plugins/quarantine/restore", methods=["POST"])
@require_csrf
def restore_plugin_quarantine():
    """Restore a quarantined plugin directory."""
    data, error = _json_object_or_400()
    if error:
        return error
    quarantine_id = data.get("quarantine_id", "")
    if not isinstance(quarantine_id, str) or not quarantine_id:
        return jsonify({"error": "quarantine_id is required"}), 400
    metadata, meta_error = _read_quarantine_metadata(quarantine_id)
    if metadata is None:
        return jsonify({"error": meta_error or "Plugin quarantine entry not found"}), 404
    name = str(metadata.get("name", ""))
    if not _valid_plugin_name(name):
        return jsonify({"error": "Invalid plugin name in quarantine metadata"}), 400
    dest = os.path.join(PLUGINS_DIR, name)
    if not is_path_within(dest, PLUGINS_DIR):
        return jsonify({"error": "Invalid plugin restore path"}), 400
    if os.path.exists(dest):
        return jsonify({"error": f"Plugin '{name}' already exists"}), 409
    try:
        os.makedirs(PLUGINS_DIR, exist_ok=True)
        metadata_path = _metadata_path(metadata["quarantine_path"])
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        shutil.move(metadata["quarantine_path"], dest)
    except OSError as exc:
        return safe_error(exc, "restore_plugin_quarantine")
    logger.info("Plugin restored from quarantine: %s -> %s", quarantine_id, dest)
    return jsonify({
        "success": True,
        "name": name,
        "restored": True,
        "path": dest,
    })


@plugins_bp.route("/plugins/quarantine/delete", methods=["POST"])
@require_csrf
def delete_plugin_quarantine():
    """Permanently delete a quarantined plugin directory."""
    data, error = _json_object_or_400()
    if error:
        return error
    quarantine_id = data.get("quarantine_id", "")
    if not isinstance(quarantine_id, str) or not quarantine_id:
        return jsonify({"error": "quarantine_id is required"}), 400
    metadata, meta_error = _read_quarantine_metadata(quarantine_id)
    if metadata is None:
        return jsonify({"error": meta_error or "Plugin quarantine entry not found"}), 404
    name = str(metadata.get("name", ""))
    if data.get("confirm_name", "") != name:
        return jsonify({
            "error": "confirm_name must match the plugin name before permanent delete",
            "code": "CONFIRMATION_REQUIRED",
        }), 400
    try:
        shutil.rmtree(metadata["quarantine_path"])
    except OSError as exc:
        return safe_error(exc, "delete_plugin_quarantine")
    logger.info("Plugin quarantine entry permanently deleted: %s", quarantine_id)
    return jsonify({
        "success": True,
        "name": name,
        "deleted": True,
        "quarantine_id": quarantine_id,
    })
