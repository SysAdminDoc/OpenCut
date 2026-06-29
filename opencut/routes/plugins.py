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
from opencut.security import (
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    is_path_within,
    require_csrf,
    safe_bool,
    validate_path,
    verify_destructive_confirm_token,
)

logger = logging.getLogger("opencut")

plugins_bp = Blueprint("plugins", __name__)

_CAPABILITY_LABELS = {
    "http.routes": ("HTTP routes", "runtime"),
    "jobs.register": ("Jobs", "runtime"),
    "host.filesystem": ("Host files", "sensitive"),
    "host.network": ("Network", "sensitive"),
    "models.download": ("Model downloads", "sensitive"),
    "ui.panel": ("Panel UI", "ui"),
}


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


def _path_size_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for filename in files:
                try:
                    total += os.path.getsize(os.path.join(root, filename))
                except OSError:
                    continue
    return total


def _plugin_target(path: str, *, name: str, category: str, reversible: bool) -> dict:
    return {
        "path": path,
        "name": name,
        "category": category,
        "root": PLUGINS_DIR if category == "plugin-installation" else _quarantine_root(),
        "type": "directory" if os.path.isdir(path) else "file",
        "bytes": _path_size_bytes(path),
        "reversible": reversible,
    }


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


def _message_contains(messages: list[str], *needles: str) -> bool:
    joined = "\n".join(str(m).lower() for m in messages)
    return any(needle.lower() in joined for needle in needles)


def _capability_badges(capabilities: list[str]) -> list[dict]:
    badges = []
    for capability in capabilities:
        label, kind = _CAPABILITY_LABELS.get(capability, (capability, "unknown"))
        badges.append({
            "id": capability,
            "label": label,
            "kind": kind,
        })
    return badges


def _validate_plugin_trust(plugin: dict) -> dict:
    try:
        from opencut.core.plugin_manifest import validate_plugin_manifest

        validation = validate_plugin_manifest(plugin.get("path", ""))
        return validation.as_dict()
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return {
            "valid": False,
            "errors": [f"Plugin trust validation failed: {exc}"],
            "warnings": [],
        }


def _trust_source(plugin: dict, validation: dict) -> str:
    errors = [str(e) for e in validation.get("errors") or []]
    warnings = [str(w) for w in validation.get("warnings") or []]
    if _message_contains(errors, "plugin.lock.json missing"):
        return "lock_missing"
    if _message_contains(warnings, "plugin.lock.json missing", "opencut_plugin_allow_unsigned"):
        return "unsigned_allowed"
    if _message_contains(errors, "plugin.lock.json", "sha-256", "lock declares", "files absent from lock"):
        return "lock_failed"
    if not plugin.get("valid", False) or _message_contains(errors, "plugin.json", "manifest:", "api_version", "capabilities:"):
        return "invalid_manifest"
    if validation.get("valid"):
        return "locked"
    return "failed_validation"


def _load_status(plugin: dict, loaded: dict, validation: dict) -> str:
    name = plugin.get("name", "")
    if name in loaded:
        return "loaded"
    if not plugin.get("valid", False) or not validation.get("valid", False):
        return "failed"
    if not plugin.get("enabled", False):
        return "skipped"
    return "skipped"


def _plugin_trust_entry(plugin: dict, loaded: dict) -> dict:
    validation = _validate_plugin_trust(plugin)
    capabilities = list(plugin.get("capabilities") or [])
    source = _trust_source(plugin, validation)
    status = _load_status(plugin, loaded, validation)
    loaded_info = loaded.get(plugin.get("name", ""), {})
    return {
        "name": plugin.get("name", ""),
        "version": plugin.get("version", ""),
        "description": plugin.get("description", ""),
        "author": plugin.get("author", ""),
        "enabled": bool(plugin.get("enabled", False)),
        "loaded": status == "loaded",
        "load_status": status,
        "has_routes": bool(plugin.get("routes")) or os.path.isfile(os.path.join(plugin.get("path", ""), "routes.py")),
        "routes": list(plugin.get("routes") or []),
        "jobs": list(plugin.get("jobs") or []),
        "loaded_jobs": list(loaded_info.get("jobs") or []),
        "capabilities": capabilities,
        "capability_badges": _capability_badges(capabilities),
        "ui": plugin.get("ui"),
        "valid": bool(plugin.get("valid", False)),
        "error": plugin.get("error") or "; ".join(validation.get("errors") or []),
        "trust": {
            "valid": bool(validation.get("valid", False)),
            "source": source,
            "errors": list(validation.get("errors") or []),
            "warnings": list(validation.get("warnings") or []),
            "lock_missing": source == "lock_missing",
            "unsigned_allowed": source == "unsigned_allowed",
        },
    }


def _marketplace_snapshot() -> dict:
    payload = {
        "status": "uncached",
        "plugins": [],
        "installed_plugins": [],
        "total": 0,
        "installed_total": 0,
        "error": "",
    }
    try:
        from opencut.core import plugin_marketplace as marketplace

        installed = marketplace.list_installed_plugins()
        payload["installed_plugins"] = [
            {
                "plugin_id": p.plugin_id,
                "name": p.name,
                "version": p.version,
                "installed_version": p.installed_version or p.version,
            }
            for p in installed
        ]
        payload["installed_total"] = len(installed)

        cache_path = getattr(marketplace, "REGISTRY_CACHE", "")
        if not cache_path or not os.path.exists(cache_path):
            return payload

        with open(cache_path, "r", encoding="utf-8") as fh:
            registry = json.load(fh)
        plugins = marketplace._parse_registry(registry)  # noqa: SLF001 - internal route snapshot, no network fetch
        payload["plugins"] = [
            {
                "plugin_id": p.plugin_id,
                "name": p.name,
                "version": p.version,
                "author": p.author,
                "description": p.description,
                "tags": p.tags,
                "installed": p.installed,
                "installed_version": p.installed_version,
            }
            for p in plugins
        ]
        payload["total"] = len(plugins)
        payload["status"] = "cached"
    except Exception as exc:  # pragma: no cover - defensive reporting path
        payload["status"] = "unavailable"
        payload["error"] = str(exc)
    return payload


def _plugin_trust_summary(entries: list[dict], quarantine_entries: list[dict], marketplace: dict) -> dict:
    failed = [
        p for p in entries
        if p.get("load_status") == "failed" or p.get("trust", {}).get("errors")
    ]
    skipped = [p for p in entries if p.get("load_status") == "skipped"]
    loaded = [p for p in entries if p.get("load_status") == "loaded"]
    lock_missing = [p for p in entries if p.get("trust", {}).get("lock_missing")]
    unsigned = [p for p in entries if p.get("trust", {}).get("unsigned_allowed")]
    return {
        "total": len(entries),
        "loaded": len(loaded),
        "skipped": len(skipped),
        "failed": len(failed),
        "lock_missing": len(lock_missing),
        "unsigned": len(unsigned),
        "quarantined": len(quarantine_entries),
        "marketplace": int(marketplace.get("total") or 0),
        "marketplace_installed": int(marketplace.get("installed_total") or 0),
    }


def _plugin_trust_actions() -> dict:
    return {
        "uninstall": {
            "route": "/plugins/uninstall",
            "method": "POST",
            "dry_run": True,
            "confirmation_required": True,
            "confirm_name": "name",
            "confirm_token_source": "dry_run.confirm_token",
            "result": "quarantine",
        },
        "restore_quarantine": {
            "route": "/plugins/quarantine/restore",
            "method": "POST",
            "confirmation_required": False,
        },
        "delete_quarantine": {
            "route": "/plugins/quarantine/delete",
            "method": "POST",
            "dry_run": True,
            "confirmation_required": True,
            "confirm_name": "name",
            "confirm_token_source": "dry_run.confirm_token",
            "result": "permanent_delete",
        },
        "marketplace": {
            "registry_route": "/plugins/registry",
            "install_route": "/plugins/marketplace/install",
            "update_route": "/plugins/update",
        },
    }

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


@plugins_bp.route("/plugins/trust", methods=["GET"])
def plugin_trust_dashboard():
    """Return a read-only trust dashboard for Settings panels."""
    discovered = discover_plugins()
    loaded = get_loaded_plugins()
    plugins = [_plugin_trust_entry(plugin, loaded) for plugin in discovered]
    quarantine_entries = _list_quarantine_entries()
    marketplace = _marketplace_snapshot()
    return jsonify({
        "plugins": plugins,
        "plugins_dir": PLUGINS_DIR,
        "summary": _plugin_trust_summary(plugins, quarantine_entries, marketplace),
        "quarantine": {
            "entries": quarantine_entries,
            "quarantine_dir": _quarantine_root(),
            "total": len(quarantine_entries),
        },
        "marketplace": marketplace,
        "actions": _plugin_trust_actions(),
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
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    target = _plugin_target(plugin_dir, name=name, category="plugin-installation", reversible=True)
    plan = build_destructive_plan(
        "plugins.uninstall",
        targets=[target],
        metadata={
            "route": "/plugins/uninstall",
            "name": name,
            "quarantine_root": _quarantine_root(),
        },
        reversible=True,
    )
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "plan": [target],
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
            "name": name,
            "would_quarantine": True,
        })
    confirm_name = data.get("confirm_name", "")
    if confirm_name != name:
        return jsonify({
            "error": "confirm_name must match the plugin name before uninstall",
            "code": "CONFIRMATION_REQUIRED",
        }), 400
    if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409

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
        "destructive_plan": plan,
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
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    target = _plugin_target(
        metadata["quarantine_path"],
        name=name,
        category="plugin-quarantine",
        reversible=False,
    )
    plan = build_destructive_plan(
        "plugins.quarantine.delete",
        targets=[target],
        metadata={
            "route": "/plugins/quarantine/delete",
            "name": name,
            "quarantine_id": quarantine_id,
            "created_at": metadata.get("created_at"),
            "original_path": metadata.get("original_path", ""),
        },
        reversible=False,
    )
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "plan": [target],
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
            "name": name,
            "quarantine_id": quarantine_id,
            "would_delete": True,
        })
    if data.get("confirm_name", "") != name:
        return jsonify({
            "error": "confirm_name must match the plugin name before permanent delete",
            "code": "CONFIRMATION_REQUIRED",
        }), 400
    if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409
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
        "destructive_plan": plan,
    })
