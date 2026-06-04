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
import re
import sys
import threading

logger = logging.getLogger("opencut")

PLUGINS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "plugins")

# Required fields in plugin.json
_REQUIRED_MANIFEST_FIELDS = {"name", "version", "description"}

# Loaded plugins registry
_loaded_plugins = {}
_plugin_contexts = {}
_plugins_lock = threading.Lock()
_PLUGIN_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_PLUGIN_JOB_PATH_KEYS = {
    "file",
    "filepath",
    "input_file",
    "input_path",
    "output_dir",
    "output_file",
    "output_path",
    "path",
    "source_path",
}


def _install_plugin_csrf_guard(blueprint):
    """Ensure hosted plugin mutations use the same CSRF guard as core routes."""
    if getattr(blueprint, "_opencut_csrf_guarded", False):
        return

    from opencut.security import validate_csrf_request

    @blueprint.before_request
    def _opencut_plugin_csrf_guard():
        return validate_csrf_request()

    blueprint._opencut_csrf_guarded = True


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

    jobs = manifest.get("jobs", [])
    if jobs and not isinstance(jobs, list):
        return False, "Plugin jobs must be a list"
    if isinstance(jobs, list):
        seen = set()
        for job in jobs:
            if not isinstance(job, dict):
                return False, "Plugin jobs must be objects"
            job_id = job.get("id", "")
            if not isinstance(job_id, str) or not _PLUGIN_JOB_ID_RE.fullmatch(job_id):
                return False, f"Plugin job id is invalid: {job_id!r}"
            if job_id in seen:
                return False, f"Plugin job id is duplicated: {job_id}"
            seen.add(job_id)

    return True, ""


def _plugin_data_dir(plugin_name: str, plugin_dir: str) -> str:
    return os.path.join(plugin_dir, "data")


def _get_plugin_context(plugin_name: str) -> dict:
    with _plugins_lock:
        return dict(_plugin_contexts.get(plugin_name, {}))


def _payload_path_values(value, *, parent_key: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            yield from _payload_path_values(child, parent_key=key_str)
    elif isinstance(value, list):
        for child in value:
            yield from _payload_path_values(child, parent_key=parent_key)
    elif isinstance(value, str) and value.strip():
        key = parent_key.lower()
        if (
            key in _PLUGIN_JOB_PATH_KEYS
            or key.endswith("_path")
            or key.endswith("_file")
            or key.endswith("_dir")
        ):
            yield parent_key, value


def _validate_plugin_job_payload(plugin_name: str, data: dict) -> str:
    context = _get_plugin_context(plugin_name)
    capabilities = set(context.get("capabilities") or [])
    if "jobs.register" not in capabilities:
        return "Plugin does not declare the jobs.register capability."
    if "host.filesystem" in capabilities:
        return ""

    data_dir = context.get("data_dir", "")
    if not data_dir:
        return "Plugin data directory is unavailable."
    os.makedirs(data_dir, exist_ok=True)

    from opencut.security import validate_path

    for key, value in _payload_path_values(data):
        try:
            validate_path(value, allowed_base=data_dir)
        except ValueError:
            return (
                f"Plugin job path field '{key}' is outside the plugin data directory; "
                "declare host.filesystem to access host files."
            )
    return ""


def plugin_job(plugin_name: str, job_id: str, *,
               label: str = "",
               description: str = "",
               filepath_required: bool = False,
               filepath_param: str = "filepath",
               pre_validate=None,
               disk_operation=None,
               disk_required_mb=None,
               resumable: bool = False):
    """Decorator for plugin routes that enqueue real OpenCut async jobs.

    Use below a plugin Blueprint route::

        @plugin_bp.route("/start", methods=["POST"])
        @plugin_job("my-plugin", "long_task", filepath_required=False)
        def start(job_id, filepath, data):
            ...

    The plugin manifest must declare the same job id under ``jobs`` and include
    ``jobs.register`` in ``capabilities``.
    """
    if not _PLUGIN_JOB_ID_RE.fullmatch(str(plugin_name or "")):
        raise ValueError(f"Invalid plugin name for job registration: {plugin_name!r}")
    if not _PLUGIN_JOB_ID_RE.fullmatch(str(job_id or "")):
        raise ValueError(f"Invalid plugin job id: {job_id!r}")

    from opencut.jobs import async_job

    def _combined_pre_validate(data):
        err = _validate_plugin_job_payload(plugin_name, data)
        if err:
            return err
        if pre_validate is None:
            return None
        return pre_validate(data)

    def decorator(func):
        wrapped = async_job(
            f"plugin:{plugin_name}:{job_id}",
            filepath_required=filepath_required,
            filepath_param=filepath_param,
            pre_validate=_combined_pre_validate,
            disk_operation=disk_operation,
            disk_required_mb=disk_required_mb,
            resumable=resumable,
        )(func)
        wrapped._opencut_plugin_job = {
            "plugin": plugin_name,
            "id": job_id,
            "type": f"plugin:{plugin_name}:{job_id}",
            "label": label or job_id.replace("_", " ").replace("-", " ").title(),
            "description": description,
            "resumable": bool(resumable),
            "filepath_required": bool(filepath_required),
            "filepath_param": filepath_param,
        }
        return wrapped

    return decorator


def _collect_module_plugin_jobs(module, plugin_name: str) -> list[dict]:
    jobs = []
    seen = set()
    for value in vars(module).values():
        meta = getattr(value, "_opencut_plugin_job", None)
        if not isinstance(meta, dict):
            continue
        if meta.get("plugin") != plugin_name:
            continue
        job_id = meta.get("id", "")
        if job_id in seen:
            continue
        seen.add(job_id)
        jobs.append(dict(meta))
    return sorted(jobs, key=lambda item: item.get("id", ""))


def _validate_module_plugin_jobs(plugin_name: str, plugin_info: dict, module_jobs: list[dict]) -> str:
    if not module_jobs:
        return ""
    capabilities = set(plugin_info.get("capabilities") or [])
    if "jobs.register" not in capabilities:
        return "Plugin uses plugin_job but does not declare jobs.register"
    declared = {
        job.get("id", "")
        for job in plugin_info.get("jobs", [])
        if isinstance(job, dict)
    }
    missing = [job["id"] for job in module_jobs if job.get("id") not in declared]
    if missing:
        return "Plugin job(s) not declared in plugin.json: " + ", ".join(missing)
    return ""


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
                "jobs": [],
                "capabilities": [],
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
                "jobs": [],
                "capabilities": [],
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
            "jobs": manifest.get("jobs", []),
            "capabilities": manifest.get("capabilities", []),
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
        existing = _loaded_plugins.get(plugin_name)
        if existing and existing.get("app_id") == id(app):
            logger.warning(
                "Plugin name collision: '%s' already loaded — refusing to register",
                plugin_name,
            )
            return False

    _path_inserted = False
    try:
        with _plugins_lock:
            _plugin_contexts[plugin_name] = {
                "capabilities": list(plugin_info.get("capabilities") or []),
                "data_dir": _plugin_data_dir(plugin_name, plugin_dir),
                "plugin_dir": plugin_dir,
            }
        # Add plugin dir to sys.path for the duration of exec_module only.
        # Remove it immediately after so two plugins with identically-named
        # helper modules cannot shadow each other.
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
            _path_inserted = True

        spec = importlib.util.spec_from_file_location(
            f"opencut_plugin_{plugin_name}", routes_file
        )
        if spec is None or spec.loader is None:
            logger.warning("Cannot load plugin module: %s", plugin_name)
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    except Exception as e:
        logger.error("Failed to load plugin %s: %s", plugin_name, e)
        with _plugins_lock:
            _plugin_contexts.pop(plugin_name, None)
        return False
    finally:
        if _path_inserted and plugin_dir in sys.path:
            sys.path.remove(plugin_dir)

    try:
        # Look for plugin_bp Blueprint
        bp = getattr(module, "plugin_bp", None)
        if bp is None:
            logger.warning("Plugin %s routes.py has no 'plugin_bp' Blueprint", plugin_name)
            with _plugins_lock:
                _plugin_contexts.pop(plugin_name, None)
            return False

        module_jobs = _collect_module_plugin_jobs(module, plugin_name)
        job_error = _validate_module_plugin_jobs(plugin_name, plugin_info, module_jobs)
        if job_error:
            logger.warning("Plugin %s job registration refused: %s", plugin_name, job_error)
            with _plugins_lock:
                _plugin_contexts.pop(plugin_name, None)
            return False

        _install_plugin_csrf_guard(bp)

        # Register under /plugins/<name>/ prefix
        app.register_blueprint(bp, url_prefix=f"/plugins/{plugin_name}")

        with _plugins_lock:
            _loaded_plugins[plugin_name] = {
                "info": plugin_info,
                "module": module,
                "blueprint": bp,
                "jobs": module_jobs,
                "app_id": id(app),
            }

        logger.info("Loaded plugin: %s v%s", plugin_name, plugin_info["version"])
        return True

    except Exception as e:
        logger.error("Failed to register plugin %s: %s", plugin_name, e)
        with _plugins_lock:
            _plugin_contexts.pop(plugin_name, None)
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

    # F116: refuse to load plugins whose schema or lock file is invalid.
    # The validator is intentionally separate so dev installs can opt in
    # via OPENCUT_PLUGIN_ALLOW_UNSIGNED=1.
    try:
        from opencut.core.plugin_manifest import validate_plugin_manifest
    except Exception:  # pragma: no cover - defensive
        validate_plugin_manifest = None  # type: ignore[assignment]

    for plugin in plugins:
        if not plugin["valid"] or not plugin["enabled"]:
            result["skipped"].append(plugin["name"])
            continue

        if validate_plugin_manifest is not None:
            validation = validate_plugin_manifest(plugin["path"])
            if not validation.valid:
                logger.warning(
                    "plugin %s refused by manifest validator: %s",
                    plugin["name"],
                    "; ".join(validation.errors),
                )
                result["failed"].append({"name": plugin["name"], "error": "; ".join(validation.errors)})
                continue
            for warn in validation.warnings:
                logger.info("plugin %s warning: %s", plugin["name"], warn)

        routes_file = os.path.join(plugin["path"], "routes.py")
        if not os.path.isfile(routes_file):
            # Plugin has no routes — might be UI-only or config-only
            with _plugins_lock:
                _loaded_plugins[plugin["name"]] = {
                    "info": plugin,
                    "module": None,
                    "blueprint": None,
                    "jobs": [],
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
                "jobs": list(data.get("jobs") or []),
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
            _plugin_contexts.pop(name, None)
            return True
    return False
