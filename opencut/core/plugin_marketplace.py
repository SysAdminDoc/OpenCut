"""
Plugin Marketplace.

GitHub-based plugin registry: browse, search, install, and update
plugins from a central repository index.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

PLUGINS_DIR = os.path.join(OPENCUT_DIR, "plugins")
REGISTRY_CACHE = os.path.join(OPENCUT_DIR, "plugin_registry.json")
REGISTRY_URL = "https://raw.githubusercontent.com/opencut/plugin-registry/main/registry.json"
REGISTRY_TTL = 3600  # cache for 1 hour


@dataclass
class PluginInfo:
    """Metadata for a marketplace plugin."""
    plugin_id: str
    name: str
    version: str
    author: str
    description: str
    repo_url: str
    download_url: str = ""
    tags: List[str] = field(default_factory=list)
    min_opencut_version: str = ""
    installed: bool = False
    installed_version: str = ""
    updated_at: str = ""


def _load_installed() -> Dict[str, dict]:
    """Load the installed plugins manifest."""
    manifest_path = os.path.join(PLUGINS_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_installed(manifest: Dict[str, dict]) -> None:
    os.makedirs(PLUGINS_DIR, exist_ok=True)
    manifest_path = os.path.join(PLUGINS_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _registry_cache_valid() -> bool:
    if not os.path.exists(REGISTRY_CACHE):
        return False
    age = time.time() - os.path.getmtime(REGISTRY_CACHE)
    return age < REGISTRY_TTL


def fetch_plugin_registry(
    force: bool = False,
    on_progress: Optional[Callable] = None,
) -> List[PluginInfo]:
    """Fetch the plugin registry from GitHub.

    Args:
        force: Bypass cache and re-fetch.

    Returns:
        List of PluginInfo for all available plugins.
    """
    if on_progress:
        on_progress(10, "Checking registry cache")

    if not force and _registry_cache_valid():
        try:
            with open(REGISTRY_CACHE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return _parse_registry(data)
        except (json.JSONDecodeError, OSError):
            pass

    if on_progress:
        on_progress(30, "Fetching plugin registry")

    import urllib.request
    try:
        with urllib.request.urlopen(REGISTRY_URL, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Failed to fetch plugin registry: %s", exc)
        # Fall back to cached if available
        if os.path.exists(REGISTRY_CACHE):
            with open(REGISTRY_CACHE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            raise RuntimeError(f"Cannot fetch plugin registry: {exc}")

    os.makedirs(os.path.dirname(REGISTRY_CACHE), exist_ok=True)
    with open(REGISTRY_CACHE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    if on_progress:
        on_progress(100, "Registry fetched")

    return _parse_registry(data)


def _parse_registry(data: dict) -> List[PluginInfo]:
    installed = _load_installed()
    plugins = []
    for entry in data.get("plugins", []):
        pid = entry.get("id", "")
        inst = installed.get(pid, {})
        plugins.append(PluginInfo(
            plugin_id=pid,
            name=entry.get("name", pid),
            version=entry.get("version", "0.0.0"),
            author=entry.get("author", "Unknown"),
            description=entry.get("description", ""),
            repo_url=entry.get("repo_url", ""),
            download_url=entry.get("download_url", ""),
            tags=entry.get("tags", []),
            min_opencut_version=entry.get("min_opencut_version", ""),
            installed=pid in installed,
            installed_version=inst.get("version", ""),
            updated_at=entry.get("updated_at", ""),
        ))
    return plugins


def search_plugins(
    query: str,
    on_progress: Optional[Callable] = None,
) -> List[PluginInfo]:
    """Search the plugin registry by keyword.

    Args:
        query: Search term to match against name, description, and tags.

    Returns:
        Matching PluginInfo list.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    plugins = fetch_plugin_registry(on_progress=on_progress)
    q = query.lower().strip()
    results = []
    for p in plugins:
        searchable = f"{p.name} {p.description} {' '.join(p.tags)}".lower()
        if q in searchable:
            results.append(p)
    return results


def install_plugin(
    plugin_id: str,
    on_progress: Optional[Callable] = None,
) -> PluginInfo:
    """Install a plugin from the marketplace.

    Args:
        plugin_id: The plugin identifier to install.

    Returns:
        PluginInfo for the installed plugin.
    """
    if on_progress:
        on_progress(10, "Fetching registry")

    plugins = fetch_plugin_registry()
    target = None
    for p in plugins:
        if p.plugin_id == plugin_id:
            target = p
            break

    if target is None:
        raise KeyError(f"Plugin not found: {plugin_id}")

    if on_progress:
        on_progress(30, f"Downloading {target.name}")

    plugin_dir = os.path.join(PLUGINS_DIR, plugin_id)
    os.makedirs(plugin_dir, exist_ok=True)

    # Download plugin archive
    download_url = target.download_url or f"{target.repo_url}/archive/refs/heads/main.zip"
    import tempfile
    import urllib.request
    import zipfile

    tmp_zip = os.path.join(tempfile.gettempdir(), f"{plugin_id}.zip")
    try:
        urllib.request.urlretrieve(download_url, tmp_zip)

        if on_progress:
            on_progress(60, "Extracting plugin")

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(plugin_dir)
    finally:
        if os.path.exists(tmp_zip):
            os.unlink(tmp_zip)

    if on_progress:
        on_progress(80, "Registering plugin")

    manifest = _load_installed()
    manifest[plugin_id] = {
        "version": target.version,
        "name": target.name,
        "installed_at": time.time(),
        "path": plugin_dir,
    }
    _save_installed(manifest)

    target.installed = True
    target.installed_version = target.version

    if on_progress:
        on_progress(100, f"Installed {target.name}")

    logger.info("Installed plugin %s v%s", plugin_id, target.version)
    return target


def update_plugin(
    plugin_id: str,
    on_progress: Optional[Callable] = None,
) -> PluginInfo:
    """Update an installed plugin to the latest version.

    Args:
        plugin_id: The plugin identifier to update.

    Returns:
        Updated PluginInfo.
    """
    manifest = _load_installed()
    if plugin_id not in manifest:
        raise KeyError(f"Plugin not installed: {plugin_id}")

    if on_progress:
        on_progress(10, "Checking for updates")

    # Remove old installation
    old_path = manifest[plugin_id].get("path", "")
    if old_path and os.path.isdir(old_path):
        shutil.rmtree(old_path, ignore_errors=True)

    # Re-install latest
    return install_plugin(plugin_id, on_progress=on_progress)


def list_installed_plugins(
    on_progress: Optional[Callable] = None,
) -> List[PluginInfo]:
    """List all installed plugins.

    Returns:
        List of PluginInfo for installed plugins.
    """
    manifest = _load_installed()
    results = []
    for pid, info in manifest.items():
        results.append(PluginInfo(
            plugin_id=pid,
            name=info.get("name", pid),
            version=info.get("version", "0.0.0"),
            author="",
            description="",
            repo_url="",
            installed=True,
            installed_version=info.get("version", "0.0.0"),
        ))
    return results
