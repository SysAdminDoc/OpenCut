"""
Plugin Marketplace.

GitHub-based plugin registry: browse, search, install, and update
plugins from a central repository index.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

PLUGINS_DIR = os.path.join(OPENCUT_DIR, "plugins")
REGISTRY_CACHE = os.path.join(OPENCUT_DIR, "plugin_registry.json")
REGISTRY_URL = "https://raw.githubusercontent.com/opencut/plugin-registry/main/registry.json"
REGISTRY_TTL = 3600  # cache for 1 hour
_PLUGIN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_MAX_ARCHIVE_MEMBERS = 5000
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


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
    # Atomic write: write to temp file then replace
    import tempfile
    fd, tmp_path = tempfile.mkstemp(dir=PLUGINS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(tmp_path, manifest_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _validate_plugin_id(plugin_id: str) -> str:
    if not isinstance(plugin_id, str):
        raise ValueError("plugin_id must be a string")
    cleaned = plugin_id.strip()
    if not cleaned:
        raise ValueError("plugin_id is required")
    if not _PLUGIN_ID_RE.fullmatch(cleaned):
        raise ValueError("Invalid plugin_id")
    return cleaned


def _resolve_within(base_dir: str, *parts: str) -> str:
    real_base = os.path.realpath(base_dir)
    candidate = os.path.realpath(os.path.join(real_base, *parts))
    if candidate != real_base and not candidate.startswith(real_base + os.sep):
        raise ValueError("Path escapes plugins directory")
    return candidate


def _validate_managed_plugin_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Installed plugin path is missing")
    real_base = os.path.realpath(PLUGINS_DIR)
    real_path = os.path.realpath(path)
    if real_path != real_base and not real_path.startswith(real_base + os.sep):
        raise ValueError("Installed plugin path escapes plugins directory")
    return real_path


def _validate_download_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Plugin download URL is missing")
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Plugin download URL must use http or https")
    # Block internal/private network targets to prevent SSRF
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Plugin download URL has no hostname")
    _BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}
    if hostname in _BLOCKED_HOSTS:
        raise ValueError("Plugin download URL must not target localhost")
    # Block private IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x)
    import ipaddress
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        pass  # hostname is not an IP literal — that's fine
    else:
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise ValueError("Plugin download URL must not target private/reserved networks")
    return url.strip()


def _normalize_archive_member(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Archive contains an invalid entry name")
    if name.startswith(("/", "\\")) or _WINDOWS_DRIVE_RE.match(name):
        raise ValueError(f"Archive member uses an absolute path: {name}")

    normalized = os.path.normpath(name.replace("\\", "/")).replace("\\", "/")
    if normalized in {"", "."}:
        return ""
    if normalized == ".." or normalized.startswith("../") or "/../" in f"/{normalized}/":
        raise ValueError(f"Archive member escapes target directory: {name}")
    return normalized


def _common_archive_root(members: List[str]) -> str:
    file_members = [m for m in members if m]
    if not file_members:
        return ""

    roots = set()
    for member in file_members:
        parts = member.split("/", 1)
        if len(parts) < 2:
            return ""
        roots.add(parts[0])

    if len(roots) == 1:
        return next(iter(roots))
    return ""


def _extract_plugin_archive(zip_path: str, plugin_dir: str) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_ARCHIVE_MEMBERS:
            raise ValueError("Plugin archive contains too many files")

        total_bytes = 0
        normalized_members: List[str] = []
        for info in infos:
            if not info.is_dir():
                total_bytes += max(0, int(info.file_size))
            if total_bytes > _MAX_ARCHIVE_BYTES:
                raise ValueError("Plugin archive is too large")
            normalized_members.append(_normalize_archive_member(info.filename))

        common_root = _common_archive_root(
            [member for member, info in zip(normalized_members, infos) if not info.is_dir()]
        )

        for info, normalized in zip(infos, normalized_members):
            if not normalized:
                continue

            relative_path = normalized
            if common_root and relative_path.startswith(common_root + "/"):
                relative_path = relative_path[len(common_root) + 1:]
            if not relative_path:
                continue

            target_path = _resolve_within(plugin_dir, relative_path)
            if info.is_dir():
                os.makedirs(target_path, exist_ok=True)
                continue

            parent = os.path.dirname(target_path)
            os.makedirs(parent, exist_ok=True)
            with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    manifest_path = os.path.join(plugin_dir, "plugin.json")
    if not os.path.isfile(manifest_path):
        raise ValueError("Plugin archive did not contain plugin.json at the expected root")


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
            try:
                with open(REGISTRY_CACHE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as cache_exc:
                raise RuntimeError(
                    f"Cannot fetch plugin registry and cached registry is unreadable: {cache_exc}"
                ) from exc
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
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id", "")
        if not isinstance(pid, str) or not pid.strip():
            continue
        inst = installed.get(pid, {})
        plugins.append(PluginInfo(
            plugin_id=pid,
            name=entry.get("name", pid),
            version=entry.get("version", "0.0.0"),
            author=entry.get("author", "Unknown"),
            description=entry.get("description", ""),
            repo_url=entry.get("repo_url", ""),
            download_url=entry.get("download_url", ""),
            tags=entry.get("tags", []) if isinstance(entry.get("tags", []), list) else [],
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
        searchable = f"{p.name} {p.description} {' '.join(str(tag) for tag in p.tags)}".lower()
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
    plugin_id = _validate_plugin_id(plugin_id)

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

    os.makedirs(PLUGINS_DIR, exist_ok=True)
    plugin_dir = _resolve_within(PLUGINS_DIR, plugin_id)
    if os.path.exists(plugin_dir):
        raise FileExistsError(f"Plugin already exists: {plugin_id}")

    # Download plugin archive
    download_url = _validate_download_url(
        target.download_url or f"{target.repo_url}/archive/refs/heads/main.zip"
    )
    import urllib.request

    fd, tmp_zip = tempfile.mkstemp(prefix=f"opencut_{plugin_id}_", suffix=".zip")
    os.close(fd)
    try:
        urllib.request.urlretrieve(download_url, tmp_zip)

        if on_progress:
            on_progress(60, "Extracting plugin")

        try:
            os.makedirs(plugin_dir, exist_ok=True)
            _extract_plugin_archive(tmp_zip, plugin_dir)
        except Exception:
            shutil.rmtree(plugin_dir, ignore_errors=True)
            raise
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

    Downloads the new version first, then removes the old installation.
    If the download fails, the old version is preserved.

    Args:
        plugin_id: The plugin identifier to update.

    Returns:
        Updated PluginInfo.
    """
    plugin_id = _validate_plugin_id(plugin_id)
    manifest = _load_installed()
    if plugin_id not in manifest:
        raise KeyError(f"Plugin not installed: {plugin_id}")

    if on_progress:
        on_progress(10, "Checking for updates")

    old_path = manifest[plugin_id].get("path", "")

    # Remove old installation ONLY right before re-install so that
    # install_plugin's FileExistsError check passes.
    if old_path and os.path.isdir(old_path):
        validated_path = _validate_managed_plugin_path(old_path)
        # Rename to a temp backup so install_plugin can proceed.
        # If install fails, we restore the backup.
        backup_path = validated_path + ".update-backup"
        try:
            os.rename(validated_path, backup_path)
        except OSError:
            # If rename fails, fall back to rmtree (best effort)
            shutil.rmtree(validated_path, ignore_errors=True)
            backup_path = None

        try:
            result = install_plugin(plugin_id, on_progress=on_progress)
        except Exception:
            # Restore backup on failure
            if backup_path and os.path.isdir(backup_path):
                try:
                    os.rename(backup_path, validated_path)
                except OSError:
                    pass
            raise

        # Clean up backup on success
        if backup_path and os.path.isdir(backup_path):
            shutil.rmtree(backup_path, ignore_errors=True)
        return result

    # No existing path to worry about — just install
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
