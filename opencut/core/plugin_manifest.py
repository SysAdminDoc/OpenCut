"""Plugin manifest v1 + sandbox boundary (F116).

Today's plugin loader (:mod:`opencut.core.plugins`) accepts any directory
under ``~/.opencut/plugins/`` with a ``plugin.json`` describing a name +
version + routes. That is fine for in-house experimentation, but a real
plugin marketplace needs three things on top:

1. A **declared capability** list. Plugins must enumerate the
   permissions they want (``http.routes``, ``host.filesystem``,
   ``host.network``, ``models.download``). The host decides whether to
   honour each capability.
2. A **lock file** that records the expected SHA-256 of every file the
   plugin ships. The loader refuses to mount a plugin whose contents
   drift from the lock.
3. An explicit **trust posture**: only the operator can opt in to
   unsigned plugins (``OPENCUT_PLUGIN_ALLOW_UNSIGNED=1``). The default
   refuses to load anything that doesn't have a ``plugin.lock.json``
   sibling.

This module is the schema validator + lock generator. The plugin loader
in ``opencut.core.plugins`` calls ``validate_plugin_manifest()`` before
it registers any blueprint; failure surfaces in the existing
``discover_plugins()`` payload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger("opencut")

MANIFEST_FILENAME = "plugin.json"
LOCK_FILENAME = "plugin.lock.json"
MANIFEST_VERSION = 1

# Capabilities the host knows how to honour. Anything else is rejected so
# new permissions land alongside the runtime that enforces them.
SUPPORTED_CAPABILITIES = (
    "http.routes",         # register Flask routes under /plugins/<name>/
    "host.filesystem",     # may touch the user's filesystem outside ~/.opencut
    "host.network",        # may make outbound HTTP/Network requests
    "models.download",     # may download model weights at runtime
    "ui.panel",            # may inject panel UI assets
)

# Hashes / files we deliberately skip when generating the lock — they
# would defeat the point (or aren't part of the shipped plugin).
_LOCK_IGNORE = frozenset(
    {
        LOCK_FILENAME,
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".gitignore",
        ".DS_Store",
    }
)


@dataclass
class ManifestValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _iter_plugin_files(plugin_dir: Path) -> Iterable[Path]:
    """Yield files that should be hashed for the lock."""
    for root, dirs, files in os.walk(plugin_dir):
        dirs[:] = [d for d in dirs if d not in _LOCK_IGNORE]
        for name in sorted(files):
            if name in _LOCK_IGNORE:
                continue
            if name.endswith(".pyc"):
                continue
            yield Path(root) / name


def _hash_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_plugin_lock(plugin_dir: str | os.PathLike) -> dict:
    """Return the lock payload for ``plugin_dir`` (does not write)."""
    base = Path(plugin_dir)
    if not base.is_dir():
        raise FileNotFoundError(str(base))

    entries: Dict[str, dict] = {}
    for path in _iter_plugin_files(base):
        rel = path.relative_to(base).as_posix()
        entries[rel] = {
            "sha256": _hash_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "version": MANIFEST_VERSION,
        "files": dict(sorted(entries.items())),
    }


def write_plugin_lock(plugin_dir: str | os.PathLike) -> Path:
    """Write ``plugin.lock.json`` next to the manifest. Returns the path."""
    base = Path(plugin_dir)
    payload = compute_plugin_lock(base)
    target = base / LOCK_FILENAME
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _allow_unsigned() -> bool:
    return os.environ.get("OPENCUT_PLUGIN_ALLOW_UNSIGNED", "").strip().lower() in {"1", "true", "yes", "on"}


def validate_manifest_schema(manifest: dict) -> ManifestValidationResult:
    """Validate the v1 manifest schema (without touching the filesystem)."""
    result = ManifestValidationResult(valid=True)

    name = manifest.get("name") or ""
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "-_" for c in name):
        result.errors.append("name: must be a non-empty alphanumeric / dash / underscore string")
    if "version" not in manifest or not isinstance(manifest["version"], str) or not manifest["version"].strip():
        result.errors.append("version: required non-empty string")
    if "description" not in manifest or not isinstance(manifest["description"], str):
        result.errors.append("description: required string")

    api = manifest.get("api_version")
    if api is None:
        result.errors.append("api_version: required (current schema version is 1)")
    elif api != 1:
        result.errors.append(f"api_version: unsupported value {api!r}; expected 1")

    capabilities = manifest.get("capabilities") or []
    if not isinstance(capabilities, list):
        result.errors.append("capabilities: must be a list of strings")
    else:
        unknown = [c for c in capabilities if c not in SUPPORTED_CAPABILITIES]
        if unknown:
            result.errors.append(
                "capabilities: unknown values "
                + ", ".join(repr(u) for u in unknown)
                + f" (supported: {', '.join(SUPPORTED_CAPABILITIES)})"
            )

    if "host.network" in (capabilities or []) and not manifest.get("network_targets"):
        result.warnings.append(
            "host.network capability declared without a network_targets allowlist; "
            "the plugin will be able to contact any host"
        )

    routes = manifest.get("routes") or []
    if not isinstance(routes, list):
        result.errors.append("routes: must be a list")
    else:
        for r in routes:
            if not isinstance(r, dict) or "path" not in r:
                result.errors.append(f"routes: each entry must be an object with a 'path' field, got {r!r}")
                break

    result.valid = not result.errors
    return result


def verify_plugin_lock(plugin_dir: str | os.PathLike) -> ManifestValidationResult:
    """Check the on-disk lock file against the current plugin contents."""
    base = Path(plugin_dir)
    result = ManifestValidationResult(valid=True)
    lock_path = base / LOCK_FILENAME

    if not lock_path.exists():
        if _allow_unsigned():
            result.warnings.append(
                "plugin.lock.json missing but OPENCUT_PLUGIN_ALLOW_UNSIGNED=1 is set; loading anyway"
            )
            return result
        result.valid = False
        result.errors.append(
            "plugin.lock.json missing — generate one with "
            "`opencut.core.plugin_manifest.write_plugin_lock()` or set "
            "OPENCUT_PLUGIN_ALLOW_UNSIGNED=1 to opt in to unsigned plugins"
        )
        return result

    try:
        expected = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.valid = False
        result.errors.append(f"plugin.lock.json unreadable: {exc}")
        return result

    expected_files = expected.get("files") or {}
    if expected.get("version") != MANIFEST_VERSION:
        result.warnings.append(
            f"lock file version {expected.get('version')} differs from plugin loader (expected {MANIFEST_VERSION})"
        )

    current = compute_plugin_lock(base)
    current_files = current["files"]

    missing = sorted(set(expected_files) - set(current_files))
    added = sorted(set(current_files) - set(expected_files))
    mismatched = []
    for rel, info in expected_files.items():
        live = current_files.get(rel)
        if live is None:
            continue
        if live.get("sha256") != info.get("sha256"):
            mismatched.append(rel)

    if missing:
        result.errors.append(f"lock declares missing files: {', '.join(missing[:5])}")
    if added:
        result.errors.append(
            f"plugin ships files absent from lock: {', '.join(added[:5])} (regenerate lock or remove)"
        )
    if mismatched:
        result.errors.append(f"sha-256 mismatch on: {', '.join(mismatched[:5])}")

    result.valid = not result.errors
    return result


def validate_plugin_manifest(plugin_dir: str | os.PathLike) -> ManifestValidationResult:
    """One-stop validation: schema + lock file + capability gating.

    Used by the plugin loader before it registers any blueprints.
    """
    base = Path(plugin_dir)
    result = ManifestValidationResult(valid=True)
    manifest_path = base / MANIFEST_FILENAME
    if not manifest_path.exists():
        result.valid = False
        result.errors.append("plugin.json missing")
        return result

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.valid = False
        result.errors.append(f"plugin.json unreadable: {exc}")
        return result

    schema_result = validate_manifest_schema(manifest)
    result.errors.extend(schema_result.errors)
    result.warnings.extend(schema_result.warnings)

    lock_result = verify_plugin_lock(base)
    result.errors.extend(lock_result.errors)
    result.warnings.extend(lock_result.warnings)

    result.valid = not result.errors
    return result
