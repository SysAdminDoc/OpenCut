"""Plugin manifest v1 + sandbox boundary (F116).

Today's plugin loader (:mod:`opencut.core.plugins`) accepts any directory
under ``~/.opencut/plugins/`` with a ``plugin.json`` describing a name +
version + routes. That is fine for in-house experimentation, but a real
plugin marketplace needs three things on top:

1. A **declared capability** list. Plugins must enumerate the
   permissions they want (``http.routes``, ``jobs.register``,
   ``host.filesystem``, ``host.network``, ``models.download``). The host
   decides whether to honour each capability.
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
SIGNATURE_FILENAME = "plugin.signature.json"
MANIFEST_VERSION = 1

# --- Compatibility contract -------------------------------------------------
# `api_version == 1` used to be a hard equality check, which meant the first
# host API bump would reject every installed plugin with "unsupported value 1"
# and no way to say "I work on 1 and 2". Plugins now declare a *range* and the
# host declares what it implements, so an incompatibility is reported before
# activation with something the author can act on.

#: The plugin API generation this host implements.
PLUGIN_API_VERSION = 1

#: Oldest plugin API generation this host still honours.
MIN_SUPPORTED_PLUGIN_API = 1

#: Manifest *schema* version — how the manifest file itself is written. It
#: moves independently of the API version: a manifest can gain fields without
#: the runtime contract changing.
MANIFEST_SCHEMA_VERSION = 1
MIN_SUPPORTED_MANIFEST_SCHEMA = 1

# Capabilities the host knows how to honour. Anything else is rejected so
# new permissions land alongside the runtime that enforces them.
SUPPORTED_CAPABILITIES = (
    "http.routes",         # register Flask routes under /plugins/<name>/
    "jobs.register",       # register async background jobs through plugin_job
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
        SIGNATURE_FILENAME,
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".gitignore",
        ".DS_Store",
    }
)


@dataclass
class CompatibilityReport:
    """Whether a plugin's declared API range overlaps this host's."""

    compatible: bool
    host_api_version: int = PLUGIN_API_VERSION
    host_min_api_version: int = MIN_SUPPORTED_PLUGIN_API
    plugin_min_api_version: int = PLUGIN_API_VERSION
    plugin_max_api_version: int = PLUGIN_API_VERSION
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    reason: str = ""
    remediation: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def host_api_range() -> tuple[int, int]:
    """``(min, max)`` plugin API generations this host honours."""
    return (MIN_SUPPORTED_PLUGIN_API, PLUGIN_API_VERSION)


def _declared_api_range(manifest: dict) -> tuple[int, int] | None:
    """Resolve the plugin's supported host-API range.

    A v1 manifest only has ``api_version``; that is read as the single-point
    range ``[api_version, api_version]`` so existing plugins keep their exact
    semantics without being rewritten.
    """
    api = manifest.get("api_version")
    if not isinstance(api, int) or isinstance(api, bool):
        return None
    low = manifest.get("min_api_version", api)
    high = manifest.get("max_api_version", api)
    if not isinstance(low, int) or isinstance(low, bool):
        return None
    if not isinstance(high, int) or isinstance(high, bool):
        return None
    if low > high:
        return None
    return (low, high)


def check_api_compatibility(manifest: dict) -> CompatibilityReport:
    """Report whether *manifest* can run on this host, and what to do if not."""
    host_min, host_max = host_api_range()

    schema_version = manifest.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        return CompatibilityReport(
            compatible=False,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            reason=f"schema_version must be an integer, got {schema_version!r}",
            remediation=(
                f"Set \"schema_version\": {MANIFEST_SCHEMA_VERSION} in plugin.json, "
                "or omit it to accept the default."
            ),
        )
    if schema_version > MANIFEST_SCHEMA_VERSION:
        return CompatibilityReport(
            compatible=False,
            manifest_schema_version=schema_version,
            reason=(
                f"manifest schema {schema_version} is newer than this host "
                f"understands ({MANIFEST_SCHEMA_VERSION})"
            ),
            remediation=(
                "Upgrade OpenCut, or install a build of the plugin that targets "
                f"manifest schema {MANIFEST_SCHEMA_VERSION}."
            ),
        )
    if schema_version < MIN_SUPPORTED_MANIFEST_SCHEMA:
        return CompatibilityReport(
            compatible=False,
            manifest_schema_version=schema_version,
            reason=f"manifest schema {schema_version} is no longer supported",
            remediation=(
                f"Regenerate plugin.json against schema {MANIFEST_SCHEMA_VERSION}."
            ),
        )

    declared = _declared_api_range(manifest)
    if declared is None:
        return CompatibilityReport(
            compatible=False,
            manifest_schema_version=schema_version,
            reason=(
                "api_version / min_api_version / max_api_version must be integers "
                "with min <= max"
            ),
            remediation=(
                f"Declare \"api_version\": {PLUGIN_API_VERSION}, and optionally "
                "\"min_api_version\"/\"max_api_version\" to span several host "
                "generations."
            ),
        )

    plugin_min, plugin_max = declared
    report = CompatibilityReport(
        compatible=True,
        plugin_min_api_version=plugin_min,
        plugin_max_api_version=plugin_max,
        manifest_schema_version=schema_version,
    )
    if plugin_max < host_min:
        report.compatible = False
        report.reason = (
            f"plugin targets OpenCut plugin API {plugin_min}-{plugin_max}; this "
            f"host no longer supports anything below {host_min}"
        )
        report.remediation = (
            f"Update the plugin to support API {host_min}-{host_max}, or install "
            "an OpenCut release that still honours the older API."
        )
    elif plugin_min > host_max:
        report.compatible = False
        report.reason = (
            f"plugin requires OpenCut plugin API {plugin_min}; this host "
            f"implements up to {host_max}"
        )
        report.remediation = (
            "Upgrade OpenCut, or install a build of the plugin that targets API "
            f"{host_max}."
        )
    return report


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
    if not isinstance(manifest, dict):
        result.valid = False
        result.errors.append("manifest: must be a JSON object")
        return result

    name = manifest.get("name") or ""
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "-_" for c in name):
        result.errors.append("name: must be a non-empty alphanumeric / dash / underscore string")
    if "version" not in manifest or not isinstance(manifest["version"], str) or not manifest["version"].strip():
        result.errors.append("version: required non-empty string")
    if "description" not in manifest or not isinstance(manifest["description"], str):
        result.errors.append("description: required string")

    if manifest.get("api_version") is None:
        result.errors.append(
            f"api_version: required (this host implements plugin API "
            f"{PLUGIN_API_VERSION})"
        )
    else:
        compatibility = check_api_compatibility(manifest)
        if not compatibility.compatible:
            # Carry the remediation into the error so the loader's refusal
            # message tells the operator what to do about it.
            result.errors.append(
                f"api_version: {compatibility.reason}. {compatibility.remediation}"
            )

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

    jobs = manifest.get("jobs") or []
    if not isinstance(jobs, list):
        result.errors.append("jobs: must be a list")
    else:
        if jobs and "jobs.register" not in capabilities:
            result.errors.append("jobs: declaring jobs requires the jobs.register capability")
        seen_job_ids = set()
        for job in jobs:
            if not isinstance(job, dict):
                result.errors.append(f"jobs: each entry must be an object, got {job!r}")
                break
            job_id = job.get("id")
            if not isinstance(job_id, str) or not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
                result.errors.append(f"jobs: invalid id {job_id!r}; use alphanumeric / dash / underscore")
                break
            if job_id in seen_job_ids:
                result.errors.append(f"jobs: duplicate id {job_id!r}")
                break
            seen_job_ids.add(job_id)
            if "label" in job and not isinstance(job["label"], str):
                result.errors.append(f"jobs.{job_id}.label: must be a string")
                break
            if "description" in job and not isinstance(job["description"], str):
                result.errors.append(f"jobs.{job_id}.description: must be a string")
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
    if not isinstance(expected, dict):
        result.valid = False
        result.errors.append("plugin.lock.json must be a JSON object")
        return result

    expected_files = expected.get("files") or {}
    if not isinstance(expected_files, dict):
        result.valid = False
        result.errors.append("plugin.lock.json files must be an object")
        return result
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


def doctor(plugins_dir: str | os.PathLike | None = None) -> dict:
    """Report the health of every installed plugin.

    Returns a machine-readable summary so both the CLI and the API can render
    the same verdict: what is installed, what is compatible, and for anything
    that is not, what the author or operator should do about it.
    """
    if plugins_dir is None:
        plugins_dir = Path(os.path.expanduser("~")) / ".opencut" / "plugins"
    base = Path(plugins_dir)

    host_min, host_max = host_api_range()
    summary = {
        "plugins_dir": str(base),
        "host_api_version": PLUGIN_API_VERSION,
        "host_min_api_version": host_min,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "total": 0,
        "healthy": 0,
        "incompatible": 0,
        "invalid": 0,
        "plugins": [],
    }
    if not base.is_dir():
        return summary

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / MANIFEST_FILENAME
        if not manifest_path.is_file():
            continue

        summary["total"] += 1
        record = {
            "name": entry.name,
            "path": str(entry),
            "version": "",
            "compatible": False,
            "valid": False,
            "errors": [],
            "warnings": [],
            "reason": "",
            "remediation": "",
        }
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["errors"].append(f"plugin.json unreadable: {exc}")
            record["remediation"] = "Repair or reinstall the plugin."
            summary["invalid"] += 1
            summary["plugins"].append(record)
            continue

        record["version"] = str(manifest.get("version") or "")
        compatibility = check_api_compatibility(manifest)
        record["compatible"] = compatibility.compatible
        record["reason"] = compatibility.reason
        record["remediation"] = compatibility.remediation
        record["api_range"] = [
            compatibility.plugin_min_api_version,
            compatibility.plugin_max_api_version,
        ]

        validation = validate_plugin_manifest(entry)
        record["valid"] = validation.valid
        record["errors"].extend(validation.errors)
        record["warnings"].extend(validation.warnings)

        if not compatibility.compatible:
            summary["incompatible"] += 1
        elif not validation.valid:
            summary["invalid"] += 1
        else:
            summary["healthy"] += 1
        summary["plugins"].append(record)

    return summary
