"""OpenTimelineIO adapter discovery and schema compatibility preflight.

The native OTIO JSON adapter can deliberately serialize against an older
``OTIO_CORE`` schema map.  This module keeps that versioning concern separate
from the timeline builders and makes every write pass through the same
capability and loss check.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from functools import lru_cache
from typing import Any


class OTIOPreflightError(ValueError):
    """Raised when an OTIO write is unavailable or would be lossy."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _otio():
    try:
        import opentimelineio as otio
    except ImportError as exc:
        raise ImportError(
            "OpenTimelineIO not installed. Install with: pip install opentimelineio"
        ) from exc
    return otio


@lru_cache(maxsize=1)
def _package_distributions() -> dict[str, list[str]]:
    return importlib.metadata.packages_distributions()


def _distribution_for_module(module: Any) -> tuple[str, str]:
    """Return the installed distribution and version behind an adapter."""
    module_name = str(getattr(module, "__name__", "")).split(".", 1)[0]
    candidates = _package_distributions().get(module_name, [])
    if module_name == "opentimelineio" and "OpenTimelineIO" not in candidates:
        candidates = ["OpenTimelineIO", *candidates]
    for distribution in candidates:
        try:
            return distribution, importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return module_name or "unknown", "unknown"


def _adapter_details(adapter: Any) -> dict[str, Any]:
    try:
        module = adapter.module()
        package, version = _distribution_for_module(module)
        load_error = None
    except Exception as exc:  # adapter plugins may have missing optional deps
        package, version = "unknown", "unknown"
        load_error = f"{type(exc).__name__}: {exc}"

    can_read = bool(
        adapter.has_feature("read_from_file")
        or adapter.has_feature("read_from_string")
    )
    can_write = bool(
        adapter.has_feature("write_to_file")
        or adapter.has_feature("write_to_string")
    )
    return {
        "name": str(adapter.name),
        "suffixes": [str(s).lstrip(".").lower() for s in adapter.suffixes],
        "read": can_read,
        "write": can_write,
        "schema_targeting": str(adapter.name) == "otio_json",
        "package": package,
        "version": version,
        "load_error": load_error,
    }


def get_otio_capabilities() -> dict[str, Any]:
    """Discover installed adapters and built-in OTIO schema target maps."""
    otio = _otio()
    adapters = sorted(
        (_adapter_details(adapter) for adapter in otio.plugins.ActiveManifest().adapters),
        key=lambda item: item["name"],
    )
    version_maps = otio.versioning.full_map()
    core_maps = dict(version_maps.get("OTIO_CORE", {}))
    schema_targets = [{
        "id": "current",
        "label": f"Current ({otio.__version__})",
        "family": "OTIO_CORE",
        "version": str(otio.__version__),
        "legacy": False,
    }]
    for version in sorted(core_maps, key=_version_key):
        if version == str(otio.__version__):
            continue
        schema_targets.append({
            "id": f"OTIO_CORE:{version}",
            "label": f"OTIO Core {version}",
            "family": "OTIO_CORE",
            "version": version,
            "legacy": True,
        })
    return {
        "runtime_version": str(otio.__version__),
        "adapters": adapters,
        "schema_targets": schema_targets,
    }


def _version_key(version: str) -> tuple[int, ...]:
    result = []
    for part in str(version).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        result.append(int(digits or 0))
    return tuple(result)


def _resolve_target(otio: Any, requested: str | None) -> tuple[str, dict[str, int]]:
    target = str(requested or "current").strip()
    if target.lower() == "current":
        # An explicit empty map prevents OTIO_DEFAULT_TARGET_VERSION_FAMILY_LABEL
        # from silently changing a caller's requested current-schema export.
        return "current", {}
    if ":" in target:
        family, label = target.split(":", 1)
    else:
        family, label = "OTIO_CORE", target
    if family != "OTIO_CORE":
        raise ValueError("Only built-in OTIO_CORE compatibility targets are supported")
    try:
        return f"OTIO_CORE:{label}", dict(otio.versioning.fetch_map(family, label))
    except (KeyError, ValueError) as exc:
        version_maps = otio.versioning.full_map()
        available = ", ".join(sorted(version_maps.get("OTIO_CORE", {})))
        raise ValueError(
            f"Unknown OTIO schema target '{target}'. Available legacy targets: {available}"
        ) from exc


def _find_adapter(otio: Any, adapter_name: str) -> Any | None:
    for adapter in otio.plugins.ActiveManifest().adapters:
        if str(adapter.name) == adapter_name:
            return adapter
    return None


def adapter_output_suffix(adapter_name: str = "otio_json") -> str:
    """Return the preferred output suffix for a discovered writable adapter."""
    otio = _otio()
    adapter = _find_adapter(otio, adapter_name)
    if adapter is None:
        raise ValueError(f"OTIO adapter '{adapter_name}' is not installed")
    details = _adapter_details(adapter)
    if not details["write"]:
        raise ValueError(f"OTIO adapter '{adapter_name}' does not support writing")
    return details["suffixes"][0] if details["suffixes"] else "otio"


def _loss_paths(before: Any, after: Any, path: str = "$") -> list[str]:
    """Find values that do not survive target serialization and re-upgrade."""
    losses: list[str] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key, value in before.items():
            if key == "OTIO_SCHEMA":
                continue
            child_path = f"{path}.{key}"
            if key not in after:
                losses.append(child_path)
            else:
                losses.extend(_loss_paths(value, after[key], child_path))
        return losses
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            losses.append(f"{path}.length")
        for index, value in enumerate(before[: len(after)]):
            losses.extend(_loss_paths(value, after[index], f"{path}[{index}]"))
        return losses
    if before != after:
        losses.append(path)
    return losses


def _roundtrip_loss_paths(
    otio: Any,
    timeline: Any,
    target_versions: dict[str, int],
) -> list[str]:
    current_text = otio.core.serialize_json_to_string(
        timeline, schema_version_targets={}, indent=-1
    )
    target_text = otio.core.serialize_json_to_string(
        timeline, schema_version_targets=target_versions, indent=-1
    )
    recovered = otio.core.deserialize_json_from_string(target_text)
    recovered_text = otio.core.serialize_json_to_string(
        recovered, schema_version_targets={}, indent=-1
    )
    return sorted(set(_loss_paths(json.loads(current_text), json.loads(recovered_text))))


def preflight_otio_timeline(
    timeline: Any,
    *,
    adapter_name: str = "otio_json",
    schema_target: str = "current",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Report adapter availability and downgrade loss without writing a file."""
    otio = _otio()
    errors: list[str] = []
    warnings: list[str] = []
    adapter = _find_adapter(otio, adapter_name)
    if adapter is None:
        return {
            "ready": False,
            "adapter": {"name": adapter_name, "installed": False},
            "runtime_version": str(otio.__version__),
            "schema_target": schema_target,
            "schema_versions": {},
            "lossy": False,
            "lossy_fields": [],
            "warnings": [],
            "errors": [f"OTIO adapter '{adapter_name}' is not installed"],
        }

    details = _adapter_details(adapter)
    details["installed"] = True
    if not details["write"]:
        errors.append(f"OTIO adapter '{adapter_name}' does not support writing")
    try:
        resolved_target, target_versions = _resolve_target(otio, schema_target)
    except ValueError as exc:
        resolved_target, target_versions = str(schema_target), {}
        errors.append(str(exc))

    if resolved_target != "current" and not details["schema_targeting"]:
        errors.append(
            f"OTIO adapter '{adapter_name}' cannot target native OTIO schema versions"
        )

    suffix = os.path.splitext(output_path or "")[1].lstrip(".").lower()
    if suffix and details["suffixes"] and suffix not in details["suffixes"]:
        errors.append(
            f"OTIO adapter '{adapter_name}' writes {details['suffixes']}; got '.{suffix}'"
        )

    loss_fields: list[str] = []
    if not errors and details["schema_targeting"]:
        try:
            loss_fields = _roundtrip_loss_paths(otio, timeline, target_versions)
        except Exception as exc:
            errors.append(f"Schema compatibility simulation failed: {type(exc).__name__}: {exc}")
    if adapter_name != "otio_json":
        warnings.append(
            "Non-native adapter translation may omit format-specific data; inspect the destination application."
        )

    return {
        "ready": not errors,
        "adapter": details,
        "runtime_version": str(otio.__version__),
        "schema_target": resolved_target,
        "schema_versions": target_versions,
        "lossy": bool(loss_fields),
        "lossy_fields": loss_fields,
        "warnings": warnings,
        "errors": errors,
    }


def write_otio_timeline(
    timeline: Any,
    output_path: str,
    *,
    adapter_name: str = "otio_json",
    schema_target: str = "current",
    accept_lossy: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Preflight and optionally write an OTIO timeline with provenance."""
    otio = _otio()
    adapter = _find_adapter(otio, adapter_name)
    adapter_details = _adapter_details(adapter) if adapter is not None else {
        "name": adapter_name,
        "package": "unknown",
        "version": "unknown",
    }
    timeline.metadata["opencut_export"] = {
        "adapter": adapter_name,
        "adapter_package": adapter_details.get("package", "unknown"),
        "adapter_version": adapter_details.get("version", "unknown"),
        "otio_runtime_version": str(otio.__version__),
        "schema_target": schema_target,
        "schema_versions": {},
    }
    report = preflight_otio_timeline(
        timeline,
        adapter_name=adapter_name,
        schema_target=schema_target,
        output_path=output_path,
    )
    if not report["ready"]:
        raise OTIOPreflightError("; ".join(report["errors"]), report)
    target_versions = report["schema_versions"]
    timeline.metadata["opencut_export"]["schema_target"] = report["schema_target"]
    timeline.metadata["opencut_export"]["schema_versions"] = target_versions
    report["output_path"] = output_path
    report["written"] = False
    if preflight_only:
        return report
    if report["lossy"] and not accept_lossy:
        raise OTIOPreflightError(
            "OTIO compatibility target would lose data; explicitly accept the reported fields to write",
            report,
        )

    kwargs: dict[str, Any] = {"adapter_name": adapter_name}
    if adapter_name == "otio_json":
        kwargs["target_schema_versions"] = target_versions
    otio.adapters.write_to_file(timeline, output_path, **kwargs)
    report["written"] = True
    return report
