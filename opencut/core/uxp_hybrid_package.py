"""Static validation for UXP Hybrid plugin package layout.

Adobe's Hybrid plugin packaging rules are stricter than ordinary UXP bundles:
the manifest must opt in to native addons and the compiled ``.uxpaddon``
binaries must be present for the target platform/architecture layout before
UDT packaging. This module keeps those checks cheap and deterministic so a
future native addon cannot drift into a broken release bundle unnoticed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

MANIFEST_FILENAME = "manifest.json"
REQUIRED_MARKETPLACE_ARCHITECTURES = ("mac/arm64", "mac/x64", "win/x64")
SUPPORTED_LAYOUT_ROOTS = ("addons", "")


@dataclass
class UxpHybridPackageValidation:
    valid: bool
    hybrid: bool
    addon_name: str = ""
    layout_root: str = ""
    architectures: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _load_manifest(plugin_root: Path, result: UxpHybridPackageValidation) -> dict | None:
    manifest_path = plugin_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        result.errors.append("manifest.json missing from UXP plugin root")
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.errors.append(f"manifest.json unreadable: {exc}")
        return None
    if not isinstance(payload, dict):
        result.errors.append("manifest.json must contain a JSON object")
        return None
    return payload


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _addon_name_from_manifest(manifest: dict, result: UxpHybridPackageValidation) -> str:
    addon = manifest.get("addon")
    if addon is None:
        return ""
    result.hybrid = True
    if not isinstance(addon, dict):
        result.errors.append("addon must be an object with a name field")
        return ""
    name = addon.get("name")
    if not isinstance(name, str) or not name.strip():
        result.errors.append("addon.name must be a non-empty string")
        return ""
    name = name.strip()
    if "/" in name or "\\" in name or name in {".", ".."} or ".." in Path(name).parts:
        result.errors.append("addon.name must be a filename, not a path")
    if not name.endswith(".uxpaddon"):
        result.errors.append("addon.name must end with .uxpaddon")
    result.addon_name = name
    return name


def _iter_uxpaddons(plugin_root: Path) -> Iterable[Path]:
    yield from plugin_root.rglob("*.uxpaddon")


def _host_app(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("app", ""))
    return ""


def _validate_hybrid_manifest(manifest: dict, *, require_marketplace_architectures: bool) -> list[str]:
    errors: list[str] = []
    version = manifest.get("manifestVersion")
    if not isinstance(version, int) or version < 6:
        errors.append("hybrid manifestVersion must be an integer >= 6")

    permissions = manifest.get("requiredPermissions")
    if not isinstance(permissions, dict) or permissions.get("enableAddon") is not True:
        errors.append("hybrid requiredPermissions.enableAddon must be true")

    host = manifest.get("host")
    if require_marketplace_architectures and isinstance(host, list):
        errors.append("hybrid marketplace package must use one host object, not a development host array")
    host_entries = host if isinstance(host, list) else [host]
    apps = {_host_app(entry).lower() for entry in host_entries}
    if not ({"ppro", "premierepro"} & apps):
        errors.append("hybrid manifest host must target Premiere Pro")
    return errors


def _find_architecture_files(plugin_root: Path, addon_name: str) -> tuple[str, dict[str, str], list[str]]:
    matches_by_layout: dict[str, dict[str, str]] = {}
    for layout in SUPPORTED_LAYOUT_ROOTS:
        found: dict[str, str] = {}
        layout_root = plugin_root / layout if layout else plugin_root
        for arch in REQUIRED_MARKETPLACE_ARCHITECTURES:
            path = layout_root / Path(arch) / addon_name
            if path.is_file():
                found[arch] = _relative(path, plugin_root)
        if found:
            matches_by_layout[layout] = found

    if not matches_by_layout:
        return "", {}, []
    if len(matches_by_layout) > 1:
        layouts = ", ".join(repr(layout or ".") for layout in sorted(matches_by_layout))
        return "", {}, [f"addon binaries use mixed layout roots: {layouts}"]
    layout, matches = next(iter(matches_by_layout.items()))
    empty = []
    for rel in matches.values():
        path = plugin_root / rel
        if path.stat().st_size <= 0:
            empty.append(f"{rel} is empty")
    return layout, matches, empty


def validate_uxp_hybrid_package(
    plugin_root: str | Path,
    *,
    require_marketplace_architectures: bool = True,
) -> UxpHybridPackageValidation:
    """Validate a UXP plugin directory for Hybrid addon packaging.

    Non-hybrid UXP plugins are valid when they omit both ``addon`` and shipped
    ``.uxpaddon`` files. Hybrid plugins must include manifest opt-in fields and
    at least one supported architecture layout; Marketplace mode requires all
    Adobe-listed architectures.
    """
    root = Path(plugin_root)
    result = UxpHybridPackageValidation(valid=True, hybrid=False)
    if not root.is_dir():
        result.errors.append(f"UXP plugin root is not a directory: {root}")
        result.valid = False
        return result

    manifest = _load_manifest(root, result)
    if manifest is None:
        result.valid = False
        return result

    addon_name = _addon_name_from_manifest(manifest, result)
    shipped_addons = sorted(_relative(path, root) for path in _iter_uxpaddons(root))
    if not result.hybrid:
        if shipped_addons:
            result.errors.append(
                "plugin ships .uxpaddon files but manifest.json omits the addon field: "
                + ", ".join(shipped_addons[:5])
            )
        result.valid = not result.errors
        return result

    result.errors.extend(
        _validate_hybrid_manifest(
            manifest,
            require_marketplace_architectures=require_marketplace_architectures,
        )
    )
    if not addon_name:
        result.valid = False
        return result

    layout, architectures, layout_errors = _find_architecture_files(root, addon_name)
    result.layout_root = layout
    result.architectures = architectures
    result.errors.extend(layout_errors)

    if not architectures:
        result.errors.append(
            "no addon binaries found in addons/{mac/arm64,mac/x64,win/x64} "
            "or root {mac/arm64,mac/x64,win/x64} layout"
        )
    missing = [arch for arch in REQUIRED_MARKETPLACE_ARCHITECTURES if arch not in architectures]
    if missing and require_marketplace_architectures:
        result.errors.append("missing Marketplace addon architectures: " + ", ".join(missing))
    elif missing:
        result.warnings.append(
            "partial hybrid architecture set; unsupported platforms will fail to load: "
            + ", ".join(missing)
        )

    result.valid = not result.errors
    return result
