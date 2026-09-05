"""Package and place the UXP panel.

Until now nothing in the repository could deploy ``com.opencut.uxp``. The
README advertised it, every installer lane shipped only the CEP panel, and
``docs/UXP_MIGRATION.md`` Phase 4 still listed "add a local CCX package build
script" as outstanding. With Adobe's ExtendScript support in Premiere Pro
ending in September 2026, the CEP panel is the one on a clock.

Adobe documents two ways a UXP plugin reaches Premiere:

* A ``.ccx`` package, installed through Creative Cloud, a double-click, or the
  Unified Plugin Installer Agent. This is the distribution channel, and the
  marketplace route needs an Adobe review and signing identity.
* A developer-mode sideload: the unpacked plugin placed in
  ``.../Adobe/UXP/Plugins/External/<id>_<version>`` with Developer Mode enabled
  in Premiere's Plugins preferences.

This module builds the first and computes the second. It deliberately does not
pretend the sideload is a substitute for signed distribution -- the developer
mode toggle is a manual step inside Premiere that no installer can set.

References:
https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/install
https://github.com/AdobeDocs/uxp-premiere-pro-samples
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
UXP_SOURCE_DIR = REPO_ROOT / "extension" / "com.opencut.uxp"

#: Files that never belong in a shipped plugin package.
EXCLUDED_NAMES = frozenset({
    "node_modules",
    "__pycache__",
    ".git",
    ".DS_Store",
    "eslint.config.mjs",
})
EXCLUDED_SUFFIXES = frozenset({".log", ".tmp"})


class UXPPackageError(RuntimeError):
    """The UXP plugin source is not in a shippable state."""


def read_uxp_manifest(source_dir: Path | str | None = None) -> dict:
    """Return the plugin manifest, failing loudly on the fields we depend on."""
    base = Path(source_dir) if source_dir is not None else UXP_SOURCE_DIR
    manifest_path = base / "manifest.json"
    if not manifest_path.is_file():
        raise UXPPackageError(f"UXP manifest missing at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UXPPackageError(f"UXP manifest is not valid JSON: {exc}") from exc
    for field in ("id", "version", "main"):
        if not str(manifest.get(field, "")).strip():
            raise UXPPackageError(f"UXP manifest is missing required field {field!r}")
    return manifest


def plugin_folder_name(manifest: dict) -> str:
    """Return ``<id>_<version>``, the folder name Premiere expects.

    Adobe's samples are explicit that the containing folder must be the
    manifest id, an underscore, then the manifest version.
    """
    return f"{manifest['id']}_{manifest['version']}"


def uxp_plugins_root(platform: str | None = None, environ: dict | None = None) -> Path:
    """Return the per-user ``UXP/Plugins/External`` directory for this platform."""
    system = (platform or os.name if platform else os.name)
    env = os.environ if environ is None else environ
    if str(platform or "").lower() == "windows" or (platform is None and os.name == "nt"):
        base = env.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Adobe" / "UXP" / "Plugins" / "External"
    if str(platform or "").lower() == "darwin" or (
        platform is None and system != "nt" and _is_macos(env)
    ):
        return Path(env.get("HOME", str(Path.home()))) / "Library" / "Application Support" / \
            "Adobe" / "UXP" / "Plugins" / "External"
    # Premiere does not ship on Linux; the path is still computed so the
    # installer can report where it *would* go rather than failing opaquely.
    return Path(env.get("HOME", str(Path.home()))) / ".local" / "share" / \
        "Adobe" / "UXP" / "Plugins" / "External"


def _is_macos(environ: dict) -> bool:
    import sys as _sys

    return environ.get("OPENCUT_FAKE_PLATFORM", _sys.platform) == "darwin"


def sideload_target(
    manifest: dict | None = None,
    *,
    platform: str | None = None,
    environ: dict | None = None,
) -> Path:
    """Return the exact directory a developer-mode sideload should occupy."""
    manifest = read_uxp_manifest() if manifest is None else manifest
    return uxp_plugins_root(platform, environ) / plugin_folder_name(manifest)


def iter_package_files(source_dir: Path | str | None = None):
    """Yield ``(absolute_path, archive_name)`` for every file that should ship."""
    base = Path(source_dir) if source_dir is not None else UXP_SOURCE_DIR
    if not base.is_dir():
        raise UXPPackageError(f"UXP plugin source missing at {base}")
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if EXCLUDED_NAMES & set(relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield path, relative.as_posix()


def build_ccx(
    output_path: Path | str,
    source_dir: Path | str | None = None,
) -> dict:
    """Write a ``.ccx`` package and return a manifest of what went in.

    A ``.ccx`` is a zip with the plugin's ``manifest.json`` at the archive root.
    Signing for Creative Cloud distribution is a separate, credentialed step;
    this produces the unsigned package that step consumes.
    """
    base = Path(source_dir) if source_dir is not None else UXP_SOURCE_DIR
    manifest = read_uxp_manifest(base)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    written: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in iter_package_files(base):
            archive.write(path, arcname)
            written.append(arcname)

    if "manifest.json" not in written:
        out.unlink(missing_ok=True)
        raise UXPPackageError("built package has no manifest.json at its root")

    return {
        "path": str(out),
        "plugin_id": manifest["id"],
        "version": manifest["version"],
        "folder_name": plugin_folder_name(manifest),
        "file_count": len(written),
        "bytes": out.stat().st_size,
    }
