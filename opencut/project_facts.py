"""Canonical user-facing project and distribution facts.

The generated project-facts manifest is consumed by documentation and release
gates.  Keep manually verified publication state here; derive everything else
from the runtime contracts and generated route inventory.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from opencut import __version__
from opencut.core.ffmpeg_provenance import (
    PINNED_INSTALLER_VERSION,
    RELEASE_FLOOR,
    SNAPSHOT_FLOOR_DATE,
)
from opencut.dependency_support import (
    PLATFORMS,
    PYTHON_REQUIRES,
    PYTHON_VERSIONS,
    source_extra_install_command,
)

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]

# Publication state was verified against the public registries on this date.
# Update the record when a channel is published; the generated manifest and
# README gate will then make the corresponding command available together.
PUBLICATION_VERIFIED_ON = "2026-07-22"
LATEST_GITHUB_RELEASE = "1.25.1"
LATEST_GITHUB_RELEASE_URL = (
    "https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.25.1"
)
PLATFORM_DISPLAY_NAMES = {
    "win32": "Windows",
    "darwin": "macOS",
    "linux": "Linux",
}


def _project_metadata(repo_root: Path) -> dict[str, Any]:
    with (repo_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _route_facts(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "opencut" / "_generated" / "route_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    readiness = payload.get("readiness_counts") or {}
    return {
        "total": int(payload["total_routes"]),
        "shipped": int(payload.get("shipped_route_count", payload["total_routes"])),
        "stubs": int(readiness.get("stub", 0)),
        "blueprints": len(payload.get("blueprints") or {}),
        "source": "opencut/_generated/route_manifest.json",
    }


def build_project_facts(repo_root: Path | None = None) -> dict[str, Any]:
    """Build the deterministic fact payload used by docs and release gates."""
    root = repo_root or REPO_ROOT
    project = _project_metadata(root)
    version = str(project["version"])
    release_floor = ".".join(str(part) for part in RELEASE_FLOOR)

    channels = {
        "source_checkout": {
            "available": True,
            "current": True,
            "install_command": source_extra_install_command(""),
        },
        "github_windows_installer": {
            "available": True,
            "current": LATEST_GITHUB_RELEASE == version,
            "published_version": LATEST_GITHUB_RELEASE,
            "url": LATEST_GITHUB_RELEASE_URL,
            "asset_pattern": "OpenCut-Setup-<version>.exe",
            "install_command": None,
        },
        "pypi": {
            "available": False,
            "registry": "https://pypi.org/project/opencut-ppro/",
            "install_command": None,
        },
        "homebrew": {"available": False, "install_command": None},
        "winget": {"available": False, "install_command": None},
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "name": "OpenCut",
            "distribution_name": str(project["name"]),
            "version": version,
        },
        "runtime": {
            "python_requires": PYTHON_REQUIRES,
            "python_versions": list(PYTHON_VERSIONS),
            "platforms": list(PLATFORMS),
            "platform_display": [
                PLATFORM_DISPLAY_NAMES[name]
                for name in ("win32", "darwin", "linux")
                if name in PLATFORMS
            ],
        },
        "routes": _route_facts(root),
        "ffmpeg": {
            "release_floor": release_floor,
            "snapshot_floor_date": SNAPSHOT_FLOOR_DATE,
            "pinned_installer_build": PINNED_INSTALLER_VERSION,
        },
        "distribution": {
            "verified_on": PUBLICATION_VERIFIED_ON,
            "channels": channels,
        },
        "trust": {
            "network_default": "local-by-default",
            "fresh_install_telemetry": False,
            "local_only_environment_variable": "OPENCUT_LOCAL_ONLY",
            "optional_egress": [
                "cloud AI providers",
                "Edge-TTS",
                "model, package, and update downloads",
                "OAuth and social uploads",
                "opt-in Aptabase telemetry",
            ],
            "guarded_dynamic_execution": [
                "opencut/core/expression_engine.py",
                "opencut/core/scripting_console.py",
            ],
            "restricted_checkpoint_loading": "opencut/core/model_safety.py",
        },
    }


def validate_project_facts(payload: dict[str, Any]) -> None:
    """Reject impossible publication records before they reach documentation."""
    project = payload["project"]
    if project["version"] != __version__:
        raise ValueError(
            "version drift: pyproject.toml reports "
            f"{project['version']} but opencut.__version__ reports {__version__}"
        )
    for name, channel in payload["distribution"]["channels"].items():
        command = channel.get("install_command")
        if command and not channel.get("available"):
            raise ValueError(f"unpublished distribution channel has an install command: {name}")
