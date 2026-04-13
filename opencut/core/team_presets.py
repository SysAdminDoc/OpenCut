"""
OpenCut Team Presets via Shared Folder

Scan a shared network/cloud folder for team-wide presets, workflows,
and LUT files.  Sync new or updated assets into the local
``~/.opencut/team/`` directory.

Supported file types:
  - ``.opencut-preset``  — export/processing presets
  - ``.opencut-workflow`` — automation workflows
  - ``.cube``            — 3D LUT (Resolve, Premiere, etc.)
  - ``.3dl``             — 3D LUT (legacy)
"""

import json
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_TEAM_DIR = os.path.join(_OPENCUT_DIR, "team")
_SETTINGS_FILE = os.path.join(_OPENCUT_DIR, "settings.json")

_PRESET_EXTENSIONS = {".opencut-preset", ".opencut-workflow", ".cube", ".3dl"}


def scan_shared_folder(folder_path: str) -> dict:
    """
    Scan a shared folder for preset, workflow, and LUT files.

    Args:
        folder_path: Path to the shared folder.

    Returns:
        dict with ``presets``, ``workflows``, ``luts`` lists (each entry is
        a dict with ``name``, ``path``, ``size``, ``modified``), plus
        ``total`` count.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Shared folder not found: {folder_path}")

    presets = []
    workflows = []
    luts = []

    for root, _dirs, files in os.walk(folder_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _PRESET_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            try:
                stat = os.stat(fpath)
            except OSError:
                continue

            entry = {
                "name": fname,
                "path": fpath,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }

            if ext == ".opencut-preset":
                presets.append(entry)
            elif ext == ".opencut-workflow":
                workflows.append(entry)
            elif ext in (".cube", ".3dl"):
                luts.append(entry)

    return {
        "presets": presets,
        "workflows": workflows,
        "luts": luts,
        "total": len(presets) + len(workflows) + len(luts),
    }


def sync_team_presets(
    shared_folder: str,
    local_folder: Optional[str] = None,
) -> dict:
    """
    Copy new or updated files from a shared folder into the local team
    presets directory.

    Args:
        shared_folder: Path to the shared folder to sync from.
        local_folder: Local destination; defaults to ``~/.opencut/team/``.

    Returns:
        dict with ``synced`` total, ``new`` count, ``updated`` count,
        ``skipped`` count.
    """
    local = local_folder or _TEAM_DIR
    os.makedirs(local, exist_ok=True)

    scan = scan_shared_folder(shared_folder)
    all_files = scan["presets"] + scan["workflows"] + scan["luts"]

    new = 0
    updated = 0
    skipped = 0

    for entry in all_files:
        src = entry["path"]
        # Preserve relative path structure under shared folder
        rel = os.path.relpath(src, shared_folder)
        dst = os.path.join(local, rel)

        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            new += 1
            logger.debug("Team preset new: %s", rel)
        else:
            src_mtime = entry["modified"]
            dst_mtime = os.path.getmtime(dst)
            if src_mtime > dst_mtime:
                shutil.copy2(src, dst)
                updated += 1
                logger.debug("Team preset updated: %s", rel)
            else:
                skipped += 1

    result = {
        "synced": new + updated,
        "new": new,
        "updated": updated,
        "skipped": skipped,
    }
    logger.info("Team preset sync: %s", result)
    return result


def get_shared_folder_path() -> Optional[str]:
    """
    Read the shared folder path from ``~/.opencut/settings.json``.

    Returns:
        The configured path string, or None if not set.
    """
    if not os.path.isfile(_SETTINGS_FILE):
        return None
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("team_shared_folder") or None
    except (json.JSONDecodeError, OSError):
        return None


def set_shared_folder_path(path: str) -> None:
    """
    Store the shared folder path in ``~/.opencut/settings.json``.
    """
    os.makedirs(_OPENCUT_DIR, exist_ok=True)

    data = {}
    if os.path.isfile(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}

    data["team_shared_folder"] = path

    with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Set team shared folder: %s", path)
