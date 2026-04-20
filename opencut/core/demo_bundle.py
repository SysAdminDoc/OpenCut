"""
OpenCut Demo Bundle (Wave H1.6, v1.25.0)

Serves the bundled sample footage under ``opencut/data/demo/``. Panel
calls ``GET /system/demo/sample`` to pre-fill ``filepath`` on every tab's
"Try on demo" button.

Installer builds ship the sample as part of the PyInstaller spec; pip /
git installs can populate the folder via::

    opencut-server --download-demo

which pulls the asset from the GitHub release attached to the current
version. Offline dev environments simply show an empty list + a note.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List

logger = logging.getLogger("opencut")


SAMPLE_FILENAME = "sample.mp4"
SAMPLE_DOWNLOAD_TEMPLATE = (
    "https://github.com/SysAdminDoc/OpenCut/releases/download/"
    "v{version}/{name}"
)


def _demo_dir() -> str:
    """Return ``opencut/data/demo/`` absolute path."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "demo",
    )


def check_demo_bundle_available() -> bool:
    """True when a demo asset is on disk. Dev installs return False."""
    d = _demo_dir()
    return os.path.isdir(d) and bool(os.listdir(d) if os.path.isdir(d) else [])


def list_assets() -> Dict[str, Any]:
    """List any demo assets shipped with the installer."""
    d = _demo_dir()
    if not os.path.isdir(d):
        return {
            "assets": [],
            "dir": d,
            "note": "demo folder not found — installer builds ship it; "
                    "run `opencut-server --download-demo` on dev installs",
        }

    out: List[Dict[str, Any]] = []
    try:
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 0
                out.append({"name": name, "path": p, "size": size})
    except OSError as exc:
        return {"assets": [], "dir": d, "note": f"list failed: {exc}"}
    return {"assets": out, "dir": d, "note": ""}


def get_sample() -> Dict[str, Any]:
    """Return the primary sample.mp4 path, or a diagnostic note."""
    d = _demo_dir()
    p = os.path.join(d, SAMPLE_FILENAME)
    if os.path.isfile(p):
        return {"path": p, "size": os.path.getsize(p), "exists": True}
    return {
        "path": "", "size": 0, "exists": False,
        "note": f"{SAMPLE_FILENAME} not found in {d}",
    }


def download_sample(version: str = "", timeout: float = 60.0) -> Dict[str, Any]:
    """Download sample.mp4 from the GitHub release asset for ``version``.

    Used by the ``opencut-server --download-demo`` flag; callable via
    route so UI can also trigger it. Returns ``{path, bytes, version, url}``.
    """
    if not version:
        try:
            from opencut import __version__ as ocv
            version = ocv
        except Exception:  # noqa: BLE001
            version = "latest"

    d = _demo_dir()
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, SAMPLE_FILENAME)
    url = SAMPLE_DOWNLOAD_TEMPLATE.format(version=version, name=SAMPLE_FILENAME)

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "OpenCut-Panel/1.25.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "path": "", "bytes": 0, "version": version, "url": url,
            "error": f"download failed: {exc}",
        }

    with open(dest, "wb") as fh:
        fh.write(data)
    return {
        "path": dest, "bytes": len(data),
        "version": version, "url": url,
    }


__all__ = [
    "check_demo_bundle_available",
    "list_assets",
    "get_sample",
    "download_sample",
]
