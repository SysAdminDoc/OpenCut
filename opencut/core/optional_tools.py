"""Optional tools the installers offer, put where the server loads them.

The WPF installer runs ``OpenCut-Server.exe --install-tools auto-editor,edge-tts``
so a packaged install and the server share one set of pins and one target:

* auto-editor left PyPI at 29.3.1, so it's the pinned native build. The
  download is checked against its SHA-256 and run once before it lands in
  ``<data folder>/bin``, where :mod:`opencut.core.auto_edit` looks first.
* edge-tts and mediapipe go into ``<data folder>/packages`` through
  :func:`opencut.core.package_store.install`, the same path the panel's
  installs take: a staging folder, a child server that has to import them,
  and a merge that puts the old copy back if anything fails.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from typing import Callable, Iterable

from opencut.core import package_store
from opencut.core.package_store import (  # noqa: F401 - server.py and tests use these here
    import_check_command,
    merge_staged_packages,
    run_import_check,
    smoke_import,
)

logger = logging.getLogger("opencut")

AUTO_EDITOR_VERSION = "31.8.0"
AUTO_EDITOR_URL = (
    "https://github.com/WyattBlue/auto-editor/releases/download/"
    f"{AUTO_EDITOR_VERSION}/auto-editor-windows-x86_64.exe"
)
AUTO_EDITOR_SHA256 = "ff86b58c84f8ceccd5b4069543405e545851923c056363171fe955e72a785ce4"
_AUTO_EDITOR_MAX_BYTES = 128 * 1024 * 1024

# (requirement, import name). The bounds match pyproject.toml's extras.
PIP_TOOLS: dict[str, tuple[str, str]] = {
    "edge-tts": ("edge-tts>=6.1,<8", "edge_tts"),
    "mediapipe": ("mediapipe>=0.10,<2", "mediapipe"),
}
TOOL_NAMES: tuple[str, ...] = ("auto-editor", *PIP_TOOLS)

Report = Callable[[str], None]


def _print(line: str) -> None:
    print(line, flush=True)


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def tools_bin_dir() -> str:
    """``<data folder>/bin``, where a native optional tool goes."""
    from opencut import helpers

    return os.path.join(helpers.OPENCUT_DIR, "bin")


# ---------------------------------------------------------------------------
# auto-editor
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_auto_editor(path: str, _response=None) -> bool:
    """Validator for the staged download: the pinned bytes, and they run here."""
    digest = _sha256(path)
    if digest != AUTO_EDITOR_SHA256:
        raise RuntimeError(
            f"The auto-editor download has SHA-256 {digest}, expected {AUTO_EDITOR_SHA256}"
        )
    try:
        result = _run([path, "--version"], timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"auto-editor {AUTO_EDITOR_VERSION} wouldn't start: {exc}") from exc
    reported = (result.stdout or "").strip()
    if result.returncode != 0 or reported != AUTO_EDITOR_VERSION:
        raise RuntimeError(
            f"auto-editor {AUTO_EDITOR_VERSION} didn't run here "
            f"(exit code {result.returncode}, printed {reported!r})"
        )
    return True


def install_auto_editor(report: Report = _print) -> str:
    """Download the pinned auto-editor into :func:`tools_bin_dir`. Returns its path."""
    bin_dir = tools_bin_dir()
    if os.name != "nt":
        raise RuntimeError(
            f"The pinned auto-editor build is for Windows. Download auto-editor "
            f"{AUTO_EDITOR_VERSION} for this system from "
            f"https://github.com/WyattBlue/auto-editor/releases and put it in {bin_dir}"
        )
    from opencut.core.url_safety import transactional_download

    os.makedirs(bin_dir, exist_ok=True)
    destination = os.path.join(bin_dir, "auto-editor.exe")
    if os.path.isfile(destination) and _sha256(destination) == AUTO_EDITOR_SHA256:
        report(f"auto-editor {AUTO_EDITOR_VERSION} is already in {bin_dir}")
        return destination
    report(f"Downloading auto-editor {AUTO_EDITOR_VERSION} (45 MB)...")
    transactional_download(
        AUTO_EDITOR_URL,
        destination,
        max_bytes=_AUTO_EDITOR_MAX_BYTES,
        timeout=120,
        validator=_check_auto_editor,
        label="auto-editor download",
        local_alternative=f"put auto-editor.exe {AUTO_EDITOR_VERSION} in {bin_dir}",
    )
    return destination


# ---------------------------------------------------------------------------
# pip tools
# ---------------------------------------------------------------------------
def install_pip_tool(name: str, report: Report = _print) -> None:
    """pip-install a :data:`PIP_TOOLS` entry through the package store."""
    requirement, module = PIP_TOOLS[name]
    package_store.install([requirement], modules=[module], report=report)


def remove_pip_tool(name: str, report: Report = _print) -> list[str]:
    """Take a :data:`PIP_TOOLS` entry, and what only it needed, out of the store."""
    from packaging.requirements import Requirement

    return package_store.remove([Requirement(PIP_TOOLS[name][0]).name], report=report)


# ---------------------------------------------------------------------------
# --install-tools and --remove-tools
# ---------------------------------------------------------------------------
def _tool_names(names: Iterable[str], report: Report) -> list[str] | None:
    requested = list(dict.fromkeys(n.strip().lower() for n in names if n.strip()))
    unknown = [n for n in requested if n not in TOOL_NAMES]
    if unknown or not requested:
        report(
            f"[failed] {', '.join(unknown) or 'nothing'}: choose from {', '.join(TOOL_NAMES)}"
        )
        return None
    return requested


def remove_tools(names: Iterable[str], report: Report = _print) -> int:
    """Remove each named tool. Exit codes and ``[ok]``/``[failed]`` lines as :func:`install_tools`."""
    requested = _tool_names(names, report)
    if requested is None:
        return 2
    failed = 0
    for name in requested:
        try:
            if name == "auto-editor":
                path = os.path.join(tools_bin_dir(), "auto-editor.exe")
                if os.path.lexists(path):
                    os.remove(path)
                    report(f"[ok] auto-editor is no longer in {tools_bin_dir()}")
                else:
                    report(f"[ok] auto-editor wasn't in {tools_bin_dir()}")
            elif remove_pip_tool(name, report):
                report(f"[ok] {name} is removed. Restart OpenCut if it's running.")
            else:
                report(f"[ok] {name} wasn't installed")
        except Exception as exc:  # noqa: BLE001 - report it and go on to the next tool
            logger.warning("Optional tool %s was not removed: %s", name, exc)
            report(f"[failed] {name}: {' '.join(str(exc).split())}")
            failed += 1
    return 1 if failed else 0


def install_tools(names: Iterable[str], report: Report = _print) -> int:
    """Install each named tool. 0 when all landed, 1 when any failed, 2 for unknown names.

    Every tool ends with one ``[ok] <name>`` or ``[failed] <name>`` line,
    which the WPF installer turns into its log level.
    """
    requested = _tool_names(names, report)
    if requested is None:
        return 2
    failed = 0
    for name in requested:
        try:
            if name == "auto-editor":
                path = install_auto_editor(report)
                report(f"[ok] auto-editor {AUTO_EDITOR_VERSION} is in {path}")
            else:
                install_pip_tool(name, report)
                report(f"[ok] {name} is ready")
        except Exception as exc:  # noqa: BLE001 - report it and go on to the next tool
            logger.warning("Optional tool %s was not installed: %s", name, exc)
            # One line: the installer gives only a status line's first line its level.
            report(f"[failed] {name}: {' '.join(str(exc).split())}")
            failed += 1
    return 1 if failed else 0
