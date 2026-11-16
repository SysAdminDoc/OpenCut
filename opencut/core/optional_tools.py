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
* cuda12 is CTranslate2's own CUDA 12 runtime, which faster-whisper's worker
  needs next to a CUDA 13 PyTorch (:mod:`opencut.ct2_isolation`). It goes
  into ``<data folder>/cuda12`` whole or not at all. The installer doesn't
  offer it, since only that pairing needs it.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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
#: What the WPF installer offers (DependencyInstaller.SupportedTools).
INSTALLER_TOOLS: tuple[str, ...] = ("auto-editor", *PIP_TOOLS)
TOOL_NAMES: tuple[str, ...] = (*INSTALLER_TOOLS, "cuda12")

# Below package_store's _LEFTOVER_AGE, so no sweep takes a download in progress.
CUDA12_TIMEOUT = 50 * 60

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
# cuda12
# ---------------------------------------------------------------------------
def cuda12_problems(root: str) -> list[str]:
    """What keeps ``root`` from serving the CTranslate2 worker. Empty when nothing."""
    from packaging.requirements import Requirement

    from opencut.ct2_isolation import PRELOAD_ORDER, cuda12_library_dirs
    from opencut.gpu_lanes import CT2_CUDA12_RUNTIME

    installed = package_store.read_dists(root)
    problems = []
    for text in CT2_CUDA12_RUNTIME:
        requirement = Requirement(text)
        dist = installed.get(package_store.canonical(requirement.name))
        if dist is None:
            problems.append(f"{requirement.name} is missing")
        elif not requirement.specifier.contains(dist.version, prereleases=True):
            problems.append(f"{requirement.name} is {dist.version}, not {requirement.specifier}")
    dirs = cuda12_library_dirs(root)
    if os.name == "nt":
        for dll in PRELOAD_ORDER:
            if not any(os.path.isfile(os.path.join(folder, dll)) for folder in dirs):
                problems.append(f"{dll} is missing")
    elif not dirs:
        problems.append("it has no nvidia/*/lib folders")
    return problems


def install_cuda12_runtime(report: Report = _print) -> str:
    """pip-install CTranslate2's CUDA 12 runtime into its folder. Returns the folder.

    pip fills a staging folder, which replaces the old runtime only once it
    has every library the worker loads. A runtime in use stays put.
    """
    from opencut import security
    from opencut.gpu_lanes import CT2_CUDA12_RUNTIME, cuda12_runtime_dir

    target = cuda12_runtime_dir()
    if os.path.isdir(target) and not cuda12_problems(target):
        report(f"The CUDA 12 runtime is already in {target}")
        return target
    python = security.install_python()
    data_dir = os.path.dirname(target)
    os.makedirs(data_dir, exist_ok=True)
    # Named like the store's staging folders so a killed run's copy gets swept.
    staging = tempfile.mkdtemp(prefix=".cuda12-staging-", dir=data_dir)
    aside = tempfile.mkdtemp(prefix=".cuda12-staging-", dir=data_dir)
    try:
        report(f"Downloading the CUDA 12 runtime for faster-whisper (about 1.3 GB) with {python}...")
        try:
            result = package_store._run(
                [
                    python, "-m", "pip", "install", *CT2_CUDA12_RUNTIME,
                    "--target", staging, "--only-binary", ":all:",
                    "--disable-pip-version-check", "--no-input", "-q",
                ],
                timeout=CUDA12_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"pip took longer than {CUDA12_TIMEOUT // 60} minutes") from exc
        if result.returncode != 0:
            detail = package_store._tail(result.stderr) or package_store._tail(result.stdout) or "no output"
            raise RuntimeError(f"pip couldn't install the CUDA 12 runtime: {detail}")
        problems = cuda12_problems(staging)
        if problems:
            raise RuntimeError(f"pip's CUDA 12 runtime is incomplete: {'; '.join(problems)}")
        with package_store.store_lock(report):
            old = os.path.join(aside, "cuda12")
            had_old = os.path.lexists(target)
            if had_old:
                try:
                    os.replace(target, old)
                except OSError as exc:
                    raise RuntimeError(
                        f"{target} is in use, so it wasn't replaced. Close OpenCut and run this again. ({exc})"
                    ) from exc
            try:
                os.replace(staging, target)
            except OSError:
                if had_old:
                    os.replace(old, target)
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(aside, ignore_errors=True)
    return target


def remove_cuda12_runtime(report: Report = _print) -> bool:
    """Delete the CUDA 12 runtime folder. False when there was none."""
    from opencut.gpu_lanes import cuda12_runtime_dir

    target = cuda12_runtime_dir()
    failures: list[str] = []

    def _note(func, path, exc):
        exc = exc[1] if isinstance(exc, tuple) else exc
        # A folder that kept a locked file can't go either; the file is the news.
        if func is os.rmdir and getattr(exc, "errno", None) in (errno.ENOTEMPTY, errno.EEXIST):
            return
        failures.append(f"{path}: {exc}")

    with package_store.store_lock(report):
        if not os.path.lexists(target):
            return False
        if os.path.isdir(target) and not os.path.islink(target):
            if sys.version_info >= (3, 12):
                shutil.rmtree(target, onexc=_note)
            else:
                shutil.rmtree(target, onerror=_note)
        else:
            os.remove(target)
    if failures or os.path.lexists(target):
        raise RuntimeError(
            f"{len(failures)} files in {target} couldn't be deleted. Close OpenCut and run this again."
            + (f" First one: {failures[0]}" if failures else "")
        )
    return True


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
            elif name == "cuda12":
                if remove_cuda12_runtime(report):
                    report("[ok] cuda12 is removed")
                else:
                    report("[ok] cuda12 wasn't installed")
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
            elif name == "cuda12":
                path = install_cuda12_runtime(report)
                report(f"[ok] cuda12 is in {path}. Restart OpenCut if it's running.")
            else:
                install_pip_tool(name, report)
                report(f"[ok] {name} is ready")
        except Exception as exc:  # noqa: BLE001 - report it and go on to the next tool
            logger.warning("Optional tool %s was not installed: %s", name, exc)
            # One line: the installer gives only a status line's first line its level.
            report(f"[failed] {name}: {' '.join(str(exc).split())}")
            failed += 1
    return 1 if failed else 0
