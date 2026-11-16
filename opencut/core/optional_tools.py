"""Optional tools the installers offer, put where the server loads them.

The WPF installer runs ``OpenCut-Server.exe --install-tools auto-editor,edge-tts``
so a packaged install and the server share one set of pins and one target:

* auto-editor left PyPI at 29.3.1, so it's the pinned native build. The
  download is checked against its SHA-256 and run once before it lands in
  ``<data folder>/bin``, where :mod:`opencut.core.auto_edit` looks first.
* edge-tts and mediapipe are pip installs made with a Python that shares the
  bundled runtime's ABI (:func:`opencut.security.install_python`). They go to
  a staging folder and reach ``<data folder>/packages`` only after this server
  imports them in a child process, so a wheel that would crash the bundled
  runtime never becomes visible to it.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterable

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

_PIP_TIMEOUT = 900
_IMPORT_TIMEOUT = 300
_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_STAGING_RE = re.compile(r"\.[a-z0-9-]+-staging-[^\\/]+")
_ASIDE_PREFIX = ".packages-replaced-"
_LEFTOVER_AGE = 3600  # seconds; longer than any pip run plus its import check

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


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


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
def import_check_command(module: str, extra_path: str) -> list[str]:
    """This server, importing ``module`` with ``extra_path`` ahead of the packages folder."""
    if getattr(sys, "frozen", False):
        base = [sys.executable]
    else:
        base = [sys.executable, "-m", "opencut.server"]
    return [*base, "--import-check", module, "--import-path", extra_path]


def smoke_import(module: str, extra_path: str) -> None:
    """Import ``module`` in a child server. A native crash ends the child, not us."""
    try:
        result = _run(import_check_command(module, extra_path), timeout=_IMPORT_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Importing {module} took longer than {_IMPORT_TIMEOUT} seconds"
        ) from exc
    if result.returncode != 0:
        detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
        raise RuntimeError(
            f"The OpenCut server couldn't import {module} "
            f"(exit code {result.returncode}): {detail}"
        )


def run_import_check(module: str, extra_path: str | None) -> int:
    """``--import-check``: import ``module`` with the server's ``sys.path``.

    ``extra_path`` goes just ahead of the packages folder, where its contents
    sit once they're merged in, so the bundled runtime's own modules still win.
    """
    if not _MODULE_RE.fullmatch(module or ""):
        print(f"import-check: {module!r} is not a module name", file=sys.stderr)
        return 2
    if extra_path:
        from opencut import security

        packages = security.user_packages_dir()
        position = sys.path.index(packages) if packages in sys.path else len(sys.path)
        sys.path.insert(position, extra_path)
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any failure means the tool isn't usable
        print(f"import-check: {module}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"import-check: {module} imports", flush=True)
    return 0


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dist_name(entry: str) -> str | None:
    if not entry.endswith(".dist-info"):
        return None
    return _canonical(entry[: -len(".dist-info")].rsplit("-", 1)[0])


def _remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def _put_back(placed: list[str], set_aside: list[tuple[str, str]]) -> bool:
    """Undo a merge that stopped partway. True when every old entry is back."""
    complete = True
    for path in reversed(placed):
        try:
            _remove(path)
        except OSError as exc:
            complete = False
            logger.warning("Couldn't take back %s after a failed merge: %s", path, exc)
    for original, holder in reversed(set_aside):
        try:
            os.replace(holder, original)
        except OSError as exc:
            complete = False
            logger.warning("Couldn't put %s back, the old copy is in %s: %s", original, holder, exc)
    return complete


def merge_staged_packages(staging: str, packages: str) -> None:
    """Move a ``pip install --target`` folder's contents into ``packages``.

    Same-named entries are replaced, like ``pip --target --upgrade``, and an
    older ``.dist-info`` of a replaced project goes too, which pip leaves
    behind. ``bin`` merges file by file so another tool's scripts stay.

    Whatever gets replaced is moved aside first and deleted only once every
    new entry is in place. When a move fails, say because a running server
    has one of the old files loaded, the old entries go back, so a working
    install never ends up half replaced.
    """
    staged = sorted(os.listdir(staging))
    replaced = {name for name in map(_dist_name, staged) if name}
    aside = tempfile.mkdtemp(prefix=_ASIDE_PREFIX, dir=os.path.dirname(packages))
    set_aside: list[tuple[str, str]] = []
    placed: list[str] = []

    def move_aside(path: str) -> None:
        holder = os.path.join(aside, str(len(set_aside)))
        os.replace(path, holder)
        set_aside.append((path, holder))

    try:
        for entry in os.listdir(packages):
            if entry != "bin" and (entry in staged or _dist_name(entry) in replaced):
                move_aside(os.path.join(packages, entry))
        for entry in staged:
            source = os.path.join(staging, entry)
            target = os.path.join(packages, entry)
            if entry == "bin" and os.path.isdir(source) and os.path.isdir(target):
                for script in os.listdir(source):
                    script_target = os.path.join(target, script)
                    if os.path.lexists(script_target):
                        move_aside(script_target)
                    os.replace(os.path.join(source, script), script_target)
                    placed.append(script_target)
                continue
            if os.path.lexists(target):
                move_aside(target)
            os.replace(source, target)
            placed.append(target)
    except BaseException as exc:
        restored = _put_back(placed, set_aside)
        if restored:
            shutil.rmtree(aside, ignore_errors=True)
        if isinstance(exc, OSError):
            where = "is back" if restored else f"is in {aside}"
            raise RuntimeError(
                f"couldn't replace what's in {packages} ({exc}). The copy that was there "
                f"{where}. If OpenCut is running, close it and Premiere Pro, then try again."
            ) from exc
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _sweep_leftovers(data_dir: str) -> None:
    """Delete staging and set-aside folders a killed run left behind.

    The installer kills ``--install-tools`` when it runs past its timeout, and
    then no cleanup runs. Folders younger than :data:`_LEFTOVER_AGE` may belong
    to a run that's still going, so they stay.
    """
    cutoff = time.time() - _LEFTOVER_AGE
    try:
        entries = list(os.scandir(data_dir))
    except OSError:
        return
    for entry in entries:
        if not (entry.name.startswith(_ASIDE_PREFIX) or _STAGING_RE.fullmatch(entry.name)):
            continue
        try:
            if entry.is_dir(follow_symlinks=False) and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry.path)
        except OSError as exc:
            logger.debug("Left %s in place: %s", entry.path, exc)


def install_pip_tool(name: str, report: Report = _print) -> None:
    """pip-install a :data:`PIP_TOOLS` entry, smoke-import it, then make it visible."""
    from opencut import security

    requirement, module = PIP_TOOLS[name]
    requirement = security.runtime_security_requirement(requirement)
    python = security.install_python()
    packages = security.user_packages_dir()
    os.makedirs(packages, exist_ok=True)
    _sweep_leftovers(os.path.dirname(packages))
    staging = tempfile.mkdtemp(prefix=f".{name}-staging-", dir=os.path.dirname(packages))
    try:
        report(f"Installing {requirement} with {python}...")
        try:
            result = _run(
                [
                    python, "-m", "pip", "install", requirement,
                    "--target", staging, "--prefer-binary",
                    "--disable-pip-version-check", "--no-input", "-q",
                ],
                timeout=_PIP_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"pip took longer than {_PIP_TIMEOUT} seconds") from exc
        if result.returncode != 0:
            detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
            raise RuntimeError(f"pip couldn't install {requirement}: {detail}")
        report(f"Checking that the OpenCut server can import {module}...")
        smoke_import(module, staging)
        merge_staged_packages(staging, packages)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# --install-tools
# ---------------------------------------------------------------------------
def install_tools(names: Iterable[str], report: Report = _print) -> int:
    """Install each named tool. 0 when all landed, 1 when any failed, 2 for unknown names.

    Every tool ends with one ``[ok] <name>`` or ``[failed] <name>`` line,
    which the WPF installer turns into its log level.
    """
    requested = list(dict.fromkeys(n.strip().lower() for n in names if n.strip()))
    unknown = [n for n in requested if n not in TOOL_NAMES]
    if unknown or not requested:
        report(
            f"[failed] {', '.join(unknown) or 'nothing'}: choose from {', '.join(TOOL_NAMES)}"
        )
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
