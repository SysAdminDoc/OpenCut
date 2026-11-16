"""The add-on package store, ``<data folder>/packages``.

The server appends this folder to ``sys.path`` after everything it bundles, so
an add-on package can add modules but never replace a bundled one. Every
install into it goes through :func:`install`:

1. One lock, ``<data folder>/packages.lock``, covers the installer's
   ``--install-tools``, the panel's install routes and every server process,
   so two installs never merge into the folder at once.
2. pip installs into a staging folder next to the store, with a Python that
   shares the server's ABI (:func:`opencut.security.install_python`).
3. A dependency the store already has stays as it is when the new packages
   accept that version. When they don't, the new version replaces it, but
   only if every package that stays in the store accepts the new version too.
4. A child server imports the new packages with the staging folder ahead of
   the store, so a wheel that crashes the bundled runtime only ends the child.
5. The merge moves each replaced entry aside first and journals every move.
   A failure, or a kill partway through, puts the old store back.

Folders that several projects share (``nvidia``, ``google``, ``__pycache__``)
merge entry by entry, so a new package never removes another one's files.
The first install for a given runtime also moves packages built for another
Python out of the store (:func:`inventory_once`).
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email.parser import HeaderParser
from typing import Callable, Iterable, Iterator, Sequence

logger = logging.getLogger("opencut")

PIP_TIMEOUT = 900
IMPORT_TIMEOUT = 300
LOCK_WAIT = 30 * 60  # seconds; the installer gives --install-tools 45 minutes

_MODULE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_STAGING_RE = re.compile(r"\.[a-z0-9-]+-staging-[^\\/]+")
_ASIDE_PREFIX = ".packages-replaced-"
_JOURNAL = "journal"
_LEFTOVER_AGE = 3600  # seconds; longer than any pip run plus its import check
_LOCK_FILE = "packages.lock"
_INVENTORY_FILE = "packages-inventory.json"
_QUARANTINE_DIR = "packages-quarantine"

Report = Callable[[str], None]
OnProcess = Callable[[subprocess.Popen], None]


class PipFailed(RuntimeError):
    """pip itself failed. ``result`` has its output."""

    def __init__(self, message: str, result: subprocess.CompletedProcess):
        super().__init__(message)
        self.result = result


class StoreBusy(RuntimeError):
    """Another install held the store lock for longer than the wait."""


def _log(line: str) -> None:
    logger.info("%s", line)


def _run(cmd: list[str], timeout: float, on_start: OnProcess | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if on_start is not None:
        on_start(proc)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def packages_dir() -> str:
    from opencut import security

    return security.user_packages_dir()


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------
_THREAD_LOCK = threading.RLock()
_FILE_LOCKS: dict = {}


def _busy_message(wait: float) -> str:
    minutes = max(1, round(wait / 60))
    return (
        f"Another add-on install was still running after {minutes} minute"
        f"{'s' if minutes != 1 else ''}. Let it finish, then try again."
    )


@contextmanager
def store_lock(report: Report = _log, wait: float = LOCK_WAIT) -> Iterator[None]:
    """Hold the store lock. Waits up to ``wait`` seconds for another install."""
    from filelock import FileLock, Timeout

    data_dir = os.path.dirname(packages_dir())
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _LOCK_FILE)
    deadline = time.monotonic() + wait
    told = False
    if not _THREAD_LOCK.acquire(blocking=False):
        report("Waiting for another add-on install to finish...")
        told = True
        if wait <= 0 or not _THREAD_LOCK.acquire(timeout=wait):
            raise StoreBusy(_busy_message(wait))
    try:
        lock = _FILE_LOCKS.get(path)
        if lock is None:
            lock = _FILE_LOCKS[path] = FileLock(path, thread_local=False)
        try:
            lock.acquire(timeout=0)
        except Timeout:
            if not told:
                report("Waiting for another add-on install to finish...")
            try:
                lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
            except Timeout:
                raise StoreBusy(_busy_message(wait)) from None
        try:
            yield
        finally:
            lock.release()
    finally:
        _THREAD_LOCK.release()


# ---------------------------------------------------------------------------
# What's installed
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dist:
    name: str  # canonical project name
    display: str
    version: str
    info: str  # the .dist-info folder
    requires: tuple[str, ...]
    files: frozenset[str]  # posix paths relative to the folder, from RECORD
    top_level: tuple[str, ...]
    tags: tuple[str, ...]  # WHEEL Tag lines


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _record_path(raw: str) -> str | None:
    """A RECORD path relative to the --target folder, or None when it's outside."""
    path = raw.replace("\\", "/")
    if path.startswith("../../bin/"):  # pip --target puts scripts in <target>/bin
        path = "bin/" + path[len("../../bin/"):]
    if not path or path.startswith("/") or ":" in path.split("/", 1)[0]:
        return None
    path = posixpath.normpath(path)
    if path == "." or path == ".." or path.startswith("../"):
        return None
    return path


def _dist_name(entry: str) -> str | None:
    if not entry.endswith(".dist-info"):
        return None
    return canonical(entry[: -len(".dist-info")].rsplit("-", 1)[0])


def read_dists(folder: str) -> dict[str, Dist]:
    """Every ``*.dist-info`` in ``folder``, by canonical project name."""
    dists: dict[str, Dist] = {}
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return dists
    for entry in entries:
        info = os.path.join(folder, entry)
        if not entry.endswith(".dist-info") or not os.path.isdir(info):
            continue
        stem = entry[: -len(".dist-info")]
        fallback_name, _sep, fallback_version = stem.rpartition("-")
        if not fallback_name:
            fallback_name, fallback_version = stem, ""
        meta = HeaderParser().parsestr(_read_text(os.path.join(info, "METADATA")))
        display = meta.get("Name") or fallback_name
        files = set()
        for row in csv.reader(io.StringIO(_read_text(os.path.join(info, "RECORD")))):
            if row:
                path = _record_path(row[0])
                if path:
                    files.add(path)
        top_level = tuple(
            line.strip()
            for line in _read_text(os.path.join(info, "top_level.txt")).splitlines()
            if line.strip()
        )
        wheel = HeaderParser().parsestr(_read_text(os.path.join(info, "WHEEL")))
        dists[canonical(display)] = Dist(
            name=canonical(display),
            display=display,
            version=meta.get("Version") or fallback_version,
            info=entry,
            requires=tuple(meta.get_all("Requires-Dist") or ()),
            files=frozenset(files),
            top_level=top_level,
            tags=tuple(meta_tag.strip() for meta_tag in wheel.get_all("Tag") or ()),
        )
    return dists


def _owned(dist: Dist) -> set[str]:
    """Paths ``dist`` owns. Without a RECORD that's its top-level names."""
    if dist.files:
        return set(dist.files)
    return {dist.info, *dist.top_level}


def _ancestors(paths: Iterable[str]) -> set[str]:
    out = set()
    for path in paths:
        parts = path.split("/")
        for depth in range(1, len(parts)):
            out.add("/".join(parts[:depth]))
    return out


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------
def _requirements(dist: Dist) -> list:
    from packaging.requirements import InvalidRequirement, Requirement

    out = []
    for line in dist.requires:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            logger.debug("Skipped requirement %r of %s", line, dist.display)
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        out.append(req)
    return out


def _accepts(req, version: str) -> bool:
    from packaging.version import InvalidVersion, Version

    try:
        return req.specifier.contains(Version(version), prereleases=True)
    except InvalidVersion:
        return True  # can't judge a non-PEP 440 version, so it doesn't block


def plan_merge(
    staged: dict[str, Dist], store: dict[str, Dist], requested: set[str], keep_shared: bool
) -> tuple[list[str], list[str]]:
    """Which staged dependencies to drop in favor of the store's copy, and conflicts.

    A dependency the store already has stays when every staged package that
    needs it accepts the store's version. Whatever still replaces a store copy
    has to satisfy every package that stays in the store.
    """
    keep: list[str] = []
    if keep_shared:
        for name in sorted((staged.keys() & store.keys()) - requested):
            needs = [
                req
                for dist in staged.values()
                if dist.name != name
                for req in _requirements(dist)
                if canonical(req.name) == name
            ]
            if all(_accepts(req, store[name].version) for req in needs):
                keep.append(name)
    incoming = {name: dist for name, dist in staged.items() if name not in keep}
    conflicts = []
    for stay in store.values():
        if stay.name in incoming:
            continue
        for req in _requirements(stay):
            new = incoming.get(canonical(req.name))
            if new is not None and not _accepts(req, new.version):
                old = store.get(new.name)
                change = f"{old.version} with {new.version}" if old else new.version
                conflicts.append(
                    f"{stay.display} {stay.version} needs {req}, and this would replace "
                    f"{new.display} {change}"
                )
    return keep, conflicts


def _drop_from_staging(staging: str, dist: Dist, others: Iterable[Dist]) -> None:
    """Remove ``dist``'s files from the staging folder, keeping what others own."""
    others_owned = set()
    for other in others:
        others_owned |= _owned(other)
    shared = _ancestors(others_owned) | others_owned
    for rel in sorted(dist.files):
        if rel in others_owned:
            continue
        path = os.path.join(staging, *rel.split("/"))
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
    for top in sorted({rel.split("/", 1)[0] for rel in _owned(dist)}):
        path = os.path.join(staging, top)
        if not os.path.lexists(path):
            continue
        if top not in shared:
            _remove(path)
        elif os.path.isdir(path):
            _prune_empty_dirs(path)


def _prune_empty_dirs(root: str) -> None:
    for current, _dirs, _files in os.walk(root, topdown=False):
        try:
            os.rmdir(current)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Import check
# ---------------------------------------------------------------------------
def import_check_command(module: str, extra_path: str) -> list[str]:
    """This server, importing ``module`` with ``extra_path`` ahead of the packages folder."""
    if getattr(sys, "frozen", False):
        base = [sys.executable]
    else:
        base = [sys.executable, "-m", "opencut.server"]
    return [*base, "--import-check", module, "--import-path", extra_path]


def smoke_import(module: str, extra_path: str, on_process: OnProcess | None = None) -> None:
    """Import ``module`` in a child server. A native crash ends the child, not us."""
    try:
        result = _run(import_check_command(module, extra_path), timeout=IMPORT_TIMEOUT, on_start=on_process)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Importing {module} took longer than {IMPORT_TIMEOUT} seconds") from exc
    if result.returncode != 0:
        detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
        raise RuntimeError(
            f"The OpenCut server couldn't import {module} (exit code {result.returncode}): {detail}"
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
        packages = packages_dir()
        position = sys.path.index(packages) if packages in sys.path else len(sys.path)
        sys.path.insert(position, extra_path)
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - any failure means the package isn't usable
        print(f"import-check: {module}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"import-check: {module} imports", flush=True)
    return 0


def _public_module(name: str) -> bool:
    return bool(_MODULE_RE.fullmatch(name)) and not any(part.startswith("_") for part in name.split("."))


def import_names(dist: Dist) -> list[str]:
    """The module to import-check for ``dist``.

    That's the module named like the project when there is one
    (``pyannote.audio`` for pyannote-audio), otherwise its first top-level
    module (``cv2`` for opencv-python, ``PIL`` for Pillow).
    """
    found = set()
    for rel in dist.files:
        parts = rel.split("/")
        if len(parts) >= 2 and parts[-1] == "__init__.py":
            found.add(".".join(parts[:-1]))
        elif len(parts) == 1 and rel.endswith(".py"):
            found.add(rel[:-3])
    declared = {name for name in dist.top_level if _public_module(name)}
    found = {name for name in found if _public_module(name)} | declared
    wanted = dist.name.replace("-", "_")
    for name in sorted(found, key=lambda n: (n.count("."), n)):
        if name.replace(".", "_").lower() == wanted:
            return [name]
    tops = sorted(declared) or sorted(name for name in found if "." not in name)
    return [tops[0]] if tops else [wanted]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def _remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


class _Journal:
    """Each move is written down before it happens, so a killed merge can be undone."""

    def __init__(self, path: str):
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, op: str, *paths: str) -> None:
        self._handle.write(json.dumps([op, *paths]) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


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


def merge_staged_packages(staging: str, packages: str, protected: Iterable[str] | None = None) -> None:
    """Move a ``pip install --target`` folder's contents into ``packages``.

    A staged entry replaces the one in the store, like ``pip --target
    --upgrade``, and an older ``.dist-info`` of a replaced project goes too.
    A folder that holds files of a project the merge doesn't replace (shared
    namespace folders, ``bin``, ``__pycache__``) merges entry by entry.
    ``protected`` lists those files; by default it comes from the store's
    RECORDs.

    Whatever gets replaced is moved aside first and deleted only once every
    new entry is in place. When a move fails, say because a running server
    has one of the old files loaded, the old entries go back, so a working
    install never ends up half replaced.
    """
    staged = sorted(os.listdir(staging))
    replaced = {name for name in map(_dist_name, staged) if name}
    if protected is None:
        protected = set()
        for dist in read_dists(packages).values():
            if dist.name not in replaced:
                protected |= _owned(dist)
    merge_into = _ancestors(protected) | {"bin"}
    aside = tempfile.mkdtemp(prefix=_ASIDE_PREFIX, dir=os.path.dirname(packages))
    journal = _Journal(os.path.join(aside, _JOURNAL))
    set_aside: list[tuple[str, str]] = []
    placed: list[str] = []

    def move_aside(path: str) -> None:
        holder = os.path.join(aside, str(len(set_aside)))
        journal.write("aside", path, holder)
        os.replace(path, holder)
        set_aside.append((path, holder))

    def place(rel: str) -> None:
        source = os.path.join(staging, *rel.split("/"))
        target = os.path.join(packages, *rel.split("/"))
        if os.path.lexists(target):
            if rel in merge_into and os.path.isdir(source) and os.path.isdir(target) and not os.path.islink(target):
                for child in sorted(os.listdir(source)):
                    place(f"{rel}/{child}")
                return
            move_aside(target)
        journal.write("placed", target)
        os.replace(source, target)
        placed.append(target)

    try:
        for entry in sorted(os.listdir(packages)):
            if entry not in staged and _dist_name(entry) in replaced:
                move_aside(os.path.join(packages, entry))
        for entry in staged:
            place(entry)
        journal.write("done")
    except BaseException as exc:
        restored = _put_back(placed, set_aside)
        journal.close()
        if restored:
            shutil.rmtree(aside, ignore_errors=True)
        if isinstance(exc, OSError):
            where = "is back" if restored else f"is in {aside}"
            raise RuntimeError(
                f"couldn't replace what's in {packages} ({exc}). The copy that was there "
                f"{where}. If OpenCut is running, close it and Premiere Pro, then try again."
            ) from exc
        raise
    journal.close()
    shutil.rmtree(aside, ignore_errors=True)


def _recover(aside: str) -> bool:
    """Finish or undo the merge a journal describes. True when nothing is left to do."""
    ops = []
    for line in _read_text(os.path.join(aside, _JOURNAL)).splitlines():
        try:
            ops.append(json.loads(line))
        except ValueError:
            break  # the line a kill cut short; its move never happened
    if ["done"] not in ops:
        placed = [op[1] for op in ops if op[0] == "placed" and os.path.lexists(op[1])]
        set_aside = [(op[1], op[2]) for op in ops if op[0] == "aside" and os.path.lexists(op[2])]
        if not _put_back(placed, set_aside):
            return False
        logger.warning("Put back %d package entries an interrupted install had replaced", len(set_aside))
    shutil.rmtree(aside, ignore_errors=True)
    return True


def _sweep_leftovers(data_dir: str) -> None:
    """Clean up after installs that were killed. Call it holding the store lock.

    A journaled set-aside folder is a merge that never finished, so its old
    entries go back. Staging folders, and set-aside folders from before the
    journal, go once they're older than :data:`_LEFTOVER_AGE`.
    """
    cutoff = time.time() - _LEFTOVER_AGE
    try:
        entries = list(os.scandir(data_dir))
    except OSError:
        return
    for entry in entries:
        aside = entry.name.startswith(_ASIDE_PREFIX)
        if not (aside or _STAGING_RE.fullmatch(entry.name)):
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if aside and os.path.isfile(os.path.join(entry.path, _JOURNAL)):
                _recover(entry.path)
            elif entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry.path)
        except OSError as exc:
            logger.debug("Left %s in place: %s", entry.path, exc)


# ---------------------------------------------------------------------------
# Packages built for another Python
# ---------------------------------------------------------------------------
_QUARANTINE_README = """\
OpenCut moved these add-on packages out of
{packages}
on {when}. They were built for a different Python than the one OpenCut runs
({abi}), and loading one could crash the server.

Moved:
{listing}

To get one back, install it again from the OpenCut panel, or with
OpenCut-Server.exe --install-tools for edge-tts and mediapipe.
To undo this, move everything in this folder back into the packages folder.
Once OpenCut works the way you want, you can delete this folder.
"""


def _supported_tags() -> set | None:
    from packaging import tags

    supported = set(tags.sys_tags())
    ours = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    if sys.implementation.name == "cpython" and not any(t.interpreter == ours for t in supported):
        logger.warning("Couldn't work out which wheels this Python runs, so the packages folder wasn't checked")
        return None
    return supported


def _built_elsewhere(dist: Dist, supported: set) -> bool:
    from packaging.tags import parse_tag

    if not dist.tags:
        return False
    for tag in dist.tags:
        try:
            if parse_tag(tag) & supported:
                return False
        except ValueError:
            return False  # a tag we can't read could be fine
    return True


def inventory_once(report: Report = _log) -> list[str]:
    """Move packages built for another Python out of the store. Hold the lock.

    It runs once per runtime ABI: ``packages-inventory.json`` remembers it.
    Packages without wheel tags are kept, since nothing says they're wrong.
    Returns the projects it moved.
    """
    from opencut import security

    packages = packages_dir()
    data_dir = os.path.dirname(packages)
    marker = os.path.join(data_dir, _INVENTORY_FILE)
    abi = security.runtime_abi()
    try:
        with open(marker, encoding="utf-8") as handle:
            if json.load(handle).get("abi") == abi:
                return []
    except (OSError, ValueError, AttributeError):
        pass
    supported = _supported_tags()
    if supported is None:
        return []
    dists = read_dists(packages)
    moved = [d for d in dists.values() if _built_elsewhere(d, supported)]
    quarantine = None
    if moved:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        quarantine = os.path.join(data_dir, _QUARANTINE_DIR, stamp)
        os.makedirs(quarantine, exist_ok=True)
        staying = [d for d in dists.values() if d not in moved]
        for dist in moved:
            _move_dist(packages, quarantine, dist, staying)
        listing = "\n".join(f"  {d.display} {d.version} ({', '.join(d.tags)})" for d in moved)
        with open(os.path.join(quarantine, "README.txt"), "w", encoding="utf-8") as handle:
            handle.write(
                _QUARANTINE_README.format(
                    packages=packages, when=time.strftime("%Y-%m-%d %H:%M"), abi=abi, listing=listing
                )
            )
        report(
            f"Moved {len(moved)} add-on package{'s' if len(moved) != 1 else ''} built for another "
            f"Python to {quarantine}"
        )
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "abi": abi,
                "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "kept": sorted(d.name for d in dists.values() if d not in moved),
                "moved": sorted(d.name for d in moved),
                "quarantine": quarantine,
            },
            handle,
            indent=2,
        )
    return [d.name for d in moved]


def _move_dist(packages: str, quarantine: str, dist: Dist, staying: Sequence[Dist]) -> None:
    others = set()
    for other in staying:
        others |= _owned(other)
    for rel in sorted(_owned(dist)):
        if rel in others:
            continue
        source = os.path.join(packages, *rel.split("/"))
        if not os.path.lexists(source):
            continue
        target = os.path.join(quarantine, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
    for top in sorted({rel.split("/", 1)[0] for rel in _owned(dist)}):
        path = os.path.join(packages, top)
        if os.path.isdir(path):
            _prune_empty_dirs(path)


def inventory_at_startup() -> None:
    """Run :func:`inventory_once` when no install holds the lock right now."""
    try:
        with store_lock(wait=0):
            inventory_once()
    except (ImportError, OSError, RuntimeError) as exc:
        logger.info("Skipped the add-on package check for now: %s", exc)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
def install(
    requirements: Sequence[str],
    *,
    modules: Sequence[str] | None = None,
    pip_args: Sequence[str] = (),
    python: str | None = None,
    keep_shared: bool = True,
    timeout: float = PIP_TIMEOUT,
    report: Report = _log,
    on_process: OnProcess | None = None,
) -> subprocess.CompletedProcess:
    """pip-install ``requirements`` into the store the safe way. Returns pip's result.

    ``modules`` are what the server has to import afterwards; by default each
    requested project's own top-level module. ``keep_shared=False`` replaces
    dependencies the store already has (a reinstall), still subject to the
    conflict check. Raises :class:`PipFailed` when pip fails and
    ``RuntimeError`` when the server can't import the result or it would
    break a package in the store. Either way the store is left as it was.
    """
    from packaging.requirements import Requirement

    from opencut import security

    requirements = [security.runtime_security_requirement(r) for r in requirements]
    requested = {canonical(Requirement(r).name) for r in requirements}
    python = python or security.install_python()
    packages = packages_dir()
    os.makedirs(packages, exist_ok=True)
    data_dir = os.path.dirname(packages)
    label = "-".join(sorted(requested))[:40].strip("-") or "package"
    with store_lock(report):
        _sweep_leftovers(data_dir)
        try:
            inventory_once(report)
        except OSError as exc:  # it tries again on the next install
            logger.warning("Couldn't finish checking the packages folder: %s", exc)
        staging = tempfile.mkdtemp(prefix=f".{label}-staging-", dir=data_dir)
        # Named like a staging folder so a killed install's copy gets swept too.
        constraints = tempfile.mkdtemp(prefix=f".{label}-staging-", dir=data_dir)
        try:
            report(f"Installing {' '.join(requirements)} with {python}...")
            try:
                result = _run(
                    [
                        python, "-m", "pip", "install", *requirements, *pip_args,
                        *_lane_constraints(constraints),
                        "--target", staging, "--prefer-binary",
                        "--disable-pip-version-check", "--no-input", "-q",
                    ],
                    timeout=timeout,
                    on_start=on_process,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"pip took longer than {round(timeout)} seconds") from exc
            if result.returncode != 0:
                detail = _tail(result.stderr) or _tail(result.stdout) or "no output"
                raise PipFailed(f"pip couldn't install {' '.join(requirements)}: {detail}", result)
            staged = read_dists(staging)
            store = read_dists(packages)
            keep, conflicts = plan_merge(staged, store, requested, keep_shared)
            if conflicts:
                raise RuntimeError(
                    "Nothing was changed, because installing "
                    f"{' '.join(requirements)} would break what's installed: {'; '.join(conflicts)}"
                )
            for name in keep:
                _drop_from_staging(staging, staged[name], [d for n, d in staged.items() if n not in keep])
            if keep:
                logger.info("Kept the installed copies of %s", ", ".join(keep))
            if modules is None:
                modules = [
                    module
                    for name in sorted(requested)
                    for module in (import_names(staged[name]) if name in staged else [name.replace("-", "_")])
                ]
            for module in modules:
                report(f"Checking that the OpenCut server can import {module}...")
                smoke_import(module, staging, on_process)
            # A project someone installed for itself stays theirs when a
            # reinstall of something else replaces it as a dependency.
            for name, dist in staged.items():
                if name in store and name not in keep and _requested(packages, store[name]):
                    open(os.path.join(staging, dist.info, _REQUESTED), "a").close()
            merge_staged_packages(staging, packages)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(constraints, ignore_errors=True)
    return result


def _lane_constraints(folder: str) -> list[str]:
    """pip arguments that hold GPU builds to this machine's CUDA lane.

    ``rembg[gpu]`` brings onnxruntime-gpu in as a dependency, and pip on its
    own takes the newest one, a CUDA 13 build, even beside a CUDA 12 torch.
    A constraint bounds it without asking for it.
    """
    from opencut.gpu_lanes import onnxruntime_gpu_requirement

    requirement = onnxruntime_gpu_requirement()
    if not requirement:
        return []
    path = os.path.join(folder, "lane-constraints.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(requirement + "\n")
    return ["-c", path]


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------
_REQUESTED = "REQUESTED"  # pip writes it into the .dist-info of each project it was asked for


def _requested(packages: str, dist: Dist) -> bool:
    return os.path.isfile(os.path.join(packages, dist.info, _REQUESTED))


def _dependency_names(dist: Dist, store: dict[str, Dist], every_extra: bool) -> set[str]:
    """The store projects ``dist`` requires. ``every_extra`` counts optional ones too."""
    if not every_extra:
        names = {canonical(req.name) for req in _requirements(dist)}
    else:
        from packaging.requirements import InvalidRequirement, Requirement

        names = set()
        for line in dist.requires:
            try:
                names.add(canonical(Requirement(line).name))
            except InvalidRequirement:
                continue
    names.discard(dist.name)
    return names & store.keys()


def _closure(start: Iterable[str], store: dict[str, Dist], every_extra: bool) -> set[str]:
    seen: set[str] = set()
    todo = list(start)
    while todo:
        name = todo.pop()
        if name not in seen:
            seen.add(name)
            todo.extend(_dependency_names(store[name], store, every_extra))
    return seen


def plan_removal(
    store: dict[str, Dist], targets: Iterable[str], requested: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Which projects removing ``targets`` takes out of the store, and what stops it.

    A target goes with each dependency only the projects leaving needed. A
    project stays when something that stays requires it, counting optional
    requirements, since nothing records which extras an install asked for.
    So does one someone installed for itself (``requested``, pip's
    ``REQUESTED`` marker), and so does anything nothing requires. A target
    that something staying requires comes back as a reason instead.
    """
    targets = set(targets) & store.keys()
    requested = set(requested)
    required_by: dict[str, set[str]] = {name: set() for name in store}
    for dist in store.values():
        for dep in _dependency_names(dist, store, every_extra=True):
            required_by[dep].add(dist.name)
    roots = {
        name for name in store
        if name not in targets and (name in requested or not required_by[name])
    }
    staying = _closure(roots, store, every_extra=True)
    blocked = []
    for name in sorted(targets & staying):
        needers = sorted(store[other].display for other in required_by[name] if other in staying)
        blocked.append(f"{', '.join(needers)} still need{'s' if len(needers) == 1 else ''} {store[name].display}")
    if blocked:
        return [], blocked
    return sorted(_closure(targets, store, every_extra=False) - staying), []


def _take_out(packages: str, leaving: Sequence[Dist], staying: Sequence[Dist]) -> None:
    """Move what ``leaving`` owns out of the store, keeping files ``staying`` owns.

    Like a merge, every move is journaled before it happens and nothing is
    deleted until all of them worked, so a failure or a kill partway puts
    the store back as it was.
    """
    others: set[str] = set()
    for dist in staying:
        others |= _owned(dist)
    shared = others | _ancestors(others)
    covers = set()
    for dist in leaving:
        for rel in _owned(dist):
            if rel in others:
                continue
            parts = rel.split("/")
            # The highest folder no other project has files in goes whole.
            covers.add(next("/".join(parts[:depth]) for depth in range(1, len(parts) + 1)
                            if "/".join(parts[:depth]) not in shared))
    aside = tempfile.mkdtemp(prefix=_ASIDE_PREFIX, dir=os.path.dirname(packages))
    journal = _Journal(os.path.join(aside, _JOURNAL))
    set_aside: list[tuple[str, str]] = []
    try:
        for rel in sorted(covers):
            path = os.path.join(packages, *rel.split("/"))
            if not os.path.lexists(path):
                continue
            holder = os.path.join(aside, str(len(set_aside)))
            journal.write("aside", path, holder)
            os.replace(path, holder)
            set_aside.append((path, holder))
        journal.write("done")
    except BaseException as exc:
        restored = _put_back([], set_aside)
        journal.close()
        if restored:
            shutil.rmtree(aside, ignore_errors=True)
        if isinstance(exc, OSError):
            where = "is back" if restored else f"is in {aside}"
            raise RuntimeError(
                f"couldn't remove {', '.join(d.display for d in leaving)} from {packages} ({exc}). "
                f"Everything that was there {where}. If OpenCut is running, close it and "
                "Premiere Pro, then try again."
            ) from exc
        raise
    journal.close()
    shutil.rmtree(aside, ignore_errors=True)
    for top in sorted({rel.split("/", 1)[0] for rel in covers} & shared):
        path = os.path.join(packages, top)
        if os.path.isdir(path) and not os.path.islink(path):
            _prune_empty_dirs(path)


def remove(projects: Iterable[str], *, report: Report = _log) -> list[str]:
    """Take add-on projects out of the store. Returns the names of what it removed.

    Each project goes with the dependencies nothing else needs
    (:func:`plan_removal`). It holds the store lock, so it never runs during
    an install, and raises ``RuntimeError`` without changing anything when a
    project that stays still needs one of them.
    """
    packages = packages_dir()
    targets = {canonical(project) for project in projects if project.strip()}
    with store_lock(report):
        _sweep_leftovers(os.path.dirname(packages))
        store = read_dists(packages)
        requested = {name for name, dist in store.items() if _requested(packages, dist)}
        removing, blocked = plan_removal(store, targets, requested)
        if blocked:
            raise RuntimeError(f"Nothing was removed, because {'; '.join(blocked)}")
        if not removing:
            return []
        _take_out(
            packages,
            [store[name] for name in removing],
            [dist for name, dist in store.items() if name not in removing],
        )
    removed = [f"{store[name].display} {store[name].version}" for name in removing]
    report(f"Removed {', '.join(removed)} from {packages}")
    return removed
