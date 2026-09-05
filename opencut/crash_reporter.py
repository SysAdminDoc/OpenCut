"""Persist evidence when the server dies.

``~/.opencut/crash.log`` already had three readers -- the crash packet, the
issue report builder and the ``/logs/crash`` download -- and no writer. So the
one failure that most needs evidence produced none: in issue #8 the server
stopped mid-session and the log simply ended, with no traceback and no clue
whether Python raised or the process was aborted underneath it.

Two different failures have to be captured, because they leave different
traces:

* A Python exception that escapes the main thread or a worker thread. Neither
  reaches the logging handlers once the thread is unwinding, so
  ``sys.excepthook`` and ``threading.excepthook`` are the last chance.
* A native abort -- a segfault, or an extension module built for the wrong
  CPython ABI being imported. No Python-level hook runs at all; only
  ``faulthandler`` writes anything, and only if it was armed beforehand.
"""

from __future__ import annotations

import datetime
import faulthandler
import logging
import os
import sys
import threading
import traceback

logger = logging.getLogger("opencut")

OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
CRASH_LOG = os.path.join(OPENCUT_DIR, "crash.log")

#: Keep the file bounded. A crash loop must not fill the user's disk, and the
#: readers only ever take a tail anyway.
MAX_CRASH_LOG_BYTES = 2 * 1024 * 1024

_installed = False
_faulthandler_stream = None
_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _environment_block() -> str:
    """Describe where this interpreter's code can come from.

    ``sys.path`` provenance is here because the most likely cause of a native
    abort in a packaged build is a module loaded from somewhere it should not
    have been (issue #8).
    """
    frozen = bool(getattr(sys, "frozen", False))
    external = os.environ.get("OPENCUT_EXTERNAL_SITE_PACKAGES", "")
    lines = [
        f"  python: {sys.version.splitlines()[0]}",
        f"  executable: {sys.executable}",
        f"  frozen: {frozen}",
        f"  external_site_packages: {external or '(disabled)'}",
        f"  sys.path ({len(sys.path)} entries):",
    ]
    lines.extend(f"    {entry}" for entry in sys.path)
    return "\n".join(lines)


def _trim_if_oversized(path: str) -> None:
    try:
        if os.path.getsize(path) <= MAX_CRASH_LOG_BYTES:
            return
        with open(path, "rb") as handle:
            handle.seek(-MAX_CRASH_LOG_BYTES // 2, os.SEEK_END)
            tail = handle.read()
        with open(path, "wb") as handle:
            handle.write(b"[crash.log truncated]\n")
            handle.write(tail)
    except OSError:
        pass


def write_crash_record(kind: str, exc_type, exc_value, exc_tb, *, thread: str = "") -> str:
    """Append one structured record and return the path it was written to."""
    os.makedirs(OPENCUT_DIR, exist_ok=True)
    where = thread or threading.current_thread().name
    body = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
    record = (
        f"\n===== {kind} at {_utc_now()} =====\n"
        f"  thread: {where}\n"
        f"  pid: {os.getpid()}\n"
        f"{_environment_block()}\n"
        f"  traceback:\n{body}\n"
    )
    with _lock:
        try:
            with open(CRASH_LOG, "a", encoding="utf-8", errors="replace") as handle:
                handle.write(record)
            _trim_if_oversized(CRASH_LOG)
        except OSError as write_error:  # pragma: no cover - disk full / permissions
            logger.error("Could not write crash record to %s: %s", CRASH_LOG, write_error)
    return CRASH_LOG


def _excepthook(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    write_crash_record("unhandled exception", exc_type, exc_value, exc_tb)
    logger.critical("Unhandled exception; wrote crash record to %s", CRASH_LOG, exc_info=(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _thread_excepthook(args):
    if issubclass(args.exc_type, SystemExit):
        return
    thread_name = getattr(args.thread, "name", "") or "unknown"
    write_crash_record(
        "unhandled thread exception",
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
        thread=thread_name,
    )
    logger.critical(
        "Unhandled exception in thread %s; wrote crash record to %s", thread_name, CRASH_LOG
    )


def install_crash_handlers(*, enable_faulthandler: bool = True) -> str:
    """Arm crash capture. Safe to call more than once."""
    global _installed, _faulthandler_stream
    if _installed:
        return CRASH_LOG

    os.makedirs(OPENCUT_DIR, exist_ok=True)
    if enable_faulthandler:
        try:
            # Held open for the life of the process on purpose: faulthandler
            # writes from a signal handler and cannot open a file at that point.
            _faulthandler_stream = open(CRASH_LOG, "a", encoding="utf-8", errors="replace")
            _faulthandler_stream.write(f"\n===== faulthandler armed at {_utc_now()} (pid {os.getpid()}) =====\n")
            _faulthandler_stream.flush()
            faulthandler.enable(file=_faulthandler_stream, all_threads=True)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not arm faulthandler: %s", exc)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    _installed = True
    logger.debug("Crash handlers installed; records go to %s", CRASH_LOG)
    return CRASH_LOG
