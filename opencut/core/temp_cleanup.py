"""
Startup + periodic sweep of stale OpenCut temp files.

Closes the gap flagged in the v1.14.0 strategic-gap audit: under
concurrent load, ``tempfile.mkstemp`` / ``mkdtemp`` calls across the
93 modules that use them accumulate GB-scale intermediate files
before the deferred-cleanup worker catches up. If the server crashes
mid-job, those files leak forever.

This module runs two things:

1. **Startup sweep** — on every ``create_app()``, walk
   ``tempfile.gettempdir()`` looking for ``opencut_*`` + ``opencut-*``
   files / dirs older than a configurable TTL (default 1 hour). Remove
   them. Logs a summary.
2. **Periodic background sweep** — a daemon thread re-runs the sweep
   every N minutes (default 60) so long-running servers don't
   accumulate during normal operation. Opt-out via
   ``OPENCUT_TEMP_CLEANUP_INTERVAL=0``.

Configuration
-------------
- ``OPENCUT_TEMP_CLEANUP_TTL`` seconds: files older than this get
  deleted. Default 3600.
- ``OPENCUT_TEMP_CLEANUP_INTERVAL`` seconds: background sweep period.
  ``0`` disables the periodic sweep (startup-only). Default 3600.
- ``OPENCUT_TEMP_CLEANUP_PREFIXES`` comma-separated list of prefix
  glob-stems. Default ``"opencut_,opencut-"``.

Safety
------
- Only matches files whose basename **starts with** a known prefix —
  never deletes arbitrary user files.
- Uses ``os.path.isdir`` vs ``isfile`` to pick ``shutil.rmtree`` vs
  ``os.unlink``; symlinks are followed via ``realpath()`` check.
- Anything outside ``tempfile.gettempdir()`` after ``realpath()`` is
  skipped — defence against symlinked prefix attacks.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("opencut")


def _env_int(name: str, default: int, lo: int = 0, hi: int = 86400 * 7) -> int:
    raw = os.environ.get(name) or ""
    try:
        val = int(raw.strip() or default)
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


DEFAULT_TTL = _env_int("OPENCUT_TEMP_CLEANUP_TTL", 3600, lo=60)
DEFAULT_INTERVAL = _env_int("OPENCUT_TEMP_CLEANUP_INTERVAL", 3600, lo=0)
DEFAULT_PREFIXES = tuple(
    p.strip() for p in
    (os.environ.get("OPENCUT_TEMP_CLEANUP_PREFIXES") or "opencut_,opencut-").split(",")
    if p.strip()
)


@dataclass
class SweepReport:
    """Result of one sweep pass."""
    scanned: int = 0
    removed_files: int = 0
    removed_dirs: int = 0
    bytes_reclaimed: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "scanned": self.scanned,
            "removed_files": self.removed_files,
            "removed_dirs": self.removed_dirs,
            "bytes_reclaimed": self.bytes_reclaimed,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }


# Module-level background-thread handle — started lazily on first
# :func:`start_background_sweep` call.
_BG_THREAD: "threading.Thread | None" = None
_BG_SHUTDOWN = threading.Event()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _is_under(candidate: str, root: str) -> bool:
    """True when ``candidate`` resolves to somewhere inside ``root``."""
    try:
        cand_real = os.path.realpath(candidate)
        root_real = os.path.realpath(root)
    except OSError:
        return False
    try:
        return os.path.commonpath([cand_real, root_real]) == root_real
    except ValueError:
        return False


def _entry_bytes(path: str) -> int:
    """Total on-disk bytes for a file or dir. Best-effort — returns 0 on error."""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return total
    except OSError:
        return 0
    return 0


def sweep(
    ttl_seconds: int = DEFAULT_TTL,
    prefixes=DEFAULT_PREFIXES,
) -> SweepReport:
    """Run one sweep over ``tempfile.gettempdir()``. Synchronous."""
    report = SweepReport()
    root = tempfile.gettempdir()
    now = time.time()

    try:
        entries = os.listdir(root)
    except OSError as exc:
        report.errors.append(f"listdir({root}): {exc}")
        return report

    for name in entries:
        if not any(name.startswith(p) for p in prefixes):
            continue
        full = os.path.join(root, name)
        report.scanned += 1

        # Defend against symlink-out-of-tempdir
        if not _is_under(full, root):
            report.skipped += 1
            continue

        try:
            mtime = os.path.getmtime(full)
        except OSError as exc:
            report.errors.append(f"stat({full}): {exc}")
            continue

        if now - mtime < ttl_seconds:
            report.skipped += 1
            continue

        size = _entry_bytes(full)
        try:
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full, ignore_errors=False)
                report.removed_dirs += 1
            else:
                os.unlink(full)
                report.removed_files += 1
            report.bytes_reclaimed += size
        except OSError as exc:
            report.errors.append(f"delete({full}): {exc}")

    if report.removed_files or report.removed_dirs:
        logger.info(
            "temp_cleanup: removed %d files / %d dirs / %.1f MB from %s",
            report.removed_files, report.removed_dirs,
            report.bytes_reclaimed / (1024 * 1024), root,
        )
    return report


# ---------------------------------------------------------------------------
# Startup / background wiring
# ---------------------------------------------------------------------------

def run_startup_sweep() -> SweepReport:
    """Best-effort sweep called once from ``create_app()``."""
    try:
        return sweep()
    except Exception as exc:  # noqa: BLE001
        logger.warning("temp_cleanup startup sweep raised: %s", exc)
        return SweepReport(errors=[str(exc)])


def start_background_sweep(interval_seconds: int = DEFAULT_INTERVAL) -> bool:
    """Start (or skip) the daemon sweep thread.

    Returns ``True`` when a thread was started, ``False`` when the
    feature was disabled (``interval_seconds <= 0``) or a thread was
    already running.
    """
    global _BG_THREAD
    if interval_seconds <= 0:
        logger.debug("temp_cleanup: background sweep disabled (interval=0)")
        return False
    if _BG_THREAD is not None and _BG_THREAD.is_alive():
        return False

    _BG_SHUTDOWN.clear()

    def _loop():
        while not _BG_SHUTDOWN.wait(timeout=interval_seconds):
            try:
                sweep()
            except Exception as exc:  # noqa: BLE001
                logger.warning("temp_cleanup background sweep raised: %s", exc)

    _BG_THREAD = threading.Thread(
        target=_loop, name="opencut-temp-sweep", daemon=True,
    )
    _BG_THREAD.start()
    logger.info("temp_cleanup: background sweep every %ds", interval_seconds)
    return True


def stop_background_sweep(timeout: float = 2.0) -> None:
    """Request the background sweep to stop and join briefly."""
    global _BG_THREAD
    _BG_SHUTDOWN.set()
    if _BG_THREAD is not None:
        _BG_THREAD.join(timeout=max(0.1, float(timeout)))
        _BG_THREAD = None


def check_temp_cleanup_available() -> bool:
    """Always True — stdlib only."""
    return True
