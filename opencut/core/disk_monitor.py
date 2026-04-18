"""
Disk-space monitoring + preflight.

Extends the v1.3.0 ``helpers.check_disk_space`` helper into a
structured observability surface so render routes can:

- refuse to start a long encode that would fill the disk mid-job,
- surface an ops dashboard of every tracked mount point + headroom,
- emit warning / critical thresholds that ops can alert on.

Separate from ``core/temp_cleanup.py`` (which deletes stale files) —
this module *observes*, it doesn't delete.  Pair the two: the sweep
frees space, the monitor surfaces whether there's enough left.

Defaults target a single workstation install; ops running a shared
render node can point the monitor at external volumes via env.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Warn when < N MB free; refuse a preflight when < CRITICAL MB free.
DEFAULT_WARN_MB = 2048     # 2 GB
DEFAULT_CRITICAL_MB = 500  # 500 MB
DEFAULT_PREFLIGHT_MB = 500


def _env_int(name: str, default: int, lo: int = 0, hi: int = 1024 * 1024) -> int:
    raw = os.environ.get(name) or ""
    try:
        val = int(raw.strip() or default)
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


WARN_MB = _env_int("OPENCUT_DISK_WARN_MB", DEFAULT_WARN_MB)
CRITICAL_MB = _env_int("OPENCUT_DISK_CRITICAL_MB", DEFAULT_CRITICAL_MB)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MountReport:
    """Observability snapshot of one tracked mount."""
    path: str
    mount_point: str = ""
    free_bytes: int = 0
    total_bytes: int = 0
    used_bytes: int = 0
    free_mb: int = 0
    free_pct: float = 0.0
    status: str = "ok"          # ok | warn | critical | error
    note: str = ""


@dataclass
class DiskReport:
    """Full report — one :class:`MountReport` per tracked path."""
    mounts: List[MountReport] = field(default_factory=list)
    warn_mb: int = 0
    critical_mb: int = 0
    any_critical: bool = False
    any_warn: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "mounts": [
                {
                    "path": m.path,
                    "mount_point": m.mount_point,
                    "free_bytes": m.free_bytes,
                    "total_bytes": m.total_bytes,
                    "used_bytes": m.used_bytes,
                    "free_mb": m.free_mb,
                    "free_pct": round(m.free_pct, 2),
                    "status": m.status,
                    "note": m.note,
                }
                for m in self.mounts
            ],
            "warn_mb": self.warn_mb,
            "critical_mb": self.critical_mb,
            "any_critical": self.any_critical,
            "any_warn": self.any_warn,
        }


# ---------------------------------------------------------------------------
# Tracked paths
# ---------------------------------------------------------------------------

def _default_tracked_paths() -> List[str]:
    """Paths that every install should observe by default."""
    opencut_home = os.path.join(os.path.expanduser("~"), ".opencut")
    return [
        tempfile.gettempdir(),
        opencut_home,
    ]


_extra_paths_lock = threading.Lock()
_extra_tracked_paths: List[str] = []


def register_tracked_path(path: str) -> bool:
    """Add a path to the monitor's tracked set.

    Idempotent.  Returns ``True`` when the path was newly added,
    ``False`` when already tracked or invalid.
    """
    if not path or not isinstance(path, str):
        return False
    norm = os.path.abspath(path)
    with _extra_paths_lock:
        if norm in _extra_tracked_paths:
            return False
        _extra_tracked_paths.append(norm)
    logger.debug("disk_monitor: now tracking %s", norm)
    return True


def tracked_paths() -> List[str]:
    with _extra_paths_lock:
        extras = list(_extra_tracked_paths)
    # De-dup while preserving order
    out: List[str] = []
    seen = set()
    for p in list(_default_tracked_paths()) + extras:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _status_for(free_mb: int) -> Tuple[str, str]:
    if free_mb < CRITICAL_MB:
        return "critical", f"Below critical threshold ({CRITICAL_MB} MB)"
    if free_mb < WARN_MB:
        return "warn", f"Below warn threshold ({WARN_MB} MB)"
    return "ok", ""


def _probe_path(path: str) -> MountReport:
    report = MountReport(path=path)
    target = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    if not target:
        target = path
    try:
        usage = shutil.disk_usage(target)
        report.mount_point = target
        report.free_bytes = usage.free
        report.total_bytes = usage.total
        report.used_bytes = usage.used
        report.free_mb = round(usage.free / (1024 * 1024))
        if usage.total > 0:
            report.free_pct = (usage.free / usage.total) * 100.0
        status, note = _status_for(report.free_mb)
        report.status = status
        report.note = note
    except Exception as exc:  # noqa: BLE001
        report.status = "error"
        report.note = str(exc)[:200]
    return report


def report() -> DiskReport:
    """Run a one-shot probe across every tracked path."""
    mounts = [_probe_path(p) for p in tracked_paths()]
    return DiskReport(
        mounts=mounts,
        warn_mb=WARN_MB,
        critical_mb=CRITICAL_MB,
        any_critical=any(m.status == "critical" for m in mounts),
        any_warn=any(m.status in ("warn", "critical") for m in mounts),
    )


def preflight(
    path: str,
    required_mb: int = DEFAULT_PREFLIGHT_MB,
) -> Dict[str, object]:
    """Gate a long-running operation on a free-space minimum.

    Returns ``{ok, free_mb, required_mb, note}``. On ``ok=False`` the
    caller should bail out with a ``DISK_FULL`` style error.
    """
    try:
        target = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        usage = shutil.disk_usage(target)
        free_mb = round(usage.free / (1024 * 1024))
        return {
            "ok": free_mb >= required_mb,
            "free_mb": free_mb,
            "required_mb": int(required_mb),
            "note": "" if free_mb >= required_mb else (
                f"Only {free_mb} MB free on {target}; {required_mb} MB required."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        # Fail open: report ok=True when we can't probe, and note the issue
        # so callers can surface the warning without aborting the job.
        return {
            "ok": True,
            "free_mb": 0,
            "required_mb": int(required_mb),
            "note": f"Disk probe failed ({exc}); proceeding without guarantee.",
        }


# ---------------------------------------------------------------------------
# Optional background thread (off by default)
# ---------------------------------------------------------------------------

_BG_THREAD: Optional[threading.Thread] = None
_BG_SHUTDOWN = threading.Event()
_LAST_REPORT: Dict[str, object] = {"report": None, "timestamp": 0.0}


def _default_bg_interval() -> int:
    raw = os.environ.get("OPENCUT_DISK_MONITOR_INTERVAL") or ""
    try:
        val = int(raw.strip() or 0)
    except (TypeError, ValueError):
        val = 0
    return max(0, min(86400, val))


def start_background(interval_seconds: Optional[int] = None) -> bool:
    """Run ``report()`` every ``interval_seconds`` on a daemon thread.

    When ``interval_seconds`` is ``None``, reads
    ``OPENCUT_DISK_MONITOR_INTERVAL`` (default 0 = disabled).  Returns
    ``True`` on start, ``False`` when disabled or already running.
    """
    global _BG_THREAD
    if interval_seconds is None:
        interval_seconds = _default_bg_interval()
    if interval_seconds <= 0:
        return False
    if _BG_THREAD is not None and _BG_THREAD.is_alive():
        return False

    _BG_SHUTDOWN.clear()

    def _loop():
        while not _BG_SHUTDOWN.wait(timeout=interval_seconds):
            try:
                rep = report()
                _LAST_REPORT["report"] = rep
                _LAST_REPORT["timestamp"] = time.time()
                if rep.any_critical:
                    for m in rep.mounts:
                        if m.status == "critical":
                            logger.warning(
                                "disk_monitor CRITICAL on %s — %d MB free",
                                m.path, m.free_mb,
                            )
                elif rep.any_warn:
                    for m in rep.mounts:
                        if m.status == "warn":
                            logger.info(
                                "disk_monitor WARN on %s — %d MB free",
                                m.path, m.free_mb,
                            )
            except Exception as exc:  # noqa: BLE001
                logger.debug("disk_monitor background probe failed: %s", exc)

    _BG_THREAD = threading.Thread(
        target=_loop, name="opencut-disk-monitor", daemon=True,
    )
    _BG_THREAD.start()
    logger.info("disk_monitor: background probe every %ds", interval_seconds)
    return True


def stop_background(timeout: float = 2.0) -> None:
    global _BG_THREAD
    _BG_SHUTDOWN.set()
    if _BG_THREAD is not None:
        _BG_THREAD.join(timeout=max(0.1, float(timeout)))
        _BG_THREAD = None


def last_background_report() -> Dict[str, object]:
    """Snapshot of the latest background-thread probe.

    Empty when the background thread hasn't run yet.
    """
    rep = _LAST_REPORT.get("report")
    if rep is None:
        return {"report": None, "timestamp": 0.0}
    return {
        "report": rep.to_dict() if hasattr(rep, "to_dict") else rep,
        "timestamp": _LAST_REPORT["timestamp"],
    }


def check_disk_monitor_available() -> bool:
    """Always True — stdlib only."""
    return True
