"""
GPU-exclusive semaphore + decorator.

The long-standing Wave 3A P0 from the original roadmap: prevent
concurrent GPU model loads from OOM-ing consumer cards.  This is the
*minimal* version — a process-wide semaphore that serialises
GPU-heavy work to a small concurrent count — deliberately not the
full "multi-worker-pool" architecture described in ROADMAP.md.

Why a semaphore, not a worker pool?
-----------------------------------
- Correctness bang-for-buck.  A semaphore stops the OOM storm that
  happens when two concurrent routes both call
  ``torch.cuda.set_device`` + load a 3 GB model. That accounts for
  most of the pain the Wave 3A bullet called out.
- Upgrade path.  When we need real process isolation, the decorator
  boundary stays the same — only the implementation behind it moves
  from "acquire → run in-process" to "acquire → dispatch to worker".

Configuration
-------------
- ``OPENCUT_MAX_GPU_JOBS`` env var sets the concurrent limit.
  Defaults to **3** — two live models + one warming / freeing.
- ``OPENCUT_GPU_ACQUIRE_TIMEOUT`` seconds a request waits for a slot
  before returning 429 ``GPU_BUSY``. Defaults to 0 (non-blocking:
  immediate 429).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

logger = logging.getLogger("opencut")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name) or ""
    try:
        val = int(raw.strip() or default)
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


MAX_CONCURRENT_GPU_JOBS = _env_int("OPENCUT_MAX_GPU_JOBS", 3, 1, 32)
ACQUIRE_TIMEOUT = max(0.0, float(os.environ.get("OPENCUT_GPU_ACQUIRE_TIMEOUT") or "0") or 0.0)

_semaphore = threading.Semaphore(MAX_CONCURRENT_GPU_JOBS)
_state_lock = threading.Lock()
_active_count = 0
_rejected_count = 0
_total_acquires = 0


@dataclass
class GpuSemaphoreStatus:
    """Observable state of the GPU semaphore."""
    max_concurrent: int
    active: int
    available: int
    rejected_total: int
    acquired_total: int
    acquire_timeout_seconds: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_concurrent": self.max_concurrent,
            "active": self.active,
            "available": self.available,
            "rejected_total": self.rejected_total,
            "acquired_total": self.acquired_total,
            "acquire_timeout_seconds": self.acquire_timeout_seconds,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def status() -> GpuSemaphoreStatus:
    """Return a snapshot of the semaphore state (safe to call often)."""
    with _state_lock:
        return GpuSemaphoreStatus(
            max_concurrent=MAX_CONCURRENT_GPU_JOBS,
            active=_active_count,
            available=max(0, MAX_CONCURRENT_GPU_JOBS - _active_count),
            rejected_total=_rejected_count,
            acquired_total=_total_acquires,
            acquire_timeout_seconds=ACQUIRE_TIMEOUT,
        )


def acquire(timeout: Optional[float] = None) -> bool:
    """Try to acquire a GPU slot.  Returns ``True`` on success.

    A ``timeout`` of 0 / None uses the configured default
    (``OPENCUT_GPU_ACQUIRE_TIMEOUT``). Callers that succeed **must**
    eventually call :func:`release` — wrap in try/finally or use the
    :func:`gpu_exclusive` decorator instead.
    """
    global _rejected_count, _active_count, _total_acquires
    wait = ACQUIRE_TIMEOUT if timeout is None else max(0.0, float(timeout))
    got = _semaphore.acquire(blocking=bool(wait), timeout=wait if wait else None)
    with _state_lock:
        if got:
            _active_count += 1
            _total_acquires += 1
        else:
            _rejected_count += 1
    return bool(got)


def release() -> None:
    """Release a previously-acquired GPU slot."""
    global _active_count
    with _state_lock:
        if _active_count > 0:
            _active_count -= 1
    try:
        _semaphore.release()
    except ValueError:
        # Semaphore over-released — should never happen with the
        # try/finally discipline enforced by gpu_exclusive.
        logger.warning("GPU semaphore over-released; state may be corrupt")


# ---------------------------------------------------------------------------
# Decorator — use on the @async_job inner body, not the Flask route
# ---------------------------------------------------------------------------

def gpu_exclusive(
    _func: Optional[Callable] = None,
    *,
    timeout: Optional[float] = None,
    fail_fast_raises: bool = True,
):
    """Decorator that wraps a callable in the GPU semaphore.

    Behaviour:
    - Attempts to acquire a slot using the configured timeout.
    - On success, runs the wrapped function, releases in ``finally``.
    - On failure (``fail_fast_raises=True``, default), raises
      ``RuntimeError("GPU_BUSY: …")`` so the surrounding
      ``@async_job`` reports a clean error to the job record. When
      ``fail_fast_raises=False``, returns a structured dict
      ``{"error": "GPU_BUSY", …}`` so callers can handle the rejection
      inline.

    Use on the **inner worker body** of an async route, not on the
    Flask route itself — that way the 429 budget stays aligned with
    the queue rather than with the raw HTTP request.

    Example::

        @wave_x_bp.route("/video/ai/do-thing", methods=["POST"])
        @require_csrf
        @async_job("do_thing")
        def route_do_thing(job_id, filepath, data):
            return _run_with_gpu(job_id, filepath, data)

        @gpu_exclusive
        def _run_with_gpu(job_id, filepath, data):
            # Model load + inference here...
            ...
    """
    def decorator(func: Callable):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not acquire(timeout=timeout):
                msg = (
                    f"GPU_BUSY: {MAX_CONCURRENT_GPU_JOBS} concurrent GPU "
                    "jobs already running. Retry shortly."
                )
                if fail_fast_raises:
                    raise RuntimeError(msg)
                return {"error": "GPU_BUSY", "message": msg}
            try:
                return func(*args, **kwargs)
            finally:
                release()
        return wrapper

    # Allow both @gpu_exclusive and @gpu_exclusive(timeout=X)
    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


# ---------------------------------------------------------------------------
# Helpers for test / ops
# ---------------------------------------------------------------------------

def wait_until_idle(timeout: float = 30.0, poll: float = 0.1) -> bool:
    """Block until the semaphore has zero active jobs (or timeout)."""
    deadline = time.monotonic() + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        with _state_lock:
            if _active_count == 0:
                return True
        time.sleep(poll)
    return False


def reset_counters() -> None:
    """Zero the observability counters. Test-use only."""
    global _rejected_count, _total_acquires
    with _state_lock:
        _rejected_count = 0
        _total_acquires = 0
