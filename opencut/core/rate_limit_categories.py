"""
Category-scoped rate limits.

Closes the v1.14.0 gap: only 4 % of async routes are rate-limited.
Rather than ask every route author to pick a concurrency budget,
this module defines **four categories** — ``gpu_heavy``,
``cpu_heavy``, ``io_bound``, ``light`` — each with a tunable
semaphore.  Routes opt in via decorator and inherit the category's
budget.

The category ceilings are intentionally cautious; the ``OPENCUT_*``
env vars let operators relax them per deployment.

Interaction with ``core/gpu_semaphore.py``
------------------------------------------
- ``@gpu_exclusive`` (in ``gpu_semaphore``) is the **hard** guard for
  CUDA / MPS work. Use it on every model-loading path.
- ``@rate_limit_category("gpu_heavy")`` is the **soft** budget —
  cover the *route* so HTTP requests aren't queued into the thread
  pool faster than the GPU can drain them.
- You can stack both: the decorator ordering that makes sense is
  ``@async_job`` → ``@rate_limit_category(...)`` around the route
  body, with ``@gpu_exclusive`` on the inner worker.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, Optional

logger = logging.getLogger("opencut")

from flask import jsonify  # noqa: E402 — side-effect-free


def _env_int(name: str, default: int, lo: int = 1, hi: int = 256) -> int:
    raw = os.environ.get(name) or ""
    try:
        val = int(raw.strip() or default)
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))


# Sensible defaults for a workstation-class host. Operators tune per
# deployment via env.
CATEGORY_DEFAULTS = {
    "gpu_heavy": _env_int("OPENCUT_RL_GPU_HEAVY", 3),
    "cpu_heavy": _env_int("OPENCUT_RL_CPU_HEAVY", 4),
    "io_bound": _env_int("OPENCUT_RL_IO_BOUND", 12),
    "light": _env_int("OPENCUT_RL_LIGHT", 40),
}

CATEGORIES = tuple(CATEGORY_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Semaphore registry
# ---------------------------------------------------------------------------

@dataclass
class CategoryState:
    """Observable state of one category."""
    name: str
    max_concurrent: int
    active: int
    available: int
    rejected_total: int
    acquired_total: int


_state_lock = threading.Lock()
_semaphores: Dict[str, threading.Semaphore] = {
    cat: threading.Semaphore(limit) for cat, limit in CATEGORY_DEFAULTS.items()
}
_counters: Dict[str, Dict[str, int]] = {
    cat: {"active": 0, "rejected": 0, "acquired": 0}
    for cat in CATEGORY_DEFAULTS
}


def _ensure_category(name: str) -> None:
    if name not in _semaphores:
        raise ValueError(
            f"Unknown rate-limit category {name!r}. Valid: {list(CATEGORIES)}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acquire(category: str, timeout: float = 0.0) -> bool:
    """Attempt to take a slot in ``category``. Returns ``True`` on success.

    ``timeout=0`` is non-blocking. On success the caller **must** call
    :func:`release` in ``finally``.
    """
    _ensure_category(category)
    sem = _semaphores[category]
    got = sem.acquire(blocking=bool(timeout), timeout=timeout if timeout else None)
    with _state_lock:
        if got:
            _counters[category]["active"] += 1
            _counters[category]["acquired"] += 1
        else:
            _counters[category]["rejected"] += 1
    return bool(got)


def release(category: str) -> None:
    _ensure_category(category)
    with _state_lock:
        if _counters[category]["active"] > 0:
            _counters[category]["active"] -= 1
    try:
        _semaphores[category].release()
    except ValueError:
        logger.warning(
            "Rate-limit category %s over-released; ignoring", category,
        )


def status() -> Dict[str, CategoryState]:
    """Snapshot current state of all categories."""
    out: Dict[str, CategoryState] = {}
    with _state_lock:
        for cat in CATEGORIES:
            limit = CATEGORY_DEFAULTS[cat]
            active = _counters[cat]["active"]
            out[cat] = CategoryState(
                name=cat,
                max_concurrent=limit,
                active=active,
                available=max(0, limit - active),
                rejected_total=_counters[cat]["rejected"],
                acquired_total=_counters[cat]["acquired"],
            )
    return out


def reset_counters() -> None:
    """Zero the rejected / acquired counters.  Test-use only."""
    with _state_lock:
        for cat in CATEGORIES:
            _counters[cat]["rejected"] = 0
            _counters[cat]["acquired"] = 0


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def rate_limit_category(
    category: str,
    *,
    timeout: float = 0.0,
    return_429: bool = True,
):
    """Decorator that serialises the wrapped callable via the named category.

    ``return_429=True`` (default) makes the decorator return a Flask
    JSON 429 response when a slot can't be acquired — suitable for
    Flask route handlers. Set ``return_429=False`` for non-Flask
    callables — the decorator then raises ``RuntimeError`` so the
    outer ``@async_job`` surfaces the rejection as a structured job
    error.
    """
    _ensure_category(category)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not acquire(category, timeout=timeout):
                limit = CATEGORY_DEFAULTS[category]
                msg = (
                    f"Rate limited: {limit} concurrent {category} "
                    "operations already in flight. Retry shortly."
                )
                if return_429:
                    return jsonify({
                        "error": msg,
                        "code": "RATE_LIMITED",
                        "category": category,
                        "retry_after": 5,
                    }), 429
                raise RuntimeError(msg)
            try:
                return func(*args, **kwargs)
            finally:
                release(category)
        # Stash the category on the wrapper so introspection can find it
        wrapper.__opencut_rate_category__ = category  # type: ignore[attr-defined]
        return wrapper
    return decorator


def category_of(func: Optional[Callable]) -> str:
    """Return the rate-limit category attached to ``func`` (if any)."""
    if func is None:
        return ""
    cur = func
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        cat = getattr(cur, "__opencut_rate_category__", None)
        if cat:
            return cat
        cur = getattr(cur, "__wrapped__", None)
    return ""
