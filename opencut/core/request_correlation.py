"""
Per-request correlation IDs for OpenCut.

Complements the existing ``[job_id]`` log correlation (from v2.0) with
a ``[request_id]`` tag so debugging a user-reported issue can walk the
request through *every* log line it produced — including the pre-job
validation logs that fire before a job_id exists.

Design
------
- **Thread-local + Flask-g dual storage.** Flask's ``g`` is per-request
  and dies with the request; the thread-local is what downstream log
  records (inside background workers) pick up via a filter.
- **Header echo.** The request-ID is echoed in the response ``X-Request-ID``
  header so clients can correlate error reports with server logs.
- **Propagated into job metadata.** When an ``@async_job`` route spawns
  a background worker, the request-ID is stamped on the job dict so the
  ``/status/<job_id>`` response contains the original request-ID.
- **Never trust user input.** When the incoming request carries its own
  ``X-Request-ID``, we **regenerate** and log the original as
  ``client_request_id`` — prevents log-injection via attacker-chosen
  IDs.

Usage from ``server.create_app``::

    from opencut.core.request_correlation import install_middleware
    install_middleware(_app)

After installation, every log record under the ``opencut`` logger
carries a ``request_id`` attribute (empty string when outside a
request), and every response includes an ``X-Request-ID`` header.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable, Optional

logger = logging.getLogger("opencut")


# Thread-local that worker threads can read (they don't have Flask g).
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_request_id() -> str:
    """Return the current thread's request-ID, or an empty string."""
    return getattr(_thread_local, "request_id", "") or ""


def set_request_id(value: str) -> None:
    """Set the thread-local request-ID. Trim + sanitise for log safety."""
    if not isinstance(value, str):
        value = ""
    # Strip control chars / whitespace + cap length. No dependency on
    # a regex — we're applying the rule uniformly.
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in "-_:.")[:64]
    _thread_local.request_id = cleaned


def clear_request_id() -> None:
    _thread_local.request_id = ""


def new_request_id() -> str:
    """Generate + set + return a fresh request-ID."""
    rid = f"r-{uuid.uuid4().hex[:16]}"
    set_request_id(rid)
    return rid


# ---------------------------------------------------------------------------
# Log filter
# ---------------------------------------------------------------------------

class _RequestLogFilter(logging.Filter):
    """Inject ``request_id`` attribute on every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = get_request_id()
        return True


_filter_instance = _RequestLogFilter()


def attach_log_filter(logger_obj: Optional[logging.Logger] = None) -> None:
    """Attach the request-ID filter to the OpenCut logger (idempotent)."""
    target = logger_obj if logger_obj is not None else logger
    # Avoid double-attach
    for f in target.filters:
        if isinstance(f, _RequestLogFilter):
            return
    target.addFilter(_filter_instance)


# ---------------------------------------------------------------------------
# Flask middleware
# ---------------------------------------------------------------------------

HEADER_NAME = "X-Request-ID"


def install_middleware(app) -> None:
    """Install ``before_request`` / ``after_request`` handlers on ``app``.

    - Generates a fresh request-ID on entry (never trusts client-supplied
      IDs as-is — see module docstring).
    - Attaches to ``flask.g.request_id`` **and** the thread-local.
    - Echoes on the response ``X-Request-ID`` header.
    - Clears the thread-local on teardown so thread-reuse doesn't leak
      the ID into the next request.
    """
    from flask import g, request

    attach_log_filter()

    @app.before_request
    def _assign_request_id():  # noqa: ANN202 — Flask hook
        client_id = request.headers.get(HEADER_NAME, "") or ""
        rid = new_request_id()
        try:
            g.request_id = rid
        except RuntimeError:
            # Outside request context (shouldn't happen in before_request)
            pass
        if client_id:
            # Preserve the attacker-uncontrollable form for investigation
            safe_client = "".join(
                ch for ch in client_id if ch.isalnum() or ch in "-_:."
            )[:80]
            logger.debug(
                "request received with client-supplied %s=%r (stored as client_request_id=%r)",
                HEADER_NAME, client_id, safe_client,
            )
            try:
                g.client_request_id = safe_client
            except RuntimeError:
                pass

    @app.after_request
    def _echo_request_id(resp):  # noqa: ANN202
        try:
            rid = getattr(g, "request_id", "") or get_request_id()
        except RuntimeError:
            rid = get_request_id()
        if rid:
            resp.headers[HEADER_NAME] = rid
        return resp

    @app.teardown_request
    def _teardown(_exc):  # noqa: ANN202
        clear_request_id()


def check_request_correlation_available() -> bool:
    """Always True — stdlib only."""
    return True


# ---------------------------------------------------------------------------
# Decorator (optional, for non-Flask entry points)
# ---------------------------------------------------------------------------

def with_request_id(func: Callable) -> Callable:
    """Decorate a callable so each invocation runs under a fresh request-ID.

    Intended for CLI / background entry points that don't go through
    Flask — e.g. ``opencut`` CLI commands.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rid = new_request_id()
        try:
            return func(*args, **kwargs)
        finally:
            # Only clear if we set it; preserves nesting.
            if get_request_id() == rid:
                clear_request_id()
    return wrapper
