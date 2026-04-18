"""
Self-hosted Plausible analytics — event emitter for OpenCut.

Fires custom events to a Plausible instance (https://plausible.io,
self-hosted: https://github.com/plausible/analytics, AGPL-3) so
operators get per-route latency / usage signal without standing up a
separate metrics stack.  Designed to coexist with Sentry (error
tracking) rather than replace it.

Two emission modes:
1. **Fire-and-forget** (default): POSTs to ``/api/event`` via
   ``urllib`` on a background thread.  Never blocks the request, never
   raises into the caller.
2. **Sync** (testing only): blocks until the event POST returns.

Configuration
-------------
- ``PLAUSIBLE_HOST`` — Plausible instance URL (e.g.
  ``https://plausible.example.com``). **Required** — missing env var
  disables telemetry entirely.
- ``PLAUSIBLE_DOMAIN`` — site/domain registered with Plausible (e.g.
  ``opencut.local``). Required.
- ``PLAUSIBLE_USER_AGENT`` — optional override. Defaults to
  ``opencut-telemetry/{version}`` so events are identifiable.

Privacy
-------
- No IP addresses, no paths with filenames, no user-identifying data.
  Event names + narrow ``props`` dicts only.  ``_scrub_props`` caps
  string lengths and rejects suspicious keys.
- Telemetry is opt-in by environment variable — fresh installs emit
  nothing until an operator sets the hostname.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("opencut")


# Small in-process queue + worker.  We intentionally use a single
# background thread (not a pool) because telemetry is low-volume —
# pinning one thread is simpler to reason about than cold-creating
# threads per event and cheaper than a ThreadPoolExecutor.
_EVENT_LOCK = threading.Lock()
_WORKER_THREAD: Optional[threading.Thread] = None
_SHUTDOWN = threading.Event()
_EVENT_QUEUE: "list[Dict[str, Any]]" = []
_QUEUE_COND = threading.Condition(_EVENT_LOCK)
_MAX_QUEUED = 500   # drop older events if the server is unreachable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlausibleConfig:
    """Resolved Plausible configuration — all fields derive from env."""
    host: str
    domain: str
    user_agent: str
    enabled: bool


def get_config() -> PlausibleConfig:
    host = (os.environ.get("PLAUSIBLE_HOST") or "").strip().rstrip("/")
    domain = (os.environ.get("PLAUSIBLE_DOMAIN") or "").strip()
    try:
        from opencut import __version__ as _ver
    except Exception:  # noqa: BLE001
        _ver = "unknown"
    ua = os.environ.get("PLAUSIBLE_USER_AGENT") or f"opencut-telemetry/{_ver}"
    enabled = bool(host and domain)
    return PlausibleConfig(host=host, domain=domain, user_agent=ua, enabled=enabled)


def check_plausible_available() -> bool:
    """True when env is configured AND the host looks well-formed."""
    cfg = get_config()
    if not cfg.enabled:
        return False
    return cfg.host.startswith(("http://", "https://"))


# ---------------------------------------------------------------------------
# Payload sanitisation
# ---------------------------------------------------------------------------

_MAX_PROP_VALUE_LEN = 120
_MAX_PROP_COUNT = 30


def _scrub_props(props: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Reduce free-form props to Plausible-safe strings.

    Drops keys starting with ``_``, caps length, truncates values.
    Always returns a dict (possibly empty) — never ``None``.
    """
    out: Dict[str, str] = {}
    if not isinstance(props, dict):
        return out
    i = 0
    for k, v in props.items():
        if i >= _MAX_PROP_COUNT:
            break
        if not isinstance(k, str) or not k or k.startswith("_"):
            continue
        if not all(ch.isalnum() or ch in "-_" for ch in k):
            continue
        if isinstance(v, bool):
            val = "true" if v else "false"
        elif v is None:
            val = ""
        elif isinstance(v, (int, float)):
            val = f"{v:.6g}" if isinstance(v, float) else str(v)
        elif isinstance(v, str):
            val = v
        else:
            val = str(v)
        out[k[:40]] = val[:_MAX_PROP_VALUE_LEN]
        i += 1
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _ensure_worker_started() -> None:
    global _WORKER_THREAD
    if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
        return
    _WORKER_THREAD = threading.Thread(
        target=_worker_loop, name="plausible-telemetry", daemon=True,
    )
    _WORKER_THREAD.start()


def _worker_loop() -> None:
    while not _SHUTDOWN.is_set():
        with _QUEUE_COND:
            while not _EVENT_QUEUE and not _SHUTDOWN.is_set():
                _QUEUE_COND.wait(timeout=1.0)
            if _SHUTDOWN.is_set():
                return
            event = _EVENT_QUEUE.pop(0)
        try:
            _post_event(event)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Plausible post failed: %s", exc)


def _post_event(event: Dict[str, Any]) -> None:
    cfg = get_config()
    if not cfg.enabled:
        return
    url = f"{cfg.host}/api/event"
    body = json.dumps(event).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": cfg.user_agent,
        },
    )
    # Plausible's API is fire-and-forget — we just need the request
    # to land and return 2xx.
    try:
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            _ = resp.read(16)
    except urllib.error.HTTPError as exc:
        logger.debug("Plausible %s returned HTTP %s", cfg.host, exc.code)
    except urllib.error.URLError as exc:
        logger.debug("Plausible %s unreachable: %s", cfg.host, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track(
    event_name: str,
    props: Optional[Dict[str, Any]] = None,
    *,
    url_path: str = "/event",
    sync: bool = False,
) -> bool:
    """Emit a custom Plausible event.

    Args:
        event_name: Plausible goal / event name (e.g. ``"job.complete"``).
            Short, alphanumeric / dash / underscore; we don't enforce a
            regex — Plausible itself silently normalises.
        props: Optional key-value dict attached as Plausible custom
            props. Filtered + truncated for safety; keys starting with
            ``_`` are dropped.
        url_path: The ``url`` field Plausible requires — defaults to
            ``/event`` so events aren't tied to a specific route.
        sync: When ``True``, blocks until the POST returns. Intended
            for tests — production code should always leave this False.

    Returns:
        ``True`` when the event was queued / posted, ``False`` when
        telemetry is disabled (no env var set).
    """
    cfg = get_config()
    if not cfg.enabled:
        return False

    payload: Dict[str, Any] = {
        "name": str(event_name)[:80],
        "url": f"https://{cfg.domain}{url_path}",
        "domain": cfg.domain,
    }
    scrubbed = _scrub_props(props)
    if scrubbed:
        payload["props"] = scrubbed

    if sync:
        try:
            _post_event(payload)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Plausible sync post failed: %s", exc)
            return False

    with _QUEUE_COND:
        if len(_EVENT_QUEUE) >= _MAX_QUEUED:
            # Drop the oldest event to make room
            _EVENT_QUEUE.pop(0)
        _EVENT_QUEUE.append(payload)
        _QUEUE_COND.notify()
    _ensure_worker_started()
    return True


def queue_depth() -> int:
    """Return the number of events currently queued (test / ops use)."""
    with _EVENT_LOCK:
        return len(_EVENT_QUEUE)


def shutdown(timeout: float = 2.0) -> None:
    """Stop the worker thread and drop any queued events.

    Intended for clean process exit — after this call, further
    :func:`track` calls are no-ops (the worker thread has stopped; new
    events stay queued in memory until the process exits). Use
    sparingly; normal daemon-thread behaviour is fine for most
    deployments.
    """
    global _WORKER_THREAD
    _SHUTDOWN.set()
    with _QUEUE_COND:
        _QUEUE_COND.notify_all()
    if _WORKER_THREAD is not None:
        _WORKER_THREAD.join(timeout=max(0.1, timeout))
        _WORKER_THREAD = None
    _SHUTDOWN.clear()
