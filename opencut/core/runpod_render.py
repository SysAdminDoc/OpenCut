"""
RunPod serverless render dispatch.

Submits a render job to RunPod's serverless endpoint API
(https://docs.runpod.io/serverless/endpoints/overview) — the simplest
way to burst GPU capacity when the local machine is saturated.
RunPod's client library ``runpod`` is optional; when absent, we fall
back to direct HTTPS via ``urllib`` so the feature works in minimal
installs.

Scope
-----
- This module **submits** jobs and **polls** for status.  It does NOT
  build the serverless container — that's a one-time RunPod setup
  the user does in their account dashboard.
- The caller specifies the RunPod endpoint ID (received from the
  RunPod dashboard), a payload dict that matches the user's container
  schema, and an API key.  We don't prescribe the container; any
  RunPod-compatible image works.
- Outputs are treated as opaque job-specific blobs — we surface the
  container's JSON response verbatim so callers can extract
  ``output_url`` / ``output_path`` / whatever their handler emits.

Security
--------
- ``api_key`` is read from the function argument or from the
  ``RUNPOD_API_KEY`` env var.  **Never** logged.
- The endpoint ID is validated against a simple regex so a typo
  can't become an SSRF vector.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

RUNPOD_API_BASE = "https://api.runpod.ai/v2"

# RunPod endpoint IDs are URL-safe strings; cap length + charset to
# prevent URL injection.
_ENDPOINT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

# Terminal states the polling loop can exit on.
_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})


@dataclass
class RunPodResult:
    """Structured return from a RunPod submission."""
    job_id: str = ""
    endpoint_id: str = ""
    status: str = "PENDING"
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    delay_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_runpod_available() -> bool:
    """Always True — the transport is ``urllib`` stdlib.

    The ``runpod`` pip package is an optional enhancement (nicer
    ergonomics, richer type hints) but not required.
    """
    return True


def check_runpod_sdk_installed() -> bool:
    try:
        import runpod  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_api_key(explicit: Optional[str]) -> str:
    key = (explicit or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY missing — pass api_key=... or set the "
            "RUNPOD_API_KEY env var. Get a key at "
            "https://runpod.io/console/user/settings."
        )
    return key


def _validate_endpoint_id(endpoint_id: str) -> str:
    if not _ENDPOINT_ID_RE.match(endpoint_id or ""):
        raise ValueError(
            f"Invalid RunPod endpoint_id {endpoint_id!r}: must match "
            f"{_ENDPOINT_ID_RE.pattern}"
        )
    return endpoint_id


# ---------------------------------------------------------------------------
# HTTP layer (urllib fallback)
# ---------------------------------------------------------------------------

def _http_request(
    url: str, api_key: str, payload: Optional[Dict] = None,
    method: str = "POST", timeout: float = 60.0,
) -> Dict[str, Any]:
    data_bytes: Optional[bytes] = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data_bytes, method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "opencut-runpod/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"RunPod HTTP {exc.code}: {err_body[:400]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"RunPod network error: {exc}") from exc

    try:
        return json.loads(body) if body else {}
    except ValueError as exc:
        raise RuntimeError(f"RunPod non-JSON response: {body[:200]!r}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit(
    endpoint_id: str,
    payload: Dict[str, Any],
    api_key: Optional[str] = None,
    sync: bool = False,
    sync_timeout: float = 300.0,
) -> RunPodResult:
    """Submit a job to a RunPod serverless endpoint.

    Args:
        endpoint_id: RunPod serverless endpoint ID.
        payload: Container-specific request body. Wrapped under
            ``{"input": payload}`` to match RunPod's schema.
        api_key: Optional explicit API key. Falls back to
            ``RUNPOD_API_KEY`` env var.
        sync: When ``True``, uses RunPod's ``/runsync`` endpoint
            (blocks up to ``sync_timeout`` seconds). Otherwise uses
            ``/run`` and returns immediately with a job_id to poll.
        sync_timeout: Max seconds to wait when ``sync=True``.

    Returns:
        :class:`RunPodResult` — ``status="IN_PROGRESS"`` for async
        submissions, a terminal state for ``sync=True``.
    """
    endpoint_id = _validate_endpoint_id(endpoint_id)
    key = _resolve_api_key(api_key)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON-serialisable dict")

    path = "runsync" if sync else "run"
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/{path}"
    body = {"input": payload}
    if sync:
        # RunPod's runsync endpoint accepts `execution_timeout` (seconds)
        body["policy"] = {"executionTimeout": int(max(10, sync_timeout))}

    started = time.monotonic()
    resp = _http_request(url, key, body, method="POST", timeout=max(30.0, sync_timeout + 10.0))
    elapsed = time.monotonic() - started

    status_name = str(resp.get("status") or "IN_QUEUE")
    return RunPodResult(
        job_id=str(resp.get("id") or ""),
        endpoint_id=endpoint_id,
        status=status_name,
        output=resp.get("output"),
        error=resp.get("error"),
        duration_seconds=round(elapsed, 3),
        delay_seconds=float(resp.get("delayTime") or 0.0),
        notes=[f"endpoint={endpoint_id}", f"sync={sync}"],
    )


def status_of(
    endpoint_id: str,
    job_id: str,
    api_key: Optional[str] = None,
) -> RunPodResult:
    """Poll a previously-submitted async job for status."""
    endpoint_id = _validate_endpoint_id(endpoint_id)
    if not job_id or not isinstance(job_id, str):
        raise ValueError("job_id is required")
    key = _resolve_api_key(api_key)
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/status/{job_id}"
    resp = _http_request(url, key, method="GET", timeout=30.0)
    return RunPodResult(
        job_id=job_id,
        endpoint_id=endpoint_id,
        status=str(resp.get("status") or "UNKNOWN"),
        output=resp.get("output"),
        error=resp.get("error"),
        duration_seconds=float(resp.get("executionTime") or 0.0) / 1000.0,
        delay_seconds=float(resp.get("delayTime") or 0.0) / 1000.0,
    )


def wait(
    endpoint_id: str,
    job_id: str,
    api_key: Optional[str] = None,
    poll_interval: float = 5.0,
    max_wait: float = 1800.0,
    on_progress: Optional[Callable] = None,
) -> RunPodResult:
    """Poll an async job until it hits a terminal state.

    Uses exponential-ish backoff (interval ×1.25 every 3 polls, capped
    at 30 s) to avoid hammering RunPod's status endpoint on long jobs.
    """
    key = _resolve_api_key(api_key)
    endpoint_id = _validate_endpoint_id(endpoint_id)
    started = time.monotonic()
    interval = max(1.0, float(poll_interval))
    deadline = started + max(30.0, float(max_wait))
    poll_count = 0

    while True:
        res = status_of(endpoint_id, job_id, api_key=key)
        if on_progress:
            try:
                on_progress(0, f"RunPod {job_id[:8]}… status={res.status}")
            except Exception:  # noqa: BLE001
                pass
        if res.status in _TERMINAL_STATES:
            return res
        if time.monotonic() >= deadline:
            res.notes.append(f"wait timeout after {int(max_wait)}s")
            return res
        time.sleep(interval)
        poll_count += 1
        if poll_count % 3 == 0:
            interval = min(30.0, interval * 1.25)


def cancel(
    endpoint_id: str,
    job_id: str,
    api_key: Optional[str] = None,
) -> bool:
    """Cancel a RunPod job. Returns ``True`` when the cancel call
    returned OK (job was pending / running), ``False`` when the job
    was already terminal or not found."""
    endpoint_id = _validate_endpoint_id(endpoint_id)
    if not job_id or not isinstance(job_id, str):
        raise ValueError("job_id is required")
    key = _resolve_api_key(api_key)
    url = f"{RUNPOD_API_BASE}/{endpoint_id}/cancel/{job_id}"
    try:
        resp = _http_request(url, key, method="POST", timeout=20.0)
        return str(resp.get("status") or "").upper() == "CANCELLED"
    except RuntimeError as exc:
        logger.info("RunPod cancel for %s: %s", job_id, exc)
        return False
