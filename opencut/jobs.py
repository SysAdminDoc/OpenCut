"""
OpenCut Job System

Shared job tracking state and helper functions used across all route modules.
"""

import atexit
import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple, Union

from opencut.config import OpenCutConfig

logger = logging.getLogger("opencut")

# Bounded thread pool for background I/O (persistence, time recording)
# Prevents unbounded thread creation under batch load
_io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oc-job-io")
atexit.register(_io_pool.shutdown, wait=True)

# ---------------------------------------------------------------------------
# Thread-local job ID for log correlation
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def get_current_job_id():
    """Return the job_id running on the current thread, or empty string."""
    return getattr(_thread_local, "job_id", "")


def should_skip_install_in_testing(job_type: str = "install") -> bool:
    """Return True when Flask tests should not perform real package installs."""
    if "install" not in str(job_type).lower():
        return False
    try:
        from flask import current_app

        return bool(
            current_app.config.get("TESTING")
            and not current_app.config.get("OPENCUT_RUN_INSTALLS_IN_TESTS", False)
        )
    except RuntimeError:
        return False


def testing_install_response(component_name: str):
    """Return a side-effect-free install response for Flask TESTING mode."""
    from flask import jsonify

    component = str(component_name or "install")
    return jsonify({
        "job_id": f"test-install-{component}",
        "component": component,
        "testing": True,
        "message": "Install skipped in TESTING mode.",
    })


# ---------------------------------------------------------------------------
# Job State (shared across all Blueprints)
# ---------------------------------------------------------------------------
jobs = {}
job_lock = threading.Lock()
_job_processes = {}  # job_id -> Popen, for subprocess kill support
# These mirror OpenCutConfig defaults (the single source of truth for
# documentation and test overrides).  They are kept as module-level constants
# because jobs.py operates outside Flask app context.
_JOB_CONFIG = OpenCutConfig.from_env()
JOB_MAX_AGE = _JOB_CONFIG.job_max_age
MAX_CONCURRENT_JOBS = _JOB_CONFIG.max_concurrent_jobs
MAX_BATCH_FILES = _JOB_CONFIG.max_batch_files
MAX_PERSISTED_JOB_PAYLOAD_BYTES = 64 * 1024
_MAX_PERSISTED_PAYLOAD_ITEMS = 200
_MAX_PERSISTED_PAYLOAD_DEPTH = 8
_SENSITIVE_PAYLOAD_VALUE = "[REDACTED]"
_SENSITIVE_PAYLOAD_KEY_RE = re.compile(
    r"(^|[_\-\s.])("
    r"api[_\-\s]?key|authorization|auth[_\-\s]?token|bearer|credential|"
    r"client[_\-\s]?secret|secret|password|passwd|passphrase|"
    r"access[_\-\s]?token|refresh[_\-\s]?token|id[_\-\s]?token|token|"
    r"private[_\-\s]?key|webhook[_\-\s]?secret|signing[_\-\s]?key"
    r")([_\-\s.]|$)",
    re.IGNORECASE,
)
_TERMINAL_JOB_STATUSES = frozenset({"complete", "error", "cancelled", "interrupted"})
_JOB_RESOURCE_FIELDS = ("peak_vram_mb", "peak_cpu_pct", "peak_rss_mb")


class JobRegistry:
    """Encapsulates the three module-level job-state globals.

    Exists so tests (and future multi-app setups) can create isolated
    instances rather than patching module globals.  The module-level
    ``jobs``, ``job_lock``, and ``_job_processes`` names remain as
    backward-compatible aliases pointing to the default registry.

    Usage (isolated test fixture)::

        registry = JobRegistry()
        monkeypatch.setattr("opencut.jobs.jobs", registry.jobs)
        monkeypatch.setattr("opencut.jobs.job_lock", registry.lock)
        monkeypatch.setattr("opencut.jobs._job_processes", registry.processes)
    """

    def __init__(self):
        self.jobs: dict = {}
        self.lock: threading.Lock = threading.Lock()
        self.processes: dict = {}  # job_id -> Popen

    def reset(self):
        """Clear all state; useful for test teardown."""
        with self.lock:
            self.jobs.clear()
            self.processes.clear()


# Default instance; module-level aliases point here for backward compat.
_default_registry = JobRegistry()
# NOTE: jobs / job_lock / _job_processes are intentionally NOT delegated to
# _default_registry because 80+ route files already hold direct references to
# these names.  Changing them would require a mass-import update.  Use
# JobRegistry for new code that needs isolation.


class TooManyJobsError(RuntimeError):
    """Raised when MAX_CONCURRENT_JOBS is reached."""
    pass


def apply_config(config: OpenCutConfig) -> None:
    """Apply runtime job limits from an OpenCutConfig instance.

    Acquires ``job_lock`` so concurrent readers of the globals see
    consistent values.
    """
    global JOB_MAX_AGE, MAX_CONCURRENT_JOBS, MAX_BATCH_FILES, _JOB_STUCK_TIMEOUT
    with job_lock:
        JOB_MAX_AGE = int(config.job_max_age)
        MAX_CONCURRENT_JOBS = int(config.max_concurrent_jobs)
        MAX_BATCH_FILES = int(config.max_batch_files)
        _JOB_STUCK_TIMEOUT = int(config.job_stuck_timeout)


def _classify_exit_reason(status: str, *, error: str = "", code: str = "", exc=None) -> str:
    try:
        from opencut.core.job_diagnostics import classify_exit_reason

        return classify_exit_reason(status, error=error, code=code, exc=exc)
    except Exception:  # noqa: BLE001 - terminal updates must not fail
        status_key = str(status or "").lower()
        if status_key in ("complete", "cancelled", "interrupted"):
            return status_key
        if status_key == "error":
            return "error"
        return ""


def _stop_resource_sampler(sampler) -> dict:
    if sampler is None:
        return {}
    try:
        snapshot = sampler.stop()
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail jobs
        logger.debug("Failed to stop resource sampler: %s", exc)
        return {}
    if not isinstance(snapshot, dict):
        return {}
    return {field: snapshot.get(field) for field in _JOB_RESOURCE_FIELDS}


def _new_job(job_type: str, filepath: str, *,
             resumable: bool = False,
             partial_output_path: str = "",
             resume_source_job_id: str = "",
             resume_attempt: int = 0) -> str:
    """Create a new job entry and return its ID.

    Raises ``TooManyJobsError`` (subclass of ``RuntimeError``) if the
    concurrent-job limit is reached.
    """
    job_id = uuid.uuid4().hex[:12]
    with job_lock:
        running = sum(1 for j in jobs.values() if j.get("status") == "running")
        if running >= MAX_CONCURRENT_JOBS:
            raise TooManyJobsError("Too many concurrent jobs. Please wait for existing jobs to finish.")
        jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "filepath": filepath,
            "status": "running",
            "progress": 0,
            "message": "Starting...",
            "result": None,
            "error": None,
            "created": time.time(),
            "resumable": bool(resumable),
            "partial_output_path": str(partial_output_path or ""),
            "resume_source_job_id": str(resume_source_job_id or ""),
            "resume_attempt": _coerce_nonnegative_int(resume_attempt, 0),
            "peak_vram_mb": None,
            "peak_cpu_pct": None,
            "peak_rss_mb": None,
            "exit_reason": "",
            "request_id": "",
            "client_request_id": "",
            "_thread": None,
        }
    _start_periodic_cleanup()
    return job_id


def _get_job_copy(job_id: str):
    """Return a shallow copy of the job dict, or None if not found.

    The copy is safe to read outside the lock without race conditions.
    Private keys (prefixed ``_``) are excluded.
    """
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            return None
        return {k: v for k, v in job.items() if not k.startswith("_")}


def _list_jobs_copy() -> list:
    """Return a list of shallow-copied job dicts (no private keys)."""
    with job_lock:
        return [
            {k: v for k, v in j.items() if not k.startswith("_")}
            for j in jobs.values()
        ]


def _update_job(job_id: str, **kwargs):
    """Update job fields. Records timing data on completion."""
    job_copy_to_persist = None
    status_update = kwargs.get("status")
    if status_update in _TERMINAL_JOB_STATUSES and not kwargs.get("exit_reason"):
        kwargs["exit_reason"] = _classify_exit_reason(
            status_update,
            error=kwargs.get("error", ""),
            code=kwargs.get("code", ""),
        )
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
            job = jobs[job_id]
            if status_update in _TERMINAL_JOB_STATUSES and not job.get("completed_at"):
                job["completed_at"] = time.time()
            # Record timing data when a job completes for future estimates.
            # Prefer ``started_at`` over ``created`` so the recorded elapsed
            # excludes queue wait time — otherwise a job that waited 30s in
            # the priority queue would inflate the historical ratio used by
            # compute_estimate() and mislead future estimates.
            if status_update == "complete":
                now = time.time()
                start = job.get("started_at") or job.get("created", now)
                elapsed = max(0.0, now - start)
                _schedule_record_time(job.get("type", ""), elapsed, job.get("filepath", ""))
            resource_update = any(field in kwargs for field in _JOB_RESOURCE_FIELDS)
            terminal_update = status_update in _TERMINAL_JOB_STATUSES
            existing_terminal = job.get("status") in _TERMINAL_JOB_STATUSES
            # Persist terminal job states to SQLite; if a cancellation raced
            # ahead of the worker's sampler shutdown, persist the late peaks.
            if terminal_update or (existing_terminal and resource_update):
                job_copy_to_persist = job.copy()
                _job_processes.pop(job_id, None)
    if job_copy_to_persist is not None:
        _persist_job(job_copy_to_persist)
        # Fire webhook event for terminal job states (best-effort).
        # Wrapped in try/except so a webhook failure never blocks job
        # finalisation. Dispatched asynchronously via _io_pool.
        try:
            _emit_job_webhook(job_copy_to_persist)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Webhook emit failed for job %s: %s",
                           job_copy_to_persist.get("id"), exc)


def _emit_job_webhook(job_dict):
    """Fire a ``job.{status}`` webhook event for a terminal job.

    Uses the bounded ``_io_pool`` so the caller of ``_update_job`` isn't
    blocked waiting for outbound HTTP. Also lazily imports
    ``webhook_system`` — it's optional, and the import must not cost
    startup time for installs that never register a webhook.
    """
    status = job_dict.get("status")
    if status not in ("complete", "error", "cancelled"):
        return

    def _fire():
        try:
            from opencut.core.webhook_system import fire_event
        except Exception:  # noqa: BLE001
            return
        event_type = f"job.{status}"
        details = {
            "job_id": job_dict.get("id"),
            "job_type": job_dict.get("type"),
            "filepath": job_dict.get("filepath"),
            "endpoint": job_dict.get("_endpoint"),
            "result": job_dict.get("result"),
            "error": job_dict.get("error"),
            "progress": job_dict.get("progress"),
            "exit_reason": job_dict.get("exit_reason"),
            "peak_vram_mb": job_dict.get("peak_vram_mb"),
            "peak_cpu_pct": job_dict.get("peak_cpu_pct"),
            "peak_rss_mb": job_dict.get("peak_rss_mb"),
            "request_id": job_dict.get("request_id"),
            "client_request_id": job_dict.get("client_request_id"),
        }
        try:
            fire_event(event_type, details, job_id=str(job_dict.get("id") or ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning("fire_event(%s) raised: %s", event_type, exc)

    try:
        _io_pool.submit(_fire)
    except RuntimeError:
        # Interpreter shutdown — best-effort sync fire
        _fire()


def _persist_job(job_dict, *, sync: bool = False):
    """Persist a job to SQLite via bounded I/O pool to avoid I/O under lock."""
    def _save():
        try:
            from opencut.job_store import save_job
            save_job(job_dict)
        except Exception as e:
            logger.warning("Failed to persist job %s: %s", job_dict.get("id"), e)
    if sync:
        _save()
        return
    try:
        _io_pool.submit(_save)
    except RuntimeError:
        # Interpreter shutdown can close the pool before the last persistence
        # task is submitted. Falling back to sync avoids silently dropping the
        # final job state in that window.
        _save()


def _schedule_record_time(job_type, elapsed, filepath):
    """Record job time outside of the lock to avoid I/O under lock."""
    import os

    def _record():
        try:
            from opencut.helpers import _get_file_duration, _record_job_time
            # File may have been deleted/moved by the time this runs
            file_dur = _get_file_duration(filepath) if filepath and os.path.isfile(filepath) else 0
            _record_job_time(job_type, elapsed, file_dur)
        except Exception as e:
            logger.debug("Failed to record job time for %s: %s", job_type, e)
    try:
        _io_pool.submit(_record)
    except RuntimeError:
        # Match _persist_job(): during interpreter shutdown the executor may
        # already be closed, but the final timing sample is still cheap to save.
        _record()


def _payload_key_is_sensitive(key) -> bool:
    return bool(_SENSITIVE_PAYLOAD_KEY_RE.search(str(key)))


def _redact_payload_for_storage(value, *, depth: int = 0, seen=None):
    if seen is None:
        seen = set()
    if depth > _MAX_PERSISTED_PAYLOAD_DEPTH:
        return {"_truncated": True, "_reason": "max_depth"}

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return {"_truncated": True, "_reason": "circular_reference"}
        seen.add(obj_id)
        try:
            items = list(value.items())
            result = {}
            for key, child in items[:_MAX_PERSISTED_PAYLOAD_ITEMS]:
                key_str = str(key)
                if _payload_key_is_sensitive(key_str):
                    result[key_str] = _SENSITIVE_PAYLOAD_VALUE
                else:
                    result[key_str] = _redact_payload_for_storage(
                        child,
                        depth=depth + 1,
                        seen=seen,
                    )
            if len(items) > _MAX_PERSISTED_PAYLOAD_ITEMS:
                result["_keys_trimmed"] = True
                result["_omitted_keys"] = len(items) - _MAX_PERSISTED_PAYLOAD_ITEMS
            return result
        finally:
            seen.discard(obj_id)

    if isinstance(value, (list, tuple, set)):
        obj_id = id(value)
        if obj_id in seen:
            return [{"_truncated": True, "_reason": "circular_reference"}]
        seen.add(obj_id)
        try:
            items = list(value)
            result = [
                _redact_payload_for_storage(item, depth=depth + 1, seen=seen)
                for item in items[:_MAX_PERSISTED_PAYLOAD_ITEMS]
            ]
            if len(items) > _MAX_PERSISTED_PAYLOAD_ITEMS:
                result.append({
                    "_items_trimmed": True,
                    "_omitted_items": len(items) - _MAX_PERSISTED_PAYLOAD_ITEMS,
                })
            return result
        finally:
            seen.discard(obj_id)

    return value


def _sanitize_payload_for_storage(payload):
    """Redact and cap persisted request payloads.

    Job history is useful for diagnostics, but request payloads may contain
    LLM/API credentials. Redaction runs before size checks so even small
    payloads cannot persist secrets into ``~/.opencut/jobs.db``.
    """
    if not isinstance(payload, dict):
        return {}
    try:
        redacted = _redact_payload_for_storage(payload)
        encoded = json.dumps(redacted, ensure_ascii=False, default=str)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return {
            "_truncated": True,
            "_reason": "unserializable",
            "_keys": sorted(str(k) for k in list(payload.keys())[:50]),
        }
    encoded_bytes = len(encoded.encode("utf-8"))
    if encoded_bytes <= MAX_PERSISTED_JOB_PAYLOAD_BYTES:
        return redacted
    return {
        "_truncated": True,
        "_reason": "payload_too_large",
        "_size_bytes": encoded_bytes,
        "_keys": sorted(str(k) for k in list(payload.keys())[:50]),
    }


def _coerce_nonnegative_int(value, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _is_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    with job_lock:
        return jobs.get(job_id, {}).get("status") == "cancelled"


def _register_job_process(job_id: str, proc):
    """Store a Popen handle so the cancel route can kill it."""
    with job_lock:
        _job_processes[job_id] = proc


def _unregister_job_process(job_id: str):
    """Remove a stored Popen handle without killing the process."""
    with job_lock:
        _job_processes.pop(job_id, None)


def _kill_job_process(job_id: str):
    """Terminate (then kill) a stored subprocess for the given job.

    Uses graceful terminate with 3s wait before force-kill. Closes stdin
    after terminating so the child doesn't block on a parent that has
    stopped reading. If the child has filled its stdout/stderr pipe
    buffers and the parent stopped draining (e.g., the worker thread
    abandoned the read loop on cancel), close those pipes from the parent
    side so wait() doesn't deadlock waiting for a kernel-buffered child.
    """
    proc = None
    with job_lock:
        proc = _job_processes.pop(job_id, None)
    if proc is None:
        return

    def _close_pipe(pipe):
        if pipe is None:
            return
        try:
            pipe.close()
        except (OSError, ValueError):
            pass

    try:
        proc.terminate()
    except OSError as e:
        logger.debug("Failed to terminate process for job %s: %s", job_id, e)
    # Drop our pipe ends immediately so any worker thread blocked on stdout or
    # stderr wakes up while we wait for the child to exit.
    _close_pipe(getattr(proc, "stdin", None))
    _close_pipe(getattr(proc, "stdout", None))
    _close_pipe(getattr(proc, "stderr", None))

    # Wait up to 3 seconds for graceful exit
    try:
        proc.wait(timeout=3)
        return
    except Exception as e:
        logger.debug("Process for job %s did not exit gracefully: %s", job_id, e)

    # Force kill if still alive
    try:
        proc.kill()
    except OSError as e:
        logger.debug("Failed to kill process for job %s: %s", job_id, e)
    # Close output pipes so the kernel can release their buffers.
    _close_pipe(getattr(proc, "stdout", None))
    _close_pipe(getattr(proc, "stderr", None))
    # Reap zombie process
    try:
        proc.wait(timeout=5)
    except Exception as e:
        logger.debug("Failed to reap process for job %s: %s", job_id, e)


def _cancel_job(job_id: str, *, message: str = "Cancelled by user",
                persist_sync: bool = False) -> Tuple[Optional[dict], str]:
    """Cancel a single running job and persist the terminal state."""
    cancelled_job = None
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            return None, "not_found"
        if job.get("status") != "running":
            return None, "not_running"
        job["status"] = "cancelled"
        job["message"] = message
        job["progress"] = 0
        job["exit_reason"] = "cancelled"
        job["completed_at"] = time.time()
        cancelled_job = job.copy()

    _kill_job_process(job_id)

    try:
        from opencut.workers import cancel_job as _cancel_queued_job

        if _cancel_queued_job(job_id):
            with job_lock:
                if job_id in jobs:
                    jobs[job_id]["message"] = "Cancelled before starting"
                    cancelled_job = jobs[job_id].copy()
    except Exception as e:
        logger.debug("Failed to cancel queued job %s: %s", job_id, e)

    if cancelled_job is not None:
        _persist_job(cancelled_job, sync=persist_sync)
    return cancelled_job, "cancelled"


def _cancel_running_jobs(*, message: str = "Cancelled by user",
                         persist_sync: bool = False) -> list[str]:
    """Cancel every running job and return the cancelled job IDs."""
    with job_lock:
        running_ids = [jid for jid, job in jobs.items() if job.get("status") == "running"]

    for job_id in running_ids:
        _cancel_job(job_id, message=message, persist_sync=persist_sync)
    return running_ids


_JOB_STUCK_TIMEOUT = _JOB_CONFIG.job_stuck_timeout
_CLEANUP_INTERVAL = 300  # 5 minutes — periodic cleanup interval
_cleanup_timer_started = False
_cleanup_timer_lock = threading.Lock()


def _start_periodic_cleanup():
    """Lazily start a daemon thread that runs _cleanup_old_jobs() every 5 minutes.

    Called on first job creation. The thread is a daemon so it won't block
    interpreter shutdown.  Idempotent — only one timer thread is ever started.
    """
    global _cleanup_timer_started
    with _cleanup_timer_lock:
        if _cleanup_timer_started:
            return
        _cleanup_timer_started = True

    def _periodic_cleanup_loop():
        while True:
            time.sleep(_CLEANUP_INTERVAL)
            try:
                _cleanup_old_jobs()
            except Exception as e:
                logger.debug("Periodic job cleanup error: %s", e)

    try:
        t = threading.Thread(target=_periodic_cleanup_loop, daemon=True,
                             name="opencut-job-cleanup")
        t.start()
        logger.debug("Started periodic job cleanup thread (every %ds)", _CLEANUP_INTERVAL)
    except Exception as e:
        logger.warning("Failed to start cleanup thread: %s", e)
        with _cleanup_timer_lock:
            _cleanup_timer_started = False


def _cleanup_old_jobs():
    """Mark stuck running jobs, purge old terminal jobs, and reap dead procs.

    State mutation runs under ``job_lock`` so we never interleave with workers
    that are mid-update. Process termination and persistence happen after
    releasing the lock because both can block.
    """
    now = time.time()
    jobs_to_persist = []
    processes_to_kill = set()
    stuck_hrs = _JOB_STUCK_TIMEOUT / 3600
    with job_lock:
        expired = []
        for jid, j in jobs.items():
            status = j.get("status")
            created = j.get("created", now)
            age = now - created
            if status == "running" and age > _JOB_STUCK_TIMEOUT:
                j["status"] = "error"
                j["error"] = f"Job timed out (stuck for >{stuck_hrs:.0f} hours)"
                j["message"] = "Timed out"
                j["exit_reason"] = "timeout"
                j["completed_at"] = now
                logger.warning("Marking stuck job %s as error (created %.0fs ago)", jid, age)
                jobs_to_persist.append(j.copy())
                if jid in _job_processes:
                    processes_to_kill.add(jid)
            elif status in ("complete", "error", "cancelled") and age > JOB_MAX_AGE:
                expired.append(jid)
        for jid in expired:
            del jobs[jid]
            if jid in _job_processes:
                processes_to_kill.add(jid)
        # Reap _job_processes entries whose process has already terminated
        stale_procs = []
        for jid, proc in _job_processes.items():
            try:
                if proc.poll() is not None:
                    stale_procs.append(jid)
            except Exception:
                stale_procs.append(jid)
        for jid in stale_procs:
            if jid not in processes_to_kill:
                _job_processes.pop(jid, None)
    for job_id in processes_to_kill:
        _kill_job_process(job_id)
    for job_dict in jobs_to_persist:
        _persist_job(job_dict)


def async_job(job_type: str, *, filepath_required: bool = True,
              filepath_param: str = "filepath",
              pre_validate=None,
              disk_operation: Optional[str] = None,
              disk_required_mb: Optional[int] = None,
              resumable: bool = False,
              partial_output_param: str = "partial_output_path",
              rate_limit_key: Optional[Union[str, Callable[[dict], Optional[str]]]] = None,
              rate_limit_max_concurrent: int = 1):
    """
    Decorator that wraps a route handler in the standard async job pattern.

    The decorated function receives ``(job_id, filepath, data)`` and should
    return a result dict on success.  Exceptions are caught and recorded as
    job errors automatically.  The decorator handles:

    - JSON parsing and filepath validation
    - Optional synchronous pre-validation (fail-fast 400 instead of an
      async job that errors out 200ms later)
    - ``TooManyJobsError`` → HTTP 429
    - Background thread via ``WorkerPool``
    - Optional worker-lifetime rate-limit acquisition
    - Cancellation check before marking complete
    - Job-ID log correlation on the worker thread

    Args:
        job_type: Job type string for tracking (e.g. "silence", "export").
        filepath_required: If True (default), validates that a filepath is
            present and passes ``validate_filepath()`` checks.  Set to False
            for routes that don't need an input file (install, TTS, etc.).
        filepath_param: JSON field name for the primary filepath.  Defaults
            to ``"filepath"``.  Use ``"video_path"``, ``"input_file"``, etc.
            for routes that use a different field.
        pre_validate: Optional callable ``(data: dict) -> str | None`` that
            runs synchronously before the worker thread is spawned. Return a
            non-empty string to short-circuit with a 400 ``INVALID_INPUT``
            response — useful for routes like ``/captions/translate`` that
            need a list of segments instead of a filepath, so the client
            sees the error immediately instead of via job polling.
        disk_operation: Optional operation key for disk preflight. When set,
            the wrapper estimates required space and returns HTTP 507 before
            creating a job if the output volume is below the budget.
        disk_required_mb: Optional fixed preflight budget. When set without
            ``disk_operation``, the job type is used as the operation key.
        resumable: Mark jobs from this route as safe to re-enqueue from
            persisted payload metadata when a server restart interrupts them.
        partial_output_param: Optional JSON field that stores checkpoint or
            partial-output state used by the resume route.
        rate_limit_key: Optional static key, or callable ``(data) -> key``.
            When set, the wrapper acquires the slot before creating the job,
            returns HTTP 429 if saturated, and releases the slot when the
            worker exits. Callable keys may return ``None`` or ``""`` to skip
            limiting for a specific request.
        rate_limit_max_concurrent: Maximum concurrent jobs allowed for
            ``rate_limit_key``. Defaults to 1.

    Usage::

        @audio_bp.route("/silence", methods=["POST"])
        @require_csrf
        @async_job("silence")
        def silence_remove(job_id, filepath, data):
            # ... do work, call _update_job(job_id, progress=50) ...
            return {"segments": 42}

        @video_bp.route("/video/depth/install", methods=["POST"])
        @require_csrf
        @async_job("install", filepath_required=False)
        def depth_install(job_id, filepath, data):
            # filepath will be "" — no input file needed
            safe_pip_install("transformers", timeout=600)
            return {"component": "depth_effects"}

    For edge cases (pre-validation, multi-file), handle them inside the
    handler body — raise ``ValueError`` for validation failures (recorded
    as job error). Use ``rate_limit_key`` for GPU or install slots that
    should be held for the whole worker lifetime.
    """
    import functools

    from flask import g, jsonify, request

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            from opencut.security import get_json_dict, validate_filepath
            try:
                data = get_json_dict()
            except ValueError as e:
                return jsonify({
                    "error": str(e),
                    "code": "INVALID_INPUT",
                    "suggestion": "Send a top-level JSON object in the request body.",
                }), 400
            # Per CLAUDE.md convention, the CEP panel always sends `filepath`.
            # Routes that use a non-default ``filepath_param`` (e.g.
            # ``melody_path``, ``music_path``, ``file``) still need to accept
            # ``filepath`` as a fallback so frontend + backend stay in sync.
            filepath = data.get(filepath_param, "")
            if filepath_param != "filepath" and not filepath:
                filepath = data.get("filepath", "")
            if isinstance(filepath, str):
                filepath = filepath.strip()

            if filepath_required:
                if not filepath:
                    return jsonify({
                        "error": "No file path provided",
                        "code": "INVALID_INPUT",
                        "suggestion": "Select a clip in Premiere or pass `filepath` in the request body.",
                    }), 400
                try:
                    filepath = validate_filepath(filepath)
                except ValueError as e:
                    msg = str(e)
                    # Classify: path traversal / null byte / UNC → INVALID_INPUT,
                    # not-found / not-a-file → FILE_NOT_FOUND.
                    lower = msg.lower()
                    if "not found" in lower or "not a file" in lower or "does not exist" in lower:
                        code, hint = "FILE_NOT_FOUND", "Check the path exists and is accessible."
                    else:
                        code, hint = "INVALID_INPUT", "Use a plain absolute path with no traversal, null bytes, or UNC prefix."
                    return jsonify({"error": msg, "code": code, "suggestion": hint}), 400
            elif filepath:
                # Optional filepath provided — still validate it
                try:
                    filepath = validate_filepath(filepath)
                except ValueError as e:
                    msg = str(e)
                    lower = msg.lower()
                    if "not found" in lower or "not a file" in lower or "does not exist" in lower:
                        code, hint = "FILE_NOT_FOUND", "Check the path exists and is accessible."
                    else:
                        code, hint = "INVALID_INPUT", "Use a plain absolute path with no traversal, null bytes, or UNC prefix."
                    return jsonify({"error": msg, "code": code, "suggestion": hint}), 400

            # Optional synchronous pre-validation — fail fast instead of
            # spawning a worker thread that immediately raises ValueError.
            if pre_validate is not None:
                try:
                    err = pre_validate(data)
                except Exception as exc:  # noqa: BLE001
                    err = f"Validation error: {exc}"
                if err:
                    return jsonify({
                        "error": str(err),
                        "code": "INVALID_INPUT",
                        "suggestion": "Check the request body against the route's documented schema.",
                    }), 400

            if disk_operation or disk_required_mb is not None:
                try:
                    from opencut.core.preflight import ensure_disk_for

                    disk = ensure_disk_for(
                        disk_operation or job_type,
                        filepath,
                        data,
                        required_mb=disk_required_mb,
                    )
                except ValueError as e:
                    return jsonify({
                        "error": str(e),
                        "code": "INVALID_INPUT",
                        "suggestion": "Check the output path or output directory.",
                    }), 400
                if not disk.get("ok", True):
                    output_dir = disk.get("output_dir", "")
                    required_mb = int(disk.get("required_mb") or 0)
                    free_mb = int(disk.get("free_mb") or 0)
                    return jsonify({
                        "error": "Insufficient disk space",
                        "code": "INSUFFICIENT_STORAGE",
                        "operation": disk.get("operation") or job_type,
                        "required_mb": required_mb,
                        "free_mb": free_mb,
                        "output_dir": output_dir,
                        "note": disk.get("note", ""),
                        "suggestion": (
                            "Free disk space or choose another output directory "
                            f"with at least {required_mb} MB available."
                        ),
                    }), 507

            if should_skip_install_in_testing(job_type):
                component_name = data.get("component") or job_type
                return testing_install_response(component_name)

            job_label = filepath or job_type
            partial_output_path = ""
            if partial_output_param:
                partial_output_path = data.get(partial_output_param, "")
                if partial_output_param != "partial_output_path" and not partial_output_path:
                    partial_output_path = data.get("partial_output_path", "")
            if partial_output_path is None:
                partial_output_path = ""
            partial_output_path = str(partial_output_path).strip()
            resume_source_job_id = str(
                data.get("resume_source_job_id")
                or data.get("resume_from_job_id")
                or ""
            ).strip()
            resume_attempt = _coerce_nonnegative_int(data.get("resume_attempt"), 0)
            acquired_rate_limit_key = ""
            if rate_limit_key is not None:
                try:
                    resolved_key = rate_limit_key(data) if callable(rate_limit_key) else rate_limit_key
                except Exception as exc:  # noqa: BLE001
                    return jsonify({
                        "error": f"Rate-limit key resolution failed: {exc}",
                        "code": "INVALID_INPUT",
                        "suggestion": "Check the request body against the route's documented schema.",
                    }), 400
                resolved_key = str(resolved_key or "").strip()
                if resolved_key:
                    from opencut.security import rate_limit, rate_limit_release

                    if not rate_limit(resolved_key, rate_limit_max_concurrent):
                        return jsonify({
                            "error": f"Another {resolved_key} operation is already running. Please wait.",
                            "code": "RATE_LIMITED",
                            "suggestion": "Wait for the current operation to finish, then retry.",
                        }), 429
                    acquired_rate_limit_key = resolved_key

            try:
                job_id = _new_job(
                    job_type,
                    job_label,
                    resumable=resumable,
                    partial_output_path=partial_output_path,
                    resume_source_job_id=resume_source_job_id,
                    resume_attempt=resume_attempt,
                )
            except TooManyJobsError as e:
                if acquired_rate_limit_key:
                    from opencut.security import rate_limit_release

                    rate_limit_release(acquired_rate_limit_key)
                return jsonify({
                    "error": str(e),
                    "code": "TOO_MANY_JOBS",
                    "suggestion": "Wait for a job to finish or cancel one from the processing bar.",
                }), 429

            request_id = ""
            client_request_id = ""
            try:
                from opencut.core.request_correlation import get_request_id

                request_id = get_request_id()
                client_request_id = str(getattr(g, "client_request_id", "") or "")
            except Exception:  # noqa: BLE001 - request metadata is optional
                pass

            job_to_persist = None
            with job_lock:
                if job_id in jobs:
                    jobs[job_id]["_endpoint"] = request.path
                    jobs[job_id]["_payload"] = _sanitize_payload_for_storage(data)
                    jobs[job_id]["request_id"] = request_id
                    jobs[job_id]["client_request_id"] = client_request_id
                    job_to_persist = jobs[job_id].copy()
            if job_to_persist is not None:
                _persist_job(job_to_persist, sync=True)

            def _process():
                _thread_local.job_id = job_id
                resource_sampler = None
                worker_request_id = ""
                try:
                    from opencut.core.request_correlation import set_request_id

                    with job_lock:
                        worker_request_id = str(jobs.get(job_id, {}).get("request_id") or "")
                    if worker_request_id:
                        set_request_id(worker_request_id)
                except Exception:  # noqa: BLE001 - logging correlation is optional
                    worker_request_id = ""
                try:
                    from opencut.server import _log_thread_local
                    _log_thread_local.job_id = job_id
                except ImportError:
                    pass
                try:
                    if _is_cancelled(job_id):
                        _update_job(job_id, message="Cancelled before starting", progress=0)
                        return
                    _update_job(job_id, started_at=time.time())
                    try:
                        from opencut.core.job_diagnostics import JobResourceSampler

                        resource_sampler = JobResourceSampler()
                        resource_sampler.start()
                    except Exception as exc:  # noqa: BLE001 - diagnostics only
                        logger.debug("Failed to start resource sampler for %s: %s", job_id, exc)
                        resource_sampler = None
                    result = f(job_id, filepath, data)
                    resource_update = _stop_resource_sampler(resource_sampler)
                    resource_sampler = None
                    if not _is_cancelled(job_id):
                        _update_job(job_id, status="complete", progress=100,
                                    result=result, message="Done",
                                    **resource_update)
                    else:
                        _update_job(job_id, exit_reason="cancelled", **resource_update)
                except Exception as e:
                    logger.exception("Job %s error in %s", job_id, job_type)
                    resource_update = _stop_resource_sampler(resource_sampler)
                    resource_sampler = None
                    update = {
                        "status": "error",
                        "error": str(e),
                        "message": f"Error: {e}",
                    }
                    if getattr(e, "code", ""):
                        update["code"] = getattr(e, "code")
                    if getattr(e, "retry_after", None) is not None:
                        update["retry_after"] = getattr(e, "retry_after")
                    if getattr(e, "queue_depth", None) is not None:
                        update["queue_depth"] = getattr(e, "queue_depth")
                    update["exit_reason"] = _classify_exit_reason(
                        "error",
                        error=str(e),
                        code=str(update.get("code") or ""),
                        exc=e,
                    )
                    update.update(resource_update)
                    try:
                        from opencut.core.install_hints import suggestion_for_exception

                        suggestion = suggestion_for_exception(e, context=job_type)
                        if suggestion:
                            update["suggestion"] = suggestion
                    except Exception:  # noqa: BLE001 - job finalisation must not fail
                        pass
                    _update_job(job_id, **update)
                finally:
                    resource_update = _stop_resource_sampler(resource_sampler)
                    if resource_update:
                        _update_job(job_id, **resource_update)
                    if worker_request_id:
                        try:
                            from opencut.core.request_correlation import clear_request_id, get_request_id

                            if get_request_id() == worker_request_id:
                                clear_request_id()
                        except Exception:  # noqa: BLE001
                            pass
                    _thread_local.job_id = ""
                    try:
                        from opencut.server import _log_thread_local
                        _log_thread_local.job_id = ""
                    except ImportError:
                        pass
                    if acquired_rate_limit_key:
                        try:
                            from opencut.security import rate_limit_release

                            rate_limit_release(acquired_rate_limit_key)
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "Failed to release rate-limit key %s for job %s",
                                acquired_rate_limit_key,
                                job_id,
                            )

            from opencut.workers import get_pool
            try:
                future = get_pool().submit(job_id, _process)
            except Exception:
                if acquired_rate_limit_key:
                    from opencut.security import rate_limit_release

                    rate_limit_release(acquired_rate_limit_key)
                raise
            # Store future immediately (before returning) so cancel can find it
            with job_lock:
                if job_id in jobs:
                    jobs[job_id]["_future"] = future
                    jobs[job_id]["_thread"] = future
            return jsonify({"job_id": job_id})
        wrapper._opencut_async_job = True
        wrapper._opencut_job_type = job_type
        wrapper._opencut_resumable = bool(resumable)
        return wrapper
    return decorator


def make_install_route(blueprint, url_path, component_name, packages,
                       *, doc=None):
    """Factory that creates a standard install route on *blueprint*.

    Eliminates boilerplate for the ~10 identical install endpoints.  Each
    generated route:

    * ``POST url_path`` with CSRF
    * Rate-limits under ``"model_install"`` for the worker lifetime and returns
      **429 synchronously** when another install is already running.
    * Iterates *packages* with progress updates
    * Returns ``{"component": component_name}``

    Args:
        blueprint: Flask Blueprint to register the route on.
        url_path:  URL rule, e.g. ``"/video/depth/install"``.
        component_name: Value returned in ``{"component": ...}``.
        packages:  List of pip package specifiers.
        doc:       Optional docstring override.
    """
    from opencut.security import require_csrf as _csrf
    from opencut.security import safe_pip_install as _spi

    # Inner function that actually performs the install. Runs in a worker thread
    # via @async_job while the wrapper holds the model-install slot.
    @async_job("install", filepath_required=False, rate_limit_key="model_install")
    def _install_body(job_id, filepath, data):
        for i, pkg in enumerate(packages):
            pct = int((i / len(packages)) * 90)
            _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
            _spi(pkg, timeout=600)
        # Invalidate the capabilities cache so the next health check
        # reflects the newly installed package.
        try:
            from opencut.routes.system import invalidate_caps_cache
            invalidate_caps_cache()
        except ImportError:
            pass
        return {"component": component_name}

    @blueprint.route(url_path, methods=["POST"])
    @_csrf
    def _install_handler():
        if should_skip_install_in_testing("install"):
            return testing_install_response(component_name)

        return _install_body()

    _install_handler.__name__ = f"install_{component_name}"
    _install_handler.__qualname__ = f"install_{component_name}"
    if doc:
        _install_handler.__doc__ = doc
    return _install_handler
