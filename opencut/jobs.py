"""
OpenCut Job System

Shared job tracking state and helper functions used across all route modules.
"""

import atexit
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

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


# ---------------------------------------------------------------------------
# Job State (shared across all Blueprints)
# ---------------------------------------------------------------------------
jobs = {}
job_lock = threading.Lock()
_job_processes = {}  # job_id -> Popen, for subprocess kill support
# These mirror OpenCutConfig defaults (the single source of truth for
# documentation and test overrides).  They are kept as module-level constants
# because jobs.py operates outside Flask app context.
JOB_MAX_AGE = 3600              # see OpenCutConfig.job_max_age
MAX_CONCURRENT_JOBS = 10        # see OpenCutConfig.max_concurrent_jobs
MAX_BATCH_FILES = 100           # see OpenCutConfig.max_batch_files


def _safe_error(e, context=""):
    """Log the real exception and return a structured error response.

    Delegates to opencut.errors.safe_error for exception classification
    so the frontend receives an error code and recovery suggestion.
    """
    from opencut.errors import safe_error
    return safe_error(e, context=context)


class TooManyJobsError(RuntimeError):
    """Raised when MAX_CONCURRENT_JOBS is reached."""
    pass


def _new_job(job_type: str, filepath: str) -> str:
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
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
            # Record timing data when a job completes for future estimates
            if kwargs.get("status") == "complete":
                job = jobs[job_id]
                elapsed = time.time() - job.get("created", time.time())
                _schedule_record_time(job.get("type", ""), elapsed, job.get("filepath", ""))
            # Persist terminal job states to SQLite
            if kwargs.get("status") in ("complete", "error", "cancelled"):
                job_copy_to_persist = jobs[job_id].copy()
                _job_processes.pop(job_id, None)
    if job_copy_to_persist is not None:
        _persist_job(job_copy_to_persist)


def _persist_job(job_dict):
    """Persist a job to SQLite via bounded I/O pool to avoid I/O under lock."""
    def _save():
        try:
            from opencut.job_store import save_job
            save_job(job_dict)
        except Exception as e:
            logger.debug("Failed to persist job %s: %s", job_dict.get("id"), e)
    _io_pool.submit(_save)


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
    _io_pool.submit(_record)


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
    Uses graceful terminate with 3s wait before force-kill."""
    proc = None
    with job_lock:
        proc = _job_processes.pop(job_id, None)
    if proc is not None:
        try:
            proc.terminate()
        except OSError as e:
            logger.debug("Failed to terminate process for job %s: %s", job_id, e)
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
        # Reap zombie process
        try:
            proc.wait(timeout=5)
        except Exception as e:
            logger.debug("Failed to reap process for job %s: %s", job_id, e)


_JOB_STUCK_TIMEOUT = 7200  # see OpenCutConfig.job_stuck_timeout
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
    """Remove completed/errored jobs older than JOB_MAX_AGE.
    Also mark stuck 'running' jobs as error after _JOB_STUCK_TIMEOUT."""
    now = time.time()
    with job_lock:
        # Mark stuck running jobs as error
        for jid, j in jobs.items():
            if j["status"] == "running" and (now - j["created"]) > _JOB_STUCK_TIMEOUT:
                j["status"] = "error"
                j["error"] = "Job timed out (stuck for >2 hours)"
                j["message"] = "Timed out"
                logger.warning("Marking stuck job %s as error (created %.0fs ago)", jid, now - j["created"])
                _persist_job(j.copy())
        # Clean up old finished jobs
        expired = [
            jid for jid, j in jobs.items()
            if j["status"] in ("complete", "error", "cancelled")
            and (now - j["created"]) > JOB_MAX_AGE
        ]
        for jid in expired:
            del jobs[jid]
            _job_processes.pop(jid, None)
        # Clean up _job_processes entries where the process has already terminated
        stale_procs = []
        for jid, proc in _job_processes.items():
            try:
                if proc.poll() is not None:
                    stale_procs.append(jid)
            except Exception:
                stale_procs.append(jid)
        for jid in stale_procs:
            _job_processes.pop(jid, None)


def async_job(job_type: str, *, filepath_required: bool = True,
              filepath_param: str = "filepath"):
    """
    Decorator that wraps a route handler in the standard async job pattern.

    The decorated function receives ``(job_id, filepath, data)`` and should
    return a result dict on success.  Exceptions are caught and recorded as
    job errors automatically.  The decorator handles:

    - JSON parsing and filepath validation
    - ``TooManyJobsError`` → HTTP 429
    - Background thread via ``WorkerPool``
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

    For edge cases (GPU rate limiting, pre-validation, multi-file),
    handle them inside the handler body — raise ``ValueError`` for
    validation failures (recorded as job error).
    """
    import functools

    from flask import jsonify, request

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            from opencut.security import validate_filepath
            data = request.get_json(force=True) or {}
            filepath = data.get(filepath_param, "")
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

            job_label = filepath or job_type
            try:
                job_id = _new_job(job_type, job_label)
            except TooManyJobsError as e:
                return jsonify({
                    "error": str(e),
                    "code": "TOO_MANY_JOBS",
                    "suggestion": "Wait for a job to finish or cancel one from the processing bar.",
                }), 429

            def _process():
                _thread_local.job_id = job_id
                try:
                    from opencut.server import _log_thread_local
                    _log_thread_local.job_id = job_id
                except ImportError:
                    pass
                try:
                    result = f(job_id, filepath, data)
                    if not _is_cancelled(job_id):
                        _update_job(job_id, status="complete", progress=100,
                                    result=result, message="Done")
                except Exception as e:
                    logger.exception("Job %s error in %s", job_id, job_type)
                    _update_job(job_id, status="error", error=str(e),
                                message=f"Error: {e}")
                finally:
                    _thread_local.job_id = ""
                    try:
                        from opencut.server import _log_thread_local
                        _log_thread_local.job_id = ""
                    except ImportError:
                        pass

            from opencut.workers import get_pool
            future = get_pool().submit(job_id, _process)
            # Store future immediately (before returning) so cancel can find it
            with job_lock:
                if job_id in jobs:
                    jobs[job_id]["_future"] = future
                    jobs[job_id]["_thread"] = future
            return jsonify({"job_id": job_id})
        return wrapper
    return decorator


def make_install_route(blueprint, url_path, component_name, packages,
                       *, doc=None):
    """Factory that creates a standard install route on *blueprint*.

    Eliminates boilerplate for the ~10 identical install endpoints.  Each
    generated route:

    * ``POST url_path`` with CSRF
    * Rate-limits under ``"model_install"`` — returns **429 synchronously** when
      another install is already running (the caller must wait). Previously the
      rate limit check happened inside the async body which meant clients got
      an initial 200 + job_id and only discovered the failure by polling the
      job status.
    * Iterates *packages* with progress updates
    * Returns ``{"component": component_name}``

    Args:
        blueprint: Flask Blueprint to register the route on.
        url_path:  URL rule, e.g. ``"/video/depth/install"``.
        component_name: Value returned in ``{"component": ...}``.
        packages:  List of pip package specifiers.
        doc:       Optional docstring override.
    """
    from flask import jsonify

    from opencut.security import rate_limit as _rl
    from opencut.security import rate_limit_release as _rlr
    from opencut.security import require_csrf as _csrf
    from opencut.security import safe_pip_install as _spi

    # Inner function that actually performs the install. Runs in a worker thread
    # via @async_job. The outer Flask handler below acquires the rate-limit slot
    # before spawning the job, so `rate_limit` is already held when _install_body
    # starts — we just need to release it in the `finally`.
    @async_job("install", filepath_required=False)
    def _install_body(job_id, filepath, data):
        try:
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
        finally:
            _rlr("model_install")

    @blueprint.route(url_path, methods=["POST"])
    @_csrf
    def _install_handler():
        if not _rl("model_install"):
            return jsonify({
                "error": "A model_install operation is already running. Please wait.",
                "code": "RATE_LIMITED",
                "suggestion": "Wait for the current install to finish, then retry.",
            }), 429
        try:
            return _install_body()
        except Exception:
            # _install_body only raises before spawning the worker thread
            # (e.g., TooManyJobsError). Release the slot so the next caller
            # isn't permanently locked out.
            _rlr("model_install")
            raise

    _install_handler.__name__ = f"install_{component_name}"
    _install_handler.__qualname__ = f"install_{component_name}"
    if doc:
        _install_handler.__doc__ = doc
    return _install_handler
