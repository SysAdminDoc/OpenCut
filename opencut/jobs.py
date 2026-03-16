"""
OpenCut Job System

Shared job tracking state and helper functions used across all route modules.
"""

import logging
import os
import subprocess as _sp
import threading
import time
import uuid

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Job State (shared across all Blueprints)
# ---------------------------------------------------------------------------
jobs = {}
job_lock = threading.Lock()
_job_processes = {}  # job_id -> Popen, for subprocess kill support
JOB_MAX_AGE = 3600  # Auto-clean jobs older than 1 hour
MAX_CONCURRENT_JOBS = 10  # Prevent job spam / GPU OOM on consumer hardware
MAX_BATCH_FILES = 100  # Max files per batch request


def _safe_error(e, context=""):
    """Log the real exception and return a generic error response."""
    from flask import jsonify
    logger.exception("Internal error%s: %s", f" in {context}" if context else "", e)
    return jsonify({"error": "An internal error occurred. Check server logs for details."}), 500


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
    _cleanup_old_jobs()
    return job_id


def _get_job_copy(job_id: str) -> dict | None:
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
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
            # Record timing data when a job completes for future estimates
            if kwargs.get("status") == "complete":
                job = jobs[job_id]
                elapsed = time.time() - job.get("created", time.time())
                _schedule_record_time(job.get("type", ""), elapsed, job.get("filepath", ""))


def _schedule_record_time(job_type, elapsed, filepath):
    """Record job time outside of the lock to avoid I/O under lock."""
    def _record():
        try:
            from opencut.helpers import _get_file_duration, _record_job_time
            _record_job_time(job_type, elapsed, _get_file_duration(filepath))
        except Exception:
            pass
    threading.Thread(target=_record, daemon=True).start()


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
        except OSError:
            pass
        # Wait up to 3 seconds for graceful exit
        try:
            proc.wait(timeout=3)
            return
        except Exception:
            pass
        # Force kill if still alive
        try:
            proc.kill()
        except OSError:
            pass


def _cleanup_old_jobs():
    """Remove completed/errored jobs older than JOB_MAX_AGE."""
    now = time.time()
    with job_lock:
        expired = [
            jid for jid, j in jobs.items()
            if j["status"] in ("complete", "error", "cancelled")
            and (now - j["created"]) > JOB_MAX_AGE
        ]
        for jid in expired:
            del jobs[jid]
            _job_processes.pop(jid, None)


def async_job(job_type: str):
    """
    Decorator that wraps a route handler in the standard async job pattern.

    The decorated function receives ``(job_id, filepath, data)`` and should
    return a result dict on success.  Exceptions are caught and recorded as
    job errors automatically.

    Usage::

        @audio_bp.route("/silence", methods=["POST"])
        @require_csrf
        @async_job("silence")
        def silence_remove(job_id, filepath, data):
            # ... do work, call _update_job(job_id, progress=50) ...
            return {"segments": 42}
    """
    import functools
    from flask import request, jsonify

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            from opencut.security import validate_filepath
            data = request.get_json(force=True) or {}
            filepath = data.get("filepath", "")
            try:
                filepath = validate_filepath(filepath)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

            job_id = _new_job(job_type, filepath)

            def _process():
                try:
                    result = f(job_id, filepath, data)
                    if not _is_cancelled(job_id):
                        _update_job(job_id, status="complete", progress=100,
                                    result=result, message="Done")
                except Exception as e:
                    logger.exception("Job %s error in %s", job_id, job_type)
                    _update_job(job_id, status="error", error=str(e),
                                message=f"Error: {e}")

            import threading as _t
            _t.Thread(target=_process, daemon=True).start()
            return jsonify({"job_id": job_id})
        return wrapper
    return decorator
