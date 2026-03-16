"""
OpenCut Job Routes

Job status, cancel, list, SSE streaming, queue management.
"""

import json
import logging
import threading
import time
import uuid

from flask import Blueprint, request, jsonify, Response

from opencut.jobs import (
    jobs, job_lock, _new_job, _update_job, _safe_error,
    _is_cancelled, _kill_job_process,
    _get_job_copy, _list_jobs_copy,
)
from opencut.security import require_csrf

logger = logging.getLogger("opencut")

jobs_bp = Blueprint("jobs", __name__)


# ---------------------------------------------------------------------------
# Job Status / Cancel / List
# ---------------------------------------------------------------------------
@jobs_bp.route("/status/<job_id>", methods=["GET"])
def job_status(job_id):
    """Check the status of a processing job."""
    safe = _get_job_copy(job_id)
    if not safe:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(safe)


@jobs_bp.route("/cancel/<job_id>", methods=["POST"])
@require_csrf
def cancel_job(job_id):
    """Cancel a running job."""
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job["status"] != "running":
            return jsonify({"error": "Job is not running"}), 400
        job["status"] = "cancelled"
        job["message"] = "Cancelled by user"
        job["progress"] = 0
    _kill_job_process(job_id)
    return jsonify({"status": "cancelled", "job_id": job_id})


@jobs_bp.route("/cancel-all", methods=["POST"])
@require_csrf
def cancel_all_jobs():
    """Cancel all running jobs."""
    cancelled = []
    with job_lock:
        for jid, job in jobs.items():
            if job.get("status") == "running":
                job["status"] = "cancelled"
                job["message"] = "Cancelled by user"
                job["progress"] = 0
                cancelled.append(jid)
    for jid in cancelled:
        _kill_job_process(jid)
    return jsonify({"cancelled": cancelled, "count": len(cancelled)})


@jobs_bp.route("/jobs", methods=["GET"])
def list_jobs():
    """List all jobs."""
    return jsonify(_list_jobs_copy())


# ---------------------------------------------------------------------------
# Server-Sent Events (SSE) job stream
# ---------------------------------------------------------------------------
@jobs_bp.route("/stream/<job_id>", methods=["GET"])
def stream_job(job_id):
    """Stream job status via Server-Sent Events. Replaces polling."""
    def generate():
        while True:
            # Copy job data under the lock so we don't hold it during yield
            safe = _get_job_copy(job_id)
            if not safe:
                yield f"data: {json.dumps({'status': 'not_found', 'error': 'Job not found'})}\n\n"
                break
            status = safe.get("status")
            yield f"data: {json.dumps(safe)}\n\n"
            if status in ("complete", "error", "cancelled"):
                break
            time.sleep(0.5)

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    # CEP panels run with origin "null" (no scheme); this is intentional
    # and safe because the server only binds to 127.0.0.1.
    resp.headers["Access-Control-Allow-Origin"] = "null"
    return resp


# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------
job_queue = []
job_queue_lock = threading.Lock()
_queue_state = {"running": False}


@jobs_bp.route("/queue/add", methods=["POST"])
@require_csrf
def queue_add():
    """Add a job to the queue."""
    data = request.get_json(force=True)
    entry = {
        "id": str(uuid.uuid4())[:8],
        "endpoint": data.get("endpoint", ""),
        "payload": data.get("payload", {}),
        "status": "queued",
        "added": time.time(),
    }
    with job_queue_lock:
        job_queue.append(entry)
    _process_queue()
    return jsonify({"queue_id": entry["id"], "position": len(job_queue)})


@jobs_bp.route("/queue/list", methods=["GET"])
def queue_list():
    """List queued jobs."""
    with job_queue_lock:
        return jsonify(list(job_queue))


@jobs_bp.route("/queue/clear", methods=["POST"])
@require_csrf
def queue_clear():
    """Clear all queued (not running) jobs."""
    with job_queue_lock:
        removed = len([e for e in job_queue if e["status"] == "queued"])
        job_queue[:] = [e for e in job_queue if e["status"] != "queued"]
    return jsonify({"removed": removed})


def _dispatch_queue_entry(entry):
    """Dispatch a queue entry by calling the route handler directly
    via Flask's test_request_context (no HTTP round-trip, no CSRF issues)."""
    from flask import current_app
    endpoint = entry.get("endpoint", "")
    payload = entry.get("payload", {})

    with current_app.test_request_context(endpoint, method="POST",
                                          json=payload,
                                          headers={"Content-Type": "application/json"}):
        try:
            # Look up the route function and call it directly
            adapter = current_app.url_map.bind("")
            rule, view_args = adapter.match(endpoint, method="POST")
            view_func = current_app.view_functions.get(rule.endpoint)
            if view_func is None:
                entry["status"] = "error"
                return
            resp = view_func(**view_args)
            # Flask view functions return (response, status) or a Response
            if isinstance(resp, tuple):
                resp_obj = resp[0]
            else:
                resp_obj = resp
            result = resp_obj.get_json() if hasattr(resp_obj, "get_json") else {}
            entry["job_id"] = result.get("job_id", "")
            entry["status"] = "started"
        except Exception as e:
            entry["status"] = "error"
            logger.exception("Queue dispatch error for %s: %s", endpoint, e)


def _process_queue():
    """Process the next item in the queue (fire-and-forget)."""
    with job_queue_lock:
        if _queue_state["running"]:
            return
        pending = [e for e in job_queue if e["status"] == "queued"]
        if not pending:
            return
        _queue_state["running"] = True
        entry = pending[0]
        entry["status"] = "running"

    def _run():
        try:
            _dispatch_queue_entry(entry)
            # Wait for the job to finish
            if entry.get("job_id"):
                while True:
                    safe = _get_job_copy(entry["job_id"])
                    if safe and safe.get("status") in ("complete", "error", "cancelled"):
                        entry["status"] = safe["status"]
                        break
                    time.sleep(1)
        except Exception as e:
            entry["status"] = "error"
            logger.exception("Queue processing error: %s", e)
        finally:
            with job_queue_lock:
                _queue_state["running"] = False
                # Remove completed entries
                job_queue[:] = [e for e in job_queue if e["status"] in ("queued", "running", "started")]
            # Process next
            _process_queue()

    threading.Thread(target=_run, daemon=True).start()
