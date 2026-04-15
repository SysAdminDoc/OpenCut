"""
OpenCut Job Routes

Job status, cancel, list, SSE streaming, queue management.
"""

import json
import logging
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, request

from opencut.jobs import (
    _cancel_job,
    _cancel_running_jobs,
    _get_job_copy,
    _kill_job_process,
    _list_jobs_copy,
    job_lock,
    jobs,
)
from opencut.security import get_json_dict, require_csrf

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
    _job, state = _cancel_job(job_id)
    if state == "not_found":
        return jsonify({"error": "Job not found"}), 404
    if state == "not_running":
        return jsonify({"error": "Job is not running"}), 400
    _kill_job_process(job_id)
    return jsonify({"status": "cancelled", "job_id": job_id})


@jobs_bp.route("/cancel-all", methods=["POST"])
@require_csrf
def cancel_all_jobs():
    """Cancel all running jobs."""
    cancelled = _cancel_running_jobs()
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
_sse_state = {"connections": 0}
_sse_lock = threading.Lock()
MAX_SSE_CONNECTIONS = 20


@jobs_bp.route("/stream/<job_id>", methods=["GET"])
def stream_job(job_id):
    """Stream job status via Server-Sent Events. Replaces polling."""
    with _sse_lock:
        if _sse_state["connections"] >= MAX_SSE_CONNECTIONS:
            return jsonify({"error": "Too many streaming connections"}), 429
        # Increment inside the same lock acquisition that checks the limit,
        # preventing a race where multiple requests pass the check concurrently.
        _sse_state["connections"] += 1

    def generate():
        try:
            deadline = time.time() + 1800  # 30 minute timeout
            while time.time() < deadline:
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
        finally:
            with _sse_lock:
                _sse_state["connections"] -= 1

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    # CEP panels run with origin "null" or "file://" depending on setup;
    # safe because the server only binds to 127.0.0.1.
    req_origin = request.headers.get("Origin", "null")
    if req_origin in ("null", "file://"):
        resp.headers["Access-Control-Allow-Origin"] = req_origin
    else:
        resp.headers["Access-Control-Allow-Origin"] = "null"
    return resp


# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------
job_queue = []
job_queue_lock = threading.Lock()
_queue_state = {"running": False}
MAX_QUEUE_SIZE = 100

# Only processing-oriented routes may be invoked via the queue.
_ALLOWED_QUEUE_ENDPOINTS = frozenset({
    "/silence", "/silence/speed-up", "/fillers",
    "/audio/denoise", "/audio/normalize", "/audio/enhance",
    "/audio/pro/apply", "/audio/separate", "/audio/tts/generate",
    "/audio/tts/subtitled", "/audio/duck",
    "/styled-captions", "/captions/translate", "/captions/karaoke",
    "/captions/burnin/file", "/captions/burnin/segments",
    "/captions/animated/render", "/transcript/summarize",
    "/video/scenes", "/video/auto-edit", "/video/fx/apply",
    "/video/face/blur", "/video/face/enhance", "/video/face/swap",
    "/video/reframe", "/video/reframe/face",
    "/video/chromakey", "/video/highlights",
    "/video/lut/apply", "/video/lut/generate-from-ref",
    "/video/color/correct", "/video/color/convert",
    "/video/speed/ramp", "/video/shorts-pipeline",
    "/video/title/render", "/video/title/overlay", "/video/preview-frame",
    "/video/pip", "/video/blend", "/video/merge", "/video/trim",
    "/video/object/remove", "/video/watermark",
    "/export-video",
    "/video/ai/upscale", "/video/ai/denoise",
    "/video/style/apply", "/video/style/arbitrary", "/video/ai/rembg",
    "/video/lut/generate-ai", "/video/lut/blend",
    "/audio/music-ai/ace-step",
    "/audio/music-ai/stable-audio",
    # v1.5.0 additions
    "/audio/beat-markers", "/audio/loudness-match",
    "/captions/chapters", "/captions/repeat-detect",
    "/video/color-match", "/video/auto-zoom", "/video/multicam-cuts",
    "/timeline/export-from-markers", "/search/index",
    # v2.0 additions
    "/workflow/run",
    # v1.9.0 additions
    "/video/ai/interpolate",
    "/video/depth/map", "/video/depth/bokeh", "/video/depth/parallax",
    "/video/broll-plan",
    "/video/remove/watermark",
    "/video/upscale/run",
    "/video/multicam-xml",
    # v1.9.18 additions
    "/captions",
    "/captions/whisperx",
    "/full",
    "/transcript",
    "/video/ai/install",
    "/video/color/convert",
    "/video/color/correct",
    "/video/color/external-lut",
    "/video/emotion-highlights",
    "/video/face/blur",
    "/video/face/swap",
    "/video/lut/generate-ai",
    "/video/lut/generate-all",
    "/video/lut/generate-from-ref",
    "/video/particles/apply",
    "/video/speed/change",
    "/video/speed/ramp",
    "/video/speed/reverse",
    # v1.9.20 additions — previously missing async routes
    "/audio/beats",
    "/audio/duck-video",
    "/audio/effects/apply",
    "/audio/gen/sfx",
    "/audio/gen/tone",
    "/audio/isolate",
    "/audio/mix",
    "/audio/mix-duck",
    "/audio/music-ai/generate",
    "/audio/music-ai/melody",
    "/audio/pro/deepfilter",
    "/audio/waveform",
    "/export/preset",
    "/export/thumbnails",
    "/social/upload",
    "/video/broll-generate",
    "/video/multimodal-diarize",
    "/video/transitions/apply",
    "/video/transitions/join",
})


@jobs_bp.route("/queue/add", methods=["POST"])
@require_csrf
def queue_add():
    """Add a job to the queue."""
    try:
        data = get_json_dict()
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "code": "INVALID_INPUT",
            "suggestion": "Send a top-level JSON object in the request body.",
        }), 400
    endpoint = data.get("endpoint", "")
    if endpoint not in _ALLOWED_QUEUE_ENDPOINTS:
        return jsonify({"error": f"Endpoint not queueable: {endpoint}"}), 400
    entry = {
        "id": str(uuid.uuid4())[:8],
        "endpoint": endpoint,
        "payload": data.get("payload", {}),
        "status": "queued",
        "added": time.time(),
    }
    with job_queue_lock:
        if len(job_queue) >= MAX_QUEUE_SIZE:
            return jsonify({"error": "Queue full (max 100)"}), 429
        job_queue.append(entry)
        position = len(job_queue)
    _process_queue()
    return jsonify({"queue_id": entry["id"], "position": position})


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
    via Flask's test_request_context (no HTTP round-trip).
    Includes the CSRF token so @require_csrf doesn't reject the call."""
    from flask import current_app

    from opencut.security import get_csrf_token
    endpoint = entry.get("endpoint", "")
    payload = entry.get("payload", {})

    dispatch_timeout = 60  # seconds max for route handler to return a job_id

    app = current_app._get_current_object()
    csrf_token = get_csrf_token()

    try:
        # Look up the route function (needs a request context for url_map)
        with app.test_request_context(endpoint, method="POST",
                                      json=payload,
                                      headers={
                                          "Content-Type": "application/json",
                                          "X-OpenCut-Token": csrf_token,
                                      }):
            adapter = current_app.url_map.bind("")
            ep_name, view_args = adapter.match(endpoint, method="POST")
            view_func = current_app.view_functions.get(ep_name)
        if view_func is None:
            entry["status"] = "error"
            return

        # Run the handler in a sub-thread with its own request context
        _dispatch_result = [None, None]  # [response, exception]

        def _call():
            with app.test_request_context(endpoint, method="POST",
                                          json=payload,
                                          headers={
                                              "Content-Type": "application/json",
                                              "X-OpenCut-Token": csrf_token,
                                          }):
                try:
                    _dispatch_result[0] = view_func(**view_args)
                except Exception as exc:
                    _dispatch_result[1] = exc

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=dispatch_timeout)
        if t.is_alive():
            entry["status"] = "error"
            logger.warning("Queue dispatch timed out after %ds for %s", dispatch_timeout, endpoint)
            return
        if _dispatch_result[1]:
            raise _dispatch_result[1]

        resp = _dispatch_result[0]
        # Flask view functions return (response, status) or a Response
        if isinstance(resp, tuple):
            resp_obj = resp[0]
            status_code = resp[1] if len(resp) > 1 else 200
        else:
            resp_obj = resp
            status_code = getattr(resp_obj, "status_code", 200)
        result = resp_obj.get_json() if hasattr(resp_obj, "get_json") else {}
        if not isinstance(result, dict):
            result = {}
        if status_code >= 400:
            entry["status"] = "error"
            entry["error"] = result.get("error") or f"Route failed with HTTP {status_code}"
            entry["code"] = result.get("code", "")
            return
        job_id = result.get("job_id", "")
        if not job_id:
            entry["status"] = "error"
            entry["error"] = result.get("error") or "Route did not return a job ID"
            entry["code"] = result.get("code", "")
            return
        entry["job_id"] = job_id
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
            # Wait for the job to finish (timeout after 30 minutes)
            job_id = entry.get("job_id")
            if job_id:
                deadline = time.time() + 1800
                while time.time() < deadline:
                    # Call _get_job_copy outside job_queue_lock to avoid nested lock deadlock
                    safe = _get_job_copy(job_id)
                    if safe and safe.get("status") in ("complete", "error", "cancelled"):
                        with job_queue_lock:
                            entry["status"] = safe["status"]
                        break
                    time.sleep(1)
                else:
                    with job_queue_lock:
                        entry["status"] = "error"
                    logger.warning("Queue job %s timed out after 30 minutes", job_id)
            elif entry.get("status") not in ("started", "error"):
                with job_queue_lock:
                    entry["status"] = "error"
        except Exception as e:
            with job_queue_lock:
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


# ---------------------------------------------------------------------------
# Job History (SQLite-backed persistent storage)
# ---------------------------------------------------------------------------

@jobs_bp.route("/jobs/history", methods=["GET"])
def job_history():
    """Return historical jobs from persistent storage.

    Query params:
        status (str): Filter by status (complete, error, cancelled, interrupted)
        limit (int): Max results (default 50, max 200)
        offset (int): Pagination offset
    """
    try:
        from opencut.job_store import list_jobs as db_list_jobs
    except ImportError:
        return jsonify([])

    from opencut.security import safe_int
    status_filter = request.args.get("status", None)
    limit = min(safe_int(request.args.get("limit", 50), default=50, min_val=1, max_val=200), 200)
    offset = safe_int(request.args.get("offset", 0), default=0, min_val=0)
    results = db_list_jobs(status=status_filter, limit=limit, offset=offset)
    return jsonify(results)


@jobs_bp.route("/jobs/stream-result/<job_id>", methods=["GET"])
def stream_job_result(job_id):
    """Stream a completed job's result as NDJSON.

    Useful for large results (caption segments, scene lists, thumbnails)
    that would be too large for a single JSON response.

    The job must be complete. Returns 404 if not found, 409 if still running.
    """
    safe = _get_job_copy(job_id)
    if not safe:
        return jsonify({"error": "Job not found"}), 404

    if safe.get("status") != "complete":
        return jsonify({"error": "Job not yet complete", "status": safe.get("status")}), 409

    result = safe.get("result", {})
    if not result:
        return jsonify({"error": "Job has no result data"}), 404

    # Find the streamable array in the result
    stream_data = None
    for key in ("segments", "scenes", "thumbnails", "cuts", "results",
                "items", "chapters", "speakers", "keyframes"):
        if key in result and isinstance(result[key], list):
            stream_data = result[key]
            break

    if stream_data is None:
        return jsonify(result)

    try:
        from opencut.core.streaming import make_ndjson_response, ndjson_generator
        gen = ndjson_generator(stream_data, chunk_size=50)
        return make_ndjson_response(gen, Response)
    except ImportError:
        return jsonify(result)


@jobs_bp.route("/jobs/stats", methods=["GET"])
def job_stats():
    """Return aggregate job statistics."""
    try:
        from opencut.job_store import get_job_stats
        return jsonify(get_job_stats())
    except ImportError:
        return jsonify({"total": 0})


@jobs_bp.route("/jobs/interrupted", methods=["GET"])
def interrupted_jobs():
    """Return jobs that were interrupted by a server restart.

    The frontend can offer to retry these.
    """
    try:
        from opencut.job_store import get_interrupted_jobs
        return jsonify(get_interrupted_jobs())
    except ImportError:
        return jsonify([])
