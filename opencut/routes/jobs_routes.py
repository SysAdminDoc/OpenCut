"""
OpenCut Job Routes

Job status, cancel, list, SSE streaming, queue management.
"""

import json
import logging
import os
import re
import threading
import time
import uuid

from flask import Blueprint, Response, current_app, jsonify, request

from opencut.jobs import (
    _cancel_job,
    _cancel_running_jobs,
    _get_job_copy,
    _kill_job_process,
    _list_jobs_copy,
)
from opencut.queue_store import (
    QUEUE_SCHEMA_VERSION,
    QueueDocumentError,
)
from opencut.queue_store import (
    build_document as build_queue_document,
)
from opencut.queue_store import (
    load_queue as load_persisted_queue,
)
from opencut.queue_store import (
    parse_document as parse_queue_document,
)
from opencut.queue_store import (
    save_queue as save_persisted_queue,
)
from opencut.security import (
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    require_csrf,
    safe_bool,
    verify_destructive_confirm_token,
)

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


@jobs_bp.route("/jobs/<job_id>", methods=["GET"])
def job_detail(job_id):
    """Return one live or persisted job record."""
    safe = _get_job_copy(job_id)
    if safe:
        return jsonify(safe)
    try:
        from opencut.job_store import get_job as db_get_job

        persisted = db_get_job(job_id)
    except ImportError:
        persisted = None
    if not persisted:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(persisted)


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
                # 'interrupted' is set on startup for jobs that were running
                # when the server died. It's a terminal state — don't keep
                # an SSE connection alive trying to watch progress that
                # will never come.
                if status in ("complete", "error", "cancelled", "interrupted"):
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
_queue_persistence_enabled = False
_queue_app = None
_queue_storage_error = None
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
    # ``/video/object/remove`` is stale — actual route is /video/remove/watermark
    # (already listed in the v1.9.0 batch below). Removing prevents the
    # queue from emitting a misleading 400 ``Endpoint not queueable`` for
    # callers that try to use the obsolete name.
    "/video/watermark",
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
    "/timeline/export-from-markers", "/timeline/beat-cut", "/search/index",
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
    # v1.17.0 additions
    "/video/interpolate/neural",
    "/compose/render",
    # v1.18.0 additions
    "/audio/tts/f5",
    "/audio/beats/beatnet",
    "/video/scenes/auto",
    "/video/quality/score",
    "/video/quality/rank",
    "/video/emotion/arc",
    "/video/encode/vmaf-target",
    "/events/moments",
    # v1.19.0 additions
    "/video/matte/birefnet",
    "/video/encode/svtav1-psy",
    "/video/restore/colorize",
    "/video/restore/vrt",
    "/video/restore/deflicker",
    # v1.19.1 additions — AAF/OTIOZ promoted to async_job
    "/timeline/export/aaf",
    "/timeline/export/otioz",
    # v1.20.0 additions — Wave C (observability + quality harness + diff)
    "/timeline/otio-diff",
    "/video/quality/compare",
    "/video/quality/batch-compare",
    # v1.21.0 additions — Wave D (delivery + colour + voice grammar)
    "/video/encode/vvc",
    "/video/stream/srt/start",
    "/video/scopes/pro",
    # v1.22.0 additions — Wave E (Shaka packaging async)
    "/delivery/shaka/package",
    # v1.25.0 additions — Wave H (commercial parity)
    "/analyze/virality",
    "/analyze/virality/rank",
    "/video/cursor-zoom/resolve",
    # v1.28.2 additions — Wave K async_job routes
    "/audio/watermark/embed", "/audio/watermark/detect",
    "/settings/brand-kit/preview", "/video/reframe/batch",
    "/audio/censor/profanity", "/audio/spectral-match",
    "/video/lottie/render", "/search/ai", "/search/ai/index",
    # v1.29.0 additions — Wave L async_job routes
    "/audio/tts/elevenlabs",
    "/audio/tts/spark",
    "/audio/transcribe/moonshine",
    "/audio/music/acestep",
    "/audio/music/acestep/edit",
    "/generate/framepack",
    "/video/upscale/smart",
    "/video/face/reshape",
    "/video/face/retouch",
    # v1.30.0 additions — Wave M (K-stubs filled)
    "/video/dub",
    "/video/highlights/sports",
    # v1.33.0 additions — L2.4 VidMuse + L3.2/L3.5 promotions
    "/audio/music/vidmuse",
    # M1 additions — Chatterbox + Kokoro TTS + DiffRhythm music
    "/audio/tts/chatterbox",
    "/audio/tts/kokoro",
    "/audio/music/diffrhythm",
    # M2 — FLUX Kontext + Wan2.2 video generation
    "/image/edit/kontext",
    "/generate/wan2.2/t2v",
    "/generate/wan2.2/i2v",
    "/generate/wan2.2/s2v",
    "/generate/wan2.2/animate",
    # M3.2 — Digital twin pipeline
    "/pipeline/digital_twin",
    # N2.1 — SAM 2.1 video segmentation
    "/video/segment/sam2",
    # Wave N — inference accel + scene intel + content intel
    "/generate/wan2.2/fast",
    "/generate/wan2.2/i2v/quantized",
    "/video/depth/estimate-v2",
    "/video/depth/parallax-v2",
    "/video/compose/depth_segment",
    "/generate/cogvideox",
    "/generate/cogvideox/i2v",
    "/analyze/video/vl",
    "/audio/speech/csm",
    # Wave O — TTS expansion + LTX-Video + YuE music
    "/audio/speech/dia",
    "/audio/speech/parler",
    "/generate/ltxv/t2v",
    "/generate/ltxv/i2v",
    "/generate/ltxv/extend",
    "/audio/music/yue",
    # Wave P — identity video + SOTA T2I + multimodal intel
    "/generate/consisid",
    "/generate/allegro/t2v",
    "/generate/allegro/ti2v",
    "/image/generate/hidream",
    "/image/edit/hidream",
    "/image/generate/cogview4",
    "/analyze/video/narrate",
    "/analyze/video/qa",
    "/generate/opensora2",
    # Wave Q — compositing, voice gen, infinite video
    "/video/compose/vace",
    "/audio/tts/cosyvoice",
    "/audio/tts/maskgct",
    "/image/generate/omnigen2",
    "/generate/skyreels2/t2v",
    "/generate/skyreels3/avatar",
    # Wave R — foley + lip sync + camera-controlled I2V + T2V
    "/audio/foley/ezaudio",
    "/lipsync/musetalk",
    "/generate/videox-fun",
    "/generate/mochi",
    "/generate/stepvideo",
    # Wave S — relighting, VSR, ASR, VLM, face tools
    "/video/relight/iclight",
    "/video/relight/lav",
    "/video/relight/diffrenderer",
    "/video/upscale/seedvr2",
    "/audio/transcribe/parakeet",
    "/audio/transcribe/canary",
    "/analyze/video/qwen3vl",
    "/analyze/video/internvl3",
    "/video/face/reage",
    "/audio/music/heartmula",
    # Enhance macro (RESEARCH_FEATURE_PLAN_2026-05-25 Q3)
    "/enhance/auto",
    # Shorts A/B variants (RESEARCH_FEATURE_PLAN_2026-05-25 Q8)
    "/shorts/variants",
    # SAM 3 text-prompted object removal
    "/video/object-remove/sam3",
    # Collapsed stubs → canonical pipelines
    "/video/trailer/generate",
    "/video/outpaint",
    "/agent/search-footage",
    "/agent/storyboard",
    # Import-by-link footage ingest
    "/search/ingest",
    # Multimodal index (transcript + OCR + audio tags)
    "/search/multimodal-index",
    # CineFocus rack focus — promoted from sync handler to async_job
    "/video/cinefocus",
})

_QUEUE_ENTRY_STATUSES = frozenset({
    "queued", "running", "started", "interrupted", "complete", "error", "cancelled",
})
_QUEUE_TRANSIENT_STATUSES = frozenset({"running", "started"})
_QUEUE_REMOVABLE_STATUSES = frozenset({"queued", "interrupted"})
_QUEUE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OUTPUT_PATH_KEYS = frozenset({
    "output", "output_file", "output_filepath", "output_path",
    "destination", "destination_file", "destination_path",
})


def _normalize_queue_entry(raw, *, require_queueable=True):
    """Return a validated, detached queue entry."""
    if not isinstance(raw, dict):
        raise ValueError("Queue entry must be a JSON object")
    queue_id = raw.get("id")
    if not isinstance(queue_id, str) or not _QUEUE_ID_RE.fullmatch(queue_id):
        raise ValueError("Queue entry id must use 1-64 letters, numbers, '_' or '-'")
    endpoint = raw.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("/") or len(endpoint) > 256:
        raise ValueError("Queue entry endpoint must be an absolute route path")
    if require_queueable and endpoint not in _ALLOWED_QUEUE_ENDPOINTS:
        raise ValueError(f"Endpoint not queueable: {endpoint}")
    payload = raw.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("Queue entry payload must be a JSON object")
    status = raw.get("status", "queued")
    if status not in _QUEUE_ENTRY_STATUSES:
        raise ValueError(f"Unsupported queue entry status: {status!r}")
    added = raw.get("added", time.time())
    if isinstance(added, bool) or not isinstance(added, (int, float)):
        raise ValueError("Queue entry added timestamp must be numeric")

    entry = {
        "id": queue_id,
        "endpoint": endpoint,
        "payload": json.loads(json.dumps(payload)),
        "status": status,
        "added": float(added),
    }
    for key in ("job_id", "error", "code"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise ValueError(f"Queue entry {key} must be a string")
            entry[key] = raw[key][:1000]
    for key in ("interrupted_at", "replayed_at"):
        if key in raw:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Queue entry {key} must be numeric")
            entry[key] = float(value)
    if "attempts" in raw:
        attempts = raw["attempts"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError("Queue entry attempts must be a non-negative integer")
        entry["attempts"] = attempts
    return entry


def _persist_queue_locked():
    """Persist the queue while ``job_queue_lock`` is held."""
    if _queue_storage_error:
        raise RuntimeError(_queue_storage_error)
    if _queue_persistence_enabled:
        save_persisted_queue(job_queue)


def _persist_queue_best_effort_locked():
    try:
        _persist_queue_locked()
    except Exception as exc:  # noqa: BLE001 - queue worker must finish coherently
        logger.error("Could not persist job queue state: %s", exc)


def _update_queue_entry(entry, status, **details):
    with job_queue_lock:
        entry["status"] = status
        entry.update(details)
        _persist_queue_best_effort_locked()


def _find_output_collisions(value):
    """Return existing output paths referenced by a queued payload."""
    collisions = []

    def _walk(item):
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).strip().lower()
                if normalized in _OUTPUT_PATH_KEYS and isinstance(child, str) and child.strip():
                    candidate = os.path.abspath(os.path.expanduser(child.strip()))
                    if os.path.exists(candidate):
                        collisions.append(candidate)
                else:
                    _walk(child)
        elif isinstance(item, list):
            for child in item:
                _walk(child)

    _walk(value)
    return list(dict.fromkeys(collisions))


def _validate_queue_replay(entry, app, *, check_output=True):
    """Revalidate a recovered entry against the live route graph and outputs."""
    try:
        normalized = _normalize_queue_entry(entry)
    except ValueError as exc:
        return {"code": "INVALID_QUEUE_ENTRY", "error": str(exc)}
    try:
        adapter = app.url_map.bind("localhost")
        endpoint_name, _view_args = adapter.match(normalized["endpoint"], method="POST")
        if endpoint_name not in app.view_functions:
            raise LookupError("route handler is unavailable")
    except Exception:  # noqa: BLE001 - Werkzeug raises several route exceptions
        return {
            "code": "QUEUE_ROUTE_UNAVAILABLE",
            "error": f"Queued route is no longer available: {normalized['endpoint']}",
        }
    collisions = _find_output_collisions(normalized["payload"]) if check_output else []
    if collisions:
        return {
            "code": "OUTPUT_COLLISION",
            "error": "A queued output already exists; choose a new output path before replaying",
            "collisions": collisions,
        }
    return None


def initialize_job_queue(app, *, start_processing=True):
    """Load the durable queue and recover in-flight entries after a restart."""
    global _queue_app, _queue_persistence_enabled, _queue_storage_error

    try:
        stored_entries, migrated = load_persisted_queue()
    except QueueDocumentError as exc:
        logger.error("Job queue was not loaded: %s", exc)
        _queue_persistence_enabled = False
        _queue_storage_error = str(exc)
        return {"loaded": 0, "interrupted": 0, "invalid": 0, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - startup remains available for recovery
        logger.error("Job queue storage could not be read: %s", exc)
        _queue_persistence_enabled = False
        _queue_storage_error = str(exc)
        return {"loaded": 0, "interrupted": 0, "invalid": 0, "error": str(exc)}

    normalized_entries = []
    seen_ids = set()
    interrupted = 0
    invalid = 0
    changed = migrated
    now = time.time()
    for raw in stored_entries:
        try:
            entry = _normalize_queue_entry(raw, require_queueable=False)
        except (TypeError, ValueError) as exc:
            invalid += 1
            changed = True
            logger.warning("Skipping invalid persisted queue entry: %s", exc)
            continue
        if entry["id"] in seen_ids:
            invalid += 1
            changed = True
            logger.warning("Skipping duplicate persisted queue id: %s", entry["id"])
            continue
        seen_ids.add(entry["id"])
        if entry["status"] in ("complete", "error", "cancelled"):
            changed = True
            continue
        if entry["endpoint"] not in _ALLOWED_QUEUE_ENDPOINTS:
            entry["status"] = "interrupted"
            entry["code"] = "QUEUE_ROUTE_UNAVAILABLE"
            entry["error"] = f"Queued route is no longer supported: {entry['endpoint']}"
            entry["interrupted_at"] = now
            interrupted += 1
            changed = True
        elif entry["status"] == "queued":
            validation_error = _validate_queue_replay(entry, app)
            if validation_error:
                entry["status"] = "interrupted"
                entry["code"] = validation_error["code"]
                entry["error"] = validation_error["error"]
                entry["interrupted_at"] = now
                interrupted += 1
                changed = True
        elif entry["status"] in _QUEUE_TRANSIENT_STATUSES:
            entry["status"] = "interrupted"
            entry["code"] = "SERVER_RESTARTED"
            entry["error"] = "The server stopped while this queued job was active"
            entry["interrupted_at"] = now
            interrupted += 1
            changed = True
        normalized_entries.append(entry)

    if len(normalized_entries) > MAX_QUEUE_SIZE:
        invalid += len(normalized_entries) - MAX_QUEUE_SIZE
        normalized_entries = normalized_entries[:MAX_QUEUE_SIZE]
        changed = True

    with job_queue_lock:
        job_queue[:] = normalized_entries
        _queue_state["running"] = False
        _queue_app = app
        _queue_persistence_enabled = True
        _queue_storage_error = None
        if changed:
            _persist_queue_best_effort_locked()

    if start_processing and any(entry["status"] == "queued" for entry in normalized_entries):
        _process_queue(app)
    return {
        "loaded": len(normalized_entries),
        "interrupted": interrupted,
        "invalid": invalid,
    }


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
    try:
        entry = _normalize_queue_entry({
            "id": str(uuid.uuid4())[:8],
            "endpoint": endpoint,
            "payload": data.get("payload", {}),
            "status": "queued",
            "added": time.time(),
        })
    except (TypeError, ValueError) as exc:
        return jsonify({
            "error": str(exc),
            "code": "INVALID_QUEUE_ENTRY",
            "suggestion": "Send a JSON-object payload compatible with the selected route.",
        }), 400
    with job_queue_lock:
        if len(job_queue) >= MAX_QUEUE_SIZE:
            return jsonify({"error": "Queue full (max 100)"}), 429
        job_queue.append(entry)
        position = len(job_queue)
        try:
            _persist_queue_locked()
        except Exception as exc:  # noqa: BLE001 - roll back an uncommitted enqueue
            job_queue.remove(entry)
            logger.error("Could not persist queued job: %s", exc)
            return jsonify({
                "error": "Could not save the queued job",
                "code": "QUEUE_STORAGE_ERROR",
                "suggestion": "Check that the OpenCut data directory is writable, then retry.",
            }), 503
    _process_queue(current_app._get_current_object())
    return jsonify({"queue_id": entry["id"], "position": position})


@jobs_bp.route("/queue/list", methods=["GET"])
def queue_list():
    """List queued jobs."""
    with job_queue_lock:
        return jsonify(list(job_queue))


@jobs_bp.route("/queue/clear", methods=["POST"])
@require_csrf
def queue_clear():
    """Clear all queued or interrupted (not running) jobs."""
    data = get_json_dict() if request.data else {}
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    with job_queue_lock:
        removable = [e for e in job_queue if e["status"] in _QUEUE_REMOVABLE_STATUSES]
        records = [
            {
                "id": entry.get("id", ""),
                "endpoint": entry.get("endpoint", ""),
                "status": entry.get("status", ""),
            }
            for entry in removable
        ]
        plan = build_destructive_plan(
            "queue.clear",
            records=records,
            metadata={
                "queued_count": sum(e["status"] == "queued" for e in removable),
                "interrupted_count": sum(e["status"] == "interrupted" for e in removable),
            },
            reversible=False,
        )
        if dry_run:
            return jsonify({"success": True, "dry_run": True, "removed": 0, "plan": plan})
        if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409
        before = list(job_queue)
        removed = len(removable)
        job_queue[:] = [e for e in job_queue if e["status"] not in _QUEUE_REMOVABLE_STATUSES]
        try:
            _persist_queue_locked()
        except Exception as exc:  # noqa: BLE001 - restore before returning failure
            job_queue[:] = before
            logger.error("Could not persist queue clear: %s", exc)
            return jsonify({
                "error": "Could not save the cleared queue",
                "code": "QUEUE_STORAGE_ERROR",
                "suggestion": "Check that the OpenCut data directory is writable, then retry.",
            }), 503
    return jsonify({"success": True, "dry_run": False, "removed": removed, "plan": plan})


@jobs_bp.route("/queue/export", methods=["GET"])
def queue_export():
    """Export the queue as a versioned JSON document."""
    with job_queue_lock:
        document = build_queue_document(job_queue)
    response = jsonify(document)
    response.headers["Content-Disposition"] = "attachment; filename=opencut-job-queue.json"
    return response


@jobs_bp.route("/queue/import", methods=["POST"])
@require_csrf
def queue_import():
    """Import validated queue entries without duplicating stable IDs."""
    data = get_json_dict()
    try:
        raw_entries, _migrated = parse_queue_document(data)
    except QueueDocumentError as exc:
        return jsonify({
            "error": str(exc),
            "code": "INVALID_QUEUE_DOCUMENT",
            "expected_schema_version": QUEUE_SCHEMA_VERSION,
        }), 400

    app = current_app._get_current_object()
    imported = []
    skipped = []
    rejected = []
    candidates = []
    document_ids = set()
    for index, raw in enumerate(raw_entries):
        try:
            entry = _normalize_queue_entry(raw)
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "error": str(exc)})
            continue
        queue_id = entry["id"]
        if queue_id in document_ids:
            skipped.append(queue_id)
            continue
        document_ids.add(queue_id)
        if entry["status"] in _QUEUE_TRANSIENT_STATUSES:
            entry["status"] = "interrupted"
            entry["code"] = "IMPORTED_ACTIVE_ENTRY"
            entry["error"] = "Imported active work requires an explicit replay"
            entry["interrupted_at"] = time.time()
        elif entry["status"] not in ("queued", "interrupted"):
            rejected.append({
                "index": index,
                "id": queue_id,
                "error": f"Status cannot be imported: {entry['status']}",
            })
            continue
        validation_error = _validate_queue_replay(
            entry,
            app,
            check_output=entry["status"] == "queued",
        )
        if validation_error:
            rejected.append({"index": index, "id": queue_id, **validation_error})
            continue
        candidates.append(entry)

    with job_queue_lock:
        existing_ids = {entry["id"] for entry in job_queue}
        new_entries = []
        for entry in candidates:
            if entry["id"] in existing_ids:
                skipped.append(entry["id"])
                continue
            if len(job_queue) + len(new_entries) >= MAX_QUEUE_SIZE:
                rejected.append({
                    "id": entry["id"],
                    "code": "QUEUE_FULL",
                    "error": f"Queue full (max {MAX_QUEUE_SIZE})",
                })
                continue
            existing_ids.add(entry["id"])
            new_entries.append(entry)
            imported.append(entry["id"])
        job_queue.extend(new_entries)
        try:
            _persist_queue_locked()
        except Exception as exc:  # noqa: BLE001 - imports are transactional
            if new_entries:
                del job_queue[-len(new_entries):]
            logger.error("Could not persist queue import: %s", exc)
            return jsonify({
                "error": "Could not save the imported queue",
                "code": "QUEUE_STORAGE_ERROR",
                "suggestion": "Check that the OpenCut data directory is writable, then retry.",
            }), 503

    if any(entry["status"] == "queued" for entry in candidates):
        _process_queue(app)
    return jsonify({
        "schema_version": QUEUE_SCHEMA_VERSION,
        "imported": imported,
        "skipped": list(dict.fromkeys(skipped)),
        "rejected": rejected,
    })


@jobs_bp.route("/queue/replay/<queue_id>", methods=["POST"])
@require_csrf
def queue_replay(queue_id):
    """Replay an interrupted entry after revalidating its route and outputs."""
    app = current_app._get_current_object()
    with job_queue_lock:
        entry = next((item for item in job_queue if item.get("id") == queue_id), None)
        if entry is None:
            return jsonify({"error": "Queue entry not found"}), 404
        if entry.get("status") != "interrupted":
            return jsonify({
                "error": "Only interrupted queue entries can be replayed",
                "code": "QUEUE_ENTRY_NOT_INTERRUPTED",
            }), 409
        validation_error = _validate_queue_replay(entry, app)
        if validation_error:
            return jsonify(validation_error), 409

        before = dict(entry)
        original_index = job_queue.index(entry)
        job_queue.remove(entry)
        entry["status"] = "queued"
        entry["added"] = time.time()
        entry["replayed_at"] = entry["added"]
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        for key in ("job_id", "error", "code", "interrupted_at"):
            entry.pop(key, None)
        job_queue.append(entry)
        try:
            _persist_queue_locked()
        except Exception as exc:  # noqa: BLE001 - restore exact record and ordering
            job_queue.remove(entry)
            entry.clear()
            entry.update(before)
            job_queue.insert(original_index, entry)
            logger.error("Could not persist queue replay: %s", exc)
            return jsonify({
                "error": "Could not save the replayed queue entry",
                "code": "QUEUE_STORAGE_ERROR",
                "suggestion": "Check that the OpenCut data directory is writable, then retry.",
            }), 503

    _process_queue(app)
    return jsonify({"queue_id": queue_id, "status": "queued"})


def _dispatch_queue_entry(entry, app):
    """Dispatch a queue entry by calling the route handler directly
    via Flask's test_request_context (no HTTP round-trip).
    Includes the CSRF token so @require_csrf doesn't reject the call."""
    from opencut.security import get_csrf_token
    endpoint = entry.get("endpoint", "")
    payload = entry.get("payload", {})

    dispatch_timeout = 60  # seconds max for route handler to return a job_id

    csrf_token = get_csrf_token()

    try:
        # Look up the route function (needs a request context for url_map)
        with app.test_request_context(endpoint, method="POST",
                                      json=payload,
                                      headers={
                                          "Content-Type": "application/json",
                                          "X-OpenCut-Token": csrf_token,
                                      }):
            adapter = app.url_map.bind("")
            ep_name, view_args = adapter.match(endpoint, method="POST")
            view_func = app.view_functions.get(ep_name)
        if view_func is None:
            _update_queue_entry(
                entry,
                "error",
                code="QUEUE_ROUTE_UNAVAILABLE",
                error=f"Queued route is unavailable: {endpoint}",
            )
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
            _update_queue_entry(
                entry,
                "error",
                code="QUEUE_DISPATCH_TIMEOUT",
                error=f"Queue dispatch timed out after {dispatch_timeout} seconds",
            )
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
            _update_queue_entry(
                entry,
                "error",
                error=result.get("error") or f"Route failed with HTTP {status_code}",
                code=result.get("code", ""),
            )
            return
        job_id = result.get("job_id", "")
        if not job_id:
            _update_queue_entry(
                entry,
                "error",
                error=result.get("error") or "Route did not return a job ID",
                code=result.get("code", ""),
            )
            return
        _update_queue_entry(entry, "started", job_id=job_id)
    except Exception as e:
        _update_queue_entry(
            entry,
            "error",
            code="QUEUE_DISPATCH_ERROR",
            error="The queued route could not be started",
        )
        logger.exception("Queue dispatch error for %s: %s", endpoint, e)


def _process_queue(app=None):
    """Process the next item in the queue (fire-and-forget)."""
    resolved_app = app or _queue_app
    if resolved_app is None:
        try:
            resolved_app = current_app._get_current_object()
        except RuntimeError:
            logger.error("Cannot process queue without a Flask application")
            return
    with job_queue_lock:
        if _queue_state["running"]:
            return
        pending = [e for e in job_queue if e["status"] == "queued"]
        if not pending:
            return
        _queue_state["running"] = True
        entry = pending[0]
        entry["status"] = "running"
        try:
            _persist_queue_locked()
        except Exception as exc:  # noqa: BLE001 - do not run work we cannot recover
            entry["status"] = "queued"
            _queue_state["running"] = False
            logger.error("Queue processing paused because state could not be saved: %s", exc)
            return

    def _run():
        try:
            _dispatch_queue_entry(entry, resolved_app)
            # Wait for the job to finish (timeout after 30 minutes)
            with job_queue_lock:
                job_id = entry.get("job_id")
                entry_status = entry.get("status")
            if job_id:
                deadline = time.time() + 1800
                while time.time() < deadline:
                    # Call _get_job_copy outside job_queue_lock to avoid nested lock deadlock
                    safe = _get_job_copy(job_id)
                    if safe and safe.get("status") in ("complete", "error", "cancelled"):
                        _update_queue_entry(entry, safe["status"])
                        break
                    time.sleep(1)
                else:
                    _update_queue_entry(
                        entry,
                        "error",
                        code="QUEUE_JOB_TIMEOUT",
                        error="Queued job timed out after 30 minutes",
                    )
                    logger.warning("Queue job %s timed out after 30 minutes", job_id)
            elif entry_status not in ("started", "error"):
                _update_queue_entry(
                    entry,
                    "error",
                    code="QUEUE_JOB_ID_MISSING",
                    error="Queued route did not start a trackable job",
                )
        except Exception as e:
            _update_queue_entry(
                entry,
                "error",
                code="QUEUE_PROCESSING_ERROR",
                error="The queued job could not be processed",
            )
            logger.exception("Queue processing error: %s", e)
        finally:
            with job_queue_lock:
                _queue_state["running"] = False
                # Live jobs are represented by the job history once terminal.
                # Keep interrupted entries visible until the user replays or
                # clears them.
                job_queue[:] = [
                    e for e in job_queue
                    if e["status"] in ("queued", "running", "started", "interrupted")
                ]
                _persist_queue_best_effort_locked()
            # Process next
            _process_queue(resolved_app)

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
    limit = safe_int(request.args.get("limit", 50), default=50, min_val=1, max_val=200)
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


@jobs_bp.route("/jobs/db-diagnostics", methods=["GET"])
def job_db_diagnostics():
    """Return read-only diagnostics for the persisted job SQLite store."""
    try:
        from opencut.job_store import get_db_diagnostics
        return jsonify(get_db_diagnostics())
    except ImportError:
        return jsonify({"error": "Job history is unavailable"}), 503
    except Exception as exc:
        logger.exception("job_db_diagnostics failed")
        return jsonify({"error": str(exc)}), 500


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


def _validate_resume_endpoint(endpoint: str):
    if not isinstance(endpoint, str) or not endpoint.startswith("/") or endpoint.startswith("//"):
        return "Resume endpoint is missing or invalid."
    if "\x00" in endpoint or "://" in endpoint:
        return "Resume endpoint must be an internal OpenCut route."
    try:
        adapter = current_app.url_map.bind("")
        endpoint_name, _view_args = adapter.match(endpoint, method="POST")
    except Exception:
        return "Resume endpoint is no longer registered."
    view_func = current_app.view_functions.get(endpoint_name)
    if view_func is None or not getattr(view_func, "_opencut_async_job", False):
        return "Resume endpoint is not an async job route."
    if not getattr(view_func, "_opencut_resumable", False):
        return "Resume endpoint is not marked resumable."
    return ""


@jobs_bp.route("/jobs/<job_id>/resume", methods=["POST"])
@require_csrf
def resume_job(job_id):
    """Resume an interrupted, checkpointable job from persisted payload data."""
    try:
        from opencut.job_store import get_job as db_get_job
        from opencut.security import get_csrf_token
    except ImportError:
        return jsonify({"error": "Job history is unavailable", "code": "JOB_STORE_UNAVAILABLE"}), 503

    job = db_get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found", "code": "JOB_NOT_FOUND"}), 404
    if job.get("status") != "interrupted":
        return jsonify({
            "error": "Only interrupted jobs can be resumed",
            "code": "JOB_NOT_INTERRUPTED",
            "status": job.get("status"),
        }), 409
    if not job.get("resumable"):
        return jsonify({
            "error": "Job is not marked resumable",
            "code": "JOB_NOT_RESUMABLE",
            "job_id": job_id,
        }), 409

    endpoint = job.get("endpoint") or ""
    endpoint_error = _validate_resume_endpoint(endpoint)
    if endpoint_error:
        return jsonify({
            "error": endpoint_error,
            "code": "JOB_RESUME_UNAVAILABLE",
            "job_id": job_id,
            "endpoint": endpoint,
        }), 409

    payload = job.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    if job.get("filepath") and "filepath" not in payload:
        payload["filepath"] = job["filepath"]
    if not payload:
        return jsonify({
            "error": "Original job payload is unavailable",
            "code": "JOB_RESUME_UNAVAILABLE",
            "job_id": job_id,
            "endpoint": endpoint,
        }), 409

    resume_payload = dict(payload)
    partial_output_path = job.get("partial_output_path") or resume_payload.get("partial_output_path") or ""
    if partial_output_path:
        resume_payload["partial_output_path"] = partial_output_path
    resume_attempt = int(job.get("resume_attempt") or 0) + 1
    resume_payload["resume_source_job_id"] = job_id
    resume_payload["resume_from_job_id"] = job_id
    resume_payload["resume_attempt"] = resume_attempt

    inner = current_app.test_client().post(
        endpoint,
        json=resume_payload,
        headers={
            "Content-Type": "application/json",
            "X-OpenCut-Token": get_csrf_token(),
        },
    )
    inner_body = inner.get_json(silent=True)
    if not isinstance(inner_body, dict):
        inner_body = {}
    if inner.status_code >= 400:
        return jsonify({
            "error": "Resume dispatch failed",
            "code": "JOB_RESUME_FAILED",
            "job_id": job_id,
            "endpoint": endpoint,
            "status_code": inner.status_code,
            "details": inner_body,
        }), inner.status_code

    resumed_job_id = inner_body.get("job_id", "")
    if not resumed_job_id:
        return jsonify({
            "error": "Resume endpoint did not return a job ID",
            "code": "JOB_RESUME_FAILED",
            "job_id": job_id,
            "endpoint": endpoint,
            "details": inner_body,
        }), 502

    return jsonify({
        "success": True,
        "job_id": resumed_job_id,
        "resumed_job_id": resumed_job_id,
        "source_job_id": job_id,
        "endpoint": endpoint,
        "partial_output_path": partial_output_path,
        "resume_attempt": resume_attempt,
    }), 202


@jobs_bp.route("/jobs/<job_id>/diagnostics", methods=["GET"])
def job_diagnostics(job_id: str):
    """Return a diagnostic payload for ``job_id`` (F010).

    Combines the persisted job store row with the in-memory job state
    and a relevant slice of ``opencut.log`` filtered to lines that
    mention the job_id or its request_id. The response is scrubbed of
    home directory paths via :mod:`opencut.core.issue_report`.
    """
    try:
        from flask import request as _request

        from opencut.core.job_diagnostics import build_diagnostic
        from opencut.errors import safe_error as _safe_error

        log_tail = _request.args.get("log_tail_lines", "200")
        try:
            log_tail_lines = int(log_tail)
        except (TypeError, ValueError):
            log_tail_lines = 200

        diag = build_diagnostic(job_id, log_tail_lines=log_tail_lines)
        payload = diag.as_dict()
        status = 200 if diag.found else 404
        return jsonify(payload), status
    except Exception as exc:  # pragma: no cover - defensive
        from opencut.errors import safe_error as _safe_error
        return _safe_error(exc, "job_diagnostics")
