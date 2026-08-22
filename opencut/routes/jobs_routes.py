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

import opencut.jobs as jobs_module
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
    QueueSchemaVersionError,
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

# Only processing-oriented routes may be invoked via the queue. This is the
# full set of async POST routes minus _QUEUE_EXCLUDED_ENDPOINTS below, and it
# is pinned by tests/test_queue_coverage.py: adding an @async_job POST route
# without listing it in one of the two sets fails the suite, so a route can no
# longer fall out of the queue silently.
_ALLOWED_QUEUE_ENDPOINTS = frozenset({
    "/accessibility/colorblind-sim",
    "/accessibility/flash-detect",
    "/adr/cue-sheet",
    "/adr/guide",
    "/adr/sync",
    "/ai-gen/extend-scene",
    "/ai-gen/img-to-video",
    "/ai-gen/outpaint",
    "/ai-gen/replace-bg",
    "/ai-gen/summarize",
    "/ai/audio/classify",
    "/ai/audio/classify-timeline",
    "/ai/auto-grade",
    "/ai/broll-suggest",
    "/ai/character/create",
    "/ai/character/generate",
    "/ai/color-match",
    "/ai/color-match/batch",
    "/ai/content-scan",
    "/ai/deepfake-detect",
    "/ai/engagement-predict",
    "/ai/extend-spatial",
    "/ai/extend-temporal",
    "/ai/eye-contact",
    "/ai/foley/detect-cues",
    "/ai/foley/generate",
    "/ai/generate-broll",
    "/ai/generate-broll/batch",
    "/ai/lip-sync",
    "/ai/morph-cut",
    "/ai/music/fit-duration",
    "/ai/music/remix",
    "/ai/overdub",
    "/ai/pacing-analysis",
    "/ai/seo-optimize",
    "/ai/storyboard",
    "/ai/video-llm/find-moment",
    "/ai/video-llm/query",
    "/ai/voice-avatar",
    "/ai/voice-convert",
    "/analysis/shot-classify",
    "/analyze/video/internvl3",
    "/analyze/video/narrate",
    "/analyze/video/qa",
    "/analyze/video/qwen3vl",
    "/analyze/video/vl",
    "/analyze/virality",
    "/analyze/virality/rank",
    "/api/adjustment-layers/apply",
    "/api/ai/batch-command",
    "/api/ai/emotion-timeline",
    "/api/ai/ocr",
    "/api/ai/organize-project",
    "/api/ai/scene-describe",
    "/api/ai/summarize",
    "/api/audio/auto-dub",
    "/api/audio/generate-sfx",
    "/api/audio/overdub",
    "/api/audio/overdub/clone-voice",
    "/api/audio/spatial",
    "/api/audio/stem-remix",
    "/api/audio/transition-sfx",
    "/api/audio/voice-convert",
    "/api/audio/voice-convert/profile",
    "/api/batch/execute",
    "/api/dev/macro/play",
    "/api/display/test-pattern",
    "/api/drone/flight-map",
    "/api/drone/hyperlapse",
    "/api/drone/map-overlay",
    "/api/education/pip-detect",
    "/api/education/pip-extract",
    "/api/education/pip-side-by-side",
    "/api/education/quiz-insert",
    "/api/education/quiz-render",
    "/api/education/slide-detect",
    "/api/education/slide-extract",
    "/api/encoding/intermediate",
    "/api/filter-chain/preview",
    "/api/fonts/download",
    "/api/lens/auto-detect",
    "/api/lens/chromatic-aberration",
    "/api/lens/correct-distortion",
    "/api/motion/data-animation",
    "/api/motion/expression/evaluate",
    "/api/motion/expression/timeline",
    "/api/motion/kinetic-text",
    "/api/motion/kinetic-text/preview",
    "/api/motion/particles",
    "/api/motion/particles/preview",
    "/api/motion/shape-animate",
    "/api/pipeline/estimate",
    "/api/pipeline/estimate/batch",
    "/api/podcast/extract-audio",
    "/api/project/archive",
    "/api/project/restore",
    "/api/sfx/download",
    "/api/sfx/search",
    "/api/timeline/assemble",
    "/api/timeline/auto-mix",
    "/api/timeline/auto-mix/preview",
    "/api/timeline/batch-ops",
    "/api/timeline/batch-ops/preview",
    "/api/timeline/narrative",
    "/api/timeline/quality",
    "/api/timeline/rough-cut",
    "/api/timeline/score",
    "/api/timeline/smart-trim",
    "/api/timeline/smart-trim/batch",
    "/api/transcript/edit",
    "/api/transcript/parse",
    "/api/transcript/preview",
    "/api/tutorial/callout",
    "/api/tutorial/click-overlay",
    "/api/tutorial/keystroke-overlay",
    "/api/tutorial/screenshot-video",
    "/api/tutorial/spotlight",
    "/api/version/compare",
    "/api/video/cinemagraph",
    "/api/video/classify-content",
    "/api/video/color-intent",
    "/api/video/color-intent/preview",
    "/api/video/eye-contact",
    "/api/video/eye-contact/preview",
    "/api/video/hyperlapse",
    "/api/video/lip-sync",
    "/api/video/lip-sync/preview",
    "/api/video/log-apply",
    "/api/video/log-detect",
    "/api/video/lut-stack",
    "/api/video/sky-replace",
    "/api/vr/extract-fov",
    "/api/vr/reframe",
    "/api/vr/spatial-audio",
    "/api/vr/stabilize",
    "/audio/adr/record",
    "/audio/ambient/generate",
    "/audio/analyze-features",
    "/audio/beat-detect",
    "/audio/beat-markers",
    "/audio/beat-sync",
    "/audio/beats",
    "/audio/beats/beatnet",
    "/audio/censor/profanity",
    "/audio/declip",
    "/audio/decrackle",
    "/audio/dehum",
    "/audio/denoise",
    "/audio/dereverb",
    "/audio/description/gaps",
    "/audio/description/generate",
    "/audio/description/microsoft-draft",
    "/audio/description/synthesize",
    "/audio/dewind",
    "/audio/dialogue-premix",
    "/audio/dialogue-premix/basic",
    "/audio/duck",
    "/audio/duck-video",
    "/audio/effects/apply",
    "/audio/energy-envelope",
    "/audio/enhance",
    "/audio/enhance-speech",
    "/audio/enhance-speech/preview",
    "/audio/fingerprint",
    "/audio/fingerprint/add",
    "/audio/fingerprint/scan",
    "/audio/foley/analyze",
    "/audio/foley/ezaudio",
    "/audio/foley/place",
    "/audio/gen/sfx",
    "/audio/gen/tone",
    "/audio/isolate",
    "/audio/loudness",
    "/audio/loudness-check",
    "/audio/loudness-match",
    "/audio/ltc-extract",
    "/audio/me-mix",
    "/audio/me-mix/basic",
    "/audio/mix",
    "/audio/mix-duck",
    "/audio/mood-morph",
    "/audio/music-ai/ace-step",
    "/audio/music-ai/generate",
    "/audio/music-ai/melody",
    "/audio/music-ai/stable-audio",
    "/audio/music/acestep",
    "/audio/music/acestep/edit",
    "/audio/music/diffrhythm",
    "/audio/music/heartmula",
    "/audio/music/vidmuse",
    "/audio/music/yue",
    "/audio/normalize",
    "/audio/pro/apply",
    "/audio/pro/deepfilter",
    "/audio/pro/install",
    "/audio/room-tone/extract",
    "/audio/room-tone/fill",
    "/audio/room-tone/generate",
    "/audio/separate",
    "/audio/sound-design",
    "/audio/spectral-match",
    "/audio/spectrum",
    "/audio/speech/csm",
    "/audio/speech/dia",
    "/audio/speech/parler",
    "/audio/stem-remix",
    "/audio/stem-remix/preview",
    "/audio/surround-upmix",
    "/audio/surround/downmix",
    "/audio/surround/export",
    "/audio/surround/pan",
    "/audio/surround/upmix",
    "/audio/transcribe/canary",
    "/audio/transcribe/moonshine",
    "/audio/transcribe/parakeet",
    "/audio/tts/chatterbox",
    "/audio/tts/cosyvoice",
    "/audio/tts/elevenlabs",
    "/audio/tts/f5",
    "/audio/tts/generate",
    "/audio/tts/install",
    "/audio/tts/kokoro",
    "/audio/tts/maskgct",
    "/audio/tts/spark",
    "/audio/tts/subtitled",
    "/audio/visualizer",
    "/audio/watermark/detect",
    "/audio/watermark/embed",
    "/audio/waveform",
    "/audiogram/generate",
    "/batch-conform/run",
    "/batch-metadata/template",
    "/batch-metadata/write",
    "/batch/contact-sheet",
    "/batch/thumbnails",
    "/batch/transcode",
    "/beat-cuts/assemble",
    "/beat-cuts/generate",
    "/brand/auto-correct",
    "/brand/check",
    "/brand/load",
    "/camera-solver/export",
    "/camera-solver/ground-plane",
    "/camera-solver/render",
    "/camera-solver/solve",
    "/caption/compliance",
    "/caption/compliance/fix",
    "/captions",
    "/captions/animated/render",
    "/captions/burnin/file",
    "/captions/burnin/segments",
    "/captions/chapters",
    "/captions/enhanced/install",
    "/captions/export/essential-graphics",
    "/captions/export/premiere-xml",
    "/captions/karaoke",
    "/captions/repeat-detect",
    "/captions/style/apply",
    "/captions/style/preview",
    "/captions/translate",
    "/captions/whisperx",
    "/ceremony/auto-edit",
    "/ceremony/score",
    "/cleanup/apply",
    "/cleanup/preview",
    "/cloud/render",
    "/comparison/export",
    "/compose/render",
    "/composition/analyze-pacing",
    "/composition/classify-shot",
    "/composition/guide",
    "/composition/saliency-crop",
    "/conform/analyze",
    "/conform/batch",
    "/conform/clip",
    "/construction-timelapse/align",
    "/construction-timelapse/build",
    "/construction-timelapse/fill-frames",
    "/content/ab-variants",
    "/content/apply-hook",
    "/content/chapter-art",
    "/content/compare-thumbnails",
    "/content/generate-hook",
    "/content/predict-ctr",
    "/copilot/execute",
    "/copilot/query",
    "/credits/generate",
    "/data-video/batch",
    "/data-video/create",
    "/delivery/broadcast-qc",
    "/delivery/broadcast-qc/audio",
    "/delivery/broadcast-qc/report",
    "/delivery/caption/ebu-tt",
    "/delivery/caption/embed-cc",
    "/delivery/caption/ttml",
    "/delivery/dash",
    "/delivery/end-screen",
    "/delivery/fcpxml",
    "/delivery/hls",
    "/delivery/news-ticker",
    "/delivery/news-ticker/standalone",
    "/delivery/shaka/package",
    "/delivery/thumbnail-ab",
    "/delivery/transfer-bundle",
    "/delivery/validate",
    "/dubbing/emotion-transfer",
    "/dubbing/full-pipeline",
    "/dubbing/isochronous",
    "/dubbing/manage-tracks",
    "/effects/film-stock",
    "/effects/glitch",
    "/effects/glitch-sequence",
    "/effects/light-leak",
    "/effects/retro",
    "/effects/tilt-shift",
    "/engagement/predict",
    "/enhance/auto",
    "/events/moments",
    "/export-video",
    "/export/apng",
    "/export/av1",
    "/export/dcp",
    "/export/dnxhr",
    "/export/gif",
    "/export/imf",
    "/export/preset",
    "/export/prores",
    "/export/thumbnails",
    "/export/webp",
    "/farm/render",
    "/fillers",
    "/filter/preview",
    "/fingerprint/generate",
    "/fingerprint/search",
    "/frameio/upload",
    "/full",
    "/gaming/auto-montage/assemble",
    "/gaming/auto-montage/score",
    "/gaming/chat-replay",
    "/gaming/instant-replay",
    "/gaming/instant-replay/batch",
    "/gaming/iso-ingest/detect",
    "/gaming/iso-ingest/sync",
    "/gaming/iso-ingest/timeline",
    "/gaming/multi-pov/detect-audio",
    "/gaming/multi-pov/export-xml",
    "/gaming/multi-pov/sync",
    "/gaming/stream-highlights/assemble",
    "/gaming/stream-highlights/extract",
    "/gaming/stream-highlights/score",
    "/gaussian-splat/render",
    "/generate/allegro/t2v",
    "/generate/allegro/ti2v",
    "/generate/cogvideox",
    "/generate/cogvideox/i2v",
    "/generate/consisid",
    "/generate/framepack",
    "/generate/ltxv/extend",
    "/generate/ltxv/i2v",
    "/generate/ltxv/t2v",
    "/generate/mochi",
    "/generate/opensora2",
    "/generate/skyreels2/t2v",
    "/generate/skyreels3/avatar",
    "/generate/stepvideo",
    "/generate/videox-fun",
    "/generate/wan2.2/animate",
    "/generate/wan2.2/fast",
    "/generate/wan2.2/i2v",
    "/generate/wan2.2/i2v/quantized",
    "/generate/wan2.2/s2v",
    "/generate/wan2.2/t2v",
    "/guest/compile",
    "/guest/name-card",
    "/guest/process-single",
    "/hw/encode",
    "/image-sequence/assemble",
    "/image/edit/hidream",
    "/image/edit/kontext",
    "/image/generate/cogview4",
    "/image/generate/hidream",
    "/image/generate/omnigen2",
    "/ingest/run",
    "/install-whisper",
    "/interview-polish",
    "/lipsync/musetalk",
    "/lower-thirds/batch",
    "/lower-thirds/generate",
    "/media/find-duplicates",
    "/metadata/copy",
    "/metadata/strip",
    "/montage/create",
    "/montage/ken-burns",
    "/motion/data-animation",
    "/motion/expression/evaluate",
    "/motion/expression/timeline",
    "/motion/kinetic-text",
    "/motion/kinetic-text/preview",
    "/motion/particles",
    "/motion/particles/preview",
    "/motion/shape-animate",
    "/multicam-grid/export",
    "/multilang/create",
    "/multilang/export",
    "/multilang/sync",
    "/multilang/update",
    "/mxf/convert",
    "/mxf/export",
    "/object-effects/apply",
    "/object-effects/generate-mask",
    "/object-effects/preview",
    "/onnx/inference",
    "/overlay/countdown",
    "/overlay/elapsed-timer",
    "/overlay/safe-zones",
    "/overlay/timecode",
    "/paper-edit/assemble",
    "/paper-edit/create",
    "/paper-edit/export",
    "/pipeline/digital_twin",
    "/planar-track/export",
    "/planar-track/insert",
    "/planar-track/preview",
    "/planar-track/track",
    "/podcast/detect-speakers",
    "/podcast/per-speaker-process",
    "/podcast/polish",
    "/podcast/show-notes",
    "/privacy/anonymize-speaker",
    "/privacy/doc-redact",
    "/privacy/pii-redact",
    "/privacy/plate-blur",
    "/privacy/profanity-bleep",
    "/provenance/generate",
    "/proxy/auto-ingest",
    "/proxy/detect-duplicates",
    "/proxy/relink",
    "/proxy/swap-check",
    "/qc/audio-phase",
    "/qc/black-frames",
    "/qc/dropouts",
    "/qc/frozen-frames",
    "/qc/full-check",
    "/qc/leader-detect",
    "/qc/report",
    "/qc/silence-gaps",
    "/reaction/create",
    "/recap/generate",
    "/recap/score",
    "/redact/faces",
    "/redact/region",
    "/reframe/vertical",
    "/remote/submit",
    "/render/multi",
    "/repair/deinterlace",
    "/repair/framerate",
    "/repair/recover",
    "/repair/restore",
    "/repair/sdr-to-hdr",
    "/repurpose/extract-shorts",
    "/repurpose/podcast-bundle",
    "/repurpose/social-captions",
    "/repurpose/video-to-blog",
    "/review/create",
    "/review/integrity/verify",
    "/rough-cut/analyze",
    "/rough-cut/auto",
    "/rough-cut/execute",
    "/rough-cut/from-script",
    "/rough-cut/plan",
    "/screen/cursor-zoom",
    "/script/align",
    "/script/broll",
    "/script/parse",
    "/search/ai",
    "/search/ai/index",
    "/search/auto-index",
    "/search/federated/index",
    "/search/index",
    "/search/ingest",
    "/search/multimodal-index",
    "/search/semantic",
    "/search/semantic/index",
    "/selects/export",
    "/selects/metadata",
    "/selects/rate",
    "/selects/search",
    "/selects/tag",
    "/settings/brand-kit/preview",
    "/shorts/variants",
    "/silence",
    "/silence/speed-up",
    "/smart-render",
    "/social/generate-captions",
    "/social/upload",
    "/speaker-layout/active",
    "/speaker-layout/create",
    "/spectral/classify-noise",
    "/spectral/edit",
    "/spectral/repair",
    "/spectral/room-tone-fill",
    "/split-screen/create",
    "/star-trail/animation",
    "/star-trail/composite",
    "/star-trail/remove-streaks",
    "/stock/download",
    "/storage/scan",
    "/storyboard/from-script",
    "/storyboard/mood-board",
    "/storyboard/shot-list",
    "/stream/auto-chapter",
    "/stringout/chapters",
    "/stringout/generate",
    "/styled-captions",
    "/subtitle-position/apply",
    "/subtitle/auto-snap",
    "/subtitle/embed",
    "/subtitle/sdh-format",
    "/subtitle/snap-to-cuts",
    "/subtitles/auto-position",
    "/subtitles/broadcast-export",
    "/subtitles/multilang/create",
    "/subtitles/multilang/export",
    "/subtitles/multilang/import",
    "/subtitles/sdh-format",
    "/subtitles/shot-aware",
    "/takes/find-repeats",
    "/takes/score",
    "/talking-head/generate",
    "/talking-head/simple",
    "/team/sync",
    "/telemetry/overlay",
    "/template/assemble",
    "/template/fill",
    "/template/list",
    "/timelapse/analyze-flicker",
    "/timelapse/deflicker",
    "/timeline/beat-cut",
    "/timeline/export-from-markers",
    "/timeline/export/aaf",
    "/timeline/export/otioz",
    "/timeline/otio-diff",
    "/transcript",
    "/transcript-edit/apply-edits",
    "/transcript-edit/build-map",
    "/transcript-edit/delete-words",
    "/transcript-edit/export",
    "/transcript-edit/rearrange",
    "/transcript/summarize",
    "/ux/preview",
    "/ux/suggest",
    "/video/360/convert",
    "/video/360/crop",
    "/video/360/stabilize",
    "/video/aces/apply",
    "/video/ai-metadata/batch",
    "/video/ai-metadata/enrich",
    "/video/ai/denoise",
    "/video/ai/install",
    "/video/ai/interpolate",
    "/video/ai/rembg",
    "/video/ai/upscale",
    "/video/annotate-tracked",
    "/video/audio-reactive",
    "/video/audio-sync",
    "/video/audio-sync/multi",
    "/video/authenticity-report",
    "/video/auto-edit",
    "/video/auto-zoom",
    "/video/blend",
    "/video/body-effects",
    "/video/body-effects/detect",
    "/video/broll-generate",
    "/video/broll-plan",
    "/video/c2pa/create",
    "/video/c2pa/embed",
    "/video/c2pa/read",
    "/video/chromakey",
    "/video/cinefocus",
    "/video/clean-plate",
    "/video/color-match",
    "/video/color-wheels/apply",
    "/video/color/convert",
    "/video/color/correct",
    "/video/color/external-lut",
    "/video/colorspace/batch-detect",
    "/video/colorspace/convert",
    "/video/compare",
    "/video/compose/depth_segment",
    "/video/compose/vace",
    "/video/cursor-zoom/resolve",
    "/video/custody/create",
    "/video/custody/export",
    "/video/custody/finalize",
    "/video/custody/log",
    "/video/data-animation/bar-chart",
    "/video/data-animation/counter",
    "/video/data-animation/create",
    "/video/dead-time/detect",
    "/video/dead-time/speed-ramp",
    "/video/deinterlace",
    "/video/depth/bokeh",
    "/video/depth/estimate-v2",
    "/video/depth/map",
    "/video/depth/parallax",
    "/video/depth/parallax-v2",
    "/video/detect-deepfake",
    "/video/detect-edits",
    "/video/detect-faces",
    "/video/detect-highlights",
    "/video/detect-interlace",
    "/video/dub",
    "/video/emotion-highlights",
    "/video/emotion/arc",
    "/video/encode/apv",
    "/video/encode/svtav1-psy",
    "/video/encode/vmaf-target",
    "/video/encode/vvc",
    "/video/enhance-low-light",
    "/video/enhance-low-light/preview",
    "/video/extract-highlights",
    "/video/face-restore",
    "/video/face-restore/detect",
    "/video/face-restore/preview",
    "/video/face/blur",
    "/video/face/enhance",
    "/video/face/reage",
    "/video/face/reshape",
    "/video/face/retouch",
    "/video/face/swap",
    "/video/fit-to-fill",
    "/video/fx/apply",
    "/video/gamut/check-clipping",
    "/video/gamut/convert",
    "/video/generate-intro",
    "/video/generative-extend",
    "/video/hdr/tonemap",
    "/video/highlights",
    "/video/highlights/sports",
    "/video/holy-grail",
    "/video/horizon-level",
    "/video/hsl-qualifier/qualify",
    "/video/hsl-qualifier/secondary",
    "/video/interpolate/neural",
    "/video/kinetic-text/animate",
    "/video/kinetic-text/custom",
    "/video/kinetic-text/render",
    "/video/lens-correct",
    "/video/lip-sync-verify",
    "/video/lottie/render",
    "/video/lut/apply",
    "/video/lut/blend",
    "/video/lut/generate-ai",
    "/video/lut/generate-all",
    "/video/lut/generate-from-ref",
    "/video/matte/birefnet",
    "/video/merge",
    "/video/motion-brush",
    "/video/motion-brush/preview",
    "/video/motion-transfer",
    "/video/motion-transfer/preview",
    "/video/mouth-energy",
    "/video/multicam-cuts",
    "/video/multicam-xml",
    "/video/multimodal-diarize",
    "/video/nd-filter",
    "/video/object-remove/sam3",
    "/video/one-click-enhance",
    "/video/outpaint",
    "/video/particles/apply",
    "/video/physics-remove",
    "/video/physics-remove/detect-shadow",
    "/video/pip",
    "/video/power-windows/apply",
    "/video/power-windows/track",
    "/video/preview-frame",
    "/video/proxy/batch",
    "/video/proxy/generate",
    "/video/quality/batch-compare",
    "/video/quality/compare",
    "/video/quality/rank",
    "/video/quality/score",
    "/video/reframe",
    "/video/reframe-multi",
    "/video/reframe/batch",
    "/video/reframe/face",
    "/video/relight",
    "/video/relight/diffrenderer",
    "/video/relight/iclight",
    "/video/relight/lav",
    "/video/remove/watermark",
    "/video/repair",
    "/video/repair/diagnose",
    "/video/replace-background",
    "/video/replace-background/preview",
    "/video/restore/colorize",
    "/video/restore/deflicker",
    "/video/restore/vrt",
    "/video/rhythm-effects",
    "/video/rolling-shutter",
    "/video/scenes",
    "/video/scenes/auto",
    "/video/scopes/pro",
    "/video/segment/sam2",
    "/video/shape-animation/fill-transition",
    "/video/shape-animation/morph",
    "/video/shape-animation/stroke-draw",
    "/video/shorts-pipeline",
    "/video/speed/change",
    "/video/speed/ramp",
    "/video/speed/reverse",
    "/video/stabilize-advanced",
    "/video/stream/srt/start",
    "/video/style/apply",
    "/video/style/arbitrary",
    "/video/tc-common-range",
    "/video/tc-offsets",
    "/video/tc-sync",
    "/video/text-segment",
    "/video/text-segment/preview",
    "/video/title/overlay",
    "/video/title/render",
    "/video/track-object",
    "/video/track-overlay",
    "/video/trailer/generate",
    "/video/transitions/apply",
    "/video/transitions/join",
    "/video/trim",
    "/video/upscale/run",
    "/video/upscale/seedvr2",
    "/video/upscale/smart",
    "/video/vitc-extract",
    "/video/watermark",
    "/video/watermark/embed",
    "/video/watermark/extract",
    "/video/watermark/verify",
    "/voice/commands/start",
    "/voice/commands/stop",
    "/voice/convert/start",
    "/voice/convert/stop",
    "/watermark/apply",
    "/watermark/batch",
    "/waveform/data",
    "/waveform/image",
    "/waveform/region",
    "/whisper/reinstall",
    "/workflow/conditional",
    "/workflow/run",
})


# Async POST routes deliberately kept out of the queue. Each carries the
# reason it cannot or should not be replayed from a stored path.
_QUEUE_EXCLUDED_ENDPOINTS = frozenset({
    # Orchestrate other routes, so queueing one would re-enter the queue.
    "/agent/auto-edit",
    "/agent/create-plan",
    "/agent/execute-plan",
    "/agent/search-footage",
    "/agent/storyboard",
    # Interactive and latency-sensitive; queueing defeats the point.
    "/api/preview/cache/warm",
    "/api/preview/pipeline",
    "/api/preview/scrub",
    # Install or update code. Installs are rate-limited, not queued.
    "/plugins/marketplace/install",
    "/plugins/registry/install",
    "/plugins/registry/update",
    "/plugins/update",
    # Interactive and latency-sensitive; queueing defeats the point.
    "/preview/clip",
    "/preview/frame",
    "/preview/thumbnails",
    # Dispatch to an external service; replaying a stored path could publish twice.
    "/publish/export",
    "/publish/prepare",
    "/publish/queue",
    "/publish/upload",
    # Moves or deletes source media; not safe to replay from a stored path.
    "/storage/archive",
    # Moves or deletes source media; not safe to replay from a stored path.
    "/storage/restore",
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


def _quarantine_corrupt_queue_file():
    """Move a structurally corrupt job_queue.json aside, preserving its bytes.

    Returns the quarantine path, or ``None`` when the rename failed (a
    genuine I/O problem the caller should fail closed on).
    """
    from opencut.queue_store import QUEUE_FILE
    from opencut.user_data import OPENCUT_DIR

    source = os.path.join(OPENCUT_DIR, QUEUE_FILE)
    if not os.path.isfile(source):
        return None
    target = f"{source}.corrupt"
    try:
        os.replace(source, target)  # overwrites any previous .corrupt file
    except OSError as exc:
        logger.error("Could not quarantine corrupt queue file %s: %s", source, exc)
        return None
    return target


def initialize_job_queue(app, *, start_processing=True):
    """Load the durable queue and recover in-flight entries after a restart."""
    global _queue_app, _queue_persistence_enabled, _queue_storage_error

    try:
        stored_entries, migrated = load_persisted_queue()
    except QueueSchemaVersionError as exc:
        # A newer schema likely holds real work from a newer OpenCut: fail
        # closed instead of quarantining the file.
        logger.error("Job queue was not loaded: %s", exc)
        _queue_persistence_enabled = False
        _queue_storage_error = str(exc)
        return {"loaded": 0, "interrupted": 0, "invalid": 0, "error": str(exc)}
    except QueueDocumentError as exc:
        quarantined = _quarantine_corrupt_queue_file()
        if quarantined is None:
            logger.error("Job queue was not loaded: %s", exc)
            _queue_persistence_enabled = False
            _queue_storage_error = str(exc)
            return {"loaded": 0, "interrupted": 0, "invalid": 0, "error": str(exc)}
        logger.warning(
            "Corrupt job queue file quarantined to %s (%s); continuing with "
            "an empty queue",
            quarantined,
            exc,
        )
        stored_entries, migrated = [], False
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
        return jsonify({
            "error": f"Endpoint not queueable: {endpoint}",
            "code": "ENDPOINT_NOT_QUEUEABLE",
            "endpoint": endpoint,
            "suggestion": (
                "Run this route directly, or check GET /queue/coverage to see "
                "whether it is an async job that is simply not allowlisted yet."
            ),
        }), 400
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
                "suggestion": f"Queue persistence failed: {exc}. Check that "
                              "the OpenCut data directory is writable, then retry.",
            }), 503
    _process_queue(current_app._get_current_object())
    return jsonify({"queue_id": entry["id"], "position": position})


# ---------------------------------------------------------------------------
# Queue coverage (F306)
# ---------------------------------------------------------------------------
def queue_coverage() -> dict:
    """Report which async-job routes are queueable and which are not.

    The allowlist is hand-maintained, so it silently certifies whatever is not
    in it: a route omitted by accident and a route excluded on purpose look
    identical from outside. This computes the gap instead of asserting it, so
    the omission is visible without pre-empting the curation decision about
    which routes *should* be queueable.
    """
    allowlisted: list[str] = []
    missing: list[dict] = []
    seen: set[str] = set()

    for rule in current_app.url_map.iter_rules():
        methods = (rule.methods or set()) - {"HEAD", "OPTIONS"}
        if "POST" not in methods:
            continue
        if rule.arguments:
            # Parameterised routes cannot be replayed from a stored path.
            continue
        view = current_app.view_functions.get(rule.endpoint)
        if not getattr(view, "_opencut_async_job", False):
            continue
        path = str(rule.rule)
        if path in seen:
            continue
        seen.add(path)
        if path in _ALLOWED_QUEUE_ENDPOINTS:
            allowlisted.append(path)
        else:
            missing.append({
                "endpoint": path,
                "job_type": getattr(view, "_opencut_job_type", "") or "",
                "blueprint": rule.endpoint.rsplit(".", 1)[0] if "." in rule.endpoint else "",
            })

    allowlisted.sort()
    missing.sort(key=lambda item: item["endpoint"])
    total = len(allowlisted) + len(missing)
    # Entries in either set that no longer match a live async route.
    stale = sorted(set(_ALLOWED_QUEUE_ENDPOINTS) - seen)
    stale_excluded = sorted(set(_QUEUE_EXCLUDED_ENDPOINTS) - seen)
    # A route in neither set is an omission, not a decision. `missing` used to
    # mean "not queueable" and so counted the deliberate exclusions too, which
    # is why a 28% coverage figure looked like a curation choice rather than a
    # backlog. Report the two separately.
    unclassified = [item for item in missing if item["endpoint"] not in _QUEUE_EXCLUDED_ENDPOINTS]
    excluded = [item for item in missing if item["endpoint"] in _QUEUE_EXCLUDED_ENDPOINTS]
    return {
        "async_post_routes": total,
        "queueable": len(allowlisted),
        # Every async POST route is exactly one of these three.
        "not_queueable": len(missing),
        "excluded": len(excluded),
        "unclassified": len(unclassified),
        "coverage_percent": round((len(allowlisted) / total) * 100, 1) if total else 0.0,
        "allowlist_size": len(_ALLOWED_QUEUE_ENDPOINTS),
        "excluded_size": len(_QUEUE_EXCLUDED_ENDPOINTS),
        "stale_allowlist_entries": stale,
        "stale_excluded_entries": stale_excluded,
        "missing": unclassified,
    }


@jobs_bp.route("/queue/coverage", methods=["GET"])
def queue_coverage_route():
    """Read-only queue allowlist coverage. Changes no behaviour."""
    from opencut.errors import safe_error

    try:
        return jsonify(queue_coverage())
    except Exception as exc:  # noqa: BLE001 - diagnostics must not 500 the panel
        return safe_error(exc, "queue_coverage")


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
                "suggestion": f"Queue persistence failed: {exc}. Check that "
                              "the OpenCut data directory is writable, then retry.",
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
                "suggestion": f"Queue persistence failed: {exc}. Check that "
                              "the OpenCut data directory is writable, then retry.",
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
                "suggestion": f"Queue persistence failed: {exc}. Check that "
                              "the OpenCut data directory is writable, then retry.",
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
            # Wait for the child job using the same stuck-job timeout the
            # rest of the worker pool uses (default 7200s), then cancel it.
            with job_queue_lock:
                job_id = entry.get("job_id")
                entry_status = entry.get("status")
            if job_id:
                job_timeout = jobs_module._JOB_STUCK_TIMEOUT
                deadline = time.time() + job_timeout
                while time.time() < deadline:
                    # Call _get_job_copy outside job_queue_lock to avoid nested lock deadlock
                    safe = _get_job_copy(job_id)
                    if safe and safe.get("status") in (
                        "complete", "error", "cancelled", "interrupted"
                    ):
                        _update_queue_entry(entry, safe["status"])
                        break
                    time.sleep(1)
                else:
                    try:
                        _cancel_job(
                            job_id,
                            message=f"Queued job timed out after {job_timeout} seconds",
                        )
                    except Exception as exc:  # noqa: BLE001 - still fail the queue entry
                        logger.debug(
                            "Failed to cancel timed-out queue job %s: %s",
                            job_id,
                            exc,
                        )
                        try:
                            _kill_job_process(job_id)
                        except Exception:  # noqa: BLE001
                            pass
                    _update_queue_entry(
                        entry,
                        "error",
                        code="QUEUE_JOB_TIMEOUT",
                        error=f"Queued job timed out after {job_timeout} seconds",
                    )
                    logger.warning("Queue job %s timed out after %s seconds", job_id, job_timeout)
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
