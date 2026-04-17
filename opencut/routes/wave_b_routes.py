"""
OpenCut Wave B Routes (v1.19.0)

Restoration + encoding + matting + advanced captions layer on top of
the core modules added after v1.18.0.

Routes:
- ``POST /video/matte/birefnet``          — still/keyframe high-edge matte
- ``POST /captions/karaoke-adv/render``   — ASS karaoke renderer
- ``GET  /captions/karaoke-adv/presets``  — list karaoke presets
- ``POST /video/encode/svtav1-psy``       — SVT-AV1-PSY encoder
- ``GET  /video/encode/svtav1-psy/info``  — list presets + backend status
- ``POST /video/restore/colorize``        — DDColor B&W colorisation
- ``POST /video/restore/vrt``             — VRT/RVRT unified restoration
- ``POST /video/restore/deflicker``       — neural deflicker (FFmpeg fallback)
- ``GET  /video/restore/backends``        — report which restoration backends install
"""

import logging

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, validate_path

logger = logging.getLogger("opencut")

wave_b_bp = Blueprint("wave_b", __name__)


# ---------------------------------------------------------------------------
# BiRefNet matte
# ---------------------------------------------------------------------------

@wave_b_bp.route("/video/matte/birefnet", methods=["POST"])
@require_csrf
@async_job("birefnet_matte")
def route_birefnet_matte(job_id, filepath, data):
    from opencut.core import matte_birefnet

    mode = str(data.get("mode") or "rgba").lower()
    backend = str(data.get("backend") or "auto").lower()
    output = (data.get("output") or "").strip()
    hf_model = str(data.get("hf_model") or "ZhengPeng7/BiRefNet")[:160]
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = matte_birefnet.matte_image(
        filepath,
        output_path=output or None,
        mode=mode,
        backend=backend,
        hf_model=hf_model,
        on_progress=_on_progress,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Karaoke captions (advanced ASS)
# ---------------------------------------------------------------------------

@wave_b_bp.route("/captions/karaoke-adv/presets", methods=["GET"])
def route_karaoke_presets():
    from opencut.core import captions_karaoke_adv as ka
    return jsonify({
        "presets": list(ka.KARAOKE_PRESETS),
        "pyonfx_installed": ka.check_pyonfx_available(),
    })


@wave_b_bp.route("/captions/karaoke-adv/render", methods=["POST"])
@require_csrf
def route_karaoke_render():
    """Synchronous render — ASS generation is fast enough to skip @async_job."""
    try:
        from opencut.core import captions_karaoke_adv as ka
        from opencut.security import get_json_dict, safe_int

        data = get_json_dict()
        preset = str(data.get("preset") or "fill")
        output = str(data.get("output") or "").strip()
        if not output:
            return jsonify({
                "error": "output path required",
                "code": "INVALID_INPUT",
            }), 400
        output = validate_path(output)
        raw_segments = data.get("segments") or []
        if not isinstance(raw_segments, list) or not raw_segments:
            return jsonify({
                "error": "segments: non-empty WhisperX-style segment list required",
                "code": "INVALID_INPUT",
            }), 400
        segments = ka.segments_from_whisperx_dicts(raw_segments)
        if not segments:
            return jsonify({
                "error": "no valid segments after normalisation",
                "code": "INVALID_INPUT",
            }), 400

        res_x = safe_int(data.get("resx", 1920), 1920, min_val=64, max_val=7680)
        res_y = safe_int(data.get("resy", 1080), 1080, min_val=64, max_val=4320)
        font = str(data.get("font") or "Inter")[:80]
        font_size = safe_int(data.get("font_size", 64), 64, min_val=12, max_val=220)
        margin_v = safe_int(data.get("margin_v", 90), 90, min_val=0, max_val=500)
        prefer_pyonfx = bool(data.get("prefer_pyonfx", True))

        result = ka.render_karaoke_ass(
            segments,
            output_path=output,
            preset=preset,
            resolution=(res_x, res_y),
            font=font,
            font_size=font_size,
            margin_v=margin_v,
            prefer_pyonfx=prefer_pyonfx,
        )
        return jsonify(dict(result))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "karaoke_render")


# ---------------------------------------------------------------------------
# SVT-AV1-PSY encoder
# ---------------------------------------------------------------------------

@wave_b_bp.route("/video/encode/svtav1-psy/info", methods=["GET"])
def route_svtav1_psy_info():
    from opencut.core import svtav1_psy
    return jsonify({
        "installed": svtav1_psy.check_svtav1_psy_available(),
        "presets": svtav1_psy.list_presets(),
    })


@wave_b_bp.route("/video/encode/svtav1-psy", methods=["POST"])
@require_csrf
@async_job("svtav1_psy")
def route_svtav1_psy_encode(job_id, filepath, data):
    from opencut.core import svtav1_psy
    from opencut.security import safe_int

    preset = str(data.get("preset") or "web").lower()
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)
    crf_override = data.get("crf")
    if crf_override is not None:
        crf_override = safe_int(crf_override, 28, min_val=10, max_val=63)
    params_override = data.get("svtav1_params")
    if params_override is not None and not isinstance(params_override, str):
        params_override = None
    if params_override and len(params_override) > 500:
        raise ValueError("svtav1_params too long (max 500 chars)")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = svtav1_psy.encode_svtav1_psy(
        filepath,
        preset=preset,
        output=output or None,
        crf_override=crf_override,
        svtav1_params_override=params_override,
        on_progress=_on_progress,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Restoration — DDColor, VRT, deflicker
# ---------------------------------------------------------------------------

@wave_b_bp.route("/video/restore/backends", methods=["GET"])
def route_restore_backends():
    from opencut.core import colorize_ddcolor, deflicker_neural, restore_vrt
    return jsonify({
        "ddcolor": {
            "installed": colorize_ddcolor.check_ddcolor_available(),
            "hint": "pip install modelscope, or set OPENCUT_DDCOLOR_ONNX to a .onnx path",
        },
        "vrt": {
            "installed": restore_vrt.check_vrt_available(),
            "tasks": list(restore_vrt.TASKS),
            "hint": "pip install basicsr, or set OPENCUT_VRT_ONNX to a .onnx path",
        },
        "deflicker": {
            "neural_installed": deflicker_neural.check_neural_deflicker_available(),
            "ffmpeg_fallback": deflicker_neural.check_ffmpeg_deflicker_available(),
            "hint": "set OPENCUT_DEFLICKER_ONNX to a .onnx path for neural mode",
        },
    })


@wave_b_bp.route("/video/restore/colorize", methods=["POST"])
@require_csrf
@async_job("ddcolor")
def route_colorize(job_id, filepath, data):
    from opencut.core import colorize_ddcolor

    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = colorize_ddcolor.colourise_video(
        filepath, output=output or None, on_progress=_on_progress,
    )
    return dict(result)


@wave_b_bp.route("/video/restore/vrt", methods=["POST"])
@require_csrf
@async_job("vrt_restore")
def route_vrt(job_id, filepath, data):
    from opencut.core import restore_vrt
    from opencut.security import safe_int

    task = str(data.get("task") or "unified").lower()
    window = safe_int(data.get("window", 8), 8, min_val=2, max_val=32)
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = restore_vrt.restore_video(
        filepath, task=task, window=window,
        output=output or None, on_progress=_on_progress,
    )
    return dict(result)


@wave_b_bp.route("/video/restore/deflicker", methods=["POST"])
@require_csrf
@async_job("deflicker")
def route_deflicker(job_id, filepath, data):
    from opencut.core import deflicker_neural
    from opencut.security import safe_int

    backend = str(data.get("backend") or "auto").lower()
    strength = safe_int(data.get("strength", 3), 3, min_val=1, max_val=10)
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = deflicker_neural.deflicker_video(
        filepath, backend=backend, strength=strength,
        output=output or None, on_progress=_on_progress,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Webhook job-event passthrough (documentation endpoint)
# ---------------------------------------------------------------------------

@wave_b_bp.route("/webhooks/events", methods=["GET"])
def route_webhook_events():
    """List the event types OpenCut emits automatically.

    Jobs emit ``job.complete``, ``job.error``, and ``job.cancelled``
    events on terminal status. Webhook registrations filter by event
    type; an empty filter list subscribes to all events.
    """
    return jsonify({
        "auto_emitted": [
            {"event": "job.complete", "description": "Job finished successfully."},
            {"event": "job.error", "description": "Job ended with an error."},
            {"event": "job.cancelled", "description": "Job cancelled by user or cancel-all."},
        ],
        "payload_schema": {
            "event_type": "string",
            "timestamp": "ISO 8601 UTC",
            "job_id": "string",
            "details": {
                "job_id": "string",
                "job_type": "string",
                "filepath": "string (if present)",
                "endpoint": "string (e.g. /audio/pro/deepfilter)",
                "result": "object (final job result on success)",
                "error": "string (on error)",
                "progress": "int",
            },
        },
        "retry_policy": "up to 3 attempts, exponential backoff (1s, 5s, 15s)",
    })
