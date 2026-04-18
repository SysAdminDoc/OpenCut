"""
OpenCut Wave D Routes (v1.21.0)

Delivery / color-science / voice-command endpoints:
- ``POST /video/encode/vvc``              — VVC / H.266 encode via libvvenc
- ``GET  /video/encode/vvc/info``         — libvvenc availability + presets
- ``POST /video/stream/srt/start``        — start SRT push / listener
- ``POST /video/stream/srt/stop``         — stop an SRT stream by pid
- ``GET  /video/stream/srt/info``         — SRT capability + defaults
- ``POST /video/scopes/pro``              — colour-science scopes (xy, LUV, gamut)
- ``GET  /video/scopes/pro/info``         — backend availability
- ``POST /voice/grammar/parse``           — timeline voice-command grammar
- ``GET  /voice/grammar/catalogue``       — UI helper — list verbs
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, validate_path

logger = logging.getLogger("opencut")

wave_d_bp = Blueprint("wave_d", __name__)


# ---------------------------------------------------------------------------
# VVC / H.266
# ---------------------------------------------------------------------------

@wave_d_bp.route("/video/encode/vvc/info", methods=["GET"])
def route_vvc_info():
    from opencut.core import vvc_export
    return jsonify({
        "installed": vvc_export.check_vvc_available(),
        "presets": vvc_export.list_presets(),
        "hint": (
            "Requires an FFmpeg build with --enable-libvvenc. "
            "See https://github.com/fraunhoferhhi/vvenc for upstream builds."
        ),
    })


@wave_d_bp.route("/video/encode/vvc", methods=["POST"])
@require_csrf
@async_job("vvc_encode")
def route_vvc_encode(job_id, filepath, data):
    from opencut.core import vvc_export
    from opencut.security import safe_int

    preset = str(data.get("preset") or "balanced").lower()
    container = str(data.get("container") or ".mp4").lower()
    if not container.startswith("."):
        container = "." + container
    output = (data.get("output") or "").strip()
    if output:
        output = validate_path(output)
    qp_override = data.get("qp")
    if qp_override is not None:
        qp_override = safe_int(qp_override, 32, min_val=0, max_val=63)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = vvc_export.encode_vvc(
        filepath,
        preset=preset,
        output=output or None,
        qp_override=qp_override,
        container=container,
        on_progress=_on_progress,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# SRT streaming
# ---------------------------------------------------------------------------

@wave_d_bp.route("/video/stream/srt/info", methods=["GET"])
def route_srt_info():
    from opencut.core import srt_streaming
    return jsonify({
        "installed": srt_streaming.check_srt_available(),
        "modes": list(srt_streaming.MODES),
        "defaults": {
            "latency_ms": srt_streaming.DEFAULT_LATENCY_MS,
            "pkt_size": srt_streaming.DEFAULT_PKT_SIZE,
        },
        "hint": (
            "Requires FFmpeg built with --enable-libsrt. "
            "See https://github.com/Haivision/srt."
        ),
    })


@wave_d_bp.route("/video/stream/srt/start", methods=["POST"])
@require_csrf
@async_job("srt_stream_start")
def route_srt_start(job_id, filepath, data):
    from opencut.core import srt_streaming
    from opencut.security import safe_int

    host = str(data.get("host") or "").strip()
    port = safe_int(data.get("port", 9998), 9998, min_val=1, max_val=65535)
    mode = str(data.get("mode") or "caller").lower()
    video_codec = str(data.get("video_codec") or "libx264")[:32]
    audio_codec = str(data.get("audio_codec") or "aac")[:32]
    video_bitrate = str(data.get("video_bitrate") or "4500k")[:16]
    audio_bitrate = str(data.get("audio_bitrate") or "192k")[:16]
    latency_ms = safe_int(data.get("latency_ms", 200), 200, min_val=20, max_val=5000)
    pkt_size = safe_int(data.get("pkt_size", 1316), 1316, min_val=64, max_val=1500)
    passphrase = data.get("passphrase")
    if passphrase is not None and not isinstance(passphrase, str):
        passphrase = None
    stream_id = data.get("stream_id")
    if stream_id is not None and not isinstance(stream_id, str):
        stream_id = None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = srt_streaming.start_stream(
        filepath,
        host=host, port=port, mode=mode,
        video_codec=video_codec, audio_codec=audio_codec,
        video_bitrate=video_bitrate, audio_bitrate=audio_bitrate,
        latency_ms=latency_ms, pkt_size=pkt_size,
        passphrase=passphrase, stream_id=stream_id,
        on_progress=_on_progress,
        job_id=job_id,
    )
    return dict(result)


@wave_d_bp.route("/video/stream/srt/stop", methods=["POST"])
@require_csrf
def route_srt_stop():
    try:
        from flask import request

        from opencut.core import srt_streaming
        from opencut.security import safe_float, safe_int

        data = request.get_json(force=True) or {}
        pid = safe_int(data.get("pid"), 0, min_val=1, max_val=2**31 - 1)
        if not pid:
            return jsonify({
                "error": "'pid' is required",
                "code": "INVALID_INPUT",
            }), 400
        timeout = safe_float(data.get("timeout", 3.0), 3.0, min_val=0.1, max_val=60.0)
        stopped = srt_streaming.stop_stream(pid=pid, timeout=timeout)
        return jsonify({"pid": pid, "stopped": bool(stopped)})
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "srt_stop")


# ---------------------------------------------------------------------------
# Colour-science scopes
# ---------------------------------------------------------------------------

@wave_d_bp.route("/video/scopes/pro/info", methods=["GET"])
def route_scopes_pro_info():
    from opencut.core import color_scopes_pro
    return jsonify({
        "colour_science_installed": color_scopes_pro.check_colour_science_available(),
        "matplotlib_installed": color_scopes_pro.check_matplotlib_available(),
        "gamuts": list(color_scopes_pro.GAMUTS),
        "hint": (
            "pip install colour-science  (scope math). "
            "pip install matplotlib  (plot PNG output; optional)."
        ),
    })


@wave_d_bp.route("/video/scopes/pro", methods=["POST"])
@require_csrf
@async_job("scopes_pro")
def route_scopes_pro(job_id, filepath, data):
    from opencut.core import color_scopes_pro
    from opencut.security import safe_bool, safe_int

    sample_count = safe_int(data.get("sample_count", 24), 24, min_val=1, max_val=120)
    render_plots = safe_bool(data.get("render_plots", True), True)
    output_dir = (data.get("output_dir") or "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = color_scopes_pro.analyse_scopes(
        filepath,
        sample_count=sample_count,
        output_dir=output_dir or None,
        render_plots=render_plots,
        on_progress=_on_progress,
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Voice-command grammar
# ---------------------------------------------------------------------------

@wave_d_bp.route("/voice/grammar/catalogue", methods=["GET"])
def route_grammar_catalogue():
    from opencut.core import voice_command_grammar as vcg
    return jsonify({
        "verbs": vcg.list_grammar(),
        "units": sorted(set(vcg._UNIT_ALIASES.values())),
    })


@wave_d_bp.route("/voice/grammar/parse", methods=["POST"])
@require_csrf
def route_grammar_parse():
    """Parse a single voice utterance into a :class:`VoiceAction`.

    Synchronous — parsing is pure-Python microseconds.
    """
    try:
        from flask import request

        from opencut.core import voice_command_grammar as vcg
        from opencut.security import safe_float

        data = request.get_json(force=True) or {}
        utterance = str(data.get("utterance") or "").strip()
        if not utterance:
            return jsonify({
                "error": "'utterance' is required",
                "code": "INVALID_INPUT",
            }), 400
        if len(utterance) > 600:
            return jsonify({
                "error": "utterance too long (max 600 chars)",
                "code": "INVALID_INPUT",
            }), 400
        fps = safe_float(data.get("fps", 30.0), 30.0, min_val=1.0, max_val=240.0)
        bpm = safe_float(data.get("bpm", 120.0), 120.0, min_val=20.0, max_val=400.0)

        action = vcg.parse(utterance, fps=fps, bpm=bpm)
        return jsonify(dict(action))
    except Exception as exc:  # noqa: BLE001
        return safe_error(exc, "voice_grammar_parse")
