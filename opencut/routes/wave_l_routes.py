"""
OpenCut Wave L Routes v1.33.0

Wave L modules:
  L1.1  ElevenLabs TTS      — /audio/tts/elevenlabs
  L1.2  Smart Upscaling Hub — /video/upscale/smart
  L1.3  AI Face Reshape     — /video/face/reshape
  L1.4  AI Skin Retouch     — /video/face/retouch
  L2.3  Spark-TTS           — /audio/tts/spark
  L2.6  Moonshine ASR       — /audio/transcribe/moonshine
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify

from opencut.errors import error_response, safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

logger = logging.getLogger("opencut")
wave_l_bp = Blueprint("wave_l", __name__)


def _stub_503(name: str, hint: str = "") -> tuple:
    return error_response(
        "DEPENDENCY_NOT_INSTALLED",
        f"{name} dependency is not installed or not configured.",
        status=503,
        suggestion=hint or "Check the module's INSTALL_HINT.",
    )


# =============================================================
# L1.1 — ElevenLabs TTS
# =============================================================

@wave_l_bp.route("/audio/tts/elevenlabs", methods=["POST"])
@require_csrf
@async_job("tts_elevenlabs", filepath_required=False)
def route_tts_elevenlabs(job_id, filepath, data):
    """Synthesize speech via ElevenLabs cloud TTS (3,000+ voices, 32 languages)."""
    from opencut.core import tts_elevenlabs
    if not tts_elevenlabs.check_elevenlabs_available():
        raise RuntimeError(
            "ElevenLabs is not available. "
            + tts_elevenlabs.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = tts_elevenlabs.synthesize(
        text=str(data.get("text") or ""),
        voice=str(data.get("voice") or "Rachel"),
        model=str(data.get("model") or tts_elevenlabs.DEFAULT_MODEL),
        stability=safe_float(data.get("stability"), 0.5),
        similarity_boost=safe_float(data.get("similarity_boost"), 0.75),
        style=safe_float(data.get("style"), 0.0),
        use_speaker_boost=safe_bool(data.get("use_speaker_boost"), True),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "voice_id": result.voice_id,
        "voice_name": result.voice_name,
        "model": result.model,
        "characters": result.characters,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/tts/elevenlabs/voices", methods=["GET"])
def route_tts_elevenlabs_voices():
    """Return the ElevenLabs voice catalogue."""
    try:
        from opencut.core import tts_elevenlabs
        available = tts_elevenlabs.check_elevenlabs_available()
        voices = tts_elevenlabs.list_voices() if available else []
        models = tts_elevenlabs.list_models()
        return jsonify({
            "available": available,
            "voices": voices,
            "models": models,
            "install_hint": tts_elevenlabs.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_elevenlabs_voices")


@wave_l_bp.route("/audio/tts/elevenlabs/info", methods=["GET"])
def route_tts_elevenlabs_info():
    """Return ElevenLabs availability status."""
    try:
        import os

        from opencut.core import tts_elevenlabs
        from opencut.helpers import _try_import
        has_key = bool(os.environ.get("OPENCUT_ELEVENLABS_API_KEY", "").strip())
        return jsonify({
            "available": tts_elevenlabs.check_elevenlabs_available(),
            "sdk_installed": _try_import("elevenlabs") is not None,
            "api_key_set": has_key,
            "models": tts_elevenlabs.ELEVENLABS_MODELS,
            "install_hint": tts_elevenlabs.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_elevenlabs_info")


# =============================================================
# L1.2 — Smart Upscaling Hub
# =============================================================

@wave_l_bp.route("/video/upscale/smart", methods=["POST"])
@require_csrf
@async_job("upscale_smart")
def route_upscale_smart(job_id, filepath, data):
    """
    Upscale video using the best available backend (AI or lanczos fallback).

    Body params:
      scale   int     Scale factor (default 2).
      hint    str     Content hint: auto|fast|quality|anime|face|film.
      backend str     Force backend: auto|lanczos|realesrgan|video2x|flashvsr.
      output  str     Output path (optional).
    """
    from opencut.core import upscale_hub

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = upscale_hub.upscale(
        video_path=filepath,
        scale=safe_int(data.get("scale"), 2),
        hint=str(data.get("hint") or "auto"),
        backend=str(data.get("backend") or "auto"),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "backend": result.backend,
        "scale": result.scale,
        "width_out": result.width_out,
        "height_out": result.height_out,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/upscale/smart/info", methods=["GET"])
def route_upscale_smart_info():
    """Return available upscaling backends and auto-selection logic."""
    try:
        from opencut.core import upscale_hub
        return jsonify(upscale_hub.get_hub_info())
    except Exception as exc:
        return safe_error(exc, "upscale_smart_info")


# =============================================================
# L1.3 — AI Face Reshape
# =============================================================

@wave_l_bp.route("/video/face/reshape", methods=["POST"])
@require_csrf
@async_job("face_reshape")
def route_face_reshape(job_id, filepath, data):
    """
    Liquify-style face warp via MediaPipe FaceMesh.

    Body params:
      operation  str    slim_face|enlarge_eyes|shrink_nose|raise_cheekbones|smooth_jaw
      strength   float  0–1 warp intensity (default 0.5).
      output     str    Output path (optional).
    """
    from opencut.core import face_reshape
    if not face_reshape.check_face_reshape_available():
        raise RuntimeError(
            "face_reshape dependencies not installed. " + face_reshape.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = face_reshape.reshape(
        video_path=filepath,
        operation=str(data.get("operation") or "slim_face"),
        strength=safe_float(data.get("strength"), 0.5),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "operation": result.operation,
        "frames_processed": result.frames_processed,
        "faces_found": result.faces_found,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/face/reshape/info", methods=["GET"])
def route_face_reshape_info():
    """Return face reshape capabilities."""
    try:
        from opencut.core import face_reshape
        return jsonify({
            "available": face_reshape.check_face_reshape_available(),
            "operations": list(face_reshape.OPERATIONS),
            "install_hint": face_reshape.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "face_reshape_info")


# =============================================================
# L1.4 — AI Skin Retouch
# =============================================================

@wave_l_bp.route("/video/face/retouch", methods=["POST"])
@require_csrf
@async_job("skin_retouch")
def route_skin_retouch(job_id, filepath, data):
    """
    Skin retouching and blemish removal for video.

    Body params:
      intensity  float  0–1 retouching strength (default 0.6).
      mode       str    bilateral (fast) or gfpgan (deep, GPU).
      radiance   float  0–1 brightness boost on face (default 0).
      output     str    Output path (optional).
    """
    from opencut.core import skin_retouch
    if not skin_retouch.check_skin_retouch_available():
        raise RuntimeError(
            "skin_retouch dependencies not installed. " + skin_retouch.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = skin_retouch.retouch(
        video_path=filepath,
        intensity=safe_float(data.get("intensity"), 0.6),
        mode=str(data.get("mode") or "bilateral"),
        radiance=safe_float(data.get("radiance"), 0.0),
        output=str(data.get("output") or "") or None,
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "frames_processed": result.frames_processed,
        "faces_found": result.faces_found,
        "mode": result.mode,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/face/retouch/info", methods=["GET"])
def route_skin_retouch_info():
    """Return skin retouch capabilities."""
    try:
        from opencut.core import skin_retouch
        return jsonify({
            "available": skin_retouch.check_skin_retouch_available(),
            "gfpgan_available": skin_retouch.check_gfpgan_available(),
            "modes": ["bilateral", "gfpgan"],
            "install_hint": skin_retouch.INSTALL_HINT,
            "gfpgan_hint": skin_retouch.GFPGAN_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "skin_retouch_info")


# =============================================================
# L2.3 — Spark-TTS (CPU-native zero-shot TTS)
# =============================================================

@wave_l_bp.route("/audio/tts/spark", methods=["POST"])
@require_csrf
@async_job("tts_spark", filepath_required=False)
def route_tts_spark(job_id, filepath, data):
    """Synthesize speech via Spark-TTS (CPU-native, Apache-2, zero-shot voice clone).

    Body params:
      text             str    Text to synthesize (required).
      voice            str    Preset voice: default|warm|bright|calm|deep.
      reference_audio  str    Path to 3-10s reference clip for voice cloning (optional).
      speed            float  Playback speed 0.5-2.0 (default 1.0).
      output           str    Output WAV path (optional, auto-generated).
    """
    from opencut.core import tts_sparktts
    if not tts_sparktts.check_sparktts_available():
        raise RuntimeError(
            "Spark-TTS is not installed. " + tts_sparktts.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    voice = str(data.get("voice") or "default").strip().lower()
    if voice not in tts_sparktts.SPARK_VOICE_PRESETS:
        voice = "default"

    ref_audio = str(data.get("reference_audio") or "").strip()
    if ref_audio:
        from opencut.security import validate_filepath
        ref_audio = validate_filepath(ref_audio)

    result = tts_sparktts.synthesize(
        text=text,
        voice=voice,
        reference_audio=ref_audio,
        speed=safe_float(data.get("speed"), 1.0, min_val=0.5, max_val=2.0),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "voice": result.voice,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "sample_rate": result.sample_rate,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/tts/spark/voices", methods=["GET"])
def route_tts_spark_voices():
    """Return available Spark-TTS voice presets."""
    try:
        from opencut.core import tts_sparktts
        return jsonify({
            "available": tts_sparktts.check_sparktts_available(),
            "voices": tts_sparktts.list_voices(),
            "install_hint": tts_sparktts.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_spark_voices")


@wave_l_bp.route("/audio/tts/spark/info", methods=["GET"])
def route_tts_spark_info():
    """Return Spark-TTS availability and capabilities."""
    try:
        from opencut.core import tts_sparktts
        return jsonify({
            "available": tts_sparktts.check_sparktts_available(),
            "model": "spark-tts",
            "licence": "Apache-2.0",
            "cpu_native": True,
            "voice_cloning": True,
            "presets": list(tts_sparktts.SPARK_VOICE_PRESETS.keys()),
            "install_hint": tts_sparktts.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_spark_info")


# =============================================================
# L2.6 — Moonshine Real-Time ASR
# =============================================================

@wave_l_bp.route("/audio/transcribe/moonshine", methods=["POST"])
@require_csrf
@async_job("moonshine_asr")
def route_moonshine_transcribe(job_id, filepath, data):
    """Transcribe audio/video via Moonshine ASR (CPU-optimized, 10x realtime).

    Body params:
      model  str  moonshine-tiny or moonshine-base (default: moonshine-base).
    """
    from opencut.core import asr_moonshine
    if not asr_moonshine.check_moonshine_available():
        raise RuntimeError(
            "Moonshine ASR is not installed. " + asr_moonshine.INSTALL_HINT
        )

    model = str(data.get("model") or "moonshine-base").strip()
    if model not in asr_moonshine.MOONSHINE_MODELS:
        model = "moonshine-base"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = asr_moonshine.transcribe(
        audio_path=filepath,
        model=model,
        on_progress=_prog,
    )
    return {
        "text": result.text,
        "segments": result.segments,
        "language": result.language,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "processing_seconds": result.processing_seconds,
        "realtime_factor": result.realtime_factor,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/transcribe/moonshine/info", methods=["GET"])
def route_moonshine_info():
    """Return Moonshine ASR availability and model catalogue."""
    try:
        from opencut.core import asr_moonshine
        return jsonify({
            "available": asr_moonshine.check_moonshine_available(),
            "models": asr_moonshine.MOONSHINE_MODELS,
            "licence": "MIT (English models)",
            "cpu_native": True,
            "streaming": True,
            "languages": ["en"],
            "install_hint": asr_moonshine.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "moonshine_info")
