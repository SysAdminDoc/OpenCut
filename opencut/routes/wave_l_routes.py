"""
OpenCut Wave L Routes v1.33.0

Wave L modules:
  L1.1  ElevenLabs TTS      — /audio/tts/elevenlabs
  L1.2  Smart Upscaling Hub — /video/upscale/smart
  L1.3  AI Face Reshape     — /video/face/reshape
  L1.4  AI Skin Retouch     — /video/face/retouch
  L2.3  Spark-TTS           — /audio/tts/spark
  L2.1  FramePack I2V       — /generate/framepack
  L2.2  ACE-Step Music      — /audio/music/acestep
  L2.6  Moonshine ASR       — /audio/transcribe/moonshine
  M1.1  Chatterbox TTS      — /audio/tts/chatterbox
  M1.2  Kokoro TTS          — /audio/tts/kokoro
  M1.3  DiffRhythm Music    — /audio/music/diffrhythm
  M2.1  Wan2.2 T2V/I2V      — /generate/wan2.2/t2v, /generate/wan2.2/i2v
  M2.2  Wan2.2-S2V          — /generate/wan2.2/s2v
  M2.3  Wan2.2-Animate      — /generate/wan2.2/animate
  M2.4  FLUX Kontext Edit   — /image/edit/kontext
  M3.2  Digital Twin        — /pipeline/digital_twin
  N2.1  SAM 2.1 Segment    — /video/segment/sam2
  N1.1  FastVideo           — /generate/wan2.2/fast
  N1.2  LightX2V            — /generate/wan2.2/i2v/quantized
  N2.2  Depth-Anything-V2   — /video/depth/estimate-v2, /video/depth/parallax-v2
  N2.3  Depth+Segment Comp  — /video/compose/depth_segment
  N3.1  CogVideoX           — /generate/cogvideox
  N3.2  Qwen2.5-VL          — /analyze/video/vl
  N3.4  CSM-1B Speech       — /audio/speech/csm
  O1.1  Dia Dialogue TTS    — /audio/speech/dia
  O1.2  Parler-TTS          — /audio/speech/parler
  O2.1  LTX-Video           — /generate/ltxv/*
  O3.1  YuE Music           — /audio/music/yue
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


# =============================================================
# L2.2 — ACE-Step Full-Song Music Generation
# =============================================================

@wave_l_bp.route("/audio/music/acestep", methods=["POST"])
@require_csrf
@async_job("music_acestep", filepath_required=False)
def route_music_acestep(job_id, filepath, data):
    """Generate full-length music (up to 4 min) with optional lyrics.

    Body params:
      prompt           str    Music description (e.g., "upbeat indie pop").
      lyrics           str    Lyrics to align (optional).
      genre            str    Genre hint (pop/rock/electronic/...).
      mood             str    Mood hint (happy/sad/energetic/...).
      duration         float  Target seconds, max 240 (default 60).
      reference_audio  str    Style reference path (optional).
      output           str    Output WAV path (optional).
    """
    from opencut.core import music_acestep
    if not music_acestep.check_acestep_available():
        raise RuntimeError(
            "ACE-Step is not installed. " + music_acestep.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    genre = str(data.get("genre") or "pop").strip().lower()
    mood = str(data.get("mood") or "happy").strip().lower()

    ref_audio = str(data.get("reference_audio") or "").strip()
    if ref_audio:
        from opencut.security import validate_filepath
        ref_audio = validate_filepath(ref_audio)

    result = music_acestep.generate(
        prompt=str(data.get("prompt") or ""),
        lyrics=str(data.get("lyrics") or ""),
        genre=genre,
        mood=mood,
        duration=safe_float(data.get("duration"), 60.0, min_val=5.0, max_val=240.0),
        reference_audio=ref_audio,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "duration_seconds": result.duration_seconds,
        "genre": result.genre,
        "mood": result.mood,
        "has_lyrics": result.has_lyrics,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/music/acestep/edit", methods=["POST"])
@require_csrf
@async_job("music_acestep_edit")
def route_music_acestep_edit(job_id, filepath, data):
    """Edit lyrics in an ACE-Step generated track via flow-edit."""
    from opencut.core import music_acestep
    if not music_acestep.check_acestep_available():
        raise RuntimeError(
            "ACE-Step is not installed. " + music_acestep.INSTALL_HINT
        )

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    new_lyrics = str(data.get("lyrics") or data.get("new_lyrics") or "")
    if not new_lyrics.strip():
        raise ValueError("lyrics (or new_lyrics) field is required")

    result = music_acestep.edit_lyrics(
        audio_path=filepath,
        new_lyrics=new_lyrics,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "has_lyrics": result.has_lyrics,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/music/acestep/info", methods=["GET"])
def route_music_acestep_info():
    """Return ACE-Step availability and capabilities."""
    try:
        from opencut.core import music_acestep
        return jsonify({
            "available": music_acestep.check_acestep_available(),
            "model": "ace-step",
            "licence": "Apache-2.0",
            "max_duration_seconds": 240,
            "genres": music_acestep.ACESTEP_GENRES,
            "moods": music_acestep.ACESTEP_MOODS,
            "features": ["text-to-music", "lyrics-alignment",
                         "voice-cloning", "lyric-editing"],
            "install_hint": music_acestep.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "music_acestep_info")


# =============================================================
# L2.1 — FramePack Image-to-Video
# =============================================================

@wave_l_bp.route("/generate/framepack", methods=["POST"])
@require_csrf
@async_job("framepack_i2v", filepath_required=True, filepath_param="image_path")
def route_framepack_generate(job_id, filepath, data):
    """Generate video from a single image using FramePack (6 GB VRAM).

    Body params:
      image_path       str    Path to source image (required).
      prompt           str    Text description of motion/action.
      duration         float  Target seconds, max 60 (default 5).
      fps              float  Output frame rate (default 24).
      model            str    framepack-standard or framepack-fast.
      negative_prompt  str    Things to avoid.
      seed             int    Random seed (-1 for random).
      output           str    Output MP4 path (optional).
    """
    from opencut.core import gen_video_framepack
    if not gen_video_framepack.check_framepack_available():
        raise RuntimeError(
            "FramePack is not installed. " + gen_video_framepack.INSTALL_HINT
        )

    model = str(data.get("model") or "framepack-standard").strip()
    if model not in gen_video_framepack.FRAMEPACK_MODELS:
        model = "framepack-standard"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_framepack.generate(
        image_path=filepath,
        prompt=str(data.get("prompt") or ""),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=60.0),
        fps=safe_float(data.get("fps"), 24.0, min_val=8.0, max_val=60.0),
        model=model,
        negative_prompt=str(data.get("negative_prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "prompt": result.prompt,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/framepack/info", methods=["GET"])
def route_framepack_info():
    """Return FramePack availability and model catalogue."""
    try:
        from opencut.core import gen_video_framepack
        return jsonify({
            "available": gen_video_framepack.check_framepack_available(),
            "models": gen_video_framepack.FRAMEPACK_MODELS,
            "licence": "Apache-2.0",
            "min_vram_gb": 6,
            "max_duration_seconds": 60,
            "install_hint": gen_video_framepack.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "framepack_info")


# =============================================================
# M1.2 — Kokoro Ultralight TTS (82M, CPU-only)
# =============================================================

@wave_l_bp.route("/audio/tts/kokoro", methods=["POST"])
@require_csrf
@async_job("tts_kokoro", filepath_required=False)
def route_tts_kokoro(job_id, filepath, data):
    """Synthesize speech via Kokoro TTS (82M, CPU-only, 9 languages).

    Body params:
      text      str    Text to synthesize (required).
      voice     str    Voice preset ID (default: af_heart).
      language  str    Language code: en-us, en-gb, es, fr, hi, it, ja, pt-br, zh.
      speed     float  Speed 0.5-2.0 (default 1.0).
      output    str    Output WAV path (optional).
    """
    from opencut.core import tts_kokoro
    if not tts_kokoro.check_kokoro_available():
        raise RuntimeError(
            "Kokoro TTS is not installed. " + tts_kokoro.INSTALL_HINT
        )

    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    language = str(data.get("language") or "en-us").strip().lower()
    if language not in tts_kokoro.KOKORO_LANGUAGES:
        language = "en-us"

    voice = str(data.get("voice") or "af_heart").strip()

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = tts_kokoro.synthesize(
        text=text,
        voice=voice,
        language=language,
        speed=safe_float(data.get("speed"), 1.0, min_val=0.5, max_val=2.0),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "voice": result.voice,
        "language": result.language,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "sample_rate": result.sample_rate,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/tts/kokoro/voices", methods=["GET"])
def route_tts_kokoro_voices():
    """Return Kokoro TTS voice catalogue."""
    try:
        from flask import request as _req

        from opencut.core import tts_kokoro
        lang = _req.args.get("language", "")
        return jsonify({
            "available": tts_kokoro.check_kokoro_available(),
            "voices": tts_kokoro.list_voices(lang),
            "languages": tts_kokoro.KOKORO_LANGUAGES,
            "install_hint": tts_kokoro.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_kokoro_voices")


@wave_l_bp.route("/audio/tts/kokoro/info", methods=["GET"])
def route_tts_kokoro_info():
    """Return Kokoro TTS availability."""
    try:
        from opencut.core import tts_kokoro
        return jsonify({
            "available": tts_kokoro.check_kokoro_available(),
            "model": "kokoro",
            "params": "82M",
            "licence": "Apache-2.0",
            "cpu_native": True,
            "languages": list(tts_kokoro.KOKORO_LANGUAGES.keys()),
            "sample_rate": 24000,
            "install_hint": tts_kokoro.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_kokoro_info")


# =============================================================
# M1.3 — DiffRhythm Full-Song Generation
# =============================================================

@wave_l_bp.route("/audio/music/diffrhythm", methods=["POST"])
@require_csrf
@async_job("diffrhythm", filepath_required=False)
def route_music_diffrhythm(job_id, filepath, data):
    """Generate a full-length song from style prompt + optional LRC lyrics.

    Body params:
      style_prompt     str    Style description (e.g., "Jazzy Nightclub Vibe").
      lyrics_lrc       str    LRC-format lyrics (optional).
      style_reference  str    Path to audio style reference (optional).
      model            str    base (95s max) or full (285s max, default).
      chunked          bool   Use chunked mode for lower VRAM (default true).
      output           str    Output WAV path (optional).
    """
    from opencut.core import music_diffrhythm
    if not music_diffrhythm.check_diffrhythm_available():
        raise RuntimeError(
            "DiffRhythm is not installed. " + music_diffrhythm.INSTALL_HINT
        )

    model_variant = str(data.get("model") or "full").strip().lower()
    if model_variant not in music_diffrhythm.DIFFRHYTHM_MODELS:
        model_variant = "full"

    ref_audio = str(data.get("style_reference") or "").strip()
    if ref_audio:
        from opencut.security import validate_filepath
        ref_audio = validate_filepath(ref_audio)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = music_diffrhythm.generate(
        style_prompt=str(data.get("style_prompt") or data.get("prompt") or ""),
        lyrics_lrc=str(data.get("lyrics_lrc") or data.get("lyrics") or ""),
        style_reference=ref_audio,
        model_variant=model_variant,
        chunked=safe_bool(data.get("chunked"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "duration_seconds": result.duration_seconds,
        "style": result.style,
        "has_lyrics": result.has_lyrics,
        "model_variant": result.model_variant,
        "generation_seconds": result.generation_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/music/diffrhythm/styles", methods=["GET"])
def route_music_diffrhythm_styles():
    """Return DiffRhythm style catalogue."""
    try:
        from opencut.core import music_diffrhythm
        return jsonify({
            "available": music_diffrhythm.check_diffrhythm_available(),
            "styles": music_diffrhythm.DIFFRHYTHM_STYLES,
            "models": music_diffrhythm.DIFFRHYTHM_MODELS,
            "install_hint": music_diffrhythm.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "diffrhythm_styles")


@wave_l_bp.route("/audio/music/diffrhythm/info", methods=["GET"])
def route_music_diffrhythm_info():
    """Return DiffRhythm availability and capabilities."""
    try:
        from opencut.core import music_diffrhythm
        return jsonify({
            "available": music_diffrhythm.check_diffrhythm_available(),
            "licence": "Apache-2.0",
            "models": music_diffrhythm.DIFFRHYTHM_MODELS,
            "max_duration": {"base": 95, "full": 285},
            "min_vram_gb": 8,
            "features": ["text-to-song", "lyrics-alignment",
                         "style-reference", "continuation", "instrumental"],
            "install_hint": music_diffrhythm.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "diffrhythm_info")


# =============================================================
# M1.1 — Chatterbox TTS (emotional + voice clone)
# =============================================================

@wave_l_bp.route("/audio/tts/chatterbox", methods=["POST"])
@require_csrf
@async_job("tts_chatterbox", filepath_required=False)
def route_tts_chatterbox(job_id, filepath, data):
    """Synthesize emotional speech via Chatterbox TTS (MIT, 350M-500M).

    Body params:
      text             str    Text with optional emotion tags [laugh] [sigh] etc.
      reference_audio  str    Path to 10s clip for voice cloning (optional).
      model            str    turbo (English) or multilingual (23 langs).
      language         str    Language code for multilingual model.
      exaggeration     float  Emotion intensity 0-1 (default 0.5).
      speed            float  Speed 0.5-2.0 (default 1.0).
      output           str    Output WAV path (optional).
    """
    from opencut.core import tts_chatterbox
    if not tts_chatterbox.check_chatterbox_available():
        raise RuntimeError(
            "Chatterbox TTS is not installed. " + tts_chatterbox.INSTALL_HINT
        )

    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    model = str(data.get("model") or "turbo").strip().lower()
    if model not in tts_chatterbox.CHATTERBOX_MODELS:
        model = "turbo"

    ref_audio = str(data.get("reference_audio") or "").strip()
    if ref_audio:
        from opencut.security import validate_filepath
        ref_audio = validate_filepath(ref_audio)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = tts_chatterbox.synthesize(
        text=text,
        reference_audio=ref_audio,
        model=model,
        language=str(data.get("language") or "en").strip().lower(),
        exaggeration=safe_float(data.get("exaggeration"), 0.5, min_val=0.0, max_val=1.0),
        speed=safe_float(data.get("speed"), 1.0, min_val=0.5, max_val=2.0),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "voice": result.voice,
        "model": result.model,
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "sample_rate": result.sample_rate,
        "has_emotion_tags": result.has_emotion_tags,
        "voice_cloned": result.voice_cloned,
        "notes": result.notes,
    }


@wave_l_bp.route("/audio/tts/chatterbox/info", methods=["GET"])
def route_tts_chatterbox_info():
    """Return Chatterbox TTS availability and capabilities."""
    try:
        from opencut.core import tts_chatterbox
        return jsonify({
            "available": tts_chatterbox.check_chatterbox_available(),
            "multilingual_available": tts_chatterbox.check_chatterbox_multilingual_available(),
            "models": tts_chatterbox.CHATTERBOX_MODELS,
            "languages": tts_chatterbox.CHATTERBOX_LANGUAGES,
            "emotion_tags": tts_chatterbox.EMOTION_TAGS,
            "licence": "MIT",
            "voice_cloning": True,
            "install_hint": tts_chatterbox.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "tts_chatterbox_info")


# =============================================================
# M2.4 — FLUX.1 Kontext Image Editing
# =============================================================

@wave_l_bp.route("/image/edit/kontext", methods=["POST"])
@require_csrf
@async_job("kontext_edit", filepath_required=True, filepath_param="image_path")
def route_kontext_edit(job_id, filepath, data):
    """Edit image using FLUX.1 Kontext natural language instructions.

    Body params:
      image_path           str    Path to source image (required).
      instruction          str    Edit instruction (required).
      num_inference_steps  int    Diffusion steps 4-100 (default 28).
      guidance_scale       float  CFG scale 1-15 (default 7.5).
      seed                 int    Random seed (-1 for random).
      output               str    Output path (optional).
    """
    from opencut.core import image_edit_kontext
    if not image_edit_kontext.check_kontext_available():
        raise RuntimeError(
            "FLUX Kontext not available. " + image_edit_kontext.INSTALL_HINT
        )

    instruction = str(data.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("instruction is required")

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = image_edit_kontext.edit_image(
        image_path=filepath,
        instruction=instruction,
        num_inference_steps=safe_int(data.get("num_inference_steps"), 28, min_val=4, max_val=100),
        guidance_scale=safe_float(data.get("guidance_scale"), 7.5, min_val=1.0, max_val=15.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "instruction": result.instruction,
        "width": result.width,
        "height": result.height,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/image/edit/kontext/info", methods=["GET"])
def route_kontext_info():
    """Return FLUX Kontext model availability and download status."""
    try:
        from opencut.core import image_edit_kontext
        return jsonify(image_edit_kontext.get_model_info())
    except Exception as exc:
        return safe_error(exc, "kontext_info")


# =============================================================
# M2.1 — Wan2.2 T2V / I2V / TI2V Video Generation
# =============================================================

@wave_l_bp.route("/generate/wan2.2/t2v", methods=["POST"])
@require_csrf
@async_job("wan22_t2v", filepath_required=False)
def route_wan22_t2v(job_id, filepath, data):
    """Generate video from text using Wan2.2 (MoE, cinematic aesthetics).

    Body params:
      prompt           str    Video description (required).
      duration         float  Target seconds, max 16 (default 5).
      model            str    ti2v-5b (consumer) / t2v-14b / i2v-14b.
      negative_prompt  str    Things to avoid.
      fps              float  Frame rate (default 24).
      width            int    Output width (default 1280).
      height           int    Output height (default 720).
      seed             int    Random seed (-1 for random).
      offload_model    bool   CPU offload for VRAM savings (default true).
      output           str    Output MP4 path (optional).
    """
    from opencut.core import gen_video_wan22
    if not gen_video_wan22.check_wan22_available():
        raise RuntimeError(
            "Wan2.2 not installed. " + gen_video_wan22.INSTALL_HINT
        )

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "ti2v-5b").strip()
    if model not in gen_video_wan22.WAN22_MODELS:
        model = "ti2v-5b"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_wan22.generate_t2v(
        prompt=prompt,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=16.0),
        model=model,
        negative_prompt=str(data.get("negative_prompt") or ""),
        fps=safe_float(data.get("fps"), 24.0, min_val=8.0, max_val=60.0),
        width=safe_int(data.get("width"), 1280, min_val=256, max_val=2048),
        height=safe_int(data.get("height"), 720, min_val=256, max_val=2048),
        seed=safe_int(data.get("seed"), -1),
        offload_model=safe_bool(data.get("offload_model"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "mode": result.mode,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "generation_seconds": result.generation_seconds,
        "prompt": result.prompt,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/wan2.2/i2v", methods=["POST"])
@require_csrf
@async_job("wan22_i2v", filepath_required=True, filepath_param="image_path")
def route_wan22_i2v(job_id, filepath, data):
    """Generate video from image + optional text using Wan2.2.

    Body params:
      image_path     str    Source image (required).
      prompt         str    Optional motion description.
      duration       float  Target seconds, max 16.
      model          str    Model variant.
      seed           int    Random seed.
      offload_model  bool   CPU offload.
      output         str    Output MP4 path.
    """
    from opencut.core import gen_video_wan22
    if not gen_video_wan22.check_wan22_available():
        raise RuntimeError(
            "Wan2.2 not installed. " + gen_video_wan22.INSTALL_HINT
        )

    model = str(data.get("model") or "ti2v-5b").strip()
    if model not in gen_video_wan22.WAN22_MODELS:
        model = "ti2v-5b"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_wan22.generate_i2v(
        image_path=filepath,
        prompt=str(data.get("prompt") or ""),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=16.0),
        model=model,
        seed=safe_int(data.get("seed"), -1),
        offload_model=safe_bool(data.get("offload_model"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "mode": result.mode,
        "model": result.model,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "generation_seconds": result.generation_seconds,
        "prompt": result.prompt,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/wan2.2/info", methods=["GET"])
def route_wan22_info():
    """Return Wan2.2 model availability and variants."""
    try:
        from opencut.core import gen_video_wan22
        return jsonify({
            "available": gen_video_wan22.check_wan22_available(),
            "models": gen_video_wan22.WAN22_MODELS,
            "modes": gen_video_wan22.WAN22_MODES,
            "licence": "Apache-2.0",
            "max_duration_seconds": 16,
            "consumer_model": "ti2v-5b",
            "install_hint": gen_video_wan22.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "wan22_info")


# =============================================================
# M2.2 — Wan2.2-S2V Speech-to-Video (talking head)
# =============================================================

@wave_l_bp.route("/generate/wan2.2/s2v", methods=["POST"])
@require_csrf
@async_job("wan22_s2v", filepath_required=True, filepath_param="audio_path")
def route_wan22_s2v(job_id, filepath, data):
    """Generate talking-head video from audio + portrait image.

    Body params:
      audio_path     str   Path to speech audio (required).
      portrait_path  str   Path to reference portrait (required).
      prompt         str   Optional scene context.
      offload_model  bool  CPU offload (default true).
      half_body      bool  Upper-body mode (default false).
      seed           int   Random seed.
      output         str   Output MP4 path.
    """
    from opencut.core import gen_video_wan22_s2v
    if not gen_video_wan22_s2v.check_s2v_available():
        raise RuntimeError(
            "Wan2.2-S2V not installed. " + gen_video_wan22_s2v.INSTALL_HINT
        )

    portrait = str(data.get("portrait_path") or data.get("portrait") or "").strip()
    if not portrait:
        raise ValueError("portrait_path is required")
    from opencut.security import validate_filepath
    portrait = validate_filepath(portrait)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_wan22_s2v.generate(
        audio_path=filepath,
        portrait_path=portrait,
        prompt=str(data.get("prompt") or ""),
        offload_model=safe_bool(data.get("offload_model"), True),
        half_body=safe_bool(data.get("half_body"), False),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "audio_source": result.audio_source,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/wan2.2/s2v/info", methods=["GET"])
def route_wan22_s2v_info():
    """Return Wan2.2-S2V availability."""
    try:
        from opencut.core import gen_video_wan22_s2v
        return jsonify({
            "available": gen_video_wan22_s2v.check_s2v_available(),
            "model": "wan2.2-s2v-14b",
            "licence": "Apache-2.0",
            "min_vram_gb": 80,
            "offload_supported": True,
            "install_hint": gen_video_wan22_s2v.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "wan22_s2v_info")


# =============================================================
# M2.3 — Wan2.2-Animate Character Animation/Replacement
# =============================================================

@wave_l_bp.route("/generate/wan2.2/animate", methods=["POST"])
@require_csrf
@async_job("wan22_animate", filepath_required=True, filepath_param="motion_video")
def route_wan22_animate(job_id, filepath, data):
    """Animate character or replace character in video.

    Body params:
      character_image  str   Path to character/appearance image (required).
      motion_video     str   Path to motion reference video (required).
      mode             str   motion_transfer or character_replace.
      offload_model    bool  CPU offload (default true).
      seed             int   Random seed.
      output           str   Output MP4 path.
    """
    from opencut.core import gen_video_wan22_animate
    if not gen_video_wan22_animate.check_animate_available():
        raise RuntimeError(
            "Wan2.2-Animate not installed. " + gen_video_wan22_animate.INSTALL_HINT
        )

    char_img = str(data.get("character_image") or data.get("character") or "").strip()
    if not char_img:
        raise ValueError("character_image is required")
    from opencut.security import validate_filepath
    char_img = validate_filepath(char_img)

    mode = str(data.get("mode") or "motion_transfer").strip().lower()
    if mode not in gen_video_wan22_animate.ANIMATE_MODES:
        mode = "motion_transfer"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_wan22_animate.animate(
        character_image=char_img,
        motion_video=filepath,
        mode=mode,
        offload_model=safe_bool(data.get("offload_model"), True),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "mode": result.mode,
        "duration_seconds": result.duration_seconds,
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "model": result.model,
        "generation_seconds": result.generation_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/generate/wan2.2/animate/info", methods=["GET"])
def route_wan22_animate_info():
    """Return Wan2.2-Animate availability and modes."""
    try:
        from opencut.core import gen_video_wan22_animate
        return jsonify({
            "available": gen_video_wan22_animate.check_animate_available(),
            "model": "wan2.2-animate-14b",
            "modes": gen_video_wan22_animate.ANIMATE_MODES,
            "licence": "Apache-2.0",
            "install_hint": gen_video_wan22_animate.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "wan22_animate_info")


# =============================================================
# M3.2 — Digital Twin / AI Avatar Pipeline
# =============================================================

@wave_l_bp.route("/pipeline/digital_twin", methods=["POST"])
@require_csrf
@async_job("digital_twin", filepath_required=False)
def route_digital_twin(job_id, filepath, data):
    """Run the full digital twin localisation pipeline.

    Body params:
      script                str    Script text (required).
      voice_ref_path        str    10s voice reference clip for cloning.
      face_ref_path         str    Portrait image for talking head.
      target_languages      list   ISO 639-1 codes to dub into.
      talking_head_backend  str    auto / wan22_s2v / echomimic / skip.
      skip_stages           list   Stages to skip.
      pre_narration_path    str    Pre-generated narration (skips TTS).
      pre_avatar_path       str    Pre-generated avatar (skips head gen).
      tts_engine            str    chatterbox / edge / kokoro / spark.
      whisper_model         str    Whisper model for dub STT.
      output_dir            str    Output directory.
    """
    from opencut.core import pipeline_digital_twin
    from opencut.security import validate_filepath, validate_path

    script = str(data.get("script") or "").strip()
    if not script:
        raise ValueError("script is required")

    voice_ref = str(data.get("voice_ref_path") or "").strip()
    if voice_ref:
        voice_ref = validate_filepath(voice_ref)

    face_ref = str(data.get("face_ref_path") or "").strip()
    if face_ref:
        face_ref = validate_filepath(face_ref)

    pre_narration = str(data.get("pre_narration_path") or "").strip()
    if pre_narration:
        pre_narration = validate_filepath(pre_narration)

    pre_avatar = str(data.get("pre_avatar_path") or "").strip()
    if pre_avatar:
        pre_avatar = validate_filepath(pre_avatar)

    out_dir = str(data.get("output_dir") or "").strip()
    if out_dir:
        out_dir = validate_path(out_dir)

    backend = str(data.get("talking_head_backend") or "auto").strip().lower()
    if backend not in pipeline_digital_twin.TALKING_HEAD_BACKENDS:
        backend = "auto"

    tts_engine = str(data.get("tts_engine") or "chatterbox").strip().lower()
    if tts_engine not in ("chatterbox", "edge", "kokoro", "spark"):
        tts_engine = "chatterbox"

    target_langs = data.get("target_languages") or []
    if not isinstance(target_langs, list):
        target_langs = [str(target_langs)]
    target_langs = [str(lang).strip().lower() for lang in target_langs if isinstance(lang, str)][:10]

    skip = data.get("skip_stages") or []
    if not isinstance(skip, list):
        skip = []

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = pipeline_digital_twin.run_pipeline(
        script=script,
        voice_ref_path=voice_ref,
        face_ref_path=face_ref,
        target_languages=target_langs,
        talking_head_backend=backend,
        skip_stages=skip,
        pre_narration_path=pre_narration,
        pre_avatar_path=pre_avatar,
        tts_engine=tts_engine,
        whisper_model=str(data.get("whisper_model") or "base"),
        output_dir=out_dir,
        on_progress=_prog,
    )
    return {
        "outputs": result.outputs,
        "stages_completed": result.stages_completed,
        "stages_skipped": result.stages_skipped,
        "target_languages": result.target_languages,
        "narration_path": result.narration_path,
        "avatar_video_path": result.avatar_video_path,
        "talking_head_backend": result.talking_head_backend,
        "total_duration_seconds": result.total_duration_seconds,
        "notes": result.notes,
    }


@wave_l_bp.route("/pipeline/digital_twin/info", methods=["GET"])
def route_digital_twin_info():
    """Return digital twin pipeline capabilities."""
    try:
        from opencut.core import pipeline_digital_twin
        backend = pipeline_digital_twin._detect_talking_head_backend()
        return jsonify({
            "available": True,
            "stages": pipeline_digital_twin.PIPELINE_STAGES,
            "talking_head_backends": pipeline_digital_twin.TALKING_HEAD_BACKENDS,
            "detected_backend": backend,
            "install_hint": pipeline_digital_twin.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "digital_twin_info")


@wave_l_bp.route("/pipeline/digital_twin/stages", methods=["GET"])
def route_digital_twin_stages():
    """Return pipeline stage descriptions."""
    try:
        from opencut.core import pipeline_digital_twin
        return jsonify({
            "stages": pipeline_digital_twin.PIPELINE_STAGES,
            "descriptions": {
                "voice_clone": "Clone voice from 10s reference clip",
                "narrate": "Generate speech audio from script text",
                "talking_head": "Generate lip-sync video from speech + portrait",
                "translate_dub": "Translate script and dub into target languages",
                "composite": "Composite avatar onto original footage",
            },
        })
    except Exception as exc:
        return safe_error(exc, "digital_twin_stages")


# =============================================================
# N2.1 — SAM 2.1 Video Object Segmentation
# =============================================================

@wave_l_bp.route("/video/segment/sam2", methods=["POST"])
@require_csrf
@async_job("sam2_segment")
def route_sam2_segment(job_id, filepath, data):
    """Segment objects in video using SAM 2.1 with prompt propagation.

    Body params:
      prompts         list   Prompt objects with type/frame/coordinates.
      model           str    tiny / small / base_plus / large.
      output_format   str    alpha_video / matted_video / mask_frames / coco_json.
      propagate       bool   Propagate across all frames (default true).
      output          str    Output path (optional).
    """
    from opencut.core import segment_sam2
    if not segment_sam2.check_sam2_available():
        raise RuntimeError(
            "SAM 2.1 not installed. " + segment_sam2.INSTALL_HINT
        )

    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty list")

    model = str(data.get("model") or "small").strip()
    if model not in segment_sam2.SAM2_MODELS:
        model = "small"

    output_format = str(data.get("output_format") or "alpha_video").strip()
    if output_format not in segment_sam2.SAM2_OUTPUT_FORMATS:
        output_format = "alpha_video"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = segment_sam2.segment_video(
        video_path=filepath,
        prompts=prompts,
        model=model,
        output_format=output_format,
        propagate=safe_bool(data.get("propagate"), True),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {
        "output": result.output,
        "output_format": result.output_format,
        "frames_processed": result.frames_processed,
        "objects_tracked": result.objects_tracked,
        "model": result.model,
        "processing_seconds": result.processing_seconds,
        "mask_count": result.mask_count,
        "notes": result.notes,
    }


@wave_l_bp.route("/video/segment/sam2/info", methods=["GET"])
def route_sam2_info():
    """Return SAM 2.1 availability and model catalogue."""
    try:
        from opencut.core import segment_sam2
        return jsonify({
            "available": segment_sam2.check_sam2_available(),
            "models": segment_sam2.SAM2_MODELS,
            "prompt_types": segment_sam2.SAM2_PROMPT_TYPES,
            "output_formats": segment_sam2.SAM2_OUTPUT_FORMATS,
            "licence": "Apache-2.0",
            "install_hint": segment_sam2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "sam2_info")


# =============================================================
# N1.1 — FastVideo Inference Acceleration
# =============================================================

@wave_l_bp.route("/generate/wan2.2/fast", methods=["POST"])
@require_csrf
@async_job("fastvideo", filepath_required=False)
def route_fastvideo(job_id, filepath, data):
    """Generate video using FastVideo distilled Wan2.2 (>50x speedup)."""
    from opencut.core import gen_video_fastvideo
    if not gen_video_fastvideo.check_fastvideo_available():
        raise RuntimeError("FastVideo not installed. " + gen_video_fastvideo.INSTALL_HINT)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "FastWan2.2-TI2V-5B").strip()
    if model not in gen_video_fastvideo.FAST_MODELS:
        model = "FastWan2.2-TI2V-5B"

    image = str(data.get("image_path") or "").strip()
    if image:
        from opencut.security import validate_filepath
        image = validate_filepath(image)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_fastvideo.generate(
        prompt=prompt, model=model, image_path=image,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=10.0),
        negative_prompt=str(data.get("negative_prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/wan2.2/fast/info", methods=["GET"])
def route_fastvideo_info():
    try:
        from opencut.core import gen_video_fastvideo
        return jsonify({
            "available": gen_video_fastvideo.check_fastvideo_available(),
            "models": gen_video_fastvideo.FAST_MODELS,
            "licence": "Apache-2.0",
            "install_hint": gen_video_fastvideo.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "fastvideo_info")


# =============================================================
# N1.2 — LightX2V Quantized I2V
# =============================================================

@wave_l_bp.route("/generate/wan2.2/i2v/quantized", methods=["POST"])
@require_csrf
@async_job("lightx2v", filepath_required=True, filepath_param="image_path")
def route_lightx2v(job_id, filepath, data):
    """Generate I2V with quantized/distilled Wan2.2 A14B (24 GB VRAM)."""
    from opencut.core import gen_video_lightx2v
    if not gen_video_lightx2v.check_lightx2v_available():
        raise RuntimeError("LightX2V not installed. " + gen_video_lightx2v.INSTALL_HINT)

    quant = str(data.get("quant") or "fp8").strip()
    if quant not in gen_video_lightx2v.QUANT_MODES:
        quant = "fp8"

    steps = safe_int(data.get("steps"), 4)
    if steps not in gen_video_lightx2v.STEP_PRESETS:
        steps = 4

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_lightx2v.generate_i2v(
        image_path=filepath, prompt=str(data.get("prompt") or ""),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=16.0),
        quant=quant, steps=steps, seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/wan2.2/i2v/quantization/info", methods=["GET"])
def route_lightx2v_info():
    try:
        from opencut.core import gen_video_lightx2v
        return jsonify({
            "available": gen_video_lightx2v.check_lightx2v_available(),
            "quant_modes": gen_video_lightx2v.QUANT_MODES,
            "step_presets": {str(k): v for k, v in gen_video_lightx2v.STEP_PRESETS.items()},
            "licence": "Apache-2.0",
            "install_hint": gen_video_lightx2v.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "lightx2v_info")


# =============================================================
# N2.2 — Depth-Anything-V2 Depth Estimation
# =============================================================

@wave_l_bp.route("/video/depth/estimate-v2", methods=["POST"])
@require_csrf
@async_job("depth_estimate_v2")
def route_depth_estimate(job_id, filepath, data):
    """Estimate per-frame depth maps using Depth-Anything-V2."""
    from opencut.core import depth_anything_v2
    if not depth_anything_v2.check_depth_anything_v2_available():
        raise RuntimeError("Depth deps not installed. " + depth_anything_v2.INSTALL_HINT)

    model = str(data.get("model") or "small").strip()
    if model not in depth_anything_v2.DA2_MODELS:
        model = "small"

    fmt = str(data.get("output_format") or "video").strip()
    if fmt not in ("video", "numpy"):
        fmt = "video"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = depth_anything_v2.estimate_depth(
        video_path=filepath, model=model, output_format=fmt,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/video/depth/parallax-v2", methods=["POST"])
@require_csrf
@async_job("depth_parallax_v2")
def route_depth_parallax(job_id, filepath, data):
    """Generate 2.5D parallax via Depth-Anything-V2 layer separation."""
    from opencut.core import depth_anything_v2
    if not depth_anything_v2.check_depth_anything_v2_available():
        raise RuntimeError("Depth deps not installed. " + depth_anything_v2.INSTALL_HINT)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = depth_anything_v2.generate_parallax(
        video_path=filepath,
        shift_x=safe_float(data.get("shift_x"), 20.0, min_val=-100.0, max_val=100.0),
        shift_y=safe_float(data.get("shift_y"), 0.0, min_val=-100.0, max_val=100.0),
        model=str(data.get("model") or "small"),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/video/depth/info", methods=["GET"])
def route_depth_info():
    try:
        from opencut.core import depth_anything_v2
        return jsonify({
            "available": depth_anything_v2.check_depth_anything_v2_available(),
            "models": depth_anything_v2.DA2_MODELS,
            "licence": "Apache-2.0",
            "install_hint": depth_anything_v2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "depth_info")


# =============================================================
# N2.3 — SAM2 + Depth Compositor Pipeline
# =============================================================

@wave_l_bp.route("/video/compose/depth_segment", methods=["POST"])
@require_csrf
@async_job("depth_segment_compose")
def route_depth_segment_compose(job_id, filepath, data):
    """SAM2 segmentation + Depth estimation + layered compositing."""
    from opencut.core import compose_depth_segment
    if not compose_depth_segment.check_composite_available():
        raise RuntimeError("Pipeline deps not installed. " + compose_depth_segment.INSTALL_HINT)

    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a non-empty list")

    effects = data.get("effects", ["blur_background"])
    if not isinstance(effects, list):
        effects = ["blur_background"]

    bg_img = str(data.get("background_image") or "").strip()
    if bg_img:
        from opencut.security import validate_filepath
        bg_img = validate_filepath(bg_img)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = compose_depth_segment.compose(
        video_path=filepath, prompts=prompts, effects=effects,
        background_color=str(data.get("background_color") or ""),
        background_image=bg_img,
        blur_strength=safe_float(data.get("blur_strength"), 15.0, min_val=1.0, max_val=99.0),
        parallax_shift=safe_float(data.get("parallax_shift"), 20.0, min_val=-100.0, max_val=100.0),
        sam_model=str(data.get("sam_model") or "small"),
        depth_model=str(data.get("depth_model") or "small"),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/video/compose/depth_segment/info", methods=["GET"])
def route_compose_info():
    try:
        from opencut.core import compose_depth_segment
        return jsonify({
            "available": compose_depth_segment.check_composite_available(),
            "effects": compose_depth_segment.COMPOSITE_EFFECTS,
            "install_hint": compose_depth_segment.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "compose_info")


# =============================================================
# N3.1 — CogVideoX T2V + I2V
# =============================================================

@wave_l_bp.route("/generate/cogvideox", methods=["POST"])
@require_csrf
@async_job("cogvideox_t2v", filepath_required=False)
def route_cogvideox_t2v(job_id, filepath, data):
    """Generate video from text using CogVideoX (12 GB VRAM)."""
    from opencut.core import gen_video_cogvideox
    if not gen_video_cogvideox.check_cogvideox_available():
        raise RuntimeError("CogVideoX deps not installed. " + gen_video_cogvideox.INSTALL_HINT)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "CogVideoX-5B").strip()
    if model not in gen_video_cogvideox.COGVIDEOX_MODELS:
        model = "CogVideoX-5B"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_cogvideox.generate_t2v(
        prompt=prompt, model=model,
        num_frames=safe_int(data.get("num_frames"), 49, min_val=8, max_val=81),
        guidance_scale=safe_float(data.get("guidance_scale"), 6.0, min_val=1.0, max_val=15.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/cogvideox/i2v", methods=["POST"])
@require_csrf
@async_job("cogvideox_i2v", filepath_required=True, filepath_param="image_path")
def route_cogvideox_i2v(job_id, filepath, data):
    """Generate video from image using CogVideoX I2V."""
    from opencut.core import gen_video_cogvideox
    if not gen_video_cogvideox.check_cogvideox_available():
        raise RuntimeError("CogVideoX deps not installed. " + gen_video_cogvideox.INSTALL_HINT)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_cogvideox.generate_i2v(
        image_path=filepath, prompt=str(data.get("prompt") or ""),
        model=str(data.get("model") or "CogVideoX1.5-5B"),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/cogvideox/info", methods=["GET"])
def route_cogvideox_info():
    try:
        from opencut.core import gen_video_cogvideox
        return jsonify({
            "available": gen_video_cogvideox.check_cogvideox_available(),
            "models": gen_video_cogvideox.COGVIDEOX_MODELS,
            "licence": "Apache-2.0",
            "min_vram_gb": 12,
            "install_hint": gen_video_cogvideox.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "cogvideox_info")


# =============================================================
# N3.2 — Qwen2.5-VL Smart Timeline Analysis
# =============================================================

@wave_l_bp.route("/analyze/video/vl", methods=["POST"])
@require_csrf
@async_job("qwen_vl_analyze")
def route_vl_analyze(job_id, filepath, data):
    """Analyze video content using Qwen2.5-VL vision-language model."""
    from opencut.core import analyze_vl_qwen
    if not analyze_vl_qwen.check_qwen_vl_available():
        raise RuntimeError("Qwen VL deps not installed. " + analyze_vl_qwen.INSTALL_HINT)

    analysis_type = str(data.get("analysis_type") or "describe_scenes").strip()
    if analysis_type not in analyze_vl_qwen.ANALYSIS_TYPES:
        analysis_type = "describe_scenes"

    model = str(data.get("model") or "Qwen2.5-VL-7B").strip()
    if model not in analyze_vl_qwen.VL_MODELS:
        model = "Qwen2.5-VL-7B"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = analyze_vl_qwen.analyze_video(
        video_path=filepath, analysis_type=analysis_type,
        custom_query=str(data.get("query") or data.get("custom_query") or ""),
        model=model,
        max_frames=safe_int(data.get("max_frames"), 16, min_val=1, max_val=64),
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/analyze/video/vl/info", methods=["GET"])
def route_vl_info():
    try:
        from opencut.core import analyze_vl_qwen
        return jsonify({
            "available": analyze_vl_qwen.check_qwen_vl_available(),
            "models": analyze_vl_qwen.VL_MODELS,
            "analysis_types": analyze_vl_qwen.ANALYSIS_TYPES,
            "licence": "Apache-2.0",
            "install_hint": analyze_vl_qwen.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "vl_info")


# =============================================================
# N3.4 — CSM-1B Conversational Speech
# =============================================================

@wave_l_bp.route("/audio/speech/csm", methods=["POST"])
@require_csrf
@async_job("csm_speech", filepath_required=False)
def route_csm_speech(job_id, filepath, data):
    """Generate contextual conversational speech using CSM-1B."""
    from opencut.core import tts_csm
    if not tts_csm.check_csm_available():
        raise RuntimeError("CSM-1B not available. " + tts_csm.INSTALL_HINT)

    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    context = data.get("context", [])
    if not isinstance(context, list):
        context = []

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = tts_csm.generate(
        text=text, context=context,
        speaker_id=safe_int(data.get("speaker_id"), 0, min_val=0, max_val=1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/audio/speech/csm/info", methods=["GET"])
def route_csm_info():
    try:
        from opencut.core import tts_csm
        return jsonify({
            "available": tts_csm.check_csm_available(),
            "model": "CSM-1B",
            "licence": "Apache-2.0 (code); Meta Llama Community License (backbone)",
            "requires_ack": True,
            "install_hint": tts_csm.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "csm_info")


# =============================================================
# O1.1 — Dia 1.6B Dialogue TTS
# =============================================================

@wave_l_bp.route("/audio/speech/dia", methods=["POST"])
@require_csrf
@async_job("dia_dialogue", filepath_required=False)
def route_dia_speech(job_id, filepath, data):
    """Generate multi-speaker dialogue with nonverbal sounds via Dia."""
    from opencut.core import tts_dia
    if not tts_dia.check_dia_available():
        raise RuntimeError("Dia not installed. " + tts_dia.INSTALL_HINT)

    turns = data.get("turns", [])
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns must be a non-empty list of {speaker, text}")

    ref = str(data.get("reference_audio") or "").strip()
    if ref:
        from opencut.security import validate_filepath
        ref = validate_filepath(ref)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = tts_dia.generate_dialogue(
        turns=turns, reference_audio=ref,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/audio/speech/dia/info", methods=["GET"])
def route_dia_info():
    try:
        from opencut.core import tts_dia
        return jsonify({
            "available": tts_dia.check_dia_available(),
            "nonverbals": tts_dia.NONVERBALS,
            "licence": "Apache-2.0",
            "install_hint": tts_dia.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "dia_info")


# =============================================================
# O1.2 — Parler-TTS Voice Description
# =============================================================

@wave_l_bp.route("/audio/speech/parler", methods=["POST"])
@require_csrf
@async_job("parler_tts", filepath_required=False)
def route_parler_speech(job_id, filepath, data):
    """Synthesize speech from natural language voice description."""
    from opencut.core import tts_parler
    if not tts_parler.check_parler_available():
        raise RuntimeError("parler-tts not installed. " + tts_parler.INSTALL_HINT)

    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")

    model = str(data.get("model") or "mini").strip()
    if model not in tts_parler.PARLER_MODELS:
        model = "mini"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = tts_parler.synthesize(
        text=text,
        description=str(data.get("description") or ""),
        speaker=str(data.get("speaker") or ""),
        model=model,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/audio/speech/parler/speakers", methods=["GET"])
def route_parler_speakers():
    try:
        from opencut.core import tts_parler
        return jsonify({
            "available": tts_parler.check_parler_available(),
            "speakers": tts_parler.list_speakers(),
            "models": tts_parler.PARLER_MODELS,
            "install_hint": tts_parler.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "parler_speakers")


@wave_l_bp.route("/audio/speech/parler/info", methods=["GET"])
def route_parler_info():
    try:
        from opencut.core import tts_parler
        return jsonify({
            "available": tts_parler.check_parler_available(),
            "models": tts_parler.PARLER_MODELS,
            "licence": "Apache-2.0",
            "install_hint": tts_parler.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "parler_info")


# =============================================================
# O2.1 — LTX-Video T2V + I2V + Extension
# =============================================================

@wave_l_bp.route("/generate/ltxv/t2v", methods=["POST"])
@require_csrf
@async_job("ltxv_t2v", filepath_required=False)
def route_ltxv_t2v(job_id, filepath, data):
    """Generate video from text using LTX-Video (fastest Apache-2 DiT)."""
    from opencut.core import gen_video_ltxv
    if not gen_video_ltxv.check_ltxv_available():
        raise RuntimeError("LTX-Video deps not installed. " + gen_video_ltxv.INSTALL_HINT)

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    model = str(data.get("model") or "LTXV-2B").strip()
    if model not in gen_video_ltxv.LTXV_MODELS:
        model = "LTXV-2B"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_ltxv.generate_t2v(
        prompt=prompt, model=model,
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=60.0),
        negative_prompt=str(data.get("negative_prompt") or ""),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/ltxv/i2v", methods=["POST"])
@require_csrf
@async_job("ltxv_i2v", filepath_required=True, filepath_param="image_path")
def route_ltxv_i2v(job_id, filepath, data):
    """Generate video from image using LTX-Video I2V."""
    from opencut.core import gen_video_ltxv
    if not gen_video_ltxv.check_ltxv_available():
        raise RuntimeError("LTX-Video deps not installed. " + gen_video_ltxv.INSTALL_HINT)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_ltxv.generate_i2v(
        image_path=filepath, prompt=str(data.get("prompt") or ""),
        model=str(data.get("model") or "LTXV-2B"),
        duration=safe_float(data.get("duration"), 5.0, min_val=1.0, max_val=60.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/ltxv/extend", methods=["POST"])
@require_csrf
@async_job("ltxv_extend")
def route_ltxv_extend(job_id, filepath, data):
    """Extend a video forward or backward in time via LTX-Video."""
    from opencut.core import gen_video_ltxv
    if not gen_video_ltxv.check_ltxv_available():
        raise RuntimeError("LTX-Video deps not installed. " + gen_video_ltxv.INSTALL_HINT)

    direction = str(data.get("direction") or "forward").strip()
    if direction not in ("forward", "backward"):
        direction = "forward"

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = gen_video_ltxv.extend_video(
        video_path=filepath, direction=direction,
        duration_sec=safe_float(data.get("duration"), 3.0, min_val=1.0, max_val=30.0),
        model=str(data.get("model") or "LTXV-2B"),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/ltxv/info", methods=["GET"])
def route_ltxv_info():
    try:
        from opencut.core import gen_video_ltxv
        return jsonify({
            "available": gen_video_ltxv.check_ltxv_available(),
            "models": gen_video_ltxv.LTXV_MODELS,
            "licence": "Apache-2.0",
            "max_duration_seconds": 60,
            "install_hint": gen_video_ltxv.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "ltxv_info")


# =============================================================
# O3.1 — YuE Lyrics-to-Full-Song
# =============================================================

@wave_l_bp.route("/audio/music/yue", methods=["POST"])
@require_csrf
@async_job("yue_music", filepath_required=False)
def route_yue_music(job_id, filepath, data):
    """Generate complete song from structured lyrics via YuE."""
    from opencut.core import music_yue
    if not music_yue.check_yue_available():
        raise RuntimeError("YuE not installed. " + music_yue.INSTALL_HINT)

    lyrics = str(data.get("lyrics") or "").strip()
    if not lyrics:
        raise ValueError("lyrics is required (use [verse]/[chorus] tags)")

    genre = str(data.get("genre") or "pop").strip().lower()
    if genre not in music_yue.YUE_GENRES:
        genre = "pop"

    language = str(data.get("language") or "en").strip().lower()
    if language not in music_yue.YUE_LANGUAGES:
        language = "en"

    ref_vocal = str(data.get("reference_vocal") or "").strip()
    if ref_vocal:
        from opencut.security import validate_filepath
        ref_vocal = validate_filepath(ref_vocal)

    ref_backing = str(data.get("reference_backing") or "").strip()
    if ref_backing:
        from opencut.security import validate_filepath
        ref_backing = validate_filepath(ref_backing)

    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))

    result = music_yue.generate(
        lyrics=lyrics, genre=genre, language=language,
        instrumental_only=safe_bool(data.get("instrumental_only"), False),
        reference_vocal=ref_vocal, reference_backing=ref_backing,
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/audio/music/yue/info", methods=["GET"])
def route_yue_info():
    try:
        from opencut.core import music_yue
        return jsonify({
            "available": music_yue.check_yue_available(),
            "genres": music_yue.YUE_GENRES,
            "languages": music_yue.YUE_LANGUAGES,
            "licence": "Apache-2.0",
            "install_hint": music_yue.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "yue_info")
