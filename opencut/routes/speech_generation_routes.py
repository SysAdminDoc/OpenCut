"""Speech synthesis and transcription routes formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

from .wave_l_contract import wave_l_bp

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
