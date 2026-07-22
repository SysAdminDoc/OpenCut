"""Identity, animation, and digital-avatar generation routes formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

from .wave_l_contract import wave_l_bp

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
# P1.1 — ConsisID Identity-Preserving T2V
# =============================================================

@wave_l_bp.route("/generate/consisid", methods=["POST"])
@require_csrf
@async_job("consisid", filepath_required=True, filepath_param="face_image")
def route_consisid(job_id, filepath, data):
    """Generate identity-preserving video from face + prompt."""
    from opencut.core import gen_video_consisid
    if not gen_video_consisid.check_consisid_available():
        raise RuntimeError("ConsisID not installed. " + gen_video_consisid.INSTALL_HINT)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = gen_video_consisid.generate(
        face_image=filepath, prompt=prompt,
        duration=safe_float(data.get("duration"), 6.0, min_val=2.0, max_val=12.0),
        seed=safe_int(data.get("seed"), -1),
        output_path=str(data.get("output") or "") or "",
        on_progress=_prog,
    )
    return {k: getattr(result, k) for k in result.keys()}


@wave_l_bp.route("/generate/consisid/info", methods=["GET"])
def route_consisid_info():
    try:
        from opencut.core import gen_video_consisid
        return jsonify({"available": gen_video_consisid.check_consisid_available(),
                        "licence": "Apache-2.0", "min_vram_gb": 18,
                        "install_hint": gen_video_consisid.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "consisid_info")
