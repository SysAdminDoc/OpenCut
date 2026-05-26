"""
OpenCut Wave Q + R + S Routes v1.33.0

Closes the routing gap reported in RESEARCH_FEATURE_PLAN_2026-05-25.md (Q1):
Waves Q (commit b3201f1), R (commit b3201f1), and S (commit a8f62c0) merged
core modules without registering HTTP routes. This file routes every entry
point as a 503-stub guarded by ``check_X_available()``. The pattern matches
``wave_l_routes.py`` (and Wave H's ``_stub_503`` convention): each route is
``@require_csrf`` + ``@async_job`` for processing endpoints and a plain GET
``/info`` reporter alongside.

Wave Q — Compositing, Voice Gen, Infinite Video, Foley, Lip Sync
  Q1.1  VACE compositing     — /video/compose/vace
  Q2.1  CosyVoice 2.0        — /audio/tts/cosyvoice
  Q2.2  MaskGCT TTS          — /audio/tts/maskgct
  Q3.1  OmniGen2             — /image/generate/omnigen2
  Q3.2  SkyReels V2 T2V      — /generate/skyreels2/t2v
  Q3.3  SkyReels V3 avatar   — /generate/skyreels3/avatar

Wave R — Foley + Lip Sync + Camera-controlled I2V + Consumer/HPC T2V
  R1.1  EzAudio foley/SFX    — /audio/foley/ezaudio
  R2.1  MuseTalk 1.5 lip-sync— /lipsync/musetalk
  R2.2  VideoX-Fun control   — /generate/videox-fun
  R3.1  Mochi-1 T2V          — /generate/mochi
  R3.2  Step-Video T2V       — /generate/stepvideo

Wave S — Relighting, VSR, ASR, VLM, Face Tools
  S1.1  IC-Light V2 relight  — /video/relight/iclight
  S1.2  Light-A-Video relight— /video/relight/lav
  S1.3  DiffusionRenderer    — /video/relight/diffrenderer
  S2.1  SeedVR2 VSR          — /video/upscale/seedvr2
  S2.2  Parakeet TDT ASR     — /audio/transcribe/parakeet
  S2.3  Canary-1B-Flash ASR  — /audio/transcribe/canary
  S3.1  Qwen3-VL analysis    — /analyze/video/qwen3vl
  S3.2  InternVL3 analysis   — /analyze/video/internvl3
  S3.3  Face Re-aging FRAN   — /video/face/reage
  S3.4  HeartMuLa music gen  — /audio/music/heartmula
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int

logger = logging.getLogger("opencut")
wave_qrs_bp = Blueprint("wave_qrs", __name__)


def _result_dict(result) -> dict:
    """Serialize a wave-style dataclass result by walking its ``keys()``."""
    try:
        return {k: getattr(result, k) for k in result.keys()}
    except Exception:
        return {"output": getattr(result, "output", "")}


def _prog_factory(job_id):
    def _prog(p, m=""):
        _update_job(job_id, progress=int(p), message=str(m))
    return _prog


# =============================================================
# Wave Q
# =============================================================

# Q1.1 — VACE compositing -------------------------------------
@wave_qrs_bp.route("/video/compose/vace", methods=["POST"])
@require_csrf
@async_job("vace_compose", filepath_required=False)
def route_vace_compose(job_id, filepath, data):
    """Mask + prompt compositing via VACE (V2V / MV2V / R2V modes)."""
    from opencut.core import video_compose_vace
    if not video_compose_vace.check_wan_available():
        raise RuntimeError("VACE not installed. " + video_compose_vace.INSTALL_HINT)
    result = video_compose_vace.compose(
        mode=str(data.get("mode") or "v2v"),
        video_path=str(data.get("video_path") or ""),
        mask_path=str(data.get("mask_path") or ""),
        reference_paths=list(data.get("reference_paths") or []),
        prompt=str(data.get("prompt") or ""),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/video/compose/vace/info", methods=["GET"])
def route_vace_info():
    try:
        from opencut.core import video_compose_vace
        return jsonify({
            "available": video_compose_vace.check_wan_available(),
            "modes": ["v2v", "mv2v", "r2v"],
            "licence": "Apache-2.0",
            "install_hint": video_compose_vace.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "vace_info")


# Q2.1 — CosyVoice 2.0 ----------------------------------------
@wave_qrs_bp.route("/audio/tts/cosyvoice", methods=["POST"])
@require_csrf
@async_job("cosyvoice_tts", filepath_required=False)
def route_cosyvoice_tts(job_id, filepath, data):
    """Zero-shot voice clone + 9-language streaming via CosyVoice 2.0."""
    from opencut.core import tts_cosyvoice
    if not tts_cosyvoice.check_cosyvoice_available():
        raise RuntimeError("CosyVoice not installed. " + tts_cosyvoice.INSTALL_HINT)
    result = tts_cosyvoice.synthesize(
        text=str(data.get("text") or ""),
        language=str(data.get("language") or "en"),
        reference_audio=str(data.get("reference_audio") or "") or None,
        speaker=str(data.get("speaker") or "") or None,
        stream=bool(data.get("stream")),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/audio/tts/cosyvoice/info", methods=["GET"])
def route_cosyvoice_info():
    try:
        from opencut.core import tts_cosyvoice
        return jsonify({
            "available": tts_cosyvoice.check_cosyvoice_available(),
            "licence": "Apache-2.0",
            "install_hint": tts_cosyvoice.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "cosyvoice_info")


# Q2.2 — MaskGCT ---------------------------------------------
@wave_qrs_bp.route("/audio/tts/maskgct", methods=["POST"])
@require_csrf
@async_job("maskgct_tts", filepath_required=False)
def route_maskgct_tts(job_id, filepath, data):
    """Non-autoregressive parallel TTS via MaskGCT (Amphion)."""
    from opencut.core import tts_maskgct
    if not tts_maskgct.check_amphion_available():
        raise RuntimeError("Amphion/MaskGCT not installed. " + tts_maskgct.INSTALL_HINT)
    result = tts_maskgct.synthesize(
        text=str(data.get("text") or ""),
        reference_audio=str(data.get("reference_audio") or "") or None,
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/audio/tts/maskgct/info", methods=["GET"])
def route_maskgct_info():
    try:
        from opencut.core import tts_maskgct
        return jsonify({
            "available": tts_maskgct.check_amphion_available(),
            "licence": "MIT",
            "install_hint": tts_maskgct.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "maskgct_info")


# Q3.1 — OmniGen2 --------------------------------------------
@wave_qrs_bp.route("/image/generate/omnigen2", methods=["POST"])
@require_csrf
@async_job("omnigen2", filepath_required=False)
def route_omnigen2(job_id, filepath, data):
    """Multi-reference image generation via OmniGen2."""
    from opencut.core import t2i_omnigen2
    if not t2i_omnigen2.check_omnigen2_available():
        raise RuntimeError("OmniGen2 not installed. " + t2i_omnigen2.INSTALL_HINT)
    result = t2i_omnigen2.generate(
        prompt=str(data.get("prompt") or ""),
        reference_images=list(data.get("reference_images") or []),
        width=safe_int(data.get("width"), 1024, min_val=64, max_val=4096),
        height=safe_int(data.get("height"), 1024, min_val=64, max_val=4096),
        guidance_scale=safe_float(data.get("guidance_scale"), 5.0),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/image/generate/omnigen2/info", methods=["GET"])
def route_omnigen2_info():
    try:
        from opencut.core import t2i_omnigen2
        return jsonify({
            "available": t2i_omnigen2.check_omnigen2_available(),
            "licence": "Apache-2.0",
            "install_hint": t2i_omnigen2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "omnigen2_info")


# Q3.2 — SkyReels V2 ------------------------------------------
@wave_qrs_bp.route("/generate/skyreels2/t2v", methods=["POST"])
@require_csrf
@async_job("skyreels2_t2v", filepath_required=False)
def route_skyreels2_t2v(job_id, filepath, data):
    """Infinite-length text-to-video via SkyReels V2 Diffusion Forcing."""
    from opencut.core import gen_video_skyreels2
    if not gen_video_skyreels2.check_diffusers_available():
        raise RuntimeError("diffusers not installed. " + gen_video_skyreels2.INSTALL_HINT)
    result = gen_video_skyreels2.generate_t2v(
        prompt=str(data.get("prompt") or ""),
        duration_seconds=safe_float(data.get("duration_seconds"), 8.0),
        width=safe_int(data.get("width"), 720, min_val=64, max_val=4096),
        height=safe_int(data.get("height"), 1280, min_val=64, max_val=4096),
        fps=safe_int(data.get("fps"), 24, min_val=1, max_val=120),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/generate/skyreels2/info", methods=["GET"])
def route_skyreels2_info():
    try:
        from opencut.core import gen_video_skyreels2
        return jsonify({
            "available": gen_video_skyreels2.check_diffusers_available(),
            "licence": "Apache-2.0",
            "install_hint": gen_video_skyreels2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "skyreels2_info")


# Q3.3 — SkyReels V3 ------------------------------------------
@wave_qrs_bp.route("/generate/skyreels3/avatar", methods=["POST"])
@require_csrf
@async_job("skyreels3_avatar", filepath_required=False)
def route_skyreels3_avatar(job_id, filepath, data):
    """Talking-avatar generation up to 200 s via SkyReels V3."""
    from opencut.core import gen_video_skyreels3
    if not gen_video_skyreels3.check_skyreels_available():
        raise RuntimeError("skyreels not installed. " + gen_video_skyreels3.INSTALL_HINT)
    result = gen_video_skyreels3.generate_avatar(
        portrait_path=str(data.get("portrait_path") or ""),
        audio_path=str(data.get("audio_path") or ""),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/generate/skyreels3/info", methods=["GET"])
def route_skyreels3_info():
    try:
        from opencut.core import gen_video_skyreels3
        return jsonify({
            "available": gen_video_skyreels3.check_skyreels_available(),
            "licence": "Skywork Community License",
            "requires_ack": True,
            "install_hint": gen_video_skyreels3.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "skyreels3_info")


# =============================================================
# Wave R
# =============================================================

# R1.1 — EzAudio foley/SFX -----------------------------------
@wave_qrs_bp.route("/audio/foley/ezaudio", methods=["POST"])
@require_csrf
@async_job("ezaudio_foley", filepath_required=False)
def route_ezaudio_foley(job_id, filepath, data):
    """Text-to-foley / SFX generation via EzAudio."""
    from opencut.core import ezaudio_service
    if not ezaudio_service.check_ezaudio_available():
        raise RuntimeError("EzAudio not installed. " + ezaudio_service.INSTALL_HINT)
    result = ezaudio_service.generate(
        prompt=str(data.get("prompt") or ""),
        duration_seconds=safe_float(data.get("duration_seconds"), 4.0),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/audio/foley/ezaudio/info", methods=["GET"])
def route_ezaudio_info():
    try:
        from opencut.core import ezaudio_service
        return jsonify({
            "available": ezaudio_service.check_ezaudio_available(),
            "licence": "MIT",
            "sample_rate": 44100,
            "install_hint": ezaudio_service.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "ezaudio_info")


# R2.1 — MuseTalk 1.5 -----------------------------------------
@wave_qrs_bp.route("/lipsync/musetalk", methods=["POST"])
@require_csrf
@async_job("musetalk_lipsync", filepath_required=False)
def route_musetalk_lipsync(job_id, filepath, data):
    """30 fps+ real-time lip-sync via MuseTalk."""
    from opencut.core import musetalk_service
    if not musetalk_service.check_musetalk_available():
        raise RuntimeError("MuseTalk not installed. " + musetalk_service.INSTALL_HINT)
    result = musetalk_service.lip_sync(
        video_path=str(data.get("video_path") or ""),
        audio_path=str(data.get("audio_path") or ""),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/lipsync/musetalk/info", methods=["GET"])
def route_musetalk_info():
    try:
        from opencut.core import musetalk_service
        return jsonify({
            "available": musetalk_service.check_musetalk_available(),
            "licence": "MIT (code) / CreativeML-OpenRAIL-M (weights)",
            "install_hint": musetalk_service.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "musetalk_info")


# R2.2 — VideoX-Fun control I2V -------------------------------
@wave_qrs_bp.route("/generate/videox-fun", methods=["POST"])
@require_csrf
@async_job("videox_fun", filepath_required=False)
def route_videox_fun(job_id, filepath, data):
    """Camera-controlled / structural-controlled image-to-video via VideoX-Fun."""
    from opencut.core import videox_fun_service
    if not videox_fun_service.check_videox_fun_available():
        raise RuntimeError("videox-fun not installed. " + videox_fun_service.INSTALL_HINT)
    result = videox_fun_service.generate(
        image_path=str(data.get("image_path") or ""),
        prompt=str(data.get("prompt") or ""),
        control_type=str(data.get("control_type") or "camera"),
        control_path=str(data.get("control_path") or "") or None,
        duration_seconds=safe_float(data.get("duration_seconds"), 4.0),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/generate/videox-fun/info", methods=["GET"])
def route_videox_fun_info():
    try:
        from opencut.core import videox_fun_service
        return jsonify({
            "available": videox_fun_service.check_videox_fun_available(),
            "licence": "Apache-2.0",
            "install_hint": videox_fun_service.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "videox_fun_info")


# R3.1 — Mochi-1 -----------------------------------------------
@wave_qrs_bp.route("/generate/mochi", methods=["POST"])
@require_csrf
@async_job("mochi_t2v", filepath_required=False)
def route_mochi(job_id, filepath, data):
    """Consumer-GPU 10B-parameter text-to-video via Mochi-1."""
    from opencut.core import mochi_service
    if not mochi_service.check_diffusers_available():
        raise RuntimeError("diffusers not installed. " + mochi_service.INSTALL_HINT)
    result = mochi_service.generate(
        prompt=str(data.get("prompt") or ""),
        duration_seconds=safe_float(data.get("duration_seconds"), 5.0),
        width=safe_int(data.get("width"), 848, min_val=64, max_val=4096),
        height=safe_int(data.get("height"), 480, min_val=64, max_val=4096),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/generate/mochi/info", methods=["GET"])
def route_mochi_info():
    try:
        from opencut.core import mochi_service
        return jsonify({
            "available": mochi_service.check_diffusers_available(),
            "licence": "Apache-2.0",
            "install_hint": mochi_service.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "mochi_info")


# R3.2 — Step-Video --------------------------------------------
@wave_qrs_bp.route("/generate/stepvideo", methods=["POST"])
@require_csrf
@async_job("stepvideo_t2v", filepath_required=False)
def route_stepvideo(job_id, filepath, data):
    """HPC-class 30B-parameter text-to-video via Step-Video."""
    from opencut.core import stepvideo_service
    if not stepvideo_service.check_stepvideo_available():
        raise RuntimeError("stepvideo not installed. " + stepvideo_service.INSTALL_HINT)
    result = stepvideo_service.generate_t2v(
        prompt=str(data.get("prompt") or ""),
        duration_seconds=safe_float(data.get("duration_seconds"), 8.0),
        width=safe_int(data.get("width"), 1024, min_val=64, max_val=4096),
        height=safe_int(data.get("height"), 576, min_val=64, max_val=4096),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/generate/stepvideo/info", methods=["GET"])
def route_stepvideo_info():
    try:
        from opencut.core import stepvideo_service
        return jsonify({
            "available": stepvideo_service.check_stepvideo_available(),
            "licence": "MIT",
            "hardware": "Linux, sm_80+ GPU required",
            "install_hint": stepvideo_service.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "stepvideo_info")


# =============================================================
# Wave S
# =============================================================

# S1.1 — IC-Light V2 -----------------------------------------
@wave_qrs_bp.route("/video/relight/iclight", methods=["POST"])
@require_csrf
@async_job("iclight_relight", filepath_required=True)
def route_iclight_relight(job_id, filepath, data):
    """Per-frame relighting via IC-Light V2 (text or background reference)."""
    from opencut.core import relight_iclight
    if not relight_iclight.check_diffusers_available():
        raise RuntimeError("diffusers not installed. " + relight_iclight.INSTALL_HINT)
    result = relight_iclight.relight(
        video_path=filepath,
        mode=str(data.get("mode") or "text"),
        prompt=str(data.get("prompt") or ""),
        background_path=str(data.get("background_path") or "") or None,
        strength=safe_float(data.get("strength"), 0.7),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/video/relight/iclight/info", methods=["GET"])
def route_iclight_info():
    try:
        from opencut.core import relight_iclight
        return jsonify({
            "available": relight_iclight.check_diffusers_available(),
            "modes": ["text", "background"],
            "licence": "Apache-2.0 (code); IC-Light weights research-friendly",
            "install_hint": relight_iclight.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "iclight_info")


# S1.2 — Light-A-Video ---------------------------------------
@wave_qrs_bp.route("/video/relight/lav", methods=["POST"])
@require_csrf
@async_job("lav_relight", filepath_required=True)
def route_lav_relight(job_id, filepath, data):
    """Training-free temporal video relighting via Light-A-Video."""
    from opencut.core import relight_video_lav
    if not relight_video_lav.check_diffusers_available():
        raise RuntimeError("diffusers not installed. " + relight_video_lav.INSTALL_HINT)
    result = relight_video_lav.relight_video(
        video_path=filepath,
        prompt=str(data.get("prompt") or ""),
        strength=safe_float(data.get("strength"), 0.6),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/video/relight/lav/info", methods=["GET"])
def route_lav_info():
    try:
        from opencut.core import relight_video_lav
        return jsonify({
            "available": relight_video_lav.check_diffusers_available(),
            "licence": "Apache-2.0",
            "install_hint": relight_video_lav.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "lav_info")


# S1.3 — DiffusionRenderer -----------------------------------
@wave_qrs_bp.route("/video/relight/diffrenderer", methods=["POST"])
@require_csrf
@async_job("diffrenderer", filepath_required=True)
def route_diffrenderer(job_id, filepath, data):
    """Physically-grounded inverse + forward rendering."""
    from opencut.core import relight_diffrenderer
    if not relight_diffrenderer.check_diffusionrenderer_available():
        raise RuntimeError(
            "diffusionrenderer not installed. " + relight_diffrenderer.INSTALL_HINT
        )
    result = relight_diffrenderer.relight(
        video_path=filepath,
        light_prompt=str(data.get("light_prompt") or ""),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/video/relight/diffrenderer/info", methods=["GET"])
def route_diffrenderer_info():
    try:
        from opencut.core import relight_diffrenderer
        return jsonify({
            "available": relight_diffrenderer.check_diffusionrenderer_available(),
            "licence": "Apache-2.0",
            "install_hint": relight_diffrenderer.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "diffrenderer_info")


# S2.1 — SeedVR2 ---------------------------------------------
@wave_qrs_bp.route("/video/upscale/seedvr2", methods=["POST"])
@require_csrf
@async_job("seedvr2_vsr", filepath_required=True)
def route_seedvr2_upscale(job_id, filepath, data):
    """One-step diffusion video super-resolution via SeedVR2."""
    from opencut.core import upscale_seedvr2
    if not upscale_seedvr2.check_diffusers_available():
        raise RuntimeError("diffusers not installed. " + upscale_seedvr2.INSTALL_HINT)
    result = upscale_seedvr2.upscale(
        video_path=filepath,
        scale=safe_int(data.get("scale"), 2, min_val=2, max_val=4),
        model=str(data.get("model") or "3b"),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/video/upscale/seedvr2/info", methods=["GET"])
def route_seedvr2_info():
    try:
        from opencut.core import upscale_seedvr2
        return jsonify({
            "available": upscale_seedvr2.check_diffusers_available(),
            "models": ["3b", "7b"],
            "licence": "Apache-2.0",
            "install_hint": upscale_seedvr2.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "seedvr2_info")


# S2.2 — Parakeet TDT ----------------------------------------
@wave_qrs_bp.route("/audio/transcribe/parakeet", methods=["POST"])
@require_csrf
@async_job("parakeet_asr", filepath_required=True)
def route_parakeet_transcribe(job_id, filepath, data):
    """Sub-200 ms streaming ASR via NVIDIA Parakeet TDT."""
    from opencut.core import asr_parakeet
    if not asr_parakeet.check_nemo_toolkit_available():
        raise RuntimeError("NeMo toolkit not installed. " + asr_parakeet.INSTALL_HINT)
    result = asr_parakeet.transcribe(
        audio_path=filepath,
        language=str(data.get("language") or "en"),
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/audio/transcribe/parakeet/info", methods=["GET"])
def route_parakeet_info():
    try:
        from opencut.core import asr_parakeet
        return jsonify({
            "available": asr_parakeet.check_nemo_toolkit_available(),
            "licence": "Apache-2.0 (code); CC-BY-4.0 (model)",
            "latency_ms": 200,
            "install_hint": asr_parakeet.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "parakeet_info")


# S2.3 — Canary-1B-Flash -------------------------------------
@wave_qrs_bp.route("/audio/transcribe/canary", methods=["POST"])
@require_csrf
@async_job("canary_asr", filepath_required=True)
def route_canary_transcribe(job_id, filepath, data):
    """Batch ASR (RTFx 1000+) via NVIDIA Canary-1B-Flash."""
    from opencut.core import asr_canary
    if not asr_canary.check_nemo_toolkit_available():
        raise RuntimeError("NeMo toolkit not installed. " + asr_canary.INSTALL_HINT)
    result = asr_canary.transcribe_batch(
        audio_paths=list(data.get("audio_paths") or [filepath]),
        language=str(data.get("language") or "en"),
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/audio/transcribe/canary/info", methods=["GET"])
def route_canary_info():
    try:
        from opencut.core import asr_canary
        return jsonify({
            "available": asr_canary.check_nemo_toolkit_available(),
            "licence": "Apache-2.0 (code); CC-BY-4.0 (model)",
            "install_hint": asr_canary.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "canary_info")


# S3.1 — Qwen3-VL --------------------------------------------
@wave_qrs_bp.route("/analyze/video/qwen3vl", methods=["POST"])
@require_csrf
@async_job("qwen3vl_analyze", filepath_required=True)
def route_qwen3vl_analyze(job_id, filepath, data):
    """Multimodal video analysis (256k tokens, up to 2-hour clips)."""
    from opencut.core import multimodal_qwen3vl
    if not multimodal_qwen3vl.check_transformers_available():
        raise RuntimeError("transformers>=4.45 not installed. " + multimodal_qwen3vl.INSTALL_HINT)
    result = multimodal_qwen3vl.analyze(
        video_path=filepath,
        prompt=str(data.get("prompt") or "Summarise the video."),
        max_tokens=safe_int(data.get("max_tokens"), 1024, min_val=1, max_val=8192),
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/analyze/video/qwen3vl/info", methods=["GET"])
def route_qwen3vl_info():
    try:
        from opencut.core import multimodal_qwen3vl
        return jsonify({
            "available": multimodal_qwen3vl.check_transformers_available(),
            "context_tokens": 256000,
            "licence": "Apache-2.0",
            "install_hint": multimodal_qwen3vl.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "qwen3vl_info")


# S3.2 — InternVL3 --------------------------------------------
@wave_qrs_bp.route("/analyze/video/internvl3", methods=["POST"])
@require_csrf
@async_job("internvl3_analyze", filepath_required=True)
def route_internvl3_analyze(job_id, filepath, data):
    """Alternative multimodal VLM for vendor diversity."""
    from opencut.core import multimodal_internvl3
    if not multimodal_internvl3.check_transformers_available():
        raise RuntimeError("transformers>=4.45 not installed. " + multimodal_internvl3.INSTALL_HINT)
    result = multimodal_internvl3.analyze(
        video_path=filepath,
        prompt=str(data.get("prompt") or "Describe the video."),
        max_tokens=safe_int(data.get("max_tokens"), 1024, min_val=1, max_val=8192),
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/analyze/video/internvl3/info", methods=["GET"])
def route_internvl3_info():
    try:
        from opencut.core import multimodal_internvl3
        return jsonify({
            "available": multimodal_internvl3.check_transformers_available(),
            "licence": "Apache-2.0 (code); InternVL Community License (weights)",
            "install_hint": multimodal_internvl3.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "internvl3_info")


# S3.3 — Face Re-aging FRAN ----------------------------------
@wave_qrs_bp.route("/video/face/reage", methods=["POST"])
@require_csrf
@async_job("face_reage", filepath_required=True)
def route_face_reage(job_id, filepath, data):
    """Production VFX-quality age progression / regression (-30 to +30 years)."""
    from opencut.core import face_reage
    if not face_reage.check_face_reaging_available():
        raise RuntimeError("face_reaging not installed. " + face_reage.INSTALL_HINT)
    result = face_reage.transform(
        video_path=filepath,
        age_delta=safe_int(data.get("age_delta"), 0, min_val=-30, max_val=30),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/video/face/reage/info", methods=["GET"])
def route_face_reage_info():
    try:
        from opencut.core import face_reage
        return jsonify({
            "available": face_reage.check_face_reaging_available(),
            "age_range": [-30, 30],
            "licence": "MIT",
            "install_hint": face_reage.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "face_reage_info")


# S3.4 — HeartMuLa lyric-aligned music gen -------------------
@wave_qrs_bp.route("/audio/music/heartmula", methods=["POST"])
@require_csrf
@async_job("heartmula_music", filepath_required=False)
def route_heartmula(job_id, filepath, data):
    """Lyric-aligned music generation via HeartMuLa."""
    from opencut.core import music_heartmula
    if not music_heartmula.check_transformers_available():
        raise RuntimeError("transformers>=4.45 not installed. " + music_heartmula.INSTALL_HINT)
    result = music_heartmula.generate(
        lyrics=str(data.get("lyrics") or ""),
        style=str(data.get("style") or "pop"),
        duration_seconds=safe_float(data.get("duration_seconds"), 30.0),
        output=str(data.get("output") or "") or None,
        on_progress=_prog_factory(job_id),
    )
    return _result_dict(result)


@wave_qrs_bp.route("/audio/music/heartmula/info", methods=["GET"])
def route_heartmula_info():
    try:
        from opencut.core import music_heartmula
        return jsonify({
            "available": music_heartmula.check_transformers_available(),
            "licence": "Apache-2.0",
            "install_hint": music_heartmula.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "heartmula_info")
