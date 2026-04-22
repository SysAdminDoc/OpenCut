"""
OpenCut Wave K Routes v1.28.0

Routes for Wave K modules: AudioSeal, Brand Kit, Batch Reframe,
Clip Rating, Subtitle QA, Profanity Censor, Spectral Match, Lottie,
Semantic Search, and all Tier 2/3 stubs.
"""
from __future__ import annotations

import json
import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import error_response, safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int, validate_path

logger = logging.getLogger("opencut")
wave_k_bp = Blueprint("wave_k", __name__)

_BK_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "brand_kit_settings.json")


def _stub_503(name: str, hint: str = "") -> tuple:
    return error_response(
        "DEPENDENCY_NOT_INSTALLED",
        f"{name} optional dependency is not installed.",
        status=503,
        suggestion=hint or "Check INSTALL_HINT in the module for install instructions.",
    )


def _stub_501(name: str) -> tuple:
    return error_response(
        "ROUTE_STUBBED",
        f"{name} is a Wave K Tier 3 strategic stub in v1.28.0.",
        status=501,
        suggestion="Track the ROADMAP-NEXT.md Wave K section.",
    )


# =============================================================
# K1.1 — AudioSeal Watermark
# =============================================================

@wave_k_bp.route("/audio/watermark/embed", methods=["POST"])
@require_csrf
@async_job("audio_watermark_embed")
def route_audio_watermark_embed(job_id, filepath, data):
    from opencut.core import audio_watermark
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    message = str(data.get("message") or "opencut")
    output = str(data.get("output") or "") or None
    result = audio_watermark.embed(filepath, message=message, output=output, on_progress=_prog)
    return {"output": result.output, "method": result.method, "notes": result.notes}


@wave_k_bp.route("/audio/watermark/detect", methods=["POST"])
@require_csrf
@async_job("audio_watermark_detect")
def route_audio_watermark_detect(job_id, filepath, data):
    from opencut.core import audio_watermark
    return audio_watermark.detect(filepath)


@wave_k_bp.route("/audio/watermark/info", methods=["GET"])
def route_audio_watermark_info():
    try:
        from opencut.core import audio_watermark
        return jsonify({
            "available": audio_watermark.check_audioseal_available(),
            "install_hint": audio_watermark.INSTALL_HINT,
        })
    except Exception as exc:
        return safe_error(exc, "audio_watermark_info")


# =============================================================
# K1.2 — Brand Kit Settings
# =============================================================

@wave_k_bp.route("/settings/brand-kit", methods=["GET"])
def route_brand_kit_get():
    try:
        if os.path.isfile(_BK_SETTINGS_PATH):
            with open(_BK_SETTINGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({})
    except Exception as exc:
        return safe_error(exc, "brand_kit_get")


@wave_k_bp.route("/settings/brand-kit", methods=["POST"])
@require_csrf
def route_brand_kit_save():
    try:
        data = request.get_json(force=True, silent=True) or {}
        os.makedirs(os.path.dirname(_BK_SETTINGS_PATH), exist_ok=True)
        with open(_BK_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return jsonify({"saved": True})
    except Exception as exc:
        return safe_error(exc, "brand_kit_save")


@wave_k_bp.route("/settings/brand-kit", methods=["DELETE"])
@require_csrf
def route_brand_kit_delete():
    try:
        if os.path.isfile(_BK_SETTINGS_PATH):
            os.remove(_BK_SETTINGS_PATH)
        return jsonify({"deleted": True})
    except Exception as exc:
        return safe_error(exc, "brand_kit_delete")


@wave_k_bp.route("/settings/brand-kit/preview", methods=["POST"])
@require_csrf
@async_job("brand_kit_preview")
def route_brand_kit_preview(job_id, filepath, data):
    from opencut.core import brand_kit as bk_mod
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    if os.path.isfile(_BK_SETTINGS_PATH):
        with open(_BK_SETTINGS_PATH, encoding="utf-8") as f:
            settings = json.load(f)
    else:
        settings = data.get("brand_kit") or {}
    fields = {f for f in bk_mod.BrandKit.__dataclass_fields__} if hasattr(bk_mod.BrandKit, "__dataclass_fields__") else set()
    kit_kwargs = {k: v for k, v in settings.items() if k in fields}
    kit = bk_mod.BrandKit(**kit_kwargs)
    output = str(data.get("output") or "") or None
    result = bk_mod.auto_correct_brand(filepath, brand_kit=kit, output_path_str=output, on_progress=_prog)
    out = result.get("output", "") if isinstance(result, dict) else getattr(result, "output", "")
    return {"output": str(out)}


# =============================================================
# K1.4 — Batch Reframe
# =============================================================

@wave_k_bp.route("/video/reframe/batch", methods=["POST"])
@require_csrf
@async_job("batch_reframe", filepath_required=False)
def route_batch_reframe(job_id, filepath, data):
    from opencut.core import batch_reframe
    input_path = validate_path(str(data.get("input_path") or ""))
    ratios = data.get("ratios") or None
    output_dir = str(data.get("output_dir") or "") or None
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    result = batch_reframe.batch_reframe(input_path, ratios=ratios, output_dir=output_dir, on_progress=_prog)
    return {"outputs": result.outputs, "ratios": result.ratios, "notes": result.notes}


@wave_k_bp.route("/video/reframe/batch/presets", methods=["GET"])
def route_batch_reframe_presets():
    try:
        from opencut.core.batch_reframe import RATIO_PRESETS
        return jsonify({r: {"width": w, "height": h} for r, (w, h) in RATIO_PRESETS.items()})
    except Exception as exc:
        return safe_error(exc, "batch_reframe_presets")


# =============================================================
# K1.5 — Clip Rating
# =============================================================

@wave_k_bp.route("/clips/rate", methods=["POST"])
@require_csrf
def route_clip_rate():
    try:
        from opencut.core import clip_rating
        data = request.get_json(force=True, silent=True) or {}
        clip_path = validate_path(str(data.get("clip_path") or ""))
        rating = data.get("rating")
        status = data.get("status")
        entry = clip_rating.rate(clip_path, rating=rating, status=status)
        return jsonify({"path": entry.path, "rating": entry.rating, "status": entry.status,
                        "tags": entry.tags, "updated": entry.updated})
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "clip_rate")


@wave_k_bp.route("/clips/tag", methods=["POST"])
@require_csrf
def route_clip_tag():
    try:
        from opencut.core import clip_rating
        data = request.get_json(force=True, silent=True) or {}
        clip_path = validate_path(str(data.get("clip_path") or ""))
        tags = list(data.get("tags") or [])
        entry = clip_rating.tag(clip_path, tags)
        return jsonify({"path": entry.path, "rating": entry.rating, "status": entry.status,
                        "tags": entry.tags, "updated": entry.updated})
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "clip_tag")


@wave_k_bp.route("/clips/untag", methods=["POST"])
@require_csrf
def route_clip_untag():
    try:
        from opencut.core import clip_rating
        data = request.get_json(force=True, silent=True) or {}
        clip_path = validate_path(str(data.get("clip_path") or ""))
        tags = list(data.get("tags") or [])
        entry = clip_rating.untag(clip_path, tags)
        return jsonify({"path": entry.path, "rating": entry.rating, "status": entry.status,
                        "tags": entry.tags, "updated": entry.updated})
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "clip_untag")


@wave_k_bp.route("/clips/search", methods=["POST"])
@require_csrf
def route_clip_search():
    try:
        from opencut.core import clip_rating
        data = request.get_json(force=True, silent=True) or {}
        query = str(data.get("query") or "")
        rating_min = safe_int(data.get("rating_min"), default=0)
        status = data.get("status") or None
        tags = list(data.get("tags") or []) or None
        results = clip_rating.search(query=query, rating_min=rating_min, status=status, tags=tags)
        return jsonify([{"path": e.path, "rating": e.rating, "status": e.status,
                         "tags": e.tags, "updated": e.updated} for e in results])
    except Exception as exc:
        return safe_error(exc, "clip_search")


@wave_k_bp.route("/clips/get", methods=["GET"])
def route_clip_get():
    try:
        from opencut.core import clip_rating
        clip_path = validate_path(str(request.args.get("clip_path") or ""))
        entry = clip_rating.get(clip_path)
        if entry is None:
            return jsonify(None)
        return jsonify({"path": entry.path, "rating": entry.rating, "status": entry.status,
                        "tags": entry.tags, "updated": entry.updated})
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "clip_get")


# =============================================================
# K1.6 — Subtitle QA
# =============================================================

@wave_k_bp.route("/captions/qa/validate", methods=["POST"])
@require_csrf
def route_subtitle_qa_validate():
    try:
        from opencut.core import subtitle_qa
        data = request.get_json(force=True, silent=True) or {}
        subtitle_path = validate_path(str(data.get("subtitle_path") or ""))
        profile = str(data.get("profile") or "netflix")
        report = subtitle_qa.validate(subtitle_path, profile=profile)
        return jsonify({
            "issues": [{"rule": i.rule, "severity": i.severity, "index": i.index,
                        "start": i.start, "end": i.end, "text": i.text, "detail": i.detail}
                       for i in report.issues],
            "passed": report.passed,
            "total_cues": report.total_cues,
            "profile": report.profile,
            "notes": report.notes,
        })
    except ValueError as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "subtitle_qa_validate")


@wave_k_bp.route("/captions/qa/profiles", methods=["GET"])
def route_subtitle_qa_profiles():
    try:
        from opencut.core.subtitle_qa import PROFILES
        return jsonify(PROFILES)
    except Exception as exc:
        return safe_error(exc, "subtitle_qa_profiles")


# =============================================================
# K1.7 — Profanity Censor
# =============================================================

@wave_k_bp.route("/audio/censor/profanity", methods=["POST"])
@require_csrf
@async_job("profanity_censor")
def route_profanity_censor(job_id, filepath, data):
    from opencut.core import profanity_censor
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    segments = list(data.get("transcript_segments") or [])
    mode = str(data.get("mode") or "bleep")
    custom_words = list(data.get("custom_words") or []) or None
    output = str(data.get("output") or "") or None
    result = profanity_censor.censor(filepath, transcript_segments=segments, mode=mode,
                                     custom_words=custom_words, output=output, on_progress=_prog)
    return {"output": result.output, "censor_count": result.censor_count,
            "censored_words": result.censored_words, "mode": result.mode, "notes": result.notes}


@wave_k_bp.route("/audio/censor/profanity/wordlists", methods=["GET"])
def route_profanity_wordlists():
    try:
        from opencut.core.profanity_censor import list_wordlists
        return jsonify(list_wordlists())
    except Exception as exc:
        return safe_error(exc, "profanity_wordlists")


# =============================================================
# K1.8 — Spectral Match
# =============================================================

@wave_k_bp.route("/audio/spectral-match", methods=["POST"])
@require_csrf
@async_job("spectral_match")
def route_spectral_match(job_id, filepath, data):
    from opencut.core import spectral_match
    if not spectral_match.check_spectral_match_available():
        raise RuntimeError(spectral_match.INSTALL_HINT)
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    reference_path = validate_path(str(data.get("reference_path") or ""))
    strength = safe_float(data.get("strength"), default=1.0, min_val=0.0, max_val=2.0)
    output = str(data.get("output") or "") or None
    result = spectral_match.match(filepath, reference_path=reference_path, output=output,
                                  strength=strength, on_progress=_prog)
    return {"output": result.output, "filter_db": result.filter_db, "notes": result.notes}


@wave_k_bp.route("/audio/spectral-match/preview", methods=["POST"])
@require_csrf
def route_spectral_match_preview():
    try:
        from opencut.core import spectral_match
        if not spectral_match.check_spectral_match_available():
            return _stub_503("spectral_match", spectral_match.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        input_path = validate_path(str(data.get("input_path") or ""))
        reference_path = validate_path(str(data.get("reference_path") or ""))
        result = spectral_match.preview(input_path, reference_path)
        return jsonify(result)
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "spectral_match_preview")


@wave_k_bp.route("/audio/spectral-match/info", methods=["GET"])
def route_spectral_match_info():
    try:
        from opencut.core import spectral_match
        return jsonify({"available": spectral_match.check_spectral_match_available(),
                        "install_hint": spectral_match.INSTALL_HINT})
    except Exception as exc:
        return safe_error(exc, "spectral_match_info")


# =============================================================
# K1.9 — Lottie Import
# =============================================================

@wave_k_bp.route("/video/lottie/render", methods=["POST"])
@require_csrf
@async_job("lottie_render")
def route_lottie_render(job_id, filepath, data):
    from opencut.core import lottie_import
    if not lottie_import.check_lottie_available():
        raise RuntimeError(lottie_import.INSTALL_HINT)
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    output = str(data.get("output") or "") or None
    width = safe_int(data.get("width"), default=1920)
    height = safe_int(data.get("height"), default=1080)
    fps = safe_float(data.get("fps"), default=30.0)
    bg_color = data.get("bg_color") or None
    result = lottie_import.render(filepath, output=output, width=width, height=height,
                                  fps=fps, bg_color=bg_color, on_progress=_prog)
    return {"output": result.output, "width": result.width, "height": result.height,
            "fps": result.fps, "duration": result.duration, "frames": result.frames,
            "notes": result.notes}


@wave_k_bp.route("/video/lottie/info", methods=["POST"])
@require_csrf
def route_lottie_info():
    try:
        from opencut.core import lottie_import
        data = request.get_json(force=True, silent=True) or {}
        lottie_path = validate_path(str(data.get("filepath") or data.get("lottie_path") or ""))
        meta = lottie_import.info(lottie_path)
        return jsonify(meta)
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "lottie_info")


# =============================================================
# K1.10 — Semantic Search (using /search/ai to avoid collision)
# =============================================================

@wave_k_bp.route("/search/ai", methods=["POST"])
@require_csrf
@async_job("semantic_search", filepath_required=False)
def route_semantic_search(job_id, filepath, data):
    from opencut.core import semantic_search
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    query = str(data.get("query") or "")
    media_paths = list(data.get("media_paths") or [])
    mode = str(data.get("mode") or "all")
    top_k = safe_int(data.get("top_k"), default=10)
    results = semantic_search.search(query, media_paths=media_paths, mode=mode,
                                     top_k=top_k, on_progress=_prog)
    return {"results": [{"path": r.path, "score": r.score, "type": r.type,
                         "timestamp": r.timestamp} for r in results]}


@wave_k_bp.route("/search/ai/index", methods=["POST"])
@require_csrf
@async_job("semantic_search_index", filepath_required=False)
def route_semantic_search_index(job_id, filepath, data):
    from opencut.core import semantic_search
    def _prog(p, m=""): _update_job(job_id, progress=int(p), message=str(m))
    media_paths = list(data.get("media_paths") or [])
    status = semantic_search.build_index(media_paths, on_progress=_prog)
    return {"indexed": status.indexed, "pending": status.pending,
            "last_updated": status.last_updated}


@wave_k_bp.route("/search/ai/index/status", methods=["GET"])
def route_semantic_search_index_status():
    try:
        from opencut.core import semantic_search
        status = semantic_search.get_index_status()
        return jsonify({"indexed": status.indexed, "pending": status.pending,
                        "last_updated": status.last_updated})
    except Exception as exc:
        return safe_error(exc, "semantic_search_index_status")


# =============================================================
# Tier 2 routes — check + stub 503
# =============================================================

@wave_k_bp.route("/tts/gptsovits", methods=["POST"])
@require_csrf
def route_tts_gptsovits():
    try:
        from opencut.core import tts_gptsovits
        if not tts_gptsovits.check_gptsovits_available():
            return _stub_503("GPT-SoVITS", tts_gptsovits.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("text") or "")
        voice = str(data.get("voice") or "")
        language = str(data.get("language") or "en")
        output = str(data.get("output") or "") or None
        result = tts_gptsovits.synthesize(text, voice=voice, language=language, output=output)
        return jsonify({"output": result.output, "voice": result.voice, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "tts_gptsovits")


@wave_k_bp.route("/tts/gptsovits/voices", methods=["GET"])
def route_tts_gptsovits_voices():
    try:
        from opencut.core import tts_gptsovits
        return jsonify({"voices": tts_gptsovits.list_voices(),
                        "available": tts_gptsovits.check_gptsovits_available()})
    except Exception as exc:
        return safe_error(exc, "tts_gptsovits_voices")


@wave_k_bp.route("/tts/amphion", methods=["POST"])
@require_csrf
def route_tts_amphion():
    try:
        from opencut.core import tts_amphion
        if not tts_amphion.check_amphion_available():
            return _stub_503("Amphion TTS", tts_amphion.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = tts_amphion.synthesize(
            str(data.get("text") or ""),
            reference_audio=str(data.get("reference_audio") or "") or None,
            model=str(data.get("model") or "maskgct"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "model": result.model, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "tts_amphion")


@wave_k_bp.route("/tts/cosyvoice2", methods=["POST"])
@require_csrf
def route_tts_cosyvoice2():
    try:
        from opencut.core import tts_cosyvoice2
        if not tts_cosyvoice2.check_cosyvoice2_available():
            return _stub_503("CosyVoice2", tts_cosyvoice2.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = tts_cosyvoice2.synthesize(str(data.get("text") or ""),
                                            output=str(data.get("output") or "") or None)
        return jsonify({"output": result.output, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "tts_cosyvoice2")


@wave_k_bp.route("/audio/singing/vevo2", methods=["POST"])
@require_csrf
def route_singing_vevo2():
    try:
        from opencut.core import singing_vevo2
        if not singing_vevo2.check_vevo2_available():
            return _stub_503("Vevo2", singing_vevo2.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        audio_path = validate_path(str(data.get("audio_path") or ""))
        reference_path = validate_path(str(data.get("reference_path") or ""))
        result = singing_vevo2.convert(audio_path, reference_path,
                                       pitch_shift=safe_int(data.get("pitch_shift"), 0),
                                       output=str(data.get("output") or "") or None)
        return jsonify({"output": result.output, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "singing_vevo2")


@wave_k_bp.route("/video/lipsync/echomimic", methods=["POST"])
@require_csrf
def route_lipsync_echomimic():
    try:
        from opencut.core import lipsync_echomimic
        if not lipsync_echomimic.check_echomimic_available():
            return _stub_503("EchoMimic", lipsync_echomimic.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = lipsync_echomimic.animate(
            validate_path(str(data.get("image_path") or "")),
            validate_path(str(data.get("audio_path") or "")),
            mode=str(data.get("mode") or "portrait"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "mode": result.mode, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "lipsync_echomimic")


@wave_k_bp.route("/video/style/tokenflow", methods=["POST"])
@require_csrf
def route_style_tokenflow():
    try:
        from opencut.core import style_tokenflow
        if not style_tokenflow.check_tokenflow_available():
            return _stub_503("TokenFlow", style_tokenflow.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = style_tokenflow.restyle(
            validate_path(str(data.get("video_path") or "")),
            style_prompt=str(data.get("style_prompt") or ""),
            strength=safe_float(data.get("strength"), default=0.7),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "style_prompt": result.style_prompt, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "style_tokenflow")


@wave_k_bp.route("/video/track/cutie", methods=["POST"])
@require_csrf
def route_track_cutie():
    try:
        from opencut.core import track_cutie
        if not track_cutie.check_cutie_available():
            return _stub_503("Cutie", track_cutie.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = track_cutie.track(
            validate_path(str(data.get("video_path") or "")),
            validate_path(str(data.get("mask_frame0_path") or "")),
            object_id=safe_int(data.get("object_id"), default=1),
            output_dir=str(data.get("output_dir") or "") or None,
        )
        return jsonify({"output_mask_path": result.output_mask_path,
                        "tracked_frames": result.tracked_frames, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "track_cutie")


@wave_k_bp.route("/video/track/deva", methods=["POST"])
@require_csrf
def route_track_deva():
    try:
        from opencut.core import track_deva
        if not track_deva.check_deva_available():
            return _stub_503("DEVA", track_deva.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = track_deva.track(
            validate_path(str(data.get("video_path") or "")),
            text_prompt=str(data.get("text_prompt") or ""),
            output_dir=str(data.get("output_dir") or "") or None,
        )
        return jsonify({"output_mask_path": result.output_mask_path,
                        "tracked_objects": result.tracked_objects, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "track_deva")


@wave_k_bp.route("/video/flow/searaft", methods=["POST"])
@require_csrf
def route_flow_searaft():
    try:
        from opencut.core import flow_searaft
        if not flow_searaft.check_searaft_available():
            return _stub_503("SEA-RAFT", flow_searaft.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = flow_searaft.compute_flow(
            validate_path(str(data.get("video_path") or "")),
            output=str(data.get("output") or "") or None,
            max_resolution=safe_int(data.get("max_resolution"), default=1080),
        )
        return jsonify({"output_flow_path": result.output_flow_path, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "flow_searaft")


@wave_k_bp.route("/video/restore/diffbir", methods=["POST"])
@require_csrf
def route_restore_diffbir():
    try:
        from opencut.core import restore_diffbir
        if not restore_diffbir.check_diffbir_available():
            return _stub_503("DiffBIR", restore_diffbir.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = restore_diffbir.restore(
            validate_path(str(data.get("video_path") or "")),
            tile_size=safe_int(data.get("tile_size"), default=512),
            fast_mode=safe_bool(data.get("fast_mode"), default=False),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "restore_diffbir")


@wave_k_bp.route("/video/stabilize/gyroflow", methods=["POST"])
@require_csrf
def route_stabilize_gyroflow():
    try:
        from opencut.core import stabilize_gyroflow
        if not stabilize_gyroflow.check_gyroflow_available():
            return _stub_503("Gyroflow", stabilize_gyroflow.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = stabilize_gyroflow.stabilize(
            validate_path(str(data.get("video_path") or "")),
            gyro_data_path=str(data.get("gyro_data_path") or "") or None,
            lens_profile=str(data.get("lens_profile") or "") or None,
            horizon_lock=safe_bool(data.get("horizon_lock"), default=False),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "stabilize_gyroflow")


@wave_k_bp.route("/video/deblur", methods=["POST"])
@require_csrf
def route_deblur_motion():
    try:
        from opencut.core import deblur_motion
        if not deblur_motion.check_deblur_motion_available():
            return _stub_503("Motion Deblur", deblur_motion.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = deblur_motion.deblur(
            validate_path(str(data.get("video_path") or "")),
            backend=str(data.get("backend") or "auto"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "backend": result.backend, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "deblur_motion")


@wave_k_bp.route("/video/deblur/backends", methods=["GET"])
def route_deblur_backends():
    try:
        from opencut.core.deblur_motion import list_backends
        return jsonify(list_backends())
    except Exception as exc:
        return safe_error(exc, "deblur_backends")


@wave_k_bp.route("/video/depth/depthpro", methods=["POST"])
@require_csrf
def route_depth_depthpro():
    try:
        from opencut.core import depth_depthpro
        if not depth_depthpro.check_depthpro_available():
            return _stub_503("Depth Pro", depth_depthpro.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = depth_depthpro.estimate(
            validate_path(str(data.get("video_path") or "")),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output_depth_path": result.output_depth_path, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "depth_depthpro")


@wave_k_bp.route("/video/depth/depthpro/backends", methods=["GET"])
def route_depth_backends():
    try:
        from opencut.core.depth_depthpro import list_backends
        return jsonify(list_backends())
    except Exception as exc:
        return safe_error(exc, "depth_backends")


@wave_k_bp.route("/video/depth-flow", methods=["POST"])
@require_csrf
def route_depth_flow():
    try:
        from opencut.core import depth_flow
        if not depth_flow.check_depthflow_available():
            return _stub_503("DepthFlow", depth_flow.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = depth_flow.generate(
            validate_path(str(data.get("image_path") or "")),
            depth_path=str(data.get("depth_path") or "") or None,
            duration=safe_float(data.get("duration"), default=5.0),
            parallax_intensity=safe_float(data.get("parallax_intensity"), default=0.5),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "duration": result.duration, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "depth_flow")


@wave_k_bp.route("/audio/sfx/generate", methods=["POST"])
@require_csrf
def route_sfx_audiogen():
    try:
        from opencut.core import sfx_audiogen
        if not sfx_audiogen.check_audiogen_available():
            return _stub_503("AudioGen", sfx_audiogen.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = sfx_audiogen.generate(
            str(data.get("prompt") or ""),
            duration=safe_float(data.get("duration"), default=3.0),
            model_size=str(data.get("model_size") or "medium"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "prompt": result.prompt, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "sfx_audiogen")


@wave_k_bp.route("/generate/opensora", methods=["POST"])
@require_csrf
def route_gen_video_opensora():
    try:
        from opencut.core import gen_video_opensora
        if not gen_video_opensora.check_opensora_available():
            return _stub_503("OpenSora", gen_video_opensora.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = gen_video_opensora.generate(
            str(data.get("prompt") or ""),
            duration=safe_float(data.get("duration"), default=4.0),
            resolution=str(data.get("resolution") or "720p"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "prompt": result.prompt, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "gen_video_opensora")


@wave_k_bp.route("/generate/ltx/v2", methods=["POST"])
@require_csrf
def route_gen_video_ltx():
    try:
        from opencut.core import gen_video_ltx
        if not gen_video_ltx.check_ltx_available():
            return _stub_503("LTX-Video", gen_video_ltx.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = gen_video_ltx.generate(
            str(data.get("prompt") or ""),
            duration=safe_float(data.get("duration"), default=5.0),
            resolution=str(data.get("resolution") or "720p"),
            audio_prompt=str(data.get("audio_prompt") or "") or None,
            version=str(data.get("version") or "ltx-0.9.8"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "version": result.version, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "gen_video_ltx")


@wave_k_bp.route("/generate/ltx/backends", methods=["GET"])
def route_gen_ltx_backends():
    try:
        from opencut.core.gen_video_ltx import list_backends
        return jsonify(list_backends())
    except Exception as exc:
        return safe_error(exc, "gen_ltx_backends")


@wave_k_bp.route("/video/audio-reactive", methods=["POST"])
@require_csrf
def route_audio_reactive_fx():
    try:
        from opencut.core import audio_reactive_fx
        if not audio_reactive_fx.check_audio_reactive_available():
            return _stub_503("Audio Reactive FX", audio_reactive_fx.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = audio_reactive_fx.render(
            validate_path(str(data.get("video_path") or "")),
            audio_path=str(data.get("audio_path") or ""),
            preset=str(data.get("preset") or "boom"),
            custom_params=data.get("custom_params") or None,
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "preset": result.preset,
                        "beat_count": result.beat_count, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "audio_reactive_fx")


@wave_k_bp.route("/video/audio-reactive/presets", methods=["GET"])
def route_audio_reactive_presets():
    try:
        from opencut.core.audio_reactive_fx import list_presets
        return jsonify(list_presets())
    except Exception as exc:
        return safe_error(exc, "audio_reactive_presets")


@wave_k_bp.route("/video/cinefocus", methods=["POST"])
@require_csrf
def route_cinefocus():
    try:
        from opencut.core import cinefocus
        if not cinefocus.check_cinefocus_available():
            return _stub_503("CineFocus", cinefocus.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = cinefocus.render(
            validate_path(str(data.get("video_path") or "")),
            focal_z_start=safe_float(data.get("focal_z_start"), default=0.5),
            focal_z_end=safe_float(data.get("focal_z_end"), default=0.5),
            focal_frame_start=safe_int(data.get("focal_frame_start"), default=0),
            focal_frame_end=safe_int(data.get("focal_frame_end"), default=0),
            aperture_f=safe_float(data.get("aperture_f"), default=2.8),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "notes": result.notes})
    except Exception as exc:
        return safe_error(exc, "cinefocus")


# =============================================================
# Tier 3 routes — 501
# =============================================================

@wave_k_bp.route("/video/dub", methods=["POST"])
@require_csrf
def route_dub_pipeline():
    return _stub_501("Dubbing Pipeline (K3.1)")


@wave_k_bp.route("/video/trailer/generate", methods=["POST"])
@require_csrf
def route_trailer_gen():
    return _stub_501("Trailer Generator (K3.2)")


@wave_k_bp.route("/screenplay/parse", methods=["POST"])
@require_csrf
def route_screenplay_parse():
    try:
        from opencut.core import screenplay_parser
        data = request.get_json(force=True, silent=True) or {}
        screenplay_path = validate_path(str(data.get("screenplay_path") or ""))
        ext = screenplay_path.lower().rsplit(".", 1)[-1]
        if ext == "fdx":
            scenes = screenplay_parser.parse_fdx(screenplay_path)
        else:
            scenes = screenplay_parser.parse_fountain(screenplay_path)
        return jsonify([{"heading": s.heading, "action": s.action,
                         "characters": s.characters, "dialogue": s.dialogue} for s in scenes])
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "screenplay_parse")


@wave_k_bp.route("/video/face-age", methods=["POST"])
@require_csrf
def route_face_age_transform():
    return _stub_501("Face Age Transform (K3.4)")


@wave_k_bp.route("/video/slate/identify", methods=["POST"])
@require_csrf
def route_slate_id():
    try:
        from opencut.core import slate_id
        data = request.get_json(force=True, silent=True) or {}
        video_path = validate_path(str(data.get("video_path") or data.get("filepath") or ""))
        max_head = safe_int(data.get("max_head_frames"), default=60)
        info = slate_id.identify(video_path, max_head_frames=max_head)
        return jsonify({"scene": info.scene, "take": info.take, "camera": info.camera,
                        "roll": info.roll, "fps": info.fps, "date": info.date, "notes": info.notes})
    except (ValueError, KeyError) as exc:
        return error_response("INVALID_INPUT", str(exc), status=400)
    except Exception as exc:
        return safe_error(exc, "slate_id")


@wave_k_bp.route("/video/outpaint", methods=["POST"])
@require_csrf
def route_outpaint_video():
    return _stub_501("Video Outpainting (K3.6)")


@wave_k_bp.route("/generate/wan-vace", methods=["POST"])
@require_csrf
def route_gen_video_wan_vace():
    try:
        from opencut.core import gen_video_wan_vace
        data = request.get_json(force=True, silent=True) or {}
        result = gen_video_wan_vace.edit(
            validate_path(str(data.get("video_path") or "")),
            prompt=str(data.get("prompt") or ""),
            edit_type=str(data.get("edit_type") or "background"),
            output=str(data.get("output") or "") or None,
        )
        return jsonify({"output": result.output, "edit_type": result.edit_type, "notes": result.notes})
    except NotImplementedError as exc:
        return _stub_501("Wan2.1 VACE (K3.7)")
    except Exception as exc:
        return safe_error(exc, "gen_video_wan_vace")


@wave_k_bp.route("/video/highlights/sports", methods=["POST"])
@require_csrf
def route_highlights_sports():
    try:
        from opencut.core import highlights_sports
        if not highlights_sports.check_sports_highlights_available():
            return _stub_503("Sports Highlights", highlights_sports.INSTALL_HINT)
        data = request.get_json(force=True, silent=True) or {}
        result = highlights_sports.extract(
            validate_path(str(data.get("video_path") or "")),
            genre=str(data.get("genre") or "sports"),
            top_n=safe_int(data.get("top_n"), default=5),
        )
        return jsonify([{"start": s.start, "end": s.end, "score": s.score} for s in result])
    except NotImplementedError as exc:
        return _stub_501("Sports Highlights (K3.8)")
    except Exception as exc:
        return safe_error(exc, "highlights_sports")


@wave_k_bp.route("/video/highlights/genres", methods=["GET"])
def route_highlight_genres():
    try:
        from opencut.core.highlights_sports import GENRES
        return jsonify(GENRES)
    except Exception as exc:
        return safe_error(exc, "highlight_genres")
