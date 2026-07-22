"""Local song and music generation routes formerly grouped in Wave L."""

from __future__ import annotations

from flask import jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float

from .wave_l_contract import wave_l_bp

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
