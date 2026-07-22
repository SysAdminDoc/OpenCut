"""Caption generation, styling, display settings, and cache routes."""

from .captions import (
    VALID_WHISPER_MODELS,
    CaptionConfig,
    _asr_provenance_payload,
    _caption_review_summary,
    _export_srt_with_policy,
    _is_cancelled,
    _legacy_srt_bom_requested,
    _resolve_output_dir,
    _safe_probe,
    _update_job,
    _write_caption_roundtrip_sidecar,
    async_job,
    build_destructive_plan,
    captions_bp,
    destructive_confirmation_required_response,
    export_ass,
    export_json,
    export_vtt,
    get_json_dict,
    jsonify,
    logger,
    os,
    request,
    require_csrf,
    safe_bool,
    safe_error,
    safe_float,
    validate_path,
    verify_destructive_confirm_token,
    workflow_step,
)


# Captions
# ---------------------------------------------------------------------------
@captions_bp.route("/captions", methods=["POST"])
@require_csrf
@workflow_step("Generating captions")
@async_job("captions", disk_operation="transcribe", resumable=True)
def generate_captions(job_id, filepath, data):
    """Generate captions/subtitles."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")
    language = data.get("language", None)
    engine = data.get("engine", None)
    model_revision = data.get("model_revision", None)
    sub_format = data.get("format", "srt")
    if sub_format not in ("srt", "vtt", "json", "ass"):
        sub_format = "srt"
    word_timestamps = safe_bool(data.get("word_timestamps", True), True)
    legacy_srt_bom = _legacy_srt_bom_requested(data)

    from opencut.core.captions import check_whisper_available, transcribe

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

    _update_job(job_id, progress=10, message=f"Loading {model} model ({backend})...")

    config = CaptionConfig(
        engine=engine,
        model=model,
        model_revision=model_revision,
        language=language,
        word_timestamps=word_timestamps,
    )

    force_retranscribe = safe_bool(data.get("force_retranscribe", False), False)
    _update_job(job_id, progress=20, message="Transcribing audio (this takes a while for long files)...")
    result = transcribe(filepath, config=config, use_cache=not force_retranscribe)
    if getattr(result, "cache_hit", False):
        _update_job(job_id, progress=25, message="Using cached transcript...")

    if _is_cancelled(job_id):
        return {"cancelled": True}

    _update_job(job_id, progress=80, message="Writing subtitle file...")

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(effective_dir, f"{base_name}.{sub_format}")

    if sub_format == "vtt":
        export_vtt(result, out_path)
    elif sub_format == "json":
        export_json(result, out_path)
    elif sub_format == "ass":
        info = _safe_probe(filepath)
        vid_w = info.video.width if info and info.video else 1920
        vid_h = info.video.height if info and info.video else 1080
        export_ass(
            result, out_path,
            video_width=vid_w, video_height=vid_h,
            karaoke=word_timestamps,
        )
    else:
        _export_srt_with_policy(result, out_path, legacy_windows_bom=legacy_srt_bom)

    sidecar_path, sidecar_warnings = _write_caption_roundtrip_sidecar(result, out_path, sub_format, filepath, data)
    response = {
        "output_path": out_path,
        "sidecar_path": sidecar_path,
        "metadata_preserved": bool(sidecar_path),
        "language": getattr(result, "language", language or "en"),
        "segments": len(result.segments),
        "caption_segments": len(result.segments),
        "words": getattr(result, "word_count", 0),
        "transcript_cache_hit": bool(getattr(result, "cache_hit", False)),
        "transcript_cache_key": getattr(result, "cache_key", None),
        "asr_provenance": _asr_provenance_payload(result),
    }
    if sidecar_warnings:
        response["warnings"] = sidecar_warnings
    response.update(_caption_review_summary(result))
    if sub_format == "srt":
        response["srt_encoding"] = "utf-8-sig" if legacy_srt_bom else "utf-8"
    return response


# ---------------------------------------------------------------------------
# Styled Captions (video overlay)
# ---------------------------------------------------------------------------
@captions_bp.route("/caption-styles", methods=["GET"])
def get_caption_styles():
    """Return available caption style presets for the panel preview."""
    try:
        from opencut.core.styled_captions import get_style_info
        return jsonify({"styles": get_style_info()})
    except Exception as e:
        return safe_error(e, "caption styles")


@captions_bp.route("/captions/display-settings/tokens", methods=["GET"])
def caption_display_setting_tokens():
    """Return user-overridable closed-caption display setting tokens."""
    try:
        from opencut.core.caption_display_settings import token_schema
        return jsonify(token_schema())
    except Exception as e:
        return safe_error(e, "caption_display_setting_tokens")


@captions_bp.route("/captions/display-settings/preview", methods=["POST"])
@require_csrf
def caption_display_setting_preview():
    """Return normalized display settings and preview styles for a sample caption."""
    try:
        from opencut.core.caption_display_settings import build_preview_payload

        data = get_json_dict() or {}
        settings = data.get("settings") or {}
        sample_text = str(data.get("sample_text") or "Caption preview")
        return jsonify(build_preview_payload(settings=settings, sample_text=sample_text))
    except Exception as e:
        return safe_error(e, "caption_display_setting_preview")


@captions_bp.route("/captions/cache/stats", methods=["GET"])
def caption_cache_stats():
    """Return persistent transcript-cache inventory and hit counters."""
    try:
        from opencut.core.transcript_cache import cache_stats

        return jsonify(cache_stats())
    except Exception as e:
        return safe_error(e, "caption_cache_stats")


@captions_bp.route("/captions/cache/provenance/<cache_key>", methods=["GET"])
def caption_cache_provenance(cache_key):
    """Return cache/ASR identity without returning transcript contents."""
    try:
        from opencut.core.asr_provenance import provenance_to_dict
        from opencut.core.transcript_cache import load_transcript

        cached = load_transcript(cache_key)
        if not cached:
            return jsonify({"error": "Transcript cache entry not found"}), 404
        metadata = cached.get("metadata") or {}
        result = cached.get("result") or {}
        return jsonify({
            "cache_key": cache_key,
            "cache_schema_version": cached.get("schema_version"),
            "source_sha256": (metadata.get("source") or {}).get("source_sha256"),
            "asr_provenance": provenance_to_dict(result.get("provenance")),
        })
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid transcript cache key"}), 400
    except Exception as e:
        return safe_error(e, "caption_cache_provenance")


@captions_bp.route("/captions/cache/clear", methods=["DELETE"])
@require_csrf
def caption_cache_clear():
    """Clear persistent transcript-cache entries."""
    try:
        data = get_json_dict() if request.data else {}
        dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
        from opencut.core.transcript_cache import cache_stats, clear_cache

        stats = cache_stats()
        entries = []
        if int(stats.get("entries", 0)) > 0:
            entries.append({
                "path": str(stats.get("cache_dir", "")),
                "category": "transcript-cache",
                "root": str(stats.get("cache_dir", "")),
                "type": "directory",
                "bytes": int(stats.get("bytes", 0)),
                "reversible": False,
            })
        plan = build_destructive_plan(
            "captions.cache.clear",
            targets=entries,
            metadata={
                "route": "/captions/cache/clear",
                "entries": stats.get("entries", 0),
                "total_bytes": int(stats.get("bytes", 0)),
            },
            reversible=False,
        )
        if dry_run:
            return jsonify({
                "success": True,
                "dry_run": True,
                "plan": entries,
                "destructive_plan": plan,
                "confirm_token": plan["confirm_token"],
                "removed_entries": 0,
                "removed_bytes": 0,
                "cache_dir": stats.get("cache_dir", ""),
            })
        if entries and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409
        result = clear_cache()
        result["dry_run"] = False
        result["destructive_plan"] = plan
        return jsonify(result)
    except Exception as e:
        return safe_error(e, "caption_cache_clear")


@captions_bp.route("/styled-captions", methods=["POST"])
@require_csrf
@workflow_step("Generating styled captions")
@async_job("styled-captions")
def styled_captions_route(job_id, filepath, data):
    """Generate a transparent video overlay with styled, animated captions."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    style_name = data.get("style", "youtube_bold")
    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")
    language = data.get("language", None)
    engine = data.get("engine", None)
    model_revision = data.get("model_revision", None)
    custom_action_words = data.get("action_words", [])
    # Bound the list + per-item size so malicious/misconfigured clients can't
    # drive the styled-captions renderer into unbounded regex work or OOM.
    if not isinstance(custom_action_words, list):
        custom_action_words = []
    else:
        custom_action_words = [
            str(w)[:64] for w in custom_action_words[:500]
            if isinstance(w, (str, int, float))
        ]
    auto_detect_energy = safe_bool(data.get("auto_detect_energy", True), True)
    # Optional: pre-existing speech segments for remapping
    remap_segments_raw = data.get("remap_segments", None)

    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.core.styled_captions import (
        detect_action_words_by_energy,
        get_action_word_indices,
        render_styled_caption_video,
    )

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("Whisper is required for styled captions. Install from the Captions tab.")

    # Get video resolution
    info = _safe_probe(filepath)
    vid_w = info.video.width if info and info.video else 1920
    vid_h = info.video.height if info and info.video else 1080
    vid_fps = info.video.fps if info and info.video else 30.0

    # Step 1: Transcribe
    _update_job(job_id, progress=10, message=f"Transcribing with {backend} ({model})...")
    config = CaptionConfig(
        engine=engine,
        model=model,
        model_revision=model_revision,
        language=language,
        word_timestamps=True,
    )
    transcription = transcribe(filepath, config=config)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Optional: remap captions to edited timeline
    if remap_segments_raw and isinstance(remap_segments_raw, list):
        from opencut.core.captions import remap_captions_to_segments
        from opencut.core.silence import TimeSegment
        remap_segs = []
        for s in remap_segments_raw:
            if isinstance(s, dict) and "start" in s and "end" in s:
                remap_segs.append(TimeSegment(
                    safe_float(s["start"], 0.0, min_val=0.0),
                    safe_float(s["end"], 0.0, min_val=0.0),
                ))
        if remap_segs:
            transcription = remap_captions_to_segments(transcription, remap_segs)

    # Step 2: Detect action words
    _update_job(job_id, progress=50, message="Detecting action words...")
    all_words = []
    for seg in transcription.segments:
        all_words.extend(seg.words)

    energy_indices = set()
    if auto_detect_energy and all_words:
        try:
            energy_indices = detect_action_words_by_energy(filepath, all_words)
        except Exception as e:
            logger.warning(f"Energy detection failed: {e}")

    action_indices = get_action_word_indices(
        all_words,
        custom_words=custom_action_words if custom_action_words else None,
        use_keywords=True,
        energy_indices=energy_indices,
    )

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step 3: Render overlay video
    _update_job(job_id, progress=60, message="Rendering styled captions...")
    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    overlay_path = os.path.join(effective_dir, f"{base_name}_captions.mov")

    def _on_render_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    # Resolve total duration defensively — info may be None if probing failed,
    # and transcription.duration may be 0/None. Prefer the last segment end as
    # a guaranteed-available fallback so the renderer always gets a sane value.
    _last_end = 0.0
    try:
        _last_end = max(
            (float(getattr(s, "end", 0) or 0) for s in transcription.segments),
            default=0.0,
        )
    except (TypeError, ValueError):
        _last_end = 0.0
    _info_dur = 0.0
    try:
        _info_dur = float(info.duration) if info and getattr(info, "duration", None) else 0.0
    except (TypeError, ValueError):
        _info_dur = 0.0
    total_duration = (
        getattr(transcription, "duration", 0) or _info_dur or _last_end or 0.0
    )

    result = render_styled_caption_video(
        transcription,
        overlay_path,
        style_name=style_name,
        video_width=vid_w,
        video_height=vid_h,
        fps=vid_fps,
        action_indices=action_indices,
        total_duration=total_duration,
        on_progress=_on_render_progress,
    )

    return {
        "output_path": overlay_path,
        "overlay_path": overlay_path,
        "style": style_name,
        "frames_rendered": result["frames_rendered"],
        "action_words_found": len(action_indices),
        "caption_segments": len(transcription.segments),
        "words": sum(len(getattr(s, "words", []) or []) for s in transcription.segments),
        "language": getattr(transcription, "language", language or "en"),
        "asr_provenance": _asr_provenance_payload(transcription),
    }


# ---------------------------------------------------------------------------
