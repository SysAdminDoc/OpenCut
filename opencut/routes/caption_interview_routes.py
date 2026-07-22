"""Resumable interview-polish workflow routes."""

from .captions import (
    CaptionConfig,
    ExportConfig,
    _asr_provenance_payload,
    _caption_review_summary,
    _export_srt_with_policy,
    _is_cancelled,
    _legacy_srt_bom_requested,
    _make_sequence_name,
    _resolve_output_dir,
    _safe_probe,
    _segment_payload,
    _update_job,
    async_job,
    captions_bp,
    copy,
    detect_speech,
    export_premiere_xml,
    get_preset,
    jsonify,
    logger,
    os,
    request,
    require_csrf,
    safe_bool,
    validate_filepath,
    validate_path,
)


# Interview Polish — one-click 6-step pipeline for podcasts/interviews (v1.9.29)
# ---------------------------------------------------------------------------
@captions_bp.route("/interview-polish", methods=["POST"])
@require_csrf
@async_job("interview_polish")
def interview_polish(job_id, filepath, data):
    """One-click pipeline for podcast / interview content.

    Runs in order:

    1. Silence detection (speech-only segments)
    2. Repeated-take detection (keeps the best of each restart)
    3. Transcription + filler-word removal
    4. Speaker diarization (when pyannote is available)
    5. Chapter generation (LLM or pause-heuristic fallback)
    6. Export Premiere XML + SRT + chapter markdown

    The result dict contains every per-step artifact so the panel can
    show a step-by-step progress checklist and expose individual
    outputs (SRT, chapter markdown, XML).
    """
    from opencut.core.captions import (
        check_whisper_available,
        remap_captions_to_segments,
        transcribe,
    )
    from opencut.core.fillers import (
        build_boundary_review,
        detect_fillers,
        remove_fillers_from_segments,
    )
    from opencut.core.repeat_detect import detect_repeated_takes, merge_repeat_ranges
    from opencut.polish_state import (
        _transcription_from_dict,
        _transcription_to_dict,
        load_state,
        save_state,
    )

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    preset = data.get("preset", "youtube")
    seq_name = data.get("sequence_name", "")
    # v1.10.5 (Q): resume-from-cache is opt-in via request flag; when true
    # we skip transcription if a valid cache exists for this file.
    resume = safe_bool(data.get("resume", True), True)
    generate_chapters_flag = safe_bool(data.get("generate_chapters", True), True)
    diarize_flag = safe_bool(data.get("diarize", True), True)
    remove_fillers_flag = safe_bool(data.get("remove_fillers", True), True)
    accept_low_confidence_boundaries = safe_bool(
        data.get("accept_low_confidence_boundaries", False),
        False,
    )
    detect_repeats_flag = safe_bool(data.get("detect_repeats", True), True)
    legacy_srt_bom = _legacy_srt_bom_requested(data)

    cfg = get_preset(preset)
    effective_name = seq_name or _make_sequence_name(filepath, "Interview Polish")
    ecfg = ExportConfig(sequence_name=effective_name)
    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    xml_path = os.path.join(effective_dir, f"{base_name}_interview.xml")
    srt_path = os.path.join(effective_dir, f"{base_name}_interview.srt")
    chapters_path = os.path.join(effective_dir, f"{base_name}_chapters.md")

    _finfo = _safe_probe(filepath)
    _fdur = _finfo.duration if _finfo else 0.0

    # Per-step result carrier — the frontend renders this as a
    # checklist so users can see what ran, what was skipped, and why.
    steps_report = []

    def record_step(key, label, ok, **extra):
        entry = {"key": key, "label": label, "ok": bool(ok)}
        entry.update(extra)
        steps_report.append(entry)
        return entry

    total_steps = 6
    step_idx = [0]

    def step_progress(label):
        step_idx[0] += 1
        pct = int(5 + (step_idx[0] / total_steps) * 85)
        _update_job(job_id, progress=pct,
                    message=f"Step {step_idx[0]}/{total_steps}: {label}")

    # 1. Silence detection -----------------------------------------
    step_progress("Detecting speech segments")
    segments = detect_speech(filepath, config=cfg.silence, file_duration=_fdur)
    record_step("silence", "Detect speech", True,
                kept_segments=len(segments),
                speech_duration=round(sum(s.end - s.start for s in segments), 2))

    if _is_cancelled(job_id):
        return {"cancelled": True, "steps": steps_report}

    # 2. Transcription (used by repeat-detect, fillers, captions) --
    step_progress("Transcribing")
    whisper_ok, whisper_backend = check_whisper_available()
    transcription = None
    cached_state = load_state(filepath) if resume else None
    if cached_state and cached_state.get("transcription"):
        transcription = _transcription_from_dict(cached_state["transcription"])
        record_step(
            "transcribe", "Transcribe audio", True,
            backend="cache",
            word_count=getattr(transcription, "word_count", 0),
            language=getattr(transcription, "language", ""),
            cached=True,
        )
    elif whisper_ok:
        try:
            transcription_config = CaptionConfig(
                engine=data.get("engine", None),
                model=getattr(cfg.captions, "model", "base"),
                model_revision=data.get("model_revision", None),
                language=data.get(
                    "language",
                    getattr(cfg.captions, "language", None),
                ),
                word_timestamps=True,
            )
            transcription = transcribe(
                filepath,
                config=transcription_config,
                timeout=1800,
            )
            # Persist immediately so a crash in steps 3-6 still lets us
            # resume from here on the next attempt.
            try:
                save_state(filepath, _transcription_to_dict(transcription))
            except Exception as e:
                logger.warning("Polish state save failed: %s", e)
            record_step("transcribe", "Transcribe audio", True,
                        backend=whisper_backend,
                        word_count=getattr(transcription, "word_count", 0),
                        language=getattr(transcription, "language", ""))
        except Exception as e:
            logger.warning("Interview polish transcription failed: %s", e)
            record_step("transcribe", "Transcribe audio", False, reason=str(e))
    else:
        record_step("transcribe", "Transcribe audio", False,
                    reason="Whisper not installed — run Install Whisper in Settings.")

    if _is_cancelled(job_id):
        return {"cancelled": True, "steps": steps_report}

    # 3. Repeated-take detection -----------------------------------
    if detect_repeats_flag and transcription and transcription.segments:
        step_progress("Finding repeated takes")
        try:
            seg_dicts = [
                _segment_payload(s, include_words=False)
                for s in transcription.segments
            ]
            repeats = detect_repeated_takes(seg_dicts, threshold=0.6, gap_tolerance=2.0)
            repeat_ranges = merge_repeat_ranges(repeats.get("repeats", []))
            if repeat_ranges:
                # Subtract the repeat ranges from our speech segments.
                # Inlined because this is the only caller; no need for a
                # shared helper until another route needs it.
                def _subtract(segs, holes):
                    out = []
                    for s in segs:
                        start, end = s.start, s.end
                        pieces = [(start, end)]
                        for h in holes:
                            hs, he = h.get("start", 0.0), h.get("end", 0.0)
                            new_pieces = []
                            for (ps, pe) in pieces:
                                if he <= ps or hs >= pe:
                                    new_pieces.append((ps, pe))
                                else:
                                    if hs > ps:
                                        new_pieces.append((ps, hs))
                                    if he < pe:
                                        new_pieces.append((he, pe))
                            pieces = new_pieces
                        for (ps, pe) in pieces:
                            if pe - ps > 0.05:
                                # Rebuild a segment-like object preserving original type
                                new_seg = copy.copy(s)
                                new_seg.start, new_seg.end = ps, pe
                                out.append(new_seg)
                    return out
                segments = _subtract(segments, repeat_ranges)
            record_step("repeats", "Find repeated takes", True,
                        removed_ranges=len(repeat_ranges))
        except Exception as e:
            logger.warning("Interview polish repeat-detect failed: %s", e)
            record_step("repeats", "Find repeated takes", False, reason=str(e))
    else:
        record_step("repeats", "Find repeated takes", False,
                    reason="Skipped" if not detect_repeats_flag else "Needs transcript")

    if _is_cancelled(job_id):
        return {"cancelled": True, "steps": steps_report}

    # 4. Filler-word removal ---------------------------------------
    if remove_fillers_flag and transcription:
        step_progress("Removing filler words")
        try:
            analysis = detect_fillers(transcription, include_context_fillers=True)
            if analysis.hits:
                boundary_review = build_boundary_review(
                    analysis.hits,
                    filepath=filepath,
                )
                if (
                    boundary_review["required"]
                    and not accept_low_confidence_boundaries
                ):
                    record_step(
                        "fillers",
                        "Review filler boundaries",
                        False,
                        reason="Boundary audition required before timeline mutation",
                    )
                    return {
                        "preview_only": True,
                        "mutation_blocked": True,
                        "boundary_review": boundary_review,
                        "steps": steps_report,
                        "asr_provenance": _asr_provenance_payload(transcription),
                    }
                segments = remove_fillers_from_segments(segments, analysis.hits)
            record_step("fillers", "Remove filler words", True,
                        removed_fillers=len(analysis.hits),
                        removed_seconds=round(analysis.total_filler_time, 2))
        except Exception as e:
            logger.warning("Interview polish filler removal failed: %s", e)
            record_step("fillers", "Remove filler words", False, reason=str(e))
    else:
        record_step("fillers", "Remove filler words", False,
                    reason="Skipped" if not remove_fillers_flag else "Needs transcript")

    if _is_cancelled(job_id):
        return {"cancelled": True, "steps": steps_report}

    # 5. Speaker diarization (optional, WhisperX-backed) ----------
    # WhisperX diarization runs inside whisperx_transcribe with diarize=True,
    # but that requires a second full-audio pass + HF token. We flag it in
    # the report so users know why no speaker labels landed in the output
    # and can re-run explicitly via the Captions tab when they have the
    # token configured.
    diarization_available = False
    try:
        from opencut.core.captions_enhanced import check_whisperx_available
        diarization_available = check_whisperx_available()
    except Exception:
        pass

    if diarize_flag and transcription and diarization_available:
        step_progress("Identifying speakers")
        record_step("diarize", "Identify speakers", False,
                    reason="Run Captions → WhisperX with an HF token to label speakers in this transcript.")
    elif diarize_flag:
        record_step("diarize", "Identify speakers", False,
                    reason="WhisperX/pyannote not installed"
                           if not diarization_available else "Needs transcript")
    else:
        record_step("diarize", "Identify speakers", False, reason="Skipped")

    if _is_cancelled(job_id):
        return {"cancelled": True, "steps": steps_report}

    # 6. Chapters + caption export ---------------------------------
    step_progress("Writing captions and chapters")
    captions_result = None
    if transcription:
        try:
            captions_result = remap_captions_to_segments(transcription, segments)
            _export_srt_with_policy(captions_result, srt_path, legacy_windows_bom=legacy_srt_bom)
        except Exception as e:
            logger.warning("Caption export failed: %s", e)

    chapters_data = None
    if generate_chapters_flag and transcription:
        try:
            from opencut.core.chapter_gen import generate_chapters
            from opencut.core.llm import LLMConfig
            seg_dicts = [
                _segment_payload(s, include_words=False)
                for s in transcription.segments
            ]
            llm = None
            try:
                llm = LLMConfig(provider=data.get("llm_provider", "ollama"))
            except Exception as exc:
                logger.warning("LLM config failed, chapters will use fallback: %s", exc)
            chapters_data = generate_chapters(seg_dicts, llm_config=llm, max_chapters=8)
            desc = (chapters_data or {}).get("description_block", "")
            if desc:
                with open(chapters_path, "w", encoding="utf-8") as f:
                    f.write(desc)
            record_step("chapters", "Generate chapters", True,
                        count=len((chapters_data or {}).get("chapters", [])),
                        description_path=chapters_path if desc else "")
        except Exception as e:
            logger.warning("Chapter generation failed: %s", e)
            record_step("chapters", "Generate chapters", False, reason=str(e))
    else:
        record_step("chapters", "Generate chapters", False,
                    reason="Skipped" if not generate_chapters_flag else "Needs transcript")

    # Export Premiere XML with the final condensed segments
    _update_job(job_id, progress=92, message="Exporting Premiere XML…")
    export_premiere_xml(filepath, segments, xml_path, config=ecfg)

    result_data = {
        "xml_path": xml_path,
        "srt_path": srt_path if captions_result else "",
        "srt_encoding": "utf-8-sig" if captions_result and legacy_srt_bom else "utf-8" if captions_result else "",
        "chapters_path": chapters_path if chapters_data and chapters_data.get("description_block") else "",
        "segments": len(segments),
        "speech_duration": round(sum(s.end - s.start for s in segments), 2),
        "original_duration": round(_fdur, 2),
        "compression_ratio": round(
            sum(s.end - s.start for s in segments) / max(_fdur, 0.001), 4
        ),
        "steps": steps_report,
        "sequence_name": effective_name,
    }
    if chapters_data:
        result_data["chapters"] = chapters_data.get("chapters", [])
    if captions_result:
        _caption_segments = (
            captions_result.get("segments", [])
            if isinstance(captions_result, dict)
            else getattr(captions_result, "segments", [])
        )
        result_data["caption_segments"] = len(_caption_segments)
        result_data["words"] = (
            captions_result.get("word_count", 0)
            if isinstance(captions_result, dict)
            else getattr(captions_result, "word_count", 0)
        )
        result_data["language"] = (
            captions_result.get("language", "")
            if isinstance(captions_result, dict)
            else getattr(captions_result, "language", "")
        )
        result_data.update(_caption_review_summary(captions_result))
    return result_data


@captions_bp.route("/interview-polish/state", methods=["DELETE"])
@require_csrf
def interview_polish_clear_state():
    """Drop the cached transcript for a file so the next /interview-polish
    call does a fresh transcription. Used by the panel's "Re-transcribe"
    button in the Interview Polish card.
    """
    from opencut.polish_state import clear_state

    data = request.get_json(force=True, silent=True) or {}
    filepath = (data.get("filepath") or "").strip()
    if not filepath:
        return jsonify({"error": "filepath is required"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    removed = clear_state(filepath)
    return jsonify({"ok": True, "removed": removed, "filepath": filepath})


# ---------------------------------------------------------------------------
