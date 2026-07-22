"""Full caption and edit pipeline routes."""

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
    _update_job,
    async_job,
    captions_bp,
    detect_speech,
    export_premiere_xml,
    generate_zoom_events,
    get_edit_summary,
    get_preset,
    logger,
    os,
    require_csrf,
    safe_bool,
    validate_path,
)


# Full Pipeline
# ---------------------------------------------------------------------------
@captions_bp.route("/full", methods=["POST"])
@require_csrf
@async_job("full", disk_operation="full_pipeline")
def full_pipeline(job_id, filepath, data):
    """Run silence removal + zoom + optional captions."""
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    preset = data.get("preset", "youtube")
    skip_captions = safe_bool(data.get("skip_captions", False), False)
    skip_zoom = safe_bool(data.get("skip_zoom", False), False)
    remove_fillers = safe_bool(data.get("remove_fillers", False), False)
    accept_low_confidence_boundaries = safe_bool(
        data.get("accept_low_confidence_boundaries", False),
        False,
    )
    seq_name = data.get("sequence_name", "")
    legacy_srt_bom = _legacy_srt_bom_requested(data)

    cfg = get_preset(preset)
    effective_name = seq_name or _make_sequence_name(filepath, "Full Edit")
    ecfg = ExportConfig(sequence_name=effective_name)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")
    srt_path = os.path.join(effective_dir, f"{base_name}_opencut.srt")

    # Probe once for all pipeline steps
    _finfo = _safe_probe(filepath)
    _fdur = _finfo.duration if _finfo else 0.0

    # Calculate total steps
    total_steps = 2  # silence + export always
    if not skip_zoom:
        total_steps += 1
    if not skip_captions:
        total_steps += 1
    if remove_fillers:
        total_steps += 1
    step = [0]

    def next_step(msg):
        step[0] += 1
        pct = int(5 + (step[0] / total_steps) * 85)
        _update_job(job_id, progress=pct, message=f"Step {step[0]}/{total_steps}: {msg}")

    # Step: Silence detection
    next_step("Detecting silences...")
    segments = detect_speech(filepath, config=cfg.silence, file_duration=_fdur)
    summary = get_edit_summary(filepath, segments, file_duration=_fdur)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step: Filler word removal (requires Whisper)
    filler_stats = None
    transcription_result = None
    if remove_fillers:
        from opencut.core.captions import check_whisper_available, transcribe
        from opencut.core.fillers import (
            build_boundary_review,
            detect_fillers,
            remove_fillers_from_segments,
        )

        available, backend = check_whisper_available()
        if available:
            next_step(f"Detecting filler words ({backend})...")
            filler_cfg = CaptionConfig(
                engine=data.get("engine", None),
                model=cfg.captions.model,
                model_revision=data.get("model_revision", None),
                language=cfg.captions.language,
                word_timestamps=True,
            )
            # Use timeout to prevent hanging (10 min max)
            try:
                transcription_result = transcribe(filepath, config=filler_cfg, timeout=600)
            except TimeoutError as te:
                logger.warning(f"Filler detection timed out: {te}")
                # Continue without filler removal
                _update_job(job_id, message="Filler detection timed out, continuing without it...")
                transcription_result = None
            except Exception as te:
                logger.warning(f"Filler detection failed: {te}")
                _update_job(job_id, message="Filler detection failed, continuing without it...")
                transcription_result = None

            if transcription_result:
                analysis = detect_fillers(transcription_result, include_context_fillers=True)

                if analysis.hits:
                    boundary_review = build_boundary_review(
                        analysis.hits,
                        filepath=filepath,
                    )
                    if (
                        boundary_review["required"]
                        and not accept_low_confidence_boundaries
                    ):
                        return {
                            "preview_only": True,
                            "mutation_blocked": True,
                            "boundary_review": boundary_review,
                            "filler_stats": {
                                "total_fillers": len(analysis.hits),
                                "removed_fillers": 0,
                                "planned_fillers": len(analysis.hits),
                                "total_filler_time": analysis.total_filler_time,
                            },
                            "asr_provenance": _asr_provenance_payload(
                                transcription_result
                            ),
                        }
                    segments = remove_fillers_from_segments(segments, analysis.hits)
                    # Recalculate summary with filler-cleaned segments
                    summary = get_edit_summary(filepath, segments, file_duration=_fdur)
                    logger.info(
                        f"Fillers removed: {len(analysis.hits)} instances, "
                        f"{analysis.total_filler_time:.1f}s"
                    )

                filler_stats = {
                    "total_fillers": len(analysis.hits),
                    "removed_fillers": len(analysis.hits),
                    "total_filler_time": analysis.total_filler_time,
                    "filler_percentage": analysis.filler_percentage,
                    "total_words": analysis.total_words,
                    "breakdown": [
                        {"word": k, "count": c,
                         "time": round(sum(h.duration for h in analysis.hits if h.filler_key == k), 2),
                         "removed": True}
                        for k, c in sorted(analysis.filler_counts.items(), key=lambda x: -x[1])
                    ],
                }
        else:
            logger.warning("Filler removal requested but Whisper not installed, skipping")

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step: Zoom
    zoom_events = None
    if not skip_zoom:
        next_step("Analyzing emphasis points for zoom...")
        zoom_events = generate_zoom_events(filepath, config=cfg.zoom, speech_segments=segments)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Step: Captions
    captions_result = None
    if not skip_captions:
        from opencut.core.captions import check_whisper_available
        available, backend = check_whisper_available()
        if available:
            next_step("Generating captions...")
            # Reuse transcription from filler step if available
            if transcription_result:
                captions_result = transcription_result
            else:
                from opencut.core.captions import transcribe
                captions_result = transcribe(filepath, config=cfg.captions)

            # Remap caption timestamps to the condensed timeline
            from opencut.core.captions import remap_captions_to_segments
            captions_result = remap_captions_to_segments(captions_result, segments)
            logger.info(
                f"Captions remapped: {len(captions_result.segments)} segments, "
                f"condensed duration {captions_result.duration:.1f}s"
            )

            _export_srt_with_policy(captions_result, srt_path, legacy_windows_bom=legacy_srt_bom)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Export XML
    _update_job(job_id, progress=92, message="Exporting Premiere XML...")
    export_premiere_xml(filepath, segments, xml_path, config=ecfg, zoom_events=zoom_events)

    result_data = {
        "xml_path": xml_path,
        "summary": summary,
        "segments": len(segments),
        "zoom_events": len(zoom_events) if zoom_events else 0,
        "segments_data": [
            {"start": round(s.start, 4), "end": round(s.end, 4)}
            for s in segments
        ],
    }
    if captions_result:
        result_data["srt_path"] = srt_path
        result_data["srt_encoding"] = "utf-8-sig" if legacy_srt_bom else "utf-8"
        result_data["caption_segments"] = len(captions_result.segments)
        result_data["words"] = captions_result.word_count
        result_data["language"] = captions_result.language
        result_data.update(_caption_review_summary(captions_result))
    if filler_stats:
        result_data["filler_stats"] = filler_stats

    return result_data


# ---------------------------------------------------------------------------
