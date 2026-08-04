"""Full caption and edit pipeline routes."""

import math
import os

__all__ = ["full_pipeline", "cleanup_preview", "cleanup_apply"]

from opencut.core.workflow import build_cleanup_plan, validate_cleanup_plan

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
# Reviewable cleanup chain
# ---------------------------------------------------------------------------
def _cleanup_options(data):
    """Normalize the public cleanup verb's small option surface."""
    preset = data.get("cleanup_preset", data.get("preset", "podcast"))
    if not isinstance(preset, str) or not preset.strip():
        preset = "podcast"
    preset_name = preset.strip().lower()
    target_lufs = data.get("target_lufs")
    if target_lufs is None:
        from opencut.core.loudness_standards import get_loudness_preset

        target_lufs = get_loudness_preset(preset_name)["i"]
    try:
        target_lufs = float(target_lufs)
    except (TypeError, ValueError):
        target_lufs = -16.0
    if not math.isfinite(target_lufs):
        target_lufs = -16.0
    target_lufs = max(-70.0, min(0.0, target_lufs))
    return {
        "preset": preset_name,
        "denoise": not safe_bool(data.get("skip_denoise", False), False),
        "denoise_method": data.get("denoise_method", "afftdn"),
        "denoise_strength": data.get("denoise_strength", 0.7),
        "loudness": not safe_bool(data.get("skip_loudness", False), False),
        "target_lufs": target_lufs,
        "captions": not safe_bool(
            data.get("skip_captions", False),
            False,
        ) and safe_bool(data.get("captions", True), True),
    }


def _cleanup_output_dir(filepath, requested_dir, *, create):
    """Resolve output storage without creating anything during preview."""
    requested = str(requested_dir or "").strip()
    if requested:
        resolved = validate_path(requested)
        if create:
            os.makedirs(resolved, exist_ok=True)
        return resolved
    source_dir = os.path.dirname(os.path.abspath(filepath))
    if source_dir and os.path.isdir(source_dir):
        return source_dir
    fallback = os.path.join(os.path.abspath(os.path.expanduser("~")), "opencut_output")
    if create:
        os.makedirs(fallback, exist_ok=True)
    return fallback


def _cleanup_capabilities():
    """Return optional capability state for honest preview degradation."""
    from opencut.checks import ffmpeg_security_available
    from opencut.core.captions import check_whisper_available

    ffmpeg_available = bool(ffmpeg_security_available())
    captions_available, captions_backend = check_whisper_available()
    return {
        "ffmpeg": ffmpeg_available,
        "captions_available": bool(captions_available),
        "captions_backend": captions_backend,
        "captions_reason": (
            "Install faster-whisper, openai-whisper, or WhisperX"
            if not captions_available
            else ""
        ),
    }


def _cleanup_plan_for_source(job_id, filepath, data, *, preview):
    """Analyze a source and build the side-effect-free cleanup plan."""
    options = _cleanup_options(data)
    output_dir = _cleanup_output_dir(
        filepath,
        data.get("output_dir", ""),
        create=not preview,
    )
    cfg = get_preset(options["preset"])
    info = _safe_probe(filepath)
    duration = info.duration if info else 0.0
    if duration <= 0:
        raise ValueError("Could not determine the source duration for cleanup preview")

    capabilities = _cleanup_capabilities()
    _update_job(job_id, progress=10, message="Analyzing the cleanup chain...")
    segments = detect_speech(
        filepath,
        config=cfg.silence,
        file_duration=duration,
    )
    if _is_cancelled(job_id):
        return None, options, cfg

    source_loudness = {}
    if capabilities["ffmpeg"]:
        try:
            from opencut.core.audio_suite import measure_loudness

            _update_job(job_id, progress=42, message="Measuring source loudness...")
            measured = measure_loudness(filepath)
            source_loudness = {
                "integrated_lufs": round(measured.input_i, 2),
                "true_peak_dbtp": round(measured.input_tp, 2),
                "loudness_range_lu": round(measured.input_lra, 2),
            }
        except Exception as exc:  # noqa: BLE001 - measurement is preview-only
            logger.info("Cleanup loudness preview unavailable: %s", exc)

    plan = build_cleanup_plan(
        filepath=filepath,
        duration=duration,
        speech_segments=segments,
        options=options,
        capabilities=capabilities,
        output_dir=output_dir,
        source_loudness=source_loudness,
    )
    plan["summary"].update({
        "original_duration": round(duration, 4),
        "kept_seconds": round(sum(segment.duration for segment in segments), 4),
    })
    # The plan hash must cover every field the apply request will validate.
    from opencut.core.workflow import cleanup_plan_id

    plan["plan_id"] = cleanup_plan_id(plan)
    _update_job(job_id, progress=92, message="Cleanup preview ready for review")
    return plan, options, cfg


@captions_bp.route("/cleanup/preview", methods=["POST"])
@require_csrf
@async_job("cleanup-preview")
def cleanup_preview(job_id, filepath, data):
    """Analyze the standard cleanup chain without writing files or Premiere."""
    plan, _options, _cfg = _cleanup_plan_for_source(
        job_id,
        filepath,
        data,
        preview=True,
    )
    if plan is None:
        return {"cancelled": True}
    return {
        "preview_only": True,
        "requires_confirmation": True,
        "reversible": True,
        "cleanup_plan": plan,
        "plan_id": plan["plan_id"],
    }


def _cleanup_step(plan, step_id):
    return next(
        step for step in plan["steps"]
        if isinstance(step, dict) and step.get("id") == step_id
    )


@captions_bp.route("/cleanup/apply", methods=["POST"])
@require_csrf
@async_job("cleanup-apply", disk_operation="full_pipeline")
def cleanup_apply(job_id, filepath, data):
    """Apply one reviewed cleanup plan and return host-import artifacts."""
    plan = data.get("cleanup_plan", data.get("plan"))
    valid, error = validate_cleanup_plan(plan, filepath=filepath)
    if not valid:
        raise ValueError(error)

    options = dict(plan["options"])
    artifacts = dict(plan["artifacts"])
    output_dir = validate_path(str(artifacts.get("output_dir") or ""))
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_preset(options.get("preset", "podcast"))
    from opencut.checks import ffmpeg_security_available

    ffmpeg_available = bool(ffmpeg_security_available())
    from opencut.core.silence import TimeSegment

    segments = [
        TimeSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            label=str(segment.get("label") or "speech"),
        )
        for segment in plan["segments_data"]
    ]
    current_input = filepath
    completed_steps = []
    degraded_steps = []

    def progress(percent, message):
        _update_job(job_id, progress=percent, message=message)

    denoise_step = _cleanup_step(plan, "denoise")
    if denoise_step.get("state") == "ready" and ffmpeg_available:
        from opencut.core.audio_suite import denoise_audio

        progress(12, "Removing background noise...")
        current_input = denoise_audio(
            current_input,
            artifacts["denoised_path"],
            method=options.get("denoise_method", "afftdn"),
            strength=float(options.get("denoise_strength", 0.7)),
            on_progress=lambda pct, msg="": progress(12 + int(pct * 0.22), msg),
        )
        completed_steps.append("denoise")
    else:
        degraded_steps.append({
            "id": "denoise",
            "state": "skipped",
            "reason": denoise_step.get("reason")
            or "FFmpeg became unavailable after preview",
        })

    if _is_cancelled(job_id):
        return {"cancelled": True}

    loudness_step = _cleanup_step(plan, "loudness")
    if loudness_step.get("state") == "ready" and ffmpeg_available:
        from opencut.core.audio_suite import normalize_loudness

        progress(38, "Normalizing loudness...")
        normalized_path, loudness_info = normalize_loudness(
            current_input,
            artifacts["normalized_path"],
            preset=options.get("preset", "podcast"),
            target_lufs=float(options["target_lufs"]),
            on_progress=lambda pct, msg="": progress(38 + int(pct * 0.22), msg),
        )
        current_input = normalized_path
        completed_steps.append("loudness")
    else:
        loudness_info = None
        degraded_steps.append({
            "id": "loudness",
            "state": "skipped",
            "reason": loudness_step.get("reason")
            or "FFmpeg became unavailable after preview",
        })

    if _is_cancelled(job_id):
        return {"cancelled": True}

    captions_result = None
    captions_step = _cleanup_step(plan, "captions")
    if captions_step.get("state") == "ready":
        from opencut.core.captions import check_whisper_available, transcribe

        available, backend = check_whisper_available()
        if available:
            progress(64, f"Generating captions ({backend})...")
            try:
                captions_result = transcribe(current_input, config=cfg.captions, timeout=600)
                from opencut.core.captions import remap_captions_to_segments

                captions_result = remap_captions_to_segments(captions_result, segments)
                _export_srt_with_policy(
                    captions_result,
                    artifacts["srt_path"],
                    legacy_windows_bom=False,
                )
                completed_steps.append("captions")
            except Exception as exc:  # noqa: BLE001 - captions are optional
                logger.warning("Cleanup captions degraded: %s", exc)
                degraded_steps.append({
                    "id": "captions",
                    "state": "skipped",
                    "reason": f"Caption generation failed: {exc}",
                })
        else:
            degraded_steps.append({
                "id": "captions",
                "state": "skipped",
                "reason": "Caption backend is no longer available",
            })
    else:
        degraded_steps.append({
            "id": "captions",
            "state": "skipped",
            "reason": captions_step.get("reason", "Skipped by preview plan"),
        })

    if _is_cancelled(job_id):
        return {"cancelled": True}

    progress(90, "Writing the reviewed cleanup interchange...")
    effective_name = _make_sequence_name(filepath, "Cleanup")
    export_premiere_xml(
        current_input,
        segments,
        artifacts["xml_path"],
        config=ExportConfig(sequence_name=effective_name),
    )
    completed_steps.insert(0, "silence_trim")

    result = {
        "cleanup_chain": True,
        "chain": "standard-cleanup",
        "plan_id": plan["plan_id"],
        "preview_only": False,
        "requires_host_review": True,
        "reversible": True,
        "journal_action": "import_sequence",
        "xml_path": artifacts["xml_path"],
        "media_path": current_input,
        "summary": plan["summary"],
        "segments": len(segments),
        "segments_data": [
            {"start": round(segment.start, 4), "end": round(segment.end, 4)}
            for segment in segments
        ],
        "completed_steps": completed_steps,
        "degraded_steps": degraded_steps,
        "output_dir": output_dir,
    }
    if captions_result:
        result.update({
            "srt_path": artifacts["srt_path"],
            "caption_segments": len(captions_result.segments),
            "words": captions_result.word_count,
            "language": captions_result.language,
        })
        result.update(_caption_review_summary(captions_result))
    if loudness_info is not None:
        result["input_loudness"] = loudness_info.input_i
        result["target_loudness"] = float(options["target_lufs"])
    return result


# ---------------------------------------------------------------------------
