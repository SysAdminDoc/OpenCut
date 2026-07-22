"""Transcript creation, editing, export, and round-trip routes."""

from .captions import (
    VALID_WHISPER_MODELS,
    CaptionConfig,
    _asr_provenance_payload,
    _caption_review_summary,
    _export_srt_with_policy,
    _is_cancelled,
    _legacy_srt_bom_requested,
    _resolve_output_dir,
    _roundtrip_diff_from_data,
    _safe_probe,
    _segment_payload,
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
    os,
    request,
    require_csrf,
    safe_bool,
    safe_error,
    safe_float,
    store_caption_revision,
    validate_filepath,
    validate_path,
    verify_destructive_confirm_token,
    workflow_step,
)


# Transcript Editor
# ---------------------------------------------------------------------------
@captions_bp.route("/transcript", methods=["POST"])
@require_csrf
@workflow_step("Transcribing")
@async_job("transcript", disk_operation="transcribe", resumable=True)
def get_transcript(job_id, filepath, data):
    """Transcribe and return full word-level transcript for editing."""
    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        raise ValueError(f"Invalid model: {model}")
    language = data.get("language", None)
    engine = data.get("engine", None)
    model_revision = data.get("model_revision", None)

    from opencut.core.captions import check_whisper_available, transcribe

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("Whisper is required for transcription.")

    _update_job(job_id, progress=10, message=f"Transcribing with {backend} ({model})...")
    config = CaptionConfig(
        engine=engine,
        model=model,
        model_revision=model_revision,
        language=language,
        word_timestamps=True,
    )
    result = transcribe(filepath, config=config)

    if _is_cancelled(job_id):
        return {"cancelled": True}

    # Build editable transcript structure
    return {
        "language": result.language,
        "duration": result.duration,
        "word_count": result.word_count,
        "full_text": result.text,
        "language_confidence": getattr(result, "language_confidence", 1.0),
        "asr_provenance": _asr_provenance_payload(result),
        **_caption_review_summary(result),
        "segments": [
            {
                **_segment_payload(seg, include_words=False, precision=3),
                "id": i,
                "words": [
                    {
                        "text": w.text.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "confidence": round(w.confidence, 3),
                        "boundary_confidence": (
                            round(w.boundary_confidence, 3)
                            if getattr(w, "boundary_confidence", None) is not None
                            else None
                        ),
                    }
                    for w in seg.words
                ],
            }
            for i, seg in enumerate(result.segments)
        ],
    }


@captions_bp.route("/transcript/export", methods=["POST"])
@require_csrf
def export_edited_transcript():
    """Export an edited transcript to SRT/VTT/ASS/JSON."""
    data = get_json_dict()
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    segments_data = data.get("segments", [])
    sub_format = data.get("format", "srt")
    if sub_format not in ("srt", "vtt", "json", "ass"):
        sub_format = "srt"
    language = data.get("language", "en")
    legacy_srt_bom = _legacy_srt_bom_requested(data)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not segments_data:
        return jsonify({"error": "No transcript segments provided"}), 400

    try:
        from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

        # Reconstruct TranscriptionResult from edited segments
        caption_segments = []
        for seg in segments_data:
            words = [
                Word(
                    text=w.get("text", ""),
                    start=safe_float(w.get("start", 0.0)),
                    end=safe_float(w.get("end", 0.0)),
                    confidence=safe_float(w.get("confidence", 1.0), default=1.0),
                )
                for w in seg.get("words", [])
            ]
            caption_segments.append(CaptionSegment(
                text=seg.get("text", ""),
                start=safe_float(seg.get("start", 0.0)),
                end=safe_float(seg.get("end", 0.0)),
                words=words,
                speaker=seg.get("speaker"),
                language=seg.get("language") or language,
                language_confidence=safe_float(
                    seg.get("language_confidence", 1.0),
                    default=1.0,
                    min_val=0.0,
                    max_val=1.0,
                ),
                confidence=safe_float(
                    seg.get("confidence", 1.0),
                    default=1.0,
                    min_val=0.0,
                    max_val=1.0,
                ),
                human_review_recommended=safe_bool(
                    seg.get("human_review_recommended", False),
                    default=False,
                ),
                review_reasons=[
                    str(reason)
                    for reason in (seg.get("review_reasons") or [])
                    if isinstance(reason, (str, int, float))
                ][:16],
            ))

        # Use max() instead of last-element access so out-of-order segments
        # (possible after user edits) still produce a correct total duration.
        result = TranscriptionResult(
            segments=caption_segments,
            language=language,
            duration=max((s.end for s in caption_segments), default=0.0) if caption_segments else 0.0,
        )

        effective_dir = _resolve_output_dir(filepath, output_dir)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(effective_dir, f"{base_name}_edited.{sub_format}")

        if sub_format == "vtt":
            export_vtt(result, out_path)
        elif sub_format == "json":
            export_json(result, out_path)
        elif sub_format == "ass":
            info = _safe_probe(filepath)
            vid_w = info.video.width if info and info.video else 1920
            vid_h = info.video.height if info and info.video else 1080
            export_ass(result, out_path, video_width=vid_w, video_height=vid_h, karaoke=True)
        else:
            _export_srt_with_policy(result, out_path, legacy_windows_bom=legacy_srt_bom)

        # F111: caption QC gate. Failures return 422 with diagnostics; pass
        # `?force=1` (or `force: true` in the JSON body) to override.
        qc_payload = None
        if sub_format in {"srt", "vtt"}:
            force_qc = (
                safe_bool(request.args.get("force"), default=False)
                or safe_bool(data.get("force"), default=False)
            )
            try:
                from opencut.core.caption_qc import qc_captions

                qc_result = qc_captions(srt_path=out_path)
                qc_payload = qc_result.as_dict()
                if not qc_result.overall_pass and not force_qc:
                    return (
                        jsonify(
                            {
                                "error": "caption_qc_failed",
                                "output_path": out_path,
                                "format": sub_format,
                                "qc": qc_payload,
                            }
                        ),
                        422,
                    )
            except Exception:
                # Never block exports on a QC failure that's itself broken —
                # the diagnostics are advisory in that case.
                qc_payload = {"overall_pass": True, "error_count": 0, "warning_count": 0, "diagnostics": [], "skipped": True}

        sidecar_path, sidecar_warnings = _write_caption_roundtrip_sidecar(result, out_path, sub_format, filepath, data)
        payload = {
            "output_path": out_path,
            "sidecar_path": sidecar_path,
            "metadata_preserved": bool(sidecar_path),
            "format": sub_format,
            "qc": qc_payload,
        }
        if sidecar_warnings:
            payload["warnings"] = sidecar_warnings
        if sub_format == "srt":
            payload["srt_encoding"] = "utf-8-sig" if legacy_srt_bom else "utf-8"
        return jsonify(payload)
    except Exception as e:
        return safe_error(e, "export_edited_transcript")


@captions_bp.route("/captions/round-trip/diff", methods=["POST"])
@require_csrf
def captions_round_trip_diff():
    """Compare edited timeline captions against a sidecar or lossy original segments."""
    data = get_json_dict()
    try:
        return jsonify(_roundtrip_diff_from_data(data))
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:
        return safe_error(exc, "captions_round_trip_diff")


@captions_bp.route("/captions/round-trip/apply", methods=["POST"])
@require_csrf
def captions_round_trip_apply():
    """Persist a confirmed caption round-trip diff as a new revision."""
    data = get_json_dict()
    try:
        diff_payload = _roundtrip_diff_from_data(data)
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    except Exception as exc:
        return safe_error(exc, "captions_round_trip_apply")

    plan = build_destructive_plan(
        "caption_roundtrip_apply",
        records=[{
            "changed_cues": diff_payload["summary"]["changed_cues"],
            "total_before": diff_payload["summary"]["total_before"],
            "total_after": diff_payload["summary"]["total_after"],
        }],
        metadata={
            "transcript_cache_key": diff_payload["source"].get("transcript_cache_key"),
            "source_file_hash": diff_payload["source"].get("source_file_hash"),
            "metadata_preserved": diff_payload["metadata_preserved"],
        },
        reversible=True,
    )
    if safe_bool(data.get("dry_run", False), default=False):
        return jsonify({
            "dry_run": True,
            "confirm_token": plan["confirm_token"],
            "plan": plan,
            "diff": diff_payload,
        })
    if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409
    try:
        revision = store_caption_revision(diff_payload)
    except Exception as exc:
        return safe_error(exc, "captions_round_trip_apply")
    return jsonify({
        "applied": True,
        "revision": revision,
        "diff": diff_payload,
    })


# ---------------------------------------------------------------------------
# Emoji Map for Captions
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "laugh": "\U0001f602", "lol": "\U0001f602", "haha": "\U0001f602", "funny": "\U0001f602",
    "love": "\u2764\ufe0f", "heart": "\u2764\ufe0f", "beautiful": "\u2764\ufe0f",
    "fire": "\U0001f525", "hot": "\U0001f525", "lit": "\U0001f525", "amazing": "\U0001f525",
    "cry": "\U0001f622", "sad": "\U0001f622", "tears": "\U0001f622",
    "wow": "\U0001f62e", "shocked": "\U0001f62e", "omg": "\U0001f62e", "unbelievable": "\U0001f62e",
    "think": "\U0001f914", "hmm": "\U0001f914", "wondering": "\U0001f914",
    "thumbs": "\U0001f44d", "great": "\U0001f44d", "good": "\U0001f44d", "nice": "\U0001f44d",
    "clap": "\U0001f44f", "bravo": "\U0001f44f", "congrats": "\U0001f44f",
    "money": "\U0001f4b0", "rich": "\U0001f4b0", "dollar": "\U0001f4b0", "expensive": "\U0001f4b0",
    "star": "\u2b50", "perfect": "\u2b50", "excellent": "\u2b50",
    "rocket": "\U0001f680", "launch": "\U0001f680", "skyrocket": "\U0001f680", "moon": "\U0001f680",
    "brain": "\U0001f9e0", "smart": "\U0001f9e0", "genius": "\U0001f9e0", "intelligent": "\U0001f9e0",
    "muscle": "\U0001f4aa", "strong": "\U0001f4aa", "power": "\U0001f4aa", "strength": "\U0001f4aa",
    "warning": "\u26a0\ufe0f", "danger": "\u26a0\ufe0f", "careful": "\u26a0\ufe0f",
    "check": "\u2705", "correct": "\u2705", "yes": "\u2705", "done": "\u2705",
    "wrong": "\u274c", "no": "\u274c", "false": "\u274c", "bad": "\u274c",
    "question": "\u2753", "why": "\u2753", "how": "\u2753",
    "eyes": "\U0001f440", "look": "\U0001f440", "see": "\U0001f440", "watch": "\U0001f440",
    "celebrate": "\U0001f389", "party": "\U0001f389", "winner": "\U0001f389", "victory": "\U0001f389",
    "sleep": "\U0001f634", "tired": "\U0001f634", "boring": "\U0001f634",
    "music": "\U0001f3b5", "song": "\U0001f3b5", "singing": "\U0001f3b5",
    "camera": "\U0001f4f8", "photo": "\U0001f4f8", "picture": "\U0001f4f8",
    "time": "\u23f0", "clock": "\u23f0", "hurry": "\u23f0", "fast": "\u23f0",
    "earth": "\U0001f30d", "world": "\U0001f30d", "global": "\U0001f30d",
    "sun": "\u2600\ufe0f", "sunny": "\u2600\ufe0f", "bright": "\u2600\ufe0f",
    "rain": "\U0001f327\ufe0f", "storm": "\U0001f327\ufe0f", "weather": "\U0001f327\ufe0f",
    "food": "\U0001f355", "eat": "\U0001f355", "hungry": "\U0001f355", "delicious": "\U0001f355",
    "coffee": "\u2615", "morning": "\u2615", "energy": "\u2615",
    "book": "\U0001f4da", "read": "\U0001f4da", "study": "\U0001f4da", "learn": "\U0001f4da",
    "computer": "\U0001f4bb", "code": "\U0001f4bb", "tech": "\U0001f4bb", "programming": "\U0001f4bb",
    "phone": "\U0001f4f1", "call": "\U0001f4f1", "mobile": "\U0001f4f1",
    "key": "\U0001f511", "secret": "\U0001f511", "unlock": "\U0001f511",
    "light": "\U0001f4a1", "idea": "\U0001f4a1", "tip": "\U0001f4a1", "insight": "\U0001f4a1",
    "target": "\U0001f3af", "goal": "\U0001f3af", "focus": "\U0001f3af", "aim": "\U0001f3af",
    "gem": "\U0001f48e", "diamond": "\U0001f48e", "valuable": "\U0001f48e", "premium": "\U0001f48e",
    "crown": "\U0001f451", "king": "\U0001f451", "queen": "\U0001f451", "best": "\U0001f451",
    "ghost": "\U0001f47b", "scary": "\U0001f47b", "spooky": "\U0001f47b", "halloween": "\U0001f47b",
    "skull": "\U0001f480", "dead": "\U0001f480", "kill": "\U0001f480", "destroy": "\U0001f480",
    "hundred": "\U0001f4af", "percent": "\U0001f4af",
}


@captions_bp.route("/captions/emoji-map", methods=["GET"])
def get_emoji_map():
    """Return the keyword->emoji mapping for auto-emoji insertion."""
    return jsonify({"emoji_map": EMOJI_MAP})


# ---------------------------------------------------------------------------
