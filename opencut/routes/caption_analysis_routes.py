"""Transcript summarization, chapter, repeat, QC, and export-preflight routes."""

from .captions import (
    VALID_WHISPER_MODELS,
    CaptionConfig,
    LLMConfig,
    _segment_payload,
    _update_job,
    async_job,
    captions_bp,
    get_json_dict,
    jsonify,
    request,
    require_csrf,
    safe_error,
    safe_float,
    safe_int,
    validate_filepath,
    workflow_step,
)


# Transcript Summarization (LLM)
# ---------------------------------------------------------------------------
@captions_bp.route("/transcript/summarize", methods=["POST"])
@require_csrf
@async_job("summarize")
def transcript_summarize(job_id, filepath, data):
    """Summarize a video transcript using LLM."""
    style = data.get("style", "bullets").strip()
    if style not in ("bullets", "paragraph", "detailed"):
        style = "bullets"
    transcript = data.get("transcript", None)

    # LLM config from request
    llm_provider = data.get("llm_provider", "ollama")
    if llm_provider not in ("ollama", "openai", "anthropic", "gemini"):
        llm_provider = "ollama"
    llm_model = data.get("llm_model", "")[:100]
    llm_api_key = data.get("llm_api_key", "")
    llm_base_url = data.get("llm_base_url", "")
    if llm_base_url:
        from urllib.parse import urlparse
        parsed = urlparse(llm_base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("llm_base_url must use http or https")

    from opencut.core.highlights import summarize_video
    from opencut.core.llm import LLMConfig

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    llm_config = LLMConfig(
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
    )

    # If no transcript provided, transcribe first
    transcript_segments = transcript
    if not transcript_segments:
        _update_job(job_id, progress=5, message="Transcribing video first...")
        from opencut.core.captions import transcribe as _transcribe
        t_result = _transcribe(filepath)
        transcript_segments = [
            _segment_payload(s, include_words=False)
            for s in t_result.segments
        ]

    summary = summarize_video(
        transcript_segments,
        style=style,
        llm_config=llm_config,
        on_progress=_on_progress,
    )

    return {
        "text": summary.text,
        "bullet_points": summary.bullet_points,
        "topics": summary.topics,
        "word_count": summary.word_count,
    }


# ---------------------------------------------------------------------------
# Captions: Chapter Generation
# ---------------------------------------------------------------------------
def _validate_chapters_input(data):
    """At least one of file/filepath/segments must be present."""
    segs = data.get("segments")
    has_segs = isinstance(segs, list) and len(segs) > 0
    has_file = bool((data.get("filepath") or data.get("file") or "").strip())
    if not has_segs and not has_file:
        return "file or segments required"
    return None


@captions_bp.route("/captions/chapters", methods=["POST"])
@require_csrf
@workflow_step("Generating chapters")
@async_job("chapters", filepath_required=False, pre_validate=_validate_chapters_input)
def captions_chapters(job_id, filepath, data):
    """Generate YouTube-style chapters from a transcript using an LLM."""
    # filepath may come as "filepath" or "file" -- decorator gets "filepath",
    # but also check "file" fallback
    if not filepath:
        filepath = data.get("file", "").strip()
        if filepath:
            filepath = validate_filepath(filepath)

    segments = data.get("segments", None)
    llm_provider = data.get("llm_provider", "ollama")
    if llm_provider not in ("ollama", "openai", "anthropic", "gemini"):
        llm_provider = "ollama"
    llm_model = data.get("llm_model", "llama3")
    api_key = data.get("api_key", "")
    max_chapters = safe_int(data.get("max_chapters", 15), 15, min_val=1, max_val=100)
    transcribe_model = data.get("model", "base")

    # Validate segments if provided directly
    if segments is not None:
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        if len(segments) > 50000:
            raise ValueError("Too many segments (max 50000)")

    if transcribe_model not in VALID_WHISPER_MODELS:
        transcribe_model = "base"

    effective_segments = segments

    # Transcribe if segments not provided
    if effective_segments is None:
        if not filepath:
            raise ValueError("file or segments required")

        from opencut.core.captions import check_whisper_available, transcribe

        available, backend = check_whisper_available()
        if not available:
            raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

        _update_job(job_id, progress=10, message="Transcribing for chapter generation...")
        config = CaptionConfig(model=transcribe_model, word_timestamps=False)

        result = transcribe(filepath, config=config)
        if getattr(result, "cache_hit", False):
            _update_job(job_id, progress=30, message="Using cached transcript...")
        if hasattr(result, "segments"):
            effective_segments = [
                _segment_payload(s, include_words=False)
                for s in result.segments
            ]
        else:
            effective_segments = []

    if not effective_segments:
        raise ValueError("No transcript segments available")

    _update_job(job_id, progress=50, message="Generating chapters with LLM...")

    llm_config = None
    if LLMConfig is not None:
        llm_config = LLMConfig(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key or "",
        )

    from opencut.core import chapter_gen
    result = chapter_gen.generate_chapters(
        effective_segments,
        llm_config=llm_config,
        max_chapters=max_chapters,
    )

    chapters = result.get("chapters", []) if isinstance(result, dict) else []
    description_block = result.get("description_block", "") if isinstance(result, dict) else str(result)

    return {"chapters": chapters, "description_block": description_block}


# ---------------------------------------------------------------------------
# Captions: Repeat / Duplicate Take Detection
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/repeat-detect", methods=["POST"])
@require_csrf
@workflow_step("Detecting repeats")
@async_job("repeat-detect")
def captions_repeat_detect(job_id, filepath, data):
    """Detect repeated takes in a recording and identify clean ranges."""
    model = data.get("model", "base")
    threshold = safe_float(data.get("threshold", 0.6), 0.6, min_val=0.0, max_val=1.0)
    gap_tolerance = safe_float(data.get("gap_tolerance", 2.0), 2.0, min_val=0.0, max_val=30.0)

    if model not in VALID_WHISPER_MODELS:
        model = "base"

    from opencut.core.captions import check_whisper_available, transcribe
    from opencut.utils.config import CaptionConfig as _CC

    available, backend = check_whisper_available()
    if not available:
        raise ValueError("No Whisper backend installed. Run: pip install faster-whisper")

    _update_job(job_id, progress=10, message="Transcribing with word-level timestamps...")

    config = _CC(model=model, word_timestamps=True)

    result = transcribe(filepath, config=config)
    if getattr(result, "cache_hit", False):
        _update_job(job_id, progress=30, message="Using cached transcript...")
    if hasattr(result, "segments"):
        segments = [
            _segment_payload(s, include_words=True)
            for s in result.segments
        ]
    else:
        segments = []

    if not segments:
        return {"repeats": [], "clean_ranges": [], "total_removed_seconds": 0.0}

    _update_job(job_id, progress=60, message="Detecting repeated takes...")

    from opencut.core import repeat_detect
    detection = repeat_detect.detect_repeated_takes(
        segments, threshold=threshold, gap_tolerance=gap_tolerance
    )

    repeats = detection.get("repeats", []) if isinstance(detection, dict) else []
    clean_ranges = detection.get("clean_ranges", []) if isinstance(detection, dict) else []
    total_removed = float(detection.get("total_removed_seconds", 0.0)) if isinstance(detection, dict) else 0.0

    return {
        "repeats": repeats,
        "clean_ranges": clean_ranges,
        "total_removed_seconds": total_removed,
    }


@captions_bp.route("/captions/qc", methods=["POST"])
@require_csrf
def captions_qc():
    """Run the caption QC gate (F111).

    Body fields::

        {
            "srt_path": "/abs/path/to/file.srt",   # OR srt_text
            "srt_text": "1\\n00:00:01,000 --> ...",
            "standard": "accessibility",            # see /captions/compliance/standards
            "reading_profile": "netflix-children",  # optional; see /captions/qc/reading-profiles
            "mode": "strict"                        # or "advisory"
        }

    Returns ``{ overall_pass, error_count, warning_count, diagnostics: [...] }``.
    """
    from flask import jsonify

    from opencut.core.caption_qc import qc_captions
    from opencut.errors import safe_error

    try:
        data = get_json_dict() or {}
        srt_path = (data.get("srt_path") or "").strip() or None
        srt_text = data.get("srt_text")
        standard = (data.get("standard") or "accessibility").strip()
        if standard not in ("accessibility", "broadcast", "ebu-tt-d"):
            standard = "accessibility"
        mode = (data.get("mode") or "strict").strip()
        if mode not in ("strict", "advisory"):
            mode = "strict"
        reading_profile = (
            data.get("reading_profile")
            or data.get("profile")
            or data.get("speed_profile")
            or None
        )
        if isinstance(reading_profile, str):
            reading_profile = reading_profile.strip() or None

        if not srt_path and not srt_text:
            return jsonify({"error": "srt_path or srt_text required"}), 400

        result = qc_captions(
            srt_path=srt_path,
            srt_text=srt_text,
            standard=standard,
            mode=mode,
            reading_profile=reading_profile,
        )
        return jsonify(result.as_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"srt_path not found: {exc.filename}"}), 404
    except Exception as exc:
        return safe_error(exc, "captions_qc")


@captions_bp.route("/captions/export/preflight", methods=["POST"])
@require_csrf
def captions_export_preflight():
    """Check caption export readiness and recommend fallback strategy.

    Body fields::

        {
            "segments": [...],         // OR srt_text
            "srt_text": "...",
            "video_duration": 120.5,   // optional
            "host_version": "26.1",    // optional, from panel
            "force_strategy": null,    // optional override
            "target_profile": "imsc1.3", // optional XML conformance target
            "language": "en"
        }

    Returns ``{ ready, fallback_strategy, diagnostics, caption_count, ... }``.
    """
    try:
        from opencut.core.caption_export_preflight import run_caption_export_preflight
    except ImportError:
        from opencut.core.caption_export_preflight import run_caption_export_preflight
    data = request.get_json(silent=True) or {}
    try:
        result = run_caption_export_preflight(
            segments=data.get("segments"),
            srt_text=data.get("srt_text"),
            video_duration=safe_float(data.get("video_duration", 0), default=0, min_val=0),
            host_version=data.get("host_version"),
            force_strategy=data.get("force_strategy"),
            target_profile=data.get("target_profile"),
            language=str(data.get("language", "en")),
        )
        return jsonify(result)
    except Exception as exc:
        return safe_error(exc, "captions_export_preflight")


@captions_bp.route("/captions/qc/reading-profiles", methods=["GET"])
def captions_qc_reading_profiles():
    """Return source-backed caption reading-speed profiles (F240)."""
    try:
        from opencut.core.caption_reading_profiles import (
            CORRECTION_NOTE,
            SOURCE_URLS,
            get_reading_speed_profiles,
        )

        return jsonify(
            {
                "profiles": get_reading_speed_profiles(),
                "source_urls": SOURCE_URLS,
                "correction_note": CORRECTION_NOTE,
            }
        )
    except Exception as e:
        return safe_error(e, "captions_qc_reading_profiles")
