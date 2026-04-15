"""
OpenCut AI Content Intelligence Routes

Endpoints for AI-powered content intelligence features:
- Scene description / alt-text generation
- Video summarization (text + visual)
- OCR text extraction
- Emotion timeline analysis
- Project organization
- Natural language batch operations
"""

import logging

from flask import Blueprint

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath, validate_path

logger = logging.getLogger("opencut")

ai_intel_bp = Blueprint("ai_intel", __name__)


# ---------------------------------------------------------------------------
# POST /api/ai/scene-describe
# ---------------------------------------------------------------------------
@ai_intel_bp.route("/api/ai/scene-describe", methods=["POST"])
@require_csrf
@async_job("scene_describe")
def ai_scene_describe(job_id, filepath, data):
    """Generate descriptions for video scenes."""
    from opencut.core.scene_description import describe_all_scenes, describe_scene

    timestamps = data.get("timestamps")
    llm_config = data.get("llm_config")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if timestamps is not None and isinstance(timestamps, list):
        # Describe specific timestamps
        timestamps = [float(t) for t in timestamps if isinstance(t, (int, float))]
    else:
        timestamps = None

    # Single scene or all scenes
    single_timestamp = data.get("timestamp")
    if single_timestamp is not None and timestamps is None:
        single_timestamp = safe_float(single_timestamp, 0.0, 0.0)
        result = describe_scene(
            video_path=filepath,
            timestamp=single_timestamp,
            llm_config=llm_config,
            on_progress=_on_progress,
        )
        return {
            "timestamp": result.timestamp,
            "description": result.description,
            "alt_text": result.alt_text,
            "dominant_colors": result.dominant_colors,
            "brightness": result.brightness,
            "edge_density": result.edge_density,
            "tags": result.tags,
            "method": result.method,
        }

    result = describe_all_scenes(
        video_path=filepath,
        scene_timestamps=timestamps,
        llm_config=llm_config,
        on_progress=_on_progress,
    )

    return {
        "descriptions": [
            {
                "timestamp": d.timestamp,
                "description": d.description,
                "alt_text": d.alt_text,
                "dominant_colors": d.dominant_colors,
                "brightness": d.brightness,
                "edge_density": d.edge_density,
                "tags": d.tags,
                "method": d.method,
            }
            for d in result.descriptions
        ],
        "total_scenes": result.total_scenes,
        "duration": result.duration,
        "method": result.method,
    }


# ---------------------------------------------------------------------------
# POST /api/ai/summarize
# ---------------------------------------------------------------------------
@ai_intel_bp.route("/api/ai/summarize", methods=["POST"])
@require_csrf
@async_job("video_summarize", filepath_required=False)
def ai_summarize(job_id, filepath, data):
    """Generate text or visual video summary."""
    mode = data.get("mode", "text").strip()
    llm_config = data.get("llm_config")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if mode == "text":
        from opencut.core.video_summary import text_summary

        transcript = data.get("transcript", "")
        if not transcript:
            raise ValueError("transcript is required for text summarization")

        max_sentences = safe_int(data.get("max_sentences"), 5, 1, 20)

        result = text_summary(
            transcript=transcript,
            max_sentences=max_sentences,
            llm_config=llm_config,
            on_progress=_on_progress,
        )

        return {
            "summary": result.summary,
            "keywords": result.keywords,
            "key_sentences": result.key_sentences,
            "word_count": result.word_count,
            "method": result.method,
        }

    elif mode == "visual":
        from opencut.core.video_summary import visual_summary

        if not filepath:
            raise ValueError("filepath is required for visual summarization")

        scenes = data.get("scenes", [])
        if not scenes or not isinstance(scenes, list):
            raise ValueError("scenes list with timestamp and score is required")

        top_n = safe_int(data.get("top_n"), 5, 1, 20)
        clip_duration = safe_float(data.get("clip_duration"), 3.0, 0.5, 30.0)
        output_dir = data.get("output_dir", "").strip()
        if output_dir:
            output_dir = validate_path(output_dir)

        result = visual_summary(
            video_path=filepath,
            scenes=scenes,
            top_n=top_n,
            clip_duration=clip_duration,
            output_dir=output_dir,
            on_progress=_on_progress,
        )

        return {
            "output_path": result.output_path,
            "selected_scenes": result.selected_scenes,
            "total_duration": result.total_duration,
            "summary_duration": result.summary_duration,
            "scene_count": result.scene_count,
        }

    else:
        raise ValueError(f"Unknown summary mode: {mode}. Use 'text' or 'visual'.")


# ---------------------------------------------------------------------------
# POST /api/ai/ocr
# ---------------------------------------------------------------------------
@ai_intel_bp.route("/api/ai/ocr", methods=["POST"])
@require_csrf
@async_job("ocr_extract")
def ai_ocr(job_id, filepath, data):
    """Extract text from video via OCR."""
    from opencut.core.ocr_extract import extract_text_frames, search_text_in_video

    query = data.get("query", "").strip()
    interval = safe_float(data.get("interval"), 1.0, 0.1, 30.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if query:
        # Search mode
        case_sensitive = bool(data.get("case_sensitive", False))

        result = search_text_in_video(
            video_path=filepath,
            query=query,
            interval=interval,
            case_sensitive=case_sensitive,
            on_progress=_on_progress,
        )

        return {
            "query": result.query,
            "hits": [
                {
                    "timestamp": h.timestamp,
                    "text": h.text,
                    "context": h.context,
                    "confidence": h.confidence,
                }
                for h in result.hits
            ],
            "total_hits": result.total_hits,
        }

    else:
        # Full extraction mode
        similarity_threshold = safe_float(data.get("similarity_threshold"), 0.8, 0.0, 1.0)

        result = extract_text_frames(
            video_path=filepath,
            interval=interval,
            similarity_threshold=similarity_threshold,
            on_progress=_on_progress,
        )

        return {
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "text": f.text,
                    "confidence": f.confidence,
                }
                for f in result.frames if f.text.strip()
            ],
            "unique_texts": result.unique_texts,
            "total_frames_sampled": result.total_frames_sampled,
            "frames_with_text": result.frames_with_text,
            "duration": result.duration,
        }


# ---------------------------------------------------------------------------
# POST /api/ai/emotion-timeline
# ---------------------------------------------------------------------------
@ai_intel_bp.route("/api/ai/emotion-timeline", methods=["POST"])
@require_csrf
@async_job("emotion_timeline")
def ai_emotion_timeline(job_id, filepath, data):
    """Build emotion/energy timeline for a video."""
    from opencut.core.emotion_timeline import build_emotion_timeline

    transcript = data.get("transcript")  # Optional dict
    interval = safe_float(data.get("interval"), 1.0, 0.25, 10.0)
    audio_weight = safe_float(data.get("audio_weight"), 0.4, 0.0, 1.0)
    speech_weight = safe_float(data.get("speech_weight"), 0.3, 0.0, 1.0)
    sentiment_weight = safe_float(data.get("sentiment_weight"), 0.3, 0.0, 1.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = build_emotion_timeline(
        video_path=filepath,
        transcript=transcript,
        interval=interval,
        audio_weight=audio_weight,
        speech_weight=speech_weight,
        sentiment_weight=sentiment_weight,
        on_progress=_on_progress,
    )

    return {
        "timeline": [
            {
                "time": p.time,
                "energy": p.energy,
                "audio_rms": p.audio_rms,
                "speech_rate": p.speech_rate,
                "sentiment": p.sentiment,
            }
            for p in result.timeline
        ],
        "peaks": [
            {
                "time": p.time,
                "energy": p.energy,
                "start": p.start,
                "end": p.end,
                "duration": p.duration,
                "dominant_signal": p.dominant_signal,
                "label": p.label,
            }
            for p in result.peaks
        ],
        "duration": result.duration,
        "avg_energy": result.avg_energy,
        "max_energy": result.max_energy,
        "sample_interval": result.sample_interval,
        "signals_used": result.signals_used,
    }


# ---------------------------------------------------------------------------
# POST /api/ai/organize-project
# ---------------------------------------------------------------------------
@ai_intel_bp.route("/api/ai/organize-project", methods=["POST"])
@require_csrf
@async_job("organize_project", filepath_required=False)
def ai_organize_project(job_id, filepath, data):
    """Analyze project media and suggest organization."""
    from opencut.core.project_organizer import (
        analyze_project_media,
        generate_bin_structure,
    )

    file_list = data.get("file_list", [])
    if not file_list or not isinstance(file_list, list):
        raise ValueError("file_list is required (list of file paths)")

    # Validate each file path
    validated_files = []
    for fp in file_list:
        if isinstance(fp, str) and fp.strip():
            try:
                validated = validate_filepath(fp.strip())
                validated_files.append(validated)
            except ValueError:
                logger.debug("Skipping invalid file path: %s", fp)

    if not validated_files:
        raise ValueError("No valid file paths in file_list")

    strategy = data.get("strategy", "auto").strip()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _on_progress(5, "Analyzing media files...")

    analysis = analyze_project_media(
        file_list=validated_files,
        on_progress=lambda pct, msg: _on_progress(int(pct * 0.6), msg),
    )

    _on_progress(65, "Generating bin structure...")

    bin_structure = generate_bin_structure(
        analysis=analysis,
        strategy=strategy,
        on_progress=lambda pct, msg: _on_progress(65 + int(pct * 0.35), msg),
    )

    return {
        "analysis": {
            "total_files": analysis.total_files,
            "total_duration": analysis.total_duration,
            "media_types": analysis.media_types,
            "shot_types": analysis.shot_types,
            "resolutions": analysis.resolutions,
            "codecs": analysis.codecs,
            "date_groups": {k: len(v) for k, v in analysis.date_groups.items()},
            "scene_groups": {k: len(v) for k, v in analysis.scene_groups.items()},
        },
        "bins": [
            {
                "name": b.name,
                "path": b.path,
                "files": b.files,
                "file_count": b.file_count,
                "description": b.description,
            }
            for b in bin_structure.bins
        ],
        "total_bins": bin_structure.total_bins,
        "strategy": bin_structure.strategy,
    }


# ---------------------------------------------------------------------------
# POST /api/ai/batch-command
# ---------------------------------------------------------------------------
@ai_intel_bp.route("/api/ai/batch-command", methods=["POST"])
@require_csrf
@async_job("nl_batch", filepath_required=False)
def ai_batch_command(job_id, filepath, data):
    """Parse and execute a natural language batch command."""
    from opencut.core.nl_batch import execute_batch, parse_batch_command

    command_text = data.get("command", "").strip()
    if not command_text:
        raise ValueError("command text is required")

    file_list = data.get("file_list", [])
    dry_run = bool(data.get("dry_run", True))  # Default to dry run for safety

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _on_progress(10, "Parsing command...")

    command = parse_batch_command(command_text)

    parse_result = {
        "original_text": command.original_text,
        "parsed": command.parsed,
        "confidence": command.confidence,
        "explanation": command.explanation,
        "operation": command.operation.action,
        "parameters": command.operation.parameters,
    }

    if not command.parsed:
        return {
            "parse_result": parse_result,
            "executed": False,
            "error": "Command could not be parsed",
        }

    # If file_list provided and not just parse-only, execute
    if file_list and isinstance(file_list, list):
        # Validate file paths
        valid_files = []
        for fp in file_list:
            if isinstance(fp, str) and fp.strip():
                try:
                    validated = validate_filepath(fp.strip())
                    valid_files.append(validated)
                except ValueError:
                    pass

        if valid_files:
            _on_progress(30, "Executing batch command...")

            result = execute_batch(
                command=command,
                file_list=valid_files,
                dry_run=dry_run,
                on_progress=lambda pct, msg: _on_progress(30 + int(pct * 0.7), msg),
            )

            return {
                "parse_result": parse_result,
                "executed": True,
                "dry_run": dry_run,
                "files_matched": result.files_matched,
                "files_processed": result.files_processed,
                "files_failed": result.files_failed,
                "results": result.results,
                "errors": result.errors,
            }

    _on_progress(100, "Command parsed (no files to process)")

    return {
        "parse_result": parse_result,
        "executed": False,
        "message": "Command parsed successfully. Provide file_list to execute.",
    }
