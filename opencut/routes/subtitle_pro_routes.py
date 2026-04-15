"""
OpenCut Professional Subtitling Routes

Endpoints for professional subtitle features: shot-change-aware timing,
multi-language editing, broadcast caption export, SDH formatting, and
dynamic subtitle positioning.
"""

import logging
import os
import tempfile

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

subtitle_pro_bp = Blueprint("subtitle_pro", __name__)


# ---------------------------------------------------------------------------
# POST /subtitles/shot-aware — Apply shot-change-aware timing
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/shot-aware", methods=["POST"])
@require_csrf
@async_job("shot_aware", filepath_required=False)
def shot_aware_timing(job_id, filepath, data):
    """Apply shot-change-aware timing adjustments to subtitles."""
    from opencut.core.subtitle_shot_aware import (
        export_to_file,
        process_shot_aware_dicts,
    )

    subtitles = data.get("subtitles")
    if not subtitles or not isinstance(subtitles, list):
        raise ValueError("'subtitles' must be a non-empty list of segments")

    cuts = data.get("cuts", [])
    if not isinstance(cuts, list):
        raise ValueError("'cuts' must be a list of timestamps")

    profile = str(data.get("profile", "netflix")).strip()
    custom_settings = data.get("custom_settings")
    export_format = str(data.get("format", "")).strip().lower()
    output_dir = str(data.get("output_dir", "")).strip()

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Processing ({pct}%)")

    result = process_shot_aware_dicts(
        subtitle_dicts=subtitles,
        cut_times=[float(c) for c in cuts],
        profile=profile,
        custom_settings=custom_settings,
        on_progress=_on_progress,
    )

    response = {
        "adjusted_subtitles": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in result.adjusted_subtitles
        ],
        "splits_made": result.splits_made,
        "gaps_enforced": result.gaps_enforced,
        "violations_fixed": result.violations_fixed,
        "profile_used": result.profile_used,
        "total_segments": result.total_segments,
        "merge_count": result.merge_count,
        "line_wraps": result.line_wraps,
    }

    # Export to file if format requested
    if export_format:
        out_dir = output_dir or tempfile.gettempdir()
        ext = export_format if export_format in ("srt", "vtt", "ass") else "srt"
        out_path = os.path.join(out_dir, f"shot_aware_output.{ext}")
        export_to_file(result.adjusted_subtitles, out_path, fmt=ext)
        response["output_path"] = out_path

    return response


# ---------------------------------------------------------------------------
# GET /subtitles/profiles — List timing profiles
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/profiles", methods=["GET"])
def list_timing_profiles():
    """Return available shot-aware timing profiles."""
    from opencut.core.subtitle_shot_aware import list_profiles

    profiles = list_profiles()
    return jsonify({"profiles": profiles, "count": len(profiles)})


# ---------------------------------------------------------------------------
# POST /subtitles/multilang/create — Create multi-language project
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/multilang/create", methods=["POST"])
@require_csrf
@async_job("multilang_create", filepath_required=False)
def multilang_create(job_id, filepath, data):
    """Create a new multi-language subtitle project."""
    from opencut.core.multilang_subtitle import create_project

    name = str(data.get("name", "Untitled")).strip()
    timing = data.get("timing_segments", [])
    base_language = str(data.get("base_language", "en")).strip()
    base_texts = data.get("base_texts")
    video_path = str(data.get("video_path", "")).strip()
    fps = safe_float(data.get("fps"), 24.0, 1.0, 120.0)

    if video_path:
        video_path = validate_filepath(video_path)

    _update_job(job_id, progress=20, message="Creating project...")

    project = create_project(
        name=name,
        timing_segments=timing,
        base_language=base_language,
        base_texts=base_texts,
        video_path=video_path,
        fps=fps,
    )

    info = project.info()
    return {
        "project_id": info.project_id,
        "name": info.name,
        "languages": info.languages,
        "segment_count": info.segment_count,
        "total_duration": info.total_duration,
    }


# ---------------------------------------------------------------------------
# POST /subtitles/multilang/import — Import language track
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/multilang/import", methods=["POST"])
@require_csrf
@async_job("multilang_import", filepath_required=False)
def multilang_import(job_id, filepath, data):
    """Import an SRT/VTT file as a language track into a project."""
    from opencut.core.multilang_subtitle import bulk_import

    project_id = str(data.get("project_id", "")).strip()
    if not project_id:
        raise ValueError("'project_id' is required")

    language = str(data.get("language", "")).strip()
    if not language:
        raise ValueError("'language' code is required")

    content = str(data.get("content", "")).strip()
    fmt = str(data.get("format", "srt")).strip().lower()
    align = data.get("align_to_timing", True)

    if not content:
        # Try reading from file path
        import_path = str(data.get("import_path", "")).strip()
        if import_path:
            import_path = validate_filepath(import_path)
            with open(import_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            raise ValueError("'content' or 'import_path' is required")

    _update_job(job_id, progress=30, message=f"Importing {language}...")

    result = bulk_import(
        project_id=project_id,
        language_code=language,
        content=content,
        fmt=fmt,
        align_to_timing=align,
    )

    info = result.info()
    return {
        "project_id": info.project_id,
        "languages": info.languages,
        "segment_count": info.segment_count,
        "imported_language": language,
    }


# ---------------------------------------------------------------------------
# POST /subtitles/multilang/export — Export per-language files
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/multilang/export", methods=["POST"])
@require_csrf
@async_job("multilang_export", filepath_required=False)
def multilang_export(job_id, filepath, data):
    """Export per-language subtitle files from a project."""
    from opencut.core.multilang_subtitle import export_language_files

    project_id = str(data.get("project_id", "")).strip()
    if not project_id:
        raise ValueError("'project_id' is required")

    fmt = str(data.get("format", "srt")).strip().lower()
    languages = data.get("languages")  # None = all
    output_dir = str(data.get("output_dir", "")).strip()
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_multilang_")

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Exporting ({pct}%)")

    result = export_language_files(
        project_id=project_id,
        output_dir=output_dir,
        fmt=fmt,
        languages=languages,
        on_progress=_on_progress,
    )

    return {
        "output_dir": output_dir,
        "format": fmt,
        "files": result,
        "languages_exported": len(result),
    }


# ---------------------------------------------------------------------------
# GET /subtitles/multilang/languages — List languages in project
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/multilang/languages", methods=["GET"])
def multilang_languages():
    """List languages in a multi-language project."""
    from opencut.core.multilang_subtitle import SUPPORTED_LANGUAGES, load_project

    project_id = request.args.get("project_id", "").strip()
    if not project_id:
        # Return all supported languages
        return jsonify({
            "supported": [
                {"code": k, "name": v}
                for k, v in sorted(SUPPORTED_LANGUAGES.items())
            ],
            "count": len(SUPPORTED_LANGUAGES),
        })

    data = load_project(project_id)
    info = data.info()
    return jsonify({
        "project_id": info.project_id,
        "languages": [
            {
                "code": lang,
                "name": SUPPORTED_LANGUAGES.get(lang, lang),
                "segment_count": data.segment_count,
            }
            for lang in info.languages
        ],
        "count": len(info.languages),
    })


# ---------------------------------------------------------------------------
# POST /subtitles/broadcast-export — Export to broadcast format
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/broadcast-export", methods=["POST"])
@require_csrf
@async_job("broadcast_export", filepath_required=False)
def broadcast_export(job_id, filepath, data):
    """Export subtitles to a broadcast caption format."""
    from opencut.core.broadcast_caption import export_broadcast, segments_from_dicts

    segments_data = data.get("segments")
    if not segments_data or not isinstance(segments_data, list):
        raise ValueError("'segments' must be a non-empty list")

    fmt = str(data.get("format", "cea608")).strip().lower()
    lang = str(data.get("language", "en")).strip()
    service_channel = safe_int(data.get("service_channel"), 1, 1, 8)
    output_dir = str(data.get("output_dir", "")).strip()

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="opencut_broadcast_")

    # Determine file extension
    ext_map = {
        "cea608": "scc",
        "cea708": "mcc",
        "ebu_tt": "xml",
        "ttml": "ttml",
        "imsc1": "xml",
        "webvtt_pos": "vtt",
    }
    ext = ext_map.get(fmt, "txt")
    output_path = os.path.join(output_dir, f"captions.{ext}")

    _update_job(job_id, progress=10, message=f"Exporting {fmt}...")

    segments = segments_from_dicts(segments_data)

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Exporting ({pct}%)")

    result = export_broadcast(
        segments=segments,
        output_path=output_path,
        fmt=fmt,
        lang=lang,
        service_channel=service_channel,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "format": result.format,
        "segments_exported": result.segments_exported,
        "validation_errors": result.validation_errors,
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------
# GET /subtitles/broadcast-formats — List supported broadcast formats
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/broadcast-formats", methods=["GET"])
def broadcast_formats():
    """Return list of supported broadcast caption formats."""
    from opencut.core.broadcast_caption import list_formats

    formats = list_formats()
    return jsonify({"formats": formats, "count": len(formats)})


# ---------------------------------------------------------------------------
# POST /subtitles/sdh-format — Apply SDH formatting
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/sdh-format", methods=["POST"])
@require_csrf
@async_job("sdh_format", filepath_required=False)
def sdh_format(job_id, filepath, data):
    """Apply SDH formatting to subtitle segments."""
    from opencut.core.sdh_formatter import SDHConfig, format_sdh

    subtitles = data.get("subtitles")
    if not subtitles or not isinstance(subtitles, list):
        raise ValueError("'subtitles' must be a non-empty list")

    diarization = data.get("diarization")
    audio_events = data.get("audio_events")
    stem_metadata = data.get("stem_metadata")
    tone_annotations = data.get("tone_annotations")

    # Build config from request
    config_data = data.get("config", {})
    config = SDHConfig(
        uppercase_speakers=config_data.get("uppercase_speakers", True),
        bracket_style=str(config_data.get("bracket_style", "square")),
        music_symbol=str(config_data.get("music_symbol", "\u266a")),
        speaker_separator=str(config_data.get("speaker_separator", ":")),
        include_sound_events=config_data.get("include_sound_events", True),
        include_music_notation=config_data.get("include_music_notation", True),
        include_tone_markers=config_data.get("include_tone_markers", True),
        sound_event_position=str(
            config_data.get("sound_event_position", "before")
        ),
    )

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Formatting ({pct}%)")

    result = format_sdh(
        subtitles=subtitles,
        diarization=diarization,
        audio_events=audio_events,
        stem_metadata=stem_metadata,
        tone_annotations=tone_annotations,
        config=config,
        on_progress=_on_progress,
    )

    return {
        "formatted_subtitles": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "original_text": s.original_text,
                "formatted_text": s.formatted_text,
                "speaker": s.speaker,
                "is_music": s.is_music,
                "sound_events": s.sound_events,
            }
            for s in result.formatted_subtitles
        ],
        "speaker_labels_added": result.speaker_labels_added,
        "sound_events_added": result.sound_events_added,
        "music_segments_marked": result.music_segments_marked,
        "tone_markers_added": result.tone_markers_added,
        "total_segments": result.total_segments,
    }


# ---------------------------------------------------------------------------
# POST /subtitles/auto-position — Apply dynamic positioning
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/auto-position", methods=["POST"])
@require_csrf
@async_job("auto_position", filepath_required=False)
def auto_position(job_id, filepath, data):
    """Apply dynamic positioning to subtitle segments."""
    from opencut.core.subtitle_position import export_to_file, position_subtitles

    subtitles = data.get("subtitles")
    if not subtitles or not isinstance(subtitles, list):
        raise ValueError("'subtitles' must be a non-empty list")

    frame_analyses = data.get("frame_analyses", {})
    video_width = safe_int(data.get("video_width"), 1920, 320, 7680)
    video_height = safe_int(data.get("video_height"), 1080, 240, 4320)
    export_format = str(data.get("format", "")).strip().lower()
    output_dir = str(data.get("output_dir", "")).strip()

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Positioning ({pct}%)")

    result = position_subtitles(
        subtitles=subtitles,
        frame_analyses=frame_analyses,
        video_width=video_width,
        video_height=video_height,
        on_progress=_on_progress,
    )

    response = {
        "positioned_subtitles": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "zone": s.zone,
                "x": s.x,
                "y": s.y,
                "repositioned": s.repositioned,
                "obstruction_reason": s.obstruction_reason,
            }
            for s in result.positioned_subtitles
        ],
        "repositioned_count": result.repositioned_count,
        "obstruction_types": result.obstruction_types,
        "total_segments": result.total_segments,
        "frames_analyzed": result.frames_analyzed,
    }

    # Export to ASS if requested
    if export_format == "ass":
        out_dir = output_dir or tempfile.mkdtemp(prefix="opencut_position_")
        out_path = os.path.join(out_dir, "positioned_subtitles.ass")
        export_to_file(result, out_path, video_width, video_height)
        response["output_path"] = out_path

    return response


# ---------------------------------------------------------------------------
# POST /subtitles/auto-position/preview — Preview single frame position
# ---------------------------------------------------------------------------
@subtitle_pro_bp.route("/subtitles/auto-position/preview", methods=["POST"])
@require_csrf
def position_preview():
    """Preview subtitle positioning for a single frame."""
    from opencut.core.subtitle_position import preview_position

    data = request.get_json(force=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    frame_data = data.get("frame_data", {})
    video_width = safe_int(data.get("video_width"), 1920, 320, 7680)
    video_height = safe_int(data.get("video_height"), 1080, 240, 4320)

    result = preview_position(
        text=text,
        frame_data=frame_data,
        video_width=video_width,
        video_height=video_height,
    )

    return jsonify(result)
