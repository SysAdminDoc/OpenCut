"""
OpenCut Subtitle, Dead-Time, Stream Chapter, ND Filter & Timecode Routes

Blueprint providing endpoints for:
- Soft subtitle embedding and track listing
- Subtitle timing resynchronisation with preview-before-apply
- SDH / HoH formatting
- Dead-time detection and speed ramping
- Stream recording auto-chaptering
- ND filter simulation
- Timecode format detection and conversion
"""

import logging
import os
import re

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import (
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_json_dict,
    require_csrf,
    safe_bool,
    safe_float,
    validate_filepath,
    validate_output_path,
    verify_destructive_confirm_token,
)

logger = logging.getLogger("opencut")

subtitle_bp = Blueprint("subtitle", __name__)


# =========================================================================
# Subtitle Resynchronisation (preview/apply)
# =========================================================================

@subtitle_bp.route("/subtitle/resync", methods=["POST"])
@require_csrf
def subtitle_resync():
    """Preview or apply text-assisted SRT timing resynchronisation.

    The default request is a no-write preview.  Applying the returned plan
    requires the explicit ``apply`` flag and its confirmation token, which
    keeps a panel or API client from overwriting a subtitle library by
    accident.
    """
    from opencut.core.subtitle_resync import resync_subtitles, write_resynced_srt

    data = get_json_dict() or {}
    srt_value = data.get("srt_path") or data.get("subtitle_path")
    if not isinstance(srt_value, str) or not srt_value.strip():
        return jsonify({
            "error": "srt_path is required",
            "code": "INVALID_INPUT",
        }), 400
    try:
        srt_path = validate_filepath(srt_value.strip())
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400

    video_value = data.get("video_path") or data.get("filepath")
    video_path = None
    if video_value:
        if not isinstance(video_value, str):
            return jsonify({
                "error": "video_path must be a file path",
                "code": "INVALID_INPUT",
            }), 400
        try:
            video_path = validate_filepath(video_value.strip())
        except ValueError as exc:
            return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400

    reference_segments = data.get("reference_segments")
    if reference_segments is None and not video_path:
        return jsonify({
            "error": "Provide video_path or reference_segments",
            "code": "INVALID_INPUT",
        }), 400
    if reference_segments is not None and not isinstance(reference_segments, (list, dict)):
        return jsonify({
            "error": "reference_segments must be a list or an object containing segments",
            "code": "INVALID_INPUT",
        }), 400

    fps = safe_float(data.get("fps"), default=30.0, min_val=1.0, max_val=120.0)
    match_threshold = safe_float(
        data.get("match_threshold"),
        default=0.72,
        min_val=0.5,
        max_val=1.0,
    )
    model = str(data.get("model") or "base").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", model):
        return jsonify({
            "error": "model must contain only letters, numbers, '.', '_' or '-'",
            "code": "INVALID_INPUT",
        }), 400
    language = data.get("language")
    if language is not None and not isinstance(language, str):
        return jsonify({
            "error": "language must be a string",
            "code": "INVALID_INPUT",
        }), 400

    output_value = data.get("output_path")
    if output_value:
        try:
            output_path = validate_output_path(str(output_value).strip())
        except ValueError as exc:
            return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    else:
        output_path = validate_output_path(
            f"{os.path.splitext(srt_path)[0]}_resynced.srt"
        )

    apply = safe_bool(data.get("apply"), default=False)
    overwrite = safe_bool(data.get("overwrite"), default=False)
    try:
        preview = resync_subtitles(
            srt_path,
            reference_segments=reference_segments,
            video_path=video_path,
            fps=fps,
            match_threshold=match_threshold,
            model=model,
            language=language,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Subtitle resync failed for %s: %s", srt_path, exc)
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400

    plan = build_destructive_plan(
        "subtitle_resync",
        targets=[srt_path, output_path],
        records=[
            {
                "matched_count": preview["matched_count"],
                "source_cue_count": preview["source_cue_count"],
            }
        ],
        metadata={
            "fps": preview["fps"],
            "rate": preview["rate"],
            "offset_seconds": preview["offset_seconds"],
            "overwrite": overwrite,
        },
        reversible=True,
    )

    response = {
        "preview": True,
        "applied": False,
        "output_path": output_path,
        "plan": plan,
        "result": preview,
    }
    if not apply:
        return jsonify(response)

    if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409
    try:
        written = write_resynced_srt(
            preview,
            output_path,
            overwrite=overwrite,
        )
    except (OSError, ValueError) as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    response.update({
        "preview": False,
        "applied": True,
        "write": written,
    })
    return jsonify(response)


# =========================================================================
# Soft Subtitle Embedding
# =========================================================================

@subtitle_bp.route("/subtitle/embed", methods=["POST"])
@require_csrf
@async_job("subtitle_embed")
def subtitle_embed(job_id, filepath, data):
    """Embed soft subtitle tracks into a video container."""
    from opencut.core.soft_subtitles import embed_subtitles

    subtitle_paths = data.get("subtitle_paths", [])
    if not subtitle_paths:
        raise ValueError("subtitle_paths is required (list of subtitle file paths)")

    languages = data.get("languages", None)
    container = data.get("container", "mp4")
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = embed_subtitles(
        filepath,
        subtitle_paths=subtitle_paths,
        languages=languages,
        output_path_override=output,
        container=container,
        on_progress=_on_progress,
    )
    return result


# =========================================================================
# Subtitle Track Listing (sync)
# =========================================================================

@subtitle_bp.route("/subtitle/tracks", methods=["POST"])
@require_csrf
def subtitle_tracks():
    """List subtitle tracks in a media file (synchronous)."""
    from opencut.core.soft_subtitles import list_subtitle_tracks
    from opencut.security import validate_filepath

    data = get_json_dict() or {}
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    tracks = list_subtitle_tracks(filepath)
    return jsonify({"tracks": tracks, "count": len(tracks)})


# =========================================================================
# SDH / HoH Formatting
# =========================================================================

@subtitle_bp.route("/subtitle/sdh-format", methods=["POST"])
@require_csrf
@async_job("sdh_format", filepath_param="srt_path")
def sdh_format(job_id, filepath, data):
    """Format an SRT file with SDH conventions."""
    from opencut.core.sdh_format import format_sdh

    diarization_data = data.get("diarization_data", None)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = format_sdh(
        filepath,
        diarization_data=diarization_data,
        output_path=output,
        on_progress=_on_progress,
    )
    return result


# =========================================================================
# Dead-Time Detection
# =========================================================================

@subtitle_bp.route("/video/dead-time/detect", methods=["POST"])
@require_csrf
@async_job("dead_time_detect")
def dead_time_detect(job_id, filepath, data):
    """Detect dead-time segments in a video."""
    from opencut.core.dead_time import detect_dead_time

    motion_threshold = safe_float(data.get("motion_threshold"), default=0.001,
                                  min_val=0.0001, max_val=1.0)
    min_duration = safe_float(data.get("min_duration"), default=3.0,
                              min_val=0.5, max_val=60.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_dead_time(
        filepath,
        motion_threshold=motion_threshold,
        min_duration=min_duration,
        on_progress=_on_progress,
    )
    return {
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "motion_score": s.motion_score,
            }
            for s in result.segments
        ],
        "total_dead_time": result.total_dead_time,
        "total_duration": result.total_duration,
        "dead_percentage": result.dead_percentage,
    }


# =========================================================================
# Dead-Time Speed Ramp
# =========================================================================

@subtitle_bp.route("/video/dead-time/speed-ramp", methods=["POST"])
@require_csrf
@async_job("dead_time_speed_ramp")
def dead_time_speed_ramp(job_id, filepath, data):
    """Speed-ramp dead-time segments in a video."""
    from opencut.core.dead_time import speed_ramp_dead_time

    dead_segments = data.get("dead_segments", [])
    if not dead_segments:
        raise ValueError("dead_segments is required (list of {start, end})")

    speed_factor = safe_float(data.get("speed_factor"), default=8.0,
                              min_val=1.5, max_val=100.0)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = speed_ramp_dead_time(
        filepath,
        dead_segments=dead_segments,
        speed_factor=speed_factor,
        output_path_override=output,
        on_progress=_on_progress,
    )
    return result


# =========================================================================
# Stream Auto-Chaptering
# =========================================================================

@subtitle_bp.route("/stream/auto-chapter", methods=["POST"])
@require_csrf
@async_job("stream_auto_chapter")
def stream_auto_chapter(job_id, filepath, data):
    """Auto-detect chapters in a stream recording."""
    from opencut.core.stream_chapters import auto_chapter_stream, export_youtube_chapters

    methods = data.get("methods", None)
    export_format = data.get("export_format", "json")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_chapter_stream(
        filepath,
        methods=methods,
        on_progress=_on_progress,
    )

    chapters_data = [
        {"start": ch.start, "end": ch.end, "title": ch.title}
        for ch in result.chapters
    ]

    response = {
        "chapters": chapters_data,
        "total_chapters": result.total_chapters,
        "total_duration": result.total_duration,
        "methods_used": result.methods_used,
    }

    # Optionally include YouTube-format export
    if export_format == "youtube":
        response["youtube_chapters"] = export_youtube_chapters(chapters_data)

    return response


# =========================================================================
# ND Filter Simulation
# =========================================================================

@subtitle_bp.route("/video/nd-filter", methods=["POST"])
@require_csrf
@async_job("nd_filter")
def nd_filter_sim(job_id, filepath, data):
    """Simulate ND filter motion blur on a video."""
    from opencut.core.nd_filter_sim import simulate_nd_filter

    shutter_angle = safe_float(data.get("shutter_angle"), default=180.0,
                               min_val=1.0, max_val=360.0)
    output = data.get("output_path", None)
    if output:
        output = validate_output_path(output)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = simulate_nd_filter(
        filepath,
        shutter_angle=shutter_angle,
        output_path_override=output,
        on_progress=_on_progress,
    )
    return result


# =========================================================================
# Timecode Detection (sync)
# =========================================================================

@subtitle_bp.route("/timecode/detect", methods=["POST"])
@require_csrf
def timecode_detect():
    """Detect timecode format of a media file (synchronous)."""
    from opencut.core.timecode_utils import detect_timecode_format
    from opencut.security import validate_filepath

    data = get_json_dict() or {}
    filepath = data.get("filepath", "").strip()
    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    info = detect_timecode_format(filepath)
    return jsonify({
        "fps": info.fps,
        "is_drop_frame": info.is_drop_frame,
        "detected_tc": info.detected_tc,
    })


# =========================================================================
# Timecode Conversion (sync)
# =========================================================================

@subtitle_bp.route("/timecode/convert", methods=["POST"])
@require_csrf
def timecode_convert():
    """Convert timecode between frame rates / formats (synchronous)."""
    from opencut.core.timecode_utils import convert_timecode

    data = get_json_dict() or {}
    tc = data.get("timecode", "").strip()
    if not tc:
        return jsonify({"error": "No timecode provided"}), 400

    source_fps = safe_float(data.get("source_fps"), default=29.97,
                            min_val=1.0, max_val=120.0)
    target_fps = safe_float(data.get("target_fps"), default=29.97,
                            min_val=1.0, max_val=120.0)
    source_df = safe_bool(data.get("source_df"), default=False)
    target_df = safe_bool(data.get("target_df"), default=False)

    try:
        result = convert_timecode(
            tc,
            source_fps=source_fps,
            target_fps=target_fps,
            source_df=source_df,
            target_df=target_df,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "timecode": result,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "source_df": source_df,
        "target_df": target_df,
    })
