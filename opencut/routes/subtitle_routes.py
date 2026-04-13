"""
OpenCut Subtitle, Dead-Time, Stream Chapter, ND Filter & Timecode Routes

Blueprint providing endpoints for:
- Soft subtitle embedding and track listing
- SDH / HoH formatting
- Dead-time detection and speed ramping
- Stream recording auto-chaptering
- ND filter simulation
- Timecode format detection and conversion
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float

logger = logging.getLogger("opencut")

subtitle_bp = Blueprint("subtitle", __name__)


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

    data = request.get_json(force=True) or {}
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

    data = request.get_json(force=True) or {}
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

    data = request.get_json(force=True) or {}
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
