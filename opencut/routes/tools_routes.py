"""
OpenCut Production Tools Routes

Cursor zoom, lower-thirds, beat-synced cuts, redaction,
vertical reframing, and telemetry overlay endpoints.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

tools_bp = Blueprint("tools", __name__)


# ---------------------------------------------------------------------------
# Cursor Zoom
# ---------------------------------------------------------------------------
@tools_bp.route("/screen/cursor-zoom", methods=["POST"])
@require_csrf
@async_job("cursor_zoom")
def cursor_zoom(job_id, filepath, data):
    """Detect cursor clicks in screen recording and apply smooth zoom."""
    zoom_factor = safe_float(data.get("zoom_factor", 2.0), 2.0, min_val=1.1, max_val=5.0)
    zoom_duration = safe_float(data.get("zoom_duration", 1.5), 1.5, min_val=0.5, max_val=5.0)

    out_dir = data.get("output_dir", "")
    if out_dir:
        try:
            out_dir = validate_path(out_dir)
        except ValueError:
            out_dir = ""
    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _update_job(job_id, progress=5, message="Detecting cursor clicks...")

    from opencut.core.cursor_zoom import apply_cursor_zoom, detect_click_regions

    click_regions = detect_click_regions(filepath, on_progress=_p)

    _update_job(job_id, progress=50, message=f"Found {len(click_regions)} clicks, applying zoom...")

    result = apply_cursor_zoom(
        input_path=filepath,
        click_regions=click_regions,
        zoom_factor=zoom_factor,
        zoom_duration=zoom_duration,
        output_path_str=out_path,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Lower-Thirds
# ---------------------------------------------------------------------------
@tools_bp.route("/lower-thirds/generate", methods=["POST"])
@require_csrf
@async_job("lower_thirds_generate", filepath_required=False)
def lower_thirds_generate(job_id, filepath, data):
    """Generate a single lower-third overlay video."""
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("'name' is required")

    title = str(data.get("title", ""))[:200]
    organization = str(data.get("organization", ""))[:200]
    style = str(data.get("style", "modern"))[:20]
    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=1.0, max_val=30.0)
    width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)

    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.lower_thirds import generate_lower_third

    result = generate_lower_third(
        name=name,
        title=title,
        organization=organization,
        style=style,
        duration=duration,
        width=width,
        height=height,
        output_path_str=out_path,
        on_progress=_p,
    )
    return result


@tools_bp.route("/lower-thirds/batch", methods=["POST"])
@require_csrf
@async_job("lower_thirds_batch", filepath_required=False)
def lower_thirds_batch(job_id, filepath, data):
    """Generate lower-thirds in batch from data source."""
    data_source = data.get("data_source")
    if data_source is None:
        raise ValueError("'data_source' is required (list of dicts or CSV file path)")

    # If data_source is a string, treat as CSV path
    if isinstance(data_source, str):
        try:
            data_source = validate_filepath(data_source)
        except ValueError as e:
            raise ValueError(f"Invalid CSV path: {e}")

    style = str(data.get("style", "modern"))[:20]
    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=1.0, max_val=30.0)
    width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)

    out_dir = data.get("output_dir", "")
    if out_dir:
        try:
            out_dir = validate_path(out_dir)
        except ValueError:
            out_dir = None
    else:
        out_dir = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.lower_thirds import batch_lower_thirds

    result = batch_lower_thirds(
        data_source=data_source,
        style=style,
        duration=duration,
        width=width,
        height=height,
        output_dir=out_dir,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Beat-Synced Cuts
# ---------------------------------------------------------------------------
@tools_bp.route("/beat-cuts/generate", methods=["POST"])
@require_csrf
@async_job("beat_cuts_generate", filepath_required=True, filepath_param="music_path")
def beat_cuts_generate(job_id, filepath, data):
    """Generate a beat-synchronized cut list from music and clips."""
    clip_paths = data.get("clip_paths", [])
    if not clip_paths or not isinstance(clip_paths, list):
        raise ValueError("'clip_paths' must be a non-empty list of video file paths")

    # Validate clip paths
    validated_clips = []
    for cp in clip_paths:
        try:
            validated_clips.append(validate_filepath(str(cp)))
        except ValueError as e:
            raise ValueError(f"Invalid clip path: {e}")

    density = str(data.get("density", "every_beat"))[:20]
    assignment = str(data.get("assignment", "round_robin"))[:20]

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.beat_cuts import generate_beat_cut_list

    result = generate_beat_cut_list(
        music_path=filepath,
        clip_paths=validated_clips,
        density=density,
        assignment=assignment,
        on_progress=_p,
    )

    # Serialize dataclass result
    return {
        "cuts": [{"clip_path": c.clip_path, "start": c.start, "duration": c.duration, "beat_time": c.beat_time} for c in result.cuts],
        "total_duration": result.total_duration,
        "density": result.density,
        "bpm": result.bpm,
        "beat_count": len(result.beats),
        "cut_count": len(result.cuts),
    }


@tools_bp.route("/beat-cuts/assemble", methods=["POST"])
@require_csrf
@async_job("beat_cuts_assemble", filepath_required=True, filepath_param="music_path")
def beat_cuts_assemble(job_id, filepath, data):
    """Assemble beat-synced clips into final video with music."""
    cut_list = data.get("cut_list", [])
    if not cut_list or not isinstance(cut_list, list):
        raise ValueError("'cut_list' must be a non-empty list of cut dicts")

    transition = str(data.get("transition", "cut"))[:20]

    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.beat_cuts import assemble_beat_synced

    result = assemble_beat_synced(
        music_path=filepath,
        cut_list=cut_list,
        output_path_str=out_path,
        transition=transition,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
@tools_bp.route("/redact/region", methods=["POST"])
@require_csrf
@async_job("redact_region")
def redact_region_route(job_id, filepath, data):
    """Apply redaction to specified regions of a video."""
    regions = data.get("regions", [])
    if not regions or not isinstance(regions, list):
        raise ValueError("'regions' must be a non-empty list of region dicts (x, y, w, h, start_time, end_time)")

    method = str(data.get("method", "blur"))[:20]
    blur_strength = safe_int(data.get("blur_strength", 20), 20, min_val=1, max_val=100)

    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.redaction import redact_region

    result = redact_region(
        input_path=filepath,
        regions=regions,
        method=method,
        blur_strength=blur_strength,
        output_path_str=out_path,
        on_progress=_p,
    )
    return result


@tools_bp.route("/redact/faces", methods=["POST"])
@require_csrf
@async_job("redact_faces")
def redact_faces_route(job_id, filepath, data):
    """Detect and redact faces in a video."""
    method = str(data.get("method", "blur"))[:20]
    blur_strength = safe_int(data.get("blur_strength", 30), 30, min_val=1, max_val=100)
    sample_interval = safe_float(data.get("sample_interval", 1.0), 1.0, min_val=0.1, max_val=10.0)

    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.redaction import redact_faces

    result = redact_faces(
        input_path=filepath,
        method=method,
        blur_strength=blur_strength,
        sample_interval=sample_interval,
        output_path_str=out_path,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Vertical Reframe
# ---------------------------------------------------------------------------
@tools_bp.route("/reframe/vertical", methods=["POST"])
@require_csrf
@async_job("reframe_vertical")
def reframe_vertical_route(job_id, filepath, data):
    """Reframe a horizontal video for vertical/portrait display."""
    target_aspect = str(data.get("target_aspect", "9:16"))[:10]
    method = str(data.get("method", "auto"))[:20]

    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.smart_reframe import reframe_vertical

    result = reframe_vertical(
        input_path=filepath,
        target_aspect=target_aspect,
        method=method,
        output_path_str=out_path,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Telemetry Overlay
# ---------------------------------------------------------------------------
@tools_bp.route("/telemetry/parse-srt", methods=["POST"])
@require_csrf
def telemetry_parse_srt():
    """Parse a DJI SRT file into telemetry data (sync)."""
    try:
        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        try:
            filepath = validate_filepath(filepath)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        from opencut.core.telemetry_overlay import parse_dji_srt

        frames = parse_dji_srt(filepath)
        return jsonify({
            "frames": [
                {
                    "timestamp": f.timestamp,
                    "latitude": f.latitude,
                    "longitude": f.longitude,
                    "altitude": f.altitude,
                    "speed": f.speed,
                    "distance": f.distance,
                    "battery": f.battery,
                    "gimbal_angle": f.gimbal_angle,
                }
                for f in frames
            ],
            "count": len(frames),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "telemetry_parse_srt")


@tools_bp.route("/telemetry/overlay", methods=["POST"])
@require_csrf
@async_job("telemetry_overlay", filepath_required=True, filepath_param="video_path")
def telemetry_overlay_route(job_id, filepath, data):
    """Overlay telemetry data onto video."""
    telemetry = data.get("telemetry", [])

    # If telemetry is not provided directly, try loading from srt_path or csv_path
    if not telemetry:
        srt_path = data.get("srt_path", "").strip()
        csv_path = data.get("csv_path", "").strip()

        if srt_path:
            try:
                srt_path = validate_filepath(srt_path)
            except ValueError as e:
                raise ValueError(f"Invalid SRT path: {e}")
            from opencut.core.telemetry_overlay import parse_dji_srt
            telem_frames = parse_dji_srt(srt_path)
            telemetry = [
                {
                    "timestamp": f.timestamp,
                    "latitude": f.latitude,
                    "longitude": f.longitude,
                    "altitude": f.altitude,
                    "speed": f.speed,
                    "distance": f.distance,
                    "battery": f.battery,
                    "gimbal_angle": f.gimbal_angle,
                }
                for f in telem_frames
            ]
        elif csv_path:
            try:
                csv_path = validate_filepath(csv_path)
            except ValueError as e:
                raise ValueError(f"Invalid CSV path: {e}")
            from opencut.core.telemetry_overlay import parse_telemetry_csv
            telem_frames = parse_telemetry_csv(csv_path)
            telemetry = [
                {
                    "timestamp": f.timestamp,
                    "latitude": f.latitude,
                    "longitude": f.longitude,
                    "altitude": f.altitude,
                    "speed": f.speed,
                    "distance": f.distance,
                    "battery": f.battery,
                    "gimbal_angle": f.gimbal_angle,
                }
                for f in telem_frames
            ]

    if not telemetry:
        raise ValueError("Telemetry data required: provide 'telemetry' list, 'srt_path', or 'csv_path'")

    fields = data.get("fields")
    position = str(data.get("position", "bottom-left"))[:20]
    font_size = safe_int(data.get("font_size", 18), 18, min_val=8, max_val=72)
    font_color = str(data.get("font_color", "white"))[:20]

    out_path = data.get("output_path")
    if out_path:
        try:
            out_path = validate_path(out_path)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.telemetry_overlay import overlay_telemetry

    result = overlay_telemetry(
        video_path=filepath,
        telemetry=telemetry,
        fields=fields,
        position=position,
        font_size=font_size,
        font_color=font_color,
        output_path_str=out_path,
        on_progress=_p,
    )
    return result
