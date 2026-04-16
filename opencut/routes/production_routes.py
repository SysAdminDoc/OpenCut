"""
OpenCut Production Routes

Credits generation, image sequence assembly, timelapse deflicker,
and audiogram generation endpoints.
"""

import logging
import os
import tempfile

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
    validate_output_path,
    validate_path,
)

logger = logging.getLogger("opencut")

production_bp = Blueprint("production", __name__)


# ---------------------------------------------------------------------------
# Credits Generation
# ---------------------------------------------------------------------------
@production_bp.route("/credits/generate", methods=["POST"])
@require_csrf
@async_job("credits_generate", filepath_required=False)
def credits_generate(job_id, filepath, data):
    """Generate a scrolling credits video from structured data."""
    credits_data = data.get("credits_data", [])
    if not credits_data or not isinstance(credits_data, list):
        raise ValueError("credits_data must be a non-empty list of section dicts")

    # Validate each section
    for i, section in enumerate(credits_data):
        if not isinstance(section, dict):
            raise ValueError(f"credits_data[{i}] must be a dict with 'section' and 'names'")
        if "names" not in section or not isinstance(section["names"], list):
            raise ValueError(f"credits_data[{i}] must have a 'names' list")

    width = safe_int(data.get("width", 1920), 1920, min_val=320, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
    fps = safe_int(data.get("fps", 24), 24, min_val=1, max_val=120)
    scroll_speed = safe_int(data.get("scroll_speed", 60), 60, min_val=10, max_val=500)
    font_size = safe_int(data.get("font_size", 36), 36, min_val=8, max_val=200)
    font_color = str(data.get("font_color", "white"))[:20]
    bg_color = str(data.get("bg_color", "black"))[:20]
    font_path = data.get("font_path")

    # Validate font path if provided
    if font_path:
        try:
            font_path = validate_filepath(font_path)
        except ValueError:
            font_path = None

    # Output path
    out_dir = data.get("output_dir", "")
    if out_dir:
        try:
            out_dir = validate_path(out_dir)
        except ValueError:
            out_dir = tempfile.gettempdir()
    else:
        out_dir = tempfile.gettempdir()

    out_path = os.path.join(out_dir, "credits.mp4")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.credits_gen import generate_credits
    result = generate_credits(
        credits_data=credits_data,
        output_path_str=out_path,
        width=width,
        height=height,
        fps=fps,
        scroll_speed=scroll_speed,
        font_size=font_size,
        font_color=font_color,
        bg_color=bg_color,
        font_path=font_path,
        on_progress=_p,
    )
    return result


@production_bp.route("/credits/parse", methods=["POST"])
@require_csrf
def credits_parse():
    """Parse a credits text file into structured data (sync)."""
    try:
        data = request.get_json(force=True) or {}
        filepath = data.get("filepath", "").strip()
        if not filepath:
            return jsonify({"error": "No file path provided"}), 400
        try:
            filepath = validate_filepath(filepath)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        from opencut.core.credits_gen import parse_credits_file
        sections = parse_credits_file(filepath)
        return jsonify({"sections": sections, "count": len(sections)})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "credits_parse")


# ---------------------------------------------------------------------------
# Image Sequence
# ---------------------------------------------------------------------------
@production_bp.route("/image-sequence/detect", methods=["POST"])
@require_csrf
def image_sequence_detect():
    """Detect an image sequence in a folder (sync)."""
    try:
        data = request.get_json(force=True) or {}
        folder = data.get("folder_path", "").strip()
        if not folder:
            return jsonify({"error": "No folder_path provided"}), 400
        try:
            folder = validate_path(folder)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        from opencut.core.image_sequence import detect_image_sequence
        info = detect_image_sequence(folder)
        return jsonify({
            "pattern": info.pattern,
            "first_frame": info.first_frame,
            "last_frame": info.last_frame,
            "total_frames": info.total_frames,
            "extension": info.extension,
            "detected_pattern_str": info.detected_pattern_str,
            "gaps": info.gaps,
            "folder": info.folder,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return safe_error(e, "image_sequence_detect")


@production_bp.route("/image-sequence/assemble", methods=["POST"])
@require_csrf
@async_job("image_sequence_assemble", filepath_required=False)
def image_sequence_assemble(job_id, filepath, data):
    """Assemble an image sequence into a video file."""
    folder = data.get("folder_path", "").strip()
    if not folder:
        raise ValueError("No folder_path provided")
    try:
        folder = validate_path(folder)
    except ValueError as e:
        raise ValueError(str(e))

    fps = safe_int(data.get("fps", 24), 24, min_val=1, max_val=240)
    pattern = data.get("pattern")
    start_frame = data.get("start_frame")
    end_frame = data.get("end_frame")
    codec = str(data.get("codec", "prores_ks"))[:20]
    quality = str(data.get("quality", "high"))[:10]

    if start_frame is not None:
        start_frame = safe_int(start_frame, 0, min_val=0, max_val=999999)
    if end_frame is not None:
        end_frame = safe_int(end_frame, 999, min_val=0, max_val=999999)

    out_path = data.get("output_path")
    if out_path:
        out_path = validate_output_path(out_path)
    if out_path:
        try:
            out_dir = os.path.dirname(out_path)
            validate_path(out_dir)
        except ValueError:
            out_path = None

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.image_sequence import assemble_image_sequence
    result = assemble_image_sequence(
        folder_path=folder,
        output_path_str=out_path,
        fps=fps,
        pattern=pattern,
        start_frame=start_frame,
        end_frame=end_frame,
        codec=codec,
        quality=quality,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Timelapse Deflicker
# ---------------------------------------------------------------------------
@production_bp.route("/timelapse/analyze-flicker", methods=["POST"])
@require_csrf
@async_job("timelapse_analyze")
def timelapse_analyze_flicker(job_id, filepath, data):
    """Analyze flicker in timelapse footage."""
    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.timelapse import analyze_flicker
    analysis = analyze_flicker(filepath, on_progress=_p)
    return {
        "flicker_score": analysis.flicker_score,
        "needs_deflicker": analysis.needs_deflicker,
        "frame_count": analysis.frame_count,
        "avg_luminance": analysis.avg_luminance,
        "min_luminance": analysis.min_luminance,
        "max_luminance": analysis.max_luminance,
        "std_dev": analysis.std_dev,
        # Don't return full per_frame_luminance in response (too large)
        "sample_luminance": analysis.per_frame_luminance[:100],
    }


@production_bp.route("/timelapse/deflicker", methods=["POST"])
@require_csrf
@async_job("timelapse_deflicker")
def timelapse_deflicker(job_id, filepath, data):
    """Remove flicker from timelapse footage."""
    window_size = safe_int(data.get("window_size", 15), 15, min_val=3, max_val=99)
    strength = safe_float(data.get("strength", 0.8), 0.8, min_val=0.0, max_val=1.0)
    method = str(data.get("method", "auto"))[:10]
    if method not in ("auto", "simple", "smooth"):
        method = "auto"

    out_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    out_path = data.get("output_path")
    if out_path:
        out_path = validate_output_path(out_path)
    if not out_path:
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(out_dir, f"{base}_deflickered.mp4")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.timelapse import deflicker
    result = deflicker(
        input_path=filepath,
        output_path_str=out_path,
        window_size=window_size,
        strength=strength,
        method=method,
        on_progress=_p,
    )
    return result


# ---------------------------------------------------------------------------
# Audiogram Generator
# ---------------------------------------------------------------------------
@production_bp.route("/audiogram/generate", methods=["POST"])
@require_csrf
@async_job("audiogram_generate", filepath_param="audio_path")
def audiogram_generate(job_id, filepath, data):
    """Generate an audiogram video from an audio file."""
    style = str(data.get("style", "bars"))[:20]
    width = safe_int(data.get("width", 1080), 1080, min_val=320, max_val=7680)
    height = safe_int(data.get("height", 1080), 1080, min_val=240, max_val=4320)
    bg_color = str(data.get("bg_color", "#1a1a2e"))[:20]
    wave_color = str(data.get("wave_color", "#e94560"))[:20]
    duration = data.get("duration")
    if duration is not None:
        duration = safe_float(duration, None, min_val=0.1, max_val=7200)
    title_text = data.get("title_text")
    if title_text:
        title_text = str(title_text)[:200]

    artwork_path = data.get("artwork_path")
    if artwork_path:
        try:
            artwork_path = validate_filepath(artwork_path)
        except ValueError:
            artwork_path = None

    caption_srt = data.get("caption_srt")
    if caption_srt:
        try:
            caption_srt = validate_filepath(caption_srt)
        except ValueError:
            caption_srt = None

    out_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(out_dir, f"{base}_audiogram.mp4")

    def _p(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    from opencut.core.audiogram import generate_audiogram
    result = generate_audiogram(
        audio_path=filepath,
        output_path_str=out_path,
        style=style,
        width=width,
        height=height,
        bg_color=bg_color,
        wave_color=wave_color,
        duration=duration,
        artwork_path=artwork_path,
        title_text=title_text,
        caption_srt=caption_srt,
        on_progress=_p,
    )
    return result
