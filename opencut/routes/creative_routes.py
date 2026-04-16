"""
OpenCut Creative Routes

Video comparison, retro/analog effects, tilt-shift, light leaks,
color blindness simulation, and photosensitive flash detection.
"""

import logging

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

creative_bp = Blueprint("creative", __name__)


# ---------------------------------------------------------------------------
# Video Comparison / Diff — async (full video)
# ---------------------------------------------------------------------------
@creative_bp.route("/video/compare", methods=["POST"])
@require_csrf
@async_job("video_compare")
def video_compare(job_id, filepath, data):
    """Compare two videos using a visual diff mode."""
    from opencut.core.video_compare import compare_videos

    input_b = data.get("input_b", "").strip()
    if not input_b:
        raise ValueError("No second input file (input_b) provided")
    input_b = validate_filepath(input_b)

    mode = data.get("mode", "sidebyside").strip()
    timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"compare_{mode}", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = compare_videos(
        input_a=filepath,
        input_b=input_b,
        mode=mode,
        output_path=output,
        timestamp=timestamp,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Video Comparison / Diff — sync (single frame)
# ---------------------------------------------------------------------------
@creative_bp.route("/video/compare-frame", methods=["POST"])
@require_csrf
def video_compare_frame():
    """Compare a single frame from two videos (synchronous)."""
    try:
        from opencut.core.video_compare import compare_frames

        data = request.get_json(force=True) or {}
        input_a = data.get("filepath", "").strip()
        input_b = data.get("input_b", "").strip()

        if not input_a:
            return jsonify({"error": "No file path provided"}), 400
        if not input_b:
            return jsonify({"error": "No second input file (input_b) provided"}), 400

        input_a = validate_filepath(input_a)
        input_b = validate_filepath(input_b)

        mode = data.get("mode", "sidebyside").strip()
        timestamp = safe_float(data.get("timestamp", 0), 0, min_val=0.0)

        result = compare_frames(
            input_a=input_a,
            input_b=input_b,
            timestamp=timestamp,
            mode=mode,
        )

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return safe_error(e, "video_compare_frame")


# ---------------------------------------------------------------------------
# Retro / Analog Effects — async
# ---------------------------------------------------------------------------
@creative_bp.route("/effects/retro", methods=["POST"])
@require_csrf
@async_job("retro_effect")
def effects_retro(job_id, filepath, data):
    """Apply a retro / analog video effect."""
    from opencut.core.retro_effects import apply_retro_effect

    effect = data.get("effect", "vhs").strip()
    intensity = safe_float(data.get("intensity", 0.7), 0.7, min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"retro_{effect}", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_retro_effect(
        input_path=filepath,
        effect=effect,
        intensity=intensity,
        output_path=output,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Tilt-Shift Miniature — async
# ---------------------------------------------------------------------------
@creative_bp.route("/effects/tilt-shift", methods=["POST"])
@require_csrf
@async_job("tilt_shift")
def effects_tilt_shift(job_id, filepath, data):
    """Apply a tilt-shift miniature effect."""
    from opencut.core.retro_effects import apply_tilt_shift

    focus_y = safe_float(data.get("focus_y", 0.5), 0.5, min_val=0.0, max_val=1.0)
    focus_width = safe_float(data.get("focus_width", 0.3), 0.3, min_val=0.05, max_val=1.0)
    blur_amount = safe_int(data.get("blur_amount", 10), 10, min_val=1, max_val=30)
    saturation = safe_float(data.get("saturation", 1.5), 1.5, min_val=0.5, max_val=3.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, "tiltshift", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_tilt_shift(
        input_path=filepath,
        focus_y=focus_y,
        focus_width=focus_width,
        blur_amount=blur_amount,
        saturation=saturation,
        output_path=output,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Light Leaks & Lens Flares — async
# ---------------------------------------------------------------------------
@creative_bp.route("/effects/light-leak", methods=["POST"])
@require_csrf
@async_job("light_leak")
def effects_light_leak(job_id, filepath, data):
    """Apply a procedural light leak / lens flare overlay."""
    from opencut.core.retro_effects import apply_light_leak

    style = data.get("style", "warm_amber").strip()
    intensity = safe_float(data.get("intensity", 0.5), 0.5, min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"lightleak_{style}", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_light_leak(
        input_path=filepath,
        style=style,
        intensity=intensity,
        output_path=output,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Color Blindness Simulation — async
# ---------------------------------------------------------------------------
@creative_bp.route("/accessibility/colorblind-sim", methods=["POST"])
@require_csrf
@async_job("colorblind_sim")
def accessibility_colorblind_sim(job_id, filepath, data):
    """Simulate a color vision deficiency on the input video."""
    from opencut.core.accessibility import simulate_color_blindness

    condition = data.get("condition", "deuteranopia").strip()
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    if output is None and output_dir:
        effective_dir = _resolve_output_dir(filepath, output_dir)
        from opencut.helpers import output_path as _op
        output = _op(filepath, f"cb_{condition}", effective_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = simulate_color_blindness(
        input_path=filepath,
        condition=condition,
        output_path=output,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Photosensitive Flash Detection — async
# ---------------------------------------------------------------------------
@creative_bp.route("/accessibility/flash-detect", methods=["POST"])
@require_csrf
@async_job("flash_detect")
def accessibility_flash_detect(job_id, filepath, data):
    """Detect potentially seizure-inducing flash sequences."""
    from opencut.core.accessibility import detect_flashing

    max_flashes = safe_int(data.get("max_flashes_per_sec", 3), 3, min_val=1, max_val=30)
    min_change = safe_float(data.get("min_luminance_change", 0.2), 0.2, min_val=0.01, max_val=1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_flashing(
        input_path=filepath,
        max_flashes_per_sec=max_flashes,
        min_luminance_change=min_change,
        on_progress=_progress,
    )

    return result
