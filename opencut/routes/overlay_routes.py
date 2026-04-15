"""
OpenCut Overlay Routes

Platform safe zone overlay, timecode burn-in, countdown/elapsed timer.
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
    validate_output_path,
)

logger = logging.getLogger("opencut")

overlay_bp = Blueprint("overlay", __name__)


# ---------------------------------------------------------------------------
# Feature 36.1 - Safe Zone Overlay (async)
# ---------------------------------------------------------------------------
@overlay_bp.route("/overlay/safe-zones", methods=["POST"])
@require_csrf
@async_job("safe_zone")
def overlay_safe_zones(job_id, filepath, data):
    """Burn platform safe zone rectangles onto a video copy."""
    platform = (data.get("platform") or "").strip().lower()
    if not platform:
        raise ValueError("Missing required field: platform")
    opacity = safe_float(data.get("opacity", 0.3), 0.3, min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")

    from opencut.core.safe_zones import generate_safe_zone_overlay
    from opencut.helpers import output_path as _output_path

    out = data.get("output_path") or _output_path(filepath, f"safezone_{platform}", output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_safe_zone_overlay(
        input_path=filepath,
        platform=platform,
        out_path=out,
        opacity=opacity,
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Feature 36.1 - Safe Zone Data (sync, no job)
# ---------------------------------------------------------------------------
@overlay_bp.route("/overlay/safe-zones/data", methods=["POST"])
@require_csrf
def overlay_safe_zones_data():
    """Return safe zone coordinates for a platform and resolution (no FFmpeg)."""
    data = request.get_json(force=True) or {}
    platform = (data.get("platform") or "").strip().lower()
    if not platform:
        return jsonify({"error": "Missing required field: platform"}), 400

    # Width/height can come from a filepath probe or be passed directly
    filepath = (data.get("filepath") or "").strip()
    width = safe_int(data.get("width", 0), 0, min_val=0)
    height = safe_int(data.get("height", 0), 0, min_val=0)

    if filepath and (width == 0 or height == 0):
        try:
            filepath = validate_filepath(filepath)
            from opencut.helpers import get_video_info
            info = get_video_info(filepath)
            width = width or info["width"]
            height = height or info["height"]
        except (ValueError, Exception) as e:
            return jsonify({"error": f"Could not read video info: {e}"}), 400

    if width <= 0 or height <= 0:
        return jsonify({
            "error": "Provide width/height or a valid filepath to probe resolution.",
        }), 400

    try:
        from opencut.core.safe_zones import SUPPORTED_PLATFORMS, get_safe_zones
        zones = get_safe_zones(platform, width, height)
        return jsonify({
            "platform": platform,
            "width": width,
            "height": height,
            "zones": [
                {
                    "x": z.x,
                    "y": z.y,
                    "w": z.w,
                    "h": z.h,
                    "label": z.label,
                    "color": z.color,
                }
                for z in zones
            ],
            "supported_platforms": SUPPORTED_PLATFORMS,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        return safe_error(exc, "overlay_safe_zones_data")


# ---------------------------------------------------------------------------
# Feature 44.1 - Timecode Burn-In (async)
# ---------------------------------------------------------------------------
@overlay_bp.route("/overlay/timecode", methods=["POST"])
@require_csrf
@async_job("timecode")
def overlay_timecode(job_id, filepath, data):
    """Burn timecode (HH:MM:SS:FF) onto a video."""
    position = (data.get("position") or "top-left").strip()
    font_size = safe_int(data.get("font_size", 24), 24, min_val=8, max_val=200)
    color = (data.get("color") or "white").strip()
    bg_color = (data.get("bg_color") or "black@0.5").strip()
    start_tc = (data.get("start_tc") or "00:00:00:00").strip()

    from opencut.core.overlays import burn_timecode

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = burn_timecode(
        input_path=filepath,
        output_path_override=data.get("output_path"),
        position=position,
        font_size=font_size,
        color=color,
        bg_color=bg_color,
        start_tc=start_tc,
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Feature 34.4 - Countdown Timer (async)
# ---------------------------------------------------------------------------
@overlay_bp.route("/overlay/countdown", methods=["POST"])
@require_csrf
@async_job("countdown")
def overlay_countdown(job_id, filepath, data):
    """Burn a countdown timer onto a video."""
    duration_seconds = data.get("duration_seconds")
    if duration_seconds is not None:
        duration_seconds = safe_float(duration_seconds, None, min_val=0.0)
    position = (data.get("position") or "center").strip()
    font_size = safe_int(data.get("font_size", 48), 48, min_val=8, max_val=300)
    color = (data.get("color") or "white").strip()
    bg_color = (data.get("bg_color") or "black@0.7").strip()
    timer_format = (data.get("format") or "MM:SS").strip()

    from opencut.core.overlays import burn_countdown

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = burn_countdown(
        input_path=filepath,
        output_path_override=data.get("output_path"),
        duration_seconds=duration_seconds,
        position=position,
        font_size=font_size,
        color=color,
        bg_color=bg_color,
        timer_format=timer_format,
        on_progress=_progress,
    )
    return result


# ---------------------------------------------------------------------------
# Feature 34.4 - Elapsed Timer (async)
# ---------------------------------------------------------------------------
@overlay_bp.route("/overlay/elapsed-timer", methods=["POST"])
@require_csrf
@async_job("elapsed_timer")
def overlay_elapsed_timer(job_id, filepath, data):
    """Burn an elapsed (count-up) timer onto a video."""
    start_seconds = safe_float(data.get("start_seconds", 0), 0.0, min_val=0.0)
    position = (data.get("position") or "bottom-right").strip()
    font_size = safe_int(data.get("font_size", 24), 24, min_val=8, max_val=200)
    color = (data.get("color") or "white").strip()

    from opencut.core.overlays import burn_elapsed_timer

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = burn_elapsed_timer(
        input_path=filepath,
        output_path_override=data.get("output_path"),
        start_seconds=start_seconds,
        position=position,
        font_size=font_size,
        color=color,
        on_progress=_progress,
    )
    return result
