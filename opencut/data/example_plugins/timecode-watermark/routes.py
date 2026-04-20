"""Timecode Watermark Plugin — burns timecode overlay onto videos."""

import logging
import re

from flask import Blueprint, jsonify, request

logger = logging.getLogger("opencut")

plugin_bp = Blueprint("timecode_watermark", __name__)

_POSITIONS = {
    "top-left": "x=10:y=10",
    "top-right": "x=w-tw-10:y=10",
    "bottom-left": "x=10:y=h-th-10",
    "bottom-right": "x=w-tw-10:y=h-th-10",
}
_COLOR_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9_]*(?:@[01](?:\.\d{1,3})?)?|#?[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?)$"
)
_TIMECODE_RE = re.compile(r"^\d{2,3}:\d{2}:\d{2}:\d{2}$")


def _get_json_object():
    data = request.get_json(force=True, silent=True)
    return data if isinstance(data, dict) else {}


def _validate_timecode(value):
    value = str(value or "00:00:00:00").strip()
    if not _TIMECODE_RE.match(value):
        raise ValueError("start_timecode must use HH:MM:SS:FF")
    _hours, minutes, seconds, frames = (int(part) for part in value.split(":"))
    if minutes > 59 or seconds > 59 or frames > 24:
        raise ValueError("start_timecode contains an out-of-range component")
    return value


def _validate_color(value):
    value = str(value or "white").strip()
    if not _COLOR_RE.match(value):
        raise ValueError("color must be a simple FFmpeg color name or hex value")
    return value


@plugin_bp.route("/apply", methods=["POST"])
def apply_timecode():
    """Burn timecode overlay onto a video using FFmpeg drawtext filter."""
    data = _get_json_object()
    filepath = data.get("filepath", "")

    try:
        from opencut.helpers import output_path
        from opencut.security import safe_int, validate_filepath, validate_output_path

        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Get options
    position = str(data.get("position", "top-left")).strip().lower()
    if position not in _POSITIONS:
        return jsonify({"error": f"position must be one of: {', '.join(sorted(_POSITIONS))}"}), 400

    try:
        font_size = safe_int(data.get("font_size", 24), default=24, min_val=8, max_val=200)
        color = _validate_color(data.get("color", "white"))
        start_tc = _validate_timecode(data.get("start_timecode", "00:00:00:00"))
        output = validate_output_path(output_path(filepath, "tc"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        from opencut.helpers import run_ffmpeg

        drawtext = (
            f"drawtext=timecode='{start_tc}':rate=25:fontsize={font_size}"
            f":fontcolor={color}:{_POSITIONS[position]}"
        )

        cmd = [
            "ffmpeg", "-y", "-i", filepath,
            "-vf", drawtext,
            "-c:a", "copy",
            output,
        ]
        run_ffmpeg(cmd)

        return jsonify({
            "success": True,
            "output": output,
            "message": "Timecode watermark applied",
        })
    except Exception as e:
        logger.error("Timecode watermark failed: %s", e)
        return jsonify({"error": str(e)}), 500
