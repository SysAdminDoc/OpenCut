"""Timecode Watermark Plugin — burns timecode overlay onto videos."""

import logging
import os

from flask import Blueprint, jsonify, request

logger = logging.getLogger("opencut")

plugin_bp = Blueprint("timecode_watermark", __name__)


@plugin_bp.route("/apply", methods=["POST"])
def apply_timecode():
    """Burn timecode overlay onto a video using FFmpeg drawtext filter."""
    data = request.get_json(force=True, silent=True) or {}
    filepath = data.get("filepath", "")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Valid filepath is required"}), 400

    # Get options
    position = data.get("position", "top-left")  # top-left, top-right, bottom-left, bottom-right
    font_size = data.get("font_size", 24)
    color = data.get("color", "white")
    start_tc = data.get("start_timecode", "00:00:00:00")

    # Build position coordinates
    positions = {
        "top-left": "x=10:y=10",
        "top-right": "x=w-tw-10:y=10",
        "bottom-left": "x=10:y=h-th-10",
        "bottom-right": "x=w-tw-10:y=h-th-10",
    }
    pos_expr = positions.get(position, positions["top-left"])

    # Build output path
    base, ext = os.path.splitext(filepath)
    output = f"{base}_tc{ext}"

    try:
        from opencut.helpers import run_ffmpeg

        drawtext = (
            f"drawtext=timecode='{start_tc}':rate=25:fontsize={font_size}"
            f":fontcolor={color}:{pos_expr}"
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
