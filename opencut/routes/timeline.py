"""
OpenCut Timeline Routes

Marker-based export, batch rename, smart bins, SRT-to-captions, index status.
"""

import logging
import os
import subprocess as _sp

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import get_ffmpeg_path
from opencut.jobs import (
    _is_cancelled,
    _update_job,
    async_job,
    make_install_route,
)
from opencut.security import (
    require_csrf,
    safe_float,
    validate_filepath,
    validate_path,
)

logger = logging.getLogger("opencut")

timeline_bp = Blueprint("timeline", __name__)

# Valid smart-bin rule types and fields
_VALID_RULE_TYPES = {"contains", "starts_with", "ends_with", "equals", "regex"}
_VALID_RULE_FIELDS = {"name", "label", "comment", "media_type", "file_path", "duration"}

make_install_route(
    timeline_bp,
    "/timeline/otio/install",
    "otio",
    ["opentimelineio"],
    doc="Install OpenTimelineIO dependencies.",
)


# ---------------------------------------------------------------------------
# Timeline: Export Clips from Markers
# ---------------------------------------------------------------------------
def _validate_timeline_export(data):
    """Sync validation: at least an input file (under either key) is required."""
    input_file = (data.get("input_file") or data.get("filepath") or "").strip()
    if not input_file:
        return "No input_file provided"
    return None


@timeline_bp.route("/timeline/export-from-markers", methods=["POST"])
@require_csrf
@async_job("timeline-export", filepath_required=False, pre_validate=_validate_timeline_export)
def timeline_export_from_markers(job_id, filepath, data):
    """Use FFmpeg to extract clip segments defined by timeline markers."""
    input_file = data.get("input_file", data.get("filepath", "")).strip()
    if not input_file:
        raise ValueError("No input_file provided")
    input_file = validate_filepath(input_file)

    markers = data.get("markers", [])
    output_dir = data.get("output_dir", "").strip()
    if output_dir:
        output_dir = validate_path(output_dir)
    fmt = data.get("format", "mp4").strip().lower()

    if not isinstance(markers, list) or not markers:
        raise ValueError("No markers provided")

    if len(markers) > 500:
        raise ValueError("Too many markers (max 500)")

    if fmt not in {"mp4", "mov", "mkv", "mxf"}:
        fmt = "mp4"

    # Resolve output directory
    if output_dir:
        output_dir = validate_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(input_file)

    outputs = []
    valid_markers = [
        m for m in markers
        if isinstance(m, dict) and safe_float(m.get("duration", 0), 0) > 0
    ]
    total = len(valid_markers)
    if total == 0:
        return {"outputs": [], "count": 0}

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for idx, marker in enumerate(valid_markers):
        if _is_cancelled(job_id):
            return {"outputs": outputs, "count": len(outputs)}

        pct = int((idx / total) * 90)
        name = str(marker.get("name", f"marker_{idx}"))[:100]
        start = safe_float(marker.get("time", 0), 0, min_val=0.0)
        duration = safe_float(marker.get("duration", 0), 0, min_val=0.0)

        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        out_path = os.path.join(output_dir, f"{base_name}_{safe_name}_{idx}.{fmt}")

        _update_job(job_id, progress=pct, message=f"Extracting marker '{name}' ({idx + 1}/{total})...")

        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", input_file,
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            out_path,
        ]
        try:
            result = _sp.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                outputs.append({"marker": name, "path": out_path})
            else:
                logger.warning("FFmpeg failed for marker '%s': %s", name, result.stderr.decode(errors="replace")[:200])
        except Exception as ffmpeg_exc:
            logger.warning("FFmpeg error for marker '%s': %s", name, ffmpeg_exc)

    return {"outputs": outputs, "count": len(outputs)}


# ---------------------------------------------------------------------------
# Timeline: Batch Rename (validation only — execution via ExtendScript)
# ---------------------------------------------------------------------------
@timeline_bp.route("/timeline/batch-rename", methods=["POST"])
@require_csrf
def timeline_batch_rename():
    """Validate a list of rename operations; returns validated/invalid sets for ExtendScript."""
    data = request.get_json(force=True)
    renames = data.get("renames", [])

    if not isinstance(renames, list):
        return jsonify({"error": "renames must be a list"}), 400

    if len(renames) > 1000:
        return jsonify({"error": "Too many renames (max 1000)"}), 400

    validated = []
    invalid = []

    for item in renames:
        if not isinstance(item, dict):
            invalid.append({"item": item, "reason": "not an object"})
            continue

        node_id = str(item.get("nodeId", "")).strip()
        current_name = str(item.get("currentName", "")).strip()
        new_name = str(item.get("newName", "")).strip()

        # Validate: newName must not be empty
        if not new_name:
            invalid.append({"nodeId": node_id, "currentName": current_name, "newName": new_name, "reason": "newName is empty"})
            continue

        # Validate: must not contain path separators or null bytes
        if "/" in new_name or "\\" in new_name or "\x00" in new_name:
            invalid.append({"nodeId": node_id, "currentName": current_name, "newName": new_name, "reason": "newName contains invalid characters"})
            continue

        validated.append({"nodeId": node_id, "currentName": current_name, "newName": new_name})

    return jsonify({"validated_renames": validated, "invalid": invalid})


# ---------------------------------------------------------------------------
# Timeline: Smart Bins (validation only — execution via ExtendScript)
# ---------------------------------------------------------------------------
@timeline_bp.route("/timeline/smart-bins", methods=["POST"])
@require_csrf
def timeline_smart_bins():
    """Validate smart-bin rules; returns validated/invalid sets for ExtendScript."""
    data = request.get_json(force=True)
    rules = data.get("rules", [])

    if not isinstance(rules, list):
        return jsonify({"error": "rules must be a list"}), 400

    if len(rules) > 200:
        return jsonify({"error": "Too many rules (max 200)"}), 400

    validated = []
    invalid = []

    for item in rules:
        if not isinstance(item, dict):
            invalid.append({"item": item, "reason": "not an object"})
            continue

        bin_name = str(item.get("binName", item.get("bin_name", ""))).strip()
        rule_type = str(item.get("rule", item.get("rule_type", ""))).strip()
        field = str(item.get("field", "")).strip()
        value = str(item.get("value", "")).strip()

        reasons = []
        if not bin_name:
            reasons.append("binName is empty")
        if rule_type not in _VALID_RULE_TYPES:
            reasons.append(f"rule '{rule_type}' not in {sorted(_VALID_RULE_TYPES)}")
        if field not in _VALID_RULE_FIELDS:
            reasons.append(f"field '{field}' not in {sorted(_VALID_RULE_FIELDS)}")
        if not value:
            reasons.append("value is empty")

        if reasons:
            invalid.append({"binName": bin_name, "rule": rule_type, "field": field, "value": value, "reason": "; ".join(reasons)})
        else:
            validated.append({"binName": bin_name, "rule": rule_type, "field": field, "value": value})

    return jsonify({"validated_rules": validated, "invalid": invalid})


# ---------------------------------------------------------------------------
# Timeline: SRT to Captions Segments
# ---------------------------------------------------------------------------
@timeline_bp.route("/timeline/srt-to-captions", methods=["POST"])
@require_csrf
def timeline_srt_to_captions():
    """Parse an SRT file into caption segments, or pass through provided segments."""
    data = request.get_json(force=True)
    srt_path = data.get("srt_path", "").strip()
    segments = data.get("segments", None)

    # If segments are already provided, pass through
    if segments is not None:
        if not isinstance(segments, list):
            return jsonify({"error": "segments must be a list"}), 400
        cleaned = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            cleaned.append({
                "start": safe_float(seg.get("start", 0), 0.0, min_val=0.0),
                "end": safe_float(seg.get("end", 0), 0.0, min_val=0.0),
                "text": str(seg.get("text", "")).strip(),
            })
        return jsonify({"segments": cleaned, "count": len(cleaned)})

    # Parse SRT file
    if not srt_path:
        return jsonify({"error": "srt_path or segments required"}), 400

    try:
        srt_path = validate_filepath(srt_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        parsed = _parse_srt(srt_path)
        return jsonify({"segments": parsed, "count": len(parsed)})
    except Exception as exc:
        return safe_error(exc, "timeline_srt_to_captions")


def _parse_srt(path: str) -> list:
    """Parse an SRT subtitle file into a list of segment dicts."""
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    segments = []
    import re as _re
    # Each SRT block: index, timecode line, text lines, blank line
    blocks = _re.split(r"\n\s*\n", content.strip())
    _time_re = _re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")

    def _tc_to_sec(tc: str) -> float:
        tc = tc.replace(",", ".")
        parts = tc.split(":")
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find the timecode line (may have index number before it)
        tc_line_idx = None
        for i, line in enumerate(lines):
            if _time_re.match(line.strip()):
                tc_line_idx = i
                break
        if tc_line_idx is None:
            continue
        m = _time_re.match(lines[tc_line_idx].strip())
        start = _tc_to_sec(m.group(1))
        end = _tc_to_sec(m.group(2))
        text = " ".join(lines[tc_line_idx + 1:]).strip()
        # Strip HTML-like tags common in SRT
        text = _re.sub(r"<[^>]+>", "", text)
        if text:
            segments.append({"start": start, "end": end, "text": text})

    return segments


# ---------------------------------------------------------------------------
# Timeline: Footage Index Status
# ---------------------------------------------------------------------------
@timeline_bp.route("/timeline/index-status", methods=["GET"])
def timeline_index_status():
    """Return footage search index statistics."""
    try:
        from opencut.core import footage_search
        stats = footage_search.get_index_stats()
        return jsonify({
            "total_files": stats.get("total_files", 0),
            "total_segments": stats.get("total_segments", 0),
            "index_size_bytes": stats.get("index_size_bytes", 0),
        })
    except ImportError:
        return jsonify({"total_files": 0, "total_segments": 0, "index_size_bytes": 0})
    except Exception as exc:
        return safe_error(exc, "timeline_index_status")


# ---------------------------------------------------------------------------
# Timeline: OTIO Export (Universal Timeline Interchange)
# ---------------------------------------------------------------------------
@timeline_bp.route("/timeline/export-otio", methods=["POST"])
@require_csrf
def timeline_export_otio():
    """Export timeline edits as OpenTimelineIO file (universal NLE format).

    Supports three modes:
    - "cuts": Provide cut regions to remove, exports kept segments
    - "segments": Provide speech/kept segments directly
    - "markers": Export markers as OTIO markers on a full clip

    Body JSON:
        filepath (str): Source media path
        mode (str): "cuts", "segments", or "markers"
        cuts (list): [{start, end}, ...] — regions to remove (mode=cuts)
        segments (list): [{start, end}, ...] — regions to keep (mode=segments)
        markers (list): [{time, name, color}, ...] — marker list (mode=markers)
        output_dir (str): Output directory
        sequence_name (str): Name for the timeline
    """
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    mode = data.get("mode", "cuts")
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    sequence_name = data.get("sequence_name", "OpenCut Edit")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if output_dir:
        try:
            output_dir = validate_path(output_dir)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    try:
        from opencut.export.otio_export import (
            check_otio_available,
            export_otio,
            export_otio_from_cuts,
            export_otio_markers,
        )

        if not check_otio_available():
            return jsonify({
                "error": "OpenTimelineIO not installed.",
                "suggestion": "Install with: pip install opentimelineio",
            }), 400

        # Determine output path
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        effective_dir = output_dir or os.path.dirname(filepath)
        otio_path = os.path.join(effective_dir, f"{base_name}_opencut.otio")

        # Probe for framerate
        from opencut.utils.media import probe
        info = probe(filepath)
        fps = info.fps if hasattr(info, "fps") and info.fps > 0 else 24.0
        total_duration = info.duration if info.duration > 0 else 0.0

        if mode == "markers":
            markers = data.get("markers", [])
            if not markers:
                return jsonify({"error": "No markers provided"}), 400
            result_path = export_otio_markers(
                filepath, markers, otio_path,
                sequence_name=sequence_name,
                framerate=fps,
                total_duration=total_duration,
            )
        elif mode == "segments":
            segments_data = data.get("segments", [])
            if not segments_data:
                return jsonify({"error": "No segments provided"}), 400
            from opencut.core.silence import TimeSegment
            segments = [
                TimeSegment(start=safe_float(s.get("start", 0), 0.0, min_val=0.0), end=safe_float(s.get("end", 0), 0.0, min_val=0.0), label="speech")
                for s in segments_data
            ]
            result_path = export_otio(
                filepath, segments, otio_path,
                sequence_name=sequence_name,
                framerate=fps,
            )
        else:
            # mode == "cuts" (default)
            cuts = data.get("cuts", [])
            if not cuts:
                return jsonify({"error": "No cuts provided"}), 400
            result_path = export_otio_from_cuts(
                filepath, cuts, otio_path,
                sequence_name=sequence_name,
                framerate=fps,
                total_duration=total_duration,
            )

        return jsonify({
            "output_path": result_path,
            "format": "otio",
            "message": f"Exported OTIO timeline: {os.path.basename(result_path)}",
        })

    except ImportError as e:
        return jsonify({
            "error": f"OpenTimelineIO not available: {e}",
            "suggestion": "Install with: pip install opentimelineio",
        }), 400
    except Exception as exc:
        return safe_error(exc, "timeline_export_otio")
