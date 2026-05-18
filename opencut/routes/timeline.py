"""
OpenCut Timeline Routes

Marker-based export, batch rename, smart bins, SRT-to-captions, index status.
"""

import logging
import os
import subprocess as _sp

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.helpers import get_ffmpeg_path
from opencut.jobs import (
    _is_cancelled,
    _update_job,
    async_job,
    make_install_route,
)
from opencut.security import (
    get_json_dict,
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
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

    # Resolve output directory — already validated above when present.
    if output_dir:
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

        # Sanitise marker name for filesystem use: keep alphanumerics and a
        # few punctuation characters, collapse runs of ``_``, strip leading
        # or trailing dots/spaces (Windows silently strips trailing spaces
        # on disk so returned path disagreed with what ended up written).
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        safe_name = "_".join(p for p in safe_name.split("_") if p)
        safe_name = safe_name.strip(" .") or "marker"
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
# Characters Premiere/NTFS reject in item names. We block them up front
# so the ExtendScript side never has to surface obscure Premiere errors.
_INVALID_NAME_CHARS = set('/\\\x00<>:"|?*')


@timeline_bp.route("/timeline/batch-rename", methods=["POST"])
@require_csrf
def timeline_batch_rename():
    """Validate a list of rename operations; returns validated/invalid sets for ExtendScript."""
    try:
        data = get_json_dict()
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
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

        # Validate: no path separators, null bytes, NTFS-forbidden chars, or
        # ASCII control bytes (which break logs and some NLE UIs).
        bad_chars = _INVALID_NAME_CHARS & set(new_name)
        has_control = any(ord(ch) < 0x20 for ch in new_name)
        if bad_chars or has_control:
            invalid.append({
                "nodeId": node_id,
                "currentName": current_name,
                "newName": new_name,
                "reason": "newName contains invalid characters",
            })
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
    try:
        data = get_json_dict()
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
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
    try:
        data = get_json_dict()
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    srt_path = str(data.get("srt_path", "")).strip()
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
    _max_srt_bytes = 16 * 1024 * 1024  # 16 MB
    try:
        if os.path.getsize(path) > _max_srt_bytes:
            raise ValueError(f"SRT file exceeds {_max_srt_bytes} byte cap: {path}")
    except OSError as exc:
        raise ValueError(f"Cannot stat SRT file: {exc}") from exc
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read(_max_srt_bytes + 1)
    if len(content) > _max_srt_bytes:
        raise ValueError(f"SRT file exceeds {_max_srt_bytes} byte cap: {path}")

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
    try:
        data = get_json_dict()
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_INPUT"}), 400
    filepath = str(data.get("filepath", "")).strip()
    mode = data.get("mode", "cuts")
    output_dir = str(data.get("output_dir", "")).strip()
    sequence_name = data.get("sequence_name", "OpenCut Edit")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Single validation pass — the previous version validated twice; the
    # first call could crash the worker with a 500 on bad input because it
    # wasn't wrapped.
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
            if not isinstance(markers, list) or not markers:
                return jsonify({"error": "No markers provided (must be a non-empty list)"}), 400
            if len(markers) > 5000:
                return jsonify({"error": "Too many markers (max 5000)"}), 400
            result_path = export_otio_markers(
                filepath, markers, otio_path,
                sequence_name=sequence_name,
                framerate=fps,
                total_duration=total_duration,
            )
        elif mode == "segments":
            segments_data = data.get("segments", [])
            if not isinstance(segments_data, list) or not segments_data:
                return jsonify({"error": "No segments provided (must be a non-empty list)"}), 400
            if len(segments_data) > 10000:
                return jsonify({"error": "Too many segments (max 10000)"}), 400
            from opencut.core.silence import TimeSegment
            segments = [
                TimeSegment(
                    start=safe_float(s.get("start", 0), 0.0, min_val=0.0),
                    end=safe_float(s.get("end", 0), 0.0, min_val=0.0),
                    label="speech",
                )
                for s in segments_data if isinstance(s, dict)
            ]
            if not segments:
                return jsonify({"error": "No valid segments provided"}), 400
            result_path = export_otio(
                filepath, segments, otio_path,
                sequence_name=sequence_name,
                framerate=fps,
            )
        else:
            # mode == "cuts" (default)
            cuts = data.get("cuts", [])
            if not isinstance(cuts, list) or not cuts:
                return jsonify({"error": "No cuts provided (must be a non-empty list)"}), 400
            if len(cuts) > 10000:
                return jsonify({"error": "Too many cuts (max 10000)"}), 400
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


@timeline_bp.route("/markers/import", methods=["POST"])
@require_csrf
def markers_import():
    """Import CSV / Premiere CSV / EDL markers (F102).

    Request body (JSON)::

        {
            "format": "csv" | "premiere_csv" | "edl",      # optional; auto-detected when omitted
            "fps": 30.0,                                    # optional; default 30
            "path": "/abs/path/to/markers.csv",             # OR
            "text": "..."                                    # inline content
        }

    The response payload is the normalised marker list. Callers (panel
    or JSX bridge) are responsible for actually inserting them into the
    active Premiere sequence — we keep this route data-only so the
    same import can be re-used by tests, CLI tooling, and the MCP
    server.
    """
    from opencut.core.marker_import import detect_format, import_markers

    try:
        data = get_json_dict()
        explicit_format = (data.get("format") or "").strip().lower() or None
        fps = safe_float(data.get("fps", 30.0), default=30.0)
        text = data.get("text")
        path = (data.get("path") or "").strip() or None
        if path:
            try:
                path = validate_filepath(path)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        if not text and not path:
            return jsonify({"error": "supply either 'text' or 'path'"}), 400
        if text and path:
            return jsonify({"error": "supply only one of 'text' or 'path'"}), 400

        if not explicit_format and path:
            explicit_format = detect_format(path)

        result = import_markers(
            text=text if text is not None else None,
            path=path if text is None else None,
            fps=fps,
            format=explicit_format,
        )
        payload = result.as_dict()
        payload["count"] = len(result.markers)
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": f"path not found: {exc.filename}"}), 404
    except Exception as exc:
        return safe_error(exc, "markers_import")


@timeline_bp.route("/review/bundle", methods=["POST"])
@require_csrf
def review_bundle():
    """Build a portable review bundle (F105).

    Request body::

        {
            "output_path": "/abs/path/to/output.zip",
            "job_label": "Rough Cut v3",
            "media_path": "/abs/path/to/render.mp4",   # optional
            "captions_path": "/abs/path/to/captions.srt",
            "markers_payload": {...},                  # arbitrary JSON
            "notes": "Free-form notes for the reviewer",
            "extra_files": ["/abs/path/to/lut.cube"],
            "include_media": true,
            "framerate": 30.0,
            "duration_seconds": 90.0,
            "annotation_width": 1920,
            "annotation_height": 1080
        }

    Returns the manifest of the produced bundle (sha-256, byte count,
    contained entries). When markers are supplied, the zip also includes
    ``markers.otio`` with an OpenTimelineIO Marker timeline and, when
    drawing annotations exist, SVG overlays under ``annotations/``.
    """
    from opencut.core.review_bundle import build_review_bundle

    try:
        data = get_json_dict()
        output_path = (data.get("output_path") or "").strip()
        if not output_path:
            return jsonify({"error": "output_path required"}), 400
        try:
            from opencut.security import validate_output_path
            output_path = validate_output_path(output_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        # Optional inputs — each is validated only when supplied so the
        # route is useful even when only markers + notes are bundled.
        def _maybe_path(key):
            raw = (data.get(key) or "").strip() or None
            if raw is None:
                return None
            return validate_filepath(raw)

        extra_files_raw = data.get("extra_files") or []
        if not isinstance(extra_files_raw, list):
            return jsonify({"error": "extra_files must be a list"}), 400
        extra_files = []
        for raw in extra_files_raw:
            try:
                extra_files.append(validate_filepath(str(raw)))
            except ValueError as exc:
                return jsonify({"error": f"extra_files: {exc}"}), 400

        bundle = build_review_bundle(
            output_path=output_path,
            job_label=str(data.get("job_label") or "").strip(),
            media_path=_maybe_path("media_path"),
            captions_path=_maybe_path("captions_path"),
            markers_payload=data.get("markers_payload"),
            notes=str(data.get("notes") or ""),
            extra_files=extra_files or None,
            include_media=safe_bool(data.get("include_media"), default=True),
            framerate=safe_float(data.get("framerate"), default=30.0, min_val=1.0, max_val=240.0),
            duration_seconds=safe_float(data.get("duration_seconds"), default=0.0, min_val=0.0),
            annotation_width=safe_int(data.get("annotation_width"), default=1920, min_val=1, max_val=16384),
            annotation_height=safe_int(data.get("annotation_height"), default=1080, min_val=1, max_val=16384),
        )
        return jsonify(bundle.as_dict())
    except FileNotFoundError as exc:
        return jsonify({"error": f"path not found: {exc.filename}"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return safe_error(exc, "review_bundle")


@timeline_bp.route("/provenance/c2pa", methods=["POST"])
@require_csrf
def provenance_c2pa_sidecar():
    """Write a C2PA sidecar next to a rendered asset (F110)."""
    from opencut.core.c2pa_sidecar import (
        C2paAction,
        C2paIngredient,
        build_sidecar,
    )

    try:
        data = get_json_dict()
        asset_path = (data.get("asset_path") or "").strip()
        if not asset_path:
            return jsonify({"error": "asset_path required"}), 400
        try:
            asset_path = validate_filepath(asset_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        ingredients_raw = data.get("ingredients") or []
        if not isinstance(ingredients_raw, list):
            return jsonify({"error": "ingredients must be a list"}), 400
        ingredients = [
            C2paIngredient(
                title=str(item.get("title") or ""),
                sha256=str(item.get("sha256") or ""),
                bytes=int(item.get("bytes") or 0),
                role=str(item.get("role") or "source"),
            )
            for item in ingredients_raw
            if isinstance(item, dict)
        ]

        actions_raw = data.get("actions") or []
        if not isinstance(actions_raw, list):
            return jsonify({"error": "actions must be a list"}), 400
        actions = [
            C2paAction(
                action=str(item.get("action") or "c2pa.unknown"),
                when=str(item.get("when") or ""),
                parameters=item.get("parameters") or {},
            )
            for item in actions_raw
            if isinstance(item, dict)
        ]

        build_kwargs = dict(
            asset_path=asset_path,
            ingredients=ingredients,
            actions=actions,
            title=str(data.get("title") or "").strip() or None,
        )
        claim_generator_override = str(data.get("claim_generator") or "").strip()
        if claim_generator_override:
            build_kwargs["claim_generator"] = claim_generator_override
        result = build_sidecar(**build_kwargs)
        return jsonify(result.as_dict())
    except FileNotFoundError as exc:
        return jsonify({"error": f"path not found: {exc.filename}"}), 404
    except Exception as exc:
        return safe_error(exc, "provenance_c2pa_sidecar")


@timeline_bp.route("/provenance/verify", methods=["POST"])
@require_csrf
def provenance_verify():
    """Verify a C2PA sidecar against the referenced asset (F110)."""
    from opencut.core.c2pa_sidecar import verify_sidecar

    try:
        data = get_json_dict()
        sidecar_path = (data.get("sidecar_path") or "").strip()
        if not sidecar_path:
            return jsonify({"error": "sidecar_path required"}), 400
        try:
            sidecar_path = validate_filepath(sidecar_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        result = verify_sidecar(sidecar_path)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": f"sidecar not found: {exc.filename}"}), 404
    except Exception as exc:
        return safe_error(exc, "provenance_verify")
