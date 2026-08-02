"""
Routes for the Sequence Index panel (RESEARCH_FEATURE_PLAN_2026-05-25 Q7 / F273).

The CEP/UXP panel POSTs the JSON returned by
``host/index.jsx::ocGetSequenceInfo()`` (plus optional transcript
segments, ratings, and tags) and gets back a flat row list. Sort and
filter happen on the result via subsequent POSTs so the panel can
re-paginate without re-walking the timeline.

Routes:
  POST /timeline/sequence-index             build index from sequence JSON
  POST /timeline/sequence-index/filter      filter + sort a previously built index
  POST /timeline/sequence-index/export-csv  write the filtered view to a CSV file
  GET  /timeline/sequence-index/info        report capability + sort keys
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf, safe_bool, safe_int, validate_path

logger = logging.getLogger("opencut")
sequence_index_bp = Blueprint("sequence_index", __name__)


def _dict_to_row(d: dict):
    """Rehydrate a row dict into an IndexRow for sort/filter."""
    from opencut.core.sequence_index import IndexRow
    return IndexRow(
        track_type=str(d.get("track_type") or ""),
        track_index=int(d.get("track_index", 0) or 0),
        clip_index=int(d.get("clip_index", 0) or 0),
        name=str(d.get("name") or ""),
        path=str(d.get("path") or ""),
        start_s=float(d.get("start_s", 0.0) or 0.0),
        end_s=float(d.get("end_s", 0.0) or 0.0),
        duration_s=float(d.get("duration_s", 0.0) or 0.0),
        timecode_in=str(d.get("timecode_in") or ""),
        timecode_out=str(d.get("timecode_out") or ""),
        effects=list(d.get("effects") or []),
        rating=int(d.get("rating", 0) or 0),
        tags=list(d.get("tags") or []),
        transcript_excerpt=str(d.get("transcript_excerpt") or ""),
        locator_id=str(d.get("locator_id") or ""),
        host_locators=dict(d.get("host_locators") or {}),
        offline=safe_bool(d.get("offline", False), False),
        flash_frame=safe_bool(d.get("flash_frame", False), False),
    )


@sequence_index_bp.route("/timeline/sequence-index", methods=["POST"])
@require_csrf
def route_build_sequence_index():
    """Build the index from a Premiere sequence JSON payload.

    Body params:
      sequence              dict   the JSON returned by ocGetSequenceInfo (required)
      transcript_segments   list   optional [{start,end,text}]
      ratings               dict   optional {locator_id|clip_path: int} 0..5
      tags                  dict   optional {locator_id|clip_path: [str]}
      excerpt_chars         int    cap on transcript_excerpt length (default 240)
    """
    try:
        from opencut.core import sequence_index

        data = request.get_json(silent=True) or {}
        seq = data.get("sequence")
        if not isinstance(seq, dict):
            raise ValueError("'sequence' must be an object (output of ocGetSequenceInfo)")

        transcript_segments = data.get("transcript_segments")
        if transcript_segments is not None and not isinstance(transcript_segments, list):
            raise ValueError("'transcript_segments' must be a list")

        ratings = data.get("ratings")
        if ratings is not None and not isinstance(ratings, dict):
            raise ValueError("'ratings' must be an object")

        tags = data.get("tags")
        if tags is not None and not isinstance(tags, dict):
            raise ValueError("'tags' must be an object")

        excerpt_chars = safe_int(data.get("excerpt_chars", 240), 240, min_val=0, max_val=4096)
        flash_frame_frames = safe_int(
            data.get("flash_frame_frames", sequence_index.DEFAULT_FLASH_FRAME_FRAMES),
            sequence_index.DEFAULT_FLASH_FRAME_FRAMES,
            min_val=0,
            max_val=240,
        )

        result = sequence_index.build_index(
            sequence_payload=seq,
            transcript_segments=transcript_segments,
            ratings=ratings,
            tags=tags,
            excerpt_chars=excerpt_chars,
            flash_frame_frames=flash_frame_frames,
        )
        return jsonify({k: result[k] for k in result.keys()})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_build")


def _filter_and_sort(data: dict):
    """Shared filter+sort body for the filter and CSV-export routes."""
    from opencut.core import sequence_index

    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("'rows' must be a list")

    rows = [_dict_to_row(r) for r in raw_rows if isinstance(r, dict)]

    track_type = data.get("track_type") or None
    if track_type and track_type not in ("video", "audio"):
        raise ValueError("'track_type' must be 'video' | 'audio' | omitted")

    has_effects = data.get("has_effects", None)
    if has_effects is not None:
        has_effects = safe_bool(has_effects, False)

    offline = data.get("offline", None)
    if offline is not None:
        offline = safe_bool(offline, False)

    flash_frame = data.get("flash_frame", None)
    if flash_frame is not None:
        flash_frame = safe_bool(flash_frame, False)

    min_rating = safe_int(data.get("min_rating", 0), 0, min_val=0, max_val=5)
    sort_key = str(data.get("sort_key") or "start_s")
    if sort_key not in sequence_index.SORT_KEYS:
        raise ValueError(
            f"Unknown sort_key '{sort_key}'. Valid: {sorted(sequence_index.SORT_KEYS)}"
        )
    descending = safe_bool(data.get("descending", False), False)

    filtered = sequence_index.filter_rows(
        rows,
        query=str(data.get("query") or ""),
        track_type=track_type,
        min_rating=min_rating,
        has_effects=has_effects,
        offline=offline,
        flash_frame=flash_frame,
    )
    return sequence_index.sort_rows(filtered, sort_key, descending), sort_key, descending


@sequence_index_bp.route("/timeline/sequence-index/filter", methods=["POST"])
@require_csrf
def route_filter_sequence_index():
    """Filter + sort an existing index (rows previously emitted by build).

    Body params:
      rows           list  required, list of row dicts (round-tripped from build)
      query          str   substring against name/path/transcript/tags/effects
      track_type     str   "video" | "audio" | "" (any)
      min_rating     int   drop rows below this rating
      has_effects    bool  optional (true=must have, false=must not have)
      offline        bool  optional (true=offline media only, false=linked only)
      flash_frame    bool  optional (true=flash frames only, false=exclude them)
      sort_key       str   one of SORT_KEYS (default "start_s")
      descending     bool  default false
      offset         int   first matching row to return (default 0)
      limit          int   max rows to return, 0 = all (default 0)

    ``total_rows`` is always the number of rows that matched the filter, so
    the panel can render "showing N of M" without a second request.
    """
    try:
        data = request.get_json(silent=True) or {}
        sorted_rows, sort_key, descending = _filter_and_sort(data)

        total = len(sorted_rows)
        offset = safe_int(data.get("offset", 0), 0, min_val=0, max_val=1_000_000)
        limit = safe_int(data.get("limit", 0), 0, min_val=0, max_val=100_000)
        window = sorted_rows[offset:] if offset else sorted_rows
        if limit:
            window = window[:limit]

        return jsonify({
            "rows": [r.to_dict() for r in window],
            "total_rows": total,
            "returned_rows": len(window),
            "offset": offset,
            "limit": limit,
            "sort_key": sort_key,
            "descending": descending,
        })
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_filter")


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _csv_output_path(data: dict, sequence_name: str) -> str:
    """Resolve the CSV destination inside a validated directory."""
    output_dir = str(data.get("output_dir") or "").strip()
    if output_dir:
        resolved_dir = validate_path(output_dir)
        os.makedirs(resolved_dir, exist_ok=True)
    else:
        resolved_dir = tempfile.gettempdir()

    stem = _SAFE_STEM_RE.sub("_", sequence_name or "sequence").strip("._-") or "sequence"
    return os.path.join(
        resolved_dir, f"{stem[:80]}_index_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )


@sequence_index_bp.route("/timeline/sequence-index/export-csv", methods=["POST"])
@require_csrf
def route_export_sequence_index_csv():
    """Export the *filtered* index view to a CSV file.

    Accepts the same filter/sort params as ``/filter`` so the sheet matches
    exactly what the panel is showing, plus:

      columns        list  subset of CSV_COLUMNS, in display order
      output_dir     str   destination directory (default: system temp)
      sequence_name  str   used for the generated filename
    """
    try:
        from opencut.core import sequence_index

        data = request.get_json(silent=True) or {}
        sorted_rows, _sort_key, _descending = _filter_and_sort(data)

        columns = data.get("columns")
        if columns is not None:
            if not isinstance(columns, list):
                raise ValueError("'columns' must be a list")
            columns = [str(c) for c in columns]

        output = _csv_output_path(data, str(data.get("sequence_name") or ""))
        result = sequence_index.export_csv(sorted_rows, output, columns)
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError as exc:
        return jsonify({"error": f"Could not write CSV: {exc}"}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_export_csv")


@sequence_index_bp.route("/timeline/sequence-index/info", methods=["GET"])
def route_sequence_index_info():
    """Report module availability + the sort-key allowlist."""
    try:
        from opencut.core import sequence_index
        return jsonify({
            "available": sequence_index.check_sequence_index_available(),
            "sort_keys": sorted(sequence_index.SORT_KEYS),
            "csv_columns": list(sequence_index.CSV_COLUMNS),
            "hideable_columns": sorted(sequence_index.HIDEABLE_COLUMNS),
            "filters": ["query", "track_type", "min_rating", "has_effects",
                        "offline", "flash_frame"],
            "flash_frame_frames": sequence_index.DEFAULT_FLASH_FRAME_FRAMES,
            "host_locator_fields": [
                "sequence_guid",
                "sequence_name",
                "track_type",
                "track_index",
                "clip_index",
                "track_item_id",
                "project_item_id",
                "marker_index",
                "marker_id",
                "marker_time_s",
            ],
            "install_hint": sequence_index.INSTALL_HINT,
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_info")
