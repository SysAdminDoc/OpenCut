"""
Routes for the Sequence Index panel (RESEARCH_FEATURE_PLAN_2026-05-25 Q7 / F273).

The CEP/UXP panel POSTs the JSON returned by
``host/index.jsx::ocGetSequenceInfo()`` (plus optional transcript
segments, ratings, and tags) and gets back a flat row list. Sort and
filter happen on the result via subsequent POSTs so the panel can
re-paginate without re-walking the timeline.

Routes:
  POST /timeline/sequence-index           build index from sequence JSON
  POST /timeline/sequence-index/filter    filter + sort a previously built index
  GET  /timeline/sequence-index/info      report capability + sort keys
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import require_csrf, safe_bool, safe_int

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
    )


@sequence_index_bp.route("/timeline/sequence-index", methods=["POST"])
@require_csrf
def route_build_sequence_index():
    """Build the index from a Premiere sequence JSON payload.

    Body params:
      sequence              dict   the JSON returned by ocGetSequenceInfo (required)
      transcript_segments   list   optional [{start,end,text}]
      ratings               dict   optional {clip_path: int} 0..5
      tags                  dict   optional {clip_path: [str]}
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

        result = sequence_index.build_index(
            sequence_payload=seq,
            transcript_segments=transcript_segments,
            ratings=ratings,
            tags=tags,
            excerpt_chars=excerpt_chars,
        )
        return jsonify({k: result[k] for k in result.keys()})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_build")


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
      sort_key       str   one of SORT_KEYS (default "start_s")
      descending     bool  default false
    """
    try:
        from opencut.core import sequence_index

        data = request.get_json(silent=True) or {}
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
        )
        sorted_rows = sequence_index.sort_rows(filtered, sort_key, descending)
        return jsonify({
            "rows": [r.to_dict() for r in sorted_rows],
            "total_rows": len(sorted_rows),
            "sort_key": sort_key,
            "descending": descending,
        })
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_filter")


@sequence_index_bp.route("/timeline/sequence-index/info", methods=["GET"])
def route_sequence_index_info():
    """Report module availability + the sort-key allowlist."""
    try:
        from opencut.core import sequence_index
        return jsonify({
            "available": sequence_index.check_sequence_index_available(),
            "sort_keys": sorted(sequence_index.SORT_KEYS),
            "install_hint": sequence_index.INSTALL_HINT,
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "sequence_index_info")
