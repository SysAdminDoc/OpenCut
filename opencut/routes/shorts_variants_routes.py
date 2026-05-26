"""
Routes for the shorts A/B variant generator (RESEARCH_FEATURE_PLAN_2026-05-25 Q8).

Three surfaces:

  POST /shorts/variants            — render N variants (async)
  POST /shorts/variants/dry-run    — return the variant plan (sync)
  GET  /shorts/variants/info       — list caption styles + variant bounds
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool, safe_float, safe_int

logger = logging.getLogger("opencut")
shorts_variants_bp = Blueprint("shorts_variants", __name__)


def _coerce_segments(raw) -> list:
    """Normalize the optional transcript_segments payload to ``[{start,end,text},…]``."""
    out: list = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            s = float(entry.get("start", 0.0))
            e = float(entry.get("end", s))
            t = str(entry.get("text", ""))
        except (TypeError, ValueError):
            continue
        out.append({"start": s, "end": e, "text": t})
    return out


@shorts_variants_bp.route("/shorts/variants", methods=["POST"])
@require_csrf
@async_job("shorts_variants", filepath_required=True)
def route_shorts_variants(job_id, filepath, data):
    """Render N short-form variants of ``filepath[start:end]``.

    Body params:
      start              float  required, seconds
      end                float  required, seconds (> start)
      n_variants         int    2..6 (default 3, clamped)
      width, height      int    target dims (default 1080×1920)
      transcript_segments list  optional [{start,end,text}] for burn-in
      burn_captions      bool   default True
      output_dir         str    optional
    """
    from opencut.core import shorts_variants

    start = safe_float(data.get("start"), 0.0, min_val=0.0, max_val=86400.0)
    end = safe_float(data.get("end"), 0.0, min_val=0.0, max_val=86400.0)
    if end <= start:
        raise ValueError("end must be greater than start")

    n_variants = safe_int(
        data.get("n_variants", shorts_variants.DEFAULT_VARIANTS),
        shorts_variants.DEFAULT_VARIANTS,
        min_val=shorts_variants.MIN_VARIANTS,
        max_val=shorts_variants.MAX_VARIANTS,
    )
    width = safe_int(data.get("width", 1080), 1080, min_val=100, max_val=7680)
    height = safe_int(data.get("height", 1920), 1920, min_val=100, max_val=7680)
    burn_captions = safe_bool(data.get("burn_captions", True), True)
    output_dir = str(data.get("output_dir") or "").strip()
    transcript_segments = _coerce_segments(data.get("transcript_segments"))

    def _prog(p, m=""):
        _update_job(job_id, progress=int(p), message=str(m))

    result = shorts_variants.generate_variants(
        input_path=filepath,
        start=start,
        end=end,
        n_variants=n_variants,
        width=width,
        height=height,
        transcript_segments=transcript_segments or None,
        burn_captions=burn_captions,
        output_dir=output_dir,
        on_progress=_prog,
    )
    return {k: result[k] for k in result.keys()}


@shorts_variants_bp.route("/shorts/variants/dry-run", methods=["POST"])
@require_csrf
def route_shorts_variants_dry_run():
    """Return the planned variant set without executing.

    Same body shape as ``/shorts/variants`` minus ``transcript_segments``
    (which is not needed for planning).
    """
    try:
        from opencut.core import shorts_variants
        from opencut.security import validate_filepath

        data = request.get_json(silent=True) or {}
        raw_path = str(data.get("filepath", "")).strip()
        if not raw_path:
            raise ValueError("filepath is required")
        path = validate_filepath(raw_path)

        start = float(data.get("start", 0.0))
        end = float(data.get("end", 0.0))
        if end <= start:
            raise ValueError("end must be greater than start")

        n_variants = int(data.get("n_variants", shorts_variants.DEFAULT_VARIANTS))
        width = int(data.get("width", 1080))
        height = int(data.get("height", 1920))

        result = shorts_variants.plan_variants(
            input_path=path,
            start=start,
            end=end,
            n_variants=n_variants,
            width=width,
            height=height,
        )
        return jsonify({k: result[k] for k in result.keys()})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "shorts_variants_dry_run")


@shorts_variants_bp.route("/shorts/variants/info", methods=["GET"])
def route_shorts_variants_info():
    """Return caption styles + variant bounds + module availability."""
    try:
        from opencut.core import shorts_variants
        return jsonify({
            "available": shorts_variants.check_shorts_variants_available(),
            "caption_styles": list(shorts_variants.CAPTION_STYLES),
            "default_variants": shorts_variants.DEFAULT_VARIANTS,
            "min_variants": shorts_variants.MIN_VARIANTS,
            "max_variants": shorts_variants.MAX_VARIANTS,
            "install_hint": shorts_variants.INSTALL_HINT,
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "shorts_variants_info")
