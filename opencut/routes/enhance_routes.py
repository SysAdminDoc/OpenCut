"""
Routes for the one-click Enhance macro (RESEARCH_FEATURE_PLAN_2026-05-25 Q3).

Two surfaces:

  POST /enhance/auto          — execute the macro (returns job_id)
  POST /enhance/auto/dry-run  — return the planned pipeline only
  GET  /enhance/auto/styles   — list available style presets

The execute path is wrapped in ``@async_job`` so cancellation,
SQLite persistence, and progress streaming work the same way every
other long-running route does.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_bool

logger = logging.getLogger("opencut")
enhance_bp = Blueprint("enhance", __name__)


@enhance_bp.route("/enhance/auto", methods=["POST"])
@require_csrf
@async_job("enhance_auto", filepath_required=True)
def route_enhance_auto(job_id, filepath, data):
    """Run the one-click Enhance macro on ``filepath``.

    Body params:
      style    str   One of ``social`` | ``speech`` | ``cinematic`` (default ``social``).
      output   str   Final output path (auto under source dir when omitted).
      dry_run  bool  When true, return the planned pipeline only (no work).
    """
    from opencut.core import enhance_auto

    style = str(data.get("style") or enhance_auto.DEFAULT_STYLE)
    if style not in enhance_auto.VALID_STYLES:
        raise ValueError(
            f"Invalid style '{style}'. Must be one of: "
            + ", ".join(enhance_auto.VALID_STYLES)
        )

    dry_run = safe_bool(data.get("dry_run"), False)
    output = str(data.get("output") or "").strip() or None

    def _prog(p, m=""):
        _update_job(job_id, progress=int(p), message=str(m))

    result = enhance_auto.enhance(
        input_path=filepath,
        style=style,
        output=output,
        dry_run=dry_run,
        on_progress=_prog,
    )
    # EnhanceResult is subscriptable; jsonify happy path.
    return {k: result[k] for k in result.keys()}


@enhance_bp.route("/enhance/auto/dry-run", methods=["POST"])
@require_csrf
def route_enhance_auto_dry_run():
    """Return the planned Enhance pipeline without executing it.

    Synchronous (no job_id) so the panel can render the plan instantly
    before the user clicks Apply.
    """
    try:
        from flask import request

        from opencut.core import enhance_auto
        from opencut.security import validate_filepath

        data = request.get_json(silent=True) or {}
        raw_path = str(data.get("filepath", "")).strip()
        if not raw_path:
            raise ValueError("filepath is required")
        path = validate_filepath(raw_path)

        style = str(data.get("style") or enhance_auto.DEFAULT_STYLE)
        if style not in enhance_auto.VALID_STYLES:
            raise ValueError(
                f"Invalid style '{style}'. Must be one of: "
                + ", ".join(enhance_auto.VALID_STYLES)
            )

        result = enhance_auto.enhance(
            input_path=path,
            style=style,
            dry_run=True,
        )
        return jsonify({k: result[k] for k in result.keys()})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover — fallthrough
        return safe_error(exc, "enhance_auto_dry_run")


@enhance_bp.route("/enhance/auto/styles", methods=["GET"])
def route_enhance_auto_styles():
    """Return the available Enhance style presets and their defaults."""
    try:
        from opencut.core import enhance_auto
        return jsonify({
            "styles": list(enhance_auto.VALID_STYLES),
            "default": enhance_auto.DEFAULT_STYLE,
            "presets": enhance_auto.STYLES,
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "enhance_auto_styles")
