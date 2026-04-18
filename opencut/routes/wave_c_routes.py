"""
OpenCut Wave C Routes (v1.20.0)

Observability + quality harness + timeline-diff endpoints introduced
in v1.20.0.  The LLM-enrichment extensions for ``audio_description``
and ``quiz_overlay`` live inside those modules' existing blueprints;
this blueprint only owns the *new* surfaces.

Routes:
- ``POST /timeline/diff``                  — semantic OTIO/FCP-XML diff
- ``POST /video/quality/compare``          — VMAF/SSIM/PSNR vs reference
- ``POST /video/quality/batch-compare``    — same, for CI golden suites
- ``GET  /video/quality/backends``         — report quality-metric capability
- ``GET  /observability/status``           — Sentry / GlitchTip wiring report
"""

from __future__ import annotations

import logging
import os

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, validate_filepath

logger = logging.getLogger("opencut")

wave_c_bp = Blueprint("wave_c", __name__)


# ---------------------------------------------------------------------------
# OTIO timeline diff
# ---------------------------------------------------------------------------

def _diff_pre_validate(data):
    left = str(data.get("left") or "").strip()
    right = str(data.get("right") or "").strip()
    if not left or not right:
        return "Both 'left' and 'right' timeline paths are required."
    return None


@wave_c_bp.route("/timeline/diff", methods=["POST"])
@require_csrf
@async_job("otio_diff", filepath_required=False, pre_validate=_diff_pre_validate)
def route_timeline_diff(job_id, filepath, data):
    """Semantic diff between two OTIO-compatible timelines.

    Reads both sides via OpenTimelineIO adapters (.otio, .otioz, FCP XML,
    EDL, AAF when the optional adapter is installed) and emits a
    structured diff of tracks, clips (retimed/moved/added/removed), and
    markers (shifted/added/removed).
    """
    from opencut.export import otio_diff

    left = validate_filepath(str(data.get("left")))
    right = validate_filepath(str(data.get("right")))

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    result = otio_diff.diff_timelines(left, right, on_progress=_on_progress)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Objective quality metrics (VMAF / SSIM / PSNR)
# ---------------------------------------------------------------------------

@wave_c_bp.route("/video/quality/backends", methods=["GET"])
def route_quality_backends():
    from opencut.core import quality_metrics
    return jsonify({
        "ffmpeg": quality_metrics.check_quality_metrics_available(),
        "vmaf": quality_metrics.check_vmaf_available(),
        "metrics": list(quality_metrics.METRICS),
    })


def _quality_pre_validate(data):
    if not str(data.get("distorted") or "").strip():
        return "'distorted' path is required."
    if not str(data.get("reference") or "").strip():
        return "'reference' path is required."
    return None


@wave_c_bp.route("/video/quality/compare", methods=["POST"])
@require_csrf
@async_job("quality_compare", filepath_required=False, pre_validate=_quality_pre_validate)
def route_quality_compare(job_id, filepath, data):
    """Compute VMAF/SSIM/PSNR of a distorted file against a reference."""
    from opencut.core import quality_metrics
    from opencut.security import safe_float

    distorted = validate_filepath(str(data.get("distorted")))
    reference = validate_filepath(str(data.get("reference")))

    raw_metrics = data.get("metrics")
    if raw_metrics and isinstance(raw_metrics, list):
        metrics = [str(m).lower() for m in raw_metrics if isinstance(m, str)]
    else:
        metrics = None

    threshold = data.get("threshold_vmaf")
    if threshold is not None:
        threshold = safe_float(threshold, 0.0, min_val=0.0, max_val=100.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    report = quality_metrics.compare_videos(
        distorted, reference,
        metrics=metrics,
        threshold_vmaf=threshold,
        on_progress=_on_progress,
    )
    return dict(report)


@wave_c_bp.route("/video/quality/batch-compare", methods=["POST"])
@require_csrf
@async_job("quality_batch", filepath_required=False)
def route_quality_batch(job_id, filepath, data):
    """Run compare_videos over a list of ``{distorted, reference}`` pairs.

    Useful for CI golden-regression suites — one HTTP call, one job,
    N reports keyed by input order.  Invalid pairs become report
    entries with a ``notes`` explanation instead of aborting the batch.
    """
    from opencut.core import quality_metrics
    from opencut.security import safe_float

    pairs = data.get("pairs") or []
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("'pairs' must be a non-empty list of {distorted, reference} dicts")

    cleaned_pairs = []
    for entry in pairs[:60]:
        if not isinstance(entry, dict):
            continue
        d = str(entry.get("distorted") or "").strip()
        r = str(entry.get("reference") or "").strip()
        if not d or not r:
            cleaned_pairs.append({"distorted": d, "reference": r})
            continue
        try:
            d = validate_filepath(d)
            r = validate_filepath(r)
        except ValueError:
            # Leave as-is so the batch reporter marks it failed instead
            # of aborting the whole batch.
            pass
        cleaned_pairs.append({"distorted": d, "reference": r})

    raw_metrics = data.get("metrics")
    if raw_metrics and isinstance(raw_metrics, list):
        metrics = [str(m).lower() for m in raw_metrics if isinstance(m, str)]
    else:
        metrics = None

    threshold = data.get("threshold_vmaf")
    if threshold is not None:
        threshold = safe_float(threshold, 0.0, min_val=0.0, max_val=100.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=int(pct), message=str(msg))

    reports = quality_metrics.batch_compare(
        cleaned_pairs,
        metrics=metrics,
        threshold_vmaf=threshold,
        on_progress=_on_progress,
    )
    return {
        "reports": [dict(r) for r in reports],
        "pair_count": len(reports),
        "pass_count": sum(1 for r in reports if r.passes is True),
        "fail_count": sum(1 for r in reports if r.passes is False),
    }


# ---------------------------------------------------------------------------
# Observability status
# ---------------------------------------------------------------------------

@wave_c_bp.route("/observability/status", methods=["GET"])
def route_observability_status():
    """Report whether Sentry / GlitchTip is wired up for this process.

    Reads from the module-level ``_SENTRY_INITIALISED`` flag set by
    ``server._init_sentry_if_configured()`` on ``create_app()``.
    """
    try:
        from opencut.server import _SENTRY_INITIALISED
        initialised = bool(_SENTRY_INITIALISED)
    except Exception:  # noqa: BLE001
        initialised = False

    dsn_set = bool(os.environ.get("SENTRY_DSN", "").strip())
    try:
        import sentry_sdk  # noqa: F401
        sdk_installed = True
    except ImportError:
        sdk_installed = False

    return jsonify({
        "sentry_initialised": initialised,
        "sentry_dsn_set": dsn_set,
        "sentry_sdk_installed": sdk_installed,
        "env": os.environ.get("OPENCUT_SENTRY_ENV", "production" if initialised else None),
        "release": os.environ.get("OPENCUT_SENTRY_RELEASE"),
        "hint": (
            "Set SENTRY_DSN and pip install sentry-sdk[flask]"
            if not initialised else None
        ),
    })
