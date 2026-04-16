"""
OpenCut Processing Routes

Audio analysis (loudness, spectrum, platform compliance),
deinterlacing, and lens distortion correction endpoints.
"""

import logging

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, validate_path

logger = logging.getLogger("opencut")

processing_bp = Blueprint("processing", __name__)


# ---------------------------------------------------------------------------
# Audio Loudness Measurement
# ---------------------------------------------------------------------------
@processing_bp.route("/audio/loudness", methods=["POST"])
@require_csrf
@async_job("audio_loudness")
def audio_loudness(job_id, filepath, data):
    """Measure EBU R128 loudness of an audio/video file."""
    from opencut.core.audio_analysis import measure_loudness

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = measure_loudness(filepath, on_progress=_on_progress)

    return {
        "integrated_lufs": result.integrated_lufs,
        "true_peak_dbtp": result.true_peak_dbtp,
        "lra": result.lra,
        "momentary_max": result.momentary_max,
        "short_term_max": result.short_term_max,
    }


# ---------------------------------------------------------------------------
# Audio Spectrum Analysis
# ---------------------------------------------------------------------------
@processing_bp.route("/audio/spectrum", methods=["POST"])
@require_csrf
@async_job("audio_spectrum")
def audio_spectrum(job_id, filepath, data):
    """Analyse frequency spectrum of an audio/video file."""
    from opencut.core.audio_analysis import analyze_spectrum

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = analyze_spectrum(filepath, on_progress=_on_progress)

    return {
        "sub_bass": result.sub_bass,
        "bass": result.bass,
        "low_mid": result.low_mid,
        "mid": result.mid,
        "upper_mid": result.upper_mid,
        "presence": result.presence,
        "brilliance": result.brilliance,
    }


# ---------------------------------------------------------------------------
# Platform Loudness Compliance Check
# ---------------------------------------------------------------------------
@processing_bp.route("/audio/loudness-check", methods=["POST"])
@require_csrf
@async_job("audio_loudness_check")
def audio_loudness_check(job_id, filepath, data):
    """Check audio loudness against a platform's target."""
    from opencut.core.audio_analysis import PLATFORM_TARGETS, check_platform_loudness

    platform = data.get("platform", "youtube").strip().lower()
    if platform not in PLATFORM_TARGETS:
        platform = "youtube"

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = check_platform_loudness(
        filepath, platform=platform, on_progress=_on_progress,
    )

    return {
        "platform": result.platform,
        "target_lufs": result.target_lufs,
        "actual_lufs": result.actual_lufs,
        "passes": result.passes,
        "adjustment_needed_db": result.adjustment_needed_db,
    }


# ---------------------------------------------------------------------------
# Interlace Detection
# ---------------------------------------------------------------------------
@processing_bp.route("/video/detect-interlace", methods=["POST"])
@require_csrf
@async_job("detect_interlace")
def video_detect_interlace(job_id, filepath, data):
    """Detect interlaced content in a video file."""
    from opencut.core.deinterlace import detect_interlaced

    _update_job(job_id, progress=10, message="Detecting interlacing...")

    info = detect_interlaced(filepath)

    _update_job(job_id, progress=100, message="Detection complete")

    return {
        "is_interlaced": info.is_interlaced,
        "field_order": info.field_order,
        "detection_confidence": info.detection_confidence,
    }


# ---------------------------------------------------------------------------
# Deinterlace
# ---------------------------------------------------------------------------
@processing_bp.route("/video/deinterlace", methods=["POST"])
@require_csrf
@async_job("deinterlace")
def video_deinterlace(job_id, filepath, data):
    """Deinterlace a video file."""
    from opencut.core.deinterlace import deinterlace

    method = data.get("method", "bwdif").strip().lower()
    if method not in ("yadif", "bwdif", "nnedi"):
        method = "bwdif"

    field_order = data.get("field_order", "auto").strip().lower()
    if field_order not in ("tff", "bff", "auto"):
        field_order = "auto"

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    # Build output path
    out = None
    if output_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out = os.path.join(output_dir, f"{base}_deinterlaced_{method}.mp4")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = deinterlace(
        filepath,
        output_path_override=out,
        method=method,
        field_order=field_order,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Lens Distortion Correction
# ---------------------------------------------------------------------------
@processing_bp.route("/video/lens-correct", methods=["POST"])
@require_csrf
@async_job("lens_correct")
def video_lens_correct(job_id, filepath, data):
    """Correct lens distortion in a video file."""
    from opencut.core.lens_correction import correct_lens_distortion

    preset = data.get("preset", "").strip() or None
    k1 = data.get("k1")
    k2 = data.get("k2")

    if k1 is not None:
        k1 = safe_float(k1, default=None, min_val=-2.0, max_val=2.0)
    if k2 is not None:
        k2 = safe_float(k2, default=0.0, min_val=-2.0, max_val=2.0)

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out = None
    if output_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        suffix = f"lens_{preset}" if preset else "lens_corrected"
        out = os.path.join(output_dir, f"{base}_{suffix}.mp4")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = correct_lens_distortion(
        filepath,
        output_path_override=out,
        k1=k1,
        k2=k2,
        preset=preset,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Horizon Levelling
# ---------------------------------------------------------------------------
@processing_bp.route("/video/horizon-level", methods=["POST"])
@require_csrf
@async_job("horizon_level")
def video_horizon_level(job_id, filepath, data):
    """Level a tilted horizon by rotating the video."""
    from opencut.core.lens_correction import level_horizon

    angle = safe_float(data.get("angle", 0.0), 0.0, min_val=-45.0, max_val=45.0)

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    out = None
    if output_dir:
        import os
        base = os.path.splitext(os.path.basename(filepath))[0]
        out = os.path.join(output_dir, f"{base}_levelled.mp4")

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = level_horizon(
        filepath,
        output_path_override=out,
        angle=angle,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# Lens Presets (SYNC — no async_job needed)
# ---------------------------------------------------------------------------
@processing_bp.route("/video/lens-presets", methods=["GET"])
def video_lens_presets():
    """Return available lens correction presets."""
    from opencut.core.lens_correction import list_lens_presets
    return jsonify(list_lens_presets())
