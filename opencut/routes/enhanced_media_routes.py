"""
OpenCut Enhanced Media Routes

Audio/video enhancement endpoints: speech restoration, one-click enhance,
low-light recovery, scene edit detection. Includes preview endpoints for
quick feedback before committing to full processing.
"""

import logging
import os
import tempfile

from flask import Blueprint

from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float

logger = logging.getLogger("opencut")

enhanced_media_bp = Blueprint("enhanced_media", __name__)


# ---------------------------------------------------------------------------
# POST /audio/enhance-speech -- full speech restoration (async)
# ---------------------------------------------------------------------------
@enhanced_media_bp.route("/audio/enhance-speech", methods=["POST"])
@require_csrf
@async_job("enhance_speech")
def enhance_speech_route(job_id, filepath, data):
    """AI speech restoration: denoise + clarity enhance + bandwidth extension."""
    from opencut.core.enhanced_speech import enhance_speech

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    mode = data.get("mode", "full")
    if mode not in ("denoise_only", "enhance", "full"):
        mode = "full"

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(effective_dir, f"{base}_enhanced_{mode}.wav")

    result = enhance_speech(
        filepath,
        output_path=out_path,
        mode=mode,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "mode": result.mode,
        "original_stats": result.original_stats,
        "enhanced_stats": result.enhanced_stats,
    }


# ---------------------------------------------------------------------------
# POST /audio/enhance-speech/preview -- preview on 5-second sample (async)
# ---------------------------------------------------------------------------
@enhanced_media_bp.route("/audio/enhance-speech/preview", methods=["POST"])
@require_csrf
@async_job("enhance_speech_preview")
def enhance_speech_preview_route(job_id, filepath, data):
    """Preview enhanced speech on a 5-second sample."""
    from opencut.core.enhanced_speech import enhance_speech
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg

    mode = data.get("mode", "full")
    if mode not in ("denoise_only", "enhance", "full"):
        mode = "full"
    start_time = safe_float(data.get("start_time", 0), 0, min_val=0.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(5, "Extracting 5-second sample...")

    # Extract 5-second sample
    fd, sample_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_esp_")
    os.close(fd)

    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", filepath,
            "-ss", str(start_time),
            "-t", "5",
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1",
            sample_path,
        ]
        run_ffmpeg(cmd)

        _progress(15, "Enhancing sample...")

        output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(output_dir, f"{base}_preview_{mode}.wav")

        result = enhance_speech(
            sample_path,
            output_path=out_path,
            mode=mode,
            on_progress=lambda pct, msg="": _progress(15 + int(pct * 0.85), msg),
        )

        return {
            "output_path": result.output_path,
            "mode": result.mode,
            "preview": True,
            "sample_start": start_time,
            "sample_duration": 5.0,
        }
    finally:
        try:
            os.unlink(sample_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# POST /video/one-click-enhance -- full enhance pipeline (async)
# ---------------------------------------------------------------------------
@enhanced_media_bp.route("/video/one-click-enhance", methods=["POST"])
@require_csrf
@async_job("one_click_enhance")
def one_click_enhance_route(job_id, filepath, data):
    """One-click video enhancement: auto-detect and fix all issues."""
    from opencut.core.one_click_enhance import one_click_enhance

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    preset = data.get("preset", "balanced")
    if preset not in ("fast", "balanced", "quality"):
        preset = "balanced"

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_enhanced_{preset}{ext}")

    result = one_click_enhance(
        filepath,
        output_path=out_path,
        preset=preset,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "steps_applied": result.steps_applied,
        "duration_seconds": result.duration_seconds,
        "preset": preset,
    }


# ---------------------------------------------------------------------------
# POST /video/enhance-low-light -- low-light enhancement (async)
# ---------------------------------------------------------------------------
@enhanced_media_bp.route("/video/enhance-low-light", methods=["POST"])
@require_csrf
@async_job("enhance_low_light")
def enhance_low_light_route(job_id, filepath, data):
    """Low-light video enhancement with shadow lift and detail recovery."""
    from opencut.core.low_light import enhance_low_light

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    strength = safe_float(data.get("strength", 1.0), 1.0, min_val=0.0, max_val=2.0)
    denoise = data.get("denoise", True)
    if isinstance(denoise, str):
        denoise = denoise.lower() in ("true", "1", "yes")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_lowlight{ext}")

    result = enhance_low_light(
        filepath,
        output_path=out_path,
        strength=strength,
        denoise=denoise,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "original_avg_luminance": result.original_avg_luminance,
        "enhanced_avg_luminance": result.enhanced_avg_luminance,
        "denoise_applied": result.denoise_applied,
    }


# ---------------------------------------------------------------------------
# POST /video/enhance-low-light/preview -- preview on single frame (async)
# ---------------------------------------------------------------------------
@enhanced_media_bp.route("/video/enhance-low-light/preview", methods=["POST"])
@require_csrf
@async_job("low_light_preview")
def enhance_low_light_preview_route(job_id, filepath, data):
    """Preview low-light enhancement on a single frame."""
    from opencut.core.low_light import (
        _build_curves_filter,
        _build_denoise_filter,
        _build_unsharp_filter,
        _measure_avg_luminance,
    )
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg

    strength = safe_float(data.get("strength", 1.0), 1.0, min_val=0.0, max_val=2.0)
    denoise = data.get("denoise", True)
    if isinstance(denoise, str):
        denoise = denoise.lower() in ("true", "1", "yes")
    timestamp = safe_float(data.get("timestamp", 1.0), 1.0, min_val=0.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    _progress(10, "Extracting frame...")

    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(output_dir, f"{base}_lowlight_preview.jpg")

    # Build filter chain
    filters = [_build_curves_filter(strength), _build_unsharp_filter(strength)]
    if denoise:
        filters.append(_build_denoise_filter(strength))
    vf = ",".join(filters)

    _progress(30, "Applying low-light enhancement to frame...")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-ss", str(timestamp),
        "-i", filepath,
        "-vf", vf,
        "-vframes", "1",
        "-q:v", "2",
        out_path,
    ]
    run_ffmpeg(cmd)

    _progress(80, "Measuring result...")
    original_lum = _measure_avg_luminance(filepath, sample_seconds=2.0)

    _progress(100, "Preview ready")

    return {
        "output_path": out_path,
        "preview": True,
        "timestamp": timestamp,
        "strength": strength,
        "original_avg_luminance": original_lum,
    }


# ---------------------------------------------------------------------------
# POST /video/detect-edits -- scene edit detection (async)
# ---------------------------------------------------------------------------
@enhanced_media_bp.route("/video/detect-edits", methods=["POST"])
@require_csrf
@async_job("detect_edits")
def detect_edits_route(job_id, filepath, data):
    """Detect edit points (cuts, dissolves, fades) in pre-edited footage."""
    from opencut.core.scene_edit_detect import detect_edits

    threshold = safe_float(data.get("threshold", 0.3), 0.3, min_val=0.01, max_val=1.0)
    min_scene_duration = safe_float(
        data.get("min_scene_duration", 0.5), 0.5, min_val=0.0, max_val=60.0,
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = detect_edits(
        filepath,
        threshold=threshold,
        min_scene_duration=min_scene_duration,
        on_progress=_progress,
    )

    return result.to_dict()
