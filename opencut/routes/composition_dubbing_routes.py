"""
OpenCut Composition & Dubbing Routes

Composition analysis (guide overlays, shot classification, pacing,
saliency crop) and AI dubbing (full pipeline, isochronous translation,
multi-language track management, emotion-preserving dubbing).
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

composition_dubbing_bp = Blueprint("composition_dubbing", __name__)


# ---------------------------------------------------------------------------
# POST /composition/guide — Generate composition overlay
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/composition/guide", methods=["POST"])
@require_csrf
@async_job("composition_guide")
def composition_guide(job_id, filepath, data):
    """Generate composition guide overlay on a video frame."""
    from opencut.core.composition_guide import generate_guide_overlay

    guides = data.get("guides", ["rule_of_thirds"])
    if isinstance(guides, str):
        guides = [g.strip() for g in guides.split(",")]

    timestamp = safe_float(data.get("timestamp", 0), 0.0, min_val=0.0)
    opacity = safe_float(data.get("opacity", 0.5), 0.5, min_val=0.0, max_val=1.0)
    output_dir = data.get("output_dir", "")

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_guide_overlay(
        input_path=filepath,
        guides=guides,
        timestamp=timestamp,
        output_dir=output_dir,
        opacity=opacity,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /composition/classify-shot — Classify shot type
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/composition/classify-shot", methods=["POST"])
@require_csrf
@async_job("shot_classify")
def classify_shot(job_id, filepath, data):
    """Classify shot type from a frame or video."""
    mode = data.get("mode", "frame").strip().lower()

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    if mode == "video":
        from opencut.core.shot_classify import classify_shots
        result = classify_shots(filepath, on_progress=_progress)
        return {
            "shots": [
                {
                    "start": s.start,
                    "end": s.end,
                    "shot_type": s.shot_type,
                    "confidence": s.confidence,
                }
                for s in result.shots
            ],
            "total_shots": result.total_shots,
            "duration": result.duration,
            "type_distribution": result.type_distribution,
        }
    else:
        # Frame mode — use the enhanced classify_shot_type
        _progress(10, "Extracting frame...")

        frame_path = data.get("frame_path", "").strip()
        if frame_path:
            frame_path = validate_filepath(frame_path)
        else:
            # Extract frame from video at timestamp
            import os
            import subprocess
            import tempfile
            from opencut.helpers import get_ffmpeg_path

            timestamp = safe_float(data.get("timestamp", 0), 0.0, min_val=0.0)
            tmp_dir = tempfile.mkdtemp(prefix="opencut_shotclass_")
            frame_path = os.path.join(tmp_dir, "frame.jpg")

            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                "-ss", str(timestamp), "-i", filepath,
                "-frames:v", "1", "-q:v", "2", "-y", frame_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)

            if not os.path.isfile(frame_path):
                raise RuntimeError("Failed to extract frame from video")

        _progress(50, "Classifying shot type...")

        from opencut.core.shot_classify import classify_shot_type
        result = classify_shot_type(frame_path)

        _progress(100, f"Classified: {result.get('shot_type', 'unknown')}")
        return result


# ---------------------------------------------------------------------------
# POST /composition/analyze-pacing — Pacing analysis
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/composition/analyze-pacing", methods=["POST"])
@require_csrf
@async_job("pacing_analysis", filepath_required=False)
def analyze_pacing_route(job_id, filepath, data):
    """Analyze edit pacing from cut points or video file."""

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    cut_points = data.get("cut_points")
    total_duration = safe_float(data.get("total_duration", 0), 0.0)
    genre = data.get("genre", "general").strip()

    if cut_points is not None and total_duration > 0:
        # Use cut-point-based analysis
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts

        anomaly_threshold = safe_float(
            data.get("anomaly_threshold", 2.0), 2.0,
            min_val=1.0, max_val=5.0,
        )

        result = analyze_pacing_from_cuts(
            cut_points=cut_points,
            total_duration=total_duration,
            genre=genre,
            anomaly_threshold=anomaly_threshold,
            on_progress=_progress,
        )
        return result
    else:
        # File-based analysis
        if not filepath:
            raise ValueError(
                "Either 'filepath' or 'cut_points' + 'total_duration' required"
            )
        from opencut.core.pacing_analysis import analyze_pacing

        threshold = safe_float(
            data.get("threshold", 0.3), 0.3,
            min_val=0.0, max_val=1.0,
        )

        result = analyze_pacing(
            input_path=filepath,
            genre=genre,
            threshold=threshold,
            on_progress=_progress,
        )
        return result


# ---------------------------------------------------------------------------
# POST /composition/saliency-crop — Saliency-guided auto-crop
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/composition/saliency-crop", methods=["POST"])
@require_csrf
@async_job("saliency_crop")
def saliency_crop_route(job_id, filepath, data):
    """Perform saliency-guided auto-crop on a video."""
    from opencut.core.saliency_crop import saliency_crop

    target_aspect = data.get("target_aspect", "9:16").strip()
    sample_interval = safe_float(
        data.get("sample_interval", 2.0), 2.0,
        min_val=0.5, max_val=30.0,
    )
    smoothing = safe_float(
        data.get("smoothing", 0.7), 0.7,
        min_val=0.0, max_val=1.0,
    )
    output_dir = data.get("output_dir", "")

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = saliency_crop(
        input_path=filepath,
        target_aspect=target_aspect,
        sample_interval=sample_interval,
        smoothing=smoothing,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /dubbing/full-pipeline — End-to-end AI dubbing
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/dubbing/full-pipeline", methods=["POST"])
@require_csrf
@async_job("ai_dubbing")
def dubbing_full_pipeline(job_id, filepath, data):
    """Run the complete AI dubbing pipeline."""
    from opencut.core.ai_dubbing import run_dubbing_pipeline

    target_language = data.get("target_language", "").strip()
    if not target_language:
        raise ValueError("target_language is required")

    source_language = data.get("source_language", "").strip()
    voice_reference = data.get("voice_reference", "").strip()
    if voice_reference:
        voice_reference = validate_filepath(voice_reference)

    background_volume = safe_float(
        data.get("background_volume", 0.8), 0.8,
        min_val=0.0, max_val=1.0,
    )
    output_dir = data.get("output_dir", "")

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = run_dubbing_pipeline(
        input_path=filepath,
        target_language=target_language,
        source_language=source_language,
        voice_reference=voice_reference,
        background_volume=background_volume,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /dubbing/isochronous — Time-constrained translation
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/dubbing/isochronous", methods=["POST"])
@require_csrf
@async_job("isochronous_translate", filepath_required=False)
def isochronous_translate_route(job_id, filepath, data):
    """Perform isochronous (time-constrained) translation."""
    from opencut.core.isochronous_translate import translate_isochronous

    segments = data.get("segments")
    if not segments:
        raise ValueError("segments list is required")

    source_language = data.get("source_language", "en").strip()
    target_language = data.get("target_language", "").strip()
    if not target_language:
        raise ValueError("target_language is required")

    tolerance = safe_float(
        data.get("tolerance", 0.10), 0.10,
        min_val=0.01, max_val=0.50,
    )
    max_iterations = safe_int(
        data.get("max_iterations", 5), 5,
        min_val=1, max_val=20,
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = translate_isochronous(
        segments=segments,
        source_language=source_language,
        target_language=target_language,
        tolerance=tolerance,
        max_iterations=max_iterations,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /dubbing/manage-tracks — Multi-language track management
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/dubbing/manage-tracks", methods=["POST"])
@require_csrf
@async_job("multilang_tracks", filepath_required=False)
def manage_tracks_route(job_id, filepath, data):
    """Manage multi-language audio tracks."""
    from opencut.core.multilang_audio import manage_tracks

    operation = data.get("operation", "list").strip()
    input_path = filepath or data.get("input_path", "").strip()

    if not input_path:
        raise ValueError("filepath or input_path is required")
    input_path = validate_filepath(input_path)

    audio_tracks = data.get("audio_tracks")
    if audio_tracks:
        # Validate audio file paths
        for track in audio_tracks:
            if "path" in track:
                track["path"] = validate_filepath(track["path"])

    track_indices = data.get("track_indices")
    labels = data.get("labels")
    track_index = safe_int(data.get("track_index", 0), 0, min_val=0)
    audio_only = safe_bool(data.get("audio_only", False))
    output_dir = data.get("output_dir", "")

    if output_dir:
        output_dir = _resolve_output_dir(input_path, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = manage_tracks(
        input_path=input_path,
        operation=operation,
        audio_tracks=audio_tracks,
        track_indices=track_indices,
        labels=labels,
        track_index=track_index,
        audio_only=audio_only,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /dubbing/emotion-transfer — Prosody-preserving dubbing
# ---------------------------------------------------------------------------
@composition_dubbing_bp.route("/dubbing/emotion-transfer", methods=["POST"])
@require_csrf
@async_job("emotion_voice_dub")
def emotion_transfer_route(job_id, filepath, data):
    """Perform voice dubbing with emotion/prosody preservation."""
    from opencut.core.emotion_voice import emotion_preserving_dub

    target_language = data.get("target_language", "").strip()
    if not target_language:
        raise ValueError("target_language is required")

    source_language = data.get("source_language", "en").strip()
    output_dir = data.get("output_dir", "")

    if output_dir:
        output_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = emotion_preserving_dub(
        input_path=filepath,
        target_language=target_language,
        source_language=source_language,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return result
