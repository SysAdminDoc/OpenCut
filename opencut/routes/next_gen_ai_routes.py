"""
OpenCut Next-Gen AI Routes

Endpoints for video LLM queries, AI music remixing, audio classification,
and AI color matching between shots.
"""

import logging
import os

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

next_gen_ai_bp = Blueprint("next_gen_ai", __name__)


# ---------------------------------------------------------------------------
# POST /ai/video-llm/query -- ask a question about video content (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/video-llm/query", methods=["POST"])
@require_csrf
@async_job("video_llm_query")
def video_llm_query_route(job_id, filepath, data):
    """Query video content with natural language via multimodal LLM."""
    from opencut.core.video_llm import query_video

    question = data.get("question", "")
    if not question or not question.strip():
        raise ValueError("Question is required")

    model = data.get("model", "auto")
    max_frames = safe_int(data.get("max_frames", 16), 16, min_val=1, max_val=64)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = query_video(
        filepath,
        question=question,
        model=model,
        max_frames=max_frames,
        on_progress=_progress,
    )

    return {
        "answer": result.answer,
        "timestamps": result.timestamps,
        "confidence": result.confidence,
        "frames_analyzed": result.frames_analyzed,
        "model_used": result.model_used,
    }


# ---------------------------------------------------------------------------
# POST /ai/video-llm/find-moment -- find specific moment in video (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/video-llm/find-moment", methods=["POST"])
@require_csrf
@async_job("video_llm_find_moment")
def video_llm_find_moment_route(job_id, filepath, data):
    """Find specific moments in video matching a natural language description."""
    from opencut.core.video_llm import find_moments

    description = data.get("description", "")
    if not description or not description.strip():
        raise ValueError("Description is required")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    moments = find_moments(
        filepath,
        description=description,
        on_progress=_progress,
    )

    return {
        "moments": [
            {
                "timestamp": m.timestamp,
                "confidence": m.confidence,
                "description": m.description,
                "frame_index": m.frame_index,
            }
            for m in moments
        ],
        "count": len(moments),
    }


# ---------------------------------------------------------------------------
# POST /ai/music/remix -- remix/adjust music duration (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/music/remix", methods=["POST"])
@require_csrf
@async_job("music_remix")
def music_remix_route(job_id, filepath, data):
    """Remix or adjust music track duration using beat-aware processing."""
    from opencut.core.music_remix import fit_music_to_duration

    target_duration = safe_float(data.get("target_duration", 0), 0, min_val=0.1)
    if target_duration <= 0:
        raise ValueError("target_duration is required and must be positive")

    mode = data.get("mode", "smart")
    if mode not in ("smart", "stretch", "fade"):
        mode = "smart"

    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(effective_dir, f"{base}_remix_{mode}.aac")

    result = fit_music_to_duration(
        filepath,
        target_duration=target_duration,
        output_path=out_path,
        mode=mode,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "original_duration": result.original_duration,
        "target_duration": result.target_duration,
        "actual_duration": result.actual_duration,
        "mode_used": result.mode_used,
        "edit_points": result.edit_points,
    }


# ---------------------------------------------------------------------------
# POST /ai/music/fit-duration -- fit music to target duration (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/music/fit-duration", methods=["POST"])
@require_csrf
@async_job("music_fit_duration")
def music_fit_duration_route(job_id, filepath, data):
    """Fit music to a specific target duration with mode selection."""
    from opencut.core.music_remix import fit_music_to_duration

    target_duration = safe_float(data.get("target_duration", 0), 0, min_val=0.1)
    if target_duration <= 0:
        raise ValueError("target_duration is required and must be positive")

    mode = data.get("mode", "smart")
    if mode not in ("smart", "stretch", "fade"):
        mode = "smart"

    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(effective_dir, f"{base}_fit_{mode}.aac")

    result = fit_music_to_duration(
        filepath,
        target_duration=target_duration,
        output_path=out_path,
        mode=mode,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "original_duration": result.original_duration,
        "target_duration": result.target_duration,
        "actual_duration": result.actual_duration,
        "mode_used": result.mode_used,
        "edit_points": result.edit_points,
    }


# ---------------------------------------------------------------------------
# POST /ai/audio/classify -- classify audio category (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/audio/classify", methods=["POST"])
@require_csrf
@async_job("audio_classify")
def audio_classify_route(job_id, filepath, data):
    """Classify audio into categories: speech, music, SFX, ambience, silence."""
    from opencut.core.audio_category import classify_audio

    segment_duration = safe_float(
        data.get("segment_duration", 2.0), 2.0, min_val=0.5, max_val=30.0
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = classify_audio(
        filepath,
        segment_duration=segment_duration,
        on_progress=_progress,
    )

    return {
        "segments": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "category": s.category,
                "confidence": s.confidence,
            }
            for s in result.segments
        ],
        "summary": result.summary,
    }


# ---------------------------------------------------------------------------
# POST /ai/audio/classify-timeline -- classify audio across timeline (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/audio/classify-timeline", methods=["POST"])
@require_csrf
@async_job("audio_classify_timeline")
def audio_classify_timeline_route(job_id, filepath, data):
    """Classify audio segments across the full timeline."""
    from opencut.core.audio_category import classify_audio

    segment_duration = safe_float(
        data.get("segment_duration", 1.0), 1.0, min_val=0.5, max_val=30.0
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = classify_audio(
        filepath,
        segment_duration=segment_duration,
        on_progress=_progress,
    )

    # Build timeline-oriented response with contiguous category spans
    timeline = []
    if result.segments:
        current_cat = result.segments[0].category
        span_start = result.segments[0].start_time
        span_end = result.segments[0].end_time
        span_conf_sum = result.segments[0].confidence
        span_count = 1

        for seg in result.segments[1:]:
            if seg.category == current_cat:
                span_end = seg.end_time
                span_conf_sum += seg.confidence
                span_count += 1
            else:
                timeline.append({
                    "start_time": span_start,
                    "end_time": span_end,
                    "category": current_cat,
                    "confidence": round(span_conf_sum / span_count, 3),
                    "duration": round(span_end - span_start, 3),
                })
                current_cat = seg.category
                span_start = seg.start_time
                span_end = seg.end_time
                span_conf_sum = seg.confidence
                span_count = 1

        timeline.append({
            "start_time": span_start,
            "end_time": span_end,
            "category": current_cat,
            "confidence": round(span_conf_sum / span_count, 3),
            "duration": round(span_end - span_start, 3),
        })

    return {
        "timeline": timeline,
        "summary": result.summary,
        "total_segments": len(result.segments),
        "merged_spans": len(timeline),
    }


# ---------------------------------------------------------------------------
# POST /ai/color-match -- match color between shots (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/color-match", methods=["POST"])
@require_csrf
@async_job("color_match")
def color_match_route(job_id, filepath, data):
    """Match color/exposure of source to a reference video."""
    from opencut.core.color_match_shots import match_color

    reference_path = data.get("reference_path", "")
    if not reference_path:
        raise ValueError("reference_path is required")
    reference_path = validate_filepath(reference_path)

    strength = safe_float(data.get("strength", 1.0), 1.0, min_val=0.0, max_val=2.0)
    output_dir = data.get("output_dir", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = _resolve_output_dir(filepath, output_dir)
    base = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1] or ".mp4"
    out_path = os.path.join(effective_dir, f"{base}_colormatch{ext}")

    result = match_color(
        filepath,
        reference_path=reference_path,
        output_path=out_path,
        strength=strength,
        on_progress=_progress,
    )

    return {
        "output_path": result.output_path,
        "reference_stats": result.reference_stats,
        "matched_stats": result.matched_stats,
        "adjustments": result.adjustments,
    }


# ---------------------------------------------------------------------------
# POST /ai/color-match/batch -- batch color match multiple clips (async)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/color-match/batch", methods=["POST"])
@require_csrf
@async_job("color_match_batch")
def color_match_batch_route(job_id, filepath, data):
    """Batch color match multiple clips to one reference."""
    from opencut.core.color_match_shots import batch_match

    reference_path = data.get("reference_path", "")
    if not reference_path:
        raise ValueError("reference_path is required")
    reference_path = validate_filepath(reference_path)

    additional_paths = data.get("additional_paths", [])
    if isinstance(additional_paths, str):
        additional_paths = [p.strip() for p in additional_paths.split(",") if p.strip()]

    # Combine the main filepath with additional paths for batch
    all_paths = [filepath] + [validate_filepath(p) for p in additional_paths]

    output_dir = data.get("output_dir", "")
    effective_dir = _resolve_output_dir(filepath, output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    results = batch_match(
        all_paths,
        reference_path=reference_path,
        output_dir=effective_dir,
        on_progress=_progress,
    )

    return {
        "results": [
            {
                "output_path": r.output_path,
                "adjustments": r.adjustments,
            }
            for r in results
        ],
        "count": len(results),
        "reference_stats": results[0].reference_stats if results else {},
    }


# ---------------------------------------------------------------------------
# POST /ai/color-match/analyze -- analyze color stats only (sync)
# ---------------------------------------------------------------------------
@next_gen_ai_bp.route("/ai/color-match/analyze", methods=["POST"])
@require_csrf
def color_match_analyze_route():
    """Analyze color statistics of a video clip (synchronous)."""
    from opencut.core.color_match_shots import analyze_color_stats

    data = request.get_json(force=True, silent=True) or {}
    filepath = data.get("filepath", "")
    if not filepath:
        return jsonify({"error": "filepath is required"}), 400

    try:
        filepath = validate_filepath(filepath)
    except Exception as exc:
        return jsonify(safe_error(exc, "color_match_analyze")), 400

    sample_seconds = safe_float(data.get("sample_seconds", 3.0), 3.0,
                                min_val=0.5, max_val=30.0)

    try:
        stats = analyze_color_stats(filepath, sample_seconds=sample_seconds)
        return jsonify({"stats": stats, "filepath": filepath})
    except Exception as exc:
        return jsonify(safe_error(exc, "color_match_analyze")), 500
