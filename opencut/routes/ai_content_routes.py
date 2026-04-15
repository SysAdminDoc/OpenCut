"""
OpenCut AI Content Routes

Endpoints for AI-powered content analysis and optimization:
- Auto color grading (mood, reference, LUT)
- Content moderation scanning
- Pacing & rhythm analysis
- SEO optimization
- Engagement prediction
- Context-aware command suggestions
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, validate_output_path

logger = logging.getLogger("opencut")

ai_content_bp = Blueprint("ai_content", __name__)


# ---------------------------------------------------------------------------
# POST /ai/auto-grade
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/auto-grade", methods=["POST"])
@require_csrf
@async_job("auto_grade")
def ai_auto_grade(job_id, filepath, data):
    """Apply AI auto-color grading to a video."""
    from opencut.core.auto_color import auto_grade

    mood = data.get("mood", "").strip() or None
    reference_image = data.get("reference_image", "").strip() or None
    lut_name = data.get("lut_name", "").strip() or None
    out_path = data.get("output_path", "").strip() or None
    if out_path:
        out_path = validate_output_path(out_path)

    # Validate reference_image if provided
    if reference_image:
        from opencut.security import validate_filepath
        reference_image = validate_filepath(reference_image)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_grade(
        input_path=filepath,
        mood=mood,
        reference_image=reference_image,
        lut_name=lut_name,
        output_path=out_path,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# GET /ai/mood-presets
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/mood-presets", methods=["GET"])
def ai_mood_presets():
    """Return list of available mood presets for auto-color grading."""
    from opencut.core.auto_color import MOOD_PRESETS, list_mood_presets

    presets = []
    for name in list_mood_presets():
        info = MOOD_PRESETS[name]
        presets.append({
            "name": name,
            "description": info["description"],
        })

    return jsonify({"presets": presets, "count": len(presets)})


# ---------------------------------------------------------------------------
# POST /ai/content-scan
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/content-scan", methods=["POST"])
@require_csrf
@async_job("content_scan")
def ai_content_scan(job_id, filepath, data):
    """Scan content for moderation issues."""
    from opencut.core.content_moderation import scan_content

    checks = data.get("checks")  # None means all
    transcript_path = data.get("transcript_path", "").strip() or None

    # Validate transcript_path if provided
    if transcript_path:
        from opencut.security import validate_filepath
        transcript_path = validate_filepath(transcript_path)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = scan_content(
        input_path=filepath,
        checks=checks,
        transcript_path=transcript_path,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /ai/pacing-analysis
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/pacing-analysis", methods=["POST"])
@require_csrf
@async_job("pacing_analysis")
def ai_pacing_analysis(job_id, filepath, data):
    """Analyze video edit pacing and rhythm."""
    from opencut.core.pacing_analysis import analyze_pacing

    genre = data.get("genre", "general").strip()
    threshold = safe_float(data.get("threshold"), 0.3, 0.05, 0.95)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = analyze_pacing(
        input_path=filepath,
        genre=genre,
        threshold=threshold,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /ai/seo-optimize
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/seo-optimize", methods=["POST"])
@require_csrf
@async_job("seo_optimize", filepath_required=False)
def ai_seo_optimize(job_id, filepath, data):
    """Generate SEO-optimized metadata for video content."""
    from opencut.core.seo_optimizer import optimize_seo

    transcript_text = data.get("transcript_text", "").strip()
    if not transcript_text:
        raise ValueError("transcript_text is required for SEO optimization")

    platform = data.get("platform", "youtube").strip()
    custom_context = data.get("custom_context", "").strip()

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = optimize_seo(
        transcript_text=transcript_text,
        platform=platform,
        custom_context=custom_context,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /ai/engagement-predict
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/engagement-predict", methods=["POST"])
@require_csrf
@async_job("engagement_predict")
def ai_engagement_predict(job_id, filepath, data):
    """Predict video engagement metrics."""
    from opencut.core.engagement_predict import predict_engagement

    transcript_text = data.get("transcript_text", "").strip() or None

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = predict_engagement(
        input_path=filepath,
        transcript_text=transcript_text,
        on_progress=_on_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /ai/suggest (synchronous -- fast rules engine)
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/suggest", methods=["POST"])
@require_csrf
def ai_suggest():
    """Get context-aware command suggestions based on clip metadata."""
    from opencut.core.context_suggest import get_suggestions

    data = request.get_json(force=True) or {}
    clip_metadata = data.get("clip_metadata")
    if not clip_metadata or not isinstance(clip_metadata, dict):
        return jsonify({"error": "clip_metadata dict is required"}), 400

    recent_actions = data.get("recent_actions", [])
    if not isinstance(recent_actions, list):
        recent_actions = []

    suggestions = get_suggestions(
        clip_metadata=clip_metadata,
        recent_actions=recent_actions,
    )

    return jsonify({"suggestions": suggestions, "count": len(suggestions)})


# ---------------------------------------------------------------------------
# POST /ai/score-title (synchronous utility)
# ---------------------------------------------------------------------------
@ai_content_bp.route("/ai/score-title", methods=["POST"])
@require_csrf
def ai_score_title():
    """Score a video title for SEO effectiveness."""
    from opencut.core.seo_optimizer import score_title

    data = request.get_json(force=True) or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "title is required"}), 400

    platform = data.get("platform", "youtube").strip()

    result = score_title(title, platform=platform)
    return jsonify(result)
