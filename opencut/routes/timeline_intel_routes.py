"""
OpenCut AI Timeline Intelligence Routes

Endpoints for Category 68 — AI-powered timeline intelligence features:
- Timeline quality analysis
- Timeline engagement scoring
- Narrative clip assembly
- Natural language color grading
- Auto-dubbing pipeline
"""

import logging

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, validate_filepath, validate_path

logger = logging.getLogger("opencut")

timeline_intel_bp = Blueprint("timeline_intel", __name__)


# ---------------------------------------------------------------------------
# POST /api/timeline/quality
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/timeline/quality", methods=["POST"])
@require_csrf
@async_job("timeline_quality")
def timeline_quality(job_id, filepath, data):
    """Analyze holistic timeline quality: color, audio, pacing, continuity."""
    from opencut.core.timeline_quality import analyze_timeline_quality

    scene_threshold = safe_float(data.get("scene_threshold"), 0.3, 0.0, 1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = analyze_timeline_quality(
        video_path=filepath,
        scene_threshold=scene_threshold,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/score
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/timeline/score", methods=["POST"])
@require_csrf
@async_job("timeline_score")
def timeline_score(job_id, filepath, data):
    """Score engagement across timeline segments."""
    from opencut.core.timeline_score import score_timeline

    segment_duration = safe_float(data.get("segment_duration"), 10.0, 1.0, 300.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = score_timeline(
        video_path=filepath,
        segment_duration=segment_duration,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/timeline/narrative
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/timeline/narrative", methods=["POST"])
@require_csrf
@async_job("clip_narrative", filepath_required=False)
def timeline_narrative(job_id, filepath, data):
    """Build narrative assembly from multiple clips."""
    from opencut.core.clip_narrative import NARRATIVE_STYLES, build_narrative

    clip_paths = data.get("clip_paths", [])
    if not clip_paths or not isinstance(clip_paths, list):
        raise ValueError("'clip_paths' must be a non-empty list of file paths")

    # Validate each clip path
    validated = []
    for p in clip_paths:
        if isinstance(p, str) and p.strip():
            validated.append(validate_filepath(p.strip()))

    if not validated:
        raise ValueError("No valid clip paths provided")

    style = data.get("style", "documentary")
    if style not in NARRATIVE_STYLES:
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(NARRATIVE_STYLES.keys())}")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = build_narrative(
        clip_paths=validated,
        style=style,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# POST /api/video/color-intent
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/video/color-intent", methods=["POST"])
@require_csrf
@async_job("color_intent")
def video_color_intent(job_id, filepath, data):
    """Apply natural language color grading to video."""
    from opencut.core.ai_color_intent import apply_color_intent

    intent = data.get("intent", "warm sunset")
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("'intent' must be a non-empty string")
    intent = intent.strip()

    intensity = safe_float(data.get("intensity"), 1.0, 0.0, 2.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = apply_color_intent(
        video_path=filepath,
        intent=intent,
        intensity=intensity,
        output_dir=output_dir,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# GET /api/video/color-intents
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/video/color-intents", methods=["GET"])
def list_color_intents():
    """List all available color intent presets."""
    from opencut.core.ai_color_intent import COLOR_INTENTS

    intents = []
    for name, data in COLOR_INTENTS.items():
        intents.append({
            "name": name,
            "description": data.get("description", ""),
            "category": data.get("category", ""),
        })

    # Group by category
    categories = {}
    for item in intents:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)

    return jsonify({
        "intents": intents,
        "categories": categories,
        "total": len(intents),
    })


# ---------------------------------------------------------------------------
# POST /api/video/color-intent/preview
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/video/color-intent/preview", methods=["POST"])
@require_csrf
@async_job("color_intent_preview")
def video_color_intent_preview(job_id, filepath, data):
    """Preview color intent on a single frame."""
    from opencut.core.ai_color_intent import preview_color_intent

    intent = data.get("intent", "warm sunset")
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("'intent' must be a non-empty string")
    intent = intent.strip()

    intensity = safe_float(data.get("intensity"), 1.0, 0.0, 2.0)
    timestamp = safe_float(data.get("timestamp"), -1.0, -1.0)

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = preview_color_intent(
        video_path=filepath,
        intent=intent,
        intensity=intensity,
        timestamp=timestamp,
        on_progress=_progress,
    )

    return result


# ---------------------------------------------------------------------------
# POST /api/audio/auto-dub
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/audio/auto-dub", methods=["POST"])
@require_csrf
@async_job("auto_dub_pipeline")
def audio_auto_dub(job_id, filepath, data):
    """Run end-to-end auto-dubbing pipeline."""
    from opencut.core.auto_dub_pipeline import (
        SUPPORTED_LANGUAGES,
        DubConfig,
        auto_dub,
    )

    target_language = data.get("target_language", "es")
    if not isinstance(target_language, str) or target_language.strip() not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{target_language}'. "
            f"Use GET /api/audio/auto-dub/languages for the full list."
        )
    target_language = target_language.strip()

    config = DubConfig(
        target_language=target_language,
        source_language=data.get("source_language", ""),
        whisper_model=data.get("whisper_model", "base"),
        voice_clone=bool(data.get("voice_clone", True)),
        lip_sync=bool(data.get("lip_sync", True)),
        preserve_music=bool(data.get("preserve_music", True)),
        tts_engine=data.get("tts_engine", "edge"),
        output_dir=data.get("output_dir", ""),
    )

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = auto_dub(
        video_path=filepath,
        target_language=target_language,
        config=config,
        on_progress=_progress,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# GET /api/audio/auto-dub/languages
# ---------------------------------------------------------------------------
@timeline_intel_bp.route("/api/audio/auto-dub/languages", methods=["GET"])
def list_auto_dub_languages():
    """List supported languages for auto-dubbing."""
    from opencut.core.auto_dub_pipeline import LANGUAGE_NAMES, SUPPORTED_LANGUAGES

    languages = []
    for code in SUPPORTED_LANGUAGES:
        languages.append({
            "code": code,
            "name": LANGUAGE_NAMES.get(code, code),
        })

    return jsonify({
        "languages": languages,
        "total": len(languages),
    })
