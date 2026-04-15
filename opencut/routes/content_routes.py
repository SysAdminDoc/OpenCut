"""
OpenCut Content & Provenance Routes

Provenance manifest generation, social caption generation,
usage analytics dashboard, and podcast RSS feed generation.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.jobs import async_job
from opencut.security import require_csrf, safe_int, validate_output_path

logger = logging.getLogger("opencut")

content_bp = Blueprint("content", __name__)


# ---------------------------------------------------------------------------
# Provenance — async
# ---------------------------------------------------------------------------
@content_bp.route("/provenance/generate", methods=["POST"])
@require_csrf
@async_job("provenance")
def provenance_generate(job_id, filepath, data):
    """Generate a provenance manifest for the given file."""
    from opencut.core.provenance import generate_provenance_manifest
    from opencut.jobs import _update_job

    output = data.get("output_path", None) or None
    if output:
        output = validate_output_path(output)

    _update_job(job_id, progress=10, message="Computing file hash...")

    manifest = generate_provenance_manifest(filepath, output_path=output)

    _update_job(job_id, progress=100, message="Provenance manifest generated")
    return manifest


# ---------------------------------------------------------------------------
# Social Captions — async
# ---------------------------------------------------------------------------
@content_bp.route("/social/generate-captions", methods=["POST"])
@require_csrf
@async_job("social_captions", filepath_required=False)
def social_generate_captions(job_id, filepath, data):
    """Generate social media captions from transcript text."""
    from opencut.core.social_captions import generate_social_captions
    from opencut.jobs import _update_job

    transcript = data.get("transcript", "").strip()
    if not transcript:
        raise ValueError("No transcript text provided")

    platform = data.get("platform", "youtube")
    custom_instructions = data.get("custom_instructions", "")

    def _progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = generate_social_captions(
        transcript_text=transcript,
        platform=platform,
        custom_instructions=custom_instructions,
        on_progress=_progress,
    )

    return {
        "title": result.title,
        "description": result.description,
        "hashtags": result.hashtags,
        "tags": result.tags,
        "platform": result.platform,
    }


# ---------------------------------------------------------------------------
# Analytics — SYNC
# ---------------------------------------------------------------------------
@content_bp.route("/analytics/usage", methods=["GET"])
def analytics_usage():
    """Return aggregated usage statistics."""
    try:
        from opencut.core.analytics import get_usage_stats

        days = safe_int(request.args.get("days", 30), 30, min_val=1, max_val=365)
        stats = get_usage_stats(days=days)
        return jsonify(stats)
    except Exception as exc:
        return safe_error(exc, "analytics_usage")


@content_bp.route("/analytics/feature/<path:endpoint>", methods=["GET"])
def analytics_feature(endpoint):
    """Return detailed stats for a single feature endpoint."""
    try:
        from opencut.core.analytics import get_feature_stats

        days = safe_int(request.args.get("days", 30), 30, min_val=1, max_val=365)
        stats = get_feature_stats(endpoint=f"/{endpoint}", days=days)
        return jsonify(stats)
    except Exception as exc:
        return safe_error(exc, "analytics_feature")


# ---------------------------------------------------------------------------
# Podcast RSS — SYNC
# ---------------------------------------------------------------------------
@content_bp.route("/podcast/generate-rss", methods=["POST"])
@require_csrf
def podcast_generate_rss():
    """Generate a podcast RSS feed from episode data."""
    try:
        from opencut.core.podcast_rss import generate_podcast_rss

        data = request.get_json(force=True) or {}

        episodes = data.get("episodes", [])
        feed_metadata = data.get("feed_metadata", {})
        output_path = data.get("output_path", None)
        if output_path:
            output_path = validate_output_path(output_path)

        if not episodes:
            return jsonify({"error": "No episodes provided"}), 400
        if not feed_metadata:
            return jsonify({"error": "No feed_metadata provided"}), 400

        xml = generate_podcast_rss(
            episodes=episodes,
            feed_metadata=feed_metadata,
            output_path=output_path,
        )

        return jsonify({"rss_xml": xml, "episode_count": len(episodes)})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as exc:
        return safe_error(exc, "podcast_generate_rss")
