"""
OpenCut Context Awareness Routes

Analyzes clip metadata to provide contextual feature recommendations.
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.security import require_csrf, safe_bool, safe_float, safe_int

logger = logging.getLogger("opencut")

context_bp = Blueprint("context", __name__)

try:
    from ..core.context_awareness import classify_clip, get_guidance_message, score_features
except ImportError:
    from opencut.core.context_awareness import classify_clip, get_guidance_message, score_features
@context_bp.route("/context/analyze", methods=["POST"])
@require_csrf
def analyze_clip_context():
    """Analyze clip metadata and return feature relevance scores.

    Expects JSON body:
    {
        "has_audio": true/false,
        "has_video": true/false,
        "duration": 120.5,
        "width": 1920,
        "height": 1080,
        "frame_rate": 29.97,
        "codec": "h264",
        "num_audio_channels": 2,
        "track_type": "video"
    }

    Returns:
    {
        "tags": ["talking_head", "long_duration"],
        "features": [{"id": "...", "label": "...", "tab": "...", "score": 85, "relevant": true}, ...],
        "guidance": "Interview clip detected — try Clean Up for quick results.",
        "tab_scores": {"cut": 75, "captions": 80, "audio": 60, "video": 45}
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    metadata = {
        "has_audio": safe_bool(data.get("has_audio", False), False),
        "has_video": safe_bool(data.get("has_video", False), False),
        "duration": safe_float(data.get("duration", 0), min_val=0),
        "width": safe_int(data.get("width", 0), min_val=0),
        "height": safe_int(data.get("height", 0), min_val=0),
        "frame_rate": safe_float(data.get("frame_rate", 0), min_val=0),
        "codec": str(data.get("codec", "")),
        "num_audio_channels": safe_int(data.get("num_audio_channels", 2), min_val=0),
        "track_type": str(data.get("track_type", "")),
    }

    # Classify and score
    tags = classify_clip(metadata)
    features = score_features(tags)
    guidance = get_guidance_message(tags, features)

    # Aggregate per-tab scores (average of relevant features in each tab)
    tab_scores = {}
    tab_counts = {}
    for f in features:
        tab = f["tab"]
        if tab:
            tab_scores[tab] = tab_scores.get(tab, 0) + f["score"]
            tab_counts[tab] = tab_counts.get(tab, 0) + 1

    for tab in tab_scores:
        tab_scores[tab] = round(tab_scores[tab] / tab_counts[tab])

    return jsonify({
        "tags": sorted(tags),
        "features": features,
        "guidance": guidance,
        "tab_scores": tab_scores,
    })


@context_bp.route("/context/features", methods=["GET"])
def list_features():
    """List all registered features with their relevance metadata.

    Useful for the frontend to know what features exist without analyzing a clip.
    """
    from opencut.core.context_awareness import FEATURE_RELEVANCE

    features = []
    for fid, fdata in FEATURE_RELEVANCE.items():
        features.append({
            "id": fid,
            "label": fdata.get("label", fid),
            "tab": fdata.get("tab", ""),
            "requires": fdata.get("requires", []),
            "suggests": fdata.get("suggests", []),
        })

    return jsonify({"features": features})
