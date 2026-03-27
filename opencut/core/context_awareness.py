"""
OpenCut Context Awareness

Scores features by relevance based on active clip metadata from Premiere Pro.
"""

import logging

logger = logging.getLogger("opencut")


# Each feature declares what it needs/prefers
FEATURE_RELEVANCE = {
    # Cut & Clean tab features
    "silence_detect": {
        "requires": ["audio"],
        "suggests": ["talking_head", "long_duration"],
        "irrelevant": ["image_only", "music_only"],
        "tab": "cut",
        "label": "Silence Detection",
    },
    "silence_speedup": {
        "requires": ["audio"],
        "suggests": ["talking_head", "long_duration"],
        "irrelevant": ["image_only", "music_only"],
        "tab": "cut",
        "label": "Speed Up Silence",
    },
    "filler_detect": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["image_only", "music_only"],
        "tab": "cut",
        "label": "Filler Word Detection",
    },
    "repeat_detect": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["image_only", "music_only"],
        "tab": "cut",
        "label": "Repeated Take Detection",
    },
    "scene_detect": {
        "requires": ["video"],
        "suggests": ["long_duration", "multicam"],
        "irrelevant": ["audio_only"],
        "tab": "cut",
        "label": "Scene Detection",
    },
    "auto_edit": {
        "requires": ["video", "audio"],
        "suggests": ["talking_head", "long_duration"],
        "irrelevant": ["image_only"],
        "tab": "cut",
        "label": "Auto Edit",
    },
    "highlights": {
        "requires": ["audio"],
        "suggests": ["talking_head", "long_duration"],
        "irrelevant": ["music_only", "image_only"],
        "tab": "cut",
        "label": "Highlight Extraction",
    },
    # Captions tab features
    "transcribe": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["music_only", "image_only"],
        "tab": "captions",
        "label": "Transcribe",
    },
    "styled_captions": {
        "requires": ["audio", "video"],
        "suggests": ["talking_head", "short_duration"],
        "irrelevant": ["audio_only"],
        "tab": "captions",
        "label": "Styled Captions",
    },
    "translate": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["music_only", "image_only"],
        "tab": "captions",
        "label": "Translate",
    },
    "chapters": {
        "requires": ["audio"],
        "suggests": ["long_duration", "talking_head"],
        "irrelevant": ["short_duration", "music_only"],
        "tab": "captions",
        "label": "YouTube Chapters",
    },
    # Audio tab features
    "denoise": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["image_only"],
        "tab": "audio",
        "label": "Denoise",
    },
    "normalize": {
        "requires": ["audio"],
        "suggests": [],
        "irrelevant": ["image_only"],
        "tab": "audio",
        "label": "Normalize Audio",
    },
    "stem_separate": {
        "requires": ["audio"],
        "suggests": ["music_only"],
        "irrelevant": ["image_only"],
        "tab": "audio",
        "label": "Stem Separation",
    },
    "audio_enhance": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["image_only", "music_only"],
        "tab": "audio",
        "label": "Audio Enhance",
    },
    "tts": {
        "requires": [],
        "suggests": [],
        "irrelevant": [],
        "tab": "audio",
        "label": "Text-to-Speech",
    },
    "sfx": {
        "requires": [],
        "suggests": [],
        "irrelevant": [],
        "tab": "audio",
        "label": "Sound Effects",
    },
    "ducking": {
        "requires": ["audio"],
        "suggests": ["talking_head"],
        "irrelevant": ["image_only"],
        "tab": "audio",
        "label": "Audio Ducking",
    },
    "loudness_match": {
        "requires": ["audio"],
        "suggests": [],
        "irrelevant": ["image_only"],
        "tab": "audio",
        "label": "Loudness Match",
    },
    "beat_detect": {
        "requires": ["audio"],
        "suggests": ["music_only"],
        "irrelevant": ["image_only"],
        "tab": "audio",
        "label": "Beat Detection",
    },
    # Video tab features
    "stabilize": {
        "requires": ["video"],
        "suggests": ["handheld"],
        "irrelevant": ["audio_only", "image_only"],
        "tab": "video",
        "label": "Stabilize",
    },
    "chromakey": {
        "requires": ["video"],
        "suggests": ["green_screen"],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Chroma Key",
    },
    "color_correct": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Color Correction",
    },
    "color_match": {
        "requires": ["video"],
        "suggests": ["multicam"],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Color Match",
    },
    "lut_apply": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Apply LUT",
    },
    "reframe": {
        "requires": ["video"],
        "suggests": ["talking_head", "vertical_output"],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Video Reframe",
    },
    "face_reframe": {
        "requires": ["video"],
        "suggests": ["talking_head", "vertical_output"],
        "irrelevant": ["audio_only", "music_only"],
        "tab": "video",
        "label": "Face Reframe",
    },
    "auto_zoom": {
        "requires": ["video"],
        "suggests": ["talking_head"],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Auto Zoom",
    },
    "upscale": {
        "requires": ["video"],
        "suggests": ["low_resolution"],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Upscale",
    },
    "speed_ramp": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Speed Ramp",
    },
    "transitions": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Transitions",
    },
    "pip": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Picture-in-Picture",
    },
    "bg_remove": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Background Removal",
    },
    "watermark_remove": {
        "requires": ["video"],
        "suggests": [],
        "irrelevant": ["audio_only"],
        "tab": "video",
        "label": "Watermark Removal",
    },
    "multicam": {
        "requires": ["audio"],
        "suggests": ["multicam", "talking_head"],
        "irrelevant": ["image_only"],
        "tab": "video",
        "label": "Multicam Switching",
    },
    "shorts_pipeline": {
        "requires": ["video", "audio"],
        "suggests": ["talking_head", "long_duration", "vertical_output"],
        "irrelevant": ["image_only"],
        "tab": "video",
        "label": "Shorts Pipeline",
    },
}


def classify_clip(metadata):
    """Classify a clip based on its metadata into content type tags.

    Args:
        metadata: dict with keys like has_audio, has_video, duration,
                  codec, frame_rate, width, height, track_type, num_audio_channels

    Returns:
        set of string tags: audio_only, video_only, image_only,
        talking_head, music_only, long_duration, short_duration,
        low_resolution, multicam, vertical_output, handheld, green_screen
    """
    tags = set()

    has_audio = metadata.get("has_audio", False)
    has_video = metadata.get("has_video", False)
    duration = metadata.get("duration", 0)
    width = metadata.get("width", 0)
    height = metadata.get("height", 0)
    num_audio_channels = metadata.get("num_audio_channels", 2)

    # Basic type classification
    if has_audio and not has_video:
        tags.add("audio_only")
    if has_video and not has_audio:
        tags.add("video_only")
    if has_video and duration < 0.1:
        tags.add("image_only")

    # Duration classification
    if duration > 300:  # > 5 minutes
        tags.add("long_duration")
    elif duration < 60:  # < 1 minute
        tags.add("short_duration")

    # Resolution classification
    if has_video and width > 0 and height > 0:
        if max(width, height) < 720:
            tags.add("low_resolution")
        if height > width:
            tags.add("vertical_output")

    # Content hints from metadata
    if has_audio and has_video and num_audio_channels <= 2:
        # Likely talking head if has audio+video and is not multichannel
        tags.add("talking_head")

    if has_audio and not has_video and num_audio_channels >= 2:
        if duration > 30:
            tags.add("music_only")

    return tags


def score_features(clip_tags, features=None):
    """Score all features based on clip classification tags.

    Args:
        clip_tags: set of string tags from classify_clip()
        features: optional dict of feature definitions (defaults to FEATURE_RELEVANCE)

    Returns:
        list of dicts sorted by score descending:
        [{"id": "silence_detect", "label": "...", "tab": "cut", "score": 85, "relevant": True}, ...]
    """
    if features is None:
        features = FEATURE_RELEVANCE

    results = []
    for feature_id, feature in features.items():
        score = 50  # Base score

        requires = set(feature.get("requires", []))
        suggests = set(feature.get("suggests", []))
        irrelevant = set(feature.get("irrelevant", []))

        # Check hard requirements — if clip doesn't have what's required, low score
        if requires:
            # Requirements like "audio" are met if clip has audio
            # We need to check capabilities, not content tags
            # audio_only means has_audio, video_only means has_video
            has_audio_tag = "audio_only" in clip_tags or (
                "image_only" not in clip_tags and "video_only" not in clip_tags
            )
            has_video_tag = "video_only" in clip_tags or (
                "image_only" not in clip_tags and "audio_only" not in clip_tags
            )

            for req in requires:
                if req == "audio" and not has_audio_tag:
                    score -= 40
                elif req == "video" and not has_video_tag:
                    score -= 40

        # Boost for matching suggestions
        matched_suggests = suggests & clip_tags
        score += len(matched_suggests) * 15

        # Penalty for irrelevant matches
        matched_irrelevant = irrelevant & clip_tags
        score -= len(matched_irrelevant) * 25

        # Clamp to 0-100
        score = max(0, min(100, score))

        results.append({
            "id": feature_id,
            "label": feature.get("label", feature_id),
            "tab": feature.get("tab", ""),
            "score": score,
            "relevant": score >= 30,
        })

    # Sort by score descending, then alphabetically
    results.sort(key=lambda x: (-x["score"], x["label"]))
    return results


def get_guidance_message(clip_tags, top_features):
    """Generate a contextual guidance message based on clip type.

    Args:
        clip_tags: set of tags from classify_clip()
        top_features: list of scored features (from score_features)

    Returns:
        str guidance message, or empty string if no specific guidance
    """
    if not clip_tags:
        return "Select a clip in your timeline to see relevant tools."

    if "image_only" in clip_tags:
        return "Still image detected — video effects like color correction and reframe are most relevant."

    if "audio_only" in clip_tags:
        if "music_only" in clip_tags:
            return "Music track detected — try Stem Separation, Beat Detection, or Loudness Match."
        return "Audio clip detected — try Denoise, Normalize, or Enhance for quick cleanup."

    if "talking_head" in clip_tags:
        if "long_duration" in clip_tags:
            return "Interview/talking-head clip detected — try Clean Up for silence removal, or Auto Subtitle."
        if "short_duration" in clip_tags:
            return "Short talking clip — try Styled Captions or Face Reframe for social media."
        return "Talking-head clip detected — Silence Detection, Captions, and Denoise are recommended."

    if "video_only" in clip_tags:
        return "Video-only clip (no audio) — Color Correction, Stabilize, and Transitions are most useful."

    if "long_duration" in clip_tags:
        return "Long clip detected — try Scene Detection, Highlights, or YouTube Chapters."

    # Default
    relevant_count = sum(1 for f in top_features if f["relevant"])
    if relevant_count > 0:
        top = top_features[0]
        return f"{relevant_count} relevant tools available — try {top['label']} to get started."

    return ""
