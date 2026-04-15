"""
OpenCut Smart Content Routing

Analyzes video characteristics and classifies content type to automatically
suggest optimal workflows and configure operation parameters.
"""

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import get_ffprobe_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Content type definitions
# ---------------------------------------------------------------------------
CONTENT_TYPES: Dict[str, Dict[str, Any]] = {
    "interview": {
        "label": "Interview / Talking Head",
        "description": "Single or multi-person interview with primarily static camera",
        "indicators": {
            "motion_level": "low",
            "face_count": "1-3",
            "duration_range": "120-7200",
            "audio_channels": "1-2",
        },
    },
    "vlog": {
        "label": "Vlog / Personal Video",
        "description": "Personal video with mixed indoor/outdoor, moderate motion",
        "indicators": {
            "motion_level": "medium",
            "face_count": "1-2",
            "duration_range": "180-1800",
            "audio_channels": "1-2",
        },
    },
    "tutorial": {
        "label": "Tutorial / How-To",
        "description": "Screen recording or instructional content with narration",
        "indicators": {
            "motion_level": "low",
            "face_count": "0-1",
            "duration_range": "300-3600",
            "resolution_hint": ">=1080p",
        },
    },
    "music_video": {
        "label": "Music Video",
        "description": "High-motion, short-form creative content with music focus",
        "indicators": {
            "motion_level": "high",
            "duration_range": "120-420",
            "audio_channels": "2",
        },
    },
    "podcast": {
        "label": "Podcast / Audio-First",
        "description": "Long-form audio-centric content, often static video",
        "indicators": {
            "motion_level": "very_low",
            "duration_range": "1800-14400",
            "face_count": "1-4",
        },
    },
    "gaming": {
        "label": "Gaming / Screen Capture",
        "description": "Screen capture of gameplay with possible facecam overlay",
        "indicators": {
            "motion_level": "high",
            "duration_range": "300-14400",
            "resolution_hint": ">=1080p",
            "fps_hint": ">=30",
        },
    },
    "drone": {
        "label": "Drone / Aerial Footage",
        "description": "Aerial footage with smooth motion, wide aspect, high resolution",
        "indicators": {
            "motion_level": "medium",
            "resolution_hint": ">=2160p",
            "duration_range": "30-600",
        },
    },
    "timelapse": {
        "label": "Timelapse / Hyperlapse",
        "description": "Time-compressed footage with very low effective framerate",
        "indicators": {
            "motion_level": "very_low",
            "duration_range": "10-300",
            "fps_hint": "<10_effective",
        },
    },
    "corporate": {
        "label": "Corporate / Promotional",
        "description": "Polished, branded content with mixed shots and graphics",
        "indicators": {
            "motion_level": "medium",
            "duration_range": "30-600",
            "audio_channels": "2",
        },
    },
    "social_short": {
        "label": "Social Media Short",
        "description": "Very short vertical or square content for social platforms",
        "indicators": {
            "motion_level": "high",
            "duration_range": "5-90",
            "aspect_hint": "vertical_or_square",
        },
    },
}


# ---------------------------------------------------------------------------
# Workflow templates per content type
# ---------------------------------------------------------------------------
WORKFLOW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "interview": {
        "operations": ["silence_detect", "auto_caption", "color_correct", "loudness_normalize", "export"],
        "params": {
            "silence_threshold_db": -35,
            "caption_style": "clean",
            "color_preset": "natural",
            "loudness_target": -16.0,
            "export_preset": "h264_high",
        },
        "tips": ["Consider multi-cam sync if multiple camera angles", "Auto-cut silences for tighter pacing"],
    },
    "vlog": {
        "operations": ["stabilize", "color_grade", "auto_caption", "music_duck", "export"],
        "params": {
            "stabilize_strength": 0.6,
            "color_lut": "warm_cinematic",
            "caption_style": "dynamic",
            "duck_level_db": -12,
            "export_preset": "h264_high",
        },
        "tips": ["Add jump-cut removal for smoother pacing", "Background music ducking for voice clarity"],
    },
    "tutorial": {
        "operations": ["silence_detect", "zoom_to_cursor", "auto_caption", "chapter_mark", "export"],
        "params": {
            "silence_threshold_db": -30,
            "zoom_factor": 2.0,
            "caption_style": "clean",
            "export_preset": "h264_lossless_screen",
        },
        "tips": ["Zoom-to-cursor for screen recordings", "Auto-generate chapters from silence gaps"],
    },
    "music_video": {
        "operations": ["beat_detect", "color_grade", "speed_ramp", "effects", "export"],
        "params": {
            "color_lut": "cinematic_teal_orange",
            "speed_ramp_sync": "beat",
            "export_preset": "h264_high_bitrate",
        },
        "tips": ["Sync cuts to beat markers", "Consider anamorphic crop for cinematic look"],
    },
    "podcast": {
        "operations": ["loudness_normalize", "noise_reduce", "auto_caption", "chapter_mark", "export"],
        "params": {
            "loudness_target": -16.0,
            "noise_reduce_strength": 0.4,
            "caption_style": "minimal",
            "export_preset": "h264_efficient",
        },
        "tips": ["Split long episodes into chapters", "Consider audio-only export for podcast platforms"],
    },
    "gaming": {
        "operations": ["facecam_detect", "highlight_detect", "auto_caption", "export"],
        "params": {
            "highlight_threshold": 0.7,
            "caption_style": "gaming",
            "export_preset": "h264_60fps",
        },
        "tips": ["Auto-detect exciting moments via audio peaks", "Facecam isolation for thumbnail"],
    },
    "drone": {
        "operations": ["stabilize", "color_grade", "speed_adjust", "denoise", "export"],
        "params": {
            "stabilize_strength": 0.3,
            "color_lut": "aerial_vivid",
            "denoise_strength": 0.2,
            "export_preset": "h265_high",
        },
        "tips": ["Gentle stabilization to preserve natural motion", "H.265 for better 4K compression"],
    },
    "timelapse": {
        "operations": ["deflicker", "color_grade", "stabilize", "speed_adjust", "export"],
        "params": {
            "deflicker_strength": 0.5,
            "color_lut": "vivid",
            "stabilize_strength": 0.2,
            "export_preset": "h265_high",
        },
        "tips": ["Deflicker pass before color grading", "Consider adding gentle pan for static timelapses"],
    },
    "corporate": {
        "operations": ["color_correct", "loudness_normalize", "auto_caption", "watermark", "export"],
        "params": {
            "color_preset": "broadcast_safe",
            "loudness_target": -24.0,
            "caption_style": "professional",
            "export_preset": "h264_broadcast",
        },
        "tips": ["Broadcast-safe color levels", "Add lower-thirds and logo watermark"],
    },
    "social_short": {
        "operations": ["auto_crop_vertical", "auto_caption", "color_grade", "speed_ramp", "export"],
        "params": {
            "crop_aspect": "9:16",
            "caption_style": "bold_centered",
            "color_lut": "punchy",
            "export_preset": "h264_social",
        },
        "tips": ["Auto-crop to vertical for Reels/TikTok/Shorts", "Bold centered captions for mobile viewing"],
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ContentClassification:
    """Result of content type classification."""
    content_type: str = "unknown"
    confidence: float = 0.0
    label: str = ""
    description: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    video_traits: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowSuggestion:
    """Suggested workflow for a classified content type."""
    content_type: str = ""
    operations: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    tips: List[str] = field(default_factory=list)
    estimated_speedup: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SmartRouteResult:
    """Combined classification + workflow suggestion."""
    classification: ContentClassification = field(default_factory=ContentClassification)
    suggestion: WorkflowSuggestion = field(default_factory=WorkflowSuggestion)
    alternatives: List[WorkflowSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "classification": self.classification.to_dict() if isinstance(self.classification, ContentClassification) else self.classification,
            "suggestion": self.suggestion.to_dict() if isinstance(self.suggestion, WorkflowSuggestion) else self.suggestion,
            "alternatives": [
                a.to_dict() if isinstance(a, WorkflowSuggestion) else a
                for a in self.alternatives
            ],
        }
        return d


# ---------------------------------------------------------------------------
# Video analysis helpers
# ---------------------------------------------------------------------------
def _probe_video(video_path: str) -> Dict[str, Any]:
    """Extract detailed video metadata via ffprobe."""
    info = get_video_info(video_path)
    result = {
        "duration": info.get("duration", 0),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "fps": info.get("fps", 0),
        "codec": info.get("codec", ""),
        "audio_channels": info.get("audio_channels", 0),
        "audio_codec": info.get("audio_codec", ""),
        "bitrate": info.get("bitrate", 0),
        "filesize": 0,
    }
    if os.path.isfile(video_path):
        result["filesize"] = os.path.getsize(video_path)
    return result


def _estimate_motion_level(video_path: str, sample_seconds: int = 10) -> str:
    """Estimate motion level from frame difference analysis.

    Returns one of: ``very_low``, ``low``, ``medium``, ``high``.
    """
    ffprobe = get_ffprobe_path()
    try:
        cmd = [
            ffprobe, "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time,pict_type",
            "-of", "json",
            "-read_intervals", f"%+{sample_seconds}",
            video_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout) if proc.stdout.strip() else {}
        frames = data.get("frames", [])
        if not frames:
            return "medium"

        # Count I-frames vs total — high I-frame ratio suggests high motion / scene changes
        i_frames = sum(1 for f in frames if f.get("pict_type") == "I")
        total = len(frames)
        ratio = i_frames / total if total > 0 else 0

        if ratio > 0.25:
            return "high"
        elif ratio > 0.10:
            return "medium"
        elif ratio > 0.03:
            return "low"
        else:
            return "very_low"

    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        logger.debug("Motion estimation failed: %s", exc)
        return "medium"


def _compute_aspect_category(width: int, height: int) -> str:
    """Categorize aspect ratio."""
    if width == 0 or height == 0:
        return "unknown"
    ratio = width / height
    if ratio < 0.8:
        return "vertical"
    elif ratio <= 1.2:
        return "square"
    elif ratio <= 1.9:
        return "landscape"
    else:
        return "ultrawide"


def _score_content_type(
    ct_name: str,
    ct_info: dict,
    traits: Dict[str, Any],
) -> float:
    """Score how well video traits match a content type's indicators. Returns 0.0-1.0."""
    indicators = ct_info.get("indicators", {})
    if not indicators:
        return 0.0

    match_points = 0.0
    total_points = 0.0

    # Motion level check
    if "motion_level" in indicators:
        total_points += 2.0
        expected = indicators["motion_level"]
        actual = traits.get("motion_level", "medium")
        if actual == expected:
            match_points += 2.0
        elif _motion_distance(actual, expected) == 1:
            match_points += 1.0

    # Duration range check
    if "duration_range" in indicators:
        total_points += 2.0
        lo_s, hi_s = (float(x) for x in indicators["duration_range"].split("-"))
        dur = traits.get("duration", 0)
        if lo_s <= dur <= hi_s:
            match_points += 2.0
        elif dur > 0:
            # Partial credit if close
            if dur < lo_s and dur >= lo_s * 0.5:
                match_points += 0.5
            elif dur > hi_s and dur <= hi_s * 1.5:
                match_points += 0.5

    # Face count check
    if "face_count" in indicators:
        total_points += 1.5
        lo_f, hi_f = (int(x) for x in indicators["face_count"].split("-"))
        face_count = traits.get("face_count", -1)
        if face_count >= 0:
            if lo_f <= face_count <= hi_f:
                match_points += 1.5
            elif abs(face_count - lo_f) <= 1 or abs(face_count - hi_f) <= 1:
                match_points += 0.5

    # Audio channels check
    if "audio_channels" in indicators:
        total_points += 1.0
        expected_ch = indicators["audio_channels"]
        actual_ch = traits.get("audio_channels", 0)
        if "-" in expected_ch:
            lo_ch, hi_ch = (int(x) for x in expected_ch.split("-"))
            if lo_ch <= actual_ch <= hi_ch:
                match_points += 1.0
        else:
            if actual_ch == int(expected_ch):
                match_points += 1.0

    # Resolution hint
    if "resolution_hint" in indicators:
        total_points += 1.0
        hint = indicators["resolution_hint"]
        height = traits.get("height", 0)
        if hint == ">=1080p" and height >= 1080:
            match_points += 1.0
        elif hint == ">=2160p" and height >= 2160:
            match_points += 1.0
        elif hint == ">=720p" and height >= 720:
            match_points += 1.0

    # Aspect ratio hint
    if "aspect_hint" in indicators:
        total_points += 1.5
        hint = indicators["aspect_hint"]
        aspect_cat = traits.get("aspect_category", "landscape")
        if hint == "vertical_or_square" and aspect_cat in ("vertical", "square"):
            match_points += 1.5
        elif hint == "landscape" and aspect_cat == "landscape":
            match_points += 1.5
        elif hint == "ultrawide" and aspect_cat == "ultrawide":
            match_points += 1.5

    # FPS hint
    if "fps_hint" in indicators:
        total_points += 1.0
        hint = indicators["fps_hint"]
        fps = traits.get("fps", 0)
        if hint.startswith(">="):
            threshold = int(hint[2:])
            if fps >= threshold:
                match_points += 1.0
        elif hint.startswith("<"):
            val_str = hint[1:].replace("_effective", "")
            threshold = int(val_str)
            if 0 < fps < threshold:
                match_points += 1.0

    return match_points / total_points if total_points > 0 else 0.0


_MOTION_ORDER = ["very_low", "low", "medium", "high"]


def _motion_distance(a: str, b: str) -> int:
    """Distance between two motion levels."""
    try:
        return abs(_MOTION_ORDER.index(a) - _MOTION_ORDER.index(b))
    except ValueError:
        return 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_content(
    video_path: str,
    face_count: int = -1,
    on_progress: Optional[Callable] = None,
) -> ContentClassification:
    """Analyze a video and classify its content type.

    Args:
        video_path:  Path to the video file.
        face_count:  Pre-detected face count (-1 = skip face analysis).
        on_progress: Optional progress callback.

    Returns:
        ``ContentClassification`` with type, confidence, and trait details.
    """
    if on_progress:
        on_progress(10, "Probing video metadata")

    probe = _probe_video(video_path)

    if on_progress:
        on_progress(30, "Estimating motion level")

    motion = _estimate_motion_level(video_path)
    aspect_cat = _compute_aspect_category(probe.get("width", 0), probe.get("height", 0))

    traits = {
        "duration": probe.get("duration", 0),
        "width": probe.get("width", 0),
        "height": probe.get("height", 0),
        "fps": probe.get("fps", 0),
        "codec": probe.get("codec", ""),
        "audio_channels": probe.get("audio_channels", 0),
        "motion_level": motion,
        "aspect_category": aspect_cat,
        "face_count": face_count,
        "filesize_mb": round(probe.get("filesize", 0) / (1024 * 1024), 1),
    }

    if on_progress:
        on_progress(60, "Scoring content types")

    scores = {}
    for ct_name, ct_info in CONTENT_TYPES.items():
        scores[ct_name] = round(_score_content_type(ct_name, ct_info, traits), 4)

    # Pick the best match
    best_type = max(scores, key=scores.get) if scores else "unknown"
    best_score = scores.get(best_type, 0.0)

    ct_info = CONTENT_TYPES.get(best_type, {})

    if on_progress:
        on_progress(100, "Classification complete")

    return ContentClassification(
        content_type=best_type,
        confidence=round(best_score, 4),
        label=ct_info.get("label", best_type),
        description=ct_info.get("description", ""),
        scores=scores,
        video_traits=traits,
    )


def suggest_workflow(
    classification: ContentClassification,
    on_progress: Optional[Callable] = None,
) -> SmartRouteResult:
    """Suggest an optimal workflow based on content classification.

    Args:
        classification: A ``ContentClassification`` from ``classify_content``.
        on_progress:    Optional progress callback.

    Returns:
        ``SmartRouteResult`` with primary suggestion and alternatives.
    """
    if on_progress:
        on_progress(20, "Selecting workflow template")

    ct = classification.content_type
    template = WORKFLOW_TEMPLATES.get(ct, WORKFLOW_TEMPLATES.get("vlog", {}))

    primary = WorkflowSuggestion(
        content_type=ct,
        operations=template.get("operations", []),
        params=template.get("params", {}),
        tips=template.get("tips", []),
        estimated_speedup="2-5x vs manual editing",
    )

    if on_progress:
        on_progress(60, "Finding alternatives")

    # Build alternatives from runner-up scores
    sorted_scores = sorted(classification.scores.items(), key=lambda x: -x[1])
    alternatives = []
    for alt_type, alt_score in sorted_scores[1:4]:  # top 3 alternatives
        if alt_score < 0.1:
            continue
        alt_template = WORKFLOW_TEMPLATES.get(alt_type, {})
        if alt_template:
            alternatives.append(WorkflowSuggestion(
                content_type=alt_type,
                operations=alt_template.get("operations", []),
                params=alt_template.get("params", {}),
                tips=alt_template.get("tips", []),
                estimated_speedup="",
            ))

    if on_progress:
        on_progress(100, "Workflow suggestion ready")

    return SmartRouteResult(
        classification=classification,
        suggestion=primary,
        alternatives=alternatives,
    )
