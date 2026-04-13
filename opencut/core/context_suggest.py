"""
OpenCut Context-Aware Command Suggestions

Rules-based engine that analyzes clip metadata and recent user actions
to suggest the most relevant next actions. Fast, synchronous, no FFmpeg needed.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class Suggestion:
    """A single context-aware suggestion."""
    action: str
    description: str
    confidence: float    # 0.0 - 1.0
    reason: str


def get_suggestions(
    clip_metadata: Dict,
    recent_actions: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """
    Generate context-aware command suggestions based on clip metadata
    and recent user actions.

    Args:
        clip_metadata: dict with keys like duration, has_audio, has_video,
                       loudness_lufs, resolution, codec, width, height, fps, etc.
        recent_actions: List of recently performed action names.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of suggestion dicts, sorted by confidence (top 3).
    """
    if recent_actions is None:
        recent_actions = []

    recent_set = set(a.lower().strip() for a in recent_actions)

    if on_progress:
        on_progress(10, "Analyzing clip context...")

    suggestions: List[Suggestion] = []

    # Extract metadata with safe defaults
    duration = float(clip_metadata.get("duration", 0))
    has_audio = bool(clip_metadata.get("has_audio", True))
    has_video = bool(clip_metadata.get("has_video", True))
    loudness_lufs = clip_metadata.get("loudness_lufs")
    resolution = int(clip_metadata.get("resolution", clip_metadata.get("width", 0)))
    width = int(clip_metadata.get("width", resolution))
    height = int(clip_metadata.get("height", 0))
    codec = str(clip_metadata.get("codec", "")).lower()
    fps = float(clip_metadata.get("fps", 30.0))
    file_size_mb = float(clip_metadata.get("file_size_mb", 0))

    if on_progress:
        on_progress(30, "Evaluating rules...")

    # --- Audio loudness rules ---
    if loudness_lufs is not None:
        loudness_val = float(loudness_lufs)
        if loudness_val > -10:
            suggestions.append(Suggestion(
                action="normalize_audio",
                description="Normalize audio levels -- current loudness is too high",
                confidence=0.95,
                reason=f"Loudness is {loudness_val:.1f} LUFS, above -10 LUFS threshold",
            ))
        elif loudness_val < -24:
            suggestions.append(Suggestion(
                action="normalize_audio",
                description="Normalize audio levels -- current loudness is too low",
                confidence=0.90,
                reason=f"Loudness is {loudness_val:.1f} LUFS, below -24 LUFS threshold",
            ))
        elif -24 <= loudness_val <= -10:
            # Loudness is acceptable, but could still benefit from normalization
            if "normalize_audio" not in recent_set:
                suggestions.append(Suggestion(
                    action="check_loudness",
                    description="Audio levels are within range -- verify platform compliance",
                    confidence=0.30,
                    reason=f"Loudness is {loudness_val:.1f} LUFS (acceptable range)",
                ))

    # --- Resolution rules ---
    if resolution > 1920 and "upscale" not in recent_set:
        suggestions.append(Suggestion(
            action="no_upscale_needed",
            description="Resolution is already high -- no upscaling needed",
            confidence=0.80,
            reason=f"Resolution is {width}x{height}, exceeds 1920px",
        ))
    elif 0 < resolution < 1080 and "upscale" not in recent_set:
        suggestions.append(Suggestion(
            action="upscale",
            description="Consider upscaling to HD for better quality",
            confidence=0.75,
            reason=f"Resolution is {width}x{height}, below 1080p",
        ))

    # --- Duration rules ---
    if duration > 300 and "scene_detect" not in recent_set and "detect_scenes" not in recent_set:
        suggestions.append(Suggestion(
            action="detect_scenes",
            description="Detect scene boundaries for easier navigation",
            confidence=0.85,
            reason=f"Video is {duration:.0f}s ({duration / 60:.1f} min) -- scene detection helps organize long content",
        ))

    if duration > 600 and "highlights" not in recent_set:
        suggestions.append(Suggestion(
            action="extract_highlights",
            description="Extract highlight moments from this long video",
            confidence=0.70,
            reason=f"Video is {duration / 60:.1f} minutes -- highlights can create shorter engaging clips",
        ))

    # --- Audio + captions rules ---
    if has_audio and "captions" not in recent_set and "add_captions" not in recent_set:
        suggestions.append(Suggestion(
            action="add_captions",
            description="Generate captions/subtitles for accessibility",
            confidence=0.80,
            reason="Video has audio but no captions detected in recent actions",
        ))

    # --- Silence removal ---
    if has_audio and "silence" not in recent_set and "remove_silence" not in recent_set:
        suggestions.append(Suggestion(
            action="remove_silence",
            description="Detect and remove silent segments",
            confidence=0.65,
            reason="Silence removal not yet performed -- can tighten pacing",
        ))

    # --- Codec optimization ---
    if codec and codec in ("h264", "h.264", "avc") and "transcode" not in recent_set:
        if file_size_mb > 500:
            suggestions.append(Suggestion(
                action="transcode_h265",
                description="Re-encode to H.265/HEVC for smaller file size",
                confidence=0.70,
                reason=f"File is {file_size_mb:.0f}MB with H.264 -- H.265 can reduce size 30-50%",
            ))

    # --- FPS rules ---
    if fps > 30 and "export" not in recent_set:
        suggestions.append(Suggestion(
            action="check_export_fps",
            description=f"Source is {fps:.0f}fps -- ensure export settings match platform requirements",
            confidence=0.40,
            reason=f"High frame rate ({fps:.0f}fps) may need adjustment for target platform",
        ))

    # --- Thumbnail ---
    if duration > 30 and "thumbnail" not in recent_set:
        suggestions.append(Suggestion(
            action="generate_thumbnail",
            description="Generate a compelling thumbnail for this video",
            confidence=0.55,
            reason="No thumbnail generated yet -- thumbnails boost click-through rates",
        ))

    # --- Color grading ---
    if has_video and "color_grade" not in recent_set and "auto_grade" not in recent_set:
        suggestions.append(Suggestion(
            action="auto_color_grade",
            description="Apply AI color grading for a polished look",
            confidence=0.45,
            reason="No color grading applied -- auto-grading can enhance visual quality",
        ))

    if on_progress:
        on_progress(80, "Ranking suggestions...")

    # Sort by confidence descending, return top 3
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    top = suggestions[:3]

    if on_progress:
        on_progress(100, f"Generated {len(top)} suggestions")

    return [_suggestion_to_dict(s) for s in top]


def _suggestion_to_dict(s: Suggestion) -> dict:
    """Convert Suggestion dataclass to a JSON-serializable dict."""
    return {
        "action": s.action,
        "description": s.description,
        "confidence": round(s.confidence, 2),
        "reason": s.reason,
    }
