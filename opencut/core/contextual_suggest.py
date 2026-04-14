"""
OpenCut Contextual Suggest - Smart Operation Recommendations

Analyze current clip properties and suggest relevant OpenCut operations.
Uses the ClipProfile from smart_defaults and a rules engine to rank
suggestions by confidence. Factors in recent operations to avoid
redundant suggestions.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class Suggestion:
    """A contextual operation suggestion."""
    feature_id: str
    name: str
    reason: str
    confidence: float
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rules Engine
# ---------------------------------------------------------------------------
def _build_suggestions(
    profile,
    recent_set: set,
) -> List[Suggestion]:
    """Apply rules to generate ranked suggestions based on clip profile.

    Args:
        profile: ClipProfile from smart_defaults.analyze_clip_properties().
        recent_set: Set of lowercased recently-performed operation IDs.

    Returns:
        List of Suggestion objects, unsorted.
    """
    suggestions = []

    # --- Audio loudness rules ---
    if profile.has_audio and profile.avg_loudness_lufs is not None:
        lufs = profile.avg_loudness_lufs
        if lufs < -24 and "normalize_audio" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="normalize_audio",
                name="Normalize Audio",
                reason=f"Audio is too quiet ({lufs:.1f} LUFS). Normalizing will bring it to broadcast standard.",
                confidence=0.95,
                params={"target_lufs": -16.0},
            ))
        elif lufs > -10 and "normalize_audio" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="normalize_audio",
                name="Normalize Audio",
                reason=f"Audio is too loud ({lufs:.1f} LUFS). Risk of clipping on playback.",
                confidence=0.93,
                params={"target_lufs": -16.0},
            ))

    # --- No audio but video present ---
    if profile.has_video and not profile.has_audio:
        if "music_gen" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="music_gen",
                name="AI Music Generation",
                reason="Video has no audio track. Consider adding background music.",
                confidence=0.70,
                params={"style": "ambient"},
            ))

    # --- Caption suggestions ---
    if profile.has_audio and "add_captions" not in recent_set:
        confidence = 0.85
        if profile.detected_content_type in ("interview", "tutorial", "vlog", "presentation"):
            confidence = 0.92
        suggestions.append(Suggestion(
            feature_id="add_captions",
            name="Generate Captions",
            reason="No captions detected. Adding captions improves accessibility and engagement.",
            confidence=confidence,
            params={"model": "medium"},
        ))

    # --- Stabilization for shaky footage ---
    if profile.has_video and not profile.is_static_camera:
        if "stabilize" not in recent_set:
            confidence = 0.60
            if profile.detected_content_type == "vlog":
                confidence = 0.80
            elif profile.detected_content_type == "drone":
                confidence = 0.75
            elif profile.detected_content_type == "sports":
                confidence = 0.70
            suggestions.append(Suggestion(
                feature_id="stabilize",
                name="Stabilize Video",
                reason="Camera motion detected. Stabilization can improve viewing experience.",
                confidence=confidence,
                params={"smoothing": 15},
            ))

    # --- Upscale low resolution ---
    if profile.has_video and profile.resolution < 1080 and "upscale" not in recent_set:
        suggestions.append(Suggestion(
            feature_id="upscale",
            name="AI Upscale",
            reason=f"Resolution is {profile.width}x{profile.height} (below 1080p). AI upscaling can improve quality.",
            confidence=0.80,
            params={"target_resolution": "1080p"},
        ))

    # --- Denoise ---
    if profile.has_video and "denoise" not in recent_set:
        confidence = 0.40  # lower base confidence -- not always needed
        if profile.detected_content_type in ("interview", "vlog"):
            confidence = 0.55
        if profile.codec in ("h264", "h.264") and profile.bitrate_kbps > 0 and profile.bitrate_kbps < 5000:
            # Low bitrate h264 likely has compression artifacts
            confidence = 0.70
            suggestions.append(Suggestion(
                feature_id="denoise",
                name="Video Denoise",
                reason=f"Low bitrate ({profile.bitrate_kbps} kbps) may have compression artifacts. Denoising can help.",
                confidence=confidence,
                params={"strength": "moderate"},
            ))

    # --- Speaker diarization for multi-speaker content ---
    if profile.has_audio and profile.detected_content_type in ("interview", "presentation"):
        if "diarize" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="diarize",
                name="Speaker Diarization",
                reason="Interview/presentation content likely has multiple speakers. Diarization labels each speaker.",
                confidence=0.75,
                params={},
            ))

    # --- Scene detection for long videos ---
    if profile.duration_s > 300 and "scene_detect" not in recent_set:
        suggestions.append(Suggestion(
            feature_id="scene_detect",
            name="Scene Detection",
            reason=f"Video is {profile.duration_s / 60:.1f} minutes. Scene detection helps organize long content.",
            confidence=0.82,
            params={},
        ))

    # --- Highlight extraction for very long videos ---
    if profile.duration_s > 600 and "highlights" not in recent_set:
        suggestions.append(Suggestion(
            feature_id="highlights",
            name="Highlight Detection",
            reason=f"Video is {profile.duration_s / 60:.1f} minutes. Extract highlights for shorter, engaging clips.",
            confidence=0.72,
            params={},
        ))

    # --- Auto color grade ---
    if profile.has_video and "auto_color" not in recent_set and "color_grade" not in recent_set:
        confidence = 0.45
        if profile.detected_content_type in ("drone", "music_video", "vlog"):
            confidence = 0.65
        suggestions.append(Suggestion(
            feature_id="auto_color",
            name="Auto Color Grade",
            reason="No color grading applied. Auto-grading can enhance visual quality.",
            confidence=confidence,
            params={},
        ))

    # --- Silence removal for speech-heavy content ---
    if profile.has_audio and profile.detected_content_type in ("interview", "tutorial", "vlog", "presentation"):
        if "remove_silence" not in recent_set and "dead_time" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="remove_silence",
                name="Remove Silence",
                reason="Speech-heavy content often has pauses. Removing silence tightens pacing.",
                confidence=0.68,
                params={"threshold_db": -35, "min_silence_ms": 800},
            ))

    # --- Smart reframe for non-vertical content ---
    if (profile.has_video and profile.width > profile.height
            and "smart_reframe" not in recent_set):
        if profile.detected_content_type in ("interview", "vlog"):
            suggestions.append(Suggestion(
                feature_id="smart_reframe",
                name="Smart Reframe",
                reason="Landscape video can be reframed to 9:16 for mobile/social platforms.",
                confidence=0.60,
                params={"target_aspect": "9:16"},
            ))

    # --- Thumbnail for publishable content ---
    if profile.has_video and profile.duration_s > 30 and "thumbnail_gen" not in recent_set:
        suggestions.append(Suggestion(
            feature_id="thumbnail_gen",
            name="Thumbnail Generator",
            reason="Generate a compelling thumbnail before publishing.",
            confidence=0.50,
            params={},
        ))

    # --- Screen recording specific ---
    if profile.detected_content_type == "screen_recording":
        if "cursor_zoom" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="cursor_zoom",
                name="Cursor Zoom",
                reason="Screen recording detected. Cursor zoom highlights important actions.",
                confidence=0.72,
                params={},
            ))

    # --- Drone specific ---
    if profile.detected_content_type == "drone":
        if "telemetry_overlay" not in recent_set:
            suggestions.append(Suggestion(
                feature_id="telemetry_overlay",
                name="Telemetry Overlay",
                reason="Drone footage detected. Overlay GPS, altitude, and speed data.",
                confidence=0.60,
                params={},
            ))

    # --- Codec efficiency ---
    if (profile.codec in ("h264", "h.264", "avc")
            and profile.bitrate_kbps > 20000
            and "export_h265" not in recent_set):
        suggestions.append(Suggestion(
            feature_id="export_h265",
            name="Export H.265",
            reason=f"H.264 at {profile.bitrate_kbps} kbps. Re-encoding to H.265 can reduce file size 30-50%.",
            confidence=0.55,
            params={"crf": 20},
        ))

    return suggestions


def suggest_operations(
    video_path: str,
    recent_ops: Optional[List[str]] = None,
    max_suggestions: int = 3,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Analyze clip and suggest relevant OpenCut operations.

    Args:
        video_path: Path to the video file.
        recent_ops: List of recently performed operation IDs to exclude.
        max_suggestions: Maximum number of suggestions to return.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of suggestion dicts sorted by confidence, each with
        feature_id, name, reason, confidence, and params.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    import os
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if recent_ops is None:
        recent_ops = []

    recent_set = {op.lower().strip() for op in recent_ops}

    if on_progress:
        on_progress(5, "Analyzing clip properties...")

    # Import and run clip analysis
    from opencut.core.smart_defaults import analyze_clip_properties

    profile = analyze_clip_properties(video_path, on_progress=lambda p, m="": (
        on_progress(5 + int(p * 0.6), m) if on_progress else None
    ))

    if on_progress:
        on_progress(70, "Generating suggestions...")

    suggestions = _build_suggestions(profile, recent_set)

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s.confidence, reverse=True)

    # Take top N
    top = suggestions[:max_suggestions]

    if on_progress:
        on_progress(100, f"Generated {len(top)} suggestions")

    return [
        {
            "feature_id": s.feature_id,
            "name": s.name,
            "reason": s.reason,
            "confidence": round(s.confidence, 2),
            "params": s.params,
        }
        for s in top
    ]
