"""
OpenCut Sound Design & Music Routes (Category 75)

Blueprint for AI sound design and music features:
- AI-driven sound design from video analysis
- Procedural ambient soundscape generation
- Music mood morphing with keyframes
- Beat-synced video editing
- Creative stem remix with presets
"""

import logging

from flask import Blueprint, jsonify

from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
    validate_path,
    validate_output_path,
)

logger = logging.getLogger("opencut")

sound_music_bp = Blueprint("sound_music", __name__)


# ===========================================================================
# AI Sound Design
# ===========================================================================

@sound_music_bp.route("/audio/sound-design", methods=["POST"])
@require_csrf
@async_job("sound_design")
def route_sound_design(job_id, filepath, data):
    """Generate SFX from video analysis — detect motion events, synthesize sounds, mix."""
    from opencut.core.sound_design_ai import generate_sound_design

    sensitivity = safe_float(data.get("sensitivity", 0.5), 0.5, min_val=0.0, max_val=1.0)
    categories = data.get("categories")
    if categories and not isinstance(categories, list):
        categories = None
    seed = safe_int(data.get("seed"), None) if data.get("seed") is not None else None
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Sound design {pct}%")

    result = generate_sound_design(
        video_path=filepath,
        sensitivity=sensitivity,
        categories=categories,
        seed=seed,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result.to_dict()


@sound_music_bp.route("/audio/sfx-categories", methods=["GET"])
def route_sfx_categories():
    """List available SFX categories."""
    from opencut.core.sound_design_ai import list_sfx_categories
    categories = list_sfx_categories()
    return jsonify({"categories": categories, "count": len(categories)})


# ===========================================================================
# Ambient Soundscape Generation
# ===========================================================================

@sound_music_bp.route("/audio/ambient/generate", methods=["POST"])
@require_csrf
@async_job("ambient_generate", filepath_required=False)
def route_ambient_generate(job_id, filepath, data):
    """Generate ambient soundscape from a preset."""
    from opencut.core.ambient_generator import generate_ambient

    preset = data.get("preset", "forest").strip() or "forest"
    duration = safe_float(data.get("duration", 30.0), 30.0, min_val=1.0, max_val=600.0)
    intensity = safe_float(data.get("intensity", 0.5), 0.5, min_val=0.0, max_val=1.0)
    seed = safe_int(data.get("seed"), None) if data.get("seed") is not None else None
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    crossfade = safe_bool(data.get("crossfade", True), True)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Generating ambient {pct}%")

    result = generate_ambient(
        preset=preset,
        duration=duration,
        intensity=intensity,
        seed=seed,
        output_dir=output_dir,
        crossfade=crossfade,
        on_progress=_on_progress,
    )

    return result.to_dict()


@sound_music_bp.route("/audio/ambient/presets", methods=["GET"])
def route_ambient_presets():
    """List available ambient presets."""
    from opencut.core.ambient_generator import list_presets
    presets = list_presets()
    return jsonify({"presets": presets, "count": len(presets)})


# ===========================================================================
# Music Mood Morph
# ===========================================================================

@sound_music_bp.route("/audio/mood-morph", methods=["POST"])
@require_csrf
@async_job("mood_morph")
def route_mood_morph(job_id, filepath, data):
    """Apply mood transformation to audio.

    Supports single mood or keyframed mood curve.
    """
    from opencut.core.music_mood_morph import (
        MoodKeyframe,
        apply_keyframed_morph,
        apply_mood_morph,
    )

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output_path_val = data.get("output_path", "")
    if output_path_val:
        output_path_val = validate_output_path(output_path_val)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Mood morph {pct}%")

    # Check for keyframed mode
    keyframes_data = data.get("keyframes")
    if keyframes_data and isinstance(keyframes_data, list) and len(keyframes_data) > 1:
        # Parse keyframes
        keyframes = []
        for kf in keyframes_data:
            if isinstance(kf, dict):
                keyframes.append(MoodKeyframe(
                    time=safe_float(kf.get("time", 0), 0.0, min_val=0.0),
                    mood=kf.get("mood", "brighten").strip() or "brighten",
                    intensity=safe_float(kf.get("intensity", 1.0), 1.0, min_val=0.0, max_val=1.0),
                ))

        if keyframes:
            segment_duration = safe_float(
                data.get("segment_duration", 5.0), 5.0, min_val=1.0, max_val=30.0
            )
            result = apply_keyframed_morph(
                input_path=filepath,
                keyframes=keyframes,
                segment_duration=segment_duration,
                output_path_val=output_path_val,
                output_dir=output_dir,
                on_progress=_on_progress,
            )
            return result.to_dict()

    # Single mood mode
    mood = data.get("mood", "brighten").strip() or "brighten"
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=1.0)

    result = apply_mood_morph(
        input_path=filepath,
        mood=mood,
        intensity=intensity,
        output_path_val=output_path_val,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result.to_dict()


# ===========================================================================
# Beat-Synced Video Editing
# ===========================================================================

@sound_music_bp.route("/audio/beat-sync", methods=["POST"])
@require_csrf
@async_job("beat_sync")
def route_beat_sync(job_id, filepath, data):
    """Beat-synced video editing — align video cuts to music beats."""
    from opencut.core.beat_sync_edit import assemble_beat_sync

    clip_paths = data.get("clip_paths", [])
    if not isinstance(clip_paths, list) or not clip_paths:
        raise ValueError("clip_paths must be a non-empty list of video file paths")

    # Validate each clip path
    validated_clips = []
    for cp in clip_paths:
        if isinstance(cp, str) and cp.strip():
            validated_clips.append(validate_filepath(cp.strip()))
    if not validated_clips:
        raise ValueError("No valid clip paths provided")

    mode = data.get("mode", "every_beat").strip() or "every_beat"
    custom_n = safe_int(data.get("custom_n", 1), 1, min_val=1, max_val=64)
    sensitivity = safe_float(data.get("sensitivity", 0.5), 0.5, min_val=0.0, max_val=1.0)
    energy_match = safe_bool(data.get("energy_match", False), False)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output_path_val = data.get("output_path", "")
    if output_path_val:
        output_path_val = validate_output_path(output_path_val)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Beat sync {pct}%")

    result = assemble_beat_sync(
        audio_path=filepath,
        clip_paths=validated_clips,
        mode=mode,
        custom_n=custom_n,
        sensitivity=sensitivity,
        energy_match=energy_match,
        output_path_val=output_path_val,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result.to_dict()


@sound_music_bp.route("/audio/beat-detect", methods=["POST"])
@require_csrf
@async_job("beat_detect")
def route_beat_detect(job_id, filepath, data):
    """Detect beats in an audio track."""
    from opencut.core.beat_sync_edit import detect_beats

    sensitivity = safe_float(data.get("sensitivity", 0.5), 0.5, min_val=0.0, max_val=1.0)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Beat detection {pct}%")

    result = detect_beats(
        audio_path=filepath,
        sensitivity=sensitivity,
        on_progress=_on_progress,
    )

    return result.to_dict()


# ===========================================================================
# Stem Remix
# ===========================================================================

@sound_music_bp.route("/audio/stem-remix", methods=["POST"])
@require_csrf
@async_job("stem_remix", filepath_required=False)
def route_stem_remix(job_id, filepath, data):
    """Remix from separated stems with per-stem effects."""
    from opencut.core.stem_remix import remix_stems

    stem_dir = data.get("stem_dir", "").strip()
    stem_paths = data.get("stem_paths")
    if stem_paths and not isinstance(stem_paths, dict):
        stem_paths = None

    # Validate stem paths if provided
    if stem_paths:
        validated = {}
        for name, path in stem_paths.items():
            if isinstance(path, str) and path.strip():
                validated[name] = validate_filepath(path.strip())
        stem_paths = validated or None

    preset = data.get("preset", "").strip()
    custom_settings = data.get("custom_settings")
    if custom_settings and not isinstance(custom_settings, dict):
        custom_settings = None
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    output_path_val = data.get("output_path", "")
    if output_path_val:
        output_path_val = validate_output_path(output_path_val)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Stem remix {pct}%")

    result = remix_stems(
        stem_dir=stem_dir,
        stem_paths=stem_paths,
        preset=preset,
        custom_settings=custom_settings,
        output_path_val=output_path_val,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result.to_dict()


@sound_music_bp.route("/audio/remix-presets", methods=["GET"])
def route_remix_presets():
    """List available remix presets."""
    from opencut.core.stem_remix import list_remix_presets
    presets = list_remix_presets()
    return jsonify({"presets": presets, "count": len(presets)})


@sound_music_bp.route("/audio/stem-remix/preview", methods=["POST"])
@require_csrf
@async_job("stem_remix_preview", filepath_required=False)
def route_stem_remix_preview(job_id, filepath, data):
    """Preview remix settings on a short segment."""
    from opencut.core.stem_remix import preview_remix

    stem_dir = data.get("stem_dir", "").strip()
    stem_paths = data.get("stem_paths")
    if stem_paths and not isinstance(stem_paths, dict):
        stem_paths = None

    if stem_paths:
        validated = {}
        for name, path in stem_paths.items():
            if isinstance(path, str) and path.strip():
                validated[name] = validate_filepath(path.strip())
        stem_paths = validated or None

    preset = data.get("preset", "").strip()
    custom_settings = data.get("custom_settings")
    if custom_settings and not isinstance(custom_settings, dict):
        custom_settings = None
    preview_duration = safe_float(data.get("preview_duration", 15.0), 15.0, min_val=3.0, max_val=30.0)
    preview_start = safe_float(data.get("preview_start", 0.0), 0.0, min_val=0.0)
    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg or f"Preview {pct}%")

    result = preview_remix(
        stem_dir=stem_dir,
        stem_paths=stem_paths,
        preset=preset,
        custom_settings=custom_settings,
        preview_duration=preview_duration,
        preview_start=preview_start,
        output_dir=output_dir,
        on_progress=_on_progress,
    )

    return result.to_dict()
