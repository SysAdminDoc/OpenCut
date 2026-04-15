"""
OpenCut Audio Post-Production Routes (Category 82)

Blueprint for audio post-production features:
- ADR (Automated Dialogue Replacement) session and cue management
- M&E (Music & Effects) mix export
- Automated dialogue premixing with per-speaker processing
- Surround sound upmixing and panning
- Foley cueing, detection, and SFX placement
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import _update_job, async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

audio_post_bp = Blueprint("audio_post", __name__)


# ===========================================================================
# ADR - Session Management
# ===========================================================================

@audio_post_bp.route("/audio/adr/create", methods=["POST"])
@require_csrf
def route_adr_create():
    """Create a new ADR session for a project."""
    from opencut.core.adr_cue_system import create_session

    data = request.get_json(force=True) or {}
    project_name = str(data.get("project_name", "Untitled")).strip()
    source_path = str(data.get("source_path", "")).strip()
    reel = str(data.get("reel", "R1")).strip()
    fps = safe_float(data.get("fps", 24.0), 24.0, min_val=1.0, max_val=120.0)

    if source_path:
        try:
            source_path = validate_filepath(source_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    session = create_session(
        project_name=project_name,
        source_path=source_path,
        reel=reel,
        fps=fps,
    )
    return jsonify(session.to_dict()), 201


# ===========================================================================
# ADR - Cue Management
# ===========================================================================

@audio_post_bp.route("/audio/adr/cue", methods=["POST"])
@require_csrf
def route_adr_cue():
    """Add or update an ADR cue in a session."""
    from opencut.core.adr_cue_system import add_cue, update_cue

    data = request.get_json(force=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    cue_id = str(data.get("cue_id", "")).strip()

    try:
        if cue_id:
            # Update existing cue
            update_fields = {}
            for key in ("character_name", "original_line", "timecode_in",
                        "timecode_out", "scene_context", "reason",
                        "priority", "status", "notes"):
                if key in data:
                    update_fields[key] = data[key]

            cue = update_cue(session_id, cue_id, **update_fields)
        else:
            # Add new cue
            cue = add_cue(
                session_id=session_id,
                character_name=str(data.get("character_name", "")).strip(),
                original_line=str(data.get("original_line", "")).strip(),
                timecode_in=safe_float(data.get("timecode_in", 0), 0, min_val=0.0),
                timecode_out=safe_float(data.get("timecode_out", 0), 0, min_val=0.0),
                scene_context=str(data.get("scene_context", "")).strip(),
                reason=str(data.get("reason", "performance")).strip(),
                priority=safe_int(data.get("priority", 3), 3, min_val=1, max_val=5),
                notes=str(data.get("notes", "")).strip(),
            )
        return jsonify(cue.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@audio_post_bp.route("/audio/adr/cues", methods=["GET"])
def route_adr_cues():
    """List ADR cues for a session with optional filters."""
    from opencut.core.adr_cue_system import list_cues

    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    status_filter = request.args.get("status", None)
    character_filter = request.args.get("character", None)
    sort_by = request.args.get("sort", "timecode")

    try:
        cues = list_cues(
            session_id=session_id,
            status_filter=status_filter,
            character_filter=character_filter,
            sort_by=sort_by,
        )
        return jsonify({"session_id": session_id, "cues": cues, "count": len(cues)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ===========================================================================
# ADR - Cue Sheet Export
# ===========================================================================

@audio_post_bp.route("/audio/adr/cuesheet", methods=["POST"])
@require_csrf
def route_adr_cuesheet():
    """Export ADR cue sheet as CSV or JSON."""
    from opencut.core.adr_cue_system import export_cue_sheet_csv, export_cue_sheet_json

    data = request.get_json(force=True) or {}
    session_id = str(data.get("session_id", "")).strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    export_format = str(data.get("format", "csv")).strip().lower()
    output_path = str(data.get("output_path", "")).strip() or None

    try:
        if export_format == "json":
            path = export_cue_sheet_json(session_id, output_path)
        else:
            path = export_cue_sheet_csv(session_id, output_path)

        return jsonify({
            "output_path": path,
            "format": export_format,
            "session_id": session_id,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ===========================================================================
# ADR - Replacement Recording
# ===========================================================================

@audio_post_bp.route("/audio/adr/record", methods=["POST"])
@require_csrf
@async_job("adr_record")
def route_adr_record(job_id, filepath, data):
    """Process an ADR replacement recording and sync to original timecode."""
    from opencut.core.adr_cue_system import process_replacement

    session_id = str(data.get("session_id", "")).strip()
    cue_id = str(data.get("cue_id", "")).strip()
    sync_method = str(data.get("sync_method", "align")).strip()
    output_dir = str(data.get("output_dir", "")).strip()

    if not session_id or not cue_id:
        raise ValueError("session_id and cue_id are required")

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"ADR record {pct}%")

    result = process_replacement(
        session_id=session_id,
        cue_id=cue_id,
        recording_path=filepath,
        sync_method=sync_method,
        output_dir=output_dir,
        on_progress=_on_progress,
    )
    return result


# ===========================================================================
# M&E Mix
# ===========================================================================

@audio_post_bp.route("/audio/me-mix", methods=["POST"])
@require_csrf
@async_job("me_mix")
def route_me_mix(job_id, filepath, data):
    """Generate an M&E (Music & Effects) mix by removing dialogue."""
    from opencut.core.me_mix import generate_me_mix

    method = str(data.get("method", "auto")).strip()
    output_format = str(data.get("output_format", "wav")).strip()
    target_lufs = safe_float(data.get("target_lufs", -23.0), -23.0,
                             min_val=-36.0, max_val=-10.0)
    dialogue_tracks = data.get("dialogue_tracks")
    if dialogue_tracks and not isinstance(dialogue_tracks, list):
        dialogue_tracks = None
    output_path = str(data.get("output_path", "")).strip() or None

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"M&E mix {pct}%")

    result = generate_me_mix(
        input_path=filepath,
        output_path=output_path,
        method=method,
        output_format=output_format,
        target_lufs=target_lufs,
        dialogue_tracks=dialogue_tracks,
        on_progress=_on_progress,
    )
    return result.to_dict()


# ===========================================================================
# Dialogue Premix
# ===========================================================================

@audio_post_bp.route("/audio/dialogue-premix", methods=["POST"])
@require_csrf
@async_job("dialogue_premix")
def route_dialogue_premix(job_id, filepath, data):
    """Automated dialogue premix with per-speaker processing chains."""
    from opencut.core.dialogue_premix import premix_dialogue

    content_type = str(data.get("content_type", "podcast")).strip()
    target_lufs = data.get("target_lufs")
    if target_lufs is not None:
        target_lufs = safe_float(target_lufs, -16.0, min_val=-36.0, max_val=-10.0)
    deess_strength = str(data.get("deess_strength", "moderate")).strip()
    diarization = data.get("diarization_segments")
    if diarization and not isinstance(diarization, list):
        diarization = None
    output_path = str(data.get("output_path", "")).strip() or None

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Dialogue premix {pct}%")

    result = premix_dialogue(
        input_path=filepath,
        content_type=content_type,
        target_lufs=target_lufs,
        deess_strength=deess_strength,
        diarization_segments=diarization,
        output_path=output_path,
        on_progress=_on_progress,
    )
    return result.to_dict()


@audio_post_bp.route("/audio/premix-presets", methods=["GET"])
def route_premix_presets():
    """List available EQ/processing presets for dialogue premix."""
    from opencut.core.dialogue_premix import list_deess_strengths, list_presets

    presets = list_presets()
    deess = list_deess_strengths()
    return jsonify({"presets": presets, "deess_strengths": deess})


# ===========================================================================
# Surround Upmix
# ===========================================================================

@audio_post_bp.route("/audio/surround-upmix", methods=["POST"])
@require_csrf
@async_job("surround_upmix")
def route_surround_upmix(job_id, filepath, data):
    """Upmix stereo audio to surround sound."""
    from opencut.core.surround_upmix import upmix_surround

    mode = str(data.get("mode", "simple_5_1")).strip()
    export_format = str(data.get("export_format", "wav")).strip()
    output_path = str(data.get("output_path", "")).strip() or None

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Surround upmix {pct}%")

    result = upmix_surround(
        input_path=filepath,
        mode=mode,
        output_path=output_path,
        export_format=export_format,
        on_progress=_on_progress,
    )
    return result.to_dict()


@audio_post_bp.route("/audio/surround-layouts", methods=["GET"])
def route_surround_layouts():
    """List available surround layouts and upmix modes."""
    from opencut.core.surround_upmix import list_export_formats, list_layouts, list_upmix_modes

    layouts = list_layouts()
    modes = list_upmix_modes()
    formats = list_export_formats()
    return jsonify({"layouts": layouts, "modes": modes, "export_formats": formats})


# ===========================================================================
# Foley - Analyze
# ===========================================================================

@audio_post_bp.route("/audio/foley/analyze", methods=["POST"])
@require_csrf
@async_job("foley_analyze")
def route_foley_analyze(job_id, filepath, data):
    """Detect foley cue points in a video file."""
    from opencut.core.foley_cue import detect_foley_cues

    categories = data.get("categories")
    if categories and not isinstance(categories, list):
        categories = None
    sensitivity = safe_float(data.get("sensitivity", 0.5), 0.5,
                             min_val=0.0, max_val=1.0)
    max_cues = safe_int(data.get("max_cues", 500), 500, min_val=1, max_val=2000)

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"Foley analysis {pct}%")

    session = detect_foley_cues(
        video_path=filepath,
        categories=categories,
        sensitivity=sensitivity,
        max_cues=max_cues,
        on_progress=_on_progress,
    )
    return session.to_dict()


# ===========================================================================
# Foley - SFX Placement
# ===========================================================================

@audio_post_bp.route("/audio/foley/place", methods=["POST"])
@require_csrf
@async_job("foley_place")
def route_foley_place(job_id, filepath, data):
    """Place SFX at foley cue points from a sound library."""
    from opencut.core.foley_cue import FoleyCue, FoleySession, place_sfx

    sfx_library = str(data.get("sfx_library_dir", "")).strip()
    if not sfx_library:
        raise ValueError("sfx_library_dir is required")

    mix_level = safe_float(data.get("mix_level", 0.5), 0.5,
                           min_val=0.0, max_val=1.0)
    output_path = str(data.get("output_path", "")).strip() or None

    # Reconstruct session from data
    cues_data = data.get("cues", [])
    if not isinstance(cues_data, list):
        cues_data = []

    cues = []
    for cd in cues_data:
        cues.append(FoleyCue(
            cue_id=str(cd.get("cue_id", "")),
            event_type=str(cd.get("event_type", "")),
            timecode=safe_float(cd.get("timecode", 0), 0),
            duration=safe_float(cd.get("duration", 0.1), 0.1),
            intensity=safe_float(cd.get("intensity", 0.5), 0.5,
                                 min_val=0.0, max_val=1.0),
            suggested_sound=str(cd.get("suggested_sound", "")),
        ))

    session = FoleySession(
        session_id=str(data.get("session_id", "manual")),
        source_path=filepath,
        cues=cues,
        total_duration=safe_float(data.get("total_duration", 0), 0),
    )

    def _on_progress(pct):
        _update_job(job_id, progress=pct, message=f"SFX placement {pct}%")

    result = place_sfx(
        session=session,
        sfx_library_dir=sfx_library,
        source_audio_path=filepath,
        output_path=output_path,
        mix_level=mix_level,
        on_progress=_on_progress,
    )
    return result


# ===========================================================================
# Foley - Categories
# ===========================================================================

@audio_post_bp.route("/audio/foley/categories", methods=["GET"])
def route_foley_categories():
    """List available foley sound categories."""
    from opencut.core.foley_cue import list_categories

    categories = list_categories()
    return jsonify({"categories": categories, "count": len(categories)})
