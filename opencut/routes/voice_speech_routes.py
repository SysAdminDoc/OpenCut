"""
OpenCut Voice & Speech Routes

Blueprint for AI voice and speech features:
- Transcript-based timeline editing (Descript-style)
- Eye contact correction
- Voice overdub (fix mistakes by typing)
- AI lip sync
- Voice-to-voice conversion
"""

import logging

from flask import Blueprint, jsonify

from opencut.errors import safe_error
from opencut.helpers import _resolve_output_dir
from opencut.jobs import _update_job, async_job
from opencut.security import require_csrf, safe_float, safe_int, validate_filepath

logger = logging.getLogger("opencut")

voice_speech_bp = Blueprint("voice_speech", __name__)


# ===========================================================================
# Transcript-Based Timeline Editing
# ===========================================================================

@voice_speech_bp.route("/api/transcript/edit", methods=["POST"])
@require_csrf
@async_job("transcript_edit")
def route_transcript_edit(job_id, filepath, data):
    """Apply text-based video edits via transcript timeline map."""
    from opencut.core.transcript_timeline_edit import (
        apply_edits,
        delete_words,
        duplicate_segment,
        insert_pause,
        parse_transcript,
        rearrange_segments,
    )

    transcript_data = data.get("transcript")
    if not transcript_data:
        raise ValueError("Missing 'transcript' field")

    operations = data.get("operations", [])
    if not operations:
        raise ValueError("Missing 'operations' list")

    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    _update_job(job_id, progress=5, message="Parsing transcript...")
    tmap = parse_transcript(
        transcript_data,
        source_file=filepath,
        on_progress=lambda pct: _update_job(job_id, progress=5 + int(pct * 0.15)),
    )

    _update_job(job_id, progress=20, message="Applying edits...")
    cuts = []
    for i, op in enumerate(operations):
        op_type = op.get("type", "")
        if op_type == "delete_words":
            cuts = delete_words(tmap, op.get("word_indices", []))
        elif op_type == "rearrange_segments":
            cuts = rearrange_segments(tmap, op.get("new_order", []))
        elif op_type == "duplicate_segment":
            cuts = duplicate_segment(
                tmap, op.get("paragraph_index", 0), op.get("insert_after"),
            )
        elif op_type == "insert_pause":
            cuts = insert_pause(
                tmap, op.get("after_word_index", 0),
                op.get("pause_duration", 1.0),
            )
        pct = 20 + int(((i + 1) / len(operations)) * 40)
        _update_job(job_id, progress=pct, message=f"Applied {op_type}")

    if not cuts:
        raise ValueError("No edits produced any cuts")

    _update_job(job_id, progress=60, message="Rendering output...")
    result = apply_edits(
        filepath, cuts,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(job_id, progress=60 + int(pct * 0.4)),
    )

    return result.to_dict()


@voice_speech_bp.route("/api/transcript/parse", methods=["POST"])
@require_csrf
@async_job("transcript_parse", filepath_required=False)
def route_transcript_parse(job_id, filepath, data):
    """Parse transcript into editable text-timeline map."""
    from opencut.core.transcript_timeline_edit import parse_transcript

    transcript_data = data.get("transcript")
    if not transcript_data:
        raise ValueError("Missing 'transcript' field")

    source_file = data.get("source_file", filepath or "")

    tmap = parse_transcript(
        transcript_data,
        source_file=source_file,
        on_progress=lambda pct: _update_job(job_id, progress=pct),
    )

    return tmap.to_dict()


@voice_speech_bp.route("/api/transcript/preview", methods=["POST"])
@require_csrf
@async_job("transcript_preview", filepath_required=False)
def route_transcript_preview(job_id, filepath, data):
    """Preview transcript edit result without applying."""
    from opencut.core.transcript_timeline_edit import parse_transcript, preview_edits

    transcript_data = data.get("transcript")
    if not transcript_data:
        raise ValueError("Missing 'transcript' field")

    operations = data.get("operations", [])
    if not operations:
        raise ValueError("Missing 'operations' list")

    source_file = data.get("source_file", filepath or "")

    _update_job(job_id, progress=10, message="Parsing transcript...")
    tmap = parse_transcript(
        transcript_data,
        source_file=source_file,
        on_progress=lambda pct: _update_job(job_id, progress=10 + int(pct * 0.3)),
    )

    _update_job(job_id, progress=40, message="Previewing edits...")
    result = preview_edits(
        tmap, operations,
        on_progress=lambda pct: _update_job(job_id, progress=40 + int(pct * 0.6)),
    )

    return result.to_dict()


# ===========================================================================
# Eye Contact Correction
# ===========================================================================

@voice_speech_bp.route("/api/video/eye-contact", methods=["POST"])
@require_csrf
@async_job("eye_contact")
def route_eye_contact(job_id, filepath, data):
    """Apply eye contact correction to video."""
    from opencut.core.eye_contact_fix import fix_eye_contact

    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=1.0)
    smoothing = safe_float(data.get("smoothing_alpha", 0.3), 0.3, min_val=0.01, max_val=1.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = fix_eye_contact(
        input_path=filepath,
        intensity=intensity,
        smoothing_alpha=smoothing,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(
            job_id, progress=pct, message=f"Eye contact correction: {pct}%",
        ),
    )

    return result.to_dict()


@voice_speech_bp.route("/api/video/eye-contact/preview", methods=["POST"])
@require_csrf
@async_job("eye_contact_preview")
def route_eye_contact_preview(job_id, filepath, data):
    """Preview eye contact correction on a single frame."""
    from opencut.core.eye_contact_fix import preview_eye_contact

    frame_number = safe_int(data.get("frame_number", 0), 0, min_val=0)
    intensity = safe_float(data.get("intensity", 1.0), 1.0, min_val=0.0, max_val=1.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = preview_eye_contact(
        input_path=filepath,
        frame_number=frame_number,
        intensity=intensity,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(job_id, progress=pct),
    )

    return result


# ===========================================================================
# Voice Overdub
# ===========================================================================

@voice_speech_bp.route("/api/audio/overdub", methods=["POST"])
@require_csrf
@async_job("overdub")
def route_overdub(job_id, filepath, data):
    """Replace audio segment with corrected speech."""
    from opencut.core.voice_overdub import overdub

    replacements = data.get("replacements", [])
    if not replacements:
        raise ValueError("Missing 'replacements' list")

    transcript_segments = data.get("transcript_segments")
    tts_endpoint = data.get("tts_endpoint", "")
    voice_name = data.get("voice_name", "en-US-GuyNeural")
    crossfade_ms = safe_int(data.get("crossfade_ms", 50), 50, min_val=0, max_val=500)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = overdub(
        input_path=filepath,
        replacements=replacements,
        transcript_segments=transcript_segments,
        tts_endpoint=tts_endpoint,
        voice_name=voice_name,
        crossfade_ms=crossfade_ms,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(
            job_id, progress=pct, message=f"Overdubbing: {pct}%",
        ),
    )

    return result.to_dict()


@voice_speech_bp.route("/api/audio/overdub/clone-voice", methods=["POST"])
@require_csrf
@async_job("clone_voice")
def route_clone_voice(job_id, filepath, data):
    """Extract speaker voice profile for overdub cloning."""
    from opencut.core.voice_overdub import extract_speaker_audio

    segments = data.get("segments", [])
    if not segments:
        raise ValueError("Missing 'segments' list for voice extraction")

    exclude_start = safe_float(data.get("exclude_start", 0.0), 0.0, min_val=0.0)
    exclude_end = safe_float(data.get("exclude_end", 0.0), 0.0, min_val=0.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = extract_speaker_audio(
        input_path=filepath,
        segments=segments,
        exclude_start=exclude_start,
        exclude_end=exclude_end,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(job_id, progress=pct),
    )

    return result.to_dict()


# ===========================================================================
# Lip Sync
# ===========================================================================

@voice_speech_bp.route("/api/video/lip-sync", methods=["POST"])
@require_csrf
@async_job("lip_sync")
def route_lip_sync(job_id, filepath, data):
    """Apply lip sync to video with replacement audio."""
    from opencut.core.lip_sync import apply_lip_sync

    audio_path = data.get("audio_path", "")
    if not audio_path:
        raise ValueError("Missing 'audio_path' field")
    validate_filepath(audio_path)

    use_external = data.get("use_external", True)
    blend_strength = safe_float(data.get("blend_strength", 0.7), 0.7, min_val=0.0, max_val=1.0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = apply_lip_sync(
        video_path=filepath,
        audio_path=audio_path,
        use_external=bool(use_external),
        blend_strength=blend_strength,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(
            job_id, progress=pct, message=f"Lip sync: {pct}%",
        ),
    )

    return result.to_dict()


@voice_speech_bp.route("/api/video/lip-sync/preview", methods=["POST"])
@require_csrf
@async_job("lip_sync_preview")
def route_lip_sync_preview(job_id, filepath, data):
    """Preview lip sync on a single frame."""
    from opencut.core.lip_sync import preview_lip_sync

    audio_path = data.get("audio_path", "")
    if not audio_path:
        raise ValueError("Missing 'audio_path' field")
    validate_filepath(audio_path)

    frame_number = safe_int(data.get("frame_number", 0), 0, min_val=0)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = preview_lip_sync(
        video_path=filepath,
        audio_path=audio_path,
        frame_number=frame_number,
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(job_id, progress=pct),
    )

    return result


# ===========================================================================
# Voice Conversion
# ===========================================================================

@voice_speech_bp.route("/api/audio/voice-convert", methods=["POST"])
@require_csrf
@async_job("voice_convert")
def route_voice_convert(job_id, filepath, data):
    """Convert voice to target voice profile."""
    from opencut.core.voice_convert import convert_voice

    target_profile_path = data.get("target_profile_path", "")
    target_profile_name = data.get("target_profile_name", "")
    if not target_profile_path and not target_profile_name:
        raise ValueError("Provide 'target_profile_path' or 'target_profile_name'")

    pitch_shift = data.get("pitch_shift")
    if pitch_shift is not None:
        pitch_shift = safe_float(pitch_shift, 0.0, min_val=-12.0, max_val=12.0)

    use_rvc = data.get("use_rvc", True)
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    result = convert_voice(
        input_path=filepath,
        target_profile_path=target_profile_path,
        target_profile_name=target_profile_name,
        pitch_shift=pitch_shift,
        use_rvc=bool(use_rvc),
        output_dir=output_dir,
        on_progress=lambda pct: _update_job(
            job_id, progress=pct, message=f"Voice conversion: {pct}%",
        ),
    )

    return result.to_dict()


@voice_speech_bp.route("/api/audio/voice-convert/profile", methods=["POST"])
@require_csrf
@async_job("voice_profile_create")
def route_create_voice_profile(job_id, filepath, data):
    """Create target voice profile from reference audio."""
    from opencut.core.voice_convert import create_voice_profile

    name = data.get("name", "")

    result = create_voice_profile(
        audio_path=filepath,
        name=name,
        on_progress=lambda pct: _update_job(job_id, progress=pct),
    )

    return result.to_dict()


@voice_speech_bp.route("/api/audio/voice-convert/profiles", methods=["GET"])
def route_list_voice_profiles():
    """List available voice profiles."""
    from opencut.core.voice_convert import list_voice_profiles

    try:
        profiles = list_voice_profiles()
        return jsonify({"profiles": profiles, "count": len(profiles)})
    except Exception as e:
        return safe_error(e, context="list_voice_profiles")
