"""
OpenCut Audio Production Routes

Blueprint for advanced audio production features:
- Audio restoration (declip, dehum, decrackle, dewind, dereverb)
- Audio fingerprinting & copyright detection
- Room tone matching & generation
- M&E mix export
- Automated dialogue premix
- AI show notes & transcript summary
"""

import logging

from flask import Blueprint

from opencut.jobs import async_job
from opencut.security import (
    require_csrf,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

audio_prod_bp = Blueprint("audio_prod", __name__)


# ===========================================================================
# Audio Restoration Toolkit
# ===========================================================================

@audio_prod_bp.route("/audio/declip", methods=["POST"])
@require_csrf
@async_job("declip")
def route_declip(job_id, filepath, data):
    """Remove clipping distortion from audio."""
    from opencut.core.audio_restore import declip

    result = declip(
        input_path=filepath,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


@audio_prod_bp.route("/audio/dehum", methods=["POST"])
@require_csrf
@async_job("dehum")
def route_dehum(job_id, filepath, data):
    """Remove electrical hum at fundamental + harmonics."""
    from opencut.core.audio_restore import dehum

    frequency = safe_float(data.get("frequency", 60.0), 60.0, min_val=20.0, max_val=500.0)
    harmonics = safe_int(data.get("harmonics", 4), 4, min_val=1, max_val=8)

    result = dehum(
        input_path=filepath,
        frequency=frequency,
        harmonics=harmonics,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


@audio_prod_bp.route("/audio/decrackle", methods=["POST"])
@require_csrf
@async_job("decrackle")
def route_decrackle(job_id, filepath, data):
    """Reduce crackle/pop noise from audio."""
    from opencut.core.audio_restore import decrackle

    result = decrackle(
        input_path=filepath,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


@audio_prod_bp.route("/audio/dewind", methods=["POST"])
@require_csrf
@async_job("dewind")
def route_dewind(job_id, filepath, data):
    """Remove wind noise using high-pass filter."""
    from opencut.core.audio_restore import dewind

    cutoff_hz = safe_float(data.get("cutoff_hz", 80.0), 80.0, min_val=20.0, max_val=500.0)

    result = dewind(
        input_path=filepath,
        cutoff_hz=cutoff_hz,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


@audio_prod_bp.route("/audio/dereverb", methods=["POST"])
@require_csrf
@async_job("dereverb")
def route_dereverb(job_id, filepath, data):
    """Reduce room reverb from audio."""
    from opencut.core.audio_restore import dereverb

    result = dereverb(
        input_path=filepath,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


# ===========================================================================
# Audio Fingerprinting & Copyright Detection
# ===========================================================================

@audio_prod_bp.route("/audio/fingerprint", methods=["POST"])
@require_csrf
@async_job("fingerprint")
def route_fingerprint(job_id, filepath, data):
    """Generate audio fingerprint for a file."""
    from opencut.core.audio_fingerprint import generate_fingerprint

    fp = generate_fingerprint(
        input_path=filepath,
        on_progress=lambda pct, msg: None,
    )
    return {
        "hash_count": len(fp.hash_data),
        "duration": fp.duration,
        "sample_rate": fp.sample_rate,
    }


@audio_prod_bp.route("/audio/fingerprint/scan", methods=["POST"])
@require_csrf
@async_job("fingerprint_scan")
def route_fingerprint_scan(job_id, filepath, data):
    """Scan audio against local fingerprint database for copyright matches."""
    from opencut.core.audio_fingerprint import scan_against_database

    db_path = data.get("db_path")

    matches = scan_against_database(
        input_path=filepath,
        db_path=db_path,
        on_progress=lambda pct, msg: None,
    )
    return {
        "matches": matches,
        "total_checked": len(matches),
        "flagged": sum(1 for m in matches if m.get("is_match")),
    }


@audio_prod_bp.route("/audio/fingerprint/add", methods=["POST"])
@require_csrf
@async_job("fingerprint_add")
def route_fingerprint_add(job_id, filepath, data):
    """Add a track to the local fingerprint database."""
    from opencut.core.audio_fingerprint import add_to_database

    label = data.get("label", "")
    if not label:
        raise ValueError("A 'label' is required to add a track to the fingerprint database.")

    add_to_database(
        input_path=filepath,
        label=label,
        db_path=data.get("db_path"),
    )
    return {"status": "added", "label": label}


# ===========================================================================
# Room Tone Matching & Generation
# ===========================================================================

@audio_prod_bp.route("/audio/room-tone/extract", methods=["POST"])
@require_csrf
@async_job("room_tone_extract")
def route_room_tone_extract(job_id, filepath, data):
    """Extract room tone from quietest segments of audio."""
    from opencut.core.room_tone import extract_room_tone

    duration = safe_float(data.get("duration", 5.0), 5.0, min_val=0.5, max_val=60.0)

    result = extract_room_tone(
        input_path=filepath,
        duration=duration,
        on_progress=lambda pct, msg: None,
    )
    return result


@audio_prod_bp.route("/audio/room-tone/generate", methods=["POST"])
@require_csrf
@async_job("room_tone_generate")
def route_room_tone_generate(job_id, filepath, data):
    """Generate seamless room tone from a reference file."""
    from opencut.core.room_tone import generate_room_tone

    duration = safe_float(data.get("duration", 30.0), 30.0, min_val=1.0, max_val=3600.0)

    result = generate_room_tone(
        reference_path=filepath,
        duration=duration,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


@audio_prod_bp.route("/audio/room-tone/fill", methods=["POST"])
@require_csrf
@async_job("room_tone_fill")
def route_room_tone_fill(job_id, filepath, data):
    """Fill silence gaps with room tone."""
    from opencut.core.room_tone import fill_gaps_with_tone

    tone_path = data.get("tone_path", "")
    if not tone_path:
        raise ValueError("A 'tone_path' is required for gap filling.")
    validate_filepath(tone_path)

    gap_threshold_db = safe_float(
        data.get("gap_threshold_db", -50.0), -50.0, min_val=-80.0, max_val=-10.0
    )

    result = fill_gaps_with_tone(
        input_path=filepath,
        tone_path=tone_path,
        gap_threshold_db=gap_threshold_db,
        output_path=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )
    return result


# ===========================================================================
# M&E Mix Export
# ===========================================================================

@audio_prod_bp.route("/audio/me-mix", methods=["POST"])
@require_csrf
@async_job("me_mix")
def route_me_mix(job_id, filepath, data):
    """Generate M&E (Music & Effects) mix by removing vocals."""
    from opencut.core.me_mix import generate_me_mix

    method = data.get("method", "subtract")
    if method not in ("subtract", "stems"):
        method = "subtract"

    result = generate_me_mix(
        input_path=filepath,
        output_path=data.get("output_path"),
        method=method,
        on_progress=lambda pct, msg: None,
    )
    return result


# ===========================================================================
# Automated Dialogue Premix
# ===========================================================================

@audio_prod_bp.route("/audio/dialogue-premix", methods=["POST"])
@require_csrf
@async_job("dialogue_premix")
def route_dialogue_premix(job_id, filepath, data):
    """Apply broadcast-standard dialogue premix chain."""
    from opencut.core.dialogue_premix import premix_dialogue, premix_multi_speaker

    target_lufs = safe_float(data.get("target_lufs", -23.0), -23.0, min_val=-36.0, max_val=-10.0)
    segments = data.get("diarization_segments")

    if segments and isinstance(segments, list):
        result = premix_multi_speaker(
            input_path=filepath,
            diarization_segments=segments,
            target_lufs=target_lufs,
            on_progress=lambda pct, msg: None,
        )
    else:
        result = premix_dialogue(
            input_path=filepath,
            target_lufs=target_lufs,
            on_progress=lambda pct, msg: None,
        )

    return result


# ===========================================================================
# AI Show Notes & Transcript Summary
# ===========================================================================

@audio_prod_bp.route("/podcast/show-notes", methods=["POST"])
@require_csrf
@async_job("show_notes")
def route_show_notes(job_id, filepath, data):
    """Generate AI-powered show notes from transcript text."""
    from opencut.core.show_notes import export_show_notes, generate_show_notes

    transcript_text = data.get("transcript_text", "")
    if not transcript_text:
        # Try to read transcript from file if filepath points to a text file
        import os
        if filepath and os.path.exists(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            if ext in (".txt", ".srt", ".vtt", ".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    transcript_text = f.read()

    if not transcript_text:
        raise ValueError("No transcript text provided. Pass 'transcript_text' in the request body or point to a transcript file.")

    export_format = data.get("format", "markdown")
    if export_format not in ("markdown", "html", "text"):
        export_format = "markdown"

    notes = generate_show_notes(
        transcript_text=transcript_text,
        format=export_format,
        on_progress=lambda pct, msg: None,
    )

    exported = export_show_notes(notes, format=export_format)

    return {
        "summary": notes.summary,
        "key_topics": notes.key_topics,
        "quotes": notes.quotes,
        "chapter_markers": notes.chapter_markers,
        "resources_mentioned": notes.resources_mentioned,
        "formatted_output": exported,
        "format": export_format,
    }
