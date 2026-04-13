"""
OpenCut Audio Advanced Routes

Blueprint for advanced audio features:
- Podcast Production Suite
- Real-Time Voice Conversion
- ADR Cueing System
- Surround Sound Panning & Upmix
- Audio-Reactive Visualizers
- Audio Description Track Generator
- Voice Commands
"""

import logging

from flask import Blueprint, jsonify, request

from opencut.jobs import async_job
from opencut.security import (
    require_csrf,
    safe_bool,
    safe_float,
    safe_int,
    validate_filepath,
)

logger = logging.getLogger("opencut")

audio_adv_bp = Blueprint("audio_adv", __name__)


# ===========================================================================
# Podcast Production Suite
# ===========================================================================

@audio_adv_bp.route("/podcast/polish", methods=["POST"])
@require_csrf
@async_job("podcast_polish")
def route_podcast_polish(job_id, filepath, data):
    """One-click podcast polish: auto-level, EQ, intro/outro, chapters, -16 LUFS."""
    from opencut.core.podcast_suite import PodcastConfig, polish_podcast

    config = PodcastConfig(
        target_lufs=safe_float(data.get("target_lufs", -16.0), -16.0, min_val=-36.0, max_val=-10.0),
        crossfade_duration=safe_float(data.get("crossfade_duration", 2.0), 2.0, min_val=0.0, max_val=10.0),
        generate_chapters=safe_bool(data.get("generate_chapters", True), True),
        normalize_per_speaker=safe_bool(data.get("normalize_per_speaker", True), True),
        highpass_hz=safe_float(data.get("highpass_hz", 80.0), 80.0, min_val=20.0, max_val=500.0),
        compressor_threshold_db=safe_float(data.get("compressor_threshold_db", -18.0), -18.0, min_val=-60.0, max_val=0.0),
        compressor_ratio=safe_float(data.get("compressor_ratio", 3.0), 3.0, min_val=1.0, max_val=20.0),
        limiter_threshold_db=safe_float(data.get("limiter_threshold_db", -1.0), -1.0, min_val=-30.0, max_val=0.0),
    )

    intro_path = data.get("intro_path", "")
    if intro_path:
        validate_filepath(intro_path)
        config.intro_path = intro_path

    outro_path = data.get("outro_path", "")
    if outro_path:
        validate_filepath(outro_path)
        config.outro_path = outro_path

    eq_profiles = data.get("eq_profiles")
    if eq_profiles and isinstance(eq_profiles, dict):
        config.eq_profiles = eq_profiles

    result = polish_podcast(
        audio_path=filepath,
        config=config,
        output_path_val=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "speakers_detected": result.speakers_detected,
        "chapters": result.chapters,
        "loudness_lufs": result.loudness_lufs,
        "processing_steps": result.processing_steps,
    }


@audio_adv_bp.route("/podcast/detect-speakers", methods=["POST"])
@require_csrf
@async_job("detect_speakers")
def route_detect_speakers(job_id, filepath, data):
    """Detect speakers in audio for podcast processing."""
    from opencut.core.podcast_suite import detect_speakers

    segments = detect_speakers(
        audio_path=filepath,
        on_progress=lambda pct, msg: None,
    )

    return {
        "segments": [
            {
                "speaker_id": s.speaker_id,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in segments
        ],
        "speaker_count": len(set(s.speaker_id for s in segments)),
    }


@audio_adv_bp.route("/podcast/per-speaker-process", methods=["POST"])
@require_csrf
@async_job("per_speaker_process")
def route_per_speaker_process(job_id, filepath, data):
    """Apply per-speaker EQ and level processing."""
    from opencut.core.podcast_suite import (
        PodcastConfig,
        SpeakerSegment,
        apply_per_speaker_processing,
    )

    segments_data = data.get("segments", [])
    if not segments_data:
        raise ValueError("No speaker segments provided. Run detect-speakers first.")

    segments = [
        SpeakerSegment(
            speaker_id=s.get("speaker_id", "speaker_0"),
            start=float(s.get("start", 0)),
            end=float(s.get("end", 0)),
            text=s.get("text", ""),
        )
        for s in segments_data
    ]

    config = PodcastConfig(
        target_lufs=safe_float(data.get("target_lufs", -16.0), -16.0, min_val=-36.0, max_val=-10.0),
    )

    eq_profiles = data.get("eq_profiles")
    if eq_profiles and isinstance(eq_profiles, dict):
        config.eq_profiles = eq_profiles

    result = apply_per_speaker_processing(
        audio_path=filepath,
        speaker_segments=segments,
        output_path_val=data.get("output_path"),
        config=config,
        on_progress=lambda pct, msg: None,
    )

    return {"output_path": result}


# ===========================================================================
# Real-Time Voice Conversion
# ===========================================================================

@audio_adv_bp.route("/voice/models", methods=["GET"])
def route_list_voice_models():
    """List available voice conversion models."""
    from opencut.core.realtime_voice import list_voice_models
    models = list_voice_models()
    return jsonify({"models": models, "count": len(models)})


@audio_adv_bp.route("/voice/convert/start", methods=["POST"])
@require_csrf
@async_job("voice_convert_start", filepath_required=False)
def route_start_voice_conversion(job_id, filepath, data):
    """Start a real-time voice conversion session."""
    from opencut.core.realtime_voice import VoiceConversionConfig, start_voice_conversion

    model = data.get("model", "pitch_up")
    config = VoiceConversionConfig(
        pitch_shift_semitones=safe_float(data.get("pitch_shift", 0.0), 0.0, min_val=-12.0, max_val=12.0),
        formant_shift=safe_float(data.get("formant_shift", 0.0), 0.0, min_val=-1.0, max_val=1.0),
        model_name=model,
        sample_rate=safe_int(data.get("sample_rate", 48000), 48000, min_val=8000, max_val=96000),
        apply_noise_gate=safe_bool(data.get("noise_gate", True), True),
    )

    session = start_voice_conversion(
        model_path=model,
        config=config,
        output_dir=data.get("output_dir", ""),
        on_progress=lambda pct, msg: None,
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "model": session.model_path,
        "output_path": session.output_path,
    }


@audio_adv_bp.route("/voice/convert/stop", methods=["POST"])
@require_csrf
@async_job("voice_convert_stop", filepath_required=False)
def route_stop_voice_conversion(job_id, filepath, data):
    """Stop the active voice conversion session."""
    from opencut.core.realtime_voice import stop_voice_conversion

    session = stop_voice_conversion(
        on_progress=lambda pct, msg: None,
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "output_path": session.output_path,
        "duration": session.stopped_at - session.started_at if session.stopped_at else 0,
    }


# ===========================================================================
# ADR Cueing System
# ===========================================================================

@audio_adv_bp.route("/adr/cue-sheet", methods=["POST"])
@require_csrf
@async_job("adr_cue_sheet", filepath_required=False)
def route_adr_cue_sheet(job_id, filepath, data):
    """Create an ADR cue sheet from transcript with marked lines."""
    from opencut.core.adr_cueing import create_adr_cue_sheet

    transcript = data.get("transcript", [])
    if not transcript:
        raise ValueError("No transcript provided.")

    marked_lines = data.get("marked_lines", [])
    if not marked_lines:
        raise ValueError("No marked lines provided.")

    export_format = data.get("format", "json")
    if export_format not in ("json", "csv", "txt"):
        export_format = "json"

    result = create_adr_cue_sheet(
        transcript=transcript,
        marked_lines=marked_lines,
        output_path_val=data.get("output_path"),
        project_name=data.get("project_name", "Untitled"),
        export_format=export_format,
        on_progress=lambda pct, msg: None,
    )

    return {
        "project_name": result.project_name,
        "total_lines": result.total_lines,
        "total_duration": result.total_duration,
        "cues": [
            {
                "cue_id": c.cue_id,
                "character": c.character,
                "line_text": c.line_text,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "reason": c.reason,
                "priority": c.priority,
                "status": c.status,
            }
            for c in result.cues
        ],
    }


@audio_adv_bp.route("/adr/guide", methods=["POST"])
@require_csrf
@async_job("adr_guide")
def route_adr_guide(job_id, filepath, data):
    """Generate an ADR guide video for a specific cue."""
    from opencut.core.adr_cueing import ADRCue, generate_adr_guide

    cue_data = data.get("cue", {})
    if not cue_data:
        raise ValueError("No cue data provided.")

    cue = ADRCue(
        cue_id=cue_data.get("cue_id", "ADR-0001"),
        character=cue_data.get("character", "Unknown"),
        line_text=cue_data.get("line_text", ""),
        start_time=safe_float(cue_data.get("start_time", 0), 0),
        end_time=safe_float(cue_data.get("end_time", 0), 0),
        reason=cue_data.get("reason", "performance"),
        priority=cue_data.get("priority", "normal"),
    )

    if cue.end_time <= cue.start_time:
        raise ValueError("Cue end_time must be greater than start_time.")

    result = generate_adr_guide(
        video_path=filepath,
        cue=cue,
        output_path_val=data.get("output_path"),
        preroll=safe_float(data.get("preroll", 3.0), 3.0, min_val=0.0, max_val=10.0),
        postroll=safe_float(data.get("postroll", 2.0), 2.0, min_val=0.0, max_val=10.0),
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "cue_id": result.cue_id,
        "duration": result.duration,
        "preroll": result.preroll_seconds,
        "postroll": result.postroll_seconds,
    }


@audio_adv_bp.route("/adr/sync", methods=["POST"])
@require_csrf
@async_job("adr_sync")
def route_adr_sync(job_id, filepath, data):
    """Sync an ADR recording to the original timing."""
    from opencut.core.adr_cueing import ADRCue, sync_adr_recording

    recorded_path = data.get("recorded_path", "")
    if not recorded_path:
        raise ValueError("No recorded_path provided.")
    validate_filepath(recorded_path)

    cue_data = data.get("cue", {})
    if not cue_data:
        raise ValueError("No cue data provided.")

    cue = ADRCue(
        cue_id=cue_data.get("cue_id", "ADR-0001"),
        character=cue_data.get("character", "Unknown"),
        line_text=cue_data.get("line_text", ""),
        start_time=safe_float(cue_data.get("start_time", 0), 0),
        end_time=safe_float(cue_data.get("end_time", 0), 0),
    )

    result = sync_adr_recording(
        original_path=filepath,
        recorded_path=recorded_path,
        cue=cue,
        output_path_val=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "cue_id": result.cue_id,
        "time_offset": result.time_offset,
        "sync_quality": result.sync_quality,
        "original_duration": result.original_duration,
        "recorded_duration": result.recorded_duration,
    }


# ===========================================================================
# Surround Sound Panning & Upmix
# ===========================================================================

@audio_adv_bp.route("/audio/surround/upmix", methods=["POST"])
@require_csrf
@async_job("surround_upmix")
def route_surround_upmix(job_id, filepath, data):
    """Upmix stereo audio to 5.1 or 7.1 surround."""
    from opencut.core.surround_mix import upmix_to_surround

    channels = data.get("channels", "5.1")
    if channels not in ("5.1", "7.1"):
        channels = "5.1"

    result = upmix_to_surround(
        audio_path=filepath,
        channels=channels,
        output_path_val=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "source_channels": result.source_channels,
        "target_channels": result.target_channels,
        "target_layout": result.target_layout,
    }


@audio_adv_bp.route("/audio/surround/pan", methods=["POST"])
@require_csrf
@async_job("surround_pan")
def route_surround_pan(job_id, filepath, data):
    """Pan audio to a specific position in the surround field."""
    from opencut.core.surround_mix import SurroundPosition, pan_in_surround

    position = SurroundPosition(
        angle=safe_float(data.get("angle", 0.0), 0.0, min_val=-180.0, max_val=180.0),
        elevation=safe_float(data.get("elevation", 0.0), 0.0, min_val=-90.0, max_val=90.0),
        distance=safe_float(data.get("distance", 1.0), 1.0, min_val=0.0, max_val=1.0),
        lfe_amount=safe_float(data.get("lfe_amount", 0.0), 0.0, min_val=0.0, max_val=1.0),
    )

    channels = data.get("channels", "5.1")
    if channels not in ("5.1", "7.1"):
        channels = "5.1"

    result = pan_in_surround(
        audio_path=filepath,
        position=position,
        channels=channels,
        output_path_val=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "position": {
            "angle": position.angle,
            "elevation": position.elevation,
            "distance": position.distance,
            "lfe_amount": position.lfe_amount,
        },
        "channels": result.channels,
        "layout": result.layout,
    }


@audio_adv_bp.route("/audio/surround/downmix", methods=["POST"])
@require_csrf
@async_job("surround_downmix")
def route_surround_downmix(job_id, filepath, data):
    """Create a stereo downmix preview of surround audio."""
    from opencut.core.surround_mix import downmix_preview

    result = downmix_preview(
        surround_path=filepath,
        output_path_val=data.get("output_path"),
        on_progress=lambda pct, msg: None,
    )

    return {"output_path": result}


@audio_adv_bp.route("/audio/surround/export", methods=["POST"])
@require_csrf
@async_job("surround_export")
def route_surround_export(job_id, filepath, data):
    """Export multichannel audio in various formats."""
    from opencut.core.surround_mix import EXPORT_FORMATS, export_multichannel

    fmt = data.get("format", "wav")
    if fmt not in EXPORT_FORMATS:
        fmt = "wav"

    result = export_multichannel(
        audio_path=filepath,
        format=fmt,
        output_path_val=data.get("output_path"),
        bitrate=data.get("bitrate", "640k"),
        on_progress=lambda pct, msg: None,
    )

    return {"output_path": result, "format": fmt}


# ===========================================================================
# Audio-Reactive Visualizers
# ===========================================================================

@audio_adv_bp.route("/audio/visualizer", methods=["POST"])
@require_csrf
@async_job("audio_visualizer")
def route_audio_visualizer(job_id, filepath, data):
    """Generate an audio-reactive visualizer video."""
    from opencut.core.audio_visualizer import (
        VISUALIZER_STYLES,
        VisualizerConfig,
        generate_visualizer,
    )

    style = data.get("style", "waveform_bars")
    if style not in VISUALIZER_STYLES:
        style = "waveform_bars"

    config = VisualizerConfig(
        style=style,
        width=safe_int(data.get("width", 1920), 1920, min_val=320, max_val=3840),
        height=safe_int(data.get("height", 1080), 1080, min_val=240, max_val=2160),
        fps=safe_int(data.get("fps", 30), 30, min_val=15, max_val=60),
        color_primary=data.get("color_primary", "0x00AAFF"),
        color_secondary=data.get("color_secondary", "0xFF6600"),
        color_background=data.get("color_background", "0x1A1A2E"),
        bar_count=safe_int(data.get("bar_count", 64), 64, min_val=8, max_val=256),
        opacity=safe_float(data.get("opacity", 0.8), 0.8, min_val=0.0, max_val=1.0),
        as_overlay=safe_bool(data.get("as_overlay", False), False),
        overlay_position=data.get("overlay_position", "bottom"),
        overlay_height_ratio=safe_float(data.get("overlay_height_ratio", 0.2), 0.2, min_val=0.05, max_val=0.5),
    )

    video_path = data.get("video_path", "")
    if video_path:
        validate_filepath(video_path)

    result = generate_visualizer(
        audio_path=filepath,
        style=style,
        output_path_val=data.get("output_path"),
        config=config,
        video_path=video_path if video_path else None,
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "style": result.style,
        "width": result.width,
        "height": result.height,
        "fps": result.fps,
    }


@audio_adv_bp.route("/audio/visualizer/styles", methods=["GET"])
def route_visualizer_styles():
    """List available visualizer styles."""
    from opencut.core.audio_visualizer import VISUALIZER_STYLES
    return jsonify({
        "styles": [
            {"name": k, "label": v["label"], "description": v["description"]}
            for k, v in VISUALIZER_STYLES.items()
        ],
    })


# ===========================================================================
# Audio Description Track Generator
# ===========================================================================

@audio_adv_bp.route("/audio/description/generate", methods=["POST"])
@require_csrf
@async_job("audio_description")
def route_audio_description(job_id, filepath, data):
    """Generate an audio description track for a video."""
    from opencut.core.audio_description import generate_audio_description

    result = generate_audio_description(
        video_path=filepath,
        output_path_val=data.get("output_path"),
        descriptions=data.get("descriptions"),
        transcript=data.get("transcript"),
        voice=data.get("voice", "default"),
        min_gap_seconds=safe_float(data.get("min_gap_seconds", 2.0), 2.0, min_val=1.0, max_val=10.0),
        on_progress=lambda pct, msg: None,
    )

    return {
        "output_path": result.output_path,
        "gaps_found": result.gaps_found,
        "descriptions_added": result.descriptions_added,
        "total_description_duration": result.total_description_duration,
    }


@audio_adv_bp.route("/audio/description/gaps", methods=["POST"])
@require_csrf
@async_job("find_description_gaps")
def route_find_description_gaps(job_id, filepath, data):
    """Find gaps in dialogue suitable for audio descriptions."""
    from opencut.core.audio_description import find_description_gaps

    gaps = find_description_gaps(
        audio_path=filepath,
        transcript=data.get("transcript"),
        min_gap_seconds=safe_float(data.get("min_gap_seconds", 2.0), 2.0, min_val=0.5, max_val=30.0),
        max_gap_seconds=safe_float(data.get("max_gap_seconds", 15.0), 15.0, min_val=2.0, max_val=60.0),
        silence_threshold_db=safe_float(data.get("threshold_db", -35.0), -35.0, min_val=-60.0, max_val=-10.0),
        on_progress=lambda pct, msg: None,
    )

    return {
        "gaps": [
            {
                "start": g.start,
                "end": g.end,
                "duration": g.duration,
                "suitable": g.suitable,
                "max_words": g.max_words,
            }
            for g in gaps
        ],
        "total_gaps": len(gaps),
    }


@audio_adv_bp.route("/audio/description/synthesize", methods=["POST"])
@require_csrf
@async_job("synthesize_description", filepath_required=False)
def route_synthesize_description(job_id, filepath, data):
    """Synthesize audio description speech from text."""
    from opencut.core.audio_description import synthesize_description

    text = data.get("text", "")
    if not text:
        raise ValueError("No text provided for synthesis.")

    result = synthesize_description(
        text=text,
        voice=data.get("voice", "default"),
        output_path_val=data.get("output_path"),
        speed=safe_float(data.get("speed", 1.1), 1.1, min_val=0.5, max_val=2.0),
        on_progress=lambda pct, msg: None,
    )

    return {"output_path": result}


# ===========================================================================
# Voice Commands
# ===========================================================================

@audio_adv_bp.route("/voice/commands/mapping", methods=["GET"])
def route_voice_command_mapping():
    """Return the voice command mapping for UI display."""
    from opencut.core.voice_commands import get_command_mapping
    mapping = get_command_mapping()
    return jsonify({"commands": mapping})


@audio_adv_bp.route("/voice/commands/parse", methods=["POST"])
@require_csrf
def route_parse_voice_command():
    """Parse raw voice text into a structured command."""
    from opencut.core.voice_commands import parse_voice_command

    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    custom_commands = data.get("custom_commands")

    cmd = parse_voice_command(text, custom_commands=custom_commands)

    return jsonify({
        "raw_text": cmd.raw_text,
        "command": cmd.command,
        "action": cmd.action,
        "parameters": cmd.parameters,
        "confidence": cmd.confidence,
        "matched": cmd.matched,
    })


@audio_adv_bp.route("/voice/commands/start", methods=["POST"])
@require_csrf
@async_job("voice_listener_start", filepath_required=False)
def route_start_voice_listener(job_id, filepath, data):
    """Start the voice command listener."""
    from opencut.core.voice_commands import VoiceCommandConfig, start_voice_listener

    config = VoiceCommandConfig(
        wake_word=data.get("wake_word", "opencut"),
        language=data.get("language", "en-US"),
        timeout_seconds=safe_float(data.get("timeout", 5.0), 5.0, min_val=1.0, max_val=30.0),
        require_wake_word=safe_bool(data.get("require_wake_word", True), True),
        continuous=safe_bool(data.get("continuous", True), True),
    )

    custom_commands = data.get("custom_commands")
    if custom_commands and isinstance(custom_commands, dict):
        config.custom_commands = custom_commands

    session = start_voice_listener(
        config=config,
        on_progress=lambda pct, msg: None,
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "wake_word": config.wake_word,
    }


@audio_adv_bp.route("/voice/commands/stop", methods=["POST"])
@require_csrf
@async_job("voice_listener_stop", filepath_required=False)
def route_stop_voice_listener(job_id, filepath, data):
    """Stop the voice command listener."""
    from opencut.core.voice_commands import stop_voice_listener

    session = stop_voice_listener(
        on_progress=lambda pct, msg: None,
    )

    return {
        "session_id": session.session_id,
        "status": session.status,
        "commands_processed": session.commands_processed,
        "duration": session.stopped_at - session.started_at if session.stopped_at else 0,
    }
