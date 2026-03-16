"""
OpenCut Captions Routes

Transcription, caption styles, burn-in, animated captions, whisperx,
translation, karaoke, format conversion.
"""

import json
import logging
import os
import subprocess as _sp
import sys
import tempfile
import threading
import time
from flask import Blueprint, request, jsonify

from opencut.jobs import jobs, job_lock, _new_job, _update_job, _safe_error, _is_cancelled
from opencut.helpers import _try_import, _try_import_from, _resolve_output_dir, _unique_output_path, _run_ffmpeg_with_progress, _make_sequence_name, OPENCUT_DIR
from opencut.security import validate_path, validate_filepath, require_csrf, safe_pip_install, safe_float, safe_int, VALID_WHISPER_MODELS, require_rate_limit

# Core imports used by multiple routes (try relative/absolute)
try:
    from ..utils.media import probe as _probe_media
    from ..utils.config import SilenceConfig, ExportConfig, CaptionConfig, get_preset
    from ..core.silence import detect_speech, get_edit_summary
    from ..core.zoom import generate_zoom_events
    from ..export.premiere import export_premiere_xml
    from ..export.srt import export_srt, export_vtt, export_json, export_ass, rgb_to_ass_color
except ImportError:
    from opencut.utils.media import probe as _probe_media
    from opencut.utils.config import SilenceConfig, ExportConfig, CaptionConfig, get_preset
    from opencut.core.silence import detect_speech, get_edit_summary
    from opencut.core.zoom import generate_zoom_events
    from opencut.export.premiere import export_premiere_xml
    from opencut.export.srt import export_srt, export_vtt, export_json, export_ass, rgb_to_ass_color

logger = logging.getLogger("opencut")

captions_bp = Blueprint("captions", __name__)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------
@captions_bp.route("/captions", methods=["POST"])
@require_csrf
def generate_captions():
    """Generate captions/subtitles."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        return jsonify({"error": f"Invalid model: {model}"}), 400
    language = data.get("language", None)
    sub_format = data.get("format", "srt")
    word_timestamps = data.get("word_timestamps", True)

    job_id = _new_job("captions", filepath)

    def _process():
        try:
            from opencut.core.captions import transcribe, check_whisper_available

            available, backend = check_whisper_available()
            if not available:
                _update_job(
                    job_id, status="error",
                    error="No Whisper backend installed. Run: pip install faster-whisper",
                    message="Whisper not installed",
                )
                return

            _update_job(job_id, progress=10, message=f"Loading {model} model ({backend})...")

            config = CaptionConfig(
                model=model,
                language=language,
                word_timestamps=word_timestamps,
            )

            _update_job(job_id, progress=20, message="Transcribing audio (this takes a while for long files)...")
            result = transcribe(filepath, config=config)

            if _is_cancelled(job_id):
                return

            _update_job(job_id, progress=80, message="Writing subtitle file...")

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(effective_dir, f"{base_name}.{sub_format}")

            if sub_format == "vtt":
                export_vtt(result, out_path)
            elif sub_format == "json":
                export_json(result, out_path)
            elif sub_format == "ass":
                info = _probe_media(filepath)
                vid_w = info.video.width if info.video else 1920
                vid_h = info.video.height if info.video else 1080
                export_ass(
                    result, out_path,
                    video_width=vid_w, video_height=vid_h,
                    karaoke=word_timestamps,
                )
            else:
                export_srt(result, out_path)

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result={
                    "output_path": out_path,
                    "language": result.language,
                    "segments": len(result.segments),
                    "caption_segments": len(result.segments),
                    "words": result.word_count,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Captions error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Styled Captions (video overlay)
# ---------------------------------------------------------------------------
@captions_bp.route("/caption-styles", methods=["GET"])
def get_caption_styles():
    """Return available caption style presets for the panel preview."""
    try:
        from opencut.core.styled_captions import get_style_info
        return jsonify({"styles": get_style_info()})
    except Exception as e:
        return _safe_error(e, context="caption styles")


@captions_bp.route("/styled-captions", methods=["POST"])
@require_csrf
def styled_captions_route():
    """Generate a transparent video overlay with styled, animated captions."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    style_name = data.get("style", "youtube_bold")
    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        return jsonify({"error": f"Invalid model: {model}"}), 400
    language = data.get("language", None)
    custom_action_words = data.get("action_words", [])
    auto_detect_energy = data.get("auto_detect_energy", True)
    # Optional: pre-existing speech segments for remapping
    remap_segments_raw = data.get("remap_segments", None)

    job_id = _new_job("styled-captions", filepath)

    def _process():
        try:
            from opencut.core.captions import transcribe, check_whisper_available
            from opencut.core.styled_captions import (
                render_styled_caption_video, STYLES, detect_action_words_by_energy,
                get_action_word_indices,
            )

            available, backend = check_whisper_available()
            if not available:
                _update_job(
                    job_id, status="error",
                    error="Whisper is required for styled captions. Install from the Captions tab.",
                    message="Whisper not installed",
                )
                return

            # Get video resolution
            info = _probe_media(filepath)
            vid_w = info.video.width if info.video else 1920
            vid_h = info.video.height if info.video else 1080
            vid_fps = info.video.fps if info.video else 30.0

            # Step 1: Transcribe
            _update_job(job_id, progress=10, message=f"Transcribing with {backend} ({model})...")
            config = CaptionConfig(model=model, language=language, word_timestamps=True)
            transcription = transcribe(filepath, config=config)

            if _is_cancelled(job_id):
                return

            # Optional: remap captions to edited timeline
            if remap_segments_raw:
                from opencut.core.captions import remap_captions_to_segments
                from opencut.core.silence import TimeSegment
                remap_segs = [TimeSegment(s["start"], s["end"]) for s in remap_segments_raw]
                transcription = remap_captions_to_segments(transcription, remap_segs)

            # Step 2: Detect action words
            _update_job(job_id, progress=50, message="Detecting action words...")
            all_words = []
            for seg in transcription.segments:
                all_words.extend(seg.words)

            energy_indices = set()
            if auto_detect_energy and all_words:
                try:
                    energy_indices = detect_action_words_by_energy(filepath, all_words)
                except Exception as e:
                    logger.warning(f"Energy detection failed: {e}")

            action_indices = get_action_word_indices(
                all_words,
                custom_words=custom_action_words if custom_action_words else None,
                use_keywords=True,
                energy_indices=energy_indices,
            )

            if _is_cancelled(job_id):
                return

            # Step 3: Render overlay video
            _update_job(job_id, progress=60, message="Rendering styled captions...")
            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            overlay_path = os.path.join(effective_dir, f"{base_name}_captions.mov")

            def _on_render_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            result = render_styled_caption_video(
                transcription,
                overlay_path,
                style_name=style_name,
                video_width=vid_w,
                video_height=vid_h,
                fps=vid_fps,
                action_indices=action_indices,
                total_duration=transcription.duration or info.duration,
                on_progress=_on_render_progress,
            )

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result={
                    "output_path": overlay_path,
                    "overlay_path": overlay_path,
                    "style": style_name,
                    "frames_rendered": result["frames_rendered"],
                    "action_words_found": len(action_indices),
                    "caption_segments": len(transcription.segments),
                    "words": sum(len(s.words) for s in transcription.segments),
                    "language": transcription.language,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Styled captions error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Transcript Editor
# ---------------------------------------------------------------------------
@captions_bp.route("/transcript", methods=["POST"])
@require_csrf
def get_transcript():
    """Transcribe and return full word-level transcript for editing."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model = data.get("model", "base")
    if model not in VALID_WHISPER_MODELS:
        return jsonify({"error": f"Invalid model: {model}"}), 400
    language = data.get("language", None)

    job_id = _new_job("transcript", filepath)

    def _process():
        try:
            from opencut.core.captions import transcribe, check_whisper_available

            available, backend = check_whisper_available()
            if not available:
                _update_job(
                    job_id, status="error",
                    error="Whisper is required for transcription.",
                    message="Whisper not installed",
                )
                return

            _update_job(job_id, progress=10, message=f"Transcribing with {backend} ({model})...")
            config = CaptionConfig(model=model, language=language, word_timestamps=True)
            result = transcribe(filepath, config=config)

            if _is_cancelled(job_id):
                return

            # Build editable transcript structure
            transcript_data = {
                "language": result.language,
                "duration": result.duration,
                "word_count": result.word_count,
                "full_text": result.text,
                "segments": [
                    {
                        "id": i,
                        "text": seg.text.strip(),
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "speaker": seg.speaker,
                        "words": [
                            {
                                "text": w.text.strip(),
                                "start": round(w.start, 3),
                                "end": round(w.end, 3),
                                "confidence": round(w.confidence, 3),
                            }
                            for w in seg.words
                        ],
                    }
                    for i, seg in enumerate(result.segments)
                ],
            }

            _update_job(
                job_id, status="complete", progress=100,
                message="Transcript ready!",
                result=transcript_data,
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Transcript error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@captions_bp.route("/transcript/export", methods=["POST"])
@require_csrf
def export_edited_transcript():
    """Export an edited transcript to SRT/VTT/ASS/JSON."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    segments_data = data.get("segments", [])
    sub_format = data.get("format", "srt")
    language = data.get("language", "en")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not segments_data:
        return jsonify({"error": "No transcript segments provided"}), 400

    try:
        from opencut.core.captions import CaptionSegment, Word, TranscriptionResult

        # Reconstruct TranscriptionResult from edited segments
        caption_segments = []
        for seg in segments_data:
            words = [
                Word(
                    text=w.get("text", ""),
                    start=safe_float(w.get("start", 0.0)),
                    end=safe_float(w.get("end", 0.0)),
                    confidence=safe_float(w.get("confidence", 1.0), default=1.0),
                )
                for w in seg.get("words", [])
            ]
            caption_segments.append(CaptionSegment(
                text=seg.get("text", ""),
                start=safe_float(seg.get("start", 0.0)),
                end=safe_float(seg.get("end", 0.0)),
                words=words,
                speaker=seg.get("speaker"),
            ))

        result = TranscriptionResult(
            segments=caption_segments,
            language=language,
            duration=caption_segments[-1].end if caption_segments else 0.0,
        )

        effective_dir = _resolve_output_dir(filepath, output_dir)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(effective_dir, f"{base_name}_edited.{sub_format}")

        if sub_format == "vtt":
            export_vtt(result, out_path)
        elif sub_format == "json":
            export_json(result, out_path)
        elif sub_format == "ass":
            info = _probe_media(filepath) if os.path.isfile(filepath) else None
            vid_w = info.video.width if info and info.video else 1920
            vid_h = info.video.height if info and info.video else 1080
            export_ass(result, out_path, video_width=vid_w, video_height=vid_h, karaoke=True)
        else:
            export_srt(result, out_path)

        return jsonify({"output_path": out_path, "format": sub_format})
    except Exception as e:
        return _safe_error(e)


# ---------------------------------------------------------------------------
# Emoji Map for Captions
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "laugh": "\U0001f602", "lol": "\U0001f602", "haha": "\U0001f602", "funny": "\U0001f602",
    "love": "\u2764\ufe0f", "heart": "\u2764\ufe0f", "beautiful": "\u2764\ufe0f",
    "fire": "\U0001f525", "hot": "\U0001f525", "lit": "\U0001f525", "amazing": "\U0001f525",
    "cry": "\U0001f622", "sad": "\U0001f622", "tears": "\U0001f622",
    "wow": "\U0001f62e", "shocked": "\U0001f62e", "omg": "\U0001f62e", "unbelievable": "\U0001f62e",
    "think": "\U0001f914", "hmm": "\U0001f914", "wondering": "\U0001f914",
    "thumbs": "\U0001f44d", "great": "\U0001f44d", "good": "\U0001f44d", "nice": "\U0001f44d",
    "clap": "\U0001f44f", "bravo": "\U0001f44f", "congrats": "\U0001f44f",
    "money": "\U0001f4b0", "rich": "\U0001f4b0", "dollar": "\U0001f4b0", "expensive": "\U0001f4b0",
    "star": "\u2b50", "perfect": "\u2b50", "excellent": "\u2b50",
    "rocket": "\U0001f680", "launch": "\U0001f680", "skyrocket": "\U0001f680", "moon": "\U0001f680",
    "brain": "\U0001f9e0", "smart": "\U0001f9e0", "genius": "\U0001f9e0", "intelligent": "\U0001f9e0",
    "muscle": "\U0001f4aa", "strong": "\U0001f4aa", "power": "\U0001f4aa", "strength": "\U0001f4aa",
    "warning": "\u26a0\ufe0f", "danger": "\u26a0\ufe0f", "careful": "\u26a0\ufe0f",
    "check": "\u2705", "correct": "\u2705", "yes": "\u2705", "done": "\u2705",
    "wrong": "\u274c", "no": "\u274c", "false": "\u274c", "bad": "\u274c",
    "question": "\u2753", "why": "\u2753", "how": "\u2753",
    "eyes": "\U0001f440", "look": "\U0001f440", "see": "\U0001f440", "watch": "\U0001f440",
    "celebrate": "\U0001f389", "party": "\U0001f389", "winner": "\U0001f389", "victory": "\U0001f389",
    "sleep": "\U0001f634", "tired": "\U0001f634", "boring": "\U0001f634",
    "music": "\U0001f3b5", "song": "\U0001f3b5", "singing": "\U0001f3b5",
    "camera": "\U0001f4f8", "photo": "\U0001f4f8", "picture": "\U0001f4f8",
    "time": "\u23f0", "clock": "\u23f0", "hurry": "\u23f0", "fast": "\u23f0",
    "earth": "\U0001f30d", "world": "\U0001f30d", "global": "\U0001f30d",
    "sun": "\u2600\ufe0f", "sunny": "\u2600\ufe0f", "bright": "\u2600\ufe0f",
    "rain": "\U0001f327\ufe0f", "storm": "\U0001f327\ufe0f", "weather": "\U0001f327\ufe0f",
    "food": "\U0001f355", "eat": "\U0001f355", "hungry": "\U0001f355", "delicious": "\U0001f355",
    "coffee": "\u2615", "morning": "\u2615", "energy": "\u2615",
    "book": "\U0001f4da", "read": "\U0001f4da", "study": "\U0001f4da", "learn": "\U0001f4da",
    "computer": "\U0001f4bb", "code": "\U0001f4bb", "tech": "\U0001f4bb", "programming": "\U0001f4bb",
    "phone": "\U0001f4f1", "call": "\U0001f4f1", "mobile": "\U0001f4f1",
    "key": "\U0001f511", "secret": "\U0001f511", "unlock": "\U0001f511",
    "light": "\U0001f4a1", "idea": "\U0001f4a1", "tip": "\U0001f4a1", "insight": "\U0001f4a1",
    "target": "\U0001f3af", "goal": "\U0001f3af", "focus": "\U0001f3af", "aim": "\U0001f3af",
    "gem": "\U0001f48e", "diamond": "\U0001f48e", "valuable": "\U0001f48e", "premium": "\U0001f48e",
    "crown": "\U0001f451", "king": "\U0001f451", "queen": "\U0001f451", "best": "\U0001f451",
    "ghost": "\U0001f47b", "scary": "\U0001f47b", "spooky": "\U0001f47b", "halloween": "\U0001f47b",
    "skull": "\U0001f480", "dead": "\U0001f480", "kill": "\U0001f480", "destroy": "\U0001f480",
    "hundred": "\U0001f4af", "percent": "\U0001f4af",
}


@captions_bp.route("/captions/emoji-map", methods=["GET"])
def get_emoji_map():
    """Return the keyword->emoji mapping for auto-emoji insertion."""
    return jsonify({"emoji_map": EMOJI_MAP})


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
@captions_bp.route("/full", methods=["POST"])
@require_csrf
def full_pipeline():
    """Run silence removal + zoom + optional captions."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    preset = data.get("preset", "youtube")
    skip_captions = data.get("skip_captions", False)
    skip_zoom = data.get("skip_zoom", False)
    remove_fillers = data.get("remove_fillers", False)
    seq_name = data.get("sequence_name", "")

    job_id = _new_job("full", filepath)

    def _process():
        try:
            cfg = get_preset(preset)
            effective_name = seq_name or _make_sequence_name(filepath, "Full Edit")
            ecfg = ExportConfig(sequence_name=effective_name)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")
            srt_path = os.path.join(effective_dir, f"{base_name}_opencut.srt")

            # Probe once for all pipeline steps
            _finfo = _probe_media(filepath)
            _fdur = _finfo.duration if _finfo else 0.0

            # Calculate total steps
            total_steps = 2  # silence + export always
            if not skip_zoom:
                total_steps += 1
            if not skip_captions:
                total_steps += 1
            if remove_fillers:
                total_steps += 1
            step = [0]
            def next_step(msg):
                step[0] += 1
                pct = int(5 + (step[0] / total_steps) * 85)
                _update_job(job_id, progress=pct, message=f"Step {step[0]}/{total_steps}: {msg}")

            # Step: Silence detection
            next_step("Detecting silences...")
            segments = detect_speech(filepath, config=cfg.silence, file_duration=_fdur)
            summary = get_edit_summary(filepath, segments, file_duration=_fdur)

            if _is_cancelled(job_id):
                return

            # Step: Filler word removal (requires Whisper)
            filler_stats = None
            transcription_result = None
            if remove_fillers:
                from opencut.core.captions import check_whisper_available, transcribe
                from opencut.core.fillers import detect_fillers, remove_fillers_from_segments

                available, backend = check_whisper_available()
                if available:
                    next_step(f"Detecting filler words ({backend})...")
                    filler_cfg = CaptionConfig(
                        model=cfg.captions.model,
                        language=cfg.captions.language,
                        word_timestamps=True,
                    )
                    # Use timeout to prevent hanging (10 min max)
                    try:
                        transcription_result = transcribe(filepath, config=filler_cfg, timeout=600)
                    except TimeoutError as te:
                        logger.warning(f"Filler detection timed out: {te}")
                        # Continue without filler removal
                        _update_job(job_id, message="Filler detection timed out, continuing without it...")
                        transcription_result = None
                    except Exception as te:
                        logger.warning(f"Filler detection failed: {te}")
                        _update_job(job_id, message="Filler detection failed, continuing without it...")
                        transcription_result = None

                    if transcription_result:
                        analysis = detect_fillers(transcription_result, include_context_fillers=True)

                        if analysis.hits:
                            segments = remove_fillers_from_segments(segments, analysis.hits)
                            # Recalculate summary with filler-cleaned segments
                            summary = get_edit_summary(filepath, segments, file_duration=_fdur)
                            logger.info(
                                f"Fillers removed: {len(analysis.hits)} instances, "
                                f"{analysis.total_filler_time:.1f}s"
                            )

                        filler_stats = {
                            "total_fillers": len(analysis.hits),
                            "removed_fillers": len(analysis.hits),
                            "total_filler_time": analysis.total_filler_time,
                            "filler_percentage": analysis.filler_percentage,
                            "total_words": analysis.total_words,
                            "breakdown": [
                                {"word": k, "count": c,
                                 "time": round(sum(h.duration for h in analysis.hits if h.filler_key == k), 2),
                                 "removed": True}
                                for k, c in sorted(analysis.filler_counts.items(), key=lambda x: -x[1])
                            ],
                        }
                else:
                    logger.warning("Filler removal requested but Whisper not installed, skipping")

            if _is_cancelled(job_id):
                return

            # Step: Zoom
            zoom_events = None
            if not skip_zoom:
                next_step("Analyzing emphasis points for zoom...")
                zoom_events = generate_zoom_events(filepath, config=cfg.zoom, speech_segments=segments)
            else:
                pass  # Don't increment step for skipped

            if _is_cancelled(job_id):
                return

            # Step: Captions
            captions_result = None
            if not skip_captions:
                from opencut.core.captions import check_whisper_available
                available, backend = check_whisper_available()
                if available:
                    next_step("Generating captions...")
                    # Reuse transcription from filler step if available
                    if transcription_result:
                        captions_result = transcription_result
                    else:
                        from opencut.core.captions import transcribe
                        captions_result = transcribe(filepath, config=cfg.captions)

                    # Remap caption timestamps to the condensed timeline
                    from opencut.core.captions import remap_captions_to_segments
                    captions_result = remap_captions_to_segments(captions_result, segments)
                    logger.info(
                        f"Captions remapped: {len(captions_result.segments)} segments, "
                        f"condensed duration {captions_result.duration:.1f}s"
                    )

                    export_srt(captions_result, srt_path)
                else:
                    pass  # Whisper not installed, skip silently

            if _is_cancelled(job_id):
                return

            # Export XML
            _update_job(job_id, progress=92, message="Exporting Premiere XML...")
            export_premiere_xml(filepath, segments, xml_path, config=ecfg, zoom_events=zoom_events)

            result_data = {
                "xml_path": xml_path,
                "summary": summary,
                "segments": len(segments),
                "zoom_events": len(zoom_events) if zoom_events else 0,
                "segments_data": [
                    {"start": round(s.start, 4), "end": round(s.end, 4)}
                    for s in segments
                ],
            }
            if captions_result:
                result_data["srt_path"] = srt_path
                result_data["caption_segments"] = len(captions_result.segments)
                result_data["words"] = captions_result.word_count
                result_data["language"] = captions_result.language
            if filler_stats:
                result_data["filler_stats"] = filler_stats

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result=result_data,
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Full pipeline error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Enhanced Captions (WhisperX, Translation, Karaoke)
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/enhanced/capabilities", methods=["GET"])
def captions_enhanced_capabilities():
    """Return enhanced caption capabilities."""
    try:
        from opencut.core.captions_enhanced import (
            check_whisperx_available, check_nllb_available, check_pysubs2_available,
            TRANSLATION_LANGUAGES,
        )
        return jsonify({
            "whisperx": check_whisperx_available(),
            "nllb": check_nllb_available(),
            "pysubs2": check_pysubs2_available(),
            "languages": TRANSLATION_LANGUAGES,
        })
    except Exception as e:
        return _safe_error(e)


@captions_bp.route("/captions/whisperx", methods=["POST"])
@require_csrf
def captions_whisperx():
    """Transcribe with WhisperX for word-level timestamps."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    model_size = data.get("model", "base")
    if model_size not in VALID_WHISPER_MODELS:
        return jsonify({"error": f"Invalid model: {model_size}"}), 400
    language = data.get("language", "")
    diarize = data.get("diarize", False)
    hf_token = data.get("hf_token", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = _new_job("whisperx", filepath)

    def _process():
        try:
            from opencut.core.captions_enhanced import whisperx_transcribe

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            result = whisperx_transcribe(
                filepath, model_size=model_size,
                language=language, diarize=diarize,
                hf_token=hf_token, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"WhisperX: {result['word_count']} words aligned",
                result=result,
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("WhisperX error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@captions_bp.route("/captions/translate", methods=["POST"])
@require_csrf
def captions_translate():
    """Translate caption segments using NLLB."""
    data = request.get_json(force=True)
    segments = data.get("segments", [])
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "es")
    output_dir = data.get("output_dir", "")
    output_format = data.get("format", "srt")
    filepath = data.get("filepath", "")

    if filepath:
        try:
            filepath = validate_filepath(filepath)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    job_id = _new_job("translate", filepath or "captions")

    def _process():
        try:
            from opencut.core.captions_enhanced import translate_segments

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            translated = translate_segments(
                segments, source_lang=source_lang,
                target_lang=target_lang, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Translated {len(translated)} segments to {target_lang}",
                result={"segments": translated, "target_lang": target_lang},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Translation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@captions_bp.route("/captions/karaoke", methods=["POST"])
@require_csrf
def captions_karaoke():
    """Export segments as ASS karaoke subtitles."""
    data = request.get_json(force=True)
    segments = data.get("segments", [])
    filepath = data.get("filepath", "")
    output_dir = data.get("output_dir", "")

    if filepath:
        try:
            filepath = validate_filepath(filepath)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    job_id = _new_job("karaoke", filepath or "captions")

    def _process():
        try:
            from opencut.core.captions_enhanced import segments_to_ass_karaoke

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir) if filepath else tempfile.gettempdir()
            base = os.path.splitext(os.path.basename(filepath))[0] if filepath else "captions"
            out_path = os.path.join(effective_dir, f"{base}_karaoke.ass")

            segments_to_ass_karaoke(
                segments, out_path,
                font_name=data.get("font", "Arial"),
                font_size=safe_int(data.get("font_size", 48), default=48),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Karaoke ASS exported!",
                result={"output_path": out_path},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Karaoke export error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@captions_bp.route("/captions/convert", methods=["POST"])
@require_csrf
def captions_convert():
    """Convert subtitle file between formats."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    target_format = data.get("format", "srt")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        from opencut.core.captions_enhanced import convert_subtitle_format
        out = convert_subtitle_format(filepath, output_format=target_format)
        return jsonify({"output_path": out})
    except Exception as e:
        return _safe_error(e)


@captions_bp.route("/captions/enhanced/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def captions_enhanced_install():
    """Install enhanced caption dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "whisperx")

    packages = {
        "whisperx": ["whisperx"],
        "pysubs2": ["pysubs2"],
        "nllb": ["ctranslate2", "sentencepiece", "huggingface-hub"],
    }

    pkgs = packages.get(component, [])
    if not pkgs:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    job_id = _new_job("install", component)

    def _process():
        try:
            for i, pkg in enumerate(pkgs):
                pct = int((i / len(pkgs)) * 90)
                _update_job(job_id, progress=pct, message=f"Installing {pkg}...")
                safe_pip_install(pkg)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Installed {component}!",
                result={"component": component},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Caption Burn-in
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/burnin/styles", methods=["GET"])
def burnin_styles():
    """Return available burn-in styles."""
    try:
        from opencut.core.caption_burnin import get_burnin_styles
        return jsonify({"styles": get_burnin_styles()})
    except Exception as e:
        return _safe_error(e)


@captions_bp.route("/captions/burnin/file", methods=["POST"])
@require_csrf
def burnin_from_file():
    """Burn a subtitle file into video."""
    data = request.get_json(force=True)
    video_path = data.get("filepath", "").strip()
    subtitle_path = data.get("subtitle_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not video_path:
        return jsonify({"error": "Video file not found"}), 400
    try:
        video_path = validate_filepath(video_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not subtitle_path:
        return jsonify({"error": "Subtitle file not found"}), 400
    try:
        subtitle_path = validate_filepath(subtitle_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    job_id = _new_job("burnin", video_path)

    def _process():
        try:
            from opencut.core.caption_burnin import burnin_subtitles

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(video_path, output_dir)
            out = burnin_subtitles(
                video_path, subtitle_path,
                output_dir=effective_dir,
                font_size=safe_int(data.get("font_size", 0)),
                margin_bottom=safe_int(data.get("margin_bottom", 0)),
                force_style=data.get("force_style", ""),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Captions burned in!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Burn-in error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@captions_bp.route("/captions/burnin/segments", methods=["POST"])
@require_csrf
def burnin_from_segments():
    """Burn caption segments directly into video."""
    data = request.get_json(force=True)
    video_path = data.get("filepath", "").strip()
    segments = data.get("segments", [])
    style = data.get("style", "default")
    output_dir = data.get("output_dir", "")

    if not video_path:
        return jsonify({"error": "Video file not found"}), 400
    try:
        video_path = validate_filepath(video_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    job_id = _new_job("burnin-seg", video_path)

    def _process():
        try:
            from opencut.core.caption_burnin import burnin_segments

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(video_path, output_dir)
            out = burnin_segments(
                video_path, segments, output_dir=effective_dir,
                style=style, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Captions burned in!",
                result={"output_path": out, "style": style},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Burn-in segments error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Animated Captions
# ---------------------------------------------------------------------------
@captions_bp.route("/captions/animated/presets", methods=["GET"])
def animated_caption_presets():
    try:
        from opencut.core.animated_captions import get_animation_presets
        return jsonify({"presets": get_animation_presets()})
    except Exception as e:
        return _safe_error(e)


@captions_bp.route("/captions/animated/render", methods=["POST"])
@require_csrf
def animated_caption_render():
    """Render animated word-by-word captions onto video."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    words = data.get("word_segments", [])

    if not fp:
        return jsonify({"error": "No file path provided"}), 400
    try:
        fp = validate_filepath(fp)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not words:
        return jsonify({"error": "No word segments"}), 400

    job_id = _new_job("anim-cap", fp)

    def _process():
        try:
            from opencut.core.animated_captions import render_animated_captions
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = render_animated_captions(fp, words, output_dir=d,
                                            animation=data.get("animation", "pop"),
                                            font_size=safe_int(data.get("font_size", 56), default=56),
                                            max_words_per_line=safe_int(data.get("max_words", 6), default=6),
                                            on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Transcript Summarization (LLM)
# ---------------------------------------------------------------------------
@captions_bp.route("/transcript/summarize", methods=["POST"])
@require_csrf
def transcript_summarize():
    """Summarize a video transcript using LLM."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    style = data.get("style", "bullets").strip()
    if style not in ("bullets", "paragraph", "detailed"):
        style = "bullets"
    transcript = data.get("transcript", None)

    # LLM config from request
    llm_provider = data.get("llm_provider", "ollama")
    llm_model = data.get("llm_model", "")
    llm_api_key = data.get("llm_api_key", "")
    llm_base_url = data.get("llm_base_url", "")

    job_id = _new_job("summarize", filepath)

    def _process():
        try:
            from opencut.core.highlights import summarize_video
            from opencut.core.llm import LLMConfig

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            llm_config = LLMConfig(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )

            # If no transcript provided, transcribe first
            transcript_segments = transcript
            if not transcript_segments:
                _update_job(job_id, progress=5, message="Transcribing video first...")
                try:
                    from opencut.core.captions import transcribe as _transcribe
                    t_result = _transcribe(filepath)
                    transcript_segments = [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in t_result.segments
                    ]
                except Exception as te:
                    raise RuntimeError(f"Transcription failed: {te}")

            summary = summarize_video(
                transcript_segments,
                style=style,
                llm_config=llm_config,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message="Summary generated!",
                result={
                    "text": summary.text,
                    "bullet_points": summary.bullet_points,
                    "topics": summary.topics,
                    "word_count": summary.word_count,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Summarization error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})
