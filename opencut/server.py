"""
OpenCut Backend Server

Local HTTP server that the Premiere Pro CEP panel communicates with.
Runs on localhost:5679 and handles all processing requests.
"""

import json
import logging
import logging.handlers
import os
import sys
import socket
import time
import threading
import traceback
import uuid
import tempfile
import subprocess as _sp
from pathlib import Path

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

logger = logging.getLogger("opencut")
logger.setLevel(logging.DEBUG)

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("  %(message)s"))
logger.addHandler(_console_handler)

# Handle both relative and absolute imports
try:
    from .utils.media import probe as _probe_media
    from .utils.config import SilenceConfig, ExportConfig, CaptionConfig, get_preset
    from .core.silence import detect_speech, get_edit_summary
    from .core.zoom import generate_zoom_events
    from .export.premiere import export_premiere_xml
    from .export.srt import export_srt, export_vtt, export_json, export_ass, rgb_to_ass_color
except ImportError:
    from opencut.utils.media import probe as _probe_media
    from opencut.utils.config import SilenceConfig, ExportConfig, CaptionConfig, get_preset
    from opencut.core.silence import detect_speech, get_edit_summary
    from opencut.core.zoom import generate_zoom_events
    from opencut.export.premiere import export_premiere_xml
    from opencut.export.srt import export_srt, export_vtt, export_json, export_ass, rgb_to_ass_color


app = Flask(__name__)
CORS(app, origins=["*"])  # CEP panels use null origin

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------
jobs = {}
job_lock = threading.Lock()
JOB_MAX_AGE = 3600  # Auto-clean jobs older than 1 hour


def _new_job(job_type: str, filepath: str) -> str:
    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "filepath": filepath,
            "status": "running",
            "progress": 0,
            "message": "Starting...",
            "result": None,
            "error": None,
            "created": time.time(),
            "_thread": None,
        }
    _cleanup_old_jobs()
    return job_id


def _update_job(job_id: str, **kwargs):
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def _is_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    with job_lock:
        return jobs.get(job_id, {}).get("status") == "cancelled"


def _cleanup_old_jobs():
    """Remove completed/errored jobs older than JOB_MAX_AGE."""
    now = time.time()
    with job_lock:
        expired = [
            jid for jid, j in jobs.items()
            if j["status"] in ("complete", "error", "cancelled")
            and (now - j["created"]) > JOB_MAX_AGE
        ]
        for jid in expired:
            del jobs[jid]


def _resolve_output_dir(filepath: str, requested_dir: str = "") -> str:
    """
    Determine the best output directory for generated files.

    Priority:
    1. Explicitly requested directory (from panel/API)
    2. Same directory as the source file
    3. User's temp directory (last resort)
    """
    # Try requested directory first
    if requested_dir:
        requested_dir = requested_dir.strip().strip('"').strip("'")
        if os.path.isdir(requested_dir):
            return requested_dir
        try:
            os.makedirs(requested_dir, exist_ok=True)
            return requested_dir
        except OSError:
            pass

    # Try source file's directory
    source_dir = os.path.dirname(os.path.abspath(filepath))
    if source_dir and os.path.isdir(source_dir):
        # Test write access
        try:
            test_file = os.path.join(source_dir, f".opencut_test_{uuid.uuid4().hex[:6]}")
            with open(test_file, "w") as f:
                f.write("")
            os.unlink(test_file)
            return source_dir
        except (OSError, PermissionError):
            pass

    # Fallback to temp directory
    fallback = os.path.join(tempfile.gettempdir(), "opencut_output")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _make_sequence_name(filepath: str, suffix: str = "") -> str:
    """Generate a sequence name from the source filename."""
    base = os.path.splitext(os.path.basename(filepath))[0]
    if len(base) > 60:
        base = base[:57] + "..."
    name = f"OpenCut - {base}"
    if suffix:
        name += f" ({suffix})"
    return name


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    caps = {"silence": True, "zoom": True, "ffmpeg": True,
            "audio_suite": True, "scene_detect": True}
    try:
        try:
            from .core.captions import check_whisper_available
        except ImportError:
            from opencut.core.captions import check_whisper_available
        available, backend = check_whisper_available()
        caps["captions"] = available
        caps["whisper_backend"] = backend
    except Exception:
        caps["captions"] = False
        caps["whisper_backend"] = "none"
    return jsonify({"status": "ok", "version": "0.9.0", "capabilities": caps})


@app.route("/shutdown", methods=["POST"])
def shutdown_server():
    """Aggressively shut down the server. Used by new instances to kill old ones."""
    logger.info("Shutdown requested via /shutdown endpoint")

    # Cancel all running jobs first
    with job_lock:
        for jid, job in jobs.items():
            if job.get("status") == "running":
                job["status"] = "cancelled"

    # Remove PID file
    try:
        pid_file = os.path.join(os.path.expanduser("~"), ".opencut", "server.pid")
        if os.path.exists(pid_file):
            os.unlink(pid_file)
    except Exception:
        pass

    # Schedule hard exit - os._exit bypasses cleanup but guarantees death
    def _die():
        time.sleep(0.2)
        logger.info("Server shutting down NOW")
        os._exit(0)

    threading.Thread(target=_die, daemon=True).start()
    return jsonify({"status": "shutting_down", "pid": os.getpid()})


@app.route("/info", methods=["POST"])
def media_info():
    """Get media file metadata."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    try:
        info = _probe_media(filepath)
        result = {
            "filename": info.filename,
            "duration": info.duration,
            "format": info.format_name,
        }
        if info.has_video:
            result["video"] = {
                "width": info.video.width,
                "height": info.video.height,
                "fps": info.video.fps,
                "codec": info.video.codec,
            }
        if info.has_audio:
            result["audio"] = {
                "sample_rate": info.audio.sample_rate,
                "channels": info.audio.channels,
                "codec": info.audio.codec,
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Silence Removal
# ---------------------------------------------------------------------------
@app.route("/silence", methods=["POST"])
def silence_remove():
    """Remove silences and export Premiere XML."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    threshold = float(data.get("threshold", -30.0))
    min_duration = float(data.get("min_duration", 0.5))
    padding_before = float(data.get("padding_before", 0.1))
    padding_after = float(data.get("padding_after", 0.1))
    min_speech = float(data.get("min_speech", 0.25))
    preset = data.get("preset", None)
    seq_name = data.get("sequence_name", "")

    job_id = _new_job("silence", filepath)

    def _process():
        try:
            _update_job(job_id, progress=5, message="Analyzing media file...")

            if preset:
                cfg = get_preset(preset)
                scfg = cfg.silence
            else:
                scfg = SilenceConfig(
                    threshold_db=threshold,
                    min_duration=min_duration,
                    padding_before=padding_before,
                    padding_after=padding_after,
                    min_speech_duration=min_speech,
                )

            effective_name = seq_name or _make_sequence_name(filepath)
            ecfg = ExportConfig(sequence_name=effective_name)

            _update_job(job_id, progress=15, message="Detecting silences (this may take a moment for long files)...")
            segments = detect_speech(filepath, config=scfg)

            if _is_cancelled(job_id):
                return

            _update_job(job_id, progress=70, message="Building edit summary...")
            summary = get_edit_summary(filepath, segments)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")

            _update_job(job_id, progress=85, message="Exporting Premiere XML...")
            export_premiere_xml(filepath, segments, xml_path, config=ecfg)

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result={
                    "xml_path": xml_path,
                    "summary": summary,
                    "segments": len(segments),
                    "segments_data": [
                        {"start": round(s.start, 4), "end": round(s.end, 4)}
                        for s in segments
                    ],
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Filler Word Removal
# ---------------------------------------------------------------------------
@app.route("/fillers", methods=["POST"])
def filler_removal():
    """Detect and remove filler words (um, uh, like, etc.)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    model = data.get("model", "base")
    language = data.get("language", None)
    include_context = data.get("include_context_fillers", True)
    custom_words = data.get("custom_words", [])
    # Which filler types to actually remove (empty = all detected)
    remove_keys = data.get("remove_fillers", [])
    # Also remove silences?
    remove_silence = data.get("remove_silence", True)
    silence_preset = data.get("silence_preset", "youtube")
    seq_name = data.get("sequence_name", "")

    job_id = _new_job("fillers", filepath)

    def _process():
        try:
            # Check Whisper
            try:
                from .core.captions import transcribe, check_whisper_available
                from .core.fillers import detect_fillers, remove_fillers_from_segments, FILLER_WORDS
            except ImportError:
                from opencut.core.captions import transcribe, check_whisper_available
                from opencut.core.fillers import detect_fillers, remove_fillers_from_segments, FILLER_WORDS

            available, backend = check_whisper_available()
            if not available:
                _update_job(
                    job_id, status="error",
                    error="Whisper is required for filler word detection. Install it from the Captions tab.",
                    message="Whisper not installed",
                )
                return

            # Step 1: Silence detection (optional)
            segments = None
            summary = None
            if remove_silence:
                _update_job(job_id, progress=5, message="Step 1/4: Detecting silences...")
                cfg = get_preset(silence_preset)
                segments = detect_speech(filepath, config=cfg.silence)
                summary = get_edit_summary(filepath, segments)
            else:
                # Use the whole file as one segment
                info = _probe_media(filepath)
                try:
                    from .core.silence import TimeSegment
                except ImportError:
                    from opencut.core.silence import TimeSegment
                segments = [TimeSegment(start=0.0, end=info.duration, label="speech")]

            if _is_cancelled(job_id):
                return

            # Step 2: Transcribe with word timestamps
            _update_job(job_id, progress=20, message=f"Step 2/4: Transcribing with {backend} ({model})...")
            config = CaptionConfig(
                model=model,
                language=language,
                word_timestamps=True,
            )
            transcription = transcribe(filepath, config=config)

            if _is_cancelled(job_id):
                return

            # Step 3: Detect fillers
            _update_job(job_id, progress=65, message="Step 3/4: Detecting filler words...")
            analysis = detect_fillers(
                transcription,
                include_context_fillers=include_context,
                custom_words=custom_words if custom_words else None,
            )

            logger.info(
                f"Filler analysis: {len(analysis.hits)} fillers found "
                f"({analysis.filler_percentage}% of words), "
                f"{analysis.total_filler_time:.1f}s total filler time"
            )

            if _is_cancelled(job_id):
                return

            # Step 4: Remove selected fillers from segments
            _update_job(job_id, progress=75, message="Step 4/4: Removing fillers from segments...")

            # Filter hits to only the ones the user wants removed
            hits_to_remove = analysis.hits
            if remove_keys:
                key_set = set(remove_keys)
                hits_to_remove = [h for h in analysis.hits if h.filler_key in key_set]

            refined_segments = remove_fillers_from_segments(segments, hits_to_remove)

            # Build summary for refined segments
            refined_summary = get_edit_summary(filepath, refined_segments)

            # Export XML
            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            xml_path = os.path.join(effective_dir, f"{base_name}_cleancut.xml")

            effective_name = seq_name or _make_sequence_name(filepath, "Clean Cut")
            ecfg = ExportConfig(sequence_name=effective_name)

            _update_job(job_id, progress=90, message="Exporting Premiere XML...")
            export_premiere_xml(filepath, refined_segments, xml_path, config=ecfg)

            # Build per-filler breakdown for UI
            filler_breakdown = []
            for key, count in sorted(analysis.filler_counts.items(), key=lambda x: -x[1]):
                time_for_key = sum(
                    h.duration for h in analysis.hits if h.filler_key == key
                )
                filler_breakdown.append({
                    "word": key,
                    "count": count,
                    "time": round(time_for_key, 2),
                    "removed": (not remove_keys) or (key in set(remove_keys)),
                })

            _update_job(
                job_id,
                status="complete",
                progress=100,
                message="Done!",
                result={
                    "xml_path": xml_path,
                    "summary": refined_summary,
                    "segments": len(refined_segments),
                    "segments_data": [
                        {"start": round(s.start, 4), "end": round(s.end, 4)}
                        for s in refined_segments
                    ],
                    "filler_stats": {
                        "total_fillers": len(analysis.hits),
                        "removed_fillers": len(hits_to_remove),
                        "total_filler_time": analysis.total_filler_time,
                        "filler_percentage": analysis.filler_percentage,
                        "total_words": analysis.total_words,
                        "breakdown": filler_breakdown,
                    },
                    "language": transcription.language,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Filler removal error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------
@app.route("/captions", methods=["POST"])
def generate_captions():
    """Generate captions/subtitles."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    model = data.get("model", "base")
    language = data.get("language", None)
    sub_format = data.get("format", "srt")
    word_timestamps = data.get("word_timestamps", True)

    job_id = _new_job("captions", filepath)

    def _process():
        try:
            try:
                from .core.captions import transcribe, check_whisper_available
            except ImportError:
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
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Styled Captions (video overlay)
# ---------------------------------------------------------------------------
@app.route("/caption-styles", methods=["GET"])
def get_caption_styles():
    """Return available caption style presets for the panel preview."""
    try:
        from .core.styled_captions import get_style_info
    except ImportError:
        from opencut.core.styled_captions import get_style_info
    return jsonify({"styles": get_style_info()})


@app.route("/styled-captions", methods=["POST"])
def styled_captions_route():
    """Generate a transparent video overlay with styled, animated captions."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    style_name = data.get("style", "youtube_bold")
    model = data.get("model", "base")
    language = data.get("language", None)
    custom_action_words = data.get("action_words", [])
    auto_detect_energy = data.get("auto_detect_energy", True)
    # Optional: pre-existing speech segments for remapping
    remap_segments_raw = data.get("remap_segments", None)

    job_id = _new_job("styled-captions", filepath)

    def _process():
        try:
            try:
                from .core.captions import transcribe, check_whisper_available
                from .core.styled_captions import (
                    render_styled_caption_video, STYLES, detect_action_words_by_energy,
                    get_action_word_indices,
                )
            except ImportError:
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
                try:
                    from .core.captions import remap_captions_to_segments
                    from .core.silence import TimeSegment
                except ImportError:
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
@app.route("/transcript", methods=["POST"])
def get_transcript():
    """Transcribe and return full word-level transcript for editing."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    model = data.get("model", "base")
    language = data.get("language", None)

    job_id = _new_job("transcript", filepath)

    def _process():
        try:
            try:
                from .core.captions import transcribe, check_whisper_available
            except ImportError:
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


@app.route("/transcript/export", methods=["POST"])
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
    if not segments_data:
        return jsonify({"error": "No transcript segments provided"}), 400

    try:
        try:
            from .core.captions import CaptionSegment, Word, TranscriptionResult
        except ImportError:
            from opencut.core.captions import CaptionSegment, Word, TranscriptionResult

        # Reconstruct TranscriptionResult from edited segments
        caption_segments = []
        for seg in segments_data:
            words = [
                Word(
                    text=w.get("text", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("confidence", 1.0),
                )
                for w in seg.get("words", [])
            ]
            caption_segments.append(CaptionSegment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
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
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Emoji Map for Captions
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "laugh": "😂", "lol": "😂", "haha": "😂", "funny": "😂",
    "love": "❤️", "heart": "❤️", "beautiful": "❤️",
    "fire": "🔥", "hot": "🔥", "lit": "🔥", "amazing": "🔥",
    "cry": "😢", "sad": "😢", "tears": "😢",
    "wow": "😮", "shocked": "😮", "omg": "😮", "unbelievable": "😮",
    "think": "🤔", "hmm": "🤔", "wondering": "🤔",
    "thumbs": "👍", "great": "👍", "good": "👍", "nice": "👍",
    "clap": "👏", "bravo": "👏", "congrats": "👏",
    "money": "💰", "rich": "💰", "dollar": "💰", "expensive": "💰",
    "star": "⭐", "perfect": "⭐", "excellent": "⭐",
    "rocket": "🚀", "launch": "🚀", "skyrocket": "🚀", "moon": "🚀",
    "brain": "🧠", "smart": "🧠", "genius": "🧠", "intelligent": "🧠",
    "muscle": "💪", "strong": "💪", "power": "💪", "strength": "💪",
    "warning": "⚠️", "danger": "⚠️", "careful": "⚠️",
    "check": "✅", "correct": "✅", "yes": "✅", "done": "✅",
    "wrong": "❌", "no": "❌", "false": "❌", "bad": "❌",
    "question": "❓", "why": "❓", "how": "❓",
    "eyes": "👀", "look": "👀", "see": "👀", "watch": "👀",
    "celebrate": "🎉", "party": "🎉", "winner": "🎉", "victory": "🎉",
    "sleep": "😴", "tired": "😴", "boring": "😴",
    "music": "🎵", "song": "🎵", "singing": "🎵",
    "camera": "📸", "photo": "📸", "picture": "📸",
    "time": "⏰", "clock": "⏰", "hurry": "⏰", "fast": "⏰",
    "earth": "🌍", "world": "🌍", "global": "🌍",
    "sun": "☀️", "sunny": "☀️", "bright": "☀️",
    "rain": "🌧️", "storm": "🌧️", "weather": "🌧️",
    "food": "🍕", "eat": "🍕", "hungry": "🍕", "delicious": "🍕",
    "coffee": "☕", "morning": "☕", "energy": "☕",
    "book": "📚", "read": "📚", "study": "📚", "learn": "📚",
    "computer": "💻", "code": "💻", "tech": "💻", "programming": "💻",
    "phone": "📱", "call": "📱", "mobile": "📱",
    "key": "🔑", "secret": "🔑", "unlock": "🔑",
    "light": "💡", "idea": "💡", "tip": "💡", "insight": "💡",
    "target": "🎯", "goal": "🎯", "focus": "🎯", "aim": "🎯",
    "gem": "💎", "diamond": "💎", "valuable": "💎", "premium": "💎",
    "crown": "👑", "king": "👑", "queen": "👑", "best": "👑",
    "ghost": "👻", "scary": "👻", "spooky": "👻", "halloween": "👻",
    "skull": "💀", "dead": "💀", "kill": "💀", "destroy": "💀",
    "hundred": "💯", "percent": "💯",
}


@app.route("/captions/emoji-map", methods=["GET"])
def get_emoji_map():
    """Return the keyword->emoji mapping for auto-emoji insertion."""
    return jsonify({"emoji_map": EMOJI_MAP})


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
@app.route("/full", methods=["POST"])
def full_pipeline():
    """Run silence removal + zoom + optional captions."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

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
            segments = detect_speech(filepath, config=cfg.silence)
            summary = get_edit_summary(filepath, segments)

            if _is_cancelled(job_id):
                return

            # Step: Filler word removal (requires Whisper)
            filler_stats = None
            transcription_result = None
            if remove_fillers:
                try:
                    from .core.captions import check_whisper_available, transcribe
                    from .core.fillers import detect_fillers, remove_fillers_from_segments
                except ImportError:
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
                    transcription_result = transcribe(filepath, config=filler_cfg)
                    analysis = detect_fillers(transcription_result, include_context_fillers=True)

                    if analysis.hits:
                        segments = remove_fillers_from_segments(segments, analysis.hits)
                        # Recalculate summary with filler-cleaned segments
                        summary = get_edit_summary(filepath, segments)
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
                try:
                    from .core.captions import check_whisper_available
                except ImportError:
                    from opencut.core.captions import check_whisper_available
                available, backend = check_whisper_available()
                if available:
                    next_step("Generating captions...")
                    # Reuse transcription from filler step if available
                    if transcription_result:
                        captions_result = transcription_result
                    else:
                        try:
                            from .core.captions import transcribe
                        except ImportError:
                            from opencut.core.captions import transcribe
                        captions_result = transcribe(filepath, config=cfg.captions)

                    # Remap caption timestamps to the condensed timeline
                    try:
                        from .core.captions import remap_captions_to_segments
                    except ImportError:
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
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Noise Reduction
# ---------------------------------------------------------------------------
@app.route("/audio/denoise", methods=["POST"])
def audio_denoise():
    """Remove background noise from audio/video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    method = data.get("method", "afftdn")
    strength = float(data.get("strength", 0.7))
    noise_floor = float(data.get("noise_floor", -30.0))

    job_id = _new_job("denoise", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import denoise_audio
            except ImportError:
                from opencut.core.audio_suite import denoise_audio

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_denoised{ext}")

            denoise_audio(
                filepath, out_path,
                method=method, strength=strength,
                noise_floor=noise_floor,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message="Noise reduction complete!",
                result={
                    "output_path": out_path,
                    "method": method,
                    "strength": strength,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Denoise error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Voice Isolation
# ---------------------------------------------------------------------------
@app.route("/audio/isolate", methods=["POST"])
def audio_isolate():
    """Isolate vocals using bandpass filtering or Demucs AI."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    method = data.get("method", "bandpass")  # "bandpass" or "demucs"

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("isolate", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import isolate_voice, isolate_stems
            except ImportError:
                from opencut.core.audio_suite import isolate_voice, isolate_stems

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]

            if method == "demucs":
                # Use Demucs for AI-based stem separation
                stems_dir = os.path.join(effective_dir, "stems")
                result = isolate_stems(
                    filepath, stems_dir,
                    model="htdemucs",
                    stems=["vocals"],  # Only extract vocals
                    on_progress=_on_progress,
                )
                out_path = result.vocals_path
                _update_job(
                    job_id, status="complete", progress=100,
                    message="Demucs voice isolation complete!",
                    result={
                        "output_path": out_path,
                        "method": "demucs",
                        "model": result.model_name,
                    },
                )
            else:
                # Use FFmpeg bandpass filtering
                out_path = os.path.join(effective_dir, f"{base_name}_voice{ext}")
                isolate_voice(filepath, out_path, method="bandpass", on_progress=_on_progress)
                _update_job(
                    job_id, status="complete", progress=100,
                    message="Voice isolation complete!",
                    result={"output_path": out_path, "method": "bandpass"},
                )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Voice isolation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Loudness Normalization
# ---------------------------------------------------------------------------
@app.route("/audio/normalize", methods=["POST"])
def audio_normalize():
    """Normalize audio loudness to broadcast/platform standards."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    preset = data.get("preset", "youtube")
    target_lufs = data.get("target_lufs", None)
    if target_lufs is not None:
        target_lufs = float(target_lufs)

    job_id = _new_job("normalize", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import normalize_loudness, measure_loudness, LOUDNESS_PRESETS
            except ImportError:
                from opencut.core.audio_suite import normalize_loudness, measure_loudness, LOUDNESS_PRESETS

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_normalized{ext}")

            targets = LOUDNESS_PRESETS.get(preset, LOUDNESS_PRESETS["youtube"])

            out_path, info = normalize_loudness(
                filepath, out_path,
                preset=preset,
                target_lufs=target_lufs,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Normalized to {targets['i']:.1f} LUFS",
                result={
                    "output_path": out_path,
                    "preset": preset,
                    "input_loudness": info.input_i,
                    "target_loudness": targets["i"],
                    "input_peak": info.input_tp,
                    "loudness_range": info.input_lra,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Normalize error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/measure", methods=["POST"])
def audio_measure():
    """Measure audio loudness (EBU R128) without processing."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    try:
        try:
            from .core.audio_suite import measure_loudness
        except ImportError:
            from opencut.core.audio_suite import measure_loudness

        info = measure_loudness(filepath)
        return jsonify({
            "integrated_lufs": info.input_i,
            "true_peak_dbtp": info.input_tp,
            "loudness_range_lu": info.input_lra,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Audio Suite: Beat Detection
# ---------------------------------------------------------------------------
@app.route("/audio/beats", methods=["POST"])
def audio_beats():
    """Detect beats and estimate BPM."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    sensitivity = float(data.get("sensitivity", 0.5))

    job_id = _new_job("beats", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import detect_beats
            except ImportError:
                from opencut.core.audio_suite import detect_beats

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            info = detect_beats(filepath, sensitivity=sensitivity, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Found {len(info.beat_times)} beats at {info.bpm:.0f} BPM",
                result={
                    "bpm": info.bpm,
                    "total_beats": len(info.beat_times),
                    "beat_times": [round(t, 3) for t in info.beat_times[:500]],
                    "downbeat_times": [round(t, 3) for t in info.downbeat_times[:125]],
                    "confidence": round(info.confidence, 3),
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Beat detection error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Audio Effects
# ---------------------------------------------------------------------------
@app.route("/audio/effects", methods=["GET"])
def audio_effects_list():
    """Return available audio effects."""
    try:
        try:
            from .core.audio_suite import get_available_effects
        except ImportError:
            from opencut.core.audio_suite import get_available_effects
        return jsonify({"effects": get_available_effects()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/effects/apply", methods=["POST"])
def audio_effects_apply():
    """Apply an audio effect."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    if not effect:
        return jsonify({"error": "No effect specified"}), 400

    job_id = _new_job("audio-effect", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import apply_audio_effect
            except ImportError:
                from opencut.core.audio_suite import apply_audio_effect

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_{effect}{ext}")

            apply_audio_effect(filepath, effect, out_path, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Effect '{effect}' applied!",
                result={"output_path": out_path, "effect": effect},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio effect error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Loudness Presets
# ---------------------------------------------------------------------------
@app.route("/audio/loudness-presets", methods=["GET"])
def loudness_presets():
    """Return available loudness normalization presets."""
    try:
        try:
            from .core.audio_suite import LOUDNESS_PRESETS
        except ImportError:
            from opencut.core.audio_suite import LOUDNESS_PRESETS
        presets = []
        for name, vals in LOUDNESS_PRESETS.items():
            presets.append({
                "name": name,
                "label": name.replace("_", " ").title(),
                "target_lufs": vals["i"],
                "target_tp": vals["tp"],
                "target_lra": vals["lra"],
            })
        return jsonify({"presets": presets})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Audio Suite: EQ Presets
# ---------------------------------------------------------------------------
@app.route("/audio/eq-presets", methods=["GET"])
def eq_presets():
    """Return available EQ presets."""
    try:
        try:
            from .core.audio_suite import get_available_eq_presets
        except ImportError:
            from opencut.core.audio_suite import get_available_eq_presets
        return jsonify({"presets": get_available_eq_presets()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/eq", methods=["POST"])
def audio_eq():
    """Apply an EQ preset to audio/video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    preset = data.get("preset", "voice_enhance")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("eq", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import apply_eq_preset
            except ImportError:
                from opencut.core.audio_suite import apply_eq_preset

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_{preset}{ext}")

            apply_eq_preset(filepath, preset, out_path, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message=f"EQ preset '{preset}' applied!",
                result={"output_path": out_path, "preset": preset},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("EQ preset error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Audio Ducking
# ---------------------------------------------------------------------------
@app.route("/audio/duck", methods=["POST"])
def audio_duck():
    """Mix speech and music with automatic ducking."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()  # Speech track
    music_filepath = data.get("music_filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No speech file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"Speech file not found: {filepath}"}), 400
    if not music_filepath:
        return jsonify({"error": "No music file path provided"}), 400
    if not os.path.isfile(music_filepath):
        return jsonify({"error": f"Music file not found: {music_filepath}"}), 400

    duck_level = float(data.get("duck_level", -12.0))
    attack_ms = float(data.get("attack_ms", 200.0))
    release_ms = float(data.get("release_ms", 500.0))

    job_id = _new_job("duck", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import duck_audio
            except ImportError:
                from opencut.core.audio_suite import duck_audio

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            out_path = os.path.join(effective_dir, f"{base_name}_ducked.mp3")

            out_path, keyframes = duck_audio(
                filepath, music_filepath, out_path,
                duck_level=duck_level,
                attack_ms=attack_ms,
                release_ms=release_ms,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Audio ducking complete: {len(keyframes)} keyframes",
                result={
                    "output_path": out_path,
                    "duck_level": duck_level,
                    "keyframe_count": len(keyframes),
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio ducking error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/ducking-keyframes", methods=["POST"])
def audio_ducking_keyframes():
    """Generate ducking keyframes (without mixing)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    duck_level = float(data.get("duck_level", -12.0))
    attack_ms = float(data.get("attack_ms", 200.0))
    release_ms = float(data.get("release_ms", 500.0))

    job_id = _new_job("ducking-keyframes", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import generate_ducking_keyframes
            except ImportError:
                from opencut.core.audio_suite import generate_ducking_keyframes

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            keyframes = generate_ducking_keyframes(
                filepath,
                music_volume=0.0,
                duck_volume=duck_level,
                attack_ms=attack_ms,
                release_ms=release_ms,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Generated {len(keyframes)} ducking keyframes",
                result={
                    "keyframes": keyframes,
                    "keyframe_count": len(keyframes),
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Ducking keyframes error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Crossfade
# ---------------------------------------------------------------------------
@app.route("/audio/crossfade-types", methods=["GET"])
def crossfade_types():
    """Return available crossfade curve types."""
    try:
        try:
            from .core.audio_suite import get_crossfade_types
        except ImportError:
            from opencut.core.audio_suite import get_crossfade_types
        return jsonify({"types": get_crossfade_types()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/crossfade", methods=["POST"])
def audio_crossfade():
    """Create a crossfade between two audio files."""
    data = request.get_json(force=True)
    filepath1 = data.get("filepath1", "").strip()
    filepath2 = data.get("filepath2", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath1:
        return jsonify({"error": "No first file path provided"}), 400
    if not os.path.isfile(filepath1):
        return jsonify({"error": f"First file not found: {filepath1}"}), 400
    if not filepath2:
        return jsonify({"error": "No second file path provided"}), 400
    if not os.path.isfile(filepath2):
        return jsonify({"error": f"Second file not found: {filepath2}"}), 400

    duration_ms = float(data.get("duration_ms", 1000.0))
    fade_type = data.get("fade_type", "linear")

    job_id = _new_job("crossfade", filepath1)

    def _process():
        try:
            try:
                from .core.audio_suite import crossfade_audio
            except ImportError:
                from opencut.core.audio_suite import crossfade_audio

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath1, output_dir)
            base1 = os.path.splitext(os.path.basename(filepath1))[0]
            base2 = os.path.splitext(os.path.basename(filepath2))[0]
            ext = os.path.splitext(filepath1)[1]
            out_path = os.path.join(effective_dir, f"{base1}_x_{base2}{ext}")

            result = crossfade_audio(
                filepath1, filepath2, out_path,
                duration_ms=duration_ms,
                fade_type=fade_type,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Crossfade complete ({fade_type}, {duration_ms:.0f}ms)",
                result={
                    "output_path": result.output_path,
                    "duration_ms": result.duration_ms,
                    "fade_type": result.fade_type,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Crossfade error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Suite: Stem Separation (Demucs)
# ---------------------------------------------------------------------------
@app.route("/audio/stems", methods=["POST"])
def audio_stems():
    """Separate audio into stems using Demucs v4."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    stems = data.get("stems")  # None = all stems
    model = data.get("model", "htdemucs")

    job_id = _new_job("stems", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import isolate_stems
            except ImportError:
                from opencut.core.audio_suite import isolate_stems

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            stems_dir = os.path.join(effective_dir, "stems")

            result = isolate_stems(
                filepath, stems_dir,
                model=model,
                stems=stems,
                on_progress=_on_progress,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Stem separation complete ({model})",
                result={
                    "vocals_path": result.vocals_path,
                    "drums_path": result.drums_path,
                    "bass_path": result.bass_path,
                    "other_path": result.other_path,
                    "model": result.model_name,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Stem separation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Scene Detection
# ---------------------------------------------------------------------------
@app.route("/video/scenes", methods=["POST"])
def video_scenes():
    """Detect scene changes in a video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    threshold = float(data.get("threshold", 0.3))
    min_scene = float(data.get("min_scene_length", 2.0))

    job_id = _new_job("scenes", filepath)

    def _process():
        try:
            try:
                from .core.scene_detect import detect_scenes, generate_chapter_markers
            except ImportError:
                from opencut.core.scene_detect import detect_scenes, generate_chapter_markers

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            info = detect_scenes(
                filepath, threshold=threshold,
                min_scene_length=min_scene,
                on_progress=_on_progress,
            )

            # Generate YouTube chapters
            chapters_yt = generate_chapter_markers(info, format="youtube")

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Found {info.total_scenes} scenes",
                result={
                    "total_scenes": info.total_scenes,
                    "duration": info.duration,
                    "avg_scene_length": round(info.avg_scene_length, 2),
                    "boundaries": [
                        {
                            "time": round(b.time, 3),
                            "frame": b.frame,
                            "score": round(b.score, 3),
                            "label": b.label,
                        }
                        for b in info.boundaries
                    ],
                    "youtube_chapters": chapters_yt,
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Scene detection error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Speed Ramp Presets
# ---------------------------------------------------------------------------
@app.route("/video/speed-presets", methods=["GET"])
def speed_presets():
    """Return available speed ramp presets."""
    try:
        try:
            from .core.speed_ramp import get_speed_presets
        except ImportError:
            from opencut.core.speed_ramp import get_speed_presets
        return jsonify({"presets": get_speed_presets()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/speed-ramp", methods=["POST"])
def video_speed_ramp():
    """Apply speed ramp effect to a video clip."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    preset = data.get("preset", "smooth_slow")
    mode = data.get("mode", "render")  # render or xml
    quality = data.get("quality", "medium")
    smoothness = data.get("smoothness", 0.5)
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting speed ramp..."}

    def _run():
        try:
            try:
                from .core.speed_ramp import apply_speed_ramp, generate_speed_xml
            except ImportError:
                from opencut.core.speed_ramp import apply_speed_ramp, generate_speed_xml

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            if mode == "xml":
                result = generate_speed_xml(
                    input_path=input_path, preset=preset,
                    output_dir=output_dir, on_progress=cb,
                )
            else:
                result = apply_speed_ramp(
                    input_path=input_path, preset=preset,
                    quality=quality, output_dir=output_dir, on_progress=cb,
                )

            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "mode": mode,
                    "preset": preset,
                    "original_duration": result.original_duration,
                    "output_duration": result.output_duration,
                }
        except Exception as e:
            logger.exception("Speed ramp failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# LUT / Color Grading
# ---------------------------------------------------------------------------
COLOR_PRESETS = {
    "warm":           "colortemperature=7500,eq=saturation=1.1",
    "golden_hour":    "colortemperature=8500,eq=saturation=1.2:brightness=0.03",
    "sepia":          "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:0",
    "cool":           "colortemperature=4500,eq=saturation=1.05",
    "moonlight":      "colortemperature=3800,eq=saturation=0.85:brightness=-0.05,colorbalance=bs=0.15:bh=0.08",
    "cinematic":      "colorbalance=rs=-0.08:gs=0.02:bs=0.12:rh=0.06:gh=-0.02:bh=-0.08,eq=saturation=1.1:contrast=1.05",
    "vintage":        "curves=vintage,eq=saturation=0.85:contrast=1.08",
    "matte":          "eq=saturation=0.75:contrast=0.95:brightness=0.04,curves=m='0/0.04 0.5/0.54 1/0.96'",
    "bleach_bypass":  "eq=saturation=0.5:contrast=1.3:brightness=-0.02",
    "cross_process":  "curves=r='0/0.1 0.5/0.6 1/0.9':g='0/0 0.5/0.45 1/1':b='0/0.2 0.5/0.5 1/0.8',eq=saturation=1.3",
    "bw_classic":     "hue=s=0,eq=contrast=1.1",
    "bw_high_contrast": "hue=s=0,eq=contrast=1.4:brightness=-0.03",
    "vivid":          "eq=saturation=1.5:contrast=1.1",
    "desaturated":    "eq=saturation=0.4",
    "negative":       "negate",
}


@app.route("/video/lut", methods=["POST"])
def video_lut():
    """Apply a LUT file or color preset to a video."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    lut_path = data.get("lut_path", "")
    preset = data.get("preset", "")
    intensity = float(data.get("intensity", 1.0))
    quality = data.get("quality", "medium")
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    if not lut_path and not preset:
        return jsonify({"error": "Provide lut_path or preset"}), 400

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting color grade..."}

    def _run():
        try:
            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            if preset and preset in COLOR_PRESETS:
                # Use FFmpeg filter chain for built-in presets
                cb(10, f"Applying preset: {preset}")
                out_path = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(input_path))[0] + f"_color_{preset}.mp4"
                )
                crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")
                vf = COLOR_PRESETS[preset]

                if intensity < 0.99:
                    vf = f"split[orig][fx];[fx]{vf}[graded];[orig][graded]blend=all_expr='A*{1 - intensity}+B*{intensity}'"

                cmd = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-vf", vf, "-c:v", "libx264", "-crf", crf,
                    "-c:a", "copy", out_path,
                ]
                cb(30, "Encoding...")
                subprocess.run(cmd, capture_output=True, timeout=3600, check=True)
                cb(100, "Done")

                with job_lock:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["progress"] = 100
                    jobs[job_id]["result"] = {
                        "output": out_path,
                        "preset": preset,
                        "intensity": intensity,
                    }
            else:
                # Use LUT file via video_fx module
                try:
                    from .core.video_fx import apply_lut
                except ImportError:
                    from opencut.core.video_fx import apply_lut

                result = apply_lut(
                    input_path=input_path, lut_path=lut_path,
                    intensity=intensity, quality=quality,
                    output_dir=output_dir, on_progress=cb,
                )
                with job_lock:
                    jobs[job_id]["status"] = "done"
                    jobs[job_id]["progress"] = 100
                    jobs[job_id]["result"] = {
                        "output": result.output_path,
                        "lut": lut_path,
                        "intensity": intensity,
                    }
        except Exception as e:
            logger.exception("Color grading failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/lut-validate", methods=["POST"])
def video_lut_validate():
    """Validate a LUT file path."""
    data = request.get_json(force=True)
    lut_path = data.get("lut_path", "")
    if not lut_path or not os.path.isfile(lut_path):
        return jsonify({"valid": False, "error": "File not found"}), 400
    try:
        try:
            from .core.video_fx import get_lut_info
        except ImportError:
            from opencut.core.video_fx import get_lut_info
        info = get_lut_info(lut_path)
        return jsonify({"valid": True, "info": info})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


# ---------------------------------------------------------------------------
# Chroma Key
# ---------------------------------------------------------------------------
@app.route("/video/chroma-key", methods=["POST"])
def video_chroma_key():
    """Apply chroma key (green screen) removal."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    key_color = data.get("key_color", "green")
    similarity = float(data.get("similarity", 0.3))
    blend = float(data.get("blend", 0.1))
    bg_mode = data.get("bg_mode", "blur")
    bg_value = data.get("bg_value", "")
    quality = data.get("quality", "medium")
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting chroma key..."}

    def _run():
        try:
            try:
                from .core.video_fx import apply_chroma_key
            except ImportError:
                from opencut.core.video_fx import apply_chroma_key

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = apply_chroma_key(
                input_path=input_path, key_color=key_color,
                similarity=similarity, blend=blend,
                bg_mode=bg_mode, bg_value=bg_value,
                quality=quality, output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "key_color": key_color,
                    "bg_mode": bg_mode,
                }
        except Exception as e:
            logger.exception("Chroma key failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# AI Background Removal
# ---------------------------------------------------------------------------
@app.route("/video/bg-check", methods=["GET"])
def video_bg_check():
    """Check if rembg is installed."""
    try:
        try:
            from .core.video_fx import check_rembg_available
        except ImportError:
            from opencut.core.video_fx import check_rembg_available
        ok, msg = check_rembg_available()
        return jsonify({"installed": ok, "message": msg})
    except Exception as e:
        return jsonify({"installed": False, "message": str(e)})


@app.route("/video/bg-install", methods=["POST"])
def video_bg_install():
    """Install rembg + onnxruntime."""
    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Installing rembg...", "type": "install-rembg"}

    def _run():
        try:
            try:
                from .core.video_fx import install_rembg
            except ImportError:
                from opencut.core.video_fx import install_rembg

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            ok = install_rembg(progress_callback=cb)
            with job_lock:
                jobs[job_id]["status"] = "done" if ok else "error"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["message"] = "rembg installed" if ok else "Installation failed"
                jobs[job_id]["result"] = {"installed": ok}
        except Exception as e:
            logger.exception("rembg install failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/bg-remove", methods=["POST"])
def video_bg_remove():
    """Remove background from image or video."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    model = data.get("model", "birefnet-general")
    bg_mode = data.get("bg_mode", "transparent")
    bg_value = data.get("bg_value", "")
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Removing background..."}

    def _run():
        try:
            try:
                from .core.video_fx import remove_background
            except ImportError:
                from opencut.core.video_fx import remove_background

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = remove_background(
                input_path=input_path, model=model,
                bg_mode=bg_mode, bg_value=bg_value,
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "model": model,
                    "bg_mode": bg_mode,
                }
        except Exception as e:
            logger.exception("Background removal failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# AI Slow Motion (RIFE)
# ---------------------------------------------------------------------------
@app.route("/video/slowmo-check", methods=["GET"])
def video_slowmo_check():
    """Check if RIFE is installed."""
    try:
        try:
            from .core.video_fx import check_rife_available
        except ImportError:
            from opencut.core.video_fx import check_rife_available
        ok, msg = check_rife_available()
        return jsonify({"installed": ok, "message": msg})
    except Exception as e:
        return jsonify({"installed": False, "message": str(e)})


@app.route("/video/slowmo-install", methods=["POST"])
def video_slowmo_install():
    """Install Practical-RIFE."""
    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Installing RIFE...", "type": "install-rife"}

    def _run():
        try:
            try:
                from .core.video_fx import install_rife
            except ImportError:
                from opencut.core.video_fx import install_rife

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            ok = install_rife(progress_callback=cb)
            with job_lock:
                jobs[job_id]["status"] = "done" if ok else "error"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["message"] = "RIFE installed" if ok else "Installation failed"
                jobs[job_id]["result"] = {"installed": ok}
        except Exception as e:
            logger.exception("RIFE install failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/slowmo", methods=["POST"])
def video_slowmo():
    """Generate AI slow motion video."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    multiplier = int(data.get("multiplier", 2))
    scale = float(data.get("scale", 1.0))
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting slow motion..."}

    def _run():
        try:
            try:
                from .core.video_fx import apply_slow_motion
            except ImportError:
                from opencut.core.video_fx import apply_slow_motion

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = apply_slow_motion(
                input_path=input_path, multiplier=multiplier,
                scale=scale, output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "multiplier": multiplier,
                    "original_duration": result.original_duration,
                    "output_duration": result.output_duration,
                }
        except Exception as e:
            logger.exception("Slow motion failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# AI Upscaling (Real-ESRGAN)
# ---------------------------------------------------------------------------
@app.route("/video/upscale-check", methods=["GET"])
def video_upscale_check():
    """Check if Real-ESRGAN is installed."""
    try:
        try:
            from .core.video_fx import check_realesrgan_available
        except ImportError:
            from opencut.core.video_fx import check_realesrgan_available
        ok, msg = check_realesrgan_available()
        return jsonify({"installed": ok, "message": msg})
    except Exception as e:
        return jsonify({"installed": False, "message": str(e)})


@app.route("/video/upscale-install", methods=["POST"])
def video_upscale_install():
    """Install Real-ESRGAN."""
    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Installing Real-ESRGAN...", "type": "install-esrgan"}

    def _run():
        try:
            try:
                from .core.video_fx import install_realesrgan
            except ImportError:
                from opencut.core.video_fx import install_realesrgan

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            ok = install_realesrgan(progress_callback=cb)
            with job_lock:
                jobs[job_id]["status"] = "done" if ok else "error"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["message"] = "Real-ESRGAN installed" if ok else "Installation failed"
                jobs[job_id]["result"] = {"installed": ok}
        except Exception as e:
            logger.exception("Real-ESRGAN install failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/upscale", methods=["POST"])
def video_upscale():
    """Upscale video using AI."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    model = data.get("model", "x4plus")
    scale = int(data.get("scale", 4))
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting upscale..."}

    def _run():
        try:
            try:
                from .core.video_fx import upscale_video
            except ImportError:
                from opencut.core.video_fx import upscale_video

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = upscale_video(
                input_path=input_path, model=model,
                scale=scale, output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "model": model,
                    "scale": scale,
                    "width": result.width,
                    "height": result.height,
                }
        except Exception as e:
            logger.exception("Upscale failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Auto-Reframe
# ---------------------------------------------------------------------------
@app.route("/video/reframe-presets", methods=["GET"])
def video_reframe_presets():
    """Return available aspect ratio presets."""
    try:
        try:
            from .core.video_fx import ASPECT_PRESETS
        except ImportError:
            from opencut.core.video_fx import ASPECT_PRESETS
        return jsonify({"presets": {k: v["label"] for k, v in ASPECT_PRESETS.items()}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/reframe", methods=["POST"])
def video_reframe():
    """Reframe video to a different aspect ratio."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    target_aspect = data.get("target_aspect") or data.get("aspect", "9:16")
    mode = data.get("mode", "center")
    face_detection = data.get("face_detection", False)
    quality = data.get("quality", "medium")
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    if face_detection:
        mode = "smart"

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting reframe..."}

    def _run():
        try:
            try:
                from .core.video_fx import auto_reframe
            except ImportError:
                from opencut.core.video_fx import auto_reframe

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = auto_reframe(
                input_path=input_path, target_aspect=target_aspect,
                mode=mode, quality=quality,
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "target_aspect": target_aspect,
                    "width": result.width,
                    "height": result.height,
                }
        except Exception as e:
            logger.exception("Reframe failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Video FX Capabilities
# ---------------------------------------------------------------------------
@app.route("/video/fx-capabilities", methods=["GET"])
def video_fx_capabilities():
    """Return all video FX features and installation status."""
    try:
        try:
            from .core.video_fx import get_video_fx_capabilities
        except ImportError:
            from opencut.core.video_fx import get_video_fx_capabilities
        caps = get_video_fx_capabilities()
        caps["color_presets"] = list(COLOR_PRESETS.keys())
        return jsonify(caps)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# GPU / System Info
# ---------------------------------------------------------------------------
@app.route("/system/gpu", methods=["GET"])
def system_gpu():
    """Check GPU availability for AI features."""
    gpu_info = {"available": False, "name": "None", "vram_mb": 0}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu_info["available"] = True
            gpu_info["name"] = parts[0].strip()
            gpu_info["vram_mb"] = int(parts[1].strip()) if len(parts) > 1 else 0
    except Exception:
        pass
    return jsonify(gpu_info)


# ---------------------------------------------------------------------------
# Job Status / Cancel
# ---------------------------------------------------------------------------
@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id):
    """Check the status of a processing job."""
    with job_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    safe = {k: v for k, v in job.items() if not k.startswith("_")}
    return jsonify(safe)


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running job."""
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        if job["status"] != "running":
            return jsonify({"error": "Job is not running"}), 400
        job["status"] = "cancelled"
        job["message"] = "Cancelled by user"
        job["progress"] = 0
    return jsonify({"status": "cancelled", "job_id": job_id})


@app.route("/jobs", methods=["GET"])
def list_jobs():
    """List all jobs."""
    with job_lock:
        safe_list = []
        for j in jobs.values():
            safe_list.append({k: v for k, v in j.items() if not k.startswith("_")})
        return jsonify(safe_list)


# ---------------------------------------------------------------------------
# Server-Sent Events (SSE) job stream
# ---------------------------------------------------------------------------
@app.route("/stream/<job_id>", methods=["GET"])
def stream_job(job_id):
    """Stream job status via Server-Sent Events. Replaces polling."""
    def generate():
        while True:
            with job_lock:
                job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found', 'error': 'Job not found'})}\n\n"
                break
            safe = {k: v for k, v in job.items() if not k.startswith("_")}
            yield f"data: {json.dumps(safe)}\n\n"
            if job["status"] in ("complete", "error", "cancelled"):
                break
            time.sleep(0.5)

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ---------------------------------------------------------------------------
# Install Whisper
# ---------------------------------------------------------------------------
@app.route("/install-whisper", methods=["POST"])
def install_whisper():
    """Install faster-whisper via pip (on-demand from panel)."""
    data = request.get_json(force=True) if request.data else {}
    backend = data.get("backend", "faster-whisper")

    allowed = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed:
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    job_id = _new_job("install-whisper", backend)

    def _process():
        import subprocess as _sp

        _update_job(job_id, progress=5, message=f"Installing {backend}...")
        logger.info(f"Starting Whisper install: {backend}")

        # Strategy list - try each in order until one works
        strategies = []
        if backend == "faster-whisper":
            strategies = [
                {
                    "label": "Install tokenizers wheel first, then faster-whisper",
                    "pre": [sys.executable, "-m", "pip", "install",
                            "tokenizers", "--only-binary", "tokenizers",
                            "--quiet"],
                    "cmd": [sys.executable, "-m", "pip", "install",
                            "faster-whisper", "--progress-bar", "on"],
                },
                {
                    "label": "Upgrade pip + prefer binary wheels",
                    "pre": [sys.executable, "-m", "pip", "install",
                            "--upgrade", "pip", "setuptools", "wheel",
                            "--quiet"],
                    "cmd": [sys.executable, "-m", "pip", "install",
                            "faster-whisper", "--prefer-binary",
                            "--progress-bar", "on"],
                },
                {
                    "label": "Pin older tokenizers with wheel support",
                    "pre": [sys.executable, "-m", "pip", "install",
                            "tokenizers>=0.13,<0.20", "--only-binary",
                            "tokenizers", "--quiet"],
                    "cmd": [sys.executable, "-m", "pip", "install",
                            "faster-whisper", "--progress-bar", "on"],
                },
                {
                    "label": "Fallback to openai-whisper (no Rust needed)",
                    "cmd": [sys.executable, "-m", "pip", "install",
                            "openai-whisper", "--progress-bar", "on"],
                    "verify": [sys.executable, "-c",
                               "import whisper; print('ok')"],
                    "backend_name": "openai-whisper",
                },
            ]
        else:
            strategies = [
                {
                    "label": "Standard install",
                    "cmd": [sys.executable, "-m", "pip", "install",
                            backend, "--progress-bar", "on"],
                },
            ]

        last_error = ""
        for si, strat in enumerate(strategies):
            if _is_cancelled(job_id):
                return

            pct_base = int(5 + (si / len(strategies)) * 70)
            _update_job(job_id, progress=pct_base,
                        message=f"Strategy {si+1}/{len(strategies)}: {strat['label']}...")
            logger.info(f"Whisper install strategy {si+1}: {strat['label']}")

            # Run pre-command if present (e.g. upgrade pip)
            if "pre" in strat:
                try:
                    pre_result = _sp.run(strat["pre"], capture_output=True, text=True, timeout=120)
                    logger.debug(f"Pre-command exit {pre_result.returncode}: {pre_result.stdout[-200:]}")
                except Exception as e:
                    logger.warning(f"Pre-command failed: {e}")

            # Run main install command
            try:
                proc = _sp.Popen(strat["cmd"], stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
                lines = []
                for line in proc.stdout:
                    line = line.strip()
                    if line:
                        lines.append(line)
                        lower = line.lower()
                        if "downloading" in lower:
                            _update_job(job_id, progress=pct_base + 10,
                                        message=f"Downloading packages...")
                        elif "installing" in lower or "building" in lower:
                            _update_job(job_id, progress=pct_base + 20,
                                        message=f"Installing packages...")
                        elif "successfully installed" in lower:
                            _update_job(job_id, progress=85,
                                        message="Verifying installation...")

                proc.wait()
                logger.debug(f"pip exit code: {proc.returncode}")

                if proc.returncode != 0:
                    last_error = "\n".join(lines[-8:])
                    logger.warning(f"Strategy {si+1} failed: {last_error[-300:]}")
                    continue  # Try next strategy

                # Verify import
                _update_job(job_id, progress=90, message="Verifying import...")

                verify_cmd = strat.get("verify", None)
                actual_backend = strat.get("backend_name", backend)

                if verify_cmd is None:
                    if actual_backend == "faster-whisper":
                        verify_cmd = [sys.executable, "-c", "from faster_whisper import WhisperModel; print('ok')"]
                    elif actual_backend == "openai-whisper":
                        verify_cmd = [sys.executable, "-c", "import whisper; print('ok')"]
                    else:
                        verify_cmd = [sys.executable, "-c", f"import {actual_backend}; print('ok')"]

                verify = _sp.run(verify_cmd, capture_output=True, text=True, timeout=30)

                if verify.returncode == 0 and "ok" in verify.stdout:
                    note = ""
                    if actual_backend != backend:
                        note = f" (used {actual_backend} as fallback)"
                    _update_job(
                        job_id, status="complete", progress=100,
                        message=f"Whisper installed successfully!{note}",
                        result={"backend": actual_backend, "installed": True},
                    )
                    logger.info(f"Whisper installed via strategy {si+1}: {actual_backend}")
                    return
                else:
                    last_error = f"Import failed: {verify.stderr[:300]}"
                    logger.warning(f"Strategy {si+1} verify failed: {last_error}")
                    continue

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Strategy {si+1} exception: {e}")
                continue

        # All strategies failed
        # Provide a helpful error message about Rust/tokenizers
        helpful = last_error
        if "metadata-generation-failed" in helpful or "rust" in helpful.lower() or "tokenizers" in helpful.lower():
            helpful = (
                "Could not install Whisper. The 'tokenizers' package needs either "
                "a pre-built wheel for your Python version or Rust to compile from source.\n\n"
                "Try one of these:\n"
                "1. Update Python to 3.10-3.12 (pre-built wheels available)\n"
                "2. Install Rust: https://rustup.rs/\n"
                "3. Run manually: pip install openai-whisper\n\n"
                f"Last error:\n{last_error[-200:]}"
            )

        _update_job(
            job_id, status="error",
            error=helpful,
            message="All install methods failed",
        )
        logger.error(f"All whisper install strategies failed. Last error: {last_error[:500]}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Export Rendered Video (FFmpeg concat)
# ---------------------------------------------------------------------------
@app.route("/export-video", methods=["POST"])
def export_video():
    """Render a new video file with silences removed (no visible cuts)."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    segments_data = data.get("segments", [])

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    if not segments_data:
        return jsonify({"error": "No segments provided"}), 400

    # Options
    video_codec = data.get("video_codec", "libx264")
    audio_codec = data.get("audio_codec", "aac")
    quality = data.get("quality", "medium")  # low, medium, high, lossless
    output_format = data.get("output_format", "mp4")
    audio_only = data.get("audio_only", False)

    # Audio-only: override format and extension
    audio_format_ext = data.get("audio_format", "mp3")  # mp3, wav, flac
    if audio_only:
        output_format = audio_format_ext

    job_id = _new_job("export-video", filepath)

    def _process():
        import subprocess as _sp

        try:
            _update_job(job_id, progress=5, message="Preparing export...")

            info = _probe_media(filepath)
            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            suffix = "_audio" if audio_only else "_opencut"
            output_path = os.path.join(effective_dir, f"{base_name}{suffix}.{output_format}")

            # Build filter_complex for segment extraction and concatenation
            # Using the segment approach: extract each segment, then concat
            total_segments = len(segments_data)

            if total_segments == 0:
                _update_job(job_id, status="error", error="No segments to export", message="No segments")
                return

            # Determine CRF / quality
            crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
            crf = crf_map.get(quality, "23")

            if audio_only:
                # Audio-only: just use atrim + concat (no video)
                filter_parts = []
                concat_inputs = []

                for i, seg in enumerate(segments_data):
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    if end <= start:
                        continue
                    filter_parts.append(
                        f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
                    )
                    concat_inputs.append(f"[a{i}]")

                if not concat_inputs:
                    _update_job(job_id, status="error", error="No valid segments", message="No valid segments")
                    return

                n = len(concat_inputs)
                concat_str = "".join(concat_inputs) + f"concat=n={n}:v=0:a=1[outa]"
                filter_parts.append(concat_str)
                filter_complex = ";\n".join(filter_parts)

                _update_job(job_id, progress=15, message=f"Rendering {n} segments to {output_format.upper()} (audio only)...")

                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-i", filepath,
                    "-filter_complex", filter_complex,
                    "-map", "[outa]", "-vn",
                ]

                # Audio codec by format
                if output_format == "wav":
                    cmd.extend(["-acodec", "pcm_s16le"])
                elif output_format == "flac":
                    cmd.extend(["-acodec", "flac"])
                else:
                    cmd.extend(["-acodec", "libmp3lame", "-b:a", "192k"])

                cmd.append(output_path)

            else:
                # Video + audio export
                filter_parts = []
                concat_inputs = []

                for i, seg in enumerate(segments_data):
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    if end <= start:
                        continue
                    filter_parts.append(
                        f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{i}]"
                    )
                    filter_parts.append(
                        f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]"
                    )
                    concat_inputs.append(f"[v{i}][a{i}]")

                    if i % 10 == 0:
                        pct = int(5 + (i / total_segments) * 10)
                        _update_job(job_id, progress=pct, message=f"Building filter graph ({i+1}/{total_segments} segments)...")

                if not concat_inputs:
                    _update_job(job_id, status="error", error="No valid segments after filtering", message="No valid segments")
                    return

                n = len(concat_inputs)
                concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
                filter_parts.append(concat_str)
                filter_complex = ";\n".join(filter_parts)

                _update_job(job_id, progress=15, message=f"Rendering {n} segments to {output_format.upper()}...")

                cmd = [
                    "ffmpeg", "-hide_banner", "-y",
                    "-i", filepath,
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]",
                ]

                if video_codec == "copy":
                    cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "fast"])
                elif video_codec == "libx264":
                    cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", "fast"])
                elif video_codec == "libx265":
                    cmd.extend(["-c:v", "libx265", "-crf", crf, "-preset", "fast"])
                else:
                    cmd.extend(["-c:v", video_codec])

                cmd.extend(["-c:a", audio_codec, "-b:a", "192k"])
                cmd.extend(["-pix_fmt", "yuv420p"])
                cmd.append(output_path)

            _update_job(job_id, progress=20, message="FFmpeg encoding in progress...")

            # Run FFmpeg with progress monitoring
            proc = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True
            )

            # Calculate expected output duration for progress estimation
            total_kept = sum(
                seg.get("end", 0) - seg.get("start", 0) for seg in segments_data
                if seg.get("end", 0) > seg.get("start", 0)
            )

            import re as _re
            for line in proc.stdout:
                line = line.strip()
                # Parse FFmpeg time= progress
                time_match = _re.search(r"time=(\d+):(\d+):(\d+\.?\d*)", line)
                if time_match and total_kept > 0:
                    h, m, s = float(time_match.group(1)), float(time_match.group(2)), float(time_match.group(3))
                    current_time = h * 3600 + m * 60 + s
                    pct = min(95, int(20 + (current_time / total_kept) * 75))
                    elapsed_str = f"{int(current_time // 60)}:{int(current_time % 60):02d}"
                    _update_job(job_id, progress=pct, message=f"Encoding... {elapsed_str} / {int(total_kept // 60)}:{int(total_kept % 60):02d}")

                if _is_cancelled(job_id):
                    proc.kill()
                    # Clean up partial file
                    if os.path.exists(output_path):
                        try:
                            os.unlink(output_path)
                        except OSError:
                            pass
                    return

            proc.wait()

            if proc.returncode != 0:
                _update_job(
                    job_id, status="error",
                    error=f"FFmpeg exited with code {proc.returncode}",
                    message="Encoding failed",
                )
                return

            # Get output file size
            output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            size_mb = output_size / (1024 * 1024)

            _update_job(
                job_id, status="complete", progress=100,
                message="Export complete!",
                result={
                    "output_path": output_path,
                    "segments": n,
                    "duration": round(total_kept, 2),
                    "file_size_mb": round(size_mb, 1),
                    "codec": video_codec,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ===========================================================================
# EXPORT & PUBLISH ENDPOINTS (Phase 6 - v0.9.0)
# ===========================================================================

@app.route("/export/platform-presets", methods=["GET"])
def export_platform_presets():
    """Return available platform export presets."""
    try:
        try:
            from .core.export import get_export_capabilities
        except ImportError:
            from opencut.core.export import get_export_capabilities
        caps = get_export_capabilities()
        return jsonify(caps)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export/render", methods=["POST"])
def export_render():
    """Render video with platform preset or custom settings."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    preset = data.get("preset", "")
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting render..."}

    def _run():
        try:
            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            if preset:
                try:
                    from .core.export import render_platform
                except ImportError:
                    from opencut.core.export import render_platform
                result = render_platform(
                    input_path=input_path, preset=preset,
                    output_dir=output_dir, on_progress=cb,
                )
            else:
                try:
                    from .core.export import render_custom
                except ImportError:
                    from opencut.core.export import render_custom
                result = render_custom(
                    input_path=input_path,
                    video_codec=data.get("video_codec", "libx264"),
                    audio_codec=data.get("audio_codec", "aac"),
                    resolution=data.get("resolution", ""),
                    width=int(data.get("width", 0)),
                    height=int(data.get("height", 0)),
                    crf=int(data.get("crf", 23)),
                    video_bitrate=data.get("video_bitrate", ""),
                    audio_bitrate=data.get("audio_bitrate", "192k"),
                    fps=float(data.get("fps", 0)),
                    pixel_format=data.get("pixel_format", "yuv420p"),
                    container=data.get("container", "mp4"),
                    output_dir=output_dir, on_progress=cb,
                )

            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration": result.duration,
                    "width": result.width,
                    "height": result.height,
                    "operation": result.operation,
                    "details": result.details,
                }
        except Exception as e:
            logger.exception("Export render failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/export/thumbnail", methods=["POST"])
def export_thumbnail():
    """Extract thumbnail(s) from video."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    timestamp = float(data.get("timestamp", -1))
    mode = data.get("mode", "single")
    count = int(data.get("count", 1))
    width = int(data.get("width", 1920))
    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Extracting thumbnail..."}

    def _run():
        try:
            try:
                from .core.export import extract_thumbnail
            except ImportError:
                from opencut.core.export import extract_thumbnail

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = extract_thumbnail(
                input_path=input_path, timestamp=timestamp,
                mode=mode, count=count, width=width,
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "mode": mode,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "details": result.details,
                }
        except Exception as e:
            logger.exception("Thumbnail extraction failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/export/burn-subs", methods=["POST"])
def export_burn_subs():
    """Burn subtitles into video (hardcode)."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    subtitle_path = data.get("subtitle_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400
    if not subtitle_path or not os.path.isfile(subtitle_path):
        return jsonify({"error": "Valid subtitle file required"}), 400

    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Burning subtitles..."}

    def _run():
        try:
            try:
                from .core.export import burn_subtitles
            except ImportError:
                from opencut.core.export import burn_subtitles

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = burn_subtitles(
                input_path=input_path, subtitle_path=subtitle_path,
                font_size=int(data.get("font_size", 24)),
                font_name=data.get("font_name", "Arial"),
                font_color=data.get("font_color", "white"),
                outline_color=data.get("outline_color", "black"),
                outline_width=int(data.get("outline_width", 2)),
                position=data.get("position", "bottom"),
                margin_v=int(data.get("margin_v", 30)),
                quality=data.get("quality", "medium"),
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration": result.duration,
                }
        except Exception as e:
            logger.exception("Subtitle burn-in failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/export/watermark", methods=["POST"])
def export_watermark():
    """Add watermark to video."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Adding watermark..."}

    def _run():
        try:
            try:
                from .core.export import add_watermark
            except ImportError:
                from opencut.core.export import add_watermark

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = add_watermark(
                input_path=input_path,
                watermark_type=data.get("watermark_type", "text"),
                text=data.get("text", ""),
                image_path=data.get("image_path", ""),
                position=data.get("position", "bottom-right"),
                opacity=float(data.get("opacity", 0.5)),
                font_size=int(data.get("font_size", 24)),
                font_color=data.get("font_color", "white"),
                margin=int(data.get("margin", 20)),
                scale=float(data.get("scale", 0.15)),
                quality=data.get("quality", "medium"),
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "file_size_mb": round(result.file_size_mb, 2),
                }
        except Exception as e:
            logger.exception("Watermark failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/export/gif", methods=["POST"])
def export_gif():
    """Export video segment as animated GIF."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Generating GIF..."}

    def _run():
        try:
            try:
                from .core.export import export_gif as _export_gif
            except ImportError:
                from opencut.core.export import export_gif as _export_gif

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = _export_gif(
                input_path=input_path,
                start=float(data.get("start", 0)),
                duration=float(data.get("duration", 5)),
                width=int(data.get("width", 480)),
                fps=int(data.get("fps", 15)),
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration": result.duration,
                }
        except Exception as e:
            logger.exception("GIF export failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/export/audio-extract", methods=["POST"])
def export_audio_extract():
    """Extract audio from video."""
    data = request.get_json(force=True)
    input_path = data.get("input") or data.get("input_path", "")
    if not input_path or not os.path.isfile(input_path):
        return jsonify({"error": "Valid input file required"}), 400

    output_dir = data.get("output_dir") or os.path.dirname(input_path)

    job_id = str(uuid.uuid4())[:8]
    with job_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "Extracting audio..."}

    def _run():
        try:
            try:
                from .core.export import extract_audio
            except ImportError:
                from opencut.core.export import extract_audio

            def cb(pct, msg):
                with job_lock:
                    jobs[job_id]["progress"] = pct
                    jobs[job_id]["message"] = msg

            result = extract_audio(
                input_path=input_path,
                codec=data.get("codec", "libmp3lame"),
                bitrate=data.get("bitrate", "192k"),
                sample_rate=int(data.get("sample_rate", 44100)),
                normalize=data.get("normalize", False),
                output_dir=output_dir, on_progress=cb,
            )
            with job_lock:
                jobs[job_id]["status"] = "done"
                jobs[job_id]["progress"] = 100
                jobs[job_id]["result"] = {
                    "output": result.output_path,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration": result.duration,
                }
        except Exception as e:
            logger.exception("Audio extraction failed")
            with job_lock:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "running"})



# ===========================================================================
# BATCH & WORKFLOW ENDPOINTS (Phase 7 - v1.0.0)
# ===========================================================================

@app.route("/batch/capabilities", methods=["GET"])
def batch_capabilities():
    """Return batch processing features and workflow presets."""
    try:
        try:
            from .core.batch import get_batch_capabilities
        except ImportError:
            from opencut.core.batch import get_batch_capabilities
        return jsonify(get_batch_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/inspect", methods=["POST"])
def batch_inspect():
    """Detailed media file inspection."""
    data = request.get_json(force=True)
    filepath = data.get("filepath") or data.get("input", "")
    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Valid file path required"}), 400
    try:
        try:
            from .core.batch import inspect_media
        except ImportError:
            from opencut.core.batch import inspect_media
        result = inspect_media(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/scan", methods=["POST"])
def batch_scan():
    """Scan folder for media files."""
    data = request.get_json(force=True)
    folder = data.get("folder", "")
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "Valid folder path required"}), 400
    try:
        try:
            from .core.batch import scan_folder
        except ImportError:
            from opencut.core.batch import scan_folder
        files = scan_folder(
            folder=folder,
            recursive=data.get("recursive", False),
        )
        return jsonify({"folder": folder, "count": len(files), "files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/workflow-presets", methods=["GET"])
def batch_workflow_presets():
    """Return available workflow presets."""
    try:
        try:
            from .core.batch import WORKFLOW_PRESETS
        except ImportError:
            from opencut.core.batch import WORKFLOW_PRESETS
        presets = {
            k: {"label": v["label"], "description": v["description"],
                 "steps": v["steps"]}
            for k, v in WORKFLOW_PRESETS.items()
        }
        return jsonify(presets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _batch_executor(input_path, steps, output_dir, progress_cb):
    """Execute workflow steps sequentially on a single file, chaining outputs."""
    current_input = input_path
    results = []

    for i, step in enumerate(steps):
        op = step.get("operation", "")
        params = step.get("params", {})
        step_pct = int(((i) / max(len(steps), 1)) * 100)
        progress_cb(step_pct, f"Step {i+1}/{len(steps)}: {op}")

        try:
            output = _run_workflow_step(current_input, op, params, output_dir)
            if output and isinstance(output, str) and os.path.isfile(output):
                current_input = output
            results.append({"step": i+1, "operation": op, "output": output or current_input, "status": "done"})
        except Exception as e:
            results.append({"step": i+1, "operation": op, "error": str(e), "status": "error"})
            logger.warning(f"Workflow step {op} failed: {e}")

    progress_cb(100, "Complete")
    return {"final_output": current_input, "steps": results}


def _run_workflow_step(input_path, operation, params, output_dir):
    """Execute a single workflow operation and return output path."""
    # Audio operations
    if operation == "denoise":
        try:
            from .core.audio_suite import denoise_audio
        except ImportError:
            from opencut.core.audio_suite import denoise_audio
        r = denoise_audio(input_path, strength=params.get("strength", "moderate"),
                          output_dir=output_dir)
        return r.get("output_path") or r.get("output")

    elif operation == "normalize":
        try:
            from .core.audio_suite import normalize_loudness
        except ImportError:
            from opencut.core.audio_suite import normalize_loudness
        r = normalize_loudness(input_path, target_lufs=params.get("target_lufs", -14),
                               output_dir=output_dir)
        return r.get("output_path") or r.get("output")

    elif operation == "eq":
        try:
            from .core.audio_suite import apply_eq
        except ImportError:
            from opencut.core.audio_suite import apply_eq
        r = apply_eq(input_path, preset=params.get("preset", "voice_clarity"),
                     output_dir=output_dir)
        return r.get("output_path") or r.get("output")

    elif operation == "silence_remove":
        try:
            from .core.silence import detect_silence
        except ImportError:
            from opencut.core.silence import detect_silence
        # Silence remove returns segments, not a file - skip chaining
        return None

    elif operation == "isolation":
        try:
            from .core.audio_suite import isolate_vocals
        except ImportError:
            from opencut.core.audio_suite import isolate_vocals
        r = isolate_vocals(input_path, output_dir=output_dir)
        return r.get("vocals_path") or r.get("output")

    # Video operations
    elif operation == "color_grade":
        try:
            from .core.video_fx import apply_lut
        except ImportError:
            from opencut.core.video_fx import apply_lut
        r = apply_lut(input_path, color_preset=params.get("preset", "cinematic"),
                      intensity=params.get("intensity", 0.5), output_dir=output_dir)
        return r.get("output_path") or r.get("output")

    elif operation == "reframe":
        try:
            from .core.video_fx import auto_reframe
        except ImportError:
            from opencut.core.video_fx import auto_reframe
        r = auto_reframe(input_path, target_aspect=params.get("aspect", "9:16"),
                         mode=params.get("mode", "center"), output_dir=output_dir)
        return r.get("output_path") or r.get("output")

    elif operation == "speed_ramp":
        try:
            from .core.speed_ramp import apply_speed_ramp
        except ImportError:
            from opencut.core.speed_ramp import apply_speed_ramp
        r = apply_speed_ramp(input_path, preset=params.get("preset", "smooth_slow"),
                             output_dir=output_dir)
        return r.get("output_path") or r.get("output")

    # Export operations
    elif operation == "platform_export":
        try:
            from .core.export import render_platform
        except ImportError:
            from opencut.core.export import render_platform
        r = render_platform(input_path, preset=params.get("preset", "youtube_1080"),
                            output_dir=output_dir)
        return r.output_path

    elif operation == "custom_render":
        try:
            from .core.export import render_custom
        except ImportError:
            from opencut.core.export import render_custom
        r = render_custom(input_path, output_dir=output_dir, **params)
        return r.output_path

    elif operation == "thumbnail":
        try:
            from .core.export import extract_thumbnail
        except ImportError:
            from opencut.core.export import extract_thumbnail
        r = extract_thumbnail(input_path, output_dir=output_dir,
                              mode=params.get("mode", "single"),
                              timestamp=params.get("timestamp", -1))
        return r.output_path

    elif operation == "gif_export":
        try:
            from .core.export import export_gif
        except ImportError:
            from opencut.core.export import export_gif
        r = export_gif(input_path, output_dir=output_dir,
                       start=params.get("start", 0),
                       duration=params.get("duration", 5),
                       width=params.get("width", 480),
                       fps=params.get("fps", 15))
        return r.output_path

    elif operation == "audio_extract":
        try:
            from .core.export import extract_audio
        except ImportError:
            from opencut.core.export import extract_audio
        r = extract_audio(input_path, output_dir=output_dir,
                          codec=params.get("codec", "libmp3lame"),
                          bitrate=params.get("bitrate", "192k"),
                          normalize=params.get("normalize", False))
        return r.output_path

    elif operation == "watermark":
        try:
            from .core.export import add_watermark
        except ImportError:
            from opencut.core.export import add_watermark
        r = add_watermark(input_path, output_dir=output_dir, **params)
        return r.output_path

    elif operation == "burn_subtitles":
        try:
            from .core.export import burn_subtitles
        except ImportError:
            from opencut.core.export import burn_subtitles
        r = burn_subtitles(input_path, output_dir=output_dir, **params)
        return r.output_path

    else:
        logger.warning(f"Unknown workflow operation: {operation}")
        return None


@app.route("/batch/start", methods=["POST"])
def batch_start():
    """Start a batch job with workflow steps."""
    data = request.get_json(force=True)
    file_paths = data.get("files", [])
    output_dir = data.get("output_dir", "")

    # Workflow: either a preset name or custom steps
    preset_name = data.get("preset", "")
    custom_steps = data.get("steps", [])

    if not file_paths:
        return jsonify({"error": "No files provided"}), 400

    # Resolve workflow steps
    if preset_name:
        try:
            from .core.batch import WORKFLOW_PRESETS
        except ImportError:
            from opencut.core.batch import WORKFLOW_PRESETS
        if preset_name not in WORKFLOW_PRESETS:
            return jsonify({"error": f"Unknown preset: {preset_name}"}), 400
        steps = WORKFLOW_PRESETS[preset_name]["steps"]
    elif custom_steps:
        steps = custom_steps
    else:
        return jsonify({"error": "Provide 'preset' or 'steps'"}), 400

    try:
        try:
            from .core.batch import create_batch, start_batch
        except ImportError:
            from opencut.core.batch import create_batch, start_batch

        job_id = create_batch(file_paths, steps, output_dir)
        start_batch(job_id, _batch_executor)
        return jsonify({"job_id": job_id, "status": "running", "files": len(file_paths), "steps": len(steps)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/status/<job_id>", methods=["GET"])
def batch_status(job_id):
    """Get batch job status."""
    try:
        try:
            from .core.batch import get_batch_status
        except ImportError:
            from opencut.core.batch import get_batch_status
        status = get_batch_status(job_id)
        if not status:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/cancel/<job_id>", methods=["POST"])
def batch_cancel(job_id):
    """Cancel a running batch job."""
    try:
        try:
            from .core.batch import cancel_batch
        except ImportError:
            from opencut.core.batch import cancel_batch
        ok = cancel_batch(job_id)
        return jsonify({"cancelled": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/jobs", methods=["GET"])
def batch_jobs():
    """List all batch jobs."""
    try:
        try:
            from .core.batch import list_batch_jobs
        except ImportError:
            from opencut.core.batch import list_batch_jobs
        return jsonify(list_batch_jobs())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/watch/start", methods=["POST"])
def batch_watch_start():
    """Start watching a folder for new files."""
    data = request.get_json(force=True)
    folder = data.get("folder", "")
    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "Valid folder path required"}), 400

    preset_name = data.get("preset", "")
    custom_steps = data.get("steps", [])

    if preset_name:
        try:
            from .core.batch import WORKFLOW_PRESETS
        except ImportError:
            from opencut.core.batch import WORKFLOW_PRESETS
        if preset_name not in WORKFLOW_PRESETS:
            return jsonify({"error": f"Unknown preset: {preset_name}"}), 400
        steps = WORKFLOW_PRESETS[preset_name]["steps"]
    elif custom_steps:
        steps = custom_steps
    else:
        return jsonify({"error": "Provide 'preset' or 'steps'"}), 400

    try:
        try:
            from .core.batch import start_watch
        except ImportError:
            from opencut.core.batch import start_watch
        watcher_id = start_watch(
            folder=folder,
            workflow_steps=steps,
            output_dir=data.get("output_dir", ""),
            executor_fn=_batch_executor,
        )
        return jsonify({"watcher_id": watcher_id, "status": "active", "folder": folder})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/watch/stop/<watcher_id>", methods=["POST"])
def batch_watch_stop(watcher_id):
    """Stop a folder watcher."""
    try:
        try:
            from .core.batch import stop_watch
        except ImportError:
            from opencut.core.batch import stop_watch
        ok = stop_watch(watcher_id)
        return jsonify({"stopped": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/watch/status/<watcher_id>", methods=["GET"])
def batch_watch_status(watcher_id):
    """Get watcher status."""
    try:
        try:
            from .core.batch import get_watch_status
        except ImportError:
            from opencut.core.batch import get_watch_status
        status = get_watch_status(watcher_id)
        if not status:
            return jsonify({"error": "Watcher not found"}), 404
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/watch/list", methods=["GET"])
def batch_watch_list():
    """List all folder watchers."""
    try:
        try:
            from .core.batch import list_watchers
        except ImportError:
            from opencut.core.batch import list_watchers
        return jsonify(list_watchers())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# VOICE LAB ENDPOINTS (Phase 4 - v0.7.0 | Phase 5 - v0.8.0)
# ===========================================================================

@app.route("/voice/speakers", methods=["GET"])
def voice_speakers():
    """List all available TTS speakers (built-in + saved profiles)."""
    try:
        from opencut.core.voice_lab import get_speakers, get_emotion_presets, get_supported_languages
    except ImportError:
        try:
            from .core.voice_lab import get_speakers, get_emotion_presets, get_supported_languages
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    return jsonify({
        "speakers": get_speakers(),
        "emotions": get_emotion_presets(),
        "languages": get_supported_languages(),
    })


@app.route("/voice/check", methods=["GET"])
def voice_check():
    """Check if Qwen3-TTS is installed and ready."""
    try:
        from opencut.core.voice_lab import check_qwen_tts_available
    except ImportError:
        try:
            from .core.voice_lab import check_qwen_tts_available
        except ImportError:
            return jsonify({"installed": False, "message": "Voice Lab module not found"})

    installed, message = check_qwen_tts_available()
    return jsonify({"installed": installed, "message": message})


@app.route("/voice/install", methods=["POST"])
def voice_install():
    """Install qwen-tts and dependencies."""
    try:
        from opencut.core.voice_lab import install_qwen_tts
    except ImportError:
        try:
            from .core.voice_lab import install_qwen_tts
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    job_id = _new_job("install-voice", "qwen-tts")

    def _process():
        try:
            def _prog(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            _update_job(job_id, progress=5, message="Installing Qwen3-TTS...")
            success = install_qwen_tts(progress_callback=_prog)

            if success:
                _update_job(
                    job_id, status="complete", progress=100,
                    message="Qwen3-TTS installed successfully!",
                    result={"installed": True},
                )
            else:
                _update_job(
                    job_id, status="error",
                    error="Installation failed. Check logs for details.",
                    message="Installation failed",
                )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/voice/generate", methods=["POST"])
def voice_generate():
    """Generate speech from text using a preset voice or cloned voice."""
    try:
        from opencut.core.voice_lab import (
            generate_tts, generate_voice_clone, generate_voice_design,
            get_voice_profile, check_qwen_tts_available,
        )
    except ImportError:
        try:
            from .core.voice_lab import (
                generate_tts, generate_voice_clone, generate_voice_design,
                get_voice_profile, check_qwen_tts_available,
            )
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    installed, msg = check_qwen_tts_available()
    if not installed:
        return jsonify({"error": "Qwen3-TTS is not installed. Install from Settings."}), 400

    data = request.get_json(force=True) if request.data else {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    mode = data.get("mode", "tts")  # tts, clone, design
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "English")
    emotion = data.get("emotion", "neutral")
    speed = float(data.get("speed", 1.0))
    quality = data.get("quality", "auto")
    output_dir = _resolve_output_dir(data.get("filepath", ""), data.get("output_dir", ""))

    # For clone mode with a profile ID
    profile_id = data.get("profile_id", "")
    ref_audio = data.get("ref_audio", "")
    ref_text = data.get("ref_text", "")

    # For design mode
    voice_description = data.get("voice_description", "")

    job_id = _new_job("voice-generate", text[:50])

    def _process():
        try:
            def _prog(pct, msg):
                if _is_cancelled(job_id):
                    return
                _update_job(job_id, progress=pct, message=msg)

            _update_job(job_id, progress=5, message="Preparing voice generation...")

            result = None

            if mode == "clone":
                # Resolve reference audio from profile or direct path
                clone_ref = ref_audio
                clone_text = ref_text

                if profile_id:
                    profile = get_voice_profile(profile_id)
                    if not profile:
                        _update_job(job_id, status="error",
                                    error=f"Voice profile '{profile_id}' not found")
                        return
                    clone_ref = profile.ref_audio_path
                    clone_text = profile.ref_text

                if not clone_ref:
                    _update_job(job_id, status="error",
                                error="No reference audio provided for cloning")
                    return

                result = generate_voice_clone(
                    text=text, ref_audio=clone_ref, ref_text=clone_text,
                    language=language, speed=speed, quality=quality,
                    output_dir=output_dir, progress_callback=_prog,
                )

            elif mode == "design":
                if not voice_description:
                    _update_job(job_id, status="error",
                                error="No voice description provided")
                    return

                result = generate_voice_design(
                    text=text, voice_description=voice_description,
                    language=language, speed=speed, output_dir=output_dir,
                    progress_callback=_prog,
                )

            else:
                # Default: TTS with preset speaker
                result = generate_tts(
                    text=text, speaker=speaker, language=language,
                    emotion=emotion, speed=speed, quality=quality,
                    output_dir=output_dir, progress_callback=_prog,
                )

            _update_job(
                job_id, status="complete", progress=100,
                message="Speech generated!",
                result={
                    "output_path": result.output_path,
                    "duration_seconds": result.duration_seconds,
                    "sample_rate": result.sample_rate,
                    "model_used": result.model_used,
                    "speaker": result.speaker,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/voice/replace", methods=["POST"])
def voice_replace_endpoint():
    """Replace spoken words in audio using voice cloning."""
    try:
        from opencut.core.voice_lab import voice_replace, check_qwen_tts_available
    except ImportError:
        try:
            from .core.voice_lab import voice_replace, check_qwen_tts_available
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    installed, msg = check_qwen_tts_available()
    if not installed:
        return jsonify({"error": "Qwen3-TTS is not installed."}), 400

    data = request.get_json(force=True) if request.data else {}
    filepath = data.get("filepath", "")
    original_text = data.get("original_text", "").strip()
    replacement_text = data.get("replacement_text", "").strip()
    language = data.get("language", "English")
    quality = data.get("quality", "auto")
    output_dir = _resolve_output_dir(filepath, data.get("output_dir", ""))

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "Audio file not found"}), 400
    if not original_text or not replacement_text:
        return jsonify({"error": "Both original and replacement text are required"}), 400

    job_id = _new_job("voice-replace", filepath)

    def _process():
        try:
            def _prog(pct, msg):
                if _is_cancelled(job_id):
                    return
                _update_job(job_id, progress=pct, message=msg)

            _update_job(job_id, progress=5, message="Starting voice replacement...")

            result = voice_replace(
                original_audio=filepath,
                original_text=original_text,
                replacement_text=replacement_text,
                language=language, quality=quality,
                output_dir=output_dir, progress_callback=_prog,
            )

            _update_job(
                job_id, status="complete", progress=100,
                message="Voice replacement complete!",
                result={
                    "output_path": result.output_path,
                    "duration_seconds": result.duration_seconds,
                    "original_text": original_text,
                    "replacement_text": replacement_text,
                },
            )

        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/voice/profiles", methods=["GET"])
def voice_profiles_list():
    """List all saved voice profiles."""
    try:
        from opencut.core.voice_lab import list_voice_profiles
    except ImportError:
        try:
            from .core.voice_lab import list_voice_profiles
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    profiles = list_voice_profiles()
    return jsonify({
        "profiles": [p.to_dict() for p in profiles],
    })


@app.route("/voice/profiles", methods=["POST"])
def voice_profiles_save():
    """Save a new voice profile from a reference audio file."""
    try:
        from opencut.core.voice_lab import save_voice_profile
    except ImportError:
        try:
            from .core.voice_lab import save_voice_profile
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    data = request.get_json(force=True) if request.data else {}
    name = data.get("name", "").strip()
    ref_audio = data.get("ref_audio", "")
    ref_text = data.get("ref_text", "")
    language = data.get("language", "English")
    description = data.get("description", "")

    if not name:
        return jsonify({"error": "Profile name is required"}), 400
    if not ref_audio or not os.path.isfile(ref_audio):
        return jsonify({"error": "Reference audio file not found"}), 400

    try:
        profile = save_voice_profile(
            name=name, ref_audio_path=ref_audio,
            ref_text=ref_text, language=language,
            description=description,
        )
        return jsonify({"profile": profile.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice/profiles/<profile_id>", methods=["DELETE"])
def voice_profiles_delete(profile_id):
    """Delete a voice profile."""
    try:
        from opencut.core.voice_lab import delete_voice_profile
    except ImportError:
        try:
            from .core.voice_lab import delete_voice_profile
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    deleted = delete_voice_profile(profile_id)
    if deleted:
        return jsonify({"deleted": True, "id": profile_id})
    else:
        return jsonify({"error": "Profile not found"}), 404


@app.route("/voice/unload", methods=["POST"])
def voice_unload():
    """Unload all voice models from memory."""
    try:
        from opencut.core.voice_lab import unload_models
    except ImportError:
        try:
            from .core.voice_lab import unload_models
        except ImportError:
            return jsonify({"error": "Voice Lab module not found"}), 500

    unload_models()
    return jsonify({"unloaded": True, "message": "All voice models unloaded"})


@app.route("/voice/preview/<path:filename>", methods=["GET"])
def voice_preview(filename):
    """Serve a generated audio file for in-panel preview playback."""
    import mimetypes

    # Security: only serve files from cache/voice dir or output dirs
    # Resolve to absolute path and check
    safe_dirs = []
    try:
        from opencut.core.voice_lab import _get_cache_dir
        safe_dirs.append(str(_get_cache_dir()))
    except ImportError:
        try:
            from .core.voice_lab import _get_cache_dir
            safe_dirs.append(str(_get_cache_dir()))
        except ImportError:
            pass

    # Check if file is a full path passed as the "filename"
    filepath = filename
    if not os.path.isabs(filepath):
        # Try in cache dir
        for d in safe_dirs:
            candidate = os.path.join(d, filepath)
            if os.path.isfile(candidate):
                filepath = candidate
                break

    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    mime = mimetypes.guess_type(filepath)[0] or "audio/wav"

    def generate():
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return Response(generate(), mimetype=mime, headers={
        "Content-Disposition": f"inline; filename={os.path.basename(filepath)}",
        "Accept-Ranges": "bytes",
    })


# ---------------------------------------------------------------------------
# PID File Management
# ---------------------------------------------------------------------------
PID_FILE = os.path.join(os.path.expanduser("~"), ".opencut", "server.pid")


def _write_pid(port: int):
    """Write current PID and port to file so future instances can find us."""
    try:
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(f"{os.getpid()}\n{port}\n")
        logger.debug(f"Wrote PID file: pid={os.getpid()} port={port}")
    except Exception as e:
        logger.warning(f"Could not write PID file: {e}")


def _read_pid():
    """Read PID and port from file. Returns (pid, port) or (None, None)."""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                lines = f.read().strip().split("\n")
            pid = int(lines[0]) if lines else None
            port = int(lines[1]) if len(lines) > 1 else None
            return pid, port
    except Exception:
        pass
    return None, None


def _remove_pid():
    """Remove PID file."""
    try:
        if os.path.exists(PID_FILE):
            os.unlink(PID_FILE)
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid is None:
        return False
    try:
        if sys.platform == "win32":
            result = _sp.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)  # Signal 0 = check if alive
            return True
    except (OSError, _sp.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Port Checking (with SO_REUSEADDR to handle TIME_WAIT)
# ---------------------------------------------------------------------------
def _check_port(host: str, port: int) -> bool:
    """
    Check if a port is available for binding.
    Uses SO_REUSEADDR so TIME_WAIT sockets from recently-killed servers
    don't falsely report the port as busy.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _is_opencut_on_port(host: str, port: int) -> bool:
    """Check if an OpenCut server is responding on the given port."""
    import urllib.request
    try:
        req = urllib.request.Request(f"http://{host}:{port}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Kill Strategies (tried in order, each one more aggressive)
# ---------------------------------------------------------------------------
def _kill_via_shutdown_endpoint(host: str, port: int) -> bool:
    """Strategy 1: Ask the server to shut itself down via HTTP."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/shutdown",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b"{}",
        )
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        old_pid = data.get("pid")
        logger.info(f"Shutdown request accepted by server on :{port} (pid={old_pid})")
        return True
    except Exception:
        return False


def _kill_via_pid(pid: int) -> bool:
    """Strategy 2: Kill a specific PID directly."""
    if pid is None or not _is_pid_alive(pid):
        return False
    try:
        print(f"  Killing old server process (PID {pid})...")
        logger.info(f"Killing PID {pid}")
        if sys.platform == "win32":
            # /F = force, /T = kill entire process tree (bat launcher + python)
            _sp.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=10)
        else:
            os.kill(pid, 9)  # SIGKILL
            # Reap zombie if it happens to be our child
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass  # Not our child - init will reap it

        # Verify kill worked (retry a few times)
        for _ in range(6):
            time.sleep(0.3)
            if not _is_pid_alive(pid):
                return True
        logger.warning(f"PID {pid} still alive after kill attempt")
        return False
    except Exception as e:
        logger.warning(f"PID kill failed for {pid}: {e}")
        return False


def _kill_via_netstat(host: str, port: int) -> bool:
    """Strategy 3: Find PID holding the port via netstat and force-kill it."""
    killed_any = False
    try:
        if sys.platform == "win32":
            # Use findstr with word boundary pattern for exact port match
            # Match patterns like ":5679 " or ":5679\n" at end of address field
            result = _sp.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                # Only match LISTENING state with exact port
                # Format: "  TCP    127.0.0.1:5679    0.0.0.0:0    LISTENING    12345"
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    local_addr = parts[1]
                    # Check for exact port match (address ends with :PORT)
                    if local_addr.endswith(f":{port}"):
                        pid = parts[4]
                        if pid.isdigit() and int(pid) != os.getpid():
                            print(f"  Found process {pid} on port {port}, killing...")
                            logger.info(f"Killing PID {pid} found on port {port}")
                            _sp.run(["taskkill", "/F", "/T", "/PID", pid],
                                    capture_output=True, timeout=10)
                            killed_any = True
        else:
            result = _sp.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5
            )
            for pid in result.stdout.strip().split():
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"  Found process {pid} on port {port}, killing...")
                    logger.info(f"Killing PID {pid} found on port {port}")
                    os.kill(int(pid), 9)
                    killed_any = True
    except Exception as e:
        logger.warning(f"Netstat kill failed for port {port}: {e}")
    return killed_any


def _wait_for_port(host: str, port: int, timeout: float = 8.0) -> bool:
    """Wait up to `timeout` seconds for a port to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_port(host, port):
            return True
        time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Master Kill Sequence: nuke everything, reclaim the port
# ---------------------------------------------------------------------------
def _nuke_old_servers(host: str, port: int) -> bool:
    """
    Aggressively kill any existing OpenCut server(s) to reclaim the port.
    Tries multiple strategies in sequence. Returns True if port is free.
    """
    print(f"  Port {port} is in use. Cleaning up...")
    logger.info(f"Port {port} busy - starting kill sequence")

    # --- Step 1: Send /shutdown to all ports in our range ---
    for p in range(port, port + 11):
        _kill_via_shutdown_endpoint(host, p)

    if _wait_for_port(host, port, timeout=3.0):
        print(f"  Graceful shutdown succeeded.")
        return True

    # --- Step 2: Kill via PID file ---
    old_pid, old_port = _read_pid()
    if old_pid:
        _kill_via_pid(old_pid)
        _remove_pid()
        if _wait_for_port(host, port, timeout=3.0):
            print(f"  Killed old server via PID file (PID {old_pid}).")
            return True

    # --- Step 3: Kill via netstat (find whoever is holding the port) ---
    _kill_via_netstat(host, port)
    if _wait_for_port(host, port, timeout=4.0):
        print(f"  Killed process holding port {port}.")
        return True

    # --- Step 4: Last check with SO_REUSEADDR (TIME_WAIT is OK) ---
    if _check_port(host, port):
        print(f"  Port {port} available (socket in TIME_WAIT, safe to reuse).")
        return True

    print(f"  Could not free port {port}.")
    logger.warning(f"All kill strategies failed for port {port}")
    return False


# ---------------------------------------------------------------------------
# Server Startup
# ---------------------------------------------------------------------------
def run_server(host="127.0.0.1", port=5679, debug=False):
    """Start the OpenCut backend server."""
    effective_port = port

    if not _check_port(host, port):
        # Port is busy - run the kill sequence
        if _nuke_old_servers(host, port):
            effective_port = port
        else:
            # Kill sequence failed - find an alternate port
            print(f"  Searching for an open port...")
            for offset in range(1, 11):
                candidate = port + offset
                if _check_port(host, candidate):
                    effective_port = candidate
                    print(f"  Using port {effective_port} instead.")
                    break
            else:
                print(f"")
                print(f"  ERROR: Ports {port}-{port + 10} are all in use.")
                print(f"  Kill any stuck python processes and try again.")
                print(f"")
                sys.exit(1)

    # Write PID file so future instances can kill us
    _write_pid(effective_port)

    # Register cleanup on normal exit
    import atexit
    atexit.register(_remove_pid)

    print(f"")
    print(f"  OpenCut Backend Server v1.0.0")
    print(f"  Listening on http://{host}:{effective_port}")
    print(f"  PID: {os.getpid()}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Press Ctrl+C to stop")
    print(f"")
    logger.info(f"Server starting on http://{host}:{effective_port} (pid={os.getpid()})")
    app.run(host=host, port=effective_port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenCut Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5679, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug)
