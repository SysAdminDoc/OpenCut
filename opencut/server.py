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

from flask import Flask, request, jsonify, Response, send_from_directory
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

# Temp directory for audio previews
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "opencut_previews")
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Handle both relative and absolute imports
try:
    from .utils.media import probe as _probe_media
    from .utils.config import SilenceConfig, ExportConfig, CaptionConfig, get_preset
    from .core.silence import detect_speech, get_edit_summary
    from .core.zoom import generate_zoom_events
    from .export.premiere import export_premiere_xml
    from .export.srt import export_srt, export_vtt, export_json
except ImportError:
    from opencut.utils.media import probe as _probe_media
    from opencut.utils.config import SilenceConfig, ExportConfig, CaptionConfig, get_preset
    from opencut.core.silence import detect_speech, get_edit_summary
    from opencut.core.zoom import generate_zoom_events
    from opencut.export.premiere import export_premiere_xml
    from opencut.export.srt import export_srt, export_vtt, export_json


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
    return jsonify({"status": "ok", "version": "0.3.0"})


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
            # Ensure Pillow is installed
            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                _update_job(job_id, progress=5, message="Installing Pillow...")
                import subprocess as sp2
                sp2.check_call(
                    [sys.executable, "-m", "pip", "install", "Pillow", "--quiet"],
                    timeout=120,
                )

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
                except ImportError:
                    from opencut.core.captions import remap_captions_to_segments
                from .core.silence import TimeSegment
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

            result = render_styled_caption_video(
                transcription,
                overlay_path,
                style_name=style_name,
                video_width=vid_w,
                video_height=vid_h,
                fps=vid_fps,
                action_indices=action_indices,
                total_duration=transcription.duration or info.duration,
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


# ---------------------------------------------------------------------------
# Waveform Data (audio energy for visualization)
# ---------------------------------------------------------------------------
@app.route("/waveform", methods=["POST"])
def get_waveform():
    """Return downsampled audio energy peaks for waveform visualization."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    num_peaks = min(int(data.get("peaks", 500)), 2000)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    try:
        try:
            from .core.audio import analyze_energy
        except ImportError:
            from opencut.core.audio import analyze_energy

        energies = analyze_energy(filepath)

        if not energies:
            return jsonify({"peaks": [], "duration": 0, "max_rms": 0})

        # Downsample to num_peaks
        total = len(energies)
        step = max(1, total // num_peaks)
        peaks = []
        for i in range(0, total, step):
            chunk = energies[i:min(i + step, total)]
            max_rms_val = max(e.rms for e in chunk)
            avg_time = (chunk[0].time + chunk[-1].time) / 2
            peaks.append({"t": round(avg_time, 3), "r": round(max_rms_val, 5)})

        duration = energies[-1].time if energies else 0
        max_rms_global = max(e.rms for e in energies)

        return jsonify({
            "peaks": peaks,
            "duration": round(duration, 3),
            "max_rms": round(max_rms_global, 5),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Regenerate XML from edited segments (no re-detection)
# ---------------------------------------------------------------------------
@app.route("/regenerate", methods=["POST"])
def regenerate_xml():
    """Regenerate Premiere XML from provided segment list without re-analyzing."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    segments_data = data.get("segments", [])
    output_dir = data.get("output_dir", "")
    seq_name = data.get("sequence_name", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    if not segments_data:
        return jsonify({"error": "No segments provided"}), 400

    try:
        try:
            from .core.silence import TimeSegment
        except ImportError:
            from opencut.core.silence import TimeSegment

        segments = [
            TimeSegment(start=s["start"], end=s["end"], label="speech")
            for s in segments_data
            if s.get("end", 0) > s.get("start", 0)
        ]

        effective_dir = _resolve_output_dir(filepath, output_dir)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        xml_path = os.path.join(effective_dir, f"{base_name}_opencut.xml")

        effective_name = seq_name or _make_sequence_name(filepath)
        ecfg = ExportConfig(sequence_name=effective_name)

        export_premiere_xml(filepath, segments, xml_path, config=ecfg)
        summary = get_edit_summary(filepath, segments)

        return jsonify({
            "xml_path": xml_path,
            "summary": summary,
            "segments": len(segments),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Audio Preview (render short clip for previewing cuts)
# ---------------------------------------------------------------------------
@app.route("/preview-audio", methods=["POST"])
def preview_audio():
    """Generate a short audio preview clip around a given time point."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    center_time = float(data.get("time", 0))
    duration = min(float(data.get("duration", 3.0)), 10.0)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    try:
        start = max(0, center_time - duration / 2)
        preview_name = f"preview_{uuid.uuid4().hex[:8]}.wav"
        preview_path = os.path.join(PREVIEW_DIR, preview_name)

        # Clean old previews (keep last 10)
        _cleanup_previews()

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", filepath,
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            preview_path,
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode != 0:
            logger.error(f"Preview FFmpeg error: {result.stderr}")
            return jsonify({"error": "Failed to generate preview"}), 500

        return jsonify({
            "url": f"/preview-files/{preview_name}",
            "start": round(start, 3),
            "duration": round(duration, 3),
        })

    except Exception as e:
        logger.exception("preview_audio failed")
        return jsonify({"error": str(e)}), 500


@app.route("/preview-files/<filename>", methods=["GET"])
def serve_preview(filename):
    """Serve a generated audio preview file."""
    # Sanitize: only allow simple filenames
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Not found", 404
    return send_from_directory(PREVIEW_DIR, filename, mimetype="audio/wav")


def _cleanup_previews(keep=10):
    """Remove old preview files, keeping only the most recent `keep`."""
    try:
        files = sorted(
            Path(PREVIEW_DIR).glob("preview_*.wav"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for f in files[keep:]:
            try:
                f.unlink()
            except OSError:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Capabilities check
# ---------------------------------------------------------------------------
@app.route("/capabilities", methods=["GET"])
def capabilities():
    """Report which optional features are available."""
    caps = {
        "silence": True,
        "zoom": True,
        "ffmpeg": True,
    }

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

    try:
        try:
            from .core.diarize import check_pyannote_available
        except ImportError:
            from opencut.core.diarize import check_pyannote_available
        caps["diarize"] = check_pyannote_available()
    except Exception:
        caps["diarize"] = False

    import shutil
    caps["ffmpeg"] = shutil.which("ffmpeg") is not None
    caps["log_file"] = LOG_FILE

    return jsonify(caps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _check_port(host: str, port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def run_server(host="127.0.0.1", port=5679, debug=False):
    """Start the OpenCut backend server."""
    if not _check_port(host, port):
        print(f"")
        print(f"  ERROR: Port {port} is already in use.")
        print(f"")
        print(f"  This usually means another OpenCut server is already running.")
        print(f"  Either stop the other instance, or use a different port:")
        print(f"    python -m opencut.server --port {port + 1}")
        print(f"")
        sys.exit(1)

    print(f"")
    print(f"  OpenCut Backend Server v0.3.0")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Press Ctrl+C to stop")
    print(f"")
    logger.info(f"Server starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenCut Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5679, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug)
