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

# ---------------------------------------------------------------------------
# Bundled Installation Detection
# ---------------------------------------------------------------------------
# When running from the standalone installer, these env vars are set by the launcher
BUNDLED_MODE = os.environ.get("OPENCUT_BUNDLED", "").lower() == "true" or \
               os.environ.get("WHISPER_MODELS_DIR") is not None

# Model paths - use bundled paths if available, otherwise default to cache dirs
WHISPER_MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", None)
TORCH_HOME = os.environ.get("TORCH_HOME", None)
FLORENCE_MODEL_DIR = os.environ.get("OPENCUT_FLORENCE_DIR", None)
LAMA_MODEL_DIR = os.environ.get("OPENCUT_LAMA_DIR", None)

if BUNDLED_MODE:
    logger.info("Running in bundled mode")
    if WHISPER_MODELS_DIR:
        logger.info(f"  Whisper models: {WHISPER_MODELS_DIR}")
    if TORCH_HOME:
        logger.info(f"  Torch home: {TORCH_HOME}")
    if FLORENCE_MODEL_DIR:
        logger.info(f"  Florence model: {FLORENCE_MODEL_DIR}")
    if LAMA_MODEL_DIR:
        logger.info(f"  LaMA model: {LAMA_MODEL_DIR}")

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


# ---------------------------------------------------------------------------
# Whisper Settings (loaded early for health check)
# ---------------------------------------------------------------------------
_WHISPER_SETTINGS_FILE = os.path.join(
    os.path.expanduser("~"), ".opencut", "whisper_settings.json"
)


def _load_whisper_settings() -> dict:
    """Load Whisper settings from disk."""
    defaults = {"cpu_mode": False, "model": "base"}
    try:
        if os.path.exists(_WHISPER_SETTINGS_FILE):
            with open(_WHISPER_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                defaults.update(saved)
    except Exception:
        pass
    return defaults


def _save_whisper_settings(settings: dict):
    """Save Whisper settings to disk."""
    try:
        os.makedirs(os.path.dirname(_WHISPER_SETTINGS_FILE), exist_ok=True)
        with open(_WHISPER_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save Whisper settings: {e}")


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
def check_demucs_available():
    """Check if Demucs is installed and available."""
    try:
        import demucs
        return True
    except ImportError:
        return False

def check_watermark_available():
    """Check if watermark removal dependencies are installed."""
    try:
        import transformers
        from simple_lama_inpainting import SimpleLama
        return True
    except ImportError:
        return False

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
    
    # Check Demucs availability
    caps["separation"] = check_demucs_available()
    
    # Check watermark removal availability
    caps["watermark_removal"] = check_watermark_available()
    
    # Include Whisper settings
    try:
        whisper_settings = _load_whisper_settings()
        caps["whisper_cpu_mode"] = whisper_settings.get("cpu_mode", False)
    except Exception:
        caps["whisper_cpu_mode"] = False
    
    # Check video AI availability
    try:
        try:
            from .core.video_ai import get_ai_capabilities
        except ImportError:
            from opencut.core.video_ai import get_ai_capabilities
        caps["video_ai"] = get_ai_capabilities()
    except Exception:
        caps["video_ai"] = {}

    # Check pedalboard availability
    try:
        try:
            from .core.audio_pro import check_pedalboard_available, check_deepfilter_available
        except ImportError:
            from opencut.core.audio_pro import check_pedalboard_available, check_deepfilter_available
        caps["pedalboard"] = check_pedalboard_available()
        caps["deepfilter"] = check_deepfilter_available()
    except Exception:
        caps["pedalboard"] = False
        caps["deepfilter"] = False

    # Video effects always available (FFmpeg-based)
    caps["video_fx"] = True

    # Face tools
    try:
        try:
            from .core.face_tools import check_face_tools_available
        except ImportError:
            from opencut.core.face_tools import check_face_tools_available
        caps["face_tools"] = check_face_tools_available()
    except Exception:
        caps["face_tools"] = {}

    # Style transfer (always available via OpenCV DNN)
    caps["style_transfer"] = True

    # Enhanced captions
    try:
        try:
            from .core.captions_enhanced import check_whisperx_available, check_nllb_available
        except ImportError:
            from opencut.core.captions_enhanced import check_whisperx_available, check_nllb_available
        caps["whisperx"] = check_whisperx_available()
        caps["nllb"] = check_nllb_available()
    except Exception:
        caps["whisperx"] = False
        caps["nllb"] = False

    # Voice generation
    try:
        try:
            from .core.voice_gen import check_edge_tts_available, check_kokoro_available
        except ImportError:
            from opencut.core.voice_gen import check_edge_tts_available, check_kokoro_available
        caps["edge_tts"] = check_edge_tts_available()
        caps["kokoro"] = check_kokoro_available()
    except Exception:
        caps["edge_tts"] = False
        caps["kokoro"] = False

    # Caption burn-in always available (FFmpeg-based)
    caps["burnin"] = True

    return jsonify({"status": "ok", "version": "1.0.0-beta", "capabilities": caps})


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
    """Isolate vocals by emphasizing voice frequencies."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("isolate", filepath)

    def _process():
        try:
            try:
                from .core.audio_suite import isolate_voice
            except ImportError:
                from opencut.core.audio_suite import isolate_voice

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1]
            out_path = os.path.join(effective_dir, f"{base_name}_voice{ext}")

            isolate_voice(filepath, out_path, on_progress=_on_progress)

            _update_job(
                job_id, status="complete", progress=100,
                message="Voice isolation complete!",
                result={"output_path": out_path},
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
# Audio Suite: AI Stem Separation (Demucs)
# ---------------------------------------------------------------------------
@app.route("/audio/separate", methods=["POST"])
def audio_separate():
    """Separate audio into stems using Meta's Demucs AI model."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    model = data.get("model", "htdemucs")
    stems = data.get("stems", ["vocals", "no_vocals"])
    output_format = data.get("format", "wav")
    auto_import = data.get("auto_import", True)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    
    if not check_demucs_available():
        return jsonify({"error": "Demucs not installed. Please install it first."}), 400

    job_id = _new_job("separate", filepath)

    def _process():
        try:
            import subprocess
            import tempfile
            import shutil
            
            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            
            _update_job(job_id, progress=5, message="Extracting audio...")
            
            # Extract audio if video file
            video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext in video_exts:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_audio = tmp.name
                
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', filepath,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                    temp_audio
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                input_audio = temp_audio
            else:
                input_audio = filepath
            
            _update_job(job_id, progress=15, message=f"Running Demucs ({model})...")
            
            # Create temp output directory for demucs
            with tempfile.TemporaryDirectory() as temp_out:
                # Run demucs
                demucs_cmd = [
                    'python', '-m', 'demucs',
                    '--out', temp_out,
                    '-n', model,
                    '--two-stems=vocals' if set(stems) <= {'vocals', 'no_vocals'} else '',
                    input_audio
                ]
                # Remove empty strings from command
                demucs_cmd = [c for c in demucs_cmd if c]
                
                process = subprocess.Popen(
                    demucs_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # Monitor progress
                for line in process.stdout:
                    if '%' in line:
                        try:
                            pct_str = line.split('%')[0].strip().split()[-1]
                            pct = int(float(pct_str))
                            _update_job(job_id, progress=15 + int(pct * 0.7), message=f"Separating audio... {pct}%")
                        except:
                            pass
                
                process.wait()
                if process.returncode != 0:
                    raise Exception("Demucs separation failed")
                
                _update_job(job_id, progress=90, message="Saving stems...")
                
                # Find output files
                # Demucs outputs to: temp_out/model_name/track_name/stem.wav
                demucs_out_dir = None
                for root, dirs, files in os.walk(temp_out):
                    if any(f.endswith('.wav') for f in files):
                        demucs_out_dir = root
                        break
                
                if not demucs_out_dir:
                    raise Exception("Could not find Demucs output files")
                
                output_paths = []
                stem_map = {
                    'vocals': 'vocals',
                    'no_vocals': 'no_vocals', 
                    'drums': 'drums',
                    'bass': 'bass',
                    'other': 'other',
                    'piano': 'piano',
                    'guitar': 'guitar'
                }
                
                for stem in stems:
                    stem_file = stem_map.get(stem, stem)
                    src_path = os.path.join(demucs_out_dir, f"{stem_file}.wav")
                    
                    # Check alternate naming (instrumental vs no_vocals)
                    if not os.path.exists(src_path) and stem == 'no_vocals':
                        src_path = os.path.join(demucs_out_dir, "instrumental.wav")
                    
                    if os.path.exists(src_path):
                        # Convert to desired format
                        out_filename = f"{base_name}_{stem}.{output_format}"
                        out_path = os.path.join(effective_dir, out_filename)
                        
                        if output_format == 'wav':
                            shutil.copy2(src_path, out_path)
                        elif output_format == 'mp3':
                            subprocess.run([
                                'ffmpeg', '-y', '-i', src_path,
                                '-codec:a', 'libmp3lame', '-b:a', '320k',
                                out_path
                            ], check=True, capture_output=True)
                        elif output_format == 'flac':
                            subprocess.run([
                                'ffmpeg', '-y', '-i', src_path,
                                '-codec:a', 'flac',
                                out_path
                            ], check=True, capture_output=True)
                        
                        output_paths.append(out_path)
            
            # Cleanup temp audio if we extracted from video
            if file_ext in video_exts and os.path.exists(temp_audio):
                os.unlink(temp_audio)
            
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Separation complete! {len(output_paths)} stems exported.",
                result={
                    "output_paths": output_paths,
                    "auto_import": auto_import
                },
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio separation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/demucs/install", methods=["POST"])
def install_demucs():
    """Install Demucs for AI audio separation."""
    try:
        import subprocess
        import sys
        
        # Install demucs with torch
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'demucs', '--break-system-packages'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            # Try without --break-system-packages for older pip
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'demucs'],
                capture_output=True,
                text=True,
                timeout=600
            )
        
        if result.returncode == 0:
            return jsonify({"success": True, "message": "Demucs installed successfully"})
        else:
            return jsonify({"error": f"Installation failed: {result.stderr}"}), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Installation timed out. Please install manually: pip install demucs"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
# Watermark Removal (Florence-2 + LaMA)
# ---------------------------------------------------------------------------
@app.route("/video/watermark", methods=["POST"])
def video_watermark():
    """Remove watermarks from video or image using AI."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    max_bbox_percent = int(data.get("max_bbox_percent", 10))
    detection_prompt = data.get("detection_prompt", "watermark")
    detection_skip = int(data.get("detection_skip", 3))
    transparent = data.get("transparent", False)
    preview = data.get("preview", False)
    auto_import = data.get("auto_import", True)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    
    if not check_watermark_available():
        return jsonify({"error": "Watermark remover not installed. Please install it first."}), 400

    job_id = _new_job("watermark", filepath)

    def _process():
        try:
            import subprocess
            import tempfile
            
            effective_dir = _resolve_output_dir(filepath, output_dir)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1].lower()
            
            _update_job(job_id, progress=5, message="Initializing AI models...")
            
            # Determine if video or image
            video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
            is_video = ext in video_exts
            
            # Build output path
            suffix = "_preview" if preview else "_clean"
            if is_video:
                out_path = os.path.join(effective_dir, f"{base_name}{suffix}.mp4")
            else:
                out_ext = ".png" if transparent else ext
                out_path = os.path.join(effective_dir, f"{base_name}{suffix}{out_ext}")
            
            _update_job(job_id, progress=10, message="Running watermark detection...")
            
            # Build command for the watermark remover
            # We'll use subprocess to call the remwm.py script or use the modules directly
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                from simple_lama_inpainting import SimpleLama
                from PIL import Image
                import numpy as np
                import cv2
                import torch
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load Florence-2 model for detection
                _update_job(job_id, progress=15, message="Loading Florence-2 model...")
                
                # Use bundled model path if available
                if FLORENCE_MODEL_DIR and os.path.isdir(FLORENCE_MODEL_DIR):
                    model_id = FLORENCE_MODEL_DIR
                    logger.info(f"Using bundled Florence model: {model_id}")
                else:
                    model_id = "microsoft/Florence-2-base"
                    
                florence_model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True
                ).to(device)
                florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                
                # Load LaMA model for inpainting
                _update_job(job_id, progress=25, message="Loading LaMA inpainting model...")
                
                # SimpleLama uses its own model path, but we can set TORCH_HOME
                if LAMA_MODEL_DIR and os.path.isdir(LAMA_MODEL_DIR):
                    os.environ["LAMA_MODEL"] = os.path.join(LAMA_MODEL_DIR, "big-lama.pt")
                    logger.info(f"Using bundled LaMA model: {LAMA_MODEL_DIR}")
                    
                lama = SimpleLama()
                
                def detect_watermarks(image):
                    """Detect watermarks using Florence-2."""
                    task = "<OPEN_VOCABULARY_DETECTION>"
                    prompt = task + detection_prompt
                    
                    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        generated_ids = florence_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            do_sample=False,
                            num_beams=3
                        )
                    
                    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    parsed = florence_processor.post_process_generation(
                        generated_text, task=task, image_size=(image.width, image.height)
                    )
                    
                    bboxes = []
                    if task in parsed:
                        result = parsed[task]
                        if "bboxes" in result:
                            img_area = image.width * image.height
                            max_area = img_area * (max_bbox_percent / 100.0)
                            
                            for bbox in result["bboxes"]:
                                x1, y1, x2, y2 = bbox
                                bbox_area = (x2 - x1) * (y2 - y1)
                                if bbox_area <= max_area:
                                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                    
                    return bboxes
                
                def create_mask(image_size, bboxes, expand=5):
                    """Create binary mask from bounding boxes."""
                    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        x1 = max(0, x1 - expand)
                        y1 = max(0, y1 - expand)
                        x2 = min(image_size[0], x2 + expand)
                        y2 = min(image_size[1], y2 + expand)
                        mask[y1:y2, x1:x2] = 255
                    return mask
                
                def process_image(img_pil):
                    """Process a single image."""
                    bboxes = detect_watermarks(img_pil)
                    
                    if not bboxes:
                        return img_pil, False
                    
                    if preview:
                        # Draw detection boxes
                        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        for bbox in bboxes:
                            cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)), True
                    
                    if transparent:
                        # Make watermark area transparent
                        img_rgba = img_pil.convert("RGBA")
                        img_array = np.array(img_rgba)
                        mask = create_mask((img_pil.width, img_pil.height), bboxes)
                        img_array[:, :, 3] = np.where(mask > 0, 0, 255)
                        return Image.fromarray(img_array), True
                    
                    # Inpaint using LaMA
                    mask = create_mask((img_pil.width, img_pil.height), bboxes)
                    mask_pil = Image.fromarray(mask)
                    result = lama(img_pil, mask_pil)
                    return result, True
                
                if is_video:
                    # Process video
                    _update_job(job_id, progress=30, message="Processing video frames...")
                    
                    cap = cv2.VideoCapture(filepath)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Create temp output
                    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_video = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
                    
                    # Two-pass: detect watermarks on key frames, then apply to all
                    detected_masks = {}
                    frame_idx = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Detect on every N frames
                        if frame_idx % detection_skip == 0:
                            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            bboxes = detect_watermarks(img_pil)
                            if bboxes:
                                detected_masks[frame_idx] = bboxes
                        
                        progress = 30 + int((frame_idx / total_frames) * 30)
                        if frame_idx % 30 == 0:
                            _update_job(job_id, progress=progress, message=f"Detecting... frame {frame_idx}/{total_frames}")
                        
                        frame_idx += 1
                    
                    cap.release()
                    
                    # Second pass: process all frames
                    _update_job(job_id, progress=60, message="Inpainting frames...")
                    
                    cap = cv2.VideoCapture(filepath)
                    frame_idx = 0
                    current_mask = None
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Find nearest detected mask
                        if frame_idx in detected_masks:
                            current_mask = detected_masks[frame_idx]
                        
                        if current_mask and not preview:
                            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            mask = create_mask((width, height), current_mask)
                            mask_pil = Image.fromarray(mask)
                            result = lama(img_pil, mask_pil)
                            frame = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                        elif current_mask and preview:
                            for bbox in current_mask:
                                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        
                        out_video.write(frame)
                        
                        progress = 60 + int((frame_idx / total_frames) * 35)
                        if frame_idx % 30 == 0:
                            _update_job(job_id, progress=progress, message=f"Processing... frame {frame_idx}/{total_frames}")
                        
                        frame_idx += 1
                    
                    cap.release()
                    out_video.release()
                    
                    # Merge audio if ffmpeg available
                    _update_job(job_id, progress=95, message="Merging audio...")
                    try:
                        subprocess.run([
                            'ffmpeg', '-y',
                            '-i', temp_video,
                            '-i', filepath,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                            '-c:a', 'aac', '-b:a', '192k',
                            '-map', '0:v', '-map', '1:a?',
                            '-shortest',
                            out_path
                        ], check=True, capture_output=True)
                        os.unlink(temp_video)
                    except:
                        # No ffmpeg or no audio, just rename
                        import shutil
                        shutil.move(temp_video, out_path)
                
                else:
                    # Process image
                    _update_job(job_id, progress=50, message="Processing image...")
                    img_pil = Image.open(filepath).convert("RGB")
                    result, had_watermark = process_image(img_pil)
                    
                    if transparent and not preview:
                        result.save(out_path, "PNG")
                    else:
                        result.save(out_path)
                
                # Clean up models
                del florence_model, florence_processor, lama
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                _update_job(
                    job_id, status="complete", progress=100,
                    message="Watermark removal complete!" if not preview else "Preview complete!",
                    result={
                        "output_path": out_path,
                        "auto_import": auto_import
                    },
                )
                
            except Exception as e:
                raise Exception(f"Processing error: {str(e)}")
                
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Watermark removal error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/watermark/install", methods=["POST"])
def install_watermark():
    """Install watermark removal dependencies."""
    try:
        import subprocess
        import sys
        
        # Install required packages
        packages = [
            'transformers',
            'simple-lama-inpainting',
            'torch',
            'torchvision',
            'pillow',
            'opencv-python'
        ]
        
        for pkg in packages:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', pkg, '--break-system-packages'],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode != 0:
                # Try without --break-system-packages
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', pkg],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
        
        return jsonify({"success": True, "message": "Watermark remover installed successfully"})
        
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Installation timed out. Please install manually."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
            from .core.scene_detect import SPEED_RAMP_PRESETS
        except ImportError:
            from opencut.core.scene_detect import SPEED_RAMP_PRESETS
        return jsonify({"presets": SPEED_RAMP_PRESETS})
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
# Whisper Settings & Reinstall
# ---------------------------------------------------------------------------
@app.route("/whisper/settings", methods=["GET", "POST"])
def whisper_settings():
    """Get or update Whisper settings (CPU mode, default model)."""
    if request.method == "GET":
        settings = _load_whisper_settings()
        return jsonify(settings)
    
    # POST - update settings
    data = request.get_json(force=True) if request.data else {}
    settings = _load_whisper_settings()
    
    if "cpu_mode" in data:
        settings["cpu_mode"] = bool(data["cpu_mode"])
    if "model" in data:
        settings["model"] = str(data["model"])
    
    _save_whisper_settings(settings)
    logger.info(f"Whisper settings updated: {settings}")
    
    return jsonify({"success": True, "settings": settings})


@app.route("/whisper/clear-cache", methods=["POST"])
def whisper_clear_cache():
    """Clear downloaded Whisper model cache."""
    import shutil
    
    cleared = []
    errors = []
    
    # Common cache locations
    cache_paths = [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenCut", "models"),
    ]
    
    for cache_dir in cache_paths:
        if not cache_dir or not os.path.exists(cache_dir):
            continue
        
        try:
            # For huggingface, only remove whisper-related models
            if "huggingface" in cache_dir:
                for item in os.listdir(cache_dir):
                    if "whisper" in item.lower():
                        item_path = os.path.join(cache_dir, item)
                        shutil.rmtree(item_path, ignore_errors=True)
                        cleared.append(item_path)
            else:
                # Remove entire directory
                shutil.rmtree(cache_dir, ignore_errors=True)
                cleared.append(cache_dir)
        except Exception as e:
            errors.append(f"{cache_dir}: {e}")
    
    return jsonify({
        "success": len(errors) == 0,
        "cleared": cleared,
        "errors": errors,
        "message": f"Cleared {len(cleared)} cache location(s)"
    })


@app.route("/whisper/reinstall", methods=["POST"])
def whisper_reinstall():
    """Complete Whisper reinstall: uninstall, clear cache, reinstall fresh."""
    data = request.get_json(force=True) if request.data else {}
    backend = data.get("backend", "faster-whisper")
    cpu_mode = data.get("cpu_mode", False)
    
    job_id = _new_job("reinstall-whisper", backend)
    
    def _process():
        import subprocess as _sp
        import shutil
        
        try:
            # Step 1: Uninstall existing packages
            _update_job(job_id, progress=5, message="Uninstalling existing Whisper packages...")
            logger.info("Reinstall: Uninstalling existing packages")
            
            uninstall_pkgs = ["faster-whisper", "openai-whisper", "whisperx"]
            for pkg in uninstall_pkgs:
                try:
                    _sp.run(
                        [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                        capture_output=True, timeout=60
                    )
                except Exception:
                    pass
            
            if _is_cancelled(job_id):
                return
            
            # Step 2: Clear model cache
            _update_job(job_id, progress=20, message="Clearing model cache...")
            logger.info("Reinstall: Clearing cache")
            
            cache_paths = [
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
                os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenCut", "models"),
            ]
            
            for cache_dir in cache_paths:
                if not cache_dir or not os.path.exists(cache_dir):
                    continue
                try:
                    if "huggingface" in cache_dir:
                        for item in os.listdir(cache_dir):
                            if "whisper" in item.lower():
                                shutil.rmtree(os.path.join(cache_dir, item), ignore_errors=True)
                    else:
                        shutil.rmtree(cache_dir, ignore_errors=True)
                except Exception:
                    pass
            
            if _is_cancelled(job_id):
                return
            
            # Step 3: Clear pip cache for these packages
            _update_job(job_id, progress=30, message="Clearing pip cache...")
            try:
                _sp.run(
                    [sys.executable, "-m", "pip", "cache", "remove", "faster_whisper"],
                    capture_output=True, timeout=30
                )
                _sp.run(
                    [sys.executable, "-m", "pip", "cache", "remove", "ctranslate2"],
                    capture_output=True, timeout=30
                )
            except Exception:
                pass
            
            if _is_cancelled(job_id):
                return
            
            # Step 4: Install fresh
            _update_job(job_id, progress=40, message=f"Installing {backend} fresh...")
            logger.info(f"Reinstall: Installing {backend}")
            
            if backend == "faster-whisper":
                # Install with specific options for CPU mode
                install_cmd = [
                    sys.executable, "-m", "pip", "install",
                    "faster-whisper", "--force-reinstall", "--no-cache-dir"
                ]
                
                # For CPU mode, also install CPU-only ctranslate2
                if cpu_mode:
                    _update_job(job_id, progress=45, message="Installing CPU-optimized version...")
                    # First install ctranslate2 without CUDA
                    try:
                        _sp.run(
                            [sys.executable, "-m", "pip", "install",
                             "ctranslate2", "--force-reinstall", "--no-cache-dir"],
                            capture_output=True, timeout=300
                        )
                    except Exception:
                        pass
            else:
                install_cmd = [
                    sys.executable, "-m", "pip", "install",
                    backend, "--force-reinstall", "--no-cache-dir"
                ]
            
            proc = _sp.Popen(install_cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
            
            for line in proc.stdout:
                if _is_cancelled(job_id):
                    proc.kill()
                    return
                line = line.strip().lower()
                if "downloading" in line:
                    _update_job(job_id, progress=55, message="Downloading packages...")
                elif "installing" in line:
                    _update_job(job_id, progress=70, message="Installing packages...")
            
            proc.wait()
            
            if proc.returncode != 0:
                _update_job(
                    job_id, status="error",
                    error="pip install failed",
                    message="Installation failed. Try running: pip install faster-whisper --force-reinstall"
                )
                return
            
            # Step 5: Save CPU mode setting
            _update_job(job_id, progress=85, message="Saving settings...")
            settings = _load_whisper_settings()
            settings["cpu_mode"] = cpu_mode
            _save_whisper_settings(settings)
            
            # Step 6: Verify
            _update_job(job_id, progress=90, message="Verifying installation...")
            
            if backend == "faster-whisper":
                verify_cmd = [sys.executable, "-c", "from faster_whisper import WhisperModel; print('ok')"]
            else:
                verify_cmd = [sys.executable, "-c", "import whisper; print('ok')"]
            
            verify = _sp.run(verify_cmd, capture_output=True, text=True, timeout=30)
            
            if verify.returncode == 0 and "ok" in verify.stdout:
                mode_str = " (CPU mode)" if cpu_mode else ""
                _update_job(
                    job_id, status="complete", progress=100,
                    message=f"Whisper reinstalled successfully!{mode_str}",
                    result={"backend": backend, "cpu_mode": cpu_mode, "installed": True}
                )
                logger.info(f"Whisper reinstalled: {backend}, cpu_mode={cpu_mode}")
            else:
                _update_job(
                    job_id, status="error",
                    error=f"Verification failed: {verify.stderr[:200]}",
                    message="Installation completed but import failed"
                )
        
        except Exception as e:
            logger.exception("Whisper reinstall error")
            _update_job(
                job_id, status="error",
                error=str(e),
                message=f"Reinstall failed: {e}"
            )
    
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
# Video Effects (FFmpeg-based, always available)
# ---------------------------------------------------------------------------
@app.route("/video/fx/list", methods=["GET"])
def video_fx_list():
    """Return available FFmpeg-based video effects."""
    try:
        try:
            from .core.video_fx import get_available_video_effects
        except ImportError:
            from opencut.core.video_fx import get_available_video_effects
        return jsonify({"effects": get_available_video_effects()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/fx/apply", methods=["POST"])
def video_fx_apply():
    """Apply a video effect."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "").strip()
    params = data.get("params", {})

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    if not effect:
        return jsonify({"error": "No effect specified"}), 400

    job_id = _new_job("video-fx", filepath)

    def _process():
        try:
            try:
                from .core import video_fx
            except ImportError:
                from opencut.core import video_fx

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)

            if effect == "stabilize":
                out = video_fx.stabilize_video(
                    filepath, output_dir=effective_dir,
                    smoothing=params.get("smoothing", 10),
                    crop=params.get("crop", "keep"),
                    zoom=params.get("zoom", 0),
                    on_progress=_on_progress,
                )
            elif effect == "chromakey":
                out = video_fx.chromakey(
                    filepath, output_dir=effective_dir,
                    color=params.get("color", "0x00FF00"),
                    similarity=params.get("similarity", 0.3),
                    blend=params.get("blend", 0.1),
                    background=params.get("background", ""),
                    on_progress=_on_progress,
                )
            elif effect == "lut":
                lut_path = params.get("lut_path", "")
                if not lut_path:
                    raise ValueError("No LUT file path provided")
                out = video_fx.apply_lut(
                    filepath, lut_path, output_dir=effective_dir,
                    intensity=params.get("intensity", 1.0),
                    on_progress=_on_progress,
                )
            elif effect == "vignette":
                out = video_fx.apply_vignette(
                    filepath, output_dir=effective_dir,
                    intensity=params.get("intensity", 0.5),
                    on_progress=_on_progress,
                )
            elif effect == "film_grain":
                out = video_fx.apply_film_grain(
                    filepath, output_dir=effective_dir,
                    intensity=params.get("intensity", 0.5),
                    on_progress=_on_progress,
                )
            elif effect == "letterbox":
                out = video_fx.apply_letterbox(
                    filepath, output_dir=effective_dir,
                    aspect=params.get("aspect", "2.39:1"),
                    on_progress=_on_progress,
                )
            elif effect == "color_match":
                out = video_fx.color_match(
                    filepath, output_dir=effective_dir,
                    reference_path=params.get("reference_path", ""),
                    on_progress=_on_progress,
                )
            else:
                raise ValueError(f"Unknown video effect: {effect}")

            _update_job(
                job_id, status="complete", progress=100,
                message=f"Effect '{effect}' applied!",
                result={"output_path": out, "effect": effect},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Video FX error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Video AI (upscale, bg removal, interpolation, denoise)
# ---------------------------------------------------------------------------
@app.route("/video/ai/capabilities", methods=["GET"])
def video_ai_capabilities():
    """Return available AI video capabilities."""
    try:
        try:
            from .core.video_ai import get_ai_capabilities
        except ImportError:
            from opencut.core.video_ai import get_ai_capabilities
        return jsonify(get_ai_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/ai/upscale", methods=["POST"])
def video_ai_upscale():
    """AI upscale video using Real-ESRGAN."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    scale = data.get("scale", 2)
    model = data.get("model", "realesrgan-x4plus")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("ai-upscale", filepath)

    def _process():
        try:
            try:
                from .core.video_ai import upscale_video
            except ImportError:
                from opencut.core.video_ai import upscale_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = upscale_video(
                filepath, output_dir=effective_dir,
                scale=scale, model=model,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Upscaled {scale}x!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI upscale error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/ai/rembg", methods=["POST"])
def video_ai_rembg():
    """AI background removal using rembg."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    model = data.get("model", "u2net")
    bg_color = data.get("bg_color", "")
    alpha_only = data.get("alpha_only", False)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("ai-rembg", filepath)

    def _process():
        try:
            try:
                from .core.video_ai import remove_background
            except ImportError:
                from opencut.core.video_ai import remove_background

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = remove_background(
                filepath, output_dir=effective_dir,
                model=model, bg_color=bg_color,
                alpha_only=alpha_only,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Background removed!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI rembg error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/ai/interpolate", methods=["POST"])
def video_ai_interpolate():
    """AI frame interpolation."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    multiplier = data.get("multiplier", 2)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("ai-interpolate", filepath)

    def _process():
        try:
            try:
                from .core.video_ai import frame_interpolate
            except ImportError:
                from opencut.core.video_ai import frame_interpolate

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = frame_interpolate(
                filepath, output_dir=effective_dir,
                multiplier=multiplier,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Interpolated {multiplier}x!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI interpolation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/ai/denoise", methods=["POST"])
def video_ai_denoise():
    """AI video noise reduction."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    method = data.get("method", "nlmeans")
    strength = data.get("strength", 0.5)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("video-denoise", filepath)

    def _process():
        try:
            try:
                from .core.video_ai import video_denoise
            except ImportError:
                from opencut.core.video_ai import video_denoise

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = video_denoise(
                filepath, output_dir=effective_dir,
                method=method, strength=strength,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Video denoised!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Video denoise error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/ai/install", methods=["POST"])
def video_ai_install():
    """Install AI video dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "upscale")

    packages = {
        "upscale": ["realesrgan", "basicsr", "opencv-python-headless"],
        "rembg": ["rembg[gpu]", "opencv-python-headless"],
        "rembg_cpu": ["rembg", "opencv-python-headless"],
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
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg,
                     "--break-system-packages", "-q"],
                    capture_output=True, timeout=600,
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace")
                    raise RuntimeError(f"pip install {pkg} failed: {err[-200:]}")
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Installed {component}!",
                result={"component": component},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("AI install error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Pro (Pedalboard studio effects, DeepFilterNet)
# ---------------------------------------------------------------------------
@app.route("/audio/pro/effects", methods=["GET"])
def audio_pro_effects():
    """Return available Pedalboard effects."""
    try:
        try:
            from .core.audio_pro import get_pedalboard_effects, check_pedalboard_available
        except ImportError:
            from opencut.core.audio_pro import get_pedalboard_effects, check_pedalboard_available
        return jsonify({
            "effects": get_pedalboard_effects(),
            "available": check_pedalboard_available(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/pro/apply", methods=["POST"])
def audio_pro_apply():
    """Apply a Pedalboard studio effect."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    effect = data.get("effect", "").strip()
    params = data.get("params", {})

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400
    if not effect:
        return jsonify({"error": "No effect specified"}), 400

    job_id = _new_job("audio-pro", filepath)

    def _process():
        try:
            try:
                from .core.audio_pro import apply_pedalboard_effect
            except ImportError:
                from opencut.core.audio_pro import apply_pedalboard_effect

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = apply_pedalboard_effect(
                filepath, effect, output_dir=effective_dir,
                params=params, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Effect applied!",
                result={"output_path": out, "effect": effect},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio pro error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/pro/deepfilter", methods=["POST"])
def audio_pro_deepfilter():
    """AI noise reduction using DeepFilterNet."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("deepfilter", filepath)

    def _process():
        try:
            try:
                from .core.audio_pro import deepfilter_denoise
            except ImportError:
                from opencut.core.audio_pro import deepfilter_denoise

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = deepfilter_denoise(
                filepath, output_dir=effective_dir,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="AI noise reduction complete!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("DeepFilterNet error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/pro/install", methods=["POST"])
def audio_pro_install():
    """Install audio pro dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "pedalboard")

    packages = {
        "pedalboard": ["pedalboard"],
        "deepfilter": ["deepfilternet"],
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
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg,
                     "--break-system-packages", "-q"],
                    capture_output=True, timeout=600,
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace")
                    raise RuntimeError(f"pip install {pkg} failed: {err[-200:]}")
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Installed {component}!",
                result={"component": component},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Audio pro install error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Face Tools (detection, blur, auto-censor)
# ---------------------------------------------------------------------------
@app.route("/video/face/capabilities", methods=["GET"])
def face_capabilities():
    """Return face tool capabilities."""
    try:
        try:
            from .core.face_tools import check_face_tools_available
        except ImportError:
            from opencut.core.face_tools import check_face_tools_available
        return jsonify(check_face_tools_available())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/face/detect", methods=["POST"])
def face_detect():
    """Detect faces in image or first frame of video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    detector = data.get("detector", "mediapipe")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 400

    try:
        try:
            from .core.face_tools import detect_faces_in_frame
        except ImportError:
            from opencut.core.face_tools import detect_faces_in_frame
        result = detect_faces_in_frame(filepath, detector=detector)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/face/blur", methods=["POST"])
def face_blur():
    """Auto-detect and blur faces in video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    method = data.get("method", "gaussian")
    strength = data.get("strength", 51)
    detector = data.get("detector", "mediapipe")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("face-blur", filepath)

    def _process():
        try:
            try:
                from .core.face_tools import blur_faces
            except ImportError:
                from opencut.core.face_tools import blur_faces

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = blur_faces(
                filepath, output_dir=effective_dir,
                method=method, strength=strength,
                detector=detector, on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Faces blurred!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Face blur error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/face/install", methods=["POST"])
def face_install():
    """Install MediaPipe for face detection."""
    job_id = _new_job("install", "mediapipe")

    def _process():
        try:
            _update_job(job_id, progress=20, message="Installing MediaPipe...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "mediapipe",
                 "opencv-python-headless", "--break-system-packages", "-q"],
                capture_output=True, timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace")[-300:])
            _update_job(
                job_id, status="complete", progress=100,
                message="MediaPipe installed!",
                result={"component": "mediapipe"},
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
# Style Transfer
# ---------------------------------------------------------------------------
@app.route("/video/style/list", methods=["GET"])
def style_list():
    """Return available style transfer models."""
    try:
        try:
            from .core.style_transfer import get_available_styles
        except ImportError:
            from opencut.core.style_transfer import get_available_styles
        return jsonify({"styles": get_available_styles()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/style/apply", methods=["POST"])
def style_apply():
    """Apply neural style transfer to video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    style_name = data.get("style", "candy")
    intensity = data.get("intensity", 1.0)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("style-transfer", filepath)

    def _process():
        try:
            try:
                from .core.style_transfer import style_transfer_video
            except ImportError:
                from opencut.core.style_transfer import style_transfer_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = style_transfer_video(
                filepath, style_name=style_name,
                output_dir=effective_dir,
                intensity=intensity,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Style '{style_name}' applied!",
                result={"output_path": out, "style": style_name},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Style transfer error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Enhanced Captions (WhisperX, Translation, Karaoke)
# ---------------------------------------------------------------------------
@app.route("/captions/enhanced/capabilities", methods=["GET"])
def captions_enhanced_capabilities():
    """Return enhanced caption capabilities."""
    try:
        try:
            from .core.captions_enhanced import (
                check_whisperx_available, check_nllb_available, check_pysubs2_available,
                TRANSLATION_LANGUAGES,
            )
        except ImportError:
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
        return jsonify({"error": str(e)}), 500


@app.route("/captions/whisperx", methods=["POST"])
def captions_whisperx():
    """Transcribe with WhisperX for word-level timestamps."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    model_size = data.get("model", "base")
    language = data.get("language", "")
    diarize = data.get("diarize", False)
    hf_token = data.get("hf_token", "")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("whisperx", filepath)

    def _process():
        try:
            try:
                from .core.captions_enhanced import whisperx_transcribe
            except ImportError:
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


@app.route("/captions/translate", methods=["POST"])
def captions_translate():
    """Translate caption segments using NLLB."""
    data = request.get_json(force=True)
    segments = data.get("segments", [])
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "es")
    output_dir = data.get("output_dir", "")
    output_format = data.get("format", "srt")
    filepath = data.get("filepath", "")

    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    job_id = _new_job("translate", filepath or "captions")

    def _process():
        try:
            try:
                from .core.captions_enhanced import translate_segments
            except ImportError:
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


@app.route("/captions/karaoke", methods=["POST"])
def captions_karaoke():
    """Export segments as ASS karaoke subtitles."""
    data = request.get_json(force=True)
    segments = data.get("segments", [])
    filepath = data.get("filepath", "")
    output_dir = data.get("output_dir", "")

    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    job_id = _new_job("karaoke", filepath or "captions")

    def _process():
        try:
            try:
                from .core.captions_enhanced import segments_to_ass_karaoke
            except ImportError:
                from opencut.core.captions_enhanced import segments_to_ass_karaoke

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir) if filepath else tempfile.gettempdir()
            base = os.path.splitext(os.path.basename(filepath))[0] if filepath else "captions"
            out_path = os.path.join(effective_dir, f"{base}_karaoke.ass")

            segments_to_ass_karaoke(
                segments, out_path,
                font_name=data.get("font", "Arial"),
                font_size=data.get("font_size", 48),
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


@app.route("/captions/convert", methods=["POST"])
def captions_convert():
    """Convert subtitle file between formats."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    target_format = data.get("format", "srt")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 400

    try:
        try:
            from .core.captions_enhanced import convert_subtitle_format
        except ImportError:
            from opencut.core.captions_enhanced import convert_subtitle_format
        out = convert_subtitle_format(filepath, output_format=target_format)
        return jsonify({"output_path": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/captions/enhanced/install", methods=["POST"])
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
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg,
                     "--break-system-packages", "-q"],
                    capture_output=True, timeout=600,
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace")
                    raise RuntimeError(f"pip install {pkg} failed: {err[-200:]}")
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
# Export Presets
# ---------------------------------------------------------------------------
@app.route("/export/presets", methods=["GET"])
def export_presets_list():
    """Return available export presets."""
    try:
        try:
            from .core.export_presets import get_export_presets, get_preset_categories
        except ImportError:
            from opencut.core.export_presets import get_export_presets, get_preset_categories
        return jsonify({
            "presets": get_export_presets(),
            "categories": get_preset_categories(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export/preset", methods=["POST"])
def export_with_preset_route():
    """Export video using a preset profile."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    preset_name = data.get("preset", "youtube_1080p")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("export-preset", filepath)

    def _process():
        try:
            try:
                from .core.export_presets import export_with_preset
            except ImportError:
                from opencut.core.export_presets import export_with_preset

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = export_with_preset(
                filepath, preset_name,
                output_dir=effective_dir,
                on_progress=_on_progress,
            )
            size_mb = os.path.getsize(out) / (1024 * 1024)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Exported ({preset_name}, {size_mb:.1f}MB)",
                result={"output_path": out, "preset": preset_name, "size_mb": round(size_mb, 1)},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Export preset error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Auto-Thumbnail Generation
# ---------------------------------------------------------------------------
@app.route("/export/thumbnails", methods=["POST"])
def generate_thumbnails_route():
    """Generate thumbnail candidates from video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")
    count = data.get("count", 5)
    width = data.get("width", 1920)
    use_faces = data.get("use_faces", True)

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File not found: {filepath}"}), 400

    job_id = _new_job("thumbnails", filepath)

    def _process():
        try:
            try:
                from .core.thumbnail import generate_thumbnails
            except ImportError:
                from opencut.core.thumbnail import generate_thumbnails

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            results = generate_thumbnails(
                filepath, output_dir=effective_dir,
                count=count, width=width,
                use_faces=use_faces,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Generated {len(results)} thumbnails",
                result={"thumbnails": results},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Thumbnail generation error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------
@app.route("/batch/create", methods=["POST"])
def batch_create():
    """Create a batch processing job."""
    data = request.get_json(force=True)
    operation = data.get("operation", "")
    filepaths = data.get("filepaths", [])
    params = data.get("params", {})

    if not operation:
        return jsonify({"error": "No operation specified"}), 400
    if not filepaths:
        return jsonify({"error": "No files specified"}), 400

    try:
        try:
            from .core.batch_process import create_batch, update_batch_item, finalize_batch
        except ImportError:
            from opencut.core.batch_process import create_batch, update_batch_item, finalize_batch

        batch_id = str(uuid.uuid4())[:8]
        batch = create_batch(batch_id, operation, filepaths, params)

        # Process items sequentially in a background thread
        def _process_batch():
            for idx, item in enumerate(batch.items):
                if batch.status == "cancelled":
                    break
                if item.status == "skipped":
                    continue

                update_batch_item(batch_id, idx,
                                  status="running", started_at=time.time(),
                                  message="Processing...")

                try:
                    # Route the operation to the appropriate function
                    result = _execute_batch_item(
                        operation, item.filepath, params,
                        lambda pct, msg: update_batch_item(
                            batch_id, idx, progress=pct, message=msg
                        ),
                    )
                    update_batch_item(batch_id, idx,
                                      status="complete", progress=100,
                                      output_path=result,
                                      message="Done",
                                      finished_at=time.time())
                except Exception as e:
                    update_batch_item(batch_id, idx,
                                      status="error", error=str(e),
                                      message=f"Error: {e}",
                                      finished_at=time.time())

            finalize_batch(batch_id)

        thread = threading.Thread(target=_process_batch, daemon=True)
        thread.start()

        return jsonify({"batch_id": batch_id, "status": "running", "total": batch.total})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/<batch_id>", methods=["GET"])
def batch_status(batch_id):
    """Get batch job status."""
    try:
        try:
            from .core.batch_process import get_batch
        except ImportError:
            from opencut.core.batch_process import get_batch
        batch = get_batch(batch_id)
        if not batch:
            return jsonify({"error": "Batch not found"}), 404
        return jsonify(batch.summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/<batch_id>/cancel", methods=["POST"])
def batch_cancel(batch_id):
    """Cancel a running batch."""
    try:
        try:
            from .core.batch_process import cancel_batch
        except ImportError:
            from opencut.core.batch_process import cancel_batch
        if cancel_batch(batch_id):
            return jsonify({"status": "cancelled"})
        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch/list", methods=["GET"])
def batch_list():
    """List recent batch jobs."""
    try:
        try:
            from .core.batch_process import list_batches
        except ImportError:
            from opencut.core.batch_process import list_batches
        return jsonify({"batches": list_batches()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _execute_batch_item(operation, filepath, params, on_progress):
    """Execute a single batch item based on operation type."""
    output_dir = params.get("output_dir", os.path.dirname(filepath))

    if operation == "denoise":
        try:
            from .core.audio_suite import denoise_audio
        except ImportError:
            from opencut.core.audio_suite import denoise_audio
        return denoise_audio(
            filepath, output_dir=output_dir,
            strength=params.get("strength", 0.5),
            on_progress=on_progress,
        )
    elif operation == "normalize":
        try:
            from .core.audio_suite import normalize_audio
        except ImportError:
            from opencut.core.audio_suite import normalize_audio
        return normalize_audio(
            filepath, output_dir=output_dir,
            target_lufs=params.get("target_lufs", -16),
            on_progress=on_progress,
        )
    elif operation == "stabilize":
        try:
            from .core.video_fx import stabilize_video
        except ImportError:
            from opencut.core.video_fx import stabilize_video
        return stabilize_video(
            filepath, output_dir=output_dir,
            smoothing=params.get("smoothing", 10),
            on_progress=on_progress,
        )
    elif operation == "face_blur":
        try:
            from .core.face_tools import blur_faces
        except ImportError:
            from opencut.core.face_tools import blur_faces
        return blur_faces(
            filepath, output_dir=output_dir,
            method=params.get("method", "gaussian"),
            strength=params.get("strength", 51),
            on_progress=on_progress,
        )
    elif operation == "export_preset":
        try:
            from .core.export_presets import export_with_preset
        except ImportError:
            from opencut.core.export_presets import export_with_preset
        return export_with_preset(
            filepath, params.get("preset", "youtube_1080p"),
            output_dir=output_dir,
            on_progress=on_progress,
        )
    elif operation == "vignette":
        try:
            from .core.video_fx import apply_vignette
        except ImportError:
            from opencut.core.video_fx import apply_vignette
        return apply_vignette(
            filepath, output_dir=output_dir,
            intensity=params.get("intensity", 0.5),
            on_progress=on_progress,
        )
    elif operation == "film_grain":
        try:
            from .core.video_fx import apply_film_grain
        except ImportError:
            from opencut.core.video_fx import apply_film_grain
        return apply_film_grain(
            filepath, output_dir=output_dir,
            intensity=params.get("intensity", 0.5),
            on_progress=on_progress,
        )
    elif operation == "letterbox":
        try:
            from .core.video_fx import apply_letterbox
        except ImportError:
            from opencut.core.video_fx import apply_letterbox
        return apply_letterbox(
            filepath, output_dir=output_dir,
            aspect=params.get("aspect", "2.39:1"),
            on_progress=on_progress,
        )
    elif operation == "style_transfer":
        try:
            from .core.style_transfer import style_transfer_video
        except ImportError:
            from opencut.core.style_transfer import style_transfer_video
        return style_transfer_video(
            filepath, style_name=params.get("style", "candy"),
            output_dir=output_dir,
            intensity=params.get("intensity", 1.0),
            on_progress=on_progress,
        )
    else:
        raise ValueError(f"Unsupported batch operation: {operation}")


# ---------------------------------------------------------------------------
# Voice Generation (TTS)
# ---------------------------------------------------------------------------
@app.route("/audio/tts/voices", methods=["GET"])
def tts_voices():
    """Return available TTS voices."""
    try:
        try:
            from .core.voice_gen import get_voice_list, KOKORO_VOICES, check_edge_tts_available, check_kokoro_available
        except ImportError:
            from opencut.core.voice_gen import get_voice_list, KOKORO_VOICES, check_edge_tts_available, check_kokoro_available
        return jsonify({
            "edge_tts": {
                "available": check_edge_tts_available(),
                "voices": get_voice_list("edge"),
            },
            "kokoro": {
                "available": check_kokoro_available(),
                "voices": KOKORO_VOICES,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/tts/generate", methods=["POST"])
def tts_generate():
    """Generate speech from text."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    engine = data.get("engine", "edge")
    voice = data.get("voice", "en-US-AriaNeural")
    rate = data.get("rate", "+0%")
    pitch = data.get("pitch", "+0Hz")
    speed = data.get("speed", 1.0)
    output_dir = data.get("output_dir", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    job_id = _new_job("tts", text[:50])

    def _process():
        try:
            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()

            if engine == "kokoro":
                try:
                    from .core.voice_gen import kokoro_generate
                except ImportError:
                    from opencut.core.voice_gen import kokoro_generate
                out = kokoro_generate(
                    text, voice=voice, output_dir=effective_dir,
                    speed=speed, on_progress=_on_progress,
                )
            else:
                try:
                    from .core.voice_gen import edge_tts_generate
                except ImportError:
                    from opencut.core.voice_gen import edge_tts_generate
                out = edge_tts_generate(
                    text, voice=voice, output_dir=effective_dir,
                    rate=rate, pitch=pitch, on_progress=_on_progress,
                )

            _update_job(
                job_id, status="complete", progress=100,
                message="Speech generated!",
                result={"output_path": out, "engine": engine, "voice": voice},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("TTS error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/tts/subtitled", methods=["POST"])
def tts_subtitled():
    """Generate speech + word-level subtitles."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    voice = data.get("voice", "en-US-AriaNeural")
    output_dir = data.get("output_dir", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    job_id = _new_job("tts-sub", text[:50])

    def _process():
        try:
            try:
                from .core.voice_gen import edge_tts_with_subtitles
            except ImportError:
                from opencut.core.voice_gen import edge_tts_with_subtitles

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            result = edge_tts_with_subtitles(
                text, voice=voice, output_dir=effective_dir,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Speech + subtitles generated!",
                result=result,
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("TTS subtitled error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/tts/install", methods=["POST"])
def tts_install():
    """Install TTS engine dependencies."""
    data = request.get_json(force=True)
    component = data.get("component", "edge_tts")

    packages = {
        "edge_tts": ["edge-tts"],
        "kokoro": ["kokoro>=0.3", "soundfile"],
    }
    pkgs = packages.get(component, [])
    if not pkgs:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    job_id = _new_job("install", component)

    def _process():
        try:
            for pkg in pkgs:
                _update_job(job_id, progress=30, message=f"Installing {pkg}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg,
                     "--break-system-packages", "-q"],
                    capture_output=True, timeout=600,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode(errors="replace")[-200:])
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
# Music & SFX Generation
# ---------------------------------------------------------------------------
@app.route("/audio/gen/capabilities", methods=["GET"])
def audio_gen_capabilities():
    """Return available audio generation capabilities."""
    try:
        try:
            from .core.music_gen import get_audio_generators
        except ImportError:
            from opencut.core.music_gen import get_audio_generators
        return jsonify(get_audio_generators())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/gen/tone", methods=["POST"])
def audio_gen_tone():
    """Generate a tone."""
    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "")

    job_id = _new_job("tone", f"{data.get('frequency', 440)}Hz")

    def _process():
        try:
            try:
                from .core.music_gen import generate_tone
            except ImportError:
                from opencut.core.music_gen import generate_tone

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            out = generate_tone(
                output_dir=effective_dir,
                frequency=data.get("frequency", 440),
                duration=data.get("duration", 3.0),
                waveform=data.get("waveform", "sine"),
                volume=data.get("volume", 0.5),
                fade_in=data.get("fade_in", 0.1),
                fade_out=data.get("fade_out", 0.5),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Tone generated!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/gen/sfx", methods=["POST"])
def audio_gen_sfx():
    """Generate a synthesized sound effect."""
    data = request.get_json(force=True)
    preset = data.get("preset", "swoosh")
    output_dir = data.get("output_dir", "")

    job_id = _new_job("sfx", preset)

    def _process():
        try:
            try:
                from .core.music_gen import generate_sfx
            except ImportError:
                from opencut.core.music_gen import generate_sfx

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            out = generate_sfx(
                preset=preset,
                output_dir=effective_dir,
                duration=data.get("duration", 1.0),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"SFX '{preset}' generated!",
                result={"output_path": out, "preset": preset},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/gen/silence", methods=["POST"])
def audio_gen_silence():
    """Generate silence/padding audio."""
    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "")
    duration = data.get("duration", 1.0)

    try:
        try:
            from .core.music_gen import generate_silence
        except ImportError:
            from opencut.core.music_gen import generate_silence
        out = generate_silence(duration=duration, output_dir=output_dir or tempfile.gettempdir())
        return jsonify({"output_path": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Caption Burn-in
# ---------------------------------------------------------------------------
@app.route("/captions/burnin/styles", methods=["GET"])
def burnin_styles():
    """Return available burn-in styles."""
    try:
        try:
            from .core.caption_burnin import get_burnin_styles
        except ImportError:
            from opencut.core.caption_burnin import get_burnin_styles
        return jsonify({"styles": get_burnin_styles()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/captions/burnin/file", methods=["POST"])
def burnin_from_file():
    """Burn a subtitle file into video."""
    data = request.get_json(force=True)
    video_path = data.get("filepath", "").strip()
    subtitle_path = data.get("subtitle_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Video file not found"}), 400
    if not subtitle_path or not os.path.isfile(subtitle_path):
        return jsonify({"error": "Subtitle file not found"}), 400

    job_id = _new_job("burnin", video_path)

    def _process():
        try:
            try:
                from .core.caption_burnin import burnin_subtitles
            except ImportError:
                from opencut.core.caption_burnin import burnin_subtitles

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(video_path, output_dir)
            out = burnin_subtitles(
                video_path, subtitle_path,
                output_dir=effective_dir,
                font_size=data.get("font_size", 0),
                margin_bottom=data.get("margin_bottom", 0),
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


@app.route("/captions/burnin/segments", methods=["POST"])
def burnin_from_segments():
    """Burn caption segments directly into video."""
    data = request.get_json(force=True)
    video_path = data.get("filepath", "").strip()
    segments = data.get("segments", [])
    style = data.get("style", "default")
    output_dir = data.get("output_dir", "")

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Video file not found"}), 400
    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    job_id = _new_job("burnin-seg", video_path)

    def _process():
        try:
            try:
                from .core.caption_burnin import burnin_segments
            except ImportError:
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
# Speed Ramp
# ---------------------------------------------------------------------------
@app.route("/video/speed/presets", methods=["GET"])
def speed_ramp_presets():
    """Return available speed ramp presets."""
    try:
        try:
            from .core.speed_ramp import get_speed_ramp_presets, EASING_FUNCTIONS
        except ImportError:
            from opencut.core.speed_ramp import get_speed_ramp_presets, EASING_FUNCTIONS
        return jsonify({
            "presets": get_speed_ramp_presets(),
            "easings": list(EASING_FUNCTIONS.keys()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/speed/change", methods=["POST"])
def speed_change_route():
    """Apply constant speed change."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    speed = data.get("speed", 2.0)
    output_dir = data.get("output_dir", "")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 400

    job_id = _new_job("speed", filepath)

    def _process():
        try:
            try:
                from .core.speed_ramp import change_speed
            except ImportError:
                from opencut.core.speed_ramp import change_speed

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = change_speed(
                filepath, speed=speed, output_dir=effective_dir,
                maintain_pitch=data.get("maintain_pitch", False),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Speed changed to {speed}x",
                result={"output_path": out, "speed": speed},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Speed change error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/speed/reverse", methods=["POST"])
def speed_reverse_route():
    """Reverse video playback."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    output_dir = data.get("output_dir", "")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 400

    job_id = _new_job("reverse", filepath)

    def _process():
        try:
            try:
                from .core.speed_ramp import reverse_video
            except ImportError:
                from opencut.core.speed_ramp import reverse_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = reverse_video(
                filepath, output_dir=effective_dir,
                reverse_audio=data.get("reverse_audio", True),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Video reversed!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/speed/ramp", methods=["POST"])
def speed_ramp_route():
    """Apply keyframe-based speed ramp or preset."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    preset = data.get("preset", "")
    keyframes = data.get("keyframes", [])
    output_dir = data.get("output_dir", "")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 400

    job_id = _new_job("speed-ramp", filepath)

    def _process():
        try:
            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)

            if preset:
                try:
                    from .core.speed_ramp import apply_speed_ramp_preset
                except ImportError:
                    from opencut.core.speed_ramp import apply_speed_ramp_preset
                out = apply_speed_ramp_preset(
                    filepath, preset, output_dir=effective_dir,
                    on_progress=_on_progress,
                )
            else:
                try:
                    from .core.speed_ramp import speed_ramp
                except ImportError:
                    from opencut.core.speed_ramp import speed_ramp
                out = speed_ramp(
                    filepath, keyframes, output_dir=effective_dir,
                    easing=data.get("easing", "ease_in_out"),
                    on_progress=_on_progress,
                )

            _update_job(
                job_id, status="complete", progress=100,
                message="Speed ramp applied!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Speed ramp error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Audio Ducking
# ---------------------------------------------------------------------------
@app.route("/audio/duck", methods=["POST"])
def audio_duck_route():
    """Duck music under voice using sidechain compression."""
    data = request.get_json(force=True)
    music_path = data.get("music_path", "").strip()
    voice_path = data.get("voice_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not music_path or not os.path.isfile(music_path):
        return jsonify({"error": "Music file not found"}), 400
    if not voice_path or not os.path.isfile(voice_path):
        return jsonify({"error": "Voice file not found"}), 400

    job_id = _new_job("duck", music_path)

    def _process():
        try:
            try:
                from .core.audio_duck import sidechain_duck
            except ImportError:
                from opencut.core.audio_duck import sidechain_duck

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(music_path, output_dir)
            out = sidechain_duck(
                music_path, voice_path, output_dir=effective_dir,
                duck_amount=data.get("duck_amount", 0.7),
                threshold=data.get("threshold", 0.015),
                attack=data.get("attack", 20),
                release=data.get("release", 300),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Audio ducking applied!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/mix-duck", methods=["POST"])
def audio_mix_duck_route():
    """Mix voice + music with auto-ducking."""
    data = request.get_json(force=True)
    voice_path = data.get("voice_path", "").strip()
    music_path = data.get("music_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not voice_path or not os.path.isfile(voice_path):
        return jsonify({"error": "Voice file not found"}), 400
    if not music_path or not os.path.isfile(music_path):
        return jsonify({"error": "Music file not found"}), 400

    job_id = _new_job("mix-duck", voice_path)

    def _process():
        try:
            try:
                from .core.audio_duck import mix_with_duck
            except ImportError:
                from opencut.core.audio_duck import mix_with_duck

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(voice_path, output_dir)
            out = mix_with_duck(
                voice_path, music_path, output_dir=effective_dir,
                music_volume=data.get("music_volume", 0.3),
                duck_amount=data.get("duck_amount", 0.6),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Audio mixed with ducking!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/duck-video", methods=["POST"])
def audio_duck_video_route():
    """Add music to video with auto-ducking under speech."""
    data = request.get_json(force=True)
    video_path = data.get("filepath", "").strip()
    music_path = data.get("music_path", "").strip()
    output_dir = data.get("output_dir", "")

    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "Video file not found"}), 400
    if not music_path or not os.path.isfile(music_path):
        return jsonify({"error": "Music file not found"}), 400

    job_id = _new_job("duck-video", video_path)

    def _process():
        try:
            try:
                from .core.audio_duck import auto_duck_video
            except ImportError:
                from opencut.core.audio_duck import auto_duck_video

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(video_path, output_dir)
            out = auto_duck_video(
                video_path, music_path, output_dir=effective_dir,
                music_volume=data.get("music_volume", 0.25),
                duck_amount=data.get("duck_amount", 0.7),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message="Music added with ducking!",
                result={"output_path": out},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("Duck video error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/mix", methods=["POST"])
def audio_mix_route():
    """Mix multiple audio tracks."""
    data = request.get_json(force=True)
    tracks = data.get("tracks", [])
    output_dir = data.get("output_dir", "")

    if not tracks:
        return jsonify({"error": "No tracks to mix"}), 400

    job_id = _new_job("mix", f"{len(tracks)} tracks")

    def _process():
        try:
            try:
                from .core.audio_duck import mix_audio_tracks
            except ImportError:
                from opencut.core.audio_duck import mix_audio_tracks

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = output_dir or tempfile.gettempdir()
            out = mix_audio_tracks(
                tracks, output_dir=effective_dir,
                duration_mode=data.get("duration_mode", "longest"),
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Mixed {len(tracks)} tracks!",
                result={"output_path": out},
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
# LUT Library
# ---------------------------------------------------------------------------
@app.route("/video/lut/list", methods=["GET"])
def lut_list():
    """Return available LUTs."""
    try:
        try:
            from .core.lut_library import get_lut_list
        except ImportError:
            from opencut.core.lut_library import get_lut_list
        return jsonify({"luts": get_lut_list()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/lut/apply", methods=["POST"])
def lut_apply():
    """Apply a color LUT to video."""
    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()
    lut_name = data.get("lut", "teal_orange")
    intensity = data.get("intensity", 1.0)
    output_dir = data.get("output_dir", "")

    if not filepath or not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 400

    job_id = _new_job("lut", filepath)

    def _process():
        try:
            try:
                from .core.lut_library import apply_lut
            except ImportError:
                from opencut.core.lut_library import apply_lut

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            effective_dir = _resolve_output_dir(filepath, output_dir)
            out = apply_lut(
                filepath, lut_name, intensity=intensity,
                output_dir=effective_dir,
                on_progress=_on_progress,
            )
            _update_job(
                job_id, status="complete", progress=100,
                message=f"LUT applied: {lut_name}",
                result={"output_path": out, "lut": lut_name},
            )
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
            logger.exception("LUT apply error")

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/lut/generate-all", methods=["POST"])
def lut_generate_all():
    """Pre-generate all built-in LUT .cube files."""
    job_id = _new_job("lut-gen", "all")

    def _process():
        try:
            try:
                from .core.lut_library import generate_all_luts
            except ImportError:
                from opencut.core.lut_library import generate_all_luts

            def _on_progress(pct, msg):
                _update_job(job_id, progress=pct, message=msg)

            count = generate_all_luts(on_progress=_on_progress)
            _update_job(
                job_id, status="complete", progress=100,
                message=f"Generated {count} LUT files",
                result={"count": count},
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
# Chromakey & Compositing
# ---------------------------------------------------------------------------
@app.route("/video/chromakey", methods=["POST"])
def chromakey_route():
    """Chromakey (green/blue screen) removal + compositing."""
    data = request.get_json(force=True)
    fg = data.get("filepath", "").strip()
    bg = data.get("background", "").strip()
    output_dir = data.get("output_dir", "")
    if not fg or not os.path.isfile(fg):
        return jsonify({"error": "Foreground file not found"}), 400
    if not bg or not os.path.isfile(bg):
        return jsonify({"error": "Background file not found"}), 400
    job_id = _new_job("chromakey", fg)
    def _process():
        try:
            try:
                from .core.chromakey import chromakey_video
            except ImportError:
                from opencut.core.chromakey import chromakey_video
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fg, output_dir)
            out = chromakey_video(fg, bg, output_dir=d, color=data.get("color", "green"),
                                  tolerance=data.get("tolerance", 0.5),
                                  spill_suppress=data.get("spill_suppress", 0.5),
                                  edge_blur=data.get("edge_blur", 3), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, message="Chromakey complete!",
                        result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/pip", methods=["POST"])
def pip_route():
    """Picture-in-picture overlay."""
    data = request.get_json(force=True)
    main = data.get("filepath", "").strip()
    pip = data.get("pip_path", "").strip()
    if not main or not os.path.isfile(main): return jsonify({"error": "Main file not found"}), 400
    if not pip or not os.path.isfile(pip): return jsonify({"error": "PiP file not found"}), 400
    job_id = _new_job("pip", main)
    def _process():
        try:
            try:
                from .core.chromakey import picture_in_picture
            except ImportError:
                from opencut.core.chromakey import picture_in_picture
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(main, data.get("output_dir", ""))
            out = picture_in_picture(main, pip, output_dir=d,
                                      position=data.get("position", "bottom_right"),
                                      scale=data.get("scale", 0.25), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/blend", methods=["POST"])
def blend_route():
    """Blend two videos with a blend mode."""
    data = request.get_json(force=True)
    base = data.get("filepath", "").strip()
    overlay = data.get("overlay_path", "").strip()
    if not base or not os.path.isfile(base): return jsonify({"error": "Base file not found"}), 400
    if not overlay or not os.path.isfile(overlay): return jsonify({"error": "Overlay not found"}), 400
    job_id = _new_job("blend", base)
    def _process():
        try:
            try:
                from .core.chromakey import blend_videos
            except ImportError:
                from opencut.core.chromakey import blend_videos
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(base, data.get("output_dir", ""))
            out = blend_videos(base, overlay, output_dir=d,
                                mode=data.get("mode", "overlay"),
                                opacity=data.get("opacity", 0.5), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Animated Captions
# ---------------------------------------------------------------------------
@app.route("/captions/animated/presets", methods=["GET"])
def animated_caption_presets():
    try:
        try:
            from .core.animated_captions import get_animation_presets
        except ImportError:
            from opencut.core.animated_captions import get_animation_presets
        return jsonify({"presets": get_animation_presets()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/captions/animated/render", methods=["POST"])
def animated_caption_render():
    """Render animated word-by-word captions onto video."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    words = data.get("word_segments", [])
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    if not words: return jsonify({"error": "No word segments"}), 400
    job_id = _new_job("anim-cap", fp)
    def _process():
        try:
            try:
                from .core.animated_captions import render_animated_captions
            except ImportError:
                from opencut.core.animated_captions import render_animated_captions
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = render_animated_captions(fp, words, output_dir=d,
                                            animation=data.get("animation", "pop"),
                                            font_size=data.get("font_size", 56),
                                            max_words_per_line=data.get("max_words", 6),
                                            on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Object Removal
# ---------------------------------------------------------------------------
@app.route("/video/remove/capabilities", methods=["GET"])
def removal_capabilities():
    try:
        try:
            from .core.object_removal import get_removal_capabilities
        except ImportError:
            from opencut.core.object_removal import get_removal_capabilities
        return jsonify(get_removal_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/remove/watermark", methods=["POST"])
def remove_watermark_route():
    """Remove watermark using delogo or LaMA."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    region = data.get("region", {})
    method = data.get("method", "delogo")
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    if not region: return jsonify({"error": "No region specified"}), 400
    job_id = _new_job("watermark", fp)
    def _process():
        try:
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            if method == "lama":
                try:
                    from .core.object_removal import remove_watermark_lama
                except ImportError:
                    from opencut.core.object_removal import remove_watermark_lama
                out = remove_watermark_lama(fp, region, output_dir=d, on_progress=_p)
            else:
                try:
                    from .core.object_removal import remove_watermark_delogo
                except ImportError:
                    from opencut.core.object_removal import remove_watermark_delogo
                out = remove_watermark_delogo(fp, region, output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Motion Graphics / Titles
# ---------------------------------------------------------------------------
@app.route("/video/title/presets", methods=["GET"])
def title_presets():
    try:
        try:
            from .core.motion_graphics import get_title_presets
        except ImportError:
            from opencut.core.motion_graphics import get_title_presets
        return jsonify({"presets": get_title_presets()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/title/render", methods=["POST"])
def title_render():
    """Render a standalone title card video."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text: return jsonify({"error": "No text"}), 400
    job_id = _new_job("title", text[:40])
    def _process():
        try:
            try:
                from .core.motion_graphics import render_title_card
            except ImportError:
                from opencut.core.motion_graphics import render_title_card
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = data.get("output_dir", "") or tempfile.gettempdir()
            out = render_title_card(text, output_dir=d,
                                     preset=data.get("preset", "fade_center"),
                                     duration=data.get("duration", 5.0),
                                     font_size=data.get("font_size", 72),
                                     subtitle=data.get("subtitle", ""),
                                     on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/title/overlay", methods=["POST"])
def title_overlay():
    """Overlay animated title onto existing video."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    text = data.get("text", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    if not text: return jsonify({"error": "No text"}), 400
    job_id = _new_job("title-overlay", fp)
    def _process():
        try:
            try:
                from .core.motion_graphics import overlay_title
            except ImportError:
                from opencut.core.motion_graphics import overlay_title
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = overlay_title(fp, text, output_dir=d,
                                 preset=data.get("preset", "fade_center"),
                                 font_size=data.get("font_size", 72),
                                 start_time=data.get("start_time", 0),
                                 duration=data.get("duration", 5.0),
                                 subtitle=data.get("subtitle", ""),
                                 on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------
@app.route("/video/transitions/list", methods=["GET"])
def transitions_list():
    try:
        try:
            from .core.transitions_3d import get_transition_list
        except ImportError:
            from opencut.core.transitions_3d import get_transition_list
        return jsonify({"transitions": get_transition_list()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/transitions/apply", methods=["POST"])
def transitions_apply():
    """Apply transition between two clips."""
    data = request.get_json(force=True)
    a = data.get("clip_a", "").strip()
    b = data.get("clip_b", "").strip()
    if not a or not os.path.isfile(a): return jsonify({"error": "Clip A not found"}), 400
    if not b or not os.path.isfile(b): return jsonify({"error": "Clip B not found"}), 400
    job_id = _new_job("transition", a)
    def _process():
        try:
            try:
                from .core.transitions_3d import apply_transition
            except ImportError:
                from opencut.core.transitions_3d import apply_transition
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(a, data.get("output_dir", ""))
            out = apply_transition(a, b, output_dir=d,
                                    transition=data.get("transition", "fade"),
                                    duration=data.get("duration", 1.0),
                                    on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/transitions/join", methods=["POST"])
def transitions_join():
    """Join multiple clips with transitions."""
    data = request.get_json(force=True)
    clips = data.get("clips", [])
    if len(clips) < 2: return jsonify({"error": "Need 2+ clips"}), 400
    job_id = _new_job("join", f"{len(clips)} clips")
    def _process():
        try:
            try:
                from .core.transitions_3d import join_with_transitions
            except ImportError:
                from opencut.core.transitions_3d import join_with_transitions
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = data.get("output_dir", "") or os.path.dirname(clips[0])
            out = join_with_transitions(clips, output_dir=d,
                                         transition=data.get("transition", "fade"),
                                         duration=data.get("duration", 1.0),
                                         on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------
@app.route("/video/particles/presets", methods=["GET"])
def particle_presets():
    try:
        try:
            from .core.particles import get_particle_presets
        except ImportError:
            from opencut.core.particles import get_particle_presets
        return jsonify({"presets": get_particle_presets()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/particles/apply", methods=["POST"])
def particle_apply():
    """Overlay particle effects on video."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    job_id = _new_job("particles", fp)
    def _process():
        try:
            try:
                from .core.particles import overlay_particles
            except ImportError:
                from opencut.core.particles import overlay_particles
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = overlay_particles(fp, output_dir=d,
                                     preset=data.get("preset", "confetti"),
                                     density=data.get("density", 1.0),
                                     on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Face Swap & Enhancement
# ---------------------------------------------------------------------------
@app.route("/video/face/capabilities", methods=["GET"])
def face_capabilities():
    try:
        try:
            from .core.face_swap import get_face_capabilities
        except ImportError:
            from opencut.core.face_swap import get_face_capabilities
        return jsonify(get_face_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/face/enhance", methods=["POST"])
def face_enhance_route():
    """Enhance/restore faces in video using GFPGAN."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    job_id = _new_job("face-enhance", fp)
    def _process():
        try:
            try:
                from .core.face_swap import enhance_faces
            except ImportError:
                from opencut.core.face_swap import enhance_faces
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = enhance_faces(fp, output_dir=d, upscale=data.get("upscale", 2), on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/face/swap", methods=["POST"])
def face_swap_route():
    """Swap faces in video with a reference image."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    ref = data.get("reference_face", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "Video not found"}), 400
    if not ref or not os.path.isfile(ref): return jsonify({"error": "Reference face not found"}), 400
    job_id = _new_job("face-swap", fp)
    def _process():
        try:
            try:
                from .core.face_swap import swap_face
            except ImportError:
                from opencut.core.face_swap import swap_face
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = swap_face(fp, ref, output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Pro Upscaling
# ---------------------------------------------------------------------------
@app.route("/video/upscale/capabilities", methods=["GET"])
def upscale_capabilities():
    try:
        try:
            from .core.upscale_pro import get_upscale_capabilities
        except ImportError:
            from opencut.core.upscale_pro import get_upscale_capabilities
        return jsonify(get_upscale_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/upscale/run", methods=["POST"])
def upscale_run():
    """Upscale video with quality preset."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    job_id = _new_job("upscale", fp)
    def _process():
        try:
            try:
                from .core.upscale_pro import upscale_with_preset
            except ImportError:
                from opencut.core.upscale_pro import upscale_with_preset
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = upscale_with_preset(fp, preset=data.get("preset", "fast"),
                                       scale=data.get("scale", 2),
                                       output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Color Management
# ---------------------------------------------------------------------------
@app.route("/video/color/capabilities", methods=["GET"])
def color_capabilities():
    try:
        try:
            from .core.color_management import get_color_capabilities
        except ImportError:
            from opencut.core.color_management import get_color_capabilities
        return jsonify(get_color_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/color/correct", methods=["POST"])
def color_correct_route():
    """Apply color correction (exposure, contrast, saturation, temperature, etc.)."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    job_id = _new_job("color", fp)
    def _process():
        try:
            try:
                from .core.color_management import color_correct
            except ImportError:
                from opencut.core.color_management import color_correct
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = color_correct(fp, output_dir=d,
                                 exposure=data.get("exposure", 0),
                                 contrast=data.get("contrast", 1.0),
                                 saturation=data.get("saturation", 1.0),
                                 temperature=data.get("temperature", 0),
                                 tint=data.get("tint", 0),
                                 shadows=data.get("shadows", 0),
                                 midtones=data.get("midtones", 0),
                                 highlights=data.get("highlights", 0),
                                 on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/color/convert", methods=["POST"])
def color_convert_route():
    """Convert video color space."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "File not found"}), 400
    job_id = _new_job("colorspace", fp)
    def _process():
        try:
            try:
                from .core.color_management import convert_colorspace
            except ImportError:
                from opencut.core.color_management import convert_colorspace
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = convert_colorspace(fp, target=data.get("target", "rec709"),
                                      output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/video/color/external-lut", methods=["POST"])
def color_external_lut_route():
    """Apply external .cube/.3dl LUT file."""
    data = request.get_json(force=True)
    fp = data.get("filepath", "").strip()
    lut = data.get("lut_path", "").strip()
    if not fp or not os.path.isfile(fp): return jsonify({"error": "Video not found"}), 400
    if not lut or not os.path.isfile(lut): return jsonify({"error": "LUT file not found"}), 400
    job_id = _new_job("ext-lut", fp)
    def _process():
        try:
            try:
                from .core.color_management import apply_external_lut
            except ImportError:
                from opencut.core.color_management import apply_external_lut
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = _resolve_output_dir(fp, data.get("output_dir", ""))
            out = apply_external_lut(fp, lut, intensity=data.get("intensity", 1.0),
                                      output_dir=d, on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# AI Music Generation
# ---------------------------------------------------------------------------
@app.route("/audio/music-ai/capabilities", methods=["GET"])
def music_ai_capabilities():
    try:
        try:
            from .core.music_ai import get_music_ai_capabilities
        except ImportError:
            from opencut.core.music_ai import get_music_ai_capabilities
        return jsonify(get_music_ai_capabilities())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/audio/music-ai/generate", methods=["POST"])
def music_ai_generate():
    """Generate music from text prompt via MusicGen."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt: return jsonify({"error": "No prompt"}), 400
    job_id = _new_job("musicgen", prompt[:40])
    def _process():
        try:
            try:
                from .core.music_ai import generate_music
            except ImportError:
                from opencut.core.music_ai import generate_music
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = data.get("output_dir", "") or tempfile.gettempdir()
            out = generate_music(prompt, output_dir=d,
                                  duration=data.get("duration", 10),
                                  model_size=data.get("model", "small"),
                                  temperature=data.get("temperature", 1.0),
                                  on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


@app.route("/audio/music-ai/melody", methods=["POST"])
def music_ai_melody():
    """Generate music with melody conditioning."""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    melody = data.get("melody_path", "").strip()
    if not prompt: return jsonify({"error": "No prompt"}), 400
    if not melody or not os.path.isfile(melody): return jsonify({"error": "Melody not found"}), 400
    job_id = _new_job("musicgen-melody", prompt[:40])
    def _process():
        try:
            try:
                from .core.music_ai import generate_music_with_melody
            except ImportError:
                from opencut.core.music_ai import generate_music_with_melody
            def _p(pct, msg): _update_job(job_id, progress=pct, message=msg)
            d = data.get("output_dir", "") or tempfile.gettempdir()
            out = generate_music_with_melody(prompt, melody, output_dir=d,
                                              duration=data.get("duration", 10),
                                              on_progress=_p)
            _update_job(job_id, status="complete", progress=100, result={"output_path": out})
        except Exception as e:
            _update_job(job_id, status="error", error=str(e), message=f"Error: {e}")
    thread = threading.Thread(target=_process, daemon=True); thread.start()
    with job_lock:
        if job_id in jobs: jobs[job_id]["_thread"] = thread
    return jsonify({"job_id": job_id, "status": "running"})


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
    print(f"  OpenCut Backend Server v0.9.0")
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
