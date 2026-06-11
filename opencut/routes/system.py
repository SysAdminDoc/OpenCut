"""
OpenCut System Routes

Health, shutdown, GPU, dependencies, model management, whisper installation.
"""

import importlib
import logging
import os
import platform
import shutil
import subprocess as _sp
import sys
import threading
import time
from contextlib import suppress

from flask import Blueprint, jsonify, request, send_file

try:
    import psutil
except ImportError:
    psutil = None

from opencut import __version__
from opencut.errors import safe_error
from opencut.helpers import _try_import, _try_import_from, get_ffmpeg_path
from opencut.jobs import (
    _cancel_running_jobs,
    _is_cancelled,
    _kill_job_process,
    _list_jobs_copy,
    _register_job_process,
    _unregister_job_process,
    _update_job,
    async_job,
    job_lock,
    jobs,
    should_skip_install_in_testing,
    testing_install_response,
)
from opencut.security import (
    VALID_WHISPER_MODELS,
    build_destructive_plan,
    destructive_confirmation_required_response,
    get_csrf_token,
    get_json_dict,
    is_path_within_any,
    require_csrf,
    require_rate_limit,
    safe_bool,
    safe_float,
    safe_int,
    safe_pip_install,
    validate_filepath,
    validate_path,
    verify_destructive_confirm_token,
)
from opencut.user_data import load_whisper_settings, save_whisper_settings

logger = logging.getLogger("opencut")

_OPEN_PATH_ALLOWED_EXTS = frozenset({
    ".aac", ".aif", ".aiff", ".ass", ".avi", ".bmp", ".csv", ".edl",
    ".fcpxml", ".flac", ".gif", ".jpeg", ".jpg", ".json", ".log", ".m4a",
    ".m4v", ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".mxf",
    ".ogg", ".opus", ".otio", ".png", ".srt", ".tif", ".tiff", ".tsv",
    ".txt", ".vtt", ".wav", ".webm", ".webp", ".wmv", ".xml",
})

system_bp = Blueprint("system", __name__)

# ---------------------------------------------------------------------------
# Model Paths (from env / bundled installer)
# ---------------------------------------------------------------------------
WHISPER_MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", None)

# ---------------------------------------------------------------------------
# Models list cache (TTL-based)
# ---------------------------------------------------------------------------
_models_cache = {"data": None, "ts": 0}
_models_cache_lock = threading.Lock()
_MODELS_CACHE_TTL = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Server Start Time (for uptime calculation)
# ---------------------------------------------------------------------------
_server_start_time = time.time()


def _path_size_bytes(path: str) -> tuple[int, list[str]]:
    """Return recursive size and non-fatal read errors for a file or directory."""
    errors: list[str] = []
    if os.path.isfile(path):
        try:
            return os.path.getsize(path), errors
        except OSError as exc:
            return 0, [f"{path}: {exc}"]
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            item = os.path.join(root, filename)
            try:
                total += os.path.getsize(item)
            except OSError as exc:
                errors.append(f"{item}: {exc}")
    return total, errors


def _delete_cache_target(path: str) -> tuple[bool, str | None]:
    """Delete one planned cache target and return a per-path error when it fails."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            return False, f"{path}: path not found"
    except OSError as exc:
        return False, f"{path}: {exc}"
    return True, None


def _cache_plan_entry(path: str, *, category: str) -> dict:
    size, errors = _path_size_bytes(path)
    root = os.path.dirname(path) if os.path.isfile(path) else path
    entry = {
        "path": path,
        "category": category,
        "root": root,
        "type": "directory" if os.path.isdir(path) else "file",
        "bytes": size,
        "reversible": False,
    }
    if errors:
        entry["errors"] = errors
    return entry


def _build_whisper_cache_plan() -> dict:
    """Enumerate Whisper cache targets without deleting them."""
    entries: list[dict] = []
    errors: list[str] = []
    cache_paths = [
        ("huggingface-whisper", os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")),
        ("whisper-cache", os.path.join(os.path.expanduser("~"), ".cache", "whisper")),
        ("opencut-models", os.path.join(os.environ.get("LOCALAPPDATA", ""), "OpenCut", "models")),
    ]
    for category, cache_dir in cache_paths:
        if not cache_dir or not os.path.exists(cache_dir):
            continue
        if category == "huggingface-whisper":
            try:
                for item in os.listdir(cache_dir):
                    if "whisper" in item.lower():
                        entries.append(_cache_plan_entry(os.path.join(cache_dir, item), category=category))
            except OSError as exc:
                errors.append(f"{cache_dir}: {exc}")
        else:
            entries.append(_cache_plan_entry(cache_dir, category=category))
    for entry in entries:
        errors.extend(entry.get("errors", []))
    return {
        "entries": entries,
        "total_bytes": sum(int(entry.get("bytes", 0)) for entry in entries),
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Availability Helpers (local to this module)
# ---------------------------------------------------------------------------
def check_demucs_available():
    """Check if Demucs is installed and available."""
    return _try_import("demucs") is not None


def check_watermark_available():
    """Check if watermark removal dependencies are installed."""
    transformers = _try_import("transformers")
    if transformers is None:
        return False
    sli = _try_import("simple_lama_inpainting")
    if sli is None:
        return False
    return hasattr(sli, "SimpleLama")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
# Capabilities change only when deps are installed/uninstalled — no need to
# re-probe 20+ modules on every 4-second health check.  Cache for 30 s.
_caps_cache = {"data": None, "ts": 0.0}
_caps_cache_lock = threading.Lock()
_CAPS_CACHE_TTL = 30.0
_CSRF_BOOTSTRAP_BLOCKED_ORIGINS = {"null", "file://"}


def invalidate_caps_cache():
    """Called after install routes modify available dependencies.

    Both the per-route capability cache (used by /health-style probes) and
    the heavy /system/dependencies cache need invalidation — otherwise the
    Settings tab's dependency grid keeps showing the just-installed package
    as missing for up to 60 seconds, and the user re-clicks Install.
    """
    with _caps_cache_lock:
        _caps_cache["data"] = None
        _caps_cache["ts"] = 0.0
    # Also clear the /system/dependencies TTL cache so the next render
    # reflects the new install. Defined later in this module.
    try:
        with _deps_cache_lock:
            _deps_cache["data"] = None
            _deps_cache["ts"] = 0.0
    except NameError:
        # _deps_cache is defined later in module load order; if this
        # function is somehow called before that, the deps cache is
        # already empty.
        pass


def _build_capabilities():
    """Probe all optional dependencies and return a capabilities dict."""
    caps = {"silence": True, "zoom": True, "ffmpeg": True,
            "audio_suite": True, "scene_detect": True}
    try:
        check_whisper_available = _try_import_from("opencut.core.captions", "check_whisper_available")
        if check_whisper_available:
            available, backend = check_whisper_available()
            caps["captions"] = available
            caps["whisper_backend"] = backend
        else:
            caps["captions"] = False
            caps["whisper_backend"] = "none"
    except Exception:
        caps["captions"] = False
        caps["whisper_backend"] = "none"

    # Check Silero VAD availability (neural silence detection)
    from opencut.checks import check_crisper_whisper_available, check_otio_available, check_silero_vad_available
    caps["silero_vad"] = check_silero_vad_available()
    caps["crisper_whisper"] = check_crisper_whisper_available()
    caps["otio"] = check_otio_available()

    # Check Demucs availability
    caps["separation"] = check_demucs_available()

    # Check watermark removal availability
    caps["watermark_removal"] = check_watermark_available()

    # Include Whisper settings
    try:
        whisper_settings = load_whisper_settings()
        caps["whisper_cpu_mode"] = whisper_settings.get("cpu_mode", False)
    except Exception:
        caps["whisper_cpu_mode"] = False

    # Check video AI availability
    try:
        get_ai_capabilities = _try_import_from("opencut.core.video_ai", "get_ai_capabilities")
        if get_ai_capabilities:
            caps["video_ai"] = get_ai_capabilities()
        else:
            caps["video_ai"] = {}
    except Exception:
        caps["video_ai"] = {}

    # Check pedalboard availability
    try:
        check_pedalboard_available = _try_import_from("opencut.core.audio_pro", "check_pedalboard_available")
        check_deepfilter_available = _try_import_from("opencut.core.audio_pro", "check_deepfilter_available")
        if check_pedalboard_available:
            caps["pedalboard"] = check_pedalboard_available()
            caps["deepfilter"] = check_deepfilter_available()
        else:
            caps["pedalboard"] = False
            caps["deepfilter"] = False
    except Exception:
        caps["pedalboard"] = False
        caps["deepfilter"] = False

    # Video effects always available (FFmpeg-based)
    caps["video_fx"] = True

    # Face tools
    try:
        check_face_tools_available = _try_import_from("opencut.core.face_tools", "check_face_tools_available")
        if check_face_tools_available:
            caps["face_tools"] = check_face_tools_available()
        else:
            caps["face_tools"] = {}
    except Exception:
        caps["face_tools"] = {}

    # Style transfer (always available via OpenCV DNN)
    caps["style_transfer"] = True

    # Enhanced captions
    try:
        check_whisperx_available = _try_import_from("opencut.core.captions_enhanced", "check_whisperx_available")
        check_nllb_available = _try_import_from("opencut.core.captions_enhanced", "check_nllb_available")
        if check_whisperx_available:
            caps["whisperx"] = check_whisperx_available()
            caps["nllb"] = check_nllb_available()
        else:
            caps["whisperx"] = False
            caps["nllb"] = False
    except Exception:
        caps["whisperx"] = False
        caps["nllb"] = False

    # Voice generation
    try:
        check_edge_tts_available = _try_import_from("opencut.core.voice_gen", "check_edge_tts_available")
        check_kokoro_available = _try_import_from("opencut.core.voice_gen", "check_kokoro_available")
        if check_edge_tts_available:
            caps["edge_tts"] = check_edge_tts_available()
            caps["kokoro"] = check_kokoro_available()
        else:
            caps["edge_tts"] = False
            caps["kokoro"] = False
    except Exception:
        caps["edge_tts"] = False
        caps["kokoro"] = False

    # Caption burn-in always available (FFmpeg-based)
    caps["burnin"] = True

    # Depth effects (Depth Anything V2)
    from opencut.checks import check_depth_available
    caps["depth_effects"] = check_depth_available()

    # DaVinci Resolve bridge
    from opencut.checks import check_resolve_available
    caps["resolve"] = check_resolve_available()

    # Emotion analysis (deepface)
    from opencut.checks import check_deepface_available
    caps["deepface"] = check_deepface_available()

    # Multimodal diarization (audio + face recognition)
    from opencut.checks import check_multimodal_diarize_available
    caps["multimodal_diarize"] = check_multimodal_diarize_available()

    # AI B-roll generation (text-to-video)
    from opencut.checks import check_broll_generate_available
    caps["broll_generate"] = check_broll_generate_available()

    # WebSocket bridge
    from opencut.checks import check_websocket_available
    caps["websocket"] = check_websocket_available()

    # Social media posting
    from opencut.checks import check_social_post_available
    caps["social_post"] = check_social_post_available()

    # Engine registry status
    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        caps["engine_registry"] = {
            "domains": len(reg.get_all_domains()),
            "total_engines": sum(len(reg.get_engines(d)) for d in reg.get_all_domains()),
        }
    except Exception:
        caps["engine_registry"] = {}

    return caps


def _health_should_expose_csrf_token() -> bool:
    origin = (request.headers.get("Origin") or "").strip().lower()
    return origin not in _CSRF_BOOTSTRAP_BLOCKED_ORIGINS


@system_bp.route("/health", methods=["GET"])
def health():
    # Serve cached capabilities; rebuild at most once per 30 s.
    with _caps_cache_lock:
        if _caps_cache["data"] is not None and (time.time() - _caps_cache["ts"]) < _CAPS_CACHE_TTL:
            caps = _caps_cache["data"]
        else:
            caps = None

    if caps is None:
        caps = _build_capabilities()
        with _caps_cache_lock:
            _caps_cache["data"] = caps
            _caps_cache["ts"] = time.time()

    payload = {
        "status": "ok",
        "version": __version__,
        "capabilities": caps,
    }
    if _health_should_expose_csrf_token():
        payload["csrf_token"] = get_csrf_token()
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
@system_bp.route("/shutdown", methods=["POST"])
@require_csrf
def shutdown_server():
    """Aggressively shut down the server. Used by new instances to kill old ones."""
    # Only allow shutdown from localhost
    remote = request.remote_addr or ""
    if remote not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("Shutdown attempt from non-localhost IP: %s", remote)
        return jsonify({"error": "Shutdown only allowed from localhost"}), 403
    logger.info("Shutdown requested via /shutdown endpoint")

    # Cancel all running jobs first and persist synchronously because the
    # process exits hard a moment later.
    cancelled = _cancel_running_jobs(
        message="Cancelled due to server shutdown",
        persist_sync=True,
    )
    for jid in cancelled:
        _kill_job_process(jid)

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


# ---------------------------------------------------------------------------
# Media Info
# ---------------------------------------------------------------------------
@system_bp.route("/info", methods=["POST"])
@require_csrf
def media_info():
    """Get media file metadata."""
    from opencut.utils.media import probe as _probe_media

    try:
        data = get_json_dict()
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "code": "INVALID_INPUT",
            "suggestion": "Send a top-level JSON object in the request body.",
        }), 400
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({
            "error": "No file path provided",
            "code": "INVALID_INPUT",
            "suggestion": "Pass `filepath` in the request body.",
        }), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        msg = str(e)
        lower = msg.lower()
        if "not found" in lower or "not a file" in lower or "does not exist" in lower:
            code, hint = "FILE_NOT_FOUND", "Check that the file exists and is accessible."
        else:
            code, hint = "INVALID_INPUT", "Use a plain absolute path with no traversal or UNC prefix."
        return jsonify({"error": msg, "code": code, "suggestion": hint}), 400

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
        return safe_error(e, "media_info")


# ---------------------------------------------------------------------------
# GPU / System Info
# ---------------------------------------------------------------------------
_gpu_cache = {"info": None, "ts": 0}
_gpu_cache_lock = threading.Lock()
_GPU_CACHE_TTL = 30  # seconds


def _detect_gpu():
    """Detect GPU via nvidia-smi with 30s cache."""
    now = time.time()
    with _gpu_cache_lock:
        if _gpu_cache["info"] is not None and (now - _gpu_cache["ts"]) < _GPU_CACHE_TTL:
            return dict(_gpu_cache["info"])
    gpu_info = {"available": False, "name": "None", "vram_mb": 0}
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]  # First GPU only
            # Use rsplit to handle GPU names containing commas (e.g. "NVIDIA GeForce RTX 4090, 24564")
            parts = line.rsplit(",", 1)
            gpu_info["available"] = True
            gpu_info["name"] = parts[0].strip()
            gpu_info["vram_mb"] = safe_int(parts[1].strip()) if len(parts) > 1 else 0
    except Exception:
        pass
    with _gpu_cache_lock:
        _gpu_cache["info"] = dict(gpu_info)
        _gpu_cache["ts"] = now
    return gpu_info


@system_bp.route("/system/gpu", methods=["GET"])
def system_gpu():
    """Check GPU availability for AI features."""
    return jsonify(_detect_gpu())


# ---------------------------------------------------------------------------
# System Status (lightweight, polled every 5s by frontend status bar)
# ---------------------------------------------------------------------------
_vram_cache = {"used_mb": 0, "ts": 0}
_vram_cache_lock = threading.Lock()
_VRAM_CACHE_TTL = 30  # seconds — same as GPU info cache


def _get_vram_used():
    """Get VRAM usage with 30s cache to avoid frequent nvidia-smi calls."""
    now = time.time()
    with _vram_cache_lock:
        if (now - _vram_cache["ts"]) < _VRAM_CACHE_TTL:
            return _vram_cache["used_mb"]
    used = 0
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            used = safe_int(result.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    with _vram_cache_lock:
        _vram_cache["used_mb"] = used
        _vram_cache["ts"] = now
    return used


@system_bp.route("/system/status", methods=["GET"])
def system_status():
    """Return comprehensive system status for the status bar."""
    now = time.time()
    uptime = now - _server_start_time

    # CPU / RAM via psutil or fallback
    cpu_percent = 0.0
    ram_used_mb = 0
    ram_total_mb = 0
    disk_free_gb = 0.0

    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            ram_used_mb = int(mem.used / (1024 * 1024))
            ram_total_mb = int(mem.total / (1024 * 1024))
        except Exception:
            pass
        try:
            disk = psutil.disk_usage(os.path.abspath(os.sep))
            disk_free_gb = round(disk.free / (1024 * 1024 * 1024), 1)
        except Exception:
            pass
    else:
        # Fallback: disk via shutil
        try:
            usage = shutil.disk_usage(os.path.abspath(os.sep))
            disk_free_gb = round(usage.free / (1024 * 1024 * 1024), 1)
        except Exception:
            pass

    # GPU info (reuses cached _detect_gpu + cached VRAM usage)
    gpu_raw = _detect_gpu()
    gpu_info = {
        "available": gpu_raw.get("available", False),
        "name": gpu_raw.get("name", "None"),
        "vram_used_mb": 0,
        "vram_total_mb": gpu_raw.get("vram_mb", 0),
    }
    if gpu_raw.get("available"):
        gpu_info["vram_used_mb"] = _get_vram_used()

    # Job counts
    running = 0
    queued = 0
    completed_today = 0
    with job_lock:
        for j in jobs.values():
            status = j.get("status", "")
            if status == "running":
                running += 1
            elif status == "queued":
                queued += 1
            elif status == "complete":
                # Count jobs completed today (use created time as proxy)
                created = j.get("created", 0)
                if created and (now - created) < 86400:
                    completed_today += 1

    return jsonify({
        "connected": True,
        "uptime_seconds": int(uptime),
        "cpu_percent": cpu_percent,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "gpu": gpu_info,
        "disk_free_gb": disk_free_gb,
        "jobs": {
            "running": running,
            "queued": queued,
            "completed_today": completed_today,
        },
        "python_version": platform.python_version(),
        "server_version": __version__,
    })


# ---------------------------------------------------------------------------
# GPU Recommendation
# ---------------------------------------------------------------------------
@system_bp.route("/system/gpu-recommend", methods=["GET"])
def gpu_recommend():
    """Recommend model sizes and settings based on GPU."""
    gpu_info = _detect_gpu()

    vram = gpu_info["vram_mb"]
    rec = {
        "gpu": gpu_info,
        "whisper_model": "tiny",
        "whisper_device": "cpu",
        "caption_quality": "fast",
        "batch_size": 1,
        "notes": []
    }
    if gpu_info["available"]:
        rec["whisper_device"] = "cuda"
        if vram >= 10000:
            rec["whisper_model"] = "large-v3"
            rec["caption_quality"] = "best"
            rec["batch_size"] = 4
            rec["notes"].append("High-end GPU: all features at max quality")
        elif vram >= 6000:
            rec["whisper_model"] = "medium"
            rec["caption_quality"] = "great"
            rec["batch_size"] = 2
            rec["notes"].append("Mid-range GPU: most features at high quality")
        elif vram >= 4000:
            rec["whisper_model"] = "small"
            rec["caption_quality"] = "good"
            rec["batch_size"] = 1
            rec["notes"].append("Entry GPU: good quality, some features may be slow")
        else:
            rec["whisper_model"] = "base"
            rec["caption_quality"] = "balanced"
            rec["batch_size"] = 1
            rec["notes"].append("Low VRAM: use smaller models for best performance")
    else:
        rec["notes"].append("No NVIDIA GPU detected. Using CPU mode (slower).")
    return jsonify(rec)


# ---------------------------------------------------------------------------
# Dependency Health Dashboard
# ---------------------------------------------------------------------------
# Cold call invokes 20+ importlib.import_module() + ffprobe/ffmpeg subprocess +
# an ollama HTTP probe. On first hit (Settings tab open) this takes ~5 s
# which blocks the frontend. Cache the result for 60 s across all requests.
_deps_cache = {"data": None, "ts": 0.0}
_deps_cache_lock = threading.Lock()
_DEPS_CACHE_TTL = 60.0


@system_bp.route("/system/dependencies", methods=["GET"])
def check_dependencies():
    """Check all optional dependencies and return their status.

    Supports `?fresh=1` to bypass the 60 s TTL cache when the user explicitly
    triggers a re-check (e.g., after an install finishes).
    """
    fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
    if not fresh:
        with _deps_cache_lock:
            if _deps_cache["data"] is not None and (time.time() - _deps_cache["ts"]) < _DEPS_CACHE_TTL:
                return jsonify(_deps_cache["data"])

    deps = {}
    checks = {
        "faster-whisper": "faster_whisper",
        "whisperx": "whisperx",
        "demucs": "demucs",
        "pedalboard": "pedalboard",
        "deepfilternet": "df",
        "noisereduce": "noisereduce",
        "librosa": "librosa",
        "pydub": "pydub",
        "opencv": "cv2",
        "Pillow": "PIL",
        "numpy": "numpy",
        "rembg": "rembg",
        "realesrgan": "realesrgan",
        "gfpgan": "gfpgan",
        "insightface": "insightface",
        "edge-tts": "edge_tts",
        "audiocraft": "audiocraft",
        "scenedetect": "scenedetect",
        "pyannote.audio": "pyannote.audio.pipelines",
        "mediapipe": "mediapipe",
        "torch": "torch",
        "onnxruntime": "onnxruntime",
    }
    for name, module in checks.items():
        try:
            mod = importlib.import_module(module.split(".")[0])
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
            deps[name] = {"installed": True, "version": str(version)}
        except ImportError:
            deps[name] = {"installed": False, "version": None}

    # Check FFmpeg
    try:
        r = _sp.run(
            [get_ffmpeg_path(), "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        line = r.stdout.split("\n")[0].strip() if r.stdout else ""
        deps["ffmpeg"] = {
            "installed": r.returncode == 0 and bool(line),
            "version": line if r.returncode == 0 and line else None,
        }
    except Exception:
        deps["ffmpeg"] = {"installed": False, "version": None}

    # Color match (cv2 + numpy)
    try:
        from opencut.checks import check_color_match_available
        ok, backend = check_color_match_available()
        deps["color_match"] = {"installed": ok, "version": backend if ok else None}
    except Exception:
        deps["color_match"] = {"installed": False, "version": None}

    # Auto zoom (cv2)
    try:
        from opencut.checks import check_auto_zoom_available
        ok, backend = check_auto_zoom_available()
        deps["auto_zoom"] = {"installed": ok, "version": backend if ok else None}
    except Exception:
        deps["auto_zoom"] = {"installed": False, "version": None}

    # Footage search (stdlib only — always available)
    try:
        from opencut.checks import check_footage_search_available
        ok, backend = check_footage_search_available()
        deps["footage_search"] = {"installed": ok, "version": backend if ok else None}
    except Exception:
        deps["footage_search"] = {"installed": True, "version": "stdlib"}

    # Loudness match (ffmpeg in PATH)
    try:
        from opencut.checks import check_loudness_match_available
        ok, backend = check_loudness_match_available()
        deps["loudness_match"] = {"installed": ok, "version": backend if ok else None}
    except Exception:
        deps["loudness_match"] = {"installed": False, "version": None}

    # Deliverables (csv stdlib — always available)
    deps["deliverables"] = {"installed": True, "version": "stdlib"}

    # Multicam (pure Python — always available)
    deps["multicam"] = {"installed": True, "version": "stdlib"}

    # NLP command (ollama running or openai/anthropic key set)
    try:
        from opencut.checks import check_ollama_available
        ollama_ok = check_ollama_available()
        openai_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        nlp_ok = ollama_ok or openai_key or anthropic_key
        if ollama_ok:
            nlp_backend = "ollama"
        elif openai_key:
            nlp_backend = "openai"
        elif anthropic_key:
            nlp_backend = "anthropic"
        else:
            nlp_backend = None
        deps["nlp_command"] = {"installed": nlp_ok, "version": nlp_backend}
    except Exception:
        deps["nlp_command"] = {"installed": False, "version": None}

    with _deps_cache_lock:
        _deps_cache["data"] = deps
        _deps_cache["ts"] = time.time()

    return jsonify(deps)


# ---------------------------------------------------------------------------
# Project Media (stub for non-Premiere dev mode)
# ---------------------------------------------------------------------------
@system_bp.route("/project/media", methods=["GET"])
def project_media():
    """Return project media list. Stub for dev mode (outside Premiere).

    Inside Premiere the CEP panel uses ExtendScript directly; this endpoint
    exists so the panel doesn't 404 when running in a browser for development.
    """
    return jsonify({"media": [], "projectFolder": "", "rootChildren": 0})


# ---------------------------------------------------------------------------
# Open / Reveal Path (for session context "Open Output" / "Reveal in Folder")
# ---------------------------------------------------------------------------
@system_bp.route("/system/open-path", methods=["POST"])
@require_csrf
def open_path():
    """Open a file (``mode=open``) or reveal it in the OS file manager
    (``mode=reveal``). Used by the session-context overlay's job-result
    quick actions.

    The path must pass ``validate_filepath`` — no traversal, null bytes,
    or UNC paths. Non-existent files are rejected early.
    """
    data = request.get_json(force=True, silent=True) or {}
    raw_path = data.get("path", "")
    mode = str(data.get("mode", "open")).strip().lower()
    if mode not in ("open", "reveal"):
        return jsonify({"error": "mode must be 'open' or 'reveal'"}), 400
    if not raw_path:
        return jsonify({"error": "path is required"}), 400

    try:
        filepath = validate_filepath(raw_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ext = os.path.splitext(filepath)[1].lower()
    if mode == "open" and ext not in _OPEN_PATH_ALLOWED_EXTS:
        return jsonify({"error": f"Cannot open unsupported file type: {ext or '(none)'}"}), 403

    try:
        if sys.platform == "win32":
            if mode == "reveal":
                _sp.Popen(["explorer", "/select,", filepath],
                          creationflags=_sp.CREATE_NEW_PROCESS_GROUP
                          if hasattr(_sp, "CREATE_NEW_PROCESS_GROUP") else 0)
            else:
                os.startfile(filepath)  # noqa: S606 — validated path, open-mode extension allowlisted above
        elif sys.platform == "darwin":
            if mode == "reveal":
                _sp.Popen(["open", "-R", filepath], start_new_session=True)
            else:
                _sp.Popen(["open", filepath], start_new_session=True)
        else:
            # Linux / BSD — xdg-open has no "reveal" equivalent; open the
            # containing directory when reveal is requested.
            target = os.path.dirname(filepath) if mode == "reveal" else filepath
            _sp.Popen(["xdg-open", target], start_new_session=True)
    except Exception as e:
        logger.exception("open_path failed for %s", filepath)
        return jsonify({"error": f"Could not open path: {e}"}), 500

    return jsonify({"ok": True, "path": filepath, "mode": mode})


# ---------------------------------------------------------------------------
# Sequence Assistant (v1.10.0, feature E)
# ---------------------------------------------------------------------------
@system_bp.route("/assistant/suggest", methods=["POST"])
@require_csrf
def assistant_suggest():
    """Analyze a sequence snapshot and return ranked editing suggestions.

    Body::

        {
          "sequence": {...},               # shape of ocGetSequenceInfo JSON
          "dismissed": ["silence-dead-air"],  # optional, session-scoped
          "sequence_key": "project.prproj"    # optional, v1.10.2 persistent
        }

    v1.10.2: persisted dismissals are keyed by *sequence_key* (typically
    the Premiere project path). The server unions session dismissals
    with the stored per-sequence set so either flavor hides the card.
    """
    from opencut.core.assistant import analyze_sequence
    from opencut.user_data import load_assistant_dismissed

    data = request.get_json(force=True, silent=True) or {}
    sequence = data.get("sequence") or {}
    dismissed = data.get("dismissed") or []
    sequence_key = (data.get("sequence_key") or "").strip() or "default"
    if not isinstance(sequence, dict):
        return jsonify({"error": "sequence must be an object"}), 400
    if not isinstance(dismissed, list):
        return jsonify({"error": "dismissed must be a list"}), 400

    persisted = load_assistant_dismissed(sequence_key)
    all_dismissed = list({str(d) for d in dismissed} | set(persisted))

    try:
        suggestions = analyze_sequence(sequence, dismissed_ids=all_dismissed)
    except Exception as e:
        logger.exception("assistant_suggest failed")
        return jsonify({"error": f"Analyze failed: {e}"}), 500
    return jsonify({
        "suggestions": suggestions,
        "count": len(suggestions),
        "persisted_dismissed": persisted,
    })


@system_bp.route("/assistant/dismiss", methods=["POST"])
@require_csrf
def assistant_dismiss():
    """Persistently dismiss a suggestion for a sequence.

    Body: ``{"sequence_key": "...", "id": "silence-dead-air"}``
    Returns the updated list so the panel can reconcile state.
    """
    from opencut.user_data import load_assistant_dismissed, save_assistant_dismissed

    data = request.get_json(force=True, silent=True) or {}
    sequence_key = (data.get("sequence_key") or "").strip() or "default"
    sug_id = (data.get("id") or "").strip()
    if not sug_id:
        return jsonify({"error": "id is required"}), 400

    current = load_assistant_dismissed(sequence_key)
    if sug_id not in current:
        current.append(sug_id)
        save_assistant_dismissed(sequence_key, current)
    return jsonify({"sequence_key": sequence_key, "dismissed": current})


@system_bp.route("/assistant/dismiss-clear", methods=["POST"])
@require_csrf
def assistant_dismiss_clear():
    """Clear every persisted dismissal for a sequence so they reappear."""
    from opencut.user_data import (
        build_user_data_destructive_record,
        create_user_tombstone,
        load_assistant_dismissed,
        save_assistant_dismissed,
        summarize_user_tombstone,
    )

    data = get_json_dict() if request.data else {}
    sequence_key = (data.get("sequence_key") or "").strip() or "default"
    current = load_assistant_dismissed(sequence_key)
    records = []
    if current:
        records.append(build_user_data_destructive_record(
            "assistant_dismissed",
            sequence_key,
            current,
            source_file="assistant_dismissed.json",
            route="/assistant/dismiss-clear",
            action="clear",
        ))
    plan = build_destructive_plan(
        "assistant.dismiss_clear",
        records=records,
        metadata={"route": "/assistant/dismiss-clear", "sequence_key": sequence_key},
        reversible=True,
    )
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "sequence_key": sequence_key,
            "dismissed": current,
            "would_clear": len(current),
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
        })
    if records and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409
    tombstone = create_user_tombstone(
        "assistant_dismissed",
        sequence_key,
        current,
        source_file="assistant_dismissed.json",
        action="clear",
        metadata={"route": "/assistant/dismiss-clear"},
    )
    save_assistant_dismissed(sequence_key, [])
    return jsonify({
        "sequence_key": sequence_key,
        "dismissed": [],
        "tombstone": summarize_user_tombstone(tombstone),
    })


# ---------------------------------------------------------------------------
# Live audio preview (v1.9.36, feature C)
# ---------------------------------------------------------------------------
# Renders a short slice of a clip with an effect applied and returns the raw
# WAV bytes so the panel can A/B-compare settings in ~1s instead of waiting
# for a full-file job. Budget: 15s slice + FFmpeg-native filters only. Heavy
# neural processors (Resemble Enhance, Demucs) fall back to full-file jobs.
_PREVIEW_MAX_SECONDS = 15
_PREVIEW_FILTERS = {"denoise", "normalize", "compress", "eq", "silence"}


@system_bp.route("/preview/audio", methods=["POST"])
@require_csrf
def preview_audio():
    """Render a short audio slice with a filter applied.

    Body::

        {
          "filepath": "...",
          "start": 0,                   # seconds into the file
          "duration": 10,               # seconds of preview (max 15)
          "filter": "denoise|normalize|compress|eq|silence",
          "params": {...}               # filter-specific
        }

    Returns ``audio/wav`` on success, 400 on bad input.
    """
    import tempfile

    data = request.get_json(force=True, silent=True) or {}
    filepath = (data.get("filepath") or "").strip()
    start = safe_float(data.get("start", 0), 0.0, min_val=0.0)
    duration = safe_float(data.get("duration", 10), 10.0,
                          min_val=1.0, max_val=float(_PREVIEW_MAX_SECONDS))
    flt = str(data.get("filter", "denoise")).lower()
    params = data.get("params") or {}

    if flt not in _PREVIEW_FILTERS:
        return jsonify({
            "error": f"Unsupported preview filter: {flt}",
            "supported": sorted(_PREVIEW_FILTERS),
        }), 400
    if not filepath:
        return jsonify({"error": "filepath is required"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Build an FFmpeg filter chain for the requested filter type. All are
    # lightweight built-in filters that run in near-realtime on a short
    # slice — no ML model loads, no model downloads.
    afilter = None
    if flt == "denoise":
        # afftdn is an FFT-based denoiser — maps 0..1 strength to noise
        # reduction level in dB (0..24).
        strength = safe_float(params.get("strength", 0.5), 0.5, min_val=0.0, max_val=1.0)
        nr = round(strength * 24, 1)
        afilter = f"afftdn=nr={nr}"
    elif flt == "normalize":
        target = safe_float(params.get("target_lufs", -16.0), -16.0,
                            min_val=-40.0, max_val=0.0)
        afilter = f"loudnorm=I={target}:TP=-1.5:LRA=11"
    elif flt == "compress":
        threshold = safe_float(params.get("threshold_db", -20), -20.0,
                               min_val=-40.0, max_val=0.0)
        ratio = safe_float(params.get("ratio", 4.0), 4.0, min_val=1.0, max_val=20.0)
        # acompressor threshold is linear (0..1); approximate from dB.
        thr_lin = 10 ** (threshold / 20.0)
        afilter = f"acompressor=threshold={thr_lin}:ratio={ratio}:attack=5:release=50"
    elif flt == "eq":
        low = safe_float(params.get("low_db", 0), 0.0, min_val=-24.0, max_val=24.0)
        mid = safe_float(params.get("mid_db", 0), 0.0, min_val=-24.0, max_val=24.0)
        high = safe_float(params.get("high_db", 0), 0.0, min_val=-24.0, max_val=24.0)
        afilter = (
            f"equalizer=f=120:t=h:w=100:g={low},"
            f"equalizer=f=1000:t=h:w=800:g={mid},"
            f"equalizer=f=8000:t=h:w=4000:g={high}"
        )
    elif flt == "silence":
        # Preview the impact of a silence threshold by *muting* detected
        # silent regions in the slice so users can hear what gets cut.
        thr_db = safe_float(params.get("threshold_db", -30), -30.0,
                            min_val=-60.0, max_val=0.0)
        min_dur = safe_float(params.get("min_silence", 0.4), 0.4,
                             min_val=0.05, max_val=5.0)
        afilter = f"silenceremove=start_periods=0:stop_periods=-1:stop_threshold={thr_db}dB:stop_duration={min_dur}"

    tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="opencut_preview_", delete=False
    )
    out_path = tmp.name
    tmp.close()

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", filepath,
        "-vn",
        "-af", afilter,
        "-ac", "2",
        "-ar", "44100",
        "-f", "wav",
        out_path,
    ]
    try:
        proc = _sp.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0:
            os.unlink(out_path)
            return jsonify({
                "error": "Preview render failed",
                "detail": proc.stderr.decode("utf-8", "ignore")[-300:],
            }), 500
    except _sp.TimeoutExpired:
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return jsonify({"error": "Preview render timed out"}), 504
    except Exception as e:
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return jsonify({"error": f"Preview render failed: {e}"}), 500

    # Read back + return as audio/wav. Schedule deletion of the temp file.
    try:
        with open(out_path, "rb") as f:
            data_bytes = f.read()
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass

    from flask import Response
    return Response(data_bytes, mimetype="audio/wav",
                    headers={"Content-Length": str(len(data_bytes))})


# ---------------------------------------------------------------------------
# Preflight — "can this pipeline succeed?" check (v1.9.33, feature G)
# ---------------------------------------------------------------------------
@system_bp.route("/preflight/<pipeline>", methods=["POST"])
@require_csrf
def preflight_check(pipeline: str):
    """Run the preflight checklist for *pipeline* before kicking off the job.

    Body::

        {"filepath": "...", "output_dir": "..."}

    Response carries blocking issues (must fix), soft warnings (might
    degrade but won't abort), and a top-level pass:true/false.
    """
    from opencut.preflight import run_preflight

    data = request.get_json(force=True, silent=True) or {}
    filepath = (data.get("filepath") or "").strip()
    output_dir = (data.get("output_dir") or "").strip()

    try:
        report = run_preflight(pipeline, filepath=filepath, output_dir=output_dir)
    except Exception as e:
        logger.exception("preflight for %s failed", pipeline)
        return jsonify({"error": f"Preflight failed: {e}"}), 500
    if "error" in report:
        return jsonify(report), 400
    return jsonify(report)


# ---------------------------------------------------------------------------
# File Serving (for audio preview player)
# ---------------------------------------------------------------------------
@system_bp.route("/file", methods=["GET"])
def serve_file():
    """Serve a validated local media file for preview.

    Defence-in-depth:

    1. ``validate_filepath`` rejects null bytes, traversal, and UNC paths.
    2. The resolved realpath must live under an approved root — either the
       per-user ``~/.opencut`` directory (job outputs) or the system temp
       directory (short-lived work files). Arbitrary local media files
       elsewhere on disk are NOT served even if they look like audio/video.
       This prevents any same-browser origin from enumerating personal
       media via ``<img>``/``<audio>`` side-channels.
    3. The guessed MIME type must be audio/video/image so this endpoint
       can never act as a generic local-file reader.
    """
    import mimetypes
    import tempfile

    from opencut.helpers import OPENCUT_DIR

    filepath = request.args.get("path", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 404

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Approved roots: OpenCut user data dir + system temp dir. All legitimate
    # preview files (TTS/SFX/music outputs, intermediate renders) land here.
    allowed_roots = [
        tempfile.gettempdir(),
        OPENCUT_DIR,
    ]
    if not is_path_within_any(filepath, allowed_roots):
        return jsonify({"error": "Access denied"}), 403

    mime_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    if not mime_type.startswith(("audio/", "video/", "image/")):
        return jsonify({"error": "Unsupported preview file type"}), 403

    return send_file(filepath, mimetype=mime_type)


# ---------------------------------------------------------------------------
# Output Browser (recent outputs)
# ---------------------------------------------------------------------------
@system_bp.route("/outputs/recent", methods=["GET"])
def recent_outputs():
    """List recent output files from completed jobs."""
    limit = safe_int(request.args.get("limit", 20), default=20, min_val=1, max_val=100)
    outputs = []
    all_jobs = _list_jobs_copy()
    sorted_jobs = sorted(all_jobs, key=lambda j: j.get("created", 0), reverse=True)
    for job in sorted_jobs[:limit * 2]:
        if job.get("status") != "complete":
            continue
        result = job.get("result", {})
        path = result.get("output_path", result.get("output", ""))
        if isinstance(path, list):
            for p in path:
                if os.path.isfile(p):
                    try:
                        stat = os.stat(p)
                        outputs.append({
                            "path": p,
                            "name": os.path.basename(p),
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": stat.st_mtime,
                            "type": job.get("type", "unknown"),
                        })
                    except OSError:
                        pass
        elif isinstance(path, str) and path and os.path.isfile(path):
            try:
                stat = os.stat(path)
                outputs.append({
                    "path": path,
                    "name": os.path.basename(path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "type": job.get("type", "unknown"),
                })
            except OSError:
                pass
        if len(outputs) >= limit:
            break
    return jsonify(outputs)


# ---------------------------------------------------------------------------
# Install Whisper
# ---------------------------------------------------------------------------
@system_bp.route("/install-whisper", methods=["POST"])
@require_csrf
@async_job("install-whisper", filepath_required=False, rate_limit_key="model_install")
def install_whisper(job_id, filepath, data):
    """Install faster-whisper via pip (on-demand from panel)."""
    backend = data.get("backend", "faster-whisper")
    allowed = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed:
        raise ValueError(f"Unknown backend: {backend}")

    try:
        import subprocess as _sp

        _update_job(job_id, progress=5, message=f"Installing {backend}...")
        logger.info(f"Starting Whisper install: {backend}")

        # Resolve Python executable (frozen builds can't use sys.executable for pip)
        from opencut.security import _find_system_python
        if getattr(sys, "frozen", False):
            _pip_python = _find_system_python() or sys.executable
        else:
            _pip_python = sys.executable

        # Permission fallback: --target to a writable directory
        _target_dir = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
        os.makedirs(_target_dir, exist_ok=True)
        if _target_dir not in sys.path:
            sys.path.insert(0, _target_dir)

        # Build pip commands with permission fallbacks baked in
        def _pip_cmd(pkg, *extra_flags):
            """Return list of pip commands: normal, --user, --target."""
            base = [_pip_python, "-m", "pip", "install"] + list(extra_flags) + [pkg]
            return [
                base + ["--progress-bar", "on"],
                base + ["--user", "--progress-bar", "on"],
                base + ["--target", _target_dir, "--progress-bar", "on"],
            ]

        def _pip_pre(pkg, *extra_flags):
            """Return list of pre-install commands with fallbacks."""
            base = [_pip_python, "-m", "pip", "install"] + list(extra_flags) + [pkg, "--quiet"]
            return [
                base,
                base + ["--user"],
                base + ["--target", _target_dir],
            ]

        # Strategy list - try each in order until one works
        strategies = []
        if backend == "faster-whisper":
            strategies = [
                {
                    "label": "Install tokenizers wheel first, then faster-whisper",
                    "pre_cmds": _pip_pre("tokenizers", "--only-binary", "tokenizers"),
                    "cmds": _pip_cmd("faster-whisper"),
                },
                {
                    "label": "Upgrade pip + prefer binary wheels",
                    "pre_cmds": _pip_pre("pip setuptools wheel", "--upgrade"),
                    "cmds": _pip_cmd("faster-whisper", "--prefer-binary"),
                },
                {
                    "label": "Pin older tokenizers with wheel support",
                    "pre_cmds": _pip_pre("tokenizers>=0.13,<0.20", "--only-binary", "tokenizers"),
                    "cmds": _pip_cmd("faster-whisper"),
                },
                {
                    "label": "Fallback to openai-whisper (no Rust needed)",
                    "cmds": _pip_cmd("openai-whisper"),
                    "verify": [_pip_python, "-c",
                               "import whisper; print('ok')"],
                    "backend_name": "openai-whisper",
                },
            ]
        else:
            strategies = [
                {
                    "label": "Standard install",
                    "cmds": _pip_cmd(backend),
                },
            ]

        last_error = ""
        for si, strat in enumerate(strategies):
            if _is_cancelled(job_id):
                return {"backend": backend, "installed": False, "cancelled": True}

            pct_base = int(5 + (si / len(strategies)) * 70)
            _update_job(job_id, progress=pct_base,
                        message=f"Strategy {si+1}/{len(strategies)}: {strat['label']}...")
            logger.info(f"Whisper install strategy {si+1}: {strat['label']}")

            # Run pre-commands if present (try each fallback until one succeeds)
            if "pre_cmds" in strat:
                for pre_cmd in strat["pre_cmds"]:
                    try:
                        pre_result = _sp.run(
                            pre_cmd,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            check=False,
                        )
                        if pre_result.returncode == 0:
                            logger.debug(f"Pre-command succeeded: {' '.join(pre_cmd[:5])}")
                            break
                        logger.debug(f"Pre-command exit {pre_result.returncode}, trying next fallback")
                    except Exception as e:
                        logger.warning(f"Pre-command failed: {e}")

            # Run main install command (try each permission fallback)
            cmds_to_try = strat.get("cmds", [strat["cmd"]] if "cmd" in strat else [])
            pip_ok = False
            lines = []
            for cmd_variant in cmds_to_try:
                try:
                    proc = _sp.Popen(cmd_variant, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
                    _register_job_process(job_id, proc)

                    # Safety timeout: kill pip if it hangs (10 minutes)
                    _pip_timeout = 600
                    _pip_timer = threading.Timer(_pip_timeout, lambda: proc.kill())
                    _pip_timer.daemon = True
                    _pip_timer.start()

                    lines = []
                    for line in proc.stdout:
                        line = line.strip()
                        if line:
                            lines.append(line)
                            lower = line.lower()
                            if "downloading" in lower:
                                _update_job(job_id, progress=pct_base + 10,
                                            message="Downloading packages...")
                            elif "installing" in lower or "building" in lower:
                                _update_job(job_id, progress=pct_base + 20,
                                            message="Installing packages...")
                            elif "successfully installed" in lower:
                                _update_job(job_id, progress=85,
                                            message="Verifying installation...")

                    _pip_timer.cancel()
                    proc.wait(timeout=30)
                    _unregister_job_process(job_id)
                    logger.debug(f"pip exit code: {proc.returncode}")

                    if proc.returncode == 0:
                        pip_ok = True
                        break  # This permission variant worked
                    else:
                        last_error = "\n".join(lines[-8:])
                        logger.warning(f"Strategy {si+1} cmd variant failed (trying next): {last_error[-200:]}")
                        continue  # Try next permission variant (--user, --target)

                except Exception as e:
                    _unregister_job_process(job_id)
                    last_error = str(e)
                    logger.warning(f"Strategy {si+1} cmd variant exception: {e}")
                    continue  # Try next permission variant

            if not pip_ok:
                logger.warning(f"Strategy {si+1} failed all permission variants")
                continue  # Try next strategy

            # Verify import
            try:
                _update_job(job_id, progress=90, message="Verifying import...")

                verify_cmd = strat.get("verify", None)
                actual_backend = strat.get("backend_name", backend)

                if verify_cmd is None:
                    if actual_backend == "faster-whisper":
                        verify_cmd = [_pip_python, "-c", "from faster_whisper import WhisperModel; print('ok')"]
                    elif actual_backend == "openai-whisper":
                        verify_cmd = [_pip_python, "-c", "import whisper; print('ok')"]
                    else:
                        if actual_backend not in allowed:
                            last_error = f"Unknown backend: {actual_backend}"
                            continue
                        verify_cmd = [_pip_python, "-c", f"import {actual_backend}; print('ok')"]

                # Also try importing directly in-process (for --target installs)
                verify = _sp.run(
                    verify_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                if verify.returncode == 0 and "ok" in verify.stdout:
                    logger.info(f"Whisper installed via strategy {si+1}: {actual_backend}")
                    return {"backend": actual_backend, "installed": True}
                else:
                    last_error = f"Import failed: {verify.stderr[:300]}"
                    logger.warning(f"Strategy {si+1} verify failed: {last_error}")
                    continue

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Strategy {si+1} verify exception: {e}")
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

        logger.error(f"All whisper install strategies failed. Last error: {last_error[:500]}")
        raise RuntimeError(helpful)
    finally:
        _unregister_job_process(job_id)


# ---------------------------------------------------------------------------
# Whisper Settings & Reinstall
# ---------------------------------------------------------------------------
@system_bp.route("/whisper/settings", methods=["GET", "POST"])
@require_csrf
def whisper_settings():
    """Get or update Whisper settings (CPU mode, default model)."""
    if request.method == "GET":
        settings = load_whisper_settings()
        return jsonify(settings)

    # POST - update settings
    data = get_json_dict() if request.data else {}
    settings = load_whisper_settings()

    if "cpu_mode" in data:
        settings["cpu_mode"] = safe_bool(data["cpu_mode"], False)
    if "model" in data:
        _m = str(data["model"])
        if _m in VALID_WHISPER_MODELS:
            settings["model"] = _m

    save_whisper_settings(settings)
    logger.info(f"Whisper settings updated: {settings}")

    return jsonify({"success": True, "settings": settings})


@system_bp.route("/whisper/clear-cache", methods=["POST"])
@require_csrf
def whisper_clear_cache():
    """Clear downloaded Whisper model cache."""
    data = get_json_dict() if request.data else {}
    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    cache_plan = _build_whisper_cache_plan()
    plan = build_destructive_plan(
        "whisper.clear_cache",
        targets=cache_plan["entries"],
        metadata={"route": "/whisper/clear-cache", "total_bytes": cache_plan["total_bytes"]},
        reversible=False,
    )
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "plan": cache_plan["entries"],
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
            "total_bytes": cache_plan["total_bytes"],
            "cleared": [],
            "errors": cache_plan["errors"],
            "message": f"Previewed {len(cache_plan['entries'])} cache location(s)",
        })
    if plan["targets"] and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409

    cleared = []
    errors = list(cache_plan["errors"])
    for entry in plan["targets"]:
        path = str(entry.get("path", ""))
        deleted, error = _delete_cache_target(path)
        if deleted:
            cleared.append(path)
        elif error:
            errors.append(error)

    return jsonify({
        "success": len(errors) == 0,
        "dry_run": False,
        "plan": cache_plan["entries"],
        "destructive_plan": plan,
        "total_bytes": cache_plan["total_bytes"],
        "cleared": cleared,
        "errors": errors,
        "message": f"Cleared {len(cleared)} cache location(s)"
    })


@system_bp.route("/whisper/reinstall", methods=["POST"])
@require_csrf
@async_job("reinstall-whisper", filepath_required=False, rate_limit_key="model_install")
def whisper_reinstall(job_id, filepath, data):
    """Complete Whisper reinstall: uninstall, clear cache, reinstall fresh."""
    backend = data.get("backend", "faster-whisper")
    allowed_backends = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed_backends:
        raise ValueError(f"Unknown backend: {backend}")

    cpu_mode = safe_bool(data.get("cpu_mode", False), False)

    try:
        import subprocess as _sp

        # Resolve Python executable (frozen builds can't use sys.executable for pip)
        from opencut.security import _find_system_python
        if getattr(sys, "frozen", False):
            _pip_python = _find_system_python() or sys.executable
        else:
            _pip_python = sys.executable

        _target_dir = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
        os.makedirs(_target_dir, exist_ok=True)
        if _target_dir not in sys.path:
            sys.path.insert(0, _target_dir)

        def _run_pip_with_fallback(args, timeout=300):
            """Try pip command, then --user, then --target fallback."""
            base = [_pip_python, "-m", "pip"] + args
            result = None
            for variant in [base, base + ["--user"], base + ["--target", _target_dir]]:
                try:
                    result = _sp.run(
                        variant,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                    if result.returncode == 0:
                        return result
                except Exception:
                    continue
            return result  # Return last result even if failed

        # Step 1: Uninstall existing packages
        _update_job(job_id, progress=5, message="Uninstalling existing Whisper packages...")
        logger.info("Reinstall: Uninstalling existing packages")

        uninstall_pkgs = ["faster-whisper", "openai-whisper", "whisperx"]
        for pkg in uninstall_pkgs:
            with suppress(Exception):
                _sp.run(
                    [_pip_python, "-m", "pip", "uninstall", pkg, "-y"],
                    capture_output=True, timeout=60, check=False
                )

        if _is_cancelled(job_id):
            return {"backend": backend, "installed": False, "cancelled": True}

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
            return {"backend": backend, "installed": False, "cancelled": True}

        # Step 3: Clear pip cache for these packages
        _update_job(job_id, progress=30, message="Clearing pip cache...")
        with suppress(Exception):
            _sp.run(
                [_pip_python, "-m", "pip", "cache", "remove", "faster_whisper"],
                capture_output=True, timeout=30, check=False
            )
            _sp.run(
                [_pip_python, "-m", "pip", "cache", "remove", "ctranslate2"],
                capture_output=True, timeout=30, check=False
            )

        if _is_cancelled(job_id):
            return {"backend": backend, "installed": False, "cancelled": True}

        # Step 4: Install fresh (with permission fallbacks)
        _update_job(job_id, progress=40, message=f"Installing {backend} fresh...")
        logger.info(f"Reinstall: Installing {backend}")

        if backend == "faster-whisper":
            install_base = ["install", "faster-whisper", "--force-reinstall", "--no-cache-dir"]

            # For CPU mode, also install CPU-only ctranslate2
            if cpu_mode:
                _update_job(job_id, progress=45, message="Installing CPU-optimized version...")
                _run_pip_with_fallback(
                    ["install", "ctranslate2", "--force-reinstall", "--no-cache-dir"],
                    timeout=300
                )
        else:
            install_base = ["install", backend, "--force-reinstall", "--no-cache-dir"]

        # Try each permission variant for the main install
        install_ok = False
        for install_cmd in [
            [_pip_python, "-m", "pip"] + install_base,
            [_pip_python, "-m", "pip"] + install_base + ["--user"],
            [_pip_python, "-m", "pip"] + install_base + ["--target", _target_dir],
        ]:
            proc = _sp.Popen(install_cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True)
            _register_job_process(job_id, proc)

            for line in proc.stdout:
                if _is_cancelled(job_id):
                    proc.kill()
                    _unregister_job_process(job_id)
                    return {"backend": backend, "installed": False, "cancelled": True}
                line = line.strip().lower()
                if "downloading" in line:
                    _update_job(job_id, progress=55, message="Downloading packages...")
                elif "installing" in line:
                    _update_job(job_id, progress=70, message="Installing packages...")

            try:
                proc.wait(timeout=600)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)
            _unregister_job_process(job_id)

            if proc.returncode == 0:
                install_ok = True
                break
            logger.warning("Reinstall pip variant failed, trying next")

        if not install_ok:
            raise RuntimeError(
                "pip install failed -- permission denied on all attempts. "
                "Try running as administrator or: pip install faster-whisper --user --force-reinstall"
            )

        # Step 5: Save CPU mode setting
        _update_job(job_id, progress=85, message="Saving settings...")
        settings = load_whisper_settings()
        settings["cpu_mode"] = cpu_mode
        save_whisper_settings(settings)

        # Step 6: Verify
        _update_job(job_id, progress=90, message="Verifying installation...")

        if backend == "faster-whisper":
            verify_cmd = [_pip_python, "-c", "from faster_whisper import WhisperModel; print('ok')"]
        else:
            verify_cmd = [_pip_python, "-c", "import whisper; print('ok')"]

        verify = _sp.run(
            verify_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        if verify.returncode == 0 and "ok" in verify.stdout:
            logger.info(f"Whisper reinstalled: {backend}, cpu_mode={cpu_mode}")
            return {"backend": backend, "cpu_mode": cpu_mode, "installed": True}
        else:
            raise RuntimeError(f"Verification failed: {verify.stderr[:200]}")

    finally:
        _unregister_job_process(job_id)


# ---------------------------------------------------------------------------
# Demucs Install
# ---------------------------------------------------------------------------
@system_bp.route("/demucs/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def install_demucs():
    """Install Demucs for AI audio separation."""
    if should_skip_install_in_testing("install"):
        return testing_install_response("demucs")

    try:
        safe_pip_install("demucs", timeout=600)
        return jsonify({"success": True, "message": "Demucs installed successfully"})
    except RuntimeError as e:
        return safe_error(e, "install_demucs")
    except Exception as e:
        return safe_error(e, "install_demucs")


# ---------------------------------------------------------------------------
# Watermark Install
# ---------------------------------------------------------------------------
@system_bp.route("/watermark/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def install_watermark():
    """Install watermark removal dependencies."""
    if should_skip_install_in_testing("install"):
        return testing_install_response("watermark")

    try:
        packages = [
            'transformers',
            'simple-lama-inpainting',
            'torch',
            'torchvision',
            'pillow',
            'opencv-python'
        ]

        failed = []
        for pkg in packages:
            try:
                safe_pip_install(pkg, timeout=600)
            except RuntimeError:
                failed.append(pkg)

        if failed:
            from opencut.errors import error_response
            return error_response("INSTALL_FAILED",
                                  f"Failed to install: {', '.join(failed)}",
                                  status=500,
                                  suggestion="Try running as administrator or install manually via pip.")
        return jsonify({"success": True, "message": "Watermark remover installed successfully"})

    except Exception as e:
        return safe_error(e, "install_watermark")


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------
@system_bp.route("/models/list", methods=["GET"])
def list_models():
    """List downloaded AI models and their sizes (cached with 5-min TTL)."""
    now = time.time()
    with _models_cache_lock:
        if _models_cache["data"] is not None and (now - _models_cache["ts"]) < _MODELS_CACHE_TTL:
            return jsonify(_models_cache["data"])

    models = []
    # Check HuggingFace cache
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hf_cache):
        for entry in os.listdir(hf_cache):
            path = os.path.join(hf_cache, entry)
            if os.path.isdir(path) and entry.startswith("models--"):
                name = entry.replace("models--", "").replace("--", "/")
                size = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        with suppress(OSError):
                            size += os.path.getsize(os.path.join(root, f))
                models.append({"name": name, "path": path, "size_mb": round(size / (1024 * 1024), 1), "source": "huggingface"})
    # Check torch hub cache
    torch_cache = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub"))
    if os.path.isdir(torch_cache):
        checkpoints = os.path.join(torch_cache, "checkpoints")
        if os.path.isdir(checkpoints):
            for f in os.listdir(checkpoints):
                fp = os.path.join(checkpoints, f)
                if os.path.isfile(fp):
                    size = os.path.getsize(fp) / (1024 * 1024)
                    models.append({"name": f, "path": fp, "size_mb": round(size, 1), "source": "torch"})
    # Check custom whisper models dir
    if WHISPER_MODELS_DIR and os.path.isdir(WHISPER_MODELS_DIR):
        for entry in os.listdir(WHISPER_MODELS_DIR):
            path = os.path.join(WHISPER_MODELS_DIR, entry)
            if os.path.isdir(path):
                size = 0
                for root, dirs, files in os.walk(path):
                    for f in files:
                        with suppress(OSError):
                            size += os.path.getsize(os.path.join(root, f))
                models.append({"name": "whisper/" + entry, "path": path, "size_mb": round(size / (1024 * 1024), 1), "source": "whisper"})

    result = {"models": models, "total_mb": round(sum(m["size_mb"] for m in models), 1)}
    with _models_cache_lock:
        _models_cache["data"] = result
        _models_cache["ts"] = now
    return jsonify(result)


@system_bp.route("/models/delete", methods=["POST"])
@require_csrf
def delete_model():
    """Delete a downloaded model."""
    try:
        data = get_json_dict()
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "code": "INVALID_INPUT",
            "suggestion": "Send a top-level JSON object in the request body.",
        }), 400
    path = data.get("path", "")
    if path and not isinstance(path, str):
        return jsonify({
            "error": "path must be a string",
            "code": "INVALID_INPUT",
            "suggestion": "Pass the exact model cache file or directory path as a string.",
        }), 400
    path = path.strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    # Validate path for traversal attacks (may be file or directory)
    try:
        path = validate_path(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Security: only allow deletion within known cache directories
    allowed_roots = [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        os.path.join(os.path.expanduser("~"), ".cache", "torch"),
    ]
    if WHISPER_MODELS_DIR:
        allowed_roots.append(WHISPER_MODELS_DIR)
    if not is_path_within_any(path, allowed_roots):
        return jsonify({"error": "Cannot delete files outside of model cache directories"}), 403
    if not os.path.isdir(path) and not os.path.isfile(path):
        return jsonify({"error": "Path not found"}), 404

    dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
    plan_entry = _cache_plan_entry(path, category="model-cache")
    entries = [plan_entry]
    plan = build_destructive_plan(
        "models.delete",
        targets=entries,
        metadata={"route": "/models/delete", "total_bytes": plan_entry["bytes"]},
        reversible=False,
    )
    if dry_run:
        return jsonify({
            "success": True,
            "dry_run": True,
            "plan": entries,
            "destructive_plan": plan,
            "confirm_token": plan["confirm_token"],
            "total_bytes": plan_entry["bytes"],
            "deleted": [],
            "errors": plan_entry.get("errors", []),
        })
    if not verify_destructive_confirm_token(plan, data.get("confirm_token")):
        return jsonify(destructive_confirmation_required_response(plan)), 409

    deleted, delete_error = _delete_cache_target(path)
    errors = list(plan_entry.get("errors", []))
    if delete_error:
        errors.append(delete_error)
    try:
        if errors or not deleted:
            return jsonify({
                "success": False,
                "dry_run": False,
                "plan": entries,
                "destructive_plan": plan,
                "total_bytes": plan_entry["bytes"],
                "deleted": [],
                "errors": errors,
            }), 500
        # Invalidate models cache after deletion
        with _models_cache_lock:
            _models_cache["data"] = None
            _models_cache["ts"] = 0
        return jsonify({
            "success": True,
            "dry_run": False,
            "plan": entries,
            "destructive_plan": plan,
            "total_bytes": plan_entry["bytes"],
            "deleted": [path],
            "errors": errors,
        })
    except Exception as e:
        return safe_error(e, "delete_model")


# ---------------------------------------------------------------------------
# LLM Status & Test
# ---------------------------------------------------------------------------
@system_bp.route("/llm/status", methods=["GET"])
def llm_status():
    """Check LLM provider availability and list Ollama models if running."""
    from opencut.checks import check_ollama_available
    ollama_ok = check_ollama_available()

    ollama_models = []
    if ollama_ok:
        try:
            from opencut.core.llm import list_ollama_models
            ollama_models = list_ollama_models()
        except Exception:
            pass

    return jsonify({
        "ollama": {"available": ollama_ok, "models": ollama_models},
        "openai": {"available": True, "note": "Requires API key"},
        "anthropic": {"available": True, "note": "Requires API key"},
    })


@system_bp.route("/llm/test", methods=["POST"])
@require_csrf
def llm_test():
    """Test LLM connectivity with a simple prompt."""
    data = get_json_dict()

    _VALID_LLM_PROVIDERS = {"ollama", "openai", "anthropic", "gemini"}
    provider = data.get("provider", "ollama").strip().lower()
    if provider not in _VALID_LLM_PROVIDERS:
        return jsonify({"success": False, "error": f"Invalid provider: {provider}. Must be one of: {', '.join(sorted(_VALID_LLM_PROVIDERS))}"}), 400
    model = data.get("model", "").strip()
    api_key = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()

    try:
        from opencut.core.llm import LLMConfig, query_llm

        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=50,
        )

        response = query_llm("Say 'Hello from OpenCut!' in exactly those words.", config)

        return jsonify({
            "success": True,
            "response": response.text,
            "provider": response.provider,
            "model": response.model,
        })
    except Exception as e:
        return safe_error(e, "llm_test")


# ---------------------------------------------------------------------------
# Update Check (GitHub releases, 1-hour cache)
# ---------------------------------------------------------------------------
_update_cache = {"data": None, "ts": 0}
_update_cache_lock = threading.Lock()
_UPDATE_CACHE_TTL = 3600  # 1 hour


@system_bp.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Return the OpenAPI 3.0 specification for this server."""
    from flask import current_app

    from opencut.openapi import generate_openapi_spec
    return jsonify(generate_openapi_spec(current_app))


@system_bp.route("/system/update-check", methods=["GET"])
def check_for_update():
    """Check GitHub for a newer release. Cached for 1 hour."""
    import json
    import urllib.request

    now = time.time()
    with _update_cache_lock:
        if _update_cache["data"] is not None and (now - _update_cache["ts"]) < _UPDATE_CACHE_TTL:
            return jsonify(_update_cache["data"])

    current = __version__
    result = {
        "current_version": current,
        "latest_version": current,
        "update_available": False,
        "release_url": "https://github.com/SysAdminDoc/OpenCut/releases",
    }

    try:
        url = "https://api.github.com/repos/SysAdminDoc/OpenCut/releases/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "OpenCut"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            _max_bytes = 10 * 1024 * 1024
            raw = resp.read(_max_bytes + 1)
            if len(raw) > _max_bytes:
                raise ValueError(f"GitHub response exceeds {_max_bytes} byte cap")
            data = json.loads(raw.decode("utf-8"))

        tag = data.get("tag_name", "").lstrip("vV")
        html_url = data.get("html_url", result["release_url"])

        if tag:
            result["latest_version"] = tag
            result["release_url"] = html_url

            def _parse_version(v: str) -> tuple:
                """Parse version like '1.16.0' or '1.16.0-rc1' into comparable tuple.

                Keeps only leading integer components, ignoring any -rc/.dev/+build
                suffix. Falls back to (0,) if no digits are present so comparison
                never raises on unexpected inputs.
                """
                parts = []
                for seg in str(v).split("."):
                    # Strip any trailing non-digit suffix (e.g. "0-rc1" -> "0")
                    digits = ""
                    for ch in seg:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        parts.append(int(digits))
                    else:
                        break
                return tuple(parts) if parts else (0,)

            try:
                current_parts = _parse_version(current)
                latest_parts = _parse_version(tag)
                result["update_available"] = latest_parts > current_parts
            except Exception as ve:
                logger.debug("Version compare failed (%s vs %s): %s", current, tag, ve)
    except Exception as exc:
        logger.debug("Update check failed: %s", exc)
        result["error"] = "offline"

    with _update_cache_lock:
        _update_cache["data"] = result
        _update_cache["ts"] = now

    return jsonify(result)


# ---------------------------------------------------------------------------
# DaVinci Resolve Integration
# ---------------------------------------------------------------------------
@system_bp.route("/resolve/status", methods=["GET"])
def resolve_status():
    """Check DaVinci Resolve connection status and project info."""
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"connected": False, "message": "DaVinci Resolve not running or not accessible"})
        info = bridge.get_project_info()
        return jsonify({"connected": True, "project": info})
    except ImportError:
        return jsonify({"connected": False, "message": "Resolve scripting module not found"})
    except Exception as e:
        return jsonify({"connected": False, "message": str(e)})


@system_bp.route("/resolve/media", methods=["GET"])
def resolve_media():
    """Get all clips from the Resolve media pool."""
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        clips = bridge.get_media_pool_clips()
        return jsonify({"media": clips, "count": len(clips)})
    except Exception as e:
        return safe_error(e, "resolve_media")


@system_bp.route("/resolve/import", methods=["POST"])
@require_csrf
def resolve_import():
    """Import one file, or a list of files, into the Resolve media pool."""
    data = get_json_dict()
    filepath = str(data.get("filepath", "") or "").strip()
    raw_paths = data.get("paths", [])
    bin_name = data.get("bin_name", "OpenCut Output")

    if raw_paths not in (None, []) and not isinstance(raw_paths, list):
        return jsonify({"error": "paths must be a list"}), 400

    if isinstance(raw_paths, list) and raw_paths:
        cleaned_paths = []
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                return jsonify({"error": "paths must contain only non-empty strings"}), 400
            try:
                cleaned_paths.append(validate_filepath(raw_path.strip()))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        try:
            from opencut.core.resolve_integration import resolve_import_media

            return jsonify(resolve_import_media(cleaned_paths))
        except Exception as exc:
            return safe_error(exc, "resolve_import")

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400
    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        success = bridge.import_file(filepath, bin_name=bin_name)
        if success:
            return jsonify({"success": True, "message": f"Imported to {bin_name}"})
        from opencut.errors import error_response
        return error_response("OPERATION_FAILED", "Resolve import failed",
                              status=500, suggestion="Check that the file format is supported by DaVinci Resolve.")
    except Exception as e:
        return safe_error(e, "resolve_import")


@system_bp.route("/resolve/markers", methods=["POST"])
@require_csrf
def resolve_markers():
    """Add markers to the current Resolve timeline."""
    data = get_json_dict()
    markers = data.get("markers", [])
    if not markers:
        return jsonify({"error": "No markers provided"}), 400
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        added = bridge.add_markers(markers)
        return jsonify({"added": added, "total": len(markers)})
    except Exception as e:
        return safe_error(e, "resolve_markers")


@system_bp.route("/resolve/timeline", methods=["GET"])
def resolve_timeline():
    """Get current Resolve timeline info."""
    try:
        from opencut.core.resolve_bridge import ResolveBridge
        bridge = ResolveBridge()
        if not bridge.is_connected():
            return jsonify({"error": "Resolve not connected"}), 503
        info = bridge.get_timeline_info()
        if info:
            return jsonify(info)
        return jsonify({"error": "No timeline open"}), 404
    except Exception as e:
        return safe_error(e, "resolve_timeline")


# ---------------------------------------------------------------------------
# Chat-Driven Editing Assistant
# ---------------------------------------------------------------------------
@system_bp.route("/chat", methods=["POST"])
@require_csrf
def chat_message():
    """Send a message to the AI editing assistant.

    Maintains conversation context across messages. Returns the assistant's
    response with any editing actions to execute.
    """
    data = get_json_dict()
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    filepath = data.get("filepath", "")
    clip_info = data.get("clip_info", {})

    if not message:
        return jsonify({"error": "No message provided"}), 400
    if len(message) > 2000:
        return jsonify({"error": "Message too long (max 2000 chars)"}), 400

    try:
        from opencut.core.chat_editor import chat
        from opencut.core.llm import LLMConfig

        # Build LLM config from settings or request
        provider = data.get("llm_provider", "ollama")
        if provider not in ("ollama", "openai", "anthropic", "gemini"):
            return jsonify({"error": "Invalid provider", "code": "INVALID_INPUT", "suggestion": "Use ollama, openai, anthropic, or gemini"}), 400
        model = data.get("llm_model", "")
        api_key = data.get("llm_api_key", "")

        llm_config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
        )

        result = chat(
            session_id=session_id,
            user_message=message,
            filepath=filepath,
            clip_info=clip_info,
            llm_config=llm_config,
        )

        return jsonify(result)

    except ImportError as e:
        return jsonify({"error": f"Chat requires LLM module: {e}"}), 400
    except Exception as e:
        logger.exception("Chat error")
        return safe_error(e, "chat_message")


@system_bp.route("/chat/clear", methods=["POST"])
@require_csrf
def chat_clear():
    """Clear a chat session's history."""
    data = get_json_dict()
    session_id = data.get("session_id", "default")
    try:
        from opencut.core.chat_editor import clear_session, list_sessions
        session = next((item for item in list_sessions() if item.get("session_id") == session_id), None)
        records = []
        if session and int(session.get("message_count", 0) or 0) > 0:
            records.append({
                "id": str(session_id),
                "category": "chat-session",
                "type": "session",
                "message_count": int(session.get("message_count", 0) or 0),
                "filepath": str(session.get("filepath", "")),
                "bytes": 0,
                "reversible": False,
            })
        plan = build_destructive_plan(
            "chat.clear_session",
            records=records,
            metadata={"route": "/chat/clear", "session_id": str(session_id)},
            reversible=False,
        )
        dry_run = safe_bool(data.get("dry_run", data.get("preview", False)), False)
        if dry_run:
            return jsonify({
                "success": True,
                "dry_run": True,
                "message": "Session clear previewed",
                "would_clear": records[0]["message_count"] if records else 0,
                "destructive_plan": plan,
                "confirm_token": plan["confirm_token"],
            })
        if records and not verify_destructive_confirm_token(plan, data.get("confirm_token")):
            return jsonify(destructive_confirmation_required_response(plan)), 409
        clear_session(session_id)
        return jsonify({"success": True, "message": "Session cleared", "destructive_plan": plan})
    except Exception as e:
        return safe_error(e, "chat_clear")


@system_bp.route("/chat/sessions", methods=["GET"])
def chat_sessions():
    """List all active chat sessions."""
    try:
        from opencut.core.chat_editor import list_sessions
        return jsonify({"sessions": list_sessions()})
    except Exception as e:
        return safe_error(e, "chat_sessions")


# ---------------------------------------------------------------------------
# Multimodal Diarization (Audio + Face)
# ---------------------------------------------------------------------------
@system_bp.route("/video/multimodal-diarize", methods=["POST"])
@require_csrf
@async_job("multimodal-diarize", rate_limit_key="ai_gpu")
def video_multimodal_diarize(job_id, filepath, data):
    """Run multimodal speaker diarization combining audio and face recognition.

    Returns speaker segments enriched with face IDs for accurate multicam switching.
    """
    num_speakers = data.get("num_speakers")
    if num_speakers is not None:
        num_speakers = safe_int(num_speakers, None, min_val=1, max_val=20)
    sample_fps = safe_float(data.get("sample_fps", 2.0), 2.0, min_val=0.5, max_val=10.0)
    min_face_confidence = safe_float(data.get("min_face_confidence", 0.5), 0.5, min_val=0.1, max_val=1.0)

    from opencut.core.multimodal_diarize import multimodal_diarize

    def _on_progress(pct, msg=""):
        if _is_cancelled(job_id):
            raise InterruptedError("Job cancelled")
        _update_job(job_id, progress=pct, message=msg)

    result = multimodal_diarize(
        filepath,
        num_speakers=num_speakers,
        sample_fps=sample_fps,
        min_face_confidence=min_face_confidence,
        on_progress=_on_progress,
    )

    # Build enriched cuts
    cuts = result.to_enriched_cuts()

    return {
        "speaker_segments": result.speaker_segments,
        "face_segments": [
            {"face_id": f.face_id, "start": f.start, "end": f.end,
             "confidence": f.confidence, "bbox": list(f.bbox)}
            for f in result.face_segments
        ],
        "mappings": [
            {"speaker": m.speaker, "face_id": m.face_id,
             "confidence": m.confidence, "overlap_seconds": m.overlap_seconds}
            for m in result.mappings
        ],
        "cuts": cuts,
        "num_speakers": result.num_speakers,
        "num_faces": result.num_faces,
    }


# ---------------------------------------------------------------------------
# AI B-Roll Generation (Text-to-Video)
# ---------------------------------------------------------------------------
def _validate_broll_prompt(data):
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return "No prompt provided"
    if len(prompt) > 500:
        return "Prompt too long (max 500 chars)"
    return None


@system_bp.route("/video/broll-generate", methods=["POST"])
@require_csrf
@async_job("broll-generate", filepath_required=False, pre_validate=_validate_broll_prompt, rate_limit_key="ai_gpu")
def video_broll_generate(job_id, filepath, data):
    """Generate a B-roll video clip from a text description using AI."""
    prompt = data.get("prompt", "").strip()

    if not prompt:
        raise ValueError("No prompt provided")
    if len(prompt) > 500:
        raise ValueError("Prompt too long (max 500 chars)")

    output_dir = data.get("output_dir", "")
    if output_dir:
        output_dir = validate_path(output_dir)
    backend = data.get("backend", "auto")
    _valid_backends = {"auto", "stable_diffusion", "dall_e", "replicate"}
    if backend not in _valid_backends:
        backend = "auto"
    seed = data.get("seed")
    if seed is not None:
        seed = safe_int(seed, None, min_val=0, max_val=2**31)
    reference_image = data.get("reference_image", "")
    if reference_image:
        reference_image = validate_filepath(reference_image)

    from opencut.core.broll_generate import generate_broll

    def _on_progress(pct, msg=""):
        if _is_cancelled(job_id):
            raise InterruptedError("Job cancelled")
        _update_job(job_id, progress=pct, message=msg)

    effective_dir = None
    if output_dir:
        try:
            effective_dir = validate_path(output_dir)
        except ValueError:
            effective_dir = None

    result = generate_broll(
        prompt=prompt,
        output_dir=effective_dir,
        backend=backend,
        seed=seed,
        reference_image=reference_image if reference_image else None,
        on_progress=_on_progress,
    )

    return {
        "output_path": result.output_path,
        "prompt": result.prompt,
        "duration": result.duration,
        "resolution": result.resolution,
        "backend": result.backend,
        "generation_time": result.generation_time,
        "seed": result.seed,
    }


@system_bp.route("/video/broll-backends", methods=["GET"])
def video_broll_backends():
    """List available text-to-video backends."""
    try:
        from opencut.core.broll_generate import get_available_backends
        return jsonify({"backends": get_available_backends()})
    except Exception:
        return jsonify({"backends": []})


# ---------------------------------------------------------------------------
# Social Media Direct Posting
# ---------------------------------------------------------------------------
_VALID_SOCIAL_UPLOAD_PLATFORMS = {"youtube", "tiktok", "instagram", "twitter", "linkedin", "snapchat", "facebook", "pinterest"}


def _magic_clips_bundle_manifest_path(data):
    return str(
        data.get("magic_clips_bundle_manifest")
        or data.get("bundle_manifest_path")
        or ""
    ).strip()


def _validate_social_upload(data):
    if _magic_clips_bundle_manifest_path(data):
        if not safe_bool(data.get("dry_run", False), False):
            return "Magic Clips bundle uploads require dry_run=true"
        return None
    platform = str(data.get("platform") or "").strip().lower()
    if not platform:
        return "No platform specified"
    if platform not in _VALID_SOCIAL_UPLOAD_PLATFORMS:
        return f"Invalid platform. Use one of: {', '.join(sorted(_VALID_SOCIAL_UPLOAD_PLATFORMS))}"
    if not str(data.get("filepath") or "").strip():
        return "No file path provided"
    return None


@system_bp.route("/social/platforms", methods=["GET"])
def social_platforms():
    """List connected social media platforms."""
    try:
        from opencut.core.social_post import get_connected_platforms
        return jsonify({"platforms": get_connected_platforms()})
    except Exception as e:
        return safe_error(e, "social_platforms")


@system_bp.route("/social/auth-url", methods=["POST"])
@require_csrf
def social_auth_url():
    """Get OAuth authorization URL for a platform."""
    data = get_json_dict()
    platform = data.get("platform", "").strip().lower()

    if platform not in ("youtube", "tiktok", "instagram"):
        return jsonify({"error": "Unsupported platform. Use: youtube, tiktok, instagram"}), 400

    try:
        from opencut.core.social_post import get_oauth_url
        url = get_oauth_url(platform)
        if not url:
            return jsonify({"error": f"OAuth not configured for {platform}. Set API credentials in env vars."}), 400
        return jsonify({"auth_url": url, "platform": platform})
    except Exception as e:
        return safe_error(e, "social_auth_url")


@system_bp.route("/social/connect", methods=["POST"])
@require_csrf
def social_connect():
    """Store OAuth credentials after authorization callback."""
    data = get_json_dict()
    platform = data.get("platform", "").strip().lower()
    access_token = data.get("access_token", "").strip()

    if not platform or not access_token:
        return jsonify({"error": "Platform and access_token required"}), 400

    if platform not in ("youtube", "tiktok", "instagram"):
        return jsonify({"error": "Unsupported platform. Use: youtube, tiktok, instagram"}), 400

    try:
        from opencut.core.social_post import store_auth
        store_auth(
            platform=platform,
            access_token=access_token,
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            user_id=data.get("user_id"),
            username=data.get("username"),
        )
        return jsonify({"success": True, "message": f"Connected to {platform}"})
    except Exception as e:
        return safe_error(e, "social_connect")


@system_bp.route("/social/disconnect", methods=["POST"])
@require_csrf
def social_disconnect():
    """Remove stored credentials for a platform."""
    data = get_json_dict()
    platform = data.get("platform", "").strip().lower()

    if not platform:
        return jsonify({"error": "Platform required"}), 400

    if platform not in ("youtube", "tiktok", "instagram"):
        return jsonify({"error": "Unsupported platform. Use: youtube, tiktok, instagram"}), 400

    try:
        from opencut.core.social_post import disconnect_platform
        disconnect_platform(platform)
        return jsonify({"success": True, "message": f"Disconnected from {platform}"})
    except Exception as e:
        return safe_error(e, "social_disconnect")


@system_bp.route("/social/upload", methods=["POST"])
@require_csrf
@async_job("social-upload", filepath_required=False, pre_validate=_validate_social_upload)
def social_upload(job_id, filepath, data):
    """Upload a video to a social media platform."""
    bundle_manifest_path = _magic_clips_bundle_manifest_path(data)
    if bundle_manifest_path:
        bundle_manifest_path = validate_filepath(bundle_manifest_path)
        platform_filter = str(data.get("platform") or "").strip().lower()
        privacy = str(data.get("privacy") or "private").strip() or "private"
        from opencut.core.social_post import build_magic_clips_social_upload_plan

        _update_job(job_id, progress=20, message="Reading Magic Clips bundle manifest...")
        plan = build_magic_clips_social_upload_plan(
            bundle_manifest_path,
            platform=platform_filter,
            candidate_ids=data.get("candidate_ids", data.get("candidate_id")),
            privacy=privacy,
        )
        _update_job(job_id, progress=100, message=f"Prepared {plan['upload_count']} social upload item(s).")
        plan["dry_run"] = True
        return plan

    platform = data.get("platform", "").strip().lower()

    title = data.get("title", "")[:100]
    description = data.get("description", "")[:5000]
    tags = data.get("tags", [])
    if isinstance(tags, list):
        tags = tags[:30]
    privacy = data.get("privacy", "private")

    from opencut.core.social_post import upload_to_platform

    def _on_progress(pct, msg=""):
        _update_job(job_id, progress=pct, message=msg)

    result = upload_to_platform(
        filepath=filepath,
        platform=platform,
        title=title,
        description=description,
        tags=tags,
        privacy=privacy,
        on_progress=_on_progress,
    )

    if result.success:
        return {
            "platform": result.platform,
            "video_id": result.video_id,
            "url": result.url,
            "upload_time": result.upload_time,
        }
    else:
        raise RuntimeError(f"Upload failed: {result.error}")


# ---------------------------------------------------------------------------
# WebSocket Bridge Control
# ---------------------------------------------------------------------------
@system_bp.route("/ws/status", methods=["GET"])
def ws_status():
    """Get WebSocket bridge status."""
    try:
        from opencut.core.ws_bridge import get_bridge
        bridge = get_bridge()
        if bridge and bridge.is_running:
            return jsonify({
                "running": True,
                "clients": bridge.client_count,
            })
        return jsonify({"running": False, "clients": 0})
    except Exception:
        return jsonify({"running": False, "clients": 0})


@system_bp.route("/ws/start", methods=["POST"])
@require_csrf
def ws_start():
    """Start the WebSocket bridge."""
    try:
        from opencut.core.ws_bridge import check_websocket_available, init_bridge
        if not check_websocket_available():
            return jsonify({"error": "websockets package not installed. pip install websockets"}), 400
        data = get_json_dict() if request.is_json else {}
        port = safe_int(data.get("port", 5680), 5680, min_val=1024, max_val=65535)
        init_bridge(port=port)
        return jsonify({"success": True, "message": f"WebSocket bridge started on port {port}"})
    except Exception as e:
        return safe_error(e, "ws_start")


@system_bp.route("/ws/stop", methods=["POST"])
@require_csrf
def ws_stop():
    """Stop the WebSocket bridge."""
    try:
        from opencut.core.ws_bridge import stop_bridge
        stop_bridge()
        return jsonify({"success": True, "message": "WebSocket bridge stopped"})
    except Exception as e:
        return safe_error(e, "ws_stop")


# ---------------------------------------------------------------------------
# Engine Registry (Multi-Engine Backend)
# ---------------------------------------------------------------------------
@system_bp.route("/engines", methods=["GET"])
def engine_list():
    """List all registered AI engine backends and their status."""
    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        return jsonify({"engines": reg.get_status()})
    except Exception as e:
        return safe_error(e, "engine_list")


@system_bp.route("/engines/preference", methods=["POST"])
@require_csrf
def engine_set_preference():
    """Set the preferred engine for a feature domain."""
    data = get_json_dict()
    domain = data.get("domain", "").strip()
    engine = str(data.get("engine", "") or "").strip()

    if not domain:
        return jsonify({"error": "domain required"}), 400

    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()

        if not engine or engine.lower() == "auto":
            reg.clear_preference(domain)
            return jsonify({
                "success": True,
                "domain": domain,
                "engine": "",
                "mode": "auto",
            })

        # Validate domain and engine exist
        info = reg.get_engine(domain, engine)
        if not info:
            available = [e.name for e in reg.get_engines(domain)]
            return jsonify({
                "error": f"Engine '{engine}' not found in domain '{domain}'",
                "available": available,
            }), 400

        reg.set_preference(domain, engine)
        return jsonify({
            "success": True,
            "domain": domain,
            "engine": engine,
            "available": info.is_available,
        })
    except Exception as e:
        return safe_error(e, "engine_set_preference")


@system_bp.route("/engines/preferences", methods=["GET"])
def engine_get_preferences():
    """Get all engine preferences."""
    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        return jsonify({"preferences": reg.export_preferences()})
    except Exception as e:
        return safe_error(e, "engine_get_preferences")


@system_bp.route("/engines/resolve", methods=["POST"])
@require_csrf
def engine_resolve():
    """Resolve which engine to use for a domain, considering availability and preferences."""
    data = get_json_dict()
    domain = data.get("domain", "").strip()
    requested = data.get("engine", "")

    if not domain:
        return jsonify({"error": "domain required"}), 400

    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        engine = reg.resolve_engine(domain, requested if requested else None)

        if engine:
            return jsonify({
                "domain": domain,
                "engine": engine.name,
                "display_name": engine.display_name,
                "available": engine.is_available,
                "vram_mb": engine.vram_mb,
                "speed": engine.speed_rating,
                "quality": engine.quality_rating,
            })
        else:
            return jsonify({"error": f"No available engine for domain: {domain}"}), 404
    except Exception as e:
        return safe_error(e, "engine_resolve")


@system_bp.route("/auth/info", methods=["GET"])
def auth_info():
    """Return non-sensitive metadata about the local auth token (F112).

    The actual token is **never** included in this response — clients
    must read it from ``~/.opencut/auth.json`` (which is created with
    0600 mode on POSIX). This endpoint exists so the panel can render a
    "Auth required" banner with the exact instructions when the
    operator switches to a remote bind.
    """
    try:
        from opencut import auth as _auth

        remote_required = _auth.is_remote_bind_enabled()
        token = _auth.current_token()
        return jsonify(
            {
                "remote_bind_enabled": remote_required,
                "auth_required_for_remote": remote_required,
                "token_issued": token is not None,
                "token_issued_at": token.issued_at if token else None,
                "token_label": token.label if token else None,
                "token_file": str(_auth.AUTH_FILE),
                "header": _auth.AUTH_HEADER,
            }
        )
    except Exception as exc:
        return safe_error(exc, "auth_info")


@system_bp.route("/auth/rotate", methods=["POST"])
@require_csrf
def auth_rotate():
    """Issue a fresh local auth token (F112)."""
    try:
        from opencut import auth as _auth

        token = _auth.rotate_token()
        return jsonify(
            {
                "ok": True,
                "issued_at": token.issued_at,
                "label": token.label,
                "token_file": str(_auth.AUTH_FILE),
                "note": "Token written to disk; read it with the same OS user that runs the server.",
            }
        )
    except Exception as exc:
        return safe_error(exc, "auth_rotate")


@system_bp.route("/system/ai-eval/<feature_id>", methods=["GET"])
def system_ai_eval(feature_id: str):
    """Return the aggregated eval history for ``feature_id`` (F120)."""
    try:
        from opencut.core.ai_eval_harness import summarise_results

        return jsonify(summarise_results(feature_id))
    except Exception as exc:
        return safe_error(exc, "system_ai_eval")


@system_bp.route("/system/ai-eval", methods=["GET"])
def system_ai_eval_list():
    """List feature_ids that have a registered evaluation (F120)."""
    try:
        from opencut.core.ai_eval_harness import list_evaluations

        defs = list_evaluations()
        return jsonify(
            {
                "count": len(defs),
                "evaluations": [
                    {
                        "feature_id": d.feature_id,
                        "description": d.description,
                        "sample_type": d.sample_type,
                        "metric_name": d.metric_name,
                    }
                    for d in defs
                ],
            }
        )
    except Exception as exc:
        return safe_error(exc, "system_ai_eval_list")


@system_bp.route("/system/eval-datasets", methods=["GET"])
def system_eval_datasets():
    """Return the F176 public eval-dataset registry.

    Query params:

    * ``modality`` — filter by one of MODALITIES (``video``, ``audio``,
      ``music``, ``speech``, ``captions``, ``interchange``, ``provenance``).
    * ``target`` — filter by a benchmark target tag (``t2v``, ``vsr``,
      ``lip_sync``, etc.).
    * ``commercial_only=true`` — return only datasets whose license
      allows commercial eval use.
    * ``compact=true`` — strip verbose fields (citation, sha256, size)
      for panel rendering.

    Downloads are gated separately by the ``OPENCUT_DOWNLOAD_EVAL=1``
    environment variable; this route is purely informational.
    """
    try:
        from opencut.core import eval_datasets

        modality = (request.args.get("modality") or "").strip().lower()
        target = (request.args.get("target") or "").strip()
        commercial_only = (request.args.get("commercial_only") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        compact = (request.args.get("compact") or "").strip().lower() in (
            "1", "true", "yes", "on",
        )

        payload = eval_datasets.manifest(include_metadata=not compact)
        datasets = payload["datasets"]
        if modality:
            datasets = [d for d in datasets if d.get("modality") == modality]
        if target:
            datasets = [
                d for d in datasets
                if target in (d.get("benchmark_targets") or ())
            ]
        if commercial_only:
            datasets = [d for d in datasets if d.get("commercial_use_ok")]

        return jsonify({
            "version": payload["version"],
            "modalities": payload["modalities"],
            "count": len(datasets),
            "auto_download_count": sum(
                1 for d in datasets if d.get("acquisition") == "auto"
            ),
            "commercial_safe_count": sum(
                1 for d in datasets if d.get("commercial_use_ok")
            ),
            "download_opt_in": eval_datasets.download_opt_in(),
            "datasets": datasets,
        })
    except Exception as exc:
        return safe_error(exc, "system_eval_datasets")


@system_bp.route("/system/eval-datasets/<dataset_id>", methods=["GET"])
def system_eval_dataset_detail(dataset_id: str):
    """Return one F176 eval-dataset entry by id, or 404 if unknown."""
    try:
        from opencut.core import eval_datasets

        entry = eval_datasets.get_dataset(dataset_id)
        if entry is None:
            return jsonify({
                "error": f"Unknown eval dataset: {dataset_id}",
                "code": "EVAL_DATASET_NOT_FOUND",
                "suggestion": "Call GET /system/eval-datasets for the full list of supported IDs.",
            }), 404
        return jsonify(entry.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_eval_dataset_detail")


@system_bp.route("/system/ai-eval/<feature_id>/compare-backends", methods=["GET"])
def system_ai_eval_compare_backends(feature_id: str):
    """Cross-backend comparison summary for ``feature_id`` (F178).

    Groups persisted evaluation runs by backend (``cpu`` / ``cuda`` /
    ``mps`` / ``rocm`` / ``directml`` / ``unknown``) and emits relative
    quality + latency + VRAM stats. The route never picks a winner —
    it returns the data so the panel can render the comparison and
    let the user choose between "fastest" and "highest quality".
    """
    try:
        from opencut.core.ai_eval_harness import compare_backends

        return jsonify(compare_backends(feature_id))
    except Exception as exc:
        return safe_error(exc, "system_ai_eval_compare_backends")


@system_bp.route("/system/crash-packet", methods=["POST"])
@require_csrf
def system_crash_packet():
    """Build a crash + recovery diagnostic packet zip (F066).

    Body fields::

        {
            "output_path": "/abs/path/to/packet.zip",
            "log_tail_lines": 500,
            "crash_tail_bytes": 20000,
            "include_jobs": true
        }

    Returns the manifest of the produced packet.
    """
    try:
        from opencut.core.crash_packet import build_packet
        from opencut.security import (
            get_json_dict,
            validate_output_path,
        )

        data = get_json_dict()
        output_path = (data.get("output_path") or "").strip()
        if not output_path:
            return jsonify({"error": "output_path required"}), 400
        try:
            output_path = validate_output_path(output_path)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        packet = build_packet(
            output_path=output_path,
            log_tail_lines=int(data.get("log_tail_lines") or 500),
            crash_tail_bytes=int(data.get("crash_tail_bytes") or 20_000),
            include_jobs=safe_bool(data.get("include_jobs"), default=True),
        )
        return jsonify(packet.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_crash_packet")


@system_bp.route("/system/project-health", methods=["POST"])
@require_csrf
def system_project_health():
    """Run a project + media health report against a directory (F011).

    Body fields::

        {
            "project_root": "/abs/path/to/project_dir",
            "media_paths": ["/abs/path/to/source.mp4"],   # optional
            "min_free_mb": 2048                            # optional
        }
    """
    try:
        from opencut.core.project_health import build_report
        from opencut.security import (
            get_json_dict,
            validate_path,
        )

        data = get_json_dict()
        project_root = (data.get("project_root") or "").strip()
        if not project_root:
            return jsonify({"error": "project_root required"}), 400
        try:
            project_root = validate_path(project_root)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        media_paths_raw = data.get("media_paths") or []
        if not isinstance(media_paths_raw, list):
            return jsonify({"error": "media_paths must be a list"}), 400

        report = build_report(
            project_root,
            media_paths=[str(p) for p in media_paths_raw],
            min_free_mb=int(data.get("min_free_mb") or 2048),
        )
        return jsonify(report.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_project_health")


@system_bp.route("/system/ocio", methods=["GET"])
def system_ocio_validate():
    """Return the OCIO validation summary (F109).

    Reports availability, the active config, roles, colour spaces,
    looks, and findings — including non-fatal warnings like
    "no ACES space found" so the panel can suggest installing the
    Studio config.
    """
    try:
        from opencut.core.ocio_validate import validate_ocio

        result = validate_ocio()
        return jsonify(result.as_dict())
    except Exception as exc:
        return safe_error(exc, "system_ocio_validate")


@system_bp.route("/system/capabilities", methods=["GET"])
def system_capabilities():
    """Return the capability profile (F106).

    Cheap probe of FFmpeg, ffprobe, GPU, disk, and the Python runtime.
    Used by the panel + CLI to surface friendly fallback messages
    *before* a job runs.
    """
    try:
        from opencut.core.capability_profile import build_profile

        return jsonify(build_profile())
    except Exception as exc:
        return safe_error(exc, "system_capabilities")


@system_bp.route("/system/audit-log", methods=["GET"])
def system_audit_log():
    """Return recent structured security rejection audit events."""
    try:
        from opencut.security_audit import read_security_events, security_audit_log_path

        limit = safe_int(request.args.get("limit", 100), default=100, min_val=1, max_val=1000)
        events = read_security_events(limit)
        return jsonify(
            {
                "events": events,
                "count": len(events),
                "log_path": security_audit_log_path(),
            }
        )
    except Exception as exc:
        return safe_error(exc, "system_audit_log")


@system_bp.route("/system/check-failures", methods=["GET", "DELETE"])
@require_csrf
def check_failures():
    """Structured failure reasons for ``check_X_available()`` probes (E5).

    GET: returns ``{check_name: {exception, message, ts}}`` for every
    probe that raised since process start (or since last DELETE). Empty
    `{}` when all probes have returned cleanly. Useful for support
    triage — users can see *why* a feature is unavailable instead of a
    silent 503.

    DELETE: clears the registry (CSRF-protected).
    """
    try:
        from opencut import checks as _checks
        if request.method == "DELETE":
            _checks.clear_check_failures()
            return jsonify({"ok": True, "cleared": True})
        return jsonify({"failures": _checks.get_check_failures()})
    except Exception as e:
        return safe_error(e, "check_failures")


@system_bp.route("/system/feature-state", methods=["GET"])
def feature_state():
    """Return the feature readiness manifest (F100).

    Used by the CEP / UXP panels to grey out actions whose backend isn't
    ``available`` instead of waiting for a 503/501 at click time. The
    response is intentionally cheap to render: it does not boot any
    heavy AI extras; the only work is calling each feature's lightweight
    ``check_X_available()`` probe.
    """
    try:
        from opencut.registry import feature_manifest

        manifest = feature_manifest()
        feature_id = request.args.get("feature_id", "").strip()
        if feature_id:
            for record in manifest["features"]:
                if record["feature_id"] == feature_id:
                    return jsonify(record)
            return jsonify({"error": f"unknown feature_id: {feature_id}"}), 404
        return jsonify(manifest)
    except Exception as e:
        return safe_error(e, "feature_state")
