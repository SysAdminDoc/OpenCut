"""
OpenCut System Routes

Health, shutdown, GPU, dependencies, model management, whisper installation.
"""

import logging
import os
import platform
import shutil
import subprocess as _sp
import sys
import tempfile
import threading
import time

from flask import Blueprint, jsonify, request, send_file

try:
    import psutil
except ImportError:
    psutil = None

from opencut import __version__
from opencut.helpers import OPENCUT_DIR, _try_import, _try_import_from
from opencut.jobs import (
    _is_cancelled,
    _list_jobs_copy,
    _new_job,
    _register_job_process,
    _safe_error,
    _unregister_job_process,
    _update_job,
    job_lock,
    jobs,
)
from opencut.security import (
    VALID_WHISPER_MODELS,
    get_csrf_token,
    rate_limit,
    rate_limit_release,
    require_csrf,
    require_rate_limit,
    safe_int,
    safe_pip_install,
    validate_filepath,
    validate_path,
)
from opencut.user_data import load_whisper_settings, save_whisper_settings

logger = logging.getLogger("opencut")

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
@system_bp.route("/health", methods=["GET"])
def health():
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
    from opencut.checks import check_silero_vad_available, check_otio_available, check_crisper_whisper_available
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

    return jsonify({
        "status": "ok",
        "version": __version__,
        "capabilities": caps,
        "csrf_token": get_csrf_token(),
    })


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


# ---------------------------------------------------------------------------
# Media Info
# ---------------------------------------------------------------------------
@system_bp.route("/info", methods=["POST"])
@require_csrf
def media_info():
    """Get media file metadata."""
    from opencut.utils.media import probe as _probe_media

    data = request.get_json(force=True)
    filepath = data.get("filepath", "").strip()

    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

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
        return _safe_error(e)


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
            capture_output=True, text=True, timeout=5,
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
            capture_output=True, text=True, timeout=5,
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
@system_bp.route("/system/dependencies", methods=["GET"])
def check_dependencies():
    """Check all optional dependencies and return their status."""
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
        "deep-translator": "deep_translator",
        "pyannote.audio": "pyannote.audio.pipelines",
        "mediapipe": "mediapipe",
        "torch": "torch",
        "onnxruntime": "onnxruntime",
    }
    for name, module in checks.items():
        try:
            mod = __import__(module.split(".")[0])
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
            deps[name] = {"installed": True, "version": str(version)}
        except ImportError:
            deps[name] = {"installed": False, "version": None}

    # Check FFmpeg
    try:
        r = _sp.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        line = r.stdout.split("\n")[0] if r.stdout else ""
        deps["ffmpeg"] = {"installed": True, "version": line}
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

    return jsonify(deps)


# ---------------------------------------------------------------------------
# File Serving (for audio preview player)
# ---------------------------------------------------------------------------
@system_bp.route("/file", methods=["GET"])
def serve_file():
    """Serve a local file for audio/video preview. Only serves from output dirs."""
    import mimetypes
    filepath = request.args.get("path", "")

    if not filepath:
        return jsonify({"error": "File not found"}), 404

    try:
        filepath = validate_filepath(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Security: only serve files from temp or opencut output directories
    abs_path = os.path.realpath(filepath)
    allowed_prefixes = [
        os.path.realpath(tempfile.gettempdir()),
        os.path.realpath(OPENCUT_DIR),
    ]
    # Ensure prefix check uses path separators to avoid partial matches
    if not any(abs_path == p or abs_path.startswith(p + os.sep) for p in allowed_prefixes):
        return jsonify({"error": "Access denied"}), 403
    mime_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    return send_file(filepath, mimetype=mime_type)


# ---------------------------------------------------------------------------
# Output Browser (recent outputs)
# ---------------------------------------------------------------------------
@system_bp.route("/outputs/recent", methods=["GET"])
def recent_outputs():
    """List recent output files from completed jobs."""
    limit = safe_int(request.args.get("limit", 20), default=20)
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
def install_whisper():
    """Install faster-whisper via pip (on-demand from panel)."""
    if not rate_limit("model_install"):
        return jsonify({"error": "Another model_install operation is already running. Please wait."}), 429
    data = request.get_json(force=True) if request.data else {}
    backend = data.get("backend", "faster-whisper")

    allowed = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed:
        rate_limit_release("model_install")
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    try:
        job_id = _new_job("install-whisper", backend)
    except Exception:
        rate_limit_release("model_install")
        raise

    def _process():
        import subprocess as _sp

        try:
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
                    return

                pct_base = int(5 + (si / len(strategies)) * 70)
                _update_job(job_id, progress=pct_base,
                            message=f"Strategy {si+1}/{len(strategies)}: {strat['label']}...")
                logger.info(f"Whisper install strategy {si+1}: {strat['label']}")

                # Run pre-commands if present (try each fallback until one succeeds)
                if "pre_cmds" in strat:
                    for pre_cmd in strat["pre_cmds"]:
                        try:
                            pre_result = _sp.run(pre_cmd, capture_output=True, text=True, timeout=120)
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

            _update_job(
                job_id, status="error",
                error=helpful,
                message="All install methods failed",
            )
            logger.error(f"All whisper install strategies failed. Last error: {last_error[:500]}")
        finally:
            rate_limit_release("model_install")
            _unregister_job_process(job_id)

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


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
    data = request.get_json(force=True) if request.data else {}
    settings = load_whisper_settings()

    if "cpu_mode" in data:
        settings["cpu_mode"] = bool(data["cpu_mode"])
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


@system_bp.route("/whisper/reinstall", methods=["POST"])
@require_csrf
def whisper_reinstall():
    """Complete Whisper reinstall: uninstall, clear cache, reinstall fresh."""
    if not rate_limit("model_install"):
        return jsonify({"error": "Another model_install operation is already running. Please wait."}), 429
    data = request.get_json(force=True) if request.data else {}
    backend = data.get("backend", "faster-whisper")

    allowed_backends = {"faster-whisper", "openai-whisper", "whisperx"}
    if backend not in allowed_backends:
        rate_limit_release("model_install")
        return jsonify({"error": f"Unknown backend: {backend}"}), 400

    cpu_mode = data.get("cpu_mode", False)

    try:
        job_id = _new_job("reinstall-whisper", backend)
    except Exception:
        rate_limit_release("model_install")
        raise

    def _process():
        import subprocess as _sp

        try:
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
                for variant in [base, base + ["--user"], base + ["--target", _target_dir]]:
                    try:
                        result = _sp.run(variant, capture_output=True, text=True, timeout=timeout)
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
                try:
                    _sp.run(
                        [_pip_python, "-m", "pip", "uninstall", pkg, "-y"],
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
                    [_pip_python, "-m", "pip", "cache", "remove", "faster_whisper"],
                    capture_output=True, timeout=30
                )
                _sp.run(
                    [_pip_python, "-m", "pip", "cache", "remove", "ctranslate2"],
                    capture_output=True, timeout=30
                )
            except Exception:
                pass

            if _is_cancelled(job_id):
                return

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
                        return
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
                logger.warning(f"Reinstall pip variant failed, trying next")

            if not install_ok:
                _update_job(
                    job_id, status="error",
                    error="pip install failed — permission denied on all attempts",
                    message="Installation failed. Try running as administrator or: pip install faster-whisper --user --force-reinstall"
                )
                return

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
        finally:
            rate_limit_release("model_install")
            _unregister_job_process(job_id)

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    with job_lock:
        if job_id in jobs:
            jobs[job_id]["_thread"] = thread

    return jsonify({"job_id": job_id, "status": "running"})


# ---------------------------------------------------------------------------
# Demucs Install
# ---------------------------------------------------------------------------
@system_bp.route("/demucs/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def install_demucs():
    """Install Demucs for AI audio separation."""
    try:
        safe_pip_install("demucs", timeout=600)
        return jsonify({"success": True, "message": "Demucs installed successfully"})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return _safe_error(e)
    finally:
        rate_limit_release("model_install")


# ---------------------------------------------------------------------------
# Watermark Install
# ---------------------------------------------------------------------------
@system_bp.route("/watermark/install", methods=["POST"])
@require_csrf
@require_rate_limit("model_install")
def install_watermark():
    """Install watermark removal dependencies."""
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
            return jsonify({"error": f"Failed to install: {', '.join(failed)}"}), 500
        return jsonify({"success": True, "message": "Watermark remover installed successfully"})

    except Exception as e:
        return _safe_error(e)
    finally:
        rate_limit_release("model_install")


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
                        try:
                            size += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
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
                        try:
                            size += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
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
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    # Validate path for traversal attacks (may be file or directory)
    try:
        path = validate_path(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Security: only allow deletion within known cache directories
    allowed_roots = [
        os.path.realpath(os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        os.path.realpath(os.path.join(os.path.expanduser("~"), ".cache", "torch")),
    ]
    if WHISPER_MODELS_DIR:
        allowed_roots.append(os.path.realpath(WHISPER_MODELS_DIR))
    if not any(path == r or path.startswith(r + os.sep) for r in allowed_roots):
        return jsonify({"error": "Cannot delete files outside of model cache directories"}), 403
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            return jsonify({"error": "Path not found"}), 404
        # Invalidate models cache after deletion
        with _models_cache_lock:
            _models_cache["data"] = None
            _models_cache["ts"] = 0
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    data = request.get_json(force=True)

    _VALID_LLM_PROVIDERS = {"ollama", "openai", "anthropic"}
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
        return jsonify({"success": False, "error": str(e)}), 500


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
            data = json.loads(resp.read().decode("utf-8"))

        tag = data.get("tag_name", "").lstrip("vV")
        html_url = data.get("html_url", result["release_url"])

        if tag:
            result["latest_version"] = tag
            result["release_url"] = html_url
            current_parts = tuple(int(x) for x in current.split("."))
            latest_parts = tuple(int(x) for x in tag.split("."))
            result["update_available"] = latest_parts > current_parts
    except Exception as exc:
        logger.debug("Update check failed: %s", exc)
        result["error"] = "offline"

    with _update_cache_lock:
        _update_cache["data"] = result
        _update_cache["ts"] = now

    return jsonify(result)
