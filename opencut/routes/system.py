"""
OpenCut System Routes

Health, shutdown, GPU, dependencies, model management, whisper installation.
"""

import logging
import os
import shutil
import subprocess as _sp
import sys
import tempfile
import threading
import time

from flask import Blueprint, jsonify, request, send_file

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
    get_csrf_token,
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
@require_rate_limit("model_install")
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

        try:
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
                            if actual_backend not in allowed:
                                last_error = f"Unknown backend: {actual_backend}"
                                continue
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
                    _unregister_job_process(job_id)
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
        settings["model"] = str(data["model"])

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
@require_rate_limit("model_install")
def whisper_reinstall():
    """Complete Whisper reinstall: uninstall, clear cache, reinstall fresh."""
    data = request.get_json(force=True) if request.data else {}
    backend = data.get("backend", "faster-whisper")
    cpu_mode = data.get("cpu_mode", False)

    job_id = _new_job("reinstall-whisper", backend)

    def _process():
        import subprocess as _sp

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

            proc.wait()
            _unregister_job_process(job_id)

            if proc.returncode != 0:
                _update_job(
                    job_id, status="error",
                    error="pip install failed",
                    message="Installation failed. Try running: pip install faster-whisper --force-reinstall"
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
