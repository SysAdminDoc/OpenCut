"""Runtime status and dependency routes registered on the shared system blueprint."""

__all__ = [
    "shutdown_server",
    "media_info",
    "system_gpu",
    "select_system_gpu",
    "system_status",
    "gpu_recommend",
    "check_dependencies",
]

from opencut.gpu import GPUSelectionError, gpu_selection_status, save_gpu_selection

from .system import (
    __version__,
    _cancel_running_jobs,
    _kill_job_process,
    _server_start_time,
    _sp,
    get_ffmpeg_path,
    get_json_dict,
    importlib,
    job_lock,
    jobs,
    jsonify,
    logger,
    os,
    platform,
    psutil,
    request,
    require_csrf,
    safe_error,
    safe_int,
    shutil,
    system_bp,
    threading,
    time,
    validate_filepath,
)


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
    """Detect all GPUs and annotate the configured adapter with 30s cache."""
    now = time.time()
    with _gpu_cache_lock:
        if _gpu_cache["info"] is not None and (now - _gpu_cache["ts"]) < _GPU_CACHE_TTL:
            return dict(_gpu_cache["info"])
    selection = gpu_selection_status()
    devices = selection.get("devices", [])
    selected_index = selection.get("selected_index")
    selected = next(
        (device for device in devices if device.get("index") == selected_index),
        None,
    )
    gpu_info = {
        "available": bool(devices),
        "name": (selected or {}).get("name", "None"),
        "vram_mb": (selected or {}).get("memory_total_mb", 0),
        "index": selected_index,
        "configured_index": selection.get("configured_index"),
        "selected_index": selected_index,
        "selection_mode": selection.get("selection_mode", "auto"),
        "device": selection.get("device", "cpu"),
        "devices": devices,
        "selection_error": selection.get("selection_error"),
        # The panel needs this to tell "no such adapter" apart from "this build
        # has no kernels for that adapter" — two failures with opposite fixes.
        "runtime_support": selection.get("runtime_support", {}),
        "architecture": (selected or {}).get(
            "architecture",
            selection.get("faster_whisper", {}).get("architecture", "unknown"),
        ),
        "faster_whisper": selection.get("faster_whisper", {}),
    }
    with _gpu_cache_lock:
        _gpu_cache["info"] = dict(gpu_info)
        _gpu_cache["ts"] = now
    return gpu_info


@system_bp.route("/system/gpu", methods=["GET"])
def system_gpu():
    """Check GPU availability for AI features."""
    return jsonify(_detect_gpu())


@system_bp.route("/system/gpu", methods=["POST"])
@require_csrf
def select_system_gpu():
    """Persist and immediately activate a selected CUDA adapter."""
    try:
        data = get_json_dict()
        value = data.get("gpu_index", data.get("index", "auto"))
        save_gpu_selection(value)
    except GPUSelectionError as exc:
        return jsonify(exc.to_dict()), exc.status_code
    except ValueError as exc:
        return jsonify({
            "error": str(exc),
            "code": "INVALID_INPUT",
            "suggestion": "Send gpu_index as a non-negative integer or auto.",
        }), 400
    with _gpu_cache_lock:
        _gpu_cache["info"] = None
        _gpu_cache["ts"] = 0
    with _vram_cache_lock:
        _vram_cache["used_mb"] = 0
        _vram_cache["index"] = None
        _vram_cache["ts"] = 0
    return jsonify({"success": True, **gpu_selection_status()})


# ---------------------------------------------------------------------------
# System Status (lightweight, polled every 5s by frontend status bar)
# ---------------------------------------------------------------------------
_vram_cache = {"used_mb": 0, "index": None, "ts": 0}
_vram_cache_lock = threading.Lock()
_VRAM_CACHE_TTL = 30  # seconds — same as GPU info cache


def _get_vram_used(selected_index=None):
    """Get VRAM usage with 30s cache to avoid frequent nvidia-smi calls."""
    now = time.time()
    with _vram_cache_lock:
        if (
            _vram_cache["index"] == selected_index
            and (now - _vram_cache["ts"]) < _VRAM_CACHE_TTL
        ):
            return _vram_cache["used_mb"]
    used = 0
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2:
                    continue
                if selected_index is None or safe_int(parts[0]) == selected_index:
                    used = safe_int(parts[-1])
                    break
    except Exception:
        pass
    with _vram_cache_lock:
        _vram_cache["used_mb"] = used
        _vram_cache["index"] = selected_index
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
    if gpu_raw.get("available") and gpu_raw.get("selected_index") is not None:
        gpu_info["vram_used_mb"] = _get_vram_used(gpu_raw.get("selected_index"))
    gpu_info.update({
        "index": gpu_raw.get("index"),
        "configured_index": gpu_raw.get("configured_index"),
        "selected_index": gpu_raw.get("selected_index"),
        "selection_mode": gpu_raw.get("selection_mode", "auto"),
        "device": gpu_raw.get("device", "cpu"),
        "devices": gpu_raw.get("devices", []),
        "selection_error": gpu_raw.get("selection_error"),
        "architecture": gpu_raw.get("architecture", "unknown"),
        "faster_whisper": gpu_raw.get("faster_whisper", {}),
    })

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

    # Which server is actually handling traffic. A bug report that pastes the
    # console cannot otherwise tell waitress from the Werkzeug threaded server,
    # and they behave differently under concurrent SSE.
    from opencut.server import active_wsgi_server

    return jsonify({
        "connected": True,
        "uptime_seconds": int(uptime),
        "cpu_percent": cpu_percent,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "wsgi_server": active_wsgi_server(),
        "gpu": gpu_info,
        "disk_free_gb": disk_free_gb,
        "jobs": {
            "running": running,
            "queued": queued,
            "completed_today": completed_today,
        },
        "python_version": platform.python_version(),
        "server_version": __version__,
        "sqlite": _sqlite_safety_payload(),
    })


# ---------------------------------------------------------------------------
# GPU Recommendation
# ---------------------------------------------------------------------------
def _sqlite_safety_payload() -> dict:
    """FTS5 runtime floor (F309). Never raises into /system/status."""
    try:
        from opencut.core.sqlite_safety import fts5_safety_report

        return fts5_safety_report()
    except Exception as exc:  # noqa: BLE001 - status must render regardless
        logger.warning("sqlite safety report unavailable: %s", exc)
        return {}


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
        "gpu_index": gpu_info.get("selected_index"),
        "faster_whisper": gpu_info.get("faster_whisper", {}),
        "notes": []
    }
    if gpu_info["available"]:
        rec["whisper_device"] = "cuda"
        rec["whisper_compute_type"] = rec["faster_whisper"].get("compute_type", "auto")
        if rec["faster_whisper"].get("changed"):
            rec["notes"].append(rec["faster_whisper"].get("warning", ""))
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
        rec["whisper_compute_type"] = "int8"
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

    # Map of dependency display name -> (import module, pip install command).
    # The install_hint is surfaced to the panel so users know how to install
    # missing packages without leaving the UI.
    _DEP_INSTALL_HINTS = {
        "faster-whisper": 'pip install "opencut-ppro[captions]"',
        "audio-separator": 'python -m pip install -e ".[audio]"',
        "demucs": "pip install demucs",
        "pedalboard": 'pip install "opencut-ppro[audio]"',
        "deepfilternet": 'pip install "opencut-ppro[audio]"',
        "librosa": 'pip install "opencut-ppro[audio]"',
        "pydub": "pip install pydub",
        "opencv": 'pip install "opencut-ppro[video]"',
        "Pillow": 'pip install "Pillow>=12.3.0,<13"',
        "numpy": 'pip install "opencut-ppro[video]"',
        "rembg": 'pip install "opencut-ppro[ai]"',
        "realesrgan": 'pip install "opencut-ppro[ai]"',
        "gfpgan": 'pip install "opencut-ppro[ai]"',
        "insightface": 'pip install "opencut-ppro[ai]"',
        "edge-tts": 'pip install "opencut-ppro[tts]"',
        "scenedetect": 'pip install "opencut-ppro[video]"',
        "pyannote.audio": 'pip install "opencut-ppro[diarize]"',
        "nemo-toolkit": 'python -m pip install -e ".[nemo-asr]"',
        "mediapipe": 'pip install "opencut-ppro[reframe]"',
        "torch": 'pip install "opencut-ppro[torch-stack]"',
        "onnxruntime": 'pip install "opencut-ppro[ai]"',
    }

    # Backends kept only for compatibility. The dashboard states why rather
    # than presenting them as a recommended install path.
    _DEP_LEGACY_NOTES = {
        "demucs": (
            "Archived upstream on 2024-04-24. Legacy backend; new work should "
            "use audio-separator, which is the /audio/separate default."
        ),
    }

    def _apply_maintenance(entry: dict, name: str) -> None:
        """Attach the upstream maintenance record, when one is on file.

        A package can install and import perfectly while being abandoned, so
        the dashboard reports maintenance separately from availability.
        """
        from opencut.dependency_support import maintenance_status

        status = maintenance_status(name)
        if status["abandoned"]:
            entry["maintenance"] = status["state"]
            entry["maintenance_note"] = status["warning"]
            entry["maintenance_checked"] = status["checked"]

    deps = {}
    checks = {
        "faster-whisper": "faster_whisper",
        "whisperx": "whisperx",
        "audio-separator": "audio_separator",
        "demucs": "demucs",
        "pedalboard": "pedalboard",
        "deepfilternet": "df",
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
        "nemo-toolkit": "nemo",
        "mediapipe": "mediapipe",
        "torch": "torch",
        "onnxruntime": "onnxruntime",
    }
    from opencut.dependency_support import dependency_support

    for name, module in checks.items():
        support = dependency_support(name)
        try:
            mod = importlib.import_module(module.split(".")[0])
            version = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
            deps[name] = {
                "installed": True,
                "version": str(version),
                "supported": support["supported"],
                "support_reason": support["reason"],
            }
            if name in _DEP_LEGACY_NOTES:
                deps[name]["legacy_note"] = _DEP_LEGACY_NOTES[name]
            _apply_maintenance(deps[name], name)
        except ImportError:
            deps[name] = {
                "installed": False,
                "version": None,
                "supported": support["supported"],
                "support_reason": support["reason"],
                "install_hint": (
                    (support.get("install_hint") or _DEP_INSTALL_HINTS.get(name, ""))
                    if support["supported"]
                    else ""
                ),
            }
            if name in _DEP_LEGACY_NOTES:
                deps[name]["legacy_note"] = _DEP_LEGACY_NOTES[name]
            _apply_maintenance(deps[name], name)

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
        deps["ffmpeg"] = {
            "installed": False,
            "version": None,
            "install_hint": "winget install ffmpeg  |  brew install ffmpeg  |  apt install ffmpeg",
        }

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
