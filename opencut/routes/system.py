# ruff: noqa: F401,F405,I001
"""
OpenCut System Routes

Health, shutdown, GPU, dependencies, model management, whisper installation.
"""

import importlib
import logging
import os
import platform
import shutil
import socket
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

def _user_data_dir() -> str:
    """Return the OpenCut user-data directory (resolved at call time)."""
    from opencut.user_data import OPENCUT_DIR

    return OPENCUT_DIR


#: Server-owned diagnostic targets the panels may open by name.
#:
#: The panels must never assemble a filesystem path themselves — doing so was
#: the only reason the CEP panel held Node privileges. Each entry maps a fixed
#: identifier to a resolver that returns a path this server owns, so the
#: browser context only ever sends an opaque name.
_OPEN_PATH_FIXED_TARGETS = {
    "server_log": lambda: os.path.join(_user_data_dir(), "server.log"),
    "crash_log": lambda: os.path.join(_user_data_dir(), "crash.log"),
    "security_audit_log": lambda: os.path.join(_user_data_dir(), "security_audit.jsonl"),
    "log_dir": _user_data_dir,
}


def resolve_fixed_open_target(name: str) -> str:
    """Resolve a fixed diagnostic target name to an absolute path.

    Raises ``KeyError`` for unknown names so callers fail closed rather than
    falling back to a caller-supplied path.
    """
    key = str(name or "").strip().lower()
    resolver = _OPEN_PATH_FIXED_TARGETS[key]
    return os.path.abspath(resolver())

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
    # reflects the new install. _deps_cache and _deps_cache_lock live in
    # system_runtime_routes and are imported at the bottom of this module;
    # mutating the dict in place (never rebinding it) keeps this module and
    # system_runtime_routes operating on the same shared object.
    try:
        with _deps_cache_lock:
            _deps_cache["data"] = None
            _deps_cache["ts"] = 0.0
    except NameError:
        # The bottom-of-module import from system_runtime_routes has not
        # run yet; if this function is somehow called before that, the
        # deps cache is already empty.
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

    # Stem separation is available through either backend; the maintained one
    # is python-audio-separator (Demucs was archived upstream 2024-04-24).
    from opencut.checks import check_stem_separation_available
    caps["separation"] = check_stem_separation_available()

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


def _record_bootstrap_withheld(origin: str, *, token_presented: bool) -> None:
    """Audit a refused CSRF bootstrap so the cause is diagnosable.

    Without this, a panel that cannot bootstrap reports only "Invalid or
    missing CSRF token" on every later mutation, with nothing naming the
    origin that was actually refused.
    """
    try:
        from opencut.security_audit import record_security_event

        record_security_event(
            "csrf_bootstrap_withheld",
            "CSRF bootstrap token withheld from /health for a non-allowlisted origin",
            metadata={
                "origin": origin,
                "panel_bootstrap_presented": bool(token_presented),
            },
        )
    except Exception as exc:  # noqa: BLE001 - health must answer even if auditing fails
        logger.warning("csrf bootstrap audit failed: %s", exc)


def _health_should_expose_csrf_token() -> bool:
    """Expose the bootstrap CSRF token on an allowlist basis.

    A request with no Origin header, an exact same-origin match, and explicitly
    configured CORS origins are allowed. Any other cross-origin browser context
    is denied so a hostile page cannot harvest the loopback mutation token.

    Host-embedded panels are the exception that the blocklist alone cannot
    serve: the CEP panel is a ``file://`` document, so its XHR carries
    ``Origin: null`` — indistinguishable from a hostile local page by headers
    alone. It proves itself instead by presenting the 0600 bootstrap secret
    from :mod:`opencut.panel_bootstrap`, which a browser page cannot read.
    """
    from flask import current_app

    from opencut.panel_bootstrap import BOOTSTRAP_HEADER, is_bootstrap_secret_valid

    origin = (request.headers.get("Origin") or "").strip()
    presented = request.headers.get(BOOTSTRAP_HEADER, "")
    # Checked before the blocklist: the whole point is to readmit the real
    # panel from an origin the blocklist must keep refusing for everyone else.
    if presented and is_bootstrap_secret_valid(presented):
        return True
    if not origin:
        return True
    origin_l = origin.lower()
    # The blocklist takes precedence, even over an (ill-advised) CORS entry.
    if origin_l in _CSRF_BOOTSTRAP_BLOCKED_ORIGINS:
        _record_bootstrap_withheld(origin, token_presented=bool(presented))
        return False
    origin_norm = origin_l.rstrip("/")
    host = (request.host_url or "").strip().lower().rstrip("/")
    if host and origin_norm == host:
        return True
    try:
        configured = current_app.config["OPENCUT"].cors_origins or []
    except Exception:
        configured = []
    allowed = {str(o).strip().lower().rstrip("/") for o in configured}
    if origin_norm in allowed:
        return True
    _record_bootstrap_withheld(origin, token_presented=bool(presented))
    return False


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

# Import purpose-focused route modules after the shared blueprint and helpers exist.
from .system_runtime_routes import *  # noqa: E402,F401,F403
from .system_runtime_routes import _deps_cache, _deps_cache_lock  # noqa: E402,F401
from .system_workspace_routes import *  # noqa: E402,F401,F403
from .system_whisper_routes import *  # noqa: E402,F401,F403
from .system_model_routes import *  # noqa: E402,F401,F403
from .system_integration_routes import *  # noqa: E402,F401,F403
from .system_social_routes import *  # noqa: E402,F401,F403
from .system_realtime_routes import *  # noqa: E402,F401,F403
from .system_diagnostics_routes import *  # noqa: E402,F401,F403
