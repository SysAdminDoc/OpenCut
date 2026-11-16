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
from opencut.core import capability_probe
from opencut.errors import safe_error
from opencut.helpers import _try_import, forget_retryable_import_failures, get_ffmpeg_path
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
from opencut.server_log import server_log_path
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
    "server_log": lambda: server_log_path(),
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
    from opencut.core import model_weights

    entries: list[dict] = []
    errors: list[str] = []
    cache_paths = [
        ("huggingface-whisper", model_weights.hf_cache_dir()),
        ("whisper-cache", model_weights.whisper_cache_dir()),
    ]
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        cache_paths.append(("opencut-models", os.path.join(local_appdata, "OpenCut", "models")))
    addon_repos = model_weights.addon_hf_repo_dir_names()
    for category, cache_dir in cache_paths:
        # An empty XDG_CACHE_HOME or HF_HOME (or LOCALAPPDATA off Windows)
        # leaves a path relative to whatever folder the server started in.
        if not cache_dir or not os.path.isabs(cache_dir) or not os.path.exists(cache_dir):
            continue
        if category == "huggingface-whisper":
            try:
                for item in os.listdir(cache_dir):
                    if "whisper" in item.lower() and item.lower() not in addon_repos:
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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
# The cheap capability facts only change with settings; cache them for 30 s.
# Optional native stacks are probed in a child process instead
# (opencut.core.capability_probe), so nothing here can crash the server.
# "gen" counts forgets, so a build that started before one isn't stored.
_caps_cache = {"data": None, "ts": 0.0, "gen": 0}
_caps_cache_lock = threading.Lock()
_CAPS_CACHE_TTL = 30.0
def _bootstrap_origin_blocked(origin_key: str) -> bool:
    """True for ``null`` and every ``file:`` origin, given lowercased with no trailing slash."""
    return origin_key == "null" or origin_key.startswith("file:")


def forget_capability_facts():
    """Drop the cached cheap facts after a setting they read is saved.

    Without this /health kept the old Whisper CPU mode or social login for
    up to 30 s, so the panel redrew from stale state on the next poll.
    """
    with _caps_cache_lock:
        _caps_cache["data"] = None
        _caps_cache["ts"] = 0.0
        _caps_cache["gen"] += 1


def invalidate_caps_cache():
    """Called after install routes modify available dependencies.

    Both the per-route capability cache (used by /health-style probes) and
    the heavy /system/dependencies cache need invalidation — otherwise the
    Settings tab's dependency grid keeps showing the just-installed package
    as missing for up to 60 seconds, and the user re-clicks Install.
    """
    forget_capability_facts()
    # An install can supply a dependency an optional module was missing.
    forget_retryable_import_failures()
    # ...and changes what the probe child can import, so probe again.
    capability_probe.invalidate()
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
    """Capability facts that need no optional import.

    Everything here is a constant, a setting or a file lookup. The optional
    native stacks (Torch, ONNX Runtime, TensorFlow and what sits on them) are
    probed in a child process by :mod:`opencut.core.capability_probe`, so a
    library that aborts on import takes down that child, not the server.
    """
    caps = {
        "silence": True,
        "zoom": True,
        "ffmpeg": True,
        "audio_suite": True,
        "scene_detect": True,
        # FFmpeg-based, always available.
        "video_fx": True,
        "burnin": True,
        # OpenCV DNN, a core dependency.
        "style_transfer": True,
    }
    try:
        caps["whisper_cpu_mode"] = load_whisper_settings().get("cpu_mode", False)
    except Exception:
        caps["whisper_cpu_mode"] = False

    # Social posting depends on saved credentials, not on an optional import.
    from opencut.checks import check_social_post_available
    caps["social_post"] = check_social_post_available()

    # Engine registry status (find_spec lookups only, no imports).
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


def _cached_core_capabilities():
    with _caps_cache_lock:
        if _caps_cache["data"] is not None and (time.time() - _caps_cache["ts"]) < _CAPS_CACHE_TTL:
            return _caps_cache["data"]
        generation = _caps_cache["gen"]
    caps = _build_capabilities()
    with _caps_cache_lock:
        # A setting saved while this built may not be in it. Answer with it
        # once, but don't keep it for 30 s over the save.
        if _caps_cache["gen"] == generation:
            _caps_cache["data"] = caps
            _caps_cache["ts"] = time.time()
    return caps


def _capability_discovery_allowed() -> bool:
    """Discovery spawns child processes, so test apps opt in explicitly."""
    from flask import current_app, has_app_context

    if not has_app_context():
        return False
    configured = current_app.config.get("OPENCUT_CAPABILITY_DISCOVERY")
    if configured is not None:
        return bool(configured)
    return not current_app.config.get("TESTING", False)


def current_capabilities():
    """Cheap facts plus every probed capability known so far.

    Never waits for discovery: a probe that hasn't finished leaves its keys
    out, which clients read as "not known yet". Returns ``(caps, discovery)``.
    """
    discovery = capability_probe.get_discovery()
    if _capability_discovery_allowed():
        discovery.ensure_started()
    caps = dict(_cached_core_capabilities())
    caps.update(discovery.capabilities())
    # Weights come last: they read the probed package results, and a feature
    # whose packages are installed but whose model isn't on disk turns false.
    # File stats and package metadata only, so this stays import-free.
    from opencut.core import model_weights
    model_weights.apply_to_capabilities(caps)
    return caps, discovery


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


def _health_presents_auth_token() -> bool:
    """True when the request carries a valid ``X-OpenCut-Auth`` token."""
    from opencut import auth as _auth

    presented = _auth.extract_request_token(request.headers)
    return bool(presented) and _auth.is_token_valid(presented)


def _offer_secret_to_uxp_panel() -> None:
    """Refresh the UXP panel's copy of the secret after a refused bootstrap.

    The panel's data folder may not have existed when the server started.
    """
    try:
        from opencut.panel_bootstrap import mirror_secret_to_uxp

        mirror_secret_to_uxp()
    except Exception as exc:  # noqa: BLE001 - health must answer even if the copy fails
        logger.warning("panel bootstrap mirror failed: %s", exc)


def _health_should_expose_csrf_token() -> bool:
    """Hand the CSRF token only to callers that prove they're this user.

    Proof is the bootstrap secret from :mod:`opencut.panel_bootstrap` (a file
    only the account that started the server can read) or the remote auth
    token. Headers alone prove nothing on loopback: any local process, in any
    user's session, can send no ``Origin`` or a copy of the server's own
    origin, so neither earns a token any more (F471).

    ``OPENCUT_CORS_ORIGINS`` is the one exception. An operator who lists a
    web origin there has chosen to let that site's pages bootstrap without a
    secret. ``null`` and ``file://`` are refused even when listed, because a
    hostile local page presents exactly those.
    """
    from flask import current_app

    from opencut.panel_bootstrap import BOOTSTRAP_HEADER, is_bootstrap_secret_valid

    origin = (request.headers.get("Origin") or "").strip()
    presented = request.headers.get(BOOTSTRAP_HEADER, "")
    # Checked before the blocklist: the whole point is to readmit the real
    # panel from an origin the blocklist must keep refusing for everyone else.
    if presented and is_bootstrap_secret_valid(presented):
        return True
    if _health_presents_auth_token():
        return True
    if origin:
        origin_key = origin.lower().rstrip("/")
        try:
            configured = current_app.config["OPENCUT"].cors_origins or []
        except Exception:
            configured = []
        allowed = {str(o).strip().lower().rstrip("/") for o in configured}
        # The blocklist takes precedence, even over an (ill-advised) CORS entry.
        # It compares the same trimmed form as the allow list, or ``file:///``
        # and ``null/`` would slip past it and match a listed entry.
        if not _bootstrap_origin_blocked(origin_key) and origin_key in allowed:
            return True
        _record_bootstrap_withheld(origin, token_presented=bool(presented))
    # No Origin is a native client. They poll /health for liveness, so the
    # refusal isn't audited, but the UXP panel may be waiting for its copy.
    _offer_secret_to_uxp_panel()
    return False


@system_bp.route("/health", methods=["GET"])
def health():
    # Liveness first: this answers from cheap facts and never imports an
    # optional native package. Discovery runs (once) in a child process.
    caps, discovery = current_capabilities()
    payload = {
        "status": "ok",
        "version": __version__,
        "capabilities": caps,
        "capability_discovery": discovery.snapshot(),
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
