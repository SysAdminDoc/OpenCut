"""
OpenCut Backend Server

Local HTTP server that the Premiere Pro CEP panel communicates with.
Runs on localhost:5679 and handles all processing requests.
"""

import ipaddress
import logging
import logging.handlers
import os
import subprocess as _sp
import sys
import threading as _log_threading
import time
import traceback
from contextlib import suppress

from flask import Flask, jsonify, request
from flask_cors import CORS

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

logger = logging.getLogger("opencut")
logger.setLevel(logging.DEBUG)

# Job-ID correlation filter — injects job_id into every log record
_log_thread_local = _log_threading.local()

class _JobLogFilter(logging.Filter):
    def filter(self, record):
        record.job_id = getattr(_log_thread_local, "job_id", "")
        return True

_job_filter = _JobLogFilter()
logger.addFilter(_job_filter)

# JSON formatter for structured log file (Phase 0.3)
try:
    from pythonjsonlogger import jsonlogger

    class _OpenCutJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record["level"] = record.levelname
            log_record["module"] = record.module
            log_record["job_id"] = getattr(record, "job_id", "")
            log_record.pop("levelname", None)
            log_record.pop("taskName", None)

    _json_formatter = _OpenCutJsonFormatter(
        "%(asctime)s %(levelname)s %(module)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
except ImportError:
    _json_formatter = None

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(logging.DEBUG)
if _json_formatter:
    _file_handler.setFormatter(_json_formatter)
else:
    # _JobLogFilter (attached above) injects ``record.job_id`` on every
    # record, so the format string is always resolvable without relying on
    # Formatter defaults.
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(job_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("  %(message)s"))
logger.addHandler(_console_handler)

# ---------------------------------------------------------------------------
# Bundled Installation Detection (centralized in opencut.config)
# ---------------------------------------------------------------------------
from opencut.config import OpenCutConfig  # noqa: E402

_default_config = OpenCutConfig.from_env()

# Backward-compat aliases — existing code may reference these module globals
BUNDLED_MODE = _default_config.bundled_mode
WHISPER_MODELS_DIR = _default_config.whisper_models_dir
TORCH_HOME = _default_config.torch_home
FLORENCE_MODEL_DIR = _default_config.florence_model_dir
LAMA_MODEL_DIR = _default_config.lama_model_dir

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

# ---------------------------------------------------------------------------
# FFmpeg Path Resolution
# ---------------------------------------------------------------------------
# Check for bundled ffmpeg next to the exe (installer puts it in {app}\ffmpeg\)
# or in a sibling ffmpeg/ folder. Prepend to PATH so all subprocess calls find it.
def _setup_ffmpeg_path():
    """Find bundled FFmpeg and prepend its directory to PATH."""
    candidates = []
    # When running as PyInstaller exe, sys._MEIPASS is the temp extract dir
    # but the ffmpeg folder is next to the exe, not inside the bundle
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(os.path.join(exe_dir, "..", "ffmpeg"))  # {app}\ffmpeg from {app}\server\exe
        candidates.append(os.path.join(exe_dir, "ffmpeg"))
    # When running from source
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ffmpeg"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg"))

    for candidate in candidates:
        ffmpeg_exe = os.path.join(candidate, "ffmpeg.exe") if sys.platform == "win32" else os.path.join(candidate, "ffmpeg")
        if os.path.isfile(ffmpeg_exe):
            resolved = os.path.abspath(candidate)
            os.environ["PATH"] = resolved + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"  Bundled FFmpeg: {resolved}")
            return True
    return False

_bundled_ffmpeg = _setup_ffmpeg_path()
if not _bundled_ffmpeg:
    # Verify FFmpeg is available on system PATH
    import shutil
    if shutil.which("ffmpeg"):
        logger.info("  FFmpeg: system PATH")
    else:
        logger.warning("  FFmpeg: NOT FOUND — many features will not work")

# Pre-warm FFmpeg/FFprobe path cache now that PATH is fully configured
from opencut.helpers import get_ffmpeg_path, get_ffprobe_path  # noqa: E402

get_ffmpeg_path()
get_ffprobe_path()


# ---------------------------------------------------------------------------
# System Site-Packages Discovery (frozen builds)
# ---------------------------------------------------------------------------
def _setup_system_site_packages():
    """Add system Python's site-packages to sys.path for frozen builds.

    When running as a PyInstaller exe, optional packages installed via pip
    into the system Python are invisible. This finds system Python in PATH,
    queries its site-packages dirs, and appends them so _try_import() can
    discover packages like auto-editor, mediapipe, edge-tts, etc.
    """
    if not getattr(sys, "frozen", False):
        return

    import shutil
    for name in ("python", "python3", "py"):
        python = shutil.which(name)
        if not python:
            continue
        try:
            result = _sp.run(
                [python, "-c", "import site, json; print(json.dumps(site.getsitepackages()))"],
                capture_output=True, text=True, timeout=10, check=False
            )
            if result.returncode == 0:
                import json
                paths = json.loads(result.stdout.strip())
                added = 0
                for p in paths:
                    if os.path.isdir(p) and p not in sys.path:
                        sys.path.append(p)
                        added += 1
                # Also check user site-packages
                result2 = _sp.run(
                    [python, "-c", "import site; print(site.getusersitepackages())"],
                    capture_output=True, text=True, timeout=10, check=False
                )
                if result2.returncode == 0:
                    user_sp = result2.stdout.strip()
                    if os.path.isdir(user_sp) and user_sp not in sys.path:
                        sys.path.append(user_sp)
                        added += 1
                if added:
                    logger.info("  System site-packages: added %d paths from %s", added, python)
                return
        except Exception as e:
            logger.debug("  Could not query site-packages from %s: %s", name, e)

    logger.debug("  System Python not found — optional deps from pip unavailable")


# Ensure ~/.opencut/packages (pip --target fallback) is importable EARLY —
# this is the primary writable install directory for non-admin users
_opencut_packages = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
os.makedirs(_opencut_packages, exist_ok=True)
if _opencut_packages not in sys.path:
    sys.path.insert(0, _opencut_packages)
    logger.info("  Added ~/.opencut/packages to sys.path (priority)")

_setup_system_site_packages()

# Blueprints handle their own imports; this block pre-loads for backward compat
try:
    from .core.silence import detect_speech, get_edit_summary  # noqa: F401
    from .core.zoom import generate_zoom_events  # noqa: F401
    from .export.premiere import export_premiere_xml  # noqa: F401
    from .export.srt import export_ass, export_json, export_srt, export_vtt, rgb_to_ass_color  # noqa: F401
    from .utils.config import CaptionConfig, ExportConfig, SilenceConfig, get_preset  # noqa: F401
    from .utils.media import probe as _probe_media  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Optional observability — Sentry / GlitchTip
# ---------------------------------------------------------------------------

_SENTRY_INITIALISED = False


def _init_sentry_if_configured() -> bool:
    """Initialise Sentry / GlitchTip when configured via env.

    Reads:
    - ``SENTRY_DSN`` — GlitchTip or Sentry DSN. Empty / unset → skipped.
    - ``OPENCUT_SENTRY_ENV`` — environment tag (default: "production").
    - ``OPENCUT_SENTRY_RELEASE`` — release tag (default: current version).
    - ``OPENCUT_SENTRY_SAMPLE_RATE`` — float 0..1 (default: 1.0).

    Safe when ``sentry_sdk`` isn't installed — logs at INFO and returns
    ``False``. Idempotent — returns ``True`` on re-entry once initialised.
    """
    global _SENTRY_INITIALISED
    if _SENTRY_INITIALISED:
        return True

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    try:
        import sentry_sdk  # type: ignore
        from sentry_sdk.integrations.flask import FlaskIntegration  # type: ignore
    except ImportError:
        logger.info(
            "SENTRY_DSN set but `sentry_sdk` not installed — "
            "observability disabled. pip install sentry-sdk[flask]"
        )
        return False

    try:
        sample_rate = float(os.environ.get("OPENCUT_SENTRY_SAMPLE_RATE") or "1.0")
    except (TypeError, ValueError):
        sample_rate = 1.0
    sample_rate = max(0.0, min(1.0, sample_rate))

    env = os.environ.get("OPENCUT_SENTRY_ENV", "production")
    try:
        from opencut import __version__ as _ver
    except Exception:  # noqa: BLE001
        _ver = "unknown"
    release = os.environ.get("OPENCUT_SENTRY_RELEASE", f"opencut@{_ver}")

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=env,
            release=release,
            sample_rate=sample_rate,
            traces_sample_rate=sample_rate,
            integrations=[FlaskIntegration()],
            # Don't capture request bodies — they often contain media
            # paths that are irrelevant to crash triage.
            send_default_pii=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentry_sdk.init failed: %s — observability disabled", exc)
        return False

    _SENTRY_INITIALISED = True
    logger.info(
        "Sentry initialised (env=%s, release=%s, sample_rate=%.2f)",
        env, release, sample_rate,
    )
    return True


# ---------------------------------------------------------------------------
# Flask App Factory
# ---------------------------------------------------------------------------
def create_app(config=None):
    """Create and configure the Flask application.

    Args:
        config: An ``OpenCutConfig`` instance.  When *None* (the default),
            the module-level ``_default_config`` built from env vars is used.

    Returns:
        Configured Flask app with all blueprints registered.
    """
    if config is None:
        config = _default_config

    # Keep jobs.py runtime limits in sync with the app config used for this
    # process, including tests that supply a custom OpenCutConfig instance.
    try:
        from opencut.jobs import apply_config as _apply_job_config

        _apply_job_config(config)
    except Exception as e:
        logger.warning("Could not apply job runtime config: %s", e)

    # Optional observability — Sentry / GlitchTip. Driven by env vars so
    # deployments opt in without the module dependency becoming a hard
    # requirement. Safe to call multiple times (sentry_sdk.init is
    # idempotent-ish — subsequent calls just reconfigure the hub).
    _init_sentry_if_configured()

    # Startup + periodic sweep of stale opencut_* temp files. Best-effort;
    # failure never blocks app startup.
    try:
        from opencut.core import temp_cleanup as _temp_cleanup
        _temp_cleanup.run_startup_sweep()
        _temp_cleanup.start_background_sweep()
    except Exception as _cleanup_exc:  # noqa: BLE001
        logger.warning("temp_cleanup bootstrap failed: %s", _cleanup_exc)

    # Optional disk-space background monitor (off by default, opt-in
    # via OPENCUT_DISK_MONITOR_INTERVAL).
    try:
        from opencut.core import disk_monitor as _disk_monitor
        _disk_monitor.start_background()
    except Exception as _disk_exc:  # noqa: BLE001
        logger.warning("disk_monitor bootstrap failed: %s", _disk_exc)

    from opencut.security import OpenCutRequest  # noqa: E402

    _app = Flask(__name__)
    _app.request_class = OpenCutRequest
    _app.config["OPENCUT"] = config
    _app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    CORS(_app, origins=config.cors_origins)

    from opencut.errors import register_error_handlers  # noqa: E402
    register_error_handlers(_app)

    # Per-request correlation ID middleware — must run before route
    # blueprints register so every view sees the populated ``g.request_id``.
    try:
        from opencut.core.request_correlation import install_middleware
        install_middleware(_app)
    except Exception as _rc_exc:  # noqa: BLE001
        logger.warning("request_correlation install failed: %s", _rc_exc)

    # F112: per-install API token gate for non-loopback binds. The gate
    # only fires when ``OPENCUT_ALLOW_REMOTE=1`` is set AND the request
    # peer is non-loopback — local panel/CLI traffic stays trusted.
    try:
        from opencut import auth as _auth

        _AUTH_EXEMPT_PATHS = frozenset({"/health", "/auth/info"})

        @_app.before_request
        def _enforce_remote_auth_token():  # noqa: D401 - tiny middleware
            if request.path in _AUTH_EXEMPT_PATHS:
                return None
            if not _auth.request_requires_auth_token(request.remote_addr):
                return None
            token = _auth.extract_request_token(request.headers, request.args)
            if _auth.is_token_valid(token):
                return None
            return jsonify(
                {
                    "error": "Missing or invalid X-OpenCut-Auth token",
                    "code": "AUTH_REQUIRED",
                    "suggestion": (
                        "OPENCUT_ALLOW_REMOTE=1 is enabled. Read the token from "
                        "~/.opencut/auth.json or rotate it via "
                        "`opencut-server --rotate-auth`."
                    ),
                }
            ), 401

    except Exception as _auth_exc:  # noqa: BLE001
        logger.warning("auth middleware install failed: %s", _auth_exc)

    @_app.errorhandler(413)
    def handle_large_request(e):
        """Return 413 when request payload exceeds MAX_CONTENT_LENGTH."""
        max_mb = config.max_content_length / (1024 * 1024)
        return jsonify({
            "error": f"Request too large (max {max_mb:.0f} MB)",
            "code": "REQUEST_TOO_LARGE",
            "suggestion": "Reduce the payload size or process the file in smaller batches.",
        }), 413

    @_app.errorhandler(RuntimeError)
    def handle_runtime_error(e):
        """Return a structured error for unhandled RuntimeErrors.

        Delegates to ``safe_error`` so FFmpeg/ImportError/MemoryError
        RuntimeErrors get classified into specific codes with recovery
        suggestions. Falls back to a generic INTERNAL_ERROR response if
        classification fails.
        """
        logger.exception("Unhandled RuntimeError: %s", e)
        try:
            from opencut.errors import safe_error as _safe_error
            return _safe_error(e, context="runtime_error_handler")
        except Exception:
            err_msg = str(e)[:200] if str(e) else "Internal server error"
            return jsonify({
                "error": err_msg,
                "code": "INTERNAL_ERROR",
                "suggestion": "Check the server logs for details.",
            }), 500

    @_app.errorhandler(500)
    def handle_internal_error(e):
        """Log unhandled 500 errors to crash log for post-mortem debugging."""
        import traceback as _tb
        logger.exception("Unhandled 500 error: %s", e)
        try:
            _crash_log = os.path.join(LOG_DIR, "crash.log")
            with open(_crash_log, "a", encoding="utf-8") as _f:
                _f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                _f.write(f"Endpoint: {request.path} [{request.method}]\n")
                _tb.print_exc(file=_f)
        except Exception:
            pass
        return jsonify({
            "error": "An internal error occurred. Check server logs for details.",
            "code": "INTERNAL_ERROR",
            "suggestion": "Retry the request; if it persists check ~/.opencut/crash.log.",
        }), 500

    # Register Blueprints (all routes are in opencut/routes/)
    from opencut.routes import assert_no_route_collisions, register_blueprints  # noqa: E402
    register_blueprints(_app)

    # Load Plugins
    try:
        from opencut.core.plugins import load_all_plugins
        plugin_result = load_all_plugins(_app)
        if plugin_result["loaded"]:
            logger.info("Plugins: %d loaded", len(plugin_result["loaded"]))
    except Exception as e:
        logger.warning("Plugin loading failed: %s", e)

    # Route ownership must stay unambiguous. Fail fast if any later blueprint or
    # plugin registration reintroduces a method/path collision.
    assert_no_route_collisions(_app)

    return _app


# Backward-compat: module-level singleton used by entry points and tests.
# Access via `from opencut.server import app` triggers __getattr__ below,
# which creates the singleton on first use rather than at import time.
_app_singleton = None
_app_lock = _log_threading.Lock()


def _get_app():
    """Return the shared Flask app singleton, creating it on first call."""
    global _app_singleton
    with _app_lock:
        if _app_singleton is None:
            _app_singleton = create_app()
    return _app_singleton


def __getattr__(name: str):
    """Lazy attribute access for module-level names.

    Supports ``from opencut.server import app`` without triggering
    ``create_app()`` at import time.
    """
    if name == "app":
        inst = _get_app()
        # Populate __dict__ so subsequent accesses bypass this hook.
        globals()["app"] = inst
        return inst
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# PID File Management, Port Checking, and Kill Strategies
# ---------------------------------------------------------------------------
# Extracted to opencut/pid.py.  Re-exported here so any code that previously
# did `from opencut.server import _read_pid` (or similar) continues to work.
from opencut.pid import (  # noqa: E402, F401
    PID_FILE,
    _check_port,
    _is_opencut_on_port,
    _is_pid_alive,
    _kill_via_netstat,
    _kill_via_pid,
    _kill_via_shutdown_endpoint,
    _nuke_old_servers,
    _read_pid,
    _remove_pid,
    _wait_for_port,
    _write_pid,
)


# ---------------------------------------------------------------------------
# Windows Toast Notification
# ---------------------------------------------------------------------------
def _show_startup_notification(port):
    """Show a Windows toast notification so user knows the server started."""
    if sys.platform != "win32":
        return
    # Coerce to int to prevent injection into the PowerShell command string
    safe_port = int(port)
    with suppress(Exception):
        # Use PowerShell to show a native Windows toast (no extra deps needed)
        _sp.run(
            ["powershell", "-WindowStyle", "Hidden", "-Command",
             "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
             "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null; "
             "$xml = [Windows.Data.Xml.Dom.XmlDocument]::new(); "
             "$xml.LoadXml('<toast><visual><binding template=\"ToastText02\">"
             "<text id=\"1\">OpenCut Server Running</text>"
             f"<text id=\"2\">Listening on port {safe_port}. Open Premiere Pro to connect.</text>"
             "</binding></visual></toast>'); "
             "$toast = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('OpenCut'); "
             "$toast.Show([Windows.UI.Notifications.ToastNotification]::new($xml))"],
            creationflags=0x08000000,  # CREATE_NO_WINDOW
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            timeout=10, check=False
        )


# ---------------------------------------------------------------------------
# Server Startup
# ---------------------------------------------------------------------------
def run_server(host="127.0.0.1", port=5679, debug=False):
    """Start the OpenCut backend server."""
    import tempfile

    # Mark any jobs that were running when the server last died as interrupted
    try:
        from opencut.job_store import cleanup_old_jobs, mark_interrupted
        mark_interrupted()
        cleanup_old_jobs()
    except Exception as e:
        logger.warning("Job store startup failed: %s", e)

    # Clean up stale preview temp files from previous runs
    try:
        import glob as _glob
        _tmp = tempfile.gettempdir()
        _now = time.time()
        for _f in _glob.glob(os.path.join(_tmp, "opencut_preview_*.jpg")):
            try:
                if (_now - os.path.getmtime(_f)) > 3600:
                    os.unlink(_f)
            except OSError:
                pass
    except Exception:
        pass

    effective_port = port

    if not _check_port(host, port):
        # Port is busy - run the kill sequence
        if _nuke_old_servers(host, port):
            effective_port = port
        else:
            # Kill sequence failed - find an alternate port
            print("  Searching for an open port...")
            for offset in range(1, 11):
                candidate = port + offset
                if _check_port(host, candidate):
                    effective_port = candidate
                    print(f"  Using port {effective_port} instead.")
                    break
            else:
                print("")
                print(f"  ERROR: Ports {port}-{port + 10} are all in use.")
                print("  Kill any stuck python processes and try again.")
                print("")
                sys.exit(1)

    # Write PID file so future instances can kill us
    _write_pid(effective_port)

    # Register cleanup on normal exit
    import atexit
    atexit.register(_remove_pid)
    from opencut.workers import shutdown_pool
    atexit.register(shutdown_pool)
    # Close persistent SQLite connections so the WAL files aren't held
    # open on Windows after shutdown.
    try:
        from opencut.job_store import close_all_connections as _close_job_db
        atexit.register(_close_job_db)
    except ImportError:
        pass
    try:
        from opencut.core.footage_index_db import close_all_connections as _close_footage_db
        atexit.register(_close_footage_db)
    except ImportError:
        pass
    try:
        from opencut.journal import close_all_connections as _close_journal_db
        atexit.register(_close_journal_db)
    except ImportError:
        pass

    print("")
    from opencut import __version__
    print(f"  OpenCut Backend Server v{__version__}")
    print(f"  Listening on http://{host}:{effective_port}")
    print(f"  PID: {os.getpid()}")
    print(f"  Log file: {LOG_FILE}")
    print("  Press Ctrl+C to stop")
    print("")
    logger.info(f"Server starting on http://{host}:{effective_port} (pid={os.getpid()})")

    # Show Windows toast notification so user knows server started (especially
    # when launched via VBS hidden launcher where console is invisible)
    _show_startup_notification(effective_port)

    _get_app().run(host=host, port=effective_port, debug=debug, threaded=True)


def download_models(model_size="base"):
    """Download Whisper model for offline use. Called by installer."""
    import warnings
    warnings.filterwarnings("ignore")

    # Suppress noisy huggingface_hub warnings
    for name in ["huggingface_hub", "huggingface_hub.file_download",
                 "huggingface_hub.utils", "huggingface_hub._commit_api",
                 "urllib3", "filelock"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Model size info for progress display
    model_sizes = {
        "tiny": "~75 MB", "tiny.en": "~75 MB",
        "base": "~150 MB", "base.en": "~150 MB",
        "small": "~500 MB", "small.en": "~500 MB",
        "medium": "~1.5 GB", "medium.en": "~1.5 GB",
        "large-v1": "~3 GB", "large-v2": "~3 GB", "large-v3": "~3 GB",
        "turbo": "~1.6 GB", "large-v3-turbo": "~1.6 GB",
    }
    size_str = model_sizes.get(model_size, "unknown size")

    # Resolve model name to HF repo ID
    try:
        from faster_whisper.utils import _MODELS
        repo_id = _MODELS.get(model_size, model_size)
    except Exception:
        repo_id = f"Systran/faster-whisper-{model_size}"

    print("")
    print("  OpenCut Model Downloader")
    print("  ========================")
    print("")
    print(f"  Model:  Whisper '{model_size}' ({size_str})")
    print(f"  Repo:   {repo_id}")
    print("  Source: Hugging Face (huggingface.co)")
    print("")

    # Step 1: Download model files with progress
    try:
        from huggingface_hub import list_repo_files, snapshot_download

        print("  Fetching file list...")
        files = list_repo_files(repo_id)
        total_files = len(files)
        print(f"  Found {total_files} files to download.")
        print("")

        _file_count = [0]
        _start_time = [time.time()]

        def _progress_callback(step_name: str):
            """Called when tqdm updates — we intercept to show our own progress."""
            pass

        # Monkey-patch tqdm to show cleaner progress
        _orig_tqdm = None
        _tqdm_mod = None
        try:
            import tqdm as _tqdm_mod
            _orig_tqdm = _tqdm_mod.tqdm

            class _InstallerTqdm(_orig_tqdm):
                def __init__(self, *args, **kwargs):
                    kwargs["bar_format"] = "  {desc} {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} {rate_fmt}"
                    kwargs["ncols"] = 75
                    kwargs.setdefault("file", sys.stdout)
                    super().__init__(*args, **kwargs)

            _tqdm_mod.tqdm = _InstallerTqdm
        except Exception:
            pass

        try:
            print("  Downloading from Hugging Face...")
            print("")
            snapshot_download(repo_id, local_files_only=False)
            print("")
        finally:
            # Always restore tqdm even if download fails
            if _tqdm_mod is not None and _orig_tqdm is not None:
                _tqdm_mod.tqdm = _orig_tqdm

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print("")
        print("  You can download it later from the OpenCut panel in Premiere Pro.")
        return 1

    # Step 2: Verify model loads correctly
    print("")
    print("  Verifying model...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        del model
        print(f"  [OK] Whisper '{model_size}' model downloaded and verified!")
    except Exception as e:
        print(f"  [WARNING] Model downloaded but verification failed: {e}")
        print("  The model may still work — try it from the OpenCut panel.")

    print("")
    print("  Models are cached in your user profile and will be")
    print("  available immediately when you use OpenCut.")
    print("")
    return 0


def _pause_on_fatal_exit() -> None:
    """Wait for user acknowledgement only in interactive terminals."""
    stdin = getattr(sys, "stdin", None)
    try:
        if stdin is not None and stdin.isatty():
            input("  Press Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


def _is_loopback_host(host: str) -> bool:
    """Return True when ``host`` cannot expose the API beyond this machine."""
    normalized = str(host or "").strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized.strip("[]")).is_loopback
    except ValueError:
        return False


def _remote_bind_allowed() -> bool:
    return os.environ.get("OPENCUT_ALLOW_REMOTE", "").strip().lower() in _TRUE_ENV_VALUES


def main():
    """Entry point for `opencut-server` console script (pyproject.toml).

    CLI flags override env vars; env vars override the built-in defaults.
    Supported env vars:
        OPENCUT_HOST   — bind address (default 127.0.0.1)
        OPENCUT_PORT   — listen port (default 5679)
        OPENCUT_DEBUG  — enable Flask debug mode when set to "true"/"1"
        OPENCUT_ALLOW_REMOTE — set to "1" to allow non-loopback binds
    """
    try:
        import argparse

        # Env var defaults — let docker-compose / launchers configure without --flags.
        env_host = os.environ.get("OPENCUT_HOST", "127.0.0.1").strip() or "127.0.0.1"
        try:
            env_port = int(os.environ.get("OPENCUT_PORT", "5679"))
            if not (1 <= env_port <= 65535):
                env_port = 5679
        except ValueError:
            env_port = 5679
        env_debug = os.environ.get("OPENCUT_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

        parser = argparse.ArgumentParser(description="OpenCut Backend Server")
        parser.add_argument("--host", default=env_host, help="Host to bind to")
        parser.add_argument("--port", type=int, default=env_port, help="Port to listen on")
        parser.add_argument("--debug", action="store_true", default=env_debug, help="Enable debug mode")
        parser.add_argument("--download-models", nargs="?", const="base", default=None,
                            metavar="MODEL",
                            help="Download Whisper model (tiny/base/small/medium/large-v3/turbo)")
        parser.add_argument(
            "--rotate-auth",
            action="store_true",
            help="Rotate the persistent API token (~/.opencut/auth.json) and exit",
        )
        parser.add_argument(
            "--print-auth",
            action="store_true",
            help="Print the persisted API token (creating one on first call) and exit",
        )
        args = parser.parse_args()

        if args.rotate_auth or args.print_auth:
            from opencut import auth as _auth

            token = _auth.rotate_token() if args.rotate_auth else _auth.ensure_token()
            print("")
            print(f"  OpenCut API token: {token.token}")
            print(f"  Stored in:         {_auth.AUTH_FILE}")
            print(f"  Header to use:     {_auth.AUTH_HEADER}")
            print("  This token is only required for non-loopback requests when")
            print("  OPENCUT_ALLOW_REMOTE=1 is set.")
            return 0

        if args.download_models is not None:
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
            sys.exit(download_models(args.download_models))
        else:
            if not _is_loopback_host(args.host):
                if not _remote_bind_allowed():
                    print("")
                    print("  ERROR: Refusing to bind OpenCut to a non-loopback host.")
                    print(f"  Requested host: {args.host}")
                    print("  OpenCut's backend is a local, single-user API. Keep it on")
                    print("  127.0.0.1 unless you intentionally expose it on a trusted network.")
                    print("  Set OPENCUT_ALLOW_REMOTE=1 to allow a remote bind.")
                    return 2
                # F112: ensure a token exists before announcing remote binding so
                # operators can read it from ~/.opencut/auth.json on first boot.
                try:
                    from opencut import auth as _auth

                    issued = _auth.ensure_token()
                except Exception as _auth_exc:  # noqa: BLE001
                    issued = None
                    print("")
                    print(f"  WARNING: could not initialise auth token: {_auth_exc}")
                print("")
                print(f"  WARNING: Binding OpenCut to non-loopback host {args.host}.")
                print("  Non-loopback requests must carry X-OpenCut-Auth.")
                if issued is not None:
                    print(f"  Token file: {_auth.AUTH_FILE}")
                    print("  Rotate with: opencut-server --rotate-auth")
            run_server(host=args.host, port=args.port, debug=args.debug)
            return 0
    except Exception as _fatal:
        print("")
        print("  " + "=" * 50)
        print("  FATAL ERROR — OpenCut Server failed to start")
        print("  " + "=" * 50)
        print("")
        traceback.print_exc()
        print("")
        try:
            _crash_log = os.path.join(os.path.expanduser("~"), ".opencut", "crash.log")
            with open(_crash_log, "a", encoding="utf-8") as _f:
                _f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                _f.write("OpenCut Server startup crash\n\n")
                traceback.print_exc(file=_f)
            print(f"  Crash log saved to: {_crash_log}")
        except Exception:
            pass
        print("")
        _pause_on_fatal_exit()
        return 1


if __name__ == "__main__":
    sys.exit(main())
