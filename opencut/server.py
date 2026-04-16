"""
OpenCut Backend Server

Local HTTP server that the Premiere Pro CEP panel communicates with.
Runs on localhost:5679 and handles all processing requests.
"""

import json
import logging
import logging.handlers
import os
import socket
import subprocess as _sp
import sys
import threading as _log_threading
import time
import traceback
from contextlib import suppress

from flask import Flask, jsonify, request
from flask_cors import CORS

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
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(job_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        defaults={"job_id": ""},
    ))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(
    "  %(message)s",
    defaults={"job_id": ""},
))
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

    from opencut.security import OpenCutRequest  # noqa: E402

    _app = Flask(__name__)
    _app.request_class = OpenCutRequest
    _app.config["OPENCUT"] = config
    _app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    CORS(_app, origins=config.cors_origins)

    from opencut.errors import register_error_handlers  # noqa: E402
    register_error_handlers(_app)

    @_app.errorhandler(413)
    def handle_large_request(e):
        """Return 413 when request payload exceeds MAX_CONTENT_LENGTH."""
        max_mb = config.max_content_length / (1024 * 1024)
        return jsonify({"error": f"Request too large (max {max_mb:.0f} MB)"}), 413

    @_app.errorhandler(RuntimeError)
    def handle_runtime_error(e):
        """Return 500 for unhandled RuntimeErrors."""
        logger.exception("Unhandled RuntimeError: %s", e)
        err_msg = str(e)[:200] if str(e) else "Internal server error"
        return jsonify({"error": err_msg, "code": "INTERNAL_ERROR"}), 500

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
        return jsonify({"error": "An internal error occurred. Check server logs for details."}), 500

    # Register Blueprints (all routes are in opencut/routes/)
    from opencut.routes import register_blueprints  # noqa: E402
    register_blueprints(_app)

    # Load Plugins
    try:
        from opencut.core.plugins import load_all_plugins
        plugin_result = load_all_plugins(_app)
        if plugin_result["loaded"]:
            logger.info("Plugins: %d loaded", len(plugin_result["loaded"]))
    except Exception as e:
        logger.warning("Plugin loading failed: %s", e)

    return _app


# Backward-compat: module-level singleton used by entry points and tests
app = create_app()


# ---------------------------------------------------------------------------
# PID File Management
# ---------------------------------------------------------------------------
PID_FILE = os.path.join(os.path.expanduser("~"), ".opencut", "server.pid")


def _write_pid(port: int):
    """Write current PID and port to file so future instances can find us."""
    try:
        import tempfile
        pid_dir = os.path.dirname(PID_FILE)
        os.makedirs(pid_dir, exist_ok=True)
        # Atomic write: temp file + rename to prevent partial reads
        fd, tmp_path = tempfile.mkstemp(dir=pid_dir, suffix=".tmp", prefix="server.pid.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n{port}\n")
            os.replace(tmp_path, PID_FILE)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp_path)
            raise
        logger.debug(f"Wrote PID file: pid={os.getpid()} port={port}")
    except Exception as e:
        logger.warning(f"Could not write PID file: {e}")


def _read_pid():
    """Read PID and port from file. Returns (pid, port) or (None, None)."""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r", encoding="utf-8") as f:
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
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=5, check=False
            )
            # CSV format: "name","pid","session","session#","mem"
            # Check that the PID appears as an exact CSV field
            return f'"{pid}"' in result.stdout
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
        # First fetch CSRF token from /health
        csrf_token = ""
        try:
            health_req = urllib.request.Request(f"http://{host}:{port}/health", method="GET")
            with urllib.request.urlopen(health_req, timeout=2) as health_resp:
                health_data = json.loads(health_resp.read())
                csrf_token = health_data.get("csrf_token", "")
        except Exception:
            pass

        req = urllib.request.Request(
            f"http://{host}:{port}/shutdown",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-OpenCut-Token": csrf_token,
            },
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
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
                    capture_output=True, timeout=10, check=False)
        else:
            os.kill(pid, 9)  # SIGKILL
            # Reap zombie if it happens to be our child
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)

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
            result = _sp.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10, check=False
            )
            for line in result.stdout.splitlines():
                # Only match LISTENING state with exact port
                # Format: "  TCP    127.0.0.1:5679    0.0.0.0:0    LISTENING    12345"
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port}"):
                        pid = parts[4]
                        if pid.isdigit() and int(pid) != os.getpid():
                            print(f"  Found process {pid} on port {port}, killing...")
                            logger.info(f"Killing PID {pid} found on port {port}")
                            _sp.run(["taskkill", "/F", "/T", "/PID", pid],
                                    capture_output=True, timeout=10, check=False)
                            killed_any = True
        else:
            result = _sp.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5, check=False
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

    # --- Step 1: Send /shutdown to the target port ---
    _kill_via_shutdown_endpoint(host, port)

    if _wait_for_port(host, port, timeout=3.0):
        print("  Graceful shutdown succeeded.")
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
# Windows Toast Notification
# ---------------------------------------------------------------------------
def _show_startup_notification(port):
    """Show a Windows toast notification so user knows the server started."""
    if sys.platform != "win32":
        return
    with suppress(Exception):
        # Use PowerShell to show a native Windows toast (no extra deps needed)
        _sp.run(
            ["powershell", "-WindowStyle", "Hidden", "-Command",
             "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
             "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null; "
             "$xml = [Windows.Data.Xml.Dom.XmlDocument]::new(); "
             "$xml.LoadXml('<toast><visual><binding template=\"ToastText02\">"
             f"<text id=\"1\">OpenCut Server Running</text>"
             f"<text id=\"2\">Listening on port {port}. Open Premiere Pro to connect.</text>"
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

    app.run(host=host, port=effective_port, debug=debug, threaded=True)


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


def main():
    """Entry point for `opencut-server` console script (pyproject.toml).

    CLI flags override env vars; env vars override the built-in defaults.
    Supported env vars:
        OPENCUT_HOST   — bind address (default 127.0.0.1)
        OPENCUT_PORT   — listen port (default 5679)
        OPENCUT_DEBUG  — enable Flask debug mode when set to "true"/"1"
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
        args = parser.parse_args()

        if args.download_models is not None:
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
            sys.exit(download_models(args.download_models))
        else:
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
