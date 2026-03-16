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
import subprocess as _sp
from pathlib import Path

from flask import Flask, request, jsonify
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
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB request size limit
CORS(app, origins=["null", "file://"])  # CEP panels use null origin; file:// for local dev


from opencut.jobs import TooManyJobsError
from opencut.errors import register_error_handlers
register_error_handlers(app)


@app.errorhandler(413)
def handle_large_request(e):
    """Return 413 when request payload exceeds MAX_CONTENT_LENGTH."""
    return jsonify({"error": "Request too large (max 100 MB)"}), 413


@app.errorhandler(TooManyJobsError)
def handle_too_many_jobs(e):
    """Return 429 when the concurrent-job limit is reached."""
    return jsonify({"error": str(e)}), 429


@app.errorhandler(RuntimeError)
def handle_runtime_error(e):
    """Return 500 for unhandled RuntimeErrors."""
    logger.exception("Unhandled RuntimeError: %s", e)
    return jsonify({"error": str(e)}), 500


@app.errorhandler(500)
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


# ---------------------------------------------------------------------------
# Register Blueprints (all routes are in opencut/routes/)
# ---------------------------------------------------------------------------
from opencut.routes import register_blueprints
register_blueprints(app)


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
# Windows Toast Notification
# ---------------------------------------------------------------------------
def _show_startup_notification(port):
    """Show a Windows toast notification so user knows the server started."""
    if sys.platform != "win32":
        return
    try:
        # Use PowerShell to show a native Windows toast (no extra deps needed)
        _sp.Popen(
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
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
        )
    except Exception:
        pass  # Non-critical — if toast fails, server still runs


# ---------------------------------------------------------------------------
# Server Startup
# ---------------------------------------------------------------------------
def run_server(host="127.0.0.1", port=5679, debug=False):
    """Start the OpenCut backend server."""
    import tempfile

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
    print(f"  OpenCut Backend Server v1.2.0")
    print(f"  Listening on http://{host}:{effective_port}")
    print(f"  PID: {os.getpid()}")
    print(f"  Log file: {LOG_FILE}")
    print(f"  Press Ctrl+C to stop")
    print(f"")
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

    print(f"")
    print(f"  OpenCut Model Downloader")
    print(f"  ========================")
    print(f"")
    print(f"  Model:  Whisper '{model_size}' ({size_str})")
    print(f"  Repo:   {repo_id}")
    print(f"  Source: Hugging Face (huggingface.co)")
    print(f"")

    # Step 1: Download model files with progress
    try:
        from huggingface_hub import snapshot_download, list_repo_files

        print(f"  Fetching file list...")
        files = list_repo_files(repo_id)
        total_files = len(files)
        print(f"  Found {total_files} files to download.")
        print(f"")

        _file_count = [0]
        _start_time = [time.time()]

        def _progress_callback(step_name: str):
            """Called when tqdm updates — we intercept to show our own progress."""
            pass

        # Monkey-patch tqdm to show cleaner progress
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

        print(f"  Downloading from Hugging Face...")
        print(f"")
        snapshot_download(repo_id, local_files_only=False)
        print(f"")

        # Restore tqdm
        try:
            _tqdm_mod.tqdm = _orig_tqdm
        except Exception:
            pass

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"")
        print(f"  You can download it later from the OpenCut panel in Premiere Pro.")
        return 1

    # Step 2: Verify model loads correctly
    print(f"")
    print(f"  Verifying model...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        del model
        print(f"  [OK] Whisper '{model_size}' model downloaded and verified!")
    except Exception as e:
        print(f"  [WARNING] Model downloaded but verification failed: {e}")
        print(f"  The model may still work — try it from the OpenCut panel.")

    print(f"")
    print(f"  Models are cached in your user profile and will be")
    print(f"  available immediately when you use OpenCut.")
    print(f"")
    return 0


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description="OpenCut Backend Server")
        parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
        parser.add_argument("--port", type=int, default=5679, help="Port to listen on")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--download-models", nargs="?", const="base", default=None,
                            metavar="MODEL",
                            help="Download Whisper model (tiny/base/small/medium/large-v3/turbo)")
        args = parser.parse_args()

        if args.download_models is not None:
            # Suppress HF warnings before any huggingface imports happen
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
            sys.exit(download_models(args.download_models))
        else:
            run_server(host=args.host, port=args.port, debug=args.debug)
    except Exception as _fatal:
        # Catch startup crashes so the console window stays open
        print("")
        print("  " + "=" * 50)
        print("  FATAL ERROR — OpenCut Server failed to start")
        print("  " + "=" * 50)
        print("")
        traceback.print_exc()
        print("")
        # Also write to log file
        try:
            _crash_log = os.path.join(os.path.expanduser("~"), ".opencut", "crash.log")
            with open(_crash_log, "w", encoding="utf-8") as _f:
                _f.write(f"OpenCut Server crash at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                traceback.print_exc(file=_f)
            print(f"  Crash log saved to: {_crash_log}")
        except Exception:
            pass
        print("")
        input("  Press Enter to close...")
