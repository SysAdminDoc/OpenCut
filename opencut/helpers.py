"""
OpenCut Shared Helpers

Utility functions used across multiple route modules: FFmpeg progress,
output path resolution, lazy imports, job time tracking.
"""

import importlib
import json
import logging
import os
import shutil
import subprocess as _sp
import tempfile
import threading
import time
import uuid

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Ensure ~/.opencut/packages is on sys.path (pip --target fallback dir)
# ---------------------------------------------------------------------------
import sys as _sys

_opencut_pkg_dir = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
if os.path.isdir(_opencut_pkg_dir) and _opencut_pkg_dir not in _sys.path:
    _sys.path.insert(0, _opencut_pkg_dir)

# ---------------------------------------------------------------------------
# FFmpeg / FFprobe Path Cache
# ---------------------------------------------------------------------------
_ffmpeg_path = None
_ffprobe_path = None


def get_ffmpeg_path() -> str:
    """Return cached path to ffmpeg binary, resolving via shutil.which on first call."""
    global _ffmpeg_path
    if _ffmpeg_path is not None:
        return _ffmpeg_path
    found = shutil.which("ffmpeg")
    if not found:
        logger.warning("ffmpeg not found in PATH — subprocess calls may fail")
    _ffmpeg_path = found if found else "ffmpeg"
    return _ffmpeg_path


def get_ffprobe_path() -> str:
    """Return cached path to ffprobe binary, resolving via shutil.which on first call."""
    global _ffprobe_path
    if _ffprobe_path is not None:
        return _ffprobe_path
    found = shutil.which("ffprobe")
    if not found:
        logger.warning("ffprobe not found in PATH — media probing may fail")
    _ffprobe_path = found if found else "ffprobe"
    return _ffprobe_path

# ---------------------------------------------------------------------------
# OpenCut user data directory
# ---------------------------------------------------------------------------
OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")


def _ensure_opencut_dir():
    """Create the OpenCut user directory if it doesn't exist."""
    os.makedirs(OPENCUT_DIR, exist_ok=True)

# Named constants
DEFAULT_CRF = 18


def output_path(input_path: str, suffix: str, output_dir: str = "") -> str:
    """Generate output path with suffix, preserving extension."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1] or ".mp4"
    directory = output_dir or os.path.dirname(input_path)
    return os.path.join(directory, f"{base}_{suffix}{ext}")

# ---------------------------------------------------------------------------
# Deferred Temp File Cleanup
# ---------------------------------------------------------------------------
_cleanup_scheduled = set()
_cleanup_schedule_lock = threading.Lock()


def _schedule_temp_cleanup(filepath: str, delay: float = 5.0, retries: int = 3):
    """Schedule a temp file for deferred deletion.

    On Windows, FFmpeg may still hold a file handle when the job thread
    finishes.  This retries deletion with exponential backoff.
    Deduplicates: won't schedule the same path twice.
    """
    norm = os.path.normcase(os.path.abspath(filepath))
    with _cleanup_schedule_lock:
        if norm in _cleanup_scheduled:
            return
        _cleanup_scheduled.add(norm)

    def _try_delete(path, attempt, max_attempts, wait):
        try:
            if os.path.isfile(path):
                os.unlink(path)
                logger.debug("Cleaned up temp file: %s", path)
            with _cleanup_schedule_lock:
                _cleanup_scheduled.discard(os.path.normcase(os.path.abspath(path)))
        except OSError:
            if attempt < max_attempts:
                threading.Timer(wait, _try_delete,
                                args=(path, attempt + 1, max_attempts, wait * 2)).start()
            else:
                with _cleanup_schedule_lock:
                    _cleanup_scheduled.discard(os.path.normcase(os.path.abspath(path)))
                logger.warning("Failed to clean up temp file after %d attempts: %s",
                               max_attempts, path)

    threading.Timer(delay, _try_delete,
                    args=(filepath, 1, retries, delay)).start()

# ---------------------------------------------------------------------------
# Disk Space Preflight Check
# ---------------------------------------------------------------------------
def check_disk_space(path: str, min_bytes: int = 500 * 1024 * 1024) -> dict:
    """Check available disk space at path. Returns dict with 'ok', 'free_bytes', 'free_mb'.

    Args:
        path: Directory or file path to check (uses its mount point).
        min_bytes: Minimum required free space in bytes (default 500 MB).
    """
    try:
        check_dir = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        usage = shutil.disk_usage(check_dir)
        return {
            "ok": usage.free >= min_bytes,
            "free_bytes": usage.free,
            "free_mb": round(usage.free / (1024 * 1024)),
            "required_mb": round(min_bytes / (1024 * 1024)),
        }
    except Exception as e:
        logger.debug("Disk space check failed for %s: %s", path, e)
        return {"ok": True, "free_bytes": 0, "free_mb": 0, "required_mb": 0}


# ---------------------------------------------------------------------------
# Shared FFmpeg Runner
# ---------------------------------------------------------------------------
def run_ffmpeg(cmd: list, timeout: int = 3600, stderr_cap: int = 0) -> str:
    """Run FFmpeg command, raise RuntimeError on failure. Returns stderr."""
    if cmd and cmd[0] == "ffmpeg":
        cmd = [get_ffmpeg_path()] + cmd[1:]
    result = _sp.run(cmd, capture_output=True, timeout=timeout)
    stderr = result.stderr.decode(errors="replace")
    if stderr_cap > 0 and len(stderr) > stderr_cap:
        stderr = "...[truncated] " + stderr[-stderr_cap:]
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr[-500:]}")
    return stderr


# ---------------------------------------------------------------------------
# Shared Package Installer
# ---------------------------------------------------------------------------
def ensure_package(pkg: str, pip_name: str = None, on_progress=None) -> bool:
    """Import package, auto-install via safe_pip_install if missing. Returns True on success."""
    import importlib
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        pip_name = pip_name or pkg
        if on_progress:
            on_progress(5, f"Installing {pip_name}...")
        logger.info(f"Installing missing dependency: {pip_name}")
        from opencut.security import safe_pip_install
        try:
            safe_pip_install(pip_name)
        except RuntimeError:
            return False
        try:
            importlib.import_module(pkg)
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Shared Video Info Helper
# ---------------------------------------------------------------------------
def get_video_info(filepath: str) -> dict:
    """Get video width, height, fps, duration via ffprobe. Returns safe defaults on error."""
    import json as _json
    cmd = [
        get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", filepath,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    _defaults = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0}
    if result.returncode != 0:
        logger.warning("ffprobe failed (rc=%d) for %s — using defaults", result.returncode, filepath)
        return _defaults
    try:
        data = _json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        if not streams:
            logger.warning("ffprobe returned no streams for %s — using defaults", filepath)
            return _defaults
        s = streams[0]
        fps_p = s.get("r_frame_rate", "30/1").split("/")
        fps = (float(fps_p[0]) / float(fps_p[1])) if len(fps_p) == 2 and float(fps_p[1]) else 30.0
        duration = float(s.get("duration", 0))
        if duration <= 0:
            duration = float(data.get("format", {}).get("duration", 0))
        return {
            "width": int(s.get("width", 1920)),
            "height": int(s.get("height", 1080)),
            "fps": fps,
            "duration": duration,
        }
    except Exception as e:
        logger.warning("Failed to parse ffprobe output for %s: %s — using defaults", filepath, e)
        return _defaults


# ---------------------------------------------------------------------------
# Lazy Import Helper
# ---------------------------------------------------------------------------
def _try_import(name: str):
    """
    Import a module by name, returning None on failure.

    Replaces the 50+ try/except ImportError blocks scattered throughout
    the codebase with a single-line call.

    Usage:
        torch = _try_import("torch")
        if torch is None:
            return jsonify({"error": "torch not installed"}), 400
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _try_import_from(package: str, name: str):
    """
    Import *name* from *package*, trying relative then absolute.

    Usage:
        check_whisper = _try_import_from("opencut.core.captions", "check_whisper_available")
    """
    try:
        mod = importlib.import_module(package)
        return getattr(mod, name)
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Output Path Helpers
# ---------------------------------------------------------------------------
def _resolve_output_dir(filepath: str, requested_dir: str = "") -> str:
    """
    Determine the best output directory for generated files.

    Priority:
    1. Explicitly requested directory (from panel/API)
    2. Same directory as the source file
    3. User's temp directory (last resort)
    """
    from opencut.security import validate_path

    # Try requested directory first
    if requested_dir:
        requested_dir = requested_dir.strip().strip('"').strip("'")
        try:
            requested_dir = validate_path(requested_dir)
        except ValueError:
            logger.warning("Invalid output dir rejected: %s", requested_dir)
            requested_dir = ""
        else:
            if os.path.isdir(requested_dir):
                return requested_dir
            else:
                try:
                    os.makedirs(requested_dir, exist_ok=True)
                    return requested_dir
                except OSError as e:
                    logger.debug("Failed to create output dir %s: %s", requested_dir, e)

    # Try source file's directory
    source_dir = os.path.dirname(os.path.abspath(filepath))
    if source_dir and os.path.isdir(source_dir):
        test_file = os.path.join(source_dir, f".opencut_test_{uuid.uuid4().hex[:6]}")
        try:
            with open(test_file, "w") as f:
                f.write("")
            return source_dir
        except (OSError, PermissionError) as e:
            logger.debug("Source dir %s not writable: %s", source_dir, e)
        finally:
            try:
                os.unlink(test_file)
            except OSError:
                pass  # test file may not have been created

    # Fallback to temp directory
    fallback = os.path.join(tempfile.gettempdir(), "opencut_output")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _unique_output_path(path: str) -> str:
    """
    If *path* already exists on disk, append _2, _3, ... before the extension
    until we find a name that doesn't collide.  Caps at 9999 to prevent
    infinite loops from filesystem errors.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for counter in range(2, 102):
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError(f"Could not find unique output path after 100 attempts: {path}")


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
# FFmpeg Command Builder
# ---------------------------------------------------------------------------
class FFmpegCmd:
    """Fluent builder for FFmpeg command lists.

    Usage:
        cmd = (FFmpegCmd()
               .input(filepath)
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .filter_complex(fc, maps=["[outv]", "[outa]"])
               .faststart()
               .output(out_path)
               .build())
    """

    def __init__(self):
        self._inputs = []
        self._pre_input = []      # flags before first -i (e.g. -ss for input seeking)
        self._vf = []             # -vf filters
        self._af = []             # -af filters
        self._fc = []             # -filter_complex
        self._vcodec = []
        self._acodec = []
        self._maps = []
        self._extra = []
        self._output = None
        self._hide_banner = True
        self._overwrite = True

    def input(self, path, **kwargs):
        """Add an input file. kwargs become options before -i (e.g. ss="10")."""
        opts = []
        for k, v in kwargs.items():
            opts.extend([f"-{k}", str(v)])
        self._inputs.append((opts, path))
        return self

    def video_codec(self, codec, crf=None, preset=None, pix_fmt="yuv420p"):
        """Set video codec. Use codec="copy" for stream copy."""
        self._vcodec = ["-c:v", codec]
        if codec != "copy":
            if crf is not None:
                self._vcodec.extend(["-crf", str(crf)])
            if preset:
                self._vcodec.extend(["-preset", preset])
            if pix_fmt:
                self._vcodec.extend(["-pix_fmt", pix_fmt])
        return self

    def audio_codec(self, codec, bitrate=None):
        """Set audio codec. Use codec="copy" for stream copy."""
        self._acodec = ["-c:a", codec]
        if codec != "copy" and bitrate:
            self._acodec.extend(["-b:a", bitrate])
        return self

    def copy_streams(self):
        """Copy all streams without re-encoding."""
        self._vcodec = ["-c", "copy"]
        self._acodec = []
        return self

    def no_video(self):
        """Strip video stream."""
        self._extra.extend(["-vn"])
        return self

    def pre_input(self, key, value=None):
        """Add a flag before all -i inputs (e.g. -ss for input seeking)."""
        self._pre_input.append(f"-{key}" if not key.startswith("-") else key)
        if value is not None:
            self._pre_input.append(str(value))
        return self

    def video_filter(self, vf):
        """Set -vf (simple video filter). Cannot combine with filter_complex."""
        self._vf = ["-vf", vf]
        return self

    def audio_filter(self, af):
        """Set -af (simple audio filter). Cannot combine with filter_complex."""
        self._af = ["-af", af]
        return self

    def filter_complex(self, fc, maps=None):
        """Set -filter_complex with optional -map entries.
        Overrides any -vf/-af set earlier (FFmpeg doesn't allow both)."""
        self._fc = ["-filter_complex", fc]
        self._vf = []  # filter_complex supersedes simple filters
        self._af = []
        if maps:
            for m in maps:
                self._maps.extend(["-map", m])
        return self

    def map(self, *streams):
        """Add -map entries."""
        for s in streams:
            self._maps.extend(["-map", s])
        return self

    def seek(self, start=None, end=None):
        """Add -ss / -to seek options (placed before output)."""
        if start is not None:
            self._extra.extend(["-ss", str(start)])
        if end is not None:
            self._extra.extend(["-to", str(end)])
        return self

    def faststart(self):
        """Add -movflags +faststart for MP4 streaming."""
        self._extra.extend(["-movflags", "+faststart"])
        return self

    def frames(self, n):
        """Limit output to n video frames."""
        self._extra.extend(["-vframes", str(n)])
        return self

    def option(self, key, value=None):
        """Add arbitrary -key [value] option."""
        self._extra.append(f"-{key}" if not key.startswith("-") else key)
        if value is not None:
            self._extra.append(str(value))
        return self

    def format(self, fmt):
        """Set output format (-f)."""
        self._extra.extend(["-f", fmt])
        return self

    def output(self, path):
        """Set output file path (always last in command)."""
        self._output = path
        return self

    def build(self):
        """Build the command list."""
        cmd = [get_ffmpeg_path()]
        if self._hide_banner:
            cmd.append("-hide_banner")
        if self._overwrite:
            cmd.append("-y")
        cmd.extend(self._pre_input)
        for opts, path in self._inputs:
            cmd.extend(opts)
            cmd.extend(["-i", path])
        # filter_complex takes priority; otherwise use simple vf/af
        if self._fc:
            cmd.extend(self._fc)
        else:
            cmd.extend(self._vf)
            cmd.extend(self._af)
        cmd.extend(self._maps)
        cmd.extend(self._vcodec)
        cmd.extend(self._acodec)
        cmd.extend(self._extra)
        if self._output:
            cmd.append(self._output)
        return cmd


# ---------------------------------------------------------------------------
# FFmpeg Progress Runner
# ---------------------------------------------------------------------------
def _run_ffmpeg_with_progress(job_id: str, cmd: list, duration_sec: float):
    """
    Run an FFmpeg command via Popen, parsing ``-progress pipe:1`` output
    to update job progress.  Returns a (returncode, stderr) tuple.
    Registers the process for kill-on-cancel support.
    """
    from opencut.jobs import _is_cancelled, _job_processes, _register_job_process, _update_job, job_lock

    full_cmd = list(cmd) + ["-progress", "pipe:1"]
    proc = _sp.Popen(full_cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
    _register_job_process(job_id, proc)

    stderr_lines = []
    _max_stderr_bytes = 32768  # 32 KB cap on accumulated stderr

    # Drain stderr in a background thread to prevent pipe deadlock:
    # if FFmpeg writes >4KB stderr (Windows) while stdout blocks, both
    # pipes stall permanently.
    def _drain_stderr():
        try:
            data = proc.stderr.read()
            if data:
                stderr_lines.append(data[-_max_stderr_bytes:])
        except Exception as e:
            logger.debug("Error draining FFmpeg stderr for job %s: %s", job_id, e)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    last_pct = 0

    try:
        for line in proc.stdout:
            if _is_cancelled(job_id):
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
                break

            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                    if duration_sec > 0:
                        pct = min(int((us / 1_000_000) / duration_sec * 100), 99)
                        if pct > last_pct:
                            last_pct = pct
                            _update_job(job_id, progress=pct,
                                        message=f"Processing... {pct}%")
                except (ValueError, ZeroDivisionError):
                    pass
    except Exception as e:
        logger.debug("Error reading FFmpeg stdout for job %s: %s", job_id, e)

    try:
        proc.wait(timeout=10)
    except Exception as e:
        logger.debug("FFmpeg process did not exit cleanly for job %s, killing: %s", job_id, e)
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    stderr_thread.join(timeout=10)

    # Cleanup process registration
    with job_lock:
        _job_processes.pop(job_id, None)

    return proc.returncode, "".join(stderr_lines)


# ---------------------------------------------------------------------------
# Job Time Estimation
# ---------------------------------------------------------------------------
_JOB_TIMES_FILE = os.path.join(OPENCUT_DIR, "job_times.json")
_job_times_lock = threading.Lock()


def _get_file_duration(filepath):
    """Get media file duration in seconds via ffprobe. Returns 0 on failure."""
    if not filepath or not os.path.isfile(filepath):
        return 0
    try:
        cmd = [get_ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
               "-of", "json", filepath]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        logger.debug("Failed to get duration for %s: %s", filepath, e)
        return 0


def _load_job_times():
    """Load historical job timing data. Must be called under _job_times_lock."""
    try:
        if os.path.isfile(_JOB_TIMES_FILE):
            with open(_JOB_TIMES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.debug("Failed to load job times file: %s", e)
    return {}


def _record_job_time(job_type, duration_sec, file_duration_sec):
    """Record how long a job took for future estimates. Thread-safe."""
    with _job_times_lock:
        times = _load_job_times()
        if job_type not in times:
            times[job_type] = []
        times[job_type].append({
            "job_secs": round(duration_sec, 1),
            "file_secs": round(file_duration_sec, 1) if file_duration_sec else 0,
            "ratio": round(duration_sec / max(file_duration_sec, 0.1), 3) if file_duration_sec else 0,
            "ts": time.time()
        })
        # Keep last 20 entries per type
        times[job_type] = times[job_type][-20:]
        _ensure_opencut_dir()
        fd, tmp_path = tempfile.mkstemp(
            dir=OPENCUT_DIR, suffix=".tmp", prefix="job_times_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(times, f, indent=2)
            os.replace(tmp_path, _JOB_TIMES_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def compute_estimate(job_type: str, file_duration: float) -> dict:
    """Compute a time estimate for a job type. Used by /system/estimate-time."""
    with _job_times_lock:
        times = _load_job_times()
    entries = times.get(job_type, [])
    if not entries:
        return {"estimate_seconds": None, "confidence": "none", "message": "No historical data"}
    ratios = [e["ratio"] for e in entries if e["ratio"] > 0]
    if ratios and file_duration > 0:
        avg_ratio = sum(ratios) / len(ratios)
        estimate = file_duration * avg_ratio
        confidence = "high" if len(ratios) >= 5 else "medium" if len(ratios) >= 2 else "low"
    else:
        avg_times = [e["job_secs"] for e in entries]
        estimate = sum(avg_times) / len(avg_times)
        confidence = "low"
    return {
        "estimate_seconds": round(estimate, 1),
        "confidence": confidence,
        "based_on": len(entries),
        "message": f"~{int(estimate)}s based on {len(entries)} previous runs"
    }
