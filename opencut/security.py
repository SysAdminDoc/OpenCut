"""
OpenCut Security Utilities

Path validation, CSRF protection, and safe pip install helpers.
"""

import functools
import logging
import os
import secrets
import shutil
import subprocess
import sys

from flask import request, jsonify

logger = logging.getLogger("opencut")

# Valid Whisper model names for input validation
VALID_WHISPER_MODELS = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "turbo", "large-v3-turbo",
})

# ---------------------------------------------------------------------------
# CSRF Token
# ---------------------------------------------------------------------------
_csrf_token = secrets.token_hex(32)


def get_csrf_token() -> str:
    """Return the current session CSRF token."""
    return _csrf_token


def require_csrf(f):
    """
    Decorator that rejects POST/PUT/DELETE requests missing a valid
    ``X-OpenCut-Token`` header.  The token is handed to the panel via
    the ``/health`` response so no extra round-trip is needed.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if request.method in ("POST", "PUT", "DELETE"):
            token = request.headers.get("X-OpenCut-Token", "")
            if token != _csrf_token:
                return jsonify({"error": "Invalid or missing CSRF token"}), 403
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Path Validation
# ---------------------------------------------------------------------------
def validate_path(path: str, allowed_base: str = None) -> str:
    """
    Validate and resolve a file path, blocking traversal attacks.

    - Rejects null bytes
    - Rejects ``..`` components
    - Resolves symlinks via ``os.path.realpath``
    - Optionally checks the resolved path is under *allowed_base*

    Returns the resolved absolute path.
    Raises ``ValueError`` on invalid/dangerous input.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Empty or invalid path")

    # Block null bytes
    if "\x00" in path:
        raise ValueError("Null byte in path")

    # Normalise and block .. components
    normed = os.path.normpath(path)
    parts = normed.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError("Path traversal blocked")

    # Resolve to absolute real path (follows symlinks)
    resolved = os.path.realpath(path)

    # Optional base-directory confinement
    if allowed_base is not None:
        real_base = os.path.realpath(allowed_base)
        if not (resolved == real_base or resolved.startswith(real_base + os.sep)):
            raise ValueError("Path outside allowed directory")

    return resolved


# ---------------------------------------------------------------------------
# Safe pip install
# ---------------------------------------------------------------------------
def _find_system_python() -> str | None:
    """Find system Python executable when running as a frozen (PyInstaller) build."""
    for name in ("python", "python3", "py"):
        path = shutil.which(name)
        if path:
            try:
                result = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    logger.debug("Found system Python: %s (%s)", path, result.stdout.strip())
                    return path
            except Exception:
                continue
    return None


def safe_pip_install(package: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """
    Install a pip package with a safe fallback chain:

    1. Normal ``pip install <package>``
    2. ``pip install --user <package>``
    3. ``pip install --break-system-packages <package>``  (only inside a venv)

    When running as a frozen (PyInstaller) build, uses system Python
    from PATH instead of ``sys.executable`` (which points to the exe).

    Returns the ``CompletedProcess`` on success.
    Raises ``RuntimeError`` if all strategies fail.
    """
    # Frozen builds can't use sys.executable for pip — find system Python
    if getattr(sys, "frozen", False):
        python = _find_system_python()
        if not python:
            raise RuntimeError(
                f"Cannot install {package}: Python not found in PATH. "
                "Install Python from python.org and ensure it is on PATH."
            )
    else:
        python = sys.executable

    strategies = [
        [python, "-m", "pip", "install", package, "-q"],
        [python, "-m", "pip", "install", package, "--user", "-q"],
    ]
    # Only allow --break-system-packages inside a virtual environment
    if _in_virtualenv():
        strategies.append(
            [python, "-m", "pip", "install", package, "--break-system-packages", "-q"]
        )

    last_result = None
    for cmd in strategies:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                logger.info("Installed %s via: %s", package, " ".join(cmd))
                return result
            last_result = result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"pip install {package} timed out after {timeout}s")
        except Exception as e:
            logger.warning("pip install attempt failed: %s", e)
            last_result = None

    stderr = last_result.stderr[-500:] if last_result and last_result.stderr else "unknown error"
    raise RuntimeError(f"Failed to install {package}: {stderr}")


def validate_filepath(filepath: str) -> str:
    """
    Validate a user-supplied file path for input files.

    Runs ``validate_path`` checks and verifies the file exists on disk.
    Returns the resolved absolute path.
    Raises ``ValueError`` if the path is invalid or missing.
    """
    resolved = validate_path(filepath)
    if not os.path.isfile(resolved):
        raise ValueError(f"File not found: {filepath}")
    return resolved


def safe_float(value, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
    """Convert *value* to float, returning *default* on failure.

    If *min_val* or *max_val* are given, clamp the result to that range.
    Rejects inf/nan values.
    """
    try:
        result = float(value)
        if result != result or result == float("inf") or result == float("-inf"):
            return default
    except (TypeError, ValueError):
        return default
    if min_val is not None and result < min_val:
        result = min_val
    if max_val is not None and result > max_val:
        result = max_val
    return result


def safe_int(value, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    """Convert *value* to int, returning *default* on failure.

    If *min_val* or *max_val* are given, clamp the result to that range.
    """
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if min_val is not None and result < min_val:
        result = min_val
    if max_val is not None and result > max_val:
        result = max_val
    return result


# ---------------------------------------------------------------------------
# Rate Limiting (in-memory, per-endpoint)
# ---------------------------------------------------------------------------
import threading as _threading
import time as _time

_rate_limits = {}
_rate_lock = _threading.Lock()


def rate_limit(key: str, max_concurrent: int = 1) -> bool:
    """
    Check if an operation identified by *key* is already at its
    concurrency limit.  Returns ``True`` if the caller is allowed to
    proceed, ``False`` if rate-limited.

    Call ``rate_limit_release(key)`` when the operation finishes.
    """
    with _rate_lock:
        current = _rate_limits.get(key, 0)
        if current >= max_concurrent:
            return False
        _rate_limits[key] = current + 1
        return True


def rate_limit_release(key: str):
    """Decrement the concurrency counter for *key*."""
    with _rate_lock:
        current = _rate_limits.get(key, 0)
        if current > 0:
            _rate_limits[key] = current - 1


def require_rate_limit(key: str, max_concurrent: int = 1):
    """Decorator that rejects requests when concurrency limit is reached.

    Acquires the rate limit slot before calling the wrapped function and
    releases it when the function returns (or raises).  For async job
    routes that spawn a background thread, the thread's ``finally``
    block should call ``rate_limit_release(key)`` itself — in that case
    use the lower-level ``rate_limit()`` / ``rate_limit_release()``
    functions directly instead of this decorator.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not rate_limit(key, max_concurrent):
                return jsonify({
                    "error": f"Another {key} operation is already running. Please wait."
                }), 429
            try:
                return f(*args, **kwargs)
            finally:
                rate_limit_release(key)
        return wrapper
    return decorator


def _in_virtualenv() -> bool:
    """Return True if running inside a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
