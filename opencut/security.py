"""
OpenCut Security Utilities

Path validation, CSRF protection, and safe pip install helpers.
"""

import functools
import hmac
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

from flask import Request, jsonify, request
from werkzeug.exceptions import BadRequest

logger = logging.getLogger("opencut")

# Valid Whisper model names for input validation
VALID_WHISPER_MODELS = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "turbo", "large-v3-turbo",
    "distil-large-v2", "distil-large-v3", "distil-large-v3.5",
    "distil-medium.en", "distil-small.en",
})


class OpenCutRequest(Request):
    """Flask request class that rejects top-level non-object JSON bodies.

    The backend routes overwhelmingly expect JSON objects and immediately call
    ``data.get(...)``. A top-level array/string/number would otherwise make it
    through ``request.get_json()`` and crash later as an ``AttributeError``,
    surfacing as a 500 instead of a clear client error. Centralizing the guard
    here hardens every route that still uses Flask's request parsing directly.
    """

    _OBJECT_ONLY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def get_json(self, force=False, silent=False, cache=True):
        data = super().get_json(force=force, silent=silent, cache=cache)
        if silent or data is None:
            return data
        expects_object = self.method in self._OBJECT_ONLY_METHODS and (force or self.is_json)
        if expects_object and not isinstance(data, dict):
            raise BadRequest("JSON body must be an object.")
        return data

# ---------------------------------------------------------------------------
# CSRF Token (rotating with TTL)
# ---------------------------------------------------------------------------
CSRF_TTL = 3600          # seconds — tokens expire after 1 hour
CSRF_MAX_TOKENS = 10     # keep at most this many valid tokens

_csrf_tokens: dict[str, float] = {}   # token_str -> expiry_timestamp
_csrf_lock = threading.Lock()


def _purge_expired_tokens() -> None:
    """Remove expired tokens from ``_csrf_tokens``.  Caller must hold ``_csrf_lock``."""
    now = time.monotonic()
    expired = [t for t, exp in _csrf_tokens.items() if exp <= now]
    for t in expired:
        del _csrf_tokens[t]


def _newest_token() -> tuple[str | None, float]:
    """Return ``(token, expiry)`` for the newest token, or ``(None, 0)``."""
    if not _csrf_tokens:
        return None, 0.0
    return max(_csrf_tokens.items(), key=lambda kv: kv[1])


def get_csrf_token() -> str:
    """Return the current (newest) CSRF token, generating a new one if needed."""
    with _csrf_lock:
        _purge_expired_tokens()
        tok, exp = _newest_token()
        now = time.monotonic()
        # Generate a new token when none exist or the newest is past half-life
        if tok is None or (exp - now) < (CSRF_TTL / 2):
            new_tok = secrets.token_hex(32)
            _csrf_tokens[new_tok] = now + CSRF_TTL
            # Evict oldest if over the cap
            while len(_csrf_tokens) > CSRF_MAX_TOKENS:
                oldest = min(_csrf_tokens, key=_csrf_tokens.get)
                del _csrf_tokens[oldest]
            tok = new_tok
        return tok


def require_csrf(f):
    """
    Decorator that rejects POST/PUT/DELETE requests missing a valid
    ``X-OpenCut-Token`` header.  The token is handed to the panel via
    the ``/health`` response so no extra round-trip is needed.

    Accepts ANY non-expired token from the rotating pool, giving clients
    a grace window when tokens rotate.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if request.method in ("POST", "PUT", "DELETE"):
            header_token = request.headers.get("X-OpenCut-Token", "")
            with _csrf_lock:
                _purge_expired_tokens()
                valid = any(
                    hmac.compare_digest(header_token, t)
                    for t in _csrf_tokens
                )
            if not valid:
                return jsonify({"error": "Invalid or missing CSRF token"}), 403
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# JSON Body Validation
# ---------------------------------------------------------------------------
def get_json_dict(*, force: bool = True, silent: bool = False) -> dict:
    """Return the request JSON body as a dict.

    Many OpenCut routes expect a top-level JSON object and immediately call
    ``data.get(...)``. When a client sends a JSON array, number, or string, that
    pattern turns into an ``AttributeError`` and surfaces as a 500. This helper
    normalizes the happy path and turns non-object JSON into a clear 400-level
    validation error instead.

    Malformed JSON is still raised by Flask/Werkzeug so the app's centralized
    error handlers can return a structured ``INVALID_JSON`` response.
    """
    data = request.get_json(force=force, silent=silent)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


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

    # Block UNC/network paths (SSRF/NTLM hash leak risk)
    if path.startswith("\\\\") or path.startswith("//"):
        raise ValueError("UNC/network paths are not allowed")

    # Block Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
    stem = os.path.splitext(os.path.basename(path))[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL"} or re.match(r"^(COM|LPT)\d$", stem):
        raise ValueError("Windows reserved device name in path")

    # Normalise and block .. components
    normed = os.path.normpath(path)
    # Re-check after normpath (could produce UNC from edge cases)
    if normed.startswith("\\\\") or normed.startswith("//"):
        raise ValueError("UNC/network paths are not allowed")
    parts = normed.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError("Path traversal blocked")

    # Resolve to absolute real path (follows symlinks)
    resolved = os.path.realpath(normed)

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


_SAFE_PACKAGE_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?(\[.+\])?(==.+|>=.+|<=.+|~=.+|!=.+)?$")


def _verify_package_importable(python: str, package: str, target_dir: str = None) -> bool:
    """Check if a package is importable by running a quick subprocess check.

    Uses the import name (strips extras/version specifiers and normalises
    hyphens to underscores).  When *target_dir* is given it is prepended
    to ``sys.path`` inside the subprocess so ``--target`` installs are
    found.
    """
    # Derive the bare import name: "whisperx[all]>=0.1" -> "whisperx"
    import_name = re.split(r"[\[>=<!~]", package)[0].replace("-", "_")
    setup = ""
    if target_dir:
        setup = f"import sys; sys.path.insert(0, {target_dir!r}); "
    cmd = [python, "-c", f"{setup}import {import_name}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def safe_pip_install(package: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """
    Install a pip package with a safe fallback chain:

    1. ``pip install --target ~/.opencut/packages`` — always writable, no admin needed
    2. Normal ``pip install <package>``
    3. ``pip install --user <package>``
    4. ``pip install --break-system-packages <package>``  (only inside a venv)

    When running as a frozen (PyInstaller) build, uses system Python
    from PATH instead of ``sys.executable`` (which points to the exe).

    Permission errors (Errno 13) are caught and immediately skip to the
    next strategy instead of failing the entire install.

    Returns the ``CompletedProcess`` on success.
    Raises ``RuntimeError`` if all strategies fail.
    """
    if not package or not _SAFE_PACKAGE_RE.match(package):
        raise ValueError(f"Invalid package name: {package!r}")

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

    # Prefer pre-built binary wheels (avoids needing Rust/C compilers)
    _prefer = "--prefer-binary"

    # ~/.opencut/packages — the ONLY directory guaranteed writable without admin
    _target_dir = os.path.join(os.path.expanduser("~"), ".opencut", "packages")
    os.makedirs(_target_dir, exist_ok=True)
    # Add to sys.path so subsequent imports find the package immediately
    if _target_dir not in sys.path:
        sys.path.insert(0, _target_dir)

    # Build strategy list — --target first to avoid permission errors
    strategies = [
        ("--target (user-local)", [python, "-m", "pip", "install", package, _prefer, "--target", _target_dir, "-q"]),
        ("default pip install", [python, "-m", "pip", "install", package, _prefer, "-q"]),
        ("--user install", [python, "-m", "pip", "install", package, _prefer, "--user", "-q"]),
    ]

    # Only allow --break-system-packages inside a virtual environment
    if _in_virtualenv():
        strategies.append(
            ("--break-system-packages", [python, "-m", "pip", "install", package, _prefer, "--break-system-packages", "-q"])
        )

    last_result = None
    for label, cmd in strategies:
        logger.info("Trying to install %s via strategy: %s", package, label)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                logger.info("Successfully installed %s via strategy: %s", package, label)
                # Verify the package is actually importable
                if _verify_package_importable(python, package, _target_dir):
                    logger.info("Verified %s is importable after install", package)
                else:
                    logger.warning(
                        "Package %s installed via %s but import check failed — "
                        "this may be OK for packages with different import names",
                        package, label,
                    )
                return result

            # Check stderr for permission denied (Errno 13) — skip immediately
            stderr_text = result.stderr or ""
            if "Permission denied" in stderr_text or "Errno 13" in stderr_text or "[WinError 5]" in stderr_text:
                logger.warning(
                    "Permission denied installing %s via %s — skipping to next strategy. "
                    "stderr: %s", package, label, stderr_text[-300:]
                )
                last_result = result
                continue

            logger.warning(
                "Strategy %s failed for %s (rc=%d): %s",
                label, package, result.returncode, (result.stderr or "")[-300:]
            )
            last_result = result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"pip install {package} timed out after {timeout}s")
        except OSError as e:
            # Catch OS-level permission errors (e.g. Errno 13 from subprocess itself)
            if e.errno == 13:
                logger.warning("OS permission error for strategy %s: %s — skipping", label, e)
                continue
            logger.warning("pip install attempt (%s) failed: %s", label, e)
            last_result = None
        except Exception as e:
            logger.warning("pip install attempt (%s) failed: %s", label, e)
            last_result = None

    stderr = last_result.stderr[-500:] if last_result and last_result.stderr else "unknown error"
    # Provide actionable hint for Rust/compiler errors
    if "rust" in stderr.lower() or "cargo" in stderr.lower() or "metadata-generation-failed" in stderr.lower():
        raise RuntimeError(
            f"Failed to install {package}: this package requires Rust to compile from source, "
            f"and no pre-built wheel is available for your Python version.\n\n"
            f"Fix: Install Rust from https://rustup.rs/ and restart, "
            f"or upgrade Python to 3.10-3.12 where pre-built wheels are available.\n\n"
            f"Details: {stderr[-300:]}"
        )
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


def validate_output_path(path: str) -> str:
    """Validate a user-supplied output file path.

    Like ``validate_path`` but does NOT require the file to exist (it will
    be created).  Ensures the *parent directory* exists and is writable.
    Returns the resolved absolute path.
    """
    resolved = validate_path(path)
    parent = os.path.dirname(resolved)
    if parent and not os.path.isdir(parent):
        raise ValueError(f"Output directory does not exist: {parent}")
    if parent and not os.access(parent, os.W_OK):
        raise ValueError(f"Output directory is not writable: {parent}")
    if os.path.exists(resolved):
        if os.path.isdir(resolved):
            raise ValueError(f"Output path is a directory: {resolved}")
        if not os.access(resolved, os.W_OK):
            raise ValueError(f"Output file is not writable: {resolved}")
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
        result = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    if min_val is not None and result < min_val:
        result = min_val
    if max_val is not None and result > max_val:
        result = max_val
    return result


def safe_bool(value, default: bool = False) -> bool:
    """Convert common string/numeric flag values to bool.

    Accepts native booleans, 0/1 style numbers, and common string forms
    like ``"true"`` / ``"false"``. Falls back to *default* for unknown,
    ambiguous, or structurally unsafe inputs instead of treating every
    non-empty value as ``True``.

    Hardened edge cases:

    * ``None`` returns ``default``.
    * ``NaN`` / ``inf`` floats return ``default`` (consistent with
      ``safe_float``).
    * ``bytes`` / ``bytearray`` are decoded as UTF-8 then parsed as
      strings, so ``b"false"`` correctly returns ``False``.
    * Lists, tuples, sets, and dicts return ``default`` — a flag with a
      container value is a client bug, not a truth signal.
    * Unknown string forms return ``default``.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, float):
        # NaN and infinities are never valid flag values.
        if value != value or value == float("inf") or value == float("-inf"):
            return default
        return value != 0
    if isinstance(value, int):
        return value != 0
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", "", "null", "none"}:
            return False
        return default
    # Containers and arbitrary objects should not be silently coerced.
    return default


# ---------------------------------------------------------------------------
# Rate Limiting (in-memory, per-endpoint)
# ---------------------------------------------------------------------------
_rate_limits = {}
_rate_lock = threading.Lock()


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
