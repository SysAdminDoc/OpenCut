"""Local bootstrap secret that lets a host-embedded panel obtain a CSRF token.

The CEP panel is loaded from a ``file://`` document, so its XHR to the loopback
backend carries ``Origin: null``. ``/health`` deliberately withholds the CSRF
bootstrap token from that origin, because a hostile local page presents exactly
the same origin and would otherwise be able to harvest the loopback mutation
token. Without a second channel the panel can never authenticate: it connects,
reports healthy, and then fails every mutating request.

This module is that second channel. The server writes a random secret to a
0600 file under the OpenCut user directory at startup. The panel reads it
through ExtendScript — a capability the browser context itself does not have —
and presents it back on the bootstrap request. A web page cannot open the file,
so the origin blocklist stays fully effective while the real panel recovers.

The secret only unlocks the CSRF bootstrap. It is not an authentication token
and grants nothing on its own; remote binds still require the separate
``opencut.auth`` bearer token.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import stat
import tempfile
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("opencut")

#: Header the panel uses to present the bootstrap secret on ``GET /health``.
BOOTSTRAP_HEADER = "X-OpenCut-Panel-Bootstrap"

#: File name inside the OpenCut user directory.
BOOTSTRAP_FILENAME = "panel_bootstrap.token"

#: Environment override, mainly for tests and containerised layouts.
BOOTSTRAP_PATH_ENV = "OPENCUT_PANEL_BOOTSTRAP_FILE"

#: Length of the generated secret in bytes before hex expansion.
_SECRET_BYTES = 32

#: Reject anything implausible before doing constant-time work.
_MAX_SECRET_CHARS = 256

_lock = threading.Lock()
_cached_secret: Optional[str] = None


def bootstrap_path() -> str:
    """Return the absolute path of the panel bootstrap secret file."""
    configured = os.environ.get(BOOTSTRAP_PATH_ENV)
    if configured is not None:
        configured = configured.strip()
        if not configured:
            return ""
        return os.path.abspath(os.path.expanduser(configured))

    from opencut.user_data import OPENCUT_DIR

    return os.path.join(OPENCUT_DIR, BOOTSTRAP_FILENAME)


def _restrict_windows_acl(path: Path) -> None:
    """Restrict the secret to the current user via icacls (best-effort)."""
    import subprocess as _sp

    try:
        username = os.environ.get("USERNAME", "")
        if not username:
            return
        _sp.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(R,W)"],
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001 - ACL hardening must not block startup
        logger.warning("opencut.panel_bootstrap: could not restrict ACL on %s: %s", path, exc)


def _atomic_write_secret(path: Path, secret: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".opencut-panel-")
    try:
        if os.name == "posix":
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as handle:
            handle.write(secret.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        if os.name == "nt":
            _restrict_windows_acl(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_secret_file(path: Path) -> Optional[str]:
    """Return the stored secret, or ``None`` when it is absent or unusable."""
    try:
        info = path.lstat()
    except OSError:
        return None

    if not stat.S_ISREG(info.st_mode):
        logger.warning("opencut.panel_bootstrap: %s is not a regular file; ignoring.", path)
        return None
    file_attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    if file_attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
        logger.warning("opencut.panel_bootstrap: %s is a reparse point; ignoring.", path)
        return None
    if os.name == "posix" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(
            "opencut.panel_bootstrap: %s has group or world permissions; ignoring.", path
        )
        return None

    try:
        payload = path.read_bytes()
    except OSError as exc:
        logger.warning("opencut.panel_bootstrap: could not read %s: %s", path, exc)
        return None

    if len(payload) > _MAX_SECRET_CHARS + 2:
        logger.warning("opencut.panel_bootstrap: %s is too large; ignoring.", path)
        return None
    try:
        secret = payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        logger.warning("opencut.panel_bootstrap: %s is not valid UTF-8; ignoring.", path)
        return None
    if not secret or any(char.isspace() for char in secret):
        return None
    return secret


def ensure_bootstrap_secret() -> str:
    """Create the bootstrap secret if needed and return it.

    Called once during app creation. Returns an empty string when the feature
    is disabled or the secret cannot be persisted; the panel then falls back to
    the existing same-origin bootstrap path.
    """
    global _cached_secret

    target = bootstrap_path()
    if not target:
        with _lock:
            _cached_secret = None
        return ""

    path = Path(target)
    with _lock:
        existing = _read_secret_file(path)
        if existing:
            _cached_secret = existing
            return existing

        secret = secrets.token_hex(_SECRET_BYTES)
        try:
            _atomic_write_secret(path, secret)
        except Exception as exc:  # noqa: BLE001 - startup must survive a read-only home
            logger.warning(
                "opencut.panel_bootstrap: could not write %s: %s. "
                "Host-embedded panels will fall back to same-origin bootstrap.",
                path,
                exc,
            )
            _cached_secret = None
            return ""
        _cached_secret = secret
        return secret


def current_secret() -> Optional[str]:
    """Return the active bootstrap secret, loading it from disk on demand."""
    global _cached_secret

    with _lock:
        if _cached_secret:
            return _cached_secret

    target = bootstrap_path()
    if not target:
        return None
    secret = _read_secret_file(Path(target))
    if secret:
        with _lock:
            _cached_secret = secret
    return secret


def is_bootstrap_secret_valid(candidate: Optional[str]) -> bool:
    """Return True when *candidate* matches the active bootstrap secret."""
    if not isinstance(candidate, str):
        return False
    candidate = candidate.strip()
    if not candidate or len(candidate) > _MAX_SECRET_CHARS or not candidate.isascii():
        return False
    secret = current_secret()
    if not secret:
        return False
    return hmac.compare_digest(candidate, secret)


def clear_cached_secret() -> None:
    """Drop the in-process cache. Used by tests and after rotation."""
    global _cached_secret

    with _lock:
        _cached_secret = None
