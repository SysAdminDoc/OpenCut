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

F471: every native client presents it too. ``/health`` used to hand the token
to any request without an ``Origin`` header, and any local process can send
one of those, including a process in another user's session on a shared
Windows PC. The file is readable only by the account that started the server,
so the CLI, the MCP server and the UXP panel (through a copy in its own plugin
data folder, see :func:`mirror_secret_to_uxp`) prove they run as that account.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
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

#: Plugin id from ``extension/com.opencut.uxp/manifest.json``.
UXP_PLUGIN_ID = "com.opencut.uxp"

#: UXP's code for Premiere Pro in its plugin storage tree.
UXP_HOST_APP = "PPRO"

#: Environment override for the UXP ``PluginsStorage`` root (tests, odd layouts).
UXP_STORAGE_ENV = "OPENCUT_UXP_PLUGIN_STORAGE"

#: A refused ``/health`` retries the mirror at most this often.
_MIRROR_MIN_INTERVAL_S = 1.0

_lock = threading.Lock()
_cached_secret: Optional[str] = None
_mirror_lock = threading.Lock()
_last_mirror_at = 0.0


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
    try:
        username = os.environ.get("USERNAME", "")
        if not username:
            return
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(R,W)"],
            capture_output=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
    is disabled or the secret cannot be persisted. Local clients then get no
    CSRF token from ``/health`` unless they send the remote auth token.
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
                "Local panels and tools won't get a CSRF token from /health.",
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


def is_loopback_url(url: str) -> bool:
    """Return True when *url* points at a loopback host."""
    try:
        host = (urllib.parse.urlsplit(str(url or "")).hostname or "").strip().lower()
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _lsof_listener_pids(port: int) -> Optional[set[int]]:
    """PIDs that ``lsof`` run as this account sees listening on *port*."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("lsof could not list listeners on port %d: %s", port, exc)
        return None
    return {int(token) for token in out.split() if token.isdigit()}


def _listener_pids(port: int) -> Optional[set[int]]:
    """PIDs of every TCP listener on *port*, or ``None`` when one can't be named."""
    import psutil

    try:
        connections = psutil.net_connections(kind="tcp")
    except psutil.AccessDenied:
        # macOS lists every socket only for root. lsof run as this account
        # sees this account's listeners, and one it can't see isn't ours.
        return _lsof_listener_pids(port)
    except psutil.Error as exc:
        logger.debug("Could not list listeners on port %d: %s", port, exc)
        return None
    pids: set[int] = set()
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or not conn.laddr or conn.laddr.port != port:
            continue
        if not conn.pid:
            return None  # Linux hides the owner of another account's socket.
        pids.add(conn.pid)
    return pids


def _runs_as_this_account(pid: int) -> bool:
    import psutil

    try:
        if os.name == "nt":
            return psutil.Process(pid).username() == psutil.Process().username()
        return psutil.Process(pid).uids().effective == os.geteuid()
    except (psutil.Error, OSError):
        # Another account's process refuses the query unless we're an admin.
        return False


def listener_is_ours(base_url: str) -> bool:
    """Return True when every listener on *base_url*'s port runs as this account.

    Loopback is shared by every account on the machine. A process in another
    account that takes the port first would otherwise collect the secret.
    """
    try:
        parts = urllib.parse.urlsplit(str(base_url or ""))
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    try:
        pids = _listener_pids(port)
    except (ImportError, OSError) as exc:
        logger.debug("Could not list listeners on port %d: %s", port, exc)
        return False
    if not pids:
        return False
    if all(_runs_as_this_account(pid) for pid in pids):
        return True
    logger.warning(
        "Not sending the panel bootstrap secret to port %d: the process listening there runs as another account",
        port,
    )
    return False


def bootstrap_headers(base_url: str) -> dict[str, str]:
    """Headers a native client adds to ``GET /health`` on *base_url*.

    Only a loopback backend run by this account gets the secret. A remote one
    keeps its own secret and must never learn this machine's.
    """
    if not is_loopback_url(base_url) or not listener_is_ours(base_url):
        return {}
    secret = current_secret()
    return {BOOTSTRAP_HEADER: secret} if secret else {}


def uxp_storage_root() -> str:
    """Return UXP's ``PluginsStorage`` folder for this user, or ``""``."""
    configured = os.environ.get(UXP_STORAGE_ENV)
    if configured is not None:
        configured = configured.strip()
        return os.path.abspath(os.path.expanduser(configured)) if configured else ""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "").strip()
        return os.path.join(appdata, "Adobe", "UXP", "PluginsStorage") if appdata else ""
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "Adobe",
            "UXP",
            "PluginsStorage",
        )
    return ""


def uxp_plugin_data_dirs() -> list[Path]:
    """Every existing data folder UXP gave the OpenCut panel in Premiere.

    UXP keeps one per host version and install kind:
    ``<root>/PPRO/<version>/<Internal|External|Developer>/<plugin id>/PluginData``.
    It's the folder ``localFileSystem.getDataFolder()`` returns, which the
    panel can read without a file-system permission in its manifest.
    """
    root = uxp_storage_root()
    if not root:
        return []
    host_dir = Path(root) / UXP_HOST_APP
    try:
        candidates = sorted(host_dir.glob(f"*/*/{UXP_PLUGIN_ID}/PluginData"))
    except OSError:
        return []
    return [path for path in candidates if path.is_dir() and not path.is_symlink()]


def mirror_secret_to_uxp(*, force: bool = False) -> list[str]:
    """Copy the secret into the UXP panel's data folders and return the paths written.

    UXP can't open the OpenCut data folder without a broad file-system
    permission, so the server puts a copy where the panel can read it. The
    folder sits in this account's profile, so the copy crosses no account
    boundary. Folders are never created: a missing one means the panel hasn't
    run in that Premiere version yet, and a refused ``/health`` from it calls
    this again (at most once per :data:`_MIRROR_MIN_INTERVAL_S` unless
    *force*). Files that already hold the secret aren't rewritten.
    """
    global _last_mirror_at

    with _mirror_lock:
        now = time.monotonic()
        if not force and now - _last_mirror_at < _MIRROR_MIN_INTERVAL_S:
            return []
        _last_mirror_at = now

    secret = current_secret()
    if not secret:
        return []
    written: list[str] = []
    for data_dir in uxp_plugin_data_dirs():
        target = data_dir / BOOTSTRAP_FILENAME
        if _read_secret_file(target) == secret:
            continue
        try:
            _atomic_write_secret(target, secret)
        except Exception as exc:  # noqa: BLE001 - one bad folder must not stop the rest
            logger.warning("opencut.panel_bootstrap: could not copy the secret to %s: %s", target, exc)
            continue
        written.append(str(target))
    return written
