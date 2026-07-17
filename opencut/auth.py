"""Local auth tokens for non-loopback binds (F112).

OpenCut's HTTP server is single-user and ships bound to ``127.0.0.1`` by
default. When the operator opts into a non-loopback bind via
``OPENCUT_ALLOW_REMOTE=1``, the surface area changes — anyone on the
network can hit the API — so we add a second gate: a persistent API
token stored in the operating-system credential vault, required on every request
when the server isn't bound to a loopback address.

Design choices:

* **Secure secret sources.** Desktop installs keep issuance metadata in
  ``auth.json`` and the value in the OS vault. Headless deployments can point
  ``OPENCUT_REMOTE_AUTH_TOKEN_FILE`` at a locked-down mounted secret instead;
  that mode never writes the value or metadata into OpenCut's data directory.
* **Atomic metadata writes.** Vault writes are verified before the JSON metadata
  is replaced, and both sides roll back if either step fails.
* **Rotation is explicit.** Callers ask for ``rotate_token()``; the
  generator never silently invalidates existing tokens. If you want
  to invalidate, call ``clear_token()`` so both vault and metadata are removed.
* **Loopback requests bypass the gate.** The CEP/UXP panel runs in the
  same machine as the server; requiring a token in that path would
  break the single-user UX without buying any security.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import secrets
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from opencut.credential_store import (
    load_and_migrate_secrets,
    persist_secret_changes,
    secret_id,
)
from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

AUTH_FILE = Path(OPENCUT_DIR) / "auth.json"
AUTH_HEADER = "X-OpenCut-Auth"
REMOTE_AUTH_TOKEN_FILE_ENV = "OPENCUT_REMOTE_AUTH_TOKEN_FILE"
TOKEN_BYTES = 32  # 256 bits
MIN_TOKEN_CHARS = 32
MAX_TOKEN_CHARS = 512
_AUTH_SECRET_ID = secret_id("server/api-token")

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

# Hostnames we never require an auth token on. Numeric addresses are
# classified via ``ipaddress.is_loopback`` so the full ``127.0.0.0/8``
# range plus ``::1`` are recognised — a strict literal match would have
# let ``127.0.0.2`` (still loopback) bypass the gate when the operator
# binds to ``0.0.0.0`` with ``OPENCUT_ALLOW_REMOTE=1``.
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})

_lock = threading.Lock()


class RemoteAuthTokenFileError(RuntimeError):
    """A configured secret-file token is unsafe or cannot be used."""

    code = "REMOTE_AUTH_TOKEN_FILE_INVALID"
    status = 503
    suggestion = (
        f"Mount a regular non-symlink file through {REMOTE_AUTH_TOKEN_FILE_ENV}, "
        "set mode 0400 or 0600, and provide at least 32 non-whitespace characters."
    )


class RemoteAuthTokenFileReadOnlyError(RemoteAuthTokenFileError):
    code = "REMOTE_AUTH_TOKEN_FILE_READ_ONLY"
    suggestion = (
        "Replace the externally managed secret and restart/recreate the service; "
        "server-side rotation is intentionally disabled for read-only mounts."
    )


def _token_file_path() -> Optional[Path]:
    raw = os.environ.get(REMOTE_AUTH_TOKEN_FILE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise RemoteAuthTokenFileError(f"{REMOTE_AUTH_TOKEN_FILE_ENV} must name an absolute path.")
    return path


def using_token_file() -> bool:
    """Return whether the headless secret-file backend is configured."""
    return bool(os.environ.get(REMOTE_AUTH_TOKEN_FILE_ENV, "").strip())


def credential_storage() -> str:
    return "secret_file" if using_token_file() else "os_vault"


def auth_recovery_suggestion() -> str:
    if using_token_file():
        return (
            "Use the value from the mounted secret file named by "
            f"{REMOTE_AUTH_TOKEN_FILE_ENV}; replace the external secret and "
            "restart/recreate the service to rotate a read-only mount."
        )
    return (
        "Reveal the OS-vault token with `opencut-server --print-auth` or rotate it with `opencut-server --rotate-auth`."
    )


def _validate_token_file_stat(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise RemoteAuthTokenFileError("Configured auth token source is not a regular file.")
    file_attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    if file_attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
        raise RemoteAuthTokenFileError("Configured auth token source is a reparse point.")
    if os.name == "posix" and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RemoteAuthTokenFileError("Configured auth token file has group or world permissions.")


def _validate_token_value(payload: bytes) -> str:
    if len(payload) > MAX_TOKEN_CHARS + 2:
        raise RemoteAuthTokenFileError("Configured auth token is too long.")
    try:
        token = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RemoteAuthTokenFileError("Configured auth token is not valid UTF-8.") from exc
    if not token:
        raise RemoteAuthTokenFileError("Configured auth token is empty.")
    if len(token) < MIN_TOKEN_CHARS:
        raise RemoteAuthTokenFileError(f"Configured auth token must contain at least {MIN_TOKEN_CHARS} characters.")
    if len(token) > MAX_TOKEN_CHARS:
        raise RemoteAuthTokenFileError("Configured auth token is too long.")
    if any(char.isspace() for char in token):
        raise RemoteAuthTokenFileError("Configured auth token contains whitespace.")
    return token


def _read_token_file(path: Path) -> AuthToken:
    try:
        before = path.lstat()
        if path.is_symlink():
            raise RemoteAuthTokenFileError("Configured auth token source is a symlink.")
        _validate_token_file_stat(before)
        flags = os.O_RDONLY | int(getattr(os, "O_NOFOLLOW", 0))
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            _validate_token_file_stat(opened)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise RemoteAuthTokenFileError("Configured auth token file changed while it was being opened.")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                payload = handle.read(MAX_TOKEN_CHARS + 3)
        finally:
            os.close(fd)
    except RemoteAuthTokenFileError:
        raise
    except (OSError, ValueError) as exc:
        raise RemoteAuthTokenFileError("Configured auth token file cannot be read safely.") from exc
    return AuthToken(
        token=_validate_token_value(payload),
        issued_at=float(opened.st_mtime),
        label="secret-file",
    )


def _token_file_is_writable(path: Path) -> bool:
    info = path.lstat()
    _validate_token_file_stat(info)
    if os.name == "posix":
        return bool(info.st_mode & stat.S_IWUSR)
    return bool(info.st_mode & stat.S_IWRITE) and os.access(path, os.W_OK)


def _atomic_write_token_file(path: Path, token: str) -> None:
    if not _token_file_is_writable(path):
        raise RemoteAuthTokenFileReadOnlyError(
            "Configured auth token file is read-only; server-side rotation is disabled."
        )
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".opencut-auth-")
    try:
        if os.name == "posix":
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as handle:
            handle.write(token.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        if os.name == "nt":
            _restrict_windows_acl(path)
        _read_token_file(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class AuthToken:
    token: str
    issued_at: float
    label: str = "default"

    def as_dict(self) -> dict:
        return {
            "issued_at": self.issued_at,
            "label": self.label,
            "token_set": bool(self.token),
        }


def _restrict_windows_acl(path: Path) -> None:
    """Restrict file to current user only via icacls (best-effort)."""
    import subprocess as _sp

    try:
        target = str(path)
        username = os.environ.get("USERNAME", "")
        if not username:
            return
        _sp.run(
            ["icacls", target, "/inheritance:r", "/grant:r", f"{username}:(R,W)"],
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        logger.warning("opencut.auth: could not restrict ACL on %s: %s", path, exc)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="auth_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name == "posix":
            try:
                os.chmod(tmp_name, stat.S_IRUSR | stat.S_IWUSR)
            except OSError as exc:  # pragma: no cover - filesystem oddity
                logger.warning("opencut.auth: could not chmod %s: %s", tmp_name, exc)
        os.replace(tmp_name, path)
        if os.name == "nt":
            _restrict_windows_acl(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_metadata() -> Optional[dict]:
    if not AUTH_FILE.exists():
        return None
    try:
        with AUTH_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("opencut.auth: ignoring corrupt %s: %s", AUTH_FILE, exc)
        return None
    return data if isinstance(data, dict) else None


def _load() -> Optional[AuthToken]:
    token_path = _token_file_path()
    if token_path is not None:
        return _read_token_file(token_path)
    data = _read_metadata()
    if data is None:
        return None

    def persist_sanitized() -> None:
        metadata = dict(data)
        value = str(metadata.pop("token", "") or "").strip()
        metadata["token_set"] = bool(value)
        metadata["_credential_storage"] = "os_vault"
        _atomic_write(AUTH_FILE, metadata)

    token = load_and_migrate_secrets(
        {"token": _AUTH_SECRET_ID},
        data,
        persist_sanitized,
    )["token"].strip()
    if not token:
        return None
    return AuthToken(
        token=token,
        issued_at=float(data.get("issued_at") or 0.0),
        label=str(data.get("label") or "default"),
    )


def current_token() -> Optional[AuthToken]:
    """Return the persisted token, or ``None`` if not yet issued."""
    with _lock:
        return _load()


def ensure_token(label: str = "default") -> AuthToken:
    """Return the persisted token, generating one on first call."""
    with _lock:
        existing = _load()
        if existing is not None:
            return existing
        if using_token_file():  # Defensive: configured files fail closed in _load().
            raise RemoteAuthTokenFileError("Configured auth token file did not provide a token.")
        token = AuthToken(
            token=secrets.token_urlsafe(TOKEN_BYTES),
            issued_at=time.time(),
            label=label,
        )
        _persist_token(token)
        return token


def rotate_token(label: str = "default") -> AuthToken:
    """Generate a fresh token, replacing any existing one."""
    with _lock:
        token = AuthToken(
            token=secrets.token_urlsafe(TOKEN_BYTES),
            issued_at=time.time(),
            label=label,
        )
        token_path = _token_file_path()
        if token_path is not None:
            _atomic_write_token_file(token_path, token.token)
            return _read_token_file(token_path)
        _persist_token(token)
        return token


def _persist_token(token: AuthToken) -> None:
    def persist_metadata(secure: bool) -> None:
        metadata = token.as_dict()
        metadata["_credential_storage"] = "os_vault" if secure else "plaintext-opt-in"
        if not secure:
            metadata["token"] = token.token
        _atomic_write(AUTH_FILE, metadata)

    persist_secret_changes({_AUTH_SECRET_ID: token.token}, persist_metadata)


def clear_token() -> bool:
    """Delete persisted token metadata and its OS-vault value."""
    with _lock:
        token_path = _token_file_path()
        if token_path is not None:
            try:
                if not _token_file_is_writable(token_path):
                    raise RemoteAuthTokenFileReadOnlyError(
                        "Configured auth token file is read-only and cannot be cleared."
                    )
                token_path.unlink()
                return True
            except FileNotFoundError:
                return False
        metadata = _read_metadata()
        if metadata is None:
            return False

        def remove_metadata(_secure: bool) -> None:
            try:
                AUTH_FILE.unlink()
            except FileNotFoundError:
                pass

        changes = {}
        if metadata.get("token") or metadata.get("token_set"):
            changes[_AUTH_SECRET_ID] = None
        persist_secret_changes(changes, remove_metadata)
        return True


def is_token_valid(candidate: Optional[str]) -> bool:
    """Constant-time compare ``candidate`` against the persisted token."""
    if not candidate:
        return False
    persisted = current_token()
    if persisted is None:
        return False
    return hmac.compare_digest(persisted.token, candidate)


def is_remote_bind_enabled() -> bool:
    return os.environ.get("OPENCUT_ALLOW_REMOTE", "").strip().lower() in _TRUE_ENV_VALUES


def _request_is_loopback(remote_addr: Optional[str]) -> bool:
    """Return True when ``remote_addr`` is a loopback address.

    Accepts either a hostname (``localhost``) or a numeric address. For
    numeric addresses we delegate to ``ipaddress.is_loopback`` so the full
    IPv4 ``127.0.0.0/8`` range and IPv6 ``::1`` are covered. IPv6 scoped
    addresses (``fe80::1%eth0``) and bracketed forms (``[::1]``) are
    normalised first so the classification matches the underlying interface.
    """
    if not remote_addr:
        return False
    addr = remote_addr.strip()
    if not addr:
        return False
    # Strip IPv6 zone suffix and surrounding brackets that Werkzeug or
    # downstream proxies may leave attached.
    addr = addr.split("%", 1)[0].strip("[]")
    if addr.lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return False


def request_requires_auth_token(remote_addr: Optional[str]) -> bool:
    """Return True when the current request must carry ``X-OpenCut-Auth``.

    Auth is required when the operator opted into a non-loopback bind
    **and** the request is coming from a non-loopback peer. Loopback
    requests are still trusted by default to keep the single-user
    workflow snappy.
    """
    if not is_remote_bind_enabled():
        return False
    return not _request_is_loopback(remote_addr)


def extract_request_token(headers: Iterable) -> str:
    """Extract the auth token from the ``X-OpenCut-Auth`` header.

    Tokens are deliberately NOT accepted via a ``?auth=`` query parameter:
    query strings leak into server access logs, proxy logs, browser history,
    and Referer headers. No shipped client uses query auth — the extension's
    SSE streams connect from loopback, which never requires a token.
    """
    if hasattr(headers, "get"):
        return (headers.get(AUTH_HEADER) or "").strip()
    return ""
