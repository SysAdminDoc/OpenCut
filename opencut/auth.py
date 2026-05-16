"""Local auth tokens for non-loopback binds (F112).

OpenCut's HTTP server is single-user and ships bound to ``127.0.0.1`` by
default. When the operator opts into a non-loopback bind via
``OPENCUT_ALLOW_REMOTE=1``, the surface area changes — anyone on the
network can hit the API — so we add a second gate: a persistent API
token stored at ``~/.opencut/auth.json``, required on every request
when the server isn't bound to a loopback address.

Design choices:

* **No new dependency.** Pure stdlib (``secrets``, ``json``, ``os``).
* **Atomic writes.** The token file is written to a sibling
  ``auth.json.tmp`` then renamed so a crash mid-write never leaves a
  zero-byte file.
* **0600 file mode** on POSIX. On Windows we rely on user-profile
  inheritance — there is no portable way to mark a file "user-only" in
  stdlib without ``ctypes``.
* **Rotation is explicit.** Callers ask for ``rotate_token()``; the
  generator never silently invalidates existing tokens. If you want
  to invalidate, delete ``~/.opencut/auth.json`` and the next read
  returns ``None``.
* **Loopback requests bypass the gate.** The CEP/UXP panel runs in the
  same machine as the server; requiring a token in that path would
  break the single-user UX without buying any security.
"""

from __future__ import annotations

import hmac
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

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

AUTH_FILE = Path(OPENCUT_DIR) / "auth.json"
AUTH_HEADER = "X-OpenCut-Auth"
TOKEN_BYTES = 32  # 256 bits

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

# Hosts we never require an auth token on. The match is performed against the
# Flask request remote_addr; localhost names are resolved to loopback by the
# OS so this list is intentionally tiny.
_LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "::1", "localhost"})

_lock = threading.Lock()


@dataclass
class AuthToken:
    token: str
    issued_at: float
    label: str = "default"

    def as_dict(self) -> dict:
        return {"token": self.token, "issued_at": self.issued_at, "label": self.label}


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix="auth_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        if os.name == "posix":
            try:
                os.chmod(tmp_name, stat.S_IRUSR | stat.S_IWUSR)
            except OSError as exc:  # pragma: no cover - filesystem oddity
                logger.warning("opencut.auth: could not chmod %s: %s", tmp_name, exc)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load() -> Optional[AuthToken]:
    if not AUTH_FILE.exists():
        return None
    try:
        with AUTH_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("opencut.auth: ignoring corrupt %s: %s", AUTH_FILE, exc)
        return None
    token = str(data.get("token") or "").strip()
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
        token = AuthToken(
            token=secrets.token_urlsafe(TOKEN_BYTES),
            issued_at=time.time(),
            label=label,
        )
        _atomic_write(AUTH_FILE, token.as_dict())
        return token


def rotate_token(label: str = "default") -> AuthToken:
    """Generate a fresh token, replacing any existing one."""
    with _lock:
        token = AuthToken(
            token=secrets.token_urlsafe(TOKEN_BYTES),
            issued_at=time.time(),
            label=label,
        )
        _atomic_write(AUTH_FILE, token.as_dict())
        return token


def clear_token() -> bool:
    """Delete the persisted token. Returns True if a file was removed."""
    with _lock:
        try:
            AUTH_FILE.unlink()
            return True
        except FileNotFoundError:
            return False


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
    if not remote_addr:
        return False
    # Werkzeug normalises addresses to strings; strip IPv6 zone suffix.
    addr = remote_addr.split("%", 1)[0]
    return addr in _LOOPBACK_ADDRESSES


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


def extract_request_token(headers: Iterable, args: Optional[dict] = None) -> str:
    """Extract a token from the headers (preferred) or ``?auth=`` query."""
    if hasattr(headers, "get"):
        header_token = (headers.get(AUTH_HEADER) or "").strip()
        if header_token:
            return header_token
    args = args or {}
    return str(args.get("auth", "") or "").strip()
