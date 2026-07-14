"""Local auth tokens for non-loopback binds (F112).

OpenCut's HTTP server is single-user and ships bound to ``127.0.0.1`` by
default. When the operator opts into a non-loopback bind via
``OPENCUT_ALLOW_REMOTE=1``, the surface area changes — anyone on the
network can hit the API — so we add a second gate: a persistent API
token stored in the operating-system credential vault, required on every request
when the server isn't bound to a loopback address.

Design choices:

* **OS-vault secret.** ``auth.json`` contains only issuance metadata; keyring
  selects Windows Credential Locker, macOS Keychain, Secret Service, or KWallet.
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
TOKEN_BYTES = 32  # 256 bits
_AUTH_SECRET_ID = secret_id("server/api-token")

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

# Hostnames we never require an auth token on. Numeric addresses are
# classified via ``ipaddress.is_loopback`` so the full ``127.0.0.0/8``
# range plus ``::1`` are recognised — a strict literal match would have
# let ``127.0.0.2`` (still loopback) bypass the gate when the operator
# binds to ``0.0.0.0`` with ``OPENCUT_ALLOW_REMOTE=1``.
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})

_lock = threading.Lock()


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
            capture_output=True, timeout=10,
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
        _persist_token(token)
        return token


def _persist_token(token: AuthToken) -> None:
    def persist_metadata(secure: bool) -> None:
        metadata = token.as_dict()
        metadata["_credential_storage"] = (
            "os_vault" if secure else "plaintext-opt-in"
        )
        if not secure:
            metadata["token"] = token.token
        _atomic_write(AUTH_FILE, metadata)

    persist_secret_changes({_AUTH_SECRET_ID: token.token}, persist_metadata)


def clear_token() -> bool:
    """Delete persisted token metadata and its OS-vault value."""
    with _lock:
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


def extract_request_token(headers: Iterable, args: Optional[dict] = None) -> str:
    """Extract a token from the headers (preferred) or ``?auth=`` query."""
    if hasattr(headers, "get"):
        header_token = (headers.get(AUTH_HEADER) or "").strip()
        if header_token:
            return header_token
    args = args or {}
    return str(args.get("auth", "") or "").strip()
