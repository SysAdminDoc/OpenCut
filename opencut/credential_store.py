"""OS credential-vault boundary and transactional plaintext migration."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger("opencut")

SERVICE_NAME = "OpenCut"
INSECURE_OPT_IN_ENV = "OPENCUT_ALLOW_INSECURE_SECRET_STORAGE"
UNAVAILABLE_CODE = "CREDENTIAL_STORE_UNAVAILABLE"
WRITE_FAILED_CODE = "CREDENTIAL_STORE_WRITE_FAILED"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_SECURE_BACKEND_MARKERS = (
    "keyring.backends.windows",
    "keyring.backends.macos",
    "keyring.backends.secretservice",
    "keyring.backends.kwallet",
    "keyring.backends.libsecret",
)
_UNSAFE_BACKEND_MARKERS = (
    "keyring.backends.fail",
    "keyring.backends.null",
    "keyrings.alt",
    "plaintext",
)

_lock = threading.RLock()
_backend_override: Any = None
_last_error = ""


class CredentialStoreError(RuntimeError):
    code = WRITE_FAILED_CODE
    status = 503
    suggestion = (
        "Unlock or configure the OS credential vault, then retry. "
        f"Use {INSECURE_OPT_IN_ENV}=1 only if plaintext storage is explicitly accepted."
    )


class CredentialStoreUnavailableError(CredentialStoreError):
    code = UNAVAILABLE_CODE


class CredentialStoreWriteError(CredentialStoreError):
    code = WRITE_FAILED_CODE


@dataclass(frozen=True)
class CredentialStoreStatus:
    available: bool
    secure: bool
    backend: str
    insecure_opt_in: bool
    service: str = SERVICE_NAME
    last_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SecretPersistenceResult:
    secure: bool
    backend: str


class MemoryCredentialBackend:
    """In-memory secure backend used only by the hermetic test fixture."""

    priority = 100
    _opencut_test_backend = True

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> Optional[str]:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self.values:
            raise KeyError(username)
        del self.values[key]


def _set_backend_for_tests(backend: Any) -> None:
    global _backend_override, _last_error
    with _lock:
        _backend_override = backend
        _last_error = ""


def insecure_storage_allowed() -> bool:
    return os.environ.get(INSECURE_OPT_IN_ENV, "").strip().lower() in _TRUE_VALUES


def secret_id(namespace: str, identity: str = "") -> str:
    """Build a stable vault username without exposing node URLs or account IDs."""
    clean_namespace = str(namespace or "").strip().strip("/")
    if not clean_namespace:
        raise ValueError("credential namespace is required")
    if not identity:
        return clean_namespace
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
    return f"{clean_namespace}/{digest}"


def _backend_name(backend: Any) -> str:
    cls = backend.__class__
    return f"{cls.__module__}.{cls.__name__}"


def _secure_backend(backend: Any) -> Any:
    if getattr(backend, "_opencut_test_backend", False):
        return backend
    name = _backend_name(backend).lower()
    if any(marker in name for marker in _UNSAFE_BACKEND_MARKERS):
        return None
    if any(marker in name for marker in _SECURE_BACKEND_MARKERS):
        try:
            return backend if float(backend.priority) > 0 else None
        except Exception:
            return None
    for candidate in getattr(backend, "backends", ()) or ():
        selected = _secure_backend(candidate)
        if selected is not None:
            return selected
    return None


def _select_backend() -> Any:
    global _last_error
    if _backend_override is not None:
        backend = _secure_backend(_backend_override)
        _last_error = (
            "" if backend is not None
            else "No supported OS credential-vault backend is active."
        )
        return backend
    try:
        import keyring

        backend = _secure_backend(keyring.get_keyring())
        if backend is None:
            _last_error = "No supported OS credential-vault backend is active."
        else:
            _last_error = ""
        return backend
    except Exception as exc:  # noqa: BLE001 - status must remain diagnostic
        _last_error = f"Credential vault initialization failed: {type(exc).__name__}"
        return None


def status() -> CredentialStoreStatus:
    with _lock:
        backend = _select_backend()
        return CredentialStoreStatus(
            available=backend is not None,
            secure=backend is not None,
            backend=_backend_name(backend) if backend is not None else "unavailable",
            insecure_opt_in=insecure_storage_allowed(),
            last_error=_last_error,
        )


def _unavailable() -> CredentialStoreUnavailableError:
    return CredentialStoreUnavailableError(
        "No supported OS credential vault is available; secret persistence was refused."
    )


def get_secret(identifier: str, default: str = "", *, required: bool = False) -> str:
    global _last_error
    with _lock:
        backend = _select_backend()
        if backend is None:
            if required:
                raise _unavailable()
            return default
        try:
            value = backend.get_password(SERVICE_NAME, identifier)
        except Exception as exc:  # noqa: BLE001
            _last_error = f"Credential vault read failed: {type(exc).__name__}"
            if required:
                raise CredentialStoreWriteError(
                    "The OS credential vault could not read the requested secret."
                ) from exc
            return default
        return str(value) if value is not None else default


def _restore_snapshot(backend: Any, snapshot: Mapping[str, Optional[str]]) -> None:
    global _last_error
    failures = 0
    for identifier, previous in snapshot.items():
        try:
            current = backend.get_password(SERVICE_NAME, identifier)
            if previous is None:
                if current is not None:
                    backend.delete_password(SERVICE_NAME, identifier)
                if backend.get_password(SERVICE_NAME, identifier) is not None:
                    failures += 1
            else:
                backend.set_password(SERVICE_NAME, identifier, previous)
                restored = backend.get_password(SERVICE_NAME, identifier)
                if restored is None or not hmac.compare_digest(
                    str(restored), previous
                ):
                    failures += 1
        except Exception:  # noqa: BLE001 - best-effort rollback diagnostics
            failures += 1
    if failures:
        _last_error = f"Credential rollback failed for {failures} item(s)."


def persist_secret_changes(
    changes: Mapping[str, Optional[str]],
    persist_metadata: Callable[[bool], None],
) -> SecretPersistenceResult:
    """Apply vault changes, verify them, then atomically persist redacted metadata.

    ``persist_metadata`` receives ``True`` for OS-vault storage. When no secure
    backend exists and explicit insecure storage is enabled, it receives
    ``False`` and the caller is responsible for retaining plaintext metadata.
    """
    global _last_error
    normalized = {
        str(identifier): (None if value is None or value == "" else str(value))
        for identifier, value in changes.items()
    }
    if not normalized:
        persist_metadata(True)
        return SecretPersistenceResult(secure=True, backend="not-required")

    with _lock:
        backend = _select_backend()
        if backend is None:
            if insecure_storage_allowed():
                persist_metadata(False)
                return SecretPersistenceResult(secure=False, backend="plaintext-opt-in")
            raise _unavailable()

        snapshot: dict[str, Optional[str]] = {}
        try:
            for identifier in normalized:
                snapshot[identifier] = backend.get_password(SERVICE_NAME, identifier)
            for identifier, value in normalized.items():
                previous = snapshot[identifier]
                if value is None:
                    if previous is not None:
                        backend.delete_password(SERVICE_NAME, identifier)
                    if backend.get_password(SERVICE_NAME, identifier) is not None:
                        raise CredentialStoreWriteError(
                            "The OS credential vault did not verify a secret deletion."
                        )
                else:
                    backend.set_password(SERVICE_NAME, identifier, value)
                    verified = backend.get_password(SERVICE_NAME, identifier)
                    if verified is None or not hmac.compare_digest(str(verified), value):
                        raise CredentialStoreWriteError(
                            "The OS credential vault did not verify a secret write."
                        )
            persist_metadata(True)
        except Exception as exc:
            _restore_snapshot(backend, snapshot)
            if not _last_error:
                _last_error = f"Credential persistence failed: {type(exc).__name__}"
            if isinstance(exc, CredentialStoreError):
                raise
            raise CredentialStoreWriteError(
                "Credential persistence failed; prior vault values were restored."
            ) from exc
        return SecretPersistenceResult(secure=True, backend=_backend_name(backend))


def load_and_migrate_secrets(
    identifiers: Mapping[str, str],
    legacy_plaintext: Mapping[str, Any],
    persist_sanitized: Callable[[], None],
) -> dict[str, str]:
    """Resolve secrets and migrate legacy plaintext in one verified transaction."""
    legacy = {
        field: str(legacy_plaintext.get(field) or "")
        for field in identifiers
    }
    pending = {
        identifiers[field]: value
        for field, value in legacy.items()
        if value
    }
    if pending:
        backend = _select_backend()
        if backend is None and not insecure_storage_allowed():
            logger.warning(
                "Legacy plaintext credentials remain on disk because no supported "
                "OS vault is available; they are disabled until the vault is ready "
                "or %s=1 is set explicitly.",
                INSECURE_OPT_IN_ENV,
            )
            return {field: "" for field in identifiers}
        if backend is None:
            return legacy
        try:
            persist_secret_changes(pending, lambda secure: persist_sanitized())
        except CredentialStoreError as exc:
            logger.warning("Credential migration was rolled back: %s", exc)
            return legacy

    return {
        field: get_secret(identifier, legacy.get(field, ""))
        for field, identifier in identifiers.items()
    }


def redact_secret_mapping(value: Any) -> Any:
    """Recursively redact common secret fields for exports and diagnostics."""
    if isinstance(value, dict):
        redacted = {}
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "access_token",
                "api_key",
                "app_key",
                "auth_token",
                "client_secret",
                "password",
                "refresh_token",
                "secret",
                "signing_secret",
                "token",
            }:
                redacted[key] = "[REDACTED]" if child else ""
            else:
                redacted[key] = redact_secret_mapping(child)
        return redacted
    if isinstance(value, list):
        return [redact_secret_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secret_mapping(item) for item in value)
    return value


def run_startup_migrations() -> dict:
    """Invoke every current credential-bearing loader without network I/O."""
    migrations = []

    from opencut import auth, user_data
    from opencut.core import (
        cloud_render,
        notion_sync,
        remote_process,
        social_post,
        webhook_system,
    )

    migrations.extend([
        ("llm", user_data.load_llm_settings),
        ("telemetry", user_data.load_telemetry_settings),
        ("social", social_post._load_credentials),
        ("notion", notion_sync.load_notion_config),
        ("cloud-render", cloud_render.load_nodes),
        ("remote-nodes", remote_process.reload_registry),
        ("webhooks", webhook_system._load_configs),
        ("server-auth", auth.current_token),
    ])
    migrated: list[str] = []
    failed: list[str] = []
    for name, loader in migrations:
        try:
            loader()
            migrated.append(name)
        except Exception as exc:  # noqa: BLE001 - startup remains available
            logger.warning("Credential migration check failed for %s: %s", name, exc)
            failed.append(name)
    return {
        "checked": migrated,
        "failed": failed,
        "vault": status().to_dict(),
    }
