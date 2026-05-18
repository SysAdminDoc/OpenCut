"""
Opt-in Aptabase telemetry for OpenCut.

The repository's default posture remains local-first: fresh installs do not
emit telemetry.  This module only sends events after an operator explicitly
enables Aptabase in ``~/.opencut/telemetry_settings.json`` or via environment
variables, and it strips user media paths, transcript text, prompts, and
secrets before payloads leave the process.

The wire format follows Aptabase's client-SDK contract:
``POST {host}/api/v0/events`` with an ``App-Key`` header and a JSON list of up
to 25 events.
"""

from __future__ import annotations

import json
import locale
import logging
import os
import platform
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from opencut.core.url_safety import validate_public_http_url
from opencut.user_data import load_telemetry_settings, save_telemetry_settings

logger = logging.getLogger("opencut")

APTABASE_PROVIDER = "aptabase"
APTABASE_EVENTS_PATH = "/api/v0/events"
SDK_VERSION = "opencut-aptabase@1"
MAX_BATCH_SIZE = 25
MAX_QUEUED_EVENTS = 500
MAX_PROP_COUNT = 30
MAX_PROP_KEY_LEN = 48
MAX_PROP_VALUE_LEN = 160
SESSION_TIMEOUT_SECONDS = 60 * 60

_HOSTS = {
    "EU": "https://eu.aptabase.com",
    "US": "https://us.aptabase.com",
}

_EVENT_LOCK = threading.Lock()
_QUEUE_COND = threading.Condition(_EVENT_LOCK)
_EVENT_QUEUE: list[Dict[str, Any]] = []
_WORKER_THREAD: Optional[threading.Thread] = None
_SHUTDOWN = threading.Event()

_SESSION_ID: Optional[str] = None
_SESSION_TOUCHED_AT = 0.0

_EVENT_NAME_RE = re.compile(r"[^A-Za-z0-9_.:-]+")
_PROP_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SENSITIVE_KEY_EXACT = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "email",
    "filename",
    "filepath",
    "media_path",
    "password",
    "prompt",
    "secret",
    "token",
    "transcript",
}
_SENSITIVE_KEY_NORMALIZED = {
    re.sub(r"[^a-z0-9]", "", key) for key in _SENSITIVE_KEY_EXACT
}
_SENSITIVE_KEY_TOKENS = {
    "authorization",
    "cookie",
    "email",
    "file",
    "password",
    "path",
    "prompt",
    "secret",
    "token",
    "transcript",
}


@dataclass(frozen=True)
class AptabaseConfig:
    """Resolved Aptabase configuration."""

    enabled: bool
    configured: bool
    app_key: str
    base_url: str
    endpoint: str
    app_version: str
    is_debug: bool
    timeout: float
    include_diagnostics: bool
    source: str
    disabled_reason: str = ""


def _app_version() -> str:
    try:
        from opencut import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_first(*names: str) -> tuple[str, str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value, name
    return "", ""


def _coerce_bool(value: Any, *, default: bool = False, label: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "on", "enabled"}:
            return True
        if cleaned in {"0", "false", "no", "off", "disabled"}:
            return False
    raise ValueError(f"{label} must be a boolean")


def _validate_app_key(app_key: str) -> str:
    key = str(app_key or "").strip()
    if not key:
        raise ValueError("Aptabase app_key is required")
    if not key.startswith(("A-EU-", "A-US-", "A-SH-")):
        raise ValueError("Aptabase app_key must start with A-EU-, A-US-, or A-SH-")
    if len(key) > 200 or any(ch.isspace() for ch in key):
        raise ValueError("Aptabase app_key is invalid")
    return key


def _base_url_for_app_key(app_key: str, custom_base_url: str = "") -> str:
    key = _validate_app_key(app_key)
    region = key.split("-", 2)[1]
    if region == "SH":
        if not custom_base_url:
            raise ValueError("Aptabase self-hosted app keys require base_url")
        return validate_public_http_url(
            custom_base_url, label="Aptabase base_url"
        ).rstrip("/")
    return _HOSTS[region]


def mask_app_key(app_key: str) -> str:
    """Return a display-safe app-key mask."""
    key = str(app_key or "").strip()
    if not key:
        return ""
    if len(key) <= 12:
        return "****"
    return f"{key[:5]}...{key[-4:]}"


def get_config() -> AptabaseConfig:
    """Resolve persisted settings plus environment overrides."""
    settings = load_telemetry_settings()
    enabled = _coerce_bool(settings.get("enabled"), default=False, label="enabled")
    source = "settings"

    env_enabled, enabled_source = _env_first("OPENCUT_TELEMETRY_ENABLED")
    if enabled_source:
        enabled = _coerce_bool(env_enabled, default=False, label=enabled_source)
        source = enabled_source

    app_key, key_source = _env_first("OPENCUT_APTABASE_APP_KEY", "APTABASE_APP_KEY")
    if not key_source:
        app_key = str(settings.get("app_key") or "").strip()
    else:
        app_key = app_key.strip()
        source = key_source

    base_url, base_source = _env_first(
        "OPENCUT_APTABASE_BASE_URL", "APTABASE_BASE_URL"
    )
    if not base_source:
        base_url = str(settings.get("base_url") or "").strip().rstrip("/")
    else:
        base_url = base_url.strip().rstrip("/")

    is_debug = _coerce_bool(
        os.environ.get("OPENCUT_TELEMETRY_DEBUG"),
        default=False,
        label="OPENCUT_TELEMETRY_DEBUG",
    )
    include_diagnostics = _coerce_bool(
        settings.get("include_diagnostics"),
        default=False,
        label="include_diagnostics",
    )
    try:
        timeout = max(1.0, min(30.0, float(os.environ.get("OPENCUT_TELEMETRY_TIMEOUT", "4"))))
    except ValueError:
        timeout = 4.0

    if not enabled:
        return AptabaseConfig(
            enabled=False,
            configured=bool(app_key),
            app_key=app_key,
            base_url=base_url,
            endpoint="",
            app_version=_app_version(),
            is_debug=is_debug,
            timeout=timeout,
            include_diagnostics=include_diagnostics,
            source=source,
            disabled_reason="telemetry disabled until the user opts in",
        )

    try:
        resolved_base_url = _base_url_for_app_key(app_key, base_url)
    except ValueError as exc:
        logger.warning("Aptabase telemetry disabled: %s", exc)
        return AptabaseConfig(
            enabled=False,
            configured=False,
            app_key=app_key,
            base_url=base_url,
            endpoint="",
            app_version=_app_version(),
            is_debug=is_debug,
            timeout=timeout,
            include_diagnostics=include_diagnostics,
            source=source,
            disabled_reason=str(exc),
        )

    return AptabaseConfig(
        enabled=True,
        configured=True,
        app_key=app_key,
        base_url=resolved_base_url,
        endpoint=f"{resolved_base_url}{APTABASE_EVENTS_PATH}",
        app_version=_app_version(),
        is_debug=is_debug,
        timeout=timeout,
        include_diagnostics=include_diagnostics,
        source=source,
    )


def check_aptabase_available() -> bool:
    """Return true when telemetry is explicitly enabled and configured."""
    return get_config().enabled


def public_settings(settings: Optional[dict] = None) -> dict:
    """Return settings without exposing the raw app key."""
    settings = dict(settings or load_telemetry_settings())
    app_key = str(settings.get("app_key") or "")
    return {
        "provider": APTABASE_PROVIDER,
        "enabled": bool(settings.get("enabled")),
        "app_key_set": bool(app_key),
        "app_key_masked": mask_app_key(app_key),
        "base_url": str(settings.get("base_url") or ""),
        "include_diagnostics": bool(settings.get("include_diagnostics")),
    }


def telemetry_info() -> dict:
    """Return route-safe telemetry status and privacy metadata."""
    cfg = get_config()
    return {
        "provider": APTABASE_PROVIDER,
        "default_provider": True,
        "enabled": cfg.enabled,
        "configured": cfg.configured,
        "host": cfg.base_url,
        "endpoint_path": APTABASE_EVENTS_PATH,
        "app_key_set": bool(cfg.app_key),
        "app_key_masked": mask_app_key(cfg.app_key),
        "queue_depth": queue_depth(),
        "source": cfg.source,
        "disabled_reason": cfg.disabled_reason,
        "privacy": {
            "opt_in_required": True,
            "fresh_install_emits": False,
            "scrubs_media_paths": True,
            "scrubs_transcripts_prompts_and_secrets": True,
            "max_batch_size": MAX_BATCH_SIZE,
        },
        "settings": public_settings(),
        "links": {
            "aptabase": "https://aptabase.com",
            "sdk_contract": "https://github.com/aptabase/aptabase/wiki/How-to-build-your-own-SDK",
        },
    }


def update_settings(payload: dict) -> dict:
    """Validate and save route-provided telemetry settings."""
    if not isinstance(payload, dict):
        raise ValueError("settings payload must be an object")
    current = load_telemetry_settings()
    updated = dict(current)

    if "enabled" in payload:
        updated["enabled"] = _coerce_bool(payload.get("enabled"), label="enabled")
    if "include_diagnostics" in payload:
        updated["include_diagnostics"] = _coerce_bool(
            payload.get("include_diagnostics"), label="include_diagnostics"
        )
    if "app_key" in payload:
        app_key = str(payload.get("app_key") or "").strip()
        if app_key.startswith("***") or "..." in app_key:
            app_key = str(current.get("app_key") or "").strip()
        if app_key:
            app_key = _validate_app_key(app_key)
        updated["app_key"] = app_key
    if "base_url" in payload:
        base_url = str(payload.get("base_url") or "").strip().rstrip("/")
        if base_url:
            base_url = validate_public_http_url(
                base_url, label="Aptabase base_url"
            ).rstrip("/")
        updated["base_url"] = base_url

    if updated.get("enabled"):
        app_key = _validate_app_key(str(updated.get("app_key") or ""))
        _base_url_for_app_key(app_key, str(updated.get("base_url") or ""))

    save_telemetry_settings(updated)
    return load_telemetry_settings()


def _clean_event_name(event_name: str) -> str:
    cleaned = _EVENT_NAME_RE.sub("_", str(event_name or "").strip())[:80]
    cleaned = cleaned.strip("._:-")
    if not cleaned:
        raise ValueError("event_name is required")
    return cleaned


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    if lowered in _SENSITIVE_KEY_EXACT or normalized in _SENSITIVE_KEY_NORMALIZED:
        return True
    tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
    return any(token in _SENSITIVE_KEY_TOKENS for token in tokens)


def _looks_sensitive_value(value: str) -> bool:
    lowered = value.lower()
    if "://" in lowered or "@" in value:
        return True
    if ":\\" in value or "\\\\" in value:
        return True
    path_markers = ("/users/", "/home/", "/volumes/", "/mnt/", "/media/")
    return any(marker in lowered.replace("\\", "/") for marker in path_markers)


def _scrub_props(props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce free-form event props to primitive, non-identifying values."""
    out: Dict[str, Any] = {}
    if not isinstance(props, dict):
        return out
    for key, value in props.items():
        if len(out) >= MAX_PROP_COUNT:
            break
        if not isinstance(key, str):
            continue
        clean_key = key.strip()[:MAX_PROP_KEY_LEN]
        if not clean_key or clean_key.startswith("_"):
            continue
        if not _PROP_KEY_RE.match(clean_key):
            continue
        if _looks_sensitive_key(clean_key):
            continue
        if isinstance(value, bool):
            out[clean_key] = value
        elif isinstance(value, int):
            out[clean_key] = value
        elif isinstance(value, float):
            out[clean_key] = round(value, 6)
        elif value is None:
            out[clean_key] = ""
        else:
            clean_value = str(value).strip()
            if _looks_sensitive_value(clean_value):
                continue
            out[clean_key] = clean_value[:MAX_PROP_VALUE_LEN]
    return out


def _current_session_id() -> str:
    global _SESSION_ID, _SESSION_TOUCHED_AT
    now = time.monotonic()
    with _EVENT_LOCK:
        if (
            _SESSION_ID is None
            or now - _SESSION_TOUCHED_AT > SESSION_TIMEOUT_SECONDS
        ):
            epoch_seconds = int(datetime.now(timezone.utc).timestamp())
            _SESSION_ID = str(epoch_seconds * 100000000 + random.randint(0, 99999999))
        _SESSION_TOUCHED_AT = now
        return _SESSION_ID


def _system_props(cfg: AptabaseConfig) -> dict:
    loc = locale.getlocale()[0] or "en-US"
    return {
        "locale": loc.replace("_", "-"),
        "osName": platform.system() or "unknown",
        "osVersion": platform.release() or "unknown",
        "deviceModel": platform.machine() or "unknown",
        "isDebug": cfg.is_debug,
        "appVersion": cfg.app_version,
        "sdkVersion": SDK_VERSION,
    }


def build_event(
    event_name: str,
    props: Optional[Dict[str, Any]] = None,
    *,
    cfg: Optional[AptabaseConfig] = None,
) -> dict:
    """Build one Aptabase event payload without sending it."""
    cfg = cfg or get_config()
    return {
        "timestamp": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "sessionId": _current_session_id(),
        "eventName": _clean_event_name(event_name),
        "systemProps": _system_props(cfg),
        "props": _scrub_props(props),
    }


def _post_events(events: list[Dict[str, Any]], cfg: Optional[AptabaseConfig] = None) -> None:
    cfg = cfg or get_config()
    if not cfg.enabled or not events:
        return
    body = json.dumps(events[:MAX_BATCH_SIZE], separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        cfg.endpoint,
        data=body,
        method="POST",
        headers={
            "App-Key": cfg.app_key,
            "Content-Type": "application/json",
            "User-Agent": f"opencut-telemetry/{cfg.app_version}",
        },
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
        resp.read(64)


def _ensure_worker_started() -> None:
    global _WORKER_THREAD
    if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
        return
    _WORKER_THREAD = threading.Thread(
        target=_worker_loop,
        name="aptabase-telemetry",
        daemon=True,
    )
    _WORKER_THREAD.start()


def _worker_loop() -> None:
    while not _SHUTDOWN.is_set():
        with _QUEUE_COND:
            while not _EVENT_QUEUE and not _SHUTDOWN.is_set():
                _QUEUE_COND.wait(timeout=1.0)
            if _SHUTDOWN.is_set():
                return
            batch = _EVENT_QUEUE[:MAX_BATCH_SIZE]
            del _EVENT_QUEUE[:MAX_BATCH_SIZE]
        try:
            _post_events(batch)
        except urllib.error.HTTPError as exc:
            logger.debug("Aptabase telemetry HTTP %s", exc.code)
        except urllib.error.URLError as exc:
            logger.debug("Aptabase telemetry unreachable: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Aptabase telemetry post failed: %s", exc)


def track(
    event_name: str,
    props: Optional[Dict[str, Any]] = None,
    *,
    sync: bool = False,
) -> bool:
    """Queue or send one Aptabase event.

    Returns ``False`` when telemetry is disabled or invalidly configured.  The
    caller should not treat telemetry failure as a user-facing operation error.
    """
    cfg = get_config()
    if not cfg.enabled:
        return False
    event = build_event(event_name, props=props, cfg=cfg)
    if sync:
        try:
            _post_events([event], cfg=cfg)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Aptabase sync post failed: %s", exc)
            return False

    with _QUEUE_COND:
        if len(_EVENT_QUEUE) >= MAX_QUEUED_EVENTS:
            _EVENT_QUEUE.pop(0)
        _EVENT_QUEUE.append(event)
        _QUEUE_COND.notify()
    _ensure_worker_started()
    return True


def queue_depth() -> int:
    with _EVENT_LOCK:
        return len(_EVENT_QUEUE)


def shutdown(timeout: float = 2.0) -> None:
    """Stop the worker thread and clear in-memory telemetry state."""
    global _WORKER_THREAD
    _SHUTDOWN.set()
    with _QUEUE_COND:
        _EVENT_QUEUE.clear()
        _QUEUE_COND.notify_all()
    if _WORKER_THREAD is not None:
        _WORKER_THREAD.join(timeout=max(0.1, timeout))
        _WORKER_THREAD = None
    _SHUTDOWN.clear()


def _reset_for_tests() -> None:
    """Reset module globals used by tests."""
    global _SESSION_ID, _SESSION_TOUCHED_AT
    shutdown(timeout=0.1)
    with _EVENT_LOCK:
        _EVENT_QUEUE.clear()
        _SESSION_ID = None
        _SESSION_TOUCHED_AT = 0.0
