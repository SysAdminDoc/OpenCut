"""
OpenCut Event-Driven Webhook System

Register webhook URLs for events (job_complete, job_failed,
render_complete, export_ready, error).  POST JSON payloads with
retry logic and exponential backoff.  Persistent configuration in
``~/.opencut/webhooks.json``.  Delivery log for the last 100 events.
Uses ``urllib.request`` only -- no external dependencies.
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.core.url_safety import validate_public_http_url
from opencut.core.webhook_signature import sign_webhook_body
from opencut.credential_store import (
    load_and_migrate_secrets,
    persist_secret_changes,
    secret_id,
)

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_WEBHOOKS_FILE = os.path.join(_OPENCUT_DIR, "webhooks.json")
_DELIVERY_LOG_FILE = os.path.join(_OPENCUT_DIR, "webhook_deliveries.json")
_config_lock = threading.RLock()
_delivery_lock = threading.RLock()

_MAX_DELIVERIES = 100

# Recognized event types
VALID_EVENTS = frozenset({
    "job_complete", "job_failed", "render_complete",
    "export_ready", "error",
    "job.complete", "job.error", "job.cancelled",
    "review.comment_added", "review.status_changed",
})

_LEGACY_EVENT_ALIASES = {
    "error": "job.error",
    "job_complete": "job.complete",
    "job_failed": "job.error",
}

_EVENT_ALIASES = {
    "job.complete": frozenset({"job.complete", "job_complete"}),
    "job.error": frozenset({"job.error", "job_failed", "error"}),
    "job.cancelled": frozenset({"job.cancelled"}),
    "job_complete": frozenset({"job_complete", "job.complete"}),
    "job_failed": frozenset({"job_failed", "job.error"}),
}

_EVENT_METADATA = {
    "error": {
        "canonical": "job.error",
        "deprecated": True,
        "description": "Legacy alias for failed async jobs.",
        "since_version": "1.32.0",
    },
    "export_ready": {
        "deprecated": False,
        "description": "A rendered export is available for pickup.",
        "since_version": "1.32.0",
    },
    "job.cancelled": {
        "deprecated": False,
        "description": "An async job was cancelled before completion.",
        "since_version": "1.32.0",
    },
    "job.complete": {
        "deprecated": False,
        "description": "An async job completed successfully.",
        "since_version": "1.32.0",
    },
    "job.error": {
        "deprecated": False,
        "description": "An async job failed with an error.",
        "since_version": "1.32.0",
    },
    "job_complete": {
        "canonical": "job.complete",
        "deprecated": True,
        "description": "Legacy alias for completed async jobs.",
        "since_version": "1.32.0",
    },
    "job_failed": {
        "canonical": "job.error",
        "deprecated": True,
        "description": "Legacy alias for failed async jobs.",
        "since_version": "1.32.0",
    },
    "render_complete": {
        "deprecated": False,
        "description": "A render operation completed.",
        "since_version": "1.32.0",
    },
    "review.comment_added": {
        "deprecated": False,
        "description": "A timestamped review comment was added.",
        "since_version": "1.32.0",
    },
    "review.status_changed": {
        "deprecated": False,
        "description": "A review status changed.",
        "since_version": "1.32.0",
    },
}

# Retry backoff delays in seconds
RETRY_DELAYS = (1, 5, 15)
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WebhookConfig:
    """A registered webhook endpoint."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = ""
    events: List[str] = field(default_factory=list)
    enabled: bool = True
    description: str = ""
    secret: str = ""
    created: float = field(default_factory=time.time)

    def to_dict(self, *, include_secret: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if include_secret:
            return data
        data.pop("secret", None)
        data["has_secret"] = bool(self.secret)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            url=data.get("url", ""),
            events=data.get("events", []),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            secret=str(data.get("secret", "") or ""),
            created=data.get("created", time.time()),
        )


@dataclass
class WebhookDelivery:
    """Record of a single webhook delivery attempt."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    webhook_id: str = ""
    url: str = ""
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status_code: int = 0
    success: bool = False
    attempts: int = 0
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Webhook configuration persistence
# ---------------------------------------------------------------------------

def _ensure_dir():
    os.makedirs(_OPENCUT_DIR, exist_ok=True)


def _unsigned_warning_file() -> str:
    return os.path.join(_OPENCUT_DIR, "webhooks_unsigned.txt")


def _write_unsigned_warning() -> None:
    """Write the one-time warning for explicitly unsigned webhook configs."""
    warning_file = _unsigned_warning_file()
    if os.path.exists(warning_file):
        return
    try:
        _ensure_dir()
        with open(warning_file, "w", encoding="utf-8") as fh:
            fh.write(
                "Unsigned OpenCut webhooks are enabled for at least one endpoint.\n"
                "New HTTP registrations require a non-empty HMAC signing secret by "
                "default; use allow_unsigned=true only for trusted local testing.\n"
            )
    except OSError as exc:
        logger.warning("Failed to write unsigned webhook warning: %s", exc)


def _validate_webhook_url(url: str) -> str:
    """Validate and normalize outbound webhook URLs."""
    return validate_public_http_url(url, label="Webhook URL")


def _atomic_write_json(path: str, payload: Any) -> None:
    """Atomically replace *path* with UTF-8 JSON content."""
    import tempfile

    _ensure_dir()
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_config_metadata() -> List[Dict[str, Any]]:
    with _config_lock:
        if not os.path.isfile(_WEBHOOKS_FILE):
            return []
        try:
            with open(_WEBHOOKS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load webhook configs: %s", exc)
            return []


def _load_configs() -> List[WebhookConfig]:
    """Load webhook metadata and signing secrets from the OS vault."""
    stored = _read_config_metadata()
    identifiers = {
        item.get("id", ""): secret_id("webhook/signing-secret", item.get("id", ""))
        for item in stored
        if item.get("id")
    }
    legacy = {
        item["id"]: str(item.get("secret") or "")
        for item in stored
        if item.get("id")
    }

    def persist_sanitized() -> None:
        sanitized = []
        for item in stored:
            metadata = dict(item)
            value = str(metadata.pop("secret", "") or "")
            metadata["has_secret"] = bool(value or metadata.get("has_secret"))
            metadata["_credential_storage"] = "os_vault"
            sanitized.append(metadata)
        with _config_lock:
            _atomic_write_json(_WEBHOOKS_FILE, sanitized)

    secrets = load_and_migrate_secrets(identifiers, legacy, persist_sanitized)
    configs = []
    for item in stored:
        metadata = dict(item)
        metadata["secret"] = secrets.get(item.get("id", ""), "")
        configs.append(WebhookConfig.from_dict(metadata))
    if any(cfg.enabled and not cfg.secret for cfg in configs):
        _write_unsigned_warning()
    return configs


def _save_configs(configs: List[WebhookConfig]) -> None:
    """Persist webhook metadata after vault verification."""
    old = {
        item.get("id"): item
        for item in _read_config_metadata()
        if item.get("id")
    }
    changes: dict[str, Optional[str]] = {}
    for config in configs:
        previous = old.get(config.id, {})
        if config.secret or previous.get("secret") or previous.get("has_secret"):
            changes[secret_id("webhook/signing-secret", config.id)] = (
                config.secret or None
            )
    for webhook_id, previous in old.items():
        if (
            webhook_id not in {config.id for config in configs}
            and (previous.get("secret") or previous.get("has_secret"))
        ):
            changes[secret_id("webhook/signing-secret", webhook_id)] = None

    def persist_metadata(secure: bool) -> None:
        payload = []
        for config in configs:
            item = config.to_dict(include_secret=not secure)
            item["_credential_storage"] = (
                "os_vault" if secure else "plaintext-opt-in"
            )
            payload.append(item)
        with _config_lock:
            _atomic_write_json(_WEBHOOKS_FILE, payload)

    persist_secret_changes(changes, persist_metadata)
    logger.debug("Saved %d webhook config(s)", len(configs))


# ---------------------------------------------------------------------------
# Delivery log persistence
# ---------------------------------------------------------------------------

def _load_deliveries() -> List[Dict[str, Any]]:
    """Load delivery log from disk."""
    with _delivery_lock:
        if not os.path.isfile(_DELIVERY_LOG_FILE):
            return []
        try:
            with open(_DELIVERY_LOG_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            return []


def _save_deliveries(deliveries: List[Dict[str, Any]]) -> None:
    """Persist delivery log to disk (keeps last N)."""
    with _delivery_lock:
        try:
            _atomic_write_json(_DELIVERY_LOG_FILE, deliveries[-_MAX_DELIVERIES:])
        except OSError as exc:
            logger.warning("Failed to save delivery log: %s", exc)


def _append_delivery(delivery: WebhookDelivery) -> None:
    """Append a delivery record to the log."""
    with _delivery_lock:
        deliveries = _load_deliveries()
        deliveries.append(delivery.to_dict())
        _save_deliveries(deliveries[-_MAX_DELIVERIES:])


# ---------------------------------------------------------------------------
# Webhook CRUD
# ---------------------------------------------------------------------------

def list_events() -> Dict[str, Any]:
    """Return the webhook event catalogue used by registration validation."""
    events = []
    for name in sorted(VALID_EVENTS):
        metadata = dict(_EVENT_METADATA.get(name, {}))
        canonical = str(metadata.get("canonical") or name)
        event = {
            "name": name,
            "canonical": canonical,
            "deprecated": bool(metadata.get("deprecated", False)),
            "since_version": str(metadata.get("since_version") or "1.32.0"),
            "schema_pointer": f"#/webhook-events/{canonical}",
            "description": str(metadata.get("description") or ""),
        }
        if canonical != name:
            event["replacement"] = canonical
        accepted = sorted(_EVENT_ALIASES.get(name, frozenset({name})))
        if len(accepted) > 1:
            event["subscription_aliases"] = accepted
        events.append(event)

    return {
        "events": events,
        "legacy_aliases": dict(sorted(_LEGACY_EVENT_ALIASES.items())),
    }


def list_event_types() -> Dict[str, Any]:
    """Compatibility wrapper with a route-oriented name."""
    return list_events()


def register_webhook(
    url: str,
    events: Optional[List[str]] = None,
    description: str = "",
    webhook_id: Optional[str] = None,
    secret: Optional[str] = None,
    allow_unsigned: bool = True,
    on_progress: Optional[Callable] = None,
) -> WebhookConfig:
    """Register or update a webhook endpoint.

    Args:
        url: The webhook URL to POST to.
        events: List of event types to subscribe to.
                If empty, subscribes to all events.
        description: Optional description.
        webhook_id: Optional ID for updating an existing webhook.
        secret: Optional HMAC signing secret. ``None`` preserves an existing
                secret during updates; an empty string clears it.
        allow_unsigned: When false, the resulting webhook must have a non-empty
                signing secret.
        on_progress: Optional progress callback (int).

    Returns:
        WebhookConfig for the registered webhook.
    """
    if on_progress:
        on_progress(30)

    url = _validate_webhook_url(url)

    # Validate event types
    if events:
        invalid = [e for e in events if e not in VALID_EVENTS]
        if invalid:
            raise ValueError(
                f"Invalid event types: {invalid}. "
                f"Valid: {sorted(VALID_EVENTS)}")

    with _config_lock:
        configs = _load_configs()

        if webhook_id:
            # Update existing
            for cfg in configs:
                if cfg.id == webhook_id:
                    effective_secret = cfg.secret if secret is None else str(secret or "").strip()
                    if not allow_unsigned and not effective_secret:
                        raise ValueError(
                            "Webhook signing secret is required. "
                            "Set secret or pass allow_unsigned=true for local testing.")
                    cfg.url = url
                    cfg.events = events or []
                    cfg.description = description
                    if secret is not None:
                        cfg.secret = effective_secret
                    _save_configs(configs)
                    if not cfg.secret:
                        _write_unsigned_warning()
                    if on_progress:
                        on_progress(100)
                    logger.info("Updated webhook '%s' -> %s", webhook_id, url)
                    return cfg
            # Not found — fall through to create new with that ID

        # Create new
        normalized_secret = str(secret or "").strip() if secret is not None else ""
        if not allow_unsigned and not normalized_secret:
            raise ValueError(
                "Webhook signing secret is required. "
                "Set secret or pass allow_unsigned=true for local testing.")
        webhook = WebhookConfig(
            id=webhook_id or uuid.uuid4().hex[:12],
            url=url,
            events=events or [],
            description=description,
            secret=normalized_secret,
        )
        configs.append(webhook)
        _save_configs(configs)
        if not webhook.secret:
            _write_unsigned_warning()

    if on_progress:
        on_progress(100)

    logger.info("Registered webhook '%s' -> %s (events=%s)",
                webhook.id, url, events or "all")
    return webhook


def unregister_webhook(
    webhook_id: str,
    on_progress: Optional[Callable] = None,
) -> bool:
    """Unregister a webhook by ID.

    Args:
        webhook_id: The webhook ID to remove.
        on_progress: Optional progress callback (int).

    Returns:
        True if removed, False if not found.
    """
    if on_progress:
        on_progress(50)

    with _config_lock:
        configs = _load_configs()
        original_count = len(configs)
        configs = [c for c in configs if c.id != webhook_id]

        if len(configs) == original_count:
            if on_progress:
                on_progress(100)
            return False

        _save_configs(configs)

    if on_progress:
        on_progress(100)

    logger.info("Unregistered webhook '%s'", webhook_id)
    return True


def list_webhooks(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all registered webhooks.

    Returns:
        List of webhook config dicts.
    """
    if on_progress:
        on_progress(50)

    with _config_lock:
        configs = _load_configs()
        result = [c.to_dict() for c in configs]

    if on_progress:
        on_progress(100)

    return result


def get_webhook(webhook_id: str) -> Optional[WebhookConfig]:
    """Get a single webhook by ID.

    Returns:
        WebhookConfig or None if not found.
    """
    with _config_lock:
        configs = _load_configs()
        for cfg in configs:
            if cfg.id == webhook_id:
                return cfg
    return None


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------

def _send_payload(
    url: str,
    event_type: str,
    payload: Dict[str, Any],
    timeout: int = 10,
    secret: str = "",
) -> tuple:
    """Send a single HTTP POST request.

    Returns:
        (status_code, success, error_message)
    """
    try:
        url = _validate_webhook_url(url)
        # Connect-time resolved-IP check (partial DNS-rebinding mitigation):
        # a hostname that validated public at registration can point private now.
        validate_public_http_url(url, label="Webhook URL", resolve=True)
    except ValueError as exc:
        return 0, False, str(exc)

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-OpenCut-Event": event_type,
        "X-OpenCut-Timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if secret:
        headers["X-OpenCut-Signature"] = sign_webhook_body(secret, body)
        headers["X-OpenCut-Signature-Algorithm"] = "HMAC-SHA256"

    try:
        from opencut.core.url_safety import build_guarded_opener

        req = urllib.request.Request(
            url, data=body, headers=headers, method="POST")
        # Follow redirects only through hops that pass the SSRF guard.
        with build_guarded_opener().open(req, timeout=timeout) as resp:
            status = resp.status
            if 200 <= status < 300:
                return status, True, ""
            return status, False, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        return exc.code, False, str(exc)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, False, str(exc)


def _event_matches_subscription(event_type: str, subscriptions: List[str]) -> bool:
    if not subscriptions:
        return True
    accepted = _EVENT_ALIASES.get(event_type, frozenset({event_type}))
    return any(event in accepted for event in subscriptions)


def fire_event(
    event_type: str,
    details: Dict[str, Any],
    job_id: str = "",
    on_progress: Optional[Callable] = None,
) -> List[WebhookDelivery]:
    """Fire an event to all registered webhooks for that event type.

    POST JSON payload:
    ``{"event_type": ..., "timestamp": ..., "job_id": ..., "details": ...}``

    Retry logic: up to 3 attempts with exponential backoff (1s, 5s, 15s).

    Args:
        event_type: Event type string.
        details: Event-specific payload data.
        job_id: Optional job ID for correlation.
        on_progress: Optional progress callback (int).

    Returns:
        List of WebhookDelivery records.
    """
    if on_progress:
        on_progress(10)

    with _config_lock:
        configs = _load_configs()
        # Filter to enabled webhooks subscribing to this event
        targets = [
            c for c in configs
            if c.enabled and _event_matches_subscription(event_type, c.events)
        ]

    if not targets:
        if on_progress:
            on_progress(100)
        return []

    payload = {
        "event_type": event_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": job_id,
        "details": details,
    }

    deliveries: List[WebhookDelivery] = []
    total = len(targets)

    for idx, webhook in enumerate(targets):
        if on_progress:
            pct = 10 + int(((idx + 1) / total) * 80)
            on_progress(pct)

        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            url=webhook.url,
            event_type=event_type,
            payload=payload,
        )

        success = False
        for attempt in range(MAX_RETRIES):
            status_code, ok, error_msg = _send_payload(
                webhook.url, event_type, payload, secret=webhook.secret)
            delivery.attempts = attempt + 1
            delivery.status_code = status_code

            if ok:
                delivery.success = True
                delivery.error = ""
                logger.info(
                    "Webhook delivered to %s (attempt %d, status %d)",
                    webhook.url, attempt + 1, status_code)
                success = True
                break

            delivery.error = error_msg
            logger.warning(
                "Webhook %s failed (attempt %d/%d): %s",
                webhook.url, attempt + 1, MAX_RETRIES, error_msg)

            # Backoff before retry (skip on last attempt)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                time.sleep(delay)

        if not success:
            logger.error(
                "Webhook delivery failed for %s after %d attempts",
                webhook.url, MAX_RETRIES)

        _append_delivery(delivery)
        deliveries.append(delivery)

    if on_progress:
        on_progress(100)

    return deliveries


def test_webhook(
    webhook_id: str,
    on_progress: Optional[Callable] = None,
) -> WebhookDelivery:
    """Send a test event to a specific webhook.

    Args:
        webhook_id: The webhook to test.
        on_progress: Optional progress callback (int).

    Returns:
        WebhookDelivery record for the test.

    Raises:
        ValueError: If the webhook is not found.
    """
    if on_progress:
        on_progress(20)

    webhook = get_webhook(webhook_id)
    if not webhook:
        raise ValueError(f"Webhook '{webhook_id}' not found")

    test_payload = {
        "event_type": "test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": "test_" + uuid.uuid4().hex[:8],
        "details": {
            "message": "This is a test event from OpenCut",
            "webhook_id": webhook_id,
        },
    }

    delivery = WebhookDelivery(
        webhook_id=webhook_id,
        url=webhook.url,
        event_type="test",
        payload=test_payload,
    )

    if on_progress:
        on_progress(50)

    status_code, ok, error_msg = _send_payload(
        webhook.url, "test", test_payload, secret=webhook.secret)
    delivery.status_code = status_code
    delivery.success = ok
    delivery.error = error_msg
    delivery.attempts = 1

    _append_delivery(delivery)

    if on_progress:
        on_progress(100)

    if ok:
        logger.info("Test webhook delivered to %s (status %d)",
                     webhook.url, status_code)
    else:
        logger.warning("Test webhook failed for %s: %s",
                        webhook.url, error_msg)

    return delivery


# ---------------------------------------------------------------------------
# Delivery log queries
# ---------------------------------------------------------------------------

def get_delivery_log(
    limit: int = _MAX_DELIVERIES,
    webhook_id: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Return the delivery log.

    Args:
        limit: Maximum entries to return.
        webhook_id: Optional filter by webhook ID.
        on_progress: Optional progress callback (int).

    Returns:
        List of delivery record dicts, newest last.
    """
    if on_progress:
        on_progress(50)

    with _delivery_lock:
        deliveries = _load_deliveries()

        if webhook_id:
            deliveries = [d for d in deliveries
                          if d.get("webhook_id") == webhook_id]

        result = deliveries[-limit:]

    if on_progress:
        on_progress(100)

    return result


def clear_delivery_log(
    on_progress: Optional[Callable] = None,
) -> None:
    """Clear the delivery log."""
    if on_progress:
        on_progress(50)
    _save_deliveries([])
    if on_progress:
        on_progress(100)
    logger.info("Cleared webhook delivery log")


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers
# ---------------------------------------------------------------------------

def send_webhook(
    url: str,
    event_type: str,
    payload: dict,
    timeout: int = 10,
) -> bool:
    """POST JSON payload to a webhook URL (legacy API).

    Retries once on failure.

    Returns:
        True if delivered (2xx), False otherwise.
    """
    for attempt in range(2):
        status, ok, error = _send_payload(url, event_type, payload, timeout)
        if ok:
            return True
        logger.warning("Webhook %s failed (attempt %d): %s",
                        url, attempt + 1, error)
    return False


def notify_job_complete(
    job_id: str,
    job_type: str,
    result: dict,
    webhook_urls: List[str],
) -> None:
    """Send a ``job_complete`` notification to a list of URLs (legacy API)."""
    payload = {
        "event": "job_complete",
        "job_id": job_id,
        "job_type": job_type,
        "status": "success" if not result.get("error") else "error",
        "result_summary": result,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for url in webhook_urls:
        send_webhook(url, "job_complete", payload)


def load_webhook_config() -> List[Dict]:
    """Load webhook configurations (legacy API)."""
    return [c.to_dict() for c in _load_configs()]


def save_webhook_config(configs: List[Dict]) -> None:
    """Persist webhook configurations (legacy API)."""
    existing = {config.id: config for config in _load_configs()}
    parsed = []
    for config in configs:
        cfg = WebhookConfig.from_dict(config)
        if "secret" not in config and cfg.id in existing:
            cfg.secret = existing[cfg.id].secret
        cfg.url = _validate_webhook_url(cfg.url)
        if cfg.events:
            invalid = [e for e in cfg.events if e not in VALID_EVENTS]
            if invalid:
                raise ValueError(
                    f"Invalid event types: {invalid}. Valid: {sorted(VALID_EVENTS)}"
                )
        parsed.append(cfg)
    _save_configs(parsed)
