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
from urllib.parse import urlparse

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
})

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
    created: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            url=data.get("url", ""),
            events=data.get("events", []),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
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


def _validate_webhook_url(url: str) -> str:
    """Validate and normalize outbound webhook URLs."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError("Webhook URL is required")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Webhook URL must use http:// or https:// and include a host")
    if any(ch in cleaned for ch in ("\r", "\n", "\x00")):
        raise ValueError("Webhook URL contains invalid characters")
    return cleaned


def _atomic_write_json(path: str, payload: Any) -> None:
    """Atomically replace *path* with UTF-8 JSON content."""
    import tempfile

    _ensure_dir()
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_configs() -> List[WebhookConfig]:
    """Load webhook configs from disk."""
    with _config_lock:
        if not os.path.isfile(_WEBHOOKS_FILE):
            return []
        try:
            with open(_WEBHOOKS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [WebhookConfig.from_dict(d) for d in data]
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load webhook configs: %s", exc)
            return []


def _save_configs(configs: List[WebhookConfig]) -> None:
    """Persist webhook configs to disk."""
    with _config_lock:
        _atomic_write_json(_WEBHOOKS_FILE, [c.to_dict() for c in configs])
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

def register_webhook(
    url: str,
    events: Optional[List[str]] = None,
    description: str = "",
    webhook_id: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> WebhookConfig:
    """Register or update a webhook endpoint.

    Args:
        url: The webhook URL to POST to.
        events: List of event types to subscribe to.
                If empty, subscribes to all events.
        description: Optional description.
        webhook_id: Optional ID for updating an existing webhook.
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
                    cfg.url = url
                    cfg.events = events or []
                    cfg.description = description
                    _save_configs(configs)
                    if on_progress:
                        on_progress(100)
                    logger.info("Updated webhook '%s' -> %s", webhook_id, url)
                    return cfg
            # Not found — fall through to create new with that ID

        # Create new
        webhook = WebhookConfig(
            id=webhook_id or uuid.uuid4().hex[:12],
            url=url,
            events=events or [],
            description=description,
        )
        configs.append(webhook)
        _save_configs(configs)

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
) -> tuple:
    """Send a single HTTP POST request.

    Returns:
        (status_code, success, error_message)
    """
    try:
        url = _validate_webhook_url(url)
    except ValueError as exc:
        return 0, False, str(exc)

    headers = {
        "Content-Type": "application/json",
        "X-OpenCut-Event": event_type,
        "X-OpenCut-Timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if 200 <= status < 300:
                return status, True, ""
            return status, False, f"HTTP {status}"
    except urllib.error.HTTPError as exc:
        return exc.code, False, str(exc)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, False, str(exc)


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
            if c.enabled and (not c.events or event_type in c.events)
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
                webhook.url, event_type, payload)
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
        webhook.url, "test", test_payload)
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
    parsed = []
    for config in configs:
        cfg = WebhookConfig.from_dict(config)
        cfg.url = _validate_webhook_url(cfg.url)
        if cfg.events:
            invalid = [e for e in cfg.events if e not in VALID_EVENTS]
            if invalid:
                raise ValueError(
                    f"Invalid event types: {invalid}. Valid: {sorted(VALID_EVENTS)}"
                )
        parsed.append(cfg)
    _save_configs(parsed)
