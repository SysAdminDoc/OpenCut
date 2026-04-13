"""
OpenCut Zapier / Make Webhook Integrations Module

Outbound webhook triggers on events, and inbound webhook handlers
that map incoming payloads to OpenCut operations.

Registered webhooks are stored in ``~/.opencut/webhook_triggers.json``.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_TRIGGERS_FILE = os.path.join(_OPENCUT_DIR, "webhook_triggers.json")


@dataclass
class WebhookTrigger:
    """A registered outbound webhook trigger."""
    event: str = ""
    url: str = ""
    created: float = field(default_factory=time.time)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WebhookTrigger":
        return cls(
            event=d.get("event", ""),
            url=d.get("url", ""),
            created=d.get("created", time.time()),
            active=d.get("active", True),
        )


@dataclass
class WebhookResult:
    """Result of a webhook operation."""
    success: bool = False
    status_code: int = 0
    response_body: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Trigger persistence
# ---------------------------------------------------------------------------

def _load_triggers() -> List[WebhookTrigger]:
    """Load registered webhook triggers from disk."""
    if not os.path.isfile(_TRIGGERS_FILE):
        return []
    try:
        with open(_TRIGGERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [WebhookTrigger.from_dict(d) for d in data if isinstance(d, dict)]
    except (json.JSONDecodeError, OSError):
        return []


def _save_triggers(triggers: List[WebhookTrigger]) -> None:
    """Persist webhook triggers to disk."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    with open(_TRIGGERS_FILE, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in triggers], f, indent=2)


# ---------------------------------------------------------------------------
# Public API — Outbound
# ---------------------------------------------------------------------------

def send_webhook(
    url: str,
    event_type: str,
    payload: Dict[str, Any],
    timeout: int = 10,
    on_progress: Optional[Callable] = None,
) -> WebhookResult:
    """Send an outbound webhook POST to a Zapier/Make/custom endpoint.

    Args:
        url: Target webhook URL.
        event_type: Event identifier (e.g. ``export_complete``).
        payload: Arbitrary JSON-serialisable payload.
        timeout: HTTP timeout in seconds.
        on_progress: Optional progress callback.

    Returns:
        WebhookResult with delivery status.
    """
    if on_progress:
        on_progress(30, f"Sending webhook for {event_type}...")

    headers = {
        "Content-Type": "application/json",
        "X-OpenCut-Event": event_type,
        "X-OpenCut-Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    body = json.dumps({
        "event": event_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": payload,
    }).encode("utf-8")

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8", errors="replace")[:1000]
                if 200 <= resp.status < 300:
                    logger.info("Webhook delivered to %s (event=%s)", url, event_type)
                    if on_progress:
                        on_progress(100, "Webhook delivered")
                    return WebhookResult(
                        success=True,
                        status_code=resp.status,
                        response_body=resp_body,
                    )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
            if attempt == 1:
                if on_progress:
                    on_progress(100, "Webhook failed")
                return WebhookResult(
                    success=False, status_code=e.code, error=err_body,
                )
        except (urllib.error.URLError, OSError) as e:
            if attempt == 1:
                if on_progress:
                    on_progress(100, "Webhook failed")
                return WebhookResult(success=False, error=str(e))

    return WebhookResult(success=False, error="All retries failed")


def register_webhook_trigger(
    event: str,
    url: str,
    on_progress: Optional[Callable] = None,
) -> WebhookTrigger:
    """Register a webhook URL to be triggered on a specific event.

    Args:
        event: Event name (e.g. ``export_complete``, ``job_error``).
        url: Webhook endpoint URL.
        on_progress: Optional progress callback.

    Returns:
        The created WebhookTrigger.
    """
    if on_progress:
        on_progress(30, "Registering webhook trigger...")

    trigger = WebhookTrigger(event=event, url=url)

    triggers = _load_triggers()
    triggers.append(trigger)
    _save_triggers(triggers)

    if on_progress:
        on_progress(100, "Webhook trigger registered")

    logger.info("Registered webhook trigger: event=%s, url=%s", event, url)
    return trigger


def list_registered_webhooks(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all registered webhook triggers.

    Returns:
        List of webhook trigger dicts.
    """
    if on_progress:
        on_progress(50, "Loading webhook triggers...")

    triggers = _load_triggers()

    if on_progress:
        on_progress(100, "Done")

    return [t.to_dict() for t in triggers]


def remove_webhook_trigger(
    event: str,
    url: str,
    on_progress: Optional[Callable] = None,
) -> bool:
    """Remove a registered webhook trigger.

    Args:
        event: Event name.
        url: Webhook URL.

    Returns:
        True if a trigger was removed, False if not found.
    """
    triggers = _load_triggers()
    original_len = len(triggers)
    triggers = [t for t in triggers if not (t.event == event and t.url == url)]
    removed = len(triggers) < original_len

    if removed:
        _save_triggers(triggers)
        logger.info("Removed webhook trigger: event=%s, url=%s", event, url)

    if on_progress:
        on_progress(100, "Done")

    return removed


# ---------------------------------------------------------------------------
# Public API — Inbound
# ---------------------------------------------------------------------------

# Supported inbound operations
_INBOUND_OPERATIONS = {
    "export": "Start an export job",
    "trim": "Trim a video clip",
    "silence_remove": "Remove silence from audio",
    "caption": "Generate captions",
    "thumbnail": "Generate thumbnail",
    "transcode": "Transcode video",
}


def handle_inbound_webhook(
    data: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Handle an inbound webhook payload from Zapier/Make.

    The payload should contain:
      - ``operation``: The OpenCut operation to perform.
      - ``params``: Parameters for the operation.

    This function validates the request and returns a response dict.
    Actual execution should be dispatched by the route handler.

    Args:
        data: Inbound webhook payload.
        on_progress: Optional progress callback.

    Returns:
        Dict with ``accepted``, ``operation``, and ``params`` keys.
    """
    if on_progress:
        on_progress(30, "Processing inbound webhook...")

    operation = str(data.get("operation", "")).strip().lower()
    params = data.get("params", {})

    if not operation:
        return {
            "accepted": False,
            "error": "Missing 'operation' field",
            "supported_operations": list(_INBOUND_OPERATIONS.keys()),
        }

    if operation not in _INBOUND_OPERATIONS:
        return {
            "accepted": False,
            "error": f"Unknown operation: {operation}",
            "supported_operations": list(_INBOUND_OPERATIONS.keys()),
        }

    if on_progress:
        on_progress(100, f"Inbound webhook accepted: {operation}")

    logger.info("Inbound webhook accepted: operation=%s", operation)

    return {
        "accepted": True,
        "operation": operation,
        "params": params,
        "description": _INBOUND_OPERATIONS[operation],
    }
