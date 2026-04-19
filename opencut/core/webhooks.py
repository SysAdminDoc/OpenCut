"""
OpenCut Webhook / API Notifications Module

Send POST-based webhook notifications for job events.
Supports retry on failure, configurable per-URL endpoints,
and persistent config in ``~/.opencut/webhooks.json``.
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List

from opencut.core.url_safety import validate_public_http_url

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_WEBHOOKS_FILE = os.path.join(_OPENCUT_DIR, "webhooks.json")
_config_lock = threading.RLock()


def _validate_webhook_url(url: str) -> str:
    """Validate and normalize a legacy webhook URL."""
    return validate_public_http_url(url, label="Webhook URL")


def _atomic_write_json(path: str, payload: object) -> None:
    """Atomically replace *path* with UTF-8 JSON content."""
    import tempfile

    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def send_webhook(
    url: str,
    event_type: str,
    payload: dict,
    timeout: int = 10,
) -> bool:
    """
    POST JSON payload to a webhook URL.

    Headers sent:
      - Content-Type: application/json
      - X-OpenCut-Event: <event_type>
      - X-OpenCut-Timestamp: <ISO 8601>

    Retries once on failure.

    Returns:
        True if the webhook was delivered (2xx), False otherwise.
    """
    try:
        url = _validate_webhook_url(url)
    except ValueError as exc:
        logger.warning("Rejected webhook URL: %s", exc)
        return False

    headers = {
        "Content-Type": "application/json",
        "X-OpenCut-Event": event_type,
        "X-OpenCut-Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    body = json.dumps(payload).encode("utf-8")

    for attempt in range(2):  # initial + 1 retry
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    logger.info("Webhook delivered to %s (attempt %d)", url, attempt + 1)
                    return True
                logger.warning("Webhook %s returned %d (attempt %d)", url, resp.status, attempt + 1)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("Webhook %s failed (attempt %d): %s", url, attempt + 1, exc)
        except Exception as exc:
            logger.error("Webhook %s unexpected error (attempt %d): %s", url, attempt + 1, exc)
            break  # don't retry on unexpected errors

    return False


def notify_job_complete(
    job_id: str,
    job_type: str,
    result: dict,
    webhook_urls: List[str],
) -> None:
    """
    Send a ``job_complete`` notification to all configured webhook URLs.

    Payload format::

        {
            "event": "job_complete",
            "job_id": "abc123",
            "job_type": "export",
            "status": "success",
            "result_summary": { ... },
            "timestamp": "2026-04-13T12:00:00Z"
        }
    """
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
    """
    Load webhook configurations from ``~/.opencut/webhooks.json``.

    Returns a list of webhook config dicts, each with at least ``url`` and
    optional ``events`` filter list.  Returns empty list if the file does
    not exist.
    """
    with _config_lock:
        if not os.path.isfile(_WEBHOOKS_FILE):
            return []

        try:
            with open(_WEBHOOKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load webhook config: %s", exc)
            return []


def save_webhook_config(configs: List[Dict]) -> None:
    """
    Persist webhook configurations to ``~/.opencut/webhooks.json``.

    Each config dict should have at minimum a ``url`` key.
    """
    sanitized = []
    for cfg in configs:
        if not isinstance(cfg, dict):
            raise ValueError("Each webhook config must be an object")
        clean_cfg = dict(cfg)
        clean_cfg["url"] = _validate_webhook_url(clean_cfg.get("url", ""))
        sanitized.append(clean_cfg)
    with _config_lock:
        _atomic_write_json(_WEBHOOKS_FILE, sanitized)
    logger.info("Saved %d webhook config(s)", len(sanitized))
