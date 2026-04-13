"""
OpenCut Slack / Discord Notification Module

Send rich notifications to Slack and Discord webhooks on events
such as export completion, job errors, and render milestones.

Both platforms use incoming webhook URLs.  Messages are formatted
as rich embeds (Slack blocks / Discord embeds).
"""

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class NotificationResult:
    """Result of a notification send operation."""
    success: bool = False
    platform: str = ""
    status_code: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

def send_slack_notification(
    webhook_url: str,
    message: str,
    title: str = "OpenCut",
    color: str = "#36a64f",
    fields: Optional[List[Dict[str, str]]] = None,
    on_progress: Optional[Callable] = None,
) -> NotificationResult:
    """Send a rich notification to a Slack incoming webhook.

    Args:
        webhook_url: Slack incoming webhook URL.
        message: Main message text.
        title: Attachment title.
        color: Sidebar color hex.
        fields: Optional list of ``{"title": ..., "value": ..., "short": True/False}`` dicts.
        on_progress: Optional progress callback.

    Returns:
        NotificationResult with delivery status.
    """
    if on_progress:
        on_progress(30, "Sending Slack notification...")

    attachment = {
        "color": color,
        "title": title,
        "text": message,
        "ts": int(time.time()),
    }

    if fields:
        attachment["fields"] = [
            {
                "title": f.get("title", ""),
                "value": f.get("value", ""),
                "short": f.get("short", True),
            }
            for f in fields[:10]
        ]

    payload = {
        "attachments": [attachment],
    }

    result = _post_webhook(webhook_url, payload, "slack")

    if on_progress:
        on_progress(100, "Slack notification sent" if result.success else "Slack send failed")

    return result


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def send_discord_notification(
    webhook_url: str,
    message: str,
    title: str = "OpenCut",
    color: int = 0x36A64F,
    fields: Optional[List[Dict[str, str]]] = None,
    on_progress: Optional[Callable] = None,
) -> NotificationResult:
    """Send a rich notification to a Discord webhook.

    Args:
        webhook_url: Discord webhook URL.
        message: Main message / description text.
        title: Embed title.
        color: Embed sidebar color as integer.
        fields: Optional list of ``{"name": ..., "value": ..., "inline": True/False}`` dicts.
        on_progress: Optional progress callback.

    Returns:
        NotificationResult with delivery status.
    """
    if on_progress:
        on_progress(30, "Sending Discord notification...")

    embed = {
        "title": title,
        "description": message,
        "color": color,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if fields:
        embed["fields"] = [
            {
                "name": f.get("name", f.get("title", "")),
                "value": f.get("value", ""),
                "inline": f.get("inline", f.get("short", True)),
            }
            for f in fields[:25]
        ]

    payload = {
        "embeds": [embed],
    }

    result = _post_webhook(webhook_url, payload, "discord")

    if on_progress:
        on_progress(100, "Discord notification sent" if result.success else "Discord send failed")

    return result


# ---------------------------------------------------------------------------
# Job notification formatter
# ---------------------------------------------------------------------------

def format_job_notification(
    job_data: Dict[str, Any],
    format: str = "slack",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Format job completion data into a platform-specific notification payload.

    Args:
        job_data: Job result dict with keys like ``job_id``, ``job_type``,
            ``status``, ``progress``, ``output_path``, ``error``, ``duration``.
        format: Target platform: ``"slack"`` or ``"discord"``.
        on_progress: Optional progress callback.

    Returns:
        Formatted payload dict ready for the corresponding send function.
    """
    if on_progress:
        on_progress(50, "Formatting notification...")

    status = job_data.get("status", "unknown")
    job_type = job_data.get("job_type", "job")
    job_id = job_data.get("job_id", "")[:8]

    is_success = status in ("complete", "success", "done")
    is_error = status in ("error", "failed")

    if is_success:
        emoji = ":white_check_mark:" if format == "slack" else "\\u2705"
        title = f"{emoji} {job_type.title()} Complete"
        color_slack = "#36a64f"
        color_discord = 0x36A64F
    elif is_error:
        emoji = ":x:" if format == "slack" else "\\u274C"
        title = f"{emoji} {job_type.title()} Failed"
        color_slack = "#e74c3c"
        color_discord = 0xE74C3C
    else:
        emoji = ":hourglass:" if format == "slack" else "\\u23F3"
        title = f"{emoji} {job_type.title()} - {status.title()}"
        color_slack = "#f39c12"
        color_discord = 0xF39C12

    message = f"Job `{job_id}` — **{job_type}** completed with status: {status}"
    if job_data.get("error"):
        message += f"\nError: {str(job_data['error'])[:200]}"
    if job_data.get("output_path"):
        message += f"\nOutput: `{job_data['output_path']}`"

    fields = []
    if job_data.get("duration"):
        fields.append({"title": "Duration", "name": "Duration",
                        "value": f"{job_data['duration']:.1f}s", "short": True, "inline": True})
    if job_data.get("progress") is not None:
        fields.append({"title": "Progress", "name": "Progress",
                        "value": f"{job_data['progress']}%", "short": True, "inline": True})

    result = {
        "message": message,
        "title": title,
        "fields": fields,
    }

    if format == "slack":
        result["color"] = color_slack
    else:
        result["color"] = color_discord

    if on_progress:
        on_progress(100, "Notification formatted")

    return result


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _post_webhook(
    url: str,
    payload: Dict[str, Any],
    platform: str,
    timeout: int = 10,
) -> NotificationResult:
    """POST a JSON payload to a webhook URL with retry."""
    headers = {"Content-Type": "application/json"}
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if 200 <= status < 300:
                    logger.info("%s notification delivered (attempt %d)", platform, attempt + 1)
                    return NotificationResult(
                        success=True, platform=platform, status_code=status,
                    )
                logger.warning("%s webhook returned %d (attempt %d)", platform, status, attempt + 1)
                return NotificationResult(
                    success=False, platform=platform, status_code=status,
                    error=f"HTTP {status}",
                )
        except urllib.error.HTTPError as e:
            logger.warning("%s webhook HTTP error %d (attempt %d)", platform, e.code, attempt + 1)
            if attempt == 1:
                return NotificationResult(
                    success=False, platform=platform, status_code=e.code,
                    error=f"HTTP {e.code}",
                )
        except (urllib.error.URLError, OSError) as e:
            logger.warning("%s webhook failed (attempt %d): %s", platform, attempt + 1, e)
            if attempt == 1:
                return NotificationResult(
                    success=False, platform=platform,
                    error=str(e),
                )

    return NotificationResult(success=False, platform=platform, error="All retries failed")
