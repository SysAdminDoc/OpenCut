"""Best-effort JSONL audit trail for security rejections."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("opencut")

AUDIT_ENV = "OPENCUT_SECURITY_AUDIT_LOG"
DEFAULT_SECURITY_AUDIT_LOG = os.path.join(
    os.path.expanduser("~"),
    ".opencut",
    "security_audit.jsonl",
)
SCHEMA = "opencut.security-audit.v1"

_audit_lock = threading.Lock()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _audit_path() -> str:
    configured = os.environ.get(AUDIT_ENV)
    if configured is not None:
        configured = configured.strip()
        if not configured:
            return ""
        return os.path.abspath(os.path.expanduser(configured))
    try:
        from flask import current_app, has_app_context

        if has_app_context() and current_app.config.get("TESTING"):
            return ""
    except Exception:  # noqa: BLE001 - audit path discovery must stay best-effort
        pass
    return DEFAULT_SECURITY_AUDIT_LOG


def security_audit_log_path() -> str:
    """Return the active audit log path, or an empty string when disabled."""
    return _audit_path()


def _safe_preview(value: Any, *, limit: int = 180) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "\\0").replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _hash_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    return _safe_preview(value)


def _request_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        from flask import g, has_request_context, request

        if has_request_context():
            fields.update(
                {
                    "method": request.method,
                    "path": request.path,
                    "remote_addr": request.remote_addr or "",
                    "request_id": getattr(g, "request_id", "") or "",
                }
            )
            user_agent = _safe_preview(request.headers.get("User-Agent", ""), limit=120)
            if user_agent:
                fields["user_agent"] = user_agent
    except Exception:  # noqa: BLE001 - audit collection must not affect requests
        pass

    if not fields.get("request_id"):
        try:
            from opencut.core.request_correlation import get_request_id

            request_id = get_request_id()
            if request_id:
                fields["request_id"] = request_id
        except Exception:  # noqa: BLE001 - best-effort enrichment only
            pass
    return fields


def record_security_event(
    event: str,
    reason: str,
    *,
    severity: str = "warning",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Append one security audit record and return it for callers/tests."""
    entry: dict[str, Any] = {
        "schema": SCHEMA,
        "timestamp": _utc_timestamp(),
        "event": _safe_preview(event, limit=80),
        "severity": _safe_preview(severity, limit=40),
        "reason": _safe_preview(reason),
    }
    entry.update(_request_fields())
    if metadata:
        entry["metadata"] = _json_ready(metadata)

    target = _audit_path()
    if not target:
        return {"path": "", "entry": entry, "written": False}

    try:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
        with _audit_lock:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return {"path": target, "entry": entry, "written": True}
    except Exception as exc:  # noqa: BLE001 - rejected requests must still reject normally
        logger.warning("security audit write failed: %s", exc)
        return {"path": target, "entry": entry, "written": False}


def record_csrf_rejection(*, token_present: bool) -> dict[str, Any]:
    return record_security_event(
        "csrf_rejected",
        "Invalid or missing CSRF token",
        metadata={"token_present": bool(token_present)},
    )


def record_path_validation_rejection(
    path: Any,
    reason: str,
    *,
    allowed_base: Optional[str] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path_preview": _safe_preview(path),
        "path_sha256": _hash_value(path),
    }
    if allowed_base is not None:
        metadata["allowed_base_preview"] = _safe_preview(allowed_base)
        metadata["allowed_base_sha256"] = _hash_value(allowed_base)
    return record_security_event(
        "path_validation_rejected",
        reason,
        metadata=metadata,
    )


def record_rate_limit_rejection(key: str, *, current: int, max_concurrent: int) -> dict[str, Any]:
    return record_security_event(
        "rate_limit_rejected",
        "Concurrency limit reached",
        metadata={
            "key": _safe_preview(key, limit=120),
            "current": int(current),
            "max_concurrent": int(max_concurrent),
        },
    )


def record_auth_token_rejection() -> dict[str, Any]:
    return record_security_event(
        "auth_token_rejected",
        "Missing or invalid X-OpenCut-Auth token",
    )


def read_security_events(limit: int = 100) -> list[dict[str, Any]]:
    """Read the most recent audit entries in chronological order."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 1000))

    target = _audit_path()
    if not target or not os.path.exists(target):
        return []

    events: deque[dict[str, Any]] = deque(maxlen=limit)
    with _audit_lock:
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    events.append(parsed)
    return list(events)
