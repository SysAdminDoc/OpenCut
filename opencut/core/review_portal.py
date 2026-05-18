"""Local-LAN review portal share links (F231).

The portal uses existing review-link data and adds a short-lived HMAC URL
that can be shared on a trusted LAN. It deliberately does not start Caddy or
publish mDNS records in-process; callers get deterministic descriptors they can
hand to the desktop launcher or installer-managed sidecars.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
from urllib.parse import urlencode

from opencut.core.review_links import _load_reviews

_SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass
class PortalShare:
    review_id: str
    url: str
    expires_at: int
    host: str
    port: int
    service_name: str
    hmac_algorithm: str
    caddyfile: str
    mdns: Dict[str, Any]

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_host(host: str) -> str:
    value = str(host or "").strip()
    if not value or len(value) > 253 or not _SAFE_HOST_RE.fullmatch(value):
        raise ValueError("host must be a DNS name, IP address, or mDNS name without a scheme")
    return value


def _safe_scheme(scheme: str) -> str:
    value = str(scheme or "http").strip().lower()
    if value not in {"http", "https"}:
        raise ValueError("scheme must be http or https")
    return value


def _safe_service_name(service_name: str) -> str:
    value = re.sub(r"\s+", " ", str(service_name or "OpenCut Review").strip())
    return value[:63] or "OpenCut Review"


def _review_data(review_id: str) -> dict:
    data = _load_reviews().get(review_id)
    if not isinstance(data, dict):
        raise KeyError(f"Review not found: {review_id}")
    return data


def _canonical_payload(review_id: str, expires_at: int) -> str:
    return f"review_id={review_id}&expires_at={int(expires_at)}"


def sign_portal_url(review_id: str, expires_at: int, secret: str) -> str:
    if not secret:
        raise ValueError("portal signing secret is required")
    digest = hmac.new(
        secret.encode("utf-8"),
        _canonical_payload(review_id, expires_at).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def verify_portal_signature(
    review_id: str,
    expires_at: int,
    signature: str,
    secret: str,
    *,
    now: float | None = None,
) -> bool:
    if int(expires_at) < int(now if now is not None else time.time()):
        return False
    expected = sign_portal_url(review_id, int(expires_at), secret)
    return hmac.compare_digest(expected, str(signature or "").strip())


def build_caddyfile(host: str, port: int, *, upstream_host: str = "127.0.0.1", upstream_port: int = 5679) -> str:
    host_value = _safe_host(host)
    port_value = max(1, min(65535, int(port)))
    upstream = f"{_safe_host(upstream_host)}:{max(1, min(65535, int(upstream_port)))}"
    return (
        f"{host_value}:{port_value} {{\n"
        f"    reverse_proxy {upstream}\n"
        "    header {\n"
        "        X-OpenCut-Review-Portal \"lan\"\n"
        "    }\n"
        "}\n"
    )


def build_mdns_descriptor(host: str, port: int, service_name: str = "OpenCut Review") -> Dict[str, Any]:
    return {
        "instance": _safe_service_name(service_name),
        "service_type": "_http._tcp.local.",
        "host": _safe_host(host),
        "port": max(1, min(65535, int(port))),
        "txt": {
            "app": "OpenCut",
            "purpose": "review-portal",
            "auth": "hmac-url",
        },
    }


def build_portal_share(
    *,
    review_id: str,
    host: str,
    port: int,
    scheme: str = "http",
    ttl_seconds: int = 86400,
    service_name: str = "OpenCut Review",
    now: float | None = None,
) -> PortalShare:
    review = _review_data(review_id)
    token = str(review.get("token") or "")
    if not token:
        raise ValueError(f"Review has no token: {review_id}")
    issued = int(now if now is not None else time.time())
    ttl = max(60, min(604800, int(ttl_seconds)))
    expires_at = issued + ttl
    safe_host = _safe_host(host)
    safe_port = max(1, min(65535, int(port)))
    safe_scheme = _safe_scheme(scheme)
    signature = sign_portal_url(review_id, expires_at, token)
    query = urlencode({"expires": expires_at, "sig": signature})
    url = f"{safe_scheme}://{safe_host}:{safe_port}/review/portal/{review_id}?{query}"
    return PortalShare(
        review_id=review_id,
        url=url,
        expires_at=expires_at,
        host=safe_host,
        port=safe_port,
        service_name=_safe_service_name(service_name),
        hmac_algorithm="HMAC-SHA256",
        caddyfile=build_caddyfile(safe_host, safe_port),
        mdns=build_mdns_descriptor(safe_host, safe_port, service_name),
    )


def resolve_portal_review(review_id: str, expires_at: int, signature: str) -> Dict[str, Any]:
    review = _review_data(review_id)
    token = str(review.get("token") or "")
    if not verify_portal_signature(review_id, int(expires_at), signature, token):
        raise PermissionError("invalid or expired portal signature")
    comments: List[dict] = []
    for raw in review.get("comments", []) or []:
        if not isinstance(raw, dict):
            continue
        comments.append(
            {
                "comment_id": raw.get("comment_id", ""),
                "timestamp": float(raw.get("timestamp") or 0.0),
                "text": str(raw.get("text") or ""),
                "author": str(raw.get("author") or ""),
                "created_at": float(raw.get("created_at") or 0.0),
            }
        )
    comments.sort(key=lambda item: (item["timestamp"], item["created_at"], item["comment_id"]))
    return {
        "review_id": review_id,
        "title": review.get("title") or os.path.basename(str(review.get("video_path") or "")),
        "status": review.get("status", "pending"),
        "video_basename": os.path.basename(str(review.get("video_path") or "")),
        "expires_at": int(expires_at),
        "comments": comments,
        "comment_count": len(comments),
    }


def render_portal_html(payload: Dict[str, Any]) -> str:
    comments = payload.get("comments") or []
    rows = []
    for comment in comments:
        rows.append(
            "<li>"
            f"<time>{float(comment.get('timestamp') or 0.0):.2f}s</time> "
            f"<strong>{html.escape(str(comment.get('author') or 'Anonymous'))}</strong>"
            f"<p>{html.escape(str(comment.get('text') or ''))}</p>"
            "</li>"
        )
    comments_html = "\n".join(rows) if rows else "<li>No comments yet.</li>"
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>OpenCut Review</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#17202a}"
        "h1{font-size:1.5rem}li{border-bottom:1px solid #d6dde5;padding:.75rem 0;list-style:none}"
        "time{color:#52616f;margin-right:.5rem}p{margin:.35rem 0 0}</style></head><body>"
        f"<h1>{html.escape(str(payload.get('title') or 'OpenCut Review'))}</h1>"
        f"<p>Status: <strong>{html.escape(str(payload.get('status') or 'pending'))}</strong><br>"
        f"Media: <code>{html.escape(str(payload.get('video_basename') or ''))}</code><br>"
        f"Comments: {int(payload.get('comment_count') or 0)}</p>"
        f"<ul>{comments_html}</ul>"
        "</body></html>\n"
    )
