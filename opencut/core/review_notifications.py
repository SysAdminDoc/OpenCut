"""Review notification helpers for Atom feeds and webhook details (F233)."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import quote
from xml.etree.ElementTree import Element, SubElement, register_namespace, tostring

from opencut.core.review_links import _load_reviews

ATOM_NS = "http://www.w3.org/2005/Atom"
register_namespace("", ATOM_NS)


def _atom(tag: str) -> str:
    return f"{{{ATOM_NS}}}{tag}"


def _iso_timestamp(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        ts = time.time()
    if ts <= 0:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _project_id(review: Dict[str, Any]) -> str:
    return str(review.get("project_id") or review.get("project_path") or "default")


def _review_title(review: Dict[str, Any], review_id: str) -> str:
    return str(review.get("title") or os.path.basename(str(review.get("video_path") or "")) or review_id)


def _version_data(review: Dict[str, Any], version_id: str = "") -> Dict[str, Any]:
    selected_id = version_id or str(review.get("current_version_id") or "")
    for version in review.get("versions", []) or []:
        if isinstance(version, dict) and version.get("version_id") == selected_id:
            return version
    return review


def _review_event(review_id: str, review: Dict[str, Any]) -> Dict[str, Any]:
    version_id = str(review.get("current_version_id") or "")
    version = _version_data(review, version_id)
    status = str(version.get("status") or "pending")
    updated_at = float(version.get("status_updated_at") or version.get("created_at") or time.time())
    title = _review_title(review, review_id)
    return {
        "id": f"urn:opencut:review:{review_id}:{version_id}:status:{status}:{int(updated_at)}",
        "event_type": "review.status_changed",
        "review_id": review_id,
        "project_id": _project_id(review),
        "title": f"Review {status}: {title}",
        "summary": f"Review status is {status}.",
        "updated_at": updated_at,
        "status": status,
        "version_id": version_id,
        "video_basename": os.path.basename(str(version.get("video_path") or "")),
    }


def _comment_event(review_id: str, review: Dict[str, Any], comment: Dict[str, Any]) -> Dict[str, Any]:
    comment_id = str(comment.get("comment_id") or "")
    updated_at = float(comment.get("created_at") or review.get("created_at") or time.time())
    author = str(comment.get("author") or "Anonymous")
    title = _review_title(review, review_id)
    timestamp = float(comment.get("timestamp") or 0.0)
    text = str(comment.get("text") or "")
    version_id = str(comment.get("version_id") or review.get("current_version_id") or "")
    version = _version_data(review, version_id)
    return {
        "id": f"urn:opencut:review:{review_id}:{version_id}:comment:{comment_id or int(updated_at)}",
        "event_type": "review.comment_added",
        "review_id": review_id,
        "project_id": _project_id(review),
        "title": f"Comment from {author} on {title}",
        "summary": f"{timestamp:.2f}s - {text}",
        "updated_at": updated_at,
        "status": str(version.get("status") or "pending"),
        "version_id": version_id,
        "video_basename": os.path.basename(str(version.get("video_path") or "")),
        "comment": {
            "comment_id": comment_id,
            "timestamp": timestamp,
            "text": text,
            "author": author,
            "created_at": updated_at,
            "version_id": version_id,
        },
    }


def collect_review_events(
    *,
    project_id: str = "",
    review_id: str = "",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Return newest-first review notification events from saved review links."""
    project_filter = str(project_id or "").strip()
    review_filter = str(review_id or "").strip()
    events: List[Dict[str, Any]] = []
    for current_review_id, review in _load_reviews().items():
        if not isinstance(review, dict):
            continue
        if review_filter and current_review_id != review_filter:
            continue
        if project_filter and _project_id(review) != project_filter:
            continue
        events.append(_review_event(current_review_id, review))
        for raw_comment in review.get("comments", []) or []:
            if isinstance(raw_comment, dict):
                events.append(_comment_event(current_review_id, review, raw_comment))
    events.sort(key=lambda item: (float(item.get("updated_at") or 0.0), item.get("id", "")), reverse=True)
    return events[: max(1, min(500, int(limit)))]


def build_review_webhook_details(
    *,
    review_id: str,
    event_type: str,
    comment: Dict[str, Any] | None = None,
    status: str = "",
    version_id: str = "",
) -> Dict[str, Any]:
    """Build the review-specific details object sent through webhook_system."""
    reviews = _load_reviews()
    review = reviews.get(review_id)
    if not isinstance(review, dict):
        raise KeyError(f"Review not found: {review_id}")
    selected_version_id = version_id or str(review.get("current_version_id") or "")
    version = _version_data(review, selected_version_id)
    details: Dict[str, Any] = {
        "event_type": event_type,
        "review_id": review_id,
        "project_id": _project_id(review),
        "title": _review_title(review, review_id),
        "status": status or str(version.get("status") or "pending"),
        "version_id": selected_version_id,
        "video_basename": os.path.basename(str(version.get("video_path") or "")),
    }
    if comment:
        details["comment"] = {
            "comment_id": str(comment.get("comment_id") or ""),
            "timestamp": float(comment.get("timestamp") or 0.0),
            "text": str(comment.get("text") or ""),
            "author": str(comment.get("author") or "Anonymous"),
            "created_at": float(comment.get("created_at") or time.time()),
            "version_id": str(comment.get("version_id") or selected_version_id),
        }
    return details


def build_review_atom_feed(
    *,
    project_id: str = "",
    review_id: str = "",
    base_url: str = "",
    limit: int = 100,
) -> str:
    """Render an Atom feed for review comments/status changes."""
    events = collect_review_events(project_id=project_id, review_id=review_id, limit=limit)
    label = project_id or review_id or "all"
    feed = Element(_atom("feed"))
    SubElement(feed, _atom("id")).text = f"urn:opencut:review-feed:{label}"
    SubElement(feed, _atom("title")).text = f"OpenCut Review Notifications - {label}"
    SubElement(feed, _atom("updated")).text = _iso_timestamp(events[0]["updated_at"] if events else time.time())
    SubElement(feed, _atom("generator")).text = "OpenCut"

    if base_url:
        href = f"{base_url.rstrip('/')}/review/feed.atom"
        params = []
        if project_id:
            params.append(f"project_id={quote(project_id)}")
        if review_id:
            params.append(f"review_id={quote(review_id)}")
        if params:
            href += "?" + "&".join(params)
        link = SubElement(feed, _atom("link"))
        link.set("rel", "self")
        link.set("href", href)

    for event in events:
        entry = SubElement(feed, _atom("entry"))
        SubElement(entry, _atom("id")).text = str(event["id"])
        SubElement(entry, _atom("title")).text = str(event["title"])
        SubElement(entry, _atom("updated")).text = _iso_timestamp(event["updated_at"])
        SubElement(entry, _atom("summary")).text = str(event["summary"])
        category = SubElement(entry, _atom("category"))
        category.set("term", str(event["event_type"]))
        category.set("label", str(event["event_type"]))
        review_link = SubElement(entry, _atom("link"))
        review_link.set("rel", "alternate")
        review_link.set("href", f"urn:opencut:review:{event['review_id']}")

    return "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n" + tostring(feed, encoding="unicode")
