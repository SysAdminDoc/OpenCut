"""
OpenCut Clip Rating v1.28.0

Star rating, status, and tag management for clips. Stored in ~/.opencut/clip_db.json.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "clip_db.json")


def check_clip_rating_available() -> bool:
    """Always True — stdlib JSON only."""
    return True


@dataclass
class ClipEntry:
    path: str = ""
    rating: int = 0
    status: str = "neutral"
    tags: List[str] = field(default_factory=list)
    updated: str = ""

    def __getitem__(self, k: str):
        return getattr(self, k)

    def keys(self):
        return ("path", "rating", "status", "tags", "updated")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def _load_db() -> dict:
    try:
        with open(_DB_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_db(db: dict) -> None:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with open(_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def _entry_from_dict(path: str, d: dict) -> ClipEntry:
    return ClipEntry(
        path=path,
        rating=int(d.get("rating", 0)),
        status=str(d.get("status", "neutral")),
        tags=list(d.get("tags", [])),
        updated=str(d.get("updated", "")),
    )


def rate(clip_path: str, rating: Optional[int] = None, status: Optional[str] = None) -> ClipEntry:
    """Set rating (1-5) and/or status (good/neutral/rejected) for a clip."""
    db = _load_db()
    entry = db.get(clip_path, {"rating": 0, "status": "neutral", "tags": []})
    if rating is not None:
        entry["rating"] = max(0, min(5, int(rating)))
    if status is not None:
        if status not in ("good", "neutral", "rejected"):
            raise ValueError("status must be 'good', 'neutral', or 'rejected'")
        entry["status"] = status
    entry["updated"] = datetime.now(timezone.utc).isoformat()
    db[clip_path] = entry
    _save_db(db)
    return _entry_from_dict(clip_path, entry)


def tag(clip_path: str, tags: List[str]) -> ClipEntry:
    """Add tags to a clip (deduped)."""
    db = _load_db()
    entry = db.get(clip_path, {"rating": 0, "status": "neutral", "tags": []})
    existing = set(entry.get("tags", []))
    existing.update(str(t) for t in tags)
    entry["tags"] = sorted(existing)
    entry["updated"] = datetime.now(timezone.utc).isoformat()
    db[clip_path] = entry
    _save_db(db)
    return _entry_from_dict(clip_path, entry)


def untag(clip_path: str, tags: List[str]) -> ClipEntry:
    """Remove tags from a clip."""
    db = _load_db()
    entry = db.get(clip_path, {"rating": 0, "status": "neutral", "tags": []})
    remove = set(str(t) for t in tags)
    entry["tags"] = [t for t in entry.get("tags", []) if t not in remove]
    entry["updated"] = datetime.now(timezone.utc).isoformat()
    db[clip_path] = entry
    _save_db(db)
    return _entry_from_dict(clip_path, entry)


def search(
    query: str = "",
    rating_min: int = 0,
    status: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[ClipEntry]:
    """Filter clips by query, rating, status, tags."""
    db = _load_db()
    results = []
    q = query.lower()
    for path, d in db.items():
        if rating_min and int(d.get("rating", 0)) < rating_min:
            continue
        if status and d.get("status") != status:
            continue
        if tags:
            clip_tags = set(d.get("tags", []))
            if not all(t in clip_tags for t in tags):
                continue
        if q and q not in path.lower():
            continue
        results.append(_entry_from_dict(path, d))
    return results


def get(clip_path: str) -> Optional[ClipEntry]:
    """Get a single clip entry or None."""
    db = _load_db()
    d = db.get(clip_path)
    if d is None:
        return None
    return _entry_from_dict(clip_path, d)


__all__ = ["check_clip_rating_available", "ClipEntry", "rate", "tag", "untag", "search", "get"]
