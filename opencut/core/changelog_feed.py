"""
OpenCut Changelog Feed (Wave H1.4, v1.25.0)

Fetches recent GitHub releases for the panel's "new version" toast.
Pure-stdlib (``urllib.request``) with a 15 min process-local cache and
graceful fallback when GitHub is unreachable — the panel must keep
working offline.

State persisted in ``~/.opencut/changelog_seen.json`` via user_data
wrappers so multiple Flask threads can't race on the JSON write.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from opencut.user_data import read_user_file, write_user_file

logger = logging.getLogger("opencut")

GITHUB_REPO = "SysAdminDoc/OpenCut"
GITHUB_API_RELEASES = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
DEFAULT_USER_AGENT = "OpenCut-Panel/1.25.0"
SEEN_FILE = "changelog_seen.json"

_CACHE: Dict[str, Any] = {"releases": None, "expires": 0.0, "fetched_at": 0.0}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 900.0


def check_changelog_available() -> bool:
    """Always True — stdlib urllib. Fails gracefully on network errors."""
    return True


def fetch_releases(limit: int = 5, timeout: float = 6.0) -> Dict[str, Any]:
    """Fetch the N most-recent GitHub releases. Never raises.

    Returns::

        {"releases": [...], "source": "github"|"cache"|"fallback",
         "fetched_at": <unix ts>, "note": "..."}
    """
    limit = max(1, min(int(limit or 5), 25))
    now = time.monotonic()
    with _CACHE_LOCK:
        cache = _CACHE.get("releases")
        if cache is not None and now < _CACHE.get("expires", 0.0):
            return {
                "releases": cache[:limit],
                "source": "cache",
                "fetched_at": _CACHE.get("fetched_at", 0.0),
                "note": "",
            }

    req = urllib.request.Request(
        GITHUB_API_RELEASES,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "releases": [], "source": "fallback",
            "fetched_at": 0.0,
            "note": f"github fetch failed: {exc}",
        }

    if not isinstance(payload, list):
        return {
            "releases": [], "source": "fallback",
            "fetched_at": 0.0,
            "note": "unexpected response shape",
        }

    trimmed: List[Dict[str, Any]] = []
    for r in payload[:limit]:
        if not isinstance(r, dict):
            continue
        trimmed.append({
            "tag": str(r.get("tag_name") or "").strip(),
            "name": str(r.get("name") or "").strip(),
            "published_at": str(r.get("published_at") or ""),
            "url": str(r.get("html_url") or ""),
            "draft": bool(r.get("draft")),
            "prerelease": bool(r.get("prerelease")),
            "body": str(r.get("body") or "")[:4000],
        })
    fetched_at = time.time()
    with _CACHE_LOCK:
        _CACHE["releases"] = trimmed
        _CACHE["expires"] = time.monotonic() + _CACHE_TTL
        _CACHE["fetched_at"] = fetched_at
    return {
        "releases": trimmed,
        "source": "github",
        "fetched_at": fetched_at,
        "note": "",
    }


def latest_unseen(last_seen_tag: str = "", limit: int = 3) -> Dict[str, Any]:
    """Return releases newer than ``last_seen_tag`` for the toast banner."""
    feed = fetch_releases(limit=max(5, limit + 2))
    releases = feed.get("releases") or []
    if not last_seen_tag:
        state = read_user_file(SEEN_FILE, default={}) or {}
        last_seen_tag = str(state.get("last_seen") or "")

    if not last_seen_tag:
        return {"unseen": releases[:limit], **feed}

    unseen: List[Dict[str, Any]] = []
    for r in releases:
        if r.get("tag") == last_seen_tag:
            break
        unseen.append(r)
    return {"unseen": unseen[:limit], **feed}


def mark_seen(tag: str) -> Dict[str, Any]:
    """Persist the last-seen release tag."""
    tag = str(tag or "").strip()
    if not tag:
        raise ValueError("tag must be non-empty")
    state = read_user_file(SEEN_FILE, default={}) or {}
    state["last_seen"] = tag
    state["last_seen_at"] = time.time()
    write_user_file(SEEN_FILE, state)
    return {"tag": tag, "saved": True}


__all__ = [
    "check_changelog_available",
    "fetch_releases",
    "latest_unseen",
    "mark_seen",
]
