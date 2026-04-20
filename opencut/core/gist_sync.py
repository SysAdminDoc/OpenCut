"""
OpenCut Gist Preset Sync (Wave H1.7, v1.25.0)

Exports / imports workflow presets, LUT configs, and favorites through
GitHub Gists.  Pure-stdlib ``urllib.request`` — no PyGitHub dep.

Security
--------
- ``push`` defaults to ``public=False`` (secret gist). Passing
  ``public=True`` requires an explicit opt-in per request.
- Authenticated push requires ``GITHUB_TOKEN`` in the environment.
  Unauthenticated push hits GitHub's anonymous-gist endpoint, which is
  IP rate limited and always public. We refuse anonymous push when
  ``public=False`` — nothing stays secret on an anonymous gist.
- ``pull`` accepts any gist URL that resolves to a JSON-only payload.
  Files whose size exceeds ``MAX_PAYLOAD_BYTES`` are rejected up front.
- We NEVER shell out to ``git`` — the gist is transferred through the
  GitHub REST API using stdlib HTTPS only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

logger = logging.getLogger("opencut")


GISTS_URL = "https://api.github.com/gists"
GIST_ID_URL = "https://api.github.com/gists/{id}"
DEFAULT_USER_AGENT = "OpenCut-Panel/1.25.0"
MAX_PAYLOAD_BYTES = 2_000_000   # 2 MB per file — plenty for JSON presets
MAX_FILES = 16

_GIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,60}$")
_GIST_URL_RE = re.compile(
    r"^https?://gist\.github\.com/(?:[A-Za-z0-9_-]+/)?([A-Za-z0-9_-]{20,60})(?:[/?#].*)?$",
    re.IGNORECASE,
)


def check_gist_sync_available() -> bool:
    """Always True — stdlib urllib only. Fails gracefully on network errors."""
    return True


def _auth_header() -> Dict[str, str]:
    tok = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    return {}


def _base_headers() -> Dict[str, str]:
    return {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/vnd.github+json",
    }


# ---------------------------------------------------------------------------
# Gist ID / URL helpers
# ---------------------------------------------------------------------------

def _parse_gist_id(url_or_id: str) -> str:
    """Accept a gist ID or URL, return the 20-60 char ID. Raises on invalid."""
    s = str(url_or_id or "").strip()
    if not s:
        raise ValueError("gist URL or ID is required")

    m = _GIST_URL_RE.match(s)
    if m:
        return m.group(1)
    if _GIST_ID_RE.match(s):
        return s
    raise ValueError(f"not a valid gist ID or URL: {s[:80]}")


def _validate_files(files: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Normalise a ``{filename: json-serialisable}`` dict for the API."""
    if not isinstance(files, dict) or not files:
        raise ValueError("at least one {filename: data} entry is required")
    if len(files) > MAX_FILES:
        raise ValueError(f"too many files (>{MAX_FILES})")

    out: Dict[str, Dict[str, str]] = {}
    for name, payload in files.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("file names must be non-empty strings")
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"invalid gist filename: {name}")
        if not name.lower().endswith((".json", ".jsonl", ".txt", ".md")):
            raise ValueError(
                f"only .json/.jsonl/.txt/.md files are allowed (got {name})"
            )
        # Serialise whatever we were given.
        if isinstance(payload, (dict, list)):
            content = json.dumps(payload, indent=2, ensure_ascii=False)
        else:
            content = str(payload)
        if len(content.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"file {name} exceeds {MAX_PAYLOAD_BYTES} bytes")
        out[name] = {"content": content}
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def push(
    files: Dict[str, Any],
    description: str = "OpenCut preset export",
    public: bool = False,
    update_id: str = "",
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Create (or update) a gist with the given files.

    Args:
        files: ``{filename: dict|list|str}``. Keys must end with
            ``.json``/``.jsonl``/``.txt``/``.md``.
        description: Gist description (shown in GitHub UI).
        public: ``True`` to create a public gist, ``False`` for secret.
            Unauthenticated requests cannot create secret gists — we
            refuse and raise.
        update_id: If set, PATCH an existing gist instead of creating a
            new one. Requires ``GITHUB_TOKEN``.
        timeout: Network timeout in seconds.

    Returns ``{id, url, html_url, files, public, updated}``.
    """
    files_clean = _validate_files(files)
    auth = _auth_header()
    if not public and not auth:
        raise ValueError(
            "secret gists require GITHUB_TOKEN; anonymous gists are public"
        )
    if update_id and not auth:
        raise ValueError("updating an existing gist requires GITHUB_TOKEN")

    body = {
        "description": str(description or "OpenCut preset export")[:256],
        "public": bool(public),
        "files": files_clean,
    }

    if update_id:
        gist_id = _parse_gist_id(update_id)
        url = GIST_ID_URL.format(id=gist_id)
        method = "PATCH"
    else:
        url = GISTS_URL
        method = "POST"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={**_base_headers(), "Content-Type": "application/json", **auth},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"gist push failed: HTTP {exc.code} {exc.reason} {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f"gist push failed: {exc}") from exc

    return {
        "id": str(data.get("id") or ""),
        "url": str(data.get("url") or ""),
        "html_url": str(data.get("html_url") or ""),
        "files": list((data.get("files") or {}).keys()),
        "public": bool(data.get("public")),
        "updated": bool(update_id),
    }


def pull(url_or_id: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Pull a gist's JSON files into a ``{filename: parsed}`` dict.

    Non-JSON files are returned as raw strings. Files exceeding
    ``MAX_PAYLOAD_BYTES`` are rejected up front.
    """
    gist_id = _parse_gist_id(url_or_id)
    url = GIST_ID_URL.format(id=gist_id)
    req = urllib.request.Request(
        url, headers={**_base_headers(), **_auth_header()},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"gist pull failed: HTTP {exc.code} {exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f"gist pull failed: {exc}") from exc

    files = data.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise RuntimeError("gist has no files")

    out: Dict[str, Any] = {}
    for name, meta in files.items():
        if not isinstance(meta, dict):
            continue
        size = int(meta.get("size") or 0)
        if size > MAX_PAYLOAD_BYTES:
            out[name] = {"__skipped__": True, "reason": f"size {size} > {MAX_PAYLOAD_BYTES}"}
            continue
        content = meta.get("content")
        if content is None and meta.get("raw_url"):
            try:
                raw_req = urllib.request.Request(
                    meta["raw_url"],
                    headers={**_base_headers(), **_auth_header()},
                )
                with urllib.request.urlopen(raw_req, timeout=timeout) as rr:
                    raw = rr.read()
                if len(raw) > MAX_PAYLOAD_BYTES:
                    out[name] = {"__skipped__": True, "reason": "raw payload too large"}
                    continue
                content = raw.decode("utf-8", errors="replace")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                out[name] = {"__skipped__": True, "reason": f"raw fetch failed: {exc}"}
                continue
        if not isinstance(content, str):
            out[name] = {"__skipped__": True, "reason": "content missing"}
            continue

        # Parse JSON when the extension says so; fall back to raw string.
        if name.lower().endswith((".json", ".jsonl")):
            try:
                out[name] = json.loads(content)
            except json.JSONDecodeError:
                out[name] = {"__raw__": content}
        else:
            out[name] = content

    return {
        "id": str(data.get("id") or ""),
        "description": str(data.get("description") or ""),
        "public": bool(data.get("public")),
        "html_url": str(data.get("html_url") or ""),
        "files": out,
    }


def info(url_or_id: str, timeout: float = 6.0) -> Dict[str, Any]:
    """Return lightweight metadata for a gist without pulling files."""
    gist_id = _parse_gist_id(url_or_id)
    url = GIST_ID_URL.format(id=gist_id)
    req = urllib.request.Request(
        url, headers={**_base_headers(), **_auth_header()},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"gist info failed: HTTP {exc.code} {exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f"gist info failed: {exc}") from exc

    files = data.get("files") or {}
    file_list: List[Dict[str, Any]] = []
    if isinstance(files, dict):
        for name, meta in files.items():
            if not isinstance(meta, dict):
                continue
            file_list.append({
                "name": name,
                "size": int(meta.get("size") or 0),
                "type": str(meta.get("type") or ""),
            })
    return {
        "id": str(data.get("id") or ""),
        "description": str(data.get("description") or ""),
        "public": bool(data.get("public")),
        "html_url": str(data.get("html_url") or ""),
        "updated_at": str(data.get("updated_at") or ""),
        "files": file_list,
    }


__all__ = [
    "check_gist_sync_available",
    "push",
    "pull",
    "info",
]
