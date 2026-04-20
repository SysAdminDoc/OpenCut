"""
OpenCut Issue Report Bundle (Wave H1.5, v1.25.0)

Bundles recent logs + crash excerpt into a pre-filled GitHub issue URL.
The panel shows the URL; the user reviews and submits manually — we
never post on their behalf.

Security: every path under ``$HOME`` (or ``%USERPROFILE%`` on Windows)
is scrubbed to ``~`` before the body is returned. This is defence in
depth against private directory structures leaking into a public
bug tracker. Callers MUST still prompt the user before opening the URL.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import urllib.parse
from typing import Any, Dict, List

logger = logging.getLogger("opencut")

GITHUB_REPO = "SysAdminDoc/OpenCut"
ISSUE_URL = f"https://github.com/{GITHUB_REPO}/issues/new"


def check_issue_report_available() -> bool:
    """Always True — stdlib only."""
    return True


_HOME = os.path.expanduser("~")
_HOME_PATTERNS = [
    re.compile(re.escape(_HOME), re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\Users\\[^\\]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:/Users/[^/]+", re.IGNORECASE),
]


def _scrub_paths(text: str) -> str:
    """Replace absolute HOME paths with ``~``.  Does not touch relative paths."""
    if not text:
        return text
    for pat in _HOME_PATTERNS:
        text = pat.sub("~", text)
    return text


def _tail_file(path: str, max_bytes: int = 20_000) -> str:
    """Return the last ``max_bytes`` of ``path`` with best-effort decoding."""
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError as exc:
        return f"[could not read {os.path.basename(path)}: {exc}]"
    return data.decode("utf-8", errors="replace")


def _tail_lines(path: str, max_lines: int = 200) -> str:
    """Return the last ``max_lines`` of ``path``."""
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"[could not read {os.path.basename(path)}: {exc}]"
    return "".join(lines[-max_lines:])


def bundle(
    title: str = "",
    description: str = "",
    log_tail_lines: int = 200,
    include_crash: bool = True,
    include_logs: bool = True,
) -> Dict[str, Any]:
    """Assemble a report bundle. Returns ``{title, body, url, size_bytes}``.

    Body is scrubbed of HOME paths and capped at ~60 KB so the GitHub
    URL doesn't exceed the server's URL length limit when passed as
    query parameters.
    """
    log_tail_lines = max(10, min(int(log_tail_lines or 200), 2000))
    data_dir = os.path.join(_HOME, ".opencut")
    log_path = os.path.join(data_dir, "opencut.log")
    crash_path = os.path.join(data_dir, "crash.log")

    try:
        from opencut import __version__ as ocv
    except Exception:  # noqa: BLE001
        ocv = "?"

    parts: List[str] = []
    if description:
        parts.append("## Description\n\n" + _scrub_paths(str(description)) + "\n")
    parts.append(
        "## Environment\n\n"
        f"- OpenCut: {ocv}\n"
        f"- Platform: {platform.system()} {platform.release()}\n"
        f"- Python: {platform.python_version()}\n"
    )

    if include_crash:
        tail = _tail_file(crash_path, max_bytes=20_000)
        if tail:
            parts.append("## Crash log (tail)\n\n```\n"
                         + _scrub_paths(tail) + "\n```\n")

    if include_logs:
        tail = _tail_lines(log_path, max_lines=log_tail_lines)
        if tail:
            parts.append("## Recent logs\n\n```\n"
                         + _scrub_paths(tail) + "\n```\n")

    body = "\n".join(parts)
    if len(body) > 60_000:
        body = body[:60_000] + "\n\n_[truncated]_\n"

    effective_title = (title or f"OpenCut v{ocv} issue report").strip()
    q = urllib.parse.urlencode({
        "title": effective_title,
        "body": body,
        "labels": "bug,from-panel",
    })
    url = f"{ISSUE_URL}?{q}"
    return {
        "title": effective_title,
        "body": body,
        "url": url,
        "size_bytes": len(body.encode("utf-8")),
    }


__all__ = [
    "check_issue_report_available",
    "bundle",
]
