"""Where the server writes its log, and how diagnostics read it back.

``opencut.server`` logs to ``~/.opencut/server.log`` through a
``RotatingFileHandler``, which renames a full file to ``server.log.1`` and
shifts older ones up to ``server.log.3``. Issue reports, crash packets, job
diagnostics and ``/logs/export`` all take the location from here, so the
bundles they build carry the log the server actually wrote.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger("opencut")

SERVER_LOG_NAME = "server.log"
SERVER_LOG_MAX_BYTES = 5 * 1024 * 1024
SERVER_LOG_BACKUP_COUNT = 3
DEFAULT_TAIL_BYTES = 4 * 1024 * 1024


def server_log_path(home: Optional[str] = None) -> str:
    """Path of the live server log under ``home`` (the user's home by default)."""
    base = os.path.expanduser("~") if home is None else home
    return os.path.join(base, ".opencut", SERVER_LOG_NAME)


def server_log_files(home: Optional[str] = None) -> List[str]:
    """The live log first, then its rotated siblings from newest to oldest."""
    path = server_log_path(home)
    return [path] + [f"{path}.{n}" for n in range(1, SERVER_LOG_BACKUP_COUNT + 1)]


def server_log_tail(
    max_lines: Optional[int] = None,
    max_bytes: int = DEFAULT_TAIL_BYTES,
    home: Optional[str] = None,
) -> List[str]:
    """Last lines of the server log, oldest first.

    When the live file holds fewer than ``max_lines`` lines (right after a
    rotation, say) the rotated siblings fill in the older lines. ``max_bytes``
    caps the total read across every file.
    """
    if max_lines is not None and max_lines <= 0:
        return []
    newest_first: List[List[str]] = []
    budget = max(0, int(max_bytes))
    collected = 0
    for path in server_log_files(home):
        if budget <= 0 or (max_lines is not None and collected >= max_lines):
            break
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                start = max(0, size - budget)
                fh.seek(start)
                data = fh.read()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.debug("server_log: cannot read %s: %s", path, exc)
            continue
        budget -= len(data)
        if start > 0:
            # The read began mid-line; drop the fragment.
            newline = data.find(b"\n")
            data = data[newline + 1:] if newline >= 0 else b""
        lines = data.decode("utf-8", errors="replace").splitlines()
        newest_first.append(lines)
        collected += len(lines)
    ordered = [line for chunk in reversed(newest_first) for line in chunk]
    if max_lines is not None:
        ordered = ordered[-max_lines:]
    return ordered


__all__ = [
    "DEFAULT_TAIL_BYTES",
    "SERVER_LOG_BACKUP_COUNT",
    "SERVER_LOG_MAX_BYTES",
    "SERVER_LOG_NAME",
    "server_log_files",
    "server_log_path",
    "server_log_tail",
]
