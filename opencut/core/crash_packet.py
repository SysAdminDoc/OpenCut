"""Crash + recovery diagnostic packet (F066).

The companion to :mod:`opencut.core.issue_report`. ``issue_report``
builds a *URL* you paste into a GitHub issue. ``crash_packet`` builds a
*zip* the user can attach to a support email or drop into a bug
report. The two share the same path-scrubbing rules so neither one
leaks home directories.

Bundle contents:

* ``manifest.json`` — version, generated_at, python+platform, scrubbed
  paths to the source artefacts, sizes.
* ``crash.log`` — tail of ``~/.opencut/crash.log`` (scrubbed).
* ``opencut.log`` — last 500 lines of ``~/.opencut/opencut.log``
  (scrubbed, cap configurable).
* ``environment.txt`` — Python + platform + bundled FFmpeg version +
  installed pip distributions (names only; no versions, to avoid
  fingerprinting private wheels).
* ``recent_jobs.json`` — last 50 entries from the job store (best
  effort; missing when the in-process job store isn't available).

The module is stdlib-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from opencut.core.issue_report import _scrub_paths, _tail_file, _tail_lines

logger = logging.getLogger("opencut")

PACKET_VERSION = 1
DEFAULT_LOG_LINES = 500
DEFAULT_CRASH_BYTES = 20_000


@dataclass
class PacketEntry:
    arcname: str
    bytes: int
    sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CrashPacketResult:
    output_path: str
    bundle_sha256: str
    total_bytes: int
    entries: List[PacketEntry] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "version": PACKET_VERSION,
            "output_path": self.output_path,
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "generated_at": self.generated_at,
            "entries": [e.as_dict() for e in self.entries],
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pip_freeze_names() -> List[str]:
    """Return installed distribution *names* only — no versions."""
    try:
        from importlib.metadata import distributions
    except Exception:
        return []
    try:
        names = sorted({d.metadata["Name"] for d in distributions() if d.metadata and d.metadata["Name"]})
    except Exception:
        names = []
    return names


def _environment_block() -> str:
    rows = [
        f"python_implementation={platform.python_implementation()}",
        f"python_version={platform.python_version()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"argv={' '.join(sys.argv)}",
    ]
    try:
        from opencut.helpers import get_ffmpeg_path

        rows.append(f"ffmpeg={get_ffmpeg_path() or 'not on PATH'}")
    except Exception:
        rows.append("ffmpeg=unknown")

    pip_names = _pip_freeze_names()
    if pip_names:
        rows.append(f"pip_distribution_count={len(pip_names)}")
        rows.append("pip_distributions=" + ",".join(pip_names[:80]))
        if len(pip_names) > 80:
            rows.append(f"pip_distributions_truncated={len(pip_names) - 80}")

    body = _scrub_paths("\n".join(rows) + "\n")
    return body


def _recent_jobs_block(limit: int = 50) -> Optional[str]:
    try:
        from opencut.job_store import iter_recent_jobs  # type: ignore[attr-defined]
    except Exception:
        try:
            from opencut.jobs import _jobs

            entries = list(_jobs.values())[-limit:]
            return _scrub_paths(json.dumps(entries, default=str, indent=2)) + "\n"
        except Exception:
            return None
    try:
        entries = list(iter_recent_jobs(limit))
        return _scrub_paths(json.dumps(entries, default=str, indent=2)) + "\n"
    except Exception:
        return None


def build_packet(
    *,
    output_path: str | os.PathLike,
    log_tail_lines: int = DEFAULT_LOG_LINES,
    crash_tail_bytes: int = DEFAULT_CRASH_BYTES,
    include_jobs: bool = True,
) -> CrashPacketResult:
    """Write the crash/recovery packet zip and return its manifest."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(os.path.expanduser("~")) / ".opencut"
    crash_log = data_dir / "crash.log"
    opencut_log = data_dir / "opencut.log"

    crash_text = _scrub_paths(_tail_file(str(crash_log), max_bytes=crash_tail_bytes)) if crash_log.exists() else ""
    log_text = _scrub_paths(_tail_lines(str(opencut_log), max_lines=log_tail_lines)) if opencut_log.exists() else ""
    environment_text = _environment_block()
    jobs_text = _recent_jobs_block() if include_jobs else None

    entries: List[PacketEntry] = []

    def _entry(arcname: str, payload: bytes) -> tuple:
        return arcname, payload, PacketEntry(arcname=arcname, bytes=len(payload), sha256=_sha256_bytes(payload))

    pre_entries = []
    pre_entries.append(_entry("environment.txt", environment_text.encode("utf-8")))
    if crash_text:
        pre_entries.append(_entry("crash.log", crash_text.encode("utf-8")))
    if log_text:
        pre_entries.append(_entry("opencut.log", log_text.encode("utf-8")))
    if jobs_text:
        pre_entries.append(_entry("recent_jobs.json", jobs_text.encode("utf-8")))

    manifest = {
        "version": PACKET_VERSION,
        "generated_at": time.time(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "entries": [entry.as_dict() for _arcname, _payload, entry in pre_entries],
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    pre_entries.append(_entry("manifest.json", manifest_bytes))

    pre_entries.sort(key=lambda item: item[0])

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload, entry in pre_entries:
            zi = zipfile.ZipInfo(filename=arcname, date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)
            entries.append(entry)

    raw = out.read_bytes()
    return CrashPacketResult(
        output_path=str(out),
        bundle_sha256=_sha256_bytes(raw),
        total_bytes=len(raw),
        entries=entries,
    )
