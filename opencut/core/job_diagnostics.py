"""Job request/observability diagnostics (F010 / v4.2).

Surface a single job's full timeline (status, request_id, error path,
log slice) so support tickets stop being "it didn't work" and start
being "here is the request_id, here are the 200 log lines around it,
here is the related job row from the in-process store."

The data sources are:

* :mod:`opencut.job_store` (SQLite-backed history) for the persisted
  metadata.
* :mod:`opencut.jobs` (in-memory dict) for the live status of any job
  that hasn't yet been flushed to the store.
* The opencut log file under ``~/.opencut/opencut.log`` for the actual
  log slice — we tail the last N MB and filter for lines containing
  either the ``job_id`` or, when known, the ``request_id``.

The module is stdlib-only and never performs network IO.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from opencut.core.issue_report import _scrub_paths

logger = logging.getLogger("opencut")


@dataclass
class LogSlice:
    """Tail-filtered slice of opencut.log relevant to a job/request."""

    source: str
    line_count: int
    body: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobDiagnostic:
    """Per-job diagnostic payload returned by the route."""

    job_id: str
    found: bool = False
    request_id: str = ""
    status: str = ""
    job_type: str = ""
    progress: int = 0
    created_at: float = 0.0
    finished_at: Optional[float] = None
    error: str = ""
    metadata: dict = field(default_factory=dict)
    log_slice: Optional[LogSlice] = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        if self.log_slice is not None:
            payload["log_slice"] = self.log_slice.as_dict()
        return payload


_LOG_PATH = Path(os.path.expanduser("~")) / ".opencut" / "opencut.log"
_MAX_TAIL_BYTES = 4 * 1024 * 1024  # 4 MB cap on the log slice scan


def _tail_log(max_bytes: int = _MAX_TAIL_BYTES) -> List[str]:
    if not _LOG_PATH.exists():
        return []
    try:
        size = _LOG_PATH.stat().st_size
        with _LOG_PATH.open("rb") as fh:
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError as exc:
        logger.debug("job_diagnostics: cannot read log: %s", exc)
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def _filter_lines(lines: Iterable[str], tokens: Iterable[str]) -> List[str]:
    needles = [t for t in tokens if t]
    if not needles:
        return []
    out: List[str] = []
    for line in lines:
        if any(token in line for token in needles):
            out.append(line)
    return out


def _resolve_job(job_id: str) -> dict:
    """Look up ``job_id`` in store + live registry. Returns the merged dict."""
    persisted: dict = {}
    live: dict = {}
    try:
        from opencut.job_store import get_job

        persisted = get_job(job_id) or {}
    except Exception:
        persisted = {}
    try:
        from opencut.jobs import _jobs

        live = (_jobs.get(job_id) or {})
        if isinstance(live, dict):
            live = dict(live)
        else:
            live = {}
    except Exception:
        live = {}
    return {**persisted, **live}


def _extract_metadata(record: dict) -> dict:
    """Pull non-sensitive metadata for the diagnostic payload."""
    out: dict = {}
    for key in ("filepath", "output_path", "queue_position", "duration_seconds", "tool", "params"):
        if key in record and record[key] is not None:
            value = record[key]
            if isinstance(value, str):
                value = _scrub_paths(value)
            elif isinstance(value, (list, dict)):
                try:
                    value = json.loads(_scrub_paths(json.dumps(value, default=str)))
                except Exception:
                    pass
            out[key] = value
    return out


def build_diagnostic(job_id: str, *, log_tail_lines: int = 200) -> JobDiagnostic:
    """Assemble the diagnostic payload for ``job_id``."""
    diag = JobDiagnostic(job_id=job_id)
    record = _resolve_job(job_id)
    if not record:
        return diag

    diag.found = True
    diag.status = str(record.get("status") or "")
    diag.job_type = str(record.get("type") or record.get("job_type") or "")
    diag.progress = int(record.get("progress") or 0)
    diag.request_id = str(record.get("request_id") or record.get("client_request_id") or "")
    diag.error = _scrub_paths(str(record.get("error") or record.get("error_message") or ""))
    diag.created_at = float(record.get("created_at") or 0.0)
    finished = record.get("finished_at")
    diag.finished_at = float(finished) if finished is not None else None
    diag.metadata = _extract_metadata(record)

    lines = _tail_log()
    if lines:
        matches = _filter_lines(lines, (job_id, diag.request_id))
        if matches:
            cap = max(10, int(log_tail_lines))
            body = "\n".join(matches[-cap:])
            diag.log_slice = LogSlice(
                source=str(_LOG_PATH),
                line_count=len(matches),
                body=_scrub_paths(body),
            )
    return diag


def export_diagnostic(job_id: str, output_path: str | os.PathLike, *, log_tail_lines: int = 200) -> dict:
    """Write the diagnostic payload as a JSON file."""
    diag = build_diagnostic(job_id, log_tail_lines=log_tail_lines)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(diag.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output_path": str(target), "found": diag.found, "job_id": job_id}
