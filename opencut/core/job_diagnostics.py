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

The diagnostic payload path never performs network IO. Resource sampling
uses the core ``psutil`` dependency and optional ``pynvml`` integration.
"""

from __future__ import annotations

import json
import logging
import os
import threading
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
RESOURCE_METADATA_FIELDS = ("peak_vram_mb", "peak_cpu_pct", "peak_rss_mb")
EXIT_REASONS = frozenset({
    "complete",
    "error",
    "cancelled",
    "interrupted",
    "oom",
    "timeout",
    "preflight_failed",
})
_DEFAULT_RESOURCE_SAMPLE_INTERVAL = 5.0


def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:  # noqa: BLE001 - optional diagnostics dependency
        return None


def _coerce_nonnegative_int(value):
    if value is None:
        return None
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError, OverflowError):
        return None


def _bytes_to_mb(value) -> Optional[int]:
    try:
        return max(0, int(round(float(value) / (1024 * 1024))))
    except (TypeError, ValueError, OverflowError):
        return None


def _resource_sample_interval() -> float:
    raw = os.environ.get("OPENCUT_JOB_RESOURCE_SAMPLE_INTERVAL", "")
    if not raw:
        return _DEFAULT_RESOURCE_SAMPLE_INTERVAL
    try:
        return max(0.1, min(60.0, float(raw)))
    except (TypeError, ValueError):
        return _DEFAULT_RESOURCE_SAMPLE_INTERVAL


def _normalised_error_tokens(*parts) -> set[str]:
    text = " ".join(str(part or "") for part in parts).lower()
    for ch in ("_", "-", ".", ":", ";", ",", "(", ")", "[", "]", "{", "}"):
        text = text.replace(ch, " ")
    return {token for token in text.split() if token}


def classify_exit_reason(status: str, *, error: str = "", code: str = "", exc=None) -> str:
    """Map a terminal job state to the persisted exit-reason enum."""
    status_key = str(status or "").strip().lower()
    if status_key in {"complete", "cancelled", "interrupted"}:
        return status_key
    if status_key != "error":
        return ""

    tokens = _normalised_error_tokens(error, code, exc)
    raw_text = str(error or exc or "").lower()
    if isinstance(exc, TimeoutError) or "timeout" in tokens or "timed" in tokens:
        return "timeout"
    if (
        isinstance(exc, MemoryError)
        or "oom" in tokens
        or "out of memory" in raw_text
        or "cuda out of memory" in raw_text
    ):
        return "oom"
    if (
        "insufficient" in tokens
        and ("storage" in tokens or "disk" in tokens)
    ) or "preflight" in tokens:
        return "preflight_failed"
    return "error"


def _process_tree(root_process) -> list:
    processes = [root_process]
    try:
        processes.extend(root_process.children(recursive=True))
    except Exception:  # noqa: BLE001 - process may exit while sampling
        pass
    return processes


def _sample_vram_for_pids(pids: set[int], pynvml_module=None) -> Optional[int]:
    if not pids:
        return None
    nvml = pynvml_module if pynvml_module is not None else _try_import("pynvml")
    if nvml is None:
        return None

    initialised = False
    try:
        nvml.nvmlInit()
        initialised = True
        total_bytes = 0
        device_count = int(nvml.nvmlDeviceGetCount())
        for idx in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(idx)
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                getter = getattr(nvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    running = getter(handle) or []
                except Exception:  # noqa: BLE001 - unsupported on some drivers
                    continue
                for proc in running:
                    try:
                        proc_pid = int(getattr(proc, "pid"))
                        used = int(getattr(proc, "usedGpuMemory") or 0)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if proc_pid in pids and used > 0:
                        total_bytes += used
        return _bytes_to_mb(total_bytes) if total_bytes else 0
    except Exception:  # noqa: BLE001 - optional GPU diagnostics
        return None
    finally:
        if initialised:
            try:
                nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


def sample_process_resources(
    pid: Optional[int] = None,
    *,
    psutil_module=None,
    pynvml_module=None,
) -> dict:
    """Return a point-in-time resource snapshot for a process tree."""
    psutil = psutil_module if psutil_module is not None else _try_import("psutil")
    if psutil is None:
        return {field: None for field in RESOURCE_METADATA_FIELDS}

    target_pid = pid or os.getpid()
    try:
        root = psutil.Process(target_pid)
    except Exception:  # noqa: BLE001 - process may have exited
        return {field: None for field in RESOURCE_METADATA_FIELDS}

    total_cpu = 0.0
    total_rss = 0
    saw_cpu = False
    saw_rss = False
    pids: set[int] = set()

    for proc in _process_tree(root):
        try:
            pids.add(int(proc.pid))
        except Exception:  # noqa: BLE001
            pass
        try:
            total_cpu += float(proc.cpu_percent(interval=None) or 0.0)
            saw_cpu = True
        except Exception:  # noqa: BLE001
            pass
        try:
            total_rss += int(getattr(proc.memory_info(), "rss", 0) or 0)
            saw_rss = True
        except Exception:  # noqa: BLE001
            pass

    return {
        "peak_cpu_pct": _coerce_nonnegative_int(total_cpu) if saw_cpu else None,
        "peak_rss_mb": _bytes_to_mb(total_rss) if saw_rss else None,
        "peak_vram_mb": _sample_vram_for_pids(pids, pynvml_module=pynvml_module),
    }


class JobResourceSampler:
    """Daemon sampler that keeps peak process-tree resource usage."""

    def __init__(self, pid: Optional[int] = None, *,
                 interval_seconds: Optional[float] = None,
                 sampler=None):
        self.pid = pid or os.getpid()
        self.interval_seconds = (
            _resource_sample_interval()
            if interval_seconds is None
            else max(0.1, min(60.0, float(interval_seconds)))
        )
        self._sampler = sampler or sample_process_resources
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._peaks = {field: None for field in RESOURCE_METADATA_FIELDS}

    def start(self) -> "JobResourceSampler":
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"opencut-job-resource-{self.pid}",
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample_once()

    def sample_once(self) -> dict:
        try:
            snapshot = self._sampler(self.pid)
        except TypeError:
            snapshot = self._sampler()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not fail jobs
            logger.debug("job resource sample failed: %s", exc)
            return self.snapshot()
        if not isinstance(snapshot, dict):
            return self.snapshot()

        with self._lock:
            for field in RESOURCE_METADATA_FIELDS:
                value = _coerce_nonnegative_int(snapshot.get(field))
                if value is None:
                    continue
                current = self._peaks.get(field)
                self._peaks[field] = value if current is None else max(current, value)
            return dict(self._peaks)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._peaks)

    def stop(self) -> dict:
        self._stop.set()
        self.sample_once()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        return self.snapshot()


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
        from opencut.jobs import job_lock, jobs

        with job_lock:
            live = jobs.get(job_id) or {}
            if isinstance(live, dict):
                live = {k: v for k, v in live.items() if not k.startswith("_")}
            else:
                live = {}
    except Exception:
        live = {}
    return {**persisted, **live}


def _extract_metadata(record: dict) -> dict:
    """Pull non-sensitive metadata for the diagnostic payload."""
    out: dict = {}
    metadata_keys = (
        "filepath",
        "output_path",
        "queue_position",
        "duration_seconds",
        "tool",
        "params",
        "peak_vram_mb",
        "peak_cpu_pct",
        "peak_rss_mb",
        "exit_reason",
    )
    for key in metadata_keys:
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
    diag.created_at = float(record.get("created_at") or record.get("created") or 0.0)
    finished = record.get("finished_at")
    if finished is None:
        finished = record.get("completed_at")
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
