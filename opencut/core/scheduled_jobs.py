"""
OpenCut Scheduled / Recurring Job System

Cron-like scheduling for recurring pipeline operations: batch transcodes,
workflow executions, watch-folder scans, backups, and QC checks.
Schedules are stored in ``~/.opencut/schedules.json`` and job history
in ``~/.opencut/schedule_history.json``.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_SCHEDULES_FILE = os.path.join(_OPENCUT_DIR, "schedules.json")
_HISTORY_FILE = os.path.join(_OPENCUT_DIR, "schedule_history.json")
_FILE_LOCK = threading.Lock()

# Supported job types
JOB_TYPES = {
    "workflow": "Execute a saved workflow",
    "batch_transcode": "Transcode files in a folder",
    "watch_folder": "Scan a watch folder for new media",
    "backup": "Backup project files",
    "qc_check": "Run quality control checks",
    "cleanup": "Clean temporary / old files",
    "export": "Scheduled export of a project",
    "report": "Generate pipeline health report",
}

# Cron field ranges
_CRON_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),  # 0=Monday .. 6=Sunday
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScheduleConfig:
    """Configuration for what a scheduled job does."""
    job_type: str = "workflow"
    target_path: str = ""           # file or folder path
    workflow_name: str = ""         # workflow to execute (if job_type=workflow)
    params: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    notify_on_complete: bool = False
    notify_on_failure: bool = True
    max_runtime_s: int = 3600       # kill if exceeds this

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ScheduledJob:
    """A single scheduled (recurring) job definition."""
    schedule_id: str = ""
    name: str = ""
    cron_expr: str = "0 * * * *"    # default: every hour
    job_config: ScheduleConfig = field(default_factory=ScheduleConfig)
    enabled: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0
    last_run: float = 0.0
    next_run: float = 0.0
    run_count: int = 0
    fail_count: int = 0
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.schedule_id:
            self.schedule_id = uuid.uuid4().hex[:12]
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.updated_at == 0.0:
            self.updated_at = now
        if self.next_run == 0.0 and self.enabled:
            self.next_run = get_next_run(self.cron_expr)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["job_config"] = self.job_config.to_dict() if isinstance(self.job_config, ScheduleConfig) else self.job_config
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledJob":
        raw_config = d.pop("job_config", {})
        if isinstance(raw_config, dict):
            config = ScheduleConfig.from_dict(raw_config)
        else:
            config = raw_config
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        fields["job_config"] = config
        return cls(**fields)


@dataclass
class JobHistory:
    """Record of a single scheduled job execution."""
    history_id: str = ""
    schedule_id: str = ""
    schedule_name: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_s: float = 0.0
    success: bool = True
    error_message: str = ""
    result_summary: str = ""
    job_type: str = ""

    def __post_init__(self):
        if not self.history_id:
            self.history_id = uuid.uuid4().hex[:12]
        if self.started_at == 0.0:
            self.started_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobHistory":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Cron expression parsing
# ---------------------------------------------------------------------------
def _parse_cron_field(field_str: str, field_name: str) -> List[int]:
    """Parse a single cron field into a sorted list of valid integers.

    Supports: ``*``, ``*/N``, ``N``, ``N-M``, ``N,M,O``, ``N-M/S``.
    """
    lo, hi = _CRON_RANGES[field_name]
    results = set()

    for part in field_str.split(","):
        part = part.strip()
        if part == "*":
            results.update(range(lo, hi + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            if step < 1:
                step = 1
            results.update(range(lo, hi + 1, step))
        elif "/" in part:
            range_part, step_str = part.split("/", 1)
            step = int(step_str) if step_str else 1
            if step < 1:
                step = 1
            if "-" in range_part:
                a, b = range_part.split("-", 1)
                a, b = int(a), int(b)
            else:
                a, b = int(range_part), hi
            a = max(a, lo)
            b = min(b, hi)
            results.update(range(a, b + 1, step))
        elif "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            a = max(a, lo)
            b = min(b, hi)
            results.update(range(a, b + 1))
        else:
            val = int(part)
            if lo <= val <= hi:
                results.add(val)

    return sorted(results)


def parse_cron_expr(cron_expr: str) -> Dict[str, List[int]]:
    """Parse a 5-field cron expression into expanded value lists.

    Format: ``minute hour day_of_month month day_of_week``

    Returns:
        Dict with keys ``minute``, ``hour``, ``day_of_month``, ``month``,
        ``day_of_week``, each mapped to a sorted list of valid values.

    Raises:
        ValueError: If the expression does not have exactly 5 fields.
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got {len(parts)}: {cron_expr!r}")

    field_names = ["minute", "hour", "day_of_month", "month", "day_of_week"]
    return {name: _parse_cron_field(parts[i], name) for i, name in enumerate(field_names)}


def get_next_run(cron_expr: str, after: Optional[float] = None) -> float:
    """Calculate the next run timestamp from a cron expression.

    Args:
        cron_expr: Standard 5-field cron expression.
        after:     Reference timestamp (default: now).

    Returns:
        Unix timestamp of next matching minute.
    """
    parsed = parse_cron_expr(cron_expr)
    ref = datetime.fromtimestamp(after or time.time(), tz=timezone.utc)
    # Start from the next minute
    candidate = ref.replace(second=0, microsecond=0)
    # Advance one minute from the reference point
    candidate = datetime.fromtimestamp(candidate.timestamp() + 60, tz=timezone.utc)

    # Iterate up to 366 days of minutes (527040) to find a match
    for _ in range(527040):
        if (candidate.minute in parsed["minute"]
                and candidate.hour in parsed["hour"]
                and candidate.day in parsed["day_of_month"]
                and candidate.month in parsed["month"]
                and candidate.weekday() in parsed["day_of_week"]):
            return candidate.timestamp()
        candidate = datetime.fromtimestamp(candidate.timestamp() + 60, tz=timezone.utc)

    # Fallback: 1 hour from now
    logger.warning("Could not find next run for cron expr %r, defaulting to +1h", cron_expr)
    return time.time() + 3600


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _load_schedules() -> List[dict]:
    """Load schedules from JSON file."""
    if not os.path.isfile(_SCHEDULES_FILE):
        return []
    try:
        with open(_SCHEDULES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load schedules: %s", exc)
        return []


def _save_schedules(schedules: List[dict]) -> None:
    """Save schedules to JSON file."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    tmp_path = _SCHEDULES_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(schedules, f, indent=2)
    # Atomic rename (Windows: replace if exists)
    if os.path.exists(_SCHEDULES_FILE):
        os.replace(tmp_path, _SCHEDULES_FILE)
    else:
        os.rename(tmp_path, _SCHEDULES_FILE)


def _load_history() -> List[dict]:
    """Load job history from JSON file."""
    if not os.path.isfile(_HISTORY_FILE):
        return []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load schedule history: %s", exc)
        return []


def _save_history(history: List[dict]) -> None:
    """Save job history to JSON file, keeping last 1000 entries."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)
    history = history[-1000:]  # cap at 1000 entries
    tmp_path = _HISTORY_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if os.path.exists(_HISTORY_FILE):
        os.replace(tmp_path, _HISTORY_FILE)
    else:
        os.rename(tmp_path, _HISTORY_FILE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_schedule(
    name: str,
    cron_expr: str,
    job_config: Optional[Dict[str, Any]] = None,
    enabled: bool = True,
    tags: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> ScheduledJob:
    """Create a new scheduled job.

    Args:
        name:       Human-readable name.
        cron_expr:  5-field cron expression.
        job_config: Dict of ScheduleConfig fields.
        enabled:    Whether the schedule is active.
        tags:       Optional tags for filtering.
        on_progress: Optional callback.

    Returns:
        The created ``ScheduledJob``.
    """
    # Validate cron expression
    parse_cron_expr(cron_expr)

    config = ScheduleConfig.from_dict(job_config or {})
    if config.job_type not in JOB_TYPES:
        logger.warning("Unknown job_type %r, allowing anyway", config.job_type)

    job = ScheduledJob(
        name=name,
        cron_expr=cron_expr,
        job_config=config,
        enabled=enabled,
        tags=tags or [],
    )

    if on_progress:
        on_progress(50, "Saving schedule")

    with _FILE_LOCK:
        schedules = _load_schedules()
        schedules.append(job.to_dict())
        _save_schedules(schedules)

    if on_progress:
        on_progress(100, "Schedule created")

    logger.info("Created schedule %r (id=%s, cron=%s)", name, job.schedule_id, cron_expr)
    return job


def list_schedules(
    enabled_only: bool = False,
    job_type: Optional[str] = None,
    tag: Optional[str] = None,
) -> List[ScheduledJob]:
    """List all scheduled jobs, optionally filtered.

    Args:
        enabled_only: If True, only return enabled schedules.
        job_type:     Filter by job type.
        tag:          Filter by tag.

    Returns:
        List of ``ScheduledJob`` objects.
    """
    with _FILE_LOCK:
        raw = _load_schedules()

    jobs = []
    for entry in raw:
        try:
            sj = ScheduledJob.from_dict(dict(entry))
        except (TypeError, KeyError) as exc:
            logger.warning("Skipping malformed schedule entry: %s", exc)
            continue

        if enabled_only and not sj.enabled:
            continue
        if job_type and sj.job_config.job_type != job_type:
            continue
        if tag and tag not in sj.tags:
            continue
        jobs.append(sj)

    return jobs


def get_schedule(schedule_id: str) -> Optional[ScheduledJob]:
    """Get a single schedule by ID."""
    for sj in list_schedules():
        if sj.schedule_id == schedule_id:
            return sj
    return None


def update_schedule(
    schedule_id: str,
    name: Optional[str] = None,
    cron_expr: Optional[str] = None,
    enabled: Optional[bool] = None,
    job_config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Optional[ScheduledJob]:
    """Update an existing schedule. Returns the updated job or None if not found."""
    with _FILE_LOCK:
        schedules = _load_schedules()
        found_idx = None
        for i, entry in enumerate(schedules):
            if entry.get("schedule_id") == schedule_id:
                found_idx = i
                break

        if found_idx is None:
            return None

        entry = schedules[found_idx]
        if name is not None:
            entry["name"] = name
        if cron_expr is not None:
            parse_cron_expr(cron_expr)  # validate
            entry["cron_expr"] = cron_expr
            entry["next_run"] = get_next_run(cron_expr)
        if enabled is not None:
            entry["enabled"] = enabled
        if job_config is not None:
            entry["job_config"] = ScheduleConfig.from_dict(job_config).to_dict()
        if tags is not None:
            entry["tags"] = tags
        entry["updated_at"] = time.time()

        schedules[found_idx] = entry
        _save_schedules(schedules)

    return ScheduledJob.from_dict(dict(entry))


def delete_schedule(schedule_id: str) -> bool:
    """Delete a schedule by ID. Returns True if found and deleted."""
    with _FILE_LOCK:
        schedules = _load_schedules()
        original_len = len(schedules)
        schedules = [s for s in schedules if s.get("schedule_id") != schedule_id]
        if len(schedules) == original_len:
            return False
        _save_schedules(schedules)
    logger.info("Deleted schedule %s", schedule_id)
    return True


def check_due_jobs(now: Optional[float] = None) -> List[ScheduledJob]:
    """Find schedules whose next_run has passed and need execution.

    After retrieval the caller is expected to:
      1. Execute each job.
      2. Call ``record_job_run`` and ``advance_schedule`` for each.

    Args:
        now: Reference timestamp (default: current time).

    Returns:
        List of ``ScheduledJob`` that are due.
    """
    current = now or time.time()
    due = []
    for sj in list_schedules(enabled_only=True):
        if sj.next_run > 0 and sj.next_run <= current:
            due.append(sj)
    return due


def check_missed_jobs(tolerance_minutes: int = 10) -> List[ScheduledJob]:
    """Detect schedules that missed their run window.

    A job is considered missed if ``next_run`` is more than
    *tolerance_minutes* in the past.

    Returns:
        List of ``ScheduledJob`` that missed their window.
    """
    current = time.time()
    cutoff = current - tolerance_minutes * 60
    missed = []
    for sj in list_schedules(enabled_only=True):
        if 0 < sj.next_run < cutoff:
            missed.append(sj)
    return missed


def advance_schedule(schedule_id: str) -> Optional[float]:
    """Advance a schedule's next_run to its next occurrence.

    Returns the new next_run timestamp, or None if schedule not found.
    """
    with _FILE_LOCK:
        schedules = _load_schedules()
        for entry in schedules:
            if entry.get("schedule_id") == schedule_id:
                entry["last_run"] = time.time()
                entry["next_run"] = get_next_run(entry.get("cron_expr", "0 * * * *"))
                entry["run_count"] = entry.get("run_count", 0) + 1
                _save_schedules(schedules)
                return entry["next_run"]
    return None


def record_job_run(
    schedule_id: str,
    schedule_name: str,
    success: bool,
    duration_s: float = 0.0,
    error_message: str = "",
    result_summary: str = "",
    job_type: str = "",
) -> JobHistory:
    """Record a job execution in history.

    Also updates fail_count on the schedule if the job failed.

    Returns:
        The ``JobHistory`` entry.
    """
    entry = JobHistory(
        schedule_id=schedule_id,
        schedule_name=schedule_name,
        duration_s=duration_s,
        success=success,
        error_message=error_message,
        result_summary=result_summary,
        job_type=job_type,
    )
    entry.finished_at = entry.started_at + duration_s

    with _FILE_LOCK:
        history = _load_history()
        history.append(entry.to_dict())
        _save_history(history)

        # Update fail count on the schedule
        if not success:
            schedules = _load_schedules()
            for s in schedules:
                if s.get("schedule_id") == schedule_id:
                    s["fail_count"] = s.get("fail_count", 0) + 1
                    break
            _save_schedules(schedules)

    return entry


def get_job_history(
    schedule_id: Optional[str] = None,
    limit: int = 50,
) -> List[JobHistory]:
    """Retrieve recent job execution history.

    Args:
        schedule_id: Filter to a specific schedule (optional).
        limit:       Max entries to return.

    Returns:
        List of ``JobHistory`` entries, newest first.
    """
    with _FILE_LOCK:
        raw = _load_history()

    entries = []
    for item in reversed(raw):
        if schedule_id and item.get("schedule_id") != schedule_id:
            continue
        try:
            entries.append(JobHistory.from_dict(dict(item)))
        except (TypeError, KeyError):
            continue
        if len(entries) >= limit:
            break
    return entries


def clear_history(schedule_id: Optional[str] = None) -> int:
    """Clear job history. If schedule_id given, only clear that schedule's history.

    Returns count of deleted entries.
    """
    with _FILE_LOCK:
        raw = _load_history()
        if schedule_id:
            filtered = [h for h in raw if h.get("schedule_id") != schedule_id]
            deleted = len(raw) - len(filtered)
            _save_history(filtered)
        else:
            deleted = len(raw)
            _save_history([])
    return deleted
