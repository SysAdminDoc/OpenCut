"""
OpenCut Pipeline Health Monitoring

Tracks success/failure rates, processing times, error frequency,
resource utilization trends, and per-component health scores.
Stores metrics in a lightweight SQLite database at ``~/.opencut/pipeline_health.db``.
"""

import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "pipeline_health.db")
_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False

# ---------------------------------------------------------------------------
# Alert threshold defaults
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "error_rate_warn": 0.10,        # 10 % error rate triggers warning
    "error_rate_critical": 0.25,    # 25 % triggers critical
    "avg_duration_factor": 3.0,     # 3x baseline -> degraded
    "min_health_score": 50,         # below 50 -> alert
    "throughput_drop_pct": 0.40,    # 40 % drop from peak -> alert
}

# Baseline expected durations (seconds) per operation type
OPERATION_BASELINES: Dict[str, float] = {
    "transcode": 60.0,
    "trim": 5.0,
    "concat": 15.0,
    "stabilize": 120.0,
    "denoise": 90.0,
    "caption": 30.0,
    "thumbnail": 2.0,
    "silence_detect": 10.0,
    "scene_detect": 20.0,
    "color_grade": 45.0,
    "audio_mix": 25.0,
    "export": 80.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class HealthMetric:
    """Single recorded metric event."""
    operation: str
    duration_s: float
    success: bool
    error_type: str = ""
    error_message: str = ""
    cpu_pct: float = 0.0
    gpu_pct: float = 0.0
    ram_mb: float = 0.0
    disk_write_mb: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComponentHealth:
    """Health status of a single pipeline component."""
    component: str
    health_score: int = 100          # 0-100
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    avg_duration_s: float = 0.0
    error_rate: float = 0.0
    most_common_error: str = ""
    trend: str = "stable"            # improving | stable | degrading
    alert_level: str = "ok"          # ok | warning | critical

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorSummary:
    """Summary of a particular error type."""
    error_type: str
    count: int = 0
    last_seen: float = 0.0
    affected_operations: List[str] = field(default_factory=list)
    sample_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThroughputPoint:
    """Throughput data point for time-series."""
    hour_start: float
    jobs_completed: int = 0
    jobs_failed: int = 0
    avg_duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineHealthResult:
    """Aggregate pipeline health report."""
    overall_score: int = 100
    overall_status: str = "healthy"       # healthy | degraded | critical
    total_jobs: int = 0
    total_success: int = 0
    total_failures: int = 0
    error_rate: float = 0.0
    avg_duration_s: float = 0.0
    components: List[ComponentHealth] = field(default_factory=list)
    throughput: List[ThroughputPoint] = field(default_factory=list)
    errors: List[ErrorSummary] = field(default_factory=list)
    resource_avg: Dict[str, float] = field(default_factory=dict)
    timeframe_hours: int = 24
    generated_at: float = 0.0

    def __post_init__(self):
        if self.generated_at == 0.0:
            self.generated_at = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["components"] = [c.to_dict() if isinstance(c, ComponentHealth) else c for c in self.components]
        d["throughput"] = [t.to_dict() if isinstance(t, ThroughputPoint) else t for t in self.throughput]
        d["errors"] = [e.to_dict() if isinstance(e, ErrorSummary) else e for e in self.errors]
        return d


# ---------------------------------------------------------------------------
# SQLite connection helpers
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection."""
    conn = getattr(_LOCAL, "health_conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _LOCAL.health_conn = conn
    return conn


def _init_db():
    """Create tables and indexes if they do not exist."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                operation      TEXT NOT NULL,
                duration_s     REAL NOT NULL,
                success        INTEGER NOT NULL DEFAULT 1,
                error_type     TEXT DEFAULT '',
                error_message  TEXT DEFAULT '',
                cpu_pct        REAL DEFAULT 0,
                gpu_pct        REAL DEFAULT 0,
                ram_mb         REAL DEFAULT 0,
                disk_write_mb  REAL DEFAULT 0,
                timestamp      REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hm_operation ON health_metrics (operation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hm_timestamp ON health_metrics (timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hm_success ON health_metrics (success)")
        conn.commit()
        _INITIALIZED = True
        logger.debug("Pipeline health DB initialized at %s", _DB_PATH)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def record_metric(
    operation: str,
    duration_s: float,
    success: bool,
    error_type: str = "",
    error_message: str = "",
    cpu_pct: float = 0.0,
    gpu_pct: float = 0.0,
    ram_mb: float = 0.0,
    disk_write_mb: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> HealthMetric:
    """Record a single pipeline metric event.

    Args:
        operation:     Name of the pipeline operation (e.g. ``transcode``).
        duration_s:    Wall-clock duration in seconds.
        success:       Whether the operation succeeded.
        error_type:    Error class name if failed.
        error_message: Short error description if failed.
        cpu_pct:       Average CPU usage during operation.
        gpu_pct:       Average GPU usage during operation.
        ram_mb:        Peak RAM usage in MB.
        disk_write_mb: Total bytes written to disk in MB.
        on_progress:   Optional progress callback.

    Returns:
        The recorded ``HealthMetric``.
    """
    _init_db()
    metric = HealthMetric(
        operation=operation,
        duration_s=duration_s,
        success=success,
        error_type=error_type,
        error_message=error_message,
        cpu_pct=cpu_pct,
        gpu_pct=gpu_pct,
        ram_mb=ram_mb,
        disk_write_mb=disk_write_mb,
    )
    conn = _get_conn()
    conn.execute(
        "INSERT INTO health_metrics "
        "(operation, duration_s, success, error_type, error_message, "
        " cpu_pct, gpu_pct, ram_mb, disk_write_mb, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            metric.operation, metric.duration_s, 1 if metric.success else 0,
            metric.error_type, metric.error_message,
            metric.cpu_pct, metric.gpu_pct, metric.ram_mb, metric.disk_write_mb,
            metric.timestamp,
        ),
    )
    conn.commit()
    if on_progress:
        on_progress(100, "Metric recorded")
    logger.debug("Recorded health metric: %s success=%s dur=%.1fs", operation, success, duration_s)
    return metric


# ---------------------------------------------------------------------------
# Health queries
# ---------------------------------------------------------------------------
def _compute_component_health(
    operation: str,
    rows: list,
    thresholds: Optional[Dict] = None,
) -> ComponentHealth:
    """Compute health score for a single component from its metric rows."""
    th = thresholds or DEFAULT_THRESHOLDS
    total = len(rows)
    successes = sum(1 for r in rows if r["success"])
    failures = total - successes
    error_rate = failures / total if total > 0 else 0.0
    avg_dur = sum(r["duration_s"] for r in rows) / total if total > 0 else 0.0

    # Find most common error type
    error_counts: Dict[str, int] = {}
    for r in rows:
        if not r["success"] and r["error_type"]:
            error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1
    most_common_error = max(error_counts, key=error_counts.get) if error_counts else ""

    # Health score calculation:
    #   Start at 100, deduct for error rate, slow performance, trends
    score = 100

    # Deduct for error rate (up to 50 points)
    score -= int(min(error_rate * 200, 50))

    # Deduct for slow performance relative to baseline
    baseline = OPERATION_BASELINES.get(operation, 30.0)
    if avg_dur > baseline * th["avg_duration_factor"]:
        score -= 30
    elif avg_dur > baseline * 2.0:
        score -= 15
    elif avg_dur > baseline * 1.5:
        score -= 5

    # Deduct for low sample size (less confidence)
    if total < 5:
        score -= 5

    score = max(0, min(100, score))

    # Determine trend by comparing first half vs second half error rates
    trend = "stable"
    if total >= 10:
        mid = total // 2
        first_half_err = sum(1 for r in rows[:mid] if not r["success"]) / mid
        second_half_err = sum(1 for r in rows[mid:] if not r["success"]) / (total - mid)
        if second_half_err > first_half_err + 0.05:
            trend = "degrading"
        elif second_half_err < first_half_err - 0.05:
            trend = "improving"

    # Alert level
    alert_level = "ok"
    if error_rate >= th["error_rate_critical"] or score < th["min_health_score"]:
        alert_level = "critical"
    elif error_rate >= th["error_rate_warn"] or score < 70:
        alert_level = "warning"

    return ComponentHealth(
        component=operation,
        health_score=score,
        total_jobs=total,
        successful_jobs=successes,
        failed_jobs=failures,
        avg_duration_s=round(avg_dur, 2),
        error_rate=round(error_rate, 4),
        most_common_error=most_common_error,
        trend=trend,
        alert_level=alert_level,
    )


def _compute_throughput(rows: list, timeframe_hours: int) -> List[ThroughputPoint]:
    """Bucket metric rows into hourly throughput data points."""
    if not rows:
        return []
    now = time.time()
    start = now - (timeframe_hours * 3600)
    buckets: Dict[int, list] = {}
    for r in rows:
        ts = r["timestamp"]
        if ts < start:
            continue
        bucket_key = int((ts - start) // 3600)
        buckets.setdefault(bucket_key, []).append(r)

    points = []
    for bucket_idx in range(timeframe_hours):
        bucket_rows = buckets.get(bucket_idx, [])
        completed = sum(1 for r in bucket_rows if r["success"])
        failed = sum(1 for r in bucket_rows if not r["success"])
        avg_d = (sum(r["duration_s"] for r in bucket_rows) / len(bucket_rows)) if bucket_rows else 0.0
        points.append(ThroughputPoint(
            hour_start=start + bucket_idx * 3600,
            jobs_completed=completed,
            jobs_failed=failed,
            avg_duration_s=round(avg_d, 2),
        ))
    return points


def get_pipeline_health(
    timeframe_hours: int = 24,
    thresholds: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> PipelineHealthResult:
    """Generate a full pipeline health report.

    Args:
        timeframe_hours: How far back to look (default 24h).
        thresholds:      Custom alert thresholds (merged with defaults).
        on_progress:     Optional progress callback.

    Returns:
        ``PipelineHealthResult`` with per-component scores, throughput,
        error breakdown, and resource averages.
    """
    _init_db()
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    conn = _get_conn()
    cutoff = time.time() - (timeframe_hours * 3600)

    if on_progress:
        on_progress(10, "Querying metrics")

    rows = conn.execute(
        "SELECT * FROM health_metrics WHERE timestamp >= ? ORDER BY timestamp ASC",
        (cutoff,),
    ).fetchall()

    if on_progress:
        on_progress(30, "Computing component health")

    # Group by operation
    ops: Dict[str, list] = {}
    for r in rows:
        ops.setdefault(r["operation"], []).append(r)

    components = []
    for op_name, op_rows in sorted(ops.items()):
        components.append(_compute_component_health(op_name, op_rows, th))

    if on_progress:
        on_progress(50, "Computing throughput")

    throughput = _compute_throughput(rows, timeframe_hours)

    if on_progress:
        on_progress(70, "Summarizing errors")

    # Error summary
    errors = _build_error_summary(rows)

    # Resource averages
    cpu_vals = [r["cpu_pct"] for r in rows if r["cpu_pct"] > 0]
    gpu_vals = [r["gpu_pct"] for r in rows if r["gpu_pct"] > 0]
    ram_vals = [r["ram_mb"] for r in rows if r["ram_mb"] > 0]
    resource_avg = {
        "cpu_pct": round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else 0.0,
        "gpu_pct": round(sum(gpu_vals) / len(gpu_vals), 1) if gpu_vals else 0.0,
        "ram_mb": round(sum(ram_vals) / len(ram_vals), 1) if ram_vals else 0.0,
    }

    # Overall stats
    total = len(rows)
    successes = sum(1 for r in rows if r["success"])
    failures = total - successes
    error_rate = failures / total if total > 0 else 0.0
    avg_dur = sum(r["duration_s"] for r in rows) / total if total > 0 else 0.0

    # Overall score = weighted average of component scores
    if components:
        weighted = sum(c.health_score * c.total_jobs for c in components)
        total_weight = sum(c.total_jobs for c in components)
        overall_score = int(weighted / total_weight) if total_weight else 100
    else:
        overall_score = 100

    # Overall status
    if overall_score < th["min_health_score"]:
        overall_status = "critical"
    elif overall_score < 70 or error_rate >= th["error_rate_warn"]:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    if on_progress:
        on_progress(100, "Health report complete")

    return PipelineHealthResult(
        overall_score=overall_score,
        overall_status=overall_status,
        total_jobs=total,
        total_success=successes,
        total_failures=failures,
        error_rate=round(error_rate, 4),
        avg_duration_s=round(avg_dur, 2),
        components=components,
        throughput=throughput,
        errors=errors,
        resource_avg=resource_avg,
        timeframe_hours=timeframe_hours,
    )


def _build_error_summary(rows: list) -> List[ErrorSummary]:
    """Group failed metric rows into error summaries."""
    error_map: Dict[str, dict] = {}
    for r in rows:
        if r["success"]:
            continue
        etype = r["error_type"] or "unknown"
        if etype not in error_map:
            error_map[etype] = {
                "count": 0,
                "last_seen": 0.0,
                "operations": set(),
                "sample_message": "",
            }
        entry = error_map[etype]
        entry["count"] += 1
        if r["timestamp"] > entry["last_seen"]:
            entry["last_seen"] = r["timestamp"]
            if r["error_message"]:
                entry["sample_message"] = r["error_message"]
        entry["operations"].add(r["operation"])

    summaries = []
    for etype, info in sorted(error_map.items(), key=lambda x: -x[1]["count"]):
        summaries.append(ErrorSummary(
            error_type=etype,
            count=info["count"],
            last_seen=info["last_seen"],
            affected_operations=sorted(info["operations"]),
            sample_message=info["sample_message"],
        ))
    return summaries


def get_error_summary(
    timeframe_hours: int = 24,
    on_progress: Optional[Callable] = None,
) -> List[ErrorSummary]:
    """Return error summary for the given timeframe.

    Args:
        timeframe_hours: How far back to look.
        on_progress:     Optional progress callback.

    Returns:
        List of ``ErrorSummary`` sorted by frequency descending.
    """
    _init_db()
    conn = _get_conn()
    cutoff = time.time() - (timeframe_hours * 3600)

    if on_progress:
        on_progress(20, "Querying error metrics")

    rows = conn.execute(
        "SELECT * FROM health_metrics WHERE timestamp >= ? AND success = 0 "
        "ORDER BY timestamp ASC",
        (cutoff,),
    ).fetchall()

    if on_progress:
        on_progress(60, "Building error summary")

    summaries = _build_error_summary(rows)

    if on_progress:
        on_progress(100, "Error summary complete")

    return summaries


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------
def purge_old_metrics(days: int = 90) -> int:
    """Delete metrics older than *days*. Returns count of deleted rows."""
    _init_db()
    conn = _get_conn()
    cutoff = time.time() - (days * 86400)
    cursor = conn.execute("DELETE FROM health_metrics WHERE timestamp < ?", (cutoff,))
    conn.commit()
    deleted = cursor.rowcount
    logger.info("Purged %d health metrics older than %d days", deleted, days)
    return deleted


def reset_health_db() -> None:
    """Drop all metrics. Used for testing."""
    _init_db()
    conn = _get_conn()
    conn.execute("DELETE FROM health_metrics")
    conn.commit()
