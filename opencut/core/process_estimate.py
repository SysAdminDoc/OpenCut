"""
OpenCut Processing Time Estimation

Provides accurate time estimates for video operations based on input
characteristics, operation type, hardware capabilities, and historical
performance data.  Learns from actual job durations to improve over time.

Estimate database: ``~/.opencut/estimates.db``
"""

import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(os.path.expanduser("~"), ".opencut", "estimates.db")
_LOCAL = threading.local()
_INIT_LOCK = threading.Lock()
_INITIALIZED = False

# ---------------------------------------------------------------------------
# Baseline multipliers: seconds per second of input video at 1080p
# These are conservative defaults, refined by historical data.
# ---------------------------------------------------------------------------
OPERATION_BASELINES: Dict[str, Dict[str, Any]] = {
    "transcode": {
        "base_ratio": 0.8,
        "description": "Re-encode video to different codec/quality",
        "gpu_factor": 0.3,       # GPU brings it down to 30% of CPU time
        "resolution_scale": True,
    },
    "trim": {
        "base_ratio": 0.02,
        "description": "Cut a segment from video (stream copy)",
        "gpu_factor": 1.0,
        "resolution_scale": False,
    },
    "concat": {
        "base_ratio": 0.05,
        "description": "Join multiple video segments",
        "gpu_factor": 0.8,
        "resolution_scale": False,
    },
    "stabilize": {
        "base_ratio": 2.5,
        "description": "Video stabilization (two-pass)",
        "gpu_factor": 0.4,
        "resolution_scale": True,
    },
    "denoise": {
        "base_ratio": 3.0,
        "description": "Temporal/spatial noise reduction",
        "gpu_factor": 0.25,
        "resolution_scale": True,
    },
    "color_grade": {
        "base_ratio": 0.6,
        "description": "Apply LUT or color grading",
        "gpu_factor": 0.5,
        "resolution_scale": True,
    },
    "caption": {
        "base_ratio": 1.2,
        "description": "Auto-generate captions (Whisper)",
        "gpu_factor": 0.15,
        "resolution_scale": False,
    },
    "silence_detect": {
        "base_ratio": 0.1,
        "description": "Detect silence segments",
        "gpu_factor": 1.0,
        "resolution_scale": False,
    },
    "scene_detect": {
        "base_ratio": 0.3,
        "description": "Detect scene changes",
        "gpu_factor": 0.5,
        "resolution_scale": True,
    },
    "audio_mix": {
        "base_ratio": 0.15,
        "description": "Mix/normalize audio tracks",
        "gpu_factor": 1.0,
        "resolution_scale": False,
    },
    "export": {
        "base_ratio": 1.0,
        "description": "Final render/export",
        "gpu_factor": 0.3,
        "resolution_scale": True,
    },
    "thumbnail": {
        "base_ratio": 0.01,
        "description": "Extract thumbnail frame",
        "gpu_factor": 1.0,
        "resolution_scale": False,
    },
    "speed_adjust": {
        "base_ratio": 0.5,
        "description": "Speed up or slow down video",
        "gpu_factor": 0.4,
        "resolution_scale": True,
    },
    "loudness_normalize": {
        "base_ratio": 0.2,
        "description": "Normalize audio loudness (two-pass)",
        "gpu_factor": 1.0,
        "resolution_scale": False,
    },
    "noise_reduce": {
        "base_ratio": 0.4,
        "description": "Audio noise reduction",
        "gpu_factor": 0.5,
        "resolution_scale": False,
    },
    "auto_crop": {
        "base_ratio": 0.3,
        "description": "Auto-detect and apply crop",
        "gpu_factor": 0.6,
        "resolution_scale": True,
    },
    "watermark": {
        "base_ratio": 0.7,
        "description": "Overlay watermark/logo",
        "gpu_factor": 0.5,
        "resolution_scale": True,
    },
}

# Resolution scaling factors relative to 1080p baseline
_RESOLUTION_FACTORS: Dict[str, float] = {
    "480p": 0.4,
    "720p": 0.65,
    "1080p": 1.0,
    "1440p": 1.6,
    "2160p": 3.2,
    "4320p": 8.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TimeEstimate:
    """Estimated processing time for a single operation."""
    operation: str
    estimated_seconds: float = 0.0
    human_readable: str = ""
    confidence: str = "medium"       # low | medium | high
    factors: Dict[str, Any] = field(default_factory=dict)
    input_duration_s: float = 0.0
    gpu_available: bool = False

    def __post_init__(self):
        if not self.human_readable and self.estimated_seconds > 0:
            self.human_readable = format_duration(self.estimated_seconds)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchEstimate:
    """Estimated processing time for multiple operations in sequence."""
    estimates: List[TimeEstimate] = field(default_factory=list)
    total_seconds: float = 0.0
    total_human_readable: str = ""
    parallel_seconds: float = 0.0
    parallel_human_readable: str = ""

    def __post_init__(self):
        if not self.total_human_readable and self.total_seconds > 0:
            self.total_human_readable = format_duration(self.total_seconds)
        if not self.parallel_human_readable and self.parallel_seconds > 0:
            self.parallel_human_readable = format_duration(self.parallel_seconds)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["estimates"] = [e.to_dict() if isinstance(e, TimeEstimate) else e for e in self.estimates]
        return d


# ---------------------------------------------------------------------------
# Human-readable time formatting
# ---------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Convert seconds to a human-friendly string.

    Examples: ``"< 1 second"``, ``"12 seconds"``, ``"2 min 30 sec"``,
    ``"1 hr 15 min"``, ``"3 hr 0 min"``.
    """
    if seconds < 1:
        return "< 1 second"
    if seconds < 60:
        return f"{int(seconds)} second{'s' if int(seconds) != 1 else ''}"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        parts = [f"{mins} min"]
        if secs > 0:
            parts.append(f"{secs} sec")
        return " ".join(parts)
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours} hr {mins} min"


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def _classify_resolution(height: int) -> str:
    """Map pixel height to resolution tier."""
    if height <= 0:
        return "1080p"
    if height <= 540:
        return "480p"
    if height <= 900:
        return "720p"
    if height <= 1200:
        return "1080p"
    if height <= 1800:
        return "1440p"
    if height <= 3000:
        return "2160p"
    return "4320p"


def _get_resolution_factor(height: int) -> float:
    """Get processing time multiplier for video resolution."""
    tier = _classify_resolution(height)
    return _RESOLUTION_FACTORS.get(tier, 1.0)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def _detect_gpu() -> bool:
    """Check if a GPU is available for accelerated processing."""
    # Check for NVIDIA GPU via nvidia-smi
    import shutil
    if shutil.which("nvidia-smi"):
        return True
    # Check via torch
    try:
        import torch  # noqa: lazy
        return torch.cuda.is_available()
    except ImportError:
        pass
    return False


# ---------------------------------------------------------------------------
# Historical data (SQLite)
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection for estimate history."""
    conn = getattr(_LOCAL, "est_conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _LOCAL.est_conn = conn
    return conn


def _init_db():
    """Create estimation history table."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS estimate_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                operation      TEXT NOT NULL,
                input_duration REAL DEFAULT 0,
                resolution     TEXT DEFAULT '1080p',
                gpu_used       INTEGER DEFAULT 0,
                estimated_s    REAL NOT NULL,
                actual_s       REAL NOT NULL,
                accuracy_pct   REAL DEFAULT 0,
                timestamp      REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eh_operation ON estimate_history (operation)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eh_timestamp ON estimate_history (timestamp)")
        conn.commit()
        _INITIALIZED = True
        logger.debug("Estimate history DB initialized at %s", _DB_PATH)


def _get_historical_factor(operation: str, resolution: str, gpu: bool) -> Optional[float]:
    """Look up average actual/estimated ratio from history to improve estimates.

    Returns a correction factor, or None if insufficient data.
    """
    _init_db()
    conn = _get_conn()
    rows = conn.execute(
        "SELECT estimated_s, actual_s FROM estimate_history "
        "WHERE operation = ? AND resolution = ? AND gpu_used = ? "
        "ORDER BY timestamp DESC LIMIT 20",
        (operation, resolution, 1 if gpu else 0),
    ).fetchall()

    if len(rows) < 3:
        return None

    ratios = []
    for r in rows:
        if r["estimated_s"] > 0:
            ratios.append(r["actual_s"] / r["estimated_s"])

    if not ratios:
        return None

    # Weighted average favoring recent entries
    weights = [1.0 / (i + 1) for i in range(len(ratios))]
    total_w = sum(weights)
    weighted_avg = sum(r * w for r, w in zip(ratios, weights)) / total_w
    return weighted_avg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def estimate_processing_time(
    video_path: str,
    operation: str,
    params: Optional[Dict[str, Any]] = None,
    gpu_available: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> TimeEstimate:
    """Estimate how long an operation will take on a given video.

    Args:
        video_path:    Path to input video.
        operation:     Operation name (must be in ``OPERATION_BASELINES``).
        params:        Optional operation parameters that affect timing.
        gpu_available: Override GPU detection (None = auto-detect).
        on_progress:   Optional progress callback.

    Returns:
        ``TimeEstimate`` with seconds, human-readable string, and factors.
    """
    params = params or {}

    if on_progress:
        on_progress(10, "Probing video")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    height = info.get("height", 1080)
    resolution = _classify_resolution(height)

    if on_progress:
        on_progress(30, "Checking hardware")

    gpu = gpu_available if gpu_available is not None else _detect_gpu()

    baseline = OPERATION_BASELINES.get(operation)
    if baseline is None:
        # Unknown operation — rough guess at 1:1 ratio
        logger.warning("No baseline for operation %r, using 1:1 estimate", operation)
        est = max(duration, 1.0)
        return TimeEstimate(
            operation=operation,
            estimated_seconds=round(est, 1),
            confidence="low",
            input_duration_s=duration,
            gpu_available=gpu,
            factors={"note": "unknown_operation"},
        )

    if on_progress:
        on_progress(50, "Computing estimate")

    base_ratio = baseline["base_ratio"]
    gpu_factor = baseline["gpu_factor"] if gpu else 1.0
    res_factor = _get_resolution_factor(height) if baseline.get("resolution_scale") else 1.0

    # Base estimate
    est = max(duration, 0.1) * base_ratio * gpu_factor * res_factor

    # Apply parameter adjustments
    param_factor = 1.0
    quality = params.get("quality", "")
    if quality == "high":
        param_factor *= 1.5
    elif quality == "low":
        param_factor *= 0.6

    passes = params.get("passes", 1)
    if passes > 1:
        param_factor *= passes * 0.7  # multi-pass is not perfectly linear

    strength = params.get("strength")
    if strength is not None:
        # Higher strength = more processing (for stabilize, denoise, etc.)
        param_factor *= 0.5 + float(strength)

    est *= param_factor

    # Historical correction
    historical = _get_historical_factor(operation, resolution, gpu)
    confidence = "medium"
    if historical is not None:
        est *= historical
        confidence = "high"

    # Minimum floor
    est = max(est, 0.5)

    if on_progress:
        on_progress(100, "Estimate ready")

    factors = {
        "base_ratio": base_ratio,
        "gpu_factor": gpu_factor,
        "resolution_factor": res_factor,
        "resolution_tier": resolution,
        "param_factor": round(param_factor, 3),
        "historical_correction": round(historical, 3) if historical else None,
        "input_duration_s": duration,
    }

    return TimeEstimate(
        operation=operation,
        estimated_seconds=round(est, 1),
        confidence=confidence,
        factors=factors,
        input_duration_s=duration,
        gpu_available=gpu,
    )


def batch_estimate(
    video_path: str,
    operations: List[Dict[str, Any]],
    gpu_available: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> BatchEstimate:
    """Estimate time for multiple operations in sequence.

    Args:
        video_path:    Path to input video.
        operations:    List of dicts with ``operation`` key and optional ``params``.
        gpu_available: Override GPU detection.
        on_progress:   Optional progress callback.

    Returns:
        ``BatchEstimate`` with per-operation estimates and totals.
    """
    gpu = gpu_available if gpu_available is not None else _detect_gpu()
    estimates = []
    total_ops = len(operations)

    for i, op_spec in enumerate(operations):
        op_name = op_spec.get("operation", op_spec.get("name", "unknown"))
        op_params = op_spec.get("params", {})

        if on_progress:
            pct = int(10 + (80 * i / max(total_ops, 1)))
            on_progress(pct, f"Estimating {op_name}")

        est = estimate_processing_time(
            video_path=video_path,
            operation=op_name,
            params=op_params,
            gpu_available=gpu,
        )
        estimates.append(est)

    total_s = sum(e.estimated_seconds for e in estimates)
    # Parallel estimate: assume CPU and GPU ops can overlap somewhat
    cpu_ops = [e for e in estimates if not e.gpu_available or e.operation in ("trim", "silence_detect", "audio_mix")]
    gpu_ops = [e for e in estimates if e not in cpu_ops]
    cpu_total = sum(e.estimated_seconds for e in cpu_ops)
    gpu_total = sum(e.estimated_seconds for e in gpu_ops)
    parallel_s = max(cpu_total, gpu_total) + min(cpu_total, gpu_total) * 0.2

    if on_progress:
        on_progress(100, "Batch estimate complete")

    return BatchEstimate(
        estimates=estimates,
        total_seconds=round(total_s, 1),
        parallel_seconds=round(parallel_s, 1),
    )


def record_actual_time(
    operation: str,
    estimated_s: float,
    actual_s: float,
    input_duration: float = 0.0,
    resolution: str = "1080p",
    gpu_used: bool = False,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Record actual processing time to improve future estimates.

    Args:
        operation:      Operation name.
        estimated_s:    What was estimated.
        actual_s:       What actually happened.
        input_duration: Duration of input video.
        resolution:     Resolution tier (e.g. ``1080p``).
        gpu_used:       Whether GPU was used.
        on_progress:    Optional progress callback.

    Returns:
        Dict with accuracy info.
    """
    _init_db()
    accuracy = (1 - abs(estimated_s - actual_s) / max(estimated_s, 0.1)) * 100
    accuracy = max(0.0, min(100.0, accuracy))

    conn = _get_conn()
    conn.execute(
        "INSERT INTO estimate_history "
        "(operation, input_duration, resolution, gpu_used, estimated_s, actual_s, accuracy_pct, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (operation, input_duration, resolution, 1 if gpu_used else 0,
         estimated_s, actual_s, accuracy, time.time()),
    )
    conn.commit()

    if on_progress:
        on_progress(100, "Actual time recorded")

    logger.debug(
        "Recorded estimate accuracy: %s est=%.1fs actual=%.1fs accuracy=%.1f%%",
        operation, estimated_s, actual_s, accuracy,
    )
    return {
        "operation": operation,
        "estimated_s": estimated_s,
        "actual_s": actual_s,
        "accuracy_pct": round(accuracy, 1),
        "difference_s": round(actual_s - estimated_s, 1),
    }


def get_estimate_accuracy(operation: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
    """Get estimate accuracy statistics.

    Args:
        operation: Filter by operation (None = all).
        days:      Look-back window.

    Returns:
        Dict with avg_accuracy, total_records, per-operation breakdown.
    """
    _init_db()
    conn = _get_conn()
    cutoff = time.time() - (days * 86400)

    if operation:
        rows = conn.execute(
            "SELECT operation, accuracy_pct, estimated_s, actual_s "
            "FROM estimate_history WHERE operation = ? AND timestamp >= ?",
            (operation, cutoff),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT operation, accuracy_pct, estimated_s, actual_s "
            "FROM estimate_history WHERE timestamp >= ?",
            (cutoff,),
        ).fetchall()

    if not rows:
        return {"avg_accuracy": 0.0, "total_records": 0, "operations": {}}

    total_accuracy = sum(r["accuracy_pct"] for r in rows)
    avg_accuracy = total_accuracy / len(rows)

    # Per-operation breakdown
    ops: Dict[str, list] = {}
    for r in rows:
        ops.setdefault(r["operation"], []).append(r)

    op_stats = {}
    for op_name, op_rows in ops.items():
        op_acc = sum(r["accuracy_pct"] for r in op_rows) / len(op_rows)
        avg_est = sum(r["estimated_s"] for r in op_rows) / len(op_rows)
        avg_act = sum(r["actual_s"] for r in op_rows) / len(op_rows)
        op_stats[op_name] = {
            "avg_accuracy": round(op_acc, 1),
            "avg_estimated_s": round(avg_est, 1),
            "avg_actual_s": round(avg_act, 1),
            "sample_count": len(op_rows),
        }

    return {
        "avg_accuracy": round(avg_accuracy, 1),
        "total_records": len(rows),
        "operations": op_stats,
    }


def reset_estimate_db() -> None:
    """Drop all estimate history. Used for testing."""
    _init_db()
    conn = _get_conn()
    conn.execute("DELETE FROM estimate_history")
    conn.commit()
