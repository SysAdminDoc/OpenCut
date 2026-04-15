"""
OpenCut Resource Monitor

Real-time CPU, GPU, RAM, and disk monitoring for pipeline job management.
Provides snapshots, trend history, GPU info (NVIDIA via nvidia-smi or
torch.cuda), and resource availability checks before job submission.
"""

import logging
import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Snapshot history ring buffer
# ---------------------------------------------------------------------------
_HISTORY_LOCK = threading.Lock()
_HISTORY: List[dict] = []
_HISTORY_MAX = 360          # ~6 hours at 1-minute intervals
_SNAPSHOT_INTERVAL = 60.0   # seconds between auto-snapshots
_AUTO_THREAD: Optional[threading.Thread] = None
_AUTO_STOP = threading.Event()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int = 0
    name: str = ""
    driver_version: str = ""
    utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    fan_speed_pct: float = 0.0

    @property
    def memory_used_pct(self) -> float:
        if self.memory_total_mb > 0:
            return round(self.memory_used_mb / self.memory_total_mb * 100, 1)
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["memory_used_pct"] = self.memory_used_pct
        return d


@dataclass
class DiskInfo:
    """Disk usage for a single mount/drive."""
    path: str = ""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    used_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResourceSnapshot:
    """Point-in-time resource utilization snapshot."""
    timestamp: float = 0.0
    cpu_total_pct: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    cpu_count: int = 0
    ram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    ram_used_pct: float = 0.0
    gpus: List[GPUInfo] = field(default_factory=list)
    disks: List[DiskInfo] = field(default_factory=list)
    cpu_temp_c: float = 0.0
    gpu_temp_c: float = 0.0
    process_ram_mb: float = 0.0
    process_cpu_pct: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpus"] = [g.to_dict() if isinstance(g, GPUInfo) else g for g in self.gpus]
        d["disks"] = [dk.to_dict() if isinstance(dk, DiskInfo) else dk for dk in self.disks]
        return d


@dataclass
class ResourceRequirements:
    """Minimum resource requirements for a job."""
    min_ram_mb: float = 512
    min_disk_gb: float = 1.0
    min_gpu_memory_mb: float = 0
    max_cpu_pct: float = 95.0         # refuse if CPU already above this
    max_ram_pct: float = 90.0         # refuse if RAM already above this
    prefer_gpu: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResourceMonitorResult:
    """Result of a resource availability check."""
    available: bool = True
    snapshot: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "available": self.available,
            "snapshot": self.snapshot.to_dict() if isinstance(self.snapshot, ResourceSnapshot) else self.snapshot,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }
        return d


# ---------------------------------------------------------------------------
# CPU / RAM via psutil (lazy import)
# ---------------------------------------------------------------------------
def _get_cpu_info() -> Dict[str, Any]:
    """Get CPU usage. Falls back to os.cpu_count if psutil unavailable."""
    try:
        import psutil
        cpu_total = psutil.cpu_percent(interval=0.5)
        cpu_per_core = psutil.cpu_percent(interval=0, percpu=True)
        cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        return {
            "cpu_total_pct": cpu_total,
            "cpu_per_core": cpu_per_core,
            "cpu_count": cpu_count,
        }
    except ImportError:
        cpu_count = os.cpu_count() or 1
        return {
            "cpu_total_pct": 0.0,
            "cpu_per_core": [0.0] * cpu_count,
            "cpu_count": cpu_count,
        }


def _get_ram_info() -> Dict[str, float]:
    """Get RAM usage. Falls back to zeros if psutil unavailable."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "ram_total_mb": round(mem.total / (1024 * 1024), 1),
            "ram_used_mb": round(mem.used / (1024 * 1024), 1),
            "ram_available_mb": round(mem.available / (1024 * 1024), 1),
            "ram_used_pct": mem.percent,
        }
    except ImportError:
        return {
            "ram_total_mb": 0.0,
            "ram_used_mb": 0.0,
            "ram_available_mb": 0.0,
            "ram_used_pct": 0.0,
        }


def _get_process_info() -> Dict[str, float]:
    """Get current process resource usage."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        cpu_pct = proc.cpu_percent(interval=0)
        return {
            "process_ram_mb": round(mem_info.rss / (1024 * 1024), 1),
            "process_cpu_pct": cpu_pct,
        }
    except (ImportError, Exception):
        return {"process_ram_mb": 0.0, "process_cpu_pct": 0.0}


def _get_cpu_temperature() -> float:
    """Attempt to read CPU temperature. Returns 0.0 if unavailable."""
    try:
        import psutil
        temps = psutil.sensors_temperatures()
        if not temps:
            return 0.0
        # Look for common sensor names
        for name in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if name in temps:
                entries = temps[name]
                if entries:
                    return round(entries[0].current, 1)
        # Fallback: first available sensor
        for entries in temps.values():
            if entries:
                return round(entries[0].current, 1)
    except (ImportError, AttributeError, OSError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# GPU via nvidia-smi
# ---------------------------------------------------------------------------
def get_gpu_info(on_progress: Optional[Callable] = None) -> List[GPUInfo]:
    """Query NVIDIA GPU info via nvidia-smi.

    Falls back to torch.cuda if nvidia-smi is not available.

    Args:
        on_progress: Optional progress callback.

    Returns:
        List of ``GPUInfo`` for each detected GPU.
    """
    if on_progress:
        on_progress(20, "Querying GPU")

    gpus = _query_nvidia_smi()
    if gpus:
        if on_progress:
            on_progress(100, "GPU info retrieved")
        return gpus

    # Fallback to torch.cuda
    gpus = _query_torch_cuda()
    if on_progress:
        on_progress(100, "GPU info retrieved")
    return gpus


def _query_nvidia_smi() -> List[GPUInfo]:
    """Parse nvidia-smi CSV output."""
    import shutil
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        fields = (
            "index,name,driver_version,utilization.gpu,"
            "memory.used,memory.total,memory.free,"
            "temperature.gpu,power.draw,power.limit,fan.speed"
        )
        cmd = [nvidia_smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue

            def _safe_float(val: str) -> float:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            gpus.append(GPUInfo(
                index=int(parts[0]) if parts[0].isdigit() else 0,
                name=parts[1],
                driver_version=parts[2],
                utilization_pct=_safe_float(parts[3]),
                memory_used_mb=_safe_float(parts[4]),
                memory_total_mb=_safe_float(parts[5]),
                memory_free_mb=_safe_float(parts[6]),
                temperature_c=_safe_float(parts[7]),
                power_draw_w=_safe_float(parts[8]),
                power_limit_w=_safe_float(parts[9]),
                fan_speed_pct=_safe_float(parts[10]),
            ))
        return gpus

    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        logger.debug("nvidia-smi query failed: %s", exc)
        return []


def _query_torch_cuda() -> List[GPUInfo]:
    """Query GPU info via PyTorch CUDA (fallback)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_alloc = torch.cuda.memory_allocated(i) / (1024 * 1024)
            mem_total = props.total_mem / (1024 * 1024)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                memory_total_mb=round(mem_total, 1),
                memory_used_mb=round(mem_alloc, 1),
                memory_free_mb=round(mem_total - mem_alloc, 1),
            ))
        return gpus
    except (ImportError, RuntimeError):
        return []


# ---------------------------------------------------------------------------
# Disk usage
# ---------------------------------------------------------------------------
def _get_disk_info(extra_paths: Optional[List[str]] = None) -> List[DiskInfo]:
    """Get disk usage for key paths."""
    import shutil as _shutil
    paths_to_check = set()

    # Home directory
    home = os.path.expanduser("~")
    paths_to_check.add(home)

    # OpenCut data dir
    opencut_dir = os.path.join(home, ".opencut")
    if os.path.isdir(opencut_dir):
        paths_to_check.add(opencut_dir)

    # Temp directory
    import tempfile
    paths_to_check.add(tempfile.gettempdir())

    if extra_paths:
        for p in extra_paths:
            if os.path.exists(p):
                paths_to_check.add(p)

    disks = []
    seen_drives = set()
    for path in sorted(paths_to_check):
        # Deduplicate by drive root (Windows) or mount (POSIX)
        if platform.system() == "Windows":
            drive = os.path.splitdrive(path)[0].upper()
        else:
            drive = path
        if drive in seen_drives:
            continue
        seen_drives.add(drive)

        try:
            usage = _shutil.disk_usage(path)
            disks.append(DiskInfo(
                path=path,
                total_gb=round(usage.total / (1024 ** 3), 1),
                used_gb=round(usage.used / (1024 ** 3), 1),
                free_gb=round(usage.free / (1024 ** 3), 1),
                used_pct=round(usage.used / usage.total * 100, 1) if usage.total > 0 else 0.0,
            ))
        except OSError:
            continue

    return disks


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
def get_resource_snapshot(
    extra_disk_paths: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> ResourceSnapshot:
    """Capture a full resource utilization snapshot.

    Args:
        extra_disk_paths: Additional paths to check disk usage on.
        on_progress:      Optional progress callback.

    Returns:
        ``ResourceSnapshot`` with CPU, RAM, GPU, disk, and temperature data.
    """
    if on_progress:
        on_progress(10, "Reading CPU")

    cpu = _get_cpu_info()

    if on_progress:
        on_progress(25, "Reading RAM")

    ram = _get_ram_info()
    proc = _get_process_info()

    if on_progress:
        on_progress(40, "Querying GPU")

    gpus = get_gpu_info()

    if on_progress:
        on_progress(60, "Checking disk")

    disks = _get_disk_info(extra_disk_paths)
    cpu_temp = _get_cpu_temperature()
    gpu_temp = gpus[0].temperature_c if gpus else 0.0

    if on_progress:
        on_progress(100, "Snapshot complete")

    snap = ResourceSnapshot(
        cpu_total_pct=cpu["cpu_total_pct"],
        cpu_per_core=cpu["cpu_per_core"],
        cpu_count=cpu["cpu_count"],
        ram_total_mb=ram["ram_total_mb"],
        ram_used_mb=ram["ram_used_mb"],
        ram_available_mb=ram["ram_available_mb"],
        ram_used_pct=ram["ram_used_pct"],
        gpus=gpus,
        disks=disks,
        cpu_temp_c=cpu_temp,
        gpu_temp_c=gpu_temp,
        process_ram_mb=proc["process_ram_mb"],
        process_cpu_pct=proc["process_cpu_pct"],
    )

    # Store in history
    _store_snapshot(snap)
    return snap


def _store_snapshot(snap: ResourceSnapshot) -> None:
    """Append snapshot to in-memory history ring buffer."""
    with _HISTORY_LOCK:
        _HISTORY.append(snap.to_dict())
        if len(_HISTORY) > _HISTORY_MAX:
            _HISTORY.pop(0)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def get_resource_history(
    minutes: int = 60,
    on_progress: Optional[Callable] = None,
) -> List[ResourceSnapshot]:
    """Retrieve recent resource snapshots.

    Args:
        minutes:     How many minutes of history to return.
        on_progress: Optional callback.

    Returns:
        List of ``ResourceSnapshot`` within the time window.
    """
    cutoff = time.time() - (minutes * 60)
    result = []

    if on_progress:
        on_progress(50, "Reading history")

    with _HISTORY_LOCK:
        for entry in _HISTORY:
            if entry.get("timestamp", 0) >= cutoff:
                # Reconstruct snapshot from dict
                snap = ResourceSnapshot(
                    timestamp=entry.get("timestamp", 0),
                    cpu_total_pct=entry.get("cpu_total_pct", 0),
                    cpu_per_core=entry.get("cpu_per_core", []),
                    cpu_count=entry.get("cpu_count", 0),
                    ram_total_mb=entry.get("ram_total_mb", 0),
                    ram_used_mb=entry.get("ram_used_mb", 0),
                    ram_available_mb=entry.get("ram_available_mb", 0),
                    ram_used_pct=entry.get("ram_used_pct", 0),
                    disks=[],
                    gpus=[],
                    cpu_temp_c=entry.get("cpu_temp_c", 0),
                    gpu_temp_c=entry.get("gpu_temp_c", 0),
                    process_ram_mb=entry.get("process_ram_mb", 0),
                    process_cpu_pct=entry.get("process_cpu_pct", 0),
                )
                result.append(snap)

    if on_progress:
        on_progress(100, "History loaded")

    return result


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------
def check_resource_availability(
    requirements: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable] = None,
) -> ResourceMonitorResult:
    """Check whether the system has enough resources to start a new job.

    Args:
        requirements: Dict of ``ResourceRequirements`` fields.
        on_progress:  Optional callback.

    Returns:
        ``ResourceMonitorResult`` with ``available`` flag and reasons.
    """
    reqs = ResourceRequirements()
    if requirements:
        for k, v in requirements.items():
            if hasattr(reqs, k):
                setattr(reqs, k, v)

    if on_progress:
        on_progress(20, "Taking snapshot")

    snap = get_resource_snapshot()
    reasons = []
    warnings = []

    if on_progress:
        on_progress(60, "Checking requirements")

    # RAM check
    if snap.ram_available_mb > 0 and snap.ram_available_mb < reqs.min_ram_mb:
        reasons.append(
            f"Insufficient RAM: {snap.ram_available_mb:.0f} MB available, "
            f"{reqs.min_ram_mb:.0f} MB required"
        )

    if snap.ram_used_pct > reqs.max_ram_pct:
        reasons.append(
            f"RAM usage too high: {snap.ram_used_pct:.1f}% "
            f"(max {reqs.max_ram_pct:.0f}%)"
        )

    # CPU check
    if snap.cpu_total_pct > reqs.max_cpu_pct:
        reasons.append(
            f"CPU usage too high: {snap.cpu_total_pct:.1f}% "
            f"(max {reqs.max_cpu_pct:.0f}%)"
        )

    # Disk check
    for disk in snap.disks:
        if disk.free_gb < reqs.min_disk_gb:
            reasons.append(
                f"Low disk space on {disk.path}: {disk.free_gb:.1f} GB free, "
                f"{reqs.min_disk_gb:.1f} GB required"
            )
        elif disk.free_gb < reqs.min_disk_gb * 2:
            warnings.append(
                f"Disk space getting low on {disk.path}: {disk.free_gb:.1f} GB free"
            )

    # GPU memory check
    if reqs.min_gpu_memory_mb > 0:
        gpu_ok = False
        for gpu in snap.gpus:
            if gpu.memory_free_mb >= reqs.min_gpu_memory_mb:
                gpu_ok = True
                break
        if not gpu_ok:
            if snap.gpus:
                free_list = ", ".join(f"{g.memory_free_mb:.0f} MB" for g in snap.gpus)
                reasons.append(
                    f"Insufficient GPU memory: need {reqs.min_gpu_memory_mb:.0f} MB, "
                    f"available: {free_list}"
                )
            else:
                reasons.append("No GPU detected but GPU memory required")

    # GPU preference warning
    if reqs.prefer_gpu and not snap.gpus:
        warnings.append("GPU preferred but no GPU detected; will fall back to CPU")

    # Temperature warnings
    if snap.cpu_temp_c > 90:
        warnings.append(f"CPU temperature high: {snap.cpu_temp_c:.0f} C")
    if snap.gpu_temp_c > 85:
        warnings.append(f"GPU temperature high: {snap.gpu_temp_c:.0f} C")

    available = len(reasons) == 0

    if on_progress:
        on_progress(100, "Availability check complete")

    return ResourceMonitorResult(
        available=available,
        snapshot=snap,
        reasons=reasons,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Auto-snapshot background thread
# ---------------------------------------------------------------------------
def start_auto_snapshot(interval_s: float = 60.0) -> None:
    """Start a background thread that captures periodic snapshots.

    Args:
        interval_s: Seconds between snapshots (default 60).
    """
    global _AUTO_THREAD, _SNAPSHOT_INTERVAL
    _SNAPSHOT_INTERVAL = interval_s

    if _AUTO_THREAD is not None and _AUTO_THREAD.is_alive():
        return

    _AUTO_STOP.clear()

    def _worker():
        while not _AUTO_STOP.is_set():
            try:
                get_resource_snapshot()
            except Exception as exc:
                logger.debug("Auto-snapshot failed: %s", exc)
            _AUTO_STOP.wait(timeout=_SNAPSHOT_INTERVAL)

    _AUTO_THREAD = threading.Thread(target=_worker, daemon=True, name="opencut-resource-monitor")
    _AUTO_THREAD.start()
    logger.info("Resource auto-snapshot started (interval=%.0fs)", interval_s)


def stop_auto_snapshot() -> None:
    """Stop the auto-snapshot background thread."""
    _AUTO_STOP.set()
    logger.info("Resource auto-snapshot stopped")


def clear_history() -> None:
    """Clear the in-memory snapshot history."""
    with _HISTORY_LOCK:
        _HISTORY.clear()
