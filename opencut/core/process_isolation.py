"""
OpenCut Process Isolation for GPU Workers (Feature 7.2)

Run AI models in separate Python processes to prevent OOM crashes from
taking down the main server.  Provides:

- Process pool with configurable VRAM budget tracking
- Worker lifecycle management (spawn, monitor, kill)
- VRAM monitoring via nvidia-smi subprocess calls
- Auto-kill for workers exceeding their VRAM budget
- Results communicated via multiprocessing.Queue
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import multiprocessing
import os
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

# Force spawn start method for GPU isolation (fork may copy CUDA state)
_MP_CONTEXT = multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# Dataclass types
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """Configuration for a GPU worker process."""
    model_name: str
    vram_required: int = 2048  # MB
    timeout: int = 600  # seconds
    env_vars: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class GPUWorkerProcess:
    """Represents a running GPU worker process."""
    pid: int
    model_name: str
    vram_allocated: int  # MB
    status: str = "starting"  # starting | running | completed | error | killed
    started_at: float = 0.0
    finished_at: float = 0.0
    worker_id: str = ""
    _process: Optional[Process] = field(default=None, repr=False)
    _result_queue: Optional[Queue] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "model_name": self.model_name,
            "vram_allocated": self.vram_allocated,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "worker_id": self.worker_id,
            "runtime_sec": (
                (self.finished_at or time.time()) - self.started_at
                if self.started_at else 0.0
            ),
        }


@dataclass
class IsolatedJobResult:
    """Result from an isolated GPU worker job."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    vram_peak: int = 0  # MB
    duration: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

def _query_nvidia_smi() -> List[Dict[str, Any]]:
    """Query nvidia-smi for GPU memory usage.

    Returns a list of dicts with keys:
      - index, name, memory_total, memory_used, memory_free (all in MB).
    Returns an empty list if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": int(parts[2]),
                    "memory_used": int(parts[3]),
                    "memory_free": int(parts[4]),
                })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def _get_total_vram_used() -> int:
    """Return total VRAM currently used across all GPUs (MB)."""
    gpus = _query_nvidia_smi()
    return sum(g["memory_used"] for g in gpus) if gpus else 0


def _get_total_vram_available() -> int:
    """Return total VRAM available across all GPUs (MB)."""
    gpus = _query_nvidia_smi()
    return sum(g["memory_free"] for g in gpus) if gpus else 0


def _get_vram_for_pid(pid: int) -> int:
    """Return VRAM used by a specific PID (MB).

    Uses nvidia-smi compute process queries.  Returns 0 if the PID
    is not found or nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return 0
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) == pid:
                return int(parts[1])
        return 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return 0


# ---------------------------------------------------------------------------
# Worker subprocess entry point
# ---------------------------------------------------------------------------

def _worker_entry(
    function_path: str,
    args: tuple,
    kwargs: dict,
    result_queue: multiprocessing.Queue,
    env_vars: Dict[str, str],
) -> None:
    """Entry point for a spawned worker process.

    Imports and executes ``function_path`` (e.g. ``opencut.core.llm.run_inference``),
    putting the result (or error) onto ``result_queue``.
    """
    # Apply environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

    start = time.time()
    try:
        # Resolve function from dotted path
        module_path, func_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)

        output = func(*args, **kwargs)
        duration = time.time() - start

        result_queue.put(IsolatedJobResult(
            success=True,
            output=output,
            duration=duration,
        ))
    except Exception as exc:
        duration = time.time() - start
        tb = traceback.format_exc()
        result_queue.put(IsolatedJobResult(
            success=False,
            error=f"{type(exc).__name__}: {exc}\n{tb}",
            duration=duration,
        ))


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Manages a pool of isolated GPU worker processes.

    Args:
        max_workers: Maximum number of concurrent workers.
        vram_budget_mb: Total VRAM budget across all workers (MB).
    """

    def __init__(
        self, max_workers: int = 2, vram_budget_mb: int = 8192
    ) -> None:
        self.max_workers = max(1, max_workers)
        self.vram_budget_mb = max(512, vram_budget_mb)
        self.active_workers: Dict[str, GPUWorkerProcess] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown = False

    def _next_worker_id(self) -> str:
        self._counter += 1
        return f"gpu-worker-{self._counter}"

    @property
    def vram_allocated(self) -> int:
        """Total VRAM allocated to active workers (MB)."""
        with self._lock:
            return sum(
                w.vram_allocated
                for w in self.active_workers.values()
                if w.status in ("starting", "running")
            )

    @property
    def vram_remaining(self) -> int:
        """Remaining VRAM budget (MB)."""
        return max(0, self.vram_budget_mb - self.vram_allocated)

    @property
    def worker_count(self) -> int:
        """Number of active (non-finished) workers."""
        with self._lock:
            return sum(
                1 for w in self.active_workers.values()
                if w.status in ("starting", "running")
            )

    def submit(
        self,
        model_name: str,
        function_path: str,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        config: Optional[WorkerConfig] = None,
    ) -> Tuple[str, Optional[GPUWorkerProcess]]:
        """Submit a job to run in an isolated worker process.

        Args:
            model_name: Name of the model/task being run.
            function_path: Dotted path to the function to execute
                (e.g. ``opencut.core.llm.run_inference``).
            args: Positional arguments for the function.
            kwargs: Keyword arguments for the function.
            config: Optional WorkerConfig with resource requirements.

        Returns:
            Tuple of (worker_id, GPUWorkerProcess) on success,
            or (worker_id, None) if the pool is full.
        """
        if kwargs is None:
            kwargs = {}
        if config is None:
            config = WorkerConfig(model_name=model_name)

        worker_id = self._next_worker_id()

        with self._lock:
            # Check worker limit
            active = sum(
                1 for w in self.active_workers.values()
                if w.status in ("starting", "running")
            )
            if active >= self.max_workers:
                logger.warning(
                    "Worker pool full (%d/%d), rejecting %s",
                    active, self.max_workers, model_name,
                )
                return worker_id, None

            # Check VRAM budget
            allocated = sum(
                w.vram_allocated
                for w in self.active_workers.values()
                if w.status in ("starting", "running")
            )
            if allocated + config.vram_required > self.vram_budget_mb:
                logger.warning(
                    "VRAM budget exceeded (%d + %d > %d), rejecting %s",
                    allocated, config.vram_required,
                    self.vram_budget_mb, model_name,
                )
                return worker_id, None

        # Create result queue and spawn process
        result_queue = _MP_CONTEXT.Queue()
        process = _MP_CONTEXT.Process(
            target=_worker_entry,
            args=(function_path, args, kwargs, result_queue, config.env_vars),
            daemon=True,
        )
        process.start()

        worker = GPUWorkerProcess(
            pid=process.pid or 0,
            model_name=model_name,
            vram_allocated=config.vram_required,
            status="running",
            started_at=time.time(),
            worker_id=worker_id,
            _process=process,
            _result_queue=result_queue,
        )

        with self._lock:
            self.active_workers[worker_id] = worker

        # Start monitor if not running
        self._ensure_monitor()

        logger.info(
            "Spawned GPU worker %s (pid=%d) for %s, VRAM=%dMB",
            worker_id, worker.pid, model_name, config.vram_required,
        )
        return worker_id, worker

    def get_result(
        self, worker_id: str, timeout: float = 0.0
    ) -> Optional[IsolatedJobResult]:
        """Get the result for a completed worker.

        Args:
            worker_id: The worker ID returned by ``submit()``.
            timeout: How long to wait for a result (0 = non-blocking).

        Returns:
            IsolatedJobResult if available, else None.
        """
        with self._lock:
            worker = self.active_workers.get(worker_id)
        if worker is None or worker._result_queue is None:
            return None

        try:
            result = worker._result_queue.get(
                block=timeout > 0, timeout=timeout if timeout > 0 else None
            )
            with self._lock:
                worker.status = "completed" if result.success else "error"
                worker.finished_at = time.time()
            return result
        except Exception:
            return None

    def kill_worker(self, worker_id: str) -> bool:
        """Kill a specific worker process.

        Returns True if the worker was found and killed.
        """
        with self._lock:
            worker = self.active_workers.get(worker_id)
        if worker is None:
            return False

        return self._kill_process(worker)

    def _kill_process(self, worker: GPUWorkerProcess) -> bool:
        """Terminate/kill a worker's subprocess."""
        proc = worker._process
        if proc is None:
            return False

        try:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)
        except Exception as exc:
            logger.warning("Error killing worker %s: %s", worker.worker_id, exc)

        with self._lock:
            worker.status = "killed"
            worker.finished_at = time.time()
        logger.info("Killed worker %s (pid=%d)", worker.worker_id, worker.pid)
        return True

    def cleanup(self) -> int:
        """Clean up all workers (kill active, remove finished).

        Returns the number of workers cleaned up.
        """
        self._shutdown = True
        count = 0

        with self._lock:
            worker_ids = list(self.active_workers.keys())

        for wid in worker_ids:
            with self._lock:
                worker = self.active_workers.get(wid)
            if worker is None:
                continue

            if worker.status in ("starting", "running"):
                self._kill_process(worker)

            # Close the queue
            if worker._result_queue is not None:
                try:
                    worker._result_queue.close()
                except Exception:
                    pass

            count += 1

        with self._lock:
            self.active_workers.clear()

        logger.info("Cleaned up %d workers", count)
        return count

    def get_status(self) -> dict:
        """Return pool status as a dict."""
        with self._lock:
            workers = {
                wid: w.to_dict()
                for wid, w in self.active_workers.items()
            }
            active = sum(
                1 for w in self.active_workers.values()
                if w.status in ("starting", "running")
            )
            allocated = sum(
                w.vram_allocated
                for w in self.active_workers.values()
                if w.status in ("starting", "running")
            )

        return {
            "max_workers": self.max_workers,
            "vram_budget_mb": self.vram_budget_mb,
            "active_workers": active,
            "total_workers": len(workers),
            "vram_allocated_mb": allocated,
            "vram_remaining_mb": max(0, self.vram_budget_mb - allocated),
            "workers": workers,
        }

    # ---- Monitor thread ----

    def _ensure_monitor(self) -> None:
        """Start the background monitor thread if not running."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._shutdown = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="gpu-worker-monitor"
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        """Periodically check workers for timeout and VRAM overuse."""
        while not self._shutdown:
            time.sleep(5)
            self._check_workers()

    def _check_workers(self) -> None:
        """Check all active workers for issues."""
        now = time.time()
        with self._lock:
            workers = [
                (wid, w)
                for wid, w in self.active_workers.items()
                if w.status in ("starting", "running")
            ]

        for wid, worker in workers:
            # Check process alive
            if worker._process is not None and not worker._process.is_alive():
                with self._lock:
                    if worker.status in ("starting", "running"):
                        worker.status = "completed"
                        worker.finished_at = now
                continue

            # Check timeout
            elapsed = now - worker.started_at
            # Use a generous default timeout
            timeout = 600
            if elapsed > timeout:
                logger.warning(
                    "Worker %s exceeded timeout (%ds), killing",
                    wid, timeout,
                )
                self._kill_process(worker)
                continue

            # Check VRAM usage (auto-kill if exceeding budget)
            try:
                pid_vram = _get_vram_for_pid(worker.pid)
                if pid_vram > 0 and pid_vram > worker.vram_allocated * 1.5:
                    logger.warning(
                        "Worker %s using %dMB VRAM (allocated %dMB), killing",
                        wid, pid_vram, worker.vram_allocated,
                    )
                    self._kill_process(worker)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level pool singleton
# ---------------------------------------------------------------------------

_global_pool: Optional[WorkerPool] = None
_pool_lock = threading.Lock()


def create_worker_pool(
    max_workers: int = 2, vram_budget_mb: int = 8192
) -> WorkerPool:
    """Create (or replace) the global GPU worker pool.

    Args:
        max_workers: Maximum concurrent worker processes.
        vram_budget_mb: Total VRAM budget in MB.

    Returns:
        The newly created WorkerPool.
    """
    global _global_pool
    with _pool_lock:
        if _global_pool is not None:
            _global_pool.cleanup()
        _global_pool = WorkerPool(
            max_workers=max_workers, vram_budget_mb=vram_budget_mb
        )
    return _global_pool


def get_worker_pool() -> Optional[WorkerPool]:
    """Return the global worker pool (None if not created)."""
    return _global_pool


def submit_isolated_job(
    pool: WorkerPool,
    model_name: str,
    function_path: str,
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> Tuple[str, Optional[GPUWorkerProcess]]:
    """Submit a job to run in an isolated subprocess.

    Args:
        pool: The WorkerPool to submit to.
        model_name: Model/task identifier.
        function_path: Dotted path to the function.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Tuple of (worker_id, GPUWorkerProcess or None).
    """
    return pool.submit(
        model_name=model_name,
        function_path=function_path,
        args=args,
        kwargs=kwargs,
    )


def get_pool_status(pool: WorkerPool) -> dict:
    """Return the status of a worker pool.

    Returns dict with active workers, VRAM usage, etc.
    """
    return pool.get_status()


def kill_worker(pool: WorkerPool, worker_id: str) -> bool:
    """Kill a specific worker in the pool.

    Returns True if the worker was found and killed.
    """
    return pool.kill_worker(worker_id)


def cleanup_pool(pool: WorkerPool) -> int:
    """Clean up all workers in the pool.

    Returns the count of workers cleaned up.
    """
    return pool.cleanup()
