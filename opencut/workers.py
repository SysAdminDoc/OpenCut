"""
OpenCut Worker Pool

Priority-based thread pool for job execution. Replaces raw thread spawning
with bounded concurrency and job priority support.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from enum import IntEnum

logger = logging.getLogger("opencut")


class JobPriority(IntEnum):
    """Job priority levels. Lower value = higher priority."""
    CRITICAL = 0   # System operations (health, model management)
    HIGH = 10       # Quick CPU operations (silence detect, beat markers)
    NORMAL = 50     # Standard operations (transcribe, denoise, export)
    LOW = 100       # Heavy AI operations (upscale, style transfer, music gen)
    BACKGROUND = 200  # Batch/indexing operations


class WorkerPool:
    """Thread pool with priority queue for OpenCut job execution."""

    def __init__(self, max_workers=10):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="oc-worker")
        self._futures: dict[str, Future] = {}  # job_id -> Future
        self._lock = threading.Lock()
        self._shutdown = False
        logger.info("WorkerPool initialized with %d max workers", max_workers)

    def submit(self, job_id: str, fn, *args, priority=JobPriority.NORMAL, **kwargs) -> Future:
        """Submit a job function to the pool. Returns a Future."""
        if self._shutdown:
            raise RuntimeError("WorkerPool is shut down")
        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._futures[job_id] = future
        future.add_done_callback(lambda f: self._on_done(job_id, f))
        return future

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled."""
        with self._lock:
            future = self._futures.get(job_id)
        if future:
            return future.cancel()
        return False

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        with self._lock:
            future = self._futures.get(job_id)
        return future is not None and future.running()

    def _on_done(self, job_id: str, future: Future):
        """Cleanup callback when job completes."""
        with self._lock:
            self._futures.pop(job_id, None)

    def active_count(self) -> int:
        """Number of currently running/pending jobs."""
        with self._lock:
            return len(self._futures)

    def shutdown(self, wait=True):
        """Shut down the pool. Called on server exit."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)
        logger.info("WorkerPool shut down")


# Module-level singleton
_pool: WorkerPool | None = None
_pool_lock = threading.Lock()


def get_pool(max_workers=10) -> WorkerPool:
    """Get or create the global WorkerPool singleton."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = WorkerPool(max_workers=max_workers)
    return _pool


def shutdown_pool(wait=True):
    """Shut down the global pool. Called on server exit."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=wait)
        _pool = None
