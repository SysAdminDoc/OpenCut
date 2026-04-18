"""
OpenCut Worker Pool

Priority-based thread pool for job execution. Replaces raw thread spawning
with bounded concurrency and job priority support.

Jobs with lower priority values run first (CRITICAL=0 before BACKGROUND=200).
When all worker threads are busy, queued jobs are dispatched in priority order.
"""

import logging
import queue
import threading
from concurrent.futures import Future
from enum import IntEnum

from opencut.jobs import _update_job

logger = logging.getLogger("opencut")


class JobPriority(IntEnum):
    """Job priority levels. Lower value = higher priority."""
    CRITICAL = 0   # System operations (health, model management)
    HIGH = 10       # Quick CPU operations (silence detect, beat markers)
    NORMAL = 50     # Standard operations (transcribe, denoise, export)
    LOW = 100       # Heavy AI operations (upscale, style transfer, music gen)
    BACKGROUND = 200  # Batch/indexing operations


class WorkerPool:
    """Thread pool with priority queue for OpenCut job execution.

    Unlike ThreadPoolExecutor (FIFO), this pool dispatches queued work
    items in priority order so CRITICAL/HIGH jobs leapfrog BACKGROUND work.
    """

    def __init__(self, max_workers=10):
        self._max_workers = max_workers
        self._work_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        self._workers: list[threading.Thread] = []
        self._seq = 0  # tiebreaker for equal priorities

        # Start worker threads
        for i in range(max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True,
                                 name=f"oc-worker-{i}")
            t.start()
            self._workers.append(t)

        logger.info("WorkerPool initialized with %d max workers", max_workers)

    def submit(self, job_id: str, fn, *args, priority=JobPriority.NORMAL, **kwargs) -> Future:
        """Submit a job function to the pool. Returns a Future.

        Jobs with lower priority values are executed first when workers
        are busy.  Equal-priority jobs run in submission order (FIFO).
        """
        if self._shutdown:
            raise RuntimeError("WorkerPool is shut down")

        future = Future()
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._futures[job_id] = future

        # PriorityQueue sorts by tuple: (priority, sequence, work_item)
        self._work_queue.put((int(priority), seq, (fn, args, kwargs, future, job_id)))
        return future

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled."""
        with self._lock:
            future = self._futures.get(job_id)
        if future:
            cancelled = future.cancel()
            if cancelled:
                # Remove from tracking so the entry doesn't leak
                with self._lock:
                    self._futures.pop(job_id, None)
            return cancelled
        return False

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        with self._lock:
            future = self._futures.get(job_id)
        return future is not None and future.running()

    def _worker_loop(self):
        """Worker thread main loop — pull from priority queue and execute.

        Catches ``BaseException`` rather than ``Exception`` around the user
        callable so that a KeyboardInterrupt / SystemExit from a job body
        cannot kill the worker thread without first marking the future
        complete. A killed worker that never returns means anyone awaiting
        ``future.result()`` blocks forever.
        """
        while True:
            try:
                item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutdown:
                    return
                continue

            if item[2] is None:
                # Poison pill — shutdown signal
                return

            _priority, _seq, (fn, args, kwargs, future, job_id) = item

            try:
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as exc:  # noqa: BLE001
                        # Always propagate to the future; never let a worker
                        # thread die with the future still unresolved.
                        future.set_exception(exc)
                        # Re-raise SystemExit / KeyboardInterrupt so the
                        # process actually exits as the signaller intended.
                        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                            raise
                    else:
                        future.set_result(result)
                else:
                    try:
                        _update_job(job_id, status="cancelled", message="Cancelled before starting")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Cancel-bookkeeping failed for %s: %s", job_id, exc)
            finally:
                with self._lock:
                    self._futures.pop(job_id, None)

    def active_count(self) -> int:
        """Number of currently tracked jobs (running + queued)."""
        with self._lock:
            return len(self._futures)

    def queue_size(self) -> int:
        """Number of jobs waiting in the priority queue."""
        return self._work_queue.qsize()

    def shutdown(self, wait=True):
        """Shut down the pool. Called on server exit."""
        with self._lock:
            already_shutdown = self._shutdown
            self._shutdown = True
            workers = list(self._workers)

        if not already_shutdown:
            cancelled_job_ids = []
            while True:
                try:
                    _priority, _seq, work_item = self._work_queue.get_nowait()
                except queue.Empty:
                    break
                if work_item is None:
                    continue
                _fn, _args, _kwargs, future, job_id = work_item
                if future.cancel():
                    cancelled_job_ids.append(job_id)
                    with self._lock:
                        self._futures.pop(job_id, None)

            for job_id in cancelled_job_ids:
                try:
                    _update_job(
                        job_id,
                        status="cancelled",
                        message="Cancelled during server shutdown",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Shutdown cancel bookkeeping failed for %s: %s", job_id, exc)

            # Wake any idle workers and let running jobs exit cleanly after
            # their current task completes.
            for _ in workers:
                self._work_queue.put((-1, -1, None))
        if wait:
            for t in workers:
                t.join(timeout=10)
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


def cancel_job(job_id: str) -> bool:
    """Cancel a queued job without creating the pool on demand."""
    with _pool_lock:
        pool = _pool
    if pool is None:
        return False
    return pool.cancel(job_id)


def shutdown_pool(wait=True):
    """Shut down the global pool. Called on server exit."""
    global _pool
    with _pool_lock:
        pool = _pool
        _pool = None
    if pool is not None:
        pool.shutdown(wait=wait)
