"""
OpenCut Parallel Batch Executor (Phase 5.4)

Runs multiple batch operations in parallel using ThreadPoolExecutor.
Each operation is defined by an endpoint path and payload, and is
dispatched to the appropriate handler function.  Combined progress
is tracked via the existing job update mechanism.

Features:
- Configurable max_workers (default 2) to limit parallelism
- Per-operation progress tracking with combined rollup
- Partial failure support: some ops can succeed while others fail
- Cancellation propagation from parent job
- Thread-safe progress aggregation
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class OperationSpec:
    """A single operation to run inside a parallel batch."""
    endpoint: str       # e.g. "/video/watermark", "/audio/denoise"
    payload: Dict       # JSON body that would be sent to the endpoint
    index: int = 0      # Position within the batch (set by executor)


@dataclass
class OperationResult:
    """Result of a single parallel operation."""
    index: int
    endpoint: str
    status: str = "queued"     # queued, running, complete, error
    progress: int = 0
    message: str = "Queued"
    result: Any = None
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.finished_at if self.finished_at > 0 else time.time()
        return end - self.started_at

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "endpoint": self.endpoint,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
        }


class BatchExecutor:
    """Execute a list of operations in parallel with progress tracking.

    Parameters
    ----------
    operations : list[OperationSpec]
        Operations to execute.
    max_workers : int
        Maximum number of concurrent threads (default 2).
    job_id : str | None
        Parent job ID for progress updates via ``_update_job``.
    on_progress : callable | None
        Optional ``(pct: int, message: str) -> None`` callback invoked
        whenever the combined progress changes.
    """

    def __init__(
        self,
        operations: List[OperationSpec],
        max_workers: int = 2,
        job_id: Optional[str] = None,
        on_progress: Optional[Callable] = None,
    ):
        # Assign indices
        for i, op in enumerate(operations):
            op.index = i

        self.operations = operations
        self.max_workers = max(1, min(max_workers, 8))  # clamp 1-8
        self.job_id = job_id
        self.on_progress = on_progress

        self._results: List[OperationResult] = [
            OperationResult(index=op.index, endpoint=op.endpoint)
            for op in operations
        ]
        self._lock = threading.Lock()
        self._cancelled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, handler: Callable) -> List[OperationResult]:
        """Execute all operations using *handler* and return results.

        ``handler(op: OperationSpec, progress_cb)`` should execute the
        operation and return a result value.  *progress_cb* is
        ``(pct, msg) -> None`` for per-operation progress updates.

        Returns the list of :class:`OperationResult` objects.
        """
        total = len(self.operations)
        if total == 0:
            return self._results

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="oc-batch",
        ) as pool:
            future_to_idx = {}
            for op in self.operations:
                if self._cancelled:
                    break
                future = pool.submit(self._run_one, handler, op)
                future_to_idx[future] = op.index

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                # Future exceptions are captured inside _run_one,
                # but guard against unexpected errors here too.
                try:
                    future.result()
                except Exception as exc:
                    logger.exception(
                        "Unexpected error in parallel batch op %d", idx
                    )
                    with self._lock:
                        r = self._results[idx]
                        if r.status not in ("complete", "error"):
                            r.status = "error"
                            r.error = str(exc)
                            r.message = f"Error: {exc}"
                            r.finished_at = time.time()
                    self._report_combined_progress()

        # Mark any remaining queued ops as skipped (cancellation)
        with self._lock:
            for r in self._results:
                if r.status == "queued":
                    r.status = "skipped"
                    r.message = "Cancelled"

        self._report_combined_progress()
        return self._results

    def cancel(self):
        """Signal cancellation -- running ops will finish but no new ones start."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def combined_progress(self) -> int:
        """Overall progress across all operations (0-100)."""
        with self._lock:
            if not self._results:
                return 0
            total_pct = sum(r.progress for r in self._results)
            return int(total_pct / len(self._results))

    @property
    def summary(self) -> Dict:
        """Aggregate summary of all operations."""
        with self._lock:
            results_copy = [r.to_dict() for r in self._results]
            success = sum(1 for r in self._results if r.status == "complete")
            failed = sum(1 for r in self._results if r.status == "error")
            skipped = sum(1 for r in self._results if r.status == "skipped")
        return {
            "total": len(self.operations),
            "max_workers": self.max_workers,
            "progress": self.combined_progress,
            "operations": results_copy,
            "results": {
                "success": success,
                "failed": failed,
                "skipped": skipped,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_one(self, handler: Callable, op: OperationSpec):
        """Execute a single operation with error handling."""
        if self._cancelled:
            return

        idx = op.index

        with self._lock:
            self._results[idx].status = "running"
            self._results[idx].started_at = time.time()
            self._results[idx].message = "Processing..."
        self._report_combined_progress()

        def _progress_cb(pct: int, msg: str = ""):
            with self._lock:
                self._results[idx].progress = pct
                if msg:
                    self._results[idx].message = msg
            self._report_combined_progress()

        try:
            result = handler(op, _progress_cb)
            with self._lock:
                self._results[idx].status = "complete"
                self._results[idx].progress = 100
                self._results[idx].result = result
                self._results[idx].message = "Done"
                self._results[idx].finished_at = time.time()
        except Exception as exc:
            logger.warning("Parallel batch op %d failed: %s", idx, exc)
            with self._lock:
                self._results[idx].status = "error"
                self._results[idx].error = str(exc)
                self._results[idx].message = f"Error: {exc}"
                self._results[idx].finished_at = time.time()

        self._report_combined_progress()

    def _report_combined_progress(self):
        """Push combined progress to parent job and/or callback."""
        pct = self.combined_progress

        # Update parent job if we have one
        if self.job_id:
            try:
                from opencut.jobs import _update_job
                done = sum(
                    1 for r in self._results
                    if r.status in ("complete", "error", "skipped")
                )
                total = len(self._results)
                _update_job(
                    self.job_id,
                    progress=pct,
                    message=f"Parallel batch: {done}/{total} operations complete",
                )
            except Exception:
                pass  # best-effort

        # Invoke user callback
        if self.on_progress:
            try:
                done = sum(
                    1 for r in self._results
                    if r.status in ("complete", "error", "skipped")
                )
                total = len(self._results)
                self.on_progress(
                    pct, f"Parallel batch: {done}/{total} operations complete"
                )
            except Exception:
                pass
