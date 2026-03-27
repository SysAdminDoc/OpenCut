"""
OpenCut Batch Processing Module v0.7.2

Multi-file batch processing queue:
- Queue multiple files for the same operation
- Parallel or sequential processing
- Per-file progress tracking
- Aggregate results reporting

Works with any existing OpenCut operation endpoint.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class BatchItem:
    """A single file in a batch queue."""
    filepath: str
    status: str = "queued"       # queued, running, complete, error, skipped
    progress: int = 0
    message: str = "Queued"
    output_path: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.started_at == 0:
            return 0
        end = self.finished_at if self.finished_at > 0 else time.time()
        return end - self.started_at


@dataclass
class BatchJob:
    """A batch processing job managing multiple files."""
    batch_id: str
    operation: str               # e.g. "silence", "denoise", "stabilize"
    items: List[BatchItem] = field(default_factory=list)
    params: Dict = field(default_factory=dict)
    status: str = "running"      # running, complete, cancelled, error
    created_at: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def completed(self) -> int:
        return sum(1 for i in self.items if i.status in ("complete", "error", "skipped"))

    @property
    def progress(self) -> int:
        if not self.items:
            return 0
        return min(100, int((self.completed / self.total) * 100))

    @property
    def summary(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "operation": self.operation,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "progress": self.progress,
            "items": [
                {
                    "filepath": i.filepath,
                    "filename": os.path.basename(i.filepath),
                    "status": i.status,
                    "progress": i.progress,
                    "message": i.message,
                    "output_path": i.output_path,
                    "error": i.error,
                    "elapsed": round(i.elapsed, 1),
                }
                for i in self.items
            ],
            "results": {
                "success": sum(1 for i in self.items if i.status == "complete"),
                "failed": sum(1 for i in self.items if i.status == "error"),
                "skipped": sum(1 for i in self.items if i.status == "skipped"),
            },
        }


# Global batch registry
_batches: Dict[str, BatchJob] = {}
_batch_lock = threading.Lock()
_BATCH_MAX_AGE = 3600  # Auto-clean batches older than 1 hour
_BATCH_MAX_COUNT = 200  # Hard cap on stored batches
_BATCH_CLEANUP_INTERVAL = 300  # 5 minutes
_batch_cleanup_started = False
_batch_cleanup_start_lock = threading.Lock()


def _start_periodic_batch_cleanup():
    """Lazily start a daemon thread that cleans old batches every 5 minutes.

    Called on first batch creation.  Idempotent and daemon so it won't block
    shutdown or run during tests that never create batches.
    """
    global _batch_cleanup_started
    with _batch_cleanup_start_lock:
        if _batch_cleanup_started:
            return
        _batch_cleanup_started = True

    def _loop():
        while True:
            time.sleep(_BATCH_CLEANUP_INTERVAL)
            try:
                with _batch_lock:
                    _cleanup_old_batches()
            except Exception as e:
                logger.debug("Periodic batch cleanup error: %s", e)

    t = threading.Thread(target=_loop, daemon=True, name="opencut-batch-cleanup")
    t.start()
    logger.debug("Started periodic batch cleanup thread (every %ds)", _BATCH_CLEANUP_INTERVAL)


def _cleanup_old_batches():
    """Remove finished batches older than _BATCH_MAX_AGE. Must hold _batch_lock."""
    now = time.time()
    expired = [
        bid for bid, b in _batches.items()
        if b.status not in ("running",) and (now - b.created_at) > _BATCH_MAX_AGE
    ]
    for bid in expired:
        del _batches[bid]
    # Hard cap: if still over limit, remove oldest finished batches
    if len(_batches) > _BATCH_MAX_COUNT:
        finished = sorted(
            [(bid, b) for bid, b in _batches.items() if b.status != "running"],
            key=lambda x: x[1].created_at,
        )
        to_remove = len(_batches) - _BATCH_MAX_COUNT
        for bid, _ in finished[:to_remove]:
            del _batches[bid]


def create_batch(
    batch_id: str,
    operation: str,
    filepaths: List[str],
    params: Dict = None,
) -> BatchJob:
    """Create a new batch job."""
    items = []
    for fp in filepaths:
        if os.path.isfile(fp):
            items.append(BatchItem(filepath=fp))
        else:
            items.append(BatchItem(filepath=fp, status="skipped",
                                   message=f"File not found: {fp}"))

    batch = BatchJob(
        batch_id=batch_id,
        operation=operation,
        items=items,
        params=params or {},
    )

    with _batch_lock:
        _cleanup_old_batches()
        _batches[batch_id] = batch

    _start_periodic_batch_cleanup()
    return batch


def get_batch(batch_id: str) -> Optional[BatchJob]:
    """Get batch job by ID."""
    with _batch_lock:
        return _batches.get(batch_id)


def cancel_batch(batch_id: str) -> bool:
    """Cancel a running batch."""
    with _batch_lock:
        batch = _batches.get(batch_id)
        if batch:
            batch.status = "cancelled"
            for item in batch.items:
                if item.status == "queued":
                    item.status = "skipped"
                    item.message = "Cancelled"
            return True
    return False


def update_batch_item(batch_id: str, index: int, **kwargs):
    """Update a specific item in a batch."""
    with _batch_lock:
        batch = _batches.get(batch_id)
        if batch and 0 <= index < len(batch.items):
            item = batch.items[index]
            for k, v in kwargs.items():
                if hasattr(item, k):
                    setattr(item, k, v)


def finalize_batch(batch_id: str):
    """Mark batch as complete."""
    with _batch_lock:
        batch = _batches.get(batch_id)
        if batch:
            has_errors = any(i.status == "error" for i in batch.items)
            batch.status = "complete" if not has_errors else "complete_with_errors"


def list_batches(limit: int = 20) -> List[Dict]:
    """List recent batches."""
    with _batch_lock:
        sorted_batches = sorted(
            _batches.values(),
            key=lambda b: b.created_at,
            reverse=True,
        )[:limit]
        return [b.summary for b in sorted_batches]


# ---------------------------------------------------------------------------
# Parallel Batch Processing
# ---------------------------------------------------------------------------

# GPU-bound operations need limited concurrency to prevent OOM
GPU_OPERATIONS = frozenset({
    "upscale", "rembg", "denoise-ai", "face-enhance",
    "face-swap", "style-transfer", "interpolate",
})


def process_batch_parallel(
    batch_id: str,
    items: List,
    operation: str,
    params: Dict,
    on_item_complete: Optional[Callable] = None,
    on_item_error: Optional[Callable] = None,
) -> Dict:
    """Process batch items in parallel using ThreadPoolExecutor.

    Args:
        batch_id: batch identifier
        items: list of file paths or item dicts
        operation: operation name string
        params: shared parameters dict
        on_item_complete: callback(item_index, result_dict)
        on_item_error: callback(item_index, error_str)

    Returns:
        dict with keys: completed (int), failed (int), results (list), errors (list)
    """
    # Determine worker count based on operation type
    if operation in GPU_OPERATIONS:
        max_workers = 2  # Prevent GPU OOM
    else:
        max_workers = min(os.cpu_count() or 4, len(items))

    # Ensure at least 1 worker
    max_workers = max(1, max_workers)

    results: List[Optional[Dict]] = [None] * len(items)
    errors: List[Optional[str]] = [None] * len(items)
    completed = 0
    failed = 0
    _progress_lock = threading.Lock()

    def _run_item(idx: int, item) -> tuple:
        """Execute a single item; returns (index, result_or_None, error_or_None)."""
        filepath = item if isinstance(item, str) else item.get("filepath", item)
        try:
            # Lazy import to avoid circular dependency
            from opencut.routes.video_core import _execute_batch_item

            def _item_progress(pct, msg=""):
                update_batch_item(batch_id, idx, progress=pct, message=msg)

            update_batch_item(
                batch_id, idx,
                status="running", started_at=time.time(),
                message="Processing...",
            )

            result = _execute_batch_item(operation, filepath, params, _item_progress)

            update_batch_item(
                batch_id, idx,
                status="complete", progress=100,
                output_path=result, message="Done",
                finished_at=time.time(),
            )
            return (idx, {"output_path": result}, None)
        except Exception as exc:
            err_str = str(exc)
            update_batch_item(
                batch_id, idx,
                status="error", error=err_str,
                message=f"Error: {exc}",
                finished_at=time.time(),
            )
            return (idx, None, err_str)

    logger.info(
        "Batch %s: starting parallel processing (%d items, %d workers, op=%s)",
        batch_id, len(items), max_workers, operation,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Check for cancelled batch before submitting
        batch = get_batch(batch_id)
        if batch and batch.status == "cancelled":
            return {"completed": 0, "failed": 0, "results": results, "errors": errors}

        futures = {}
        for idx, item in enumerate(items):
            # Skip items already marked (e.g. file-not-found)
            batch = get_batch(batch_id)
            if batch and batch.items[idx].status == "skipped":
                continue
            # Check for cancellation before each submission
            if batch and batch.status == "cancelled":
                break
            fut = executor.submit(_run_item, idx, item)
            futures[fut] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                _idx, result_dict, err = future.result()
            except Exception as exc:
                # Should not happen since _run_item catches all exceptions,
                # but guard against thread-pool level failures
                err = str(exc)
                result_dict = None
                update_batch_item(
                    batch_id, idx,
                    status="error", error=err,
                    message=f"Error: {exc}",
                    finished_at=time.time(),
                )

            with _progress_lock:
                if err:
                    errors[idx] = err
                    failed += 1
                    if on_item_error:
                        try:
                            on_item_error(idx, err)
                        except Exception:
                            pass
                else:
                    results[idx] = result_dict
                    completed += 1
                    if on_item_complete:
                        try:
                            on_item_complete(idx, result_dict)
                        except Exception:
                            pass

    finalize_batch(batch_id)

    logger.info(
        "Batch %s: parallel processing done — %d completed, %d failed",
        batch_id, completed, failed,
    )

    return {
        "completed": completed,
        "failed": failed,
        "results": results,
        "errors": errors,
    }
