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
        return int((self.completed / self.total) * 100)

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
        _batches[batch_id] = batch

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
