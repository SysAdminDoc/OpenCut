"""
OpenCut Render Queue with Priority

Sequential background render queue supporting priority ordering,
pause/resume, per-item progress, and OS process priority lowering.
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR, _ensure_opencut_dir

logger = logging.getLogger("opencut")

_QUEUE_PATH = os.path.join(OPENCUT_DIR, "render_queue.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RenderQueueItem:
    """A single item in the render queue."""
    id: str
    input_path: str
    preset_name: str
    params: Optional[Dict] = None
    priority: int = 3  # 1 (lowest) to 5 (highest)
    status: str = "pending"  # pending, rendering, complete, error, cancelled
    progress: int = 0
    output_path: str = ""
    created_at: float = field(default_factory=time.time)
    error: str = ""


# ---------------------------------------------------------------------------
# Queue state
# ---------------------------------------------------------------------------
_queue: List[RenderQueueItem] = []
_queue_lock = threading.Lock()
_queue_paused = threading.Event()
_queue_paused.set()  # starts "not paused" (set = running allowed)
_queue_worker: Optional[threading.Thread] = None
_queue_running = threading.Event()  # set when worker loop is active
_on_item_complete_cb: Optional[Callable] = None
_on_queue_complete_cb: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save_queue():
    """Persist queue to disk. Caller must hold _queue_lock."""
    _ensure_opencut_dir()
    data = [asdict(item) for item in _queue]
    try:
        with open(_QUEUE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        logger.debug("Failed to save render queue: %s", e)


def _load_queue():
    """Load queue from disk. Caller must hold _queue_lock."""
    global _queue
    try:
        with open(_QUEUE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _queue = []
        for item in data:
            _queue.append(RenderQueueItem(
                id=item.get("id", uuid.uuid4().hex[:12]),
                input_path=item.get("input_path", ""),
                preset_name=item.get("preset_name", ""),
                params=item.get("params"),
                priority=item.get("priority", 3),
                status=item.get("status", "pending"),
                progress=item.get("progress", 0),
                output_path=item.get("output_path", ""),
                created_at=item.get("created_at", time.time()),
                error=item.get("error", ""),
            ))
    except (FileNotFoundError, json.JSONDecodeError):
        _queue = []


# Ensure queue is loaded on import
with _queue_lock:
    _load_queue()


# ---------------------------------------------------------------------------
# OS priority
# ---------------------------------------------------------------------------
def set_process_priority(low: bool = True):
    """Lower (or restore) the current process priority for background renders.

    On Windows uses BELOW_NORMAL_PRIORITY_CLASS; on Unix uses nice(10).
    """
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            # BELOW_NORMAL = 0x4000, NORMAL = 0x20
            priority_class = 0x4000 if low else 0x0020
            kernel32.SetPriorityClass(handle, priority_class)
            logger.debug("Set process priority class to %s", "below_normal" if low else "normal")
        else:
            os.nice(10 if low else 0)
            logger.debug("Set process nice to %s", 10 if low else 0)
    except Exception as e:
        logger.debug("Failed to set process priority: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_to_queue(input_path: str, preset_name: str,
                 params: Optional[Dict] = None, priority: int = 3) -> str:
    """Add a render job to the queue.

    Args:
        input_path: Path to input media file.
        preset_name: Export preset name.
        params: Optional additional parameters.
        priority: 1 (lowest) to 5 (highest).

    Returns:
        Item ID string.
    """
    priority = max(1, min(5, priority))
    item_id = uuid.uuid4().hex[:12]
    item = RenderQueueItem(
        id=item_id,
        input_path=input_path,
        preset_name=preset_name,
        params=params,
        priority=priority,
    )
    with _queue_lock:
        _queue.append(item)
        _save_queue()
    logger.info("Render queue: added item %s (%s, priority %d)",
                item_id, preset_name, priority)
    return item_id


def remove_from_queue(item_id: str) -> bool:
    """Remove an item from the queue. Cannot remove items currently rendering.

    Returns:
        True if removed, False if not found or currently rendering.
    """
    with _queue_lock:
        for i, item in enumerate(_queue):
            if item.id == item_id:
                if item.status == "rendering":
                    return False
                _queue.pop(i)
                _save_queue()
                logger.info("Render queue: removed item %s", item_id)
                return True
    return False


def reorder_queue(item_id: str, new_priority: int) -> None:
    """Change the priority of a queued item.

    Args:
        item_id: ID of the item to reorder.
        new_priority: New priority (1-5).
    """
    new_priority = max(1, min(5, new_priority))
    with _queue_lock:
        for item in _queue:
            if item.id == item_id:
                item.priority = new_priority
                _save_queue()
                logger.info("Render queue: reordered item %s to priority %d",
                            item_id, new_priority)
                return
    raise ValueError(f"Item not found: {item_id}")


def get_queue() -> List[RenderQueueItem]:
    """Return a copy of all queue items, sorted by priority (highest first)."""
    with _queue_lock:
        items = list(_queue)
    items.sort(key=lambda x: (-x.priority, x.created_at))
    return items


def _get_next_pending() -> Optional[RenderQueueItem]:
    """Get the highest-priority pending item. Caller must hold _queue_lock."""
    pending = [item for item in _queue if item.status == "pending"]
    if not pending:
        return None
    pending.sort(key=lambda x: (-x.priority, x.created_at))
    return pending[0]


def _render_item(item: RenderQueueItem):
    """Render a single queue item using export_with_preset."""
    from opencut.core.export_presets import export_with_preset

    def _progress(pct, msg=""):
        with _queue_lock:
            item.progress = pct
            _save_queue()

    try:
        output = export_with_preset(
            input_path=item.input_path,
            preset_name=item.preset_name,
            output_dir=item.params.get("output_dir", "") if item.params else "",
            on_progress=_progress,
        )
        with _queue_lock:
            item.status = "complete"
            item.progress = 100
            item.output_path = output
            _save_queue()
        logger.info("Render queue: completed item %s -> %s", item.id, output)
    except Exception as e:
        with _queue_lock:
            item.status = "error"
            item.error = str(e)
            _save_queue()
        logger.error("Render queue: item %s failed: %s", item.id, e)


def _worker_loop():
    """Background worker that processes items sequentially."""
    set_process_priority(low=True)
    logger.info("Render queue worker started")

    try:
        while _queue_running.is_set():
            # Wait if paused
            _queue_paused.wait(timeout=1.0)
            if not _queue_running.is_set():
                break

            with _queue_lock:
                item = _get_next_pending()
                if item is None:
                    # Nothing to render -- exit worker
                    break
                item.status = "rendering"
                item.progress = 0
                _save_queue()

            _render_item(item)

            if _on_item_complete_cb:
                try:
                    _on_item_complete_cb(item)
                except Exception as e:
                    logger.debug("Render queue on_item_complete callback error: %s", e)
    finally:
        set_process_priority(low=False)
        _queue_running.clear()
        logger.info("Render queue worker stopped")

        if _on_queue_complete_cb:
            try:
                _on_queue_complete_cb()
            except Exception as e:
                logger.debug("Render queue on_queue_complete callback error: %s", e)


def start_queue_processing(on_item_complete: Optional[Callable] = None,
                           on_queue_complete: Optional[Callable] = None) -> None:
    """Start processing the render queue in a background thread.

    Args:
        on_item_complete: Optional callback ``(RenderQueueItem)`` per item.
        on_queue_complete: Optional callback when entire queue finishes.
    """
    global _queue_worker, _on_item_complete_cb, _on_queue_complete_cb

    if _queue_running.is_set():
        logger.debug("Render queue worker already running")
        return

    _on_item_complete_cb = on_item_complete
    _on_queue_complete_cb = on_queue_complete
    _queue_running.set()
    _queue_paused.set()  # ensure not paused

    _queue_worker = threading.Thread(
        target=_worker_loop,
        daemon=True,
        name="opencut-render-queue",
    )
    _queue_worker.start()


def pause_queue() -> None:
    """Pause queue processing. Current render continues, next won't start."""
    _queue_paused.clear()
    logger.info("Render queue paused")


def resume_queue() -> None:
    """Resume queue processing after pause."""
    _queue_paused.set()
    logger.info("Render queue resumed")
    # If worker died while paused, restart it
    if not _queue_running.is_set():
        with _queue_lock:
            has_pending = any(item.status == "pending" for item in _queue)
        if has_pending:
            start_queue_processing(
                on_item_complete=_on_item_complete_cb,
                on_queue_complete=_on_queue_complete_cb,
            )


def is_queue_paused() -> bool:
    """Check if the queue is currently paused."""
    return not _queue_paused.is_set()


def is_queue_running() -> bool:
    """Check if the queue worker is active."""
    return _queue_running.is_set()
