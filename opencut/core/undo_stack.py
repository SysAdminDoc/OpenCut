"""
OpenCut Undo Stack / Operation History

Track operations performed during a session (input file, output file,
parameters, timestamp).  Undo reverts to the previous output file.
Session-scoped list capped at 20 entries.
"""

import copy
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

MAX_HISTORY = 20

_lock = threading.Lock()
# session_id -> list[OperationRecord]
_sessions: Dict[str, List["OperationRecord"]] = {}


@dataclass
class OperationRecord:
    """A single tracked operation."""
    operation: str
    input_file: str
    output_file: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    undone: bool = False


def push_operation(
    op_data: Dict[str, Any],
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> OperationRecord:
    """Push an operation onto the undo stack.

    Args:
        op_data: Dict with keys 'operation', 'input_file', 'output_file',
                 and optionally 'parameters'.
        session_id: Session scope identifier.
        on_progress: Optional progress callback.

    Returns:
        The created OperationRecord.
    """
    if on_progress:
        on_progress(10, "Recording operation...")

    operation = op_data.get("operation", "unknown")
    input_file = op_data.get("input_file", "")
    output_file = op_data.get("output_file", "")
    parameters = op_data.get("parameters", {})

    if not operation:
        raise ValueError("Operation name is required")

    record = OperationRecord(
        operation=operation,
        input_file=input_file,
        output_file=output_file,
        parameters=copy.deepcopy(parameters),
    )

    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = []
        stack = _sessions[session_id]
        stack.append(record)
        # Enforce limit
        if len(stack) > MAX_HISTORY:
            _sessions[session_id] = stack[-MAX_HISTORY:]

    if on_progress:
        on_progress(100, "Operation recorded")

    logger.debug("Pushed operation '%s' to session '%s'", operation, session_id)
    return record


def undo_last(
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Undo the last operation by marking it undone and returning the revert info.

    The undo returns the input_file from the undone operation so the caller
    can revert to it.

    Args:
        session_id: Session scope identifier.
        on_progress: Optional progress callback.

    Returns:
        Dict with 'reverted_operation', 'revert_to_file', and 'undone_record',
        or None if nothing to undo.
    """
    if on_progress:
        on_progress(10, "Looking for operation to undo...")

    with _lock:
        stack = _sessions.get(session_id, [])
        # Find the last non-undone operation
        target = None
        for i in range(len(stack) - 1, -1, -1):
            if not stack[i].undone:
                target = stack[i]
                break

        if target is None:
            logger.debug("Nothing to undo in session '%s'", session_id)
            return None

        target.undone = True

    if on_progress:
        on_progress(100, "Undo complete")

    logger.debug("Undid operation '%s' in session '%s'", target.operation, session_id)
    return {
        "reverted_operation": target.operation,
        "revert_to_file": target.input_file,
        "undone_record": asdict(target),
    }


def get_history(
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Get the full operation history for a session.

    Args:
        session_id: Session scope identifier.
        on_progress: Optional progress callback.

    Returns:
        List of operation records as dicts, newest last.
    """
    if on_progress:
        on_progress(50, "Retrieving history...")

    with _lock:
        stack = _sessions.get(session_id, [])
        result = [asdict(r) for r in stack]

    if on_progress:
        on_progress(100, "History retrieved")

    return result


def clear_history(
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> int:
    """Clear the operation history for a session.

    Args:
        session_id: Session scope identifier.
        on_progress: Optional progress callback.

    Returns:
        Number of records cleared.
    """
    if on_progress:
        on_progress(50, "Clearing history...")

    with _lock:
        stack = _sessions.pop(session_id, [])
        count = len(stack)

    if on_progress:
        on_progress(100, "History cleared")

    logger.debug("Cleared %d operations from session '%s'", count, session_id)
    return count
