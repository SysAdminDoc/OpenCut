"""
OpenCut Macro Recording & Playback

Record API calls during a session as an ordered list.  Playback
executes each call, substituting the target file.  Save/load macros
as JSON files.
"""

import copy
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_lock = threading.Lock()
# session_id -> MacroSession
_sessions: Dict[str, "MacroSession"] = {}


@dataclass
class MacroAction:
    """A single recorded API action."""
    endpoint: str
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    order: int = 0


@dataclass
class Macro:
    """A complete macro with metadata and action list."""
    name: str = "Untitled Macro"
    description: str = ""
    actions: List[MacroAction] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert macro to a JSON-serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "actions": [asdict(a) for a in self.actions],
            "created": self.created,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Macro":
        """Create a Macro from a dict (e.g., loaded from JSON)."""
        actions = [
            MacroAction(
                endpoint=a["endpoint"],
                params=a.get("params", {}),
                timestamp=a.get("timestamp", 0),
                order=a.get("order", i),
            )
            for i, a in enumerate(data.get("actions", []))
        ]
        return cls(
            name=data.get("name", "Untitled Macro"),
            description=data.get("description", ""),
            actions=actions,
            created=data.get("created", time.time()),
            version=data.get("version", 1),
        )


@dataclass
class MacroSession:
    """Active recording session."""
    recording: bool = False
    actions: List[MacroAction] = field(default_factory=list)
    started: float = field(default_factory=time.time)


def start_recording(
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Start recording API calls for a session.

    Args:
        session_id: Session scope identifier.
        on_progress: Optional progress callback.

    Returns:
        Dict with 'recording' status and session info.
    """
    if on_progress:
        on_progress(50, "Starting macro recording...")

    with _lock:
        session = _sessions.get(session_id)
        if session and session.recording:
            return {
                "recording": True,
                "message": "Already recording",
                "action_count": len(session.actions),
            }

        _sessions[session_id] = MacroSession(
            recording=True,
            actions=[],
            started=time.time(),
        )

    if on_progress:
        on_progress(100, "Recording started")

    logger.info("Started macro recording for session '%s'", session_id)
    return {
        "recording": True,
        "message": "Recording started",
        "session_id": session_id,
    }


def stop_recording(
    session_id: str = "default",
    name: str = "Untitled Macro",
    description: str = "",
    on_progress: Optional[Callable] = None,
) -> Macro:
    """Stop recording and return the captured macro.

    Args:
        session_id: Session scope identifier.
        name: Name for the resulting macro.
        description: Description for the macro.
        on_progress: Optional progress callback.

    Returns:
        Macro object with all recorded actions.

    Raises:
        ValueError: If no recording is active for this session.
    """
    if on_progress:
        on_progress(50, "Stopping recording...")

    with _lock:
        session = _sessions.get(session_id)
        if not session or not session.recording:
            raise ValueError(f"No active recording for session '{session_id}'")

        session.recording = False
        actions = list(session.actions)

    macro = Macro(
        name=name,
        description=description,
        actions=actions,
        created=time.time(),
    )

    if on_progress:
        on_progress(100, "Recording stopped")

    logger.info(
        "Stopped macro recording for session '%s': %d actions captured",
        session_id, len(actions),
    )
    return macro


def record_action(
    endpoint: str,
    params: Dict[str, Any],
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> bool:
    """Record a single API action if recording is active.

    Args:
        endpoint: API endpoint path (e.g., '/silence').
        params: Parameters dict for the API call.
        session_id: Session scope identifier.
        on_progress: Optional progress callback.

    Returns:
        True if the action was recorded, False if not recording.
    """
    with _lock:
        session = _sessions.get(session_id)
        if not session or not session.recording:
            return False

        action = MacroAction(
            endpoint=endpoint,
            params=copy.deepcopy(params),
            order=len(session.actions),
        )
        session.actions.append(action)

    if on_progress:
        on_progress(100, f"Recorded action: {endpoint}")

    logger.debug("Recorded action '%s' for session '%s'", endpoint, session_id)
    return True


def play_macro(
    macro: Macro,
    target_file: str,
    executor: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Play back a macro, substituting the target file.

    The executor callback is called for each action with (endpoint, params)
    and should return a result dict.  If no executor is provided, actions
    are returned as a dry-run plan.

    Args:
        macro: Macro object to play back.
        target_file: File path to substitute in each action's filepath param.
        executor: Optional callback(endpoint, params) -> result dict.
        on_progress: Optional progress callback.

    Returns:
        List of result dicts, one per action.
    """
    if on_progress:
        on_progress(5, "Preparing macro playback...")

    if not macro.actions:
        return []

    results = []
    total = len(macro.actions)

    for i, action in enumerate(macro.actions):
        if on_progress:
            pct = 5 + int(((i + 1) / total) * 90)
            on_progress(pct, f"Executing step {i + 1}/{total}: {action.endpoint}")

        # Substitute target file in params
        params = copy.deepcopy(action.params)
        # Replace filepath-like parameters
        for key in ("filepath", "file", "input_file", "video_path", "input_path"):
            if key in params:
                params[key] = target_file

        if executor:
            try:
                result = executor(action.endpoint, params)
                results.append({
                    "step": i + 1,
                    "endpoint": action.endpoint,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "step": i + 1,
                    "endpoint": action.endpoint,
                    "success": False,
                    "error": str(e),
                })
        else:
            # Dry run
            results.append({
                "step": i + 1,
                "endpoint": action.endpoint,
                "params": params,
                "dry_run": True,
            })

    if on_progress:
        on_progress(100, "Macro playback complete")

    logger.info("Played macro '%s': %d steps", macro.name, total)
    return results


def save_macro(
    macro: Macro,
    path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Save a macro to a JSON file.

    Args:
        macro: Macro object to save.
        path: File path for the output JSON.
        on_progress: Optional progress callback.

    Returns:
        The path the macro was saved to.
    """
    if on_progress:
        on_progress(50, "Saving macro...")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(macro.to_dict(), f, indent=2)

    if on_progress:
        on_progress(100, "Macro saved")

    logger.info("Saved macro '%s' (%d actions) to %s", macro.name, len(macro.actions), path)
    return path


def load_macro(
    path: str,
    on_progress: Optional[Callable] = None,
) -> Macro:
    """Load a macro from a JSON file.

    Args:
        path: Path to the macro JSON file.
        on_progress: Optional progress callback.

    Returns:
        Macro object loaded from the file.
    """
    if on_progress:
        on_progress(50, "Loading macro...")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Macro file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    macro = Macro.from_dict(data)

    if on_progress:
        on_progress(100, "Macro loaded")

    logger.info("Loaded macro '%s' (%d actions) from %s", macro.name, len(macro.actions), path)
    return macro
