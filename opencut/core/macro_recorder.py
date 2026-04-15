"""
OpenCut Macro Recording & Playback

Record API calls during a session as an ordered list.  Playback
executes each call, substituting the target file.  Save/load macros
as JSON files.  Supports variable substitution in payloads.
"""

import copy
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_MACROS_DIR = os.path.join(_OPENCUT_DIR, "macros")

_lock = threading.Lock()
# session_id -> MacroSession
_sessions: Dict[str, "MacroSession"] = {}

# Variable patterns recognized during playback substitution
_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MacroStep:
    """A single recorded API action."""
    endpoint: str
    method: str = "POST"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    order: int = 0


@dataclass
class MacroRecording:
    """A complete macro with metadata and step list."""
    name: str = "Untitled Macro"
    description: str = ""
    steps: List[MacroStep] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert macro to a JSON-serializable dict."""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [asdict(s) for s in self.steps],
            "created": self.created,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MacroRecording":
        """Create a MacroRecording from a dict (e.g., loaded from JSON)."""
        steps = [
            MacroStep(
                endpoint=s["endpoint"],
                method=s.get("method", "POST"),
                payload=s.get("payload", s.get("params", {})),
                timestamp=s.get("timestamp", 0),
                order=s.get("order", i),
            )
            for i, s in enumerate(data.get("steps", data.get("actions", [])))
        ]
        return cls(
            name=data.get("name", "Untitled Macro"),
            description=data.get("description", ""),
            steps=steps,
            created=data.get("created", time.time()),
            version=data.get("version", 1),
        )


# Legacy aliases for backward compatibility
MacroAction = MacroStep
Macro = MacroRecording


@dataclass
class MacroSession:
    """Active recording session."""
    recording: bool = False
    steps: List[MacroStep] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    # Legacy alias
    @property
    def actions(self) -> List[MacroStep]:
        return self.steps


# ---------------------------------------------------------------------------
# Variable substitution
# ---------------------------------------------------------------------------

def _substitute_vars(
    payload: Dict[str, Any],
    variables: Dict[str, str],
) -> Dict[str, Any]:
    """Recursively substitute ``${var}`` placeholders in payload values.

    Recognized variables:
        ``${input_file}``  - input file path
        ``${output_dir}``  - output directory
        ``${timestamp}``   - current ISO timestamp

    Custom variables can be passed via *variables* dict.

    Args:
        payload: The payload dict (deep-copied internally).
        variables: Mapping of variable names to replacement values.

    Returns:
        New dict with variables substituted.
    """
    result = copy.deepcopy(payload)

    def _replace(value):
        if isinstance(value, str):
            def _repl(m):
                var_name = m.group(1)
                return variables.get(var_name, m.group(0))
            return _VAR_PATTERN.sub(_repl, value)
        if isinstance(value, dict):
            return {k: _replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_replace(item) for item in value]
        return value

    return _replace(result)


# ---------------------------------------------------------------------------
# Recording API
# ---------------------------------------------------------------------------

def start_recording(
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Start recording API calls for a session.

    Args:
        session_id: Session scope identifier.
        on_progress: Optional progress callback (int).

    Returns:
        Dict with 'recording' status and session info.
    """
    if on_progress:
        on_progress(50)

    with _lock:
        session = _sessions.get(session_id)
        if session and session.recording:
            return {
                "recording": True,
                "message": "Already recording",
                "step_count": len(session.steps),
            }

        _sessions[session_id] = MacroSession(
            recording=True,
            steps=[],
            started=time.time(),
        )

    if on_progress:
        on_progress(100)

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
) -> MacroRecording:
    """Stop recording and return the captured macro.

    Args:
        session_id: Session scope identifier.
        name: Name for the resulting macro.
        description: Description for the macro.
        on_progress: Optional progress callback (int).

    Returns:
        MacroRecording object with all recorded steps.

    Raises:
        ValueError: If no recording is active for this session.
    """
    if on_progress:
        on_progress(50)

    with _lock:
        session = _sessions.get(session_id)
        if not session or not session.recording:
            raise ValueError(f"No active recording for session '{session_id}'")

        session.recording = False
        steps = list(session.steps)

    macro = MacroRecording(
        name=name,
        description=description,
        steps=steps,
        created=time.time(),
    )

    if on_progress:
        on_progress(100)

    logger.info(
        "Stopped macro recording for session '%s': %d steps captured",
        session_id, len(steps),
    )
    return macro


def add_step(
    endpoint: str,
    method: str = "POST",
    payload: Optional[Dict[str, Any]] = None,
    session_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> bool:
    """Record a single API step if recording is active.

    Args:
        endpoint: API endpoint path (e.g., '/api/silence').
        method: HTTP method (default POST).
        payload: Parameters dict for the API call.
        session_id: Session scope identifier.
        on_progress: Optional progress callback (int).

    Returns:
        True if the step was recorded, False if not recording.
    """
    with _lock:
        session = _sessions.get(session_id)
        if not session or not session.recording:
            return False

        step = MacroStep(
            endpoint=endpoint,
            method=method,
            payload=copy.deepcopy(payload or {}),
            order=len(session.steps),
        )
        session.steps.append(step)

    if on_progress:
        on_progress(100)

    logger.debug("Recorded step '%s' for session '%s'", endpoint, session_id)
    return True


# Legacy alias
record_action = add_step


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_macro(
    macro: MacroRecording,
    target_file: str = "",
    output_dir: str = "",
    executor: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Play back a macro, substituting variables.

    The executor callback is called for each step with (endpoint, method, payload)
    and should return a result dict.  If no executor is provided, steps
    are returned as a dry-run plan.

    Args:
        macro: MacroRecording to play back.
        target_file: File path to substitute as ``${input_file}``.
        output_dir: Output directory to substitute as ``${output_dir}``.
        executor: Optional callback(endpoint, method, payload) -> result dict.
        on_progress: Optional progress callback (int).

    Returns:
        List of result dicts, one per step.
    """
    if on_progress:
        on_progress(5)

    if not macro.steps:
        return []

    # Build variable map
    variables = {
        "input_file": target_file,
        "output_dir": output_dir or os.path.dirname(target_file),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    results = []
    total = len(macro.steps)

    for i, step in enumerate(macro.steps):
        if on_progress:
            pct = 5 + int(((i + 1) / total) * 90)
            on_progress(pct)

        # Substitute variables in payload
        payload = _substitute_vars(step.payload, variables)

        # Also substitute filepath-like params with target_file
        if target_file:
            for key in ("filepath", "file", "input_file", "video_path",
                        "input_path"):
                if key in payload:
                    payload[key] = target_file

        if executor:
            try:
                result = executor(step.endpoint, step.method, payload)
                results.append({
                    "step": i + 1,
                    "endpoint": step.endpoint,
                    "method": step.method,
                    "success": True,
                    "result": result,
                })
            except Exception as exc:
                results.append({
                    "step": i + 1,
                    "endpoint": step.endpoint,
                    "method": step.method,
                    "success": False,
                    "error": str(exc),
                })
        else:
            # Dry run
            results.append({
                "step": i + 1,
                "endpoint": step.endpoint,
                "method": step.method,
                "payload": payload,
                "dry_run": True,
            })

    if on_progress:
        on_progress(100)

    logger.info("Played macro '%s': %d steps", macro.name, total)
    return results


# ---------------------------------------------------------------------------
# Macro CRUD (file-based persistence)
# ---------------------------------------------------------------------------

def _ensure_macros_dir():
    """Create the macros directory if needed."""
    os.makedirs(_MACROS_DIR, exist_ok=True)


def _macro_path(name: str) -> str:
    """Return the file path for a named macro."""
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
    safe_name = safe_name.strip().replace(" ", "_")
    if not safe_name:
        safe_name = "untitled"
    return os.path.join(_MACROS_DIR, f"{safe_name}.opencut-macro")


def save_macro(
    macro: MacroRecording,
    path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Save a macro to a JSON file.

    Args:
        macro: MacroRecording to save.
        path: Optional custom file path.  Defaults to macros dir.
        on_progress: Optional progress callback (int).

    Returns:
        The path the macro was saved to.
    """
    if on_progress:
        on_progress(50)

    if path is None:
        _ensure_macros_dir()
        path = _macro_path(macro.name)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(macro.to_dict(), fh, indent=2)

    if on_progress:
        on_progress(100)

    logger.info("Saved macro '%s' (%d steps) to %s",
                macro.name, len(macro.steps), path)
    return path


def load_macro(
    path: str,
    on_progress: Optional[Callable] = None,
) -> MacroRecording:
    """Load a macro from a JSON file.

    Args:
        path: Path to the macro file.
        on_progress: Optional progress callback (int).

    Returns:
        MacroRecording loaded from the file.
    """
    if on_progress:
        on_progress(50)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Macro file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    macro = MacroRecording.from_dict(data)

    if on_progress:
        on_progress(100)

    logger.info("Loaded macro '%s' (%d steps) from %s",
                macro.name, len(macro.steps), path)
    return macro


def list_macros(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all saved macros.

    Returns:
        List of dicts with name, description, step_count, created, path.
    """
    if on_progress:
        on_progress(30)

    _ensure_macros_dir()
    macros = []

    for fname in sorted(os.listdir(_MACROS_DIR)):
        if fname.endswith(".opencut-macro"):
            fpath = os.path.join(_MACROS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                macros.append({
                    "name": data.get("name", fname),
                    "description": data.get("description", ""),
                    "step_count": len(data.get("steps",
                                               data.get("actions", []))),
                    "created": data.get("created", 0),
                    "path": fpath,
                })
            except (json.JSONDecodeError, OSError):
                continue

    if on_progress:
        on_progress(100)

    return macros


def delete_macro(
    name: str,
    on_progress: Optional[Callable] = None,
) -> bool:
    """Delete a saved macro by name.

    Args:
        name: Macro name.
        on_progress: Optional progress callback (int).

    Returns:
        True if deleted, False if not found.
    """
    if on_progress:
        on_progress(50)

    path = _macro_path(name)
    if os.path.isfile(path):
        os.unlink(path)
        logger.info("Deleted macro '%s' at %s", name, path)
        if on_progress:
            on_progress(100)
        return True

    # Fallback: scan for matching name in all files
    _ensure_macros_dir()
    for fname in os.listdir(_MACROS_DIR):
        if fname.endswith(".opencut-macro"):
            fpath = os.path.join(_MACROS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("name") == name:
                    os.unlink(fpath)
                    logger.info("Deleted macro '%s' at %s", name, fpath)
                    if on_progress:
                        on_progress(100)
                    return True
            except (json.JSONDecodeError, OSError):
                continue

    if on_progress:
        on_progress(100)
    return False


def export_macro(
    name: str,
    export_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export a macro to a specified path.

    Args:
        name: Macro name to export.
        export_path: Destination file path.
        on_progress: Optional progress callback (int).

    Returns:
        The export path.
    """
    path = _macro_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Macro '{name}' not found")

    macro = load_macro(path)
    return save_macro(macro, path=export_path, on_progress=on_progress)


def import_macro(
    import_path: str,
    on_progress: Optional[Callable] = None,
) -> MacroRecording:
    """Import a macro from an external file into the macros directory.

    Args:
        import_path: Source file path.
        on_progress: Optional progress callback (int).

    Returns:
        The imported MacroRecording.
    """
    macro = load_macro(import_path, on_progress=on_progress)
    save_macro(macro)
    logger.info("Imported macro '%s' from %s", macro.name, import_path)
    return macro
