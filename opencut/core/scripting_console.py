"""
OpenCut Python Scripting Console

Execute Python code in a sandboxed scope with core modules pre-imported.
Returns output as string.  Restricts dangerous imports and file system
access outside the project.  Stores execution history to disk.
"""

import io
import json
import logging
import math
import os
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_HISTORY_FILE = os.path.join(_OPENCUT_DIR, "console_history.json")
_MAX_HISTORY = 50

# Modules allowed in the sandbox
_ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "functools", "operator", "string", "textwrap",
    "copy", "pprint", "decimal", "fractions", "random", "hashlib",
    "base64", "uuid", "dataclasses", "enum", "typing", "time",
}

# Modules explicitly blocked (security risk)
_BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "glob",
    "socket", "http", "urllib", "requests", "ctypes", "importlib",
    "builtins", "code", "compile", "compileall", "py_compile",
    "signal", "multiprocessing", "threading", "asyncio",
    "pickle", "shelve", "marshal", "tempfile", "io",
    "webbrowser", "ftplib", "smtplib", "telnetlib",
}

# Builtins blocked in sandbox
_BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open",
    "breakpoint", "exit", "quit", "input",
}

# Dunder attribute patterns blocked in raw source
_BLOCKED_PATTERNS = (
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__code__", "__builtins__",
)

# Maximum output length (characters)
MAX_OUTPUT_LENGTH = 50_000

# Maximum execution time (seconds)
DEFAULT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# ScriptResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScriptResult:
    """Result from executing a script in the sandbox."""
    output: str = ""
    error: str = ""
    execution_time_ms: float = 0.0
    success: bool = True


# ---------------------------------------------------------------------------
# Safe import for sandbox
# ---------------------------------------------------------------------------

def _safe_import(name, *args, **kwargs):
    """Restricted import that only allows safe modules."""
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
    if top_level not in _ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed. "
            f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    if isinstance(__builtins__, dict):
        return __builtins__["__import__"](name, *args, **kwargs)
    return getattr(__builtins__, "__import__")(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# OpenCut namespace — safe wrappers exposed to user scripts
# ---------------------------------------------------------------------------

def _build_opencut_namespace() -> Dict[str, Any]:
    """Build the curated ``opencut`` namespace for the sandbox.

    Provides safe, read-only wrappers around core operations so user
    scripts can query information without mutating state or touching
    the filesystem directly.
    """
    ns: Dict[str, Any] = {}

    def _safe_get_video_info(filepath: str) -> dict:
        """Get video metadata (width, height, fps, duration)."""
        try:
            from opencut.helpers import get_video_info
            return get_video_info(filepath)
        except Exception as exc:
            return {"error": str(exc)}

    def _safe_detect_silences(
        filepath: str,
        threshold: float = -30.0,
        min_duration: float = 0.5,
    ) -> list:
        """Detect silent segments in an audio/video file."""
        try:
            from opencut.core.silence import detect_silences
            return detect_silences(filepath, threshold=threshold,
                                   min_duration=min_duration)
        except Exception as exc:
            return [{"error": str(exc)}]

    def _safe_generate_chapters(
        filepath: str,
        interval: float = 300.0,
    ) -> list:
        """Generate chapter markers at fixed intervals."""
        try:
            from opencut.core.chapter_gen import generate_chapters
            return generate_chapters(filepath, interval=interval)
        except Exception as exc:
            return [{"error": str(exc)}]

    def _safe_get_scenes(filepath: str, threshold: float = 0.3) -> list:
        """Detect scene changes in a video file."""
        try:
            from opencut.core.scene_detect import detect_scenes
            return detect_scenes(filepath, threshold=threshold)
        except Exception as exc:
            return [{"error": str(exc)}]

    def _safe_get_loudness(filepath: str) -> dict:
        """Analyze audio loudness (LUFS) of a file."""
        try:
            from opencut.core.audio import analyze_loudness
            return analyze_loudness(filepath)
        except Exception as exc:
            return {"error": str(exc)}

    ns["get_video_info"] = _safe_get_video_info
    ns["detect_silences"] = _safe_detect_silences
    ns["generate_chapters"] = _safe_generate_chapters
    ns["get_scenes"] = _safe_get_scenes
    ns["get_loudness"] = _safe_get_loudness

    return ns


# ---------------------------------------------------------------------------
# Sandbox creation
# ---------------------------------------------------------------------------

def create_sandbox(
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a sandboxed execution scope.

    The sandbox has:
    - Restricted builtins (no exec, eval, open, etc.)
    - Safe import function that blocks dangerous modules
    - Pre-populated context variables from the caller
    - Common safe modules pre-imported
    - ``opencut`` namespace with safe wrappers

    Args:
        context: Optional dict of variables to inject into the sandbox.

    Returns:
        Dict representing the sandbox globals for exec().
    """
    import collections as _collections
    import datetime as _datetime
    import json as _json
    import re as _re

    # Build safe builtins
    if isinstance(__builtins__, dict):
        safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in _BLOCKED_BUILTINS
        }
    else:
        safe_builtins = {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k not in _BLOCKED_BUILTINS and not k.startswith("_")
        }
        # Add essential dunder items
        for k in ("__name__", "__doc__"):
            if hasattr(__builtins__, k):
                safe_builtins[k] = getattr(__builtins__, k)

    safe_builtins["__import__"] = _safe_import

    sandbox = {
        "__builtins__": safe_builtins,
        # Pre-imported safe modules
        "math": math,
        "json": _json,
        "re": _re,
        "datetime": _datetime,
        "collections": _collections,
        # Convenience math functions at top level
        "sqrt": math.sqrt,
        "ceil": math.ceil,
        "floor": math.floor,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
        # String/JSON helpers
        "dumps": _json.dumps,
        "loads": _json.loads,
        # OpenCut namespace
        "opencut": type("opencut", (), _build_opencut_namespace())(),
    }

    # Inject caller context
    if context:
        for key, value in context.items():
            if not key.startswith("_"):
                sandbox[key] = value

    return sandbox


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------

def execute_script(
    code: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    on_progress: Optional[Callable] = None,
) -> ScriptResult:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python source code to execute.
        context: Optional dict of variables available to the script.
        timeout: Maximum execution time in seconds (default 30).
        on_progress: Optional progress callback.

    Returns:
        ScriptResult with output, error, execution_time_ms, and success.
    """
    if on_progress:
        on_progress(10)

    if not code or not code.strip():
        return ScriptResult(output="", success=True)

    # Basic security check on raw source
    code_lower = code.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern in code_lower:
            result = ScriptResult(
                output="",
                success=False,
                error=f"Use of '{pattern}' is not allowed in the sandbox",
            )
            _append_history(code, result)
            return result

    sandbox = create_sandbox(context)

    if on_progress:
        on_progress(30)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    start_time = time.monotonic()
    result = ScriptResult()

    # Execute in a thread with timeout enforcement
    exec_error: List[Optional[BaseException]] = [None]
    exec_done = threading.Event()

    def _run():
        try:
            compiled = compile(code, "<opencut_script>", "exec")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, sandbox)  # noqa: S102
        except Exception as exc:
            exec_error[0] = exc
        finally:
            exec_done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    finished = exec_done.wait(timeout=timeout)
    elapsed_ms = (time.monotonic() - start_time) * 1000.0

    if not finished:
        result.success = False
        result.error = f"Script execution timed out after {timeout}s"
        result.execution_time_ms = elapsed_ms
        _append_history(code, result)
        return result

    if on_progress:
        on_progress(80)

    exc = exec_error[0]
    stdout_text = stdout_capture.getvalue()
    stderr_text = stderr_capture.getvalue()

    output = stdout_text
    if stderr_text:
        output += "\n[stderr]\n" + stderr_text

    # Truncate if too long
    if len(output) > MAX_OUTPUT_LENGTH:
        output = output[:MAX_OUTPUT_LENGTH] + "\n... [output truncated]"

    result.output = output
    result.execution_time_ms = elapsed_ms

    if exc is not None:
        result.success = False
        if isinstance(exc, SyntaxError):
            result.error = f"Syntax error at line {exc.lineno}: {exc.msg}"
        elif isinstance(exc, ImportError):
            result.error = str(exc)
        else:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            clean_lines = []
            for line in "".join(tb).split("\n"):
                if "<opencut_script>" in line or not line.strip().startswith("File"):
                    clean_lines.append(line)
            result.error = "\n".join(clean_lines).strip()
    else:
        result.success = True

    if on_progress:
        on_progress(100)

    _append_history(code, result)
    return result


# ---------------------------------------------------------------------------
# Namespace introspection
# ---------------------------------------------------------------------------

def get_available_functions() -> List[Dict[str, str]]:
    """Return the list of functions available in the ``opencut`` namespace.

    Returns:
        List of dicts with ``name`` and ``doc`` keys.
    """
    ns = _build_opencut_namespace()
    funcs = []
    for name, obj in sorted(ns.items()):
        if callable(obj):
            funcs.append({
                "name": f"opencut.{name}",
                "doc": (obj.__doc__ or "").strip(),
            })
    return funcs


def get_available_modules() -> List[str]:
    """Return the list of modules available in the sandbox.

    Returns:
        Sorted list of allowed module names.
    """
    return sorted(_ALLOWED_MODULES)


def get_namespace_info() -> Dict[str, Any]:
    """Return full sandbox namespace documentation.

    Returns:
        Dict with modules, functions, builtins, and math helpers.
    """
    return {
        "modules": get_available_modules(),
        "functions": get_available_functions(),
        "math_helpers": [
            "sqrt", "ceil", "floor", "log", "sin", "cos", "pi", "e",
        ],
        "json_helpers": ["dumps", "loads"],
        "blocked_builtins": sorted(_BLOCKED_BUILTINS),
        "blocked_modules": sorted(_BLOCKED_MODULES),
    }


# ---------------------------------------------------------------------------
# Execution history
# ---------------------------------------------------------------------------

def _ensure_dir():
    """Create the OpenCut user directory if needed."""
    os.makedirs(_OPENCUT_DIR, exist_ok=True)


def _load_history() -> List[Dict[str, Any]]:
    """Load execution history from disk."""
    if not os.path.isfile(_HISTORY_FILE):
        return []
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(history: List[Dict[str, Any]]) -> None:
    """Persist execution history to disk."""
    _ensure_dir()
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(history[-_MAX_HISTORY:], fh, indent=2)
    except OSError as exc:
        logger.warning("Failed to save console history: %s", exc)


def _append_history(code: str, result: ScriptResult) -> None:
    """Append an execution record to history."""
    history = _load_history()
    history.append({
        "code": code[:2000],
        "output": result.output[:1000],
        "error": result.error[:1000] if result.error else "",
        "success": result.success,
        "execution_time_ms": round(result.execution_time_ms, 2),
        "timestamp": time.time(),
    })
    # Keep only the last N entries
    _save_history(history[-_MAX_HISTORY:])


def get_history(limit: int = _MAX_HISTORY) -> List[Dict[str, Any]]:
    """Return the last *limit* execution history entries.

    Args:
        limit: Maximum entries to return (default 50).

    Returns:
        List of history dicts, newest last.
    """
    history = _load_history()
    return history[-limit:]


def clear_history() -> None:
    """Clear execution history."""
    _save_history([])
    logger.info("Cleared scripting console history")
