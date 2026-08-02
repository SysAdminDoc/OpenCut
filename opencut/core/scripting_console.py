"""
OpenCut Python Scripting Console

Execute Python code in a sandboxed scope with core modules pre-imported.
Returns output as string.  Restricts dangerous imports and file system
access outside the project.  Stores execution history to disk.
"""

import ast
import io
import json
import logging
import math
import os
import sys
import threading
import time
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_HISTORY_FILE = os.path.join(_OPENCUT_DIR, "console_history.json")
_MAX_HISTORY = 50
_history_lock = threading.RLock()
_execution_state = threading.local()

# Modules allowed in the sandbox
_ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "string", "textwrap",
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
    # These modules expose reflection helpers such as attrgetter/methodcaller
    # that can resolve dunder names supplied as runtime strings.
    "operator", "functools",
}

# Builtins blocked in sandbox
_BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open",
    "breakpoint", "exit", "quit", "input",
    "getattr", "setattr", "delattr",  # prevent dunder access bypass
    "globals", "locals", "vars",       # prevent scope inspection
}

# Dunder attribute patterns blocked in raw source
_BLOCKED_PATTERNS = (
    "__class__", "__subclasses__", "__bases__", "__mro__",
    "__globals__", "__code__", "__builtins__", "__dict__",
    "__import__", "__loader__", "__spec__",
    "__init__", "__new__", "__reduce__", "__getattribute__",
    "__module__", "__wrapped__", "__qualname__", "__self__",
    "__func__", "__closure__",
)


def _constant_string_value(node: ast.AST) -> Optional[str]:
    """Return a statically-known string value, if *node* has one.

    ``ast.Constant`` checks alone miss a dunder assembled from literals, such
    as ``"__glo" + "bals__"``. Keep this deliberately small and only fold
    operations whose operands are already known string literals; user code is
    never executed while validating the tree.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                return None
            parts.append(value.value)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string_value(node.left)
        right = _constant_string_value(node.right)
        if left is not None and right is not None:
            return left + right
    return None

def _check_ast_safety(code: str) -> Optional[str]:
    """Validate parsed script structure against sandbox-escape patterns.

    The lowercased substring scan in :data:`_BLOCKED_PATTERNS` is bypassable
    with obfuscation and produces false positives on strings/comments. This
    operates on the parsed AST instead, so it catches dunder attribute access
    and blocked-builtin references regardless of source formatting, and covers
    every dunder — not just the hand-maintained pattern list.

    Returns an error message if unsafe, ``None`` if safe (or unparseable, in
    which case ``compile()`` surfaces the syntax error with line info).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        # Any private attribute access is an escape vector. Blocking the
        # complete underscore-prefixed namespace also keeps implementation
        # details on callable proxies unreachable (not only familiar dunders).
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                return f"Use of '{attr}' is not allowed in the sandbox"
        # Reflection helpers can turn a runtime string into an attribute name,
        # so reject literal dunders and fragments that can be concatenated into
        # one. This closes obfuscations the raw source scan cannot see.
        elif isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            literal = _constant_string_value(node)
            if literal is not None and "__" in literal:
                return "Dunder names in strings are not allowed in the sandbox"
        # Direct references to blocked builtins, even when shadowed or aliased
        # syntactically (e.g. ``g = getattr``).
        elif isinstance(node, ast.Name):
            if node.id in _BLOCKED_BUILTINS:
                return f"Use of '{node.id}' is not allowed in the sandbox"
        # Imports must be from the explicit allowlist. _safe_import enforces
        # this at runtime too, but rejecting at parse time gives a clearer,
        # earlier refusal and keeps blocked reflection modules out of the
        # execution thread entirely.
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BLOCKED_MODULES or top not in _ALLOWED_MODULES:
                    return f"Import of '{alias.name}' is not allowed in the sandbox"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in _BLOCKED_MODULES or top not in _ALLOWED_MODULES:
                return f"Import of '{node.module}' is not allowed in the sandbox"

    return None


# Maximum output length (characters)
MAX_OUTPUT_LENGTH = 50_000

# Maximum source length accepted by the sandbox (bytes)
MAX_CODE_LENGTH_BYTES = 100 * 1024

# Maximum execution time (seconds)
DEFAULT_TIMEOUT = 30


def _safe_sleep(seconds: float) -> None:
    """Sleep in short increments so script timeouts can interrupt it cleanly."""
    try:
        remaining = float(seconds)
    except (TypeError, ValueError):
        raise ValueError("sleep duration must be numeric")
    if remaining < 0:
        raise ValueError("sleep duration must be non-negative")

    deadline = getattr(_execution_state, "deadline", None)
    timeout = getattr(_execution_state, "timeout", DEFAULT_TIMEOUT)

    while remaining > 0:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Script execution timed out after {timeout}s")
        chunk = min(remaining, 0.05)
        time.sleep(chunk)
        remaining -= chunk


# Python functions carry a ``__globals__`` reference. Exposing one directly
# in a restricted ``exec`` scope makes the scope's own module imports
# reachable, even when the function only appears to be a harmless wrapper.
# Callable proxies keep the implementation registry private to this module;
# the user-visible object has no function attribute and no ``__globals__``.
_SANDBOX_IMPLEMENTATIONS: Dict[str, Callable[..., Any]] = {}


class _SandboxCallable:
    """Callable object whose implementation is not reachable from the scope."""

    __slots__ = ("_name", "_doc")

    def __init__(self, name: str, doc: str = "") -> None:
        self._name = name
        self._doc = doc

    def __call__(self, *args, **kwargs):
        implementation = _SANDBOX_IMPLEMENTATIONS.get(self._name)
        if implementation is None:
            raise RuntimeError("Sandbox callable is unavailable")
        return implementation(*args, **kwargs)


def _make_sandbox_callable(
    name: str, implementation: Callable[..., Any]
) -> _SandboxCallable:
    _SANDBOX_IMPLEMENTATIONS[name] = implementation
    return _SandboxCallable(name, (implementation.__doc__ or "").strip())


def _is_safe_context_value(value: Any, *, depth: int = 0) -> bool:
    """Return whether a context value is inert data rather than executable code."""
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return "__" not in value
    if depth >= 8:
        return False
    if isinstance(value, list):
        return all(_is_safe_context_value(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and not key.startswith("_")
            and "__" not in key
            and _is_safe_context_value(item, depth=depth + 1)
            for key, item in value.items()
        )
    return False


def _build_safe_time_module() -> types.ModuleType:
    """Expose a constrained time module to sandboxed scripts."""
    safe_time = types.ModuleType("time")
    for attr in (
        "time", "monotonic", "perf_counter", "process_time",
        "strftime", "gmtime", "localtime", "ctime",
    ):
        setattr(safe_time, attr, getattr(time, attr))
    safe_time.sleep = _make_sandbox_callable("safe_sleep", _safe_sleep)
    return safe_time


_SAFE_TIME_MODULE = _build_safe_time_module()


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()


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
    if top_level == "time":
        return _SAFE_TIME_MODULE
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

    for name, implementation in (
        ("get_video_info", _safe_get_video_info),
        ("detect_silences", _safe_detect_silences),
        ("generate_chapters", _safe_generate_chapters),
        ("get_scenes", _safe_get_scenes),
        ("get_loudness", _safe_get_loudness),
    ):
        ns[name] = _make_sandbox_callable(name, implementation)

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

    # Build safe builtins — filter both blocked names and all dunders
    # (except the few we explicitly re-add below)
    if isinstance(__builtins__, dict):
        safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in _BLOCKED_BUILTINS and not k.startswith("_")
        }
    else:
        safe_builtins = {
            k: getattr(__builtins__, k)
            for k in dir(__builtins__)
            if k not in _BLOCKED_BUILTINS and not k.startswith("_")
        }

    safe_builtins["__import__"] = _safe_import

    sandbox = {
        "__builtins__": safe_builtins,
        # Pre-imported safe modules
        "math": math,
        "json": _json,
        "re": _re,
        "datetime": _datetime,
        "collections": _collections,
        "time": _SAFE_TIME_MODULE,
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
        "dumps": _make_sandbox_callable("json_dumps", _json.dumps),
        "loads": _make_sandbox_callable("json_loads", _json.loads),
        # OpenCut namespace
        "opencut": types.SimpleNamespace(**_build_opencut_namespace()),
    }

    # Inject caller context — reject dunder keys and non-data values to prevent
    # sandbox escapes through caller-owned functions, modules, or objects.
    if isinstance(context, dict):
        for key, value in context.items():
            if (
                not isinstance(key, str)
                or "__" in key
                or key.startswith("_")
                or not _is_safe_context_value(value)
            ):
                continue
            sandbox[key] = value

    return sandbox


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------

def code_size_bytes(code: str) -> int:
    """Return the UTF-8 byte size of a script."""
    return len(code.encode("utf-8"))


def validate_code_length(code: str) -> None:
    """Reject scripts large enough to create avoidable compile/exec pressure."""
    size = code_size_bytes(code)
    if size > MAX_CODE_LENGTH_BYTES:
        raise ValueError(
            f"Script code is too large ({size} bytes). "
            f"Maximum length is {MAX_CODE_LENGTH_BYTES} bytes."
        )


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

    try:
        validate_code_length(code)
    except ValueError as exc:
        return ScriptResult(output="", success=False, error=str(exc))

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

    # Obfuscation-proof structural check on the parsed AST (catches dunder
    # access and blocked-builtin references the substring scan above misses).
    ast_error = _check_ast_safety(code)
    if ast_error:
        result = ScriptResult(output="", success=False, error=ast_error)
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
    deadline = time.monotonic() + timeout

    def _deadline_trace(frame, event, arg):
        if time.monotonic() >= getattr(_execution_state, "deadline", deadline):
            raise TimeoutError(f"Script execution timed out after {timeout}s")
        return _deadline_trace

    def _run():
        try:
            _execution_state.deadline = deadline
            _execution_state.timeout = timeout
            sys.settrace(_deadline_trace)
            compiled = compile(code, "<opencut_script>", "exec")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, sandbox)  # noqa: S102
        except Exception as exc:
            exec_error[0] = exc
        finally:
            sys.settrace(None)
            for attr in ("deadline", "timeout"):
                if hasattr(_execution_state, attr):
                    delattr(_execution_state, attr)
            exec_done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    finished = exec_done.wait(timeout=timeout)
    elapsed_ms = (time.monotonic() - start_time) * 1000.0
    if elapsed_ms <= 0:
        elapsed_ms = 0.001

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
            doc = obj._doc if isinstance(obj, _SandboxCallable) else (obj.__doc__ or "")
            funcs.append({
                "name": f"opencut.{name}",
                "doc": doc.strip(),
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
    with _history_lock:
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
    """Persist execution history to disk (atomic write)."""
    with _history_lock:
        _ensure_dir()
        try:
            import tempfile
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(_HISTORY_FILE), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(history[-_MAX_HISTORY:], fh, indent=2)
                os.replace(tmp_path, _HISTORY_FILE)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.warning("Failed to save console history: %s", exc)


def _append_history(code: str, result: ScriptResult) -> None:
    """Append an execution record to history."""
    with _history_lock:
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
    with _history_lock:
        history = _load_history()
        return history[-limit:]


def clear_history() -> None:
    """Clear execution history."""
    _save_history([])
    logger.info("Cleared scripting console history")
