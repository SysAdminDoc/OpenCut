"""
OpenCut Python Scripting Console

Execute Python code in a sandboxed scope with core modules pre-imported.
Returns output as string.  Restricts dangerous imports and file system
access outside the project.
"""

import io
import logging
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# Modules allowed in the sandbox
_ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections",
    "itertools", "functools", "operator", "string", "textwrap",
    "copy", "pprint", "decimal", "fractions", "random", "hashlib",
    "base64", "uuid", "dataclasses", "enum", "typing",
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

# Maximum output length (characters)
MAX_OUTPUT_LENGTH = 50_000

# Maximum execution time (seconds)
MAX_EXECUTION_TIME = 10


def _safe_import(name, *args, **kwargs):
    """Restricted import that only allows safe modules."""
    # Handle 'from X import Y' — name is the top-level module
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
    if top_level not in _ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed. "
            f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) \
        else getattr(__builtins__, "__import__")(name, *args, **kwargs)


def create_sandbox(
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a sandboxed execution scope.

    The sandbox has:
    - Restricted builtins (no exec, eval, open, etc.)
    - Safe import function that blocks dangerous modules
    - Pre-populated context variables from the caller
    - Common safe modules pre-imported

    Args:
        context: Optional dict of variables to inject into the sandbox.

    Returns:
        Dict representing the sandbox globals for exec().
    """
    import collections as _collections
    import datetime as _datetime
    import json as _json
    import math
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
    }

    # Inject caller context
    if context:
        for key, value in context.items():
            if not key.startswith("_"):
                sandbox[key] = value

    return sandbox


def execute_script(
    code: str,
    context: Optional[Dict[str, Any]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python source code to execute.
        context: Optional dict of variables available to the script.
        on_progress: Optional progress callback.

    Returns:
        Dict with 'output' (captured stdout+stderr), 'success' bool,
        and 'error' string if execution failed.
    """
    if on_progress:
        on_progress(10, "Preparing sandbox...")

    if not code or not code.strip():
        return {"output": "", "success": True, "error": None}

    # Basic security check on raw source
    code_lower = code.lower()
    for pattern in ("__class__", "__subclasses__", "__bases__", "__mro__",
                    "__globals__", "__code__", "__builtins__"):
        if pattern in code_lower:
            return {
                "output": "",
                "success": False,
                "error": f"Use of '{pattern}' is not allowed in the sandbox",
            }

    sandbox = create_sandbox(context)

    if on_progress:
        on_progress(30, "Executing script...")

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        compiled = compile(code, "<opencut_script>", "exec")

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compiled, sandbox)

        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        output = stdout_text
        if stderr_text:
            output += "\n[stderr]\n" + stderr_text

        # Truncate if too long
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... [output truncated]"

        if on_progress:
            on_progress(100, "Script executed")

        return {
            "output": output,
            "success": True,
            "error": None,
        }

    except SyntaxError as e:
        return {
            "output": "",
            "success": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}",
        }
    except ImportError as e:
        return {
            "output": stdout_capture.getvalue(),
            "success": False,
            "error": str(e),
        }
    except Exception:
        tb = traceback.format_exc()
        # Strip sandbox internals from traceback
        clean_tb = []
        for line in tb.split("\n"):
            if "<opencut_script>" in line or not line.strip().startswith("File"):
                clean_tb.append(line)
        return {
            "output": stdout_capture.getvalue(),
            "success": False,
            "error": "\n".join(clean_tb).strip(),
        }


def get_available_modules() -> List[str]:
    """Return the list of modules available in the sandbox.

    Returns:
        Sorted list of allowed module names.
    """
    return sorted(_ALLOWED_MODULES)
