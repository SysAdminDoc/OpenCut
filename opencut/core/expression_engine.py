"""
OpenCut Expression Scripting Engine v1.0.0

Lightweight per-property expression evaluator with sandboxed Python
execution. Provides time, frame, audio amplitude, and math variables.
"""

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Safe evaluation infrastructure
# ---------------------------------------------------------------------------

# Allowed builtin functions in expressions
_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "round": round,
    "bool": bool,
    "len": len,
    "range": range,
    "sum": sum,
    "pow": pow,
}

# Allowed math module functions
_SAFE_MATH = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "fmod": math.fmod,
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Allowed AST node types for safe evaluation
_ALLOWED_NODES = {
    ast.Module,
    ast.Expr,
    ast.Expression,
    # Literals
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    # Operations
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    # Comparison operators
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    # Conditional
    ast.IfExp,
    # Variables and calls
    ast.Name,
    ast.Load,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    # Subscript
    ast.Subscript,
    ast.Slice,
}

# Python 3.7/3.8 compat: these were removed in 3.12+
for _compat_name in ("Num", "Str", "Index"):
    _node = getattr(ast, _compat_name, None)
    if _node is not None:
        _ALLOWED_NODES.add(_node)


class ExpressionError(ValueError):
    """Raised when an expression is invalid or unsafe."""
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_ast(node: ast.AST, depth: int = 0):
    """Recursively validate that an AST tree uses only allowed node types."""
    if depth > 50:
        raise ExpressionError("Expression too deeply nested (max depth 50)")

    if type(node) not in _ALLOWED_NODES:
        raise ExpressionError(
            f"Disallowed syntax: {type(node).__name__}. "
            f"Only arithmetic, comparisons, conditionals, and function calls are allowed."
        )

    for child in ast.iter_child_nodes(node):
        _validate_ast(child, depth + 1)


def validate_expression(expr: str) -> dict:
    """Validate an expression for syntax and safety.

    Args:
        expr: Expression string to validate.

    Returns:
        Dict with ``valid`` (bool), ``error`` (str or None),
        ``variables`` (list of referenced names).
    """
    result = {
        "valid": False,
        "error": None,
        "variables": [],
    }

    if not expr or not isinstance(expr, str):
        result["error"] = "Empty or non-string expression"
        return result

    expr = expr.strip()
    if len(expr) > 2000:
        result["error"] = "Expression too long (max 2000 characters)"
        return result

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        result["error"] = f"Syntax error: {e.msg} (line {e.lineno}, col {e.offset})"
        return result

    try:
        _validate_ast(tree)
    except ExpressionError as e:
        result["error"] = str(e)
        return result

    # Extract referenced variable names
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)

    result["valid"] = True
    result["variables"] = sorted(names)
    return result


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@dataclass
class CompiledExpression:
    """A compiled expression ready for fast evaluation."""
    source: str
    code: Any = None    # compiled code object
    variables: List[str] = field(default_factory=list)
    valid: bool = False
    error: str = ""


def compile_expression(expr: str) -> CompiledExpression:
    """Compile an expression for repeated evaluation.

    Args:
        expr: Expression string.

    Returns:
        :class:`CompiledExpression` with compiled code object.
    """
    result = CompiledExpression(source=expr)

    validation = validate_expression(expr)
    if not validation["valid"]:
        result.error = validation["error"]
        return result

    try:
        code = compile(expr.strip(), "<expression>", "eval")
        result.code = code
        result.variables = validation["variables"]
        result.valid = True
    except Exception as e:
        result.error = f"Compilation failed: {e}"

    return result


# ---------------------------------------------------------------------------
# Expression context
# ---------------------------------------------------------------------------

def create_expression_context(
    frame: int = 0,
    time: float = 0.0,
    audio_amp: float = 0.0,
    fps: float = 24.0,
    duration: float = 0.0,
    width: int = 1920,
    height: int = 1080,
    custom_vars: Optional[Dict[str, Any]] = None,
) -> dict:
    """Create an evaluation context with standard animation variables.

    Args:
        frame: Current frame number.
        time: Current time in seconds (alias: t).
        audio_amp: Audio amplitude at current time (0.0-1.0).
        fps: Frame rate.
        duration: Total duration in seconds.
        width: Composition width.
        height: Composition height.
        custom_vars: Additional custom variables.

    Returns:
        Dict of variables available in expressions.
    """
    # Normalized progress (0 to 1)
    progress = time / max(duration, 0.001) if duration > 0 else 0.0

    ctx = {
        # Time variables
        "frame": frame,
        "f": frame,
        "time": time,
        "t": time,
        "fps": fps,
        "duration": duration,
        "progress": min(progress, 1.0),

        # Audio
        "audio_amp": audio_amp,
        "audio": audio_amp,
        "amp": audio_amp,

        # Composition
        "width": width,
        "height": height,
        "w": width,
        "h": height,

        # Math functions
        **_SAFE_MATH,

        # Utility functions
        "lerp": lambda a, b, t: a + (b - a) * t,
        "clamp": lambda x, lo, hi: max(lo, min(x, hi)),
        "smoothstep": lambda edge0, edge1, x: (
            0.0 if x <= edge0 else
            1.0 if x >= edge1 else
            (lambda t: t * t * (3 - 2 * t))((x - edge0) / (edge1 - edge0))
        ),
        "wiggle": lambda freq, amp_val, t_val=time: (
            math.sin(t_val * freq * 2 * math.pi) * amp_val
        ),
        "ease_in": lambda t_val: t_val * t_val,
        "ease_out": lambda t_val: 1 - (1 - t_val) ** 2,
        "ease_in_out": lambda t_val: (
            2 * t_val * t_val if t_val < 0.5
            else 1 - (-2 * t_val + 2) ** 2 / 2
        ),
        "step": lambda edge, x: 0.0 if x < edge else 1.0,
        "pulse": lambda lo, hi, x: 1.0 if lo <= x <= hi else 0.0,
    }

    # Add builtins
    ctx.update(_SAFE_BUILTINS)

    # Add custom variables
    if custom_vars:
        ctx.update(custom_vars)

    return ctx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_expression(
    expr,
    context: dict,
) -> Any:
    """Evaluate an expression in a sandboxed context.

    Args:
        expr: Expression string or :class:`CompiledExpression`.
        context: Evaluation context from :func:`create_expression_context`.

    Returns:
        The evaluated result.

    Raises:
        ExpressionError: If the expression is invalid or evaluation fails.
    """
    if isinstance(expr, CompiledExpression):
        if not expr.valid:
            raise ExpressionError(f"Invalid compiled expression: {expr.error}")
        code = expr.code
    elif isinstance(expr, str):
        validation = validate_expression(expr)
        if not validation["valid"]:
            raise ExpressionError(validation["error"])
        code = compile(expr.strip(), "<expression>", "eval")
    else:
        raise ExpressionError(f"Expected string or CompiledExpression, got {type(expr).__name__}")

    # Sandbox: restrict globals/builtins
    safe_globals = {"__builtins__": {}}
    safe_globals.update(context)

    try:
        result = eval(code, safe_globals)
    except ExpressionError:
        raise
    except Exception as e:
        raise ExpressionError(f"Evaluation error: {e}")

    return result
