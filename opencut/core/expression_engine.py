"""
OpenCut Expression Engine (Category 79 - Motion Design)

Lightweight scripting layer to drive any animatable property. Expressions
are Python expressions evaluated per frame with a sandboxed set of globals
including time, frame, fps, audio_amplitude, beat, and math functions.

Functions:
    evaluate_expression  - Evaluate a single expression in context
    evaluate_timeline    - Evaluate expression for every frame in timeline
    create_context       - Create an ExpressionContext for evaluation
    list_functions       - List available sandbox functions
"""

import logging
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

_evaluation_state = threading.local()
_MAX_NOISE_OCTAVES = 32


def _check_deadline() -> None:
    """Raise TimeoutError when the current expression exceeded its deadline."""
    deadline = getattr(_evaluation_state, "deadline", None)
    if deadline is None or time.monotonic() < deadline:
        return
    timeout_ms = getattr(_evaluation_state, "timeout_ms", None)
    if timeout_ms is None:
        raise TimeoutError("Expression evaluation timed out")
    raise TimeoutError(f"Expression evaluation timed out after {timeout_ms}ms")


def _normalize_noise_octaves(octaves: int) -> int:
    """Validate and normalize octave counts to a sane motion-graphics range."""
    if isinstance(octaves, bool):
        raise ValueError(
            f"noise octaves must be an integer between 1 and {_MAX_NOISE_OCTAVES}"
        )
    if isinstance(octaves, float):
        if not math.isfinite(octaves) or not octaves.is_integer():
            raise ValueError(
                f"noise octaves must be an integer between 1 and {_MAX_NOISE_OCTAVES}"
            )
        octaves = int(octaves)
    else:
        try:
            octaves = int(octaves)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"noise octaves must be an integer between 1 and {_MAX_NOISE_OCTAVES}"
            ) from exc

    if octaves < 1 or octaves > _MAX_NOISE_OCTAVES:
        raise ValueError(
            f"noise octaves must be between 1 and {_MAX_NOISE_OCTAVES}"
        )
    return octaves

# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExpressionResult:
    """Result of expression evaluation over a timeline."""

    values: List[float] = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "values": self.values,
            "min": self.min_value,
            "max": self.max_value,
            "errors": self.errors,
        }


@dataclass
class CompiledExpression:
    """Pre-validated expression container for batch reuse."""

    source: str = ""
    code: Any = None
    valid: bool = False
    error: str = ""
    variables: List[str] = field(default_factory=list)


class ExpressionError(ValueError):
    """Raised when an expression cannot be evaluated safely."""


# ---------------------------------------------------------------------------
# Simplex-like Noise (Perlin-inspired via value noise)
# ---------------------------------------------------------------------------


def _noise_hash(x: int, seed: int = 0) -> float:
    """Simple hash function for noise generation."""
    n = x + seed * 131
    n = (n << 13) ^ n
    n = n * (n * n * 15731 + 789221) + 1376312589
    return (n & 0x7FFFFFFF) / 0x7FFFFFFF


def _noise_1d(x: float, seed: int = 0) -> float:
    """1D value noise, returns -1 to 1."""
    ix = int(math.floor(x))
    fx = x - ix
    # Smoothstep
    fx = fx * fx * (3.0 - 2.0 * fx)
    a = _noise_hash(ix, seed) * 2.0 - 1.0
    b = _noise_hash(ix + 1, seed) * 2.0 - 1.0
    return a + (b - a) * fx


def _noise_2d(x: float, y: float, seed: int = 0) -> float:
    """2D value noise, returns -1 to 1."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)

    n00 = _noise_hash(ix + iy * 57, seed) * 2.0 - 1.0
    n10 = _noise_hash(ix + 1 + iy * 57, seed) * 2.0 - 1.0
    n01 = _noise_hash(ix + (iy + 1) * 57, seed) * 2.0 - 1.0
    n11 = _noise_hash(ix + 1 + (iy + 1) * 57, seed) * 2.0 - 1.0

    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return nx0 + (nx1 - nx0) * fy


def noise(x: float, y: float = 0.0, octaves: int = 1,
          seed: int = 0) -> float:
    """Multi-octave noise function available in expressions.

    Args:
        x: First coordinate.
        y: Second coordinate (0 for 1D noise).
        octaves: Number of octaves for fractal noise.
        seed: Random seed.

    Returns:
        Float in range approximately -1 to 1.
    """
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("noise coordinates must be finite numbers")

    octaves = _normalize_noise_octaves(octaves)
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amp = 0.0

    for octave_idx in range(octaves):
        if octave_idx % 4 == 0:
            _check_deadline()
        if y == 0.0:
            value += _noise_1d(x * frequency, seed) * amplitude
        else:
            value += _noise_2d(x * frequency, y * frequency, seed) * amplitude
        max_amp += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return value / max_amp if max_amp > 0 else 0.0


# ---------------------------------------------------------------------------
# Sandbox Math Functions
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b by t."""
    return a + (b - a) * t


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smoothstep interpolation between edge0 and edge1."""
    t = _clamp((x - edge0) / max(edge1 - edge0, 0.0001), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _remap(value: float, in_min: float, in_max: float,
           out_min: float, out_max: float) -> float:
    """Remap value from one range to another."""
    t = (value - in_min) / max(in_max - in_min, 0.0001)
    return out_min + t * (out_max - out_min)


def _ping_pong(t: float, length: float = 1.0) -> float:
    """Ping-pong oscillation between 0 and length."""
    if length <= 0:
        return 0.0
    t = t % (length * 2)
    if t > length:
        return 2 * length - t
    return t


def _step(edge: float, x: float) -> float:
    """Step function: 0 if x < edge, 1 otherwise."""
    return 1.0 if x >= edge else 0.0


def _pulse(x: float, frequency: float = 1.0,
           duty_cycle: float = 0.5) -> float:
    """Square wave / pulse function."""
    phase = (x * frequency) % 1.0
    return 1.0 if phase < duty_cycle else 0.0


# ---------------------------------------------------------------------------
# Sandbox Globals
# ---------------------------------------------------------------------------


SANDBOX_FUNCTIONS = {
    # Math
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": math.sqrt,
    "pow": pow,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "fmod": math.fmod,
    "degrees": math.degrees,
    "radians": math.radians,
    # Constants
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
    "inf": math.inf,
    # Interpolation & utility
    "lerp": _lerp,
    "clamp": _clamp,
    "smoothstep": _smoothstep,
    "remap": _remap,
    "ping_pong": _ping_pong,
    "step": _step,
    "pulse": _pulse,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    # Noise
    "noise": noise,
}


def list_functions() -> List[dict]:
    """Return list of functions available in the expression sandbox."""
    descriptions = {
        "sin": "Sine function (radians)",
        "cos": "Cosine function (radians)",
        "tan": "Tangent function (radians)",
        "asin": "Arc sine",
        "acos": "Arc cosine",
        "atan": "Arc tangent",
        "atan2": "Two-argument arc tangent",
        "sqrt": "Square root",
        "pow": "Power function",
        "abs": "Absolute value",
        "floor": "Floor (round down)",
        "ceil": "Ceiling (round up)",
        "round": "Round to nearest integer",
        "log": "Natural logarithm",
        "log2": "Base-2 logarithm",
        "log10": "Base-10 logarithm",
        "exp": "e raised to power",
        "fmod": "Floating-point modulo",
        "degrees": "Radians to degrees",
        "radians": "Degrees to radians",
        "lerp": "Linear interpolation: lerp(a, b, t)",
        "clamp": "Clamp value: clamp(val, min, max)",
        "smoothstep": "Smooth interpolation: smoothstep(edge0, edge1, x)",
        "remap": "Remap range: remap(val, in_min, in_max, out_min, out_max)",
        "ping_pong": "Oscillate: ping_pong(t, length)",
        "step": "Step function: step(edge, x)",
        "pulse": "Square wave: pulse(x, freq, duty)",
        "noise": "Perlin-like noise: noise(x, y=0, octaves=1, seed=0)",
        "min": "Minimum of values",
        "max": "Maximum of values",
        "random": "Seeded random: random() returns 0-1",
    }
    result = []
    for name in SANDBOX_FUNCTIONS:
        if callable(SANDBOX_FUNCTIONS[name]):
            result.append({
                "name": name,
                "description": descriptions.get(name, ""),
            })
    # Add constants
    for name in ("pi", "tau", "e", "inf"):
        if name in SANDBOX_FUNCTIONS:
            result.append({
                "name": name,
                "description": f"Mathematical constant {name}",
                "type": "constant",
            })
    return result


# ---------------------------------------------------------------------------
# AST Safety Checker
# ---------------------------------------------------------------------------

import ast  # noqa: E402

_BANNED_NODE_TYPES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
    ast.Delete, ast.ClassDef, ast.AsyncFunctionDef, ast.FunctionDef,
    ast.AsyncFor, ast.AsyncWith, ast.Await,
    ast.Yield, ast.YieldFrom,
    ast.Try, ast.Raise,
)

# Banned attribute names (security-sensitive)
_BANNED_ATTRS = frozenset({
    "__import__", "__builtins__", "__globals__", "__code__",
    "__subclasses__", "__bases__", "__mro__", "__class__",
    "__dict__", "__getattr__", "__setattr__", "__delattr__",
    "__getattribute__", "__init__", "__new__", "__call__",
    "__reduce__", "__reduce_ex__", "__getnewargs__",
    "__getnewargs_ex__", "__module__", "__wrapped__",
    "__qualname__", "__self__", "__func__", "__closure__",
    "__init_subclass__", "__set_name__", "__del__",
    "eval", "exec", "compile", "open", "input",
    "__loader__", "__spec__", "__name__", "__file__",
})

_MAX_EXPRESSION_LENGTH = 2000
_RESERVED_NAMES = frozenset(SANDBOX_FUNCTIONS) | frozenset({
    "time", "t", "frame", "f", "fps", "duration", "progress",
    "audio_amplitude", "audio_amp", "amplitude", "amp",
    "beat", "seed", "width", "height", "w", "h",
    "True", "False",
})


def _check_ast_safety(expr_str: str) -> Optional[str]:
    """Check expression AST for safety violations.

    Returns error message if unsafe, None if safe.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODE_TYPES):
            return f"Forbidden construct: {type(node).__name__}"

        if isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                return f"Forbidden attribute access: {node.attr}"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "open",
                                    "__import__", "input", "getattr",
                                    "setattr", "delattr", "globals",
                                    "locals", "vars", "dir", "type",
                                    "isinstance", "issubclass"):
                    return f"Forbidden function call: {node.func.id}"

    return None


def _collect_variable_names(expr_str: str) -> List[str]:
    """Return user-defined variable references used by an expression."""
    tree = ast.parse(expr_str, mode="eval")
    names = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        name = node.id
        if name in _RESERVED_NAMES or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def validate_expression(expr_string: str) -> dict:
    """Validate an expression and report referenced custom variables."""
    expr_string = "" if expr_string is None else str(expr_string)
    expr_string = expr_string.strip()

    if not expr_string:
        return {
            "valid": False,
            "error": "Expression is empty",
            "variables": [],
        }
    if len(expr_string) > _MAX_EXPRESSION_LENGTH:
        return {
            "valid": False,
            "error": (
                f"Expression exceeds maximum length of "
                f"{_MAX_EXPRESSION_LENGTH} characters"
            ),
            "variables": [],
        }

    safety_error = _check_ast_safety(expr_string)
    if safety_error:
        return {
            "valid": False,
            "error": safety_error,
            "variables": [],
        }

    try:
        variables = _collect_variable_names(expr_string)
    except SyntaxError as exc:
        return {
            "valid": False,
            "error": f"Syntax error: {exc}",
            "variables": [],
        }

    return {
        "valid": True,
        "error": "",
        "variables": variables,
    }


def compile_expression(expr_string: str) -> CompiledExpression:
    """Compile an expression once for repeated evaluation."""
    validation = validate_expression(expr_string)
    if not validation["valid"]:
        return CompiledExpression(
            source="" if expr_string is None else str(expr_string).strip(),
            valid=False,
            error=validation["error"],
            variables=validation["variables"],
        )

    source = str(expr_string).strip()
    try:
        code = compile(source, "<opencut-expression>", "eval")
    except SyntaxError as exc:
        return CompiledExpression(
            source=source,
            valid=False,
            error=f"Syntax error: {exc}",
            variables=[],
        )

    return CompiledExpression(
        source=source,
        code=code,
        valid=True,
        error="",
        variables=validation["variables"],
    )


# ---------------------------------------------------------------------------
# Expression Context
# ---------------------------------------------------------------------------


@dataclass
class ExpressionContext:
    """Context providing variables for expression evaluation.

    Attributes:
        time: Current time in seconds.
        frame: Current frame number.
        fps: Frames per second.
        audio_amplitude: Audio amplitude 0.0-1.0.
        beat: True if current frame is on a beat.
        seed: Random seed for reproducibility.
        custom_vars: Additional variables available in expressions.
    """

    time: float = 0.0
    frame: int = 0
    fps: float = 30.0
    audio_amplitude: float = 0.0
    duration: float = 0.0
    width: int = 1920
    height: int = 1080
    beat: bool = False
    seed: int = 42
    custom_vars: Dict[str, Any] = field(default_factory=dict)

    def to_globals(self) -> dict:
        """Build the sandbox globals dict for eval()."""
        rng = random.Random(self.seed + self.frame)
        progress = 0.0
        if self.duration > 0:
            progress = _clamp(self.time / self.duration, 0.0, 1.0)
        globs = dict(SANDBOX_FUNCTIONS)
        globs.update({
            "time": self.time,
            "t": self.time,
            "frame": self.frame,
            "f": self.frame,
            "fps": self.fps,
            "audio_amplitude": self.audio_amplitude,
            "audio_amp": self.audio_amplitude,
            "amplitude": self.audio_amplitude,
            "amp": self.audio_amplitude,
            "duration": self.duration,
            "progress": progress,
            "width": self.width,
            "height": self.height,
            "w": self.width,
            "h": self.height,
            "seed": self.seed,
            "beat": self.beat,
            "random": rng.random,
            "True": True,
            "False": False,
        })
        # Sanitize custom_vars: reject dunder keys and non-scalar values
        # that could carry dangerous methods (__float__, __class__, etc.)
        for k, v in self.custom_vars.items():
            if k.startswith("_") or k in _BANNED_ATTRS:
                continue
            if not isinstance(v, (int, float, bool, str, type(None))):
                continue
            globs[k] = v
        # Remove builtins to prevent access
        globs["__builtins__"] = {}
        return globs

    def __getitem__(self, key: str) -> Any:
        return self.to_globals()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_globals().get(key, default)


def create_context(
    time: float = 0.0,
    frame: int = 0,
    fps: float = 30.0,
    audio_amplitude: float = 0.0,
    duration: float = 0.0,
    width: int = 1920,
    height: int = 1080,
    beat: bool = False,
    seed: int = 42,
    **custom_vars,
) -> ExpressionContext:
    """Create an ExpressionContext with the given parameters."""
    return ExpressionContext(
        time=time,
        frame=frame,
        fps=fps,
        audio_amplitude=audio_amplitude,
        duration=duration,
        width=width,
        height=height,
        beat=beat,
        seed=seed,
        custom_vars=custom_vars,
    )


def create_expression_context(
    time: float = 0.0,
    frame: int = 0,
    fps: float = 30.0,
    duration: float = 0.0,
    width: int = 1920,
    height: int = 1080,
    audio_amp: float = 0.0,
    beat: bool = False,
    seed: int = 42,
    custom_vars: Optional[Dict[str, Any]] = None,
) -> ExpressionContext:
    """Backward-compatible context factory used by batch-data routes."""
    return ExpressionContext(
        time=float(time),
        frame=int(frame),
        fps=float(fps),
        audio_amplitude=float(audio_amp),
        duration=float(duration),
        width=int(width),
        height=int(height),
        beat=bool(beat),
        seed=int(seed),
        custom_vars=dict(custom_vars or {}),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Default timeout per expression evaluation (ms)
DEFAULT_TIMEOUT_MS = 100


def evaluate_expression(expr_string: str,
                        context: Optional[ExpressionContext] = None,
                        timeout_ms: int = DEFAULT_TIMEOUT_MS) -> float:
    """Evaluate a single expression and return the result as a float.

    Args:
        expr_string: Python expression string.
        context: ExpressionContext providing variables.
        timeout_ms: Maximum evaluation time in milliseconds.

    Returns:
        Float result of the expression.

    Raises:
        ValueError: If expression is unsafe or has syntax errors.
        TimeoutError: If evaluation exceeds timeout.
        RuntimeError: If evaluation fails.
    """
    code = None
    source = expr_string
    if isinstance(expr_string, CompiledExpression):
        if not expr_string.valid or expr_string.code is None:
            raise ExpressionError(
                f"Evaluation error: {expr_string.error or 'Expression is invalid'}"
            )
        source = expr_string.source
        code = expr_string.code
    else:
        if not expr_string or not str(expr_string).strip():
            raise ValueError("Expression is empty")
        source = str(expr_string).strip()

        validation = validate_expression(source)
        if not validation["valid"]:
            raise ExpressionError(f"Evaluation error: {validation['error']}")
        code = compile(source, "<opencut-expression>", "eval")

    if context is None:
        context = ExpressionContext()

    globs = context.to_globals()

    # Evaluate with timeout using threading
    result_box = [None]
    error_box = [None]
    deadline = time.monotonic() + (timeout_ms / 1000.0)

    def _deadline_trace(frame, event, arg):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Expression evaluation timed out after {timeout_ms}ms"
            )
        return _deadline_trace

    def _eval():
        try:
            _evaluation_state.deadline = deadline
            _evaluation_state.timeout_ms = timeout_ms
            sys.settrace(_deadline_trace)
            result_box[0] = eval(code, globs)  # noqa: S307
        except Exception as e:
            error_box[0] = e
        finally:
            sys.settrace(None)
            _evaluation_state.deadline = None
            _evaluation_state.timeout_ms = None

    thread = threading.Thread(target=_eval, daemon=True)
    thread.start()
    thread.join(timeout=timeout_ms / 1000.0)

    if thread.is_alive():
        raise TimeoutError(
            f"Expression evaluation timed out after {timeout_ms}ms"
        )

    if error_box[0] is not None:
        if isinstance(error_box[0], TimeoutError):
            raise error_box[0]
        raise ExpressionError(f"Evaluation error: {error_box[0]}")

    result = result_box[0]

    # Convert to float — only allow primitive types to prevent
    # __float__/__int__ dunder calls on arbitrary objects
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    if isinstance(result, (int, float)):
        return float(result)
    if result is None:
        return 0.0
    if isinstance(result, str):
        try:
            return float(result)
        except (ValueError, TypeError):
            return 0.0

    # Reject non-primitive results — calling float() on arbitrary objects
    # would invoke __float__() which could be a sandbox escape
    return 0.0


def evaluate_timeline(expr_string: str,
                      fps: float = 30.0,
                      duration: float = 1.0,
                      audio_amplitudes: Optional[List[float]] = None,
                      beats: Optional[List[bool]] = None,
                      seed: int = 42,
                      timeout_ms: int = DEFAULT_TIMEOUT_MS,
                      on_progress: Optional[Callable] = None,
                      **custom_vars) -> ExpressionResult:
    """Evaluate expression for every frame in a timeline.

    Args:
        expr_string: Python expression string.
        fps: Frames per second.
        duration: Timeline duration in seconds.
        audio_amplitudes: Per-frame audio amplitude values.
        beats: Per-frame beat boolean values.
        seed: Random seed.
        timeout_ms: Timeout per frame evaluation.
        on_progress: Progress callback.
        **custom_vars: Additional variables.

    Returns:
        ExpressionResult with per-frame values and statistics.
    """
    total_frames = max(1, int(fps * duration))
    values = []
    errors = []

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        amp = 0.0
        if audio_amplitudes and frame_idx < len(audio_amplitudes):
            amp = audio_amplitudes[frame_idx]
        beat_val = False
        if beats and frame_idx < len(beats):
            beat_val = beats[frame_idx]

        ctx = ExpressionContext(
            time=time_s,
            frame=frame_idx,
            fps=fps,
            audio_amplitude=amp,
            beat=beat_val,
            seed=seed,
            custom_vars=custom_vars,
        )

        try:
            val = evaluate_expression(expr_string, ctx, timeout_ms)
            values.append(val)
        except Exception as e:
            errors.append(f"Frame {frame_idx}: {e}")
            values.append(0.0)

        if on_progress and total_frames > 1:
            pct = int((frame_idx + 1) / total_frames * 100)
            on_progress(pct)

    min_val = min(values) if values else 0.0
    max_val = max(values) if values else 0.0

    return ExpressionResult(
        values=values,
        min_value=min_val,
        max_value=max_val,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------


def evaluate_multi(expressions: Dict[str, str],
                   context: Optional[ExpressionContext] = None,
                   timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Dict[str, float]:
    """Evaluate multiple named expressions in the same context.

    Args:
        expressions: Dict mapping property names to expression strings.
        context: Shared evaluation context.
        timeout_ms: Timeout per expression.

    Returns:
        Dict mapping property names to float results.
    """
    results = {}
    for name, expr in expressions.items():
        try:
            results[name] = evaluate_expression(expr, context, timeout_ms)
        except Exception as e:
            logger.debug("Expression '%s' failed: %s", name, e)
            results[name] = 0.0
    return results
