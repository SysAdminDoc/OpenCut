"""AI evaluation harness (F120).

Optional AI backends in OpenCut share a problem: every release wave
introduces another half-dozen models, and there is no consistent way to
say "this one is fast enough for the editor's preview path" vs "this is
production-grade." Without a harness, the only way to evaluate is by
hand, which is why most of the Wave H/T stubs in the roadmap never
graduate.

The harness in this module is intentionally lightweight: it doesn't
download weights or run inference itself. Instead, it provides a
**registry of evaluations** that any model integration can register
into, plus a runner that:

* Executes the evaluation function against a sample fixture.
* Captures latency (wall clock) and a caller-supplied quality score.
* Persists the result under ``~/.opencut/ai_eval/<feature_id>.json``
  so dashboards / docs can compare runs over time.

A model integration registers itself with ``register_evaluation``
(decorator). Calling ``run_evaluation(feature_id, ...)`` resolves the
function, runs it, and returns a structured :class:`EvalResult`.

The persisted record is the canonical answer to "is this model ready
to ship?" — F115 cards point at it, the panel can render a chip, and
release_smoke can fail on missing or stale evaluations.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

EVAL_DIR = Path(os.path.expanduser("~")) / ".opencut" / "ai_eval"
EVAL_VERSION = 1


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single fixture the harness can feed an evaluation function."""

    sample_id: str
    path: str = ""           # absolute path to the sample on disk (may be empty)
    sample_type: str = "audio"  # audio | video | image | text
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResult:
    """Outcome of running an evaluation."""

    feature_id: str
    sample_id: str
    success: bool
    latency_ms: int
    quality_score: float = 0.0
    quality_metric: str = ""
    notes: str = ""
    environment: dict = field(default_factory=dict)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalDefinition:
    """Registered evaluation entry."""

    feature_id: str
    runner: Callable[[EvalSample], dict]
    description: str = ""
    sample_type: str = "audio"
    metric_name: str = "n/a"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_registry: Dict[str, EvalDefinition] = {}
_registry_lock = threading.Lock()


def register_evaluation(
    feature_id: str,
    *,
    description: str = "",
    sample_type: str = "audio",
    metric_name: str = "n/a",
) -> Callable:
    """Decorator that registers an evaluation function under ``feature_id``."""

    def _decorator(fn: Callable[[EvalSample], dict]) -> Callable[[EvalSample], dict]:
        with _registry_lock:
            if feature_id in _registry:
                raise ValueError(f"evaluation already registered for {feature_id!r}")
            _registry[feature_id] = EvalDefinition(
                feature_id=feature_id,
                runner=fn,
                description=description,
                sample_type=sample_type,
                metric_name=metric_name,
            )
        return fn

    return _decorator


def clear_registry() -> None:
    """Test-only helper — drop the registered evaluations."""
    with _registry_lock:
        _registry.clear()


def list_evaluations() -> List[EvalDefinition]:
    with _registry_lock:
        return list(_registry.values())


def get_evaluation(feature_id: str) -> Optional[EvalDefinition]:
    with _registry_lock:
        return _registry.get(feature_id)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _environment_snapshot() -> dict:
    """Capture a small, privacy-safe description of the runtime."""
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    # GPU info if available — uses the existing gpu module.
    try:
        from opencut import gpu as _gpu

        snapshot["device"] = _gpu.get_device()
        ok, info = _gpu.check_vram(0)
        snapshot["vram_total_mb"] = info.get("total_mb", 0)
        snapshot["gpu_ok"] = bool(ok)
    except Exception:
        snapshot["device"] = "unknown"
    return snapshot


def run_evaluation(
    feature_id: str,
    sample: EvalSample,
    *,
    persist: bool = True,
    eval_dir: Optional[Path] = None,
) -> EvalResult:
    """Run the registered evaluation for ``feature_id`` against ``sample``."""
    definition = get_evaluation(feature_id)
    if definition is None:
        raise KeyError(f"no evaluation registered for {feature_id!r}")

    env = _environment_snapshot()
    start = time.perf_counter()
    success = False
    payload: dict = {}
    error_str = ""
    try:
        payload = definition.runner(sample) or {}
        success = True
    except Exception as exc:
        error_str = repr(exc)
        logger.warning("ai_eval %s failed on sample %s: %s", feature_id, sample.sample_id, exc)

    elapsed = int((time.perf_counter() - start) * 1000)
    result = EvalResult(
        feature_id=feature_id,
        sample_id=sample.sample_id,
        success=success,
        latency_ms=elapsed,
        quality_score=float(payload.get("quality_score", 0.0) or 0.0),
        quality_metric=str(payload.get("quality_metric", definition.metric_name)),
        notes=str(payload.get("notes", "")),
        environment=env,
        error=error_str,
    )
    if persist:
        _append_result(result, eval_dir=eval_dir)
    return result


def _append_result(result: EvalResult, *, eval_dir: Optional[Path] = None) -> None:
    base = Path(eval_dir) if eval_dir else EVAL_DIR
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{result.feature_id}.json"

    history: List[dict] = []
    if target.exists():
        try:
            history = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except (OSError, json.JSONDecodeError):
            history = []
    history.append(result.as_dict())
    # Cap history to the most recent 200 results so the file doesn't grow
    # without bound across long-running installs.
    history = history[-200:]
    target.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def summarise_results(
    feature_id: str,
    *,
    eval_dir: Optional[Path] = None,
) -> dict:
    """Aggregate persisted evaluation results for ``feature_id``."""
    base = Path(eval_dir) if eval_dir else EVAL_DIR
    target = base / f"{feature_id}.json"
    if not target.exists():
        return {"feature_id": feature_id, "runs": 0, "history": []}

    try:
        history = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history = []

    latencies = [int(r.get("latency_ms", 0) or 0) for r in history if r.get("success")]
    quality = [float(r.get("quality_score", 0.0) or 0.0) for r in history if r.get("success")]
    success = sum(1 for r in history if r.get("success"))

    return {
        "feature_id": feature_id,
        "runs": len(history),
        "successes": success,
        "failures": len(history) - success,
        "latency_ms_p50": int(statistics.median(latencies)) if latencies else None,
        "latency_ms_p95": int(_percentile(latencies, 0.95)) if latencies else None,
        "quality_mean": (statistics.fmean(quality) if quality else None),
        "history": history[-20:],
    }


def _percentile(values, pct):
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct * (len(s) - 1)))))
    return s[k]
