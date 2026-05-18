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
from typing import Callable, Dict, List, Optional

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
    """Outcome of running an evaluation.

    F178 extends the F120 result shape with:

    * ``vram_peak_mb`` — peak GPU memory used while the runner was on
      the GPU. ``0`` when the runner was CPU-only or torch wasn't
      available.
    * ``reference_score`` — caller-supplied expected/baseline score so
      ``quality_score`` is comparable across runs even when models
      drift.
    * ``backend`` — concrete inference backend ("cpu", "cuda", "mps",
      "rocm", "directml"); set by the runner via the returned
      payload's ``backend`` key, otherwise inferred from
      ``environment["device"]``.
    * ``backend_choice_reason`` — operator-facing rationale for why
      the chosen backend won (e.g. "no CUDA available — fell back to
      CPU"). Powers the cross-backend comparison endpoint.
    """

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
    # F178 additions
    vram_peak_mb: float = 0.0
    reference_score: float = 0.0
    backend: str = ""
    backend_choice_reason: str = ""

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


def _reset_vram_counter() -> None:
    """Reset the torch CUDA peak-memory counter before a run, if torch is loaded."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover - device-dependent
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _vram_peak_mb() -> float:
    """Return peak GPU memory in MB since the last reset, or ``0.0``."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():  # pragma: no cover - device-dependent
            peak_bytes = int(torch.cuda.max_memory_allocated())
            return round(peak_bytes / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def _infer_backend(payload: dict, environment: dict) -> str:
    """Resolve the backend name from runner payload, falling back to env."""
    backend = str(payload.get("backend") or "").strip().lower()
    if backend:
        return backend
    device = str(environment.get("device") or "").strip().lower()
    if device:
        # Normalise common device strings the gpu module returns.
        if device.startswith("cuda"):
            return "cuda"
        if device in {"cpu", "mps", "rocm", "directml", "xpu"}:
            return device
    return "cpu"


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
    _reset_vram_counter()
    start = time.perf_counter()
    success = False
    payload: dict = {}
    error_str = ""
    try:
        raw_payload = definition.runner(sample)
        # Coerce to dict — a runner that returns a list / tuple / None
        # would otherwise blow up the subsequent .get() calls with
        # AttributeError.
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        success = True
    except Exception as exc:
        error_str = repr(exc)
        logger.warning("ai_eval %s failed on sample %s: %s", feature_id, sample.sample_id, exc)

    elapsed = int((time.perf_counter() - start) * 1000)
    # F178 — pick up the new fields from the runner payload (the runner
    # is the only thing that knows the actual backend / VRAM ceiling /
    # baseline reference score).
    payload_vram = payload.get("vram_peak_mb")
    if payload_vram is None:
        vram_peak_mb = _vram_peak_mb()
    else:
        try:
            vram_peak_mb = max(0.0, float(payload_vram))
        except (TypeError, ValueError):
            vram_peak_mb = 0.0
    try:
        reference_score = max(0.0, float(payload.get("reference_score", 0.0) or 0.0))
    except (TypeError, ValueError):
        reference_score = 0.0
    backend = _infer_backend(payload, env)
    backend_choice_reason = str(payload.get("backend_choice_reason", "")).strip()

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
        vram_peak_mb=vram_peak_mb,
        reference_score=reference_score,
        backend=backend,
        backend_choice_reason=backend_choice_reason,
    )
    if persist:
        _append_result(result, eval_dir=eval_dir)
    return result


_history_io_lock = threading.Lock()


def _load_history(target: Path) -> List[dict]:
    """Read an eval-history file and return a sanitised list of dicts."""
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    # Filter out non-dict items so callers can use `.get()` without
    # AttributeError on a corrupted history file.
    return [item for item in raw if isinstance(item, dict)]


def _append_result(result: EvalResult, *, eval_dir: Optional[Path] = None) -> None:
    import tempfile  # local import keeps module-import cheap

    base = Path(eval_dir) if eval_dir else EVAL_DIR
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{result.feature_id}.json"

    with _history_io_lock:
        history = _load_history(target)
        history.append(result.as_dict())
        # Cap history to the most recent 200 results so the file doesn't grow
        # without bound across long-running installs.
        history = history[-200:]
        # Atomic write — mkstemp + fsync + os.replace defeats partial-flush
        # corruption that would otherwise lose every prior result on a
        # mid-write crash.
        parent = target.parent
        fd, tmp_path = tempfile.mkstemp(
            dir=str(parent),
            prefix=target.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
                f.write("\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


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

    history = _load_history(target)

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


# ---------------------------------------------------------------------------
# F178 — cross-backend comparison
# ---------------------------------------------------------------------------


def compare_backends(
    feature_id: str,
    *,
    eval_dir: Optional[Path] = None,
) -> dict:
    """Group persisted evaluations by backend and emit relative scores.

    Returns ``{"feature_id", "backends": [...], "best_latency", "best_quality"}``.
    Each backend entry carries the runs / success count / latency
    p50/p95 / mean quality / VRAM peak / and a normalised
    ``quality_vs_reference`` ratio (quality_mean / reference_score
    mean, capped at 1.5 for display sanity).

    The harness intentionally **never** picks a "winner" because the
    user knows whether they care about latency or quality. We emit the
    summary; the UI / panel renders the recommendation.
    """
    base = Path(eval_dir) if eval_dir else EVAL_DIR
    target = base / f"{feature_id}.json"
    if not target.exists():
        return {
            "feature_id": feature_id,
            "backends": [],
            "best_latency": None,
            "best_quality": None,
            "note": "no persisted results",
        }
    history = _load_history(target)

    by_backend: Dict[str, List[dict]] = {}
    for entry in history:
        backend = str(entry.get("backend") or "unknown") or "unknown"
        by_backend.setdefault(backend, []).append(entry)

    backends_summary: List[dict] = []
    for backend, runs in sorted(by_backend.items()):
        success_runs = [r for r in runs if r.get("success")]
        latencies = [int(r.get("latency_ms", 0) or 0) for r in success_runs]
        quality = [float(r.get("quality_score", 0.0) or 0.0) for r in success_runs]
        vram = [float(r.get("vram_peak_mb", 0.0) or 0.0) for r in success_runs]
        reference = [
            float(r.get("reference_score", 0.0) or 0.0)
            for r in success_runs
            if float(r.get("reference_score", 0.0) or 0.0) > 0
        ]
        quality_mean = (statistics.fmean(quality) if quality else None)
        reference_mean = (statistics.fmean(reference) if reference else None)
        ratio = None
        if quality_mean is not None and reference_mean and reference_mean > 0:
            ratio = round(min(1.5, quality_mean / reference_mean), 4)
        backends_summary.append({
            "backend": backend,
            "runs": len(runs),
            "successes": len(success_runs),
            "failures": len(runs) - len(success_runs),
            "latency_ms_p50": int(statistics.median(latencies)) if latencies else None,
            "latency_ms_p95": int(_percentile(latencies, 0.95)) if latencies else None,
            "quality_mean": quality_mean,
            "reference_mean": reference_mean,
            "quality_vs_reference": ratio,
            "vram_peak_mb_max": max(vram) if vram else 0.0,
            "vram_peak_mb_mean": (
                round(statistics.fmean(vram), 2) if vram else 0.0
            ),
            "latest_reason": (
                str(success_runs[-1].get("backend_choice_reason", ""))
                if success_runs else ""
            ),
        })

    best_latency = None
    best_quality = None
    if backends_summary:
        with_latency = [b for b in backends_summary if b["latency_ms_p50"] is not None]
        if with_latency:
            best_latency = min(with_latency, key=lambda b: b["latency_ms_p50"])["backend"]
        with_quality = [b for b in backends_summary if b["quality_mean"] is not None]
        if with_quality:
            best_quality = max(with_quality, key=lambda b: b["quality_mean"])["backend"]

    return {
        "feature_id": feature_id,
        "backends": backends_summary,
        "best_latency": best_latency,
        "best_quality": best_quality,
    }
