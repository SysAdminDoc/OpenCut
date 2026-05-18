"""Performance benchmark registry for ML, compose, and TTS surfaces.

F214 extends the F128 regression-gate idea beyond FFmpeg filter syntax.
Normal CI should not download model weights or hit cloud TTS endpoints, so
this module keeps the benchmark contract deterministic: it names the required
surfaces, backends, sample sizes, and throughput metrics, and provides a small
measurement primitive that opt-in benchmark runners can call.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence

PERF_BENCHMARK_ENV = "OPENCUT_RUN_PERF_BENCHMARKS"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Definition of a benchmark OpenCut expects release candidates to run."""

    benchmark_id: str
    title: str
    category: str
    backends: tuple[str, ...]
    metric_name: str
    source_unit: str
    sample_units: float
    advisory_budget_seconds_per_unit: float
    timeout_seconds: int
    sample_description: str
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["backends"] = list(self.backends)
        data["tags"] = list(self.tags)
        return data


@dataclass(frozen=True)
class BenchmarkResult:
    """Single backend benchmark measurement."""

    benchmark_id: str
    backend: str
    success: bool
    elapsed_seconds: float
    units: float
    seconds_per_unit: float
    metric_name: str
    advisory_budget_seconds_per_unit: float
    within_budget: bool
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        benchmark_id="asr_transcription",
        title="Whisper-family ASR throughput",
        category="asr",
        backends=("openai-whisper", "faster-whisper", "whisperx"),
        metric_name="wall_clock_seconds_per_source_second",
        source_unit="source_second",
        sample_units=30.0,
        advisory_budget_seconds_per_unit=1.0,
        timeout_seconds=180,
        sample_description="30 seconds of mono 16 kHz speech from the F176 LibriTTS/LRS3 eval catalogue",
        notes="Run each installed ASR backend without downloading weights during CI.",
        tags=("ml", "audio", "captions", "f214"),
    ),
    BenchmarkSpec(
        benchmark_id="ai_upscale",
        title="AI upscaler throughput",
        category="upscale",
        backends=("realesrgan", "flashvsr", "seedvr"),
        metric_name="wall_clock_seconds_per_source_frame",
        source_unit="source_frame",
        sample_units=16.0,
        advisory_budget_seconds_per_unit=1.5,
        timeout_seconds=600,
        sample_description="16-frame 720p clip from the F176 REDS/DAVIS eval catalogue",
        notes="Backends may skip when model weights are absent; recorded runs still share the same metric.",
        tags=("ml", "video", "upscale", "f214"),
    ),
    BenchmarkSpec(
        benchmark_id="declarative_compose",
        title="Declarative compose render throughput",
        category="compose",
        backends=("ffmpeg-compose",),
        metric_name="wall_clock_seconds_per_source_second",
        source_unit="source_second",
        sample_units=30.0,
        advisory_budget_seconds_per_unit=0.5,
        timeout_seconds=240,
        sample_description="30-second declarative_compose spec with title/color/video clips and one transition",
        notes="This is the non-ML control benchmark for the same release candidate.",
        tags=("ffmpeg", "compose", "f214"),
    ),
    BenchmarkSpec(
        benchmark_id="tts_synthesis",
        title="TTS synthesis throughput",
        category="tts",
        backends=("edge-tts", "kokoro", "chatterbox", "f5-tts", "elevenlabs"),
        metric_name="wall_clock_seconds_per_synthesised_second",
        source_unit="synthesised_second",
        sample_units=10.0,
        advisory_budget_seconds_per_unit=1.0,
        timeout_seconds=240,
        sample_description="Text prompt calibrated to approximately 10 seconds of generated speech",
        notes="Cloud backends require explicit operator credentials; local backends skip if weights are absent.",
        tags=("ml", "tts", "audio", "f214"),
    ),
)

_BENCHMARK_BY_ID = {spec.benchmark_id: spec for spec in BENCHMARK_SPECS}


def list_benchmarks() -> tuple[BenchmarkSpec, ...]:
    """Return all benchmark specs in deterministic order."""
    return BENCHMARK_SPECS


def benchmark_ids() -> tuple[str, ...]:
    """Return the benchmark IDs in deterministic order."""
    return tuple(spec.benchmark_id for spec in BENCHMARK_SPECS)


def get_benchmark(benchmark_id: str) -> BenchmarkSpec:
    """Return a benchmark spec or raise ``KeyError`` for unknown IDs."""
    return _BENCHMARK_BY_ID[benchmark_id]


def backend_matrix() -> dict[str, list[str]]:
    """Return a JSON-friendly benchmark -> backend matrix."""
    return {spec.benchmark_id: list(spec.backends) for spec in BENCHMARK_SPECS}


def should_run_benchmarks(env: Mapping[str, str] | None = None) -> bool:
    """True when heavyweight benchmark execution is explicitly enabled."""
    if env is None:
        import os

        env = os.environ
    return str(env.get(PERF_BENCHMARK_ENV, "")).strip() == "1"


def measure_backend(
    spec: BenchmarkSpec,
    backend: str,
    runner: Callable[[], object],
    *,
    units: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Measure one backend and return a normalised throughput result.

    ``runner`` is deliberately supplied by the caller. This keeps the registry
    free of imports that could download weights or contact cloud APIs during
    normal test collection.
    """
    backend = str(backend).strip()
    if backend not in spec.backends:
        raise ValueError(
            f"backend {backend!r} is not registered for {spec.benchmark_id!r}"
        )
    measured_units = float(spec.sample_units if units is None else units)
    if measured_units <= 0:
        raise ValueError("benchmark units must be > 0")

    started = clock()
    success = False
    error = ""
    try:
        runner()
        success = True
    except Exception as exc:  # pragma: no cover - exercised by tests
        error = repr(exc)
    elapsed = max(0.0, float(clock() - started))
    seconds_per_unit = elapsed / measured_units
    return BenchmarkResult(
        benchmark_id=spec.benchmark_id,
        backend=backend,
        success=success,
        elapsed_seconds=round(elapsed, 6),
        units=measured_units,
        seconds_per_unit=round(seconds_per_unit, 6),
        metric_name=spec.metric_name,
        advisory_budget_seconds_per_unit=spec.advisory_budget_seconds_per_unit,
        within_budget=success and seconds_per_unit <= spec.advisory_budget_seconds_per_unit,
        error=error,
    )


def validate_benchmark_registry(
    specs: Sequence[BenchmarkSpec] = BENCHMARK_SPECS,
) -> list[str]:
    """Return human-readable registry errors; empty list means valid."""
    errors: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.benchmark_id in seen:
            errors.append(f"duplicate benchmark_id: {spec.benchmark_id}")
        seen.add(spec.benchmark_id)
        if not spec.backends:
            errors.append(f"{spec.benchmark_id}: at least one backend required")
        if len(set(spec.backends)) != len(spec.backends):
            errors.append(f"{spec.benchmark_id}: duplicate backend")
        if spec.sample_units <= 0:
            errors.append(f"{spec.benchmark_id}: sample_units must be > 0")
        if spec.advisory_budget_seconds_per_unit <= 0:
            errors.append(
                f"{spec.benchmark_id}: advisory_budget_seconds_per_unit must be > 0"
            )
        if spec.timeout_seconds <= 0:
            errors.append(f"{spec.benchmark_id}: timeout_seconds must be > 0")
        if not spec.metric_name.startswith("wall_clock_seconds_per_"):
            errors.append(f"{spec.benchmark_id}: metric_name must be wall-clock normalised")
    return errors


__all__ = [
    "BENCHMARK_SPECS",
    "PERF_BENCHMARK_ENV",
    "BenchmarkResult",
    "BenchmarkSpec",
    "backend_matrix",
    "benchmark_ids",
    "get_benchmark",
    "list_benchmarks",
    "measure_backend",
    "should_run_benchmarks",
    "validate_benchmark_registry",
]
