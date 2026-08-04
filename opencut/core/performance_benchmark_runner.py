"""Opt-in, provenance-first execution for the performance benchmark registry.

The registry in :mod:`opencut.core.performance_benchmarks` deliberately has no
optional model imports.  This module is the execution boundary used by the
benchmark CLI.  It keeps normal test and release paths lightweight while an
operator can explicitly run a selected set of local adapters and retain a
machine-readable receipt.

Adapters must never download weights or silently call a network service.  An
adapter either proves that its already-installed local prerequisites are
available, runs against the pinned synthetic fixture, or returns a truthful
``skipped`` result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import statistics
import subprocess
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from opencut import __version__
from opencut.core.performance_benchmarks import (
    BENCHMARK_SPECS,
    BenchmarkSpec,
    get_benchmark,
    validate_benchmark_registry,
)

RECEIPT_SCHEMA_VERSION = 1
RECEIPT_KIND = "opencut.performance-benchmark"
DEFAULT_WARMUP_RUNS = 1
DEFAULT_REPEATS = 3
DEFAULT_TOLERANCES: dict[str, float] = {
    "seconds_per_unit_relative": 0.15,
    "seconds_per_unit_absolute": 0.05,
    "memory_relative": 0.20,
    "quality_absolute": 0.01,
}


class BenchmarkOptInRequired(RuntimeError):
    """Raised when heavyweight benchmarking was not explicitly enabled."""


class BenchmarkUnavailable(RuntimeError):
    """Raised by an adapter when its local prerequisites are unavailable."""


@dataclass(frozen=True)
class BenchmarkFixture:
    """A deterministic fixture descriptor, without a bundled media payload."""

    fixture_id: str
    description: str
    license: str
    source: str
    payload: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            {
                "fixture_id": self.fixture_id,
                "description": self.description,
                "license": self.license,
                "source": self.source,
                "payload": self.payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "description": self.description,
            "license": self.license,
            "source": self.source,
            "sha256": self.sha256,
        }


BENCHMARK_FIXTURES: dict[str, BenchmarkFixture] = {
    "asr_transcription": BenchmarkFixture(
        fixture_id="synthetic-asr-v1",
        description="Deterministic speech transcript manifest for local ASR adapters",
        license="MIT",
        source="OpenCut synthetic fixture; no external download",
        payload={
            "sample_rate_hz": 16000,
            "duration_seconds": 30,
            "transcript": "OpenCut reproducible benchmark speech sample",
        },
    ),
    "ai_upscale": BenchmarkFixture(
        fixture_id="synthetic-upscale-v1",
        description="Deterministic low-resolution frame manifest for local upscaler adapters",
        license="MIT",
        source="OpenCut synthetic fixture; no external download",
        payload={"width": 16, "height": 16, "frames": 16, "pattern": "solid-black"},
    ),
    "declarative_compose": BenchmarkFixture(
        fixture_id="synthetic-compose-v1",
        description="Deterministic declarative render manifest for the FFmpeg control adapter",
        license="MIT",
        source="OpenCut synthetic fixture; no external download",
        payload={
            "duration_seconds": 30,
            "width": 64,
            "height": 64,
            "rate": 2,
            "pattern": "solid-black",
        },
    ),
    "tts_synthesis": BenchmarkFixture(
        fixture_id="synthetic-tts-v1",
        description="Deterministic text manifest for local TTS adapters",
        license="MIT",
        source="OpenCut synthetic fixture; no external download",
        payload={
            "target_duration_seconds": 10,
            "text": "OpenCut reproducible benchmark speech sample",
        },
    ),
}


@dataclass(frozen=True)
class AdapterContext:
    """Inputs made available to a backend adapter for one benchmark."""

    spec: BenchmarkSpec
    fixture: BenchmarkFixture
    seed: int
    allow_network: bool = False
    working_directory: Path | None = None


@dataclass(frozen=True)
class AdapterAvailability:
    available: bool
    reason: str = ""
    dependency_versions: Mapping[str, str] = field(default_factory=dict)


AdapterProbe = Callable[[AdapterContext], AdapterAvailability]
AdapterRun = Callable[[AdapterContext], Mapping[str, Any] | None]


@dataclass(frozen=True)
class BenchmarkAdapter:
    backend: str
    probe: AdapterProbe
    run: AdapterRun
    dependencies: tuple[str, ...] = ()


_ADAPTERS: dict[str, BenchmarkAdapter] = {}


def register_backend_adapter(
    backend: str,
    *,
    dependencies: Sequence[str] = (),
    probe: AdapterProbe | None = None,
) -> Callable[[AdapterRun], AdapterRun]:
    """Register an executable adapter without importing optional runtimes.

    Third-party integrations can use this decorator at explicit opt-in time.
    The built-in registry below supplies the safe control adapter and truthful
    optional-backend probes.
    """

    name = str(backend).strip()
    if not name:
        raise ValueError("backend name is required")

    def decorator(run: AdapterRun) -> AdapterRun:
        _ADAPTERS[name] = BenchmarkAdapter(
            backend=name,
            probe=probe or _package_probe(tuple(dependencies)),
            run=run,
            dependencies=tuple(dependencies),
        )
        return run

    return decorator


def _version_for(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"
    except Exception:
        return "unknown"


def _dependency_versions(names: Sequence[str]) -> dict[str, str]:
    return {name: _version_for(name) for name in names}


def _package_probe(packages: Sequence[str]) -> AdapterProbe:
    def probe(_context: AdapterContext) -> AdapterAvailability:
        missing = [package for package in packages if importlib.util.find_spec(package) is None]
        versions = _dependency_versions(packages)
        if missing:
            return AdapterAvailability(
                False,
                f"optional dependency unavailable: {', '.join(missing)}",
                versions,
            )
        return AdapterAvailability(
            False,
            "package is installed but no no-download fixture adapter is configured",
            versions,
        )

    return probe


def _unavailable_run(_context: AdapterContext) -> Mapping[str, Any]:
    raise BenchmarkUnavailable("adapter is not executable in this installation")


def _ffmpeg_probe(_context: AdapterContext) -> AdapterAvailability:
    try:
        from opencut.helpers import get_ffmpeg_path

        binary = get_ffmpeg_path()
        version = _ffmpeg_version(binary)
    except Exception as exc:
        return AdapterAvailability(False, f"FFmpeg unavailable: {exc}", {})
    return AdapterAvailability(True, f"FFmpeg {version or 'unknown'}", {"ffmpeg": version or "unknown"})


def _ffmpeg_version(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    first_line = (result.stdout or result.stderr or "").splitlines()
    if not first_line:
        return ""
    parts = first_line[0].split()
    try:
        return parts[2]
    except IndexError:
        return ""


def _run_ffmpeg_compose(context: AdapterContext) -> Mapping[str, Any]:
    from opencut.helpers import get_ffmpeg_path

    binary = get_ffmpeg_path()
    payload = context.fixture.payload
    duration = float(payload.get("duration_seconds", context.spec.sample_units))
    size = f"{int(payload.get('width', 64))}x{int(payload.get('height', 64))}"
    rate = str(int(payload.get("rate", 2)))
    command = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={size}:r={rate}:d={duration}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=context.spec.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out after {context.spec.timeout_seconds}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or "FFmpeg returned a non-zero exit code").strip()
        raise RuntimeError(detail[-1000:])
    return {
        "quality_metric": "synthetic_render_completed",
        "quality_score": 1.0,
        "quality_metrics": {
            "render_completed": 1.0,
            "exit_code": float(result.returncode),
        },
        "dependency_versions": {"ffmpeg": _ffmpeg_version(binary) or "unknown"},
    }


register_backend_adapter("ffmpeg-compose", probe=_ffmpeg_probe)(_run_ffmpeg_compose)

for _backend, _packages in {
    "openai-whisper": ("whisper",),
    "faster-whisper": ("faster_whisper",),
    "whisperx": ("whisperx",),
    "realesrgan": ("realesrgan",),
    "flashvsr": ("flashvsr",),
    "seedvr": ("seedvr",),
    "edge-tts": ("edge_tts",),
    "kokoro": ("kokoro",),
    "chatterbox": ("chatterbox",),
    "f5-tts": ("f5_tts",),
    "elevenlabs": ("elevenlabs",),
}.items():
    _ADAPTERS[_backend] = BenchmarkAdapter(
        backend=_backend,
        probe=_package_probe(_packages),
        run=_unavailable_run,
        dependencies=_packages,
    )


def list_backend_adapters() -> tuple[str, ...]:
    """Return executable/probed backend adapter names in stable order."""
    return tuple(sorted(_ADAPTERS))


def _memory_snapshot() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
        "memory_total_bytes": 0,
    }
    try:
        import psutil  # type: ignore

        snapshot["memory_total_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        from opencut import gpu

        snapshot["device"] = str(gpu.get_device())
        _ok, vram = gpu.check_vram(0)
        snapshot["vram_total_mb"] = int(vram.get("total_mb", 0) or 0)
    except Exception:
        snapshot["device"] = "unknown"
        snapshot["vram_total_mb"] = 0
    return snapshot


def _compatibility_key(environment: Mapping[str, Any]) -> str:
    comparable = {
        key: environment.get(key)
        for key in (
            "platform",
            "platform_release",
            "machine",
            "processor",
            "python",
            "cpu_count",
            "memory_total_bytes",
            "device",
            "vram_total_mb",
        )
    }
    canonical = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _mean_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = sorted({key for entry in metrics for key, value in entry.items() if isinstance(value, (int, float))})
    return {
        key: round(statistics.fmean(float(entry[key]) for entry in metrics if key in entry), 6)
        for key in keys
    }


def _error_result(
    *,
    spec: BenchmarkSpec,
    backend: str,
    fixture: BenchmarkFixture,
    environment: Mapping[str, Any],
    seed: int,
    warmup_runs: int,
    repeats: int,
    dependencies: Mapping[str, str],
    status: str,
    reason: str,
) -> dict[str, Any]:
    result = {
        "benchmark_id": spec.benchmark_id,
        "backend": backend,
        "status": status,
        "fixture": fixture.as_dict(),
        "seed": seed,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "dependencies": dict(dependencies),
        "timing": {
            "metric_name": spec.metric_name,
            "units": spec.sample_units,
            "samples_seconds": [],
            "seconds_per_unit": None,
            "p50_seconds_per_unit": None,
            "p95_seconds_per_unit": None,
        },
        "memory": {"rss_before_bytes": 0, "rss_peak_bytes": 0, "python_allocated_peak_bytes": 0},
        "quality_metrics": {},
        "quality_score": None,
        "error": reason if status == "failed" else "",
        "skip_reason": reason if status == "skipped" else "",
        "environment": dict(environment),
    }
    return result


def _run_one(
    spec: BenchmarkSpec,
    backend: str,
    *,
    fixture: BenchmarkFixture,
    environment: Mapping[str, Any],
    seed: int,
    warmup_runs: int,
    repeats: int,
    allow_network: bool,
    working_directory: Path,
) -> dict[str, Any]:
    adapter = _ADAPTERS.get(backend)
    if adapter is None:
        return _error_result(
            spec=spec,
            backend=backend,
            fixture=fixture,
            environment=environment,
            seed=seed,
            warmup_runs=warmup_runs,
            repeats=repeats,
            dependencies={},
            status="skipped",
            reason="no adapter registered",
        )

    context = AdapterContext(spec, fixture, seed, allow_network, working_directory)
    try:
        availability = adapter.probe(context)
    except Exception as exc:
        availability = AdapterAvailability(False, f"adapter probe failed: {exc}")
    dependencies = dict(availability.dependency_versions)
    dependencies.update(_dependency_versions(adapter.dependencies))
    if not availability.available:
        return _error_result(
            spec=spec,
            backend=backend,
            fixture=fixture,
            environment=environment,
            seed=seed,
            warmup_runs=warmup_runs,
            repeats=repeats,
            dependencies=dependencies,
            status="skipped",
            reason=availability.reason or "adapter unavailable",
        )

    random.seed(seed)
    try:
        for _ in range(warmup_runs):
            adapter.run(context)
    except BenchmarkUnavailable as exc:
        return _error_result(
            spec=spec,
            backend=backend,
            fixture=fixture,
            environment=environment,
            seed=seed,
            warmup_runs=warmup_runs,
            repeats=repeats,
            dependencies=dependencies,
            status="skipped",
            reason=str(exc),
        )
    except Exception as exc:
        return _error_result(
            spec=spec,
            backend=backend,
            fixture=fixture,
            environment=environment,
            seed=seed,
            warmup_runs=warmup_runs,
            repeats=repeats,
            dependencies=dependencies,
            status="failed",
            reason=repr(exc),
        )

    samples: list[float] = []
    rss_peaks: list[int] = []
    traced_peaks: list[int] = []
    quality_payloads: list[Mapping[str, Any]] = []
    quality_scores: list[float] = []
    quality_metric = ""
    model_version = ""
    rss_before = _memory_snapshot()
    try:
        for _ in range(repeats):
            tracemalloc.start()
            start = time.perf_counter()
            payload = adapter.run(context) or {}
            elapsed = max(0.0, time.perf_counter() - start)
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            samples.append(round(elapsed, 6))
            rss_peaks.append(max(_memory_snapshot(), rss_before))
            if isinstance(payload, Mapping):
                metrics = payload.get("quality_metrics")
                if isinstance(metrics, Mapping):
                    quality_payloads.append(metrics)
                try:
                    if payload.get("quality_score") is not None:
                        quality_scores.append(float(payload["quality_score"]))
                except (TypeError, ValueError):
                    pass
                quality_metric = str(payload.get("quality_metric") or quality_metric)
                model_version = str(payload.get("model_version") or model_version)
                supplied_dependencies = payload.get("dependency_versions")
                if isinstance(supplied_dependencies, Mapping):
                    dependencies.update({str(k): str(v) for k, v in supplied_dependencies.items()})
            traced_peaks.append(int(peak))
    except BenchmarkUnavailable as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return _error_result(
            spec=spec,
            backend=backend,
            fixture=fixture,
            environment=environment,
            seed=seed,
            warmup_runs=warmup_runs,
            repeats=repeats,
            dependencies=dependencies,
            status="skipped",
            reason=str(exc),
        )
    except Exception as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return _error_result(
            spec=spec,
            backend=backend,
            fixture=fixture,
            environment=environment,
            seed=seed,
            warmup_runs=warmup_runs,
            repeats=repeats,
            dependencies=dependencies,
            status="failed",
            reason=repr(exc),
        )

    per_unit = [sample / spec.sample_units for sample in samples]
    return {
        "benchmark_id": spec.benchmark_id,
        "backend": backend,
        "status": "success",
        "fixture": fixture.as_dict(),
        "seed": seed,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "dependencies": dependencies,
        "model_version": model_version,
        "timing": {
            "metric_name": spec.metric_name,
            "units": spec.sample_units,
            "samples_seconds": samples,
            "seconds_per_unit": round(statistics.fmean(per_unit), 6),
            "p50_seconds_per_unit": round(_percentile(per_unit, 0.50), 6),
            "p95_seconds_per_unit": round(_percentile(per_unit, 0.95), 6),
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "rss_peak_bytes": max(rss_peaks or [rss_before]),
            "python_allocated_peak_bytes": max(traced_peaks or [0]),
        },
        "quality_metric": quality_metric,
        "quality_score": round(statistics.fmean(quality_scores), 6) if quality_scores else None,
        "quality_metrics": _mean_metrics(quality_payloads),
        "environment": dict(environment),
        "error": "",
        "skip_reason": "",
    }


def _validate_selection(
    benchmark_ids: Sequence[str] | None,
    backends: Sequence[str] | None,
) -> list[tuple[BenchmarkSpec, list[str]]]:
    errors = validate_benchmark_registry()
    if errors:
        raise ValueError("invalid benchmark registry: " + "; ".join(errors))
    selected_ids = list(benchmark_ids or [spec.benchmark_id for spec in BENCHMARK_SPECS])
    selected: list[tuple[BenchmarkSpec, list[str]]] = []
    requested_backends = [str(backend).strip() for backend in (backends or []) if str(backend).strip()]
    for benchmark_id in selected_ids:
        try:
            spec = get_benchmark(str(benchmark_id).strip())
        except KeyError as exc:
            raise ValueError(f"unknown benchmark: {benchmark_id}") from exc
        chosen = [backend for backend in spec.backends if not requested_backends or backend in requested_backends]
        unknown = [backend for backend in requested_backends if backend not in spec.backends]
        if unknown:
            raise ValueError(
                f"backend(s) {', '.join(unknown)} are not registered for {spec.benchmark_id}"
            )
        if not chosen:
            raise ValueError(f"no requested backends selected for {spec.benchmark_id}")
        selected.append((spec, chosen))
    return selected


def run_benchmarks(
    benchmark_ids: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    *,
    seed: int = 0,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    repeats: int = DEFAULT_REPEATS,
    allow_network: bool = False,
    output_path: Path | None = None,
    require_opt_in: bool = True,
) -> dict[str, Any]:
    """Run selected adapters and return a JSON-serialisable receipt.

    ``require_opt_in`` is exposed for unit tests and embedding applications;
    the CLI leaves it enabled so a normal command cannot accidentally download
    models or contact a cloud provider.
    """
    if require_opt_in:
        from opencut.core.performance_benchmarks import should_run_benchmarks

        if not should_run_benchmarks():
            raise BenchmarkOptInRequired(
                "set OPENCUT_RUN_PERF_BENCHMARKS=1 to run opt-in benchmarks"
            )
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    selected = _validate_selection(benchmark_ids, backends)
    fixture_missing = [spec.benchmark_id for spec, _ in selected if spec.benchmark_id not in BENCHMARK_FIXTURES]
    if fixture_missing:
        raise ValueError("missing pinned fixtures: " + ", ".join(fixture_missing))

    environment = _hardware_snapshot()
    environment["compatibility_key"] = _compatibility_key(environment)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    base_dir = Path(output_path).parent if output_path else Path.cwd()
    base_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for spec, selected_backends in selected:
        fixture = BENCHMARK_FIXTURES[spec.benchmark_id]
        for backend in selected_backends:
            results.append(
                _run_one(
                    spec,
                    backend,
                    fixture=fixture,
                    environment=environment,
                    seed=seed,
                    warmup_runs=warmup_runs,
                    repeats=repeats,
                    allow_network=allow_network,
                    working_directory=base_dir,
                )
            )

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "created_at": created_at,
        "opencut_version": __version__,
        "seed": seed,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "allow_network": bool(allow_network),
        "environment": environment,
        "tolerances": dict(DEFAULT_TOLERANCES),
        "benchmarks": [spec.as_dict() for spec, _ in selected],
        "results": results,
    }
    if output_path:
        write_receipt(receipt, output_path)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Return structural errors for a benchmark receipt."""
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if receipt.get("kind") != RECEIPT_KIND:
        errors.append("unexpected kind")
    if not isinstance(receipt.get("environment"), Mapping):
        errors.append("environment must be an object")
    elif not receipt["environment"].get("compatibility_key"):
        errors.append("environment.compatibility_key is required")
    if not isinstance(receipt.get("tolerances"), Mapping):
        errors.append("tolerances must be an object")
    results = receipt.get("results")
    if not isinstance(results, list):
        errors.append("results must be a list")
    else:
        for index, result in enumerate(results):
            if not isinstance(result, Mapping):
                errors.append(f"results[{index}] must be an object")
                continue
            if result.get("status") not in {"success", "skipped", "failed"}:
                errors.append(f"results[{index}].status is invalid")
            fixture = result.get("fixture")
            if not isinstance(fixture, Mapping) or not fixture.get("sha256") or not fixture.get("license"):
                errors.append(f"results[{index}].fixture must include sha256 and license")
    return errors


def load_receipt(path: Path) -> dict[str, Any]:
    """Load and strictly validate a receipt from disk."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark receipt must be a JSON object")
    errors = validate_receipt(payload)
    if errors:
        raise ValueError("invalid benchmark receipt: " + "; ".join(errors))
    return payload


def write_receipt(receipt: Mapping[str, Any], path: Path) -> Path:
    """Atomically write a validated receipt."""
    errors = validate_receipt(receipt)
    if errors:
        raise ValueError("refusing to write invalid receipt: " + "; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _numeric_metric(result: Mapping[str, Any], key: str) -> float | None:
    value = result.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def compare_receipts(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare receipts only when their captured environments are compatible."""
    current_errors = validate_receipt(current)
    baseline_errors = validate_receipt(baseline)
    if current_errors or baseline_errors:
        raise ValueError(
            "cannot compare invalid receipts: "
            + "; ".join(current_errors + baseline_errors)
        )
    current_key = current["environment"].get("compatibility_key")
    baseline_key = baseline["environment"].get("compatibility_key")
    if current_key != baseline_key:
        return {
            "status": "incompatible",
            "compatible": False,
            "reason": "hardware or software compatibility keys differ",
            "baseline_compatibility_key": baseline_key,
            "current_compatibility_key": current_key,
            "comparisons": [],
            "regressions": [],
        }

    limits = dict(DEFAULT_TOLERANCES)
    limits.update({str(key): float(value) for key, value in (tolerances or {}).items()})
    baseline_results = {
        (str(item.get("benchmark_id")), str(item.get("backend"))): item
        for item in baseline["results"]
        if isinstance(item, Mapping)
    }
    comparisons: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for item in current["results"]:
        if not isinstance(item, Mapping):
            continue
        key = (str(item.get("benchmark_id")), str(item.get("backend")))
        previous = baseline_results.get(key)
        if not previous or item.get("status") != "success" or previous.get("status") != "success":
            comparisons.append({"benchmark_id": key[0], "backend": key[1], "status": "not-comparable"})
            continue
        current_timing = _numeric_metric(item.get("timing") or {}, "seconds_per_unit")
        baseline_timing = _numeric_metric(previous.get("timing") or {}, "seconds_per_unit")
        timing_regression = False
        timing_delta = None
        timing_limit = None
        if current_timing is not None and baseline_timing is not None:
            timing_delta = current_timing - baseline_timing
            timing_limit = max(
                limits["seconds_per_unit_absolute"],
                baseline_timing * limits["seconds_per_unit_relative"],
            )
            timing_regression = timing_delta > timing_limit
        current_quality = _numeric_metric(item, "quality_score")
        baseline_quality = _numeric_metric(previous, "quality_score")
        quality_regression = False
        quality_delta = None
        if current_quality is not None and baseline_quality is not None:
            quality_delta = current_quality - baseline_quality
            quality_regression = quality_delta < -limits["quality_absolute"]
        comparison = {
            "benchmark_id": key[0],
            "backend": key[1],
            "status": "regression" if timing_regression or quality_regression else "ok",
            "timing_delta_seconds_per_unit": timing_delta,
            "timing_tolerance_seconds_per_unit": timing_limit,
            "quality_delta": quality_delta,
        }
        comparisons.append(comparison)
        if comparison["status"] == "regression":
            regressions.append(comparison)

    status = "regression" if regressions else ("ok" if comparisons else "no-comparable-results")
    return {
        "status": status,
        "compatible": True,
        "tolerances": limits,
        "comparisons": comparisons,
        "regressions": regressions,
    }


__all__ = [
    "BENCHMARK_FIXTURES",
    "DEFAULT_TOLERANCES",
    "RECEIPT_KIND",
    "RECEIPT_SCHEMA_VERSION",
    "AdapterAvailability",
    "AdapterContext",
    "BenchmarkAdapter",
    "BenchmarkFixture",
    "BenchmarkOptInRequired",
    "BenchmarkUnavailable",
    "compare_receipts",
    "list_backend_adapters",
    "load_receipt",
    "register_backend_adapter",
    "run_benchmarks",
    "validate_receipt",
    "write_receipt",
]
