"""Coverage for the opt-in performance benchmark runner and receipts."""

from __future__ import annotations

import copy

import pytest

from opencut.core import performance_benchmark_runner as runner


def _fixed_environment() -> dict:
    return {
        "platform": "Windows",
        "platform_release": "test",
        "machine": "AMD64",
        "processor": "test-cpu",
        "python": "3.12.0",
        "cpu_count": 8,
        "memory_total_bytes": 16 * 1024**3,
        "device": "cpu",
        "vram_total_mb": 0,
        "compatibility_key": "same-host",
    }


def test_fixture_descriptors_are_pinned_and_licensed():
    assert set(runner.BENCHMARK_FIXTURES) == {
        "asr_transcription",
        "ai_upscale",
        "declarative_compose",
        "tts_synthesis",
    }
    for fixture in runner.BENCHMARK_FIXTURES.values():
        assert fixture.license == "MIT"
        assert len(fixture.sha256) == 64
        assert fixture.sha256 == fixture.sha256


def test_runner_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("OPENCUT_RUN_PERF_BENCHMARKS", raising=False)
    with pytest.raises(runner.BenchmarkOptInRequired, match="OPENCUT_RUN_PERF_BENCHMARKS=1"):
        runner.run_benchmarks(
            ["declarative_compose"],
            ["ffmpeg-compose"],
        )


def test_runner_records_provenance_metrics_and_writes_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_hardware_snapshot", _fixed_environment)
    adapter = runner.BenchmarkAdapter(
        backend="faster-whisper",
        probe=lambda _context: runner.AdapterAvailability(
            True,
            dependency_versions={"test-package": "1.2.3"},
        ),
        run=lambda _context: {
            "quality_metric": "accuracy",
            "quality_score": 0.91,
            "quality_metrics": {"accuracy": 0.91},
            "model_version": "fixture-model-1",
        },
        dependencies=("definitely-not-installed",),
    )
    monkeypatch.setitem(runner._ADAPTERS, "faster-whisper", adapter)

    receipt_path = tmp_path / "receipt.json"
    receipt = runner.run_benchmarks(
        ["asr_transcription"],
        ["faster-whisper"],
        seed=7,
        warmup_runs=1,
        repeats=2,
        output_path=receipt_path,
        require_opt_in=False,
    )

    assert receipt_path.exists()
    loaded = runner.load_receipt(receipt_path)
    assert loaded == receipt
    assert runner.validate_receipt(receipt) == []
    result = receipt["results"][0]
    assert result["status"] == "success"
    assert result["seed"] == 7
    assert result["warmup_runs"] == 1
    assert result["repeats"] == 2
    assert result["fixture"]["license"] == "MIT"
    assert result["fixture"]["sha256"]
    assert result["dependencies"]["test-package"] == "1.2.3"
    assert result["model_version"] == "fixture-model-1"
    assert result["quality_metrics"]["accuracy"] == pytest.approx(0.91)
    assert result["timing"]["seconds_per_unit"] >= 0
    assert result["memory"]["rss_peak_bytes"] >= 0


def test_unavailable_optional_backend_is_recorded_as_skip(monkeypatch):
    monkeypatch.setattr(runner, "_hardware_snapshot", _fixed_environment)
    receipt = runner.run_benchmarks(
        ["asr_transcription"],
        ["faster-whisper"],
        require_opt_in=False,
    )
    result = receipt["results"][0]
    assert result["status"] == "skipped"
    assert result["skip_reason"]
    assert result["error"] == ""
    assert result["fixture"]["license"]


def test_comparison_uses_declared_tolerances_and_rejects_other_hosts():
    baseline = {
        "schema_version": runner.RECEIPT_SCHEMA_VERSION,
        "kind": runner.RECEIPT_KIND,
        "environment": _fixed_environment(),
        "tolerances": dict(runner.DEFAULT_TOLERANCES),
        "results": [
            {
                "benchmark_id": "declarative_compose",
                "backend": "ffmpeg-compose",
                "status": "success",
                "fixture": {"sha256": "a" * 64, "license": "MIT"},
                "timing": {"seconds_per_unit": 1.0},
                "quality_score": 1.0,
            }
        ],
    }
    current = copy.deepcopy(baseline)
    current["results"][0]["timing"]["seconds_per_unit"] = 1.10
    assert runner.compare_receipts(current, baseline)["status"] == "ok"
    current["results"][0]["timing"]["seconds_per_unit"] = 1.30
    comparison = runner.compare_receipts(current, baseline)
    assert comparison["status"] == "regression"
    assert len(comparison["regressions"]) == 1

    other_host = copy.deepcopy(baseline)
    other_host["environment"]["compatibility_key"] = "different-host"
    comparison = runner.compare_receipts(current, other_host)
    assert comparison["status"] == "incompatible"
    assert comparison["compatible"] is False
    assert comparison["regressions"] == []


def test_tool_list_command_is_machine_readable():
    from opencut.tools.run_performance_benchmarks import main

    # The tool's output is intentionally JSON so release smoke can consume it.
    assert main(["list", "--json"]) == 0
