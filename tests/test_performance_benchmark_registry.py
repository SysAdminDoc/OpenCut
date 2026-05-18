"""F214 contract tests for ML / compose / TTS performance benchmarks."""

from __future__ import annotations

import pytest

from opencut.core import performance_benchmarks as pb

EXPECTED_BENCHMARK_IDS = (
    "asr_transcription",
    "ai_upscale",
    "declarative_compose",
    "tts_synthesis",
)


def test_f214_benchmark_inventory_is_pinned():
    assert pb.benchmark_ids() == EXPECTED_BENCHMARK_IDS


def test_f214_backend_matrix_covers_required_surfaces():
    matrix = pb.backend_matrix()
    assert matrix["asr_transcription"] == [
        "openai-whisper",
        "faster-whisper",
        "whisperx",
    ]
    assert matrix["ai_upscale"] == ["realesrgan", "flashvsr", "seedvr"]
    assert matrix["declarative_compose"] == ["ffmpeg-compose"]
    assert matrix["tts_synthesis"] == [
        "edge-tts",
        "kokoro",
        "chatterbox",
        "f5-tts",
        "elevenlabs",
    ]


def test_f214_registry_schema_is_valid():
    assert pb.validate_benchmark_registry() == []
    for spec in pb.list_benchmarks():
        assert spec.metric_name.startswith("wall_clock_seconds_per_")
        assert spec.sample_units > 0
        assert spec.timeout_seconds >= 180
        assert "f214" in spec.tags


def test_should_run_benchmarks_requires_explicit_env():
    assert pb.should_run_benchmarks({}) is False
    assert pb.should_run_benchmarks({pb.PERF_BENCHMARK_ENV: "0"}) is False
    assert pb.should_run_benchmarks({pb.PERF_BENCHMARK_ENV: "1"}) is True


def test_measure_backend_normalises_wall_clock_units():
    spec = pb.get_benchmark("asr_transcription")
    ticks = iter([10.0, 25.0])

    result = pb.measure_backend(
        spec,
        "faster-whisper",
        lambda: None,
        units=30.0,
        clock=lambda: next(ticks),
    )

    assert result.success is True
    assert result.elapsed_seconds == 15.0
    assert result.seconds_per_unit == 0.5
    assert result.metric_name == "wall_clock_seconds_per_source_second"
    assert result.within_budget is True


def test_measure_backend_records_failures_without_raising():
    spec = pb.get_benchmark("tts_synthesis")
    ticks = iter([0.0, 2.0])

    def _broken():
        raise RuntimeError("missing weights")

    result = pb.measure_backend(
        spec,
        "kokoro",
        _broken,
        units=10.0,
        clock=lambda: next(ticks),
    )

    assert result.success is False
    assert result.seconds_per_unit == 0.2
    assert result.within_budget is False
    assert "missing weights" in result.error


def test_measure_backend_rejects_unknown_backend():
    spec = pb.get_benchmark("ai_upscale")
    with pytest.raises(ValueError, match="not registered"):
        pb.measure_backend(spec, "waifu2x", lambda: None)


def test_measure_backend_rejects_zero_units():
    spec = pb.get_benchmark("declarative_compose")
    with pytest.raises(ValueError, match="units"):
        pb.measure_backend(spec, "ffmpeg-compose", lambda: None, units=0)
