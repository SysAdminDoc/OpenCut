"""Tests for the AI evaluation harness (F120)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencut.core import ai_eval_harness as eh


@pytest.fixture(autouse=True)
def _clean_registry():
    eh.clear_registry()
    yield
    eh.clear_registry()


def test_register_evaluation_records_definition():
    @eh.register_evaluation(
        "demo.feature",
        description="demo evaluator",
        sample_type="audio",
        metric_name="rmse",
    )
    def runner(sample):  # noqa: ANN001
        return {"quality_score": 0.42}

    definition = eh.get_evaluation("demo.feature")
    assert definition is not None
    assert definition.description == "demo evaluator"
    assert definition.metric_name == "rmse"

    # Second registration with same id raises.
    with pytest.raises(ValueError):
        @eh.register_evaluation("demo.feature")
        def _other(sample):  # noqa: ANN001
            return {}


def test_run_evaluation_records_success(tmp_path):
    @eh.register_evaluation("demo.audio")
    def runner(sample):  # noqa: ANN001
        return {"quality_score": 0.9, "quality_metric": "sdr"}

    sample = eh.EvalSample(sample_id="sample-1", sample_type="audio")
    result = eh.run_evaluation("demo.audio", sample, eval_dir=tmp_path)

    assert result.success is True
    assert result.quality_score == 0.9
    assert result.quality_metric == "sdr"
    assert result.latency_ms >= 0

    history = json.loads((tmp_path / "demo.audio.json").read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert history[-1]["sample_id"] == "sample-1"


def test_run_evaluation_captures_failures_without_raising(tmp_path):
    @eh.register_evaluation("demo.broken")
    def runner(sample):  # noqa: ANN001
        raise RuntimeError("kaboom")

    sample = eh.EvalSample(sample_id="x")
    result = eh.run_evaluation("demo.broken", sample, eval_dir=tmp_path)

    assert result.success is False
    assert "kaboom" in result.error
    assert result.latency_ms >= 0


def test_run_evaluation_raises_for_missing_feature(tmp_path):
    sample = eh.EvalSample(sample_id="x")
    with pytest.raises(KeyError):
        eh.run_evaluation("does.not.exist", sample, eval_dir=tmp_path)


def test_summarise_results_returns_percentiles(tmp_path):
    @eh.register_evaluation("demo.summary")
    def runner(sample):  # noqa: ANN001
        return {"quality_score": 0.7}

    sample = eh.EvalSample(sample_id="a")
    for _ in range(5):
        eh.run_evaluation("demo.summary", sample, eval_dir=tmp_path)

    summary = eh.summarise_results("demo.summary", eval_dir=tmp_path)
    assert summary["runs"] == 5
    assert summary["successes"] == 5
    assert summary["latency_ms_p50"] is not None
    assert summary["latency_ms_p95"] is not None
    assert summary["quality_mean"] == pytest.approx(0.7)


def test_summarise_results_handles_missing_history(tmp_path):
    summary = eh.summarise_results("nothing.here", eval_dir=tmp_path)
    assert summary == {"feature_id": "nothing.here", "runs": 0, "history": []}


def test_history_is_capped_to_recent_runs(tmp_path):
    @eh.register_evaluation("demo.cap")
    def runner(sample):  # noqa: ANN001
        return {"quality_score": 0.1}

    # Pre-seed a fat history file.
    sample_dict = eh.EvalResult(
        feature_id="demo.cap", sample_id="seed", success=True, latency_ms=1
    ).as_dict()
    (tmp_path / "demo.cap.json").write_text(
        json.dumps([dict(sample_dict, sample_id=f"seed-{i}") for i in range(210)]),
        encoding="utf-8",
    )

    eh.run_evaluation("demo.cap", eh.EvalSample(sample_id="new"), eval_dir=tmp_path)
    history = json.loads((tmp_path / "demo.cap.json").read_text(encoding="utf-8"))
    assert len(history) == 200
    assert history[-1]["sample_id"] == "new"


def test_environment_snapshot_includes_python_version():
    sample = eh.EvalSample(sample_id="x")

    @eh.register_evaluation("demo.env")
    def runner(s):  # noqa: ANN001
        return {}

    result = eh.run_evaluation("demo.env", sample, persist=False)
    assert "python" in result.environment
    assert "platform" in result.environment


def test_list_evaluations_returns_registered_definitions():
    @eh.register_evaluation("demo.a", description="A")
    def _a(s):  # noqa: ANN001
        return {}

    @eh.register_evaluation("demo.b", description="B")
    def _b(s):  # noqa: ANN001
        return {}

    feature_ids = {d.feature_id for d in eh.list_evaluations()}
    assert feature_ids == {"demo.a", "demo.b"}
