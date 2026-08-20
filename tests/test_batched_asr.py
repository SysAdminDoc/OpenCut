"""F336 — batched faster-whisper inference for long files.

Upstream's BatchedInferencePipeline is roughly 4x on long audio, and long audio
is where transcription cost actually lands. Batching short clips buys little and
spends VRAM, so the decision is duration-driven; these tests pin the policy, the
opt-out, and that an older wrapper degrades in speed instead of failing the job.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencut.core import captions  # noqa: E402
from opencut.core.captions import (  # noqa: E402
    BATCH_SIZE_MAX,
    plan_batched_inference,
)
from opencut.utils.config import CaptionConfig  # noqa: E402


def _plan(monkeypatch, duration, **cfg):
    monkeypatch.setattr(captions, "_audio_duration_seconds", lambda _p: duration)
    return plan_batched_inference(CaptionConfig(**cfg), "audio.wav")


class TestBatchPolicy:
    def test_long_file_batches(self, monkeypatch):
        plan = _plan(monkeypatch, 3600.0)
        assert plan["batched"] is True
        assert "3600s" in plan["reason"]

    def test_short_file_stays_sequential(self, monkeypatch):
        plan = _plan(monkeypatch, 45.0)
        assert plan["batched"] is False
        assert "threshold" in plan["reason"]

    def test_threshold_boundary_is_inclusive(self, monkeypatch):
        assert _plan(monkeypatch, 600.0)["batched"] is True
        assert _plan(monkeypatch, 599.0)["batched"] is False

    def test_opt_out_wins_over_duration(self, monkeypatch):
        plan = _plan(monkeypatch, 7200.0, batched=False)
        assert plan["batched"] is False
        assert plan["reason"] == "disabled by request"

    def test_custom_threshold_is_honoured(self, monkeypatch):
        assert _plan(monkeypatch, 120.0, batch_threshold_seconds=100.0)["batched"] is True

    def test_absurdly_low_threshold_is_floored(self, monkeypatch):
        """A 0s threshold would batch every clip, including 2-second ones."""
        assert _plan(monkeypatch, 5.0, batch_threshold_seconds=0.0)["batched"] is False

    def test_unknown_duration_does_not_change_behaviour(self, monkeypatch):
        plan = _plan(monkeypatch, 0.0)
        assert plan["batched"] is False
        assert plan["reason"] == "duration unknown"

    def test_junk_threshold_falls_back_to_the_default(self, monkeypatch):
        assert _plan(monkeypatch, 300.0, batch_threshold_seconds="abc")["batched"] is False
        assert _plan(monkeypatch, 900.0, batch_threshold_seconds="abc")["batched"] is True


class TestBatchSize:
    def test_default(self, monkeypatch):
        assert _plan(monkeypatch, 3600.0)["batch_size"] == 8

    def test_clamped_to_ceiling(self, monkeypatch):
        assert _plan(monkeypatch, 3600.0, batch_size=9999)["batch_size"] == BATCH_SIZE_MAX

    def test_never_below_one(self, monkeypatch):
        assert _plan(monkeypatch, 3600.0, batch_size=0)["batch_size"] == 1
        assert _plan(monkeypatch, 3600.0, batch_size=-4)["batch_size"] == 1

    def test_junk_falls_back_to_default(self, monkeypatch):
        assert _plan(monkeypatch, 3600.0, batch_size="abc")["batch_size"] == 8


class TestPipelineConstruction:
    def test_missing_pipeline_degrades_to_none(self, monkeypatch):
        """An older faster-whisper must slow down, not fail the job."""
        import builtins

        real_import = builtins.__import__

        def _no_batched(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("no BatchedInferencePipeline")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_batched)
        assert captions._batched_pipeline(object()) is None

    def test_construction_failure_degrades_to_none(self, monkeypatch):
        import types

        fake = types.ModuleType("faster_whisper")

        def _boom(**kwargs):
            raise RuntimeError("unsupported model")

        fake.BatchedInferencePipeline = _boom
        monkeypatch.setitem(sys.modules, "faster_whisper", fake)
        assert captions._batched_pipeline(object()) is None


class TestDurationProbe:
    def test_reads_a_real_wav_header(self, tmp_path):
        import wave

        path = tmp_path / "tone.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000 * 3)

        assert abs(captions._audio_duration_seconds(str(path)) - 3.0) < 0.01

    def test_unreadable_file_reports_zero_rather_than_raising(self, tmp_path):
        missing = tmp_path / "nope.wav"
        assert captions._audio_duration_seconds(str(missing)) == 0.0
