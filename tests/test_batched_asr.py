"""F336 — batched faster-whisper inference for long files.

Upstream's BatchedInferencePipeline is roughly 4x on long audio, and long audio
is where transcription cost actually lands. Batching short clips buys little and
spends VRAM, so the decision is duration-driven; these tests pin the policy, the
opt-out, and that an older wrapper degrades in speed instead of failing the job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import wave

import pytest

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


# ---------------------------------------------------------------------------
# F361 — decode the same speech both ways and compare what captions depend on
# ---------------------------------------------------------------------------

_ASR_MODEL = "Systran/faster-whisper-medium.en"

_SPEECH = (
    "The quick brown fox jumps over the lazy dog. "
    "Sixteen editors reviewed the timeline before the final export. "
    "Silence detection removed nine seconds from the interview. "
    "Captions were burned in at twenty four frames per second. "
) * 2


def _speak_to_wav(target) -> float:
    """Render offline speech with Windows SAPI; skip where it is unavailable."""
    if sys.platform != "win32" or not shutil.which("powershell"):
        pytest.skip("offline speech synthesis needs Windows SAPI")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{target}'); "
        f"$s.Speak(@'\n{_SPEECH}\n'@); "
        "$s.Dispose()"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not os.path.exists(target):
        pytest.skip(f"SAPI produced no audio: {result.stderr[:200]}")
    with wave.open(str(target), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _cached_model_dir():
    """Locate an already-downloaded faster-whisper snapshot, or skip.

    The test must never reach for the network, so it resolves the Hugging
    Face cache directly rather than by repo id: this cache stores real files
    under ``snapshots/`` with an empty ``blobs/``, which the hub's own
    ``local_files_only`` resolution refuses to accept.
    """
    # conftest sandboxes HOME and os.path.expanduser so OpenCut's own paths
    # stay inside the test home. The model cache belongs to the real user, so
    # anchor on USERPROFILE, which that fixture leaves alone.
    home = (
        os.environ.get("USERPROFILE")
        or os.environ.get("REAL_HOME")
        or os.path.expanduser("~")
    )
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        home, ".cache", "huggingface", "hub"
    )
    repo = os.path.join(root, "models--" + _ASR_MODEL.replace("/", "--"), "snapshots")
    if os.path.isdir(repo):
        for name in sorted(os.listdir(repo)):
            candidate = os.path.join(repo, name)
            if os.path.isfile(os.path.join(candidate, "model.bin")):
                return candidate
    pytest.skip(f"{_ASR_MODEL} is not cached under {root}")


def _load_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError:  # pragma: no cover - environment dependent
        pytest.skip("faster-whisper is not installed")
    model_dir = _cached_model_dir()
    try:
        return WhisperModel(model_dir, device="cpu", compute_type="int8")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cached model at {model_dir} would not load: {exc}")


def _words(model, path, **kwargs):
    started = time.time()
    segments, _info = model.transcribe(str(path), word_timestamps=True, **kwargs)
    segments = list(segments)
    words = [word for segment in segments for word in (segment.words or [])]
    return segments, words, time.time() - started


@pytest.mark.slow
class TestBatchedDecodingKeepsTheTranscriptCaptionsAreBuiltFrom:
    """F336 promised word timestamps and segment shape survive batching.

    Only half of that is true, and the half that is false matters less than
    it sounds. Decoded both ways on the same speech, the word sequence and
    word timings agree closely, while the segmentation does not: the batched
    pipeline re-windows the audio, so it returns a different number of
    segments for identical words. Captions are built from words, so this
    pins the word-level contract exactly and records the segment difference
    as expected behaviour rather than pretending it does not happen.

    Marked slow: it loads a medium model and decodes the same audio twice.
    """

    def test_words_survive_batching_even_though_segments_do_not(self, tmp_path):
        speech = tmp_path / "speech.wav"
        duration = _speak_to_wav(speech)
        assert duration > 20, f"fixture too short to be meaningful: {duration:.1f}s"

        model = _load_model()
        sequential_segments, sequential, sequential_s = _words(model, speech)
        batched_model = captions._batched_pipeline(model)
        if batched_model is None:
            pytest.skip("this faster-whisper has no BatchedInferencePipeline")
        batched_segments, batched, batched_s = _words(model=batched_model, path=speech, batch_size=8)

        # Hardware note for anyone reading a failure: this is CPU int8 on
        # whatever box ran it. Upstream's ~4x is a GPU, long-file figure and
        # is not reproducible here, which is why the changelog attributes the
        # number to upstream rather than claiming OpenCut measured it.
        print(
            f"\n[F361] {duration:.1f}s speech, CPU int8 {_ASR_MODEL}: "
            f"sequential {sequential_s:.1f}s / {len(sequential_segments)} segments, "
            f"batched {batched_s:.1f}s / {len(batched_segments)} segments"
        )

        assert sequential, "sequential decode produced no words"
        assert [w.word for w in sequential] == [w.word for w in batched], (
            "batching must not change the words themselves"
        )

        starts = sorted(abs(a.start - b.start) for a, b in zip(sequential, batched))
        ends = sorted(abs(a.end - b.end) for a, b in zip(sequential, batched))
        p95 = starts[int(len(starts) * 0.95) - 1]

        # Typical agreement is tight (median ~0.01s, p95 ~0.04s measured), so a
        # systematic drift breaks this even while staying inside the bound below.
        assert p95 <= 0.10, f"word starts drifted: p95 {p95:.3f}s\n{starts[-5:]}"
        # Worst case sits at a re-segmentation boundary (0.29s measured). Half a
        # second keeps every word inside its own caption cue, which is the
        # property a caption workflow actually depends on.
        assert starts[-1] <= 0.5, f"a word moved {starts[-1]:.3f}s between decoders"
        assert ends[-1] <= 0.5, f"a word end moved {ends[-1]:.3f}s between decoders"
