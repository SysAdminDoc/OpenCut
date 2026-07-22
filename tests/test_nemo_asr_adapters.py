from __future__ import annotations

import json
import wave
from types import SimpleNamespace

import pytest


def _wav(path, seconds: float = 1.0):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * int(16000 * seconds))
    return str(path)


def _hypothesis(text="hello world"):
    return SimpleNamespace(
        text=text,
        confidence=0.87,
        timestamp={
            "word": [
                {"word": "hello", "start": 0.0, "end": 0.4, "confidence": 0.91},
                {"word": "world", "start": 0.5, "end": 0.9, "confidence": 0.83},
            ],
            "segment": [
                {"segment": text, "start": 0.0, "end": 0.9, "confidence": 0.87}
            ],
        },
    )


class _ParakeetModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, paths, *, batch_size, timestamps):
        self.calls.append((list(paths), batch_size, timestamps))
        return [_hypothesis() for _ in paths]


def test_parakeet_normalizes_timestamps_confidence_and_provenance(monkeypatch, tmp_path):
    from opencut.core import asr_parakeet
    from opencut.core.asr_nemo_models import PARAKEET_SPEC

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE", "0")
    source = _wav(tmp_path / "speech.wav")
    model = _ParakeetModel()

    result = asr_parakeet.transcribe(
        filepath=source,
        language="en",
        model_instance=model,
        audio_preparer=lambda path: path,
    )

    assert result.text == "hello world"
    assert result.segments[0].words[0].confidence == 0.91
    assert result.segments[0].end == 0.9
    assert result.provenance.model_id == PARAKEET_SPEC.model_id
    assert result.provenance.model_revision == PARAKEET_SPEC.revision
    assert model.calls[0][1:] == (1, True)
    payload = result.to_dict()
    assert payload["model_revision"] == PARAKEET_SPEC.revision
    assert payload["segments"][0]["words"][1]["text"] == "world"


def test_parakeet_cache_hit_does_not_load_nemo(monkeypatch, tmp_path):
    from opencut.core import asr_parakeet, transcript_cache

    monkeypatch.delenv("OPENCUT_TRANSCRIPT_CACHE", raising=False)
    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path / "cache"))
    transcript_cache.reset_runtime_stats()
    source = _wav(tmp_path / "cached.wav")

    first = asr_parakeet.transcribe(
        filepath=source,
        model_instance=_ParakeetModel(),
        audio_preparer=lambda path: path,
    )

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("cache hit must not load NeMo")

    second = asr_parakeet.transcribe(
        filepath=source,
        model_loader=forbidden_loader,
        allow_download=False,
        device="cuda",
        audio_preparer=lambda path: path,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text
    assert second.result.cache_key == first.result.cache_key


def test_parakeet_cancellation_stops_before_inference(monkeypatch, tmp_path):
    from opencut.core import asr_parakeet

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE", "0")
    source = _wav(tmp_path / "cancel.wav")
    model = _ParakeetModel()

    with pytest.raises(InterruptedError, match="cancelled"):
        asr_parakeet.transcribe(
            filepath=source,
            model_instance=model,
            is_cancelled=lambda: True,
            audio_preparer=lambda path: path,
        )
    assert model.calls == []


def test_parakeet_rejects_language_outside_checkpoint_contract(tmp_path):
    from opencut.core import asr_parakeet

    source = _wav(tmp_path / "unsupported.wav")
    with pytest.raises(ValueError, match="Unsupported Parakeet language"):
        asr_parakeet.transcribe(
            filepath=source,
            language="ja",
            model_instance=_ParakeetModel(),
            audio_preparer=lambda path: path,
        )


def test_missing_nemo_confidence_is_not_reported_as_certain():
    from opencut.core.asr_nemo import normalize_hypothesis
    from opencut.core.asr_provenance import ASRProvenance

    result = normalize_hypothesis(
        "timestamp metadata unavailable",
        language="en",
        duration=1.0,
        provenance=ASRProvenance(),
    )
    assert result.segments[0].confidence == 0.0
    assert result.segments[0].human_review_recommended is True


class _CanaryModel:
    def __init__(self):
        self.manifest = []

    def transcribe(self, manifest_path, *, batch_size):
        with open(manifest_path, encoding="utf-8") as handle:
            self.manifest = [json.loads(line) for line in handle]
        return [_hypothesis("hello world") for _ in self.manifest]


def test_canary_batch_uses_manifest_languages_and_preserves_order(monkeypatch, tmp_path):
    from opencut.core import asr_canary

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE", "0")
    sources = [_wav(tmp_path / "a.wav"), _wav(tmp_path / "b.wav", 0.5)]
    model = _CanaryModel()

    result = asr_canary.transcribe_batch(
        sources,
        source_language="de",
        target_language="en",
        translate=True,
        batch_size=2,
        model_instance=model,
        audio_preparer=lambda path: path,
    )

    assert result.source_paths == sources
    assert len(result.results) == 2
    assert all(item["source_lang"] == "de" for item in model.manifest)
    assert all(item["target_lang"] == "en" for item in model.manifest)
    assert all(item["timestamp"] == "yes" for item in model.manifest)
    assert result.results[0].language == "en"
    assert result.to_dict()["items"][1]["source_path"] == sources[1]


def test_canary_rejects_unsupported_language(tmp_path):
    from opencut.core import asr_canary

    source = _wav(tmp_path / "language.wav")
    with pytest.raises(ValueError, match="Unsupported Canary source language"):
        asr_canary.transcribe_batch(
            [source],
            source_language="ja",
            model_instance=_CanaryModel(),
            audio_preparer=lambda path: path,
        )
