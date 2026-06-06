import json
from pathlib import Path

from tests.conftest import csrf_headers


def _sample_result():
    from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

    return TranscriptionResult(
        segments=[
            CaptionSegment(
                text="hello cache",
                start=0.0,
                end=1.25,
                words=[Word(text="hello", start=0.0, end=0.5, confidence=0.91)],
                language="en",
                confidence=0.91,
            )
        ],
        language="en",
        duration=1.25,
        language_confidence=0.99,
    )


def test_cache_key_changes_with_source_bytes_and_settings(monkeypatch, tmp_path):
    from opencut.core import transcript_cache
    from opencut.utils.config import CaptionConfig

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path / "cache"))
    media = tmp_path / "clip.wav"
    media.write_bytes(b"audio-v1")

    key1, meta1 = transcript_cache.build_cache_key(
        str(media),
        backend="faster-whisper",
        config=CaptionConfig(model="base", language="en"),
    )
    key2, _ = transcript_cache.build_cache_key(
        str(media),
        backend="faster-whisper",
        config=CaptionConfig(model="small", language="en"),
    )
    media.write_bytes(b"audio-v2")
    key3, meta3 = transcript_cache.build_cache_key(
        str(media),
        backend="faster-whisper",
        config=CaptionConfig(model="base", language="en"),
    )

    assert key1 != key2
    assert key1 != key3
    assert meta1["source"]["source_sha256"] != meta3["source"]["source_sha256"]


def test_transcribe_uses_persistent_cache(monkeypatch, tmp_path):
    import opencut.core.captions as captions
    from opencut.core import transcript_cache
    from opencut.utils.config import CaptionConfig

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path / "cache"))
    transcript_cache.reset_runtime_stats()

    media = tmp_path / "interview.mov"
    wav = tmp_path / "interview.wav"
    media.write_bytes(b"stable-media-bytes")
    wav.write_bytes(b"RIFF")

    calls = []
    monkeypatch.setattr(captions, "check_whisper_available", lambda: (True, "faster-whisper"))
    monkeypatch.setattr(captions, "extract_audio_wav", lambda filepath, sample_rate=16000: str(wav))

    def fake_backend(wav_path, config):
        calls.append((wav_path, config.model))
        return _sample_result()

    monkeypatch.setattr(captions, "_transcribe_faster_whisper", fake_backend)

    first = captions.transcribe(str(media), config=CaptionConfig(model="base"))
    second = captions.transcribe(str(media), config=CaptionConfig(model="base"))

    assert len(calls) == 1
    assert first.cache_hit is False
    assert first.cache_key
    assert first.cache_path and Path(first.cache_path).is_file()
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key
    assert second.text == "hello cache"
    assert second.segments[0].words[0].confidence == 0.91

    stats = transcript_cache.cache_stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["writes"] == 1


def test_transcribe_respects_disabled_cache(monkeypatch, tmp_path):
    import opencut.core.captions as captions
    from opencut.core import transcript_cache
    from opencut.utils.config import CaptionConfig

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE", "0")
    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path / "cache"))
    transcript_cache.reset_runtime_stats()

    media = tmp_path / "interview.mov"
    wav = tmp_path / "interview.wav"
    media.write_bytes(b"stable-media-bytes")
    wav.write_bytes(b"RIFF")

    monkeypatch.setattr(captions, "check_whisper_available", lambda: (True, "faster-whisper"))
    monkeypatch.setattr(captions, "extract_audio_wav", lambda filepath, sample_rate=16000: str(wav))
    monkeypatch.setattr(
        captions,
        "_transcribe_faster_whisper",
        lambda wav_path, config: _sample_result(),
    )
    monkeypatch.setattr(
        transcript_cache,
        "source_digest",
        lambda filepath: (_ for _ in ()).throw(AssertionError("cache disabled")),
    )

    result = captions.transcribe(str(media), config=CaptionConfig(model="base"))

    assert result.cache_hit is False
    assert result.cache_key is None
    assert result.cache_path is None
    assert transcript_cache.cache_stats()["entries"] == 0


def test_generate_captions_honors_force_retranscribe(monkeypatch, tmp_path):
    import opencut.core.captions as core_captions
    import opencut.routes.captions as captions_routes

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")

    use_cache_values = []

    def fake_transcribe(filepath, config=None, timeout=None, use_cache=True):
        use_cache_values.append(use_cache)
        result = _sample_result()
        result.cache_hit = False
        result.cache_key = "a" * 64
        return result

    monkeypatch.setattr(core_captions, "check_whisper_available", lambda: (True, "stub"))
    monkeypatch.setattr(core_captions, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        captions_routes,
        "_export_srt_with_policy",
        lambda result, out_path, legacy_windows_bom=False: Path(out_path).write_text(
            "1\n00:00:00,000 --> 00:00:01,250\nhello cache\n",
            encoding="utf-8",
        ),
    )

    response = captions_routes.generate_captions.__wrapped__.__wrapped__(
        "job-cache",
        str(source),
        {"force_retranscribe": True},
    )

    assert use_cache_values == [False]
    assert response["transcript_cache_hit"] is False
    assert response["transcript_cache_key"] == "a" * 64
    assert Path(response["output_path"]).is_file()


def test_caption_cache_routes(monkeypatch, tmp_path, client, csrf_token):
    from opencut.core import transcript_cache

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path / "cache"))
    transcript_cache.reset_runtime_stats()

    key = "b" * 64
    transcript_cache.store_transcript(
        key,
        {
            "schema_version": transcript_cache.CACHE_SCHEMA_VERSION,
            "key": key,
            "source": {"source_sha256": "c" * 64, "source_size_bytes": 5},
            "backend": "faster-whisper",
            "backend_version": "test",
            "settings": {"model": "base"},
            "extra": {},
        },
        {"language": "en", "segments": []},
    )

    stats_resp = client.get("/captions/cache/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.get_json()
    assert stats["enabled"] is True
    assert stats["entries"] == 1
    assert stats["bytes"] > 0

    rejected = client.delete(
        "/captions/cache/clear",
        data=json.dumps({}),
        headers=csrf_headers(csrf_token),
    )
    assert rejected.status_code == 409
    assert rejected.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

    preview_resp = client.delete(
        "/captions/cache/clear",
        data=json.dumps({"dry_run": True}),
        headers=csrf_headers(csrf_token),
    )
    assert preview_resp.status_code == 200
    preview = preview_resp.get_json()
    assert preview["dry_run"] is True
    assert preview["removed_entries"] == 0
    assert transcript_cache.cache_stats()["entries"] == 1

    clear_resp = client.delete(
        "/captions/cache/clear",
        data=json.dumps({"confirm_token": preview["confirm_token"]}),
        headers=csrf_headers(csrf_token),
    )
    assert clear_resp.status_code == 200
    cleared = clear_resp.get_json()
    assert cleared["removed_entries"] == 1
    assert transcript_cache.cache_stats()["entries"] == 0
