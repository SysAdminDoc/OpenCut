"""ASR provenance, cache migration, and edit-boundary contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _result(*, boundary_confidence=0.9):
    from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

    return TranscriptionResult(
        segments=[
            CaptionSegment(
                text="um continue",
                start=0.0,
                end=1.0,
                words=[
                    Word(
                        "um",
                        0.0,
                        0.2,
                        confidence=0.96,
                        boundary_confidence=boundary_confidence,
                    ),
                    Word(
                        "continue",
                        0.3,
                        1.0,
                        confidence=0.93,
                        boundary_confidence=0.92,
                    ),
                ],
                language="en",
            )
        ],
        language="en",
        duration=1.0,
    )


def test_builtin_faster_whisper_identity_is_immutable():
    from opencut.core.asr_provenance import model_identity

    model_id, revision = model_identity("faster-whisper", "large-v3-turbo")

    assert model_id == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert revision == "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"


def test_corrupt_cache_recovery_downloads_the_pinned_revision(monkeypatch):
    import opencut.core.captions as captions

    captured = {}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        type(
            "Hub",
            (),
            {
                "snapshot_download": staticmethod(
                    lambda repo_id, **kwargs: captured.update(
                        repo_id=repo_id,
                        **kwargs,
                    )
                )
            },
        )(),
    )

    captions._download_model("base")

    assert captured["repo_id"] == "Systran/faster-whisper-base"
    assert captured["revision"] == "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
    assert captured["force_download"] is True


def test_engine_override_is_strict_and_normalizes_aliases(monkeypatch):
    import opencut.core.captions as captions

    monkeypatch.setattr(
        captions,
        "_whisper_backend_available",
        lambda engine: engine == "faster-whisper",
    )

    assert captions.resolve_whisper_backend("faster_whisper") == (
        "faster-whisper",
        "",
    )
    try:
        captions.resolve_whisper_backend("whisperx")
    except RuntimeError as exc:
        assert "not installed" in str(exc)
    else:
        raise AssertionError("unavailable explicit engine must not silently fall back")


def test_text_and_boundary_confidence_are_independent():
    segment = _result(boundary_confidence=0.2).segments[0]

    assert segment.confidence > 0.9
    assert segment.boundary_confidence == 0.56
    assert "low_boundary_confidence" in segment.review_reasons
    assert "low_asr_confidence" not in segment.review_reasons


def test_result_serialization_migrates_pre_provenance_payload():
    from opencut.core.captions import transcription_result_from_dict

    restored = transcription_result_from_dict({
        "language": "en",
        "segments": [{"text": "legacy", "start": 0, "end": 1, "words": []}],
    })

    assert restored.provenance.engine == "legacy-unknown"
    assert "predates" in restored.provenance.fallback_reason


def test_legacy_caption_sidecar_migrates_in_memory(tmp_path):
    from opencut.core.caption_roundtrip import read_caption_sidecar

    path = tmp_path / "legacy.opencut-captions.json"
    path.write_text(
        json.dumps({
            "schema": "opencut.caption_sidecar",
            "schema_version": 1,
            "export": {"path": ""},
            "result": {"language": "en"},
            "cues": [{
                "text": "legacy",
                "words": [{"text": "legacy", "start": 0, "end": 1}],
            }],
        }),
        encoding="utf-8",
    )

    migrated, warnings = read_caption_sidecar(str(path))

    assert warnings == []
    assert migrated["schema_version"] == 2
    assert migrated["result"]["asr_provenance"]["engine"] == "legacy-unknown"
    assert migrated["cues"][0]["words"][0]["boundary_confidence"] is None


def test_cache_key_changes_for_provenance_affecting_options(tmp_path):
    from opencut.core import transcript_cache
    from opencut.utils.config import CaptionConfig

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    base = CaptionConfig(engine="faster-whisper", model="base")
    pinned = CaptionConfig(
        engine="faster-whisper",
        model="base",
        model_revision="operator-revision",
    )
    aligned = CaptionConfig(engine="whisperx", model="base")

    base_key, _ = transcript_cache.build_cache_key(
        str(source), backend="faster-whisper", config=base
    )
    pinned_key, _ = transcript_cache.build_cache_key(
        str(source), backend="faster-whisper", config=pinned
    )
    aligned_key, _ = transcript_cache.build_cache_key(
        str(source), backend="whisperx", config=aligned
    )

    assert len({base_key, pinned_key, aligned_key}) == 3


def test_schema_one_cache_migrates_without_retranscribing(monkeypatch, tmp_path):
    import opencut.core.captions as captions
    from opencut.core import transcript_cache
    from opencut.utils.config import CaptionConfig

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "source.mov"
    source.write_bytes(b"stable")
    config = CaptionConfig(model="base")
    monkeypatch.setattr(
        captions,
        "check_whisper_available",
        lambda: (True, "faster-whisper"),
    )
    legacy_key, legacy_metadata = transcript_cache.build_legacy_cache_key(
        str(source),
        backend="faster-whisper",
        config=config,
    )
    legacy_path = Path(transcript_cache.cache_entry_path(legacy_key))
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({
            "schema_version": 1,
            "key": legacy_key,
            "metadata": legacy_metadata,
            "result": {
                "language": "en",
                "segments": [{"text": "legacy", "start": 0, "end": 1}],
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        captions,
        "extract_audio_wav",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy cache should prevent transcription")
        ),
    )

    migrated = captions.transcribe(str(source), config=config)

    assert migrated.cache_hit is True
    assert migrated.provenance.engine == "faster-whisper"
    assert "schema 1" in migrated.provenance.fallback_reason
    current = json.loads(Path(migrated.cache_path).read_text(encoding="utf-8"))
    assert current["schema_version"] == 2
    assert current["result"]["provenance"]["model_revision"]


def test_filler_boundary_review_is_auditionable_and_non_mutating():
    from opencut.core.fillers import build_boundary_review, detect_fillers

    transcription = _result(boundary_confidence=None)
    analysis = detect_fillers(transcription)
    review = build_boundary_review(analysis.hits, filepath="C:/media/interview.mov")

    assert review["required"] is True
    assert review["review_hits"] == 1
    item = review["items"][0]
    assert item["reason"] == "boundary_confidence_unavailable"
    assert item["audition"]["endpoint"] == "/preview/audio"
    assert item["audition"]["filter"] == "raw"
    assert item["audition"]["start"] == 0.0


def test_filler_route_blocks_timeline_export_until_boundary_review(
    monkeypatch, tmp_path
):
    import opencut.core.captions as captions
    import opencut.routes.audio as audio_routes

    source = tmp_path / "interview.wav"
    source.write_bytes(b"RIFF")
    monkeypatch.setattr(
        audio_routes,
        "_probe_media",
        lambda _path: type("Probe", (), {"duration": 1.0})(),
    )
    monkeypatch.setattr(
        audio_routes,
        "export_premiere_xml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("boundary review must happen before export")
        ),
    )
    monkeypatch.setattr(
        captions,
        "check_whisper_available",
        lambda: (True, "faster-whisper"),
    )
    monkeypatch.setattr(
        captions,
        "transcribe",
        lambda *_args, **_kwargs: _result(boundary_confidence=None),
    )

    response = audio_routes.filler_removal.__wrapped__.__wrapped__(
        "boundary-job",
        str(source),
        {
            "output_dir": str(tmp_path),
            "remove_silence": False,
            "remove_fillers": ["um"],
        },
    )

    assert response["preview_only"] is True
    assert response["mutation_blocked"] is True
    assert response["filler_stats"]["planned_fillers"] == 1
    assert "xml_path" not in response


def test_provenance_diagnostics_never_returns_transcript_text(
    client, monkeypatch, tmp_path
):
    from opencut.core import transcript_cache
    from opencut.core.asr_provenance import build_provenance

    monkeypatch.setenv("OPENCUT_TRANSCRIPT_CACHE_DIR", str(tmp_path))
    key = "c" * 64
    provenance = build_provenance(
        engine="faster-whisper",
        requested_engine="faster-whisper",
        model="base",
        model_revision=None,
        requested_language="en",
        word_timestamps=True,
        translate=False,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
    )
    transcript_cache.store_transcript(
        key,
        {"source": {"source_sha256": "d" * 64}},
        {
            "language": "en",
            "provenance": provenance.to_dict(),
            "segments": [{"text": "must not leak", "start": 0, "end": 1}],
        },
    )

    response = client.get(f"/captions/cache/provenance/{key}")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["asr_provenance"]["engine"] == "faster-whisper"
    assert "must not leak" not in json.dumps(payload)
    assert "segments" not in payload


def test_asr_fixture_manifest_separates_wer_from_boundaries():
    from opencut.core.ai_eval_harness import (
        evaluate_asr_contract,
        load_asr_fixture_manifest,
    )

    manifest_path = Path(__file__).parent / "fixtures" / "asr_boundary_contract.json"
    manifest = load_asr_fixture_manifest(manifest_path)
    traits = {
        trait
        for fixture in manifest["fixtures"]
        for trait in fixture.get("traits", [])
    }

    assert {"accent", "overlap", "fillers", "VFR", "29.97", "59.94"} <= traits
    filler = next(
        fixture
        for fixture in manifest["fixtures"]
        if fixture["id"] == "filler-boundary-regression"
    )
    metrics = evaluate_asr_contract(
        filler["reference_words"],
        filler["hypothesis_words"],
        boundary_tolerance_ms=manifest["boundary_tolerance_ms"],
    )
    assert metrics["text_wer"] == 0.0
    assert metrics["boundary_within_tolerance_rate"] < 1.0
    assert metrics["boundary_mae_ms"] > 80


def test_cep_exposes_boundary_audition_and_provenance_diagnostics():
    root = (
        Path(__file__).resolve().parents[1]
        / "extension"
        / "com.opencut.panel"
        / "client"
    )
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "main.js").read_text(encoding="utf-8")

    assert 'id="fillerBoundaryReview"' in html
    assert 'id="fillerBoundaryPlayer"' in html
    assert "renderFillerBoundaryReview" in script
    assert 'filter: "raw"' not in script  # server-provided audition plan is authoritative
    assert "accept_low_confidence_boundaries" in script
    assert "r.asr_provenance" in script
