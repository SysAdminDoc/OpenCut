import json
from pathlib import Path
from types import SimpleNamespace

from tests.conftest import csrf_headers


def test_hindi_and_arabic_segments_are_flagged_for_human_review():
    from opencut.core.captions import CaptionSegment, Word, is_human_review_language

    hindi = CaptionSegment(
        text="\u0928\u092e\u0938\u094d\u0924\u0947",
        start=0.0,
        end=1.0,
        words=[Word("\u0928\u092e\u0938\u094d\u0924\u0947", 0.0, 1.0, confidence=0.96)],
        language="hi-IN",
        language_confidence=0.99,
    )
    arabic = CaptionSegment(
        text="\u0645\u0631\u062d\u0628\u0627",
        start=1.0,
        end=2.0,
        language="Arabic",
        language_confidence=0.99,
        confidence=0.95,
    )

    assert is_human_review_language("hi")
    assert is_human_review_language("ar-EG")
    assert hindi.human_review_recommended
    assert arabic.human_review_recommended
    assert "language_requires_human_review" in hindi.review_reasons
    assert "language_requires_human_review" in arabic.review_reasons


def test_low_asr_and_language_confidence_add_review_reasons():
    from opencut.core.captions import CaptionSegment, Word

    low_asr = CaptionSegment(
        text="uncertain words",
        start=0.0,
        end=1.0,
        words=[
            Word("uncertain", 0.0, 0.5, confidence=0.50),
            Word("words", 0.5, 1.0, confidence=0.60),
        ],
        language="en",
        language_confidence=0.95,
    )
    low_language = CaptionSegment(
        text="bonjour",
        start=1.0,
        end=2.0,
        language="fr",
        language_confidence=0.40,
        confidence=0.90,
    )

    assert low_asr.confidence == 0.55
    assert low_asr.human_review_recommended
    assert "low_asr_confidence" in low_asr.review_reasons
    assert low_language.human_review_recommended
    assert "low_language_confidence" in low_language.review_reasons


def test_segment_serializer_and_json_export_include_review_metadata(tmp_path):
    from opencut.core.captions import CaptionSegment, TranscriptionResult, caption_segment_to_dict
    from opencut.export.srt import export_json

    segment = CaptionSegment(
        text="\u0645\u0631\u062d\u0628\u0627",
        start=0.0,
        end=1.2,
        language="ar",
        language_confidence=0.91,
        confidence=0.88,
    )
    result = TranscriptionResult(
        segments=[segment],
        language="ar",
        duration=1.2,
        language_confidence=0.91,
    )
    payload = caption_segment_to_dict(segment, include_words=True, precision=3)

    assert payload["language"] == "ar"
    assert payload["human_review_recommended"] is True
    assert payload["review_reasons"] == ["language_requires_human_review"]

    out = tmp_path / "captions.json"
    export_json(result, str(out))
    exported = json.loads(out.read_text(encoding="utf-8"))
    assert exported["language_confidence"] == 0.91
    assert exported["human_review_recommended"] is True
    assert exported["review_segment_count"] == 1
    assert exported["segments"][0]["language"] == "ar"
    assert exported["segments"][0]["review_reasons"] == ["language_requires_human_review"]


def test_caption_roundtrip_sidecar_preserves_timeline_metadata(tmp_path):
    from opencut.core.caption_roundtrip import read_caption_sidecar, write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    result = TranscriptionResult(
        segments=[
            CaptionSegment(
                text="Needs review",
                start=1.0,
                end=2.5,
                words=[Word("Needs", 1.0, 1.4, confidence=0.82), Word("review", 1.5, 2.5, confidence=0.66)],
                speaker="A",
                language="en",
                language_confidence=0.77,
                confidence=0.74,
                human_review_recommended=True,
                review_reasons=["editor_flagged"],
            )
        ],
        language="en",
        duration=2.5,
        language_confidence=0.77,
        cache_key="a" * 64,
    )
    export_path = tmp_path / "clip.srt"

    sidecar_path = write_caption_sidecar(
        result,
        str(export_path),
        export_format="srt",
        source_path=str(source),
        display_settings={"token_ids": ["caption-style-main"]},
    )
    sidecar, warnings = read_caption_sidecar(sidecar_path, expected_export_path=str(export_path))

    assert Path(sidecar_path).name == "clip.srt.opencut-captions.json"
    assert warnings == []
    assert sidecar["schema_version"] == 1
    assert sidecar["source"]["source_file_hash"]
    assert sidecar["source"]["transcript_cache_key"] == "a" * 64
    cue = sidecar["cues"][0]
    assert cue["caption_id"] == "cap_000001"
    assert cue["source_segment_id"] == "segment_000001"
    assert cue["speaker"] == "A"
    assert cue["language_confidence"] == 0.77
    assert cue["human_review_recommended"] is True
    assert "editor_flagged" in cue["review_reasons"]
    assert cue["word_ids"] == ["cap_000001_word_0001", "cap_000001_word_0002"]
    assert cue["display_setting_token_ids"] == ["caption-style-main"]


def test_timeline_srt_to_captions_labels_srt_only_metadata_loss(client, csrf_token, tmp_path):
    srt_path = tmp_path / "lossy.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nEdited text\n", encoding="utf-8")

    response = client.post(
        "/timeline/srt-to-captions",
        json={"srt_path": str(srt_path)},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is False
    assert set(payload["warnings"]) == {"sidecar_missing", "metadata_unavailable"}
    assert payload["segments"] == [{"start": 1.0, "end": 2.0, "text": "Edited text"}]


def test_timeline_srt_to_captions_enriches_segments_from_sidecar(client, csrf_token, tmp_path):
    from opencut.core.caption_roundtrip import write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    srt_path = tmp_path / "clip.srt"
    srt_path.write_text("1\n00:00:01,200 --> 00:00:02,800\nEdited timeline text\n", encoding="utf-8")
    result = TranscriptionResult(
        segments=[
            CaptionSegment(
                text="Original sidecar text",
                start=1.0,
                end=2.5,
                words=[Word("Original", 1.0, 1.5, confidence=0.9)],
                speaker="Narrator",
                language="en",
                language_confidence=0.95,
                confidence=0.88,
            )
        ],
        language="en",
        duration=2.5,
        cache_key="b" * 64,
    )
    sidecar_path = write_caption_sidecar(result, str(srt_path), export_format="srt", source_path=str(source))

    response = client.post(
        "/timeline/srt-to-captions",
        json={"srt_path": str(srt_path), "sidecar_path": sidecar_path},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is True
    assert payload["warnings"] == []
    segment = payload["segments"][0]
    assert segment["text"] == "Edited timeline text"
    assert segment["start"] == 1.2
    assert segment["end"] == 2.8
    assert segment["speaker"] == "Narrator"
    assert segment["transcript_cache_key"] == "b" * 64
    assert segment["words"][0]["text"] == "Original"


def test_timeline_srt_to_captions_warns_on_invalid_sidecar_path(client, csrf_token, tmp_path):
    srt_path = tmp_path / "lossy.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nEdited text\n", encoding="utf-8")

    response = client.post(
        "/timeline/srt-to-captions",
        json={"srt_path": str(srt_path), "sidecar_path": "bad\x00sidecar.json"},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is False
    assert set(payload["warnings"]) == {"sidecar_missing", "metadata_unavailable"}
    assert payload["segments"] == [{"start": 1.0, "end": 2.0, "text": "Edited text"}]


def test_timeline_srt_to_captions_rejects_cueless_sidecar_metadata(client, csrf_token, tmp_path):
    from opencut.core.caption_roundtrip import SIDECAR_SCHEMA, SIDECAR_VERSION

    srt_path = tmp_path / "lossy.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nEdited text\n", encoding="utf-8")
    sidecar_path = tmp_path / "lossy.srt.opencut-captions.json"
    sidecar_path.write_text(
        json.dumps({
            "schema": SIDECAR_SCHEMA,
            "schema_version": SIDECAR_VERSION,
            "export": {"format": "srt", "path": str(srt_path)},
        }),
        encoding="utf-8",
    )

    response = client.post(
        "/timeline/srt-to-captions",
        json={"srt_path": str(srt_path), "sidecar_path": str(sidecar_path)},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is False
    assert set(payload["warnings"]) == {"sidecar_cues_missing", "metadata_unavailable"}
    assert payload["segments"] == [{"start": 1.0, "end": 2.0, "text": "Edited text"}]


def test_remap_preserves_review_metadata():
    from opencut.core.captions import CaptionSegment, TranscriptionResult, remap_captions_to_segments

    result = TranscriptionResult(
        segments=[
            CaptionSegment(
                text="low confidence",
                start=1.0,
                end=2.0,
                language="en",
                language_confidence=0.95,
                confidence=0.45,
            )
        ],
        language="en",
        duration=3.0,
        language_confidence=0.95,
    )

    remapped = remap_captions_to_segments(
        result,
        [SimpleNamespace(start=0.5, end=2.5)],
    )

    assert remapped.language_confidence == 0.95
    assert remapped.segments[0].confidence == 0.45
    assert remapped.segments[0].human_review_recommended is True
    assert remapped.segments[0].review_reasons == ["low_asr_confidence"]


def test_transcript_route_returns_segment_review_metadata(monkeypatch):
    import opencut.core.captions as core_captions
    import opencut.routes.captions as caption_routes
    from opencut.core.captions import CaptionSegment, TranscriptionResult

    result = TranscriptionResult(
        segments=[
            CaptionSegment(
                text="\u0645\u0631\u062d\u0628\u0627",
                start=0.0,
                end=1.0,
                language="ar",
                language_confidence=0.87,
                confidence=0.92,
            )
        ],
        language="ar",
        duration=1.0,
        language_confidence=0.87,
    )

    monkeypatch.setattr(core_captions, "check_whisper_available", lambda: (True, "stub"))
    monkeypatch.setattr(core_captions, "transcribe", lambda filepath, config=None: result)

    payload = caption_routes.get_transcript.__wrapped__.__wrapped__(
        "job_test",
        "input.wav",
        {"model": "base"},
    )

    assert payload["language"] == "ar"
    assert payload["language_confidence"] == 0.87
    assert payload["human_review_recommended"] is True
    assert payload["human_review_segments"] == 1
    assert payload["segments"][0]["language"] == "ar"
    assert payload["segments"][0]["human_review_recommended"] is True
    assert payload["segments"][0]["review_reasons"] == ["language_requires_human_review"]


def test_export_edited_transcript_preserves_review_metadata(client, csrf_token, tmp_path):
    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")

    response = client.post(
        "/transcript/export",
        json={
            "filepath": str(source),
            "output_dir": str(tmp_path),
            "format": "json",
            "language": "ar",
            "segments": [
                {
                    "text": "\u0645\u0631\u062d\u0628\u0627",
                    "start": 0.0,
                    "end": 1.0,
                    "language": "ar",
                    "language_confidence": 0.90,
                    "confidence": 0.86,
                    "human_review_recommended": True,
                    "review_reasons": ["language_requires_human_review"],
                    "words": [
                        {
                            "text": "\u0645\u0631\u062d\u0628\u0627",
                            "start": 0.0,
                            "end": 1.0,
                            "confidence": 0.86,
                        }
                    ],
                }
            ],
        },
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    output_path = Path(payload["output_path"])
    sidecar_path = Path(payload["sidecar_path"])
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert exported["human_review_recommended"] is True
    assert exported["segments"][0]["language"] == "ar"
    assert exported["segments"][0]["language_confidence"] == 0.90
    assert exported["segments"][0]["confidence"] == 0.86
    assert exported["segments"][0]["review_reasons"] == ["language_requires_human_review"]
    assert sidecar["cues"][0]["language"] == "ar"
    assert sidecar["cues"][0]["review_reasons"] == ["language_requires_human_review"]
