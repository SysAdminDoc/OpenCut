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
    output_path = Path(response.get_json()["output_path"])
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["human_review_recommended"] is True
    assert exported["segments"][0]["language"] == "ar"
    assert exported["segments"][0]["language_confidence"] == 0.90
    assert exported["segments"][0]["confidence"] == 0.86
    assert exported["segments"][0]["review_reasons"] == ["language_requires_human_review"]
