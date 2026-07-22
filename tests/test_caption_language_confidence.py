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
    assert sidecar["schema_version"] == 2
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
    assert set(payload["segments"][0]) == {"start", "end", "text"}
    for metadata_key in (
        "speaker",
        "review_reasons",
        "display_setting_token_ids",
        "transcript_cache_key",
        "words",
    ):
        assert metadata_key not in payload["segments"][0]


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


def test_caption_roundtrip_import_diff_preserves_sidecar_metadata(client, csrf_token, tmp_path):
    from opencut.core.caption_roundtrip import write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult, Word

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    srt_path = tmp_path / "clip.srt"
    srt_path.write_text("1\n00:00:01,100 --> 00:00:02,600\nTimeline edit\n", encoding="utf-8")
    result = TranscriptionResult(
        segments=[
            CaptionSegment(
                text="Original text",
                start=1.0,
                end=2.5,
                words=[Word("Original", 1.0, 1.8, confidence=0.91)],
                speaker="Narrator",
                language="en",
                language_confidence=0.93,
                confidence=0.82,
                human_review_recommended=True,
                review_reasons=["editor_flagged"],
            )
        ],
        language="en",
        duration=2.5,
        cache_key="f" * 64,
    )
    sidecar_path = write_caption_sidecar(
        result,
        str(srt_path),
        export_format="srt",
        source_path=str(source),
        display_settings={"token_ids": ["caption-style-main"], "font": "Inter"},
    )

    imported = client.post(
        "/timeline/srt-to-captions",
        json={"srt_path": str(srt_path), "sidecar_path": sidecar_path},
        headers=csrf_headers(csrf_token),
    )
    assert imported.status_code == 200
    imported_payload = imported.get_json()
    imported_segment = imported_payload["segments"][0]
    assert imported_payload["metadata_preserved"] is True
    assert imported_segment["text"] == "Timeline edit"
    assert imported_segment["speaker"] == "Narrator"
    assert imported_segment["human_review_recommended"] is True
    assert imported_segment["review_reasons"] == ["editor_flagged"]
    assert imported_segment["display_setting_token_ids"] == ["caption-style-main"]
    assert imported_segment["transcript_cache_key"] == "f" * 64

    diff = client.post(
        "/captions/round-trip/diff",
        json={"sidecar_path": sidecar_path, "edited_segments": imported_payload["segments"]},
        headers=csrf_headers(csrf_token),
    )
    assert diff.status_code == 200
    diff_payload = diff.get_json()
    assert diff_payload["metadata_preserved"] is True
    assert diff_payload["warnings"] == []
    assert diff_payload["source"]["transcript_cache_key"] == "f" * 64
    assert diff_payload["counts"]["text_changed"] == 1
    change = diff_payload["changes"][0]
    assert change["before"]["caption_id"] == "cap_000001"
    assert change["before"]["speaker"] == "Narrator"
    assert change["before"]["review_reasons"] == ["editor_flagged"]
    assert change["before"]["display_setting_token_ids"] == ["caption-style-main"]
    assert change["after"]["text"] == "Timeline edit"
    assert change["after"]["speaker"] == "Narrator"
    assert change["after"]["review_reasons"] == ["editor_flagged"]
    assert change["after"]["display_setting_token_ids"] == ["caption-style-main"]
    assert change["after"]["transcript_cache_key"] == "f" * 64


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


def test_timeline_srt_to_captions_warns_on_stale_sidecar_export_path(client, csrf_token, tmp_path):
    from opencut.core.caption_roundtrip import write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    old_srt = tmp_path / "old.srt"
    current_srt = tmp_path / "current.srt"
    old_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nOld\n", encoding="utf-8")
    current_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nCurrent\n", encoding="utf-8")
    result = TranscriptionResult(
        segments=[CaptionSegment(text="Old", start=1.0, end=2.0, speaker="A")],
        language="en",
        duration=2.0,
        cache_key="g" * 64,
    )
    sidecar_path = write_caption_sidecar(result, str(old_srt), export_format="srt", source_path=str(source))

    response = client.post(
        "/timeline/srt-to-captions",
        json={"srt_path": str(current_srt), "sidecar_path": sidecar_path},
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is True
    assert payload["segments"][0]["speaker"] == "A"
    assert payload["segments"][0]["text"] == "Current"
    assert "sidecar_export_path_mismatch" in payload["warnings"]


def test_caption_roundtrip_diff_classifies_split_merge_insert_and_delete():
    from opencut.core.caption_roundtrip import diff_caption_roundtrip

    sidecar = {
        "source": {"transcript_cache_key": "h" * 64, "source_file_hash": "sha"},
        "cues": [
            {
                "caption_id": "cap-1",
                "start": 0.0,
                "end": 1.0,
                "text": "First",
                "display_setting_token_ids": ["style-a"],
            },
            {"caption_id": "cap-2", "start": 1.0, "end": 2.0, "text": "Second"},
            {"caption_id": "cap-3", "start": 2.0, "end": 3.0, "text": "Third"},
        ],
    }

    changed = diff_caption_roundtrip(
        sidecar=sidecar,
        edited_segments=[
            {
                "caption_id": "cap-1a",
                "source_caption_id": "cap-1",
                "start": 0.0,
                "end": 0.5,
                "text": "First part",
                "display_setting_token_ids": ["style-b"],
            },
            {
                "caption_id": "cap-1b",
                "source_caption_id": "cap-1",
                "start": 0.5,
                "end": 1.0,
                "text": "First part two",
            },
            {
                "caption_id": "cap-2-3",
                "source_caption_ids": ["cap-2", "cap-3"],
                "start": 1.0,
                "end": 3.0,
                "text": "Second and third",
            },
            {"caption_id": "cap-4", "start": 3.0, "end": 4.0, "text": "Inserted"},
        ],
    )
    assert changed["metadata_preserved"] is True
    assert changed["counts"]["split"] >= 1
    assert changed["counts"]["merge"] == 1
    assert changed["counts"]["inserted"] == 1
    assert changed["counts"]["style_changed"] == 1
    assert any("split" in change["changes"] for change in changed["changes"])
    assert any("merge" in change["changes"] for change in changed["changes"])

    deleted = diff_caption_roundtrip(sidecar=sidecar, edited_segments=sidecar["cues"][:2])
    assert deleted["counts"]["deleted"] == 1
    assert deleted["changes"][-1]["change_type"] == "deleted"


def test_caption_roundtrip_diff_endpoint_reports_sidecar_changes(client, csrf_token, tmp_path):
    from opencut.core.caption_roundtrip import write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    export_path = tmp_path / "clip.srt"
    result = TranscriptionResult(
        segments=[CaptionSegment(text="Original text", start=1.0, end=2.0, speaker="A")],
        language="en",
        duration=2.0,
        cache_key="c" * 64,
    )
    sidecar_path = write_caption_sidecar(result, str(export_path), export_format="srt", source_path=str(source))

    response = client.post(
        "/captions/round-trip/diff",
        json={
            "sidecar_path": sidecar_path,
            "edited_segments": [{"start": 1.25, "end": 2.25, "text": "Edited text"}],
        },
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is True
    assert payload["confidence_label"] == "high"
    assert payload["counts"]["text_changed"] == 1
    assert payload["counts"]["timing_changed"] == 1
    assert payload["summary"]["changed"] is True
    assert payload["changes"][0]["before"]["text"] == "Original text"
    assert payload["changes"][0]["after"]["text"] == "Edited text"
    assert payload["changes"][0]["after"]["speaker"] == "A"
    assert payload["source"]["transcript_cache_key"] == "c" * 64


def test_caption_roundtrip_diff_endpoint_supports_lossy_no_sidecar(client, csrf_token):
    response = client.post(
        "/captions/round-trip/diff",
        json={
            "original_segments": [{"start": 0.0, "end": 1.0, "text": "Original"}],
            "edited_segments": [{"start": 0.0, "end": 1.0, "text": "Edited"}],
        },
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is False
    assert payload["confidence_label"] == "low"
    assert set(payload["warnings"]) == {"metadata_unavailable", "sidecar_missing"}
    assert payload["counts"]["text_changed"] == 1


def test_caption_roundtrip_diff_endpoint_accepts_srt_text(client, csrf_token, tmp_path):
    from opencut.core.caption_roundtrip import write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult

    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    export_path = tmp_path / "clip.srt"
    result = TranscriptionResult(
        segments=[CaptionSegment(text="Original text", start=1.0, end=2.0, speaker="A")],
        language="en",
        duration=2.0,
        cache_key="e" * 64,
    )
    sidecar_path = write_caption_sidecar(result, str(export_path), export_format="srt", source_path=str(source))

    response = client.post(
        "/captions/round-trip/diff",
        json={
            "sidecar_path": sidecar_path,
            "srt_text": "1\n00:00:01,000 --> 00:00:02,000\nOriginal text\n",
        },
        headers=csrf_headers(csrf_token),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metadata_preserved"] is True
    assert payload["summary"]["changed"] is False
    assert payload["counts"]["unchanged"] == 1


def test_caption_roundtrip_apply_requires_confirmation_and_stores_revision(client, csrf_token, tmp_path, monkeypatch):
    from opencut.core.caption_roundtrip import write_caption_sidecar
    from opencut.core.captions import CaptionSegment, TranscriptionResult

    monkeypatch.setenv("OPENCUT_CAPTION_REVISION_DIR", str(tmp_path / "revisions"))
    source = tmp_path / "clip.wav"
    source.write_bytes(b"RIFF")
    export_path = tmp_path / "clip.srt"
    result = TranscriptionResult(
        segments=[CaptionSegment(text="Original", start=0.0, end=1.0, speaker="A")],
        language="en",
        duration=1.0,
        cache_key="d" * 64,
    )
    sidecar_path = write_caption_sidecar(result, str(export_path), export_format="srt", source_path=str(source))
    request_json = {
        "sidecar_path": sidecar_path,
        "edited_segments": [{"start": 0.0, "end": 1.0, "text": "Original"}],
    }

    missing = client.post(
        "/captions/round-trip/apply",
        json=request_json,
        headers=csrf_headers(csrf_token),
    )
    assert missing.status_code == 409
    preview = client.post(
        "/captions/round-trip/apply",
        json={**request_json, "dry_run": True},
        headers=csrf_headers(csrf_token),
    )
    assert preview.status_code == 200
    token = preview.get_json()["confirm_token"]

    applied = client.post(
        "/captions/round-trip/apply",
        json={**request_json, "confirm_token": token},
        headers=csrf_headers(csrf_token),
    )
    assert applied.status_code == 200
    payload = applied.get_json()
    revision = payload["revision"]
    assert revision["transcript_cache_key"] == "d" * 64
    revision_path = Path(revision["revision_path"])
    assert revision_path.exists()
    revision_payload = json.loads(revision_path.read_text(encoding="utf-8"))
    stored_change = revision_payload["diff"]["changes"][0]
    assert stored_change["before"]["speaker"] == "A"
    assert stored_change["after"]["speaker"] == "A"
    assert stored_change["after"]["transcript_cache_key"] == "d" * 64

    applied_again = client.post(
        "/captions/round-trip/apply",
        json={**request_json, "confirm_token": token},
        headers=csrf_headers(csrf_token),
    )
    assert applied_again.status_code == 200
    assert applied_again.get_json()["revision"]["revision_id"] == revision["revision_id"]


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
