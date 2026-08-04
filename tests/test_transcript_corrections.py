"""Coverage for bulk transcript correction, undo, and glossary behavior."""

from __future__ import annotations

def _segments():
    return [
        {
            "id": 0,
            "start": 0.0,
            "end": 2.0,
            "text": "OpenCut makes wrld editing faster.",
            "words": [
                {"text": "OpenCut", "start": 0.0, "end": 0.5},
                {"text": "makes", "start": 0.5, "end": 0.9},
                {"text": "wrld", "start": 0.9, "end": 1.2},
                {"text": "editing", "start": 1.2, "end": 1.6},
                {"text": "faster.", "start": 1.6, "end": 2.0},
            ],
        },
        {"id": 1, "start": 2.5, "end": 3.5, "text": "No correction here.", "words": []},
    ]


def test_preview_replaces_literal_text_and_preserves_unchanged_word_timing():
    from opencut.core.transcript_corrections import preview_transcript_corrections

    original = _segments()
    preview = preview_transcript_corrections(
        original,
        find="wrld",
        replace="world",
        whole_word=True,
    )

    assert original[0]["text"] == "OpenCut makes wrld editing faster."
    assert preview["segments"][0]["text"] == "OpenCut makes world editing faster."
    assert preview["summary"]["total_replacements"] == 1
    assert preview["changes"][0]["segment_index"] == 0
    assert preview["segments"][0]["words"][2]["text"] == "world"
    assert preview["segments"][0]["words"][0]["start"] == 0.0
    assert preview["segments"][0]["words"][0]["end"] == 0.5


def test_case_and_whole_word_options_are_literal():
    from opencut.core.transcript_corrections import preview_transcript_corrections

    segments = [{"text": "Cat catalog CAT", "start": 0, "end": 1}]
    whole = preview_transcript_corrections(
        segments,
        find="cat",
        replace="dog",
        whole_word=True,
    )
    sensitive = preview_transcript_corrections(
        segments,
        find="Cat",
        replace="Dog",
        case_sensitive=True,
    )

    assert whole["segments"][0]["text"] == "dog catalog dog"
    assert sensitive["segments"][0]["text"] == "Dog catalog CAT"


def test_glossary_is_applied_to_transcription_without_changing_raw_cache(monkeypatch, tmp_path):
    from opencut import user_data
    from opencut.core import captions, transcript_cache
    from opencut.core.captions import CaptionConfig, CaptionSegment, TranscriptionResult, Word

    monkeypatch.setattr(user_data, "OPENCUT_DIR", str(tmp_path / "opencut"))
    project = str(tmp_path / "project.prproj")
    user_data.save_transcript_glossary(
        project,
        [{"find": "wrld", "replace": "world", "case_sensitive": False, "whole_word": True}],
    )
    media = tmp_path / "clip.wav"
    media.write_bytes(b"audio")
    wav = tmp_path / "extracted.wav"
    wav.write_bytes(b"wav")

    monkeypatch.setattr(captions, "resolve_whisper_backend", lambda _override=None: ("faster-whisper", ""))
    monkeypatch.setattr(captions, "extract_audio_wav", lambda *_args, **_kwargs: str(wav))
    monkeypatch.setattr(transcript_cache, "cache_enabled", lambda: False)
    monkeypatch.setattr(
        captions,
        "_transcribe_faster_whisper",
        lambda _wav, _config: TranscriptionResult(
            segments=[
                CaptionSegment(
                    text="Hello wrld",
                    start=0.0,
                    end=1.0,
                    words=[Word(text="Hello", start=0.0, end=0.4), Word(text="wrld", start=0.4, end=0.8)],
                )
            ],
            language="en",
        ),
    )

    result = captions.transcribe(
        str(media),
        config=CaptionConfig(model="base"),
        use_cache=False,
        project_path=project,
    )

    assert result.text == "Hello world"
    assert result.segments[0].words[1].text == "world"
    assert result.segments[0].words[1].start == 0.4


def test_routes_require_review_token_then_support_apply_and_undo(client, csrf_token):
    from tests.conftest import csrf_headers

    body = {
        "project_path": "C:/projects/interview.prproj",
        "segments": _segments(),
        "find": "wrld",
        "replace": "world",
        "whole_word": True,
        "save_to_glossary": True,
    }
    headers = csrf_headers(csrf_token)
    preview_response = client.post("/transcript-edit/corrections/preview", json=body, headers=headers)
    assert preview_response.status_code == 200, preview_response.get_json()
    preview = preview_response.get_json()
    assert preview["summary"]["total_replacements"] == 1
    assert preview["segments"][0]["text"] == "OpenCut makes world editing faster."

    missing = client.post("/transcript-edit/corrections/apply", json=body, headers=headers)
    assert missing.status_code == 409
    assert missing.get_json()["code"] == "DESTRUCTIVE_CONFIRMATION_REQUIRED"

    applied = dict(body)
    applied["confirm_token"] = preview["confirm_token"]
    applied_response = client.post("/transcript-edit/corrections/apply", json=applied, headers=headers)
    assert applied_response.status_code == 200, applied_response.get_json()
    applied_payload = applied_response.get_json()
    assert applied_payload["applied"] is True
    assert applied_payload["undo_token"]
    assert applied_payload["glossary"][0]["replace"] == "world"

    undo_response = client.post(
        "/transcript-edit/corrections/undo",
        json={
            "project_path": body["project_path"],
            "undo_token": applied_payload["undo_token"],
        },
        headers=headers,
    )
    assert undo_response.status_code == 200, undo_response.get_json()
    assert undo_response.get_json()["segments"][0]["text"] == body["segments"][0]["text"]

    glossary_response = client.get(
        "/transcript-edit/glossary",
        query_string={"project_path": body["project_path"]},
    )
    assert glossary_response.status_code == 200
    assert glossary_response.get_json()["rules"][0]["find"] == "wrld"
