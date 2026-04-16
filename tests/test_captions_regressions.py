from pathlib import Path
from types import SimpleNamespace


def test_interview_polish_uses_export_module_srt(monkeypatch, tmp_path):
    import opencut.core.captions as core_captions
    import opencut.core.fillers as fillers
    import opencut.core.repeat_detect as repeat_detect
    import opencut.export.srt as srt_export
    import opencut.polish_state as polish_state
    import opencut.routes.captions as captions_routes

    source_path = tmp_path / "input.wav"
    source_path.write_bytes(b"RIFF")

    kept_segment = SimpleNamespace(start=0.0, end=1.0)
    transcript_segment = SimpleNamespace(start=0.0, end=1.0, text="hello there")
    transcription = SimpleNamespace(
        segments=[transcript_segment],
        word_count=2,
        language="en",
    )

    monkeypatch.setattr(
        captions_routes,
        "detect_speech",
        lambda filepath, config=None, file_duration=0.0: [kept_segment],
    )
    monkeypatch.setattr(
        captions_routes,
        "get_preset",
        lambda preset: SimpleNamespace(silence=object(), captions=object()),
    )
    monkeypatch.setattr(
        captions_routes,
        "ExportConfig",
        lambda sequence_name="": SimpleNamespace(sequence_name=sequence_name),
    )
    monkeypatch.setattr(
        captions_routes,
        "_probe_media",
        lambda filepath: SimpleNamespace(duration=1.0),
    )
    monkeypatch.setattr(
        captions_routes,
        "export_premiere_xml",
        lambda filepath, segments, xml_path, config=None: Path(xml_path).write_text(
            "<xml/>", encoding="utf-8"
        ),
    )

    monkeypatch.setattr(core_captions, "check_whisper_available", lambda: (True, "stub"))
    monkeypatch.setattr(
        core_captions,
        "transcribe",
        lambda filepath, config=None, timeout=0: transcription,
    )
    monkeypatch.setattr(
        core_captions,
        "remap_captions_to_segments",
        lambda transcript, segments: {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello there"}]
        },
    )

    monkeypatch.setattr(
        srt_export,
        "export_srt",
        lambda result, out_path: Path(out_path).write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nhello there\n",
            encoding="utf-8",
        ),
    )

    monkeypatch.setattr(
        fillers,
        "detect_fillers",
        lambda transcript, include_context_fillers=True: SimpleNamespace(
            hits=[], total_filler_time=0.0
        ),
    )
    monkeypatch.setattr(
        fillers,
        "remove_fillers_from_segments",
        lambda segments, hits: segments,
    )
    monkeypatch.setattr(
        repeat_detect,
        "detect_repeated_takes",
        lambda segments, threshold=0.6, gap_tolerance=2.0: {"repeats": []},
    )
    monkeypatch.setattr(repeat_detect, "merge_repeat_ranges", lambda repeats: [])

    monkeypatch.setattr(polish_state, "load_state", lambda filepath: None)
    monkeypatch.setattr(polish_state, "save_state", lambda filepath, data: None)
    monkeypatch.setattr(polish_state, "_transcription_from_dict", lambda data: transcription)
    monkeypatch.setattr(polish_state, "_transcription_to_dict", lambda transcript: {"ok": True})

    result = captions_routes.interview_polish.__wrapped__.__wrapped__(
        "job_test",
        str(source_path),
        {
            "output_dir": str(tmp_path),
            "resume": False,
            "generate_chapters": False,
            "diarize": False,
            "remove_fillers": False,
            "detect_repeats": False,
        },
    )

    assert result["srt_path"].endswith("_interview.srt")
    assert Path(result["srt_path"]).exists()
