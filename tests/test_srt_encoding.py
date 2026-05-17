from pathlib import Path

import pytest

from opencut.core.captions import CaptionSegment, TranscriptionResult
from opencut.export.srt import UTF8_BOM, export_srt, has_utf8_bom, srt_text_encoding


def _result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            CaptionSegment(text="Plain English", start=0.0, end=1.0),
            CaptionSegment(text="Arabic مرحبا and Hindi नमस्ते", start=1.25, end=2.5),
        ],
        language="mul",
    )


def test_export_srt_defaults_to_utf8_without_bom(tmp_path):
    output = tmp_path / "default.srt"

    export_srt(_result(), str(output))

    raw = output.read_bytes()
    assert not raw.startswith(UTF8_BOM)
    assert "مرحبا" in raw.decode("utf-8")
    assert has_utf8_bom(str(output)) is False


def test_export_srt_legacy_windows_bom_is_opt_in(tmp_path):
    output = tmp_path / "legacy.srt"

    export_srt(_result(), str(output), legacy_windows_bom=True)

    raw = output.read_bytes()
    assert raw.startswith(UTF8_BOM)
    assert "नमस्ते" in raw.decode("utf-8-sig")
    assert has_utf8_bom(str(output)) is True


def test_srt_text_encoding_rejects_non_utf8_encodings():
    assert srt_text_encoding() == "utf-8"
    assert srt_text_encoding(legacy_windows_bom=True) == "utf-8-sig"
    assert srt_text_encoding(encoding="utf8-sig") == "utf-8-sig"
    with pytest.raises(ValueError, match="utf-8"):
        srt_text_encoding(encoding="cp1252")


def test_caption_route_bom_aliases_are_opt_in():
    from opencut.routes.captions import _legacy_srt_bom_requested

    assert _legacy_srt_bom_requested({}) is False
    assert _legacy_srt_bom_requested({"srt_legacy_bom": True}) is True
    assert _legacy_srt_bom_requested({"windows_legacy_bom": "yes"}) is True
    assert _legacy_srt_bom_requested({"legacy_bom": "1"}) is True


def test_shot_aware_export_to_file_supports_legacy_bom(tmp_path):
    from opencut.core.subtitle_shot_aware import SubtitleSegment, export_to_file

    output = tmp_path / "shot_aware.srt"
    segments = [SubtitleSegment(index=1, start=0.0, end=1.0, text="Hello")]

    export_to_file(segments, str(output), fmt="srt", legacy_windows_bom=True)

    assert Path(output).read_bytes().startswith(UTF8_BOM)
