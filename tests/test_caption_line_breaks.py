from opencut.core.caption_line_breaks import (
    all_lines_within,
    caption_layout_tokens,
    line_break_candidates,
    split_caption_text_chunks,
    wrap_caption_text,
)
from opencut.core.captions import CaptionSegment, TranscriptionResult
from opencut.export.srt import export_srt, export_vtt


def test_line_break_candidates_allow_no_space_cjk_without_orphaning_punctuation():
    text = "これは長い字幕です。次の行も続きます"
    candidates = line_break_candidates(text)

    assert candidates
    assert len(text) in candidates
    for candidate in candidates:
        if candidate < len(text):
            assert text[candidate] not in "。！？、"


def test_wrap_caption_text_breaks_japanese_without_spaces():
    text = "これは改行検証用の長い字幕テキストです"

    wrapped = wrap_caption_text(text, 8, 4, ellipsis=False)
    lines = wrapped.splitlines()

    assert len(lines) > 1
    assert all_lines_within(lines, 8)
    assert "".join(lines) == text


def test_wrap_caption_text_keeps_latin_space_wrapping_readable():
    wrapped = wrap_caption_text(
        "This caption should wrap at word boundaries",
        18,
        3,
        ellipsis=False,
    )

    assert wrapped.splitlines() == [
        "This caption",
        "should wrap at",
        "word boundaries",
    ]


def test_split_caption_text_chunks_handles_chinese_without_spaces():
    text = "开放剪辑字幕渲染验证需要处理没有空格的长文本"

    chunks = split_caption_text_chunks(text, 7)

    assert len(chunks) > 1
    assert all_lines_within(chunks, 7)
    assert "".join(chunks) == text


def test_caption_layout_tokens_split_cjk_for_styled_overlays():
    tokens = caption_layout_tokens("OpenCut字幕测试")

    assert tokens == ["OpenCut", "字", "幕", "测", "试"]


def test_export_srt_splits_no_space_cjk_without_truncation(tmp_path):
    text = "これは改行検証用の長い字幕テキストです"
    result = TranscriptionResult(
        segments=[CaptionSegment(text=text, start=0.0, end=4.0, language="ja")],
        language="ja",
        duration=4.0,
    )
    output = tmp_path / "ja.srt"

    export_srt(result, str(output), max_line_length=8, max_lines=2)

    exported = output.read_text(encoding="utf-8")
    payload_lines = [
        line
        for line in exported.splitlines()
        if line and "-->" not in line and not line.isdigit()
    ]
    assert "".join(payload_lines) == text
    assert all_lines_within(payload_lines, 8)
    assert "..." not in exported


def test_export_vtt_uses_same_cjk_wrapping_policy(tmp_path):
    text = "开放剪辑字幕渲染验证需要处理没有空格的长文本"
    result = TranscriptionResult(
        segments=[CaptionSegment(text=text, start=0.0, end=4.0, language="zh")],
        language="zh",
        duration=4.0,
    )
    output = tmp_path / "zh.vtt"

    export_vtt(result, str(output), max_line_length=7, max_lines=2)

    exported = output.read_text(encoding="utf-8")
    payload_lines = [
        line
        for line in exported.splitlines()
        if line and line != "WEBVTT" and "-->" not in line and not line.isdigit()
    ]
    assert "".join(payload_lines) == text
    assert all_lines_within(payload_lines, 7)


def test_shot_aware_wrap_uses_unicode_line_breaker():
    from opencut.core.subtitle_shot_aware import _wrap_text

    text = "これは改行検証用の長い字幕テキストです"
    wrapped = _wrap_text(text, 8, 4)

    assert "".join(wrapped.splitlines()) == text
    assert all_lines_within(wrapped.splitlines(), 8)
