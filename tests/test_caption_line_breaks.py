import os
import re
import shutil
import subprocess

import pytest

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


_F459_TEXT = (
    "every microphone tested camera a we afterwards it single production a understand "
    "and then the interview recording continued until the production schedule ended"
)


def _timed_segment(tokens, step=0.4):
    from opencut.core.captions import Word

    words = [Word(text=tok, start=i * step, end=i * step + step * 0.9) for i, tok in enumerate(tokens)]
    return CaptionSegment(text=" ".join(tokens), start=0.0, end=len(tokens) * step, words=words)


def _exported_cues(path):
    """Return ``(start, end, lines)`` for every cue in an SRT or VTT file."""
    def _seconds(stamp):
        hms, frac = stamp.replace(",", ".").rsplit(".", 1)
        h, m, s = (int(part) for part in hms.split(":"))
        return h * 3600 + m * 60 + s + int(frac) / 1000.0

    cues = []
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        lines = [line for line in block.splitlines() if line and line != "WEBVTT"]
        for index, line in enumerate(lines):
            if "-->" in line:
                start, end = (part.strip() for part in line.split("-->"))
                cues.append((_seconds(start), _seconds(end), lines[index + 1:]))
                break
    return cues


@pytest.mark.parametrize("exporter, suffix", [(export_srt, ".srt"), (export_vtt, ".vtt")])
def test_export_keeps_every_word_of_a_long_timed_segment(tmp_path, exporter, suffix):
    tokens = _F459_TEXT.split()
    segment = _timed_segment(tokens)
    output = tmp_path / f"f459{suffix}"

    exporter(TranscriptionResult(segments=[segment], duration=segment.end), str(output))

    cues = _exported_cues(output)
    written = " ".join(" ".join(lines) for _, _, lines in cues).split()
    assert written == tokens
    assert "..." not in output.read_text(encoding="utf-8")
    for start, end, lines in cues:
        assert 1 <= len(lines) <= 2
        assert all_lines_within(lines, 42)
        cue_words = " ".join(lines).split()
        first = segment.words[written.index(cue_words[0])]
        assert start == pytest.approx(first.start, abs=0.001)
        assert end > start


@pytest.mark.parametrize("exporter, suffix", [(export_srt, ".srt"), (export_vtt, ".vtt")])
@pytest.mark.parametrize("timed", [True, False])
def test_export_never_drops_words_across_random_cues(tmp_path, exporter, suffix, timed):
    import random

    rng = random.Random(459)
    vocabulary = [
        "a", "we", "it", "and", "the", "until", "camera", "tested", "single",
        "schedule", "production", "microphone", "afterwards", "understand",
        "interview", "recording", "continued", "documentary",
    ]
    for case in range(300):
        max_line_length = rng.randint(12, 42)
        max_lines = rng.randint(1, 3)
        tokens = [rng.choice(vocabulary) for _ in range(rng.randint(1, 40))]
        if timed:
            segment = _timed_segment(tokens)
        else:
            segment = CaptionSegment(text=" ".join(tokens), start=0.0, end=len(tokens) * 0.4)
        output = tmp_path / f"case{case}{suffix}"

        exporter(
            TranscriptionResult(segments=[segment], duration=segment.end),
            str(output),
            max_line_length=max_line_length,
            max_lines=max_lines,
        )

        cues = _exported_cues(output)
        written = " ".join(" ".join(lines) for _, _, lines in cues).split()
        assert written == tokens, (case, max_line_length, max_lines)
        for _, _, lines in cues:
            assert 1 <= len(lines) <= max_lines, (case, lines)
            assert all_lines_within(lines, max_line_length), (case, lines)


def test_export_keeps_a_word_longer_than_a_whole_cue(tmp_path):
    token = "x" * 30
    segment = _timed_segment(["see", token, "now"])
    output = tmp_path / "long.srt"

    export_srt(TranscriptionResult(segments=[segment], duration=segment.end), str(output),
               max_line_length=8, max_lines=2)

    cues = _exported_cues(output)
    assert "".join("".join(lines) for _, _, lines in cues) == "see" + token + "now"
    for start, end, lines in cues:
        assert 1 <= len(lines) <= 2
        assert all_lines_within(lines, 8)
        assert end > start


def test_shot_aware_line_limits_split_instead_of_dropping_lines():
    from opencut.core.subtitle_shot_aware import SubtitleSegment, _enforce_line_limits

    tokens = _F459_TEXT.split()
    segments = [SubtitleSegment(index=1, start=2.0, end=11.2, text=" ".join(tokens))]

    adjusted, wraps = _enforce_line_limits(segments, 42, 2)

    assert wraps == 1
    assert len(adjusted) > 1
    assert " ".join(seg.text.replace("\n", " ") for seg in adjusted).split() == tokens
    assert adjusted[0].start == 2.0 and adjusted[-1].end == 11.2
    for previous, current in zip(adjusted, adjusted[1:]):
        assert current.start == pytest.approx(previous.end)
    for seg in adjusted:
        lines = seg.text.splitlines()
        assert 1 <= len(lines) <= 2
        assert all_lines_within(lines, 42)


# Eight 4-character words make a 39-character line at 42 per line, so 16 words
# fill two lines and "zzzz" (44 characters with line two) spills into a third,
# giving pieces of 79 and 4 characters. By character share alone "zzzz" would
# get under 0.1 s of a 2 s cue.
_SHORT_TAIL_WORDS = [f"aa{n:02d}" for n in range(1, 17)] + ["zzzz"]


def _shot_aware(start, end):
    from opencut.core.subtitle_shot_aware import SubtitleSegment, process_shot_aware

    segment = SubtitleSegment(index=1, start=start, end=end, text=" ".join(_SHORT_TAIL_WORDS))
    return process_shot_aware([segment], cuts=[], profile="netflix").adjusted_subtitles


def test_shot_aware_split_keeps_the_profile_minimum_duration():
    from opencut.core.subtitle_shot_aware import get_profile

    min_dur = get_profile("netflix")["min_duration"]
    pieces = _shot_aware(10.0, 12.0)

    assert [seg.text.replace("\n", " ").split() for seg in pieces] == [_SHORT_TAIL_WORDS[:16], ["zzzz"]]
    assert pieces[0].start == 10.0 and pieces[-1].end == 12.0
    assert pieces[1].start == pytest.approx(pieces[0].end)
    # The short piece is pinned at the minimum and the long one takes the rest.
    assert pieces[1].end - pieces[1].start == pytest.approx(min_dur)
    assert pieces[0].end - pieces[0].start == pytest.approx(2.0 - min_dur)


def test_shot_aware_split_shares_equally_when_the_minimum_cannot_fit():
    # Two pieces need 2 x 0.833 s, more than this 1.5 s cue has. Keeping the
    # cue's own start and end, the fairest split is half each.
    pieces = _shot_aware(10.0, 11.5)

    assert [seg.text.replace("\n", " ").split() for seg in pieces] == [_SHORT_TAIL_WORDS[:16], ["zzzz"]]
    assert pieces[0].start == 10.0 and pieces[-1].end == 11.5
    assert [seg.end - seg.start for seg in pieces] == [pytest.approx(0.75), pytest.approx(0.75)]


def test_timed_chunks_without_a_minimum_keep_the_character_share():
    from opencut.core.caption_line_breaks import timed_caption_chunks

    cues = timed_caption_chunks(" ".join(_SHORT_TAIL_WORDS), 0.0, 8.3, 42, 2)

    assert [(start, end) for _, start, end in cues] == [(0.0, pytest.approx(7.9)), (pytest.approx(7.9), 8.3)]


def test_positioned_webvtt_splits_long_cues_instead_of_dropping_words(tmp_path):
    from opencut.core.broadcast_caption import CaptionSegment as BroadcastSegment
    from opencut.core.broadcast_caption import export_webvtt_positioned

    tokens = _F459_TEXT.split()
    long_word = "x" * 50
    segments = [
        BroadcastSegment(index=1, start=1.0, end=9.0, text=" ".join(tokens), row=2),
        BroadcastSegment(index=2, start=10.0, end=12.0, text=f"see {long_word} now", row=15),
    ]
    output = tmp_path / "positioned.vtt"

    export_webvtt_positioned(segments, str(output))

    def _seconds(stamp):
        h, m, s = stamp.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    cues = []
    for block in output.read_text(encoding="utf-8").split("\n\n"):
        lines = [line for line in block.splitlines() if line and line != "WEBVTT"]
        if lines:
            start, _, end, *settings = lines[0].split()
            cues.append((_seconds(start), _seconds(end), settings, lines[1:]))

    first = [cue for cue in cues if cue[1] <= 9.0]
    second = [cue for cue in cues if cue[0] >= 10.0]
    assert len(first) + len(second) == len(cues)
    assert len(first) > 1
    assert " ".join(" ".join(lines) for *_, lines in first).split() == tokens
    assert "".join("".join(lines) for *_, lines in second).replace(" ", "") == "see" + long_word + "now"
    for group, (lo, hi), settings in (
        (first, (1.0, 9.0), ["line:10%", "position:50%", "align:center", "size:80%"]),
        (second, (10.0, 12.0), ["line:90%", "position:50%", "align:center", "size:80%"]),
    ):
        assert group[0][0] == pytest.approx(lo) and group[-1][1] == pytest.approx(hi)
        for previous, current in zip(group, group[1:]):
            assert current[0] == pytest.approx(previous[1])
        for start, end, cue_settings, lines in group:
            assert end > start
            assert cue_settings == settings
            assert 1 <= len(lines) <= 2
            assert all_lines_within(lines, 42)


def test_shot_aware_wrap_uses_unicode_line_breaker():
    from opencut.core.subtitle_shot_aware import _wrap_text

    text = "これは改行検証用の長い字幕テキストです"
    wrapped = _wrap_text(text, 8, 4)

    assert "".join(wrapped.splitlines()) == text
    assert all_lines_within(wrapped.splitlines(), 8)


def test_ffmpeg_drawtext_cjk_smoke_when_font_available(tmp_path):
    from opencut.core.caption_styles import CaptionStyle, _build_drawtext_filter, resolve_caption_font
    from opencut.helpers import get_ffmpeg_path

    ffmpeg_cmd = get_ffmpeg_path()
    ffmpeg = shutil.which(ffmpeg_cmd) or (ffmpeg_cmd if os.path.isfile(ffmpeg_cmd) else "")
    if not ffmpeg:
        pytest.skip("FFmpeg not available for CJK drawtext smoke")

    text = "\u65e5\u672c\u8a9e\u5b57\u5e55"
    style = CaptionStyle(font_family="Arial", font_size=32, colors={"text": "#FFFFFF", "shadow": "#000000"})
    resolution = resolve_caption_font(style.font_family, text)
    if resolution.source != "script_fallback_file" or not resolution.font_path:
        pytest.skip("No CJK-capable font file available for drawtext smoke")

    vf = _build_drawtext_filter(style, text, 0.0, 0.2, width=320, height=180)
    output = tmp_path / "cjk-caption.png"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x180:d=0.2",
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert "fontfile=''" not in vf
    assert result.returncode == 0, result.stderr[-1000:]
    assert output.exists()
    assert output.stat().st_size > 0


def test_positioned_webvtt_carries_a_rounded_second_into_the_minute(tmp_path):
    from opencut.core.broadcast_caption import CaptionSegment as BroadcastSegment
    from opencut.core.broadcast_caption import export_webvtt_positioned

    text = " ".join(["abcd"] * 16 + ["efgh"] * 16)
    output = tmp_path / "positioned.vtt"
    export_webvtt_positioned(
        [BroadcastSegment(index=1, start=49.9996, end=69.9996, text=text, row=15)], str(output)
    )

    stamps = [
        stamp
        for line in output.read_text(encoding="utf-8").splitlines()
        if "-->" in line
        for stamp in line.split()[:3:2]
    ]
    assert len(stamps) >= 4
    assert all(re.fullmatch(r"\d{2}:[0-5]\d:[0-5]\d\.\d{3}", stamp) for stamp in stamps), stamps
    assert "00:01:00.000" in stamps


@pytest.mark.parametrize(
    "module_name, func_name, mark",
    [
        ("opencut.core.broadcast_caption", "_seconds_to_vtt", "."),
        ("opencut.core.broadcast_caption", "_seconds_to_ttml", "."),
        ("opencut.core.subtitle_shot_aware", "_seconds_to_srt", ","),
        ("opencut.core.subtitle_shot_aware", "_seconds_to_vtt", "."),
        ("opencut.core.sdh_formatter", "_seconds_to_srt", ","),
        ("opencut.core.sdh_formatter", "_seconds_to_vtt", "."),
        ("opencut.core.multilang_subtitle", "_seconds_to_srt", ","),
        ("opencut.core.multilang_subtitle", "_seconds_to_vtt", "."),
        ("opencut.core.project_notes", "_fmt_timestamp", "."),
        ("opencut.core.silence", "_format_time", "."),
    ],
)
def test_clock_formatters_never_print_sixty_seconds(module_name, func_name, mark):
    import importlib

    fmt = getattr(importlib.import_module(module_name), func_name)
    assert fmt(59.9996) == f"00:01:00{mark}000"
    assert fmt(3599.9996) == f"01:00:00{mark}000"
    assert fmt(61.25) == f"00:01:01{mark}250"
