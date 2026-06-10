"""FFmpeg filtergraph / concat escaping helpers.

The exact escaped strings here were verified end-to-end against a real ffmpeg
(subtitles filter with Windows drive-letter paths, drawtext with apostrophes/
colons/percent under expansion=none, and concat lists with apostrophes and
non-ASCII names). These string-level assertions lock that behavior in for CI
where ffmpeg is not available.
"""
import os

from opencut.helpers import (
    _concat_quote,
    escape_drawtext,
    escape_filter_path,
    write_concat_list,
)


class TestEscapeFilterPath:
    def test_windows_drive_letter_colon_is_escaped(self):
        # The bug: a drive-letter colon inside single quotes was read by the
        # option parser as an option separator, breaking every Windows burn-in.
        assert escape_filter_path(r"C:\Users\me\sub.srt") == "C\\:/Users/me/sub.srt"

    def test_backslashes_become_forward_slashes(self):
        assert escape_filter_path(r"D:\a\b\c.ass") == "D\\:/a/b/c.ass"

    def test_apostrophe_uses_two_level_close_reopen(self):
        assert escape_filter_path("o'brian.srt") == "o\\'\\''brian.srt"

    def test_spaces_are_preserved(self):
        assert escape_filter_path("C:/my subs/a b.srt") == "C\\:/my subs/a b.srt"


class TestEscapeDrawtext:
    def test_plain_text_unchanged(self):
        assert escape_drawtext("Hello World") == "Hello World"

    def test_apostrophe(self):
        assert escape_drawtext("don't") == "don\\'\\''t"

    def test_colon_is_escaped(self):
        assert escape_drawtext("Chapter 1: start") == "Chapter 1\\: start"

    def test_percent_passthrough_relies_on_expansion_none(self):
        # Under expansion=none a literal % needs no escaping (and must not be
        # doubled, which would render two percent signs).
        assert escape_drawtext("100% done") == "100% done"

    def test_backslash_escaped_first(self):
        assert escape_drawtext("a\\b") == "a\\\\b"


class TestConcatList:
    def test_simple_quote_wrapping(self):
        assert _concat_quote("/a/b/clip.mp4") == "/a/b/clip.mp4"

    def test_apostrophe_close_reopen(self):
        assert _concat_quote("o'brian.mp4") == "o'\\''brian.mp4"

    def test_crlf_stripped(self):
        assert _concat_quote("a\r\nb.mp4") == "ab.mp4"

    def test_write_concat_list_is_utf8_and_escaped(self, tmp_path):
        lst = tmp_path / "list.txt"
        paths = [str(tmp_path / "клип.mp4"), str(tmp_path / "o'brian.mp4")]
        write_concat_list(paths, str(lst))
        raw = lst.read_bytes()
        # Non-ASCII names must be UTF-8 (cp1252 would corrupt or raise on write).
        text = raw.decode("utf-8")
        assert "клип.mp4" in text
        assert "o'\\''brian.mp4" in text
        assert text.startswith("file '")
        assert os.path.exists(str(lst))
