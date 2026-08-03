"""FFmpeg filtergraph / concat escaping helpers.

The exact escaped strings here were verified end-to-end against a real ffmpeg
(subtitles filter with Windows drive-letter paths, drawtext with apostrophes/
colons/percent under expansion=none, and concat lists with apostrophes and
non-ASCII names). These string-level assertions lock that behavior in for CI
where ffmpeg is not available.
"""
import os
import subprocess
from pathlib import Path

import pytest

from opencut.helpers import (
    _concat_file_line,
    _concat_quote,
    escape_drawtext,
    escape_filter_path,
    get_ffmpeg_path,
    write_concat_list,
)

CORE_CONCAT_MODULES = (
    "ai_intro_gen",
    "auto_chapter_art",
    "auto_dub_pipeline",
    "auto_montage",
    "beat_cuts",
    "beat_sync_edit",
    "ceremony_autoedit",
    "cursor_zoom",
    "dialogue_premix",
    "event_recap",
    "fit_to_fill",
    "generative_extend",
    "glitch_effects",
    "guest_compilation",
    "hook_generator",
    "instant_replay",
    "music_gen",
    "music_mood_morph",
    "music_remix",
    "paper_edit",
    "photo_montage",
    "render_farm",
    "rough_cut",
    "smart_render",
    "speaker_layout",
    "speed_ramp",
    "stream_highlights",
    "stringout_reel",
    "template_assembly",
    "timeline_copilot",
    "transcript_edit",
    "video_360",
    "video_condensed",
    "video_summary",
    "voice_overdub",
)

DRAWTEXT_LITERAL_MODULES = (
    "audiogram",
    "ab_variant",
    "adr_cueing",
    "brand_kit",
    "camera_solver",
    "callout_gen",
    "caption_styles",
    "character_consistency",
    "click_overlay",
    "data_animation",
    "declarative_compose",
    "end_screen",
    "guest_compilation",
    "hook_generator",
    "instant_replay",
    "kinetic_type",
    "multicam_grid",
    "news_ticker",
    "programmatic_video",
    "quiz_overlay",
    "safe_zones",
    "motion_graphics",
    "template_assembly",
    "telemetry_overlay",
    "thumbnail_ab",
    "video_compare",
    "watermark",
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


class TestDrawtextEntrypoints:
    """The affected public renderers must accept literal hostile text."""

    def test_literal_text_renders_through_title_and_replay_entrypoints(self, tmp_path):
        try:
            ffmpeg = get_ffmpeg_path()
        except (FileNotFoundError, RuntimeError) as exc:
            pytest.skip(f"FFmpeg unavailable for drawtext smoke: {exc}")

        source = tmp_path / "source.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x90:r=6:d=1",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(source),
            ],
            check=True,
            capture_output=True,
        )

        from opencut.core.instant_replay import ReplayConfig, create_replay
        from opencut.core.motion_graphics import overlay_title, render_title_card

        texts = ("Round 2: FIGHT", r"C:\media\clip", "100%{x}")
        for index, text in enumerate(texts):
            title_output = tmp_path / f"title-{index}.mp4"
            render_title_card(
                text,
                output_path=str(title_output),
                width=160,
                height=90,
                duration=0.4,
                fps=6,
            )
            assert title_output.stat().st_size > 0

            overlay_output = tmp_path / f"overlay-{index}.mp4"
            overlay_title(
                str(source),
                text,
                output_path=str(overlay_output),
                duration=0.4,
            )
            assert overlay_output.stat().st_size > 0

            replay_output = tmp_path / f"replay-{index}.mp4"
            create_replay(
                str(source),
                timestamp=0.5,
                output_path_str=str(replay_output),
                config=ReplayConfig(
                    pre_roll=0.1,
                    post_roll=0.3,
                    slow_factor=1.0,
                    transition="none",
                    include_original=False,
                    overlay_text=text,
                    font_size=18,
                ),
            )
            assert replay_output.stat().st_size > 0


class TestConcatList:
    def test_simple_quote_wrapping(self):
        assert _concat_quote("/a/b/clip.mp4") == "/a/b/clip.mp4"

    def test_apostrophe_close_reopen(self):
        assert _concat_quote("o'brian.mp4") == "o'\\''brian.mp4"

    def test_concat_file_line_formats_escaped_entry(self):
        assert _concat_file_line("o'brian.mp4") == "file 'o'\\''brian.mp4'\n"

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

    def test_core_modules_use_shared_concat_writer(self):
        repo_root = Path(__file__).resolve().parents[1]
        for module_name in CORE_CONCAT_MODULES:
            source = (repo_root / "opencut" / "core" / f"{module_name}.py").read_text(
                encoding="utf-8"
            )
            assert "write_concat_list" in source or "_concat_file_line" in source, module_name
            assert "file '" not in source, module_name

    def test_literal_drawtext_modules_use_shared_escape_and_expansion_none(self):
        repo_root = Path(__file__).resolve().parents[1]
        for module_name in DRAWTEXT_LITERAL_MODULES:
            source = (repo_root / "opencut" / "core" / f"{module_name}.py").read_text(
                encoding="utf-8"
            )
            assert "escape_drawtext" in source, module_name
            assert "drawtext=expansion=none:" in source, module_name
            assert '.replace("%", "%%")' not in source, module_name
            assert "drawtext=text='" not in source, module_name
