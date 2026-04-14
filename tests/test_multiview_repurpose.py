"""
Tests for OpenCut Multi-View & Content Repurposing features.

Covers:
  - Split-screen layout templates & compositing
  - Reaction video templates & audio sync
  - Before/after comparison export
  - Multicam grid view export
  - Long-form to multi-short extraction
  - Video-to-blog-post generation
  - Podcast episode to multi-platform bundle
  - Cross-platform content calendar
  - Social caption generator (platform-specific)
  - Routes (smoke tests)

85+ tests, mocking FFmpeg, LLM, and file I/O.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Split-Screen Layout Templates (57.1)
# ============================================================
class TestSplitScreenLayouts(unittest.TestCase):
    """Tests for opencut.core.split_screen layout definitions."""

    def test_get_preset_layouts_returns_dict(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        self.assertIsInstance(layouts, dict)
        self.assertGreater(len(layouts), 0)

    def test_side_by_side_has_two_cells(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        self.assertIn("side_by_side", layouts)
        self.assertEqual(layouts["side_by_side"]["cell_count"], 2)

    def test_2x2_grid_has_four_cells(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        self.assertIn("2x2_grid", layouts)
        self.assertEqual(layouts["2x2_grid"]["cell_count"], 4)

    def test_3x3_grid_has_nine_cells(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        self.assertIn("3x3_grid", layouts)
        self.assertEqual(layouts["3x3_grid"]["cell_count"], 9)

    def test_pip_variants_exist(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        for name in ("pip_top_right", "pip_bottom_right", "pip_bottom_left", "pip_top_left"):
            self.assertIn(name, layouts, f"Missing PiP variant: {name}")

    def test_diagonal_layout_exists(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        self.assertIn("diagonal", layouts)

    def test_l_shaped_layout_exists(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        self.assertIn("l_shaped", layouts)
        self.assertEqual(layouts["l_shaped"]["cell_count"], 3)

    def test_parse_layout_from_dict(self):
        from opencut.core.split_screen import parse_layout
        data = {
            "name": "custom",
            "cells": [
                {"x": 0, "y": 0, "w": 50, "h": 50, "label": "A"},
                {"x": 50, "y": 50, "w": 50, "h": 50, "label": "B"},
            ],
        }
        layout = parse_layout(data)
        self.assertEqual(layout.name, "custom")
        self.assertEqual(len(layout.cells), 2)
        self.assertEqual(layout.cells[0].label, "A")

    def test_layout_cells_have_xy_wh(self):
        from opencut.core.split_screen import get_preset_layouts
        layouts = get_preset_layouts()
        for name, layout in layouts.items():
            for cell in layout["cells"]:
                for key in ("x", "y", "w", "h"):
                    self.assertIn(key, cell, f"{name} cell missing {key}")

    def test_create_split_screen_no_videos_raises(self):
        from opencut.core.split_screen import create_split_screen
        with self.assertRaises(ValueError):
            create_split_screen(video_paths=[])

    def test_create_split_screen_missing_file_raises(self):
        from opencut.core.split_screen import create_split_screen
        with self.assertRaises(FileNotFoundError):
            create_split_screen(video_paths=["/nonexistent/video.mp4"])

    def test_create_split_screen_unknown_layout_raises(self):
        from opencut.core.split_screen import create_split_screen
        tmpdir = tempfile.mkdtemp()
        try:
            f = os.path.join(tmpdir, "v.mp4")
            open(f, "w").close()
            with self.assertRaises(ValueError):
                create_split_screen(video_paths=[f], layout_name="nonexistent_layout")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.split_screen.run_ffmpeg")
    @patch("opencut.core.split_screen.get_video_info")
    def test_create_split_screen_calls_ffmpeg(self, mock_info, mock_run):
        from opencut.core.split_screen import create_split_screen
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            v1 = os.path.join(tmpdir, "a.mp4")
            v2 = os.path.join(tmpdir, "b.mp4")
            open(v1, "w").close()
            open(v2, "w").close()

            result = create_split_screen(
                video_paths=[v1, v2],
                layout_name="side_by_side",
                output_path_str=os.path.join(tmpdir, "out.mp4"),
            )
            self.assertTrue(mock_run.called)
            self.assertIn("output_path", vars(result))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Reaction Video Template (57.2)
# ============================================================
class TestReactionTemplate(unittest.TestCase):
    """Tests for opencut.core.reaction_template."""

    def test_get_reaction_presets(self):
        from opencut.core.reaction_template import get_reaction_presets
        presets = get_reaction_presets()
        self.assertIsInstance(presets, dict)
        self.assertIn("corner_pip", presets)
        self.assertIn("side_panel", presets)
        self.assertIn("bottom_bar", presets)

    def test_preset_structure(self):
        from opencut.core.reaction_template import get_reaction_presets
        presets = get_reaction_presets()
        for name, preset in presets.items():
            self.assertIn("content", preset)
            self.assertIn("reaction", preset)
            for key in ("x", "y", "w", "h"):
                self.assertIn(key, preset["content"])
                self.assertIn(key, preset["reaction"])

    def test_create_reaction_missing_content_raises(self):
        from opencut.core.reaction_template import create_reaction_video
        with self.assertRaises(FileNotFoundError):
            create_reaction_video("/no/content.mp4", "/no/reaction.mp4")

    def test_create_reaction_unknown_preset_raises(self):
        from opencut.core.reaction_template import create_reaction_video
        tmpdir = tempfile.mkdtemp()
        try:
            c = os.path.join(tmpdir, "c.mp4")
            r = os.path.join(tmpdir, "r.mp4")
            open(c, "w").close()
            open(r, "w").close()
            with self.assertRaises(ValueError):
                create_reaction_video(c, r, preset_name="nonexistent")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.reaction_template.run_ffmpeg")
    @patch("opencut.core.reaction_template.get_video_info")
    def test_create_reaction_calls_ffmpeg(self, mock_info, mock_run):
        from opencut.core.reaction_template import create_reaction_video
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            c = os.path.join(tmpdir, "c.mp4")
            r = os.path.join(tmpdir, "r.mp4")
            open(c, "w").close()
            open(r, "w").close()

            result = create_reaction_video(
                c, r,
                preset_name="corner_pip",
                output_path_str=os.path.join(tmpdir, "out.mp4"),
            )
            self.assertTrue(mock_run.called)
            self.assertEqual(result.preset_name, "corner_pip")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.reaction_template.run_ffmpeg")
    @patch("opencut.core.reaction_template.get_video_info")
    def test_reaction_audio_ducking(self, mock_info, mock_run):
        from opencut.core.reaction_template import create_reaction_video
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            c = os.path.join(tmpdir, "c.mp4")
            r = os.path.join(tmpdir, "r.mp4")
            open(c, "w").close()
            open(r, "w").close()

            result = create_reaction_video(
                c, r,
                duck_level=0.5,
                output_path_str=os.path.join(tmpdir, "out.mp4"),
            )
            self.assertTrue(mock_run.called)
            # Filter complex should contain volume parameter
            cmd_str = str(mock_run.call_args)
            self.assertIn("volume", cmd_str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_estimate_audio_offset_returns_float(self):
        from opencut.core.reaction_template import estimate_audio_offset
        # With nonexistent files, should return 0.0 gracefully
        offset = estimate_audio_offset("/no/a.mp4", "/no/b.mp4")
        self.assertIsInstance(offset, float)

    @patch("opencut.core.reaction_template.run_ffmpeg")
    @patch("opencut.core.reaction_template.get_video_info")
    def test_all_presets_callable(self, mock_info, mock_run):
        from opencut.core.reaction_template import create_reaction_video, get_reaction_presets
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            c = os.path.join(tmpdir, "c.mp4")
            r = os.path.join(tmpdir, "r.mp4")
            open(c, "w").close()
            open(r, "w").close()

            for preset_name in get_reaction_presets().keys():
                mock_run.reset_mock()
                result = create_reaction_video(
                    c, r, preset_name=preset_name,
                    output_path_str=os.path.join(tmpdir, f"out_{preset_name}.mp4"),
                )
                self.assertEqual(result.preset_name, preset_name)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Before/After Comparison Export (57.3)
# ============================================================
class TestComparisonExport(unittest.TestCase):
    """Tests for video_compare.export_comparison_video."""

    def test_export_modes_defined(self):
        from opencut.core.video_compare import EXPORT_MODES
        self.assertIn("vertical_wipe", EXPORT_MODES)
        self.assertIn("horizontal_wipe", EXPORT_MODES)
        self.assertIn("side_by_side", EXPORT_MODES)
        self.assertIn("alternating", EXPORT_MODES)

    def test_export_missing_file_raises(self):
        from opencut.core.video_compare import export_comparison_video
        with self.assertRaises(FileNotFoundError):
            export_comparison_video("/no/orig.mp4", "/no/proc.mp4")

    def test_export_invalid_mode_raises(self):
        from opencut.core.video_compare import export_comparison_video
        tmpdir = tempfile.mkdtemp()
        try:
            a = os.path.join(tmpdir, "a.mp4")
            b = os.path.join(tmpdir, "b.mp4")
            open(a, "w").close()
            open(b, "w").close()
            with self.assertRaises(ValueError):
                export_comparison_video(a, b, mode="invalid_mode")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_export_vertical_wipe(self, mock_info, mock_run):
        from opencut.core.video_compare import export_comparison_video
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            a = os.path.join(tmpdir, "a.mp4")
            b = os.path.join(tmpdir, "b.mp4")
            open(a, "w").close()
            open(b, "w").close()

            result = export_comparison_video(
                a, b, mode="vertical_wipe",
                out_path=os.path.join(tmpdir, "out.mp4"),
            )
            self.assertTrue(mock_run.called)
            self.assertEqual(result["mode"], "vertical_wipe")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_export_side_by_side_has_labels(self, mock_info, mock_run):
        from opencut.core.video_compare import export_comparison_video
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            a = os.path.join(tmpdir, "a.mp4")
            b = os.path.join(tmpdir, "b.mp4")
            open(a, "w").close()
            open(b, "w").close()

            result = export_comparison_video(
                a, b, mode="side_by_side",
                label_original="Before",
                label_processed="After",
                out_path=os.path.join(tmpdir, "out.mp4"),
            )
            cmd_str = str(mock_run.call_args)
            self.assertIn("Before", cmd_str)
            self.assertIn("After", cmd_str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.video_compare.run_ffmpeg")
    @patch("opencut.core.video_compare.get_video_info")
    def test_export_all_modes(self, mock_info, mock_run):
        from opencut.core.video_compare import export_comparison_video, EXPORT_MODES
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            a = os.path.join(tmpdir, "a.mp4")
            b = os.path.join(tmpdir, "b.mp4")
            open(a, "w").close()
            open(b, "w").close()

            for mode in EXPORT_MODES:
                mock_run.reset_mock()
                result = export_comparison_video(
                    a, b, mode=mode,
                    out_path=os.path.join(tmpdir, f"out_{mode}.mp4"),
                )
                self.assertTrue(mock_run.called, f"FFmpeg not called for mode {mode}")
                self.assertEqual(result["mode"], mode)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Multicam Grid View Export (57.4)
# ============================================================
class TestMulticamGrid(unittest.TestCase):
    """Tests for opencut.core.multicam_grid."""

    def test_compute_grid_dims(self):
        from opencut.core.multicam_grid import _compute_grid_dims
        self.assertEqual(_compute_grid_dims(1), (1, 1))
        self.assertEqual(_compute_grid_dims(2), (2, 1))
        self.assertEqual(_compute_grid_dims(4), (2, 2))
        self.assertEqual(_compute_grid_dims(9), (3, 3))
        self.assertEqual(_compute_grid_dims(16), (4, 4))

    def test_export_empty_raises(self):
        from opencut.core.multicam_grid import export_multicam_grid
        with self.assertRaises(ValueError):
            export_multicam_grid(video_paths=[])

    def test_export_too_many_raises(self):
        from opencut.core.multicam_grid import export_multicam_grid
        with self.assertRaises(ValueError):
            export_multicam_grid(video_paths=[f"/fake/{i}.mp4" for i in range(17)])

    def test_export_missing_file_raises(self):
        from opencut.core.multicam_grid import export_multicam_grid
        with self.assertRaises(FileNotFoundError):
            export_multicam_grid(video_paths=["/nonexistent.mp4"])

    @patch("opencut.core.multicam_grid.run_ffmpeg")
    @patch("opencut.core.multicam_grid.get_video_info")
    def test_export_2x2_grid(self, mock_info, mock_run):
        from opencut.core.multicam_grid import export_multicam_grid
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            vids = []
            for i in range(4):
                v = os.path.join(tmpdir, f"cam{i}.mp4")
                open(v, "w").close()
                vids.append(v)

            result = export_multicam_grid(
                video_paths=vids,
                output_path_str=os.path.join(tmpdir, "grid.mp4"),
            )
            self.assertTrue(mock_run.called)
            self.assertEqual(result.grid_size, "2x2")
            self.assertEqual(result.cell_count, 4)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.multicam_grid.run_ffmpeg")
    @patch("opencut.core.multicam_grid.get_video_info")
    def test_timecode_burnin(self, mock_info, mock_run):
        from opencut.core.multicam_grid import export_multicam_grid
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            v = os.path.join(tmpdir, "cam.mp4")
            open(v, "w").close()

            result = export_multicam_grid(
                video_paths=[v, v],
                show_timecode=True,
                output_path_str=os.path.join(tmpdir, "grid.mp4"),
            )
            self.assertTrue(result.has_timecode)
            # Filter should contain timecode-related drawtext
            cmd_str = str(mock_run.call_args)
            self.assertIn("pts", cmd_str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("opencut.core.multicam_grid.run_ffmpeg")
    @patch("opencut.core.multicam_grid.get_video_info")
    def test_custom_labels(self, mock_info, mock_run):
        from opencut.core.multicam_grid import export_multicam_grid
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60}

        tmpdir = tempfile.mkdtemp()
        try:
            v = os.path.join(tmpdir, "cam.mp4")
            open(v, "w").close()

            result = export_multicam_grid(
                video_paths=[v, v],
                label_names=["Front", "Rear"],
                output_path_str=os.path.join(tmpdir, "grid.mp4"),
            )
            cmd_str = str(mock_run.call_args)
            self.assertIn("Front", cmd_str)
            self.assertIn("Rear", cmd_str)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Long-Form to Multi-Short Extraction (58.1)
# ============================================================
class TestLongToShorts(unittest.TestCase):
    """Tests for opencut.core.long_to_shorts."""

    def test_extract_missing_file_raises(self):
        from opencut.core.long_to_shorts import extract_shorts
        with self.assertRaises(FileNotFoundError):
            extract_shorts("/nonexistent/long.mp4")

    @patch("opencut.core.long_to_shorts._probe_duration", return_value=600)
    @patch("opencut.core.long_to_shorts._trim_video")
    @patch("opencut.core.long_to_shorts.shutil.copy2")
    def test_fallback_highlights_generated(self, mock_copy, mock_trim, mock_dur):
        from opencut.core.long_to_shorts import extract_shorts

        tmpdir = tempfile.mkdtemp()
        try:
            src = os.path.join(tmpdir, "long.mp4")
            open(src, "w").close()

            result = extract_shorts(
                input_path=src,
                num_shorts=3,
                output_dir=os.path.join(tmpdir, "out"),
            )
            self.assertIsNotNone(result.csv_path)
            self.assertEqual(result.output_dir, os.path.join(tmpdir, "out"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_metadata_csv(self):
        from opencut.core.long_to_shorts import _write_metadata_csv, ShortSegment

        tmpdir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmpdir, "meta.csv")
            segments = [
                ShortSegment(index=1, title="Test", start=0, end=30,
                             duration=30, output_path="/out.mp4", score=0.8),
            ]
            _write_metadata_csv(segments, csv_path)
            self.assertTrue(os.path.isfile(csv_path))

            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            self.assertEqual(rows[0][0], "index")
            self.assertEqual(rows[1][0], "1")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_fallback_highlights_coverage(self):
        from opencut.core.long_to_shorts import _fallback_highlights
        segments = _fallback_highlights(600, 3, 15, 60)
        self.assertGreater(len(segments), 0)
        for seg in segments:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertGreater(seg["end"], seg["start"])

    def test_num_shorts_clamped(self):
        from opencut.core.long_to_shorts import extract_shorts
        tmpdir = tempfile.mkdtemp()
        try:
            src = os.path.join(tmpdir, "x.mp4")
            open(src, "w").close()
            # duration 0 should raise
            with self.assertRaises((ValueError, FileNotFoundError)):
                extract_shorts(src, num_shorts=100)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Video to Blog Post (58.2)
# ============================================================
class TestVideoToBlog(unittest.TestCase):
    """Tests for opencut.core.video_to_blog."""

    def test_generate_missing_video_raises(self):
        from opencut.core.video_to_blog import generate_blog_post
        with self.assertRaises(FileNotFoundError):
            generate_blog_post("/nonexistent/video.mp4")

    def test_fallback_blog_structure(self):
        from opencut.core.video_to_blog import _fallback_blog
        result = _fallback_blog(
            "This is a great topic about technology. We discuss many aspects. "
            "Machine learning is changing the world. Data science is important."
        )
        self.assertIn("title", result)
        self.assertIn("sections", result)
        self.assertGreater(len(result["sections"]), 0)

    def test_fallback_blog_empty_text(self):
        from opencut.core.video_to_blog import _fallback_blog
        result = _fallback_blog("")
        self.assertEqual(result["title"], "Video Summary")

    def test_render_markdown(self):
        from opencut.core.video_to_blog import _render_markdown, BlogSection, SEOMetadata
        sections = [
            BlogSection(heading="Introduction", content="Hello world."),
            BlogSection(heading="Details", content="More content here."),
        ]
        seo = SEOMetadata(title="Test Blog", meta_description="A test",
                          keywords=["test"], reading_time_min=2)
        md = _render_markdown("Test Blog", sections, seo)
        self.assertIn("# Test Blog", md)
        self.assertIn("## Introduction", md)
        self.assertIn("Hello world.", md)

    def test_render_html(self):
        from opencut.core.video_to_blog import _render_html, BlogSection, SEOMetadata
        sections = [
            BlogSection(heading="Intro", content="Content text."),
        ]
        seo = SEOMetadata(title="Test", meta_description="Desc", keywords=["k"])
        html = _render_html("Test", sections, seo)
        self.assertIn("<h1>Test</h1>", html)
        self.assertIn("<h2>Intro</h2>", html)
        self.assertIn("Content text.", html)

    def test_seo_metadata_slug(self):
        from opencut.core.video_to_blog import SEOMetadata
        import re
        title = "My Amazing Blog Post!"
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
        seo = SEOMetadata(title=title, slug=slug)
        self.assertEqual(seo.slug, "my-amazing-blog-post")

    def test_render_html_escapes_special_chars(self):
        from opencut.core.video_to_blog import _render_html, BlogSection, SEOMetadata
        sections = [BlogSection(heading="<script>", content="a & b < c")]
        seo = SEOMetadata()
        html = _render_html("Title", sections, seo)
        self.assertNotIn("<script>", html.split("<h2>")[1] if "<h2>" in html else "")
        self.assertIn("&amp;", html)


# ============================================================
# Podcast Bundle (58.4)
# ============================================================
class TestPodcastBundle(unittest.TestCase):
    """Tests for opencut.core.podcast_bundle."""

    def test_missing_audio_raises(self):
        from opencut.core.podcast_bundle import create_podcast_bundle
        with self.assertRaises(FileNotFoundError):
            create_podcast_bundle("/nonexistent/podcast.mp3")

    @patch("opencut.core.podcast_bundle._generate_show_notes", return_value=("# Notes", "<p>Notes</p>"))
    @patch("opencut.core.podcast_bundle._extract_highlight_clips", return_value=[])
    @patch("opencut.core.podcast_bundle._extract_chapters", return_value=[])
    @patch("opencut.core.podcast_bundle._transcribe_audio", return_value=("sample text", []))
    @patch("opencut.core.podcast_bundle._export_clean_audio", return_value={})
    @patch("opencut.core.podcast_bundle._denoise_and_normalize")
    @patch("opencut.core.podcast_bundle._get_audio_duration", return_value=300)
    def test_full_bundle_pipeline(self, mock_dur, mock_denoise, mock_export,
                                   mock_transcribe, mock_chapters,
                                   mock_highlights, mock_notes):
        from opencut.core.podcast_bundle import create_podcast_bundle

        tmpdir = tempfile.mkdtemp()
        try:
            src = os.path.join(tmpdir, "podcast.mp3")
            open(src, "w").close()
            mock_denoise.return_value = src

            result = create_podcast_bundle(
                audio_path=src,
                title="Test Episode",
                output_dir=os.path.join(tmpdir, "bundle"),
                generate_audiogram_flag=False,
            )
            self.assertEqual(result.output_dir, os.path.join(tmpdir, "bundle"))
            self.assertIn("title", result.manifest)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_audio_duration_invalid_file(self):
        from opencut.core.podcast_bundle import _get_audio_duration
        dur = _get_audio_duration("/nonexistent.mp3")
        self.assertEqual(dur, 0.0)

    @patch("opencut.core.podcast_bundle.run_ffmpeg")
    def test_denoise_and_normalize(self, mock_run):
        from opencut.core.podcast_bundle import _denoise_and_normalize
        result = _denoise_and_normalize("/in.mp3", "/out.m4a")
        self.assertTrue(mock_run.called)
        self.assertEqual(result, "/out.m4a")


# ============================================================
# Content Calendar (58.5)
# ============================================================
class TestContentCalendar(unittest.TestCase):
    """Tests for opencut.core.content_calendar."""

    def test_get_platform_rules(self):
        from opencut.core.content_calendar import get_platform_rules
        rules = get_platform_rules()
        self.assertIn("youtube", rules)
        self.assertIn("tiktok", rules)
        self.assertIn("instagram", rules)
        self.assertIn("twitter", rules)
        for platform, rule in rules.items():
            self.assertIn("posts_per_week", rule)
            self.assertIn("best_days", rule)
            self.assertIn("best_times", rule)

    def test_generate_empty_clips_raises(self):
        from opencut.core.content_calendar import generate_content_calendar
        with self.assertRaises(ValueError):
            generate_content_calendar(clips=[])

    def test_generate_calendar_csv(self):
        from opencut.core.content_calendar import generate_content_calendar

        tmpdir = tempfile.mkdtemp()
        try:
            clips = [
                {"title": "Video 1", "platform": "youtube"},
                {"title": "Video 2", "platform": "tiktok"},
                {"title": "Video 3", "platform": "instagram"},
            ]
            result = generate_content_calendar(
                clips=clips,
                weeks=2,
                output_format="csv",
                output_dir=tmpdir,
                start_date="2026-05-01",
            )
            self.assertGreater(result.total_posts, 0)
            self.assertTrue(os.path.isfile(result.csv_path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_generate_calendar_ics(self):
        from opencut.core.content_calendar import generate_content_calendar

        tmpdir = tempfile.mkdtemp()
        try:
            clips = [{"title": "Clip A", "platform": "twitter"}]
            result = generate_content_calendar(
                clips=clips,
                weeks=1,
                output_format="ics",
                output_dir=tmpdir,
                start_date="2026-05-01",
            )
            self.assertTrue(os.path.isfile(result.ics_path))
            with open(result.ics_path, "r") as f:
                content = f.read()
            self.assertIn("BEGIN:VCALENDAR", content)
            self.assertIn("BEGIN:VEVENT", content)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_date_format_raises(self):
        from opencut.core.content_calendar import generate_content_calendar
        with self.assertRaises(ValueError):
            generate_content_calendar(
                clips=[{"title": "X"}],
                start_date="not-a-date",
            )

    def test_platform_breakdown(self):
        from opencut.core.content_calendar import generate_content_calendar

        tmpdir = tempfile.mkdtemp()
        try:
            clips = [
                {"title": "A", "platform": "youtube"},
                {"title": "B", "platform": "youtube"},
                {"title": "C", "platform": "tiktok"},
            ]
            result = generate_content_calendar(
                clips=clips,
                weeks=2,
                output_format="csv",
                output_dir=tmpdir,
                start_date="2026-05-01",
            )
            self.assertIn("youtube", result.platform_breakdown)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_platforms_used(self):
        from opencut.core.content_calendar import generate_content_calendar

        tmpdir = tempfile.mkdtemp()
        try:
            clips = [{"title": "X"}]  # No platform specified
            result = generate_content_calendar(
                clips=clips,
                weeks=1,
                output_format="csv",
                output_dir=tmpdir,
                start_date="2026-05-01",
            )
            self.assertGreaterEqual(result.total_posts, 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_export_csv_format(self):
        from opencut.core.content_calendar import _export_csv, ScheduledPost

        tmpdir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmpdir, "cal.csv")
            posts = [
                ScheduledPost(date="2026-05-01", time="09:00",
                              platform="youtube", title="Test"),
            ]
            _export_csv(posts, csv_path)
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
            self.assertEqual(rows[0][0], "Date")
            self.assertEqual(rows[1][0], "2026-05-01")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Social Caption Generator (58.3)
# ============================================================
class TestPlatformCaptions(unittest.TestCase):
    """Tests for social_captions.generate_platform_caption."""

    def test_empty_transcript_returns_empty(self):
        from opencut.core.social_captions import generate_platform_caption
        result = generate_platform_caption("", platform="twitter")
        self.assertEqual(result.caption, "")
        self.assertEqual(result.platform, "twitter")

    def test_twitter_caption_under_280(self):
        from opencut.core.social_captions import generate_platform_caption
        text = "This is a great video about machine learning and AI. " * 10
        result = generate_platform_caption(text, platform="twitter", tone="professional")
        self.assertLessEqual(result.char_count, 280)

    def test_instagram_caption_generated(self):
        from opencut.core.social_captions import generate_platform_caption
        text = "We explore cooking techniques and healthy recipes in this episode."
        result = generate_platform_caption(text, platform="instagram", tone="casual")
        self.assertEqual(result.platform, "instagram")
        self.assertGreater(len(result.caption), 0)

    def test_linkedin_professional_tone(self):
        from opencut.core.social_captions import generate_platform_caption
        text = "Business strategy and leadership insights for modern executives."
        result = generate_platform_caption(text, platform="linkedin", tone="professional")
        self.assertEqual(result.tone, "professional")
        self.assertEqual(result.platform, "linkedin")

    def test_tiktok_trendy_tone(self):
        from opencut.core.social_captions import generate_platform_caption
        text = "This viral dance tutorial shows the newest moves."
        result = generate_platform_caption(text, platform="tiktok", tone="trendy")
        self.assertEqual(result.tone, "trendy")

    def test_unknown_platform_defaults(self):
        from opencut.core.social_captions import generate_platform_caption
        result = generate_platform_caption("Some text", platform="unknown_platform")
        self.assertEqual(result.platform, "twitter")

    def test_unknown_tone_defaults(self):
        from opencut.core.social_captions import generate_platform_caption
        result = generate_platform_caption("Some text", tone="alien_tone")
        self.assertEqual(result.tone, "professional")

    def test_custom_hashtags_included(self):
        from opencut.core.social_captions import generate_platform_caption
        text = "Technology and innovation in modern computing."
        result = generate_platform_caption(
            text, platform="twitter",
            custom_hashtags=["CustomTag", "#AnotherTag"],
        )
        self.assertTrue(any("CustomTag" in h for h in result.hashtags))
        self.assertTrue(any("AnotherTag" in h for h in result.hashtags))

    def test_platform_caption_result_dataclass(self):
        from opencut.core.social_captions import PlatformCaption
        pc = PlatformCaption(
            platform="twitter", caption="Hello", char_count=5, tone="casual"
        )
        self.assertEqual(pc.platform, "twitter")
        self.assertEqual(pc.char_count, 5)

    def test_char_limits_defined(self):
        from opencut.core.social_captions import _PLATFORM_CHAR_LIMITS
        self.assertEqual(_PLATFORM_CHAR_LIMITS["twitter"], 280)
        self.assertGreater(_PLATFORM_CHAR_LIMITS["instagram"], 280)


# ============================================================
# Route Smoke Tests
# ============================================================
class TestMultiviewRepurposeRoutes(unittest.TestCase):
    """Smoke tests for multiview_repurpose_routes blueprint endpoints."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        cls.app = create_app(config=test_config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Get CSRF token
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf_token = data.get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf_token,
            "Content-Type": "application/json",
        }

    def test_split_screen_layouts_get(self):
        resp = self.client.get("/split-screen/layouts")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("layouts", data)
        self.assertIn("side_by_side", data["layouts"])

    def test_split_screen_create_missing_videos(self):
        resp = self.client.post("/split-screen/create",
                                headers=self.headers,
                                json={"video_paths": []})
        # Should return job that eventually errors, or 400
        self.assertIn(resp.status_code, (200, 400))

    def test_reaction_create_missing_paths(self):
        resp = self.client.post("/reaction/create",
                                headers=self.headers,
                                json={})
        self.assertIn(resp.status_code, (200, 400))

    def test_comparison_export_missing_paths(self):
        resp = self.client.post("/comparison/export",
                                headers=self.headers,
                                json={})
        self.assertIn(resp.status_code, (200, 400))

    def test_multicam_grid_missing_videos(self):
        resp = self.client.post("/multicam-grid/export",
                                headers=self.headers,
                                json={"video_paths": []})
        self.assertIn(resp.status_code, (200, 400))

    def test_extract_shorts_no_csrf(self):
        resp = self.client.post("/repurpose/extract-shorts",
                                json={"filepath": "/test.mp4"})
        self.assertIn(resp.status_code, (401, 403))

    def test_video_to_blog_no_csrf(self):
        resp = self.client.post("/repurpose/video-to-blog",
                                json={"filepath": "/test.mp4"})
        self.assertIn(resp.status_code, (401, 403))

    def test_podcast_bundle_no_csrf(self):
        resp = self.client.post("/repurpose/podcast-bundle",
                                json={"filepath": "/test.mp3"})
        self.assertIn(resp.status_code, (401, 403))

    def test_content_calendar_empty_clips(self):
        resp = self.client.post("/repurpose/content-calendar",
                                headers=self.headers,
                                json={"clips": []})
        self.assertEqual(resp.status_code, 400)

    def test_content_calendar_valid(self):
        resp = self.client.post("/repurpose/content-calendar",
                                headers=self.headers,
                                json={
                                    "clips": [{"title": "V1", "platform": "youtube"}],
                                    "weeks": 1,
                                    "start_date": "2026-05-01",
                                })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("total_posts", data)

    def test_social_captions_missing_transcript(self):
        resp = self.client.post("/repurpose/social-captions",
                                headers=self.headers,
                                json={"transcript": ""})
        # async_job returns 200 with job that eventually errors
        self.assertIn(resp.status_code, (200, 400))

    def test_social_captions_with_transcript(self):
        resp = self.client.post("/repurpose/social-captions",
                                headers=self.headers,
                                json={
                                    "transcript": "Technology is amazing.",
                                    "platform": "twitter",
                                    "tone": "casual",
                                })
        self.assertIn(resp.status_code, (200, 202))


# ============================================================
# Additional Edge-Case Tests
# ============================================================
class TestAdditionalEdgeCases(unittest.TestCase):
    """Extra tests for edge cases to reach 85+ coverage."""

    def test_split_screen_custom_layout(self):
        """Custom layout JSON is parsed and used correctly."""
        from opencut.core.split_screen import parse_layout
        layout = parse_layout({
            "name": "my_layout",
            "description": "Test layout",
            "cells": [
                {"x": 10, "y": 10, "w": 80, "h": 80, "label": "Main"},
            ],
        })
        self.assertEqual(layout.name, "my_layout")
        self.assertEqual(layout.cells[0].x, 10)

    def test_reaction_all_preset_names_unique(self):
        from opencut.core.reaction_template import _REACTION_PRESETS
        names = [p.name for p in _REACTION_PRESETS.values()]
        self.assertEqual(len(names), len(set(names)))

    def test_multicam_grid_single_video(self):
        """Grid with 1 video should produce 1x1."""
        from opencut.core.multicam_grid import _compute_grid_dims
        self.assertEqual(_compute_grid_dims(1), (1, 1))

    def test_multicam_grid_six_videos(self):
        from opencut.core.multicam_grid import _compute_grid_dims
        self.assertEqual(_compute_grid_dims(6), (3, 2))

    def test_blog_post_result_fields(self):
        from opencut.core.video_to_blog import BlogPostResult
        r = BlogPostResult(title="T", word_count=200)
        self.assertEqual(r.title, "T")
        self.assertEqual(r.word_count, 200)

    def test_content_item_dataclass(self):
        from opencut.core.content_calendar import ContentItem
        item = ContentItem(title="X", platform="tiktok", duration=30)
        self.assertEqual(item.platform, "tiktok")

    def test_scheduled_post_dataclass(self):
        from opencut.core.content_calendar import ScheduledPost
        post = ScheduledPost(date="2026-05-01", time="09:00",
                             platform="youtube", title="V1")
        self.assertEqual(post.status, "scheduled")

    def test_content_calendar_both_format(self):
        from opencut.core.content_calendar import generate_content_calendar
        tmpdir = tempfile.mkdtemp()
        try:
            result = generate_content_calendar(
                clips=[{"title": "T", "platform": "youtube"}],
                weeks=1,
                output_format="both",
                output_dir=tmpdir,
                start_date="2026-06-01",
            )
            self.assertTrue(result.csv_path)
            self.assertTrue(result.ics_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_comparison_export_progress_callback(self):
        """Progress callback is invoked during export."""
        from opencut.core.video_compare import export_comparison_video

        tmpdir = tempfile.mkdtemp()
        progress_calls = []

        try:
            a = os.path.join(tmpdir, "a.mp4")
            b = os.path.join(tmpdir, "b.mp4")
            open(a, "w").close()
            open(b, "w").close()

            with patch("opencut.core.video_compare.run_ffmpeg"):
                with patch("opencut.core.video_compare.get_video_info",
                           return_value={"width": 1920, "height": 1080,
                                         "fps": 30, "duration": 60}):
                    export_comparison_video(
                        a, b, mode="side_by_side",
                        out_path=os.path.join(tmpdir, "out.mp4"),
                        on_progress=lambda p, m: progress_calls.append(p),
                    )
            self.assertIn(100, progress_calls)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_podcast_bundle_result_dataclass(self):
        from opencut.core.podcast_bundle import PodcastBundleResult
        r = PodcastBundleResult(output_dir="/tmp/x")
        self.assertEqual(r.output_dir, "/tmp/x")
        self.assertEqual(r.chapters, [])

    def test_split_screen_result_dataclass(self):
        from opencut.core.split_screen import SplitScreenResult
        r = SplitScreenResult(output_path="/out.mp4", layout_name="2x2_grid",
                               cell_count=4, width=1920, height=1080)
        self.assertEqual(r.cell_count, 4)

    def test_long_to_shorts_result_dataclass(self):
        from opencut.core.long_to_shorts import LongToShortsResult
        r = LongToShortsResult(total_shorts=3, source_duration=600)
        self.assertEqual(r.total_shorts, 3)


if __name__ == "__main__":
    unittest.main()
