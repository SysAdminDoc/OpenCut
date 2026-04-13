"""
Tests for OpenCut documentary & event features.

Covers: selects_bin, stringout_reel, archival_conform,
        brand_kit, guest_compilation, photo_montage, event_recap,
        and documentary_routes blueprint.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 14.2 — Selects Bin Tests
# ============================================================
class TestSelectsBin(unittest.TestCase):
    """Tests for opencut.core.selects_bin."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()
        self.tmp2 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp2.write(b"\x00" * 100)
        self.tmp2.close()
        # Use a temporary DB
        self._orig_get_db = None

    def tearDown(self):
        for f in (self.tmp.name, self.tmp2.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_rate_clip_basic(self, mock_db_path, mock_info):
        """Rating a clip should store the rating in SQLite."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import rate_clip
            result = rate_clip(self.tmp.name, 4)
            self.assertEqual(result["rating"], 4)
            self.assertEqual(result["status"], "ok")
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_rate_clip_update(self, mock_db_path, mock_info):
        """Re-rating a clip should update the existing rating."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import rate_clip
            rate_clip(self.tmp.name, 3)
            result = rate_clip(self.tmp.name, 5)
            self.assertEqual(result["rating"], 5)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def test_rate_clip_invalid_rating(self):
        """Rating outside 0-5 should raise ValueError."""
        from opencut.core.selects_bin import rate_clip
        with self.assertRaises(ValueError):
            rate_clip(self.tmp.name, 6)

    def test_rate_clip_missing_file(self):
        """Rating a nonexistent file should raise FileNotFoundError."""
        from opencut.core.selects_bin import rate_clip
        with self.assertRaises(FileNotFoundError):
            rate_clip("/nonexistent/clip.mp4", 3)

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_tag_clip_set(self, mock_db_path, mock_info):
        """Setting tags should replace all existing tags."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import tag_clip
            result = tag_clip(self.tmp.name, ["hero", "outdoor"])
            self.assertIn("hero", result["tags"])
            self.assertIn("outdoor", result["tags"])
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_tag_clip_add_mode(self, mock_db_path, mock_info):
        """Add mode should append tags without removing existing."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import tag_clip
            tag_clip(self.tmp.name, ["hero"])
            result = tag_clip(self.tmp.name, ["outdoor"], mode="add")
            self.assertIn("hero", result["tags"])
            self.assertIn("outdoor", result["tags"])
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_tag_clip_remove_mode(self, mock_db_path, mock_info):
        """Remove mode should remove specified tags."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import tag_clip
            tag_clip(self.tmp.name, ["hero", "outdoor", "sunset"])
            result = tag_clip(self.tmp.name, ["outdoor"], mode="remove")
            self.assertIn("hero", result["tags"])
            self.assertNotIn("outdoor", result["tags"])
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def test_tag_clip_invalid_tags(self):
        """Non-list tags should raise ValueError."""
        from opencut.core.selects_bin import tag_clip
        with self.assertRaises(ValueError):
            tag_clip(self.tmp.name, "not_a_list")

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_search_selects_by_rating(self, mock_db_path, mock_info):
        """Searching by min_rating should filter correctly."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import rate_clip, search_selects
            rate_clip(self.tmp.name, 5)
            rate_clip(self.tmp2.name, 2)
            result = search_selects({"min_rating": 4})
            self.assertEqual(len(result.clips), 1)
            self.assertEqual(result.clips[0].rating, 5)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_search_selects_by_tags(self, mock_db_path, mock_info):
        """Searching by required tags should filter correctly."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import rate_clip, search_selects, tag_clip
            rate_clip(self.tmp.name, 3)
            tag_clip(self.tmp.name, ["hero", "sunset"])
            rate_clip(self.tmp2.name, 3)
            tag_clip(self.tmp2.name, ["indoor"])
            result = search_selects({"tags": ["hero"]})
            paths = [c.clip_path for c in result.clips]
            self.assertIn(self.tmp.name, paths)
            self.assertNotIn(self.tmp2.name, paths)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_get_clip_metadata_in_bin(self, mock_db_path, mock_info):
        """Getting metadata for a clip in the bin should return full info."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        try:
            from opencut.core.selects_bin import get_clip_metadata, rate_clip
            rate_clip(self.tmp.name, 4)
            meta = get_clip_metadata(self.tmp.name)
            self.assertTrue(meta["in_bin"])
            self.assertEqual(meta["rating"], 4)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    @patch("opencut.core.selects_bin.get_video_info")
    def test_get_clip_metadata_not_in_bin(self, mock_info):
        """Getting metadata for a clip not in the bin should return in_bin=False."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.selects_bin import get_clip_metadata
        meta = get_clip_metadata(self.tmp.name)
        self.assertFalse(meta["in_bin"])

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_export_selects_json(self, mock_db_path, mock_info):
        """Exporting selects as JSON should produce valid JSON file."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        out_path = tempfile.mktemp(suffix=".json")
        try:
            from opencut.core.selects_bin import export_selects, rate_clip
            rate_clip(self.tmp.name, 5)
            result = export_selects(output_path_str=out_path, format="json")
            self.assertEqual(result["format"], "json")
            self.assertTrue(os.path.isfile(out_path))
            with open(out_path) as f:
                data = json.load(f)
            self.assertEqual(data["count"], 1)
        finally:
            for p in (db_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.selects_bin._get_db_path")
    def test_export_selects_csv(self, mock_db_path, mock_info):
        """Exporting selects as CSV should produce valid CSV file."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        db_path = tempfile.mktemp(suffix=".db")
        mock_db_path.return_value = db_path
        out_path = tempfile.mktemp(suffix=".csv")
        try:
            from opencut.core.selects_bin import export_selects, rate_clip
            rate_clip(self.tmp.name, 3)
            result = export_selects(output_path_str=out_path, format="csv")
            self.assertEqual(result["format"], "csv")
            self.assertTrue(os.path.isfile(out_path))
        finally:
            for p in (db_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_rate_clip_progress_callback(self):
        """Progress callback should be called during rate_clip."""
        from opencut.core.selects_bin import rate_clip
        progress = MagicMock()
        with patch("opencut.core.selects_bin.get_video_info") as mock_info, \
             patch("opencut.core.selects_bin._get_db_path") as mock_db:
            mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
            db_path = tempfile.mktemp(suffix=".db")
            mock_db.return_value = db_path
            try:
                rate_clip(self.tmp.name, 3, on_progress=progress)
                self.assertTrue(progress.called)
            finally:
                try:
                    os.unlink(db_path)
                except OSError:
                    pass


# ============================================================
# 14.3 — String-Out Reel Tests
# ============================================================
class TestStringOutReel(unittest.TestCase):
    """Tests for opencut.core.stringout_reel."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.stringout_reel.run_ffmpeg")
    @patch("opencut.core.stringout_reel.get_video_info")
    def test_generate_stringout_with_clips(self, mock_info, mock_ffmpeg, mock_sel_info):
        """Generating a string-out from explicit clip paths should succeed."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_sel_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.stringout_reel import generate_stringout
        result = generate_stringout(clip_paths=[self.tmp.name])
        self.assertEqual(result.clip_count, 1)
        self.assertGreater(result.duration, 0)

    def test_generate_stringout_no_inputs(self):
        """No filter or clips should raise ValueError."""
        from opencut.core.stringout_reel import generate_stringout
        with self.assertRaises(ValueError):
            generate_stringout()

    def test_generate_stringout_empty_clips(self):
        """Empty clip list should raise ValueError."""
        from opencut.core.stringout_reel import generate_stringout
        with self.assertRaises(ValueError):
            generate_stringout(clip_paths=["/nonexistent.mp4"])

    @patch("opencut.core.selects_bin.get_video_info")
    @patch("opencut.core.stringout_reel.run_ffmpeg")
    @patch("opencut.core.stringout_reel.get_video_info")
    def test_generate_stringout_chapters(self, mock_info, mock_ffmpeg, mock_sel_info):
        """String-out should generate chapter markers for each clip."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_sel_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        from opencut.core.stringout_reel import generate_stringout
        result = generate_stringout(clip_paths=[self.tmp.name])
        self.assertEqual(len(result.chapters), 1)
        self.assertEqual(result.chapters[0].start_time, 0)

    def test_add_chapter_markers_missing_file(self):
        """Adding chapters to missing file should raise FileNotFoundError."""
        from opencut.core.stringout_reel import add_chapter_markers
        with self.assertRaises(FileNotFoundError):
            add_chapter_markers([{"title": "Ch1", "start_time": 0, "end_time": 10}],
                               "/nonexistent.mp4")

    def test_add_chapter_markers_empty(self):
        """Empty chapters list should raise ValueError."""
        from opencut.core.stringout_reel import add_chapter_markers
        with self.assertRaises(ValueError):
            add_chapter_markers([], self.tmp.name)

    @patch("opencut.core.stringout_reel.run_ffmpeg")
    def test_add_chapter_markers_success(self, mock_ffmpeg):
        """Adding chapters to an existing file should succeed."""
        from opencut.core.stringout_reel import add_chapter_markers
        result = add_chapter_markers(
            [{"title": "Intro", "start_time": 0, "end_time": 30},
             {"title": "Main", "start_time": 30, "end_time": 120}],
            self.tmp.name,
        )
        self.assertEqual(result["chapter_count"], 2)


# ============================================================
# 14.4 — Archival Conform Tests
# ============================================================
class TestArchivalConform(unittest.TestCase):
    """Tests for opencut.core.archival_conform."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.archival_conform._probe_detailed")
    def test_analyze_conformance_all_match(self, mock_probe):
        """Clips matching target settings should report no issues."""
        mock_probe.return_value = {
            "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "field_order": "progressive",
        }
        from opencut.core.archival_conform import analyze_conformance
        target = {"fps": 30.0, "width": 1920, "height": 1080,
                  "pix_fmt": "yuv420p", "color_space": "bt709"}
        result = analyze_conformance([self.tmp.name], target)
        self.assertTrue(result.all_conformant)
        self.assertEqual(len(result.issues), 0)

    @patch("opencut.core.archival_conform._probe_detailed")
    def test_analyze_conformance_fps_mismatch(self, mock_probe):
        """Different frame rate should produce a frame_rate issue."""
        mock_probe.return_value = {
            "width": 1920, "height": 1080, "fps": 25.0, "duration": 10.0,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "field_order": "progressive",
        }
        from opencut.core.archival_conform import analyze_conformance
        target = {"fps": 30.0, "width": 1920, "height": 1080}
        result = analyze_conformance([self.tmp.name], target)
        self.assertFalse(result.all_conformant)
        types = [i.issue_type for i in result.issues]
        self.assertIn("frame_rate", types)

    @patch("opencut.core.archival_conform._probe_detailed")
    def test_analyze_conformance_resolution_mismatch(self, mock_probe):
        """Different resolution should produce a resolution issue."""
        mock_probe.return_value = {
            "width": 1280, "height": 720, "fps": 30.0, "duration": 10.0,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "field_order": "progressive",
        }
        from opencut.core.archival_conform import analyze_conformance
        target = {"fps": 30.0, "width": 1920, "height": 1080}
        result = analyze_conformance([self.tmp.name], target)
        types = [i.issue_type for i in result.issues]
        self.assertIn("resolution", types)

    @patch("opencut.core.archival_conform._probe_detailed")
    def test_analyze_conformance_interlaced(self, mock_probe):
        """Interlaced content against progressive target should flag."""
        mock_probe.return_value = {
            "width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "field_order": "tt",
        }
        from opencut.core.archival_conform import analyze_conformance
        target = {"fps": 30.0, "width": 1920, "height": 1080, "interlaced": False}
        result = analyze_conformance([self.tmp.name], target)
        types = [i.issue_type for i in result.issues]
        self.assertIn("interlacing", types)

    def test_analyze_conformance_empty_paths(self):
        """Empty file list should raise ValueError."""
        from opencut.core.archival_conform import analyze_conformance
        with self.assertRaises(ValueError):
            analyze_conformance([], {"fps": 30.0})

    @patch("opencut.core.archival_conform.run_ffmpeg")
    @patch("opencut.core.archival_conform._probe_detailed")
    def test_conform_clip_basic(self, mock_probe, mock_ffmpeg):
        """Conforming a clip should apply detected changes."""
        mock_probe.return_value = {
            "width": 1280, "height": 720, "fps": 25.0, "duration": 10.0,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "field_order": "progressive",
        }
        from opencut.core.archival_conform import conform_clip
        result = conform_clip(self.tmp.name, {"fps": 30.0, "width": 1920, "height": 1080})
        self.assertIn("output_path", result.__dict__)
        self.assertTrue(len(result.changes_applied) > 0)

    def test_conform_clip_missing_file(self):
        """Conforming a missing file should raise FileNotFoundError."""
        from opencut.core.archival_conform import conform_clip
        with self.assertRaises(FileNotFoundError):
            conform_clip("/nonexistent.mp4", {"fps": 30.0})

    @patch("opencut.core.archival_conform.run_ffmpeg")
    @patch("opencut.core.archival_conform._probe_detailed")
    def test_batch_conform(self, mock_probe, mock_ffmpeg):
        """Batch conform should process multiple files."""
        mock_probe.return_value = {
            "width": 1920, "height": 1080, "fps": 25.0, "duration": 10.0,
            "pix_fmt": "yuv420p", "color_space": "bt709",
            "field_order": "progressive",
        }
        from opencut.core.archival_conform import batch_conform
        result = batch_conform([self.tmp.name], {"fps": 30.0, "width": 1920, "height": 1080})
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["error_count"], 0)

    def test_batch_conform_empty(self):
        """Empty file list for batch conform should raise ValueError."""
        from opencut.core.archival_conform import batch_conform
        with self.assertRaises(ValueError):
            batch_conform([], {"fps": 30.0})


# ============================================================
# 15.1 — Brand Kit Tests
# ============================================================
class TestBrandKit(unittest.TestCase):
    """Tests for opencut.core.brand_kit."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()
        self.config_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        json.dump({
            "name": "TestBrand",
            "primary_color": "#FF0000",
            "secondary_color": "#00FF00",
            "font_family": "Helvetica",
            "logo_path": "",
            "watermark_text": "TestBrand 2026",
        }, self.config_file)
        self.config_file.close()

    def tearDown(self):
        for f in (self.tmp.name, self.config_file.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    def test_load_brand_kit_basic(self):
        """Loading a valid brand kit config should return BrandKit."""
        from opencut.core.brand_kit import load_brand_kit
        kit = load_brand_kit(self.config_file.name)
        self.assertEqual(kit.name, "TestBrand")
        self.assertEqual(kit.primary_color, "#FF0000")
        self.assertEqual(kit.font_family, "Helvetica")

    def test_load_brand_kit_missing_file(self):
        """Loading from missing config should raise FileNotFoundError."""
        from opencut.core.brand_kit import load_brand_kit
        with self.assertRaises(FileNotFoundError):
            load_brand_kit("/nonexistent/config.json")

    def test_brand_kit_dataclass_defaults(self):
        """BrandKit should have sensible defaults."""
        from opencut.core.brand_kit import BrandKit
        kit = BrandKit()
        self.assertEqual(kit.name, "Default")
        self.assertEqual(kit.primary_color, "#FFFFFF")
        self.assertEqual(kit.logo_position, "top_right")
        self.assertAlmostEqual(kit.logo_scale, 0.15)

    @patch("opencut.core.brand_kit.get_video_info")
    def test_check_brand_compliance_low_res(self, mock_info):
        """Low resolution video should flag compliance issue."""
        mock_info.return_value = {"width": 640, "height": 480, "fps": 30.0, "duration": 30.0}
        from opencut.core.brand_kit import BrandKit, check_brand_compliance
        kit = BrandKit(name="Test")
        report = check_brand_compliance(self.tmp.name, kit)
        types = [i.issue_type for i in report.issues]
        self.assertIn("low_resolution", types)

    @patch("opencut.core.brand_kit.get_video_info")
    def test_check_brand_compliance_missing_watermark(self, mock_info):
        """Brand with watermark_text should flag missing watermark."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.brand_kit import BrandKit, check_brand_compliance
        kit = BrandKit(name="Test", watermark_text="Brand 2026")
        report = check_brand_compliance(self.tmp.name, kit)
        types = [i.issue_type for i in report.issues]
        self.assertIn("missing_watermark", types)

    @patch("opencut.core.brand_kit.get_video_info")
    def test_check_brand_compliance_score(self, mock_info):
        """Compliance score should be between 0 and 100."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.brand_kit import BrandKit, check_brand_compliance
        kit = BrandKit()
        report = check_brand_compliance(self.tmp.name, kit)
        self.assertGreaterEqual(report.score, 0)
        self.assertLessEqual(report.score, 100)

    def test_check_brand_compliance_missing_video(self):
        """Checking compliance on missing video should raise FileNotFoundError."""
        from opencut.core.brand_kit import BrandKit, check_brand_compliance
        with self.assertRaises(FileNotFoundError):
            check_brand_compliance("/nonexistent.mp4", BrandKit())

    @patch("opencut.core.brand_kit.run_ffmpeg")
    @patch("opencut.core.brand_kit.get_video_info")
    def test_auto_correct_brand_watermark(self, mock_info, mock_ffmpeg):
        """Auto-correct with watermark should apply watermark correction."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.brand_kit import BrandKit, auto_correct_brand
        kit = BrandKit(watermark_text="Test Brand")
        result = auto_correct_brand(self.tmp.name, kit, add_logo=False)
        self.assertIn("watermark", result["corrections_applied"])

    @patch("opencut.core.brand_kit.run_ffmpeg")
    @patch("opencut.core.brand_kit.get_video_info")
    def test_auto_correct_brand_logo(self, mock_info, mock_ffmpeg):
        """Auto-correct with logo should apply logo overlay."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        logo_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        logo_tmp.write(b"\x89PNG" + b"\x00" * 100)
        logo_tmp.close()
        try:
            from opencut.core.brand_kit import BrandKit, auto_correct_brand
            kit = BrandKit(logo_path=logo_tmp.name)
            result = auto_correct_brand(self.tmp.name, kit)
            self.assertIn("logo_overlay", result["corrections_applied"])
        finally:
            try:
                os.unlink(logo_tmp.name)
            except OSError:
                pass

    def test_auto_correct_brand_missing_video(self):
        """Auto-correcting missing video should raise FileNotFoundError."""
        from opencut.core.brand_kit import BrandKit, auto_correct_brand
        with self.assertRaises(FileNotFoundError):
            auto_correct_brand("/nonexistent.mp4", BrandKit())


# ============================================================
# 48.2 — Guest Compilation Tests
# ============================================================
class TestGuestCompilation(unittest.TestCase):
    """Tests for opencut.core.guest_compilation."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()
        self.tmp_dir = tempfile.mkdtemp(prefix="opencut_test_guest_")

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass
        import shutil
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except OSError:
            pass

    def test_extract_name_simple(self):
        """Simple underscore name should extract correctly."""
        from opencut.core.guest_compilation import _extract_name_from_filename
        self.assertEqual(_extract_name_from_filename("John_Smith.mp4"), "John Smith")

    def test_extract_name_dashes(self):
        """Dash-separated name should extract correctly."""
        from opencut.core.guest_compilation import _extract_name_from_filename
        self.assertEqual(_extract_name_from_filename("jane-doe.mp4"), "Jane Doe")

    def test_extract_name_with_prefix(self):
        """Name with leading number prefix should strip it."""
        from opencut.core.guest_compilation import _extract_name_from_filename
        result = _extract_name_from_filename("03_Jane_Doe_congrats.mov")
        self.assertIn("Jane", result)
        self.assertIn("Doe", result)

    def test_extract_name_message_from(self):
        """'message_from_' prefix should be stripped."""
        from opencut.core.guest_compilation import _extract_name_from_filename
        result = _extract_name_from_filename("message_from_Bob.mp4")
        self.assertEqual(result, "Bob")

    def test_extract_name_empty_fallback(self):
        """Unrecognizable filename should fall back to 'Guest'."""
        from opencut.core.guest_compilation import _extract_name_from_filename
        result = _extract_name_from_filename("__.mp4")
        self.assertEqual(result, "Guest")

    @patch("opencut.core.guest_compilation.run_ffmpeg")
    @patch("opencut.core.guest_compilation.get_video_info")
    def test_generate_name_card(self, mock_info, mock_ffmpeg):
        """Name card generation should produce output path."""
        from opencut.core.guest_compilation import generate_name_card
        result = generate_name_card("Jane Doe")
        self.assertIn("output_path", result)
        self.assertEqual(result["name"], "Jane Doe")
        try:
            os.unlink(result["output_path"])
        except OSError:
            pass

    def test_generate_name_card_empty(self):
        """Empty name should raise ValueError."""
        from opencut.core.guest_compilation import generate_name_card
        with self.assertRaises(ValueError):
            generate_name_card("")

    @patch("opencut.core.guest_compilation.run_ffmpeg")
    @patch("opencut.core.guest_compilation.get_video_info")
    @patch("opencut.core.guest_compilation._detect_silence_boundaries")
    def test_process_single_message(self, mock_silence, mock_info, mock_ffmpeg):
        """Processing a single message should return ProcessedMessage."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0}
        mock_silence.return_value = (1.0, 28.0)
        from opencut.core.guest_compilation import process_single_message
        result = process_single_message(self.tmp.name, trim_silence=True)
        self.assertGreater(result.silence_removed, 0)
        self.assertEqual(result.audio_normalized, True)

    def test_process_single_message_missing_file(self):
        """Processing missing file should raise FileNotFoundError."""
        from opencut.core.guest_compilation import process_single_message
        with self.assertRaises(FileNotFoundError):
            process_single_message("/nonexistent.mp4")

    def test_compile_guest_messages_empty_folder(self):
        """Empty folder should raise ValueError."""
        from opencut.core.guest_compilation import compile_guest_messages
        with self.assertRaises(ValueError):
            compile_guest_messages(self.tmp_dir)

    def test_compile_guest_messages_missing_folder(self):
        """Missing folder should raise FileNotFoundError."""
        from opencut.core.guest_compilation import compile_guest_messages
        with self.assertRaises(FileNotFoundError):
            compile_guest_messages("/nonexistent/folder")

    @patch("opencut.core.guest_compilation.run_ffmpeg")
    @patch("opencut.core.guest_compilation.get_video_info")
    @patch("opencut.core.guest_compilation._detect_silence_boundaries")
    def test_compile_guest_messages_with_files(self, mock_silence, mock_info, mock_ffmpeg):
        """Folder with video files should compile successfully."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 20.0}
        mock_silence.return_value = (0.5, 19.0)
        # Create dummy video files in the temp dir
        for name in ["Alice_Smith.mp4", "Bob_Jones.mp4"]:
            fpath = os.path.join(self.tmp_dir, name)
            with open(fpath, "wb") as f:
                f.write(b"\x00" * 100)
        from opencut.core.guest_compilation import compile_guest_messages
        result = compile_guest_messages(self.tmp_dir, trim_silence=False)
        self.assertEqual(result.message_count, 2)
        self.assertGreater(result.total_duration, 0)


# ============================================================
# 48.3 — Photo Montage Tests
# ============================================================
class TestPhotoMontage(unittest.TestCase):
    """Tests for opencut.core.photo_montage."""

    def setUp(self):
        self.tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        self.tmp_img.write(b"\xff\xd8\xff" + b"\x00" * 100)
        self.tmp_img.close()
        self.tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp_vid.write(b"\x00" * 100)
        self.tmp_vid.close()

    def tearDown(self):
        for f in (self.tmp_img.name, self.tmp_vid.name):
            try:
                os.unlink(f)
            except OSError:
                pass

    def test_is_image(self):
        """Image extensions should be recognized."""
        from opencut.core.photo_montage import _is_image
        self.assertTrue(_is_image("photo.jpg"))
        self.assertTrue(_is_image("photo.PNG"))
        self.assertFalse(_is_image("video.mp4"))

    def test_is_video(self):
        """Video extensions should be recognized."""
        from opencut.core.photo_montage import _is_video
        self.assertTrue(_is_video("clip.mp4"))
        self.assertTrue(_is_video("clip.MOV"))
        self.assertFalse(_is_video("photo.jpg"))

    @patch("opencut.core.photo_montage.run_ffmpeg")
    def test_apply_ken_burns_zoom_in(self, mock_ffmpeg):
        """Ken Burns zoom_in should build a zoompan filter."""
        from opencut.core.photo_montage import apply_ken_burns
        result = apply_ken_burns(self.tmp_img.name, duration=3.0, effect="zoom_in")
        self.assertEqual(result["effect"], "zoom_in")
        self.assertEqual(result["duration"], 3.0)
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("zoompan", cmd_str)

    @patch("opencut.core.photo_montage.run_ffmpeg")
    def test_apply_ken_burns_zoom_out(self, mock_ffmpeg):
        """Ken Burns zoom_out should work."""
        from opencut.core.photo_montage import apply_ken_burns
        result = apply_ken_burns(self.tmp_img.name, effect="zoom_out")
        self.assertEqual(result["effect"], "zoom_out")

    @patch("opencut.core.photo_montage.run_ffmpeg")
    def test_apply_ken_burns_pan_left(self, mock_ffmpeg):
        """Ken Burns pan_left should work."""
        from opencut.core.photo_montage import apply_ken_burns
        result = apply_ken_burns(self.tmp_img.name, effect="pan_left")
        self.assertEqual(result["effect"], "pan_left")

    def test_apply_ken_burns_missing_image(self):
        """Ken Burns on missing image should raise FileNotFoundError."""
        from opencut.core.photo_montage import apply_ken_burns
        with self.assertRaises(FileNotFoundError):
            apply_ken_burns("/nonexistent.jpg")

    def test_sync_to_beats_basic(self):
        """Sync to beats should assign durations based on intervals."""
        from opencut.core.photo_montage import sync_to_beats
        clips = [{"media_path": "a.jpg"}, {"media_path": "b.jpg"}]
        beats = [0.0, 0.5, 1.0, 1.5, 2.0]
        result = sync_to_beats(clips, beats)
        self.assertEqual(len(result), 2)
        for r in result:
            self.assertIn("duration", r)
            self.assertGreater(r["duration"], 0)

    def test_sync_to_beats_empty_clips(self):
        """Empty clips should raise ValueError."""
        from opencut.core.photo_montage import sync_to_beats
        with self.assertRaises(ValueError):
            sync_to_beats([], [0.0, 0.5, 1.0])

    def test_sync_to_beats_empty_beats(self):
        """Empty beats should raise ValueError."""
        from opencut.core.photo_montage import sync_to_beats
        with self.assertRaises(ValueError):
            sync_to_beats([{"media_path": "a.jpg"}], [])

    def test_create_montage_empty_paths(self):
        """Empty media_paths should raise ValueError."""
        from opencut.core.photo_montage import create_montage
        with self.assertRaises(ValueError):
            create_montage([])

    def test_create_montage_no_valid_files(self):
        """Non-existent media paths should raise ValueError."""
        from opencut.core.photo_montage import create_montage
        with self.assertRaises(ValueError):
            create_montage(["/nonexistent.jpg", "/nonexistent.mp4"])

    @patch("shutil.copy2")
    @patch("opencut.core.photo_montage.run_ffmpeg")
    @patch("opencut.core.photo_montage.get_video_info")
    def test_create_montage_mixed_media(self, mock_info, mock_ffmpeg, mock_copy):
        """Montage with mixed images and videos should work."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 5.0}
        from opencut.core.photo_montage import create_montage
        result = create_montage(
            media_paths=[self.tmp_img.name, self.tmp_vid.name],
            sync_to_music=False,
        )
        self.assertEqual(result.image_count, 1)
        self.assertEqual(result.video_count, 1)
        self.assertEqual(result.segment_count, 2)

    @patch("shutil.copy2")
    @patch("opencut.core.photo_montage.run_ffmpeg")
    def test_create_montage_images_only(self, mock_ffmpeg, mock_copy):
        """Montage with only images should work."""
        from opencut.core.photo_montage import create_montage
        result = create_montage(
            media_paths=[self.tmp_img.name],
            sync_to_music=False,
        )
        self.assertEqual(result.image_count, 1)
        self.assertEqual(result.video_count, 0)


# ============================================================
# 48.4 — Event Recap Tests
# ============================================================
class TestEventRecap(unittest.TestCase):
    """Tests for opencut.core.event_recap."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_recap_config_defaults(self):
        """RecapConfig should have sensible defaults."""
        from opencut.core.event_recap import RecapConfig
        config = RecapConfig()
        self.assertAlmostEqual(config.target_duration, 180.0)
        self.assertAlmostEqual(config.audio_weight, 0.35)
        self.assertAlmostEqual(config.motion_weight, 0.30)

    @patch("opencut.core.event_recap._analyze_motion")
    @patch("opencut.core.event_recap._analyze_audio_energy")
    @patch("opencut.core.event_recap.get_video_info")
    def test_score_event_segments(self, mock_info, mock_audio, mock_motion):
        """Scoring should produce segments with combined scores."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        mock_audio.return_value = [0.3, 0.8, 0.5, 0.2, 0.9]
        mock_motion.return_value = [0.1, 0.7, 0.3, 0.6, 0.4]
        from opencut.core.event_recap import RecapConfig, score_event_segments
        config = RecapConfig(segment_analysis_interval=12.0)
        segments = score_event_segments(self.tmp.name, config)
        self.assertEqual(len(segments), 5)
        for seg in segments:
            self.assertGreaterEqual(seg.combined_score, 0)

    def test_score_event_segments_missing_file(self):
        """Scoring missing video should raise FileNotFoundError."""
        from opencut.core.event_recap import score_event_segments
        with self.assertRaises(FileNotFoundError):
            score_event_segments("/nonexistent.mp4")

    @patch("opencut.core.event_recap.get_video_info")
    def test_generate_recap_too_short(self, mock_info):
        """Video shorter than target should raise ValueError."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        from opencut.core.event_recap import generate_recap
        with self.assertRaises(ValueError):
            generate_recap(self.tmp.name, target_duration=120)

    def test_generate_recap_missing_file(self):
        """Recap of missing video should raise FileNotFoundError."""
        from opencut.core.event_recap import generate_recap
        with self.assertRaises(FileNotFoundError):
            generate_recap("/nonexistent.mp4")

    @patch("opencut.core.event_recap.run_ffmpeg")
    @patch("opencut.core.event_recap._analyze_motion")
    @patch("opencut.core.event_recap._analyze_audio_energy")
    @patch("opencut.core.event_recap.get_video_info")
    def test_generate_recap_success(self, mock_info, mock_audio, mock_motion, mock_ffmpeg):
        """Full recap generation should select segments and produce output."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 600.0}
        # 300 segments at 2s intervals = 600s
        mock_audio.return_value = [0.1 + (i % 10) * 0.08 for i in range(300)]
        mock_motion.return_value = [0.05 + (i % 7) * 0.1 for i in range(300)]
        from opencut.core.event_recap import RecapConfig, generate_recap
        config = RecapConfig(target_duration=30, segment_analysis_interval=2.0)
        result = generate_recap(self.tmp.name, target_duration=30, config=config)
        self.assertGreater(result.segments_selected, 0)
        self.assertGreater(result.compression_ratio, 1.0)

    @patch("opencut.core.event_recap._analyze_motion")
    @patch("opencut.core.event_recap._analyze_audio_energy")
    @patch("opencut.core.event_recap.get_video_info")
    def test_score_segments_progress(self, mock_info, mock_audio, mock_motion):
        """Progress callback should be invoked during scoring."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
        mock_audio.return_value = [0.5] * 5
        mock_motion.return_value = [0.5] * 5
        from opencut.core.event_recap import score_event_segments
        progress = MagicMock()
        score_event_segments(self.tmp.name, on_progress=progress)
        self.assertTrue(progress.called)


# ============================================================
# Documentary Routes Blueprint Tests
# ============================================================
class TestDocumentaryRoutes(unittest.TestCase):
    """Tests for the documentary_bp Flask routes."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def _get_app(self):
        """Create a test Flask app with the documentary blueprint."""
        from flask import Flask

        from opencut.routes.documentary_routes import documentary_bp
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(documentary_bp)
        return app

    def test_blueprint_registration(self):
        """Documentary blueprint should register without errors."""
        app = self._get_app()
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        self.assertIn("/selects/rate", rules)
        self.assertIn("/selects/tag", rules)
        self.assertIn("/selects/search", rules)
        self.assertIn("/selects/metadata", rules)
        self.assertIn("/selects/export", rules)
        self.assertIn("/stringout/generate", rules)
        self.assertIn("/stringout/chapters", rules)
        self.assertIn("/conform/analyze", rules)
        self.assertIn("/conform/clip", rules)
        self.assertIn("/conform/batch", rules)
        self.assertIn("/brand/load", rules)
        self.assertIn("/brand/check", rules)
        self.assertIn("/brand/auto-correct", rules)
        self.assertIn("/guest/compile", rules)
        self.assertIn("/guest/process-single", rules)
        self.assertIn("/guest/name-card", rules)
        self.assertIn("/montage/create", rules)
        self.assertIn("/montage/ken-burns", rules)
        self.assertIn("/recap/score", rules)
        self.assertIn("/recap/generate", rules)

    def test_all_routes_are_post(self):
        """All documentary routes should only accept POST."""
        app = self._get_app()
        for rule in app.url_map.iter_rules():
            if rule.rule.startswith(("/selects/", "/stringout/", "/conform/",
                                      "/brand/", "/guest/", "/montage/", "/recap/")):
                methods = rule.methods - {"OPTIONS", "HEAD"}
                self.assertEqual(methods, {"POST"},
                                 f"{rule.rule} should only accept POST, got {methods}")


if __name__ == "__main__":
    unittest.main()
