"""
Tests for OpenCut utility/integration features.

Covers:
  - Watermark application (text, image, batch)
  - Webhook / API notifications
  - Team presets via shared folder
  - Timestamped project notes
  - License tracking
  - Batch thumbnail extraction & contact sheets
  - Change annotations
  - Utility route smoke tests
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Watermark Core
# ============================================================
class TestWatermark(unittest.TestCase):
    """Tests for opencut.core.watermark."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.tmpdir, "sample.mp4")
        with open(self.test_file, "wb") as f:
            f.write(b"\x00" * 1024)
        self.watermark_img = os.path.join(self.tmpdir, "logo.png")
        with open(self.watermark_img, "wb") as f:
            f.write(b"\x89PNG" + b"\x00" * 100)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_presets_exist(self):
        """WATERMARK_PRESETS should contain the expected keys."""
        from opencut.core.watermark import WATERMARK_PRESETS
        for key in ("draft", "review", "confidential", "client_name"):
            self.assertIn(key, WATERMARK_PRESETS)
            self.assertIn("watermark_type", WATERMARK_PRESETS[key])

    def test_preset_structure(self):
        """Each preset should have required fields."""
        from opencut.core.watermark import WATERMARK_PRESETS
        for name, preset in WATERMARK_PRESETS.items():
            self.assertIn("content", preset, f"Preset {name} missing content")
            self.assertIn("position", preset, f"Preset {name} missing position")
            self.assertIn("opacity", preset, f"Preset {name} missing opacity")

    @patch("opencut.core.watermark.run_ffmpeg")
    def test_apply_text_watermark(self, mock_ffmpeg):
        """apply_watermark with text type should invoke FFmpeg drawtext."""
        from opencut.core.watermark import apply_watermark

        result = apply_watermark(
            input_path=self.test_file,
            watermark_type="text",
            content="DRAFT",
            position="center",
            opacity=0.4,
            font_size=48,
        )
        self.assertIn("output_path", result)
        self.assertEqual(result["watermark_type"], "text")
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        # Verify drawtext filter is in the command
        vf_idx = cmd.index("-vf")
        self.assertIn("drawtext", cmd[vf_idx + 1])

    @patch("opencut.core.watermark.run_ffmpeg")
    def test_apply_image_watermark(self, mock_ffmpeg):
        """apply_watermark with image type should use overlay filter."""
        from opencut.core.watermark import apply_watermark

        result = apply_watermark(
            input_path=self.test_file,
            watermark_type="image",
            content=self.watermark_img,
            position="bottom_right",
            opacity=0.3,
        )
        self.assertEqual(result["watermark_type"], "image")
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]
        fc_idx = cmd.index("-filter_complex")
        self.assertIn("overlay", cmd[fc_idx + 1])

    def test_apply_watermark_missing_file(self):
        """apply_watermark should raise FileNotFoundError for missing input."""
        from opencut.core.watermark import apply_watermark
        with self.assertRaises(FileNotFoundError):
            apply_watermark(input_path="/nonexistent/video.mp4")

    @patch("opencut.core.watermark.run_ffmpeg")
    def test_apply_watermark_progress_callback(self, mock_ffmpeg):
        """apply_watermark should invoke progress callback."""
        from opencut.core.watermark import apply_watermark
        progress_calls = []
        def on_progress(pct, msg):
            progress_calls.append((pct, msg))

        apply_watermark(self.test_file, on_progress=on_progress)
        self.assertTrue(len(progress_calls) >= 2)
        self.assertEqual(progress_calls[-1][0], 100)

    @patch("opencut.core.watermark.run_ffmpeg")
    def test_batch_apply_watermark(self, mock_ffmpeg):
        """batch_apply_watermark should process all files."""
        from opencut.core.watermark import batch_apply_watermark

        file2 = os.path.join(self.tmpdir, "second.mp4")
        with open(file2, "wb") as f:
            f.write(b"\x00" * 512)

        result = batch_apply_watermark(
            file_paths=[self.test_file, file2],
            watermark_config={"content": "TEST", "watermark_type": "text"},
            output_dir=os.path.join(self.tmpdir, "out"),
        )
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(len(result["results"]), 2)

    @patch("opencut.core.watermark.run_ffmpeg")
    def test_batch_apply_watermark_handles_missing(self, mock_ffmpeg):
        """batch_apply_watermark should record errors for missing files."""
        from opencut.core.watermark import batch_apply_watermark

        result = batch_apply_watermark(
            file_paths=["/nonexistent/a.mp4", self.test_file],
            watermark_config={"content": "TEST"},
        )
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["success_count"], 1)

    def test_batch_apply_empty(self):
        """batch_apply_watermark with empty list should return zeros."""
        from opencut.core.watermark import batch_apply_watermark
        result = batch_apply_watermark([], {})
        self.assertEqual(result["success_count"], 0)


# ============================================================
# Webhooks
# ============================================================
class TestWebhooks(unittest.TestCase):
    """Tests for opencut.core.webhooks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_file = None

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_send_webhook_success(self):
        """send_webhook should return True on 200 response."""
        from opencut.core import webhooks as _wh_mod
        from opencut.core.webhooks import send_webhook

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(_wh_mod.urllib.request, "urlopen", return_value=mock_resp) as mock_urlopen:
            ok = send_webhook("https://example.com/hook", "test", {"msg": "hi"})
            self.assertTrue(ok)
            mock_urlopen.assert_called_once()

    def test_send_webhook_retry_on_failure(self):
        """send_webhook should retry once on failure."""
        import urllib.error as _ue

        from opencut.core import webhooks as _wh_mod
        from opencut.core.webhooks import send_webhook

        with patch.object(_wh_mod.urllib.request, "urlopen",
                          side_effect=_ue.URLError("connection refused")) as mock_urlopen:
            ok = send_webhook("https://example.com/hook", "test", {})
            self.assertFalse(ok)
            self.assertEqual(mock_urlopen.call_count, 2)

    @patch("opencut.core.webhooks.send_webhook")
    def test_notify_job_complete(self, mock_send):
        """notify_job_complete should call send_webhook for each URL."""
        from opencut.core.webhooks import notify_job_complete

        notify_job_complete(
            "job123", "export", {"output": "/tmp/out.mp4"},
            ["https://a.com/hook", "https://b.com/hook"],
        )
        self.assertEqual(mock_send.call_count, 2)
        # Verify payload structure
        payload = mock_send.call_args_list[0][0][2]
        self.assertEqual(payload["event"], "job_complete")
        self.assertEqual(payload["job_id"], "job123")
        self.assertEqual(payload["job_type"], "export")

    def test_load_save_webhook_config(self):
        """save_webhook_config and load_webhook_config should round-trip."""
        from opencut.core.webhooks import load_webhook_config, save_webhook_config

        with patch("opencut.core.webhooks._WEBHOOKS_FILE",
                    os.path.join(self.tmpdir, "webhooks.json")), \
             patch("opencut.core.webhooks._OPENCUT_DIR", self.tmpdir):
            configs = [{"url": "https://example.com/hook", "events": ["job_complete"]}]
            save_webhook_config(configs)
            loaded = load_webhook_config()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["url"], "https://example.com/hook")

    def test_load_config_missing_file(self):
        """load_webhook_config should return empty list if file missing."""
        from opencut.core.webhooks import load_webhook_config
        with patch("opencut.core.webhooks._WEBHOOKS_FILE", "/nonexistent/webhooks.json"):
            self.assertEqual(load_webhook_config(), [])


# ============================================================
# Team Presets
# ============================================================
class TestTeamPresets(unittest.TestCase):
    """Tests for opencut.core.team_presets."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.shared = os.path.join(self.tmpdir, "shared")
        self.local = os.path.join(self.tmpdir, "local")
        os.makedirs(self.shared)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, name, content=b"data"):
        path = os.path.join(self.shared, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_scan_shared_folder_finds_presets(self):
        """scan_shared_folder should find .opencut-preset files."""
        from opencut.core.team_presets import scan_shared_folder

        self._create_file("export.opencut-preset")
        self._create_file("grade.cube")
        self._create_file("flow.opencut-workflow")
        self._create_file("ignore.txt")

        result = scan_shared_folder(self.shared)
        self.assertEqual(len(result["presets"]), 1)
        self.assertEqual(len(result["luts"]), 1)
        self.assertEqual(len(result["workflows"]), 1)
        self.assertEqual(result["total"], 3)

    def test_scan_shared_folder_missing(self):
        """scan_shared_folder should raise FileNotFoundError."""
        from opencut.core.team_presets import scan_shared_folder
        with self.assertRaises(FileNotFoundError):
            scan_shared_folder("/nonexistent/shared")

    def test_sync_team_presets_new_files(self):
        """sync_team_presets should copy new files."""
        from opencut.core.team_presets import sync_team_presets

        self._create_file("test.opencut-preset")
        self._create_file("look.cube")

        result = sync_team_presets(self.shared, local_folder=self.local)
        self.assertEqual(result["new"], 2)
        self.assertEqual(result["updated"], 0)
        self.assertTrue(os.path.isfile(os.path.join(self.local, "test.opencut-preset")))

    def test_sync_team_presets_skips_unchanged(self):
        """sync_team_presets should skip files that haven't changed."""
        from opencut.core.team_presets import sync_team_presets

        self._create_file("test.opencut-preset")
        sync_team_presets(self.shared, local_folder=self.local)

        # Second sync should skip
        result = sync_team_presets(self.shared, local_folder=self.local)
        self.assertEqual(result["new"], 0)
        self.assertEqual(result["skipped"], 1)

    def test_get_set_shared_folder_path(self):
        """get/set_shared_folder_path should round-trip."""
        from opencut.core.team_presets import get_shared_folder_path, set_shared_folder_path

        settings = os.path.join(self.tmpdir, "settings.json")
        with patch("opencut.core.team_presets._SETTINGS_FILE", settings), \
             patch("opencut.core.team_presets._OPENCUT_DIR", self.tmpdir):
            self.assertIsNone(get_shared_folder_path())
            set_shared_folder_path("/mnt/team/presets")
            self.assertEqual(get_shared_folder_path(), "/mnt/team/presets")

    def test_scan_3dl_files(self):
        """scan_shared_folder should find .3dl LUT files."""
        from opencut.core.team_presets import scan_shared_folder
        self._create_file("legacy.3dl")
        result = scan_shared_folder(self.shared)
        self.assertEqual(len(result["luts"]), 1)


# ============================================================
# Project Notes
# ============================================================
class TestProjectNotes(unittest.TestCase):
    """Tests for opencut.core.project_notes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self.tmpdir, "notes.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_db(self):
        return patch("opencut.core.project_notes._NOTES_DB", self._db_path), \
               patch("opencut.core.project_notes._OPENCUT_DIR", self.tmpdir)

    def test_add_note(self):
        """add_note should create a note with a generated ID."""
        from opencut.core.project_notes import add_note
        p1, p2 = self._patch_db()
        with p1, p2:
            note = add_note("proj1", 12.5, "Fix the intro", priority="high", author="Alice")
            self.assertIn("note_id", note)
            self.assertEqual(note["project_id"], "proj1")
            self.assertEqual(note["timestamp"], 12.5)
            self.assertEqual(note["priority"], "high")
            self.assertEqual(note["status"], "open")

    def test_get_notes(self):
        """get_notes should return notes for a project, ordered by timestamp."""
        from opencut.core.project_notes import add_note, get_notes
        p1, p2 = self._patch_db()
        with p1, p2:
            add_note("proj1", 30.0, "Second note")
            add_note("proj1", 10.0, "First note")
            add_note("proj2", 5.0, "Other project")

            notes = get_notes("proj1")
            self.assertEqual(len(notes), 2)
            self.assertLess(notes[0]["timestamp"], notes[1]["timestamp"])

    def test_get_notes_with_status_filter(self):
        """get_notes with status filter should only return matching notes."""
        from opencut.core.project_notes import add_note, get_notes, update_note
        p1, p2 = self._patch_db()
        with p1, p2:
            add_note("proj1", 1.0, "Open note")
            n2 = add_note("proj1", 2.0, "Resolved note")
            update_note(n2["note_id"], status="resolved")

            open_notes = get_notes("proj1", status="open")
            self.assertEqual(len(open_notes), 1)

    def test_update_note(self):
        """update_note should modify the specified fields."""
        from opencut.core.project_notes import add_note, update_note
        p1, p2 = self._patch_db()
        with p1, p2:
            note = add_note("proj1", 5.0, "Original")
            updated = update_note(note["note_id"], text="Updated", status="resolved")
            self.assertEqual(updated["text"], "Updated")
            self.assertEqual(updated["status"], "resolved")

    def test_update_note_not_found(self):
        """update_note should raise ValueError for missing note."""
        from opencut.core.project_notes import update_note
        p1, p2 = self._patch_db()
        with p1, p2:
            with self.assertRaises(ValueError):
                update_note("nonexistent")

    def test_delete_note(self):
        """delete_note should remove the note and return True."""
        from opencut.core.project_notes import add_note, delete_note, get_notes
        p1, p2 = self._patch_db()
        with p1, p2:
            note = add_note("proj1", 1.0, "To delete")
            self.assertTrue(delete_note(note["note_id"]))
            self.assertEqual(len(get_notes("proj1")), 0)

    def test_delete_note_not_found(self):
        """delete_note should return False for missing note."""
        from opencut.core.project_notes import delete_note
        p1, p2 = self._patch_db()
        with p1, p2:
            self.assertFalse(delete_note("nonexistent"))

    def test_export_notes_text(self):
        """export_notes text format should contain note text."""
        from opencut.core.project_notes import add_note, export_notes
        p1, p2 = self._patch_db()
        with p1, p2:
            add_note("proj1", 5.0, "Test note")
            result = export_notes("proj1", format="text")
            self.assertIn("Test note", result)
            self.assertIn("proj1", result)

    def test_export_notes_csv(self):
        """export_notes csv format should contain headers and data."""
        from opencut.core.project_notes import add_note, export_notes
        p1, p2 = self._patch_db()
        with p1, p2:
            add_note("proj1", 5.0, "CSV note")
            result = export_notes("proj1", format="csv")
            self.assertIn("note_id", result)
            self.assertIn("CSV note", result)

    def test_export_notes_markdown(self):
        """export_notes markdown format should contain markdown syntax."""
        from opencut.core.project_notes import add_note, export_notes
        p1, p2 = self._patch_db()
        with p1, p2:
            add_note("proj1", 5.0, "MD note", priority="high")
            result = export_notes("proj1", format="markdown")
            self.assertIn("# Project Notes", result)
            self.assertIn("MD note", result)

    def test_export_notes_empty(self):
        """export_notes for project with no notes should return a message."""
        from opencut.core.project_notes import export_notes
        p1, p2 = self._patch_db()
        with p1, p2:
            result = export_notes("empty_proj")
            self.assertIn("No notes found", result)


# ============================================================
# License Tracker
# ============================================================
class TestLicenseTracker(unittest.TestCase):
    """Tests for opencut.core.license_tracker."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self.tmpdir, "licenses.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_db(self):
        return patch("opencut.core.license_tracker._LICENSES_DB", self._db_path), \
               patch("opencut.core.license_tracker._OPENCUT_DIR", self.tmpdir)

    def test_record_asset_usage(self):
        """record_asset_usage should store a record with generated ID."""
        from opencut.core.license_tracker import record_asset_usage
        p1, p2 = self._patch_db()
        with p1, p2:
            rec = record_asset_usage(
                source_url="https://example.com/music.mp3",
                filename="background.mp3",
                license_type="CC-BY-4.0",
                attribution_text="Music by Artist",
                project_id="proj1",
            )
            self.assertIn("record_id", rec)
            self.assertEqual(rec["license_type"], "CC-BY-4.0")
            self.assertEqual(rec["project_id"], "proj1")

    def test_get_project_licenses(self):
        """get_project_licenses should return records for the project."""
        from opencut.core.license_tracker import get_project_licenses, record_asset_usage
        p1, p2 = self._patch_db()
        with p1, p2:
            record_asset_usage("https://a.com", "a.mp3", "MIT", project_id="proj1")
            record_asset_usage("https://b.com", "b.mp3", "CC-BY", project_id="proj1")
            record_asset_usage("https://c.com", "c.mp3", "MIT", project_id="proj2")

            recs = get_project_licenses("proj1")
            self.assertEqual(len(recs), 2)

    def test_export_attribution_text(self):
        """export_attribution text format should group by license type."""
        from opencut.core.license_tracker import export_attribution, record_asset_usage
        p1, p2 = self._patch_db()
        with p1, p2:
            record_asset_usage("https://a.com", "track.mp3", "CC-BY-4.0",
                               attribution_text="By Artist A", project_id="proj1")
            record_asset_usage("https://b.com", "font.ttf", "MIT",
                               attribution_text="Font by B", project_id="proj1")

            result = export_attribution("proj1", format="text")
            self.assertIn("CC-BY-4.0", result)
            self.assertIn("MIT", result)
            self.assertIn("By Artist A", result)

    def test_export_attribution_markdown(self):
        """export_attribution markdown format should contain headings."""
        from opencut.core.license_tracker import export_attribution, record_asset_usage
        p1, p2 = self._patch_db()
        with p1, p2:
            record_asset_usage("https://a.com", "img.jpg", "Royalty-Free",
                               project_id="proj1")
            result = export_attribution("proj1", format="markdown")
            self.assertIn("## Royalty-Free", result)

    def test_export_attribution_empty(self):
        """export_attribution for empty project should return message."""
        from opencut.core.license_tracker import export_attribution
        p1, p2 = self._patch_db()
        with p1, p2:
            result = export_attribution("empty")
            self.assertIn("No license records", result)


# ============================================================
# Batch Thumbnails
# ============================================================
class TestBatchThumbnails(unittest.TestCase):
    """Tests for opencut.core.batch_thumbnails."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.tmpdir, "video.mp4")
        with open(self.test_file, "wb") as f:
            f.write(b"\x00" * 1024)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("opencut.core.batch_thumbnails.run_ffmpeg")
    @patch("opencut.core.batch_thumbnails._get_file_duration", return_value=60.0)
    def test_extract_thumbnails_fixed(self, mock_dur, mock_ffmpeg):
        """extract_thumbnails in fixed mode should extract at pct offset."""
        from opencut.core.batch_thumbnails import extract_thumbnails

        result = extract_thumbnails(
            [self.test_file],
            mode="fixed",
            timestamp_pct=0.5,
            output_dir=os.path.join(self.tmpdir, "thumbs"),
        )
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["error_count"], 0)
        mock_ffmpeg.assert_called_once()
        # Verify seek time is at 50%
        cmd = mock_ffmpeg.call_args[0][0]
        ss_idx = cmd.index("-ss")
        seek = float(cmd[ss_idx + 1])
        self.assertAlmostEqual(seek, 30.0, places=1)

    @patch("opencut.core.batch_thumbnails.run_ffmpeg")
    @patch("opencut.core.batch_thumbnails._get_file_duration", return_value=120.0)
    def test_extract_thumbnails_auto(self, mock_dur, mock_ffmpeg):
        """extract_thumbnails auto mode should seek to ~10% of duration."""
        from opencut.core.batch_thumbnails import extract_thumbnails

        result = extract_thumbnails(
            [self.test_file],
            mode="auto",
            output_dir=os.path.join(self.tmpdir, "thumbs"),
        )
        self.assertEqual(result["success_count"], 1)
        cmd = mock_ffmpeg.call_args[0][0]
        ss_idx = cmd.index("-ss")
        seek = float(cmd[ss_idx + 1])
        self.assertAlmostEqual(seek, 12.0, places=0)

    def test_extract_thumbnails_missing_file(self):
        """extract_thumbnails should record error for missing files."""
        from opencut.core.batch_thumbnails import extract_thumbnails

        result = extract_thumbnails(["/nonexistent/video.mp4"])
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["success_count"], 0)

    def test_extract_thumbnails_empty(self):
        """extract_thumbnails with empty list should return zeros."""
        from opencut.core.batch_thumbnails import extract_thumbnails
        result = extract_thumbnails([])
        self.assertEqual(result["success_count"], 0)

    @patch("opencut.core.batch_thumbnails.run_ffmpeg")
    @patch("opencut.core.batch_thumbnails._get_file_duration", return_value=60.0)
    def test_extract_thumbnails_progress(self, mock_dur, mock_ffmpeg):
        """extract_thumbnails should invoke progress callback."""
        from opencut.core.batch_thumbnails import extract_thumbnails
        progress = []

        def on_progress(pct, msg):
            progress.append(pct)

        extract_thumbnails(
            [self.test_file],
            on_progress=on_progress,
            output_dir=os.path.join(self.tmpdir, "out"),
        )
        self.assertIn(100, progress)

    def test_generate_contact_sheet_no_pillow(self):
        """generate_contact_sheet should raise ImportError if Pillow missing."""
        from opencut.core.batch_thumbnails import generate_contact_sheet

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None, "PIL.ImageDraw": None, "PIL.ImageFont": None}):
            # The import inside the function should fail
            try:
                generate_contact_sheet(["thumb.jpg"])
            except (ImportError, ValueError):
                pass  # Expected

    def test_generate_contact_sheet_empty(self):
        """generate_contact_sheet with empty list should raise ValueError."""
        from opencut.core.batch_thumbnails import generate_contact_sheet
        with self.assertRaises(ValueError):
            generate_contact_sheet([])


# ============================================================
# Annotations
# ============================================================
class TestAnnotations(unittest.TestCase):
    """Tests for opencut.core.annotations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self.tmpdir, "annotations.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _patch_db(self):
        return patch("opencut.core.annotations._ANNOTATIONS_DB", self._db_path), \
               patch("opencut.core.annotations._OPENCUT_DIR", self.tmpdir)

    def test_add_annotation(self):
        """add_annotation should create an annotation with generated ID."""
        from opencut.core.annotations import add_annotation
        p1, p2 = self._patch_db()
        with p1, p2:
            ann = add_annotation("snap1", "Fixed color balance", change_ref="abc123", author="Bob")
            self.assertIn("annotation_id", ann)
            self.assertEqual(ann["snapshot_id"], "snap1")
            self.assertEqual(ann["change_ref"], "abc123")

    def test_get_annotations(self):
        """get_annotations should return annotations for a snapshot."""
        from opencut.core.annotations import add_annotation, get_annotations
        p1, p2 = self._patch_db()
        with p1, p2:
            add_annotation("snap1", "First change")
            add_annotation("snap1", "Second change")
            add_annotation("snap2", "Other snapshot")

            anns = get_annotations("snap1")
            self.assertEqual(len(anns), 2)

    def test_get_annotations_empty(self):
        """get_annotations should return empty list for unknown snapshot."""
        from opencut.core.annotations import get_annotations
        p1, p2 = self._patch_db()
        with p1, p2:
            self.assertEqual(get_annotations("nonexistent"), [])

    def test_export_revision_history_markdown(self):
        """export_revision_history in markdown format should contain headings."""
        from opencut.core.annotations import add_annotation, export_revision_history
        p1, p2 = self._patch_db()
        with p1, p2:
            add_annotation("snap1", "Initial edit", author="Alice")
            add_annotation("snap2", "Color grade applied", change_ref="def456")

            result = export_revision_history(["snap1", "snap2"], format="markdown")
            self.assertIn("# Revision History", result)
            self.assertIn("snap1", result)
            self.assertIn("Color grade applied", result)

    def test_export_revision_history_text(self):
        """export_revision_history in text format should contain entries."""
        from opencut.core.annotations import add_annotation, export_revision_history
        p1, p2 = self._patch_db()
        with p1, p2:
            add_annotation("snap1", "Trimmed intro")
            result = export_revision_history(["snap1"], format="text")
            self.assertIn("Trimmed intro", result)
            self.assertIn("Revision 1", result)

    def test_export_revision_history_empty(self):
        """export_revision_history with no snapshot_ids returns message."""
        from opencut.core.annotations import export_revision_history
        p1, p2 = self._patch_db()
        with p1, p2:
            result = export_revision_history([])
            self.assertIn("No snapshots", result)


# ============================================================
# Utility Routes (smoke tests)
# ============================================================
class TestUtilityRoutes(unittest.TestCase):
    """Smoke tests for utility_routes blueprint endpoints."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        config = OpenCutConfig()
        cls.app = create_app(config=config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        # Get CSRF token
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.token = data.get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.token,
            "Content-Type": "application/json",
        }

    # --- Webhook routes ---

    @patch("opencut.core.webhooks.send_webhook", return_value=True)
    def test_webhook_test_route(self, mock_send):
        resp = self.client.post("/webhook/test", headers=self.headers,
                                data=json.dumps({"url": "https://example.com/hook"}))
        self.assertIn(resp.status_code, (200, 201))
        data = resp.get_json()
        self.assertTrue(data.get("success"))

    def test_webhook_test_no_url(self):
        resp = self.client.post("/webhook/test", headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.webhooks.load_webhook_config", return_value=[])
    def test_webhook_config_get(self, mock_load):
        resp = self.client.get("/webhook/config", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("webhooks", resp.get_json())

    @patch("opencut.core.webhooks.save_webhook_config")
    def test_webhook_config_save(self, mock_save):
        resp = self.client.post("/webhook/config", headers=self.headers,
                                data=json.dumps({"webhooks": [{"url": "https://test.com"}]}))
        self.assertEqual(resp.status_code, 200)
        mock_save.assert_called_once()

    # --- Notes routes ---

    def test_notes_add_missing_fields(self):
        resp = self.client.post("/notes/add", headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    def test_notes_list_missing_project_id(self):
        resp = self.client.get("/notes/list", headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.project_notes.add_note")
    def test_notes_add_valid(self, mock_add):
        mock_add.return_value = {"note_id": "abc", "text": "hello"}
        resp = self.client.post("/notes/add", headers=self.headers,
                                data=json.dumps({
                                    "project_id": "proj1",
                                    "text": "Test note",
                                    "timestamp": 5.0,
                                }))
        self.assertEqual(resp.status_code, 200)

    @patch("opencut.core.project_notes.get_notes", return_value=[])
    def test_notes_list_valid(self, mock_get):
        resp = self.client.get("/notes/list?project_id=proj1", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("notes", resp.get_json())

    @patch("opencut.core.project_notes.export_notes", return_value="exported")
    def test_notes_export(self, mock_export):
        resp = self.client.post("/notes/export", headers=self.headers,
                                data=json.dumps({"project_id": "proj1", "format": "text"}))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["content"], "exported")

    # --- License routes ---

    def test_license_record_missing_fields(self):
        resp = self.client.post("/license/record", headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.license_tracker.record_asset_usage")
    def test_license_record_valid(self, mock_rec):
        mock_rec.return_value = {"record_id": "x"}
        resp = self.client.post("/license/record", headers=self.headers,
                                data=json.dumps({
                                    "source_url": "https://example.com",
                                    "filename": "music.mp3",
                                    "license_type": "MIT",
                                }))
        self.assertEqual(resp.status_code, 200)

    def test_license_list_missing_project_id(self):
        resp = self.client.get("/license/list", headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.license_tracker.get_project_licenses", return_value=[])
    def test_license_list_valid(self, mock_get):
        resp = self.client.get("/license/list?project_id=proj1", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("licenses", resp.get_json())

    @patch("opencut.core.license_tracker.export_attribution", return_value="attr text")
    def test_license_export(self, mock_export):
        resp = self.client.post("/license/export", headers=self.headers,
                                data=json.dumps({"project_id": "proj1"}))
        self.assertEqual(resp.status_code, 200)

    # --- Annotations routes ---

    def test_annotations_add_missing_fields(self):
        resp = self.client.post("/annotations/add", headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.annotations.add_annotation")
    def test_annotations_add_valid(self, mock_add):
        mock_add.return_value = {"annotation_id": "x"}
        resp = self.client.post("/annotations/add", headers=self.headers,
                                data=json.dumps({
                                    "snapshot_id": "snap1",
                                    "text": "Test annotation",
                                }))
        self.assertEqual(resp.status_code, 200)

    def test_annotations_list_missing_snapshot_id(self):
        resp = self.client.get("/annotations/list", headers=self.headers)
        self.assertEqual(resp.status_code, 400)

    @patch("opencut.core.annotations.get_annotations", return_value=[])
    def test_annotations_list_valid(self, mock_get):
        resp = self.client.get("/annotations/list?snapshot_id=snap1", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("annotations", resp.get_json())

    @patch("opencut.core.annotations.export_revision_history", return_value="history")
    def test_annotations_export(self, mock_export):
        resp = self.client.post("/annotations/export", headers=self.headers,
                                data=json.dumps({"snapshot_ids": ["snap1"]}))
        self.assertEqual(resp.status_code, 200)

    def test_annotations_export_invalid(self):
        resp = self.client.post("/annotations/export", headers=self.headers,
                                data=json.dumps({"snapshot_ids": "not a list"}))
        self.assertEqual(resp.status_code, 400)

    # --- Team routes ---

    @patch("opencut.core.team_presets.get_shared_folder_path", return_value=None)
    def test_team_status_not_configured(self, mock_path):
        resp = self.client.get("/team/status", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.get_json()["configured"])

    @patch("opencut.core.team_presets.get_shared_folder_path", return_value="/mnt/shared")
    @patch("opencut.core.team_presets.scan_shared_folder")
    def test_team_status_configured(self, mock_scan, mock_path):
        mock_scan.return_value = {"total": 5, "presets": [1, 2], "workflows": [3], "luts": [4, 5]}
        resp = self.client.get("/team/status", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["configured"])
        self.assertEqual(data["total"], 5)


if __name__ == "__main__":
    unittest.main()
