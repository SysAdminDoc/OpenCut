"""
Tests for the one-click Enhance macro (RESEARCH_FEATURE_PLAN_2026-05-25 Q3).

These exercise:
  - the style preset table (social / speech / cinematic);
  - the dry-run pipeline planner (no FFmpeg work, just plan);
  - the route surface (/enhance/auto/styles, /enhance/auto/dry-run);
  - the EnhanceResult subscript protocol so Flask jsonify works.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_silent_wav(path: str, seconds: float = 1.0, sr: int = 16000) -> None:
    n = int(sr * seconds)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


class TestEnhanceStylePresets(unittest.TestCase):
    def test_three_styles_exist(self):
        from opencut.core import enhance_auto
        self.assertEqual(set(enhance_auto.VALID_STYLES), {"social", "speech", "cinematic"})
        self.assertEqual(enhance_auto.DEFAULT_STYLE, "social")

    def test_social_targets_minus_16_lufs(self):
        from opencut.core import enhance_auto
        self.assertEqual(enhance_auto.STYLES["social"]["target_lufs"], -16.0)

    def test_cinematic_targets_minus_23_lufs(self):
        from opencut.core import enhance_auto
        self.assertEqual(enhance_auto.STYLES["cinematic"]["target_lufs"], -23.0)

    def test_speech_skips_grade(self):
        from opencut.core import enhance_auto
        self.assertIsNone(enhance_auto.STYLES["speech"]["grade_intent"])

    def test_unknown_style_raises(self):
        from opencut.core.enhance_auto import _resolve_style
        with self.assertRaises(ValueError):
            _resolve_style("not_a_style")


class TestEnhanceResultProtocol(unittest.TestCase):
    def test_result_is_subscriptable_for_jsonify(self):
        from opencut.core.enhance_auto import EnhanceResult, EnhanceStep
        r = EnhanceResult(
            output="/tmp/x.mp4",
            style="social",
            steps=[EnhanceStep(name="loudness", module="x")],
        )
        self.assertEqual(r["output"], "/tmp/x.mp4")
        self.assertEqual(r["style"], "social")
        self.assertIn("steps", r.keys())
        self.assertIsInstance(r["steps"], list)
        self.assertEqual(r["steps"][0]["name"], "loudness")


class TestEnhanceDryRun(unittest.TestCase):
    """Dry-run planning must NOT call FFmpeg; works against any extant file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="enhance_test_")
        self.path = os.path.join(self.tmp, "clip.wav")
        _write_silent_wav(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_returns_five_step_plan(self):
        from opencut.core import enhance_auto
        result = enhance_auto.enhance(self.path, style="social", dry_run=True)
        self.assertTrue(result.dry_run)
        self.assertEqual(len(result.steps), 5)
        names = [s.name for s in result.steps]
        self.assertEqual(names, ["loudness", "denoise", "stabilize", "deflicker", "grade"])

    def test_dry_run_skips_video_steps_on_audio_source(self):
        from opencut.core import enhance_auto
        result = enhance_auto.enhance(self.path, style="cinematic", dry_run=True)
        by_name = {s.name: s for s in result.steps}
        # WAV file has no video stream → stabilize / deflicker / grade are
        # skipped with a "no video stream" reason.
        self.assertEqual(by_name["stabilize"].status, "skipped")
        self.assertIn("no video", by_name["stabilize"].reason.lower())
        self.assertEqual(by_name["deflicker"].status, "skipped")
        self.assertEqual(by_name["grade"].status, "skipped")

    def test_dry_run_speech_skips_grade_step(self):
        from opencut.core import enhance_auto
        result = enhance_auto.enhance(self.path, style="speech", dry_run=True)
        by_name = {s.name: s for s in result.steps}
        self.assertEqual(by_name["grade"].status, "skipped")
        self.assertIn("intent", by_name["grade"].reason.lower())

    def test_dry_run_missing_file_raises(self):
        from opencut.core import enhance_auto
        with self.assertRaises(FileNotFoundError):
            enhance_auto.enhance("/nonexistent/path.mp4", dry_run=True)

    def test_dry_run_invalid_style_raises(self):
        from opencut.core import enhance_auto
        with self.assertRaises(ValueError):
            enhance_auto.enhance(self.path, style="bogus", dry_run=True)


class TestEnhanceRouteSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.token = cls.client.get("/health").get_json().get("csrf_token", "")
        cls.tmp = tempfile.mkdtemp(prefix="enhance_route_")
        cls.path = os.path.join(cls.tmp, "clip.wav")
        _write_silent_wav(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_styles_endpoint_lists_three_presets(self):
        resp = self.client.get("/enhance/auto/styles")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertEqual(set(body["styles"]), {"social", "speech", "cinematic"})
        self.assertEqual(body["default"], "social")

    def test_dry_run_route_returns_plan(self):
        resp = self.client.post(
            "/enhance/auto/dry-run",
            json={"filepath": self.path, "style": "social"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200, resp.data)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(body["dry_run"])
        self.assertEqual(body["style"], "social")
        self.assertEqual(len(body["steps"]), 5)

    def test_dry_run_rejects_missing_filepath(self):
        resp = self.client.post(
            "/enhance/auto/dry-run",
            json={"style": "social"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_dry_run_rejects_bad_style(self):
        resp = self.client.post(
            "/enhance/auto/dry-run",
            json={"filepath": self.path, "style": "definitely_not_valid"},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_enhance_endpoint_in_queue_allowlist(self):
        from opencut.routes.jobs_routes import _ALLOWED_QUEUE_ENDPOINTS
        self.assertIn("/enhance/auto", _ALLOWED_QUEUE_ENDPOINTS)


if __name__ == "__main__":
    unittest.main()
