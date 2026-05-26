"""
Tests for the shorts A/B variant generator (RESEARCH_FEATURE_PLAN_2026-05-25 Q8).

Covers:
  * descriptor calculation (hook tightness, caption-style cycle, focal alternation);
  * window validation (invalid range, missing file);
  * dry-run planner returns N variant descriptors without invoking FFmpeg;
  * VariantsResult subscript protocol for Flask jsonify;
  * three new routes (info, dry-run, async-enqueue).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_silent_wav(path: str, seconds: float = 10.0, sr: int = 16000) -> None:
    n = int(sr * seconds)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


class TestDescriptorMath(unittest.TestCase):
    def test_descriptor_count_clamped(self):
        from opencut.core.shorts_variants import _variant_descriptor, MIN_VARIANTS, MAX_VARIANTS
        # Below minimum.
        d = _variant_descriptor(0.0, 10.0, 1, 1080, 1920)
        self.assertEqual(len(d), MIN_VARIANTS)
        # Above maximum.
        d = _variant_descriptor(0.0, 10.0, 99, 1080, 1920)
        self.assertEqual(len(d), MAX_VARIANTS)

    def test_hook_tightness_grows_per_variant(self):
        from opencut.core.shorts_variants import _variant_descriptor
        d = _variant_descriptor(5.0, 20.0, 3, 1080, 1920)
        self.assertEqual(d[0]["hook_offset"], 0.0)
        self.assertEqual(d[1]["hook_offset"], 0.5)
        self.assertEqual(d[2]["hook_offset"], 1.0)
        self.assertEqual(d[0]["start"], 5.0)
        self.assertEqual(d[1]["start"], 5.5)

    def test_face_track_alternates(self):
        from opencut.core.shorts_variants import _variant_descriptor
        d = _variant_descriptor(0.0, 10.0, 4, 1080, 1920)
        flags = [v["face_track"] for v in d]
        # Variant 0 + 2 face-tracked; 1 + 3 fixed-crop.
        self.assertEqual(flags, [True, False, True, False])

    def test_caption_style_cycles(self):
        from opencut.core.shorts_variants import _variant_descriptor, CAPTION_STYLES
        d = _variant_descriptor(0.0, 10.0, 4, 1080, 1920)
        styles = [v["caption_style"] for v in d]
        for i, s in enumerate(styles):
            self.assertEqual(s, CAPTION_STYLES[i % len(CAPTION_STYLES)])

    def test_hook_offset_clamped_below_duration(self):
        # Very short window — hook offsets must stay inside [0, duration-0.5].
        from opencut.core.shorts_variants import _variant_descriptor
        d = _variant_descriptor(0.0, 1.0, 6, 1080, 1920)
        for v in d:
            self.assertLessEqual(v["hook_offset"], 0.5)
            self.assertGreaterEqual(v["hook_offset"], 0.0)


class TestWindowValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="variants_test_")
        self.path = os.path.join(self.tmp, "clip.wav")
        _write_silent_wav(self.path, seconds=10.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_raises(self):
        from opencut.core.shorts_variants import _validate_window
        with self.assertRaises(FileNotFoundError):
            _validate_window("/nonexistent.mp4", 0.0, 5.0)

    def test_inverted_window_raises(self):
        from opencut.core.shorts_variants import _validate_window
        with self.assertRaises(ValueError):
            _validate_window(self.path, 5.0, 3.0)

    def test_negative_start_raises(self):
        from opencut.core.shorts_variants import _validate_window
        with self.assertRaises(ValueError):
            _validate_window(self.path, -1.0, 3.0)


class TestPlanVariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="variants_plan_")
        self.path = os.path.join(self.tmp, "clip.wav")
        _write_silent_wav(self.path, seconds=15.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_returns_dry_run_result(self):
        from opencut.core.shorts_variants import plan_variants
        result = plan_variants(self.path, 1.0, 8.0, n_variants=3)
        self.assertTrue(result.dry_run)
        self.assertEqual(len(result.variants), 3)
        self.assertEqual(result.input_path, self.path)

    def test_plan_clamp_n_to_max(self):
        from opencut.core.shorts_variants import plan_variants, MAX_VARIANTS
        result = plan_variants(self.path, 0.0, 5.0, n_variants=99)
        self.assertEqual(len(result.variants), MAX_VARIANTS)


class TestVariantsResultProtocol(unittest.TestCase):
    def test_result_is_subscriptable(self):
        from opencut.core.shorts_variants import VariantsResult, ShortsVariant
        r = VariantsResult(
            input_path="/tmp/in.mp4",
            start=0.0,
            end=5.0,
            variants=[ShortsVariant(
                variant_id=0,
                output="/tmp/v0.mp4",
                start=0.0, end=5.0,
                hook_offset=0.0,
                caption_style="default",
                face_track=True,
                width=1080, height=1920,
            )],
        )
        self.assertEqual(r["input_path"], "/tmp/in.mp4")
        self.assertEqual(r["variants"][0]["variant_id"], 0)
        self.assertIn("variants", r.keys())
        self.assertIn("dry_run", r.keys())


class TestShortsVariantsRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.token = cls.client.get("/health").get_json().get("csrf_token", "")
        cls.tmp = tempfile.mkdtemp(prefix="variants_route_")
        cls.path = os.path.join(cls.tmp, "clip.wav")
        _write_silent_wav(cls.path, seconds=10.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_info_endpoint(self):
        resp = self.client.get("/shorts/variants/info")
        self.assertEqual(resp.status_code, 200)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(body["available"])
        self.assertGreater(len(body["caption_styles"]), 0)
        self.assertEqual(body["min_variants"], 2)
        self.assertEqual(body["max_variants"], 6)

    def test_dry_run_endpoint_returns_descriptor_list(self):
        resp = self.client.post(
            "/shorts/variants/dry-run",
            json={"filepath": self.path, "start": 0.0, "end": 8.0, "n_variants": 3},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 200, resp.data)
        body = json.loads(resp.data.decode("utf-8"))
        self.assertTrue(body["dry_run"])
        self.assertEqual(len(body["variants"]), 3)
        self.assertEqual(body["variants"][0]["variant_id"], 0)

    def test_dry_run_rejects_invalid_window(self):
        resp = self.client.post(
            "/shorts/variants/dry-run",
            json={"filepath": self.path, "start": 5.0, "end": 5.0},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_dry_run_rejects_missing_filepath(self):
        resp = self.client.post(
            "/shorts/variants/dry-run",
            json={"start": 0.0, "end": 5.0},
            headers={"X-OpenCut-Token": self.token},
        )
        self.assertEqual(resp.status_code, 400)

    def test_variants_endpoint_in_queue_allowlist(self):
        from opencut.routes.jobs_routes import _ALLOWED_QUEUE_ENDPOINTS
        self.assertIn("/shorts/variants", _ALLOWED_QUEUE_ENDPOINTS)


if __name__ == "__main__":
    unittest.main()
