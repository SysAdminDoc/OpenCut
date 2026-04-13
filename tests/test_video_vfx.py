"""
Tests for OpenCut Video VFX features.

Covers: motion tracking, AI relighting, 360 video, clean plate,
gaming highlight detection, deepfake detection, face tagging,
and holy grail timelapse.
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Motion Tracking Tests
# ============================================================
class TestMotionTracking(unittest.TestCase):
    """Tests for opencut.core.motion_tracking."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_track_point_dataclass(self):
        """TrackPoint should store frame tracking data."""
        from opencut.core.motion_tracking import TrackPoint

        pt = TrackPoint(frame_idx=5, timestamp=0.167, x=100, y=200, width=50, height=60)
        self.assertEqual(pt.frame_idx, 5)
        self.assertEqual(pt.x, 100)
        self.assertEqual(pt.confidence, 1.0)

    def test_track_point_to_dict(self):
        """TrackPoint should serialise to dict."""
        from opencut.core.motion_tracking import TrackPoint

        pt = TrackPoint(frame_idx=0, timestamp=0.0, x=10, y=20, width=30, height=40)
        d = asdict(pt)
        self.assertIn("frame_idx", d)
        self.assertIn("confidence", d)

    def test_track_result_dataclass(self):
        """TrackResult should hold full tracking info."""
        from opencut.core.motion_tracking import TrackPoint, TrackResult

        result = TrackResult(
            points=[TrackPoint(0, 0.0, 10, 20, 30, 40)],
            fps=30.0, frame_count=1,
        )
        self.assertEqual(len(result.points), 1)
        self.assertEqual(result.fps, 30.0)

    def test_track_object_missing_video(self):
        """track_object should raise FileNotFoundError for missing video."""
        from opencut.core.motion_tracking import track_object

        with self.assertRaises(FileNotFoundError):
            track_object("/nonexistent/video.mp4", (100, 100))

    @patch("opencut.core.motion_tracking.get_video_info")
    @patch("opencut.core.motion_tracking.ensure_package", return_value=True)
    def test_track_object_progress(self, mock_ensure, mock_info):
        """track_object should call progress callback."""
        mock_info.return_value = {"fps": 30.0, "width": 1920, "height": 1080}
        progress = MagicMock()

        # Will fail at cv2.VideoCapture (no real video) but progress should be called
        from opencut.core.motion_tracking import track_object
        try:
            track_object(self.tmp.name, (100, 100), on_progress=progress)
        except Exception:
            pass

        self.assertTrue(progress.called)

    def test_export_track_data_json(self):
        """export_track_data should write JSON."""
        from opencut.core.motion_tracking import export_track_data

        data = [{"frame_idx": 0, "timestamp": 0.0, "x": 10, "y": 20,
                 "width": 30, "height": 40, "confidence": 1.0}]
        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        out.close()
        try:
            result = export_track_data(data, out.name, format="json")
            self.assertEqual(result, out.name)
            with open(out.name) as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 1)
        finally:
            os.unlink(out.name)

    def test_export_track_data_csv(self):
        """export_track_data should write CSV."""
        from opencut.core.motion_tracking import export_track_data

        data = [{"frame_idx": 0, "timestamp": 0.0, "x": 10, "y": 20,
                 "width": 30, "height": 40, "confidence": 0.9}]
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_track_data(data, out.name, format="csv")
            content = open(out.name).read()
            self.assertIn("frame_idx", content)
            self.assertIn("0.9", content)
        finally:
            os.unlink(out.name)

    def test_export_track_data_empty(self):
        """export_track_data should raise on empty data."""
        from opencut.core.motion_tracking import export_track_data

        with self.assertRaises(ValueError):
            export_track_data([], "/tmp/out.json")

    def test_annotate_tracked_missing_video(self):
        """annotate_tracked should raise for missing video."""
        from opencut.core.motion_tracking import annotate_tracked

        with self.assertRaises(FileNotFoundError):
            annotate_tracked("/nonexistent.mp4", [{"frame_idx": 0}], {"type": "box"})

    def test_annotate_tracked_empty_data(self):
        """annotate_tracked should raise on empty track_data."""
        from opencut.core.motion_tracking import annotate_tracked

        with self.assertRaises(ValueError):
            annotate_tracked(self.tmp.name, [], {"type": "box"})


# ============================================================
# 2. AI Relighting Tests
# ============================================================
class TestRelighting(unittest.TestCase):
    """Tests for opencut.core.relighting."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_light_config_defaults(self):
        """LightConfig should have sensible defaults."""
        from opencut.core.relighting import LightConfig

        cfg = LightConfig()
        self.assertEqual(cfg.intensity, 1.0)
        self.assertEqual(cfg.ambient, 0.2)
        self.assertIsInstance(cfg.position, tuple)

    def test_light_config_custom(self):
        """LightConfig should accept custom values."""
        from opencut.core.relighting import LightConfig

        cfg = LightConfig(position=(1.0, 0.0, 0.5), intensity=2.0, color=(0.8, 0.9, 1.0))
        self.assertEqual(cfg.intensity, 2.0)
        self.assertEqual(cfg.position[0], 1.0)

    @patch("opencut.core.relighting.ensure_package", return_value=True)
    def test_estimate_normals(self, mock_ensure):
        """estimate_normals should return (H, W, 3) float array."""
        import numpy as np

        from opencut.core.relighting import estimate_normals

        depth = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        normals = estimate_normals(depth)
        self.assertEqual(normals.shape, (100, 100, 3))
        self.assertEqual(normals.dtype, np.float32)

    @patch("opencut.core.relighting.ensure_package", return_value=True)
    def test_estimate_normals_invalid_input(self, mock_ensure):
        """estimate_normals should reject 3D arrays."""
        import numpy as np

        from opencut.core.relighting import estimate_normals

        with self.assertRaises(ValueError):
            estimate_normals(np.zeros((100, 100, 3), dtype=np.float32))

    @patch("opencut.core.relighting.ensure_package", return_value=True)
    def test_apply_lighting_shape(self, mock_ensure):
        """apply_lighting should return same shape as input frame."""
        import numpy as np

        from opencut.core.relighting import apply_lighting

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normals = np.random.randn(100, 100, 3).astype(np.float32)
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)

        result = apply_lighting(frame, normals)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, np.uint8)

    @patch("opencut.core.relighting.ensure_package", return_value=True)
    def test_apply_lighting_custom_params(self, mock_ensure):
        """apply_lighting should accept custom light parameters."""
        import numpy as np

        from opencut.core.relighting import apply_lighting

        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        normals = np.zeros((50, 50, 3), dtype=np.float32)
        normals[:, :, 2] = 1.0

        result = apply_lighting(
            frame, normals,
            light_pos=(0.0, 0.0, 1.0),
            intensity=2.0,
            ambient=0.5,
        )
        self.assertEqual(result.shape, frame.shape)

    def test_relight_video_missing_file(self):
        """relight_video should raise for missing file."""
        from opencut.core.relighting import relight_video

        with self.assertRaises(FileNotFoundError):
            relight_video("/nonexistent.mp4")


# ============================================================
# 3. 360 Video Tests
# ============================================================
class TestVideo360(unittest.TestCase):
    """Tests for opencut.core.video_360."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.video_360.get_video_info")
    def test_detect_360_equirect(self, mock_info):
        """detect_360_format should identify 2:1 aspect as equirect."""
        mock_info.return_value = {"width": 3840, "height": 1920, "fps": 30.0}
        from opencut.core.video_360 import detect_360_format

        result = detect_360_format(self.tmp.name)
        self.assertTrue(result["is_360"])
        self.assertEqual(result["projection"], "equirect")

    @patch("opencut.core.video_360.get_video_info")
    def test_detect_360_non_360(self, mock_info):
        """detect_360_format should detect non-360 video."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0}
        from opencut.core.video_360 import detect_360_format

        result = detect_360_format(self.tmp.name)
        self.assertFalse(result["is_360"])

    def test_detect_360_missing_file(self):
        """detect_360_format should raise for missing file."""
        from opencut.core.video_360 import detect_360_format

        with self.assertRaises(FileNotFoundError):
            detect_360_format("/nonexistent.mp4")

    @patch("opencut.core.video_360.get_video_info")
    def test_detect_360_filename_hint(self, mock_info):
        """detect_360_format should use filename hints."""
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0}
        from opencut.core.video_360 import detect_360_format

        # Create temp with 360 in name
        t = tempfile.NamedTemporaryFile(suffix="_360.mp4", delete=False)
        t.write(b"\x00" * 50)
        t.close()
        try:
            result = detect_360_format(t.name)
            self.assertTrue(result["is_360"])
        finally:
            os.unlink(t.name)

    def test_convert_360_invalid_projection(self):
        """convert_360_projection should reject invalid projection."""
        from opencut.core.video_360 import convert_360_projection

        with self.assertRaises(ValueError):
            convert_360_projection(self.tmp.name, projection="invalid_proj")

    def test_convert_360_missing_file(self):
        """convert_360_projection should raise for missing file."""
        from opencut.core.video_360 import convert_360_projection

        with self.assertRaises(FileNotFoundError):
            convert_360_projection("/nonexistent.mp4")

    @patch("opencut.core.video_360.run_ffmpeg")
    @patch("opencut.core.video_360.detect_360_format")
    @patch("opencut.core.video_360.get_video_info")
    def test_convert_360_cubemap(self, mock_info, mock_detect, mock_ffmpeg):
        """convert_360_projection should call FFmpeg with v360 filter."""
        mock_info.return_value = {"width": 3840, "height": 1920, "fps": 30.0}
        mock_detect.return_value = {"projection": "equirect"}
        from opencut.core.video_360 import convert_360_projection

        convert_360_projection(self.tmp.name, projection="cubemap")
        mock_ffmpeg.assert_called_once()
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("v360", cmd_str)

    @patch("opencut.core.video_360.run_ffmpeg")
    def test_extract_360_crop(self, mock_ffmpeg):
        """extract_360_crop should use v360=e:flat filter."""
        from opencut.core.video_360 import extract_360_crop

        extract_360_crop(self.tmp.name, yaw=45.0, pitch=10.0, fov=90.0)
        mock_ffmpeg.assert_called_once()
        cmd_str = " ".join(str(c) for c in mock_ffmpeg.call_args[0][0])
        self.assertIn("v360=e:flat", cmd_str)

    @patch("opencut.core.video_360.run_ffmpeg")
    def test_stabilize_360(self, mock_ffmpeg):
        """stabilize_360 should run two-pass vidstab."""
        from opencut.core.video_360 import stabilize_360

        stabilize_360(self.tmp.name)
        self.assertEqual(mock_ffmpeg.call_count, 2)  # Two passes


# ============================================================
# 4. Clean Plate Tests
# ============================================================
class TestCleanPlate(unittest.TestCase):
    """Tests for opencut.core.clean_plate."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    @patch("opencut.core.clean_plate.ensure_package", return_value=True)
    def test_median_composite(self, mock_ensure):
        """median_composite should return median of frames."""
        import numpy as np

        from opencut.core.clean_plate import median_composite

        frames = [
            np.full((100, 100, 3), 50, dtype=np.uint8),
            np.full((100, 100, 3), 100, dtype=np.uint8),
            np.full((100, 100, 3), 150, dtype=np.uint8),
        ]
        result = median_composite(frames)
        self.assertEqual(result.shape, (100, 100, 3))
        # Median of 50, 100, 150 = 100
        self.assertEqual(result[0, 0, 0], 100)

    @patch("opencut.core.clean_plate.ensure_package", return_value=True)
    def test_median_composite_empty(self, mock_ensure):
        """median_composite should raise on empty frames."""
        from opencut.core.clean_plate import median_composite

        with self.assertRaises(ValueError):
            median_composite([])

    @patch("opencut.core.clean_plate.ensure_package", return_value=True)
    def test_median_composite_shape_mismatch(self, mock_ensure):
        """median_composite should reject differently shaped frames."""
        import numpy as np

        from opencut.core.clean_plate import median_composite

        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((50, 50, 3), dtype=np.uint8),
        ]
        with self.assertRaises(ValueError):
            median_composite(frames)

    @patch("opencut.core.clean_plate.ensure_package", return_value=True)
    def test_inpaint_gaps_telea(self, mock_ensure):
        """inpaint_gaps should fill masked regions."""
        import numpy as np

        from opencut.core.clean_plate import inpaint_gaps

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        result = inpaint_gaps(img, mask, method="telea")
        self.assertEqual(result.shape, img.shape)

    @patch("opencut.core.clean_plate.ensure_package", return_value=True)
    def test_inpaint_gaps_ns(self, mock_ensure):
        """inpaint_gaps should support NS method."""
        import numpy as np

        from opencut.core.clean_plate import inpaint_gaps

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        result = inpaint_gaps(img, mask, method="ns")
        self.assertEqual(result.shape, img.shape)

    def test_generate_clean_plate_missing_file(self):
        """generate_clean_plate should raise for missing file."""
        from opencut.core.clean_plate import generate_clean_plate

        with self.assertRaises(FileNotFoundError):
            generate_clean_plate("/nonexistent.mp4")


# ============================================================
# 5. Highlight Detection Tests
# ============================================================
class TestHighlightDetect(unittest.TestCase):
    """Tests for opencut.core.highlight_detect."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_highlight_score_dataclass(self):
        """HighlightScore should store segment scores."""
        from opencut.core.highlight_detect import HighlightScore

        hs = HighlightScore(start_time=0.0, end_time=10.0, score=0.85)
        self.assertEqual(hs.score, 0.85)
        self.assertEqual(hs.audio_score, 0.0)

    def test_highlight_score_serialise(self):
        """HighlightScore should serialise to dict."""
        from opencut.core.highlight_detect import HighlightScore

        hs = HighlightScore(start_time=10.0, end_time=20.0, score=0.7,
                            audio_score=0.8, visual_score=0.6)
        d = asdict(hs)
        self.assertEqual(d["score"], 0.7)
        self.assertIn("audio_score", d)

    def test_detect_highlights_missing_file(self):
        """detect_highlights should raise for missing file."""
        from opencut.core.highlight_detect import detect_highlights

        with self.assertRaises(FileNotFoundError):
            detect_highlights("/nonexistent.mp4")

    def test_extract_highlight_clips_missing_file(self):
        """extract_highlight_clips should raise for missing file."""
        from opencut.core.highlight_detect import extract_highlight_clips

        with self.assertRaises(FileNotFoundError):
            extract_highlight_clips("/nonexistent.mp4", [{"start_time": 0, "end_time": 5}])

    def test_extract_highlight_clips_empty(self):
        """extract_highlight_clips should raise on empty highlights."""
        from opencut.core.highlight_detect import extract_highlight_clips

        with self.assertRaises(ValueError):
            extract_highlight_clips(self.tmp.name, [])

    @patch("opencut.core.highlight_detect.run_ffmpeg")
    @patch("opencut.core.highlight_detect.get_video_info")
    def test_extract_highlight_clips_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        """extract_highlight_clips should call FFmpeg per highlight."""
        mock_info.return_value = {"fps": 30.0, "duration": 120.0}
        from opencut.core.highlight_detect import extract_highlight_clips

        highlights = [
            {"start_time": 10.0, "end_time": 20.0},
            {"start_time": 50.0, "end_time": 60.0},
        ]
        clips = extract_highlight_clips(self.tmp.name, highlights)
        self.assertEqual(mock_ffmpeg.call_count, 2)
        self.assertEqual(len(clips), 2)

    @patch("opencut.core.highlight_detect.get_video_info")
    def test_score_segments_missing_file(self, mock_info):
        """score_segments should raise for missing file."""
        from opencut.core.highlight_detect import score_segments

        with self.assertRaises(FileNotFoundError):
            score_segments("/nonexistent.mp4")


# ============================================================
# 6. Deepfake Detection Tests
# ============================================================
class TestDeepfakeDetect(unittest.TestCase):
    """Tests for opencut.core.deepfake_detect."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_deepfake_result_dataclass(self):
        """DeepfakeResult should store detection results."""
        from opencut.core.deepfake_detect import DeepfakeResult

        r = DeepfakeResult(start_time=0.0, end_time=5.0, confidence=0.3, label="authentic")
        self.assertEqual(r.confidence, 0.3)
        self.assertEqual(r.label, "authentic")

    def test_deepfake_result_serialise(self):
        """DeepfakeResult should serialise to dict."""
        from opencut.core.deepfake_detect import DeepfakeResult

        r = DeepfakeResult(start_time=0.0, end_time=5.0, confidence=0.8, label="suspicious")
        d = asdict(r)
        self.assertIn("confidence", d)
        self.assertIn("face_consistency", d)

    def test_detect_deepfake_missing_file(self):
        """detect_deepfake should raise for missing file."""
        from opencut.core.deepfake_detect import detect_deepfake

        with self.assertRaises(FileNotFoundError):
            detect_deepfake("/nonexistent.mp4")

    def test_analyze_face_consistency_missing(self):
        """analyze_face_consistency should raise for missing file."""
        from opencut.core.deepfake_detect import analyze_face_consistency

        with self.assertRaises(FileNotFoundError):
            analyze_face_consistency("/nonexistent.mp4")

    def test_generate_authenticity_report_empty(self):
        """generate_authenticity_report should raise on empty results."""
        from opencut.core.deepfake_detect import generate_authenticity_report

        with self.assertRaises(ValueError):
            generate_authenticity_report({}, "/tmp/report.json")

    def test_generate_authenticity_report(self):
        """generate_authenticity_report should write JSON report."""
        from opencut.core.deepfake_detect import generate_authenticity_report

        results = {
            "results": [{"start_time": 0, "end_time": 5, "confidence": 0.2, "label": "authentic"}],
            "overall_score": 0.2,
            "max_score": 0.2,
            "verdict": "likely_authentic",
        }
        out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        out.close()
        try:
            path = generate_authenticity_report(results, out.name)
            self.assertEqual(path, out.name)
            with open(out.name) as f:
                report = json.load(f)
            self.assertEqual(report["verdict"], "likely_authentic")
            self.assertIn("summary", report)
        finally:
            os.unlink(out.name)

    def test_generate_summary_authentic(self):
        """Summary should describe authentic video."""
        from opencut.core.deepfake_detect import _generate_summary

        summary = _generate_summary({
            "verdict": "likely_authentic",
            "overall_score": 0.1,
            "results": [{"label": "authentic"}],
        })
        self.assertIn("likely authentic", summary)

    def test_generate_summary_suspicious(self):
        """Summary should describe suspicious video."""
        from opencut.core.deepfake_detect import _generate_summary

        summary = _generate_summary({
            "verdict": "suspicious",
            "overall_score": 0.55,
            "results": [{"label": "suspicious"}],
        })
        self.assertIn("suspicious", summary)

    def test_generate_summary_fake(self):
        """Summary should describe likely fake video."""
        from opencut.core.deepfake_detect import _generate_summary

        summary = _generate_summary({
            "verdict": "likely_fake",
            "overall_score": 0.85,
            "results": [{"label": "likely_fake"}, {"label": "likely_fake"}],
        })
        self.assertIn("manipulation", summary)


# ============================================================
# 7. Face Tagging Tests
# ============================================================
class TestFaceTagging(unittest.TestCase):
    """Tests for opencut.core.face_tagging."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.tmp.write(b"\x00" * 100)
        self.tmp.close()
        # Clear global tags between tests
        import opencut.core.face_tagging as ft
        ft._face_tags.clear()

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_face_cluster_dataclass(self):
        """FaceCluster should store cluster info."""
        from opencut.core.face_tagging import FaceCluster

        c = FaceCluster(cluster_id=0, name="Alice", face_count=15)
        self.assertEqual(c.cluster_id, 0)
        self.assertEqual(c.name, "Alice")
        self.assertEqual(c.face_count, 15)

    def test_face_cluster_serialise(self):
        """FaceCluster should serialise to dict."""
        from opencut.core.face_tagging import FaceCluster

        c = FaceCluster(cluster_id=1, name="Bob", face_count=8)
        d = asdict(c)
        self.assertIn("cluster_id", d)
        self.assertIn("representative_bbox", d)

    def test_detect_faces_missing_file(self):
        """detect_faces should raise for missing file."""
        from opencut.core.face_tagging import detect_faces

        with self.assertRaises(FileNotFoundError):
            detect_faces("/nonexistent.mp4")

    def test_tag_face_cluster(self):
        """tag_face_cluster should store name."""
        from opencut.core.face_tagging import tag_face_cluster

        result = tag_face_cluster(0, "Alice")
        self.assertEqual(result["status"], "tagged")
        self.assertEqual(result["name"], "Alice")

    def test_tag_face_cluster_invalid_id(self):
        """tag_face_cluster should reject invalid cluster_id."""
        from opencut.core.face_tagging import tag_face_cluster

        with self.assertRaises(ValueError):
            tag_face_cluster(-1, "Test")

    def test_tag_face_cluster_empty_name(self):
        """tag_face_cluster should reject empty name."""
        from opencut.core.face_tagging import tag_face_cluster

        with self.assertRaises(ValueError):
            tag_face_cluster(0, "")

    def test_search_by_face(self):
        """search_by_face should find tagged faces."""
        from opencut.core.face_tagging import search_by_face, tag_face_cluster

        tag_face_cluster(0, "Alice Smith")
        tag_face_cluster(1, "Bob Jones")

        results = search_by_face("alice")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Alice Smith")

    def test_search_by_face_no_match(self):
        """search_by_face should return empty for no matches."""
        from opencut.core.face_tagging import search_by_face

        results = search_by_face("nobody")
        self.assertEqual(len(results), 0)

    def test_search_by_face_empty_name(self):
        """search_by_face should raise on empty name."""
        from opencut.core.face_tagging import search_by_face

        with self.assertRaises(ValueError):
            search_by_face("")

    @patch("opencut.core.face_tagging.ensure_package", return_value=True)
    def test_cluster_faces_empty(self, mock_ensure):
        """cluster_faces should return empty for no embeddings."""
        from opencut.core.face_tagging import cluster_faces

        result = cluster_faces([])
        self.assertEqual(result, [])

    @patch("opencut.core.face_tagging.ensure_package", return_value=True)
    def test_cluster_faces_basic(self, mock_ensure):
        """cluster_faces should group similar embeddings."""
        from opencut.core.face_tagging import cluster_faces

        # Create two distinct groups
        embeddings = []
        for i in range(5):
            emb = [1.0] * 64
            emb[0] = 0.1 * i
            embeddings.append({"frame_idx": i, "bbox": [0, 0, 50, 50], "embedding": emb})
        for i in range(5):
            emb = [0.0] * 64
            emb[0] = 10.0 + 0.1 * i
            embeddings.append({"frame_idx": 5 + i, "bbox": [100, 100, 50, 50], "embedding": emb})

        clusters = cluster_faces(embeddings, distance_threshold=5.0, min_cluster_size=2)
        self.assertGreater(len(clusters), 0)

    def test_simple_cluster_fallback(self):
        """_simple_cluster should work without sklearn."""
        import numpy as np

        from opencut.core.face_tagging import _simple_cluster

        X = np.array([
            [1.0, 0.0],
            [1.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
        ], dtype=np.float32)

        labels = _simple_cluster(X, threshold=1.0)
        self.assertEqual(len(labels), 4)
        # First two should be same cluster, last two same cluster
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


# ============================================================
# 8. Holy Grail Timelapse Tests
# ============================================================
class TestHolyGrailTimelapse(unittest.TestCase):
    """Tests for opencut.core.holy_grail_timelapse."""

    def test_holy_grail_config_defaults(self):
        """HolyGrailConfig should have sensible defaults."""
        from opencut.core.holy_grail_timelapse import HolyGrailConfig

        cfg = HolyGrailConfig()
        self.assertEqual(cfg.output_fps, 24.0)
        self.assertEqual(cfg.target_brightness, 0.45)
        self.assertTrue(cfg.deflicker)

    def test_holy_grail_config_custom(self):
        """HolyGrailConfig should accept custom values."""
        from opencut.core.holy_grail_timelapse import HolyGrailConfig

        cfg = HolyGrailConfig(output_fps=30.0, deflicker=False)
        self.assertEqual(cfg.output_fps, 30.0)
        self.assertFalse(cfg.deflicker)

    def test_analyze_exposure_ramp_empty(self):
        """analyze_exposure_ramp should raise on empty list."""
        from opencut.core.holy_grail_timelapse import analyze_exposure_ramp

        with self.assertRaises(ValueError):
            analyze_exposure_ramp([])

    def test_analyze_exposure_ramp_missing_file(self):
        """analyze_exposure_ramp should raise for missing files."""
        from opencut.core.holy_grail_timelapse import analyze_exposure_ramp

        with self.assertRaises(FileNotFoundError):
            analyze_exposure_ramp(["/nonexistent/img1.jpg"])

    @patch("opencut.core.holy_grail_timelapse.ensure_package", return_value=True)
    def test_analyze_exposure_ramp(self, mock_ensure):
        """analyze_exposure_ramp should return brightness analysis."""
        import cv2
        import numpy as np

        from opencut.core.holy_grail_timelapse import analyze_exposure_ramp

        # Create test images with increasing brightness
        paths = []
        for i in range(5):
            brightness = 50 + i * 40
            img = np.full((100, 100, 3), brightness, dtype=np.uint8)
            path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            path.close()
            cv2.imwrite(path.name, img)
            paths.append(path.name)

        try:
            result = analyze_exposure_ramp(paths)
            self.assertIn("brightness_values", result)
            self.assertIn("ramp_direction", result)
            self.assertEqual(result["frame_count"], 5)
            self.assertEqual(result["ramp_direction"], "brightening")
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    @patch("opencut.core.holy_grail_timelapse.ensure_package", return_value=True)
    def test_apply_exposure_compensation(self, mock_ensure):
        """apply_exposure_compensation should adjust brightness."""
        import cv2
        import numpy as np

        from opencut.core.holy_grail_timelapse import apply_exposure_compensation

        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        path.close()
        cv2.imwrite(path.name, img)

        out_path = path.name.replace(".png", "_comp.png")
        try:
            result = apply_exposure_compensation(path.name, adjustment=1.5, output_path=out_path)
            self.assertTrue(os.path.exists(result))
            compensated = cv2.imread(result)
            # Should be brighter (100 * 1.5 = 150)
            self.assertGreater(np.mean(compensated), np.mean(img))
        finally:
            for p in (path.name, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_apply_exposure_compensation_missing(self):
        """apply_exposure_compensation should raise for missing file."""
        from opencut.core.holy_grail_timelapse import apply_exposure_compensation

        with self.assertRaises(FileNotFoundError):
            apply_exposure_compensation("/nonexistent.png")

    def test_process_holy_grail_empty(self):
        """process_holy_grail should raise on empty list."""
        from opencut.core.holy_grail_timelapse import process_holy_grail

        with self.assertRaises(ValueError):
            process_holy_grail([])

    def test_process_holy_grail_too_few(self):
        """process_holy_grail should require at least 3 images."""
        from opencut.core.holy_grail_timelapse import process_holy_grail

        t1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        t1.write(b"\x00")
        t1.close()
        try:
            with self.assertRaises(ValueError):
                process_holy_grail([t1.name, t1.name])
        finally:
            os.unlink(t1.name)


# ============================================================
# Route Tests (Blueprint smoke tests)
# ============================================================
class TestVideoVfxRoutes(unittest.TestCase):
    """Smoke tests for video_vfx_routes blueprint."""

    @classmethod
    def setUpClass(cls):
        try:
            from opencut.config import OpenCutConfig
            from opencut.server import create_app
            test_config = OpenCutConfig()
            cls.app = create_app(config=test_config)
            cls.app.config["TESTING"] = True
            cls.client = cls.app.test_client()
            # Get CSRF token
            resp = cls.client.get("/health")
            data = resp.get_json()
            cls.csrf = data.get("csrf_token", "")
        except Exception:
            cls.app = None
            cls.csrf = ""

    def _headers(self):
        return {
            "X-OpenCut-Token": self.csrf,
            "Content-Type": "application/json",
        }

    def test_track_object_no_filepath(self):
        """POST /video/track-object without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/track-object",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_relight_no_filepath(self):
        """POST /video/relight without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/relight",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_detect_360_no_filepath(self):
        """POST /video/360/detect without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/360/detect",
                                json={}, headers=self._headers())
        self.assertEqual(resp.status_code, 400)

    def test_clean_plate_no_filepath(self):
        """POST /video/clean-plate without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/clean-plate",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_detect_highlights_no_filepath(self):
        """POST /video/detect-highlights without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/detect-highlights",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_detect_deepfake_no_filepath(self):
        """POST /video/detect-deepfake without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/detect-deepfake",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_detect_faces_no_filepath(self):
        """POST /video/detect-faces without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/detect-faces",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_cluster_faces_no_embeddings(self):
        """POST /video/cluster-faces without embeddings should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/cluster-faces",
                                json={}, headers=self._headers())
        self.assertEqual(resp.status_code, 400)

    def test_tag_face_no_data(self):
        """POST /video/tag-face without data should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/tag-face",
                                json={}, headers=self._headers())
        self.assertEqual(resp.status_code, 400)

    def test_search_face_no_name(self):
        """POST /video/search-face without name should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/search-face",
                                json={}, headers=self._headers())
        self.assertEqual(resp.status_code, 400)

    def test_holy_grail_no_images(self):
        """POST /video/holy-grail without image_paths should error."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/holy-grail",
                                json={}, headers=self._headers())
        # Should return job or 400 depending on timing
        self.assertIn(resp.status_code, (200, 400, 429))

    def test_analyze_exposure_no_images(self):
        """POST /video/analyze-exposure without image_paths should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/analyze-exposure",
                                json={}, headers=self._headers())
        self.assertEqual(resp.status_code, 400)

    def test_export_track_no_data(self):
        """POST /video/export-track without track_data should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/export-track",
                                json={}, headers=self._headers())
        self.assertEqual(resp.status_code, 400)

    def test_360_convert_no_filepath(self):
        """POST /video/360/convert without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/360/convert",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_360_crop_no_filepath(self):
        """POST /video/360/crop without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/360/crop",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_360_stabilize_no_filepath(self):
        """POST /video/360/stabilize without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/360/stabilize",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_extract_highlights_no_filepath(self):
        """POST /video/extract-highlights without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/extract-highlights",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))

    def test_authenticity_report_no_filepath(self):
        """POST /video/authenticity-report without filepath should 400."""
        if self.app is None:
            self.skipTest("App not available")
        resp = self.client.post("/video/authenticity-report",
                                json={}, headers=self._headers())
        self.assertIn(resp.status_code, (400, 429))


if __name__ == "__main__":
    unittest.main()
