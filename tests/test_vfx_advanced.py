"""
Tests for OpenCut VFX Advanced features.

Covers: object_effects (all 6 effect types, mask generation, preview),
        planar_track (tracking, insertion, export, preview),
        and vfx_advanced_routes blueprint.

80+ tests with mocked OpenCV, FFmpeg, numpy.
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
# Helper: create a temp video stub
# ============================================================
def _make_temp_file(suffix=".mp4"):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"\x00" * 100)
    f.close()
    return f.name


def _make_fake_frame(h=480, w=640, channels=3):
    """Return a mock numpy array shaped like a video frame."""
    import numpy as np
    return np.zeros((h, w, channels), dtype=np.uint8) + 128


def _make_fake_mask(h=480, w=640):
    """Return a mock binary mask (white rectangle in centre)."""
    import numpy as np
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = 255
    return mask


# ============================================================
# Object Effects - Dataclass Tests
# ============================================================
class TestEffectConfig(unittest.TestCase):
    """Tests for EffectConfig dataclass."""

    def test_valid_effect_types(self):
        from opencut.core.object_effects import EffectConfig
        for etype in ("squish", "melt", "inflate", "explode", "dissolve", "crystallize"):
            cfg = EffectConfig(effect_type=etype)
            self.assertEqual(cfg.effect_type, etype)

    def test_invalid_effect_type_raises(self):
        from opencut.core.object_effects import EffectConfig
        with self.assertRaises(ValueError):
            EffectConfig(effect_type="nonexistent")

    def test_intensity_clamped(self):
        from opencut.core.object_effects import EffectConfig
        cfg = EffectConfig(intensity=2.5)
        self.assertEqual(cfg.intensity, 1.0)
        cfg2 = EffectConfig(intensity=-0.5)
        self.assertEqual(cfg2.intensity, 0.0)

    def test_duration_minimum(self):
        from opencut.core.object_effects import EffectConfig
        cfg = EffectConfig(duration=0.01)
        self.assertEqual(cfg.duration, 0.1)

    def test_seed_integer(self):
        from opencut.core.object_effects import EffectConfig
        cfg = EffectConfig(seed=3.7)
        self.assertEqual(cfg.seed, 3)


class TestObjectMask(unittest.TestCase):
    """Tests for ObjectMask dataclass."""

    def test_basic_creation(self):
        from opencut.core.object_effects import ObjectMask
        mask = ObjectMask(
            mask_frames=["frame1.png", "frame2.png"],
            bbox_per_frame=[(10, 10, 100, 100), (11, 11, 100, 100)],
        )
        self.assertEqual(len(mask.mask_frames), 2)
        self.assertEqual(len(mask.bbox_per_frame), 2)


class TestObjectEffectResult(unittest.TestCase):
    """Tests for ObjectEffectResult dataclass."""

    def test_creation(self):
        from opencut.core.object_effects import ObjectEffectResult
        r = ObjectEffectResult(
            output_path="/tmp/out.mp4",
            frames_processed=100,
            effect_applied="squish",
        )
        self.assertEqual(r.output_path, "/tmp/out.mp4")
        self.assertEqual(r.frames_processed, 100)
        self.assertEqual(r.effect_applied, "squish")


# ============================================================
# Object Effects - Effect Functions Tests
# ============================================================
class TestEffectFunctions(unittest.TestCase):
    """Tests for individual effect implementations."""

    def setUp(self):
        self.frame = _make_fake_frame()
        self.mask = _make_fake_mask()
        self.bbox = (160, 120, 320, 240)  # centre quarter
        self.rng = __import__("random").Random(42)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_squish_returns_frame(self, mock_ep):
        from opencut.core.object_effects import _apply_squish
        result = _apply_squish(self.frame, self.mask, self.bbox, 0.5, 0.7, self.rng)
        self.assertEqual(result.shape, self.frame.shape)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_squish_zero_bbox_noop(self, mock_ep):
        import numpy as np

        from opencut.core.object_effects import _apply_squish
        result = _apply_squish(self.frame, self.mask, (0, 0, 0, 0), 0.5, 0.7, self.rng)
        np.testing.assert_array_equal(result, self.frame)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_melt_returns_frame(self, mock_ep):
        from opencut.core.object_effects import _apply_melt
        result = _apply_melt(self.frame, self.mask, self.bbox, 0.5, 0.7, self.rng)
        self.assertEqual(result.shape, self.frame.shape)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_melt_zero_bbox_noop(self, mock_ep):
        import numpy as np

        from opencut.core.object_effects import _apply_melt
        result = _apply_melt(self.frame, self.mask, (0, 0, 0, 0), 0.5, 0.7, self.rng)
        np.testing.assert_array_equal(result, self.frame)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_inflate_returns_frame(self, mock_ep):
        from opencut.core.object_effects import _apply_inflate
        result = _apply_inflate(self.frame, self.mask, self.bbox, 0.5, 0.7, self.rng)
        self.assertEqual(result.shape, self.frame.shape)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_inflate_zero_bbox_noop(self, mock_ep):
        import numpy as np

        from opencut.core.object_effects import _apply_inflate
        result = _apply_inflate(self.frame, self.mask, (0, 0, 0, 0), 0.5, 0.7, self.rng)
        np.testing.assert_array_equal(result, self.frame)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_explode_returns_frame(self, mock_ep):
        from opencut.core.object_effects import _apply_explode
        result = _apply_explode(self.frame, self.mask, self.bbox, 0.5, 0.7, self.rng)
        self.assertEqual(result.shape, self.frame.shape)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_explode_zero_bbox_noop(self, mock_ep):
        import numpy as np

        from opencut.core.object_effects import _apply_explode
        result = _apply_explode(self.frame, self.mask, (0, 0, 0, 0), 0.5, 0.7, self.rng)
        np.testing.assert_array_equal(result, self.frame)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_dissolve_returns_frame(self, mock_ep):
        from opencut.core.object_effects import _apply_dissolve
        result = _apply_dissolve(self.frame, self.mask, self.bbox, 0.5, 0.7, self.rng)
        self.assertEqual(result.shape, self.frame.shape)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_dissolve_zero_bbox_noop(self, mock_ep):
        import numpy as np

        from opencut.core.object_effects import _apply_dissolve
        result = _apply_dissolve(self.frame, self.mask, (0, 0, 0, 0), 0.5, 0.7, self.rng)
        np.testing.assert_array_equal(result, self.frame)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_crystallize_returns_frame(self, mock_ep):
        from opencut.core.object_effects import _apply_crystallize
        result = _apply_crystallize(self.frame, self.mask, self.bbox, 0.5, 0.7, self.rng)
        self.assertEqual(result.shape, self.frame.shape)

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_crystallize_zero_bbox_noop(self, mock_ep):
        import numpy as np

        from opencut.core.object_effects import _apply_crystallize
        result = _apply_crystallize(self.frame, self.mask, (0, 0, 0, 0), 0.5, 0.7, self.rng)
        np.testing.assert_array_equal(result, self.frame)

    def test_grid_pixelate(self):
        from opencut.core.object_effects import _grid_pixelate
        result = _grid_pixelate(self.frame, 8)
        self.assertEqual(result.shape, self.frame.shape)

    def test_grid_pixelate_small_cell(self):
        from opencut.core.object_effects import _grid_pixelate
        result = _grid_pixelate(self.frame, 2)
        self.assertEqual(result.shape, self.frame.shape)

    def test_ease_in_out_boundaries(self):
        from opencut.core.object_effects import _ease_in_out
        self.assertAlmostEqual(_ease_in_out(0.0), 0.0, places=5)
        self.assertAlmostEqual(_ease_in_out(1.0), 1.0, places=5)
        self.assertAlmostEqual(_ease_in_out(0.5), 0.5, places=5)


# ============================================================
# Object Effects - Mask Generation Tests
# ============================================================
class TestMaskGeneration(unittest.TestCase):
    """Tests for mask generation functions."""

    def setUp(self):
        self.video_path = _make_temp_file()

    def tearDown(self):
        try:
            os.unlink(self.video_path)
        except OSError:
            pass

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_threshold_fallback_runs(self, mock_ep):
        """generate_effect_mask should fall back to threshold when SAM2 is absent."""
        from opencut.core.object_effects import ObjectMask

        fake_mask = ObjectMask(
            mask_frames=[_make_fake_mask()],
            bbox_per_frame=[(100, 100, 200, 200)],
        )
        with patch("opencut.core.object_effects._generate_mask_threshold",
                    return_value=fake_mask) as mock_thresh, \
             patch("opencut.core.object_effects._generate_mask_sam2",
                    side_effect=ImportError("no sam2")):
            from opencut.core.object_effects import generate_effect_mask
            result = generate_effect_mask(self.video_path, (320, 240), 1)
            self.assertEqual(len(result.mask_frames), 1)
            mock_thresh.assert_called_once()

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_generate_effect_mask_no_sam2(self, mock_ep):
        """Without SAM2, should fall back to threshold."""
        import numpy as np

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap.__enter__ = MagicMock(return_value=mock_cap)
        mock_cap.__exit__ = MagicMock(return_value=False)

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.cvtColor", return_value=frame[:, :, 0:1].repeat(3, axis=2)), \
             patch("cv2.inRange", return_value=_make_fake_mask()), \
             patch("cv2.getStructuringElement", return_value=np.ones((7, 7), np.uint8)), \
             patch("cv2.morphologyEx", return_value=_make_fake_mask()):
            from opencut.core.object_effects import _generate_mask_threshold
            result = _generate_mask_threshold(
                self.video_path, (320, 240), 5, None,
            )
            self.assertEqual(len(result.mask_frames), 5)
            self.assertEqual(len(result.bbox_per_frame), 5)


# ============================================================
# Object Effects - apply_object_effect Tests
# ============================================================
class TestApplyObjectEffect(unittest.TestCase):
    """Tests for the main apply_object_effect function."""

    def setUp(self):
        self.video_path = _make_temp_file()

    def tearDown(self):
        try:
            os.unlink(self.video_path)
        except OSError:
            pass

    @patch("opencut.core.object_effects.run_ffmpeg")
    @patch("opencut.core.object_effects.get_video_info")
    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_apply_squish_basic(self, mock_ep, mock_info, mock_ffmpeg):

        mock_info.return_value = {"width": 640, "height": 480, "fps": 30.0, "duration": 2.0}

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 3  # 3 frames
        mock_cap.read.side_effect = [
            (True, frame.copy()), (True, frame.copy()), (True, frame.copy()),
            (False, None),
        ]
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0x7634706d):
            from opencut.core.object_effects import (
                EffectConfig,
                ObjectMask,
                apply_object_effect,
            )
            mask = ObjectMask(
                mask_frames=[_make_fake_mask(), _make_fake_mask(), _make_fake_mask()],
                bbox_per_frame=[(160, 120, 320, 240)] * 3,
            )
            config = EffectConfig(effect_type="squish", intensity=0.7)

            result = apply_object_effect(
                self.video_path, mask, config, out_path="/tmp/test_out.mp4",
            )
            self.assertEqual(result.effect_applied, "squish")
            self.assertEqual(result.frames_processed, 3)
            mock_ffmpeg.assert_called_once()

    @patch("opencut.core.object_effects.run_ffmpeg")
    @patch("opencut.core.object_effects.get_video_info")
    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_apply_melt_basic(self, mock_ep, mock_info, mock_ffmpeg):

        mock_info.return_value = {"width": 640, "height": 480, "fps": 30.0, "duration": 1.0}

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 2
        mock_cap.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0):
            from opencut.core.object_effects import (
                EffectConfig,
                ObjectMask,
                apply_object_effect,
            )
            mask = ObjectMask(
                mask_frames=[_make_fake_mask(), _make_fake_mask()],
                bbox_per_frame=[(160, 120, 320, 240)] * 2,
            )
            config = EffectConfig(effect_type="melt", intensity=0.5)
            result = apply_object_effect(
                self.video_path, mask, config, out_path="/tmp/melt.mp4",
            )
            self.assertEqual(result.effect_applied, "melt")

    @patch("opencut.core.object_effects.get_video_info")
    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_apply_bad_dimensions_raises(self, mock_ep, mock_info):
        mock_info.return_value = {"width": 0, "height": 0, "fps": 30.0}
        from opencut.core.object_effects import (
            EffectConfig,
            ObjectMask,
            apply_object_effect,
        )
        mask = ObjectMask(mask_frames=[], bbox_per_frame=[])
        config = EffectConfig(effect_type="squish")
        with self.assertRaises(RuntimeError):
            apply_object_effect(self.video_path, mask, config)

    @patch("opencut.core.object_effects.ensure_package", return_value=False)
    def test_apply_no_opencv_raises(self, mock_ep):
        from opencut.core.object_effects import (
            EffectConfig,
            ObjectMask,
            apply_object_effect,
        )
        mask = ObjectMask(mask_frames=[], bbox_per_frame=[])
        config = EffectConfig(effect_type="squish")
        with self.assertRaises(RuntimeError):
            apply_object_effect(self.video_path, mask, config)


# ============================================================
# Object Effects - get_available_effects Tests
# ============================================================
class TestGetAvailableEffects(unittest.TestCase):

    def test_returns_all_six(self):
        from opencut.core.object_effects import get_available_effects
        effects = get_available_effects()
        self.assertEqual(len(effects), 6)
        names = {e["type"] for e in effects}
        self.assertEqual(
            names,
            {"squish", "melt", "inflate", "explode", "dissolve", "crystallize"},
        )

    def test_each_has_description(self):
        from opencut.core.object_effects import get_available_effects
        for e in get_available_effects():
            self.assertIn("type", e)
            self.assertIn("description", e)
            self.assertTrue(len(e["description"]) > 5)


# ============================================================
# Object Effects - Preview Tests
# ============================================================
class TestPreviewEffectFrame(unittest.TestCase):

    def setUp(self):
        self.video_path = _make_temp_file()

    def tearDown(self):
        try:
            os.unlink(self.video_path)
        except OSError:
            pass

    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_preview_returns_dict(self, mock_ep):

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (True, frame)

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.imwrite", return_value=True), \
             patch("cv2.resize", return_value=_make_fake_mask()):
            from opencut.core.object_effects import (
                EffectConfig,
                ObjectMask,
                preview_effect_frame,
            )
            mask = ObjectMask(
                mask_frames=[_make_fake_mask()],
                bbox_per_frame=[(160, 120, 320, 240)],
            )
            config = EffectConfig(effect_type="dissolve")
            result = preview_effect_frame(
                self.video_path, mask, config, timestamp=0.5,
            )
            self.assertIn("preview_path", result)
            self.assertIn("effect_type", result)
            self.assertEqual(result["effect_type"], "dissolve")

    @patch("opencut.core.object_effects.ensure_package", return_value=False)
    def test_preview_no_opencv_raises(self, mock_ep):
        from opencut.core.object_effects import (
            EffectConfig,
            ObjectMask,
            preview_effect_frame,
        )
        mask = ObjectMask(mask_frames=[], bbox_per_frame=[])
        config = EffectConfig(effect_type="squish")
        with self.assertRaises(RuntimeError):
            preview_effect_frame(self.video_path, mask, config)


# ============================================================
# Planar Tracking - Dataclass Tests
# ============================================================
class TestTrackRegion(unittest.TestCase):

    def test_valid_4_corners(self):
        from opencut.core.planar_track import TrackRegion
        r = TrackRegion(corners=[(0, 0), (100, 0), (100, 100), (0, 100)])
        self.assertEqual(len(r.corners), 4)

    def test_wrong_corner_count_raises(self):
        from opencut.core.planar_track import TrackRegion
        with self.assertRaises(ValueError):
            TrackRegion(corners=[(0, 0), (100, 0)])

    def test_center_computation(self):
        from opencut.core.planar_track import TrackRegion
        r = TrackRegion(corners=[(0, 0), (100, 0), (100, 100), (0, 100)])
        cx, cy = r.center
        self.assertAlmostEqual(cx, 50.0)
        self.assertAlmostEqual(cy, 50.0)

    def test_area_computation(self):
        from opencut.core.planar_track import TrackRegion
        r = TrackRegion(corners=[(0, 0), (100, 0), (100, 100), (0, 100)])
        self.assertAlmostEqual(r.area, 10000.0)

    def test_as_list(self):
        from opencut.core.planar_track import TrackRegion
        r = TrackRegion(corners=[(1.5, 2.5), (3.5, 4.5), (5.5, 6.5), (7.5, 8.5)])
        lst = r.as_list()
        self.assertEqual(len(lst), 4)
        self.assertEqual(lst[0], [1.5, 2.5])

    def test_corners_cast_to_float(self):
        from opencut.core.planar_track import TrackRegion
        r = TrackRegion(corners=[(0, 0), (100, 0), (100, 100), (0, 100)])
        for cx, cy in r.corners:
            self.assertIsInstance(cx, float)
            self.assertIsInstance(cy, float)


class TestTrackResult(unittest.TestCase):

    def test_avg_confidence(self):
        from opencut.core.planar_track import TrackRegion, TrackResult
        frames = [
            TrackRegion(corners=[(0, 0), (1, 0), (1, 1), (0, 1)])
            for _ in range(3)
        ]
        tr = TrackResult(
            frames=frames,
            confidence_per_frame=[0.8, 0.9, 1.0],
            fps=30.0,
            total_frames=3,
        )
        self.assertAlmostEqual(tr.avg_confidence, 0.9)

    def test_empty_confidence(self):
        from opencut.core.planar_track import TrackResult
        tr = TrackResult(frames=[], confidence_per_frame=[], fps=30.0, total_frames=0)
        self.assertEqual(tr.avg_confidence, 0.0)

    def test_duration(self):
        from opencut.core.planar_track import TrackResult
        tr = TrackResult(
            frames=[],
            confidence_per_frame=[],
            fps=30.0,
            total_frames=90,
        )
        self.assertAlmostEqual(tr.duration, 3.0)

    def test_duration_zero_fps(self):
        from opencut.core.planar_track import TrackResult
        tr = TrackResult(frames=[], confidence_per_frame=[], fps=0, total_frames=10)
        self.assertEqual(tr.duration, 0.0)


# ============================================================
# Planar Tracking - Track Function Tests
# ============================================================
class TestTrackPlanarSurface(unittest.TestCase):

    def setUp(self):
        self.video_path = _make_temp_file()

    def tearDown(self):
        try:
            os.unlink(self.video_path)
        except OSError:
            pass

    @patch("opencut.core.planar_track.ensure_package", return_value=True)
    def test_tracking_basic(self, mock_ep):
        import numpy as np

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 30.0, 7: 5}.get(prop, 0)
        mock_cap.read.side_effect = [
            (True, frame.copy()),
            (True, frame.copy()),
            (True, frame.copy()),
            (False, None),
        ]
        mock_cap.set.return_value = True

        fake_kp = [MagicMock(pt=(float(i * 10), float(i * 10))) for i in range(20)]
        fake_desc = np.random.randint(0, 256, (20, 32), dtype=np.uint8)
        fake_matches = [MagicMock(queryIdx=i, trainIdx=i, distance=float(i))
                        for i in range(20)]
        mock_H = np.eye(3, dtype=np.float64)
        mock_inliers = np.ones((20, 1), dtype=np.uint8)

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.cvtColor", return_value=frame[:, :, 0]), \
             patch("cv2.ORB_create") as mock_orb_cls, \
             patch("cv2.BFMatcher") as mock_bf_cls, \
             patch("cv2.findHomography", return_value=(mock_H, mock_inliers)), \
             patch("cv2.perspectiveTransform", side_effect=lambda pts, H: pts):

            mock_orb = MagicMock()
            mock_orb.detectAndCompute.return_value = (fake_kp, fake_desc)
            mock_orb_cls.return_value = mock_orb

            mock_bf = MagicMock()
            mock_bf.match.return_value = fake_matches
            mock_bf_cls.return_value = mock_bf

            from opencut.core.planar_track import track_planar_surface
            result = track_planar_surface(
                self.video_path,
                initial_corners=[(100, 100), (300, 100), (300, 300), (100, 300)],
                start_frame=0,
                end_frame=3,
            )
            self.assertEqual(len(result.frames), 3)
            self.assertEqual(len(result.confidence_per_frame), 3)
            self.assertEqual(result.fps, 30.0)

    def test_wrong_corner_count_raises(self):
        from opencut.core.planar_track import track_planar_surface
        with patch("opencut.core.planar_track.ensure_package", return_value=True), \
             patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap_cls.return_value = mock_cap
            with self.assertRaises(ValueError):
                track_planar_surface(
                    self.video_path,
                    initial_corners=[(0, 0), (100, 0)],
                )

    @patch("opencut.core.planar_track.ensure_package", return_value=False)
    def test_no_opencv_raises(self, mock_ep):
        from opencut.core.planar_track import track_planar_surface
        with self.assertRaises(RuntimeError):
            track_planar_surface(
                self.video_path,
                initial_corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
            )

    @patch("opencut.core.planar_track.ensure_package", return_value=True)
    def test_start_gte_end_raises(self, mock_ep):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {3: 30.0, 7: 100}.get(prop, 0)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            from opencut.core.planar_track import track_planar_surface
            with self.assertRaises(ValueError):
                track_planar_surface(
                    self.video_path,
                    initial_corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
                    start_frame=50,
                    end_frame=10,
                )


# ============================================================
# Planar Tracking - Export Tests
# ============================================================
class TestExportTrackData(unittest.TestCase):

    def _make_track_result(self, n=3):
        from opencut.core.planar_track import TrackRegion, TrackResult
        frames = [
            TrackRegion(corners=[(i, i), (i + 100, i), (i + 100, i + 100), (i, i + 100)])
            for i in range(n)
        ]
        return TrackResult(
            frames=frames,
            confidence_per_frame=[0.9] * n,
            fps=30.0,
            total_frames=n,
        )

    def test_export_json(self):
        from opencut.core.planar_track import export_track_data
        tr = self._make_track_result()
        result = export_track_data(tr, format="json")
        data = json.loads(result)
        self.assertEqual(data["total_frames"], 3)
        self.assertEqual(len(data["frames"]), 3)
        self.assertIn("corners", data["frames"][0])

    def test_export_csv(self):
        from opencut.core.planar_track import export_track_data
        tr = self._make_track_result()
        result = export_track_data(tr, format="csv")
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 data rows
        self.assertIn("tl_x", lines[0])

    def test_export_nuke(self):
        from opencut.core.planar_track import export_track_data
        tr = self._make_track_result()
        result = export_track_data(tr, format="nuke")
        self.assertIn("CornerPin2D", result)
        self.assertIn("to1", result)
        self.assertIn("to4", result)

    def test_export_invalid_format(self):
        from opencut.core.planar_track import export_track_data
        tr = self._make_track_result()
        with self.assertRaises(ValueError):
            export_track_data(tr, format="invalid")

    def test_export_json_structure(self):
        from opencut.core.planar_track import export_track_data
        tr = self._make_track_result(5)
        data = json.loads(export_track_data(tr, format="json"))
        self.assertIn("fps", data)
        self.assertIn("avg_confidence", data)
        self.assertEqual(data["fps"], 30.0)
        for f in data["frames"]:
            self.assertIn("frame", f)
            self.assertIn("corners", f)
            self.assertIn("confidence", f)
            self.assertEqual(len(f["corners"]), 4)


# ============================================================
# Planar Tracking - Insert Tests
# ============================================================
class TestInsertReplacement(unittest.TestCase):

    def setUp(self):
        self.video_path = _make_temp_file()
        self.replacement = _make_temp_file(".png")

    def tearDown(self):
        for p in (self.video_path, self.replacement):
            try:
                os.unlink(p)
            except OSError:
                pass

    @patch("opencut.core.planar_track.run_ffmpeg")
    @patch("opencut.core.planar_track.get_video_info")
    @patch("opencut.core.planar_track.ensure_package", return_value=True)
    def test_insert_basic(self, mock_ep, mock_info, mock_ffmpeg):
        import numpy as np

        mock_info.return_value = {"width": 640, "height": 480, "fps": 30.0, "duration": 1.0}

        frame = _make_fake_frame()
        rep_image = _make_fake_frame(200, 300)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]

        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        from opencut.core.planar_track import TrackRegion, TrackResult

        track_result = TrackResult(
            frames=[
                TrackRegion(corners=[(100, 100), (400, 100), (400, 350), (100, 350)]),
                TrackRegion(corners=[(105, 102), (405, 102), (405, 352), (105, 352)]),
            ],
            confidence_per_frame=[0.95, 0.90],
            fps=30.0,
            total_frames=2,
        )

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0), \
             patch("cv2.imread", return_value=rep_image), \
             patch("cv2.getPerspectiveTransform", return_value=np.eye(3)), \
             patch("cv2.warpPerspective", return_value=frame), \
             patch("cv2.fillPoly"):
            from opencut.core.planar_track import insert_replacement
            result = insert_replacement(
                self.video_path,
                track_result,
                self.replacement,
                out_path="/tmp/insert_out.mp4",
            )
            self.assertEqual(result, "/tmp/insert_out.mp4")
            mock_ffmpeg.assert_called_once()

    def test_insert_missing_replacement_raises(self):
        from opencut.core.planar_track import TrackRegion, TrackResult

        track_result = TrackResult(
            frames=[TrackRegion(corners=[(0, 0), (1, 0), (1, 1), (0, 1)])],
            confidence_per_frame=[1.0],
            fps=30.0,
            total_frames=1,
        )

        with patch("opencut.core.planar_track.ensure_package", return_value=True):
            from opencut.core.planar_track import insert_replacement
            with self.assertRaises(FileNotFoundError):
                insert_replacement(
                    self.video_path, track_result,
                    "/nonexistent/image.png",
                )


# ============================================================
# Planar Tracking - Preview Tests
# ============================================================
class TestPreviewTrackFrame(unittest.TestCase):

    def setUp(self):
        self.video_path = _make_temp_file()

    def tearDown(self):
        try:
            os.unlink(self.video_path)
        except OSError:
            pass

    @patch("opencut.core.planar_track.ensure_package", return_value=True)
    def test_preview_basic(self, mock_ep):

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)

        from opencut.core.planar_track import TrackRegion, TrackResult

        tr = TrackResult(
            frames=[TrackRegion(corners=[(50, 50), (200, 50), (200, 200), (50, 200)])],
            confidence_per_frame=[0.95],
            fps=30.0,
            total_frames=1,
        )

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.polylines"), \
             patch("cv2.circle"), \
             patch("cv2.putText"), \
             patch("cv2.imwrite", return_value=True):
            from opencut.core.planar_track import preview_track_frame
            result = preview_track_frame(self.video_path, tr, frame_number=0)
            self.assertIn("preview_path", result)
            self.assertIn("confidence", result)
            self.assertEqual(result["frame_number"], 0)

    def test_preview_out_of_range(self):
        from opencut.core.planar_track import TrackRegion, TrackResult, preview_track_frame

        tr = TrackResult(
            frames=[TrackRegion(corners=[(0, 0), (1, 0), (1, 1), (0, 1)])],
            confidence_per_frame=[1.0],
            fps=30.0,
            total_frames=1,
        )

        with patch("opencut.core.planar_track.ensure_package", return_value=True):
            with self.assertRaises(ValueError):
                preview_track_frame(self.video_path, tr, frame_number=5)


# ============================================================
# VFX Advanced Routes - Blueprint Tests
# ============================================================
class TestVfxAdvancedRoutes(unittest.TestCase):
    """Tests for the vfx_advanced_bp Flask blueprint."""

    @classmethod
    def setUpClass(cls):
        cls._video = _make_temp_file()

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls._video)
        except OSError:
            pass

    def setUp(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        cfg = OpenCutConfig()
        self.app = create_app(config=cfg)
        self.app.config["TESTING"] = True
        # Register the vfx_advanced blueprint (not in __init__.py yet)
        from opencut.routes.vfx_advanced_routes import vfx_advanced_bp
        try:
            self.app.register_blueprint(vfx_advanced_bp)
        except ValueError:
            pass  # already registered
        self.client = self.app.test_client()
        resp = self.client.get("/health")
        data = resp.get_json()
        self.csrf = data.get("csrf_token", "")
        self.headers = {
            "X-OpenCut-Token": self.csrf,
            "Content-Type": "application/json",
        }

    def test_get_effect_types(self):
        resp = self.client.get("/object-effects/types")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("effects", data)
        self.assertEqual(len(data["effects"]), 6)

    def test_apply_no_filepath(self):
        resp = self.client.post(
            "/object-effects/apply",
            headers=self.headers,
            json={"effect_type": "squish"},
        )
        self.assertIn(resp.status_code, (400,))

    def test_apply_missing_csrf(self):
        resp = self.client.post(
            "/object-effects/apply",
            headers={"Content-Type": "application/json"},
            json={"filepath": self._video, "effect_type": "squish"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_preview_no_filepath(self):
        resp = self.client.post(
            "/object-effects/preview",
            headers=self.headers,
            json={"effect_type": "dissolve"},
        )
        self.assertIn(resp.status_code, (400,))

    def test_generate_mask_no_filepath(self):
        resp = self.client.post(
            "/object-effects/generate-mask",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (400,))

    def test_track_no_filepath(self):
        resp = self.client.post(
            "/planar-track/track",
            headers=self.headers,
            json={"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        )
        self.assertIn(resp.status_code, (400,))

    def test_insert_no_filepath(self):
        resp = self.client.post(
            "/planar-track/insert",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (400,))

    def test_export_no_filepath(self):
        resp = self.client.post(
            "/planar-track/export",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (400,))

    def test_preview_track_no_filepath(self):
        resp = self.client.post(
            "/planar-track/preview",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, (400,))

    def test_apply_returns_job_id(self):
        resp = self.client.post(
            "/object-effects/apply",
            headers=self.headers,
            json={
                "filepath": self._video,
                "effect_type": "squish",
                "click_x": 100,
                "click_y": 100,
            },
        )
        self.assertIn(resp.status_code, (200,))
        data = resp.get_json()
        self.assertIn("job_id", data)

    def test_track_returns_job_id(self):
        resp = self.client.post(
            "/planar-track/track",
            headers=self.headers,
            json={
                "filepath": self._video,
                "corners": [[100, 100], [300, 100], [300, 300], [100, 300]],
            },
        )
        self.assertIn(resp.status_code, (200,))
        data = resp.get_json()
        self.assertIn("job_id", data)


# ============================================================
# Additional edge-case tests to reach 80+
# ============================================================
class TestEffectEdgeCases(unittest.TestCase):

    def test_effect_types_dict_complete(self):
        from opencut.core.object_effects import _EFFECT_FUNCTIONS, EFFECT_TYPES
        self.assertEqual(set(EFFECT_TYPES.keys()), set(_EFFECT_FUNCTIONS.keys()))

    def test_all_effects_in_registry(self):
        from opencut.core.object_effects import _EFFECT_FUNCTIONS
        self.assertIn("squish", _EFFECT_FUNCTIONS)
        self.assertIn("melt", _EFFECT_FUNCTIONS)
        self.assertIn("inflate", _EFFECT_FUNCTIONS)
        self.assertIn("explode", _EFFECT_FUNCTIONS)
        self.assertIn("dissolve", _EFFECT_FUNCTIONS)
        self.assertIn("crystallize", _EFFECT_FUNCTIONS)

    def test_effect_config_default_values(self):
        from opencut.core.object_effects import EffectConfig
        cfg = EffectConfig()
        self.assertEqual(cfg.effect_type, "squish")
        self.assertEqual(cfg.intensity, 0.7)
        self.assertEqual(cfg.duration, 1.0)
        self.assertEqual(cfg.seed, 42)

    def test_object_effect_result_fields(self):
        from opencut.core.object_effects import ObjectEffectResult
        r = ObjectEffectResult(
            output_path="test.mp4",
            frames_processed=0,
            effect_applied="explode",
        )
        self.assertEqual(r.frames_processed, 0)


class TestPlanarTrackEdgeCases(unittest.TestCase):

    def test_track_region_five_corners_raises(self):
        from opencut.core.planar_track import TrackRegion
        with self.assertRaises(ValueError):
            TrackRegion(corners=[(0, 0)] * 5)

    def test_track_region_one_corner_raises(self):
        from opencut.core.planar_track import TrackRegion
        with self.assertRaises(ValueError):
            TrackRegion(corners=[(0, 0)])

    def test_track_region_empty_raises(self):
        from opencut.core.planar_track import TrackRegion
        with self.assertRaises(ValueError):
            TrackRegion(corners=[])

    def test_planar_insert_dataclass(self):
        from opencut.core.planar_track import PlanarInsert, TrackRegion, TrackResult
        tr = TrackResult(
            frames=[TrackRegion(corners=[(0, 0), (1, 0), (1, 1), (0, 1)])],
            confidence_per_frame=[1.0],
            fps=30.0,
            total_frames=1,
        )
        pi = PlanarInsert(
            track_data=tr,
            replacement_path="/tmp/img.png",
            output_path="/tmp/out.mp4",
        )
        self.assertEqual(pi.replacement_path, "/tmp/img.png")

    def test_export_csv_correct_columns(self):
        from opencut.core.planar_track import TrackRegion, TrackResult, export_track_data
        tr = TrackResult(
            frames=[TrackRegion(corners=[(1, 2), (3, 4), (5, 6), (7, 8)])],
            confidence_per_frame=[0.99],
            fps=24.0,
            total_frames=1,
        )
        csv = export_track_data(tr, format="csv")
        header = csv.strip().split("\n")[0]
        self.assertIn("tl_x", header)
        self.assertIn("bl_y", header)
        self.assertIn("confidence", header)

    def test_export_nuke_has_all_corners(self):
        from opencut.core.planar_track import TrackRegion, TrackResult, export_track_data
        tr = TrackResult(
            frames=[
                TrackRegion(corners=[(10, 20), (110, 20), (110, 120), (10, 120)])
                for _ in range(5)
            ],
            confidence_per_frame=[0.9] * 5,
            fps=25.0,
            total_frames=5,
        )
        nuke = export_track_data(tr, format="nuke")
        for name in ("to1", "to2", "to3", "to4"):
            self.assertIn(name, nuke)

    def test_track_region_area_zero_for_degenerate(self):
        from opencut.core.planar_track import TrackRegion
        r = TrackRegion(corners=[(0, 0), (0, 0), (0, 0), (0, 0)])
        self.assertAlmostEqual(r.area, 0.0)

    def test_track_result_duration_with_fps(self):
        from opencut.core.planar_track import TrackResult
        tr = TrackResult(frames=[], confidence_per_frame=[], fps=60.0, total_frames=120)
        self.assertAlmostEqual(tr.duration, 2.0)


class TestVoronoiCrystallize(unittest.TestCase):
    """Tests for the Voronoi crystallize helper."""

    def test_voronoi_with_scipy(self):
        """If scipy is available, _voronoi_crystallize should run."""
        try:
            from scipy.spatial import Voronoi  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        import random

        from opencut.core.object_effects import _voronoi_crystallize
        region = _make_fake_frame(100, 100)
        result = _voronoi_crystallize(region, 10, random.Random(1))
        self.assertEqual(result.shape, region.shape)

    def test_crystallize_falls_back_without_scipy(self):
        """Without scipy, crystallize should fall back to grid pixelation."""
        from opencut.core.object_effects import _apply_crystallize
        frame = _make_fake_frame()
        mask = _make_fake_mask()
        rng = __import__("random").Random(99)
        with patch("opencut.core.object_effects._voronoi_crystallize",
                    side_effect=ImportError("no scipy")):
            result = _apply_crystallize(frame, mask, (160, 120, 320, 240), 0.5, 0.7, rng)
            self.assertEqual(result.shape, frame.shape)


class TestEffectProgressCallback(unittest.TestCase):
    """Tests verifying progress callbacks fire during effect application."""

    @patch("opencut.core.object_effects.run_ffmpeg")
    @patch("opencut.core.object_effects.get_video_info")
    @patch("opencut.core.object_effects.ensure_package", return_value=True)
    def test_progress_called(self, mock_ep, mock_info, mock_ffmpeg):

        mock_info.return_value = {"width": 640, "height": 480, "fps": 30.0, "duration": 1.0}

        frame = _make_fake_frame()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 1
        mock_cap.read.side_effect = [(True, frame.copy()), (False, None)]
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        progress_calls = []

        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0):
            from opencut.core.object_effects import (
                EffectConfig,
                ObjectMask,
                apply_object_effect,
            )
            mask = ObjectMask(
                mask_frames=[_make_fake_mask()],
                bbox_per_frame=[(160, 120, 320, 240)],
            )
            config = EffectConfig(effect_type="dissolve")
            apply_object_effect(
                _make_temp_file(), mask, config,
                out_path="/tmp/prog.mp4",
                on_progress=lambda pct, msg: progress_calls.append((pct, msg)),
            )
        self.assertTrue(len(progress_calls) >= 2)


class TestExportTrackDataEdge(unittest.TestCase):

    def test_export_json_single_frame(self):
        from opencut.core.planar_track import TrackRegion, TrackResult, export_track_data
        tr = TrackResult(
            frames=[TrackRegion(corners=[(0, 0), (10, 0), (10, 10), (0, 10)])],
            confidence_per_frame=[1.0],
            fps=24.0,
            total_frames=1,
        )
        data = json.loads(export_track_data(tr, format="json"))
        self.assertEqual(len(data["frames"]), 1)

    def test_export_csv_single_frame(self):
        from opencut.core.planar_track import TrackRegion, TrackResult, export_track_data
        tr = TrackResult(
            frames=[TrackRegion(corners=[(0, 0), (10, 0), (10, 10), (0, 10)])],
            confidence_per_frame=[0.5],
            fps=24.0,
            total_frames=1,
        )
        csv = export_track_data(tr, format="csv")
        lines = csv.strip().split("\n")
        self.assertEqual(len(lines), 2)  # header + 1 row


if __name__ == "__main__":
    unittest.main()
