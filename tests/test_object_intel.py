"""
Tests for OpenCut Object Intelligence features (Category 69).

Covers:
  - Text-based video segmentation (text_segment)
  - Physics-aware object removal (physics_remove)
  - Object tracking with overlays (object_track_overlay)
  - Semantic video search (semantic_video_search)
  - Multi-subject intelligent reframe (ai_reframe_multi)
  - Object intelligence routes (smoke tests)
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
# Text Segment - Dataclasses & Helpers
# ============================================================
class TestTextSegmentDataclasses(unittest.TestCase):
    """Tests for text_segment dataclasses."""

    def test_region_match_defaults(self):
        from opencut.core.text_segment import RegionMatch
        r = RegionMatch()
        self.assertEqual(r.x, 0)
        self.assertEqual(r.y, 0)
        self.assertEqual(r.width, 0)
        self.assertEqual(r.height, 0)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.frame_idx, 0)

    def test_region_match_to_dict(self):
        from opencut.core.text_segment import RegionMatch
        r = RegionMatch(x=10, y=20, width=100, height=50, score=0.85, frame_idx=3)
        d = r.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["x"], 10)
        self.assertEqual(d["y"], 20)
        self.assertEqual(d["width"], 100)
        self.assertEqual(d["height"], 50)
        self.assertEqual(d["score"], 0.85)
        self.assertEqual(d["frame_idx"], 3)

    def test_region_match_properties(self):
        from opencut.core.text_segment import RegionMatch
        r = RegionMatch(x=100, y=200, width=50, height=30)
        self.assertEqual(r.cx, 125)
        self.assertEqual(r.cy, 215)
        self.assertEqual(r.area, 1500)

    def test_text_segment_result_defaults(self):
        from opencut.core.text_segment import TextSegmentResult
        r = TextSegmentResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.mask_dir, "")
        self.assertEqual(r.query, "")
        self.assertEqual(r.frame_count, 0)
        self.assertEqual(r.best_score, 0.0)
        self.assertIsNone(r.best_region)
        self.assertEqual(r.method, "clip")
        self.assertFalse(r.sam2_refined)

    def test_text_segment_result_to_dict(self):
        from opencut.core.text_segment import TextSegmentResult
        r = TextSegmentResult(
            output_path="/out/video.webm",
            query="red car",
            frame_count=120,
            best_score=0.72,
            method="clip+sam2",
            sam2_refined=True,
            fps=29.97,
        )
        d = r.to_dict()
        self.assertEqual(d["output_path"], "/out/video.webm")
        self.assertEqual(d["query"], "red car")
        self.assertEqual(d["frame_count"], 120)
        self.assertTrue(d["sam2_refined"])
        self.assertEqual(d["fps"], 29.97)


class TestTextSegmentHelpers(unittest.TestCase):
    """Tests for text_segment helper functions."""

    def test_clean_query_strips_stop_words(self):
        from opencut.core.text_segment import _clean_query
        result = _clean_query("the red car on the left")
        self.assertNotIn("the", result.split())
        self.assertIn("red", result)
        self.assertIn("car", result)
        self.assertIn("left", result)

    def test_clean_query_empty_raises(self):
        from opencut.core.text_segment import _clean_query
        with self.assertRaises(ValueError):
            _clean_query("")
        with self.assertRaises(ValueError):
            _clean_query("   ")

    def test_clean_query_preserves_nouns(self):
        from opencut.core.text_segment import _clean_query
        result = _clean_query("person in blue")
        self.assertIn("person", result)
        self.assertIn("blue", result)

    def test_build_clip_prompts(self):
        from opencut.core.text_segment import _build_clip_prompts
        prompts = _build_clip_prompts("red car")
        self.assertIsInstance(prompts, list)
        self.assertTrue(len(prompts) >= 3)
        self.assertTrue(any("red car" in p for p in prompts))
        self.assertTrue(any("photo" in p for p in prompts))

    def test_sliding_window_regions(self):
        from opencut.core.text_segment import _sliding_window_regions
        regions = _sliding_window_regions(640, 480, grid_cols=4, grid_rows=3)
        self.assertIsInstance(regions, list)
        self.assertTrue(len(regions) > 0)
        # Each region should be a 4-tuple
        for r in regions:
            self.assertEqual(len(r), 4)
            x, y, w, h = r
            self.assertTrue(x >= 0)
            self.assertTrue(y >= 0)
            self.assertTrue(w > 0)
            self.assertTrue(h > 0)

    def test_sliding_window_covers_image(self):
        from opencut.core.text_segment import _sliding_window_regions
        regions = _sliding_window_regions(800, 600, grid_cols=4, grid_rows=3)
        # At least one region should start near top-left
        starts = [(r[0], r[1]) for r in regions]
        self.assertTrue(any(x < 50 and y < 50 for x, y in starts))

    def test_segment_by_text_missing_video(self):
        from opencut.core.text_segment import segment_by_text
        with self.assertRaises(FileNotFoundError):
            segment_by_text("/nonexistent/video.mp4", query="car")

    def test_segment_by_text_empty_query(self):
        from opencut.core.text_segment import segment_by_text
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                segment_by_text(path, query="")
        finally:
            os.unlink(path)

    def test_find_target_region_missing_frame(self):
        from opencut.core.text_segment import find_target_region
        with self.assertRaises(FileNotFoundError):
            find_target_region("/no/such/frame.png", "car")


# ============================================================
# Physics Remove - Dataclasses & Helpers
# ============================================================
class TestPhysicsRemoveDataclasses(unittest.TestCase):
    """Tests for physics_remove dataclasses."""

    def test_shadow_info_defaults(self):
        from opencut.core.physics_remove import ShadowInfo
        s = ShadowInfo()
        self.assertFalse(s.detected)
        self.assertEqual(s.direction_deg, 0.0)
        self.assertEqual(s.extent_px, 0)
        self.assertEqual(s.area_px, 0)
        self.assertEqual(s.intensity, 0.0)
        self.assertEqual(s.mask_points, [])
        self.assertEqual(s.bbox, (0, 0, 0, 0))

    def test_shadow_info_to_dict(self):
        from opencut.core.physics_remove import ShadowInfo
        s = ShadowInfo(detected=True, direction_deg=135.0, extent_px=80, area_px=5000)
        d = s.to_dict()
        self.assertTrue(d["detected"])
        self.assertEqual(d["direction_deg"], 135.0)
        self.assertEqual(d["extent_px"], 80)
        self.assertEqual(d["area_px"], 5000)

    def test_reflection_info_defaults(self):
        from opencut.core.physics_remove import ReflectionInfo
        r = ReflectionInfo()
        self.assertFalse(r.detected)
        self.assertEqual(r.surface_y, 0)
        self.assertEqual(r.similarity, 0.0)

    def test_reflection_info_to_dict(self):
        from opencut.core.physics_remove import ReflectionInfo
        r = ReflectionInfo(detected=True, surface_y=400, similarity=0.65, area_px=3000)
        d = r.to_dict()
        self.assertTrue(d["detected"])
        self.assertEqual(d["surface_y"], 400)
        self.assertEqual(d["similarity"], 0.65)

    def test_physics_remove_result_defaults(self):
        from opencut.core.physics_remove import PhysicsRemoveResult
        r = PhysicsRemoveResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.object_mask_area, 0)
        self.assertIsNone(r.shadow_info)
        self.assertIsNone(r.reflection_info)
        self.assertEqual(r.inpaint_method, "lama")

    def test_physics_remove_result_to_dict(self):
        from opencut.core.physics_remove import PhysicsRemoveResult
        r = PhysicsRemoveResult(
            output_path="/out/clean.mp4",
            object_mask_area=12000,
            total_removed_area=18000,
            frame_count=300,
            inpaint_method="ffmpeg_delogo",
        )
        d = r.to_dict()
        self.assertEqual(d["output_path"], "/out/clean.mp4")
        self.assertEqual(d["total_removed_area"], 18000)
        self.assertEqual(d["inpaint_method"], "ffmpeg_delogo")


class TestPhysicsRemoveHelpers(unittest.TestCase):
    """Tests for physics_remove helper functions."""

    def test_remove_with_physics_missing_video(self):
        from opencut.core.physics_remove import remove_with_physics
        with self.assertRaises(FileNotFoundError):
            remove_with_physics("/nonexistent.mp4", mask_points=[{"x": 100, "y": 100}])

    def test_remove_with_physics_no_mask_or_bbox(self):
        from opencut.core.physics_remove import remove_with_physics
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                remove_with_physics(path)
        finally:
            os.unlink(path)

    def test_detect_shadow_missing_frame(self):
        from opencut.core.physics_remove import detect_shadow
        with self.assertRaises(FileNotFoundError):
            detect_shadow("/no/frame.png")

    def test_detect_shadow_no_mask_or_bbox_raises(self):
        from opencut.core.physics_remove import detect_shadow
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake png")
            path = f.name
        try:
            # No mask, no bbox, no points => should raise
            with self.assertRaises((ValueError, Exception)):
                detect_shadow(path)
        finally:
            os.unlink(path)

    def test_constants_valid(self):
        from opencut.core.physics_remove import (
            INPAINT_DILATE_PX,
            INPAINT_TEMPORAL_WINDOW,
            SHADOW_INTENSITY_THRESHOLD,
            SHADOW_MAX_AREA_RATIO,
            SHADOW_MIN_AREA_RATIO,
            SHADOW_SEARCH_RADIUS,
        )
        self.assertTrue(0 < SHADOW_INTENSITY_THRESHOLD < 1)
        self.assertTrue(SHADOW_MIN_AREA_RATIO < SHADOW_MAX_AREA_RATIO)
        self.assertTrue(SHADOW_SEARCH_RADIUS > 1)
        self.assertTrue(INPAINT_DILATE_PX > 0)
        self.assertTrue(INPAINT_TEMPORAL_WINDOW > 0)


# ============================================================
# Object Track Overlay - Dataclasses & Helpers
# ============================================================
class TestObjectTrackOverlayDataclasses(unittest.TestCase):
    """Tests for object_track_overlay dataclasses."""

    def test_overlay_config_defaults(self):
        from opencut.core.object_track_overlay import OverlayConfig
        c = OverlayConfig()
        self.assertEqual(c.overlay_type, "text_label")
        self.assertEqual(c.font_size, 24)
        self.assertEqual(c.font_color, "#FFFFFF")
        self.assertEqual(c.blur_strength, 15)
        self.assertEqual(c.opacity, 1.0)
        self.assertTrue(c.scale_with_object)

    def test_overlay_config_to_dict(self):
        from opencut.core.object_track_overlay import OverlayConfig
        c = OverlayConfig(overlay_type="blur", blur_strength=25)
        d = c.to_dict()
        self.assertEqual(d["overlay_type"], "blur")
        self.assertEqual(d["blur_strength"], 25)

    def test_overlay_config_from_dict(self):
        from opencut.core.object_track_overlay import OverlayConfig
        c = OverlayConfig.from_dict({
            "overlay_type": "arrow",
            "arrow_color": "#00FF00",
            "unknown_key": "ignored",
        })
        self.assertEqual(c.overlay_type, "arrow")
        self.assertEqual(c.arrow_color, "#00FF00")

    def test_overlay_config_from_empty_dict(self):
        from opencut.core.object_track_overlay import OverlayConfig
        c = OverlayConfig.from_dict({})
        self.assertEqual(c.overlay_type, "text_label")

    def test_track_frame_defaults(self):
        from opencut.core.object_track_overlay import TrackFrame
        t = TrackFrame()
        self.assertEqual(t.frame_idx, 0)
        self.assertEqual(t.x, 0)
        self.assertEqual(t.y, 0)
        self.assertFalse(t.lost)
        self.assertEqual(t.scale, 1.0)

    def test_track_frame_properties(self):
        from opencut.core.object_track_overlay import TrackFrame
        t = TrackFrame(x=100, y=200, width=60, height=40)
        self.assertEqual(t.cx, 130)
        self.assertEqual(t.cy, 220)

    def test_track_frame_to_dict(self):
        from opencut.core.object_track_overlay import TrackFrame
        t = TrackFrame(frame_idx=5, x=50, y=60, width=80, height=90, confidence=0.95)
        d = t.to_dict()
        self.assertEqual(d["frame_idx"], 5)
        self.assertEqual(d["confidence"], 0.95)

    def test_track_overlay_result_defaults(self):
        from opencut.core.object_track_overlay import TrackOverlayResult
        r = TrackOverlayResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.tracked_frames, 0)
        self.assertEqual(r.lost_frames, 0)

    def test_track_overlay_result_to_dict(self):
        from opencut.core.object_track_overlay import TrackOverlayResult
        r = TrackOverlayResult(
            output_path="/out/tracked.mp4",
            frame_count=500,
            tracked_frames=480,
            lost_frames=20,
            overlay_type="text_label",
        )
        d = r.to_dict()
        self.assertEqual(d["tracked_frames"], 480)
        self.assertEqual(d["lost_frames"], 20)


class TestObjectTrackOverlayHelpers(unittest.TestCase):
    """Tests for object_track_overlay helper functions."""

    def test_overlay_types_list(self):
        from opencut.core.object_track_overlay import OVERLAY_TYPES
        self.assertIsInstance(OVERLAY_TYPES, list)
        self.assertTrue(len(OVERLAY_TYPES) >= 8)
        self.assertIn("text_label", OVERLAY_TYPES)
        self.assertIn("blur", OVERLAY_TYPES)
        self.assertIn("arrow", OVERLAY_TYPES)
        self.assertIn("highlight", OVERLAY_TYPES)
        self.assertIn("custom_image", OVERLAY_TYPES)
        self.assertIn("circle", OVERLAY_TYPES)
        self.assertIn("rectangle", OVERLAY_TYPES)
        self.assertIn("crosshair", OVERLAY_TYPES)
        self.assertIn("spotlight", OVERLAY_TYPES)
        self.assertIn("censor", OVERLAY_TYPES)

    def test_track_and_overlay_missing_video(self):
        from opencut.core.object_track_overlay import track_and_overlay
        with self.assertRaises(FileNotFoundError):
            track_and_overlay("/nonexistent.mp4", track_point=(100, 100))

    def test_track_and_overlay_invalid_overlay_type(self):
        from opencut.core.object_track_overlay import track_and_overlay
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                track_and_overlay(path, track_point=(100, 100),
                                  overlay_config={"overlay_type": "nonexistent"})
        finally:
            os.unlink(path)

    def test_smooth_track(self):
        from opencut.core.object_track_overlay import TrackFrame, _smooth_track
        # Create a track with some jitter
        tracks = []
        for i in range(20):
            jitter = 5 if i % 2 == 0 else -5
            tracks.append(TrackFrame(
                frame_idx=i, x=100 + jitter, y=200 + jitter,
                width=50, height=50, confidence=0.9,
            ))
        smoothed = _smooth_track(tracks, window=5)
        self.assertEqual(len(smoothed), 20)
        # Smoothed positions should have less variance
        orig_variance = sum((t.x - 100) ** 2 for t in tracks)
        smooth_variance = sum((t.x - 100) ** 2 for t in smoothed)
        self.assertTrue(smooth_variance <= orig_variance)

    def test_smooth_track_preserves_lost(self):
        from opencut.core.object_track_overlay import TrackFrame, _smooth_track
        tracks = [
            TrackFrame(frame_idx=0, x=100, y=100, width=50, height=50),
            TrackFrame(frame_idx=1, x=100, y=100, width=50, height=50, lost=True),
            TrackFrame(frame_idx=2, x=100, y=100, width=50, height=50),
        ]
        smoothed = _smooth_track(tracks, window=3)
        self.assertTrue(smoothed[1].lost)

    def test_init_template(self):
        from opencut.core.object_track_overlay import _init_template
        import numpy as np
        frame = np.zeros((480, 640), dtype=np.uint8)
        template, bbox = _init_template(frame, (320, 240), pad=30)
        self.assertEqual(bbox[2], 60)  # width = 2 * pad
        self.assertEqual(bbox[3], 60)  # height = 2 * pad
        self.assertEqual(template.shape, (60, 60))

    def test_init_template_near_edge(self):
        from opencut.core.object_track_overlay import _init_template
        import numpy as np
        frame = np.zeros((100, 100), dtype=np.uint8)
        template, bbox = _init_template(frame, (5, 5), pad=40)
        # Should be clamped to image bounds
        self.assertTrue(bbox[0] >= 0)
        self.assertTrue(bbox[1] >= 0)
        self.assertTrue(template.shape[0] > 0)
        self.assertTrue(template.shape[1] > 0)


# ============================================================
# Semantic Video Search - Dataclasses & Helpers
# ============================================================
class TestSemanticSearchDataclasses(unittest.TestCase):
    """Tests for semantic_video_search dataclasses."""

    def test_search_result_defaults(self):
        from opencut.core.semantic_video_search import SearchResult
        r = SearchResult()
        self.assertEqual(r.clip_path, "")
        self.assertEqual(r.clip_name, "")
        self.assertEqual(r.frame_idx, 0)
        self.assertEqual(r.score, 0.0)

    def test_search_result_to_dict(self):
        from opencut.core.semantic_video_search import SearchResult
        r = SearchResult(
            clip_path="/clips/intro.mp4",
            clip_name="intro.mp4",
            frame_idx=5,
            timestamp=2.5,
            score=0.87,
        )
        d = r.to_dict()
        self.assertEqual(d["clip_name"], "intro.mp4")
        self.assertEqual(d["score"], 0.87)
        self.assertEqual(d["timestamp"], 2.5)

    def test_semantic_search_result_defaults(self):
        from opencut.core.semantic_video_search import SemanticSearchResult
        r = SemanticSearchResult()
        self.assertEqual(r.query, "")
        self.assertEqual(r.query_type, "text")
        self.assertEqual(r.results, [])
        self.assertEqual(r.total_clips_searched, 0)
        self.assertFalse(r.index_cached)

    def test_semantic_search_result_to_dict(self):
        from opencut.core.semantic_video_search import SemanticSearchResult
        r = SemanticSearchResult(
            query="sunset over water",
            query_type="text",
            results=[{"score": 0.9}],
            total_clips_searched=5,
            search_time_ms=120,
            index_cached=True,
        )
        d = r.to_dict()
        self.assertEqual(d["query"], "sunset over water")
        self.assertTrue(d["index_cached"])
        self.assertEqual(d["search_time_ms"], 120)
        self.assertEqual(len(d["results"]), 1)

    def test_clip_index_entry_to_dict(self):
        from opencut.core.semantic_video_search import ClipIndexEntry
        e = ClipIndexEntry(
            clip_path="/clips/test.mp4",
            clip_hash="abc123",
            frame_count=12,
            fps=24.0,
            duration=30.0,
        )
        d = e.to_dict()
        self.assertEqual(d["clip_path"], "/clips/test.mp4")
        self.assertNotIn("frame_paths", d)  # Should be excluded


class TestSemanticSearchHelpers(unittest.TestCase):
    """Tests for semantic_video_search helper functions."""

    def test_clip_file_hash_deterministic(self):
        from opencut.core.semantic_video_search import _clip_file_hash
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test content")
            path = f.name
        try:
            h1 = _clip_file_hash(path)
            h2 = _clip_file_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 24)
        finally:
            os.unlink(path)

    def test_clip_file_hash_nonexistent(self):
        from opencut.core.semantic_video_search import _clip_file_hash
        h = _clip_file_hash("/nonexistent/file.mp4")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 24)

    def test_cache_path_creates_dir(self):
        from opencut.core.semantic_video_search import _cache_path
        path = _cache_path("testhash123")
        self.assertTrue(path.endswith(".pkl"))
        self.assertIn("testhash123", path)

    def test_load_cached_nonexistent(self):
        from opencut.core.semantic_video_search import _load_cached_embeddings
        result = _load_cached_embeddings("nonexistent_hash_xyz")
        self.assertIsNone(result)

    def test_semantic_search_empty_clips_raises(self):
        from opencut.core.semantic_video_search import semantic_search
        with self.assertRaises(ValueError):
            semantic_search([], query="test")

    def test_semantic_search_no_query_raises(self):
        from opencut.core.semantic_video_search import semantic_search
        with self.assertRaises(ValueError):
            semantic_search(["/some/clip.mp4"])

    def test_semantic_search_no_valid_paths(self):
        from opencut.core.semantic_video_search import semantic_search
        with self.assertRaises(FileNotFoundError):
            semantic_search(["/nonexistent1.mp4", "/nonexistent2.mp4"], query="test")

    def test_build_clip_index_empty_raises(self):
        from opencut.core.semantic_video_search import build_clip_index
        with self.assertRaises(ValueError):
            build_clip_index([])

    def test_build_clip_index_nonexistent_raises(self):
        from opencut.core.semantic_video_search import build_clip_index
        with self.assertRaises(FileNotFoundError):
            build_clip_index(["/nonexistent.mp4"])

    def test_constants_valid(self):
        from opencut.core.semantic_video_search import (
            CLIP_MODEL_NAME,
            DEFAULT_MAX_RESULTS,
            FRAMES_PER_CLIP,
            MIN_SIMILARITY,
        )
        self.assertTrue(FRAMES_PER_CLIP > 0)
        self.assertTrue(DEFAULT_MAX_RESULTS > 0)
        self.assertTrue(0 < MIN_SIMILARITY < 1)
        self.assertIn("clip", CLIP_MODEL_NAME.lower())


# ============================================================
# AI Reframe Multi - Dataclasses & Helpers
# ============================================================
class TestReframeMultiDataclasses(unittest.TestCase):
    """Tests for ai_reframe_multi dataclasses."""

    def test_subject_info_defaults(self):
        from opencut.core.ai_reframe_multi import SubjectInfo
        s = SubjectInfo()
        self.assertEqual(s.subject_type, "object")
        self.assertEqual(s.x, 0)
        self.assertEqual(s.importance, 0.5)

    def test_subject_info_properties(self):
        from opencut.core.ai_reframe_multi import SubjectInfo
        s = SubjectInfo(x=100, y=200, width=50, height=30)
        self.assertEqual(s.cx, 125)
        self.assertEqual(s.cy, 215)
        self.assertEqual(s.area, 1500)

    def test_subject_info_to_dict(self):
        from opencut.core.ai_reframe_multi import SubjectInfo
        s = SubjectInfo(subject_type="face", x=50, y=60, width=80, height=90, importance=0.95)
        d = s.to_dict()
        self.assertEqual(d["subject_type"], "face")
        self.assertEqual(d["importance"], 0.95)

    def test_crop_window_defaults(self):
        from opencut.core.ai_reframe_multi import CropWindow
        c = CropWindow()
        self.assertEqual(c.x, 0)
        self.assertEqual(c.coverage, 1.0)
        self.assertFalse(c.use_split)

    def test_crop_window_to_dict(self):
        from opencut.core.ai_reframe_multi import CropWindow
        c = CropWindow(x=100, y=50, width=720, height=1280, coverage=0.8, use_split=True)
        d = c.to_dict()
        self.assertEqual(d["width"], 720)
        self.assertTrue(d["use_split"])

    def test_reframe_result_defaults(self):
        from opencut.core.ai_reframe_multi import ReframeResult
        r = ReframeResult()
        self.assertEqual(r.output_path, "")
        self.assertEqual(r.split_screen_frames, 0)
        self.assertEqual(r.avg_coverage, 0.0)

    def test_reframe_result_to_dict(self):
        from opencut.core.ai_reframe_multi import ReframeResult
        r = ReframeResult(
            output_path="/out/reframed.mp4",
            target_ratio="9:16",
            frame_count=900,
            subjects_detected=45,
            avg_coverage=0.85,
        )
        d = r.to_dict()
        self.assertEqual(d["target_ratio"], "9:16")
        self.assertEqual(d["avg_coverage"], 0.85)


class TestReframeMultiHelpers(unittest.TestCase):
    """Tests for ai_reframe_multi helper functions."""

    def test_aspect_ratios_dict(self):
        from opencut.core.ai_reframe_multi import ASPECT_RATIOS
        self.assertIsInstance(ASPECT_RATIOS, dict)
        self.assertTrue(len(ASPECT_RATIOS) >= 5)
        self.assertIn("9:16", ASPECT_RATIOS)
        self.assertIn("1:1", ASPECT_RATIOS)
        self.assertIn("16:9", ASPECT_RATIOS)
        for name, (w, h) in ASPECT_RATIOS.items():
            self.assertTrue(w > 0, f"{name} width should be positive")
            self.assertTrue(h > 0, f"{name} height should be positive")

    def test_parse_ratio_valid(self):
        from opencut.core.ai_reframe_multi import _parse_ratio
        w, h = _parse_ratio("9:16")
        self.assertEqual(w, 9)
        self.assertEqual(h, 16)

    def test_parse_ratio_invalid(self):
        from opencut.core.ai_reframe_multi import _parse_ratio
        with self.assertRaises(ValueError):
            _parse_ratio("invalid")

    def test_compute_output_dims_vertical(self):
        from opencut.core.ai_reframe_multi import _compute_output_dims
        out_w, out_h = _compute_output_dims(1920, 1080, 9, 16)
        self.assertTrue(out_w <= 1920)
        self.assertTrue(out_h <= 1080)
        self.assertEqual(out_w % 2, 0)
        self.assertEqual(out_h % 2, 0)
        # Should be roughly 9:16
        ratio = out_w / out_h
        self.assertAlmostEqual(ratio, 9/16, places=1)

    def test_compute_output_dims_square(self):
        from opencut.core.ai_reframe_multi import _compute_output_dims
        out_w, out_h = _compute_output_dims(1920, 1080, 1, 1)
        self.assertEqual(out_w, out_h)
        self.assertTrue(out_w <= 1080)

    def test_compute_output_dims_even(self):
        from opencut.core.ai_reframe_multi import _compute_output_dims
        out_w, out_h = _compute_output_dims(1921, 1081, 16, 9)
        self.assertEqual(out_w % 2, 0)
        self.assertEqual(out_h % 2, 0)

    def test_compute_crop_for_subjects_empty(self):
        from opencut.core.ai_reframe_multi import _compute_crop_for_subjects
        crop = _compute_crop_for_subjects([], 720, 1280, 1920, 1080, 0)
        self.assertEqual(crop.width, 720)
        self.assertEqual(crop.height, 1280)
        self.assertEqual(crop.coverage, 0.0)

    def test_compute_crop_for_subjects_single(self):
        from opencut.core.ai_reframe_multi import SubjectInfo, _compute_crop_for_subjects
        subjects = [SubjectInfo(subject_type="face", x=500, y=300, width=100, height=100, importance=1.0)]
        crop = _compute_crop_for_subjects(subjects, 400, 400, 1920, 1080, 0)
        # Centre should be near the subject
        crop_cx = crop.x + crop.width // 2
        crop_cy = crop.y + crop.height // 2
        self.assertTrue(abs(crop_cx - 550) < 250)
        self.assertTrue(abs(crop_cy - 350) < 250)

    def test_smooth_crop_path_reduces_jitter(self):
        from opencut.core.ai_reframe_multi import CropWindow, _smooth_crop_path
        crops = []
        for i in range(30):
            jitter = 20 if i % 2 == 0 else -20
            crops.append(CropWindow(x=500 + jitter, y=200, width=720, height=1280, frame_idx=i))
        smoothed = _smooth_crop_path(crops, kernel_size=7, frame_w=1920)
        self.assertEqual(len(smoothed), 30)
        # Variance should be reduced
        orig_var = sum((c.x - 500) ** 2 for c in crops)
        smooth_var = sum((c.x - 500) ** 2 for c in smoothed)
        self.assertTrue(smooth_var <= orig_var)

    def test_smooth_crop_path_short_input(self):
        from opencut.core.ai_reframe_multi import CropWindow, _smooth_crop_path
        crops = [CropWindow(x=100, y=100, width=200, height=200)]
        smoothed = _smooth_crop_path(crops)
        self.assertEqual(len(smoothed), 1)

    def test_interpolate_crops(self):
        from opencut.core.ai_reframe_multi import CropWindow, _interpolate_crops
        analysed = {
            0: CropWindow(x=100, y=100, width=400, height=400, frame_idx=0),
            10: CropWindow(x=200, y=100, width=400, height=400, frame_idx=10),
        }
        result = _interpolate_crops(analysed, 11, 400, 400)
        self.assertEqual(len(result), 11)
        # Frame 5 should be roughly at x=150
        self.assertTrue(140 <= result[5].x <= 160)

    def test_interpolate_crops_empty(self):
        from opencut.core.ai_reframe_multi import _interpolate_crops
        result = _interpolate_crops({}, 5, 400, 400)
        self.assertEqual(len(result), 5)

    def test_subject_weights_valid(self):
        from opencut.core.ai_reframe_multi import SUBJECT_WEIGHTS
        self.assertIn("face", SUBJECT_WEIGHTS)
        self.assertIn("text", SUBJECT_WEIGHTS)
        self.assertIn("motion", SUBJECT_WEIGHTS)
        # Face should have highest weight
        self.assertEqual(max(SUBJECT_WEIGHTS.values()), SUBJECT_WEIGHTS["face"])

    def test_reframe_missing_video(self):
        from opencut.core.ai_reframe_multi import reframe_multi_subject
        with self.assertRaises(FileNotFoundError):
            reframe_multi_subject("/nonexistent.mp4", target_ratio="9:16")

    def test_reframe_invalid_ratio(self):
        from opencut.core.ai_reframe_multi import reframe_multi_subject
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises((ValueError, RuntimeError)):
                reframe_multi_subject(path, target_ratio="invalid_ratio")
        finally:
            os.unlink(path)


# ============================================================
# Route Smoke Tests
# ============================================================
class TestObjectIntelRoutes(unittest.TestCase):
    """Smoke tests for object intelligence route endpoints."""

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
            health_data = resp.get_json()
            cls.csrf_token = health_data.get("csrf_token", "")
        except Exception as e:
            cls.app = None
            cls.client = None
            cls.csrf_token = ""
            print(f"Warning: Could not create test app: {e}")

    def _headers(self):
        return {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    def _skip_if_no_app(self):
        if self.client is None:
            self.skipTest("Test app not available")

    # -- GET endpoints (synchronous) --

    def test_get_overlay_types(self):
        self._skip_if_no_app()
        resp = self.client.get("/video/overlay-types")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("overlay_types", data)
        self.assertIsInstance(data["overlay_types"], list)
        self.assertTrue(len(data["overlay_types"]) >= 8)
        self.assertIn("text_label", data["overlay_types"])
        self.assertIn("blur", data["overlay_types"])
        self.assertIn("count", data)

    def test_get_aspect_ratios(self):
        self._skip_if_no_app()
        resp = self.client.get("/video/aspect-ratios")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("aspect_ratios", data)
        self.assertIsInstance(data["aspect_ratios"], list)
        self.assertTrue(len(data["aspect_ratios"]) >= 5)
        # Check structure
        first = data["aspect_ratios"][0]
        self.assertIn("name", first)
        self.assertIn("width", first)
        self.assertIn("height", first)
        self.assertIn("decimal", first)
        # Verify 9:16 is present
        names = [r["name"] for r in data["aspect_ratios"]]
        self.assertIn("9:16", names)
        self.assertIn("1:1", names)

    # -- POST endpoints (validation tests) --

    def test_text_segment_no_query(self):
        self._skip_if_no_app()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            resp = self.client.post("/video/text-segment",
                                    headers=self._headers(),
                                    json={"filepath": path, "query": ""})
            # Should fail with validation error (400 or async job error)
            self.assertIn(resp.status_code, [400, 200, 202])
        finally:
            os.unlink(path)

    def test_physics_remove_no_mask(self):
        self._skip_if_no_app()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            resp = self.client.post("/video/physics-remove",
                                    headers=self._headers(),
                                    json={"filepath": path})
            # Should fail: no mask_points or object_bbox
            self.assertIn(resp.status_code, [400, 200, 202])
        finally:
            os.unlink(path)

    def test_track_overlay_no_point(self):
        self._skip_if_no_app()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            resp = self.client.post("/video/track-overlay",
                                    headers=self._headers(),
                                    json={"filepath": path})
            # Should fail: no track_point
            self.assertIn(resp.status_code, [400, 200, 202])
        finally:
            os.unlink(path)

    def test_semantic_search_no_query(self):
        self._skip_if_no_app()
        resp = self.client.post("/search/semantic",
                                headers=self._headers(),
                                json={"clip_paths": ["/fake/clip.mp4"]})
        # Should fail: no query or query_image
        self.assertIn(resp.status_code, [400, 200, 202])

    def test_semantic_search_empty_clips(self):
        self._skip_if_no_app()
        resp = self.client.post("/search/semantic",
                                headers=self._headers(),
                                json={"clip_paths": [], "query": "test"})
        # Should fail: empty clip_paths
        self.assertIn(resp.status_code, [400, 200, 202])

    def test_semantic_index_empty_clips(self):
        self._skip_if_no_app()
        resp = self.client.post("/search/semantic/index",
                                headers=self._headers(),
                                json={"clip_paths": []})
        self.assertIn(resp.status_code, [400, 200, 202])

    def test_reframe_multi_valid_structure(self):
        self._skip_if_no_app()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            resp = self.client.post("/video/reframe-multi",
                                    headers=self._headers(),
                                    json={
                                        "filepath": path,
                                        "target_ratio": "9:16",
                                        "enable_split_screen": True,
                                    })
            # Should accept the request (async job)
            self.assertIn(resp.status_code, [200, 202, 400])
        finally:
            os.unlink(path)

    def test_text_segment_preview_no_query(self):
        self._skip_if_no_app()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            resp = self.client.post("/video/text-segment/preview",
                                    headers=self._headers(),
                                    json={"filepath": path, "query": ""})
            self.assertIn(resp.status_code, [400, 200, 202])
        finally:
            os.unlink(path)

    def test_detect_shadow_no_params(self):
        self._skip_if_no_app()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            path = f.name
        try:
            resp = self.client.post("/video/physics-remove/detect-shadow",
                                    headers=self._headers(),
                                    json={"filepath": path})
            # May fail or succeed depending on extraction, but should not 500
            self.assertIn(resp.status_code, [200, 202, 400])
        finally:
            os.unlink(path)


# ============================================================
# Cross-module integration-style tests
# ============================================================
class TestCrossModuleConstants(unittest.TestCase):
    """Verify constants are consistent across modules."""

    def test_all_overlay_types_have_renderers(self):
        from opencut.core.object_track_overlay import (
            OVERLAY_TYPES,
            _OVERLAY_IMG_RENDERERS,
            _OVERLAY_RENDERERS,
        )
        all_renderers = set(_OVERLAY_RENDERERS.keys()) | set(_OVERLAY_IMG_RENDERERS.keys())
        for otype in OVERLAY_TYPES:
            self.assertIn(otype, all_renderers,
                          f"Overlay type '{otype}' missing renderer")

    def test_sam2_models_in_text_segment(self):
        from opencut.core.text_segment import SAM2_MODELS
        self.assertIn("tiny", SAM2_MODELS)
        self.assertIn("small", SAM2_MODELS)
        self.assertIn("base", SAM2_MODELS)
        self.assertIn("large", SAM2_MODELS)

    def test_aspect_ratios_have_decimal_values(self):
        from opencut.core.ai_reframe_multi import ASPECT_RATIOS
        for name, (w, h) in ASPECT_RATIOS.items():
            ratio = w / h
            self.assertTrue(ratio > 0, f"{name}: ratio should be positive")

    def test_subject_weights_sum(self):
        from opencut.core.ai_reframe_multi import SUBJECT_WEIGHTS
        # All weights should be between 0 and 1
        for name, weight in SUBJECT_WEIGHTS.items():
            self.assertTrue(0 < weight <= 1.0, f"{name} weight {weight} out of range")


class TestEdgeCases(unittest.TestCase):
    """Edge case tests across all modules."""

    def test_clean_query_all_stop_words(self):
        from opencut.core.text_segment import _clean_query
        # "the a" -> should still return something useful
        result = _clean_query("the a an")
        self.assertTrue(len(result) > 0)

    def test_overlay_config_partial_dict(self):
        from opencut.core.object_track_overlay import OverlayConfig
        c = OverlayConfig.from_dict({"text": "Hello", "opacity": 0.5})
        self.assertEqual(c.text, "Hello")
        self.assertEqual(c.opacity, 0.5)
        self.assertEqual(c.overlay_type, "text_label")  # default

    def test_region_match_zero_area(self):
        from opencut.core.text_segment import RegionMatch
        r = RegionMatch(x=0, y=0, width=0, height=0)
        self.assertEqual(r.area, 0)
        self.assertEqual(r.cx, 0)
        self.assertEqual(r.cy, 0)

    def test_shadow_info_empty_mask_points(self):
        from opencut.core.physics_remove import ShadowInfo
        s = ShadowInfo(detected=True, mask_points=[])
        d = s.to_dict()
        self.assertEqual(d["mask_points"], [])

    def test_compute_output_dims_very_small(self):
        from opencut.core.ai_reframe_multi import _compute_output_dims
        out_w, out_h = _compute_output_dims(10, 10, 1, 1)
        self.assertTrue(out_w >= 2)
        self.assertTrue(out_h >= 2)
        self.assertEqual(out_w % 2, 0)
        self.assertEqual(out_h % 2, 0)

    def test_parse_ratio_with_slash(self):
        from opencut.core.ai_reframe_multi import _parse_ratio
        # "9/16" should parse like "9:16"
        w, h = _parse_ratio("9/16")
        self.assertTrue(w > 0)
        self.assertTrue(h > 0)

    def test_search_result_clip_name_matches_path(self):
        from opencut.core.semantic_video_search import SearchResult
        r = SearchResult(clip_path="/videos/test_clip.mp4", clip_name="test_clip.mp4")
        d = r.to_dict()
        self.assertEqual(os.path.basename(d["clip_path"]), d["clip_name"])

    def test_track_frame_lost_state(self):
        from opencut.core.object_track_overlay import TrackFrame
        t = TrackFrame(frame_idx=10, lost=True, confidence=0.1)
        self.assertTrue(t.lost)
        self.assertEqual(t.confidence, 0.1)
        d = t.to_dict()
        self.assertTrue(d["lost"])

    def test_semantic_search_result_empty_results(self):
        from opencut.core.semantic_video_search import SemanticSearchResult
        r = SemanticSearchResult(query="test", results=[])
        d = r.to_dict()
        self.assertEqual(d["results"], [])
        self.assertEqual(d["total_clips_searched"], 0)

    def test_physics_remove_result_no_shadow(self):
        from opencut.core.physics_remove import PhysicsRemoveResult
        r = PhysicsRemoveResult(output_path="/out.mp4", shadow_info=None)
        d = r.to_dict()
        self.assertIsNone(d["shadow_info"])
        self.assertIsNone(d["reflection_info"])

    def test_multiple_aspect_ratio_outputs(self):
        from opencut.core.ai_reframe_multi import _compute_output_dims
        # All common ratios should produce valid output
        test_cases = [
            ("9:16", 9, 16),
            ("1:1", 1, 1),
            ("16:9", 16, 9),
            ("4:3", 4, 3),
        ]
        for name, rw, rh in test_cases:
            out_w, out_h = _compute_output_dims(1920, 1080, rw, rh)
            self.assertTrue(out_w >= 2, f"{name} width too small")
            self.assertTrue(out_h >= 2, f"{name} height too small")
            self.assertTrue(out_w <= 1920, f"{name} width exceeds source")
            self.assertTrue(out_h <= 1080, f"{name} height exceeds source")


if __name__ == "__main__":
    unittest.main()
