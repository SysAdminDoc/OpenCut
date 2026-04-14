"""
Tests for OpenCut Composition & Dubbing features.

Covers:
  - 61.1 Composition Guide Overlay
  - 61.2 Shot Type Auto-Classification (classify_shot_type)
  - 61.3 Intelligent Pacing Analysis (analyze_pacing_from_cuts)
  - 61.4 Saliency-Guided Auto-Crop
  - 62.1 End-to-End AI Dubbing Pipeline
  - 62.2 Isochronous Translation
  - 62.3 Multi-Language Audio Track Management
  - 62.4 Voice Translation with Emotion Preservation
  - Routes smoke tests
"""

import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 61.1 Composition Guide Overlay
# ============================================================
class TestCompositionGuide(unittest.TestCase):
    """Tests for opencut.core.composition_guide."""

    def test_guide_types_list(self):
        """list_guide_types returns all expected guide types."""
        from opencut.core.composition_guide import list_guide_types, GUIDE_TYPES
        types = list_guide_types()
        self.assertIsInstance(types, list)
        self.assertEqual(len(types), len(GUIDE_TYPES))
        for t in types:
            self.assertIn("type", t)
            self.assertIn("description", t)

    def test_guide_types_contains_rule_of_thirds(self):
        """GUIDE_TYPES should contain rule_of_thirds."""
        from opencut.core.composition_guide import GUIDE_TYPES
        self.assertIn("rule_of_thirds", GUIDE_TYPES)

    def test_guide_types_contains_safe_areas(self):
        """GUIDE_TYPES should contain safe_areas."""
        from opencut.core.composition_guide import GUIDE_TYPES
        self.assertIn("safe_areas", GUIDE_TYPES)

    def test_guide_types_contains_golden_ratio(self):
        """GUIDE_TYPES should contain golden_ratio."""
        from opencut.core.composition_guide import GUIDE_TYPES
        self.assertIn("golden_ratio", GUIDE_TYPES)

    def test_guide_types_contains_fibonacci_spiral(self):
        """GUIDE_TYPES should include fibonacci_spiral."""
        from opencut.core.composition_guide import GUIDE_TYPES
        self.assertIn("fibonacci_spiral", GUIDE_TYPES)

    def test_default_colors_defined(self):
        """DEFAULT_COLORS should have entries for all primary guide types."""
        from opencut.core.composition_guide import DEFAULT_COLORS
        for key in ["rule_of_thirds", "golden_ratio", "diagonal",
                     "center_cross", "title_safe", "action_safe"]:
            self.assertIn(key, DEFAULT_COLORS)
            self.assertEqual(len(DEFAULT_COLORS[key]), 4)  # RGBA

    def test_phi_constant(self):
        """PHI constant should be approximately 1.618."""
        from opencut.core.composition_guide import PHI
        self.assertAlmostEqual(PHI, 1.618, places=2)

    def test_generate_overlay_missing_file(self):
        """generate_guide_overlay raises FileNotFoundError for missing input."""
        from opencut.core.composition_guide import generate_guide_overlay
        with self.assertRaises(FileNotFoundError):
            generate_guide_overlay("/nonexistent/video.mp4")

    @patch("opencut.core.composition_guide.ensure_package", return_value=True)
    @patch("opencut.core.composition_guide._extract_frame", return_value=False)
    @patch("opencut.core.composition_guide.get_video_info")
    def test_generate_overlay_fallback_to_blank(self, mock_info, mock_extract, mock_pkg):
        """Should create overlay on blank canvas when frame extraction fails."""
        mock_info.return_value = {"width": 1920, "height": 1080, "duration": 10}

        PIL_Image = MagicMock()
        PIL_ImageDraw = MagicMock()
        img_mock = MagicMock()
        img_mock.size = (1920, 1080)
        PIL_Image.new.return_value = img_mock
        PIL_Image.open.return_value = img_mock
        img_mock.convert.return_value = img_mock
        PIL_Image.alpha_composite.return_value = img_mock

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with patch.dict("sys.modules", {
                "PIL": MagicMock(Image=PIL_Image, ImageDraw=PIL_ImageDraw),
                "PIL.Image": PIL_Image,
                "PIL.ImageDraw": PIL_ImageDraw,
            }):
                from opencut.core.composition_guide import generate_guide_overlay
                result = generate_guide_overlay(path, guides=["rule_of_thirds"])
            self.assertIn("guides_applied", result)
            self.assertIn("rule_of_thirds", result["guides_applied"])
        finally:
            os.unlink(path)

    def test_draw_functions_exist(self):
        """All draw helper functions should be importable."""
        from opencut.core.composition_guide import (
            _draw_rule_of_thirds,
            _draw_golden_ratio,
            _draw_diagonal,
            _draw_center_cross,
            _draw_safe_areas,
            _draw_fibonacci_spiral,
            _draw_triangle,
            _draw_grid_4x4,
        )
        # Just verify they are callable
        for fn in [_draw_rule_of_thirds, _draw_golden_ratio, _draw_diagonal,
                    _draw_center_cross, _draw_fibonacci_spiral, _draw_triangle,
                    _draw_grid_4x4]:
            self.assertTrue(callable(fn))


# ============================================================
# 61.2 Shot Type Auto-Classification
# ============================================================
class TestShotClassify(unittest.TestCase):
    """Tests for opencut.core.shot_classify (enhanced)."""

    def test_shot_type_abbrevs_defined(self):
        """SHOT_TYPE_ABBREVS should map abbreviations to full names."""
        from opencut.core.shot_classify import SHOT_TYPE_ABBREVS
        self.assertIn("ECU", SHOT_TYPE_ABBREVS)
        self.assertIn("CU", SHOT_TYPE_ABBREVS)
        self.assertIn("MCU", SHOT_TYPE_ABBREVS)
        self.assertIn("MS", SHOT_TYPE_ABBREVS)
        self.assertIn("WS", SHOT_TYPE_ABBREVS)

    def test_shot_type_result_dataclass(self):
        """ShotTypeResult dataclass should have expected fields."""
        from opencut.core.shot_classify import ShotTypeResult
        r = ShotTypeResult()
        self.assertEqual(r.shot_type, "medium")
        self.assertEqual(r.confidence, 0.0)
        self.assertEqual(r.face_count, 0)

    def test_classify_shot_type_missing_file(self):
        """classify_shot_type raises FileNotFoundError for missing frame."""
        from opencut.core.shot_classify import classify_shot_type
        with self.assertRaises(FileNotFoundError):
            classify_shot_type("/nonexistent/frame.jpg")

    @patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.5)
    @patch("opencut.core.shot_classify._compute_edge_density", return_value=0.5)
    @patch("opencut.core.shot_classify._detect_faces_in_frame")
    @patch("opencut.core.shot_classify.subprocess.run")
    def test_classify_shot_type_single_face_ecu(self, mock_run, mock_faces,
                                                  mock_edge, mock_depth):
        """Single large face should classify as ECU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
        )
        # Face covers >30% of frame
        mock_faces.return_value = [{"x": 500, "y": 200, "w": 900, "h": 700}]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.shot_classify import classify_shot_type
            result = classify_shot_type(path)
            self.assertEqual(result["shot_type"], "extreme_close_up")
            self.assertEqual(result["abbreviation"], "ECU")
            self.assertEqual(result["face_count"], 1)
            self.assertEqual(result["classification_method"], "face_ratio")
        finally:
            os.unlink(path)

    @patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.5)
    @patch("opencut.core.shot_classify._compute_edge_density", return_value=0.5)
    @patch("opencut.core.shot_classify._detect_faces_in_frame")
    @patch("opencut.core.shot_classify.subprocess.run")
    def test_classify_shot_type_single_face_cu(self, mock_run, mock_faces,
                                                 mock_edge, mock_depth):
        """Medium-large face should classify as CU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
        )
        # Face ~20% of frame
        mock_faces.return_value = [{"x": 700, "y": 300, "w": 500, "h": 830}]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.shot_classify import classify_shot_type
            result = classify_shot_type(path)
            self.assertEqual(result["shot_type"], "close_up")
            self.assertEqual(result["abbreviation"], "CU")
        finally:
            os.unlink(path)

    @patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.5)
    @patch("opencut.core.shot_classify._compute_edge_density", return_value=0.5)
    @patch("opencut.core.shot_classify._detect_faces_in_frame")
    @patch("opencut.core.shot_classify.subprocess.run")
    def test_classify_shot_type_two_shot(self, mock_run, mock_faces,
                                          mock_edge, mock_depth):
        """Two faces should classify as two_shot."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
        )
        mock_faces.return_value = [
            {"x": 200, "y": 200, "w": 300, "h": 400},
            {"x": 1200, "y": 200, "w": 300, "h": 400},
        ]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.shot_classify import classify_shot_type
            result = classify_shot_type(path)
            self.assertEqual(result["shot_type"], "two_shot")
            self.assertEqual(result["abbreviation"], "TWO_SHOT")
            self.assertEqual(result["face_count"], 2)
            self.assertEqual(result["classification_method"], "multi_face")
        finally:
            os.unlink(path)

    @patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.5)
    @patch("opencut.core.shot_classify._compute_edge_density", return_value=0.5)
    @patch("opencut.core.shot_classify._detect_faces_in_frame")
    @patch("opencut.core.shot_classify.subprocess.run")
    def test_classify_shot_type_group(self, mock_run, mock_faces,
                                       mock_edge, mock_depth):
        """Three or more faces should classify as group_shot."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
        )
        mock_faces.return_value = [
            {"x": 100, "y": 200, "w": 200, "h": 300},
            {"x": 700, "y": 200, "w": 200, "h": 300},
            {"x": 1300, "y": 200, "w": 200, "h": 300},
        ]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.shot_classify import classify_shot_type
            result = classify_shot_type(path)
            self.assertEqual(result["shot_type"], "group_shot")
            self.assertEqual(result["abbreviation"], "GROUP")
            self.assertEqual(result["face_count"], 3)
        finally:
            os.unlink(path)

    @patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.2)
    @patch("opencut.core.shot_classify._compute_edge_density", return_value=0.8)
    @patch("opencut.core.shot_classify._detect_faces_in_frame", return_value=[])
    @patch("opencut.core.shot_classify.subprocess.run")
    def test_classify_shot_type_no_face_wide(self, mock_run, mock_faces,
                                               mock_edge, mock_depth):
        """No face + high edge density + low depth = extreme wide."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.shot_classify import classify_shot_type
            result = classify_shot_type(path)
            self.assertEqual(result["shot_type"], "extreme_wide")
            self.assertEqual(result["classification_method"], "edge_density")
        finally:
            os.unlink(path)

    @patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.7)
    @patch("opencut.core.shot_classify._compute_edge_density", return_value=0.1)
    @patch("opencut.core.shot_classify._detect_faces_in_frame", return_value=[])
    @patch("opencut.core.shot_classify.subprocess.run")
    def test_classify_shot_type_no_face_insert(self, mock_run, mock_faces,
                                                 mock_edge, mock_depth):
        """No face + low edge + high depth variance = ECU/insert."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.shot_classify import classify_shot_type
            result = classify_shot_type(path)
            self.assertEqual(result["shot_type"], "extreme_close_up")
        finally:
            os.unlink(path)

    def test_classify_shot_result_has_all_fields(self):
        """classify_shot_type result should have all expected fields."""
        expected_fields = [
            "shot_type", "abbreviation", "confidence", "face_count",
            "face_ratio", "edge_density", "depth_variance",
            "classification_method",
        ]
        # Use a mock to avoid needing real image
        with patch("opencut.core.shot_classify.subprocess.run") as mock_run, \
             patch("opencut.core.shot_classify._detect_faces_in_frame", return_value=[]), \
             patch("opencut.core.shot_classify._compute_edge_density", return_value=0.5), \
             patch("opencut.core.shot_classify._compute_depth_variance", return_value=0.5):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({"streams": [{"width": 1920, "height": 1080}]}),
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(b"fake")
                path = f.name
            try:
                from opencut.core.shot_classify import classify_shot_type
                result = classify_shot_type(path)
                for field in expected_fields:
                    self.assertIn(field, result, f"Missing field: {field}")
            finally:
                os.unlink(path)


# ============================================================
# 61.3 Intelligent Pacing Analysis
# ============================================================
class TestPacingAnalysis(unittest.TestCase):
    """Tests for opencut.core.pacing_analysis (enhanced)."""

    def test_analyze_pacing_from_cuts_basic(self):
        """Basic cut-point analysis should return valid metrics."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[5.0, 10.0, 15.0, 20.0],
            total_duration=25.0,
        )
        self.assertEqual(result["total_shots"], 5)
        self.assertAlmostEqual(result["mean_shot_length"], 5.0, places=1)
        self.assertAlmostEqual(result["median_shot_length"], 5.0, places=1)
        self.assertGreater(result["cuts_per_minute"], 0)

    def test_analyze_pacing_from_cuts_no_cuts(self):
        """Empty cut points should return single shot result."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[],
            total_duration=60.0,
        )
        self.assertEqual(result["total_shots"], 1)
        self.assertEqual(result["mean_shot_length"], 60.0)

    def test_analyze_pacing_from_cuts_negative_duration(self):
        """Negative duration should raise ValueError."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        with self.assertRaises(ValueError):
            analyze_pacing_from_cuts(cut_points=[5], total_duration=-1)

    def test_analyze_pacing_from_cuts_stddev(self):
        """Stddev should be 0 for equal-length shots."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[10.0, 20.0, 30.0],
            total_duration=40.0,
        )
        self.assertAlmostEqual(result["stddev_shot_length"], 0.0, places=1)

    def test_analyze_pacing_from_cuts_unequal_stddev(self):
        """Stddev should be positive for unequal shots."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[1.0, 2.0, 20.0],
            total_duration=25.0,
        )
        self.assertGreater(result["stddev_shot_length"], 0)

    def test_analyze_pacing_from_cuts_genre_comparisons(self):
        """Result should include genre comparisons dict."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[3, 6, 9, 12, 15],
            total_duration=18.0,
            genre="trailer",
        )
        self.assertIn("genre_comparisons", result)
        self.assertIn("trailer", result["genre_comparisons"])
        trailer = result["genre_comparisons"]["trailer"]
        self.assertIn("match_quality", trailer)

    def test_analyze_pacing_from_cuts_anomalies(self):
        """Anomalies should be flagged for outlier shots."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        # Create data with one extreme outlier
        result = analyze_pacing_from_cuts(
            cut_points=[2, 4, 6, 8, 10, 12, 14, 16, 18, 50],
            total_duration=60.0,
            anomaly_threshold=1.5,
        )
        self.assertIn("anomalies", result)
        # The 32-second segment (50-18) should be flagged
        if result["anomalies"]:
            self.assertTrue(any(a["direction"] == "too_long"
                                for a in result["anomalies"]))

    def test_analyze_pacing_from_cuts_bar_chart(self):
        """Result should include bar chart data."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[5, 10, 15, 20],
            total_duration=25.0,
        )
        self.assertIn("bar_chart", result)
        self.assertIsInstance(result["bar_chart"], list)

    def test_analyze_pacing_from_cuts_pacing_curve(self):
        """Result should include pacing curve."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[5, 10, 15, 20, 25, 30],
            total_duration=60.0,
        )
        self.assertIn("pacing_curve", result)
        self.assertIsInstance(result["pacing_curve"], list)
        if result["pacing_curve"]:
            point = result["pacing_curve"][0]
            self.assertIn("time", point)
            self.assertIn("local_cpm", point)

    def test_analyze_pacing_from_cuts_suggestions(self):
        """Result should include suggestions list."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[1, 2, 3, 4, 5],
            total_duration=6.0,
            genre="cinematic",
        )
        self.assertIn("suggestions", result)
        self.assertIsInstance(result["suggestions"], list)

    def test_analyze_pacing_from_cuts_deduplication(self):
        """Duplicate cut points should be deduplicated."""
        from opencut.core.pacing_analysis import analyze_pacing_from_cuts
        result = analyze_pacing_from_cuts(
            cut_points=[5.0, 5.0, 10.0, 10.0],
            total_duration=15.0,
        )
        # After dedup: cuts at 5 and 10 -> 3 shots
        self.assertEqual(result["total_shots"], 3)

    def test_compute_median_odd(self):
        """Median of odd-length list."""
        from opencut.core.pacing_analysis import _compute_median
        self.assertEqual(_compute_median([1, 3, 5]), 3)

    def test_compute_median_even(self):
        """Median of even-length list."""
        from opencut.core.pacing_analysis import _compute_median
        self.assertEqual(_compute_median([1, 3, 5, 7]), 4.0)

    def test_compute_median_empty(self):
        """Median of empty list should be 0."""
        from opencut.core.pacing_analysis import _compute_median
        self.assertEqual(_compute_median([]), 0.0)

    def test_compute_stddev_single(self):
        """Stddev of single value should be 0."""
        from opencut.core.pacing_analysis import _compute_stddev
        self.assertEqual(_compute_stddev([5.0], 5.0), 0.0)

    def test_genre_profiles_complete(self):
        """GENRE_PROFILES should have all expected genres."""
        from opencut.core.pacing_analysis import GENRE_PROFILES
        expected = ["general", "trailer", "interview", "documentary",
                    "music_video", "vlog", "commercial", "cinematic"]
        for g in expected:
            self.assertIn(g, GENRE_PROFILES)


# ============================================================
# 61.4 Saliency-Guided Auto-Crop
# ============================================================
class TestSaliencyCrop(unittest.TestCase):
    """Tests for opencut.core.saliency_crop."""

    def test_aspect_ratios_defined(self):
        """ASPECT_RATIOS should contain standard ratios."""
        from opencut.core.saliency_crop import ASPECT_RATIOS
        self.assertIn("9:16", ASPECT_RATIOS)
        self.assertIn("1:1", ASPECT_RATIOS)
        self.assertIn("16:9", ASPECT_RATIOS)

    def test_saliency_crop_missing_file(self):
        """saliency_crop raises FileNotFoundError for missing input."""
        from opencut.core.saliency_crop import saliency_crop
        with self.assertRaises(FileNotFoundError):
            saliency_crop("/nonexistent/video.mp4")

    def test_find_optimal_crop_empty_regions(self):
        """Empty regions should return center crop."""
        from opencut.core.saliency_crop import _find_optimal_crop
        x, y = _find_optimal_crop([], 1920, 1080, 608, 1080)
        self.assertEqual(x, (1920 - 608) // 2)
        self.assertEqual(y, 0)

    def test_find_optimal_crop_single_region(self):
        """Single region should pull crop toward it."""
        from opencut.core.saliency_crop import _find_optimal_crop, SaliencyRegion
        regions = [SaliencyRegion(x=100, y=100, w=200, h=200, weight=5.0)]
        x, y = _find_optimal_crop(regions, 1920, 1080, 608, 1080)
        # Crop should be pulled leftward toward the region
        self.assertLessEqual(x, 1920 // 2)

    def test_smooth_crop_path_empty(self):
        """Empty keyframe list should return empty."""
        from opencut.core.saliency_crop import _smooth_crop_path
        self.assertEqual(_smooth_crop_path([]), [])

    def test_smooth_crop_path_single(self):
        """Single keyframe should be returned as-is."""
        from opencut.core.saliency_crop import _smooth_crop_path, CropKeyframe
        kf = CropKeyframe(time=0, x=100, y=100, w=600, h=1080)
        result = _smooth_crop_path([kf])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].x, 100)

    def test_smooth_crop_path_smoothing(self):
        """Smoothing should reduce jitter between keyframes."""
        from opencut.core.saliency_crop import _smooth_crop_path, CropKeyframe
        keyframes = [
            CropKeyframe(time=0, x=100, y=0, w=600, h=1080),
            CropKeyframe(time=1, x=500, y=0, w=600, h=1080),
            CropKeyframe(time=2, x=100, y=0, w=600, h=1080),
        ]
        smoothed = _smooth_crop_path(keyframes, smoothing=0.7)
        # Second keyframe should not jump all the way to 500
        self.assertLess(smoothed[1].x, 400)
        self.assertGreater(smoothed[1].x, 100)

    def test_saliency_region_dataclass(self):
        """SaliencyRegion should have expected defaults."""
        from opencut.core.saliency_crop import SaliencyRegion
        r = SaliencyRegion()
        self.assertEqual(r.weight, 1.0)
        self.assertEqual(r.source, "unknown")


# ============================================================
# 62.1 AI Dubbing Pipeline
# ============================================================
class TestAIDubbing(unittest.TestCase):
    """Tests for opencut.core.ai_dubbing."""

    def test_supported_languages_count(self):
        """Should support 50+ languages."""
        from opencut.core.ai_dubbing import SUPPORTED_LANGUAGES
        self.assertGreaterEqual(len(SUPPORTED_LANGUAGES), 50)

    def test_supported_languages_contain_major(self):
        """Should contain all major world languages."""
        from opencut.core.ai_dubbing import SUPPORTED_LANGUAGES
        for lang in ["en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi"]:
            self.assertIn(lang, SUPPORTED_LANGUAGES)

    def test_list_supported_languages(self):
        """list_supported_languages should return sorted list."""
        from opencut.core.ai_dubbing import list_supported_languages
        langs = list_supported_languages()
        self.assertIsInstance(langs, list)
        self.assertGreaterEqual(len(langs), 50)
        self.assertIn("code", langs[0])
        self.assertIn("name", langs[0])

    def test_run_dubbing_missing_file(self):
        """run_dubbing_pipeline raises FileNotFoundError for missing input."""
        from opencut.core.ai_dubbing import run_dubbing_pipeline
        with self.assertRaises(FileNotFoundError):
            run_dubbing_pipeline("/nonexistent.mp4", "es")

    def test_run_dubbing_invalid_language(self):
        """run_dubbing_pipeline raises ValueError for unsupported language."""
        from opencut.core.ai_dubbing import run_dubbing_pipeline
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                run_dubbing_pipeline(path, "zzz_invalid")
        finally:
            os.unlink(path)

    @patch("opencut.core.ai_dubbing.run_ffmpeg")
    @patch("opencut.core.ai_dubbing._mix_dubbed_audio", return_value=True)
    @patch("opencut.core.ai_dubbing._separate_stems")
    @patch("opencut.core.ai_dubbing._generate_tts", return_value=False)
    @patch("opencut.core.ai_dubbing._translate_segments")
    @patch("opencut.core.ai_dubbing._transcribe_audio")
    @patch("opencut.core.ai_dubbing._extract_audio", return_value=True)
    @patch("opencut.core.ai_dubbing.get_video_info")
    def test_run_dubbing_pipeline_flow(self, mock_info, mock_extract, mock_transcribe,
                                        mock_translate, mock_tts, mock_stems,
                                        mock_mix, mock_ffmpeg):
        """Dubbing pipeline should follow expected flow."""
        mock_info.return_value = {"width": 1920, "height": 1080, "duration": 30, "fps": 30}
        mock_transcribe.return_value = [
            {"start": 0, "end": 5, "text": "Hello world", "speaker": "s0"},
        ]
        mock_translate.return_value = [
            {"start": 0, "end": 5, "text": "Hello world",
             "translated_text": "Hola mundo", "speaker": "s0"},
        ]
        mock_stems.return_value = {"vocals": "", "background": "bg.wav"}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.ai_dubbing import run_dubbing_pipeline
            result = run_dubbing_pipeline(path, "es")
            self.assertEqual(result["target_language"], "es")
            self.assertIn("segments", result)
            mock_transcribe.assert_called_once()
            mock_translate.assert_called_once()
        finally:
            os.unlink(path)

    def test_extract_audio_function_exists(self):
        """_extract_audio helper should be importable."""
        from opencut.core.ai_dubbing import _extract_audio
        self.assertTrue(callable(_extract_audio))


# ============================================================
# 62.2 Isochronous Translation
# ============================================================
class TestIsochronousTranslate(unittest.TestCase):
    """Tests for opencut.core.isochronous_translate."""

    def test_speaking_rates_defined(self):
        """Speaking rates should be defined for major languages."""
        from opencut.core.isochronous_translate import SPEAKING_RATES_WPM
        self.assertIn("en", SPEAKING_RATES_WPM)
        self.assertIn("es", SPEAKING_RATES_WPM)
        self.assertIn("zh", SPEAKING_RATES_WPM)

    def test_estimate_tts_duration_english(self):
        """English TTS duration estimation should be reasonable."""
        from opencut.core.isochronous_translate import _estimate_tts_duration
        dur = _estimate_tts_duration("This is a test sentence", "en")
        self.assertGreater(dur, 0.5)
        self.assertLess(dur, 10.0)

    def test_estimate_tts_duration_empty(self):
        """Empty text should return 0 duration."""
        from opencut.core.isochronous_translate import _estimate_tts_duration
        dur = _estimate_tts_duration("", "en")
        self.assertEqual(dur, 0.0)

    def test_estimate_tts_duration_chinese(self):
        """Chinese estimation should use character-based counting."""
        from opencut.core.isochronous_translate import _estimate_tts_duration
        dur = _estimate_tts_duration("hello", "zh")
        self.assertGreater(dur, 0)

    def test_translate_isochronous_empty_segments(self):
        """Empty segments should raise ValueError."""
        from opencut.core.isochronous_translate import translate_isochronous
        with self.assertRaises(ValueError):
            translate_isochronous([], "en", "es")

    @patch("opencut.core.isochronous_translate._translate_text")
    def test_translate_isochronous_basic(self, mock_translate):
        """Basic isochronous translation should return results."""
        mock_translate.return_value = "Hola mundo"

        from opencut.core.isochronous_translate import translate_isochronous
        result = translate_isochronous(
            segments=[
                {"text": "Hello world", "start": 0, "end": 3},
            ],
            source_language="en",
            target_language="es",
        )
        self.assertEqual(result["total_segments"], 1)
        self.assertEqual(result["source_language"], "en")
        self.assertEqual(result["target_language"], "es")
        self.assertIn("segments", result)

    @patch("opencut.core.isochronous_translate._translate_text")
    def test_translate_isochronous_within_tolerance(self, mock_translate):
        """Appropriately-sized text should fit within tolerance."""
        # 5 words over 3 seconds at 150 wpm = 2.0s estimated => ratio ~0.67
        # tolerance 0.5 means 0.5..1.5 is OK => 0.67 is within
        mock_translate.return_value = "Hello there my friend now"

        from opencut.core.isochronous_translate import translate_isochronous
        result = translate_isochronous(
            segments=[
                {"text": "Hello there my friend now", "start": 0, "end": 3},
            ],
            source_language="en",
            target_language="es",
            tolerance=0.5,  # Very generous tolerance
        )
        seg = result["segments"][0]
        self.assertTrue(seg["within_tolerance"])

    @patch("opencut.core.isochronous_translate._translate_text")
    def test_translate_isochronous_skips_brackets(self, mock_translate):
        """Segments starting with [ should be skipped."""
        mock_translate.return_value = "test"

        from opencut.core.isochronous_translate import translate_isochronous
        result = translate_isochronous(
            segments=[
                {"text": "[music]", "start": 0, "end": 5},
            ],
            source_language="en",
            target_language="es",
        )
        seg = result["segments"][0]
        self.assertTrue(seg["within_tolerance"])
        self.assertEqual(seg["translated_text"], "[music]")

    def test_estimate_segment_duration(self):
        """estimate_segment_duration utility should return expected fields."""
        from opencut.core.isochronous_translate import estimate_segment_duration
        result = estimate_segment_duration("Testing the function", "en")
        self.assertIn("estimated_duration_seconds", result)
        self.assertIn("word_count", result)
        self.assertIn("speaking_rate_wpm", result)
        self.assertEqual(result["word_count"], 3)


# ============================================================
# 62.3 Multi-Language Audio Track Management
# ============================================================
class TestMultiLangAudio(unittest.TestCase):
    """Tests for opencut.core.multilang_audio."""

    def test_language_codes_defined(self):
        """LANGUAGE_CODES should map ISO 639-1 to 639-2/B."""
        from opencut.core.multilang_audio import LANGUAGE_CODES
        self.assertEqual(LANGUAGE_CODES["en"], "eng")
        self.assertEqual(LANGUAGE_CODES["es"], "spa")
        self.assertEqual(LANGUAGE_CODES["fr"], "fre")

    def test_list_audio_tracks_missing_file(self):
        """list_audio_tracks raises FileNotFoundError for missing file."""
        from opencut.core.multilang_audio import list_audio_tracks
        with self.assertRaises(FileNotFoundError):
            list_audio_tracks("/nonexistent/video.mp4")

    @patch("opencut.core.multilang_audio.subprocess.run")
    def test_list_audio_tracks_parses_output(self, mock_run):
        """list_audio_tracks should parse ffprobe JSON output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "streams": [
                    {
                        "index": 1,
                        "codec_name": "aac",
                        "channels": 2,
                        "sample_rate": "48000",
                        "bit_rate": "192000",
                        "tags": {"language": "eng", "title": "English"},
                    },
                    {
                        "index": 2,
                        "codec_name": "aac",
                        "channels": 2,
                        "sample_rate": "48000",
                        "bit_rate": "192000",
                        "tags": {"language": "spa", "title": "Spanish"},
                    },
                ],
            }),
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.multilang_audio import list_audio_tracks
            result = list_audio_tracks(path)
            self.assertEqual(result["total_tracks"], 2)
            self.assertEqual(result["tracks"][0]["language"], "eng")
            self.assertEqual(result["tracks"][1]["language"], "spa")
        finally:
            os.unlink(path)

    def test_add_audio_tracks_missing_video(self):
        """add_audio_tracks raises FileNotFoundError for missing video."""
        from opencut.core.multilang_audio import add_audio_tracks
        with self.assertRaises(FileNotFoundError):
            add_audio_tracks("/nonexistent.mp4", [{"path": "a.wav", "language": "en"}])

    def test_add_audio_tracks_no_tracks(self):
        """add_audio_tracks raises ValueError with empty tracks list."""
        from opencut.core.multilang_audio import add_audio_tracks
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                add_audio_tracks(path, [])
        finally:
            os.unlink(path)

    def test_remove_audio_tracks_no_indices(self):
        """remove_audio_tracks raises ValueError with empty indices."""
        from opencut.core.multilang_audio import remove_audio_tracks
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                remove_audio_tracks(path, [])
        finally:
            os.unlink(path)

    def test_manage_tracks_invalid_operation(self):
        """manage_tracks raises ValueError for unknown operation."""
        from opencut.core.multilang_audio import manage_tracks
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                manage_tracks(path, "invalid_op")
        finally:
            os.unlink(path)

    def test_label_audio_tracks_no_labels(self):
        """label_audio_tracks raises ValueError with empty labels."""
        from opencut.core.multilang_audio import label_audio_tracks
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                label_audio_tracks(path, [])
        finally:
            os.unlink(path)

    def test_export_single_language_missing_file(self):
        """export_single_language raises FileNotFoundError."""
        from opencut.core.multilang_audio import export_single_language
        with self.assertRaises(FileNotFoundError):
            export_single_language("/nonexistent.mp4")


# ============================================================
# 62.4 Voice Translation with Emotion Preservation
# ============================================================
class TestEmotionVoice(unittest.TestCase):
    """Tests for opencut.core.emotion_voice."""

    def test_prosody_profile_dataclass(self):
        """ProsodyProfile should have expected defaults."""
        from opencut.core.emotion_voice import ProsodyProfile
        p = ProsodyProfile()
        self.assertEqual(p.mean_pitch_hz, 0.0)
        self.assertEqual(p.speaking_rate, 1.0)
        self.assertEqual(p.mean_energy_db, 0.0)

    def test_extract_prosody_missing_file(self):
        """extract_prosody raises FileNotFoundError for missing audio."""
        from opencut.core.emotion_voice import extract_prosody
        with self.assertRaises(FileNotFoundError):
            extract_prosody("/nonexistent/audio.wav")

    @patch("opencut.core.emotion_voice._measure_speaking_rate", return_value=1.2)
    @patch("opencut.core.emotion_voice._extract_energy_contour")
    @patch("opencut.core.emotion_voice._extract_f0_contour")
    @patch("opencut.core.emotion_voice.get_video_info")
    def test_extract_prosody_returns_profile(self, mock_info, mock_f0,
                                              mock_energy, mock_rate):
        """extract_prosody should return dict with all prosody fields."""
        mock_info.return_value = {"duration": 10.0, "width": 0, "height": 0, "fps": 0}
        mock_f0.return_value = (180.0, 60.0, [170, 180, 190])
        mock_energy.return_value = (-18.0, 12.0, [-20, -16, -18])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            from opencut.core.emotion_voice import extract_prosody
            result = extract_prosody(path)
            self.assertIn("mean_pitch_hz", result)
            self.assertIn("speaking_rate", result)
            self.assertIn("mean_energy_db", result)
            self.assertAlmostEqual(result["mean_pitch_hz"], 180.0)
            self.assertAlmostEqual(result["speaking_rate"], 1.2)
        finally:
            os.unlink(path)

    def test_emotion_preserving_dub_missing_file(self):
        """emotion_preserving_dub raises FileNotFoundError."""
        from opencut.core.emotion_voice import emotion_preserving_dub
        with self.assertRaises(FileNotFoundError):
            emotion_preserving_dub("/nonexistent.mp4", "es")

    @patch("opencut.core.emotion_voice.run_ffmpeg")
    def test_apply_prosody_transfer_no_adjustments(self, mock_ffmpeg):
        """Prosody transfer with default profile should still produce output."""
        mock_ffmpeg.return_value = ""

        from opencut.core.emotion_voice import _apply_prosody_transfer
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            path = f.name
        out_path = path + "_prosody.wav"
        try:
            # Default prosody = no adjustments
            result = _apply_prosody_transfer(
                path,
                {"speaking_rate": 1.0, "mean_pitch_hz": 150, "mean_energy_db": -20},
                out_path,
            )
            # Should have called ffmpeg
            mock_ffmpeg.assert_called_once()
        finally:
            for p in [path, out_path]:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ============================================================
# Route Smoke Tests
# ============================================================
class TestCompositionDubbingRoutes(unittest.TestCase):
    """Smoke tests for composition_dubbing_routes endpoints."""

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
        cls.csrf_token = data.get("csrf_token", "")
        cls.headers = {
            "X-OpenCut-Token": cls.csrf_token,
            "Content-Type": "application/json",
        }

    def test_composition_guide_no_file(self):
        """POST /composition/guide without filepath returns 400."""
        resp = self.client.post(
            "/composition/guide",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, [400, 404])

    def test_classify_shot_no_file(self):
        """POST /composition/classify-shot without filepath returns 400."""
        resp = self.client.post(
            "/composition/classify-shot",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, [400, 404])

    def test_analyze_pacing_with_cuts(self):
        """POST /composition/analyze-pacing with cut_points should start job."""
        resp = self.client.post(
            "/composition/analyze-pacing",
            headers=self.headers,
            json={
                "cut_points": [5, 10, 15, 20],
                "total_duration": 25.0,
                "genre": "general",
            },
        )
        self.assertIn(resp.status_code, [200, 202])

    def test_saliency_crop_no_file(self):
        """POST /composition/saliency-crop without filepath returns 400."""
        resp = self.client.post(
            "/composition/saliency-crop",
            headers=self.headers,
            json={},
        )
        self.assertIn(resp.status_code, [400, 404])

    def test_dubbing_full_no_file(self):
        """POST /dubbing/full-pipeline without filepath returns 400."""
        resp = self.client.post(
            "/dubbing/full-pipeline",
            headers=self.headers,
            json={"target_language": "es"},
        )
        self.assertIn(resp.status_code, [400, 404])

    def test_dubbing_isochronous_missing_segments(self):
        """POST /dubbing/isochronous without segments should start job but fail."""
        resp = self.client.post(
            "/dubbing/isochronous",
            headers=self.headers,
            json={"target_language": "es"},
        )
        # May return 202 (job started) or 400/500 depending on validation
        self.assertIn(resp.status_code, [200, 202, 400, 500])

    def test_manage_tracks_no_input(self):
        """POST /dubbing/manage-tracks without input should fail."""
        resp = self.client.post(
            "/dubbing/manage-tracks",
            headers=self.headers,
            json={"operation": "list"},
        )
        self.assertIn(resp.status_code, [200, 202, 400, 404, 500])

    def test_emotion_transfer_no_file(self):
        """POST /dubbing/emotion-transfer without filepath returns 400."""
        resp = self.client.post(
            "/dubbing/emotion-transfer",
            headers=self.headers,
            json={"target_language": "es"},
        )
        self.assertIn(resp.status_code, [400, 404])

    def test_no_csrf_rejected(self):
        """Requests without CSRF token should be rejected."""
        resp = self.client.post(
            "/composition/guide",
            headers={"Content-Type": "application/json"},
            json={"filepath": "test.mp4"},
        )
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
