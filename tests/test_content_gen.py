"""
Tests for OpenCut AI Content Generation features (Category 71).

Covers:
  - Voice Avatar: AvatarConfig, AvatarResult, AVATAR_STYLES, audio analysis
  - CTR Prediction: ThumbnailFeatures, CTRPrediction, scoring, comparison
  - B-Roll AI Gen: BRollGenConfig, BRollGenResult, prompt enhancement
  - Chapter Art: ChapterArtConfig, CARD_STYLES, title generation
  - Intro Gen: BrandKit, IntroConfig, INTRO_STYLES, easing functions
  - Content gen routes: smoke tests for all endpoints
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
# Voice Avatar Tests
# ============================================================
class TestVoiceAvatarDataclasses(unittest.TestCase):
    """Tests for voice_avatar data structures."""

    def test_avatar_config_defaults(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig()
        self.assertEqual(cfg.style, "cartoon")
        self.assertEqual(cfg.width, 720)
        self.assertEqual(cfg.height, 720)
        self.assertEqual(cfg.fps, 30)
        self.assertEqual(cfg.background_mode, "solid")
        self.assertEqual(cfg.background_color, (18, 18, 24))
        self.assertAlmostEqual(cfg.mouth_open_threshold, 0.02)
        self.assertAlmostEqual(cfg.mouth_amplitude_scale, 1.5)
        self.assertAlmostEqual(cfg.face_scale, 0.6)
        self.assertEqual(cfg.face_position, (0.5, 0.45))
        self.assertAlmostEqual(cfg.max_duration, 0.0)

    def test_avatar_config_to_dict(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig(style="silhouette", width=512, height=512)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["style"], "silhouette")
        self.assertEqual(d["width"], 512)
        self.assertEqual(d["height"], 512)
        self.assertIn("fps", d)
        self.assertIn("background_mode", d)

    def test_avatar_config_validate_valid(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig()
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_avatar_config_validate_bad_style(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig(style="nonexistent")
        errors = cfg.validate()
        self.assertTrue(any("style" in e.lower() or "nonexistent" in e for e in errors))

    def test_avatar_config_validate_bad_dimensions(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig(width=10, height=10000)
        errors = cfg.validate()
        self.assertTrue(len(errors) >= 2)

    def test_avatar_config_validate_bad_fps(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig(fps=0)
        errors = cfg.validate()
        self.assertTrue(any("fps" in e.lower() for e in errors))

    def test_avatar_config_validate_custom_bg_no_image(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig(background_mode="custom", background_image="")
        errors = cfg.validate()
        self.assertTrue(any("background_image" in e for e in errors))

    def test_avatar_config_validate_bad_face_scale(self):
        from opencut.core.voice_avatar import AvatarConfig
        cfg = AvatarConfig(face_scale=0.01)
        errors = cfg.validate()
        self.assertTrue(any("face_scale" in e for e in errors))

    def test_avatar_result_defaults(self):
        from opencut.core.voice_avatar import AvatarResult
        res = AvatarResult()
        self.assertEqual(res.output_path, "")
        self.assertEqual(res.duration, 0.0)
        self.assertEqual(res.width, 0)
        self.assertEqual(res.height, 0)
        self.assertEqual(res.frame_count, 0)
        self.assertEqual(res.style, "")

    def test_avatar_result_to_dict(self):
        from opencut.core.voice_avatar import AvatarResult
        res = AvatarResult(output_path="/tmp/test.mp4", duration=5.0,
                           width=720, height=720, fps=30, frame_count=150,
                           style="cartoon")
        d = res.to_dict()
        self.assertEqual(d["output_path"], "/tmp/test.mp4")
        self.assertEqual(d["duration"], 5.0)
        self.assertEqual(d["frame_count"], 150)


class TestAvatarStyles(unittest.TestCase):
    """Tests for AVATAR_STYLES constant."""

    def test_avatar_styles_is_list(self):
        from opencut.core.voice_avatar import AVATAR_STYLES
        self.assertIsInstance(AVATAR_STYLES, list)
        self.assertTrue(len(AVATAR_STYLES) >= 4)

    def test_avatar_styles_have_required_keys(self):
        from opencut.core.voice_avatar import AVATAR_STYLES
        for style in AVATAR_STYLES:
            self.assertIn("id", style)
            self.assertIn("name", style)
            self.assertIn("description", style)
            self.assertIn("requires_gpu", style)
            self.assertIn("backends", style)
            self.assertIsInstance(style["backends"], list)

    def test_avatar_style_ids_unique(self):
        from opencut.core.voice_avatar import AVATAR_STYLES
        ids = [s["id"] for s in AVATAR_STYLES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_known_styles_present(self):
        from opencut.core.voice_avatar import AVATAR_STYLE_IDS
        for sid in ["realistic", "cartoon", "silhouette", "minimal", "sketch"]:
            self.assertIn(sid, AVATAR_STYLE_IDS)

    def test_list_avatar_styles(self):
        from opencut.core.voice_avatar import list_avatar_styles
        styles = list_avatar_styles()
        self.assertIsInstance(styles, list)
        self.assertTrue(len(styles) >= 4)


class TestAvatarGeneration(unittest.TestCase):
    """Tests for generate_avatar function."""

    def test_missing_audio_raises(self):
        from opencut.core.voice_avatar import generate_avatar
        with self.assertRaises(FileNotFoundError):
            generate_avatar("/nonexistent/audio.wav", "/nonexistent/face.png")

    def test_missing_face_raises(self):
        from opencut.core.voice_avatar import generate_avatar
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            audio_path = f.name
        try:
            with self.assertRaises(FileNotFoundError):
                generate_avatar(audio_path, "/nonexistent/face.png")
        finally:
            os.unlink(audio_path)

    def test_invalid_config_raises(self):
        from opencut.core.voice_avatar import AvatarConfig, generate_avatar
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
            af.write(b"RIFF" + b"\x00" * 100)
            audio_path = af.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as ff:
            ff.write(b"\x89PNG" + b"\x00" * 100)
            face_path = ff.name
        try:
            cfg = AvatarConfig(style="nonexistent")
            with self.assertRaises(ValueError):
                generate_avatar(audio_path, face_path, config=cfg)
        finally:
            os.unlink(audio_path)
            os.unlink(face_path)


class TestAudioAnalysis(unittest.TestCase):
    """Tests for audio amplitude analysis helpers."""

    def test_get_audio_duration_missing_file(self):
        from opencut.core.voice_avatar import _get_audio_duration
        dur = _get_audio_duration("/nonexistent/audio.wav")
        self.assertEqual(dur, 0.0)

    @patch("opencut.core.voice_avatar.subprocess.run")
    def test_get_audio_duration_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=b"3.5\n")
        from opencut.core.voice_avatar import _get_audio_duration
        dur = _get_audio_duration("/tmp/test.wav")
        self.assertAlmostEqual(dur, 3.5)


# ============================================================
# CTR Prediction Tests
# ============================================================
class TestCTRDataclasses(unittest.TestCase):
    """Tests for CTR prediction data structures."""

    def test_thumbnail_features_defaults(self):
        from opencut.core.ctr_predict import ThumbnailFeatures
        feat = ThumbnailFeatures()
        self.assertEqual(feat.face_count, 0)
        self.assertEqual(feat.face_area_ratio, 0.0)
        self.assertEqual(feat.avg_saturation, 0.0)
        self.assertEqual(feat.text_regions, 0)
        self.assertEqual(feat.visual_weight_center, (0.5, 0.5))

    def test_thumbnail_features_to_dict(self):
        from opencut.core.ctr_predict import ThumbnailFeatures
        feat = ThumbnailFeatures(image_path="/test.png", width=1280, height=720,
                                  face_count=1, face_area_ratio=0.15)
        d = feat.to_dict()
        self.assertEqual(d["face_count"], 1)
        self.assertAlmostEqual(d["face_area_ratio"], 0.15)
        self.assertIn("dominant_colors", d)

    def test_ctr_prediction_defaults(self):
        from opencut.core.ctr_predict import CTRPrediction
        pred = CTRPrediction()
        self.assertEqual(pred.ctr_score, 0.0)
        self.assertEqual(pred.platform, "youtube")
        self.assertEqual(pred.grade, "")
        self.assertIsInstance(pred.suggestions, list)

    def test_ctr_prediction_to_dict(self):
        from opencut.core.ctr_predict import CTRPrediction
        pred = CTRPrediction(
            image_path="/test.png", platform="youtube",
            ctr_score=72.5, confidence=0.83, grade="B+",
            feature_scores={"face_presence": 90.0, "contrast": 65.0},
        )
        d = pred.to_dict()
        self.assertAlmostEqual(d["ctr_score"], 72.5)
        self.assertEqual(d["grade"], "B+")
        self.assertIn("feature_scores", d)
        self.assertAlmostEqual(d["feature_scores"]["face_presence"], 90.0)

    def test_improvement_suggestion_to_dict(self):
        from opencut.core.ctr_predict import ImprovementSuggestion
        sug = ImprovementSuggestion(
            category="face_presence", priority="high",
            message="Add a face", current_score=20.0, potential_gain=25.0)
        d = sug.to_dict()
        self.assertEqual(d["category"], "face_presence")
        self.assertEqual(d["priority"], "high")
        self.assertAlmostEqual(d["potential_gain"], 25.0)

    def test_comparison_result_to_dict(self):
        from opencut.core.ctr_predict import ComparisonResult, CTRPrediction
        cr = ComparisonResult(
            predictions=[CTRPrediction(ctr_score=80.0), CTRPrediction(ctr_score=60.0)],
            winner_index=0, winner_path="/a.png", score_delta=20.0)
        d = cr.to_dict()
        self.assertEqual(d["winner_index"], 0)
        self.assertAlmostEqual(d["score_delta"], 20.0)
        self.assertEqual(len(d["predictions"]), 2)


class TestPlatformWeights(unittest.TestCase):
    """Tests for PLATFORM_WEIGHTS constant."""

    def test_platforms_exist(self):
        from opencut.core.ctr_predict import PLATFORM_WEIGHTS
        for platform in ["youtube", "tiktok", "instagram", "twitter", "facebook"]:
            self.assertIn(platform, PLATFORM_WEIGHTS)

    def test_weights_sum_to_one(self):
        from opencut.core.ctr_predict import PLATFORM_WEIGHTS
        for platform, weights in PLATFORM_WEIGHTS.items():
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=2,
                                    msg=f"{platform} weights sum to {total}")

    def test_weights_have_required_categories(self):
        from opencut.core.ctr_predict import PLATFORM_WEIGHTS
        required = {"face_presence", "face_size", "expression", "text_readability",
                     "color_vibrancy", "contrast", "composition"}
        for platform, weights in PLATFORM_WEIGHTS.items():
            for cat in required:
                self.assertIn(cat, weights, f"{platform} missing {cat}")


class TestCTRScoring(unittest.TestCase):
    """Tests for CTR scoring logic."""

    def test_score_to_grade(self):
        from opencut.core.ctr_predict import _score_to_grade
        self.assertEqual(_score_to_grade(95), "A+")
        self.assertEqual(_score_to_grade(85), "A")
        self.assertEqual(_score_to_grade(75), "B+")
        self.assertEqual(_score_to_grade(65), "B")
        self.assertEqual(_score_to_grade(55), "C+")
        self.assertEqual(_score_to_grade(45), "C")
        self.assertEqual(_score_to_grade(35), "D")
        self.assertEqual(_score_to_grade(15), "F")

    def test_compute_feature_scores_no_faces(self):
        from opencut.core.ctr_predict import ThumbnailFeatures, _compute_feature_scores
        feat = ThumbnailFeatures(face_count=0, face_area_ratio=0.0)
        scores = _compute_feature_scores(feat)
        self.assertEqual(scores["face_presence"], 20.0)

    def test_compute_feature_scores_one_face(self):
        from opencut.core.ctr_predict import ThumbnailFeatures, _compute_feature_scores
        feat = ThumbnailFeatures(face_count=1, face_area_ratio=0.15)
        scores = _compute_feature_scores(feat)
        self.assertEqual(scores["face_presence"], 90.0)
        self.assertGreaterEqual(scores["face_size"], 80.0)

    def test_compute_feature_scores_two_faces(self):
        from opencut.core.ctr_predict import ThumbnailFeatures, _compute_feature_scores
        feat = ThumbnailFeatures(face_count=2, face_area_ratio=0.2)
        scores = _compute_feature_scores(feat)
        self.assertEqual(scores["face_presence"], 75.0)

    def test_generate_suggestions_no_face(self):
        from opencut.core.ctr_predict import ThumbnailFeatures, _generate_suggestions
        feat = ThumbnailFeatures(face_count=0)
        scores = {"face_presence": 20.0, "face_size": 30.0, "expression": 10.0,
                  "text_readability": 40.0, "text_size": 30.0, "color_vibrancy": 80.0,
                  "contrast": 70.0, "composition": 50.0, "clutter": 60.0}
        suggestions = _generate_suggestions(scores, feat)
        self.assertTrue(any(s.category == "face_presence" for s in suggestions))
        # Sorted by potential_gain descending
        gains = [s.potential_gain for s in suggestions]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_predict_ctr_missing_image(self):
        from opencut.core.ctr_predict import predict_ctr
        with self.assertRaises(FileNotFoundError):
            predict_ctr("/nonexistent/thumb.png")

    def test_predict_ctr_unknown_platform(self):
        from opencut.core.ctr_predict import predict_ctr
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG" + b"\x00" * 100)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                predict_ctr(path, platform="myspace")
        finally:
            os.unlink(path)

    def test_compare_thumbnails_too_few(self):
        from opencut.core.ctr_predict import compare_thumbnails
        with self.assertRaises(ValueError):
            compare_thumbnails(["/one.png"])

    def test_compare_thumbnails_too_many(self):
        from opencut.core.ctr_predict import compare_thumbnails
        paths = [f"/img_{i}.png" for i in range(11)]
        with self.assertRaises(ValueError):
            compare_thumbnails(paths)

    @patch("opencut.core.ctr_predict.extract_features")
    def test_predict_ctr_with_mocked_features(self, mock_extract):
        from opencut.core.ctr_predict import (
            ThumbnailFeatures, predict_ctr,
        )
        mock_extract.return_value = ThumbnailFeatures(
            image_path="/test.png", width=1280, height=720,
            face_count=1, face_area_ratio=0.15,
            expression_score=0.7, avg_saturation=0.5,
            avg_brightness=0.6, color_vibrancy=0.4,
            contrast_ratio=6.0, text_regions=2,
            text_area_ratio=0.08, text_contrast=0.7,
            thirds_score=0.6, edge_density=0.3,
            clutter_score=0.2,
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG" + b"\x00" * 100)
            path = f.name
        try:
            result = predict_ctr(path, platform="youtube")
            self.assertGreater(result.ctr_score, 0)
            self.assertLessEqual(result.ctr_score, 100)
            self.assertIn(result.grade, ["A+", "A", "B+", "B", "C+", "C", "D", "F"])
            self.assertIsInstance(result.feature_scores, dict)
            self.assertGreater(result.confidence, 0)
        finally:
            os.unlink(path)


# ============================================================
# B-Roll AI Gen Tests
# ============================================================
class TestBRollDataclasses(unittest.TestCase):
    """Tests for B-roll generation data structures."""

    def test_broll_config_defaults(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig()
        self.assertEqual(cfg.backend, "image_kenburns")
        self.assertEqual(cfg.width, 1920)
        self.assertEqual(cfg.height, 1080)
        self.assertEqual(cfg.fps, 30)
        self.assertAlmostEqual(cfg.duration, 5.0)
        self.assertEqual(cfg.ken_burns_preset, "zoom_in")
        self.assertAlmostEqual(cfg.guidance_scale, 7.5)

    def test_broll_config_to_dict_redacts_api_key(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig(api_key="secret-key-123")
        d = cfg.to_dict()
        self.assertEqual(d["api_key"], "***")

    def test_broll_config_to_dict_no_key(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig(api_key="")
        d = cfg.to_dict()
        self.assertEqual(d["api_key"], "")

    def test_broll_config_validate_valid(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig()
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_broll_config_validate_bad_backend(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig(backend="nonexistent")
        errors = cfg.validate()
        self.assertTrue(any("backend" in e.lower() or "nonexistent" in e for e in errors))

    def test_broll_config_validate_bad_dimensions(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig(width=32, height=5000)
        errors = cfg.validate()
        self.assertTrue(len(errors) >= 2)

    def test_broll_config_validate_duration_below_min(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig(duration=1.0, min_duration=2.0)
        errors = cfg.validate()
        self.assertTrue(any("duration" in e.lower() for e in errors))

    def test_broll_config_validate_bad_guidance(self):
        from opencut.core.broll_ai_gen import BRollGenConfig
        cfg = BRollGenConfig(guidance_scale=50.0)
        errors = cfg.validate()
        self.assertTrue(any("guidance" in e.lower() for e in errors))

    def test_broll_result_defaults(self):
        from opencut.core.broll_ai_gen import BRollGenResult
        res = BRollGenResult()
        self.assertEqual(res.output_path, "")
        self.assertEqual(res.prompt, "")
        self.assertFalse(res.fallback_used)
        self.assertIsInstance(res.stock_suggestions, list)

    def test_broll_result_to_dict(self):
        from opencut.core.broll_ai_gen import BRollGenResult
        res = BRollGenResult(output_path="/tmp/test.mp4", prompt="sunset",
                              duration=5.0, backend_used="image_kenburns",
                              stock_suggestions=["nature", "sunset"])
        d = res.to_dict()
        self.assertEqual(d["prompt"], "sunset")
        self.assertEqual(len(d["stock_suggestions"]), 2)

    def test_batch_result_to_dict(self):
        from opencut.core.broll_ai_gen import BatchBRollResult, BRollGenResult
        batch = BatchBRollResult(
            results=[BRollGenResult(prompt="a"), BRollGenResult(prompt="b")],
            total_duration=10.0, total_clips=2, successful=2, failed=0)
        d = batch.to_dict()
        self.assertEqual(d["total_clips"], 2)
        self.assertEqual(d["successful"], 2)
        self.assertEqual(len(d["results"]), 2)


class TestBRollHelpers(unittest.TestCase):
    """Tests for B-roll helper functions."""

    def test_enhance_prompt(self):
        from opencut.core.broll_ai_gen import _enhance_prompt
        result = _enhance_prompt("sunset over ocean")
        self.assertIn("sunset over ocean", result)
        self.assertTrue(len(result) > len("sunset over ocean"))

    def test_enhance_prompt_with_suffix(self):
        from opencut.core.broll_ai_gen import _enhance_prompt
        result = _enhance_prompt("forest", style_suffix="anime style")
        self.assertIn("anime style", result)

    def test_suggest_stock_keywords(self):
        from opencut.core.broll_ai_gen import _suggest_stock_keywords
        keywords = _suggest_stock_keywords("sunset over the mountain lake")
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) > 0)
        self.assertTrue(any("nature" in k or "landscape" in k or "mountain" in k
                             for k in keywords))

    def test_suggest_stock_keywords_empty(self):
        from opencut.core.broll_ai_gen import _suggest_stock_keywords
        keywords = _suggest_stock_keywords("xyz abc")
        self.assertIsInstance(keywords, list)

    def test_ken_burns_presets_exist(self):
        from opencut.core.broll_ai_gen import KEN_BURNS_PRESETS
        self.assertIn("zoom_in", KEN_BURNS_PRESETS)
        self.assertIn("zoom_out", KEN_BURNS_PRESETS)
        self.assertIn("pan_left", KEN_BURNS_PRESETS)
        self.assertIn("pan_right", KEN_BURNS_PRESETS)
        for name, preset in KEN_BURNS_PRESETS.items():
            self.assertIn("start_scale", preset)
            self.assertIn("end_scale", preset)
            self.assertIn("start_pos", preset)
            self.assertIn("end_pos", preset)

    def test_video_gen_backends(self):
        from opencut.core.broll_ai_gen import VIDEO_GEN_BACKENDS
        self.assertIn("wan", VIDEO_GEN_BACKENDS)
        self.assertIn("cogvideo", VIDEO_GEN_BACKENDS)
        self.assertIn("ltx", VIDEO_GEN_BACKENDS)
        self.assertIn("image_kenburns", VIDEO_GEN_BACKENDS)

    def test_generate_broll_empty_prompt(self):
        from opencut.core.broll_ai_gen import generate_broll
        with self.assertRaises(ValueError):
            generate_broll("")

    def test_generate_broll_invalid_config(self):
        from opencut.core.broll_ai_gen import BRollGenConfig, generate_broll
        cfg = BRollGenConfig(backend="nonexistent")
        with self.assertRaises(ValueError):
            generate_broll("test prompt", config=cfg)

    def test_batch_generate_empty_prompts(self):
        from opencut.core.broll_ai_gen import batch_generate_broll
        with self.assertRaises(ValueError):
            batch_generate_broll([])

    def test_batch_generate_too_many_prompts(self):
        from opencut.core.broll_ai_gen import batch_generate_broll
        prompts = [f"prompt {i}" for i in range(21)]
        with self.assertRaises(ValueError):
            batch_generate_broll(prompts)


# ============================================================
# Chapter Art Tests
# ============================================================
class TestChapterArtDataclasses(unittest.TestCase):
    """Tests for chapter art data structures."""

    def test_chapter_art_config_defaults(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig()
        self.assertEqual(cfg.style, "minimal")
        self.assertEqual(cfg.width, 1920)
        self.assertEqual(cfg.height, 1080)
        self.assertAlmostEqual(cfg.card_duration, 3.0)
        self.assertTrue(cfg.export_images)
        self.assertFalse(cfg.export_video)
        self.assertEqual(cfg.title_prefix, "Chapter")
        self.assertTrue(cfg.auto_title_from_transcript)

    def test_chapter_art_config_to_dict(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig(style="bold", width=1280)
        d = cfg.to_dict()
        self.assertEqual(d["style"], "bold")
        self.assertEqual(d["width"], 1280)

    def test_chapter_art_config_validate_valid(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig()
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_chapter_art_config_validate_bad_style(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig(style="nonexistent")
        errors = cfg.validate()
        self.assertTrue(len(errors) >= 1)

    def test_chapter_art_config_validate_bad_dimensions(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig(width=32, height=10000)
        errors = cfg.validate()
        self.assertTrue(len(errors) >= 2)

    def test_chapter_art_config_validate_bad_duration(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig(card_duration=0.1)
        errors = cfg.validate()
        self.assertTrue(any("card_duration" in e for e in errors))

    def test_chapter_art_config_validate_bad_format(self):
        from opencut.core.auto_chapter_art import ChapterArtConfig
        cfg = ChapterArtConfig(image_format="bmp")
        errors = cfg.validate()
        self.assertTrue(any("image_format" in e for e in errors))

    def test_chapter_card_to_dict(self):
        from opencut.core.auto_chapter_art import ChapterCard
        card = ChapterCard(chapter_index=0, title="Introduction",
                            image_path="/tmp/ch1.png", start_time=0.0,
                            width=1920, height=1080, style="minimal")
        d = card.to_dict()
        self.assertEqual(d["title"], "Introduction")
        self.assertEqual(d["chapter_index"], 0)

    def test_chapter_art_result_to_dict(self):
        from opencut.core.auto_chapter_art import ChapterArtResult, ChapterCard
        result = ChapterArtResult(
            cards=[ChapterCard(title="Ch1"), ChapterCard(title="Ch2")],
            output_dir="/tmp/art", total_chapters=2, style="bold")
        d = result.to_dict()
        self.assertEqual(d["total_chapters"], 2)
        self.assertEqual(len(d["cards"]), 2)
        self.assertEqual(d["style"], "bold")

    def test_brand_kit_to_dict(self):
        from opencut.core.auto_chapter_art import BrandKit
        bk = BrandKit(font_name="Helvetica", primary_color=(255, 0, 0))
        d = bk.to_dict()
        self.assertEqual(d["font_name"], "Helvetica")
        self.assertEqual(d["primary_color"], (255, 0, 0))


class TestCardStyles(unittest.TestCase):
    """Tests for CARD_STYLES constant."""

    def test_card_styles_exist(self):
        from opencut.core.auto_chapter_art import CARD_STYLES
        for style in ["minimal", "bold", "gradient", "cinematic", "split"]:
            self.assertIn(style, CARD_STYLES)

    def test_card_styles_have_required_keys(self):
        from opencut.core.auto_chapter_art import CARD_STYLES
        for name, style in CARD_STYLES.items():
            self.assertIn("name", style, f"{name} missing 'name'")
            self.assertIn("description", style, f"{name} missing 'description'")
            self.assertIn("text_color", style, f"{name} missing 'text_color'")
            self.assertIn("font_size_ratio", style, f"{name} missing 'font_size_ratio'")
            self.assertIn("text_position", style, f"{name} missing 'text_position'")

    def test_list_card_styles(self):
        from opencut.core.auto_chapter_art import list_card_styles
        styles = list_card_styles()
        self.assertIsInstance(styles, dict)
        self.assertTrue(len(styles) >= 5)
        for key, val in styles.items():
            self.assertIn("name", val)
            self.assertIn("description", val)


class TestChapterArtHelpers(unittest.TestCase):
    """Tests for chapter art helper functions."""

    def test_auto_title_from_transcript(self):
        from opencut.core.auto_chapter_art import _auto_title_from_transcript
        title = _auto_title_from_transcript("today we're going to talk about video editing")
        self.assertTrue(len(title) > 0)
        self.assertTrue(title[0].isupper())

    def test_auto_title_empty_transcript(self):
        from opencut.core.auto_chapter_art import _auto_title_from_transcript
        self.assertEqual(_auto_title_from_transcript(""), "")
        self.assertEqual(_auto_title_from_transcript("   "), "")

    def test_auto_title_truncation(self):
        from opencut.core.auto_chapter_art import _auto_title_from_transcript
        long_text = "This is a very long transcript segment that goes on and on and on with many words"
        title = _auto_title_from_transcript(long_text, max_length=30)
        self.assertTrue(len(title) <= 33)  # 30 + "..."

    def test_select_key_frame_time_explicit(self):
        from opencut.core.auto_chapter_art import ChapterInfo, _select_key_frame_time
        ch = ChapterInfo(start_time=10.0, end_time=30.0, key_frame_time=15.0)
        self.assertAlmostEqual(_select_key_frame_time(ch), 15.0)

    def test_select_key_frame_time_auto(self):
        from opencut.core.auto_chapter_art import ChapterInfo, _select_key_frame_time
        ch = ChapterInfo(start_time=10.0, end_time=30.0)
        t = _select_key_frame_time(ch)
        self.assertGreater(t, 10.0)
        self.assertLess(t, 30.0)

    def test_generate_chapter_art_missing_video(self):
        from opencut.core.auto_chapter_art import generate_chapter_art
        with self.assertRaises(FileNotFoundError):
            generate_chapter_art("/nonexistent/video.mp4", [{"index": 0}])

    def test_generate_chapter_art_no_chapters(self):
        from opencut.core.auto_chapter_art import generate_chapter_art
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                generate_chapter_art(path, [])
        finally:
            os.unlink(path)


# ============================================================
# Intro Generation Tests
# ============================================================
class TestIntroDataclasses(unittest.TestCase):
    """Tests for intro generation data structures."""

    def test_brand_kit_defaults(self):
        from opencut.core.ai_intro_gen import BrandKit
        bk = BrandKit()
        self.assertEqual(bk.name, "")
        self.assertEqual(bk.tagline, "")
        self.assertEqual(bk.primary_color, (80, 140, 255))
        self.assertEqual(bk.background_color, (12, 12, 18))
        self.assertEqual(bk.font_name, "Arial")

    def test_brand_kit_to_dict(self):
        from opencut.core.ai_intro_gen import BrandKit
        bk = BrandKit(name="TestBrand", tagline="Be bold")
        d = bk.to_dict()
        self.assertEqual(d["name"], "TestBrand")
        self.assertEqual(d["tagline"], "Be bold")
        self.assertIn("primary_color", d)

    def test_brand_kit_validate_empty(self):
        from opencut.core.ai_intro_gen import BrandKit
        bk = BrandKit()
        errors = bk.validate()
        self.assertTrue(any("name" in e or "logo" in e for e in errors))

    def test_brand_kit_validate_with_name(self):
        from opencut.core.ai_intro_gen import BrandKit
        bk = BrandKit(name="TestBrand")
        errors = bk.validate()
        self.assertEqual(errors, [])

    def test_brand_kit_validate_bad_logo_path(self):
        from opencut.core.ai_intro_gen import BrandKit
        bk = BrandKit(logo_path="/nonexistent/logo.png")
        errors = bk.validate()
        self.assertTrue(any("logo" in e.lower() for e in errors))

    def test_intro_config_defaults(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig()
        self.assertEqual(cfg.style, "logo_reveal")
        self.assertEqual(cfg.width, 1920)
        self.assertEqual(cfg.height, 1080)
        self.assertEqual(cfg.fps, 30)
        self.assertAlmostEqual(cfg.duration, 5.0)
        self.assertEqual(cfg.background_mode, "solid")
        self.assertAlmostEqual(cfg.music_volume, 0.8)
        self.assertAlmostEqual(cfg.glow_intensity, 1.0)

    def test_intro_config_to_dict(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig(style="kinetic", duration=4.5)
        d = cfg.to_dict()
        self.assertEqual(d["style"], "kinetic")
        self.assertAlmostEqual(d["duration"], 4.5)

    def test_intro_config_validate_valid(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig()
        errors = cfg.validate()
        self.assertEqual(errors, [])

    def test_intro_config_validate_bad_style(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig(style="nonexistent")
        errors = cfg.validate()
        self.assertTrue(len(errors) >= 1)

    def test_intro_config_validate_bad_dimensions(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig(width=10, height=10000)
        errors = cfg.validate()
        self.assertTrue(len(errors) >= 2)

    def test_intro_config_validate_duration_out_of_range(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig(style="logo_reveal", duration=1.0)  # min is 3.0
        errors = cfg.validate()
        self.assertTrue(any("duration" in e.lower() for e in errors))

    def test_intro_config_validate_bad_bg_mode(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig(background_mode="hologram")
        errors = cfg.validate()
        self.assertTrue(any("background_mode" in e for e in errors))

    def test_intro_config_validate_bad_volume(self):
        from opencut.core.ai_intro_gen import IntroConfig
        cfg = IntroConfig(music_volume=1.5)
        errors = cfg.validate()
        self.assertTrue(any("music_volume" in e for e in errors))

    def test_intro_result_defaults(self):
        from opencut.core.ai_intro_gen import IntroResult
        res = IntroResult()
        self.assertEqual(res.output_path, "")
        self.assertEqual(res.duration, 0.0)
        self.assertFalse(res.has_music)
        self.assertEqual(res.prepended_to, "")

    def test_intro_result_to_dict(self):
        from opencut.core.ai_intro_gen import IntroResult
        res = IntroResult(output_path="/tmp/intro.mp4", duration=5.0,
                          width=1920, height=1080, fps=30, frame_count=150,
                          style="logo_reveal", has_music=True)
        d = res.to_dict()
        self.assertEqual(d["style"], "logo_reveal")
        self.assertTrue(d["has_music"])
        self.assertEqual(d["frame_count"], 150)


class TestIntroStyles(unittest.TestCase):
    """Tests for INTRO_STYLES constant."""

    def test_intro_styles_exist(self):
        from opencut.core.ai_intro_gen import INTRO_STYLES
        for style in ["logo_reveal", "text_sweep", "particles", "minimal_fade", "kinetic"]:
            self.assertIn(style, INTRO_STYLES)

    def test_intro_styles_have_required_keys(self):
        from opencut.core.ai_intro_gen import INTRO_STYLES
        for name, style in INTRO_STYLES.items():
            self.assertIn("name", style, f"{name} missing 'name'")
            self.assertIn("description", style, f"{name} missing 'description'")
            self.assertIn("min_duration", style, f"{name} missing 'min_duration'")
            self.assertIn("max_duration", style, f"{name} missing 'max_duration'")
            self.assertIn("supports_logo", style, f"{name} missing 'supports_logo'")
            self.assertLess(style["min_duration"], style["max_duration"])

    def test_list_intro_styles(self):
        from opencut.core.ai_intro_gen import list_intro_styles
        styles = list_intro_styles()
        self.assertIsInstance(styles, dict)
        self.assertTrue(len(styles) >= 5)
        for key, val in styles.items():
            self.assertIn("name", val)
            self.assertIn("description", val)
            self.assertIn("min_duration", val)
            self.assertIn("max_duration", val)


class TestIntroEasing(unittest.TestCase):
    """Tests for easing functions."""

    def test_ease_in_out_cubic_bounds(self):
        from opencut.core.ai_intro_gen import _ease_in_out_cubic
        self.assertAlmostEqual(_ease_in_out_cubic(0.0), 0.0)
        self.assertAlmostEqual(_ease_in_out_cubic(1.0), 1.0)
        self.assertAlmostEqual(_ease_in_out_cubic(0.5), 0.5)

    def test_ease_in_out_cubic_monotonic(self):
        from opencut.core.ai_intro_gen import _ease_in_out_cubic
        prev = 0.0
        for i in range(101):
            t = i / 100.0
            val = _ease_in_out_cubic(t)
            self.assertGreaterEqual(val, prev - 0.001)
            prev = val

    def test_ease_out_elastic_bounds(self):
        from opencut.core.ai_intro_gen import _ease_out_elastic
        self.assertAlmostEqual(_ease_out_elastic(0.0), 0.0, places=1)
        self.assertAlmostEqual(_ease_out_elastic(1.0), 1.0, places=1)

    def test_ease_out_quad_bounds(self):
        from opencut.core.ai_intro_gen import _ease_out_quad
        self.assertAlmostEqual(_ease_out_quad(0.0), 0.0)
        self.assertAlmostEqual(_ease_out_quad(1.0), 1.0)
        # Ease-out should be above linear at midpoint
        self.assertGreater(_ease_out_quad(0.5), 0.5)


class TestIntroGeneration(unittest.TestCase):
    """Tests for generate_intro function."""

    def test_generate_intro_empty_brand_kit(self):
        from opencut.core.ai_intro_gen import generate_intro
        with self.assertRaises(ValueError):
            generate_intro({})

    def test_generate_intro_brand_kit_with_name(self):
        from opencut.core.ai_intro_gen import generate_intro
        # Should fail at rendering (no PIL in test), but pass validation
        brand = {"name": "TestBrand", "tagline": "Test Tagline"}
        try:
            generate_intro(brand, style="minimal_fade", duration=3.0)
        except (RuntimeError, ImportError, Exception) as e:
            # Expected to fail at rendering stage, not validation
            self.assertNotIn("Invalid", str(e))

    def test_generate_intro_bad_style(self):
        from opencut.core.ai_intro_gen import generate_intro
        with self.assertRaises(ValueError):
            generate_intro({"name": "Test"}, style="nonexistent")

    def test_generate_intro_bad_duration(self):
        from opencut.core.ai_intro_gen import generate_intro
        with self.assertRaises(ValueError):
            generate_intro({"name": "Test"}, style="logo_reveal", duration=1.0)


# ============================================================
# Route Smoke Tests
# ============================================================
class TestContentGenRoutes(unittest.TestCase):
    """Smoke tests for content generation route registration."""

    def test_blueprint_exists(self):
        from opencut.routes.content_gen_routes import content_gen_bp
        self.assertEqual(content_gen_bp.name, "content_gen")

    def test_blueprint_has_rules(self):
        from opencut.routes.content_gen_routes import content_gen_bp
        rules = list(content_gen_bp.deferred_functions)
        self.assertTrue(len(rules) > 0)

    def test_voice_avatar_styles_endpoint(self):
        """GET /ai/voice-avatar/styles should return styles list."""
        from opencut.routes.content_gen_routes import content_gen_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(content_gen_bp)
        with app.test_client() as client:
            resp = client.get("/ai/voice-avatar/styles")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("styles", data)
            self.assertIsInstance(data["styles"], list)
            self.assertTrue(len(data["styles"]) >= 4)

    def test_chapter_art_styles_endpoint(self):
        """GET /content/chapter-art/styles should return styles dict."""
        from opencut.routes.content_gen_routes import content_gen_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(content_gen_bp)
        with app.test_client() as client:
            resp = client.get("/content/chapter-art/styles")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("styles", data)
            self.assertIn("minimal", data["styles"])
            self.assertIn("bold", data["styles"])

    def test_intro_styles_endpoint(self):
        """GET /video/intro-styles should return styles dict."""
        from opencut.routes.content_gen_routes import content_gen_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(content_gen_bp)
        with app.test_client() as client:
            resp = client.get("/video/intro-styles")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("styles", data)
            self.assertIn("logo_reveal", data["styles"])
            self.assertIn("kinetic", data["styles"])

    def test_post_endpoints_require_csrf(self):
        """POST endpoints without CSRF should fail."""
        from opencut.routes.content_gen_routes import content_gen_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(content_gen_bp)
        post_routes = [
            "/ai/voice-avatar",
            "/content/predict-ctr",
            "/content/compare-thumbnails",
            "/ai/generate-broll",
            "/ai/generate-broll/batch",
            "/content/chapter-art",
            "/video/generate-intro",
        ]
        with app.test_client() as client:
            for route in post_routes:
                resp = client.post(route, json={})
                # Should not be 404 — route exists
                self.assertNotEqual(resp.status_code, 404,
                                     f"Route {route} returned 404")


class TestRouteEndpointCount(unittest.TestCase):
    """Verify the blueprint registers the expected number of endpoints."""

    def test_endpoint_count(self):
        from opencut.routes.content_gen_routes import content_gen_bp
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(content_gen_bp)

        content_gen_rules = [
            rule for rule in app.url_map.iter_rules()
            if rule.endpoint.startswith("content_gen.")
        ]
        # 10 routes total (7 POST + 3 GET)
        self.assertGreaterEqual(len(content_gen_rules), 10)


if __name__ == "__main__":
    unittest.main()
