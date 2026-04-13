"""
Tests for OpenCut AI content features.

Covers:
  - Auto color grading (mood presets, reference, LUT)
  - Content moderation scanner
  - Pacing & rhythm analysis
  - SEO optimizer (heuristic + title scoring)
  - Engagement prediction
  - Context-aware command suggestions
  - AI content routes (smoke tests)
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Auto Color Grading
# ============================================================
class TestAutoColor(unittest.TestCase):
    """Tests for opencut.core.auto_color."""

    def test_list_mood_presets(self):
        """list_mood_presets should return a non-empty sorted list."""
        from opencut.core.auto_color import list_mood_presets
        presets = list_mood_presets()
        self.assertIsInstance(presets, list)
        self.assertTrue(len(presets) >= 5)
        self.assertEqual(presets, sorted(presets))

    def test_mood_presets_contain_expected(self):
        """MOOD_PRESETS should contain the required moods."""
        from opencut.core.auto_color import MOOD_PRESETS
        expected = {"warm sunset", "teal orange", "horror", "noir", "vintage"}
        for mood in expected:
            self.assertIn(mood, MOOD_PRESETS, f"Missing mood: {mood}")
            self.assertIn("filters", MOOD_PRESETS[mood])
            self.assertIn("description", MOOD_PRESETS[mood])

    def test_mood_presets_filters_are_strings(self):
        """Each mood preset filter chain should be a non-empty string."""
        from opencut.core.auto_color import MOOD_PRESETS
        for name, preset in MOOD_PRESETS.items():
            self.assertIsInstance(preset["filters"], str, f"{name} filters not a string")
            self.assertTrue(len(preset["filters"]) > 10, f"{name} filters too short")

    def test_auto_grade_no_mode_raises(self):
        """auto_grade should raise ValueError when no mode is specified."""
        from opencut.core.auto_color import auto_grade
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                auto_grade(path)
        finally:
            os.unlink(path)

    def test_auto_grade_missing_file(self):
        """auto_grade should raise FileNotFoundError for missing file."""
        from opencut.core.auto_color import auto_grade
        with self.assertRaises(FileNotFoundError):
            auto_grade("/nonexistent/video.mp4", mood="noir")

    def test_auto_grade_invalid_mood(self):
        """auto_grade should raise ValueError for unknown mood."""
        from opencut.core.auto_color import auto_grade
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                auto_grade(path, mood="nonexistent_mood")
        finally:
            os.unlink(path)

    @patch("opencut.core.auto_color.run_ffmpeg")
    def test_auto_grade_mood_calls_ffmpeg(self, mock_ffmpeg):
        """auto_grade with mood should call run_ffmpeg with filter chain."""
        from opencut.core.auto_color import auto_grade
        mock_ffmpeg.return_value = ""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = auto_grade(path, mood="noir")
            mock_ffmpeg.assert_called_once()
            self.assertEqual(result["grading_method"], "mood")
            self.assertEqual(result["mood"], "noir")
            self.assertIn("output_path", result)
        finally:
            os.unlink(path)

    @patch("opencut.core.auto_color.run_ffmpeg")
    def test_auto_grade_reference_calls_ffmpeg(self, mock_ffmpeg):
        """auto_grade with reference_image should produce reference grading."""
        from opencut.core.auto_color import auto_grade
        mock_ffmpeg.return_value = ""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video")
            vid_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image")
            ref_path = f.name
        try:
            with patch("opencut.core.auto_color._extract_color_stats") as mock_stats:
                mock_stats.return_value = {"hueavg": 140, "satavg": 130, "yavg": 150}
                result = auto_grade(vid_path, reference_image=ref_path)
            self.assertEqual(result["grading_method"], "reference")
            self.assertIn("adjustments", result)
        finally:
            os.unlink(vid_path)
            os.unlink(ref_path)


# ============================================================
# Content Moderation
# ============================================================
class TestContentModeration(unittest.TestCase):
    """Tests for opencut.core.content_moderation."""

    def test_profanity_words_loaded(self):
        """PROFANITY_WORDS should be a non-empty list."""
        from opencut.core.content_moderation import PROFANITY_WORDS
        self.assertIsInstance(PROFANITY_WORDS, list)
        self.assertTrue(len(PROFANITY_WORDS) >= 5)

    def test_moderation_issue_dataclass(self):
        """ModerationIssue should hold expected fields."""
        from opencut.core.content_moderation import ModerationIssue
        issue = ModerationIssue(
            timestamp=5.0, type="profanity", severity="medium",
            description="Test issue"
        )
        self.assertEqual(issue.type, "profanity")
        self.assertEqual(issue.severity, "medium")

    def test_scan_content_missing_file(self):
        """scan_content should raise FileNotFoundError."""
        from opencut.core.content_moderation import scan_content
        with self.assertRaises(FileNotFoundError):
            scan_content("/nonexistent/video.mp4")

    @patch("opencut.core.content_moderation.get_video_info")
    @patch("opencut.core.content_moderation._check_silence")
    @patch("opencut.core.content_moderation._check_loudness")
    @patch("opencut.core.content_moderation._check_flash")
    @patch("opencut.core.content_moderation._check_profanity")
    def test_scan_content_all_checks(self, mock_prof, mock_flash, mock_loud, mock_silence, mock_info):
        """scan_content should run all checks and return proper structure."""
        from opencut.core.content_moderation import ModerationIssue, scan_content
        mock_info.return_value = {"duration": 60.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_prof.return_value = [
            ModerationIssue(timestamp=5.0, type="profanity", severity="medium", description="bad word")
        ]
        mock_flash.return_value = []
        mock_loud.return_value = []
        mock_silence.return_value = []

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = scan_content(path)
            self.assertIn("issues", result)
            self.assertIn("overall_risk", result)
            self.assertIn("checks_performed", result)
            self.assertEqual(len(result["checks_performed"]), 4)
            self.assertEqual(result["total_issues"], 1)
        finally:
            os.unlink(path)

    @patch("opencut.core.content_moderation.get_video_info")
    @patch("opencut.core.content_moderation._check_loudness")
    def test_scan_content_specific_checks(self, mock_loud, mock_info):
        """scan_content should only run requested checks."""
        from opencut.core.content_moderation import scan_content
        mock_info.return_value = {"duration": 30.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_loud.return_value = []

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = scan_content(path, checks=["loudness"])
            self.assertEqual(result["checks_performed"], ["loudness"])
        finally:
            os.unlink(path)

    def test_profanity_detection_in_srt(self):
        """_check_profanity should find profanity in SRT-style transcript."""
        from opencut.core.content_moderation import _check_profanity
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            vid_path = f.name
        srt_path = vid_path.replace(".mp4", ".srt")
        with open(srt_path, "w") as f:
            f.write("1\n00:00:05,000 --> 00:00:08,000\nWhat the fuck is going on\n\n")
            f.write("2\n00:00:10,000 --> 00:00:12,000\nThis is fine\n")
        try:
            issues = _check_profanity(vid_path, transcript_path=None)
            self.assertTrue(len(issues) >= 1)
            self.assertEqual(issues[0].type, "profanity")
        finally:
            os.unlink(vid_path)
            if os.path.exists(srt_path):
                os.unlink(srt_path)


# ============================================================
# Pacing & Rhythm Analysis
# ============================================================
class TestPacingAnalysis(unittest.TestCase):
    """Tests for opencut.core.pacing_analysis."""

    def test_genre_profiles_exist(self):
        """GENRE_PROFILES should contain expected genres."""
        from opencut.core.pacing_analysis import GENRE_PROFILES
        expected = {"general", "trailer", "interview", "documentary"}
        for genre in expected:
            self.assertIn(genre, GENRE_PROFILES)
            self.assertIn("target_cpm", GENRE_PROFILES[genre])
            self.assertIn("target_avg", GENRE_PROFILES[genre])

    def test_analyze_pacing_missing_file(self):
        """analyze_pacing should raise FileNotFoundError."""
        from opencut.core.pacing_analysis import analyze_pacing
        with self.assertRaises(FileNotFoundError):
            analyze_pacing("/nonexistent/video.mp4")

    @patch("opencut.core.scene_detect.detect_scenes")
    @patch("opencut.core.pacing_analysis.get_video_info")
    def test_analyze_pacing_returns_structure(self, mock_info, mock_scenes):
        """analyze_pacing should return a complete result dict."""
        from opencut.core.pacing_analysis import analyze_pacing
        from opencut.core.scene_detect import SceneBoundary, SceneInfo

        mock_info.return_value = {"duration": 120.0, "width": 1920, "height": 1080, "fps": 30.0}
        mock_scenes.return_value = SceneInfo(
            boundaries=[
                SceneBoundary(time=0.0, frame=0, score=1.0, label="Start"),
                SceneBoundary(time=10.0, frame=300, score=0.5),
                SceneBoundary(time=25.0, frame=750, score=0.4),
                SceneBoundary(time=45.0, frame=1350, score=0.6),
                SceneBoundary(time=70.0, frame=2100, score=0.35),
                SceneBoundary(time=100.0, frame=3000, score=0.45),
            ],
            total_scenes=6,
            duration=120.0,
            avg_scene_length=20.0,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = analyze_pacing(path, genre="documentary")
            self.assertIn("cuts_per_minute", result)
            self.assertIn("avg_shot_duration", result)
            self.assertIn("shot_duration_distribution", result)
            self.assertIn("pacing_curve", result)
            self.assertIn("suggestions", result)
            self.assertIn("shots", result)
            self.assertEqual(result["genre"], "documentary")
            self.assertEqual(result["total_shots"], 6)
            self.assertGreater(result["cuts_per_minute"], 0)
        finally:
            os.unlink(path)

    def test_pacing_curve_computation(self):
        """_compute_pacing_curve should return non-empty curve for valid shots."""
        from opencut.core.pacing_analysis import ShotInfo, _compute_pacing_curve
        shots = [
            ShotInfo(index=1, start=0.0, end=5.0, duration=5.0),
            ShotInfo(index=2, start=5.0, end=12.0, duration=7.0),
            ShotInfo(index=3, start=12.0, end=20.0, duration=8.0),
        ]
        curve = _compute_pacing_curve(shots, 20.0, window_seconds=10.0)
        self.assertIsInstance(curve, list)
        self.assertTrue(len(curve) > 0)
        for point in curve:
            self.assertIn("time", point)
            self.assertIn("avg_shot_duration", point)
            self.assertIn("local_cpm", point)

    def test_generate_suggestions_slow_pacing(self):
        """_generate_suggestions should flag slow pacing for trailers."""
        from opencut.core.pacing_analysis import GENRE_PROFILES, ShotInfo, _generate_suggestions
        shots = [
            ShotInfo(index=i, start=(i - 1) * 15.0, end=i * 15.0, duration=15.0)
            for i in range(1, 5)
        ]
        profile = GENRE_PROFILES["trailer"]
        suggestions = _generate_suggestions(
            shots, cpm=4.0, avg_shot=15.0, profile=profile, genre="trailer", duration=60.0
        )
        self.assertTrue(len(suggestions) > 0)
        self.assertTrue(any("slow" in s.lower() for s in suggestions))


# ============================================================
# SEO Optimizer
# ============================================================
class TestSEOOptimizer(unittest.TestCase):
    """Tests for opencut.core.seo_optimizer."""

    def test_optimize_seo_empty_transcript_raises(self):
        """optimize_seo should raise ValueError for empty transcript."""
        from opencut.core.seo_optimizer import optimize_seo
        with self.assertRaises(ValueError):
            optimize_seo("")

    def test_optimize_seo_heuristic_fallback(self):
        """optimize_seo should fall back to heuristic when LLM unavailable."""
        from opencut.core.seo_optimizer import optimize_seo
        transcript = (
            "Today we're going to learn about video editing techniques. "
            "Color grading is essential for creating a cinematic look. "
            "We'll cover transitions, effects, and audio mixing. "
            "Professional video editors use these techniques daily."
        )
        with patch("opencut.core.seo_optimizer._try_llm_optimization", return_value=None):
            result = optimize_seo(transcript, platform="youtube")

        self.assertIn("title_options", result)
        self.assertIn("description", result)
        self.assertIn("tags", result)
        self.assertIn("hashtags", result)
        self.assertIn("optimal_posting_time", result)
        self.assertEqual(result["platform"], "youtube")
        self.assertEqual(result["method"], "heuristic")
        self.assertTrue(len(result["title_options"]) >= 3)
        self.assertTrue(len(result["tags"]) >= 3)

    def test_score_title_good(self):
        """score_title should give high score to a well-crafted title."""
        from opencut.core.seo_optimizer import score_title
        result = score_title("Top 10 Video Editing Tips for Beginners in 2024")
        self.assertTrue(result["length_ok"])
        self.assertTrue(result["has_number"])
        self.assertTrue(result["has_power_word"])
        self.assertIn(result["estimated_ctr_tier"], ("medium", "high"))

    def test_score_title_short(self):
        """score_title should flag a very short title."""
        from opencut.core.seo_optimizer import score_title
        result = score_title("Hi")
        self.assertTrue(result["length_ok"])
        self.assertFalse(result["ideal_length"])
        self.assertEqual(result["keyword_count"], 0)

    def test_keyword_extraction(self):
        """_extract_keywords should extract meaningful keywords."""
        from opencut.core.seo_optimizer import _extract_keywords
        text = (
            "Machine learning and artificial intelligence are transforming "
            "video editing. Deep learning models can detect scenes automatically. "
            "Machine learning algorithms improve color grading."
        )
        keywords = _extract_keywords(text, max_keywords=10)
        self.assertTrue(len(keywords) > 0)
        kw_words = [kw for kw, _ in keywords]
        self.assertTrue(any("learning" in kw for kw in kw_words))

    def test_platform_limits_exist(self):
        """PLATFORM_LIMITS should cover expected platforms."""
        from opencut.core.seo_optimizer import PLATFORM_LIMITS
        for platform in ("youtube", "tiktok", "instagram", "twitter"):
            self.assertIn(platform, PLATFORM_LIMITS)


# ============================================================
# Engagement Prediction
# ============================================================
class TestEngagementPredict(unittest.TestCase):
    """Tests for opencut.core.engagement_predict."""

    def test_predict_missing_file(self):
        """predict_engagement should raise FileNotFoundError."""
        from opencut.core.engagement_predict import predict_engagement
        with self.assertRaises(FileNotFoundError):
            predict_engagement("/nonexistent/video.mp4")

    @patch("opencut.core.engagement_predict._measure_scene_change_rate")
    @patch("opencut.core.engagement_predict._build_energy_curve")
    @patch("opencut.core.engagement_predict._measure_visual_change")
    @patch("opencut.core.engagement_predict._measure_audio_energy")
    @patch("opencut.core.engagement_predict.get_video_info")
    def test_predict_engagement_structure(self, mock_info, mock_audio,
                                          mock_visual, mock_energy,
                                          mock_scene_rate):
        """predict_engagement should return complete result structure."""
        from opencut.core.engagement_predict import predict_engagement
        mock_info.return_value = {"duration": 60.0, "fps": 30.0, "width": 1920, "height": 1080}
        mock_audio.return_value = 55.0
        mock_visual.return_value = 40.0
        mock_energy.return_value = [
            {"time": 0.0, "energy": 50.0},
            {"time": 10.0, "energy": 45.0},
            {"time": 20.0, "energy": 55.0},
            {"time": 30.0, "energy": 40.0},
        ]
        mock_scene_rate.return_value = 12.0

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = predict_engagement(path, transcript_text="Hello world testing speech pace")
            self.assertIn("hook_score", result)
            self.assertIn("retention_curve", result)
            self.assertIn("virality_score", result)
            self.assertIn("overall_score", result)
            self.assertIn("suggestions", result)
            self.assertIn("analysis", result)
            self.assertIsInstance(result["hook_score"], int)
            self.assertTrue(0 <= result["hook_score"] <= 100)
            self.assertTrue(0 <= result["virality_score"] <= 100)
            self.assertTrue(len(result["retention_curve"]) > 0)
        finally:
            os.unlink(path)

    def test_speech_pace_analysis(self):
        """_analyze_speech_pace should calculate WPM from transcript."""
        from opencut.core.engagement_predict import _analyze_speech_pace
        transcript = "This is a test with exactly ten words in it"
        wpm = _analyze_speech_pace(transcript, duration=60.0)
        self.assertIsNotNone(wpm)
        self.assertAlmostEqual(wpm, 10.0, delta=1.0)

    def test_speech_pace_empty(self):
        """_analyze_speech_pace should return None for empty transcript."""
        from opencut.core.engagement_predict import _analyze_speech_pace
        self.assertIsNone(_analyze_speech_pace(None, 60.0))
        self.assertIsNone(_analyze_speech_pace("", 60.0))
        self.assertIsNone(_analyze_speech_pace("hello", 0.0))

    def test_hook_score_computation(self):
        """_compute_hook_score should return 0-100 for valid inputs."""
        from opencut.core.engagement_predict import _compute_hook_score
        # High energy + high visual = high hook
        high = _compute_hook_score(80.0, 70.0, 60.0)
        self.assertTrue(40 <= high <= 100)
        # Low energy = low hook
        low = _compute_hook_score(10.0, 10.0, 60.0)
        self.assertTrue(0 <= low <= 50)

    def test_retention_curve_shape(self):
        """_build_retention_curve should produce a decaying curve."""
        from opencut.core.engagement_predict import _build_retention_curve
        energy = [{"time": 0.0, "energy": 50.0}, {"time": 30.0, "energy": 50.0}]
        curve = _build_retention_curve(energy, 10.0, 60.0)
        self.assertTrue(len(curve) >= 2)
        # First point should have higher retention than last
        self.assertGreater(curve[0]["predicted_retention_pct"],
                          curve[-1]["predicted_retention_pct"])


# ============================================================
# Context-Aware Suggestions
# ============================================================
class TestContextSuggest(unittest.TestCase):
    """Tests for opencut.core.context_suggest."""

    def test_loud_audio_suggestion(self):
        """Should suggest normalize when loudness > -10."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={"loudness_lufs": -5.0, "duration": 60, "has_audio": True},
            recent_actions=[],
        )
        actions = [s["action"] for s in result]
        self.assertIn("normalize_audio", actions)

    def test_quiet_audio_suggestion(self):
        """Should suggest normalize when loudness < -24."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={"loudness_lufs": -30.0, "duration": 60, "has_audio": True},
            recent_actions=[],
        )
        actions = [s["action"] for s in result]
        self.assertIn("normalize_audio", actions)

    def test_long_video_scene_detect(self):
        """Should suggest scene detect for videos > 300s."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={"duration": 600, "has_audio": True, "has_video": True},
            recent_actions=[],
        )
        actions = [s["action"] for s in result]
        self.assertIn("detect_scenes", actions)

    def test_captions_suggestion(self):
        """Should suggest captions when audio present and no recent captions."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={"duration": 30, "has_audio": True, "has_video": True},
            recent_actions=[],
        )
        actions = [s["action"] for s in result]
        self.assertIn("add_captions", actions)

    def test_recent_actions_suppress(self):
        """Recent actions should suppress duplicate suggestions."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={
                "duration": 600, "has_audio": True, "has_video": True,
                "loudness_lufs": -14.0,
            },
            recent_actions=["detect_scenes", "add_captions", "remove_silence"],
        )
        actions = [s["action"] for s in result]
        self.assertNotIn("detect_scenes", actions)
        self.assertNotIn("add_captions", actions)
        self.assertNotIn("remove_silence", actions)

    def test_max_three_suggestions(self):
        """Should return at most 3 suggestions."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={
                "duration": 700, "has_audio": True, "has_video": True,
                "loudness_lufs": -5.0, "resolution": 640, "width": 640, "height": 480,
            },
            recent_actions=[],
        )
        self.assertTrue(len(result) <= 3)

    def test_sorted_by_confidence(self):
        """Suggestions should be sorted by descending confidence."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={
                "duration": 700, "has_audio": True, "has_video": True,
                "loudness_lufs": -5.0,
            },
            recent_actions=[],
        )
        if len(result) >= 2:
            for i in range(len(result) - 1):
                self.assertGreaterEqual(result[i]["confidence"], result[i + 1]["confidence"])

    def test_suggestion_dict_keys(self):
        """Each suggestion should have action, description, confidence, reason."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={"duration": 60, "has_audio": True, "loudness_lufs": -5.0},
            recent_actions=[],
        )
        for s in result:
            self.assertIn("action", s)
            self.assertIn("description", s)
            self.assertIn("confidence", s)
            self.assertIn("reason", s)

    def test_high_res_no_upscale(self):
        """High resolution should suggest no upscale needed."""
        from opencut.core.context_suggest import get_suggestions
        result = get_suggestions(
            clip_metadata={
                "duration": 30, "has_audio": False, "has_video": True,
                "resolution": 3840, "width": 3840, "height": 2160,
            },
            recent_actions=[],
        )
        actions = [s["action"] for s in result]
        self.assertIn("no_upscale_needed", actions)


# ============================================================
# Route Smoke Tests
# ============================================================
class TestAIContentRoutes(unittest.TestCase):
    """Smoke tests for AI content route endpoints."""

    def setUp(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        self.app = create_app(config=test_config)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        # Fetch CSRF token
        resp = self.client.get("/health")
        data = resp.get_json()
        self.csrf_token = data.get("csrf_token", "")
        self.headers = {
            "X-OpenCut-Token": self.csrf_token,
            "Content-Type": "application/json",
        }

    def test_mood_presets_endpoint(self):
        """GET /ai/mood-presets should return presets list."""
        resp = self.client.get("/ai/mood-presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("presets", data)
        self.assertIn("count", data)
        self.assertTrue(data["count"] >= 5)

    def test_auto_grade_no_file(self):
        """POST /ai/auto-grade without filepath should return 400."""
        resp = self.client.post("/ai/auto-grade",
                                headers=self.headers,
                                data=json.dumps({"mood": "noir"}))
        self.assertEqual(resp.status_code, 400)

    def test_content_scan_no_file(self):
        """POST /ai/content-scan without filepath should return 400."""
        resp = self.client.post("/ai/content-scan",
                                headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    def test_pacing_analysis_no_file(self):
        """POST /ai/pacing-analysis without filepath should return 400."""
        resp = self.client.post("/ai/pacing-analysis",
                                headers=self.headers,
                                data=json.dumps({"genre": "trailer"}))
        self.assertEqual(resp.status_code, 400)

    def test_seo_optimize_no_transcript(self):
        """POST /ai/seo-optimize without transcript should start job then error."""
        resp = self.client.post("/ai/seo-optimize",
                                headers=self.headers,
                                data=json.dumps({"platform": "youtube"}))
        # filepath_required=False, so it starts a job; error comes from handler
        resp.get_json()
        # Either 200 with job_id (error caught async) or input validation
        self.assertIn(resp.status_code, (200, 400))

    def test_suggest_endpoint(self):
        """POST /ai/suggest should return suggestions synchronously."""
        resp = self.client.post("/ai/suggest",
                                headers=self.headers,
                                data=json.dumps({
                                    "clip_metadata": {
                                        "duration": 120,
                                        "has_audio": True,
                                        "has_video": True,
                                        "loudness_lufs": -8.0,
                                    },
                                    "recent_actions": [],
                                }))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("suggestions", data)
        self.assertIn("count", data)
        self.assertTrue(data["count"] <= 3)

    def test_suggest_missing_metadata(self):
        """POST /ai/suggest without clip_metadata should return 400."""
        resp = self.client.post("/ai/suggest",
                                headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    def test_score_title_endpoint(self):
        """POST /ai/score-title should return title scoring."""
        resp = self.client.post("/ai/score-title",
                                headers=self.headers,
                                data=json.dumps({
                                    "title": "10 Best Video Editing Tips for Beginners",
                                    "platform": "youtube",
                                }))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("length_ok", data)
        self.assertIn("estimated_ctr_tier", data)

    def test_score_title_empty(self):
        """POST /ai/score-title without title should return 400."""
        resp = self.client.post("/ai/score-title",
                                headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)

    def test_engagement_predict_no_file(self):
        """POST /ai/engagement-predict without filepath should return 400."""
        resp = self.client.post("/ai/engagement-predict",
                                headers=self.headers,
                                data=json.dumps({}))
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
