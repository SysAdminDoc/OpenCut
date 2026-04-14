"""
Tests for OpenCut Engagement & Content features.

Covers:
  - Engagement prediction (scoring, retention curve, drop-off)
  - Caption styles (50+ styles, categories, preview, apply)
  - Hook generator (all hook types, LLM fallback, apply)
  - A/B variant generator (pacing, caption, hook variants)
  - Essential Graphics export (MOGRT JSON, SRT, Premiere XML, AE JSON)
  - Engagement content route smoke tests
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================================================
# 1. Engagement Prediction
# ========================================================================
class TestEngagementResult:
    """EngagementResult dataclass tests."""

    def test_default_values(self):
        from opencut.core.engagement_predict import EngagementResult
        r = EngagementResult()
        assert r.overall_score == 0
        assert r.hook_score == 0
        assert r.pacing_score == 0
        assert r.audio_energy_score == 0
        assert r.variety_score == 0
        assert r.retention_curve == []
        assert r.drop_off_points == []
        assert r.suggestions == []
        assert r.duration == 0.0
        assert r.silence_ratio == 0.0
        assert r.avg_cut_interval == 0.0
        assert r.total_cuts == 0

    def test_to_dict(self):
        from opencut.core.engagement_predict import EngagementResult
        r = EngagementResult(overall_score=75, hook_score=80, duration=120.0)
        d = r.to_dict()
        assert d["overall_score"] == 75
        assert d["hook_score"] == 80
        assert d["duration"] == 120.0
        assert isinstance(d, dict)

    def test_custom_values(self):
        from opencut.core.engagement_predict import EngagementResult
        r = EngagementResult(
            overall_score=60, hook_score=70, pacing_score=55,
            audio_energy_score=65, variety_score=50,
            retention_curve=[(2.5, 95.0), (7.5, 88.0)],
            drop_off_points=[7.5],
            suggestions=["Add more cuts"],
            duration=30.0, silence_ratio=0.1,
            avg_cut_interval=5.0, total_cuts=5,
        )
        assert len(r.retention_curve) == 2
        assert r.drop_off_points == [7.5]


class TestEngagementScoring:
    """Tests for engagement scoring functions."""

    def test_score_hook_short_video(self):
        from opencut.core.engagement_predict import _score_hook
        score, tips = _score_hook([-30.0], 0.5, [])
        assert score == 50
        assert tips == []

    def test_score_hook_quiet_audio(self):
        from opencut.core.engagement_predict import _score_hook
        score, tips = _score_hook([-50.0, -50.0, -50.0], 10.0, [])
        assert score < 50
        any_quiet = any("quiet" in t.lower() for t in tips)
        assert any_quiet

    def test_score_hook_loud_with_cuts(self):
        from opencut.core.engagement_predict import _score_hook
        score, _ = _score_hook([-10.0, -10.0, -10.0], 10.0, [1.0, 2.0])
        assert score >= 70

    def test_score_hook_no_cuts_long_video(self):
        from opencut.core.engagement_predict import _score_hook
        _, tips = _score_hook([-20.0], 30.0, [])
        assert any("visual changes" in t.lower() or "cut" in t.lower() for t in tips)

    def test_score_pacing_short_video(self):
        from opencut.core.engagement_predict import _score_pacing
        score, tips = _score_pacing([], 1.0)
        assert score == 50

    def test_score_pacing_no_cuts_long(self):
        from opencut.core.engagement_predict import _score_pacing
        score, tips = _score_pacing([], 30.0)
        assert score == 20
        assert any("no cuts" in t.lower() for t in tips)

    def test_score_pacing_optimal_cuts(self):
        from opencut.core.engagement_predict import _score_pacing
        cuts = [5.0, 10.0, 15.0, 20.0]
        score, _ = _score_pacing(cuts, 25.0)
        assert score >= 70

    def test_score_pacing_rapid_cuts(self):
        from opencut.core.engagement_predict import _score_pacing
        cuts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        score, tips = _score_pacing(cuts, 3.5)
        assert score >= 40
        # rapid cutting tip
        assert any("rapid" in t.lower() for t in tips)

    def test_score_pacing_slow_cuts(self):
        from opencut.core.engagement_predict import _score_pacing
        cuts = [20.0]
        score, tips = _score_pacing(cuts, 40.0)
        assert score < 90
        assert any("tighter" in t.lower() or "shot length" in t.lower() for t in tips)

    def test_score_audio_energy_empty(self):
        from opencut.core.engagement_predict import _score_audio_energy
        score, tips = _score_audio_energy([])
        assert score == 30
        assert any("no audio" in t.lower() for t in tips)

    def test_score_audio_energy_loud(self):
        from opencut.core.engagement_predict import _score_audio_energy
        score, _ = _score_audio_energy([-10.0, -12.0, -8.0])
        assert score >= 80

    def test_score_audio_energy_quiet(self):
        from opencut.core.engagement_predict import _score_audio_energy
        score, tips = _score_audio_energy([-50.0, -48.0])
        assert score < 40
        assert any("quiet" in t.lower() for t in tips)

    def test_score_audio_energy_flat(self):
        from opencut.core.engagement_predict import _score_audio_energy
        _, tips = _score_audio_energy([-25.0, -25.0, -25.0, -25.0])
        assert any("flat" in t.lower() for t in tips)

    def test_score_audio_energy_dynamic(self):
        from opencut.core.engagement_predict import _score_audio_energy
        score, _ = _score_audio_energy([-10.0, -40.0, -10.0, -40.0])
        assert score > 0

    def test_score_variety_short_video(self):
        from opencut.core.engagement_predict import _score_variety
        score, _ = _score_variety([], [], 1.0)
        assert score == 50

    def test_score_variety_high_silence(self):
        from opencut.core.engagement_predict import _score_variety
        _, tips = _score_variety([], [(0.0, 10.0)], 30.0)
        assert any("dead air" in t.lower() for t in tips)

    def test_score_variety_low_cuts(self):
        from opencut.core.engagement_predict import _score_variety
        _, tips = _score_variety([], [], 60.0)
        assert any("low visual variety" in t.lower() for t in tips)

    def test_score_variety_good_density(self):
        from opencut.core.engagement_predict import _score_variety
        cuts = [5.0 * i for i in range(1, 11)]  # ~10 cuts/min for 60s
        score, _ = _score_variety(cuts, [], 60.0)
        assert score >= 70


class TestRetentionCurve:
    """Tests for retention curve generation."""

    def test_empty_duration(self):
        from opencut.core.engagement_predict import _generate_retention_curve
        curve = _generate_retention_curve([], [], [], 0)
        assert curve == [(0.0, 100.0)]

    def test_basic_curve(self):
        from opencut.core.engagement_predict import _generate_retention_curve
        curve = _generate_retention_curve(
            [-20.0] * 10, [5.0], [], 30.0
        )
        assert len(curve) > 0
        assert curve[0][1] > curve[-1][1]  # retention decreases

    def test_curve_format(self):
        from opencut.core.engagement_predict import _generate_retention_curve
        curve = _generate_retention_curve([-20.0] * 5, [], [], 20.0)
        for ts, pct in curve:
            assert isinstance(ts, float)
            assert isinstance(pct, float)
            assert 0 <= pct <= 100

    def test_drop_off_points_empty(self):
        from opencut.core.engagement_predict import _find_drop_off_points
        # Gradual decay should not trigger drop-off
        curve = [(i * 5.0, 100.0 - i * 2.0) for i in range(10)]
        drops = _find_drop_off_points(curve, threshold=10.0)
        assert drops == []

    def test_drop_off_points_detected(self):
        from opencut.core.engagement_predict import _find_drop_off_points
        curve = [(0.0, 90.0), (5.0, 70.0), (10.0, 65.0)]
        drops = _find_drop_off_points(curve, threshold=10.0)
        assert 5.0 in drops


class TestPredictEngagement:
    """Integration tests for predict_engagement."""

    def test_missing_file(self):
        from opencut.core.engagement_predict import predict_engagement
        with pytest.raises(FileNotFoundError):
            predict_engagement("/nonexistent/video.mp4")

    @patch("opencut.core.engagement_predict._detect_scene_changes")
    @patch("opencut.core.engagement_predict._detect_silence_segments")
    @patch("opencut.core.engagement_predict._extract_audio_levels")
    @patch("opencut.core.engagement_predict.get_video_info")
    def test_full_prediction(self, mock_info, mock_audio, mock_silence, mock_scenes):
        from opencut.core.engagement_predict import predict_engagement

        mock_info.return_value = {"duration": 60.0, "fps": 30.0, "width": 1920, "height": 1080}
        mock_audio.return_value = [-20.0] * 60
        mock_silence.return_value = [(10.0, 12.0)]
        mock_scenes.return_value = [5.0, 15.0, 25.0, 35.0, 45.0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name

        try:
            result = predict_engagement(path)
            assert isinstance(result, dict)
            assert "overall_score" in result
            assert "hook_score" in result
            assert "pacing_score" in result
            assert "retention_curve" in result
            assert "drop_off_points" in result
            assert "suggestions" in result
            assert 0 <= result["overall_score"] <= 100
        finally:
            os.unlink(path)

    @patch("opencut.core.engagement_predict._detect_scene_changes")
    @patch("opencut.core.engagement_predict._detect_silence_segments")
    @patch("opencut.core.engagement_predict._extract_audio_levels")
    @patch("opencut.core.engagement_predict.get_video_info")
    def test_progress_callback(self, mock_info, mock_audio, mock_silence, mock_scenes):
        from opencut.core.engagement_predict import predict_engagement

        mock_info.return_value = {"duration": 30.0, "fps": 30.0, "width": 1920, "height": 1080}
        mock_audio.return_value = [-25.0] * 30
        mock_silence.return_value = []
        mock_scenes.return_value = [10.0, 20.0]

        progress_calls = []

        def track(pct, msg=""):
            progress_calls.append((pct, msg))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name

        try:
            predict_engagement(path, on_progress=track)
            assert len(progress_calls) > 0
            assert progress_calls[-1][0] == 100
        finally:
            os.unlink(path)

    @patch("opencut.core.engagement_predict.get_video_info")
    def test_zero_duration_raises(self, mock_info):
        from opencut.core.engagement_predict import predict_engagement
        mock_info.return_value = {"duration": 0, "fps": 30.0}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="duration"):
                predict_engagement(path)
        finally:
            os.unlink(path)


class TestEngagementConstants:
    """Verify scoring constants and weights."""

    def test_weights_sum_to_one(self):
        from opencut.core.engagement_predict import (
            AUDIO_ENERGY_WEIGHT,
            HOOK_WEIGHT,
            PACING_WEIGHT,
            VARIETY_WEIGHT,
        )
        total = HOOK_WEIGHT + PACING_WEIGHT + AUDIO_ENERGY_WEIGHT + VARIETY_WEIGHT
        assert abs(total - 1.0) < 0.001

    def test_optimal_cut_interval(self):
        from opencut.core.engagement_predict import OPTIMAL_CUT_INTERVAL
        assert OPTIMAL_CUT_INTERVAL[0] < OPTIMAL_CUT_INTERVAL[1]
        assert OPTIMAL_CUT_INTERVAL[0] > 0


# ========================================================================
# 2. Caption Styles
# ========================================================================
class TestCaptionStyleDataclass:
    """CaptionStyle dataclass tests."""

    def test_default_values(self):
        from opencut.core.caption_styles import CaptionStyle
        s = CaptionStyle()
        assert s.id == ""
        assert s.name == ""
        assert s.category == ""
        assert s.animation_type == "none"
        assert s.font_family == "Arial"
        assert s.font_size == 48
        assert s.position == "bottom"

    def test_to_dict(self):
        from opencut.core.caption_styles import CaptionStyle
        s = CaptionStyle(id="test", name="Test Style")
        d = s.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test Style"
        assert isinstance(d, dict)


class TestBuiltinStyles:
    """Tests for the 50+ built-in styles."""

    def test_at_least_50_styles(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert len(BUILTIN_STYLES) >= 50

    def test_all_styles_have_required_fields(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        for style_id, style in BUILTIN_STYLES.items():
            assert style.id, f"Style missing id: {style_id}"
            assert style.name, f"Style {style_id} missing name"
            assert style.category, f"Style {style_id} missing category"
            assert style.preview_description, f"Style {style_id} missing description"
            assert style.animation_type, f"Style {style_id} missing animation_type"
            assert style.font_family, f"Style {style_id} missing font_family"
            assert style.font_size > 0, f"Style {style_id} invalid font_size"
            assert style.position in ("top", "center", "bottom"), \
                f"Style {style_id} invalid position: {style.position}"

    def test_all_categories_represented(self):
        from opencut.core.caption_styles import BUILTIN_STYLES, CATEGORIES
        present = {s.category for s in BUILTIN_STYLES.values()}
        for cat in CATEGORIES:
            assert cat in present, f"Category '{cat}' has no styles"

    def test_unique_ids(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        ids = list(BUILTIN_STYLES.keys())
        assert len(ids) == len(set(ids)), "Duplicate style IDs found"

    def test_unique_names(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        names = [s.name for s in BUILTIN_STYLES.values()]
        assert len(names) == len(set(names)), "Duplicate style names found"

    def test_all_styles_have_colors(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        for style_id, style in BUILTIN_STYLES.items():
            assert isinstance(style.colors, dict), f"Style {style_id} colors not dict"
            assert len(style.colors) > 0, f"Style {style_id} has no colors"

    def test_valid_animation_types(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        valid = {"none", "fade", "bounce", "wave", "typewriter", "karaoke",
                 "slide", "zoom", "glow"}
        for style_id, style in BUILTIN_STYLES.items():
            assert style.animation_type in valid, \
                f"Style {style_id} invalid animation: {style.animation_type}"

    def test_tiktok_bold_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "tiktok_bold" in BUILTIN_STYLES
        s = BUILTIN_STYLES["tiktok_bold"]
        assert s.category == "bold"
        assert s.animation_type == "bounce"

    def test_youtube_clean_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "youtube_clean" in BUILTIN_STYLES
        assert BUILTIN_STYLES["youtube_clean"].category == "bold"

    def test_karaoke_highlight_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "karaoke_highlight" in BUILTIN_STYLES
        assert BUILTIN_STYLES["karaoke_highlight"].animation_type == "karaoke"

    def test_neon_glow_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "neon_glow" in BUILTIN_STYLES
        assert BUILTIN_STYLES["neon_glow"].category == "glow"

    def test_typewriter_classic_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "typewriter_classic" in BUILTIN_STYLES

    def test_bounce_pop_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "bounce_pop" in BUILTIN_STYLES

    def test_gradient_wave_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "gradient_wave" in BUILTIN_STYLES

    def test_instagram_story_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "instagram_story" in BUILTIN_STYLES

    def test_news_ticker_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "news_ticker" in BUILTIN_STYLES

    def test_cinematic_subtitle_exists(self):
        from opencut.core.caption_styles import BUILTIN_STYLES
        assert "cinematic_subtitle" in BUILTIN_STYLES


class TestGetAvailableStyles:
    """Tests for get_available_styles."""

    def test_returns_list(self):
        from opencut.core.caption_styles import CaptionStyle, get_available_styles
        styles = get_available_styles()
        assert isinstance(styles, list)
        assert all(isinstance(s, CaptionStyle) for s in styles)

    def test_count_matches_builtin(self):
        from opencut.core.caption_styles import BUILTIN_STYLES, get_available_styles
        styles = get_available_styles()
        assert len(styles) == len(BUILTIN_STYLES)


class TestGetStyle:
    """Tests for get_style."""

    def test_valid_style(self):
        from opencut.core.caption_styles import get_style
        s = get_style("tiktok_bold")
        assert s.id == "tiktok_bold"

    def test_invalid_style(self):
        from opencut.core.caption_styles import get_style
        with pytest.raises(ValueError, match="Unknown caption style"):
            get_style("nonexistent_style_xyz")


class TestCaptionHelpers:
    """Tests for caption helper functions."""

    def test_escape_drawtext(self):
        from opencut.core.caption_styles import _escape_drawtext
        assert "\\:" in _escape_drawtext("test: value")
        assert "%%" in _escape_drawtext("100%")
        assert "\\'" in _escape_drawtext("it's")

    def test_position_y_top(self):
        from opencut.core.caption_styles import _position_y
        y = _position_y("top", 1080, 48)
        assert y < 200

    def test_position_y_center(self):
        from opencut.core.caption_styles import _position_y
        y = _position_y("center", 1080, 48)
        assert 400 < y < 600

    def test_position_y_bottom(self):
        from opencut.core.caption_styles import _position_y
        y = _position_y("bottom", 1080, 48)
        assert y > 800

    def test_build_drawtext_filter(self):
        from opencut.core.caption_styles import CaptionStyle, _build_drawtext_filter
        style = CaptionStyle(
            font_size=48, position="bottom",
            colors={"text": "#FFFFFF", "shadow": "#000000"},
        )
        filt = _build_drawtext_filter(style, "Hello", 0.0, 3.0)
        assert "drawtext=" in filt
        assert "Hello" in filt
        assert "enable=" in filt


class TestApplyCaptionStyle:
    """Tests for apply_caption_style."""

    def test_missing_file(self):
        from opencut.core.caption_styles import apply_caption_style
        with pytest.raises(FileNotFoundError):
            apply_caption_style("/nonexistent.mp4", [{"text": "hi", "start": 0, "end": 1}], "tiktok_bold")

    def test_empty_captions(self):
        from opencut.core.caption_styles import apply_caption_style
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="at least one"):
                apply_caption_style(path, [], "tiktok_bold")
        finally:
            os.unlink(path)

    def test_invalid_style(self):
        from opencut.core.caption_styles import apply_caption_style
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unknown"):
                apply_caption_style(path, [{"text": "hi", "start": 0, "end": 1}], "fake_style")
        finally:
            os.unlink(path)

    @patch("opencut.core.caption_styles.run_ffmpeg")
    @patch("opencut.helpers.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0})
    def test_calls_ffmpeg(self, mock_info, mock_ffmpeg):
        from opencut.core.caption_styles import apply_caption_style
        mock_ffmpeg.return_value = ""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            captions = [{"text": "Hello", "start": 0, "end": 2}]
            result = apply_caption_style(path, captions, "tiktok_bold")
            assert result.endswith(".mp4")
            mock_ffmpeg.assert_called_once()
        finally:
            os.unlink(path)


class TestGenerateStylePreview:
    """Tests for generate_style_preview."""

    @patch("opencut.core.caption_styles.run_ffmpeg")
    def test_returns_bytes(self, mock_ffmpeg):
        from opencut.core.caption_styles import generate_style_preview
        mock_ffmpeg.return_value = ""
        # Mock the file read
        fake_png = b"\x89PNG\r\n\x1a\nfake"
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = MagicMock(return_value=fake_png)
            result = generate_style_preview("tiktok_bold")
            assert isinstance(result, bytes)

    def test_invalid_style(self):
        from opencut.core.caption_styles import generate_style_preview
        with pytest.raises(ValueError):
            generate_style_preview("nonexistent_style")


# ========================================================================
# 3. Hook Generator
# ========================================================================
class TestHookResultDataclass:
    """HookResult dataclass tests."""

    def test_default_values(self):
        from opencut.core.hook_generator import HookResult
        r = HookResult()
        assert r.hook_text == ""
        assert r.hook_type == "auto"
        assert r.insertion_method == "caption_overlay"
        assert r.preview_text == ""
        assert r.teaser_start == 0.0
        assert r.teaser_end == 0.0

    def test_to_dict(self):
        from opencut.core.hook_generator import HookResult
        r = HookResult(hook_text="Test hook", hook_type="question")
        d = r.to_dict()
        assert d["hook_text"] == "Test hook"
        assert d["hook_type"] == "question"


class TestHookHelpers:
    """Tests for hook text extraction helpers."""

    def test_extract_sentences(self):
        from opencut.core.hook_generator import _extract_sentences
        text = "This is amazing. It changed everything! Did you know?"
        sentences = _extract_sentences(text)
        assert len(sentences) == 3

    def test_extract_sentences_srt_format(self):
        from opencut.core.hook_generator import _extract_sentences
        srt = "1\n00:00:00,000 --> 00:00:02,000\nThis is the first sentence.\n\n"
        sentences = _extract_sentences(srt)
        assert len(sentences) >= 1
        assert "first sentence" in sentences[0]

    def test_find_question(self):
        from opencut.core.hook_generator import _find_question
        sentences = ["This is a statement.", "Did you know this secret?", "Another line."]
        q = _find_question(sentences)
        assert q is not None
        assert "?" in q

    def test_find_question_none(self):
        from opencut.core.hook_generator import _find_question
        sentences = ["No questions here.", "Just statements."]
        assert _find_question(sentences) is None

    def test_find_statistic(self):
        from opencut.core.hook_generator import _find_statistic
        sentences = ["Normal sentence.", "Over 90% of people fail this test.", "End."]
        stat = _find_statistic(sentences)
        assert stat is not None
        assert "90%" in stat

    def test_find_statistic_dollar(self):
        from opencut.core.hook_generator import _find_statistic
        sentences = ["It costs $1,000 per year to maintain."]
        stat = _find_statistic(sentences)
        assert stat is not None

    def test_find_statistic_none(self):
        from opencut.core.hook_generator import _find_statistic
        sentences = ["No numbers here.", "Just words."]
        assert _find_statistic(sentences) is None

    def test_find_quote(self):
        from opencut.core.hook_generator import _find_quote
        sentences = ["Normal text.", "This incredible discovery transformed everything.", "End."]
        quote = _find_quote(sentences)
        assert quote is not None
        assert "incredible" in quote.lower() or "transformed" in quote.lower()

    def test_find_quote_none(self):
        from opencut.core.hook_generator import _find_quote
        sentences = ["The cat sat on the mat.", "It was fine."]
        assert _find_quote(sentences) is None


class TestGenerateHook:
    """Tests for generate_hook."""

    def test_missing_file(self):
        from opencut.core.hook_generator import generate_hook
        with pytest.raises(FileNotFoundError):
            generate_hook("/nonexistent.mp4")

    def test_invalid_hook_type(self):
        from opencut.core.hook_generator import generate_hook
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid hook_type"):
                generate_hook(path, hook_type="invalid_type")
        finally:
            os.unlink(path)

    @patch("opencut.core.hook_generator._find_highest_energy_segment")
    @patch("opencut.core.hook_generator.get_video_info")
    def test_teaser_type(self, mock_info, mock_energy):
        from opencut.core.hook_generator import generate_hook
        mock_info.return_value = {"duration": 60.0, "fps": 30.0}
        mock_energy.return_value = (15.0, 17.0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_hook(path, hook_type="teaser")
            assert result.hook_type == "teaser"
            assert result.insertion_method == "prepend_clip"
            assert result.teaser_start > 0
        finally:
            os.unlink(path)

    @patch("opencut.core.hook_generator._try_llm_hook", return_value=None)
    @patch("opencut.core.hook_generator.get_video_info")
    def test_question_type_with_transcript(self, mock_info, mock_llm):
        from opencut.core.hook_generator import generate_hook
        mock_info.return_value = {"duration": 60.0, "fps": 30.0}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            transcript = "This is important. Did you know this one simple trick changes everything?"
            result = generate_hook(path, transcript=transcript, hook_type="question")
            assert result.hook_text
            assert "?" in result.hook_text
        finally:
            os.unlink(path)

    @patch("opencut.core.hook_generator.get_video_info")
    def test_statistic_type(self, mock_info):
        from opencut.core.hook_generator import generate_hook
        mock_info.return_value = {"duration": 60.0, "fps": 30.0}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            transcript = "A normal day. Over 85% of users experienced this problem. The end."
            result = generate_hook(path, transcript=transcript, hook_type="statistic")
            assert result.hook_text
        finally:
            os.unlink(path)

    @patch("opencut.core.hook_generator.get_video_info")
    def test_quote_type(self, mock_info):
        from opencut.core.hook_generator import generate_hook
        mock_info.return_value = {"duration": 60.0, "fps": 30.0}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            transcript = "This incredible discovery transformed the entire industry forever."
            result = generate_hook(path, transcript=transcript, hook_type="quote")
            assert result.hook_text
        finally:
            os.unlink(path)

    @patch("opencut.core.hook_generator._find_highest_energy_segment")
    @patch("opencut.core.hook_generator.get_video_info")
    def test_auto_no_transcript(self, mock_info, mock_energy):
        from opencut.core.hook_generator import generate_hook
        mock_info.return_value = {"duration": 60.0, "fps": 30.0}
        mock_energy.return_value = (20.0, 22.0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_hook(path, hook_type="auto")
            # Without transcript, auto should fall back to teaser
            assert result.insertion_method == "prepend_clip"
        finally:
            os.unlink(path)

    def test_hook_types_constant(self):
        from opencut.core.hook_generator import HOOK_TYPES
        assert "auto" in HOOK_TYPES
        assert "question" in HOOK_TYPES
        assert "statistic" in HOOK_TYPES
        assert "teaser" in HOOK_TYPES
        assert "quote" in HOOK_TYPES


class TestApplyHook:
    """Tests for apply_hook."""

    def test_missing_file(self):
        from opencut.core.hook_generator import HookResult, apply_hook
        hook = HookResult(hook_text="Test", insertion_method="caption_overlay")
        with pytest.raises(FileNotFoundError):
            apply_hook("/nonexistent.mp4", hook)

    @patch("opencut.core.hook_generator.run_ffmpeg")
    @patch("opencut.core.hook_generator.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0})
    def test_caption_overlay(self, mock_info, mock_ffmpeg):
        from opencut.core.hook_generator import HookResult, apply_hook
        mock_ffmpeg.return_value = ""
        hook = HookResult(hook_text="Watch this!", insertion_method="caption_overlay")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = apply_hook(path, hook)
            assert result.endswith(".mp4")
            mock_ffmpeg.assert_called_once()
        finally:
            os.unlink(path)

    @patch("opencut.core.hook_generator.run_ffmpeg")
    @patch("opencut.core.hook_generator.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0})
    def test_prepend_clip(self, mock_info, mock_ffmpeg):
        from opencut.core.hook_generator import HookResult, apply_hook
        mock_ffmpeg.return_value = ""
        hook = HookResult(
            hook_text="Teaser",
            insertion_method="prepend_clip",
            teaser_start=15.0, teaser_end=17.0,
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = apply_hook(path, hook)
            assert result.endswith(".mp4")
            assert mock_ffmpeg.call_count >= 2  # extract + concat
        finally:
            os.unlink(path)


# ========================================================================
# 4. A/B Variant Generator
# ========================================================================
class TestVariantDataclasses:
    """Variant and VariantSetResult dataclass tests."""

    def test_variant_defaults(self):
        from opencut.core.ab_variant import Variant
        v = Variant()
        assert v.variant_id == ""
        assert v.output_path == ""
        assert v.changes == {}
        assert v.description == ""

    def test_variant_to_dict(self):
        from opencut.core.ab_variant import Variant
        v = Variant(variant_id="A", output_path="/out/test_A.mp4",
                    changes={"hook": "teaser"}, description="Teaser hook")
        d = v.to_dict()
        assert d["variant_id"] == "A"
        assert d["changes"]["hook"] == "teaser"

    def test_variant_set_defaults(self):
        from opencut.core.ab_variant import VariantSetResult
        r = VariantSetResult()
        assert r.variants == []
        assert r.output_dir == ""

    def test_variant_set_to_dict(self):
        from opencut.core.ab_variant import Variant, VariantSetResult
        r = VariantSetResult(
            variants=[Variant(variant_id="A")],
            output_dir="/output",
        )
        d = r.to_dict()
        assert len(d["variants"]) == 1
        assert d["output_dir"] == "/output"


class TestGenerateVariants:
    """Tests for generate_variants."""

    def test_missing_file(self):
        from opencut.core.ab_variant import generate_variants
        with pytest.raises(FileNotFoundError):
            generate_variants("/nonexistent.mp4")

    def test_invalid_vary_element(self):
        from opencut.core.ab_variant import generate_variants
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Invalid vary element"):
                generate_variants(path, vary_elements=["invalid_element"])
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0})
    def test_default_variants(self, mock_info, mock_ffmpeg):
        from opencut.core.ab_variant import generate_variants
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_variants(path)
            assert len(result.variants) == 3
            letters = [v.variant_id for v in result.variants]
            assert "A" in letters
            assert "B" in letters
            assert "C" in letters
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0})
    def test_variant_count_clamped(self, mock_info, mock_ffmpeg):
        from opencut.core.ab_variant import generate_variants
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_variants(path, variant_count=10)
            assert len(result.variants) == 5  # clamped to max 5

            result2 = generate_variants(path, variant_count=0)
            assert len(result2.variants) == 1  # clamped to min 1
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0})
    def test_pacing_variants(self, mock_info, mock_ffmpeg):
        from opencut.core.ab_variant import generate_variants
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_variants(path, variant_count=2, vary_elements=["pacing"])
            assert len(result.variants) == 2
            for v in result.variants:
                assert "pacing" in v.changes or "copy" in v.changes
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0})
    def test_hook_variants(self, mock_info, mock_ffmpeg):
        from opencut.core.ab_variant import generate_variants
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_variants(path, variant_count=3, vary_elements=["hook"])
            assert len(result.variants) == 3
            for v in result.variants:
                assert "hook" in v.changes
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0})
    def test_output_naming(self, mock_info, mock_ffmpeg):
        from opencut.core.ab_variant import generate_variants
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="test_") as f:
            f.write(b"fake")
            path = f.name
        try:
            result = generate_variants(path, variant_count=2)
            for v in result.variants:
                assert "_variant_" in v.output_path
                assert v.output_path.endswith(".mp4")
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info",
           return_value={"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0})
    def test_progress_callback(self, mock_info, mock_ffmpeg):
        from opencut.core.ab_variant import generate_variants
        mock_ffmpeg.return_value = ""

        progress_calls = []

        def track(pct, msg=""):
            progress_calls.append(pct)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            generate_variants(path, variant_count=2, on_progress=track)
            assert len(progress_calls) > 0
            assert progress_calls[-1] == 100
        finally:
            os.unlink(path)

    def test_vary_elements_constant(self):
        from opencut.core.ab_variant import VARY_ELEMENTS
        assert "hook" in VARY_ELEMENTS
        assert "caption_style" in VARY_ELEMENTS
        assert "thumbnail" in VARY_ELEMENTS
        assert "music" in VARY_ELEMENTS
        assert "pacing" in VARY_ELEMENTS


# ========================================================================
# 5. Essential Graphics Export
# ========================================================================
class TestMOGRTExportResult:
    """MOGRTExportResult dataclass tests."""

    def test_defaults(self):
        from opencut.core.essential_graphics import MOGRTExportResult
        r = MOGRTExportResult()
        assert r.output_path == ""
        assert r.format == ""
        assert r.caption_count == 0
        assert r.duration == 0.0

    def test_to_dict(self):
        from opencut.core.essential_graphics import MOGRTExportResult
        r = MOGRTExportResult(output_path="/out/test.json", format="json",
                              caption_count=10, duration=120.0)
        d = r.to_dict()
        assert d["output_path"] == "/out/test.json"
        assert d["caption_count"] == 10


class TestSRTExport:
    """Tests for export_as_srt_for_premiere."""

    def test_empty_captions(self):
        from opencut.core.essential_graphics import export_as_srt_for_premiere
        with pytest.raises(ValueError, match="at least one"):
            export_as_srt_for_premiere([])

    def test_basic_srt(self):
        from opencut.core.essential_graphics import export_as_srt_for_premiere
        captions = [
            {"text": "Hello world", "start": 0.0, "end": 2.5},
            {"text": "This is a test", "start": 3.0, "end": 5.0},
        ]
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            out = f.name
        try:
            result = export_as_srt_for_premiere(captions, out_path=out)
            assert result == out
            with open(out, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Hello world" in content
            assert "00:00:00,000 --> 00:00:02,500" in content
            assert "00:00:03,000 --> 00:00:05,000" in content
        finally:
            os.unlink(out)

    def test_srt_format_time(self):
        from opencut.core.essential_graphics import _format_srt_time
        assert _format_srt_time(0.0) == "00:00:00,000"
        assert _format_srt_time(65.5) == "00:01:05,500"
        assert _format_srt_time(3661.123) == "01:01:01,123"


class TestPremiereXML:
    """Tests for generate_premiere_caption_xml."""

    def test_empty_captions(self):
        from opencut.core.essential_graphics import generate_premiere_caption_xml
        with pytest.raises(ValueError, match="at least one"):
            generate_premiere_caption_xml([])

    def test_basic_xml(self):
        from opencut.core.essential_graphics import generate_premiere_caption_xml
        captions = [
            {"text": "First caption", "start": 0.0, "end": 2.0},
            {"text": "Second caption", "start": 3.0, "end": 5.0},
        ]
        xml_str = generate_premiere_caption_xml(captions)
        assert "<?xml" in xml_str
        assert "xmeml" in xml_str
        assert "First caption" in xml_str
        assert "Second caption" in xml_str
        assert "OpenCut Captions" in xml_str

    def test_xml_with_style(self):
        from opencut.core.essential_graphics import generate_premiere_caption_xml
        captions = [{"text": "Styled", "start": 0.0, "end": 2.0}]
        xml_str = generate_premiere_caption_xml(
            captions, style={"font": "Helvetica", "size": 36, "color": "#FF0000"}
        )
        assert "Helvetica" in xml_str
        assert "36" in xml_str

    def test_xml_markers(self):
        from opencut.core.essential_graphics import generate_premiere_caption_xml
        captions = [{"text": "Marker test", "start": 0.0, "end": 2.0}]
        xml_str = generate_premiere_caption_xml(captions)
        assert "<markers>" in xml_str
        assert "Marker test" in xml_str

    def test_xml_custom_fps(self):
        from opencut.core.essential_graphics import generate_premiere_caption_xml
        captions = [{"text": "Test", "start": 1.0, "end": 2.0}]
        xml_str = generate_premiere_caption_xml(captions, fps=24.0)
        assert "<timebase>24</timebase>" in xml_str


class TestMOGRTData:
    """Tests for export_as_mogrt_data."""

    def test_empty_captions(self):
        from opencut.core.essential_graphics import export_as_mogrt_data
        with pytest.raises(ValueError, match="at least one"):
            export_as_mogrt_data([])

    def test_basic_export(self):
        from opencut.core.essential_graphics import export_as_mogrt_data
        captions = [
            {"text": "Hello", "start": 0.0, "end": 2.0},
            {"text": "World", "start": 2.5, "end": 4.0},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        try:
            result = export_as_mogrt_data(captions, out_path=out)
            assert result.output_path == out
            assert result.caption_count == 2
            assert result.duration == 4.0
            assert result.format == "opencut_essential_graphics_json"

            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["format"] == "opencut_essential_graphics"
            assert data["caption_count"] == 2
            assert len(data["captions"]) == 2
        finally:
            os.unlink(out)

    def test_export_with_style(self):
        from opencut.core.essential_graphics import export_as_mogrt_data
        captions = [{"text": "Test", "start": 0.0, "end": 1.0}]
        style = {"font": "Impact", "size": 56, "color": "#FF0000"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        try:
            export_as_mogrt_data(captions, style=style, out_path=out)
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["style"]["font_family"] == "Impact"
            assert data["style"]["font_size"] == 56
        finally:
            os.unlink(out)

    def test_export_with_word_timing(self):
        from opencut.core.essential_graphics import export_as_mogrt_data
        captions = [{
            "text": "Hello world",
            "start": 0.0, "end": 2.0,
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.8},
                {"word": "world", "start": 0.9, "end": 2.0},
            ],
        }]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        try:
            export_as_mogrt_data(captions, out_path=out)
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "words" in data["captions"][0]
            assert len(data["captions"][0]["words"]) == 2
        finally:
            os.unlink(out)


class TestAEExport:
    """Tests for export_as_ae_json."""

    def test_empty_captions(self):
        from opencut.core.essential_graphics import export_as_ae_json
        with pytest.raises(ValueError, match="at least one"):
            export_as_ae_json([])

    def test_basic_ae_export(self):
        from opencut.core.essential_graphics import export_as_ae_json
        captions = [
            {"text": "Layer one", "start": 0.0, "end": 2.0},
            {"text": "Layer two", "start": 3.0, "end": 5.0},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        try:
            result = export_as_ae_json(captions, out_path=out)
            assert result == out
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["format"] == "opencut_ae_captions"
            assert len(data["layers"]) == 2
            assert data["layers"][0]["text"] == "Layer one"
            assert data["layers"][0]["inPoint"] == 0.0
            assert data["layers"][0]["outPoint"] == 2.0
        finally:
            os.unlink(out)

    def test_ae_export_with_style(self):
        from opencut.core.essential_graphics import export_as_ae_json
        captions = [{"text": "Test", "start": 0.0, "end": 1.0}]
        style = {"font": "Helvetica", "size": 36, "width": 3840, "height": 2160}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        try:
            export_as_ae_json(captions, style=style, out_path=out)
            with open(out, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["composition"]["width"] == 3840
            assert data["text_layer_style"]["font"] == "Helvetica"
        finally:
            os.unlink(out)


# ========================================================================
# 6. Route Smoke Tests
# ========================================================================
class TestEngagementRoutes:
    """Smoke tests for engagement content routes."""

    def test_blueprint_exists(self):
        from opencut.routes.engagement_content_routes import engagement_content_bp
        assert engagement_content_bp.name == "engagement_content"

    def test_list_caption_styles_route(self, client, csrf_token):
        resp = client.get("/captions/styles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "styles" in data
        assert "count" in data
        assert "categories" in data
        assert data["count"] >= 50

    def test_list_caption_styles_filter_category(self, client, csrf_token):
        resp = client.get("/captions/styles?category=bold")
        data = resp.get_json()
        assert data["count"] > 0
        for s in data["styles"]:
            assert s["category"] == "bold"

    def test_list_caption_styles_empty_category(self, client, csrf_token):
        resp = client.get("/captions/styles?category=nonexistent")
        data = resp.get_json()
        assert data["count"] == 0

    @patch("opencut.core.engagement_predict.get_video_info")
    @patch("opencut.core.engagement_predict._extract_audio_levels")
    @patch("opencut.core.engagement_predict._detect_silence_segments")
    @patch("opencut.core.engagement_predict._detect_scene_changes")
    def test_engagement_predict_route(self, mock_scenes, mock_silence,
                                      mock_audio, mock_info, client, csrf_token):
        mock_info.return_value = {"duration": 30.0, "fps": 30.0, "width": 1920, "height": 1080}
        mock_audio.return_value = [-20.0] * 30
        mock_silence.return_value = []
        mock_scenes.return_value = [10.0, 20.0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name

        try:
            from tests.conftest import csrf_headers
            resp = client.post("/engagement/predict",
                               headers=csrf_headers(csrf_token),
                               json={"filepath": path})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "job_id" in data
        finally:
            os.unlink(path)

    def test_caption_style_preview_missing_id(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/captions/style/preview",
                           headers=csrf_headers(csrf_token),
                           json={"style_id": ""})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    @patch("opencut.core.hook_generator.get_video_info")
    @patch("opencut.core.hook_generator._find_highest_energy_segment")
    def test_generate_hook_route(self, mock_energy, mock_info, client, csrf_token):
        mock_info.return_value = {"duration": 60.0, "fps": 30.0}
        mock_energy.return_value = (15.0, 17.0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name

        try:
            from tests.conftest import csrf_headers
            resp = client.post("/content/generate-hook",
                               headers=csrf_headers(csrf_token),
                               json={"filepath": path, "hook_type": "teaser"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "job_id" in data
        finally:
            os.unlink(path)

    @patch("opencut.core.ab_variant.run_ffmpeg")
    @patch("opencut.core.ab_variant.get_video_info")
    def test_ab_variants_route(self, mock_info, mock_ffmpeg, client, csrf_token):
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name

        try:
            from tests.conftest import csrf_headers
            resp = client.post("/content/ab-variants",
                               headers=csrf_headers(csrf_token),
                               json={"filepath": path, "variant_count": 2})
            assert resp.status_code == 200
            data = resp.get_json()
            assert "job_id" in data
        finally:
            os.unlink(path)

    def test_essential_graphics_export_route(self, client, csrf_token):
        from tests.conftest import csrf_headers
        captions = [
            {"text": "Hello", "start": 0.0, "end": 2.0},
        ]
        resp = client.post("/captions/export/essential-graphics",
                           headers=csrf_headers(csrf_token),
                           json={"captions": captions})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_premiere_xml_export_route(self, client, csrf_token):
        from tests.conftest import csrf_headers
        captions = [
            {"text": "Test cap", "start": 0.0, "end": 2.0},
        ]
        resp = client.post("/captions/export/premiere-xml",
                           headers=csrf_headers(csrf_token),
                           json={"captions": captions})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_premiere_srt_export_route(self, client, csrf_token):
        from tests.conftest import csrf_headers
        captions = [
            {"text": "SRT test", "start": 0.0, "end": 3.0},
        ]
        resp = client.post("/captions/export/premiere-xml",
                           headers=csrf_headers(csrf_token),
                           json={"captions": captions, "format": "srt"})
        assert resp.status_code == 200
