"""
Tests for OpenCut AI Timeline Intelligence features (Category 68).

Covers:
  - Timeline quality analysis (dataclasses, weights, scoring, continuity)
  - Timeline engagement scoring (segment scores, visual/audio/pacing)
  - Clip narrative builder (clip info, styles, ordering, transitions, EDL)
  - AI color intent (intent library, resolution, filter scaling, application)
  - Auto-dub pipeline (config, stages, language support, orchestration)
  - Timeline intelligence routes (smoke tests for all endpoints)
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================================================
# 1. timeline_quality.py
# ========================================================================
class TestTimelineQuality:
    """Tests for opencut.core.timeline_quality."""

    def test_timeline_quality_result_defaults(self):
        """TimelineQualityResult should have sensible defaults."""
        from opencut.core.timeline_quality import TimelineQualityResult
        r = TimelineQualityResult()
        assert r.overall_score == 0.0
        assert r.color_consistency_score == 0.0
        assert r.audio_consistency_score == 0.0
        assert r.pacing_score == 0.0
        assert r.continuity_score == 0.0
        assert r.technical_score == 0.0
        assert r.total_duration == 0.0
        assert r.shot_count == 0
        assert r.shots == []
        assert r.pacing == {}
        assert r.issues == []
        assert r.recommendations == []

    def test_timeline_quality_result_to_dict(self):
        """TimelineQualityResult.to_dict() should return a plain dict."""
        from opencut.core.timeline_quality import TimelineQualityResult
        r = TimelineQualityResult(overall_score=85.0, shot_count=10)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["overall_score"] == 85.0
        assert d["shot_count"] == 10

    def test_shot_info_defaults(self):
        """ShotInfo should have correct defaults."""
        from opencut.core.timeline_quality import ShotInfo
        s = ShotInfo(index=0, start=0.0, end=5.0, duration=5.0)
        assert s.avg_brightness == 0.0
        assert s.lufs == -70.0

    def test_shot_info_to_dict(self):
        """ShotInfo.to_dict() should serialize all fields."""
        from opencut.core.timeline_quality import ShotInfo
        s = ShotInfo(index=1, start=1.0, end=3.0, duration=2.0, avg_brightness=0.5)
        d = s.to_dict()
        assert d["index"] == 1
        assert d["avg_brightness"] == 0.5

    def test_continuity_issue_defaults(self):
        """ContinuityIssue should have correct defaults."""
        from opencut.core.timeline_quality import ContinuityIssue
        c = ContinuityIssue(issue_type="flash_frame", timestamp=2.0)
        assert c.severity == "warning"
        assert c.duration == 0.0

    def test_continuity_issue_to_dict(self):
        """ContinuityIssue.to_dict() should work."""
        from opencut.core.timeline_quality import ContinuityIssue
        c = ContinuityIssue(issue_type="jump_cut", timestamp=5.0)
        d = c.to_dict()
        assert d["issue_type"] == "jump_cut"

    def test_pacing_analysis_defaults(self):
        """PacingAnalysis should have correct defaults."""
        from opencut.core.timeline_quality import PacingAnalysis
        p = PacingAnalysis()
        assert p.total_cuts == 0
        assert p.pacing_label == ""
        assert p.score == 0.0

    def test_scoring_weights_sum_to_one(self):
        """Quality scoring weights must sum to 1.0."""
        from opencut.core.timeline_quality import (
            WEIGHT_AUDIO_CONSISTENCY,
            WEIGHT_COLOR_CONSISTENCY,
            WEIGHT_CONTINUITY,
            WEIGHT_PACING,
            WEIGHT_TECHNICAL,
        )
        total = (WEIGHT_COLOR_CONSISTENCY + WEIGHT_AUDIO_CONSISTENCY +
                 WEIGHT_PACING + WEIGHT_CONTINUITY + WEIGHT_TECHNICAL)
        assert abs(total - 1.0) < 0.001

    def test_histogram_similarity_identical(self):
        """Identical histograms should have similarity 1.0."""
        from opencut.core.timeline_quality import _compute_histogram_similarity
        hist = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert _compute_histogram_similarity(hist, hist) == pytest.approx(1.0, abs=0.01)

    def test_histogram_similarity_empty(self):
        """Empty histograms should return 0.0."""
        from opencut.core.timeline_quality import _compute_histogram_similarity
        assert _compute_histogram_similarity([], []) == 0.0

    def test_histogram_similarity_different(self):
        """Different histograms should have lower similarity."""
        from opencut.core.timeline_quality import _compute_histogram_similarity
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 1.0]
        sim = _compute_histogram_similarity(a, b)
        assert sim < 0.5

    def test_analyze_pacing_empty(self):
        """_analyze_pacing with no shots should return default."""
        from opencut.core.timeline_quality import _analyze_pacing
        p = _analyze_pacing([], 60.0)
        assert p.pacing_label == "unknown"
        assert p.score == 50.0

    def test_analyze_pacing_slow(self):
        """_analyze_pacing with few long shots should label 'slow'."""
        from opencut.core.timeline_quality import ShotInfo, _analyze_pacing
        shots = [
            ShotInfo(index=0, start=0, end=30, duration=30),
            ShotInfo(index=1, start=30, end=60, duration=30),
        ]
        p = _analyze_pacing(shots, 60.0)
        assert p.pacing_label == "slow"
        assert p.total_cuts == 1

    def test_analyze_pacing_fast(self):
        """_analyze_pacing with many short shots should label 'fast'."""
        from opencut.core.timeline_quality import ShotInfo, _analyze_pacing
        shots = [ShotInfo(index=i, start=i * 2, end=(i + 1) * 2, duration=2)
                 for i in range(30)]
        p = _analyze_pacing(shots, 60.0)
        assert p.pacing_label in ("fast", "frantic")

    def test_analyze_technical_quality_4k(self):
        """4K video should score near 100."""
        from opencut.core.timeline_quality import _analyze_technical_quality
        info = {"width": 3840, "height": 2160, "fps": 60.0}
        score = _analyze_technical_quality("/fake.mp4", info)
        assert score >= 95

    def test_analyze_technical_quality_sd(self):
        """SD video should have reduced score."""
        from opencut.core.timeline_quality import _analyze_technical_quality
        info = {"width": 640, "height": 480, "fps": 30.0}
        score = _analyze_technical_quality("/fake.mp4", info)
        assert score < 75

    def test_generate_recommendations_perfect(self):
        """Perfect scores should produce 'looks great' recommendation."""
        from opencut.core.timeline_quality import PacingAnalysis, _generate_recommendations
        recs = _generate_recommendations(100, 100, PacingAnalysis(pacing_label="moderate"), [], 100)
        assert any("great" in r.lower() for r in recs)

    def test_generate_recommendations_low_color(self):
        """Low color score should recommend correction."""
        from opencut.core.timeline_quality import PacingAnalysis, _generate_recommendations
        recs = _generate_recommendations(40, 100, PacingAnalysis(pacing_label="moderate"), [], 100)
        assert any("color" in r.lower() for r in recs)

    def test_generate_recommendations_low_audio(self):
        """Low audio score should recommend normalization."""
        from opencut.core.timeline_quality import PacingAnalysis, _generate_recommendations
        recs = _generate_recommendations(100, 40, PacingAnalysis(pacing_label="moderate"), [], 100)
        assert any("audio" in r.lower() or "normalize" in r.lower() for r in recs)

    def test_detect_continuity_flash_frames(self):
        """_detect_continuity_issues should detect flash frames."""
        from opencut.core.timeline_quality import ShotInfo, _detect_continuity_issues
        shots = [
            ShotInfo(index=0, start=0, end=5, duration=5, avg_brightness=0.5, lufs=-20),
            ShotInfo(index=1, start=5, end=5.04, duration=0.04, avg_brightness=0.5, lufs=-20),
            ShotInfo(index=2, start=5.04, end=10, duration=4.96, avg_brightness=0.5, lufs=-20),
        ]
        score, issues = _detect_continuity_issues("/fake.mp4", shots)
        flash_issues = [i for i in issues if i.issue_type == "flash_frame"]
        assert len(flash_issues) == 1

    def test_detect_continuity_black_frames(self):
        """_detect_continuity_issues should detect black frames."""
        from opencut.core.timeline_quality import ShotInfo, _detect_continuity_issues
        shots = [
            ShotInfo(index=0, start=0, end=5, duration=5, avg_brightness=0.5, lufs=-20),
            ShotInfo(index=1, start=5, end=7, duration=2, avg_brightness=0.01, lufs=-60),
            ShotInfo(index=2, start=7, end=12, duration=5, avg_brightness=0.5, lufs=-20),
        ]
        score, issues = _detect_continuity_issues("/fake.mp4", shots)
        black_issues = [i for i in issues if i.issue_type == "black_frame"]
        assert len(black_issues) >= 1

    def test_analyze_timeline_quality_file_not_found(self):
        """analyze_timeline_quality should raise FileNotFoundError."""
        from opencut.core.timeline_quality import analyze_timeline_quality
        with pytest.raises(FileNotFoundError):
            analyze_timeline_quality("/nonexistent/video.mp4")


# ========================================================================
# 2. timeline_score.py
# ========================================================================
class TestTimelineScore:
    """Tests for opencut.core.timeline_score."""

    def test_segment_score_defaults(self):
        """SegmentScore should have sensible defaults."""
        from opencut.core.timeline_score import SegmentScore
        s = SegmentScore()
        assert s.index == 0
        assert s.visual_score == 0.0
        assert s.audio_score == 0.0
        assert s.overall_score == 0.0
        assert s.label == ""
        assert s.has_speech is False

    def test_segment_score_to_dict(self):
        """SegmentScore.to_dict() should return a plain dict."""
        from opencut.core.timeline_score import SegmentScore
        s = SegmentScore(index=2, visual_score=75.0, label="high")
        d = s.to_dict()
        assert d["index"] == 2
        assert d["visual_score"] == 75.0
        assert d["label"] == "high"

    def test_timeline_score_result_defaults(self):
        """TimelineScoreResult should have sensible defaults."""
        from opencut.core.timeline_score import TimelineScoreResult
        r = TimelineScoreResult()
        assert r.total_duration == 0.0
        assert r.segment_count == 0
        assert r.average_score == 0.0
        assert r.engagement_curve == []
        assert r.segments == []
        assert r.summary == ""

    def test_timeline_score_result_to_dict(self):
        """TimelineScoreResult.to_dict() should serialize."""
        from opencut.core.timeline_score import TimelineScoreResult
        r = TimelineScoreResult(total_duration=60.0, segment_count=6, average_score=72.5)
        d = r.to_dict()
        assert d["total_duration"] == 60.0
        assert d["average_score"] == 72.5

    def test_classify_score_high(self):
        """Scores >= 70 should be classified as 'high'."""
        from opencut.core.timeline_score import _classify_score
        assert _classify_score(85.0) == "high"
        assert _classify_score(70.0) == "high"

    def test_classify_score_medium(self):
        """Scores 40-69 should be classified as 'medium'."""
        from opencut.core.timeline_score import _classify_score
        assert _classify_score(55.0) == "medium"
        assert _classify_score(40.0) == "medium"

    def test_classify_score_low(self):
        """Scores < 40 should be classified as 'low'."""
        from opencut.core.timeline_score import _classify_score
        assert _classify_score(20.0) == "low"
        assert _classify_score(0.0) == "low"

    def test_score_audio_silent(self):
        """Silent audio should score low."""
        from opencut.core.timeline_score import _score_audio
        score = _score_audio(-70.0, False)
        assert score == 0.0

    def test_score_audio_with_speech(self):
        """Audio with speech should get a bonus."""
        from opencut.core.timeline_score import _score_audio
        without_speech = _score_audio(-20.0, False)
        with_speech = _score_audio(-20.0, True)
        assert with_speech > without_speech

    def test_score_visual_high_motion(self):
        """High motion should score well visually."""
        from opencut.core.timeline_score import _score_visual
        score = _score_visual(0.5, 0.7, 3, 10.0)
        assert score > 60

    def test_score_visual_static(self):
        """Very low motion should score lower."""
        from opencut.core.timeline_score import _score_visual
        score = _score_visual(0.001, 0.2, 0, 10.0)
        assert score < 40

    def test_score_pacing_intro(self):
        """Intro segment with moderate cuts should score well."""
        from opencut.core.timeline_score import _score_pacing
        score = _score_pacing(5, 10.0, 0, 20)
        assert score > 0

    def test_score_diversity_first_segment(self):
        """First segment should get neutral diversity score."""
        from opencut.core.timeline_score import _score_diversity
        score = _score_diversity([], 0.5, -20.0, True)
        assert score == 50.0

    def test_score_diversity_with_variety(self):
        """Different segment should score higher diversity."""
        from opencut.core.timeline_score import SegmentScore, _score_diversity
        recent = [
            SegmentScore(motion_level=0.1, audio_energy=-30.0, has_speech=False),
            SegmentScore(motion_level=0.1, audio_energy=-30.0, has_speech=False),
        ]
        score = _score_diversity(recent, 0.8, -10.0, True)
        assert score > 50

    def test_generate_summary(self):
        """_generate_summary should produce readable text."""
        from opencut.core.timeline_score import TimelineScoreResult, _generate_summary
        r = TimelineScoreResult(
            total_duration=60.0, segment_count=6, average_score=72.0,
            high_segments=4, medium_segments=1, low_segments=1,
            segments=[{"start": 0}, {"start": 10}, {"start": 20},
                      {"start": 30}, {"start": 40}, {"start": 50}],
            peak_segment_index=2, lowest_segment_index=5,
        )
        summary = _generate_summary(r)
        assert "6 segments" in summary
        assert "72" in summary

    def test_score_timeline_file_not_found(self):
        """score_timeline should raise FileNotFoundError."""
        from opencut.core.timeline_score import score_timeline
        with pytest.raises(FileNotFoundError):
            score_timeline("/nonexistent/video.mp4")

    def test_score_timeline_invalid_duration(self):
        """score_timeline should reject non-positive segment_duration."""
        from opencut.core.timeline_score import score_timeline
        with pytest.raises((ValueError, FileNotFoundError)):
            score_timeline("/nonexistent/video.mp4", segment_duration=-5.0)

    def test_scoring_sub_weights_sum_to_one(self):
        """Engagement sub-weights must sum to 1.0."""
        from opencut.core.timeline_score import (
            AUDIO_WEIGHT,
            DIVERSITY_WEIGHT,
            PACING_WEIGHT,
            VISUAL_WEIGHT,
        )
        total = VISUAL_WEIGHT + AUDIO_WEIGHT + PACING_WEIGHT + DIVERSITY_WEIGHT
        assert abs(total - 1.0) < 0.001


# ========================================================================
# 3. clip_narrative.py
# ========================================================================
class TestClipNarrative:
    """Tests for opencut.core.clip_narrative."""

    def test_clip_info_defaults(self):
        """ClipInfo should have sensible defaults."""
        from opencut.core.clip_narrative import ClipInfo
        c = ClipInfo()
        assert c.mood == "neutral"
        assert c.visual_type == "general"
        assert c.energy_level == 0.5
        assert c.has_speech is False
        assert c.transcript == ""

    def test_clip_info_to_dict(self):
        """ClipInfo.to_dict() should return dict with all fields."""
        from opencut.core.clip_narrative import ClipInfo
        c = ClipInfo(index=1, path="/test.mp4", mood="exciting")
        d = c.to_dict()
        assert d["index"] == 1
        assert d["mood"] == "exciting"

    def test_narrative_result_defaults(self):
        """NarrativeResult should have sensible defaults."""
        from opencut.core.clip_narrative import NarrativeResult
        r = NarrativeResult()
        assert r.style == "documentary"
        assert r.clip_count == 0
        assert r.assembly_order == []
        assert r.clips == []
        assert r.transitions == []
        assert r.narrative_score == 0.0

    def test_narrative_result_to_dict(self):
        """NarrativeResult.to_dict() should serialize all fields."""
        from opencut.core.clip_narrative import NarrativeResult
        r = NarrativeResult(style="vlog", clip_count=3, narrative_score=75.0)
        d = r.to_dict()
        assert d["style"] == "vlog"
        assert d["narrative_score"] == 75.0

    def test_transition_suggestion_defaults(self):
        """TransitionSuggestion should have correct defaults."""
        from opencut.core.clip_narrative import TransitionSuggestion
        t = TransitionSuggestion(from_clip=0, to_clip=1)
        assert t.transition_type == "cut"
        assert t.duration == 0.0

    def test_transition_suggestion_to_dict(self):
        """TransitionSuggestion.to_dict() should work."""
        from opencut.core.clip_narrative import TransitionSuggestion
        t = TransitionSuggestion(from_clip=0, to_clip=1, transition_type="crossfade")
        d = t.to_dict()
        assert d["transition_type"] == "crossfade"

    def test_narrative_styles_keys(self):
        """NARRATIVE_STYLES should contain expected styles."""
        from opencut.core.clip_narrative import NARRATIVE_STYLES
        expected = {"documentary", "vlog", "commercial", "montage",
                    "tutorial", "story", "highlight_reel", "interview"}
        assert expected.issubset(set(NARRATIVE_STYLES.keys()))

    def test_narrative_styles_have_structure(self):
        """Each narrative style should have structure and pace."""
        from opencut.core.clip_narrative import NARRATIVE_STYLES
        for name, data in NARRATIVE_STYLES.items():
            assert "structure" in data, f"{name} missing structure"
            assert "pace" in data, f"{name} missing pace"
            assert isinstance(data["structure"], list), f"{name} structure not a list"
            assert len(data["structure"]) >= 3, f"{name} structure too short"

    def test_narrative_styles_have_transitions(self):
        """Each narrative style should have preferred_transitions."""
        from opencut.core.clip_narrative import NARRATIVE_STYLES
        for name, data in NARRATIVE_STYLES.items():
            assert "preferred_transitions" in data, f"{name} missing transitions"
            assert len(data["preferred_transitions"]) > 0

    def test_transitions_dict(self):
        """TRANSITIONS dict should contain common transition types."""
        from opencut.core.clip_narrative import TRANSITIONS
        assert "cut" in TRANSITIONS
        assert "crossfade" in TRANSITIONS
        assert "dip_to_black" in TRANSITIONS

    def test_classify_mood_exciting(self):
        """High energy + high motion should be exciting."""
        from opencut.core.clip_narrative import _classify_mood
        mood = _classify_mood(energy=0.8, brightness=0.5, has_speech=False, motion=0.6)
        assert mood == "exciting"

    def test_classify_mood_calm(self):
        """Low motion + moderate brightness should be calm."""
        from opencut.core.clip_narrative import _classify_mood
        mood = _classify_mood(energy=0.3, brightness=0.6, has_speech=False, motion=0.1)
        assert mood == "calm"

    def test_classify_mood_sad(self):
        """Low energy + low brightness should be sad."""
        from opencut.core.clip_narrative import _classify_mood
        mood = _classify_mood(energy=0.1, brightness=0.2, has_speech=False, motion=0.05)
        assert mood == "sad"

    def test_classify_mood_informative(self):
        """Speech with low motion should be informative."""
        from opencut.core.clip_narrative import _classify_mood
        mood = _classify_mood(energy=0.4, brightness=0.5, has_speech=True, motion=0.1)
        assert mood == "informative"

    def test_compute_role_affinity_hook(self):
        """Exciting clip should have high affinity for hook role."""
        from opencut.core.clip_narrative import ClipInfo, _compute_role_affinity
        clip = ClipInfo(mood="exciting", energy_level=0.8, visual_type="action")
        score = _compute_role_affinity(clip, "hook")
        assert score > 0.5

    def test_compute_role_affinity_neutral(self):
        """Neutral clip should have moderate affinity for content."""
        from opencut.core.clip_narrative import ClipInfo, _compute_role_affinity
        clip = ClipInfo(mood="neutral", has_speech=True)
        score = _compute_role_affinity(clip, "content")
        assert score > 0.2

    def test_format_tc(self):
        """_format_tc should format seconds correctly."""
        from opencut.core.clip_narrative import _format_tc
        assert _format_tc(0.0) == "00:00:00:00"
        assert _format_tc(61.5) == "00:01:01:15"
        assert _format_tc(3661.0) == "01:01:01:00"

    def test_assign_narrative_order(self):
        """_assign_narrative_order should assign all clips."""
        from opencut.core.clip_narrative import ClipInfo, _assign_narrative_order
        clips = [
            ClipInfo(index=0, mood="exciting", energy_level=0.9),
            ClipInfo(index=1, mood="calm", energy_level=0.2),
            ClipInfo(index=2, mood="neutral", energy_level=0.5, has_speech=True),
        ]
        order, labels, score = _assign_narrative_order(clips, "documentary")
        assert len(order) == 3
        assert len(labels) == 3
        assert set(order) == {0, 1, 2}
        assert score > 0

    def test_build_narrative_no_clips(self):
        """build_narrative with empty list should raise ValueError."""
        from opencut.core.clip_narrative import build_narrative
        with pytest.raises(ValueError):
            build_narrative([])

    def test_build_narrative_bad_style(self):
        """build_narrative with unknown style should raise ValueError."""
        from opencut.core.clip_narrative import build_narrative
        with pytest.raises(ValueError, match="Unknown narrative style"):
            build_narrative(["/fake.mp4"], style="nonexistent_style")

    def test_build_narrative_file_not_found(self):
        """build_narrative should raise FileNotFoundError for missing clips."""
        from opencut.core.clip_narrative import build_narrative
        with pytest.raises(FileNotFoundError):
            build_narrative(["/nonexistent/clip.mp4"])


# ========================================================================
# 4. ai_color_intent.py
# ========================================================================
class TestAIColorIntent:
    """Tests for opencut.core.ai_color_intent."""

    def test_color_intent_defaults(self):
        """ColorIntent should have sensible defaults."""
        from opencut.core.ai_color_intent import ColorIntent
        c = ColorIntent()
        assert c.intent == ""
        assert c.intensity == 1.0
        assert c.source == "library"

    def test_color_intent_to_dict(self):
        """ColorIntent.to_dict() should return dict."""
        from opencut.core.ai_color_intent import ColorIntent
        c = ColorIntent(intent="noir", resolved_name="noir", category="dramatic")
        d = c.to_dict()
        assert d["intent"] == "noir"
        assert d["category"] == "dramatic"

    def test_color_intent_result_defaults(self):
        """ColorIntentResult should have sensible defaults."""
        from opencut.core.ai_color_intent import ColorIntentResult
        r = ColorIntentResult()
        assert r.output_path == ""
        assert r.applied_filters == ""
        assert r.duration == 0.0

    def test_color_intent_result_to_dict(self):
        """ColorIntentResult.to_dict() should work."""
        from opencut.core.ai_color_intent import ColorIntentResult
        r = ColorIntentResult(output_path="/out.mp4", duration=30.0)
        d = r.to_dict()
        assert d["output_path"] == "/out.mp4"

    def test_color_intents_count(self):
        """COLOR_INTENTS should have at least 30 named looks."""
        from opencut.core.ai_color_intent import COLOR_INTENTS
        assert len(COLOR_INTENTS) >= 30

    def test_color_intents_have_filters(self):
        """Every intent should have a non-empty filters string."""
        from opencut.core.ai_color_intent import COLOR_INTENTS
        for name, data in COLOR_INTENTS.items():
            assert "filters" in data, f"{name} missing filters"
            assert len(data["filters"]) > 0, f"{name} has empty filters"

    def test_color_intents_have_description(self):
        """Every intent should have a description."""
        from opencut.core.ai_color_intent import COLOR_INTENTS
        for name, data in COLOR_INTENTS.items():
            assert "description" in data, f"{name} missing description"
            assert len(data["description"]) > 5

    def test_color_intents_have_category(self):
        """Every intent should have a category."""
        from opencut.core.ai_color_intent import COLOR_INTENTS
        for name, data in COLOR_INTENTS.items():
            assert "category" in data, f"{name} missing category"

    def test_fuzzy_match_exact(self):
        """_fuzzy_match_intent should find exact matches."""
        from opencut.core.ai_color_intent import _fuzzy_match_intent
        assert _fuzzy_match_intent("noir") == "noir"
        assert _fuzzy_match_intent("warm sunset") == "warm sunset"
        assert _fuzzy_match_intent("sepia") == "sepia"

    def test_fuzzy_match_partial(self):
        """_fuzzy_match_intent should match partial keywords."""
        from opencut.core.ai_color_intent import _fuzzy_match_intent
        result = _fuzzy_match_intent("sunset warm")
        assert result == "warm sunset"

    def test_fuzzy_match_no_match(self):
        """_fuzzy_match_intent should return None for unrelated queries."""
        from opencut.core.ai_color_intent import _fuzzy_match_intent
        result = _fuzzy_match_intent("xyzzy frobnicator")
        assert result is None

    def test_apply_intensity_default(self):
        """_apply_intensity at 1.0 should return original filters."""
        from opencut.core.ai_color_intent import _apply_intensity
        f = "eq=brightness=0.04:saturation=1.3"
        assert _apply_intensity(f, 1.0) == f

    def test_apply_intensity_double(self):
        """_apply_intensity at 2.0 should double deviations."""
        from opencut.core.ai_color_intent import _apply_intensity
        f = "eq=brightness=0.04:saturation=1.3"
        result = _apply_intensity(f, 2.0)
        assert "brightness=0.080" in result
        # saturation: 1.0 + (1.3-1.0)*2 = 1.6
        assert "saturation=1.600" in result

    def test_apply_intensity_half(self):
        """_apply_intensity at 0.5 should halve deviations."""
        from opencut.core.ai_color_intent import _apply_intensity
        f = "eq=brightness=0.10"
        result = _apply_intensity(f, 0.5)
        assert "brightness=0.050" in result

    def test_resolve_intent_library_match(self):
        """resolve_color_intent should match library intents."""
        from opencut.core.ai_color_intent import resolve_color_intent
        intent = resolve_color_intent("noir")
        assert intent.resolved_name == "noir"
        assert intent.source == "library"
        assert "eq=" in intent.filters

    def test_resolve_intent_unknown_fallback(self):
        """resolve_color_intent for unknown text should fall back to clean."""
        from opencut.core.ai_color_intent import resolve_color_intent
        with patch("opencut.core.ai_color_intent._resolve_intent_via_llm", return_value=None):
            intent = resolve_color_intent("completely random gibberish xyzzy")
        assert intent.resolved_name == "clean"
        assert intent.source == "library"

    def test_resolve_intent_with_intensity(self):
        """resolve_color_intent should apply intensity."""
        from opencut.core.ai_color_intent import resolve_color_intent
        intent = resolve_color_intent("noir", intensity=0.5)
        assert intent.intensity == 0.5

    def test_apply_color_intent_file_not_found(self):
        """apply_color_intent should raise FileNotFoundError."""
        from opencut.core.ai_color_intent import apply_color_intent
        with pytest.raises(FileNotFoundError):
            apply_color_intent("/nonexistent/video.mp4", intent="noir")

    def test_preview_color_intent_file_not_found(self):
        """preview_color_intent should raise FileNotFoundError."""
        from opencut.core.ai_color_intent import preview_color_intent
        with pytest.raises(FileNotFoundError):
            preview_color_intent("/nonexistent/video.mp4", intent="noir")


# ========================================================================
# 5. auto_dub_pipeline.py
# ========================================================================
class TestAutoDubPipeline:
    """Tests for opencut.core.auto_dub_pipeline."""

    def test_dub_config_defaults(self):
        """DubConfig should have sensible defaults."""
        from opencut.core.auto_dub_pipeline import DubConfig
        c = DubConfig()
        assert c.target_language == "es"
        assert c.source_language == ""
        assert c.whisper_model == "base"
        assert c.voice_clone is True
        assert c.lip_sync is True
        assert c.preserve_music is True
        assert c.tts_engine == "edge"

    def test_dub_config_to_dict(self):
        """DubConfig.to_dict() should work."""
        from opencut.core.auto_dub_pipeline import DubConfig
        c = DubConfig(target_language="fr", voice_clone=False)
        d = c.to_dict()
        assert d["target_language"] == "fr"
        assert d["voice_clone"] is False

    def test_dub_result_defaults(self):
        """DubResult should have sensible defaults."""
        from opencut.core.auto_dub_pipeline import DubResult
        r = DubResult()
        assert r.output_path == ""
        assert r.segments_dubbed == 0
        assert r.stages_completed == []
        assert r.voice_cloned is False
        assert r.lip_synced is False
        assert r.processing_time == 0.0

    def test_dub_result_to_dict(self):
        """DubResult.to_dict() should serialize all fields."""
        from opencut.core.auto_dub_pipeline import DubResult
        r = DubResult(output_path="/out.mp4", target_language="fr",
                      segments_dubbed=5, stages_completed=["extract_audio", "transcribe"])
        d = r.to_dict()
        assert d["segments_dubbed"] == 5
        assert len(d["stages_completed"]) == 2

    def test_dub_segment_defaults(self):
        """DubSegment should have correct defaults."""
        from opencut.core.auto_dub_pipeline import DubSegment
        s = DubSegment()
        assert s.original_text == ""
        assert s.translated_text == ""
        assert s.speaker_id == "speaker_0"
        assert s.synced is False

    def test_dub_segment_to_dict(self):
        """DubSegment.to_dict() should work."""
        from opencut.core.auto_dub_pipeline import DubSegment
        s = DubSegment(index=1, start=0.0, end=5.0, original_text="Hello")
        d = s.to_dict()
        assert d["original_text"] == "Hello"

    def test_supported_languages_list(self):
        """SUPPORTED_LANGUAGES should be a list with common languages."""
        from opencut.core.auto_dub_pipeline import SUPPORTED_LANGUAGES
        assert isinstance(SUPPORTED_LANGUAGES, list)
        assert "en" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) >= 40

    def test_language_names_coverage(self):
        """LANGUAGE_NAMES should cover all SUPPORTED_LANGUAGES."""
        from opencut.core.auto_dub_pipeline import LANGUAGE_NAMES, SUPPORTED_LANGUAGES
        for code in SUPPORTED_LANGUAGES:
            assert code in LANGUAGE_NAMES, f"Missing name for language: {code}"

    def test_stage_constants_defined(self):
        """Pipeline stage constants should all be defined."""
        from opencut.core.auto_dub_pipeline import (
            STAGE_COMPOSITE,
            STAGE_EXTRACT,
            STAGE_LIP_SYNC,
            STAGE_TRANSCRIBE,
            STAGE_TRANSLATE,
            STAGE_TTS,
            STAGE_VOICE_CLONE,
        )
        stages = [STAGE_EXTRACT, STAGE_TRANSCRIBE, STAGE_TRANSLATE,
                  STAGE_VOICE_CLONE, STAGE_TTS, STAGE_LIP_SYNC, STAGE_COMPOSITE]
        assert len(stages) == 7
        assert len(set(stages)) == 7  # all unique

    def test_stage_weights_cover_all_stages(self):
        """_STAGE_WEIGHTS should cover all stages."""
        from opencut.core.auto_dub_pipeline import _STAGE_ORDER, _STAGE_WEIGHTS
        for stage in _STAGE_ORDER:
            assert stage in _STAGE_WEIGHTS, f"Missing weight for stage: {stage}"

    def test_stage_progress_helper(self):
        """_stage_progress should call on_progress with scaled percentage."""
        from opencut.core.auto_dub_pipeline import STAGE_TRANSCRIBE, _stage_progress
        calls = []
        def mock_progress(pct, msg):
            calls.append((pct, msg))
        _stage_progress(mock_progress, STAGE_TRANSCRIBE, 50, "halfway")
        assert len(calls) == 1
        assert 0 <= calls[0][0] <= 100

    def test_stage_progress_none_callback(self):
        """_stage_progress should not crash with None callback."""
        from opencut.core.auto_dub_pipeline import STAGE_EXTRACT, _stage_progress
        _stage_progress(None, STAGE_EXTRACT, 50)  # should not raise

    def test_auto_dub_file_not_found(self):
        """auto_dub should raise FileNotFoundError."""
        from opencut.core.auto_dub_pipeline import auto_dub
        with pytest.raises(FileNotFoundError):
            auto_dub("/nonexistent/video.mp4")

    def test_auto_dub_unsupported_language(self):
        """auto_dub should raise ValueError for unsupported language."""
        from opencut.core.auto_dub_pipeline import auto_dub
        with pytest.raises((ValueError, FileNotFoundError)):
            auto_dub("/nonexistent/video.mp4", target_language="zz_invalid")

    @patch("opencut.core.auto_dub_pipeline.run_ffmpeg")
    @patch("opencut.core.auto_dub_pipeline.get_video_info")
    def test_extract_source_audio(self, mock_info, mock_ffmpeg):
        """_extract_source_audio should produce WAV path."""
        from opencut.core.auto_dub_pipeline import _extract_source_audio
        mock_info.return_value = {"duration": 30.0}
        mock_ffmpeg.return_value = ""

        with tempfile.TemporaryDirectory() as td:
            # Create expected output so it passes existence check
            wav_path = os.path.join(td, "source_audio.wav")
            with open(wav_path, "w") as f:
                f.write("fake")
            result = _extract_source_audio("/fake/video.mp4", td)
            assert result.endswith("source_audio.wav")

    def test_cleanup_work_dir(self):
        """_cleanup_work_dir should remove a directory."""
        from opencut.core.auto_dub_pipeline import _cleanup_work_dir
        td = tempfile.mkdtemp()
        assert os.path.isdir(td)
        _cleanup_work_dir(td)
        assert not os.path.isdir(td)

    def test_cleanup_work_dir_nonexistent(self):
        """_cleanup_work_dir should not raise for nonexistent directory."""
        from opencut.core.auto_dub_pipeline import _cleanup_work_dir
        _cleanup_work_dir("/nonexistent/path/abc123")  # should not raise


# ========================================================================
# 6. timeline_intel_routes.py
# ========================================================================
class TestTimelineIntelRoutes:
    """Smoke tests for timeline intelligence route blueprint."""

    def test_blueprint_exists(self):
        """timeline_intel_bp should be importable."""
        from opencut.routes.timeline_intel_routes import timeline_intel_bp
        assert timeline_intel_bp.name == "timeline_intel"

    def test_blueprint_registered_in_init(self):
        """timeline_intel_bp should be listed in register_blueprints."""
        import inspect

        import opencut.routes as routes_pkg
        source = inspect.getsource(routes_pkg.register_blueprints)
        assert "timeline_intel_bp" in source

    def test_route_timeline_quality_exists(self):
        """Blueprint should have /api/timeline/quality handler."""
        from opencut.routes.timeline_intel_routes import timeline_quality
        assert callable(timeline_quality)

    def test_route_timeline_score_exists(self):
        """Blueprint should have /api/timeline/score handler."""
        from opencut.routes.timeline_intel_routes import timeline_score
        assert callable(timeline_score)

    def test_route_timeline_narrative_exists(self):
        """Blueprint should have /api/timeline/narrative handler."""
        from opencut.routes.timeline_intel_routes import timeline_narrative
        assert callable(timeline_narrative)

    def test_route_video_color_intent_exists(self):
        """Blueprint should have /api/video/color-intent handler."""
        from opencut.routes.timeline_intel_routes import video_color_intent
        assert callable(video_color_intent)

    def test_route_list_color_intents_exists(self):
        """Blueprint should have /api/video/color-intents GET handler."""
        from opencut.routes.timeline_intel_routes import list_color_intents
        assert callable(list_color_intents)

    def test_route_color_intent_preview_exists(self):
        """Blueprint should have /api/video/color-intent/preview handler."""
        from opencut.routes.timeline_intel_routes import video_color_intent_preview
        assert callable(video_color_intent_preview)

    def test_route_audio_auto_dub_exists(self):
        """Blueprint should have /api/audio/auto-dub handler."""
        from opencut.routes.timeline_intel_routes import audio_auto_dub
        assert callable(audio_auto_dub)

    def test_route_list_auto_dub_languages_exists(self):
        """Blueprint should have /api/audio/auto-dub/languages handler."""
        from opencut.routes.timeline_intel_routes import list_auto_dub_languages
        assert callable(list_auto_dub_languages)

    def test_list_color_intents_response(self, client):
        """GET /api/video/color-intents should return intent list."""
        resp = client.get("/api/video/color-intents")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "intents" in data
        assert "total" in data
        assert data["total"] >= 30
        assert "categories" in data

    def test_list_auto_dub_languages_response(self, client):
        """GET /api/audio/auto-dub/languages should return language list."""
        resp = client.get("/api/audio/auto-dub/languages")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "languages" in data
        assert "total" in data
        assert data["total"] >= 40
        codes = [lang["code"] for lang in data["languages"]]
        assert "en" in codes
        assert "es" in codes

    def test_timeline_quality_no_file(self, client, csrf_token):
        """POST /api/timeline/quality without filepath should return 400."""
        from tests.conftest import csrf_headers
        resp = client.post(
            "/api/timeline/quality",
            headers=csrf_headers(csrf_token),
            data=json.dumps({}),
        )
        assert resp.status_code == 400

    def test_timeline_score_no_file(self, client, csrf_token):
        """POST /api/timeline/score without filepath should return 400."""
        from tests.conftest import csrf_headers
        resp = client.post(
            "/api/timeline/score",
            headers=csrf_headers(csrf_token),
            data=json.dumps({}),
        )
        assert resp.status_code == 400

    def test_color_intent_no_file(self, client, csrf_token):
        """POST /api/video/color-intent without filepath should return 400."""
        from tests.conftest import csrf_headers
        resp = client.post(
            "/api/video/color-intent",
            headers=csrf_headers(csrf_token),
            data=json.dumps({"intent": "noir"}),
        )
        assert resp.status_code == 400

    def test_auto_dub_no_file(self, client, csrf_token):
        """POST /api/audio/auto-dub without filepath should return 400."""
        from tests.conftest import csrf_headers
        resp = client.post(
            "/api/audio/auto-dub",
            headers=csrf_headers(csrf_token),
            data=json.dumps({"target_language": "es"}),
        )
        assert resp.status_code == 400

    def test_narrative_no_clips(self, client, csrf_token):
        """POST /api/timeline/narrative without clip_paths should start job that errors."""
        from tests.conftest import csrf_headers
        resp = client.post(
            "/api/timeline/narrative",
            headers=csrf_headers(csrf_token),
            data=json.dumps({}),
        )
        # filepath_required=False so it starts a job, but handler raises ValueError
        # which means job starts (202) then errors internally
        assert resp.status_code in (200, 202)
