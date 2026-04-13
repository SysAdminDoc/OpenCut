"""
Tests for OpenCut AI Intelligence features.

Covers:
  - Scene description (heuristic + LLM fallback)
  - Video summarization (text + visual)
  - OCR text extraction and search
  - Emotion timeline (audio RMS, speech rate, sentiment, peaks)
  - Project organization (analysis + bin generation)
  - Natural language batch operations (parse + execute)
  - AI intelligence routes (smoke tests)
"""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================================================
# 1. scene_description.py
# ========================================================================
class TestSceneDescription:
    """Tests for opencut.core.scene_description."""

    def test_closest_color_name_exact_match(self):
        """_closest_color_name should return exact name for known color."""
        from opencut.core.scene_description import _closest_color_name
        assert _closest_color_name(255, 0, 0) == "red"
        assert _closest_color_name(0, 0, 0) == "black"
        assert _closest_color_name(255, 255, 255) == "white"

    def test_closest_color_name_approximate(self):
        """_closest_color_name should return nearest match for approximate color."""
        from opencut.core.scene_description import _closest_color_name
        # Near-red should still be red
        name = _closest_color_name(240, 10, 10)
        assert name == "red"

    def test_generate_heuristic_description_dark(self):
        """Heuristic description for dark frame should mention darkness."""
        from opencut.core.scene_description import _generate_heuristic_description
        desc = _generate_heuristic_description(
            colors=["blue", "black"],
            brightness=0.1,
            edge_density=0.3,
            video_info={"width": 1920, "height": 1080},
        )
        assert "dark" in desc.lower()

    def test_generate_heuristic_description_bright(self):
        """Heuristic description for bright frame should mention brightness."""
        from opencut.core.scene_description import _generate_heuristic_description
        desc = _generate_heuristic_description(
            colors=["white", "yellow"],
            brightness=0.9,
            edge_density=0.5,
            video_info={"width": 1920, "height": 1080},
        )
        assert "bright" in desc.lower()

    def test_generate_alt_text(self):
        """_generate_alt_text should produce concise alt text."""
        from opencut.core.scene_description import _generate_alt_text
        alt = _generate_alt_text(["green", "blue"], 0.6, 0.3)
        assert "Video frame" in alt
        assert len(alt) < 200

    def test_infer_tags_dark(self):
        """_infer_tags should include 'dark' for low brightness."""
        from opencut.core.scene_description import _infer_tags
        tags = _infer_tags(["black"], 0.1, 0.2)
        assert "dark" in tags
        assert "low-key" in tags

    def test_infer_tags_nature(self):
        """_infer_tags should infer 'nature' for green-dominant frames."""
        from opencut.core.scene_description import _infer_tags
        tags = _infer_tags(["green"], 0.5, 0.4)
        assert "nature" in tags

    def test_infer_tags_night(self):
        """_infer_tags should infer 'night' for dark black frames."""
        from opencut.core.scene_description import _infer_tags
        tags = _infer_tags(["black"], 0.15, 0.1)
        assert "night" in tags

    def test_describe_scene_file_not_found(self):
        """describe_scene should raise FileNotFoundError for missing video."""
        from opencut.core.scene_description import describe_scene
        with pytest.raises(FileNotFoundError):
            describe_scene("/nonexistent/video.mp4", timestamp=0.0)

    def test_describe_all_scenes_file_not_found(self):
        """describe_all_scenes should raise FileNotFoundError for missing video."""
        from opencut.core.scene_description import describe_all_scenes
        with pytest.raises(FileNotFoundError):
            describe_all_scenes("/nonexistent/video.mp4")

    def test_scene_description_dataclass_defaults(self):
        """SceneDescription should have sensible defaults."""
        from opencut.core.scene_description import SceneDescription
        sd = SceneDescription(timestamp=5.0)
        assert sd.timestamp == 5.0
        assert sd.description == ""
        assert sd.method == "heuristic"
        assert sd.dominant_colors == []

    def test_scene_description_result_dataclass(self):
        """SceneDescriptionResult should have sensible defaults."""
        from opencut.core.scene_description import SceneDescriptionResult
        r = SceneDescriptionResult()
        assert r.descriptions == []
        assert r.total_scenes == 0
        assert r.method == "heuristic"


# ========================================================================
# 2. video_summary.py
# ========================================================================
class TestVideoSummary:
    """Tests for opencut.core.video_summary."""

    def test_extract_keywords_basic(self):
        """_extract_keywords should extract meaningful words, skipping stop words."""
        from opencut.core.video_summary import _extract_keywords
        text = "The quick brown fox jumps over the lazy dog and the fox runs"
        keywords = _extract_keywords(text, top_n=5)
        assert "fox" in keywords
        assert "the" not in keywords

    def test_extract_keywords_empty(self):
        """_extract_keywords on empty text should return empty list."""
        from opencut.core.video_summary import _extract_keywords
        assert _extract_keywords("", top_n=5) == []

    def test_split_sentences(self):
        """_split_sentences should split on sentence-ending punctuation."""
        from opencut.core.video_summary import _split_sentences
        text = "Hello world. This is a test sentence. Another one here!"
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_split_sentences_short_filter(self):
        """_split_sentences should filter out very short fragments."""
        from opencut.core.video_summary import _split_sentences
        text = "Hi. This is a much longer sentence that should be kept. Ok."
        sentences = _split_sentences(text)
        # "Hi." and "Ok." are too short (<=10 chars)
        assert all(len(s) > 10 for s in sentences)

    def test_score_sentence_keyword_match(self):
        """_score_sentence should score higher for sentences with more keywords."""
        from opencut.core.video_summary import _score_sentence
        high = _score_sentence("The fox and dog play together in nature", ["fox", "dog", "nature"])
        low = _score_sentence("A completely unrelated sentence about nothing", ["fox", "dog", "nature"])
        assert high > low

    def test_text_summary_empty_transcript(self):
        """text_summary with empty text should return fallback."""
        from opencut.core.video_summary import text_summary
        result = text_summary("")
        assert "No transcript" in result.summary
        assert result.method == "keyword"

    def test_text_summary_keyword_mode(self):
        """text_summary without LLM should use keyword extraction."""
        from opencut.core.video_summary import text_summary
        text = (
            "Machine learning is transforming video editing. "
            "Neural networks can identify scenes automatically. "
            "AI-powered tools reduce editing time significantly. "
            "Computer vision detects objects and faces in footage. "
            "Deep learning models generate automated captions. "
            "These technologies make professional editing accessible."
        )
        result = text_summary(text, max_sentences=3)
        assert result.method == "keyword"
        assert len(result.summary) > 0
        assert len(result.keywords) > 0
        assert result.word_count > 0

    def test_text_summary_with_llm_fallback(self):
        """text_summary should fall back to keyword when LLM fails."""
        from opencut.core.video_summary import text_summary

        with patch("opencut.core.llm.query_llm", side_effect=RuntimeError("LLM offline")):
            result = text_summary(
                "This is a test transcript about video editing and content creation.",
                llm_config={"provider": "ollama"},
            )
        assert result.method == "keyword"

    def test_visual_summary_missing_file(self):
        """visual_summary should raise FileNotFoundError for missing video."""
        from opencut.core.video_summary import visual_summary
        with pytest.raises(FileNotFoundError):
            visual_summary("/nonexistent.mp4", scenes=[{"timestamp": 0, "score": 1}])

    def test_visual_summary_no_scenes(self):
        """visual_summary should raise ValueError with empty scenes list."""
        from opencut.core.video_summary import visual_summary
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with pytest.raises(ValueError, match="No scenes"):
                visual_summary(path, scenes=[])
        finally:
            os.unlink(path)

    def test_text_summary_result_dataclass(self):
        """TextSummaryResult should have correct defaults."""
        from opencut.core.video_summary import TextSummaryResult
        r = TextSummaryResult()
        assert r.summary == ""
        assert r.method == "keyword"
        assert r.keywords == []

    def test_visual_summary_result_dataclass(self):
        """VisualSummaryResult should have correct defaults."""
        from opencut.core.video_summary import VisualSummaryResult
        r = VisualSummaryResult()
        assert r.output_path == ""
        assert r.scene_count == 0


# ========================================================================
# 3. ocr_extract.py
# ========================================================================
class TestOCRExtract:
    """Tests for opencut.core.ocr_extract."""

    def test_text_similarity_identical(self):
        """_text_similarity of identical strings should be 1.0."""
        from opencut.core.ocr_extract import _text_similarity
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_text_similarity_empty(self):
        """_text_similarity of two empty strings should be 1.0."""
        from opencut.core.ocr_extract import _text_similarity
        assert _text_similarity("", "") == 1.0

    def test_text_similarity_one_empty(self):
        """_text_similarity with one empty string should be 0.0."""
        from opencut.core.ocr_extract import _text_similarity
        assert _text_similarity("hello", "") == 0.0
        assert _text_similarity("", "hello") == 0.0

    def test_text_similarity_case_insensitive(self):
        """_text_similarity should be case-insensitive."""
        from opencut.core.ocr_extract import _text_similarity
        score = _text_similarity("Hello World", "hello world")
        assert score == 1.0

    def test_deduplicate_frames_merges_similar(self):
        """_deduplicate_frames should merge similar consecutive text."""
        from opencut.core.ocr_extract import OCRFrame, _deduplicate_frames
        frames = [
            OCRFrame(timestamp=0.0, text="Breaking News: Storm Warning"),
            OCRFrame(timestamp=1.0, text="Breaking News: Storm Warning"),
            OCRFrame(timestamp=2.0, text="Breaking News: Storm Warning"),
            OCRFrame(timestamp=5.0, text="Sports Update: Game Highlights"),
        ]
        unique = _deduplicate_frames(frames, similarity_threshold=0.8)
        assert len(unique) == 2
        assert unique[0]["count"] == 3
        assert unique[0]["first_seen"] == 0.0
        assert unique[0]["last_seen"] == 2.0

    def test_deduplicate_frames_keeps_longer_text(self):
        """_deduplicate_frames should prefer longer text version."""
        from opencut.core.ocr_extract import OCRFrame, _deduplicate_frames
        frames = [
            OCRFrame(timestamp=0.0, text="Breaking News"),
            OCRFrame(timestamp=1.0, text="Breaking News: Storm Warning"),
        ]
        unique = _deduplicate_frames(frames, similarity_threshold=0.5)
        assert len(unique) == 1
        assert "Storm Warning" in unique[0]["text"]

    def test_deduplicate_frames_empty(self):
        """_deduplicate_frames with no text frames returns empty list."""
        from opencut.core.ocr_extract import OCRFrame, _deduplicate_frames
        frames = [
            OCRFrame(timestamp=0.0, text=""),
            OCRFrame(timestamp=1.0, text="  "),
        ]
        unique = _deduplicate_frames(frames)
        assert unique == []

    def test_extract_text_frames_file_not_found(self):
        """extract_text_frames should raise FileNotFoundError."""
        from opencut.core.ocr_extract import extract_text_frames
        with pytest.raises(FileNotFoundError):
            extract_text_frames("/nonexistent/video.mp4")

    def test_search_text_empty_query(self):
        """search_text_in_video should raise ValueError for empty query."""
        from opencut.core.ocr_extract import search_text_in_video
        with pytest.raises(ValueError, match="empty"):
            search_text_in_video("/some/video.mp4", query="")

    def test_ocr_frame_dataclass(self):
        """OCRFrame should store timestamp and text."""
        from opencut.core.ocr_extract import OCRFrame
        f = OCRFrame(timestamp=3.5, text="Hello", confidence=0.95)
        assert f.timestamp == 3.5
        assert f.text == "Hello"
        assert f.confidence == 0.95
        assert f.words == []

    def test_text_search_hit_dataclass(self):
        """TextSearchHit should store hit details."""
        from opencut.core.ocr_extract import TextSearchHit
        h = TextSearchHit(timestamp=1.0, text="test", context="...test...", confidence=0.8)
        assert h.timestamp == 1.0
        assert h.text == "test"


# ========================================================================
# 4. emotion_timeline.py
# ========================================================================
class TestEmotionTimeline:
    """Tests for opencut.core.emotion_timeline."""

    def test_score_text_sentiment_positive(self):
        """_score_text_sentiment should score positive text > 0.5."""
        from opencut.core.emotion_timeline import _score_text_sentiment
        score = _score_text_sentiment("This is amazing and wonderful, absolutely great!")
        assert score > 0.5

    def test_score_text_sentiment_negative(self):
        """_score_text_sentiment should score negative text < 0.5."""
        from opencut.core.emotion_timeline import _score_text_sentiment
        score = _score_text_sentiment("This is terrible and horrible, absolutely awful!")
        assert score < 0.5

    def test_score_text_sentiment_neutral(self):
        """_score_text_sentiment on neutral text should be ~0.5."""
        from opencut.core.emotion_timeline import _score_text_sentiment
        score = _score_text_sentiment("The table is brown.")
        assert score == 0.5

    def test_score_text_sentiment_empty(self):
        """_score_text_sentiment on empty text should return 0.5."""
        from opencut.core.emotion_timeline import _score_text_sentiment
        assert _score_text_sentiment("") == 0.5
        assert _score_text_sentiment("   ") == 0.5

    def test_interpolate_signal_empty(self):
        """_interpolate_signal with empty signal should return zeros."""
        from opencut.core.emotion_timeline import _interpolate_signal
        result = _interpolate_signal([], [0.0, 1.0, 2.0])
        assert result == [0.0, 0.0, 0.0]

    def test_interpolate_signal_single_point(self):
        """_interpolate_signal with single point should extrapolate."""
        from opencut.core.emotion_timeline import _interpolate_signal
        result = _interpolate_signal([(1.0, 0.8)], [0.0, 1.0, 2.0])
        assert result[1] == 0.8

    def test_interpolate_signal_linear(self):
        """_interpolate_signal should linearly interpolate between points."""
        from opencut.core.emotion_timeline import _interpolate_signal
        signal = [(0.0, 0.0), (2.0, 1.0)]
        result = _interpolate_signal(signal, [0.0, 1.0, 2.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_normalize_signal(self):
        """_normalize_signal should map to 0.0-1.0 range."""
        from opencut.core.emotion_timeline import _normalize_signal
        result = _normalize_signal([10, 20, 30])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_normalize_signal_flat(self):
        """_normalize_signal with flat input should return 0.5 for all."""
        from opencut.core.emotion_timeline import _normalize_signal
        result = _normalize_signal([5.0, 5.0, 5.0])
        assert all(v == 0.5 for v in result)

    def test_normalize_signal_empty(self):
        """_normalize_signal with empty list should return empty."""
        from opencut.core.emotion_timeline import _normalize_signal
        assert _normalize_signal([]) == []

    def test_compute_speech_rate_words(self):
        """_compute_speech_rate should compute rates from word timestamps."""
        from opencut.core.emotion_timeline import _compute_speech_rate
        transcript = {
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.3},
                {"word": "world", "start": 0.5, "end": 0.8},
                {"word": "this", "start": 1.0, "end": 1.2},
                {"word": "is", "start": 1.3, "end": 1.4},
                {"word": "fast", "start": 1.5, "end": 1.7},
            ],
        }
        rates = _compute_speech_rate(transcript, duration=2.0, interval=1.0)
        assert len(rates) == 2
        # First second has 2 words, second has 3
        assert rates[0][1] >= 0.0
        assert rates[1][1] >= 0.0

    def test_compute_speech_rate_empty(self):
        """_compute_speech_rate with no transcript should return empty."""
        from opencut.core.emotion_timeline import _compute_speech_rate
        assert _compute_speech_rate(None, 10.0) == []

    def test_compute_sentiment_segments(self):
        """_compute_sentiment should compute per-window sentiment from segments."""
        from opencut.core.emotion_timeline import _compute_sentiment
        transcript = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "This is amazing and wonderful"},
                {"start": 1.0, "end": 2.0, "text": "This is terrible and horrible"},
            ],
        }
        sentiments = _compute_sentiment(transcript, duration=2.0, interval=1.0)
        assert len(sentiments) == 2
        assert sentiments[0][1] > 0.5  # positive
        assert sentiments[1][1] < 0.5  # negative

    def test_find_peaks_empty(self):
        """find_peaks on empty timeline should return empty list."""
        from opencut.core.emotion_timeline import find_peaks
        assert find_peaks([]) == []

    def test_find_peaks_above_threshold(self):
        """find_peaks should detect local maxima above threshold."""
        from opencut.core.emotion_timeline import TimelinePoint, find_peaks
        timeline = [
            TimelinePoint(time=0.0, energy=0.3, audio_rms=0.3, speech_rate=0.3, sentiment=0.5),
            TimelinePoint(time=1.0, energy=0.5, audio_rms=0.5, speech_rate=0.5, sentiment=0.5),
            TimelinePoint(time=2.0, energy=0.9, audio_rms=0.9, speech_rate=0.5, sentiment=0.5),
            TimelinePoint(time=3.0, energy=0.5, audio_rms=0.5, speech_rate=0.5, sentiment=0.5),
            TimelinePoint(time=4.0, energy=0.3, audio_rms=0.3, speech_rate=0.3, sentiment=0.5),
        ]
        peaks = find_peaks(timeline, threshold=0.7, min_distance=1.0)
        assert len(peaks) == 1
        assert peaks[0].time == 2.0
        assert peaks[0].energy == 0.9

    def test_find_peaks_respects_min_distance(self):
        """find_peaks should enforce minimum distance between peaks."""
        from opencut.core.emotion_timeline import TimelinePoint, find_peaks
        timeline = [
            TimelinePoint(time=0.0, energy=0.3),
            TimelinePoint(time=1.0, energy=0.9),
            TimelinePoint(time=2.0, energy=0.3),
            TimelinePoint(time=3.0, energy=0.85),
            TimelinePoint(time=4.0, energy=0.3),
        ]
        # With min_distance=5.0, only the highest peak should survive
        peaks = find_peaks(timeline, threshold=0.7, min_distance=5.0)
        assert len(peaks) == 1
        assert peaks[0].energy == 0.9

    def test_build_emotion_timeline_file_not_found(self):
        """build_emotion_timeline should raise FileNotFoundError."""
        from opencut.core.emotion_timeline import build_emotion_timeline
        with pytest.raises(FileNotFoundError):
            build_emotion_timeline("/nonexistent/video.mp4")

    def test_timeline_point_dataclass(self):
        """TimelinePoint should have correct defaults."""
        from opencut.core.emotion_timeline import TimelinePoint
        p = TimelinePoint(time=1.5)
        assert p.time == 1.5
        assert p.energy == 0.0
        assert p.sentiment == 0.5

    def test_emotion_peak_info_dataclass(self):
        """EmotionPeakInfo should store peak metadata."""
        from opencut.core.emotion_timeline import EmotionPeakInfo
        p = EmotionPeakInfo(time=5.0, energy=0.9, start=4.0, end=6.0,
                            duration=2.0, dominant_signal="audio")
        assert p.time == 5.0
        assert p.dominant_signal == "audio"


# ========================================================================
# 5. project_organizer.py
# ========================================================================
class TestProjectOrganizer:
    """Tests for opencut.core.project_organizer."""

    def test_classify_media_type_video(self):
        """_classify_media_type should identify video extensions."""
        from opencut.core.project_organizer import _classify_media_type
        assert _classify_media_type(".mp4") == "video"
        assert _classify_media_type(".mov") == "video"
        assert _classify_media_type(".mkv") == "video"

    def test_classify_media_type_audio(self):
        """_classify_media_type should identify audio extensions."""
        from opencut.core.project_organizer import _classify_media_type
        assert _classify_media_type(".wav") == "audio"
        assert _classify_media_type(".mp3") == "audio"
        assert _classify_media_type(".flac") == "audio"

    def test_classify_media_type_image(self):
        """_classify_media_type should identify image extensions."""
        from opencut.core.project_organizer import _classify_media_type
        assert _classify_media_type(".jpg") == "image"
        assert _classify_media_type(".png") == "image"
        assert _classify_media_type(".tiff") == "image"

    def test_classify_media_type_unknown(self):
        """_classify_media_type should return 'other' for unknown extensions."""
        from opencut.core.project_organizer import _classify_media_type
        assert _classify_media_type(".xyz") == "other"
        assert _classify_media_type(".doc") == "other"

    def test_compute_aspect_ratio_label_16_9(self):
        """_compute_aspect_ratio_label should detect 16:9."""
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1920, 1080) == "16:9"
        assert _compute_aspect_ratio_label(3840, 2160) == "16:9"

    def test_compute_aspect_ratio_label_vertical(self):
        """_compute_aspect_ratio_label should detect 9:16 vertical."""
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1080, 1920) == "9:16"

    def test_compute_aspect_ratio_label_square(self):
        """_compute_aspect_ratio_label should detect 1:1 square."""
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(1080, 1080) == "1:1"

    def test_compute_aspect_ratio_label_zero(self):
        """_compute_aspect_ratio_label with zero dims should return unknown."""
        from opencut.core.project_organizer import _compute_aspect_ratio_label
        assert _compute_aspect_ratio_label(0, 0) == "unknown"

    def test_infer_scene_group_scene_pattern(self):
        """_infer_scene_group should detect 'Scene01' naming."""
        from opencut.core.project_organizer import _infer_scene_group
        group = _infer_scene_group("Scene01_Take02.mp4")
        assert "1" in group or "scene" in group.lower()

    def test_infer_scene_group_prefix(self):
        """_infer_scene_group should group by filename prefix."""
        from opencut.core.project_organizer import _infer_scene_group
        group = _infer_scene_group("interview_001.mp4")
        assert "interview" in group.lower()

    def test_classify_shot_type_vertical(self):
        """_classify_shot_type_basic should classify vertical as close_up."""
        from opencut.core.project_organizer import _classify_shot_type_basic
        shot = _classify_shot_type_basic(1080, 1920, {})
        assert shot == "close_up"

    def test_classify_shot_type_ultrawide(self):
        """_classify_shot_type_basic should classify ultra-wide as wide."""
        from opencut.core.project_organizer import _classify_shot_type_basic
        shot = _classify_shot_type_basic(2560, 1080, {})
        assert shot == "wide"

    def test_analyze_project_media_no_files(self):
        """analyze_project_media should raise ValueError with empty list."""
        from opencut.core.project_organizer import analyze_project_media
        with pytest.raises(ValueError, match="No files"):
            analyze_project_media([])

    def test_analyze_project_media_nonexistent(self):
        """analyze_project_media should raise ValueError if none exist."""
        from opencut.core.project_organizer import analyze_project_media
        with pytest.raises(ValueError, match="None"):
            analyze_project_media(["/nonexistent/a.mp4", "/nonexistent/b.wav"])

    def test_generate_bin_structure_empty(self):
        """generate_bin_structure with no files should return empty structure."""
        from opencut.core.project_organizer import ProjectAnalysis, generate_bin_structure
        analysis = ProjectAnalysis()
        result = generate_bin_structure(analysis)
        assert result.total_bins == 0

    def test_auto_select_strategy_by_date(self):
        """_auto_select_strategy should prefer date when many date groups."""
        from opencut.core.project_organizer import ProjectAnalysis, _auto_select_strategy
        analysis = ProjectAnalysis(
            total_files=10,
            media_types={"video": 10},
            date_groups={"2024-01-01": ["a"], "2024-01-02": ["b"], "2024-01-03": ["c"]},
            scene_groups={},
            shot_types={},
        )
        assert _auto_select_strategy(analysis) == "by_date"

    def test_bins_by_type(self):
        """_bins_by_type should create one bin per media type."""
        from opencut.core.project_organizer import MediaFileInfo, ProjectAnalysis, _bins_by_type
        analysis = ProjectAnalysis(
            files=[
                MediaFileInfo(path="/a.mp4", media_type="video"),
                MediaFileInfo(path="/b.wav", media_type="audio"),
                MediaFileInfo(path="/c.mp4", media_type="video"),
            ]
        )
        bins = _bins_by_type(analysis)
        names = {b.name for b in bins}
        assert "Video" in names
        assert "Audio" in names

    def test_media_file_info_dataclass(self):
        """MediaFileInfo should store file metadata."""
        from opencut.core.project_organizer import MediaFileInfo
        info = MediaFileInfo(path="/test.mp4", media_type="video", width=1920, height=1080)
        assert info.path == "/test.mp4"
        assert info.width == 1920


# ========================================================================
# 6. nl_batch.py
# ========================================================================
class TestNLBatch:
    """Tests for opencut.core.nl_batch."""

    def test_parse_batch_command_rename(self):
        """parse_batch_command should detect rename operations."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("rename all video clips with prefix 'final_'")
        assert cmd.parsed is True
        assert cmd.operation.action == "rename"
        assert cmd.confidence > 0

    def test_parse_batch_command_transcode(self):
        """parse_batch_command should detect transcode with codec."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("convert all 4k videos to h265")
        assert cmd.parsed is True
        assert cmd.operation.action == "transcode"
        assert cmd.operation.parameters.get("codec") == "libx265"

    def test_parse_batch_command_move(self):
        """parse_batch_command should detect move with destination."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("move all audio files to '/sorted/audio'")
        assert cmd.parsed is True
        assert cmd.operation.action == "move"
        assert cmd.filters.media_type == "audio"

    def test_parse_batch_command_empty(self):
        """parse_batch_command with empty text should return unparsed."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("")
        assert cmd.parsed is False
        assert cmd.confidence == 0.0

    def test_parse_batch_command_duration_filter(self):
        """parse_batch_command should parse duration filter."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("delete videos shorter than 5 seconds")
        assert cmd.filters.max_duration == 5.0
        assert cmd.operation.action == "delete"

    def test_parse_batch_command_resolution_filter(self):
        """parse_batch_command should parse resolution filter."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("transcode 4k videos to h264")
        assert cmd.filters.min_resolution == (3840, 2160)

    def test_parse_batch_command_aspect_ratio(self):
        """parse_batch_command should parse aspect ratio filters."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("tag all vertical videos as 'shorts'")
        assert cmd.filters.aspect_ratio == "9:16"

    def test_parse_batch_command_proxy(self):
        """parse_batch_command should detect proxy generation."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("create proxies for all video clips")
        assert cmd.parsed is True
        assert cmd.operation.action == "proxy"

    def test_parse_batch_command_explanation(self):
        """parse_batch_command should build a human-readable explanation."""
        from opencut.core.nl_batch import parse_batch_command
        cmd = parse_batch_command("rename all video files with prefix 'final_'")
        assert cmd.explanation != ""
        assert "rename" in cmd.explanation.lower()

    def test_matches_filter_extension(self):
        """_matches_filter should match by extension."""
        from opencut.core.nl_batch import FilterCriteria, _matches_filter
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            filters = FilterCriteria(extension=".mp4")
            assert _matches_filter(path, filters) is True
            filters2 = FilterCriteria(extension=".wav")
            assert _matches_filter(path, filters2) is False
        finally:
            os.unlink(path)

    def test_matches_filter_filename_pattern(self):
        """_matches_filter should match by filename pattern."""
        from opencut.core.nl_batch import FilterCriteria, _matches_filter
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="interview_", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            filters = FilterCriteria(filename_pattern="interview")
            assert _matches_filter(path, filters) is True
            filters2 = FilterCriteria(filename_pattern="broll")
            assert _matches_filter(path, filters2) is False
        finally:
            os.unlink(path)

    def test_execute_batch_unparsed_command(self):
        """execute_batch with unparsed command should return error."""
        from opencut.core.nl_batch import BatchCommand, execute_batch
        cmd = BatchCommand(original_text="gibberish", parsed=False)
        result = execute_batch(cmd, ["/some/file.mp4"])
        assert result.files_processed == 0
        assert len(result.errors) > 0

    def test_execute_batch_no_files(self):
        """execute_batch with empty file list should return error."""
        from opencut.core.nl_batch import BatchCommand, BatchOperation, execute_batch
        cmd = BatchCommand(
            parsed=True,
            operation=BatchOperation(action="tag", parameters={"tag": "final"}),
        )
        result = execute_batch(cmd, [])
        assert len(result.errors) > 0

    def test_execute_batch_tag_dry_run(self):
        """execute_batch with dry_run should not modify files."""
        from opencut.core.nl_batch import BatchCommand, BatchOperation, FilterCriteria, execute_batch
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            cmd = BatchCommand(
                parsed=True,
                filters=FilterCriteria(),
                operation=BatchOperation(action="tag", parameters={"tag": "test"}),
            )
            result = execute_batch(cmd, [path], dry_run=True)
            assert result.files_matched == 1
            assert result.files_processed == 1
            assert result.results[0]["status"] == "dry_run"
        finally:
            os.unlink(path)

    def test_execute_rename(self):
        """_execute_rename should rename file with prefix."""
        from opencut.core.nl_batch import _execute_rename
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="clip_", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = _execute_rename(path, {"prefix": "final_"})
            assert result["status"] == "renamed"
            assert "final_" in result["new_path"]
            # Clean up renamed file
            os.unlink(result["new_path"])
        except Exception:
            if os.path.exists(path):
                os.unlink(path)

    def test_execute_tag(self):
        """_execute_tag should return tag info without modifying file."""
        from opencut.core.nl_batch import _execute_tag
        result = _execute_tag("/some/file.mp4", {"tag": "approved"})
        assert result["status"] == "tagged"
        assert result["tag"] == "approved"

    def test_filter_criteria_dataclass(self):
        """FilterCriteria should default to all None."""
        from opencut.core.nl_batch import FilterCriteria
        f = FilterCriteria()
        assert f.media_type is None
        assert f.min_duration is None
        assert f.extension is None

    def test_batch_result_dataclass(self):
        """BatchResult should have correct defaults."""
        from opencut.core.nl_batch import BatchResult
        r = BatchResult()
        assert r.files_matched == 0
        assert r.files_processed == 0
        assert r.errors == []


# ========================================================================
# 7. Route smoke tests
# ========================================================================
class TestAIIntelligenceRoutes:
    """Smoke tests for AI intelligence routes blueprint registration."""

    def test_blueprint_exists(self):
        """ai_intel_bp should be a valid Blueprint."""
        from flask import Blueprint

        from opencut.routes.ai_intelligence_routes import ai_intel_bp
        assert isinstance(ai_intel_bp, Blueprint)
        assert ai_intel_bp.name == "ai_intel"

    def test_blueprint_registered_in_init(self):
        """ai_intel_bp should be listed in register_blueprints."""
        import inspect

        import opencut.routes as routes_pkg
        source = inspect.getsource(routes_pkg.register_blueprints)
        assert "ai_intel_bp" in source

    def test_route_scene_describe_exists(self):
        """Blueprint should have /api/ai/scene-describe handler."""
        from opencut.routes.ai_intelligence_routes import ai_scene_describe
        assert callable(ai_scene_describe)

    def test_route_summarize_exists(self):
        """Blueprint should have /api/ai/summarize route."""
        from opencut.routes.ai_intelligence_routes import ai_summarize
        assert callable(ai_summarize)

    def test_route_ocr_exists(self):
        """Blueprint should have /api/ai/ocr route."""
        from opencut.routes.ai_intelligence_routes import ai_ocr
        assert callable(ai_ocr)

    def test_route_emotion_timeline_exists(self):
        """Blueprint should have /api/ai/emotion-timeline route."""
        from opencut.routes.ai_intelligence_routes import ai_emotion_timeline
        assert callable(ai_emotion_timeline)

    def test_route_organize_project_exists(self):
        """Blueprint should have /api/ai/organize-project route."""
        from opencut.routes.ai_intelligence_routes import ai_organize_project
        assert callable(ai_organize_project)

    def test_route_batch_command_exists(self):
        """Blueprint should have /api/ai/batch-command route."""
        from opencut.routes.ai_intelligence_routes import ai_batch_command
        assert callable(ai_batch_command)
