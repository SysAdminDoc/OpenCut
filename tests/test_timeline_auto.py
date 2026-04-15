"""
Tests for OpenCut Advanced Timeline Automation (Category 74).

Covers:
  - Auto rough cut assembly (script parsing, keyword matching, clip selection,
    EDL generation, all assembly modes, LLM fallback)
  - Auto-mix (loudness analysis, ducking profiles, keyframe generation,
    level matching, track analysis, preview)
  - Smart trim (silence detection, speech regions, scene changes, all modes,
    batch processing, alternative points)
  - Batch timeline operations (each op type, pipeline chaining, dry-run,
    clip parsing, operation dispatch)
  - Template assembly (built-in templates, validation, slot fitting,
    assembly, EDL/OTIO export, media map)
  - Timeline automation routes (smoke tests for all endpoints)
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
# 1. auto_rough_cut.py
# ========================================================================
class TestAutoRoughCut:
    """Tests for opencut.core.auto_rough_cut."""

    def test_script_section_defaults(self):
        from opencut.core.auto_rough_cut import ScriptSection
        s = ScriptSection()
        assert s.index == 0
        assert s.text == ""
        assert s.keywords == []
        assert s.duration_hint == 0.0
        assert s.required is True

    def test_script_section_to_dict(self):
        from opencut.core.auto_rough_cut import ScriptSection
        s = ScriptSection(index=1, text="Hello world", keywords=["hello"])
        d = s.to_dict()
        assert d["index"] == 1
        assert d["text"] == "Hello world"
        assert d["keywords"] == ["hello"]

    def test_transcript_segment_defaults(self):
        from opencut.core.auto_rough_cut import TranscriptSegment
        ts = TranscriptSegment()
        assert ts.text == ""
        assert ts.duration == 0.0

    def test_transcript_segment_duration(self):
        from opencut.core.auto_rough_cut import TranscriptSegment
        ts = TranscriptSegment(start=1.0, end=5.0)
        assert ts.duration == 4.0

    def test_clip_analysis_defaults(self):
        from opencut.core.auto_rough_cut import ClipAnalysis
        ca = ClipAnalysis()
        assert ca.file_path == ""
        assert ca.duration == 0.0
        assert ca.width == 1920
        assert ca.has_audio is True

    def test_clip_analysis_to_dict(self):
        from opencut.core.auto_rough_cut import ClipAnalysis
        ca = ClipAnalysis(file_path="/test.mp4", duration=10.0)
        d = ca.to_dict()
        assert d["file_path"] == "/test.mp4"
        assert d["duration"] == 10.0

    def test_candidate_clip_duration(self):
        from opencut.core.auto_rough_cut import CandidateClip
        cc = CandidateClip(start=2.0, end=7.0)
        assert cc.duration == 5.0

    def test_candidate_clip_to_dict(self):
        from opencut.core.auto_rough_cut import CandidateClip
        cc = CandidateClip(source_file="/clip.mp4", score=0.8)
        d = cc.to_dict()
        assert d["score"] == 0.8

    def test_cut_entry_duration(self):
        from opencut.core.auto_rough_cut import CutEntry
        ce = CutEntry(source_in=0.0, source_out=5.0)
        assert ce.duration == 5.0

    def test_cut_entry_to_dict(self):
        from opencut.core.auto_rough_cut import CutEntry
        ce = CutEntry(order=0, source_file="/a.mp4", source_in=0, source_out=5)
        d = ce.to_dict()
        assert d["order"] == 0
        assert d["duration"] == 5.0

    def test_rough_cut_result_defaults(self):
        from opencut.core.auto_rough_cut import RoughCutResult
        r = RoughCutResult()
        assert r.cuts == []
        assert r.total_duration == 0.0
        assert r.mode == "strict"
        assert r.llm_used is False

    def test_rough_cut_result_to_dict(self):
        from opencut.core.auto_rough_cut import RoughCutResult
        r = RoughCutResult(total_duration=60.0, mode="loose")
        d = r.to_dict()
        assert d["total_duration"] == 60.0
        assert d["mode"] == "loose"
        assert d["clip_count"] == 0

    def test_parse_script_empty(self):
        from opencut.core.auto_rough_cut import parse_script
        assert parse_script("") == []
        assert parse_script("   ") == []

    def test_parse_script_single_section(self):
        from opencut.core.auto_rough_cut import parse_script
        sections = parse_script("This is a single section.")
        assert len(sections) >= 1
        assert sections[0].index == 0

    def test_parse_script_double_newline_split(self):
        from opencut.core.auto_rough_cut import parse_script
        text = "Section one text here.\n\nSection two follows."
        sections = parse_script(text)
        assert len(sections) == 2

    def test_parse_script_numbered_sections(self):
        from opencut.core.auto_rough_cut import parse_script
        text = "1. Introduction to the topic\n2. Main content here\n3. Conclusion"
        sections = parse_script(text)
        assert len(sections) >= 2

    def test_extract_keywords(self):
        from opencut.core.auto_rough_cut import _extract_keywords
        kws = _extract_keywords("The camera pans across the beautiful mountain landscape")
        assert "camera" in kws
        assert "mountain" in kws
        assert "landscape" in kws
        # Stop words excluded
        assert "the" not in kws
        assert "for" not in kws

    def test_extract_keywords_short_words_excluded(self):
        from opencut.core.auto_rough_cut import _extract_keywords
        kws = _extract_keywords("I am at it")
        # All words < 3 chars or stop words
        assert len(kws) == 0

    def test_compute_keyword_score_no_keywords(self):
        from opencut.core.auto_rough_cut import ClipAnalysis, ScriptSection, _compute_keyword_score
        section = ScriptSection(keywords=[])
        clip = ClipAnalysis(transcript="some text")
        assert _compute_keyword_score(section, clip) == 0.0

    def test_compute_keyword_score_match(self):
        from opencut.core.auto_rough_cut import ClipAnalysis, ScriptSection, _compute_keyword_score
        section = ScriptSection(keywords=["camera", "mountain"])
        clip = ClipAnalysis(transcript="the camera on the mountain top")
        score = _compute_keyword_score(section, clip)
        assert score > 0.0

    def test_compute_keyword_score_no_match(self):
        from opencut.core.auto_rough_cut import ClipAnalysis, ScriptSection, _compute_keyword_score
        section = ScriptSection(keywords=["ocean", "waves"])
        clip = ClipAnalysis(transcript="the camera on the mountain top")
        score = _compute_keyword_score(section, clip)
        assert score == 0.0

    def test_format_timecode(self):
        from opencut.core.auto_rough_cut import _format_timecode
        assert _format_timecode(0.0) == "00:00:00:00"
        assert _format_timecode(3661.5, fps=30.0) == "01:01:01:15"

    def test_generate_edl_empty(self):
        from opencut.core.auto_rough_cut import generate_edl
        edl = generate_edl([])
        assert "TITLE:" in edl
        assert "FCM:" in edl

    def test_generate_edl_with_cuts(self):
        from opencut.core.auto_rough_cut import CutEntry, generate_edl
        cuts = [
            CutEntry(order=0, source_file="clip1.mp4",
                     source_in=0, source_out=5,
                     record_in=0, record_out=5),
            CutEntry(order=1, source_file="clip2.mp4",
                     source_in=0, source_out=3,
                     record_in=5, record_out=8,
                     transition="dissolve", transition_duration=0.5),
        ]
        edl = generate_edl(cuts)
        assert "001" in edl
        assert "002" in edl
        assert "clip1.mp4" in edl
        assert "clip2.mp4" in edl

    @patch("opencut.core.auto_rough_cut.get_video_info")
    @patch("opencut.core.auto_rough_cut._sp.run")
    def test_assemble_rough_cut_strict(self, mock_run, mock_info):
        from opencut.core.auto_rough_cut import assemble_rough_cut
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 30.0}
        mock_run.return_value = MagicMock(returncode=0, stdout=b'{"streams":[]}', stderr=b"")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        try:
            result = assemble_rough_cut(
                script_text="First section\n\nSecond section",
                media_paths=[tmp],
                mode="strict",
                target_duration=60.0,
            )
            assert result.mode == "strict"
            assert isinstance(result.edl_text, str)
        finally:
            os.unlink(tmp)

    @patch("opencut.core.auto_rough_cut.get_video_info")
    @patch("opencut.core.auto_rough_cut._sp.run")
    def test_assemble_rough_cut_highlight(self, mock_run, mock_info):
        from opencut.core.auto_rough_cut import assemble_rough_cut
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 20.0}
        mock_run.return_value = MagicMock(returncode=0, stdout=b'{"streams":[]}', stderr=b"")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        try:
            result = assemble_rough_cut(
                script_text="",
                media_paths=[tmp],
                mode="highlight",
                target_duration=30.0,
            )
            assert result.mode == "highlight"
        finally:
            os.unlink(tmp)

    def test_assemble_rough_cut_bad_mode(self):
        from opencut.core.auto_rough_cut import assemble_rough_cut
        with pytest.raises(ValueError, match="Unknown mode"):
            assemble_rough_cut("text", ["/fake.mp4"], mode="invalid")

    def test_assemble_rough_cut_no_media(self):
        from opencut.core.auto_rough_cut import assemble_rough_cut
        with pytest.raises(ValueError, match="No media files"):
            assemble_rough_cut("text", [], mode="strict")

    def test_match_sections_keywords(self):
        from opencut.core.auto_rough_cut import (
            ClipAnalysis,
            ScriptSection,
            _match_sections_keywords,
        )
        sections = [ScriptSection(index=0, text="mountain camera landscape", keywords=["mountain", "camera"])]
        analyses = [ClipAnalysis(file_path="/a.mp4", duration=10.0, transcript="beautiful mountain camera view")]
        candidates = _match_sections_keywords(sections, analyses)
        assert 0 in candidates
        assert len(candidates[0]) > 0

    def test_select_best_clips_strict(self):
        from opencut.core.auto_rough_cut import CandidateClip, ScriptSection, _select_best_clips
        sections = [ScriptSection(index=0, text="test", duration_hint=10.0)]
        candidates = {0: [CandidateClip(source_file="/a.mp4", start=0, end=10, score=0.9)]}
        cuts, unmatched, used = _select_best_clips(sections, candidates, mode="strict")
        assert len(cuts) == 1
        assert len(unmatched) == 0

    def test_select_best_clips_strict_unmatched(self):
        from opencut.core.auto_rough_cut import ScriptSection, _select_best_clips
        sections = [ScriptSection(index=0, text="test")]
        cuts, unmatched, used = _select_best_clips(sections, {}, mode="strict")
        assert len(cuts) == 0
        assert 0 in unmatched


# ========================================================================
# 2. auto_mix.py
# ========================================================================
class TestAutoMix:
    """Tests for opencut.core.auto_mix."""

    def test_ducking_profiles_exist(self):
        from opencut.core.auto_mix import DUCKING_PROFILES
        assert "podcast" in DUCKING_PROFILES
        assert "film" in DUCKING_PROFILES
        assert "music_video" in DUCKING_PROFILES

    def test_ducking_profile_keys(self):
        from opencut.core.auto_mix import DUCKING_PROFILES
        required_keys = {"duck_amount_db", "attack_ms", "release_ms", "threshold_db",
                         "hold_ms", "dialogue_target_lufs", "music_target_lufs"}
        for name, profile in DUCKING_PROFILES.items():
            for key in required_keys:
                assert key in profile, f"Missing '{key}' in profile '{name}'"

    def test_music_video_no_duck(self):
        from opencut.core.auto_mix import DUCKING_PROFILES
        assert DUCKING_PROFILES["music_video"]["duck_amount_db"] == 0.0

    def test_track_analysis_defaults(self):
        from opencut.core.auto_mix import TrackAnalysis
        ta = TrackAnalysis()
        assert ta.duration == 0.0
        assert ta.loudness_lufs == -23.0
        assert ta.track_type == "dialogue"

    def test_track_analysis_to_dict(self):
        from opencut.core.auto_mix import TrackAnalysis
        ta = TrackAnalysis(file_path="/audio.wav", duration=60.0, loudness_lufs=-18.0)
        d = ta.to_dict()
        assert d["duration"] == 60.0
        assert d["loudness_lufs"] == -18.0

    def test_gain_keyframe_defaults(self):
        from opencut.core.auto_mix import GainKeyframe
        gk = GainKeyframe()
        assert gk.time == 0.0
        assert gk.gain_db == 0.0
        assert gk.is_ducking is False

    def test_gain_keyframe_to_dict(self):
        from opencut.core.auto_mix import GainKeyframe
        gk = GainKeyframe(time=1.5, gain_db=-12.0, is_ducking=True)
        d = gk.to_dict()
        assert d["time"] == 1.5
        assert d["gain_db"] == -12.0
        assert d["is_ducking"] is True

    def test_track_mix_params_defaults(self):
        from opencut.core.auto_mix import TrackMixParams
        mp = TrackMixParams()
        assert mp.base_gain_db == 0.0
        assert mp.keyframes == []

    def test_track_mix_params_to_dict(self):
        from opencut.core.auto_mix import TrackMixParams
        mp = TrackMixParams(file_path="/a.wav", track_type="music", base_gain_db=-3.0)
        d = mp.to_dict()
        assert d["track_type"] == "music"
        assert d["base_gain_db"] == -3.0

    def test_auto_mix_result_defaults(self):
        from opencut.core.auto_mix import AutoMixResult
        r = AutoMixResult()
        assert r.keyframes == {}
        assert r.track_analysis == []
        assert r.profile == "podcast"

    def test_auto_mix_result_to_dict(self):
        from opencut.core.auto_mix import AutoMixResult
        r = AutoMixResult(profile="film", duration=120.0)
        d = r.to_dict()
        assert d["profile"] == "film"
        assert d["duration"] == 120.0
        assert d["track_count"] == 0

    def test_compute_gain_adjustment(self):
        from opencut.core.auto_mix import _compute_gain_adjustment
        assert _compute_gain_adjustment(-23.0, -16.0) == 7.0
        assert _compute_gain_adjustment(-16.0, -16.0) == 0.0
        assert _compute_gain_adjustment(-70.1, -16.0) == 0.0  # silence

    def test_compute_gain_adjustment_clamped(self):
        from opencut.core.auto_mix import _compute_gain_adjustment
        result = _compute_gain_adjustment(-60.0, 0.0)
        assert result <= 30.0

    def test_level_match_tracks(self):
        from opencut.core.auto_mix import DUCKING_PROFILES, TrackAnalysis, _level_match_tracks
        profile = DUCKING_PROFILES["podcast"]
        tracks = [
            TrackAnalysis(track_type="dialogue", loudness_lufs=-23.0),
            TrackAnalysis(track_type="music", loudness_lufs=-20.0),
        ]
        result = _level_match_tracks(tracks, profile)
        assert result[0].gain_adjustment_db != 0.0 or result[0].loudness_lufs == profile["dialogue_target_lufs"]
        assert result[1].gain_adjustment_db != 0.0

    def test_find_dialogue_active_regions_empty(self):
        from opencut.core.auto_mix import _find_dialogue_active_regions
        assert _find_dialogue_active_regions([]) == []

    def test_find_dialogue_active_regions_merges(self):
        from opencut.core.auto_mix import TrackAnalysis, _find_dialogue_active_regions
        tracks = [
            TrackAnalysis(speech_segments=[
                {"start": 0, "end": 5, "duration": 5},
                {"start": 4.9, "end": 10, "duration": 5.1},
            ]),
        ]
        regions = _find_dialogue_active_regions(tracks)
        assert len(regions) == 1
        assert regions[0]["start"] == 0
        assert regions[0]["end"] == 10

    def test_generate_ducking_keyframes_no_duck(self):
        from opencut.core.auto_mix import (
            DUCKING_PROFILES,
            TrackAnalysis,
            _generate_ducking_keyframes,
        )
        profile = DUCKING_PROFILES["music_video"]
        track = TrackAnalysis(duration=30.0)
        regions = [{"start": 5, "end": 15, "duration": 10}]
        kfs = _generate_ducking_keyframes(track, regions, profile)
        assert len(kfs) == 0  # music_video has 0 duck

    def test_generate_ducking_keyframes_podcast(self):
        from opencut.core.auto_mix import (
            DUCKING_PROFILES,
            TrackAnalysis,
            _generate_ducking_keyframes,
        )
        profile = DUCKING_PROFILES["podcast"]
        track = TrackAnalysis(duration=30.0, track_type="music")
        regions = [{"start": 5, "end": 15, "duration": 10}]
        kfs = _generate_ducking_keyframes(track, regions, profile)
        assert len(kfs) > 0
        # Should have ducking keyframes
        ducking_kfs = [k for k in kfs if k.is_ducking]
        assert len(ducking_kfs) > 0

    def test_suggest_eq(self):
        from opencut.core.auto_mix import _suggest_eq
        eq = _suggest_eq("dialogue")
        assert "high_pass_hz" in eq
        assert eq["high_pass_hz"] == 80

    def test_suggest_compressor(self):
        from opencut.core.auto_mix import _suggest_compressor
        comp = _suggest_compressor("dialogue")
        assert "threshold_db" in comp
        assert "ratio" in comp

    def test_auto_mix_bad_profile(self):
        from opencut.core.auto_mix import auto_mix
        with pytest.raises(ValueError, match="Unknown profile"):
            auto_mix(tracks=[{"file_path": "/a.wav", "type": "dialogue"}], profile="invalid")

    def test_auto_mix_no_tracks(self):
        from opencut.core.auto_mix import auto_mix
        with pytest.raises(ValueError, match="No tracks"):
            auto_mix(tracks=[], profile="podcast")


# ========================================================================
# 3. smart_trim.py
# ========================================================================
class TestSmartTrim:
    """Tests for opencut.core.smart_trim."""

    def test_trim_modes_exist(self):
        from opencut.core.smart_trim import TRIM_MODES
        assert "tight" in TRIM_MODES
        assert "broadcast" in TRIM_MODES
        assert "social" in TRIM_MODES

    def test_trim_mode_keys(self):
        from opencut.core.smart_trim import TRIM_MODES
        required_keys = {"pre_pad_s", "post_pad_s", "silence_threshold_db",
                         "min_speech_duration_s", "scene_detect_threshold"}
        for name, mode in TRIM_MODES.items():
            for key in required_keys:
                assert key in mode, f"Missing '{key}' in mode '{name}'"

    def test_trim_point_defaults(self):
        from opencut.core.smart_trim import TrimPoint
        tp = TrimPoint()
        assert tp.time == 0.0
        assert tp.confidence == 0.0

    def test_trim_point_to_dict(self):
        from opencut.core.smart_trim import TrimPoint
        tp = TrimPoint(time=5.0, point_type="speech_onset", confidence=0.9)
        d = tp.to_dict()
        assert d["time"] == 5.0
        assert d["point_type"] == "speech_onset"

    def test_smart_trim_result_defaults(self):
        from opencut.core.smart_trim import SmartTrimResult
        r = SmartTrimResult()
        assert r.suggested_in == 0.0
        assert r.suggested_out == 0.0
        assert r.confidence == 0.0

    def test_smart_trim_result_time_saved(self):
        from opencut.core.smart_trim import SmartTrimResult
        r = SmartTrimResult(original_duration=60.0, trimmed_duration=45.0)
        assert r.time_saved == 15.0

    def test_smart_trim_result_to_dict(self):
        from opencut.core.smart_trim import SmartTrimResult
        r = SmartTrimResult(file_path="/a.mp4", original_duration=30.0,
                            suggested_in=2.0, suggested_out=28.0, trimmed_duration=26.0)
        d = r.to_dict()
        assert d["file_path"] == "/a.mp4"
        assert d["time_saved"] == 4.0

    def test_batch_trim_result_defaults(self):
        from opencut.core.smart_trim import BatchTrimResult
        r = BatchTrimResult()
        assert r.results == []
        assert r.clips_processed == 0

    def test_batch_trim_result_to_dict(self):
        from opencut.core.smart_trim import BatchTrimResult
        r = BatchTrimResult(clips_processed=5, total_time_saved=30.0)
        d = r.to_dict()
        assert d["clips_processed"] == 5
        assert d["total_time_saved"] == 30.0

    def test_detect_speech_regions_empty(self):
        from opencut.core.smart_trim import _detect_speech_regions
        regions = _detect_speech_regions([], 0.0)
        assert regions == []

    def test_detect_speech_regions_no_silence(self):
        from opencut.core.smart_trim import _detect_speech_regions
        regions = _detect_speech_regions([], 30.0)
        assert len(regions) == 1
        assert regions[0]["start"] == 0.0
        assert regions[0]["end"] == 30.0

    def test_detect_speech_regions_with_silence(self):
        from opencut.core.smart_trim import _detect_speech_regions
        silence = [{"start": 5.0, "end": 10.0, "duration": 5.0}]
        regions = _detect_speech_regions(silence, 30.0)
        assert len(regions) == 2
        assert regions[0]["end"] == 5.0
        assert regions[1]["start"] == 10.0

    def test_find_first_speech_onset(self):
        from opencut.core.smart_trim import _find_first_speech_onset
        regions = [
            {"start": 0, "end": 0.1, "duration": 0.1},
            {"start": 2.0, "end": 5.0, "duration": 3.0},
        ]
        tp = _find_first_speech_onset(regions, min_speech_duration=0.3)
        assert tp is not None
        assert tp.time == 2.0

    def test_find_first_speech_onset_none(self):
        from opencut.core.smart_trim import _find_first_speech_onset
        regions = [{"start": 0, "end": 0.1, "duration": 0.1}]
        tp = _find_first_speech_onset(regions, min_speech_duration=0.3)
        assert tp is None

    def test_find_last_speech_end(self):
        from opencut.core.smart_trim import _find_last_speech_end
        regions = [
            {"start": 2.0, "end": 5.0, "duration": 3.0},
            {"start": 10.0, "end": 25.0, "duration": 15.0},
        ]
        tp = _find_last_speech_end(regions, min_speech_duration=0.3)
        assert tp is not None
        assert tp.time == 25.0

    def test_generate_alternative_points(self):
        from opencut.core.smart_trim import _generate_alternative_points
        speech = [{"start": 2.0, "end": 5.0, "duration": 3.0}]
        scenes = [1.0, 10.0]
        silence = [{"start": 5.0, "end": 8.0, "duration": 3.0}]
        alts = _generate_alternative_points(speech, scenes, silence, 30.0)
        assert len(alts) > 0
        types = [a.point_type for a in alts]
        assert "scene_change" in types

    @patch("opencut.core.smart_trim.get_video_info")
    @patch("opencut.core.smart_trim._sp.run")
    def test_smart_trim_tight(self, mock_run, mock_info):
        from opencut.core.smart_trim import smart_trim
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 60.0}
        mock_run.return_value = MagicMock(returncode=0, stderr=b"silence_start: 0.0\nsilence_end: 3.0\n")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        try:
            result = smart_trim(tmp, mode="tight")
            assert result.mode == "tight"
            assert result.original_duration == 60.0
            assert result.suggested_in >= 0.0
            assert result.suggested_out <= 60.0
        finally:
            os.unlink(tmp)

    def test_smart_trim_bad_mode(self):
        from opencut.core.smart_trim import smart_trim
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Unknown mode"):
                smart_trim(tmp, mode="invalid")
        finally:
            os.unlink(tmp)

    def test_smart_trim_file_not_found(self):
        from opencut.core.smart_trim import smart_trim
        with pytest.raises(FileNotFoundError):
            smart_trim("/nonexistent/file.mp4", mode="tight")

    def test_batch_smart_trim_empty(self):
        from opencut.core.smart_trim import batch_smart_trim
        with pytest.raises(ValueError, match="No files"):
            batch_smart_trim([], mode="tight")

    def test_batch_smart_trim_bad_mode(self):
        from opencut.core.smart_trim import batch_smart_trim
        with pytest.raises(ValueError, match="Unknown mode"):
            batch_smart_trim(["/fake.mp4"], mode="invalid")


# ========================================================================
# 4. batch_timeline_ops.py
# ========================================================================
class TestBatchTimelineOps:
    """Tests for opencut.core.batch_timeline_ops."""

    def test_clip_info_defaults(self):
        from opencut.core.batch_timeline_ops import ClipInfo
        ci = ClipInfo()
        assert ci.file_path == ""
        assert ci.duration == 0.0
        assert ci.track == 1

    def test_clip_info_from_dict(self):
        from opencut.core.batch_timeline_ops import ClipInfo
        ci = ClipInfo.from_dict({"file_path": "/a.mp4", "start": 0, "end": 10})
        assert ci.file_path == "/a.mp4"
        assert ci.duration == 10.0

    def test_clip_info_from_dict_duration_calc(self):
        from opencut.core.batch_timeline_ops import ClipInfo
        ci = ClipInfo.from_dict({"file_path": "/a.mp4", "duration": 5.0})
        assert ci.duration == 5.0

    def test_clip_info_to_dict(self):
        from opencut.core.batch_timeline_ops import ClipInfo
        ci = ClipInfo(file_path="/a.mp4", start=0, end=10, duration=10)
        d = ci.to_dict()
        assert d["file_path"] == "/a.mp4"
        assert d["duration"] == 10.0

    def test_op_change_to_dict(self):
        from opencut.core.batch_timeline_ops import OpChange
        oc = OpChange(clip_index=0, operation="speed", parameter="factor",
                      old_value="1.0", new_value="2.0")
        d = oc.to_dict()
        assert d["operation"] == "speed"

    def test_batch_op_result_to_dict(self):
        from opencut.core.batch_timeline_ops import BatchOpResult
        r = BatchOpResult(operation="speed", clips_affected=5)
        d = r.to_dict()
        assert d["operation"] == "speed"
        assert d["clips_affected"] == 5

    def test_pipeline_result_to_dict(self):
        from opencut.core.batch_timeline_ops import PipelineResult
        r = PipelineResult(total_changes=10, dry_run=True)
        d = r.to_dict()
        assert d["total_changes"] == 10
        assert d["dry_run"] is True

    def test_parse_clip_list(self):
        from opencut.core.batch_timeline_ops import parse_clip_list
        clips = parse_clip_list([
            {"file_path": "/a.mp4", "start": 0, "end": 10},
            {"file_path": "/b.mp4", "start": 10, "end": 20},
        ])
        assert len(clips) == 2
        assert clips[0].index == 0
        assert clips[1].index == 1

    def test_batch_color_grade(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_color_grade
        clips = [ClipInfo(file_path="/a.mp4", index=0)]
        result = batch_color_grade(clips, lut_path="/some/lut.cube")
        assert result.operation == "color_grade"
        assert len(result.changes) > 0
        assert clips[0].metadata.get("lut_path") == "/some/lut.cube"

    def test_batch_color_grade_with_params(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_color_grade
        clips = [ClipInfo(file_path="/a.mp4", index=0)]
        result = batch_color_grade(clips, grade_params={"brightness": 1.1, "contrast": 1.2})
        assert len(result.changes) == 2

    def test_batch_speed(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_speed
        clips = [ClipInfo(file_path="/a.mp4", duration=10.0, index=0)]
        result = batch_speed(clips, speed_factor=2.0)
        assert result.operation == "speed"
        assert clips[0].duration == 5.0

    def test_batch_speed_clamp(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_speed
        clips = [ClipInfo(file_path="/a.mp4", duration=10.0, index=0)]
        batch_speed(clips, speed_factor=10.0)  # clamped to 4.0
        assert clips[0].metadata["speed_factor"] == 4.0

    def test_batch_normalize(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_normalize
        clips = [ClipInfo(file_path="/a.mp4", index=0)]
        result = batch_normalize(clips, target_lufs=-16.0)
        assert result.operation == "normalize"
        assert len(result.changes) == 1
        assert clips[0].metadata["normalize_target_lufs"] == -16.0

    def test_batch_transition(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_transition
        clips = [
            ClipInfo(file_path="/a.mp4", index=0),
            ClipInfo(file_path="/b.mp4", index=1),
            ClipInfo(file_path="/c.mp4", index=2),
        ]
        result = batch_transition(clips, transition_type="dissolve", transition_duration=1.0)
        assert result.operation == "transition"
        assert len(result.changes) == 2  # between a-b and b-c

    def test_batch_transition_bad_type(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_transition
        clips = [ClipInfo(index=0), ClipInfo(index=1)]
        with pytest.raises(ValueError, match="Unsupported transition"):
            batch_transition(clips, transition_type="star_wipe")

    def test_batch_crop_aspect_ratio(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_crop
        clips = [ClipInfo(file_path="/a.mp4", index=0)]
        result = batch_crop(clips, aspect_ratio="9:16")
        assert result.operation == "crop"
        assert clips[0].metadata.get("crop_aspect_ratio") == "9:16"

    def test_batch_crop_params(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_crop
        clips = [ClipInfo(file_path="/a.mp4", index=0)]
        result = batch_crop(clips, crop_params={"x": 100, "y": 50, "width": 1280, "height": 720})
        assert len(result.changes) == 4  # x, y, width, height

    def test_batch_caption(self):
        from opencut.core.batch_timeline_ops import ClipInfo, batch_caption
        clips = [ClipInfo(file_path="/a.mp4", index=0)]
        result = batch_caption(clips, language="fr", burn_in=True)
        assert result.operation == "caption"
        assert clips[0].metadata["caption_language"] == "fr"
        assert clips[0].metadata["caption_burn_in"] is True

    def test_execute_operation(self):
        from opencut.core.batch_timeline_ops import ClipInfo, execute_operation
        clips = [ClipInfo(file_path="/a.mp4", duration=10.0, index=0)]
        result = execute_operation("speed", clips, {"speed_factor": 1.5})
        assert result.operation == "speed"
        assert len(result.changes) == 1

    def test_execute_operation_unknown(self):
        from opencut.core.batch_timeline_ops import execute_operation
        with pytest.raises(ValueError, match="Unknown operation"):
            execute_operation("unknown_op", [], {})

    def test_execute_pipeline(self):
        from opencut.core.batch_timeline_ops import execute_pipeline
        clips = [
            {"file_path": "/a.mp4", "start": 0, "end": 10, "duration": 10},
            {"file_path": "/b.mp4", "start": 10, "end": 20, "duration": 10},
        ]
        ops = [
            {"operation": "speed", "params": {"speed_factor": 2.0}},
            {"operation": "normalize", "params": {"target_lufs": -16.0}},
        ]
        result = execute_pipeline(clips, ops)
        assert result.total_changes > 0
        assert len(result.operations) == 2

    def test_execute_pipeline_dry_run(self):
        from opencut.core.batch_timeline_ops import execute_pipeline
        clips = [{"file_path": "/a.mp4", "duration": 10}]
        ops = [{"operation": "caption", "params": {"language": "en"}}]
        result = execute_pipeline(clips, ops, dry_run=True)
        assert result.dry_run is True

    def test_execute_pipeline_no_clips(self):
        from opencut.core.batch_timeline_ops import execute_pipeline
        with pytest.raises(ValueError, match="No clips"):
            execute_pipeline([], [{"operation": "speed"}])

    def test_execute_pipeline_no_ops(self):
        from opencut.core.batch_timeline_ops import execute_pipeline
        with pytest.raises(ValueError, match="No operations"):
            execute_pipeline([{"file_path": "/a.mp4"}], [])

    def test_execute_pipeline_bad_op(self):
        from opencut.core.batch_timeline_ops import execute_pipeline
        with pytest.raises(ValueError, match="Unknown operation"):
            execute_pipeline(
                [{"file_path": "/a.mp4"}],
                [{"operation": "invalid_op"}],
            )


# ========================================================================
# 5. template_assembly_adv.py
# ========================================================================
class TestTemplateAssembly:
    """Tests for opencut.core.template_assembly_adv."""

    def test_template_slot_defaults(self):
        from opencut.core.template_assembly_adv import TemplateSlot
        ts = TemplateSlot()
        assert ts.name == ""
        assert ts.slot_type == "video"
        assert ts.required is True

    def test_template_slot_from_dict(self):
        from opencut.core.template_assembly_adv import TemplateSlot
        ts = TemplateSlot.from_dict({"name": "intro", "type": "video", "duration": 5.0})
        assert ts.name == "intro"
        assert ts.slot_type == "video"
        assert ts.duration == 5.0

    def test_template_slot_to_dict(self):
        from opencut.core.template_assembly_adv import TemplateSlot
        ts = TemplateSlot(name="intro", duration=5.0)
        d = ts.to_dict()
        assert d["name"] == "intro"
        assert d["duration"] == 5.0

    def test_template_defaults(self):
        from opencut.core.template_assembly_adv import Template
        t = Template()
        assert t.name == ""
        assert t.width == 1920
        assert t.height == 1080
        assert t.total_duration == 0.0

    def test_template_total_duration(self):
        from opencut.core.template_assembly_adv import Template, TemplateSlot
        t = Template(slots=[
            TemplateSlot(name="a", slot_type="video", duration=10.0),
            TemplateSlot(name="b", slot_type="video", duration=20.0),
            TemplateSlot(name="c", slot_type="audio", duration=30.0),
        ])
        assert t.total_duration == 30.0  # audio excluded

    def test_template_required_slots(self):
        from opencut.core.template_assembly_adv import Template, TemplateSlot
        t = Template(slots=[
            TemplateSlot(name="a", required=True),
            TemplateSlot(name="b", required=False),
            TemplateSlot(name="c", required=True),
        ])
        assert len(t.required_slots) == 2

    def test_template_from_dict(self):
        from opencut.core.template_assembly_adv import Template
        t = Template.from_dict({
            "name": "test",
            "width": 1280,
            "height": 720,
            "slots": [{"name": "intro", "type": "video", "duration": 5}],
        })
        assert t.name == "test"
        assert t.width == 1280
        assert len(t.slots) == 1

    def test_template_to_dict(self):
        from opencut.core.template_assembly_adv import Template, TemplateSlot
        t = Template(name="test", slots=[TemplateSlot(name="intro")])
        d = t.to_dict()
        assert d["name"] == "test"
        assert len(d["slots"]) == 1

    def test_assembled_clip_duration(self):
        from opencut.core.template_assembly_adv import AssembledClip
        ac = AssembledClip(record_in=0, record_out=10)
        assert ac.duration == 10.0

    def test_assembled_clip_to_dict(self):
        from opencut.core.template_assembly_adv import AssembledClip
        ac = AssembledClip(slot_name="intro", source_file="/a.mp4",
                           record_in=0, record_out=5)
        d = ac.to_dict()
        assert d["slot_name"] == "intro"
        assert d["duration"] == 5.0

    def test_validation_result_defaults(self):
        from opencut.core.template_assembly_adv import ValidationResult
        vr = ValidationResult()
        assert vr.valid is True
        assert vr.errors == []

    def test_validation_result_to_dict(self):
        from opencut.core.template_assembly_adv import ValidationResult
        vr = ValidationResult(valid=False, errors=["Missing name"])
        d = vr.to_dict()
        assert d["valid"] is False

    def test_assembly_result_defaults(self):
        from opencut.core.template_assembly_adv import AssemblyResult
        ar = AssemblyResult()
        assert ar.clips == []
        assert ar.total_duration == 0.0

    def test_assembly_result_to_dict(self):
        from opencut.core.template_assembly_adv import AssemblyResult
        ar = AssemblyResult(template_name="youtube_video", total_duration=120.0)
        d = ar.to_dict()
        assert d["template_name"] == "youtube_video"
        assert d["total_duration"] == 120.0

    def test_builtin_templates_exist(self):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES
        assert "youtube_video" in BUILTIN_TEMPLATES
        assert "podcast_video" in BUILTIN_TEMPLATES
        assert "tutorial" in BUILTIN_TEMPLATES
        assert "social_clip" in BUILTIN_TEMPLATES

    def test_builtin_templates_valid(self):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES, validate_template
        for name, factory in BUILTIN_TEMPLATES.items():
            tmpl = factory()
            result = validate_template(tmpl.to_dict())
            assert result.valid, f"Built-in template '{name}' failed validation: {result.errors}"

    def test_list_templates(self):
        from opencut.core.template_assembly_adv import list_templates
        templates = list_templates()
        assert len(templates) == 4
        names = [t["name"] for t in templates]
        assert "youtube_video" in names

    def test_youtube_template_structure(self):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES
        tmpl = BUILTIN_TEMPLATES["youtube_video"]()
        slot_names = [s.name for s in tmpl.slots]
        assert "intro" in slot_names
        assert "segment_1" in slot_names
        assert "outro" in slot_names

    def test_social_clip_portrait(self):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES
        tmpl = BUILTIN_TEMPLATES["social_clip"]()
        assert tmpl.width == 1080
        assert tmpl.height == 1920

    def test_validate_template_no_name(self):
        from opencut.core.template_assembly_adv import validate_template
        result = validate_template({"slots": [{"name": "a", "slot_type": "video"}]})
        assert result.valid is False
        assert any("name" in e.lower() for e in result.errors)

    def test_validate_template_no_slots(self):
        from opencut.core.template_assembly_adv import validate_template
        result = validate_template({"name": "test", "slots": []})
        assert result.valid is False

    def test_validate_template_duplicate_names(self):
        from opencut.core.template_assembly_adv import validate_template
        result = validate_template({
            "name": "test",
            "slots": [
                {"name": "intro", "slot_type": "video"},
                {"name": "intro", "slot_type": "video"},
            ],
        })
        assert result.valid is False
        assert any("Duplicate" in e for e in result.errors)

    def test_validate_template_bad_slot_type(self):
        from opencut.core.template_assembly_adv import validate_template
        result = validate_template({
            "name": "test",
            "slots": [{"name": "a", "slot_type": "hologram"}],
        })
        assert result.valid is False

    def test_validate_template_min_max_conflict(self):
        from opencut.core.template_assembly_adv import validate_template
        result = validate_template({
            "name": "test",
            "slots": [{"name": "a", "slot_type": "video",
                        "min_duration": 20, "max_duration": 5}],
        })
        assert result.valid is False

    @patch("opencut.core.template_assembly_adv.get_video_info")
    def test_assemble_youtube_template(self, mock_info):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES, assemble_from_template
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 120.0}

        tmpl = BUILTIN_TEMPLATES["youtube_video"]()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        try:
            media_map = {
                "intro": tmp,
                "segment_1": tmp,
                "outro": tmp,
            }
            result = assemble_from_template(tmpl, media_map)
            assert result.template_name == "youtube_video"
            assert len(result.clips) > 0
            assert result.total_duration > 0
        finally:
            os.unlink(tmp)

    @patch("opencut.core.template_assembly_adv.get_video_info")
    def test_assemble_social_clip(self, mock_info):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES, assemble_from_template
        mock_info.return_value = {"width": 1080, "height": 1920, "fps": 30, "duration": 30.0}

        tmpl = BUILTIN_TEMPLATES["social_clip"]()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        try:
            media_map = {"hook": tmp, "content": tmp, "cta": tmp}
            result = assemble_from_template(tmpl, media_map)
            assert result.template_name == "social_clip"
            assert len(result.clips) >= 3
        finally:
            os.unlink(tmp)

    def test_assemble_empty_template(self):
        from opencut.core.template_assembly_adv import Template, assemble_from_template
        tmpl = Template(name="empty")
        with pytest.raises(ValueError, match="no slots"):
            assemble_from_template(tmpl, {})

    def test_assemble_missing_required_slots(self):
        from opencut.core.template_assembly_adv import BUILTIN_TEMPLATES, assemble_from_template
        tmpl = BUILTIN_TEMPLATES["youtube_video"]()
        result = assemble_from_template(tmpl, {})
        assert len(result.missing_slots) > 0

    @patch("opencut.core.template_assembly_adv.get_video_info")
    def test_assemble_text_slot(self, mock_info):
        from opencut.core.template_assembly_adv import (
            Template,
            TemplateSlot,
            assemble_from_template,
        )
        mock_info.return_value = {"width": 1920, "height": 1080, "fps": 30, "duration": 10.0}
        tmpl = Template(name="text_test", slots=[
            TemplateSlot(name="title", slot_type="text", duration=5.0, required=True),
        ])
        result = assemble_from_template(tmpl, {"title": "Hello World"})
        assert len(result.clips) == 1
        assert result.clips[0].text_content == "Hello World"

    def test_generate_assembly_edl(self):
        from opencut.core.template_assembly_adv import AssembledClip, _generate_assembly_edl
        clips = [
            AssembledClip(slot_name="intro", source_file="/intro.mp4",
                          source_in=0, source_out=5,
                          record_in=0, record_out=5, slot_type="video"),
        ]
        edl = _generate_assembly_edl(clips, "Test", 30.0)
        assert "TITLE: Test" in edl
        assert "INTRO" in edl

    def test_generate_otio_json(self):
        from opencut.core.template_assembly_adv import (
            AssembledClip,
            Template,
            _generate_otio_json,
        )
        clips = [
            AssembledClip(slot_name="intro", source_file="/a.mp4",
                          source_in=0, source_out=5,
                          record_in=0, record_out=5, slot_type="video", layer=0),
        ]
        tmpl = Template(name="test", fps=30.0)
        otio = _generate_otio_json(clips, tmpl)
        assert otio["OTIO_SCHEMA"] == "Timeline.1"
        assert "tracks" in otio

    def test_fit_media_to_slot_short_media(self):
        from opencut.core.template_assembly_adv import TemplateSlot, _fit_media_to_slot
        slot = TemplateSlot(name="test", duration=30.0)
        # Non-existent file -> returns 0, slot.duration, False
        src_in, src_out, trimmed = _fit_media_to_slot("/nonexistent.mp4", slot)
        assert src_in == 0.0
        assert trimmed is False

    @patch("opencut.core.template_assembly_adv.get_video_info")
    def test_fit_media_to_slot_long_media(self, mock_info):
        from opencut.core.template_assembly_adv import TemplateSlot, _fit_media_to_slot
        mock_info.return_value = {"duration": 60.0}
        slot = TemplateSlot(name="test", duration=10.0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            tmp = f.name

        try:
            src_in, src_out, trimmed = _fit_media_to_slot(tmp, slot)
            assert trimmed is True
            assert (src_out - src_in) == pytest.approx(10.0, abs=0.1)
        finally:
            os.unlink(tmp)


# ========================================================================
# 6. Routes smoke tests
# ========================================================================
class TestTimelineAutoRoutes:
    """Smoke tests for timeline_auto_routes blueprint."""

    @pytest.fixture
    def app(self):
        from flask import Flask
        from opencut.routes.timeline_auto_routes import timeline_auto_bp
        app = Flask(__name__)
        app.register_blueprint(timeline_auto_bp)
        app.config["TESTING"] = True
        return app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    def test_list_templates_get(self, client):
        resp = client.get("/api/timeline/templates")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "templates" in data
        assert len(data["templates"]) == 4

    def test_validate_template_no_data(self, client):
        resp = client.post("/api/timeline/templates/validate",
                           data=json.dumps({}),
                           content_type="application/json")
        # May return 400 or validation error, both acceptable
        assert resp.status_code in (200, 400, 403)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_validate_template_with_csrf(self, client):
        resp = client.post(
            "/api/timeline/templates/validate",
            data=json.dumps({
                "name": "test",
                "slots": [{"name": "intro", "slot_type": "video"}],
            }),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "valid" in data

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_rough_cut_no_media(self, client):
        resp = client.post(
            "/api/timeline/rough-cut",
            data=json.dumps({"script": "test"}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        # Should fail validation (no media_paths) but return job_id or 400
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_auto_mix_no_tracks(self, client):
        resp = client.post(
            "/api/timeline/auto-mix",
            data=json.dumps({"tracks": []}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_smart_trim_no_file(self, client):
        resp = client.post(
            "/api/timeline/smart-trim",
            data=json.dumps({}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_batch_ops_no_data(self, client):
        resp = client.post(
            "/api/timeline/batch-ops",
            data=json.dumps({}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_batch_ops_preview_no_data(self, client):
        resp = client.post(
            "/api/timeline/batch-ops/preview",
            data=json.dumps({}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_assemble_no_template(self, client):
        resp = client.post(
            "/api/timeline/assemble",
            data=json.dumps({"media_map": {"intro": "/fake.mp4"}}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_smart_trim_batch_no_files(self, client):
        resp = client.post(
            "/api/timeline/smart-trim/batch",
            data=json.dumps({"file_paths": []}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)

    @patch("opencut.security._csrf_tokens", {"test": float("inf")})
    def test_auto_mix_preview(self, client):
        resp = client.post(
            "/api/timeline/auto-mix/preview",
            data=json.dumps({"tracks": []}),
            content_type="application/json",
            headers={"X-OpenCut-Token": "test"},
        )
        assert resp.status_code in (200, 400)
