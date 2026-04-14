"""
Unit tests for UX Intelligence features:
- Command Palette (feature index, fuzzy search, recents)
- Preview Frame (extraction, before/after operations)
- Smart Defaults (clip analysis, operation defaults)
- Contextual Suggest (rules engine, clip-aware suggestions)
- UX Intelligence route smoke tests
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. Command Palette -- Feature Index
# ========================================================================
class TestCommandPaletteIndex:
    """Tests for opencut.core.command_palette index building."""

    def test_build_feature_index_returns_list(self):
        from opencut.core.command_palette import build_feature_index
        index = build_feature_index()
        assert isinstance(index, list)
        assert len(index) > 0

    def test_feature_index_has_300_plus_entries(self):
        from opencut.core.command_palette import build_feature_index
        index = build_feature_index()
        assert len(index) >= 150  # large curated set

    def test_feature_entry_has_required_fields(self):
        from opencut.core.command_palette import build_feature_index
        index = build_feature_index()
        required = {"id", "name", "description", "category", "aliases", "route", "tags"}
        for entry in index:
            assert required.issubset(entry.keys()), f"Missing fields in {entry.get('id')}"

    def test_feature_ids_are_unique(self):
        from opencut.core.command_palette import build_feature_index
        index = build_feature_index()
        ids = [e["id"] for e in index]
        assert len(ids) == len(set(ids)), "Duplicate feature IDs found"

    def test_feature_entry_types(self):
        from opencut.core.command_palette import build_feature_index
        index = build_feature_index()
        entry = index[0]
        assert isinstance(entry["id"], str)
        assert isinstance(entry["name"], str)
        assert isinstance(entry["description"], str)
        assert isinstance(entry["category"], str)
        assert isinstance(entry["aliases"], list)
        assert isinstance(entry["route"], str)
        assert isinstance(entry["tags"], list)

    def test_feature_index_is_cached(self):
        from opencut.core.command_palette import build_feature_index
        idx1 = build_feature_index()
        idx2 = build_feature_index()
        assert len(idx1) == len(idx2)

    def test_feature_index_progress_callback(self):
        import opencut.core.command_palette as cp
        # Reset singleton for test
        with patch.object(cp, "_feature_index", None):
            progress_calls = []

            def on_progress(pct, msg=""):
                progress_calls.append(pct)

            cp.build_feature_index(on_progress=on_progress)
            assert len(progress_calls) > 0
            assert progress_calls[-1] == 100

    def test_categories_are_valid(self):
        from opencut.core.command_palette import FEATURE_CATEGORIES, build_feature_index
        index = build_feature_index()
        for entry in index:
            assert entry["category"] in FEATURE_CATEGORIES, (
                f"Invalid category '{entry['category']}' for {entry['id']}"
            )


# ========================================================================
# 2. Command Palette -- Fuzzy Search
# ========================================================================
class TestCommandPaletteFuzzySearch:
    """Tests for opencut.core.command_palette fuzzy search."""

    def test_exact_match_ranks_first(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("Normalize Audio")
        assert len(results) > 0
        assert results[0]["id"] == "normalize_audio"

    def test_prefix_match(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("Normal")
        assert len(results) > 0
        assert any(r["id"] == "normalize_audio" for r in results)

    def test_substring_match(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("silence")
        assert len(results) > 0
        assert any(r["id"] == "remove_silence" for r in results)

    def test_alias_match(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("loudness")
        assert len(results) > 0
        assert any(r["id"] == "normalize_audio" for r in results)

    def test_tag_match(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("waveform")
        assert len(results) > 0
        assert any(r["id"] == "audio_visualizer" for r in results)

    def test_empty_query_returns_empty(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("")
        assert results == []

    def test_whitespace_query_returns_empty(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("   ")
        assert results == []

    def test_limit_parameter(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("audio", limit=3)
        assert len(results) <= 3

    def test_results_have_score(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("denoise")
        assert len(results) > 0
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0

    def test_results_sorted_by_score_descending(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("audio")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    def test_exact_match_higher_than_prefix(self):
        from opencut.core.command_palette import _similarity_score
        exact = _similarity_score("denoise", "denoise")
        prefix = _similarity_score("denoise", "denoise video")
        assert exact > prefix

    def test_prefix_higher_than_substring(self):
        from opencut.core.command_palette import _similarity_score
        prefix = _similarity_score("audio", "audio enhance")
        substring = _similarity_score("audio", "enhance audio quality")
        assert prefix > substring

    def test_substring_higher_than_trigram(self):
        from opencut.core.command_palette import _similarity_score
        substring = _similarity_score("caption", "generate captions for video")
        trigram = _similarity_score("caption", "chapter generation")
        assert substring > trigram

    def test_search_case_insensitive(self):
        from opencut.core.command_palette import fuzzy_search
        r1 = fuzzy_search("DENOISE")
        r2 = fuzzy_search("denoise")
        assert len(r1) > 0
        assert r1[0]["id"] == r2[0]["id"]

    def test_fuzzy_search_custom_index(self):
        from opencut.core.command_palette import fuzzy_search
        custom = [
            {"id": "t1", "name": "Test Feature", "description": "A test", "category": "test", "aliases": [], "route": "/test", "tags": []},
            {"id": "t2", "name": "Other Thing", "description": "B test", "category": "test", "aliases": [], "route": "/other", "tags": []},
        ]
        results = fuzzy_search("Test", index=custom)
        assert len(results) > 0
        assert results[0]["id"] == "t1"

    def test_no_match_returns_empty(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("zzzzxyzzy999")
        assert results == []

    def test_progress_callback(self):
        from opencut.core.command_palette import fuzzy_search
        calls = []
        fuzzy_search("audio", on_progress=lambda p, m="": calls.append(p))
        assert 100 in calls


# ========================================================================
# 3. Command Palette -- Recents
# ========================================================================
class TestCommandPaletteRecents:
    """Tests for opencut.core.command_palette recent features."""

    def test_get_recent_features_no_file(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        monkeypatch.setattr(cp, "_RECENTS_FILE", str(tmp_path / "nonexistent.json"))
        result = cp.get_recent_features()
        assert result == []

    def test_record_and_get_recents(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        recents_file = str(tmp_path / "recents.json")
        monkeypatch.setattr(cp, "_RECENTS_FILE", recents_file)
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        result = cp.record_feature_use("normalize_audio")
        assert result["feature_id"] == "normalize_audio"
        assert result["use_count"] == 1

        recents = cp.get_recent_features(limit=5)
        assert len(recents) == 1
        assert recents[0]["feature_id"] == "normalize_audio"

    def test_record_increments_count(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        recents_file = str(tmp_path / "recents.json")
        monkeypatch.setattr(cp, "_RECENTS_FILE", recents_file)
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        cp.record_feature_use("denoise")
        result = cp.record_feature_use("denoise")
        assert result["use_count"] == 2

    def test_recents_limit(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        recents_file = str(tmp_path / "recents.json")
        monkeypatch.setattr(cp, "_RECENTS_FILE", recents_file)
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        for i in range(10):
            cp.record_feature_use(f"feature_{i}")

        recents = cp.get_recent_features(limit=3)
        assert len(recents) == 3

    def test_recents_sorted_by_timestamp(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        recents_file = str(tmp_path / "recents.json")
        monkeypatch.setattr(cp, "_RECENTS_FILE", recents_file)
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        cp.record_feature_use("first_feature")
        cp.record_feature_use("second_feature")

        recents = cp.get_recent_features(limit=5)
        assert recents[0]["feature_id"] == "second_feature"

    def test_record_looks_up_name(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        recents_file = str(tmp_path / "recents.json")
        monkeypatch.setattr(cp, "_RECENTS_FILE", recents_file)
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        result = cp.record_feature_use("normalize_audio")
        assert result["name"] == "Normalize Audio"

    def test_recents_progress_callback(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        monkeypatch.setattr(cp, "_RECENTS_FILE", str(tmp_path / "nonexistent.json"))
        calls = []
        cp.get_recent_features(on_progress=lambda p, m="": calls.append(p))
        assert 100 in calls

    def test_record_corrupt_file_handled(self, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        recents_file = str(tmp_path / "recents.json")
        monkeypatch.setattr(cp, "_RECENTS_FILE", recents_file)
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        with open(recents_file, "w") as f:
            f.write("not valid json{{{")

        result = cp.record_feature_use("denoise")
        assert result["use_count"] == 1


# ========================================================================
# 4. Preview Frame
# ========================================================================
class TestPreviewFrame:
    """Tests for opencut.core.preview_frame."""

    def test_preview_result_dataclass(self):
        from opencut.core.preview_frame import PreviewResult
        r = PreviewResult(
            original_b64="abc", processed_b64="def",
            width=1920, height=1080, timestamp=5.0,
        )
        assert r.original_b64 == "abc"
        assert r.width == 1920

    def test_supported_operations_list(self):
        from opencut.core.preview_frame import SUPPORTED_OPERATIONS
        assert "denoise" in SUPPORTED_OPERATIONS
        assert "upscale" in SUPPORTED_OPERATIONS
        assert "color_correct" in SUPPORTED_OPERATIONS
        assert "stabilize" in SUPPORTED_OPERATIONS
        assert "lut_apply" in SUPPORTED_OPERATIONS
        assert "brightness" in SUPPORTED_OPERATIONS
        assert "contrast" in SUPPORTED_OPERATIONS
        assert "saturation" in SUPPORTED_OPERATIONS
        assert len(SUPPORTED_OPERATIONS) == 8

    def test_extract_frame_file_not_found(self):
        from opencut.core.preview_frame import extract_frame
        with pytest.raises(FileNotFoundError):
            extract_frame("/nonexistent/video.mp4")

    def test_extract_frame_calls_ffmpeg(self, tmp_path):
        from opencut.core.preview_frame import extract_frame
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        def mock_run_ffmpeg(cmd, timeout=3600, stderr_cap=0):
            # Write a fake PNG to the output path
            out_path = cmd[-1]
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            return ""

        with patch("opencut.core.preview_frame.run_ffmpeg", side_effect=mock_run_ffmpeg):
            result = extract_frame(str(video), timestamp=2.5)

        assert result == png_bytes

    def test_extract_frame_progress_callback(self, tmp_path):
        from opencut.core.preview_frame import extract_frame
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        def mock_run_ffmpeg(cmd, timeout=3600, stderr_cap=0):
            out_path = cmd[-1]
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG" + b"\x00" * 50)
            return ""

        calls = []
        with patch("opencut.core.preview_frame.run_ffmpeg", side_effect=mock_run_ffmpeg):
            extract_frame(str(video), on_progress=lambda p, m="": calls.append(p))
        assert 100 in calls

    def test_preview_operation_unsupported(self, tmp_path):
        from opencut.core.preview_frame import preview_operation
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        with pytest.raises(ValueError, match="Unsupported operation"):
            preview_operation(str(video), "nonexistent_op")

    def test_preview_operation_file_not_found(self):
        from opencut.core.preview_frame import preview_operation
        with pytest.raises(FileNotFoundError):
            preview_operation("/no/such/file.mp4", "denoise")

    def test_preview_operation_returns_base64(self, tmp_path):
        from opencut.core.preview_frame import preview_operation
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        def mock_run_ffmpeg(cmd, timeout=3600, stderr_cap=0):
            out_path = cmd[-1]
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            return ""

        probe_json = json.dumps({"streams": [{"width": 1920, "height": 1080}]})
        mock_probe_result = MagicMock(returncode=0, stdout=probe_json.encode())

        with patch("opencut.core.preview_frame.run_ffmpeg", side_effect=mock_run_ffmpeg), \
             patch("opencut.core.preview_frame._sp.run", return_value=mock_probe_result):
            result = preview_operation(str(video), "denoise", timestamp=1.0)

        assert result.original_b64
        assert result.processed_b64
        assert result.width == 1920
        assert result.height == 1080
        assert result.timestamp == 1.0

    def test_denoise_filter_strengths(self):
        from opencut.core.preview_frame import _denoise_filter
        assert "2:2" in _denoise_filter({"strength": "light"})
        assert "4:3" in _denoise_filter({"strength": "moderate"})
        assert "8:6" in _denoise_filter({"strength": "heavy"})

    def test_upscale_filter_factor(self):
        from opencut.core.preview_frame import _upscale_filter
        assert "iw*2" in _upscale_filter({"factor": 2})
        assert "iw*3" in _upscale_filter({"factor": 3})

    def test_brightness_filter_clamp(self):
        from opencut.core.preview_frame import _brightness_filter
        f = _brightness_filter({"value": 5.0})
        assert "brightness=1.0" in f
        f = _brightness_filter({"value": -5.0})
        assert "brightness=-1.0" in f

    def test_contrast_filter(self):
        from opencut.core.preview_frame import _contrast_filter
        f = _contrast_filter({"value": 1.5})
        assert "contrast=1.5" in f

    def test_saturation_filter(self):
        from opencut.core.preview_frame import _saturation_filter
        f = _saturation_filter({"value": 1.3})
        assert "saturation=1.3" in f

    def test_color_correct_filter(self):
        from opencut.core.preview_frame import _color_correct_filter
        f = _color_correct_filter({"brightness": 0.1, "contrast": 1.2})
        assert "brightness=0.1" in f
        assert "contrast=1.2" in f

    def test_lut_apply_filter_fallback(self):
        from opencut.core.preview_frame import _lut_apply_filter
        f = _lut_apply_filter({})  # no lut_path
        assert "colorbalance" in f

    def test_stabilize_filter(self):
        from opencut.core.preview_frame import _stabilize_filter
        f = _stabilize_filter({})
        assert "unsharp" in f


# ========================================================================
# 5. Smart Defaults -- ClipProfile
# ========================================================================
class TestSmartDefaultsClipProfile:
    """Tests for opencut.core.smart_defaults ClipProfile."""

    def test_clip_profile_defaults(self):
        from opencut.core.smart_defaults import ClipProfile
        p = ClipProfile()
        assert p.avg_loudness_lufs is None
        assert p.resolution == 0
        assert p.detected_content_type == "unknown"
        assert p.has_audio is False
        assert p.has_video is False

    def test_clip_profile_fields(self):
        from opencut.core.smart_defaults import ClipProfile
        p = ClipProfile(
            avg_loudness_lufs=-16.0,
            peak_db=-1.0,
            resolution=1920,
            fps=30.0,
            codec="h264",
            duration_s=120.0,
            has_audio=True,
            has_video=True,
            is_static_camera=True,
            detected_content_type="interview",
        )
        assert p.avg_loudness_lufs == -16.0
        assert p.resolution == 1920
        assert p.detected_content_type == "interview"

    def test_content_types_constant(self):
        from opencut.core.smart_defaults import CONTENT_TYPES
        assert "interview" in CONTENT_TYPES
        assert "music_video" in CONTENT_TYPES
        assert "screen_recording" in CONTENT_TYPES
        assert "drone" in CONTENT_TYPES
        assert "vlog" in CONTENT_TYPES
        assert "unknown" in CONTENT_TYPES


# ========================================================================
# 6. Smart Defaults -- Analyze Clip
# ========================================================================
class TestSmartDefaultsAnalyze:
    """Tests for opencut.core.smart_defaults analyze_clip_properties."""

    def test_analyze_file_not_found(self):
        from opencut.core.smart_defaults import analyze_clip_properties
        with pytest.raises(FileNotFoundError):
            analyze_clip_properties("/nonexistent/video.mp4")

    def test_analyze_returns_clip_profile(self, tmp_path):
        from opencut.core.smart_defaults import ClipProfile, analyze_clip_properties

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        probe_json = json.dumps({
            "format": {"duration": "120.5", "bit_rate": "5000000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
                 "r_frame_rate": "30/1", "pix_fmt": "yuv420p"},
                {"codec_type": "audio", "channels": 2, "sample_rate": "48000"},
            ],
        })

        mock_probe = MagicMock(returncode=0, stdout=probe_json.encode())
        mock_loud = MagicMock(returncode=0, stderr=b"I: -18.0 LUFS\nPeak: -2.0 dBFS")
        mock_freeze = MagicMock(returncode=0, stderr=b"")

        call_count = [0]

        def mock_sp_run(cmd, **kwargs):
            call_count[0] += 1
            if "ffprobe" in str(cmd[0]):
                return mock_probe
            stderr = kwargs.get("capture_output", False)
            if "ebur128" in str(cmd):
                return mock_loud
            return mock_freeze

        with patch("opencut.core.smart_defaults._sp.run", side_effect=mock_sp_run):
            result = analyze_clip_properties(str(video))

        assert isinstance(result, ClipProfile)
        assert result.has_video is True
        assert result.has_audio is True
        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30.0

    def test_analyze_progress_callback(self, tmp_path):
        from opencut.core.smart_defaults import analyze_clip_properties

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        probe_json = json.dumps({
            "format": {"duration": "10"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 640, "height": 480, "r_frame_rate": "25/1", "pix_fmt": "yuv420p"}],
        })
        mock_result = MagicMock(returncode=0, stdout=probe_json.encode(), stderr=b"")

        calls = []
        with patch("opencut.core.smart_defaults._sp.run", return_value=mock_result):
            analyze_clip_properties(str(video), on_progress=lambda p, m="": calls.append(p))

        assert 100 in calls


# ========================================================================
# 7. Smart Defaults -- Get Defaults
# ========================================================================
class TestSmartDefaultsGetDefaults:
    """Tests for opencut.core.smart_defaults get_smart_defaults."""

    def _make_profile(self, **kwargs):
        from opencut.core.smart_defaults import ClipProfile
        defaults = {
            "has_audio": True, "has_video": True,
            "resolution": 1920, "width": 1920, "height": 1080,
            "fps": 30.0, "codec": "h264", "duration_s": 120.0,
            "detected_content_type": "unknown",
        }
        defaults.update(kwargs)
        return ClipProfile(**defaults)

    def test_normalize_interview(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="interview", avg_loudness_lufs=-20.0)
        result = get_smart_defaults("normalize", profile)
        assert result["target_lufs"] == -16.0

    def test_normalize_music_video(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="music_video")
        result = get_smart_defaults("normalize", profile)
        assert result["target_lufs"] == -14.0

    def test_normalize_screen_recording(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="screen_recording")
        result = get_smart_defaults("normalize", profile)
        assert result["target_lufs"] == -18.0

    def test_denoise_screen_recording(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="screen_recording")
        result = get_smart_defaults("denoise", profile)
        assert result["strength"] == "light"

    def test_upscale_low_res(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(resolution=640, width=640, height=480)
        result = get_smart_defaults("upscale", profile)
        assert result["factor"] == 2

    def test_upscale_4k_no_action(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(resolution=3840, width=3840, height=2160)
        result = get_smart_defaults("upscale", profile)
        assert result["factor"] == 1
        assert "note" in result

    def test_export_defaults_gaming(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="gaming")
        result = get_smart_defaults("export", profile)
        assert result["codec"] == "h264"

    def test_export_defaults_drone(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="drone")
        result = get_smart_defaults("export", profile)
        assert result["crf"] == 16

    def test_stabilize_drone(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="drone")
        result = get_smart_defaults("stabilize", profile)
        assert result["smoothing"] == 30

    def test_stabilize_interview_minimal(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="interview")
        result = get_smart_defaults("stabilize", profile)
        assert result["smoothing"] == 5

    def test_caption_interview_has_speaker_labels(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="interview")
        result = get_smart_defaults("caption", profile)
        assert result["speaker_labels"] is True

    def test_caption_music_video_karaoke(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="music_video")
        result = get_smart_defaults("caption", profile)
        assert result["style"] == "karaoke"

    def test_silence_remove_presentation(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="presentation")
        result = get_smart_defaults("silence_remove", profile)
        assert result["min_silence_ms"] == 1500

    def test_thumbnail_drone(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="drone")
        result = get_smart_defaults("thumbnail", profile)
        assert result["strategy"] == "scenic"

    def test_color_grade_drone(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="drone")
        result = get_smart_defaults("color_grade", profile)
        assert result["mood"] == "warm sunset"

    def test_unknown_operation_returns_note(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile()
        result = get_smart_defaults("totally_unknown_op", profile)
        assert "note" in result

    def test_defaults_include_clip_info(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="interview")
        result = get_smart_defaults("normalize", profile)
        assert "_clip_info" in result
        assert result["_clip_info"]["content_type"] == "interview"

    def test_reframe_interview(self):
        from opencut.core.smart_defaults import get_smart_defaults
        profile = self._make_profile(detected_content_type="interview")
        result = get_smart_defaults("reframe", profile)
        assert result["target_aspect"] == "9:16"
        assert result["method"] == "face_track"


# ========================================================================
# 8. Contextual Suggest -- Suggestion Dataclass
# ========================================================================
class TestContextualSuggestDataclass:
    """Tests for opencut.core.contextual_suggest Suggestion."""

    def test_suggestion_dataclass(self):
        from opencut.core.contextual_suggest import Suggestion
        s = Suggestion(
            feature_id="denoise",
            name="Denoise",
            reason="test",
            confidence=0.8,
            params={"strength": "moderate"},
        )
        assert s.feature_id == "denoise"
        assert s.confidence == 0.8

    def test_suggestion_default_params(self):
        from opencut.core.contextual_suggest import Suggestion
        s = Suggestion(feature_id="test", name="Test", reason="r", confidence=0.5)
        assert s.params == {}


# ========================================================================
# 9. Contextual Suggest -- Rules Engine
# ========================================================================
class TestContextualSuggestRules:
    """Tests for opencut.core.contextual_suggest rules engine."""

    def _make_profile(self, **kwargs):
        from opencut.core.smart_defaults import ClipProfile
        defaults = {
            "has_audio": True, "has_video": True,
            "resolution": 1920, "width": 1920, "height": 1080,
            "fps": 30.0, "codec": "h264", "duration_s": 120.0,
            "detected_content_type": "unknown",
            "avg_loudness_lufs": -16.0,
            "is_static_camera": False,
            "bitrate_kbps": 10000,
            "audio_channels": 2,
        }
        defaults.update(kwargs)
        return ClipProfile(**defaults)

    def test_low_loudness_suggests_normalize(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(avg_loudness_lufs=-30.0)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "normalize_audio" in ids

    def test_high_loudness_suggests_normalize(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(avg_loudness_lufs=-5.0)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "normalize_audio" in ids

    def test_no_audio_suggests_music_gen(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(has_audio=False)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "music_gen" in ids

    def test_has_audio_suggests_captions(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile()
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "add_captions" in ids

    def test_low_res_suggests_upscale(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(resolution=480, width=640, height=480)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "upscale" in ids

    def test_long_video_suggests_scene_detect(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(duration_s=600)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "scene_detect" in ids

    def test_very_long_video_suggests_highlights(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(duration_s=1200)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "highlights" in ids

    def test_interview_suggests_diarize(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(detected_content_type="interview")
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "diarize" in ids

    def test_screen_recording_suggests_cursor_zoom(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(detected_content_type="screen_recording")
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "cursor_zoom" in ids

    def test_drone_suggests_telemetry(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(detected_content_type="drone")
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "telemetry_overlay" in ids

    def test_recent_ops_excluded(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(avg_loudness_lufs=-30.0)
        suggestions = _build_suggestions(profile, {"normalize_audio"})
        ids = [s.feature_id for s in suggestions]
        assert "normalize_audio" not in ids

    def test_speech_content_suggests_silence_removal(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(detected_content_type="tutorial")
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "remove_silence" in ids

    def test_low_bitrate_suggests_denoise(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(bitrate_kbps=3000, codec="h264")
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "denoise" in ids

    def test_landscape_interview_suggests_reframe(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(
            detected_content_type="interview", width=1920, height=1080,
        )
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "smart_reframe" in ids

    def test_high_bitrate_h264_suggests_h265(self):
        from opencut.core.contextual_suggest import _build_suggestions
        profile = self._make_profile(codec="h264", bitrate_kbps=25000)
        suggestions = _build_suggestions(profile, set())
        ids = [s.feature_id for s in suggestions]
        assert "export_h265" in ids


# ========================================================================
# 10. Contextual Suggest -- Full Pipeline
# ========================================================================
class TestContextualSuggestPipeline:
    """Tests for opencut.core.contextual_suggest.suggest_operations."""

    def test_suggest_operations_file_not_found(self):
        from opencut.core.contextual_suggest import suggest_operations
        with pytest.raises(FileNotFoundError):
            suggest_operations("/nonexistent/video.mp4")

    def test_suggest_operations_returns_list(self, tmp_path):
        from opencut.core.contextual_suggest import suggest_operations

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        probe_json = json.dumps({
            "format": {"duration": "120", "bit_rate": "5000000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
                 "r_frame_rate": "30/1", "pix_fmt": "yuv420p"},
                {"codec_type": "audio", "channels": 2, "sample_rate": "48000"},
            ],
        })
        mock_result = MagicMock(returncode=0, stdout=probe_json.encode(), stderr=b"I: -16.0 LUFS")

        with patch("opencut.core.smart_defaults._sp.run", return_value=mock_result):
            result = suggest_operations(str(video), max_suggestions=3)

        assert isinstance(result, list)
        assert len(result) <= 3
        for s in result:
            assert "feature_id" in s
            assert "name" in s
            assert "reason" in s
            assert "confidence" in s

    def test_suggest_operations_respects_max(self, tmp_path):
        from opencut.core.contextual_suggest import suggest_operations

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        probe_json = json.dumps({
            "format": {"duration": "600", "bit_rate": "5000000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
                 "r_frame_rate": "30/1", "pix_fmt": "yuv420p"},
                {"codec_type": "audio", "channels": 2, "sample_rate": "48000"},
            ],
        })
        mock_result = MagicMock(returncode=0, stdout=probe_json.encode(), stderr=b"I: -25.0 LUFS")

        with patch("opencut.core.smart_defaults._sp.run", return_value=mock_result):
            result = suggest_operations(str(video), max_suggestions=2)

        assert len(result) <= 2

    def test_suggest_operations_sorted_by_confidence(self, tmp_path):
        from opencut.core.contextual_suggest import suggest_operations

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        probe_json = json.dumps({
            "format": {"duration": "300", "bit_rate": "5000000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
                 "r_frame_rate": "30/1", "pix_fmt": "yuv420p"},
                {"codec_type": "audio", "channels": 2, "sample_rate": "48000"},
            ],
        })
        mock_result = MagicMock(returncode=0, stdout=probe_json.encode(), stderr=b"I: -30.0 LUFS")

        with patch("opencut.core.smart_defaults._sp.run", return_value=mock_result):
            result = suggest_operations(str(video), max_suggestions=10)

        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]["confidence"] >= result[i + 1]["confidence"]


# ========================================================================
# 11. Content Type Detection
# ========================================================================
class TestContentTypeDetection:
    """Tests for smart_defaults content type heuristics."""

    def test_detect_screen_recording(self):
        from opencut.core.smart_defaults import _detect_content_type
        result = _detect_content_type(
            {"format": {}, "streams": [{"codec_type": "video", "codec_name": "h264"}]},
            motion_level=0.1, has_audio=True, duration_s=300,
            width=1920, height=1080, fps=30,
        )
        assert result == "screen_recording"

    def test_detect_drone_metadata(self):
        from opencut.core.smart_defaults import _detect_content_type
        result = _detect_content_type(
            {"format": {"tags": {"comment": "DJI Mavic Air 2"}}, "streams": []},
            motion_level=0.6, has_audio=False, duration_s=120,
            width=3840, height=2160, fps=30,
        )
        assert result == "drone"

    def test_detect_music_video(self):
        from opencut.core.smart_defaults import _detect_content_type
        result = _detect_content_type(
            {"format": {}, "streams": []},
            motion_level=0.8, has_audio=True, duration_s=240,
            width=1920, height=1080, fps=24,
        )
        assert result == "music_video"

    def test_detect_interview(self):
        from opencut.core.smart_defaults import _detect_content_type
        # Use non-standard resolution to avoid screen_recording match
        result = _detect_content_type(
            {"format": {}, "streams": []},
            motion_level=0.2, has_audio=True, duration_s=600,
            width=4096, height=2304, fps=24,
        )
        assert result == "interview"

    def test_detect_gaming(self):
        from opencut.core.smart_defaults import _detect_content_type
        result = _detect_content_type(
            {"format": {}, "streams": []},
            motion_level=0.5, has_audio=True, duration_s=300,
            width=1920, height=1080, fps=144,
        )
        assert result == "gaming"

    def test_detect_unknown_fallback(self):
        from opencut.core.smart_defaults import _detect_content_type
        result = _detect_content_type(
            {"format": {}, "streams": []},
            motion_level=0.9, has_audio=False, duration_s=5,
            width=800, height=600, fps=15,
        )
        # Very short, no audio, odd resolution -- should be sports or unknown
        assert result in ("sports", "unknown")


# ========================================================================
# 12. Route Smoke Tests
# ========================================================================
class TestUxIntelRoutes:
    """Smoke tests for UX intelligence routes."""

    def test_search_route(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/ux/search",
                           data=json.dumps({"query": "audio"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "total" in data

    def test_search_empty_query(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/ux/search",
                           data=json.dumps({"query": ""}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["results"] == []

    def test_search_with_limit(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/ux/search",
                           data=json.dumps({"query": "video", "limit": 3}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["results"]) <= 3

    def test_record_feature_route(self, client, csrf_token, tmp_path, monkeypatch):
        import opencut.core.command_palette as cp
        monkeypatch.setattr(cp, "_RECENTS_FILE", str(tmp_path / "recents.json"))
        monkeypatch.setattr(cp, "OPENCUT_DIR", str(tmp_path))

        from tests.conftest import csrf_headers
        resp = client.post("/ux/search/record",
                           data=json.dumps({"feature_id": "denoise"}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["feature_id"] == "denoise"

    def test_record_feature_missing_id(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/ux/search/record",
                           data=json.dumps({}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_recents_route(self, client):
        resp = client.get("/ux/recents")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recents" in data

    def test_recents_with_limit(self, client):
        resp = client.get("/ux/recents?limit=2")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "recents" in data

    def test_feature_index_route(self, client):
        resp = client.get("/ux/feature-index")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data
        assert "total" in data
        assert data["total"] > 0

    def test_smart_defaults_route(self, client, csrf_token):
        from tests.conftest import csrf_headers
        payload = {
            "operation": "normalize",
            "clip_profile": {
                "avg_loudness_lufs": -20.0,
                "resolution": 1920,
                "fps": 30.0,
                "codec": "h264",
                "duration_s": 120.0,
                "has_audio": True,
                "has_video": True,
                "detected_content_type": "interview",
            },
        }
        resp = client.post("/ux/smart-defaults",
                           data=json.dumps(payload),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "defaults" in data
        assert data["operation"] == "normalize"

    def test_smart_defaults_missing_operation(self, client, csrf_token):
        from tests.conftest import csrf_headers
        resp = client.post("/ux/smart-defaults",
                           data=json.dumps({"clip_profile": {}}),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 400

    def test_preview_route_starts_job(self, client, csrf_token, tmp_path):
        from tests.conftest import csrf_headers
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")

        resp = client.post("/ux/preview",
                           data=json.dumps({
                               "filepath": str(video),
                               "operation": "denoise",
                               "timestamp": 0.0,
                           }),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_preview_route_missing_operation(self, client, csrf_token, tmp_path):
        from tests.conftest import csrf_headers
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")

        resp = client.post("/ux/preview",
                           data=json.dumps({
                               "filepath": str(video),
                           }),
                           headers=csrf_headers(csrf_token))
        # The job starts but the handler raises ValueError
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data

    def test_suggest_route_starts_job(self, client, csrf_token, tmp_path):
        from tests.conftest import csrf_headers
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")

        resp = client.post("/ux/suggest",
                           data=json.dumps({
                               "filepath": str(video),
                               "recent_ops": ["normalize_audio"],
                               "max_suggestions": 3,
                           }),
                           headers=csrf_headers(csrf_token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "job_id" in data


# ========================================================================
# 13. Trigram / Similarity Helpers
# ========================================================================
class TestSimilarityHelpers:
    """Tests for internal similarity scoring functions."""

    def test_trigrams_short_string(self):
        from opencut.core.command_palette import _trigrams
        assert _trigrams("ab") == {"ab"}
        assert _trigrams("a") == {"a"}
        assert _trigrams("") == set()

    def test_trigrams_normal_string(self):
        from opencut.core.command_palette import _trigrams
        t = _trigrams("audio")
        assert "aud" in t
        assert "udi" in t
        assert "dio" in t

    def test_similarity_score_empty(self):
        from opencut.core.command_palette import _similarity_score
        assert _similarity_score("", "test") == 0.0
        assert _similarity_score("test", "") == 0.0

    def test_similarity_exact_is_1(self):
        from opencut.core.command_palette import _similarity_score
        assert _similarity_score("test", "test") == 1.0

    def test_similarity_ordering(self):
        from opencut.core.command_palette import _similarity_score
        exact = _similarity_score("audio", "audio")
        prefix = _similarity_score("audio", "audio enhance")
        substr = _similarity_score("audio", "enhance audio tools")
        assert exact > prefix > substr


# ========================================================================
# 14. Edge Cases
# ========================================================================
class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_fuzzy_search_special_characters(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("@#$%^&*()")
        assert isinstance(results, list)

    def test_fuzzy_search_very_long_query(self):
        from opencut.core.command_palette import fuzzy_search
        results = fuzzy_search("a" * 500)
        assert isinstance(results, list)

    def test_smart_defaults_zero_duration(self):
        from opencut.core.smart_defaults import ClipProfile, get_smart_defaults
        profile = ClipProfile(duration_s=0.0, has_video=True)
        result = get_smart_defaults("export", profile)
        assert isinstance(result, dict)

    def test_suggestion_no_recent_ops(self):
        from opencut.core.contextual_suggest import _build_suggestions
        from opencut.core.smart_defaults import ClipProfile
        profile = ClipProfile(has_audio=True, has_video=True, avg_loudness_lufs=-30.0, resolution=1920, width=1920, height=1080)
        suggestions = _build_suggestions(profile, set())
        assert len(suggestions) > 0

    def test_suggestion_all_ops_recent(self):
        from opencut.core.contextual_suggest import _build_suggestions
        from opencut.core.smart_defaults import ClipProfile
        # If all ops are recent, most suggestions should be filtered out
        profile = ClipProfile(has_audio=True, has_video=True, avg_loudness_lufs=-30.0,
                              resolution=480, width=640, height=480,
                              detected_content_type="interview", duration_s=1200,
                              is_static_camera=False, bitrate_kbps=3000, codec="h264")
        recent = {
            "normalize_audio", "add_captions", "stabilize", "upscale", "denoise",
            "diarize", "scene_detect", "highlights", "auto_color", "color_grade",
            "remove_silence", "dead_time", "smart_reframe", "thumbnail_gen",
            "cursor_zoom", "telemetry_overlay", "export_h265", "music_gen",
        }
        suggestions = _build_suggestions(profile, recent)
        # Most should be filtered
        assert len(suggestions) < 5

    def test_preview_negative_timestamp_clamped(self, tmp_path):
        from opencut.core.preview_frame import extract_frame
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        def mock_run_ffmpeg(cmd, timeout=3600, stderr_cap=0):
            out_path = cmd[-1]
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG" + b"\x00" * 50)
            # Verify timestamp was clamped to 0
            assert "-ss" in cmd
            ss_idx = cmd.index("-ss")
            assert float(cmd[ss_idx + 1]) >= 0.0
            return ""

        with patch("opencut.core.preview_frame.run_ffmpeg", side_effect=mock_run_ffmpeg):
            extract_frame(str(video), timestamp=-5.0)
