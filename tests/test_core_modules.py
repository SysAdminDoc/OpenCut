"""
Unit tests for 15 core OpenCut modules.

Tests pure logic functions with mocked external dependencies
(FFmpeg subprocess calls, AI models, filesystem operations).
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. silence.py
# ========================================================================
class TestSilence:
    """Tests for opencut.core.silence — silence detection engine."""

    def test_detect_silences_parses_ffmpeg_output(self):
        """detect_silences should parse silence_start/end from ffmpeg stderr."""
        ffmpeg_stderr = (
            "[silencedetect @ 0x1234] silence_start: 1.500\n"
            "[silencedetect @ 0x1234] silence_end: 3.200 | silence_duration: 1.700\n"
            "[silencedetect @ 0x1234] silence_start: 10.000\n"
            "[silencedetect @ 0x1234] silence_end: 12.500 | silence_duration: 2.500\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        with patch("opencut.core.silence.subprocess.run", return_value=mock_result), \
             patch("opencut.core.silence.probe") as mock_probe:
            mock_probe.return_value = MagicMock(duration=60.0)
            from opencut.core.silence import detect_silences
            silences = detect_silences("test.mp4", threshold_db=-30, min_duration=0.5, file_duration=60.0)

        assert len(silences) == 2
        assert silences[0].start == 1.5
        assert silences[0].end == 3.2
        assert silences[0].label == "silence"
        assert silences[1].start == 10.0
        assert silences[1].end == 12.5

    def test_detect_silences_empty_stderr(self):
        """No silence markers should return empty list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "some other output\nno silence here\n"

        with patch("opencut.core.silence.subprocess.run", return_value=mock_result), \
             patch("opencut.core.silence.probe") as mock_probe:
            mock_probe.return_value = MagicMock(duration=30.0)
            from opencut.core.silence import detect_silences
            silences = detect_silences("test.mp4", file_duration=30.0)

        assert silences == []

    def test_detect_silences_ffmpeg_not_found(self):
        """Should raise RuntimeError when ffmpeg is not installed."""
        with patch("opencut.core.silence.subprocess.run", side_effect=FileNotFoundError):
            from opencut.core.silence import detect_silences
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                detect_silences("test.mp4", file_duration=10.0)

    def test_detect_silences_timeout(self):
        """Should raise RuntimeError on timeout."""
        with patch("opencut.core.silence.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600)):
            from opencut.core.silence import detect_silences
            with pytest.raises(RuntimeError, match="timed out"):
                detect_silences("test.mp4", file_duration=10.0)

    def test_time_segment_duration(self):
        """TimeSegment.duration should compute end - start."""
        from opencut.core.silence import TimeSegment
        seg = TimeSegment(start=1.0, end=3.5, label="silence")
        assert seg.duration == 2.5

    def test_merge_overlapping(self):
        """_merge_overlapping should merge adjacent/overlapping segments."""
        from opencut.core.silence import TimeSegment, _merge_overlapping
        segs = [
            TimeSegment(start=0.0, end=2.0, label="a"),
            TimeSegment(start=1.5, end=4.0, label="a"),
            TimeSegment(start=6.0, end=8.0, label="a"),
        ]
        merged = _merge_overlapping(segs)
        assert len(merged) == 2
        assert merged[0].start == 0.0
        assert merged[0].end == 4.0
        assert merged[1].start == 6.0

    def test_format_time(self):
        """_format_time should produce HH:MM:SS.mmm."""
        from opencut.core.silence import _format_time
        assert _format_time(3661.5) == "01:01:01.500"
        assert _format_time(0.0) == "00:00:00.000"

    def test_build_atempo_chain(self):
        """_build_atempo_chain should chain atempo filters for high speeds."""
        from opencut.core.silence import _build_atempo_chain
        # 4x speed needs two stages: atempo=2.0,atempo=2.0
        result = _build_atempo_chain(4.0, "a0", "as0")
        assert "atempo=2.0" in result
        assert result.startswith("[a0]")
        assert result.endswith("[as0];")

    def test_get_edit_summary(self):
        """get_edit_summary should calculate correct stats."""
        from opencut.core.silence import TimeSegment, get_edit_summary
        segments = [
            TimeSegment(start=0.0, end=5.0, label="speech"),
            TimeSegment(start=8.0, end=10.0, label="speech"),
        ]
        with patch("opencut.core.silence.probe") as mock_probe:
            mock_probe.return_value = MagicMock(duration=20.0)
            summary = get_edit_summary("test.mp4", segments, file_duration=20.0)

        assert summary["original_duration"] == 20.0
        assert summary["kept_duration"] == 7.0
        assert summary["removed_duration"] == 13.0
        assert summary["segments_count"] == 2
        assert summary["reduction_percent"] == 65.0


# ========================================================================
# 2. fillers.py
# ========================================================================
class TestFillers:
    """Tests for opencut.core.fillers — filler word detection."""

    def _make_word(self, text, start, end, confidence=0.9):
        w = MagicMock()
        w.text = text
        w.start = start
        w.end = end
        w.confidence = confidence
        return w

    def _make_transcription(self, words_list):
        """Build a mock TranscriptionResult with one segment containing the given words."""
        seg = MagicMock()
        seg.words = words_list
        transcript = MagicMock()
        transcript.segments = [seg]
        return transcript

    def test_detect_safe_fillers(self):
        """Should detect hesitation sounds like 'um', 'uh'."""
        from opencut.core.fillers import detect_fillers
        words = [
            self._make_word("So", 0.0, 0.3),
            self._make_word("um", 0.5, 0.8),
            self._make_word("I", 1.0, 1.1),
            self._make_word("think", 1.1, 1.4),
            self._make_word("uh", 2.0, 2.3),
        ]
        transcript = self._make_transcription(words)
        result = detect_fillers(transcript, include_context_fillers=False)

        filler_texts = {h.filler_key for h in result.hits}
        assert "um" in filler_texts
        assert "uh" in filler_texts
        assert result.total_words == 5

    def test_detect_context_fillers(self):
        """Should detect context fillers like 'like', 'so' when enabled."""
        from opencut.core.fillers import detect_fillers
        words = [
            self._make_word("like", 0.0, 0.3),
            self._make_word("so", 0.5, 0.7),
            self._make_word("basically", 1.0, 1.5),
        ]
        transcript = self._make_transcription(words)
        result = detect_fillers(transcript, include_context_fillers=True)

        keys = {h.filler_key for h in result.hits}
        assert "like" in keys
        assert "so" in keys
        assert "basically" in keys

    def test_detect_phrase_fillers(self):
        """Should detect multi-word fillers like 'you know'."""
        from opencut.core.fillers import detect_fillers
        words = [
            self._make_word("you", 0.0, 0.2),
            self._make_word("know", 0.2, 0.5),
            self._make_word("it's", 0.6, 0.9),
            self._make_word("great", 0.9, 1.3),
        ]
        transcript = self._make_transcription(words)
        result = detect_fillers(transcript, include_context_fillers=True)

        assert any(h.filler_key == "you know" for h in result.hits)

    def test_custom_filler_words(self):
        """Should detect user-provided custom filler words."""
        from opencut.core.fillers import detect_fillers
        words = [
            self._make_word("dude", 0.0, 0.3),
            self._make_word("hello", 0.5, 0.8),
        ]
        transcript = self._make_transcription(words)
        result = detect_fillers(transcript, custom_words=["dude"])

        assert any(h.filler_key == "dude" for h in result.hits)

    def test_filler_percentage_calculation(self):
        """Filler percentage should be correctly computed."""
        from opencut.core.fillers import detect_fillers
        words = [
            self._make_word("um", 0.0, 0.3),
            self._make_word("hello", 0.5, 0.8),
        ]
        transcript = self._make_transcription(words)
        result = detect_fillers(transcript, include_context_fillers=False)

        assert result.total_words == 2
        assert result.filler_percentage == 50.0

    def test_remove_fillers_from_segments(self):
        """remove_fillers_from_segments should punch holes for filler words."""
        from opencut.core.fillers import FillerHit, remove_fillers_from_segments
        from opencut.core.silence import TimeSegment

        segments = [TimeSegment(start=0.0, end=10.0, label="speech")]
        hits = [
            FillerHit(text="um", filler_key="um", start=2.0, end=2.5,
                      confidence=0.9, safe=True),
        ]
        result = remove_fillers_from_segments(segments, hits, padding=0.0, min_gap=0.05)
        # Should split into: [0.0-2.0] and [2.5-10.0]
        assert len(result) == 2
        assert result[0].end <= 2.0
        assert result[1].start >= 2.5


# ========================================================================
# 3. scene_detect.py
# ========================================================================
class TestSceneDetect:
    """Tests for opencut.core.scene_detect."""

    def test_detect_scenes_parses_showinfo(self):
        """detect_scenes should parse pts_time from showinfo filter output."""
        showinfo_stderr = (
            "[Parsed_showinfo_1 @ 0x1234] n:   0 pts:    0 pts_time:0.000 ...\n"
            "[Parsed_showinfo_1 @ 0x1234] n:  150 pts: 5000 pts_time:5.000 ...\n"
            "[Parsed_showinfo_1 @ 0x1234] n:  400 pts: 13333 pts_time:13.333 ...\n"
        )
        probe_stdout = json.dumps({"format": {"duration": "60.0"}})

        mock_probe = MagicMock(returncode=0, stdout=probe_stdout, stderr="")
        mock_scene = MagicMock(returncode=0, stderr=showinfo_stderr, stdout="")

        with patch("opencut.core.scene_detect.subprocess.run",
                    side_effect=[mock_probe, mock_scene]):
            from opencut.core.scene_detect import detect_scenes
            info = detect_scenes("video.mp4", threshold=0.3, min_scene_length=2.0)

        # Should have "Start" at 0.0, plus 5.0 and 13.333
        assert info.total_scenes == 3
        assert info.boundaries[0].time == 0.0
        assert info.boundaries[1].time == 5.0
        assert abs(info.boundaries[2].time - 13.333) < 0.01

    def test_detect_scenes_min_scene_length(self):
        """Scenes too close together should be filtered by min_scene_length."""
        showinfo_stderr = (
            "[Parsed_showinfo_1 @ 0xab] n:  10 pts: 333 pts_time:1.000 showinfo stuff\n"
            "[Parsed_showinfo_1 @ 0xab] n:  20 pts: 666 pts_time:2.000 showinfo stuff\n"
        )
        probe_stdout = json.dumps({"format": {"duration": "30.0"}})
        mock_probe = MagicMock(returncode=0, stdout=probe_stdout, stderr="")
        mock_scene = MagicMock(returncode=0, stderr=showinfo_stderr, stdout="")

        with patch("opencut.core.scene_detect.subprocess.run",
                    side_effect=[mock_probe, mock_scene]):
            from opencut.core.scene_detect import detect_scenes
            # min_scene_length=5 means 1.0s and 2.0s are too close to each other
            # but 1.0s is far enough from 0.0 (Start) only if gap >= 5
            info = detect_scenes("video.mp4", min_scene_length=5.0)

        # Start at 0.0, but 1.0 and 2.0 are within 5s of 0.0 and each other
        assert info.total_scenes == 1  # Only the Start boundary

    def test_generate_chapter_markers_youtube(self):
        """YouTube format should produce 'M:SS Label' lines."""
        from opencut.core.scene_detect import SceneBoundary, SceneInfo, generate_chapter_markers
        scenes = SceneInfo(
            boundaries=[
                SceneBoundary(time=0.0, label="Intro"),
                SceneBoundary(time=65.0, label="Main"),
                SceneBoundary(time=3700.0, label="End"),
            ],
            total_scenes=3, duration=4000.0,
        )
        markers = generate_chapter_markers(scenes, format="youtube")
        lines = markers.strip().split("\n")
        assert lines[0] == "0:00 Intro"
        assert lines[1] == "1:05 Main"
        assert "1:01:40" in lines[2]

    def test_generate_speed_ramp_presets(self):
        """generate_speed_ramp should produce keyframes scaled to duration."""
        from opencut.core.scene_detect import generate_speed_ramp
        kf = generate_speed_ramp(10.0, preset="dramatic_pause")
        assert len(kf) >= 4
        assert kf[0]["time"] == 0.0
        assert kf[-1]["time"] == 10.0
        assert all("speed" in k for k in kf)

    def test_generate_chapter_markers_json(self):
        """JSON format should produce valid JSON."""
        from opencut.core.scene_detect import SceneBoundary, SceneInfo, generate_chapter_markers
        scenes = SceneInfo(
            boundaries=[SceneBoundary(time=0.0, label="Start")],
            total_scenes=1, duration=10.0,
        )
        result = generate_chapter_markers(scenes, format="json")
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["label"] == "Start"


# ========================================================================
# 4. auto_edit.py
# ========================================================================
class TestAutoEdit:
    """Tests for opencut.core.auto_edit — auto-editor integration."""

    def test_parse_auto_editor_json_v1_chunks(self, tmp_path):
        """Should parse v1 chunks format: [[start, end, speed], ...]."""
        from opencut.core.auto_edit import _parse_auto_editor_json
        chunks = [[0, 5, 1], [5, 10, 0], [10, 15, 1]]
        json_file = str(tmp_path / "ae.json")
        with open(json_file, "w") as f:
            json.dump({"chunks": chunks}, f)

        segments = _parse_auto_editor_json(json_file, 15.0)
        assert len(segments) == 3
        assert segments[0].action == "keep"
        assert segments[1].action == "cut"
        assert segments[2].action == "keep"

    def test_parse_auto_editor_json_v2_timeline(self, tmp_path):
        """Should parse v2 timeline format with tb (timebase)."""
        from opencut.core.auto_edit import _parse_auto_editor_json
        data = {
            "timeline": {
                "v": [[
                    {"offset": 0, "dur": 150, "speed": 1.0, "tb": 30},
                    {"offset": 150, "dur": 300, "speed": 99999, "tb": 30},
                ]]
            }
        }
        json_file = str(tmp_path / "ae2.json")
        with open(json_file, "w") as f:
            json.dump(data, f)

        segments = _parse_auto_editor_json(json_file, 15.0)
        assert len(segments) == 2
        assert segments[0].action == "keep"
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0  # 150/30
        assert segments[1].action == "cut"

    def test_parse_auto_editor_json_empty_fallback(self, tmp_path):
        """Empty JSON should produce single keep segment covering full duration."""
        from opencut.core.auto_edit import _parse_auto_editor_json
        json_file = str(tmp_path / "empty.json")
        with open(json_file, "w") as f:
            json.dump({}, f)

        segments = _parse_auto_editor_json(json_file, 30.0)
        assert len(segments) == 1
        assert segments[0].action == "keep"
        assert segments[0].end == 30.0

    def test_check_auto_editor_version_found(self):
        """Should return version string when auto-editor is available."""
        from opencut.core.auto_edit import check_auto_editor_version
        mock_result = MagicMock(returncode=0, stdout="24w51a\n")
        with patch("opencut.core.auto_edit.subprocess.run", return_value=mock_result):
            version = check_auto_editor_version()
        assert version == "24w51a"

    def test_check_auto_editor_version_not_found(self):
        """Should return None when auto-editor is not installed."""
        from opencut.core.auto_edit import check_auto_editor_version
        with patch("opencut.core.auto_edit.subprocess.run", side_effect=FileNotFoundError):
            version = check_auto_editor_version()
        assert version is None

    def test_edit_segment_duration(self):
        """EditSegment.duration should return end - start."""
        from opencut.core.auto_edit import EditSegment
        seg = EditSegment(start=2.0, end=7.0, action="keep")
        assert seg.duration == 5.0


# ========================================================================
# 5. highlights.py
# ========================================================================
class TestHighlights:
    """Tests for opencut.core.highlights — LLM highlight extraction."""

    def test_parse_highlights_json_valid(self):
        """Should parse well-formed JSON array of highlights."""
        from opencut.core.highlights import _parse_highlights_json
        text = json.dumps([
            {"start": 10, "end": 25, "score": 0.9, "reason": "Funny joke", "title": "LOL"},
            {"start": 60, "end": 80, "score": 0.7, "reason": "Key insight", "title": "Aha"},
        ])
        highlights = _parse_highlights_json(text)
        assert len(highlights) == 2
        assert highlights[0].start == 10.0
        assert highlights[0].score == 0.9
        assert highlights[1].title == "Aha"

    def test_parse_highlights_json_with_code_fences(self):
        """Should handle markdown code fences wrapping JSON."""
        from opencut.core.highlights import _parse_highlights_json
        text = '```json\n[{"start": 5, "end": 20, "score": 0.8, "reason": "test", "title": "t"}]\n```'
        highlights = _parse_highlights_json(text)
        assert len(highlights) == 1
        assert highlights[0].start == 5.0

    def test_parse_highlights_json_embedded_in_text(self):
        """Should find JSON array embedded in prose response."""
        from opencut.core.highlights import _parse_highlights_json
        text = 'Here are the highlights:\n[{"start": 30, "end": 45, "score": 0.6}]\nHope that helps!'
        highlights = _parse_highlights_json(text)
        assert len(highlights) == 1
        assert highlights[0].start == 30.0

    def test_parse_highlights_json_garbage(self):
        """Completely unparseable text should return empty list."""
        from opencut.core.highlights import _parse_highlights_json
        highlights = _parse_highlights_json("I don't know what you mean")
        assert highlights == []

    def test_format_transcript_for_llm(self):
        """Should format segments as timestamped lines."""
        from opencut.core.highlights import _format_transcript_for_llm
        segments = [
            {"start": 0, "end": 5, "text": "Hello world"},
            {"start": 65, "end": 70, "text": "Second part"},
        ]
        result = _format_transcript_for_llm(segments)
        assert "[00:00 - 00:05] Hello world" in result
        assert "[01:05 - 01:10] Second part" in result

    def test_format_transcript_truncation(self):
        """Should truncate to max_chars."""
        from opencut.core.highlights import _format_transcript_for_llm
        segments = [{"start": 0, "end": 1, "text": "x" * 200}]
        result = _format_transcript_for_llm(segments, max_chars=50)
        assert len(result) <= 100  # truncated + "[...transcript truncated...]"
        assert "truncated" in result

    def test_extract_highlights_with_mock_llm(self):
        """extract_highlights should parse LLM response into HighlightResult."""
        from opencut.core.highlights import extract_highlights

        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"start": 10, "end": 30, "score": 0.95, "reason": "Great moment", "title": "Wow"},
        ])
        mock_response.provider = "openai"
        mock_response.model = "gpt-4"

        with patch("opencut.core.llm.query_llm", return_value=mock_response), \
             patch("opencut.core.llm.LLMConfig"):
            result = extract_highlights(
                [{"start": 0, "end": 60, "text": "test transcript"}],
                max_highlights=5, min_duration=15.0, max_duration=60.0,
            )

        assert result.total_found == 1
        assert result.highlights[0].start == 10.0
        assert result.highlights[0].end == 30.0
        assert result.highlights[0].score > 0
        assert result.highlights[0].reason == "Great moment"
        assert result.llm_provider == "openai"


# ========================================================================
# 6. workflow.py
# ========================================================================
class TestWorkflow:
    """Tests for opencut.core.workflow — workflow validation and chaining."""

    def test_validate_valid_steps(self):
        """Should accept valid endpoint steps."""
        from opencut.core.workflow import validate_workflow_steps
        steps = [
            {"endpoint": "/silence"},
            {"endpoint": "/fillers"},
            {"endpoint": "/export-video", "params": {"preset": "youtube_1080p"}},
        ]
        ok, err = validate_workflow_steps(steps)
        assert ok is True
        assert err == ""

    def test_validate_empty_steps(self):
        """Should reject empty step list."""
        from opencut.core.workflow import validate_workflow_steps
        ok, err = validate_workflow_steps([])
        assert ok is False
        assert "at least one step" in err

    def test_validate_unknown_endpoint(self):
        """Should reject unknown endpoints."""
        from opencut.core.workflow import validate_workflow_steps
        steps = [{"endpoint": "/nonexistent/endpoint"}]
        ok, err = validate_workflow_steps(steps)
        assert ok is False
        assert "unknown endpoint" in err.lower()

    def test_validate_missing_endpoint_key(self):
        """Should reject steps without endpoint key."""
        from opencut.core.workflow import validate_workflow_steps
        steps = [{"params": {"foo": "bar"}}]
        ok, err = validate_workflow_steps(steps)
        assert ok is False
        assert "missing" in err.lower()

    def test_validate_non_dict_step(self):
        """Should reject non-dict step entries."""
        from opencut.core.workflow import validate_workflow_steps
        steps = ["not a dict"]
        ok, err = validate_workflow_steps(steps)
        assert ok is False
        assert "not a valid object" in err.lower()

    def test_extract_output_path_common_keys(self):
        """_extract_output_path should find output path from various keys."""
        from opencut.core.workflow import _extract_output_path
        with patch("os.path.isfile", return_value=True):
            assert _extract_output_path({"output_path": "/a/b.mp4"}, "/fallback") == "/a/b.mp4"
            assert _extract_output_path({"output": "/x/y.mp4"}, "/fallback") == "/x/y.mp4"
            assert _extract_output_path({"file": "/c/d.mp4"}, "/fallback") == "/c/d.mp4"

    def test_extract_output_path_fallback(self):
        """Should return fallback when no output key is found."""
        from opencut.core.workflow import _extract_output_path
        result = _extract_output_path({"no_output": True}, "/fallback.mp4")
        assert result == "/fallback.mp4"


# ========================================================================
# 7. speed_ramp.py
# ========================================================================
class TestSpeedRamp:
    """Tests for opencut.core.speed_ramp — speed change and ramping."""

    def test_easing_functions(self):
        """Easing functions should map [0,1] to [0,1]."""
        from opencut.core.speed_ramp import (
            _ease_exponential,
            _ease_in,
            _ease_in_out,
            _ease_linear,
            _ease_out,
        )
        for fn in [_ease_linear, _ease_in, _ease_out, _ease_in_out, _ease_exponential]:
            assert fn(0.0) == pytest.approx(0.0, abs=1e-9)
            assert fn(1.0) == pytest.approx(1.0, abs=1e-9)
            # Should be monotonically increasing for typical values
            assert fn(0.5) >= 0.0
            assert fn(0.5) <= 1.0

    def test_build_atempo_chain_normal(self):
        """atempo chain for 1.5x should be a single filter."""
        from opencut.core.speed_ramp import _build_atempo_chain
        result = _build_atempo_chain(1.5)
        assert "atempo=" in result
        assert result.count("atempo=") == 1

    def test_build_atempo_chain_high_speed(self):
        """atempo chain for 4x should chain two 2.0 stages."""
        from opencut.core.speed_ramp import _build_atempo_chain
        result = _build_atempo_chain(4.0)
        assert result.count("atempo=2.0") >= 1

    def test_build_atempo_chain_slow_speed(self):
        """atempo chain for 0.25x should chain multiple 0.5 stages."""
        from opencut.core.speed_ramp import _build_atempo_chain
        result = _build_atempo_chain(0.25)
        assert "atempo=0.5" in result

    def test_build_atempo_chain_unity(self):
        """atempo chain for 1.0x should return empty string."""
        from opencut.core.speed_ramp import _build_atempo_chain
        result = _build_atempo_chain(1.0)
        assert result == ""

    def test_build_atempo_maintain_pitch(self):
        """maintain_pitch should use rubberband filter."""
        from opencut.core.speed_ramp import _build_atempo_chain
        result = _build_atempo_chain(2.0, maintain_pitch=True)
        assert "rubberband" in result

    def test_speed_ramp_requires_two_keyframes(self):
        """speed_ramp should reject less than 2 keyframes."""
        from opencut.core.speed_ramp import speed_ramp
        with pytest.raises(ValueError, match="at least 2 keyframes"):
            speed_ramp("test.mp4", keyframes=[{"time": 0, "speed": 1.0}])

    def test_get_speed_ramp_presets(self):
        """get_speed_ramp_presets should return list of preset dicts."""
        from opencut.core.speed_ramp import get_speed_ramp_presets
        presets = get_speed_ramp_presets()
        assert len(presets) >= 3
        assert all("name" in p and "label" in p for p in presets)


# ========================================================================
# 8. audio.py
# ========================================================================
class TestAudio:
    """Tests for opencut.core.audio — audio analysis utilities."""

    def test_extract_audio_pcm_builds_correct_command(self):
        """extract_audio_pcm should call ffmpeg with correct args."""
        from opencut.core.audio import extract_audio_pcm
        mock_result = MagicMock(returncode=0, stdout=b"\x00\x00" * 100, stderr=b"")
        with patch("opencut.core.audio.subprocess.run", return_value=mock_result) as mock_run:
            pcm_data, sr = extract_audio_pcm("test.mp4", sample_rate=16000, mono=True)
            call_args = mock_run.call_args[0][0]
            assert "ffmpeg" in call_args[0]
            assert "-ar" in call_args
            assert "16000" in call_args
            assert "-ac" in call_args
            assert "1" in call_args
            assert sr == 16000

    def test_extract_audio_pcm_error(self):
        """Should raise RuntimeError on ffmpeg failure."""
        from opencut.core.audio import extract_audio_pcm
        mock_result = MagicMock(returncode=1, stderr=b"error occurred")
        with patch("opencut.core.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Audio extraction failed"):
                extract_audio_pcm("test.mp4")

    def test_analyze_energy_computes_rms(self):
        """analyze_energy should compute RMS and peak from PCM data."""
        import struct
        # Create a simple sine-like PCM signal: 16kHz, 1 second
        # Use a constant value for predictable RMS
        sample_rate = 16000
        num_samples = sample_rate  # 1 second
        value = 16384  # half of max 32768
        pcm_bytes = struct.pack(f"<{num_samples}h", *([value] * num_samples))

        with patch("opencut.core.audio.extract_audio_pcm",
                    return_value=(pcm_bytes, sample_rate)):
            from opencut.core.audio import analyze_energy
            energies = analyze_energy("test.mp4", window_size=0.05, hop_size=0.025)

        assert len(energies) > 0
        # Constant signal: RMS should be ~0.5 (16384/32768)
        assert energies[0].rms == pytest.approx(0.5, abs=0.01)
        assert energies[0].peak == pytest.approx(0.5, abs=0.01)

    def test_audio_energy_dataclass(self):
        """AudioEnergy should store time, rms, peak."""
        from opencut.core.audio import AudioEnergy
        e = AudioEnergy(time=1.5, rms=0.3, peak=0.8)
        assert e.time == 1.5
        assert e.rms == 0.3
        assert e.peak == 0.8


# ========================================================================
# 9. face_reframe.py
# ========================================================================
class TestFaceReframe:
    """Tests for opencut.core.face_reframe — face tracking and crop."""

    def test_smooth_tracks_fills_gaps(self):
        """_smooth_tracks should interpolate between detected frames."""
        from opencut.core.face_reframe import FaceTrack, _smooth_tracks

        tracks = [
            FaceTrack(frame=0, time=0.0, cx=0.2, cy=0.5, w=0.1, h=0.1, confidence=0.9),
            FaceTrack(frame=10, time=0.333, cx=0.8, cy=0.5, w=0.1, h=0.1, confidence=0.9),
        ]
        positions = _smooth_tracks(tracks, smoothing=0.0, total_frames=11, fps=30.0)

        assert len(positions) == 11
        # At frame 0: cx=0.2
        assert positions[0][0] == pytest.approx(0.2, abs=0.01)
        # At frame 10: cx=0.8
        assert positions[10][0] == pytest.approx(0.8, abs=0.01)
        # Midpoint (frame 5) should be between 0.2 and 0.8
        assert 0.3 < positions[5][0] < 0.7

    def test_smooth_tracks_empty(self):
        """Empty tracks should default to center (0.5, 0.5)."""
        from opencut.core.face_reframe import _smooth_tracks
        positions = _smooth_tracks([], smoothing=0.5, total_frames=10, fps=30.0)
        assert len(positions) == 10
        assert all(p == (0.5, 0.5) for p in positions)

    def test_build_crop_expression_single_window(self):
        """Single-window crop should return simple integer coordinates."""
        from opencut.core.face_reframe import _build_crop_expression
        positions = [(0.5, 0.5)] * 30  # 1 second at 30fps
        x_expr, y_expr = _build_crop_expression(
            positions, src_w=1920, src_h=1080, crop_w=608, crop_h=1080, fps=30.0,
        )
        # Should be simple integers, not complex expressions
        assert x_expr.isdigit()
        assert y_expr.isdigit()

    def test_build_crop_expression_multiple_windows(self):
        """Multiple windows should produce if(between(...)) FFmpeg expressions."""
        from opencut.core.face_reframe import _build_crop_expression
        # 3 seconds of positions with a shift at 1.5s
        positions = [(0.3, 0.5)] * 45 + [(0.7, 0.5)] * 45
        x_expr, y_expr = _build_crop_expression(
            positions, src_w=1920, src_h=1080, crop_w=608, crop_h=1080, fps=30.0,
        )
        assert "between" in x_expr

    def test_face_track_dataclass(self):
        """FaceTrack should store all detection fields."""
        from opencut.core.face_reframe import FaceTrack
        ft = FaceTrack(frame=5, time=0.167, cx=0.5, cy=0.4, w=0.2, h=0.3, confidence=0.95)
        assert ft.cx == 0.5
        assert ft.confidence == 0.95


# ========================================================================
# 10. chromakey.py
# ========================================================================
class TestChromakey:
    """Tests for opencut.core.chromakey — chroma key compositing."""

    def test_chroma_presets_exist(self):
        """Should have green, blue, red presets."""
        from opencut.core.chromakey import CHROMA_PRESETS
        assert "green" in CHROMA_PRESETS
        assert "blue" in CHROMA_PRESETS
        assert "red" in CHROMA_PRESETS
        # Each preset should have lower and upper HSV bounds
        for name, preset in CHROMA_PRESETS.items():
            assert "lower" in preset
            assert "upper" in preset
            assert len(preset["lower"]) == 3
            assert len(preset["upper"]) == 3

    def test_pip_position_map(self):
        """picture_in_picture should support various position strings."""
        # We just test that the function references known positions
        # by checking the function signature exists
        import inspect

        from opencut.core.chromakey import picture_in_picture
        sig = inspect.signature(picture_in_picture)
        assert "position" in sig.parameters

    def test_blend_modes_list(self):
        """BLEND_MODES should contain standard compositing modes."""
        from opencut.core.chromakey import BLEND_MODES
        assert "multiply" in BLEND_MODES
        assert "screen" in BLEND_MODES
        assert "overlay" in BLEND_MODES
        assert "normal" in BLEND_MODES
        assert len(BLEND_MODES) >= 10


# ========================================================================
# 11. video_fx.py
# ========================================================================
class TestVideoFx:
    """Tests for opencut.core.video_fx — FFmpeg-based video effects."""

    def test_chromakey_validates_color(self):
        """chromakey should sanitize invalid color hex to default."""
        from opencut.core.video_fx import chromakey
        with patch("opencut.core.video_fx.run_ffmpeg") as mock_run, \
             patch("opencut.core.video_fx._output_path", return_value="/out/test.mov"):
            chromakey("test.mp4", color="INVALID_COLOR")
            # Should have defaulted to 0x00FF00 (green)
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "0x00FF00" in filter_str

    def test_chromakey_with_background(self):
        """chromakey with background should use overlay filter."""
        from opencut.core.video_fx import chromakey
        with patch("opencut.core.video_fx.run_ffmpeg") as mock_run, \
             patch("opencut.core.video_fx._output_path", return_value="/out/test.mp4"), \
             patch("os.path.isfile", return_value=True):
            chromakey("fg.mp4", color="0x00FF00", background="/bg/image.jpg")
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "overlay" in filter_str

    def test_apply_vignette_builds_filter(self):
        """apply_vignette should build a vignette filter command."""
        from opencut.core.video_fx import apply_vignette
        with patch("opencut.core.video_fx.run_ffmpeg") as mock_run, \
             patch("opencut.core.video_fx._output_path", return_value="/out/test.mp4"):
            apply_vignette("test.mp4", intensity=0.5)
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "vignette" in filter_str

    def test_apply_film_grain_intensity_mapping(self):
        """Film grain strength should scale with intensity."""
        from opencut.core.video_fx import apply_film_grain
        with patch("opencut.core.video_fx.run_ffmpeg") as mock_run, \
             patch("opencut.core.video_fx._output_path", return_value="/out/test.mp4"):
            apply_film_grain("test.mp4", intensity=1.0)
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            # intensity=1.0 -> strength=5+1.0*35=40
            assert "alls=40" in filter_str

    def test_get_available_video_effects(self):
        """Should return list of effect dicts."""
        from opencut.core.video_fx import get_available_video_effects
        effects = get_available_video_effects()
        assert len(effects) >= 5
        names = {e["name"] for e in effects}
        assert "stabilize" in names
        assert "vignette" in names
        assert "film_grain" in names

    def test_apply_letterbox_invalid_aspect(self):
        """Invalid aspect ratio should raise ValueError."""
        from opencut.core.video_fx import apply_letterbox
        with patch("opencut.core.video_fx._output_path", return_value="/out/test.mp4"):
            with pytest.raises(ValueError, match="Invalid aspect ratio"):
                apply_letterbox("test.mp4", aspect="bad")


# ========================================================================
# 12. export_presets.py
# ========================================================================
class TestExportPresets:
    """Tests for opencut.core.export_presets — export profiles."""

    def test_get_export_presets_returns_all(self):
        """Should return all preset entries with required keys."""
        from opencut.core.export_presets import get_export_presets
        presets = get_export_presets()
        assert len(presets) >= 10
        for p in presets:
            assert "name" in p
            assert "label" in p
            assert "category" in p

    def test_get_preset_categories(self):
        """Should group presets into categories."""
        from opencut.core.export_presets import get_preset_categories
        cats = get_preset_categories()
        cat_names = {c["name"] for c in cats}
        assert "youtube" in cat_names
        assert "social" in cat_names
        assert "audio" in cat_names

    def test_export_with_unknown_preset(self):
        """Should raise ValueError for unknown preset name."""
        from opencut.core.export_presets import export_with_preset
        with pytest.raises(ValueError, match="Unknown preset"):
            export_with_preset("test.mp4", "nonexistent_preset_xyz")

    def test_export_preset_audio_only(self):
        """Audio-only preset should include -vn flag."""
        from opencut.core.export_presets import export_with_preset
        with patch("opencut.core.export_presets.run_ffmpeg") as mock_run:
            export_with_preset("test.mp4", "podcast_mp3", output_path="/out/test.mp3")
            call_args = mock_run.call_args[0][0]
            assert "-vn" in call_args

    def test_export_preset_video_scaling(self):
        """Video preset should include scale/pad filters for target resolution."""
        from opencut.core.export_presets import export_with_preset
        with patch("opencut.core.export_presets.run_ffmpeg") as mock_run:
            export_with_preset("test.mp4", "youtube_1080p", output_path="/out/test.mp4")
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "1920" in filter_str
            assert "1080" in filter_str

    def test_export_preset_max_duration(self):
        """Presets with max_duration should add -t flag."""
        from opencut.core.export_presets import export_with_preset
        with patch("opencut.core.export_presets.run_ffmpeg") as mock_run:
            export_with_preset("test.mp4", "youtube_shorts", output_path="/out/test.mp4")
            call_args = mock_run.call_args[0][0]
            assert "-t" in call_args
            idx = call_args.index("-t")
            assert call_args[idx + 1] == "60"

    def test_preset_data_integrity(self):
        """All video presets should have codec and resolution."""
        from opencut.core.export_presets import EXPORT_PRESETS
        for name, p in EXPORT_PRESETS.items():
            assert "ext" in p, f"Preset {name} missing ext"
            assert "category" in p, f"Preset {name} missing category"
            if not p.get("audio_only") and p.get("ext") != ".gif":
                assert "codec" in p, f"Preset {name} missing codec"


# ========================================================================
# 13. diarize.py
# ========================================================================
class TestDiarize:
    """Tests for opencut.core.diarize — speaker diarization."""

    def test_speaker_segment_duration(self):
        """SpeakerSegment.duration should compute end - start."""
        from opencut.core.diarize import SpeakerSegment
        seg = SpeakerSegment(speaker="SPEAKER_00", start=1.0, end=5.0)
        assert seg.duration == 4.0

    def test_diarization_result_speaker_durations(self):
        """get_speaker_durations should sum time per speaker."""
        from opencut.core.diarize import DiarizationResult, SpeakerSegment
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
                SpeakerSegment(speaker="SPEAKER_01", start=5.0, end=8.0),
                SpeakerSegment(speaker="SPEAKER_00", start=8.0, end=12.0),
            ],
            num_speakers=2,
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        durations = result.get_speaker_durations()
        assert durations["SPEAKER_00"] == pytest.approx(9.0)
        assert durations["SPEAKER_01"] == pytest.approx(3.0)

    def test_diarization_result_filter_by_speaker(self):
        """get_speaker_segments should return only matching speaker."""
        from opencut.core.diarize import DiarizationResult, SpeakerSegment
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker="A", start=0, end=3),
                SpeakerSegment(speaker="B", start=3, end=6),
                SpeakerSegment(speaker="A", start=6, end=9),
            ],
            num_speakers=2,
            speakers=["A", "B"],
        )
        a_segs = result.get_speaker_segments("A")
        assert len(a_segs) == 2
        assert all(s.speaker == "A" for s in a_segs)

    def test_camera_switches(self):
        """to_camera_switches should convert speakers to camera labels."""
        from opencut.core.diarize import DiarizationResult, SpeakerSegment
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker="SPEAKER_00", start=0, end=5),
                SpeakerSegment(speaker="SPEAKER_01", start=5, end=10),
            ],
            num_speakers=2,
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        switches = result.to_camera_switches()
        assert len(switches) == 2
        assert switches[0].label == "camera_0"
        assert switches[1].label == "camera_1"

    def test_merge_short_segments(self):
        """Short segments should be absorbed into previous segment."""
        from opencut.core.diarize import DiarizationResult, SpeakerSegment
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker="A", start=0, end=5),
                SpeakerSegment(speaker="B", start=5, end=5.5),  # 0.5s < 1.0s min
                SpeakerSegment(speaker="A", start=5.5, end=10),
            ],
            num_speakers=2,
            speakers=["A", "B"],
        )
        merged = result._merge_short_segments(min_duration=1.0)
        # B's 0.5s segment should be absorbed
        assert len(merged) <= 2

    def test_total_duration(self):
        """total_duration should sum all segment durations."""
        from opencut.core.diarize import DiarizationResult, SpeakerSegment
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker="A", start=0, end=3),
                SpeakerSegment(speaker="B", start=3, end=7),
            ],
        )
        assert result.total_duration == pytest.approx(7.0)


# ========================================================================
# 14. audio_duck.py
# ========================================================================
class TestAudioDuck:
    """Tests for opencut.core.audio_duck — audio ducking."""

    def test_sidechain_duck_builds_filter(self):
        """sidechain_duck should build sidechaincompress filter command."""
        from opencut.core.audio_duck import sidechain_duck
        with patch("opencut.core.audio_duck.run_ffmpeg") as mock_run:
            sidechain_duck("music.wav", "voice.wav", output_path="/out/ducked.wav",
                           duck_amount=0.7, threshold=0.015, attack=20.0, release=300.0)
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "sidechaincompress" in filter_str
            assert "threshold=0.015" in filter_str
            assert "-i" in call_args

    def test_sidechain_duck_ratio_clamping(self):
        """Duck amount should be clamped to ratio range 2-20."""
        from opencut.core.audio_duck import sidechain_duck
        with patch("opencut.core.audio_duck.run_ffmpeg") as mock_run:
            # duck_amount=0.0 -> ratio=max(2, 0*20=0)=2
            sidechain_duck("m.wav", "v.wav", output_path="/o.wav", duck_amount=0.0)
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "ratio=2:" in filter_str

    def test_mix_with_duck_includes_volume(self):
        """mix_with_duck should set music volume before ducking."""
        from opencut.core.audio_duck import mix_with_duck
        with patch("opencut.core.audio_duck.run_ffmpeg") as mock_run:
            mix_with_duck("voice.wav", "music.wav", output_path="/o.wav",
                          music_volume=0.3, duck_amount=0.6)
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "volume=0.3" in filter_str
            assert "amix" in filter_str

    def test_auto_duck_video_maps_streams(self):
        """auto_duck_video should map video and mixed audio."""
        from opencut.core.audio_duck import auto_duck_video
        with patch("opencut.core.audio_duck.run_ffmpeg") as mock_run:
            auto_duck_video("video.mp4", "music.wav", output_path="/o.mp4")
            call_args = mock_run.call_args[0][0]
            assert "-map" in call_args
            assert "0:v" in call_args

    def test_mix_audio_tracks_empty(self):
        """mix_audio_tracks should raise on empty tracks list."""
        from opencut.core.audio_duck import mix_audio_tracks
        with pytest.raises(ValueError, match="No tracks"):
            mix_audio_tracks([])

    def test_mix_audio_tracks_builds_filter(self):
        """mix_audio_tracks should chain volume filters and amix."""
        from opencut.core.audio_duck import mix_audio_tracks
        with patch("opencut.core.audio_duck.run_ffmpeg") as mock_run:
            tracks = [
                {"path": "a.wav", "volume": 1.0},
                {"path": "b.wav", "volume": 0.5},
            ]
            mix_audio_tracks(tracks, output_path="/o.wav")
            call_args = mock_run.call_args[0][0]
            filter_str = " ".join(call_args)
            assert "amix=inputs=2" in filter_str
            assert "volume=0.5" in filter_str


# ========================================================================
# 15. thumbnail.py
# ========================================================================
class TestThumbnail:
    """Tests for opencut.core.thumbnail — frame scoring for thumbnails."""

    def test_score_frame_colorful_image(self):
        """A colorful, sharp, well-exposed frame should score high."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")

        # Create a colorful gradient image (high saturation, good contrast)
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            frame[i, :, 0] = int(255 * i / h)  # B gradient
            frame[i, :, 1] = 128                # G constant
            frame[i, :, 2] = int(255 * (1 - i / h))  # R gradient

        from opencut.core.thumbnail import _score_frame
        score = _score_frame(frame)
        # Should get points for color, contrast, and exposure
        assert score > 20

    def test_score_frame_dark_image(self):
        """A very dark frame should score low."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")

        frame = np.full((480, 640, 3), 10, dtype=np.uint8)  # Near black

        from opencut.core.thumbnail import _score_frame
        score = _score_frame(frame)
        # Low saturation, low contrast, dark = low score
        assert score < 20

    def test_score_frame_uniform_penalty(self):
        """Solid color frames (title cards) should get penalized."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        from opencut.core.thumbnail import _score_frame
        score = _score_frame(frame)
        # Uniform frame: std_dev < 20, gets -15 penalty
        assert score < 15

    def test_score_frame_with_face_detector(self):
        """Face detection bonus should increase score."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")

        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

        # Mock face detector that returns one face
        def mock_face_detect(f):
            return [(200, 150, 100, 120)]  # (x, y, w, h)

        from opencut.core.thumbnail import _score_frame
        score_with_face = _score_frame(frame, has_face_detector=True,
                                       face_detector=mock_face_detect)
        score_without = _score_frame(frame, has_face_detector=False)
        assert score_with_face > score_without
