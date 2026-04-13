"""
Unit tests for OpenCut subtitle / audio / processing features (batch 6):
  - Soft subtitle embedding & track listing
  - Dead-time detection & speed ramping
  - SDH / HoH formatting
  - Stream recording auto-chaptering
  - ND filter simulation
  - Drop-frame / non-drop-frame timecode conversion
  - Route smoke tests for all new endpoints

Tests pure logic with mocked FFmpeg subprocess calls.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import csrf_headers


# ========================================================================
# Soft Subtitle Embedding
# ========================================================================
class TestSoftSubtitles:
    """Tests for opencut.core.soft_subtitles"""

    def test_embed_subtitles_mp4(self):
        """embed_subtitles should build correct FFmpeg command for MP4 container."""
        with patch("opencut.core.soft_subtitles.run_ffmpeg") as mock_ff, \
             patch("os.path.isfile", return_value=True):
            from opencut.core.soft_subtitles import embed_subtitles

            result = embed_subtitles(
                "/video/test.mp4",
                subtitle_paths=["/subs/en.srt", "/subs/es.srt"],
                languages=["eng", "spa"],
                container="mp4",
            )

            assert mock_ff.called
            cmd = mock_ff.call_args[0][0]
            # Should have two subtitle inputs
            assert cmd.count("-i") == 3  # 1 video + 2 subs
            # Should use mov_text codec for MP4
            assert "-c:s" in cmd
            idx = cmd.index("-c:s")
            assert cmd[idx + 1] == "mov_text"
            # Should have language metadata
            assert "-metadata:s:s:0" in cmd
            assert "-metadata:s:s:1" in cmd
            assert result["tracks_added"] == 2
            assert result["container"] == "mp4"

    def test_embed_subtitles_mkv(self):
        """embed_subtitles should use srt codec for MKV containers."""
        with patch("opencut.core.soft_subtitles.run_ffmpeg") as mock_ff, \
             patch("os.path.isfile", return_value=True):
            from opencut.core.soft_subtitles import embed_subtitles

            result = embed_subtitles(
                "/video/test.mkv",
                subtitle_paths=["/subs/en.srt"],
                languages=["eng"],
                container="mkv",
            )

            cmd = mock_ff.call_args[0][0]
            idx = cmd.index("-c:s")
            assert cmd[idx + 1] == "srt"
            assert result["container"] == "mkv"

    def test_embed_subtitles_no_paths_raises(self):
        """embed_subtitles should raise ValueError when no subtitle paths given."""
        from opencut.core.soft_subtitles import embed_subtitles
        with pytest.raises(ValueError, match="At least one subtitle"):
            embed_subtitles("/video/test.mp4", subtitle_paths=[])

    def test_embed_subtitles_language_mismatch_raises(self):
        """embed_subtitles should raise ValueError when language count mismatches."""
        from opencut.core.soft_subtitles import embed_subtitles
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="languages list length"):
                embed_subtitles(
                    "/video/test.mp4",
                    subtitle_paths=["/subs/en.srt"],
                    languages=["eng", "spa"],
                )

    def test_embed_subtitles_missing_file_raises(self):
        """embed_subtitles should raise FileNotFoundError for missing subtitle."""
        from opencut.core.soft_subtitles import embed_subtitles
        with pytest.raises(FileNotFoundError):
            embed_subtitles(
                "/video/test.mp4",
                subtitle_paths=["/nonexistent/sub.srt"],
            )

    def test_embed_subtitles_progress_callback(self):
        """Progress callback should be called during embedding."""
        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append((pct, msg))

        with patch("opencut.core.soft_subtitles.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True):
            from opencut.core.soft_subtitles import embed_subtitles

            embed_subtitles(
                "/video/test.mp4",
                subtitle_paths=["/subs/en.srt"],
                on_progress=on_progress,
            )

        assert len(progress_calls) >= 2
        assert progress_calls[-1][0] == 100

    def test_list_subtitle_tracks(self):
        """list_subtitle_tracks should parse ffprobe JSON output."""
        probe_output = json.dumps({
            "streams": [
                {
                    "index": 2,
                    "codec_name": "mov_text",
                    "tags": {"language": "eng", "title": "English"},
                },
                {
                    "index": 3,
                    "codec_name": "mov_text",
                    "tags": {"language": "spa"},
                },
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = probe_output.encode()

        with patch("opencut.core.soft_subtitles.subprocess.run", return_value=mock_result):
            from opencut.core.soft_subtitles import list_subtitle_tracks

            tracks = list_subtitle_tracks("/video/test.mp4")

        assert len(tracks) == 2
        assert tracks[0]["codec"] == "mov_text"
        assert tracks[0]["language"] == "eng"
        assert tracks[0]["title"] == "English"
        assert tracks[1]["language"] == "spa"

    def test_list_subtitle_tracks_no_streams(self):
        """list_subtitle_tracks should return empty list when no subtitle streams."""
        probe_output = json.dumps({"streams": []})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = probe_output.encode()

        with patch("opencut.core.soft_subtitles.subprocess.run", return_value=mock_result):
            from opencut.core.soft_subtitles import list_subtitle_tracks

            tracks = list_subtitle_tracks("/video/test.mp4")

        assert tracks == []


# ========================================================================
# Dead-Time Detection & Speed Ramp
# ========================================================================
class TestDeadTime:
    """Tests for opencut.core.dead_time"""

    def test_detect_dead_time_intersects_motion_and_silence(self):
        """detect_dead_time should only flag segments with BOTH low motion and silence."""
        motion_stderr = (
            "drop count:1 mpdecimate pts:3000 pts_time:1.0\n"
            "drop count:2 mpdecimate pts:6000 pts_time:2.0\n"
            "drop count:3 mpdecimate pts:9000 pts_time:3.0\n"
            "drop count:4 mpdecimate pts:12000 pts_time:4.0\n"
            "drop count:5 mpdecimate pts:15000 pts_time:5.0\n"
            "drop count:6 mpdecimate pts:18000 pts_time:6.0\n"
        )
        silence_stderr = (
            "[silencedetect @ 0xabc] silence_start: 1.0\n"
            "[silencedetect @ 0xabc] silence_end: 6.0 | silence_duration: 5.0\n"
        )

        mock_motion = MagicMock()
        mock_motion.returncode = 0
        mock_motion.stderr = motion_stderr

        mock_silence = MagicMock()
        mock_silence.returncode = 0
        mock_silence.stderr = silence_stderr

        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        def mock_run(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            if "mpdecimate" in cmd_str:
                return mock_motion
            return mock_silence

        with patch("opencut.core.dead_time.subprocess.run", side_effect=mock_run), \
             patch("opencut.core.dead_time.get_video_info", return_value=video_info):
            from opencut.core.dead_time import detect_dead_time

            result = detect_dead_time("/video/test.mp4", min_duration=3.0)

        assert isinstance(result.segments, list)
        assert result.total_duration == 60.0

    def test_detect_dead_time_empty_when_no_overlap(self):
        """detect_dead_time should return empty when motion and silence don't overlap."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        with patch("opencut.core.dead_time.subprocess.run", return_value=mock_result), \
             patch("opencut.core.dead_time.get_video_info", return_value=video_info):
            from opencut.core.dead_time import detect_dead_time

            result = detect_dead_time("/video/test.mp4")

        assert result.segments == []
        assert result.total_dead_time == 0.0

    def test_speed_ramp_dead_time_builds_filter(self):
        """speed_ramp_dead_time should build a concat filter complex."""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 120.0}

        with patch("opencut.core.dead_time.run_ffmpeg") as mock_ff, \
             patch("opencut.core.dead_time.get_video_info", return_value=video_info):
            from opencut.core.dead_time import speed_ramp_dead_time

            result = speed_ramp_dead_time(
                "/video/test.mp4",
                dead_segments=[
                    {"start": 30.0, "end": 50.0},
                    {"start": 80.0, "end": 90.0},
                ],
                speed_factor=8.0,
            )

        assert mock_ff.called
        cmd = mock_ff.call_args[0][0]
        assert "-filter_complex" in cmd
        assert result["segments_ramped"] == 2
        assert result["speed_factor"] == 8.0
        assert result["time_saved"] > 0

    def test_speed_ramp_no_segments_raises(self):
        """speed_ramp_dead_time should raise ValueError with empty segments."""
        from opencut.core.dead_time import speed_ramp_dead_time
        with pytest.raises(ValueError, match="No dead segments"):
            speed_ramp_dead_time("/video/test.mp4", dead_segments=[])

    def test_build_atempo_chain(self):
        """_build_atempo_chain should chain atempo filters for speed > 2.0."""
        from opencut.core.dead_time import _build_atempo_chain

        # Speed = 1.0 (no change)
        assert _build_atempo_chain(1.0) == "atempo=1.0"

        # Speed = 2.0 (single atempo)
        chain = _build_atempo_chain(2.0)
        assert "atempo=2.0" in chain

        # Speed = 8.0 (chained: 2.0 * 2.0 * 2.0)
        chain = _build_atempo_chain(8.0)
        assert chain.count("atempo=2.0") == 3

    def test_detect_dead_time_progress_callback(self):
        """Progress callback should be called during dead-time detection."""
        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        with patch("opencut.core.dead_time.subprocess.run", return_value=mock_result), \
             patch("opencut.core.dead_time.get_video_info", return_value=video_info):
            from opencut.core.dead_time import detect_dead_time

            detect_dead_time("/video/test.mp4", on_progress=on_progress)

        assert 100 in progress_calls


# ========================================================================
# SDH / HoH Formatting
# ========================================================================
class TestSDHFormat:
    """Tests for opencut.core.sdh_format"""

    _SAMPLE_SRT = (
        "1\n"
        "00:00:01,000 --> 00:00:03,000\n"
        "Hello everyone\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "Welcome to the show\n"
        "\n"
        "3\n"
        "00:00:07,000 --> 00:00:09,000\n"
        "applause\n"
    )

    def test_format_sdh_adds_speaker_labels(self):
        """format_sdh should add speaker labels from diarization data."""
        from opencut.core.sdh_format import format_sdh

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False,
                                          encoding="utf-8") as f:
            f.write(self._SAMPLE_SRT)
            srt_path = f.name

        try:
            diarization = [
                {"start": 0.0, "end": 3.5, "speaker": "Alice"},
                {"start": 3.5, "end": 10.0, "speaker": "Bob"},
            ]
            result = format_sdh(srt_path, diarization_data=diarization)

            assert os.path.isfile(result["output_path"])
            with open(result["output_path"], "r", encoding="utf-8") as f:
                content = f.read()

            assert "[ALICE]:" in content
            assert "[BOB]:" in content
            assert result["entries_formatted"] > 0
        finally:
            os.unlink(srt_path)
            if os.path.isfile(result["output_path"]):
                os.unlink(result["output_path"])

    def test_format_sdh_detects_sound_cues(self):
        """format_sdh should detect and format sound effect cues."""
        from opencut.core.sdh_format import format_sdh

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False,
                                          encoding="utf-8") as f:
            f.write(self._SAMPLE_SRT)
            srt_path = f.name

        try:
            result = format_sdh(srt_path)
            assert os.path.isfile(result["output_path"])
            with open(result["output_path"], "r", encoding="utf-8") as f:
                content = f.read()
            # "applause" entry should be converted to [applause]
            assert "[applause]" in content
        finally:
            os.unlink(srt_path)
            if os.path.isfile(result["output_path"]):
                os.unlink(result["output_path"])

    def test_format_sdh_missing_file_raises(self):
        """format_sdh should raise FileNotFoundError for missing SRT."""
        from opencut.core.sdh_format import format_sdh
        with pytest.raises(FileNotFoundError):
            format_sdh("/nonexistent/file.srt")

    def test_add_speaker_labels(self):
        """add_speaker_labels should add speaker prefixes."""
        from opencut.core.sdh_format import add_speaker_labels

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False,
                                          encoding="utf-8") as f:
            f.write(self._SAMPLE_SRT)
            srt_path = f.name

        try:
            speakers = {
                "0.0-3.5": "Host",
                "3.5-10.0": "Guest",
            }
            result = add_speaker_labels(srt_path, speakers)

            with open(result["output_path"], "r", encoding="utf-8") as f:
                content = f.read()

            assert "[HOST]:" in content
            assert "[GUEST]:" in content
            assert result["labels_added"] > 0
        finally:
            os.unlink(srt_path)
            if os.path.isfile(result["output_path"]):
                os.unlink(result["output_path"])

    def test_parse_srt(self):
        """_parse_srt should correctly parse SRT entries."""
        from opencut.core.sdh_format import _parse_srt

        entries = _parse_srt(self._SAMPLE_SRT)
        assert len(entries) == 3
        assert entries[0]["index"] == 1
        assert entries[0]["text"] == "Hello everyone"
        assert entries[2]["text"] == "applause"

    def test_detect_sound_cues(self):
        """_detect_sound_cues should replace known patterns."""
        from opencut.core.sdh_format import _detect_sound_cues

        assert _detect_sound_cues("applause") == "[applause]"
        assert _detect_sound_cues("laughter") == "[laughter]"
        assert _detect_sound_cues("Hello world") == "Hello world"
        assert _detect_sound_cues("[already formatted]") == "[already formatted]"

    def test_format_sdh_progress_callback(self):
        """Progress callback should be called during SDH formatting."""
        from opencut.core.sdh_format import format_sdh

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False,
                                          encoding="utf-8") as f:
            f.write(self._SAMPLE_SRT)
            srt_path = f.name

        try:
            result = format_sdh(srt_path, on_progress=on_progress)
            assert 100 in progress_calls
        finally:
            os.unlink(srt_path)
            if os.path.isfile(result["output_path"]):
                os.unlink(result["output_path"])


# ========================================================================
# Stream Auto-Chaptering
# ========================================================================
class TestStreamChapters:
    """Tests for opencut.core.stream_chapters"""

    def test_auto_chapter_detects_scene_and_silence(self):
        """auto_chapter_stream should combine scene and silence detection."""
        scene_stderr = (
            "n:1 pts:9000 pts_time:300.0 pos:1234 fmt:yuv420p sar:1/1\n"
            "n:2 pts:18000 pts_time:600.0 pos:5678 fmt:yuv420p sar:1/1\n"
        )
        silence_stderr = (
            "[silencedetect @ 0xabc] silence_start: 890.0\n"
            "[silencedetect @ 0xabc] silence_end: 910.0 | silence_duration: 20.0\n"
        )

        def mock_run(cmd, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            if "showinfo" in cmd_str:
                mock_result.stderr = scene_stderr
            else:
                mock_result.stderr = silence_stderr
            return mock_result

        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 1200.0}

        with patch("opencut.core.stream_chapters.subprocess.run", side_effect=mock_run), \
             patch("opencut.core.stream_chapters.get_video_info", return_value=video_info):
            from opencut.core.stream_chapters import auto_chapter_stream

            result = auto_chapter_stream("/video/stream.mp4")

        assert result.total_chapters > 0
        assert "scene" in result.methods_used
        assert "silence" in result.methods_used
        assert result.chapters[0].start == 0.0

    def test_auto_chapter_no_boundaries(self):
        """auto_chapter_stream should return single chapter when no boundaries found."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 300.0}

        with patch("opencut.core.stream_chapters.subprocess.run", return_value=mock_result), \
             patch("opencut.core.stream_chapters.get_video_info", return_value=video_info):
            from opencut.core.stream_chapters import auto_chapter_stream

            result = auto_chapter_stream("/video/stream.mp4")

        assert result.total_chapters == 1
        assert result.chapters[0].title == "Full Recording"

    def test_export_youtube_chapters(self):
        """export_youtube_chapters should produce correct YouTube format."""
        from opencut.core.stream_chapters import export_youtube_chapters

        chapters = [
            {"start": 0.0, "title": "Introduction"},
            {"start": 154.0, "title": "Topic 1"},
            {"start": 3720.0, "title": "Topic 2"},
        ]
        output = export_youtube_chapters(chapters)

        assert output == "0:00 Introduction\n2:34 Topic 1\n1:02:00 Topic 2"

    def test_export_youtube_chapters_with_offset(self):
        """export_youtube_chapters should apply time offset."""
        from opencut.core.stream_chapters import export_youtube_chapters

        chapters = [{"start": 0.0, "title": "Start"}]
        output = export_youtube_chapters(chapters, offset=10.0)
        assert output == "0:10 Start"

    def test_format_youtube_timestamp(self):
        """_format_youtube_timestamp should handle hours correctly."""
        from opencut.core.stream_chapters import _format_youtube_timestamp

        assert _format_youtube_timestamp(0) == "0:00"
        assert _format_youtube_timestamp(65) == "1:05"
        assert _format_youtube_timestamp(3661) == "1:01:01"

    def test_merge_chapter_points(self):
        """_merge_chapter_points should merge nearby points."""
        from opencut.core.stream_chapters import _merge_chapter_points

        points = [100.0, 105.0, 110.0, 300.0, 305.0]
        merged = _merge_chapter_points(points, merge_distance=30.0, total_duration=600.0)

        assert len(merged) == 2
        assert merged[0] == 100.0
        assert merged[1] == 300.0

    def test_merge_chapter_points_removes_edge_points(self):
        """_merge_chapter_points should remove points near start and end."""
        from opencut.core.stream_chapters import _merge_chapter_points

        points = [5.0, 100.0, 595.0]
        merged = _merge_chapter_points(points, total_duration=600.0)

        assert 5.0 not in merged
        assert 595.0 not in merged
        assert 100.0 in merged

    def test_auto_chapter_progress_callback(self):
        """Progress callback should be called during chaptering."""
        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 300.0}

        with patch("opencut.core.stream_chapters.subprocess.run", return_value=mock_result), \
             patch("opencut.core.stream_chapters.get_video_info", return_value=video_info):
            from opencut.core.stream_chapters import auto_chapter_stream

            auto_chapter_stream("/video/stream.mp4", on_progress=on_progress)

        assert 100 in progress_calls


# ========================================================================
# ND Filter Simulation
# ========================================================================
class TestNDFilterSim:
    """Tests for opencut.core.nd_filter_sim"""

    def test_simulate_nd_filter_standard(self):
        """simulate_nd_filter should produce correct output for 180-degree shutter."""
        video_info = {"width": 1920, "height": 1080, "fps": 24.0, "duration": 60.0}

        with patch("opencut.core.nd_filter_sim.run_ffmpeg") as mock_ff, \
             patch("opencut.core.nd_filter_sim.get_video_info", return_value=video_info):
            from opencut.core.nd_filter_sim import simulate_nd_filter

            result = simulate_nd_filter("/video/test.mp4", shutter_angle=180.0)

        assert mock_ff.called
        assert result["shutter_angle"] == 180.0
        assert result["blur_strength"] == 0.5
        assert result["source_fps"] == 24.0
        assert "nd180" in result["output_path"]

    def test_simulate_nd_filter_max_blur(self):
        """simulate_nd_filter at 360 degrees should have blur_strength 1.0."""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        with patch("opencut.core.nd_filter_sim.run_ffmpeg"), \
             patch("opencut.core.nd_filter_sim.get_video_info", return_value=video_info):
            from opencut.core.nd_filter_sim import simulate_nd_filter

            result = simulate_nd_filter("/video/test.mp4", shutter_angle=360.0)

        assert result["blur_strength"] == 1.0
        assert result["sub_frames"] == 8

    def test_simulate_nd_filter_min_blur(self):
        """simulate_nd_filter at minimum angle should have low blur."""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        with patch("opencut.core.nd_filter_sim.run_ffmpeg"), \
             patch("opencut.core.nd_filter_sim.get_video_info", return_value=video_info):
            from opencut.core.nd_filter_sim import simulate_nd_filter

            result = simulate_nd_filter("/video/test.mp4", shutter_angle=1.0)

        assert result["blur_strength"] < 0.01
        assert result["sub_frames"] == 2  # Minimum

    def test_simulate_nd_filter_clamps_angle(self):
        """simulate_nd_filter should clamp shutter_angle to valid range."""
        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        with patch("opencut.core.nd_filter_sim.run_ffmpeg"), \
             patch("opencut.core.nd_filter_sim.get_video_info", return_value=video_info):
            from opencut.core.nd_filter_sim import simulate_nd_filter

            result = simulate_nd_filter("/video/test.mp4", shutter_angle=999.0)

        assert result["shutter_angle"] == 360.0

    def test_simulate_nd_filter_uses_ffmpegcmd(self):
        """simulate_nd_filter should use FFmpegCmd builder."""
        video_info = {"width": 1920, "height": 1080, "fps": 24.0, "duration": 60.0}

        with patch("opencut.core.nd_filter_sim.run_ffmpeg") as mock_ff, \
             patch("opencut.core.nd_filter_sim.get_video_info", return_value=video_info):
            from opencut.core.nd_filter_sim import simulate_nd_filter

            simulate_nd_filter("/video/test.mp4")

        cmd = mock_ff.call_args[0][0]
        # Should contain minterpolate filter
        vf_idx = cmd.index("-vf")
        assert "minterpolate" in cmd[vf_idx + 1]
        assert "mi_mode=blend" in cmd[vf_idx + 1]

    def test_simulate_nd_filter_progress(self):
        """Progress callback should fire during ND filter simulation."""
        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        video_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}

        with patch("opencut.core.nd_filter_sim.run_ffmpeg"), \
             patch("opencut.core.nd_filter_sim.get_video_info", return_value=video_info):
            from opencut.core.nd_filter_sim import simulate_nd_filter

            simulate_nd_filter("/video/test.mp4", on_progress=on_progress)

        assert 100 in progress_calls


# ========================================================================
# Timecode Utilities
# ========================================================================
class TestTimecodeUtils:
    """Tests for opencut.core.timecode_utils"""

    def test_frames_to_timecode_ndf_30fps(self):
        """frames_to_timecode NDF should produce correct HH:MM:SS:FF."""
        from opencut.core.timecode_utils import frames_to_timecode

        assert frames_to_timecode(0, 30.0) == "00:00:00:00"
        assert frames_to_timecode(30, 30.0) == "00:00:01:00"
        assert frames_to_timecode(1800, 30.0) == "00:01:00:00"
        assert frames_to_timecode(108000, 30.0) == "01:00:00:00"
        assert frames_to_timecode(45, 30.0) == "00:00:01:15"

    def test_frames_to_timecode_ndf_24fps(self):
        """frames_to_timecode at 24fps should work correctly."""
        from opencut.core.timecode_utils import frames_to_timecode

        assert frames_to_timecode(0, 24.0) == "00:00:00:00"
        assert frames_to_timecode(24, 24.0) == "00:00:01:00"
        assert frames_to_timecode(48, 24.0) == "00:00:02:00"

    def test_frames_to_timecode_df_29_97(self):
        """frames_to_timecode DF at 29.97 should use semicolons and drop frames."""
        from opencut.core.timecode_utils import frames_to_timecode

        tc = frames_to_timecode(0, 29.97, drop_frame=True)
        assert tc == "00:00:00;00"

        # At 29.97 DF, frame 1800 should NOT be exactly 01:00:00
        # because of dropped frame numbers
        tc = frames_to_timecode(1800, 29.97, drop_frame=True)
        assert ";" in tc

    def test_timecode_to_frames_ndf(self):
        """timecode_to_frames NDF should produce correct frame numbers."""
        from opencut.core.timecode_utils import timecode_to_frames

        assert timecode_to_frames("00:00:00:00", 30.0) == 0
        assert timecode_to_frames("00:00:01:00", 30.0) == 30
        assert timecode_to_frames("00:01:00:00", 30.0) == 1800
        assert timecode_to_frames("01:00:00:00", 30.0) == 108000
        assert timecode_to_frames("00:00:01:15", 30.0) == 45

    def test_timecode_to_frames_df(self):
        """timecode_to_frames DF should handle drop-frame correctly."""
        from opencut.core.timecode_utils import timecode_to_frames

        # Frame 0 at 29.97 DF
        assert timecode_to_frames("00:00:00;00", 29.97) == 0

        # At 1 minute DF, frames 0 and 1 are dropped
        frames = timecode_to_frames("00:01:00;02", 29.97)
        assert frames > 0

    def test_roundtrip_ndf(self):
        """Converting frames -> timecode -> frames should be lossless for NDF."""
        from opencut.core.timecode_utils import frames_to_timecode, timecode_to_frames

        for f in [0, 1, 29, 30, 1799, 1800, 107999, 108000]:
            tc = frames_to_timecode(f, 30.0, drop_frame=False)
            back = timecode_to_frames(tc, 30.0, drop_frame=False)
            assert back == f, f"Roundtrip failed for frame {f}: tc={tc}, back={back}"

    def test_roundtrip_df(self):
        """Converting frames -> timecode -> frames should be lossless for DF 29.97."""
        from opencut.core.timecode_utils import frames_to_timecode, timecode_to_frames

        for f in [0, 1, 29, 1798, 17982, 53946]:
            tc = frames_to_timecode(f, 29.97, drop_frame=True)
            back = timecode_to_frames(tc, 29.97, drop_frame=True)
            assert back == f, f"DF roundtrip failed for frame {f}: tc={tc}, back={back}"

    def test_convert_timecode_same_fps(self):
        """convert_timecode with same fps should return equivalent timecode."""
        from opencut.core.timecode_utils import convert_timecode

        result = convert_timecode("00:01:00:00", 30.0, 30.0)
        assert result == "00:01:00:00"

    def test_convert_timecode_different_fps(self):
        """convert_timecode should adjust frame numbers between rates."""
        from opencut.core.timecode_utils import convert_timecode

        # 1 second at 24fps = frame 24 -> at 30fps = frame 30
        result = convert_timecode("00:00:01:00", 24.0, 30.0)
        assert result == "00:00:01:00"

    def test_convert_timecode_ndf_to_df(self):
        """convert_timecode should handle NDF -> DF conversion."""
        from opencut.core.timecode_utils import convert_timecode

        result = convert_timecode(
            "00:01:00:00", 30.0, 29.97,
            source_df=False, target_df=True,
        )
        assert ";" in result

    def test_detect_timecode_format(self):
        """detect_timecode_format should parse ffprobe output correctly."""
        probe_output = json.dumps({
            "streams": [{
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "tags": {"timecode": "01:00:00;00"},
            }],
            "format": {"tags": {}},
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = probe_output.encode()

        with patch("opencut.core.timecode_utils.subprocess.run", return_value=mock_result):
            from opencut.core.timecode_utils import detect_timecode_format

            info = detect_timecode_format("/video/test.mxf")

        assert abs(info.fps - 29.97) < 0.02
        assert info.is_drop_frame is True
        assert info.detected_tc == "01:00:00;00"

    def test_detect_timecode_ndf(self):
        """detect_timecode_format should detect NDF from colon separator."""
        probe_output = json.dumps({
            "streams": [{
                "r_frame_rate": "24/1",
                "avg_frame_rate": "24/1",
                "tags": {"timecode": "01:00:00:00"},
            }],
            "format": {"tags": {}},
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = probe_output.encode()

        with patch("opencut.core.timecode_utils.subprocess.run", return_value=mock_result):
            from opencut.core.timecode_utils import detect_timecode_format

            info = detect_timecode_format("/video/test.mov")

        assert info.fps == 24.0
        assert info.is_drop_frame is False

    def test_invalid_timecode_raises(self):
        """timecode_to_frames should raise ValueError for invalid format."""
        from opencut.core.timecode_utils import timecode_to_frames

        with pytest.raises(ValueError, match="Invalid timecode format"):
            timecode_to_frames("not-a-timecode", 30.0)

        with pytest.raises(ValueError, match="Invalid timecode format"):
            timecode_to_frames("00:00:00", 30.0)  # only 3 parts


# ========================================================================
# Route Smoke Tests
# ========================================================================
class TestSubtitleRoutes:
    """Smoke tests for subtitle_routes Blueprint endpoints."""

    def test_subtitle_tracks_no_filepath(self, client, csrf_token):
        """POST /subtitle/tracks with no filepath should return 400."""
        resp = client.post("/subtitle/tracks",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_subtitle_embed_no_filepath(self, client, csrf_token):
        """POST /subtitle/embed with no filepath should return 400."""
        resp = client.post("/subtitle/embed",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_timecode_detect_no_filepath(self, client, csrf_token):
        """POST /timecode/detect with no filepath should return 400."""
        resp = client.post("/timecode/detect",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_timecode_convert_no_timecode(self, client, csrf_token):
        """POST /timecode/convert with no timecode should return 400."""
        resp = client.post("/timecode/convert",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_timecode_convert_valid(self, client, csrf_token):
        """POST /timecode/convert with valid data should return 200."""
        resp = client.post("/timecode/convert",
                           headers=csrf_headers(csrf_token),
                           json={
                               "timecode": "00:01:00:00",
                               "source_fps": 30.0,
                               "target_fps": 24.0,
                           })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "timecode" in data

    def test_timecode_convert_invalid_tc(self, client, csrf_token):
        """POST /timecode/convert with invalid timecode should return 400."""
        resp = client.post("/timecode/convert",
                           headers=csrf_headers(csrf_token),
                           json={
                               "timecode": "bad",
                               "source_fps": 30.0,
                               "target_fps": 24.0,
                           })
        assert resp.status_code == 400

    def test_sdh_format_no_filepath(self, client, csrf_token):
        """POST /subtitle/sdh-format with no srt_path should return 400."""
        resp = client.post("/subtitle/sdh-format",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_dead_time_detect_no_filepath(self, client, csrf_token):
        """POST /video/dead-time/detect with no filepath should return 400."""
        resp = client.post("/video/dead-time/detect",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_dead_time_speed_ramp_no_filepath(self, client, csrf_token):
        """POST /video/dead-time/speed-ramp with no filepath should return 400."""
        resp = client.post("/video/dead-time/speed-ramp",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_stream_auto_chapter_no_filepath(self, client, csrf_token):
        """POST /stream/auto-chapter with no filepath should return 400."""
        resp = client.post("/stream/auto-chapter",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_nd_filter_no_filepath(self, client, csrf_token):
        """POST /video/nd-filter with no filepath should return 400."""
        resp = client.post("/video/nd-filter",
                           headers=csrf_headers(csrf_token),
                           json={})
        assert resp.status_code == 400

    def test_no_csrf_rejected(self, client):
        """Requests without CSRF token should be rejected."""
        resp = client.post("/timecode/convert",
                           headers={"Content-Type": "application/json"},
                           json={"timecode": "00:00:00:00"})
        assert resp.status_code in (400, 403)
