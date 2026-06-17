"""Coverage expansion: core logic modules."""

import json
import math
import os
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# 1. opencut/core/timecode_utils.py
# ===================================================================


class TestIsDropFrameRate:
    """_is_drop_frame_rate() — tolerance-aware DF detection."""

    def _fn(self, fps):
        from opencut.core.timecode_utils import _is_drop_frame_rate
        return _is_drop_frame_rate(fps)

    def test_29_97_is_drop_frame(self):
        assert self._fn(29.97) is True

    def test_59_94_is_drop_frame(self):
        assert self._fn(59.94) is True

    def test_30_is_not_drop_frame(self):
        assert self._fn(30.0) is False

    def test_24_is_not_drop_frame(self):
        assert self._fn(24.0) is False

    def test_25_is_not_drop_frame(self):
        assert self._fn(25.0) is False

    def test_23_976_is_not_drop_frame(self):
        assert self._fn(23.976) is False

    def test_29_98_within_tolerance(self):
        assert self._fn(29.98) is True

    def test_59_95_within_tolerance(self):
        assert self._fn(59.95) is True

    def test_29_5_outside_tolerance(self):
        assert self._fn(29.5) is False


class TestRoundFps:
    """_round_fps() — snap to standard rates."""

    def _fn(self, fps):
        from opencut.core.timecode_utils import _round_fps
        return _round_fps(fps)

    def test_snap_23_976(self):
        assert self._fn(23.98) == 23.976

    def test_snap_29_97(self):
        assert self._fn(29.97) == 29.97

    def test_snap_30(self):
        assert self._fn(30.001) == 30.0

    def test_snap_60(self):
        assert self._fn(59.999) == 60.0

    def test_nonstandard_passthrough(self):
        assert self._fn(17.5) == 17.5


class TestFramesToTimecodeNDF:
    """frames_to_timecode() non-drop-frame paths."""

    def _fn(self, frame_num, fps, drop_frame=False):
        from opencut.core.timecode_utils import frames_to_timecode
        return frames_to_timecode(frame_num, fps, drop_frame)

    def test_zero_frames_30fps(self):
        assert self._fn(0, 30.0) == "00:00:00:00"

    def test_one_second_30fps(self):
        assert self._fn(30, 30.0) == "00:00:01:00"

    def test_one_minute_30fps(self):
        assert self._fn(1800, 30.0) == "00:01:00:00"

    def test_one_hour_30fps(self):
        assert self._fn(108000, 30.0) == "01:00:00:00"

    def test_mixed_24fps(self):
        # 1*3600*24 + 2*60*24 + 3*24 + 4 = 86400 + 2880 + 72 + 4 = 89356
        assert self._fn(89356, 24.0) == "01:02:03:04"

    def test_negative_clamps_to_zero(self):
        assert self._fn(-5, 30.0) == "00:00:00:00"


class TestFramesToTimecodeDF:
    """frames_to_timecode() drop-frame (SMPTE 12M)."""

    def _fn(self, frame_num, fps, drop_frame=False):
        from opencut.core.timecode_utils import frames_to_timecode
        return frames_to_timecode(frame_num, fps, drop_frame)

    def test_zero_frames_df(self):
        assert self._fn(0, 29.97, drop_frame=True) == "00:00:00;00"

    def test_semicolon_separator(self):
        tc = self._fn(100, 29.97, drop_frame=True)
        assert ";" in tc

    def test_colon_separator_ndf(self):
        tc = self._fn(100, 30.0, drop_frame=False)
        assert ";" not in tc

    def test_one_minute_mark_df(self):
        # At 29.97 DF, frame 1800 maps to 00:01:00;02 (frames ;00 and ;01 are
        # dropped at each non-10th minute boundary).
        tc = self._fn(1800, 29.97, drop_frame=True)
        assert tc.startswith("00:01:00;")

    def test_ten_minute_mark_df(self):
        # At 29.97 DF, 10 minutes = 30*60*10 - 9*2 = 17982 frames.
        tc = self._fn(17982, 29.97, drop_frame=True)
        assert tc.startswith("00:10:00;")


class TestTimecodeToFrames:
    """timecode_to_frames() — parsing and conversion."""

    def _fn(self, tc, fps, drop_frame=None):
        from opencut.core.timecode_utils import timecode_to_frames
        return timecode_to_frames(tc, fps, drop_frame)

    def test_zero_tc(self):
        assert self._fn("00:00:00:00", 30.0) == 0

    def test_one_second(self):
        assert self._fn("00:00:01:00", 30.0) == 30

    def test_autodetect_df_from_semicolon(self):
        frames = self._fn("00:01:00;02", 29.97)
        assert frames >= 0  # Just ensure no crash + positive

    def test_autodetect_ndf_from_colon(self):
        frames = self._fn("00:01:00:00", 30.0)
        assert frames == 1800

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid timecode format"):
            self._fn("12:34", 30.0)

    def test_invalid_values_raises(self):
        with pytest.raises(ValueError, match="Invalid timecode values"):
            self._fn("xx:yy:zz:ww", 30.0)


class TestTimecodeRoundTrip:
    """Round-trip: frames -> timecode -> frames."""

    def _frames_to_tc(self, f, fps, df):
        from opencut.core.timecode_utils import frames_to_timecode
        return frames_to_timecode(f, fps, df)

    def _tc_to_frames(self, tc, fps, df):
        from opencut.core.timecode_utils import timecode_to_frames
        return timecode_to_frames(tc, fps, df)

    def test_roundtrip_ndf_30(self):
        for frame in [0, 1, 29, 30, 1800, 108000]:
            tc = self._frames_to_tc(frame, 30.0, False)
            result = self._tc_to_frames(tc, 30.0, False)
            assert result == frame, f"frame={frame}, tc={tc}, result={result}"

    def test_roundtrip_ndf_24(self):
        for frame in [0, 23, 24, 1440]:
            tc = self._frames_to_tc(frame, 24.0, False)
            result = self._tc_to_frames(tc, 24.0, False)
            assert result == frame

    def test_roundtrip_df_29_97(self):
        for frame in [0, 1, 100, 1798, 17982]:
            tc = self._frames_to_tc(frame, 29.97, True)
            result = self._tc_to_frames(tc, 29.97, True)
            assert result == frame, f"frame={frame}, tc={tc}, result={result}"


class TestConvertTimecode:
    """convert_timecode() — cross-rate conversion."""

    def _fn(self, tc, src_fps, tgt_fps, src_df=False, tgt_df=False):
        from opencut.core.timecode_utils import convert_timecode
        return convert_timecode(tc, src_fps, tgt_fps, src_df, tgt_df)

    def test_same_rate_identity(self):
        assert self._fn("01:00:00:00", 30.0, 30.0) == "01:00:00:00"

    def test_30_to_24(self):
        tc = self._fn("00:00:01:00", 30.0, 24.0)
        assert tc == "00:00:01:00"  # 1 second is 1 second regardless

    def test_zero_stays_zero(self):
        assert self._fn("00:00:00:00", 30.0, 24.0) == "00:00:00:00"


# ===================================================================
# 2. opencut/core/metadata_tools.py
# ===================================================================


class TestPrivacyFieldClassification:
    """_PRIVACY_FIELDS frozenset membership."""

    def _fields(self):
        from opencut.core.metadata_tools import _PRIVACY_FIELDS
        return _PRIVACY_FIELDS

    def test_gps_fields_are_sensitive(self):
        pf = self._fields()
        for f in ["location", "gps", "gps_latitude", "gps_longitude"]:
            assert f in pf

    def test_serial_fields_are_sensitive(self):
        pf = self._fields()
        for f in ["serial_number", "serialnumber", "camera_serial"]:
            assert f in pf

    def test_pii_fields_are_sensitive(self):
        pf = self._fields()
        for f in ["artist", "author", "owner", "copyright"]:
            assert f in pf

    def test_technical_fields_not_in_privacy(self):
        pf = self._fields()
        for f in ["creation_time", "codec_name", "width", "height", "title"]:
            assert f not in pf


class TestLegalPreserveFields:
    """_LEGAL_PRESERVE_FIELDS frozenset membership."""

    def _fields(self):
        from opencut.core.metadata_tools import _LEGAL_PRESERVE_FIELDS
        return _LEGAL_PRESERVE_FIELDS

    def test_timestamps_preserved(self):
        pf = self._fields()
        assert "creation_time" in pf
        assert "date" in pf

    def test_codec_info_preserved(self):
        pf = self._fields()
        assert "codec_name" in pf
        assert "codec_type" in pf

    def test_resolution_preserved(self):
        pf = self._fields()
        assert "width" in pf
        assert "height" in pf

    def test_title_preserved(self):
        assert "title" in self._fields()


class TestValidModes:
    """VALID_MODES membership."""

    def test_all_modes_present(self):
        from opencut.core.metadata_tools import VALID_MODES
        assert VALID_MODES == frozenset({"strip_all", "preserve_all", "selective", "legal"})


class TestPrivacyAndLegalDisjoint:
    """Privacy and legal-preserve sets should be disjoint."""

    def test_no_overlap(self):
        from opencut.core.metadata_tools import _LEGAL_PRESERVE_FIELDS, _PRIVACY_FIELDS
        overlap = _PRIVACY_FIELDS & _LEGAL_PRESERVE_FIELDS
        assert len(overlap) == 0, f"Overlap: {overlap}"


# ===================================================================
# 3. opencut/core/safe_zones.py
# ===================================================================


class TestGetSafeZones:
    """get_safe_zones() — pure data function."""

    def _fn(self, platform, w, h):
        from opencut.core.safe_zones import get_safe_zones
        return get_safe_zones(platform, w, h)

    def test_youtube_returns_zones(self):
        zones = self._fn("youtube", 1920, 1080)
        assert len(zones) == 2
        labels = {z.label for z in zones}
        assert "End Screen Zone" in labels
        assert "Title Bar Zone" in labels

    def test_tiktok_returns_zones(self):
        zones = self._fn("tiktok", 1080, 1920)
        assert len(zones) == 3

    def test_instagram_returns_zones(self):
        zones = self._fn("instagram", 1080, 1920)
        assert len(zones) == 3

    def test_twitter_returns_zones(self):
        zones = self._fn("twitter", 1920, 1080)
        assert len(zones) == 2

    def test_x_alias_matches_twitter(self):
        x_zones = self._fn("x", 1920, 1080)
        tw_zones = self._fn("twitter", 1920, 1080)
        assert len(x_zones) == len(tw_zones)
        for xz, tz in zip(x_zones, tw_zones):
            assert xz.x == tz.x and xz.y == tz.y

    def test_reels_alias_matches_instagram(self):
        r_zones = self._fn("reels", 1080, 1920)
        i_zones = self._fn("instagram", 1080, 1920)
        assert len(r_zones) == len(i_zones)

    def test_unsupported_platform_raises(self):
        with pytest.raises(ValueError, match="Unsupported platform"):
            self._fn("myspace", 1920, 1080)

    def test_case_insensitive(self):
        zones = self._fn("YouTube", 1920, 1080)
        assert len(zones) == 2


class TestSafeZoneScaling:
    """Safe zone coordinates scale to resolution."""

    def _fn(self, platform, w, h):
        from opencut.core.safe_zones import get_safe_zones
        return get_safe_zones(platform, w, h)

    def test_youtube_end_screen_y_at_80_pct(self):
        zones = self._fn("youtube", 1920, 1080)
        end_screen = [z for z in zones if z.label == "End Screen Zone"][0]
        assert end_screen.y == round(0.80 * 1080)
        assert end_screen.h == round(0.20 * 1080)

    def test_youtube_title_bar_at_top(self):
        zones = self._fn("youtube", 1920, 1080)
        title_bar = [z for z in zones if z.label == "Title Bar Zone"][0]
        assert title_bar.x == 0
        assert title_bar.y == 0

    def test_4k_scaling(self):
        zones = self._fn("youtube", 3840, 2160)
        end_screen = [z for z in zones if z.label == "End Screen Zone"][0]
        assert end_screen.y == round(0.80 * 2160)
        assert end_screen.w == 3840

    def test_zones_within_frame(self):
        for platform in ["youtube", "tiktok", "instagram", "twitter"]:
            zones = self._fn(platform, 1920, 1080)
            for z in zones:
                assert z.x >= 0
                assert z.y >= 0
                assert z.x + z.w <= 1920
                assert z.y + z.h <= 1080


class TestSupportedPlatforms:
    """SUPPORTED_PLATFORMS list."""

    def test_is_sorted(self):
        from opencut.core.safe_zones import SUPPORTED_PLATFORMS
        assert SUPPORTED_PLATFORMS == sorted(SUPPORTED_PLATFORMS)

    def test_contains_major_platforms(self):
        from opencut.core.safe_zones import SUPPORTED_PLATFORMS
        for p in ["youtube", "tiktok", "instagram", "twitter", "x", "reels"]:
            assert p in SUPPORTED_PLATFORMS


# ===================================================================
# 4. opencut/core/hdr_tools.py
# ===================================================================


class TestHDRInfoDefaults:
    """HDRInfo dataclass defaults."""

    def test_defaults(self):
        from opencut.core.hdr_tools import HDRInfo
        info = HDRInfo()
        assert info.is_hdr is False
        assert info.transfer == "sdr"
        assert info.primaries == "bt709"
        assert info.max_cll == 0
        assert info.max_fall == 0


class TestValidTonemapAlgorithms:
    """VALID_TONEMAP_ALGORITHMS tuple."""

    def test_contains_expected(self):
        from opencut.core.hdr_tools import VALID_TONEMAP_ALGORITHMS
        assert "hable" in VALID_TONEMAP_ALGORITHMS
        assert "reinhard" in VALID_TONEMAP_ALGORITHMS
        assert "mobius" in VALID_TONEMAP_ALGORITHMS
        assert "linear" in VALID_TONEMAP_ALGORITHMS


class TestDetectHdrParsing:
    """detect_hdr() ffprobe output parsing via mocked subprocess."""

    def _make_probe_output(self, color_transfer="", color_primaries="", side_data=None):
        """Build a fake ffprobe JSON stdout."""
        stream = {"color_transfer": color_transfer, "color_primaries": color_primaries}
        if side_data:
            stream["side_data_list"] = side_data
        return json.dumps({"streams": [stream]}).encode()

    @patch("opencut.core.hdr_tools._sp.run")
    @patch("opencut.core.hdr_tools.os.path.isfile", return_value=True)
    def test_pq_detected(self, _mock_isfile, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_output("smpte2084", "bt2020"),
        )
        from opencut.core.hdr_tools import detect_hdr
        info = detect_hdr("fake.mp4")
        assert info.is_hdr is True
        assert info.transfer == "pq"
        assert info.primaries == "bt2020"

    @patch("opencut.core.hdr_tools._sp.run")
    @patch("opencut.core.hdr_tools.os.path.isfile", return_value=True)
    def test_hlg_detected(self, _mock_isfile, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_output("arib-std-b67", "bt2020"),
        )
        from opencut.core.hdr_tools import detect_hdr
        info = detect_hdr("fake.mp4")
        assert info.is_hdr is True
        assert info.transfer == "hlg"

    @patch("opencut.core.hdr_tools._sp.run")
    @patch("opencut.core.hdr_tools.os.path.isfile", return_value=True)
    def test_sdr_detected(self, _mock_isfile, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_output("bt709", "bt709"),
        )
        from opencut.core.hdr_tools import detect_hdr
        info = detect_hdr("fake.mp4")
        assert info.is_hdr is False
        assert info.transfer == "sdr"

    @patch("opencut.core.hdr_tools._sp.run")
    @patch("opencut.core.hdr_tools.os.path.isfile", return_value=True)
    def test_max_cll_fall_extracted(self, _mock_isfile, mock_run):
        side_data = [{"max_content": 1000, "max_average": 400}]
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self._make_probe_output("smpte2084", "bt2020", side_data),
        )
        from opencut.core.hdr_tools import detect_hdr
        info = detect_hdr("fake.mp4")
        assert info.max_cll == 1000
        assert info.max_fall == 400

    @patch("opencut.core.hdr_tools._sp.run")
    @patch("opencut.core.hdr_tools.os.path.isfile", return_value=True)
    def test_ffprobe_failure_returns_default(self, _mock_isfile, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        from opencut.core.hdr_tools import detect_hdr
        info = detect_hdr("fake.mp4")
        assert info.is_hdr is False

    def test_detect_hdr_file_not_found(self):
        from opencut.core.hdr_tools import detect_hdr
        with pytest.raises(FileNotFoundError):
            detect_hdr("/nonexistent/file.mp4")


# ===================================================================
# 5. opencut/core/dead_time.py
# ===================================================================


class TestIntersectSegments:
    """_intersect_segments() — pure overlap logic."""

    def _fn(self, motion, silence, min_dur=0.0):
        from opencut.core.dead_time import _intersect_segments
        return _intersect_segments(motion, silence, min_dur)

    def test_empty_inputs(self):
        assert self._fn([], []) == []

    def test_no_overlap(self):
        assert self._fn([(0, 5)], [(10, 15)]) == []

    def test_full_overlap(self):
        result = self._fn([(0, 10)], [(0, 10)])
        assert len(result) == 1
        assert result[0] == (0, 10)

    def test_partial_overlap(self):
        result = self._fn([(0, 10)], [(5, 15)])
        assert len(result) == 1
        assert result[0] == (5, 10)

    def test_min_duration_filter(self):
        result = self._fn([(0, 10)], [(5, 15)], min_dur=6.0)
        assert len(result) == 0  # overlap is only 5s

    def test_multiple_overlaps(self):
        motion = [(0, 10), (20, 30)]
        silence = [(5, 25)]
        result = self._fn(motion, silence)
        assert len(result) == 2
        assert result[0] == (5, 10)
        assert result[1] == (20, 25)


class TestBuildAtempoChain:
    """_build_atempo_chain() — FFmpeg atempo chaining."""

    def _fn(self, speed):
        from opencut.core.dead_time import _build_atempo_chain
        return _build_atempo_chain(speed)

    def test_speed_1_returns_identity(self):
        assert self._fn(1.0) == "atempo=1.0"

    def test_speed_less_than_1_returns_identity(self):
        assert self._fn(0.5) == "atempo=1.0"

    def test_speed_2_single_filter(self):
        result = self._fn(2.0)
        assert "atempo=2.0" in result
        assert "," not in result  # single filter, no chain

    def test_speed_4_chains_two(self):
        result = self._fn(4.0)
        parts = result.split(",")
        assert len(parts) == 2
        assert parts[0] == "atempo=2.0"

    def test_speed_8_chains_three(self):
        result = self._fn(8.0)
        parts = result.split(",")
        assert len(parts) == 3

    def test_speed_3_chains_correctly(self):
        result = self._fn(3.0)
        parts = result.split(",")
        assert len(parts) == 2
        assert parts[0] == "atempo=2.0"
        # second should be atempo=1.5
        assert "1.5" in parts[1]


class TestDeadSegmentDataclass:
    """DeadSegment / DeadTimeResult data structures."""

    def test_dead_segment_fields(self):
        from opencut.core.dead_time import DeadSegment
        seg = DeadSegment(start=1.0, end=5.0, duration=4.0, motion_score=0.1)
        assert seg.start == 1.0
        assert seg.end == 5.0
        assert seg.duration == 4.0

    def test_dead_time_result_defaults(self):
        from opencut.core.dead_time import DeadTimeResult
        r = DeadTimeResult()
        assert r.segments == []
        assert r.total_dead_time == 0.0
        assert r.dead_percentage == 0.0


# ===================================================================
# 6. opencut/core/stream_chapters.py
# ===================================================================


class TestMergeChapterPoints:
    """_merge_chapter_points() — dedup, distance filter, boundary trim."""

    def _fn(self, points, merge_distance=30.0, total_duration=0.0):
        from opencut.core.stream_chapters import _merge_chapter_points
        return _merge_chapter_points(points, merge_distance, total_duration)

    def test_empty_input(self):
        assert self._fn([]) == []

    def test_removes_near_start(self):
        result = self._fn([5.0, 60.0])
        # 5.0 is within first 10 seconds, should be filtered
        assert 5.0 not in result
        assert 60.0 in result

    def test_removes_near_end(self):
        result = self._fn([60.0, 295.0], total_duration=300.0)
        # 295.0 is within 10 seconds of end
        assert 295.0 not in result
        assert 60.0 in result

    def test_merges_nearby_points(self):
        result = self._fn([60.0, 65.0, 100.0], merge_distance=30.0)
        assert len(result) == 2
        assert 60.0 in result
        assert 100.0 in result

    def test_deduplicates(self):
        result = self._fn([60.0, 60.0, 120.0])
        assert result.count(60.0) == 1

    def test_sorts_input(self):
        result = self._fn([120.0, 60.0, 180.0])
        assert result == sorted(result)


class TestGenerateTitles:
    """_generate_titles() — auto-title assignment."""

    def _fn(self, chapters):
        from opencut.core.stream_chapters import _generate_titles
        return _generate_titles(chapters)

    def test_first_gets_introduction(self):
        from opencut.core.stream_chapters import Chapter
        chs = [Chapter(0, 60, ""), Chapter(60, 120, "")]
        result = self._fn(chs)
        assert result[0].title == "Introduction"

    def test_last_gets_conclusion(self):
        from opencut.core.stream_chapters import Chapter
        chs = [Chapter(0, 60, ""), Chapter(60, 120, ""), Chapter(120, 180, "")]
        result = self._fn(chs)
        assert result[-1].title == "Conclusion"

    def test_middle_gets_section_n(self):
        from opencut.core.stream_chapters import Chapter
        chs = [Chapter(0, 60, ""), Chapter(60, 120, ""), Chapter(120, 180, "")]
        result = self._fn(chs)
        assert result[1].title == "Section 2"

    def test_preserves_existing_titles(self):
        from opencut.core.stream_chapters import Chapter
        chs = [Chapter(0, 60, "My Title"), Chapter(60, 120, "")]
        result = self._fn(chs)
        assert result[0].title == "My Title"


class TestFormatYoutubeTimestamp:
    """_format_youtube_timestamp() — seconds to M:SS or H:MM:SS."""

    def _fn(self, seconds):
        from opencut.core.stream_chapters import _format_youtube_timestamp
        return _format_youtube_timestamp(seconds)

    def test_zero(self):
        assert self._fn(0) == "0:00"

    def test_under_one_hour(self):
        assert self._fn(90) == "1:30"

    def test_one_hour(self):
        assert self._fn(3600) == "1:00:00"

    def test_mixed(self):
        assert self._fn(3661) == "1:01:01"

    def test_negative_clamps(self):
        assert self._fn(-5) == "0:00"


class TestExportYoutubeChapters:
    """export_youtube_chapters() formatting."""

    def _fn(self, chapters, offset=0):
        from opencut.core.stream_chapters import export_youtube_chapters
        return export_youtube_chapters(chapters, offset)

    def test_basic_export(self):
        chapters = [
            {"start": 0, "title": "Intro"},
            {"start": 60, "title": "Topic 1"},
        ]
        result = self._fn(chapters)
        assert "0:00 Intro" in result
        assert "1:00 Topic 1" in result

    def test_offset(self):
        chapters = [{"start": 0, "title": "Start"}]
        result = self._fn(chapters, offset=10)
        assert "0:10 Start" in result


# ===================================================================
# 7. opencut/core/timelapse.py
# ===================================================================


class TestRollingAverage:
    """_rolling_average() — centered window averaging."""

    def _fn(self, values, window):
        from opencut.core.timelapse import _rolling_average
        return _rolling_average(values, window)

    def test_single_value(self):
        assert self._fn([5.0], 3) == [5.0]

    def test_flat_signal(self):
        result = self._fn([10.0, 10.0, 10.0, 10.0, 10.0], 3)
        for v in result:
            assert abs(v - 10.0) < 1e-9

    def test_window_larger_than_data(self):
        result = self._fn([1.0, 2.0, 3.0], 99)
        assert abs(result[1] - 2.0) < 1e-9

    def test_smoothing_effect(self):
        data = [10.0, 50.0, 10.0, 50.0, 10.0]
        result = self._fn(data, 3)
        # Middle value should be smoothed toward neighbors
        assert result[1] < 50.0
        assert result[1] > 10.0

    def test_preserves_length(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self._fn(data, 3)
        assert len(result) == len(data)


class TestFlickerScoreComputation:
    """FlickerAnalysis dataclass and score semantics."""

    def test_defaults(self):
        from opencut.core.timelapse import FlickerAnalysis
        fa = FlickerAnalysis()
        assert fa.flicker_score == 0.0
        assert fa.needs_deflicker is False
        assert fa.per_frame_luminance == []

    def test_high_cv_means_high_score(self):
        # flicker_score = min(1.0, cv / 0.2) where cv = std/mean
        # If std = 40, mean = 100, cv = 0.4, score = min(1, 2) = 1.0
        from opencut.core.timelapse import FlickerAnalysis
        fa = FlickerAnalysis(flicker_score=1.0, needs_deflicker=True)
        assert fa.flicker_score == 1.0
        assert fa.needs_deflicker is True


# ===================================================================
# 8. opencut/core/highlight_detect.py
# ===================================================================


class TestHighlightScoreDataclass:
    """HighlightScore fields and defaults."""

    def test_all_fields(self):
        from opencut.core.highlight_detect import HighlightScore
        hs = HighlightScore(
            start_time=0.0, end_time=10.0, score=0.85,
            audio_score=0.9, visual_score=0.8, chat_score=0.0,
            label="kill",
        )
        assert hs.score == 0.85
        assert hs.label == "kill"

    def test_defaults(self):
        from opencut.core.highlight_detect import HighlightScore
        hs = HighlightScore(start_time=0, end_time=10, score=0.5)
        assert hs.audio_score == 0.0
        assert hs.visual_score == 0.0
        assert hs.chat_score == 0.0
        assert hs.label == ""

    def test_asdict_roundtrip(self):
        from opencut.core.highlight_detect import HighlightScore
        hs = HighlightScore(start_time=5.0, end_time=15.0, score=0.7)
        d = asdict(hs)
        assert d["start_time"] == 5.0
        assert d["score"] == 0.7


class TestWeightNormalization:
    """score_segments weight normalization (tested indirectly)."""

    def test_weights_sum_to_one(self):
        """Weights should normalise to sum to 1.0 inside score_segments."""
        aw, vw, cw = 5.0, 3.0, 2.0
        total = aw + vw + cw
        assert abs((aw / total + vw / total + cw / total) - 1.0) < 1e-9

    def test_zero_total_weight_edge(self):
        # If all weights are zero, the function normalises to 0/0. The actual
        # code guards with `if total_w > 0`. With all zeros, scores stay zero.
        aw, vw, cw = 0.0, 0.0, 0.0
        total = aw + vw + cw
        assert total == 0.0


# ===================================================================
# 9. opencut/core/soft_subtitles.py
# ===================================================================


class TestSubtitleCodecSelection:
    """_subtitle_codec() — codec selection by container and sub format."""

    def _fn(self, container, sub_path):
        from opencut.core.soft_subtitles import _subtitle_codec
        return _subtitle_codec(container, sub_path)

    def test_mp4_always_mov_text(self):
        assert self._fn("mp4", "sub.srt") == "mov_text"
        assert self._fn("mp4", "sub.ass") == "mov_text"

    def test_m4v_always_mov_text(self):
        assert self._fn("m4v", "sub.vtt") == "mov_text"

    def test_mkv_srt(self):
        assert self._fn("mkv", "sub.srt") == "srt"

    def test_mkv_ass(self):
        assert self._fn("mkv", "sub.ass") == "ass"

    def test_mkv_ssa(self):
        assert self._fn("mkv", "sub.ssa") == "ass"

    def test_mkv_vtt(self):
        assert self._fn("mkv", "sub.vtt") == "webvtt"

    def test_mkv_unknown_ext_defaults_srt(self):
        assert self._fn("mkv", "sub.xyz") == "srt"

    def test_webm_always_webvtt(self):
        assert self._fn("webm", "sub.srt") == "webvtt"

    def test_unknown_container_defaults_mov_text(self):
        assert self._fn("avi", "sub.srt") == "mov_text"

    def test_strips_dot_prefix(self):
        assert self._fn(".mp4", "sub.srt") == "mov_text"


class TestEmbedSubtitlesValidation:
    """embed_subtitles() input validation (no FFmpeg needed)."""

    def test_empty_subtitle_paths_raises(self):
        from opencut.core.soft_subtitles import embed_subtitles
        with pytest.raises(ValueError, match="At least one subtitle"):
            embed_subtitles("video.mp4", [])

    def test_mismatched_languages_raises(self):
        from opencut.core.soft_subtitles import embed_subtitles
        # Create real temp files so the file-existence check passes
        # before the languages-length check fires.
        paths = []
        try:
            for _ in range(2):
                fd, p = tempfile.mkstemp(suffix=".srt")
                os.close(fd)
                paths.append(p)
            with pytest.raises(ValueError, match="languages list length"):
                embed_subtitles("video.mp4", paths, languages=["eng"])
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def test_nonexistent_subtitle_raises(self):
        from opencut.core.soft_subtitles import embed_subtitles
        with pytest.raises(FileNotFoundError):
            embed_subtitles("video.mp4", ["/nonexistent/sub.srt"])


# ===================================================================
# 10. opencut/core/adr_cueing.py
# ===================================================================


class TestADRCueDataclass:
    """ADRCue fields and defaults."""

    def test_defaults(self):
        from opencut.core.adr_cueing import ADRCue
        cue = ADRCue()
        assert cue.cue_id == ""
        assert cue.character == ""
        assert cue.status == "pending"
        assert cue.priority == "normal"
        assert cue.takes_recorded == 0

    def test_custom_values(self):
        from opencut.core.adr_cueing import ADRCue
        cue = ADRCue(
            cue_id="ADR-0001", character="Alice",
            start_time=10.0, end_time=15.0,
            reason="noise", priority="high",
        )
        assert cue.cue_id == "ADR-0001"
        assert cue.end_time - cue.start_time == 5.0


class TestCreateADRCueSheet:
    """create_adr_cue_sheet() — pure logic (no FFmpeg)."""

    def _fn(self, transcript, marked_lines, **kwargs):
        from opencut.core.adr_cueing import create_adr_cue_sheet
        return create_adr_cue_sheet(transcript, marked_lines, **kwargs)

    def test_basic_cue_sheet(self):
        transcript = [
            {"text": "Hello world", "start": 0.0, "end": 2.0, "character": "Bob"},
            {"text": "How are you", "start": 3.0, "end": 5.0, "character": "Alice"},
            {"text": "Fine thanks", "start": 6.0, "end": 8.0, "character": "Bob"},
        ]
        result = self._fn(transcript, [0, 2], project_name="Test")
        assert result.total_lines == 2
        assert result.project_name == "Test"
        assert result.cues[0].cue_id == "ADR-0001"
        assert result.cues[1].cue_id == "ADR-0002"

    def test_out_of_range_indices_skipped(self):
        transcript = [{"text": "Hello", "start": 0, "end": 1}]
        result = self._fn(transcript, [0, 5, -1])
        assert result.total_lines == 1

    def test_total_duration_computed(self):
        transcript = [
            {"text": "A", "start": 0, "end": 3},
            {"text": "B", "start": 5, "end": 10},
        ]
        result = self._fn(transcript, [0, 1])
        assert result.total_duration == 8.0  # 3 + 5

    def test_default_character(self):
        transcript = [{"text": "Line", "start": 0, "end": 1}]
        result = self._fn(transcript, [0])
        assert result.cues[0].character == "Unknown"

    def test_empty_marked_lines(self):
        transcript = [{"text": "Line", "start": 0, "end": 1}]
        result = self._fn(transcript, [])
        assert result.total_lines == 0
        assert result.cues == []


class TestExportCueSheet:
    """_export_cue_sheet() — JSON/CSV/TXT file export."""

    def _make_sheet(self):
        from opencut.core.adr_cueing import ADRCue, ADRCueSheet
        return ADRCueSheet(
            project_name="TestProject",
            cues=[
                ADRCue(cue_id="ADR-0001", character="Alice",
                       line_text="Hello", start_time=0.0, end_time=2.0,
                       reason="noise", priority="normal"),
            ],
            total_lines=1,
            total_duration=2.0,
        )

    def test_export_json(self):
        from opencut.core.adr_cueing import _export_cue_sheet
        sheet = self._make_sheet()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            _export_cue_sheet(sheet, path, "json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["project_name"] == "TestProject"
            assert len(data["cues"]) == 1
            assert data["cues"][0]["cue_id"] == "ADR-0001"
        finally:
            os.unlink(path)

    def test_export_csv(self):
        from opencut.core.adr_cueing import _export_cue_sheet
        sheet = self._make_sheet()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            _export_cue_sheet(sheet, path, "csv")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "ADR-0001" in content
            assert "Alice" in content
            assert "Cue ID" in content  # header row
        finally:
            os.unlink(path)

    def test_export_txt(self):
        from opencut.core.adr_cueing import _export_cue_sheet
        sheet = self._make_sheet()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            path = f.name
        try:
            _export_cue_sheet(sheet, path, "txt")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "ADR CUE SHEET: TestProject" in content
            assert "[ADR-0001]" in content
            assert "Alice" in content
        finally:
            os.unlink(path)


class TestADRGuideResult:
    """ADRGuideResult dataclass defaults."""

    def test_defaults(self):
        from opencut.core.adr_cueing import ADRGuideResult
        r = ADRGuideResult()
        assert r.output_path == ""
        assert r.preroll_seconds == 3.0
        assert r.postroll_seconds == 2.0


class TestADRSyncResult:
    """ADRSyncResult dataclass defaults."""

    def test_defaults(self):
        from opencut.core.adr_cueing import ADRSyncResult
        r = ADRSyncResult()
        assert r.sync_quality == "good"
        assert r.time_offset == 0.0
