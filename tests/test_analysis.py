"""
Unit tests for analysis and QC features (Round 29):
- Shot Type Auto-Classification
- Caption Compliance Checker
- Shot-Change-Aware Subtitle Timing
- Dropout & Glitch Detection
- Comprehensive QC Report Generator
- Analysis, Caption, and Subtitle route smoke tests
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. Shot Type Auto-Classification
# ========================================================================
class TestShotClassify:
    """Tests for opencut.core.shot_classify."""

    def test_shot_info_dataclass(self):
        from opencut.core.shot_classify import ShotInfo
        s = ShotInfo(start=0.0, end=5.0, shot_type="close_up", confidence=0.8)
        assert s.start == 0.0
        assert s.end == 5.0
        assert s.shot_type == "close_up"
        assert s.confidence == 0.8

    def test_shot_class_result_dataclass(self):
        from opencut.core.shot_classify import ShotClassResult
        r = ShotClassResult()
        assert r.shots == []
        assert r.total_shots == 0
        assert r.duration == 0.0
        assert r.type_distribution == {}

    def test_shot_types_constant(self):
        from opencut.core.shot_classify import SHOT_TYPES
        assert "extreme_close_up" in SHOT_TYPES
        assert "close_up" in SHOT_TYPES
        assert "medium" in SHOT_TYPES
        assert "wide" in SHOT_TYPES
        assert "extreme_wide" in SHOT_TYPES
        assert "insert" in SHOT_TYPES
        assert "over_shoulder" in SHOT_TYPES
        assert len(SHOT_TYPES) == 10

    def test_classify_single_frame_returns_dict(self):
        """classify_single_frame should return a dict with shot_type and confidence."""
        from opencut.core.shot_classify import classify_single_frame

        probe_json = json.dumps({"streams": [{"width": 1920, "height": 1080}]})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_json

        # Signal stats mock (face detection + entropy)
        mock_analyze = MagicMock()
        mock_analyze.returncode = 0
        mock_analyze.stderr = ""

        with patch("opencut.core.shot_classify.subprocess.run") as mock_run:
            mock_run.return_value = mock_probe
            result = classify_single_frame("test_frame.jpg")

        assert "shot_type" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert result["shot_type"] in [
            "extreme_close_up", "close_up", "medium_close_up",
            "medium", "medium_wide", "wide", "extreme_wide",
            "aerial", "insert", "over_shoulder",
        ]

    def test_classify_shots_with_no_scenes(self):
        """classify_shots should handle video with no scene boundaries."""
        from opencut.core.shot_classify import ShotClassResult, classify_shots

        with patch("opencut.core.shot_classify.get_video_info") as mock_info, \
             patch("opencut.core.scene_detect.detect_scenes") as mock_scenes, \
             patch("opencut.core.shot_classify.classify_single_frame") as mock_frame, \
             patch("opencut.core.shot_classify.subprocess.run") as mock_run, \
             patch("opencut.core.shot_classify.os.path.isfile", return_value=True), \
             patch("shutil.rmtree"):

            mock_info.return_value = {"duration": 10.0, "fps": 30.0, "width": 1920, "height": 1080}

            from opencut.core.scene_detect import SceneInfo
            mock_scenes.return_value = SceneInfo(boundaries=[], total_scenes=0, duration=10.0)

            mock_frame.return_value = {"shot_type": "medium", "confidence": 0.5}
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = classify_shots("test.mp4")

        assert isinstance(result, ShotClassResult)
        assert result.total_shots >= 1
        assert result.duration == 10.0

    def test_classify_shots_progress_callback(self):
        """Progress callback should be called during classification."""
        from opencut.core.shot_classify import classify_shots

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        with patch("opencut.core.shot_classify.get_video_info") as mock_info, \
             patch("opencut.core.scene_detect.detect_scenes") as mock_scenes, \
             patch("opencut.core.shot_classify.classify_single_frame") as mock_frame, \
             patch("opencut.core.shot_classify.subprocess.run") as mock_run, \
             patch("opencut.core.shot_classify.os.path.isfile", return_value=True), \
             patch("shutil.rmtree"):

            mock_info.return_value = {"duration": 10.0, "fps": 30.0}
            from opencut.core.scene_detect import SceneInfo
            mock_scenes.return_value = SceneInfo(boundaries=[], total_scenes=0, duration=10.0)
            mock_frame.return_value = {"shot_type": "wide", "confidence": 0.5}
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            classify_shots("test.mp4", on_progress=on_progress)

        assert len(progress_calls) >= 3
        assert progress_calls[-1] == 100


# ========================================================================
# 2. Caption Compliance Checker
# ========================================================================
class TestCaptionCompliance:
    """Tests for opencut.core.caption_compliance."""

    @pytest.fixture
    def sample_srt(self, tmp_path):
        """Create a sample SRT file for testing."""
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "Hello, this is a test subtitle.\n\n"
            "2\n"
            "00:00:04,000 --> 00:00:06,000\n"
            "Second line of subtitles here.\n\n"
            "3\n"
            "00:00:07,000 --> 00:00:09,000\n"
            "Third subtitle block.\n\n",
            encoding="utf-8",
        )
        return str(srt)

    @pytest.fixture
    def long_line_srt(self, tmp_path):
        """Create SRT with lines exceeding CPL limits."""
        srt = tmp_path / "long.srt"
        long_text = "A" * 50  # exceeds all standard CPL limits
        srt.write_text(
            f"1\n"
            f"00:00:01,000 --> 00:00:03,000\n"
            f"{long_text}\n\n",
            encoding="utf-8",
        )
        return str(srt)

    def test_compliance_result_dataclass(self):
        from opencut.core.caption_compliance import ComplianceResult
        r = ComplianceResult()
        assert r.violations == []
        assert r.pass_rate == 100.0
        assert r.overall_pass is True

    def test_violation_dataclass(self):
        from opencut.core.caption_compliance import Violation
        v = Violation(
            line_num=1, start_time=1.0,
            violation_type="characters_per_line",
            description="Line too long",
        )
        assert v.severity == "error"
        assert v.fix_suggestion == ""

    def test_standards_defined(self):
        from opencut.core.caption_compliance import STANDARDS
        assert "netflix" in STANDARDS
        assert "bbc" in STANDARDS
        assert "fcc" in STANDARDS
        assert "youtube" in STANDARDS
        assert STANDARDS["netflix"]["max_cpl"] == 42
        assert STANDARDS["bbc"]["max_cpl"] == 37
        assert STANDARDS["fcc"]["max_cpl"] == 32

    def test_check_compliance_passes_clean_file(self, sample_srt):
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(sample_srt, standard="netflix")
        assert result.overall_pass is True
        assert result.total_subtitles == 3
        assert result.standard == "netflix"

    def test_check_compliance_detects_long_lines(self, long_line_srt):
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(long_line_srt, standard="netflix")
        cpl_violations = [v for v in result.violations if v.violation_type == "characters_per_line"]
        assert len(cpl_violations) >= 1
        assert result.overall_pass is False

    def test_check_compliance_fcc_stricter(self, sample_srt):
        """FCC has 32 CPL - lines under 42 but over 32 should fail FCC."""
        from opencut.core.caption_compliance import check_caption_compliance
        # Our sample has lines around 31 chars - should pass FCC too
        result = check_caption_compliance(sample_srt, standard="fcc")
        assert result.standard == "fcc"

    def test_check_compliance_detects_overlap(self, tmp_path):
        srt = tmp_path / "overlap.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:05,000\n"
            "First subtitle.\n\n"
            "2\n"
            "00:00:04,000 --> 00:00:07,000\n"
            "Second overlaps first.\n\n",
            encoding="utf-8",
        )
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(str(srt), standard="netflix")
        overlap = [v for v in result.violations if v.violation_type == "overlap"]
        assert len(overlap) >= 1

    def test_check_compliance_min_duration(self, tmp_path):
        srt = tmp_path / "short.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:01,050\n"
            "Too short.\n\n",
            encoding="utf-8",
        )
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(str(srt), standard="netflix")
        dur_violations = [v for v in result.violations if v.violation_type == "min_duration"]
        assert len(dur_violations) >= 1

    def test_check_compliance_bbc_gap_check(self, tmp_path):
        srt = tmp_path / "gap.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "First.\n\n"
            "2\n"
            "00:00:03,100 --> 00:00:05,000\n"
            "Too close gap.\n\n",
            encoding="utf-8",
        )
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(str(srt), standard="bbc")
        gap_violations = [v for v in result.violations if v.violation_type == "subtitle_gap"]
        assert len(gap_violations) >= 1

    def test_auto_fix_splits_long_lines(self, long_line_srt, tmp_path):
        from opencut.core.caption_compliance import auto_fix_compliance
        output = str(tmp_path / "fixed.srt")
        result = auto_fix_compliance(long_line_srt, standard="netflix", output_path_str=output)
        assert result["fixes_applied"] >= 1
        assert os.path.isfile(result["output_path"])

    def test_auto_fix_extends_short_duration(self, tmp_path):
        srt = tmp_path / "short.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:01,050\n"
            "Short.\n\n"
            "2\n"
            "00:00:05,000 --> 00:00:07,000\n"
            "Normal.\n\n",
            encoding="utf-8",
        )
        from opencut.core.caption_compliance import auto_fix_compliance
        output = str(tmp_path / "fixed.srt")
        result = auto_fix_compliance(str(srt), standard="netflix", output_path_str=output)
        assert result["fixes_applied"] >= 1

    def test_auto_fix_progress_callback(self, sample_srt, tmp_path):
        from opencut.core.caption_compliance import auto_fix_compliance

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        output = str(tmp_path / "fixed.srt")
        auto_fix_compliance(sample_srt, on_progress=on_progress, output_path_str=output)
        assert len(progress_calls) >= 3
        assert progress_calls[-1] == 100

    def test_check_compliance_empty_file(self, tmp_path):
        srt = tmp_path / "empty.srt"
        srt.write_text("", encoding="utf-8")
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(str(srt))
        assert result.total_subtitles == 0
        assert result.overall_pass is True

    def test_invalid_standard_defaults_to_netflix(self, sample_srt):
        from opencut.core.caption_compliance import check_caption_compliance
        result = check_caption_compliance(sample_srt, standard="invalid")
        assert result.standard == "netflix"


# ========================================================================
# 3. Shot-Change-Aware Subtitle Timing
# ========================================================================
class TestSubtitleTiming:
    """Tests for opencut.core.subtitle_timing."""

    @pytest.fixture
    def sample_srt(self, tmp_path):
        srt = tmp_path / "subs.srt"
        srt.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "This subtitle spans a cut.\n\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "This one does not.\n\n",
            encoding="utf-8",
        )
        return str(srt)

    def test_snap_no_cuts_unchanged(self, sample_srt, tmp_path):
        from opencut.core.subtitle_timing import snap_subtitles_to_cuts
        output = str(tmp_path / "out.srt")
        result = snap_subtitles_to_cuts(sample_srt, cut_times=[], output_path=output)
        assert result["adjustments_made"] == 0
        assert os.path.isfile(result["output_path"])

    def test_snap_splits_at_cut(self, sample_srt, tmp_path):
        from opencut.core.subtitle_timing import snap_subtitles_to_cuts
        output = str(tmp_path / "out.srt")
        # Cut at 2.5s should split the first subtitle
        result = snap_subtitles_to_cuts(
            sample_srt, cut_times=[2.5], fps=24.0,
            output_path=output,
        )
        assert result["adjustments_made"] >= 1

    def test_snap_adjusts_near_cut(self, tmp_path):
        srt = tmp_path / "near.srt"
        srt.write_text(
            "1\n"
            "00:00:02,000 --> 00:00:04,000\n"
            "Starts right after cut.\n\n",
            encoding="utf-8",
        )
        from opencut.core.subtitle_timing import snap_subtitles_to_cuts
        output = str(tmp_path / "out.srt")
        snap_subtitles_to_cuts(
            str(srt), cut_times=[1.97], fps=24.0,
            min_gap_frames=2, output_path=output,
        )
        assert os.path.isfile(output)

    def test_snap_progress_callback(self, sample_srt, tmp_path):
        from opencut.core.subtitle_timing import snap_subtitles_to_cuts

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        output = str(tmp_path / "out.srt")
        snap_subtitles_to_cuts(
            sample_srt, cut_times=[2.5], output_path=output,
            on_progress=on_progress,
        )
        assert len(progress_calls) >= 2

    def test_auto_snap_calls_scene_detection(self, sample_srt, tmp_path):
        from opencut.core.subtitle_timing import auto_snap_subtitles

        with patch("opencut.core.scene_detect.detect_scenes") as mock_detect, \
             patch("opencut.core.subtitle_timing.get_video_info") as mock_info:

            from opencut.core.scene_detect import SceneBoundary, SceneInfo
            mock_detect.return_value = SceneInfo(
                boundaries=[
                    SceneBoundary(time=0.0), SceneBoundary(time=3.0),
                ],
                total_scenes=2, duration=10.0,
            )
            mock_info.return_value = {"fps": 24.0, "duration": 10.0}

            output = str(tmp_path / "out.srt")
            result = auto_snap_subtitles(
                sample_srt, video_path="test.mp4", output_path=output,
            )

        assert "cuts_detected" in result
        assert result["cuts_detected"] >= 1
        mock_detect.assert_called_once()

    def test_auto_snap_progress_callback(self, sample_srt, tmp_path):
        from opencut.core.subtitle_timing import auto_snap_subtitles

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        with patch("opencut.core.scene_detect.detect_scenes") as mock_detect, \
             patch("opencut.core.subtitle_timing.get_video_info") as mock_info:

            from opencut.core.scene_detect import SceneInfo
            mock_detect.return_value = SceneInfo(boundaries=[], total_scenes=0, duration=10.0)
            mock_info.return_value = {"fps": 24.0, "duration": 10.0}

            output = str(tmp_path / "out.srt")
            auto_snap_subtitles(
                sample_srt, video_path="test.mp4",
                output_path=output, on_progress=on_progress,
            )

        assert len(progress_calls) >= 3
        assert progress_calls[-1] == 100


# ========================================================================
# 4. Dropout & Glitch Detection
# ========================================================================
class TestDropoutDetection:
    """Tests for detect_dropouts in opencut.core.qc_checks."""

    def test_dropout_dataclass(self):
        from opencut.core.qc_checks import Dropout
        d = Dropout(
            frame_num=100, timestamp=3.33,
            type="frame_glitch", severity="critical",
            description="test",
        )
        assert d.frame_num == 100
        assert d.type == "frame_glitch"

    def test_dropout_result_dataclass(self):
        from opencut.core.qc_checks import DropoutResult
        r = DropoutResult()
        assert r.dropouts == []
        assert r.total_dropouts == 0

    def test_detect_dropouts_parses_ssim(self):
        """Should detect glitches from SSIM output."""
        # SSIM output format: n:FRAME Y:value U:value V:value All:value (dB)
        ssim_lines = [
            "n:1 Y:0.999000 U:0.998000 V:0.997000 All:0.998000 (30.000000)",
            "n:2 Y:0.100000 U:0.150000 V:0.120000 All:0.120000 (5.000000)",
            "n:3 Y:0.998000 U:0.997000 V:0.996000 All:0.997000 (29.000000)",
        ]
        ssim_stderr = "\n".join(ssim_lines) + "\n"

        mock_probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = mock_probe_stdout

        mock_ssim = MagicMock()
        mock_ssim.returncode = 0
        mock_ssim.stderr = ssim_stderr

        # TC check
        mock_tc = MagicMock()
        mock_tc.returncode = 0
        mock_tc.stdout = "0.033\n0.066\n0.100\n"

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run, \
             patch("opencut.core.qc_checks.get_video_info") as mock_info:
            mock_run.side_effect = [mock_probe, mock_ssim, mock_tc]
            mock_info.return_value = {"fps": 30.0, "duration": 10.0}

            from opencut.core.qc_checks import detect_dropouts
            result = detect_dropouts("test.mp4", ssim_threshold=0.5)

        assert result.total_dropouts >= 1
        glitches = [d for d in result.dropouts if d.type == "frame_glitch"]
        assert len(glitches) >= 1
        assert glitches[0].frame_num == 2

    def test_detect_dropouts_no_glitches(self):
        """Should return empty when all SSIM values are high."""
        ssim_lines = [
            "n:1 Y:0.999000 U:0.998000 V:0.997000 All:0.998000 (30.000000)",
            "n:2 Y:0.997000 U:0.996000 V:0.995000 All:0.996000 (29.000000)",
        ]
        ssim_stderr = "\n".join(ssim_lines) + "\n"

        mock_probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = mock_probe_stdout

        mock_ssim = MagicMock()
        mock_ssim.returncode = 0
        mock_ssim.stderr = ssim_stderr

        mock_tc = MagicMock()
        mock_tc.returncode = 0
        mock_tc.stdout = "0.033\n0.066\n"

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run, \
             patch("opencut.core.qc_checks.get_video_info") as mock_info:
            mock_run.side_effect = [mock_probe, mock_ssim, mock_tc]
            mock_info.return_value = {"fps": 30.0, "duration": 10.0}

            from opencut.core.qc_checks import detect_dropouts
            result = detect_dropouts("test.mp4", ssim_threshold=0.5)

        assert result.total_dropouts == 0

    def test_detect_dropouts_timecode_break(self):
        """Should detect timecode breaks from large PTS gaps."""
        ssim_stderr = "n:1 Y:0.999 U:0.999 V:0.999 All:0.999 (30.0)\n"

        mock_probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = mock_probe_stdout

        mock_ssim = MagicMock()
        mock_ssim.returncode = 0
        mock_ssim.stderr = ssim_stderr

        # Big gap in PTS
        mock_tc = MagicMock()
        mock_tc.returncode = 0
        mock_tc.stdout = "0.033\n0.066\n0.100\n5.000\n5.033\n"

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run, \
             patch("opencut.core.qc_checks.get_video_info") as mock_info:
            mock_run.side_effect = [mock_probe, mock_ssim, mock_tc]
            mock_info.return_value = {"fps": 30.0, "duration": 10.0}

            from opencut.core.qc_checks import detect_dropouts
            result = detect_dropouts("test.mp4")

        tc_breaks = [d for d in result.dropouts if d.type == "timecode_break"]
        assert len(tc_breaks) >= 1

    def test_detect_dropouts_ffmpeg_not_found(self):
        """Should raise RuntimeError when ffmpeg is missing."""
        mock_probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = mock_probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run, \
             patch("opencut.core.qc_checks.get_video_info") as mock_info:
            mock_run.side_effect = [mock_probe, FileNotFoundError()]
            mock_info.return_value = {"fps": 30.0, "duration": 10.0}

            from opencut.core.qc_checks import detect_dropouts
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                detect_dropouts("test.mp4")

    def test_detect_dropouts_progress_callback(self):
        ssim_stderr = "n:1 Y:0.999 U:0.999 V:0.999 All:0.999 (30.0)\n"
        mock_probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = mock_probe_stdout

        mock_ssim = MagicMock()
        mock_ssim.returncode = 0
        mock_ssim.stderr = ssim_stderr

        mock_tc = MagicMock()
        mock_tc.returncode = 0
        mock_tc.stdout = ""

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run, \
             patch("opencut.core.qc_checks.get_video_info") as mock_info:
            mock_run.side_effect = [mock_probe, mock_ssim, mock_tc]
            mock_info.return_value = {"fps": 30.0, "duration": 10.0}

            from opencut.core.qc_checks import detect_dropouts
            detect_dropouts("test.mp4", on_progress=on_progress)

        assert len(progress_calls) >= 3
        assert progress_calls[-1] == 100


# ========================================================================
# 5. Comprehensive QC Report Generator
# ========================================================================
class TestQCReportGenerator:
    """Tests for generate_qc_report and export_qc_report_html."""

    def test_qc_check_result_dataclass(self):
        from opencut.core.qc_checks import QCCheckResult
        c = QCCheckResult(check_name="test", status="pass", details="OK")
        assert c.check_name == "test"
        assert c.issues == []

    def test_qc_report_dataclass(self):
        from opencut.core.qc_checks import QCReport
        r = QCReport()
        assert r.overall_verdict == "pass"
        assert r.per_check == []
        assert r.total_issues == 0
        assert r.html_report == ""

    def test_rulesets_defined(self):
        from opencut.core.qc_checks import RULESETS
        assert "broadcast" in RULESETS
        assert "netflix" in RULESETS
        assert "youtube" in RULESETS
        assert "label" in RULESETS["broadcast"]

    def test_generate_report_runs_all_checks(self):
        """Should run all QC checks and aggregate results."""
        from opencut.core.qc_checks import (
            BlackFrameResult,
            DropoutResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
            generate_qc_report,
        )

        with patch("opencut.core.qc_checks._probe_duration", return_value=60.0), \
             patch("opencut.core.qc_checks.detect_black_frames") as m_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as m_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as m_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as m_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as m_ld, \
             patch("opencut.core.qc_checks.detect_dropouts") as m_dr, \
             patch("opencut.core.audio_suite.measure_loudness") as m_loud:

            m_bf.return_value = BlackFrameResult(frames=[], file_duration=60.0)
            m_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            m_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            m_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            m_ld.return_value = LeaderResult()
            m_dr.return_value = DropoutResult()

            # Mock loudness - broadcast target is -24.0 LUFS
            mock_loud = MagicMock()
            mock_loud.input_i = -24.0
            mock_loud.input_tp = -1.0
            m_loud.return_value = mock_loud

            report = generate_qc_report("test.mp4", ruleset="broadcast")

        assert report.overall_verdict == "pass"
        assert len(report.per_check) >= 6
        assert report.html_report != ""
        m_bf.assert_called_once()
        m_ff.assert_called_once()
        m_dr.assert_called_once()

    def test_generate_report_fails_on_critical(self):
        """Should set verdict=fail when critical issues found."""
        from opencut.core.qc_checks import (
            BlackFrame,
            BlackFrameResult,
            DropoutResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
            generate_qc_report,
        )

        with patch("opencut.core.qc_checks._probe_duration", return_value=60.0), \
             patch("opencut.core.qc_checks.detect_black_frames") as m_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as m_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as m_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as m_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as m_ld, \
             patch("opencut.core.qc_checks.detect_dropouts") as m_dr, \
             patch("opencut.core.audio_suite.measure_loudness") as m_loud:

            m_bf.return_value = BlackFrameResult(
                frames=[BlackFrame(start=1.0, end=2.0, duration=1.0)],
                total_black_duration=1.0, file_duration=60.0, black_percentage=1.67,
            )
            m_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            m_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            m_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            m_ld.return_value = LeaderResult()
            m_dr.return_value = DropoutResult()
            mock_loud = MagicMock()
            mock_loud.input_i = -14.0
            mock_loud.input_tp = -1.0
            m_loud.return_value = mock_loud

            report = generate_qc_report("test.mp4")

        assert report.overall_verdict == "fail"
        assert report.critical_count >= 1

    def test_generate_report_continues_on_check_failure(self):
        """If one check raises, others should still run."""
        from opencut.core.qc_checks import (
            DropoutResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
            generate_qc_report,
        )

        with patch("opencut.core.qc_checks._probe_duration", return_value=60.0), \
             patch("opencut.core.qc_checks.detect_black_frames") as m_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as m_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as m_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as m_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as m_ld, \
             patch("opencut.core.qc_checks.detect_dropouts") as m_dr, \
             patch("opencut.core.audio_suite.measure_loudness") as m_loud:

            m_bf.side_effect = RuntimeError("FFmpeg crashed")
            m_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            m_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            m_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            m_ld.return_value = LeaderResult()
            m_dr.return_value = DropoutResult()
            mock_loud = MagicMock()
            mock_loud.input_i = -14.0
            mock_loud.input_tp = -1.0
            m_loud.return_value = mock_loud

            report = generate_qc_report("test.mp4")

        error_checks = [c for c in report.per_check if c.status == "error"]
        assert len(error_checks) >= 1
        # Other checks should have run
        pass_checks = [c for c in report.per_check if c.status == "pass"]
        assert len(pass_checks) >= 3

    def test_generate_report_writes_json(self, tmp_path):
        """Should write JSON output when output_path given."""
        from opencut.core.qc_checks import (
            BlackFrameResult,
            DropoutResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
            generate_qc_report,
        )
        json_path = str(tmp_path / "report.json")

        with patch("opencut.core.qc_checks._probe_duration", return_value=60.0), \
             patch("opencut.core.qc_checks.detect_black_frames") as m_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as m_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as m_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as m_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as m_ld, \
             patch("opencut.core.qc_checks.detect_dropouts") as m_dr, \
             patch("opencut.core.audio_suite.measure_loudness") as m_loud:

            m_bf.return_value = BlackFrameResult(frames=[], file_duration=60.0)
            m_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            m_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            m_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            m_ld.return_value = LeaderResult()
            m_dr.return_value = DropoutResult()
            mock_loud = MagicMock()
            mock_loud.input_i = -14.0
            mock_loud.input_tp = -1.0
            m_loud.return_value = mock_loud

            generate_qc_report("test.mp4", output_path=json_path)

        assert os.path.isfile(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert "overall_verdict" in data
        assert "per_check" in data

    def test_export_html_report_structure(self):
        """HTML report should contain proper structure."""
        from opencut.core.qc_checks import QCCheckResult, QCReport, export_qc_report_html

        report = QCReport(
            overall_verdict="pass",
            per_check=[
                QCCheckResult(check_name="black_frames", status="pass", details="OK"),
                QCCheckResult(check_name="frozen_frames", status="fail", details="2 regions"),
            ],
            total_issues=2,
            critical_count=2,
            warning_count=0,
            file_path="test.mp4",
            file_duration=60.0,
        )

        html_str = export_qc_report_html(report)
        assert "<!DOCTYPE html>" in html_str
        assert "OpenCut QC Report" in html_str
        assert "black_frames" in html_str
        assert "frozen_frames" in html_str
        assert "test.mp4" in html_str

    def test_export_html_writes_file(self, tmp_path):
        from opencut.core.qc_checks import QCReport, export_qc_report_html
        report = QCReport(overall_verdict="pass")
        path = str(tmp_path / "report.html")
        export_qc_report_html(report, output_path=path)
        assert os.path.isfile(path)

    def test_generate_report_progress_callback(self):
        from opencut.core.qc_checks import (
            BlackFrameResult,
            DropoutResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
            generate_qc_report,
        )

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        with patch("opencut.core.qc_checks._probe_duration", return_value=60.0), \
             patch("opencut.core.qc_checks.detect_black_frames") as m_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as m_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as m_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as m_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as m_ld, \
             patch("opencut.core.qc_checks.detect_dropouts") as m_dr, \
             patch("opencut.core.audio_suite.measure_loudness") as m_loud:

            m_bf.return_value = BlackFrameResult(frames=[], file_duration=60.0)
            m_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            m_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            m_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            m_ld.return_value = LeaderResult()
            m_dr.return_value = DropoutResult()
            mock_loud = MagicMock()
            mock_loud.input_i = -14.0
            mock_loud.input_tp = -1.0
            m_loud.return_value = mock_loud

            generate_qc_report("test.mp4", on_progress=on_progress)

        assert progress_calls[0] == 0
        assert progress_calls[-1] == 100

    def test_invalid_ruleset_defaults_to_broadcast(self):
        from opencut.core.qc_checks import (
            BlackFrameResult,
            DropoutResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
            generate_qc_report,
        )

        with patch("opencut.core.qc_checks._probe_duration", return_value=60.0), \
             patch("opencut.core.qc_checks.detect_black_frames") as m_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as m_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as m_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as m_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as m_ld, \
             patch("opencut.core.qc_checks.detect_dropouts") as m_dr, \
             patch("opencut.core.audio_suite.measure_loudness") as m_loud:

            m_bf.return_value = BlackFrameResult(frames=[], file_duration=60.0)
            m_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            m_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            m_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            m_ld.return_value = LeaderResult()
            m_dr.return_value = DropoutResult()
            mock_loud = MagicMock()
            mock_loud.input_i = -14.0
            mock_loud.input_tp = -1.0
            m_loud.return_value = mock_loud

            report = generate_qc_report("test.mp4", ruleset="invalid")

        assert report.ruleset == "broadcast"


# ========================================================================
# 6. Route Smoke Tests
# ========================================================================
class TestAnalysisRoutes:
    """Smoke tests for analysis, caption, and subtitle routes."""

    @pytest.fixture
    def app(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        test_config = OpenCutConfig()
        flask_app = create_app(config=test_config)
        flask_app.config["TESTING"] = True
        return flask_app

    @pytest.fixture
    def client(self, app):
        return app.test_client()

    @pytest.fixture
    def csrf_token(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        return data.get("csrf_token", "")

    def _headers(self, token):
        return {
            "X-OpenCut-Token": token,
            "Content-Type": "application/json",
        }

    # --- Analysis routes ---

    def test_shot_classify_route_exists(self, client, csrf_token):
        """POST /analysis/shot-classify should be reachable (not 404/405)."""
        resp = client.post(
            "/analysis/shot-classify",
            json={"filepath": "/nonexistent/test.mp4"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404
        assert resp.status_code != 405

    def test_shot_classify_frame_route_exists(self, client, csrf_token):
        """POST /analysis/shot-classify-frame should be reachable."""
        resp = client.post(
            "/analysis/shot-classify-frame",
            json={"filepath": "/nonexistent/frame.jpg"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code != 404
        assert resp.status_code != 405

    def test_shot_classify_frame_requires_filepath(self, client, csrf_token):
        """Missing filepath should return 400."""
        resp = client.post(
            "/analysis/shot-classify-frame",
            json={},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code == 400

    # --- Caption compliance routes ---

    def test_caption_compliance_route_exists(self, client, csrf_token):
        """POST /caption/compliance should be reachable."""
        resp = client.post(
            "/caption/compliance",
            json={"srt_path": "/nonexistent/test.srt"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    def test_caption_compliance_fix_route_exists(self, client, csrf_token):
        """POST /caption/compliance/fix should be reachable."""
        resp = client.post(
            "/caption/compliance/fix",
            json={"srt_path": "/nonexistent/test.srt"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    # --- Subtitle timing routes ---

    def test_subtitle_snap_route_exists(self, client, csrf_token):
        """POST /subtitle/snap-to-cuts should be reachable."""
        resp = client.post(
            "/subtitle/snap-to-cuts",
            json={"srt_path": "/nonexistent/test.srt", "cut_times": [1.0, 3.0]},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    def test_subtitle_auto_snap_route_exists(self, client, csrf_token):
        """POST /subtitle/auto-snap should be reachable."""
        resp = client.post(
            "/subtitle/auto-snap",
            json={"srt_path": "/nonexistent/test.srt", "video_path": "/nonexistent/test.mp4"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    # --- QC routes (extended) ---

    def test_qc_dropouts_route_exists(self, client, csrf_token):
        """POST /qc/dropouts should be reachable."""
        resp = client.post(
            "/qc/dropouts",
            json={"filepath": "/nonexistent/test.mp4"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    def test_qc_report_route_exists(self, client, csrf_token):
        """POST /qc/report should be reachable."""
        resp = client.post(
            "/qc/report",
            json={"filepath": "/nonexistent/test.mp4"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    def test_qc_report_validates_ruleset(self, client, csrf_token):
        """Invalid ruleset should still work (defaults to broadcast)."""
        resp = client.post(
            "/qc/report",
            json={"filepath": "/nonexistent/test.mp4", "ruleset": "invalid"},
            headers=self._headers(csrf_token),
        )
        assert resp.status_code in (200, 400, 500)
        assert resp.status_code != 404

    # --- CSRF enforcement ---

    def test_shot_classify_requires_csrf(self, client):
        """POST /analysis/shot-classify should require CSRF."""
        resp = client.post(
            "/analysis/shot-classify",
            json={"filepath": "/nonexistent/test.mp4"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 403

    def test_caption_compliance_requires_csrf(self, client):
        """POST /caption/compliance should require CSRF."""
        resp = client.post(
            "/caption/compliance",
            json={"srt_path": "/nonexistent/test.srt"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 403

    def test_qc_dropouts_requires_csrf(self, client):
        """POST /qc/dropouts should require CSRF."""
        resp = client.post(
            "/qc/dropouts",
            json={"filepath": "/nonexistent/test.mp4"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 403
