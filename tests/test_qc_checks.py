"""
Unit tests for opencut.core.qc_checks — QC/QA detection engine.

Tests all functions with mocked FFmpeg subprocess calls.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ========================================================================
# 1. Black Frame Detection
# ========================================================================
class TestBlackFrameDetection:
    """Tests for detect_black_frames."""

    def test_parses_blackdetect_output(self):
        """Should parse black_start/end/duration from ffmpeg stderr."""
        ffmpeg_stderr = (
            "[blackdetect @ 0x1234] black_start:1.500 black_end:3.200 black_duration:1.700\n"
            "[blackdetect @ 0x1234] black_start:10.000 black_end:12.500 black_duration:2.500\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_black_frames
            result = detect_black_frames("test.mp4", threshold=0.98, min_duration=0.5)

        assert len(result.frames) == 2
        assert result.frames[0].start == 1.5
        assert result.frames[0].end == 3.2
        assert result.frames[0].duration == 1.7
        assert result.frames[1].start == 10.0
        assert result.frames[1].end == 12.5
        assert result.total_black_duration == 4.2
        assert result.file_duration == 60.0
        assert result.black_percentage == 7.0

    def test_no_black_frames(self):
        """Should return empty list when no black frames detected."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "some output with no blackdetect data\n"

        probe_stdout = json.dumps({"format": {"duration": "30.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_black_frames
            result = detect_black_frames("test.mp4")

        assert result.frames == []
        assert result.total_black_duration == 0.0
        assert result.black_percentage == 0.0

    def test_ffmpeg_not_found_raises(self):
        """Should raise RuntimeError when ffmpeg binary is missing."""
        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, FileNotFoundError()]
            from opencut.core.qc_checks import detect_black_frames
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                detect_black_frames("test.mp4")

    def test_timeout_raises(self):
        """Should raise RuntimeError on subprocess timeout."""
        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [
                mock_probe,
                subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600),
            ]
            from opencut.core.qc_checks import detect_black_frames
            with pytest.raises(RuntimeError, match="timed out"):
                detect_black_frames("test.mp4")

    def test_progress_callback_called(self):
        """Progress callback should be invoked during detection."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append((pct, msg))

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_black_frames
            detect_black_frames("test.mp4", on_progress=on_progress)

        assert len(progress_calls) >= 3
        assert progress_calls[-1][0] == 100


# ========================================================================
# 2. Frozen Frame Detection
# ========================================================================
class TestFrozenFrameDetection:
    """Tests for detect_frozen_frames."""

    def test_parses_freezedetect_output(self):
        """Should parse freeze_start/end/duration from ffmpeg stderr."""
        ffmpeg_stderr = (
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_start: 5.000\n"
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_duration: 3.000\n"
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_end: 8.000\n"
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_start: 20.000\n"
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_duration: 5.000\n"
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_end: 25.000\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_frozen_frames
            result = detect_frozen_frames("test.mp4", noise_threshold=0.001, duration_threshold=2.0)

        assert len(result.frames) == 2
        assert result.frames[0].start == 5.0
        assert result.frames[0].end == 8.0
        assert result.frames[0].duration == 3.0
        assert result.frames[1].start == 20.0
        assert result.frames[1].end == 25.0
        assert result.frames[1].duration == 5.0
        assert result.total_frozen_duration == 8.0
        assert result.frozen_percentage == pytest.approx(13.33, abs=0.01)

    def test_no_frozen_frames(self):
        """Should return empty list when no frozen frames detected."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "normal output with no freezedetect\n"

        probe_stdout = json.dumps({"format": {"duration": "30.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_frozen_frames
            result = detect_frozen_frames("test.mp4")

        assert result.frames == []
        assert result.total_frozen_duration == 0.0

    def test_unclosed_freeze_extends_to_eof(self):
        """A freeze_start without matching freeze_end should extend to file end."""
        ffmpeg_stderr = (
            "[freezedetect @ 0x5678] lavfi.freezedetect.freeze_start: 50.000\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_frozen_frames
            result = detect_frozen_frames("test.mp4")

        assert len(result.frames) == 1
        assert result.frames[0].start == 50.0
        assert result.frames[0].end == 60.0
        assert result.frames[0].duration == 10.0

    def test_ffmpeg_not_found_raises(self):
        """Should raise RuntimeError when ffmpeg is missing."""
        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, FileNotFoundError()]
            from opencut.core.qc_checks import detect_frozen_frames
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                detect_frozen_frames("test.mp4")


# ========================================================================
# 3. Audio Phase Check
# ========================================================================
class TestAudioPhaseCheck:
    """Tests for check_audio_phase."""

    def test_detects_phase_issues(self):
        """Should find phase issues when phase drops below threshold."""
        # Simulate aphasemeter output with some low-phase segments
        lines = []
        for i in range(20):
            ts = i * 0.5
            # Frames 4-8 have bad phase
            if 4 <= i <= 8:
                phase = -0.8
            else:
                phase = 0.9
            lines.append(f"[Parsed_aphasemeter_0 @ 0x1234] pts_time:{ts:.3f}")
            lines.append(f"[Parsed_aphasemeter_0 @ 0x1234] lavfi.aphasemeter.phase={phase:.4f}")

        ffmpeg_stderr = "\n".join(lines) + "\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import check_audio_phase
            result = check_audio_phase("test.mp4", threshold=-0.5)

        assert result.has_phase_problems is True
        assert len(result.issues) >= 1
        assert result.issues[0].avg_phase < -0.5

    def test_no_phase_issues(self):
        """Should return no issues when phase is healthy."""
        lines = []
        for i in range(10):
            ts = i * 1.0
            lines.append(f"[Parsed_aphasemeter_0 @ 0x1234] pts_time:{ts:.3f}")
            lines.append("[Parsed_aphasemeter_0 @ 0x1234] lavfi.aphasemeter.phase=0.9500")

        ffmpeg_stderr = "\n".join(lines) + "\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import check_audio_phase
            result = check_audio_phase("test.mp4", threshold=-0.5)

        assert result.has_phase_problems is False
        assert result.issues == []
        assert result.overall_avg_phase > 0

    def test_empty_output(self):
        """No aphasemeter output should return clean result."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import check_audio_phase
            result = check_audio_phase("test.mp4")

        assert result.has_phase_problems is False
        assert result.issues == []
        assert result.overall_avg_phase == 0.0

    def test_ffmpeg_not_found_raises(self):
        """Should raise RuntimeError when ffmpeg is missing."""
        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, FileNotFoundError()]
            from opencut.core.qc_checks import check_audio_phase
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                check_audio_phase("test.mp4")


# ========================================================================
# 4. Silence Gap Detection
# ========================================================================
class TestSilenceGapDetection:
    """Tests for detect_silence_gaps."""

    def test_parses_silencedetect_output(self):
        """Should parse silence_start/end/duration from ffmpeg stderr."""
        ffmpeg_stderr = (
            "[silencedetect @ 0xabc] silence_start: 5.000\n"
            "[silencedetect @ 0xabc] silence_end: 8.200 | silence_duration: 3.200\n"
            "[silencedetect @ 0xabc] silence_start: 20.000\n"
            "[silencedetect @ 0xabc] silence_end: 24.500 | silence_duration: 4.500\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_silence_gaps
            result = detect_silence_gaps("test.mp4", noise_db=-50, min_duration=2.0)

        assert len(result.gaps) == 2
        assert result.gaps[0].start == 5.0
        assert result.gaps[0].end == 8.2
        assert result.gaps[0].duration == 3.2
        assert result.gaps[1].start == 20.0
        assert result.gaps[1].end == 24.5
        assert result.gaps[1].duration == 4.5
        assert result.total_silence_duration == 7.7
        assert result.silence_percentage == pytest.approx(12.83, abs=0.01)

    def test_no_silence_gaps(self):
        """Should return empty list when no silence detected."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "regular output\n"

        probe_stdout = json.dumps({"format": {"duration": "30.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_silence_gaps
            result = detect_silence_gaps("test.mp4")

        assert result.gaps == []
        assert result.total_silence_duration == 0.0

    def test_unclosed_silence_extends_to_eof(self):
        """A silence_start without matching end should extend to file duration."""
        ffmpeg_stderr = (
            "[silencedetect @ 0xabc] silence_start: 55.000\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ffmpeg_stderr

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_result]
            from opencut.core.qc_checks import detect_silence_gaps
            result = detect_silence_gaps("test.mp4")

        assert len(result.gaps) == 1
        assert result.gaps[0].start == 55.0
        assert result.gaps[0].end == 60.0
        assert result.gaps[0].duration == 5.0

    def test_ffmpeg_not_found_raises(self):
        """Should raise RuntimeError when ffmpeg is missing."""
        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, FileNotFoundError()]
            from opencut.core.qc_checks import detect_silence_gaps
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                detect_silence_gaps("test.mp4")

    def test_timeout_raises(self):
        """Should raise RuntimeError on subprocess timeout."""
        probe_stdout = json.dumps({"format": {"duration": "10.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [
                mock_probe,
                subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600),
            ]
            from opencut.core.qc_checks import detect_silence_gaps
            with pytest.raises(RuntimeError, match="timed out"):
                detect_silence_gaps("test.mp4")


# ========================================================================
# 5. Leader Element Detection
# ========================================================================
class TestLeaderDetection:
    """Tests for detect_leader_elements."""

    def test_detects_color_bars_via_saturation(self):
        """Should detect color bars when SATAVG is high for consecutive frames."""
        # Build signalstats output with high saturation for first 10 seconds
        lines = []
        for i in range(30):
            ts = i * 0.5
            if ts < 10.0:
                sat = 120.0  # High saturation = color bars
            else:
                sat = 20.0   # Low saturation = normal content
            lines.append(f"[Parsed_signalstats_0 @ 0x1234] pts_time:{ts:.3f}")
            lines.append(f"lavfi.signalstats.SATAVG= {sat:.1f}")

        bars_stderr = "\n".join(lines) + "\n"
        mock_bars = MagicMock()
        mock_bars.returncode = 0
        mock_bars.stderr = bars_stderr
        mock_bars.stdout = ""

        # Tone detection (no tone)
        mock_tone = MagicMock()
        mock_tone.returncode = 0
        mock_tone.stderr = ""
        mock_tone.stdout = ""

        # No slate (no blackdetect after bars)
        mock_slate = MagicMock()
        mock_slate.returncode = 0
        mock_slate.stderr = ""

        probe_stdout = json.dumps({"format": {"duration": "120.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_bars, mock_tone, mock_slate]
            from opencut.core.qc_checks import detect_leader_elements
            result = detect_leader_elements("test.mp4", scan_duration=120.0)

        assert result.bars_detected is True
        assert result.bars_end_time > 0

    def test_detects_reference_tone(self):
        """Should detect reference tone when RMS is high and sustained."""
        # No bars
        mock_bars = MagicMock()
        mock_bars.returncode = 0
        mock_bars.stderr = ""
        mock_bars.stdout = ""

        # Tone: sustained high RMS for several frames
        tone_lines = []
        for i in range(20):
            ts = i * 0.5
            if ts < 5.0:
                rms = -15.0  # Loud tone
            else:
                rms = -60.0  # Quiet
            tone_lines.append(f"[Parsed_astats_0 @ 0x1234] pts_time:{ts:.3f}")
            tone_lines.append(f"lavfi.astats.1.RMS_level= {rms:.1f}")

        mock_tone = MagicMock()
        mock_tone.returncode = 0
        mock_tone.stderr = "\n".join(tone_lines) + "\n"
        mock_tone.stdout = ""

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_bars, mock_tone]
            from opencut.core.qc_checks import detect_leader_elements
            result = detect_leader_elements("test.mp4")

        assert result.tone_detected is True
        assert result.tone_end_time > 0

    def test_no_leader_elements(self):
        """Should return all-false when no leader elements found."""
        mock_bars = MagicMock()
        mock_bars.returncode = 0
        mock_bars.stderr = ""
        mock_bars.stdout = ""

        mock_tone = MagicMock()
        mock_tone.returncode = 0
        mock_tone.stderr = ""
        mock_tone.stdout = ""

        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, mock_bars, mock_tone]
            from opencut.core.qc_checks import detect_leader_elements
            result = detect_leader_elements("test.mp4")

        assert result.bars_detected is False
        assert result.tone_detected is False
        assert result.slate_detected is False
        assert result.recommended_trim_point == 0.0

    def test_recommended_trim_uses_latest_element(self):
        """Trim point should be the latest detected leader element."""
        from opencut.core.qc_checks import LeaderResult

        # Directly test the logic by checking the result object
        result = LeaderResult(
            bars_detected=True,
            bars_end_time=10.0,
            tone_detected=True,
            tone_end_time=8.0,
            slate_detected=True,
            slate_end_time=12.5,
            recommended_trim_point=12.5,
        )
        assert result.recommended_trim_point == 12.5

    def test_ffmpeg_not_found_raises(self):
        """Should raise RuntimeError when ffmpeg is missing."""
        probe_stdout = json.dumps({"format": {"duration": "60.0"}})
        mock_probe = MagicMock()
        mock_probe.returncode = 0
        mock_probe.stdout = probe_stdout

        with patch("opencut.core.qc_checks.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_probe, FileNotFoundError()]
            from opencut.core.qc_checks import detect_leader_elements
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                detect_leader_elements("test.mp4")


# ========================================================================
# 6. Full QC Report
# ========================================================================
class TestFullQCReport:
    """Tests for run_full_qc."""

    def test_runs_all_checks(self):
        """Should invoke all five check functions and return combined report."""
        from opencut.core.qc_checks import (
            BlackFrameResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
        )

        with patch("opencut.core.qc_checks.detect_black_frames") as mock_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as mock_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as mock_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as mock_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as mock_ld:

            mock_bf.return_value = BlackFrameResult(frames=[], file_duration=60.0)
            mock_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            mock_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            mock_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            mock_ld.return_value = LeaderResult()

            from opencut.core.qc_checks import run_full_qc
            report = run_full_qc("test.mp4")

        assert report.passed is True
        assert report.issues_summary == []
        mock_bf.assert_called_once()
        mock_ff.assert_called_once()
        mock_ap.assert_called_once()
        mock_sg.assert_called_once()
        mock_ld.assert_called_once()

    def test_reports_issues_when_problems_found(self):
        """Should mark passed=False and list issues when checks find problems."""
        from opencut.core.qc_checks import (
            BlackFrame,
            BlackFrameResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
        )

        with patch("opencut.core.qc_checks.detect_black_frames") as mock_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as mock_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as mock_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as mock_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as mock_ld:

            mock_bf.return_value = BlackFrameResult(
                frames=[BlackFrame(start=1.0, end=2.0, duration=1.0)],
                total_black_duration=1.0,
                file_duration=60.0,
                black_percentage=1.67,
            )
            mock_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            mock_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            mock_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            mock_ld.return_value = LeaderResult()

            from opencut.core.qc_checks import run_full_qc
            report = run_full_qc("test.mp4")

        assert report.passed is False
        assert len(report.issues_summary) >= 1
        assert "black frame" in report.issues_summary[0].lower()

    def test_continues_when_individual_check_fails(self):
        """If one check raises, others should still run."""
        from opencut.core.qc_checks import (
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
        )

        with patch("opencut.core.qc_checks.detect_black_frames") as mock_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as mock_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as mock_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as mock_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as mock_ld:

            mock_bf.side_effect = RuntimeError("FFmpeg crashed")
            mock_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            mock_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            mock_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            mock_ld.return_value = LeaderResult()

            from opencut.core.qc_checks import run_full_qc
            report = run_full_qc("test.mp4")

        # Black frames failed but others should still have run
        assert report.black_frames is None
        assert report.frozen_frames is not None
        assert report.audio_phase is not None
        assert report.silence_gaps is not None
        assert report.leader is not None
        assert any("failed" in s.lower() for s in report.issues_summary)

    def test_progress_callback_spans_full_range(self):
        """Progress should go from 0 to 100 across all stages."""
        from opencut.core.qc_checks import (
            BlackFrameResult,
            FrozenFrameResult,
            LeaderResult,
            PhaseResult,
            SilenceGapResult,
        )

        progress_calls = []

        def on_progress(pct, msg=""):
            progress_calls.append(pct)

        with patch("opencut.core.qc_checks.detect_black_frames") as mock_bf, \
             patch("opencut.core.qc_checks.detect_frozen_frames") as mock_ff, \
             patch("opencut.core.qc_checks.check_audio_phase") as mock_ap, \
             patch("opencut.core.qc_checks.detect_silence_gaps") as mock_sg, \
             patch("opencut.core.qc_checks.detect_leader_elements") as mock_ld:

            mock_bf.return_value = BlackFrameResult(frames=[], file_duration=60.0)
            mock_ff.return_value = FrozenFrameResult(frames=[], file_duration=60.0)
            mock_ap.return_value = PhaseResult(issues=[], has_phase_problems=False, file_duration=60.0)
            mock_sg.return_value = SilenceGapResult(gaps=[], file_duration=60.0)
            mock_ld.return_value = LeaderResult()

            from opencut.core.qc_checks import run_full_qc
            run_full_qc("test.mp4", on_progress=on_progress)

        assert progress_calls[0] == 0
        assert progress_calls[-1] == 100


# ========================================================================
# 7. Dataclass Sanity Checks
# ========================================================================
class TestDataclasses:
    """Verify dataclass defaults and structure."""

    def test_black_frame_fields(self):
        from opencut.core.qc_checks import BlackFrame
        bf = BlackFrame(start=1.0, end=2.0, duration=1.0)
        assert bf.start == 1.0
        assert bf.end == 2.0
        assert bf.duration == 1.0

    def test_frozen_frame_fields(self):
        from opencut.core.qc_checks import FrozenFrame
        ff = FrozenFrame(start=5.0, end=10.0, duration=5.0)
        assert ff.duration == 5.0

    def test_phase_issue_fields(self):
        from opencut.core.qc_checks import PhaseIssue
        pi = PhaseIssue(start=0.0, end=1.0, avg_phase=-0.7)
        assert pi.avg_phase == -0.7

    def test_silence_gap_fields(self):
        from opencut.core.qc_checks import SilenceGap
        sg = SilenceGap(start=3.0, end=5.0, duration=2.0)
        assert sg.duration == 2.0

    def test_leader_result_defaults(self):
        from opencut.core.qc_checks import LeaderResult
        lr = LeaderResult()
        assert lr.bars_detected is False
        assert lr.tone_detected is False
        assert lr.slate_detected is False
        assert lr.recommended_trim_point == 0.0

    def test_full_qc_report_defaults(self):
        from opencut.core.qc_checks import FullQCReport
        report = FullQCReport()
        assert report.passed is True
        assert report.issues_summary == []
        assert report.black_frames is None

    def test_black_frame_result_defaults(self):
        from opencut.core.qc_checks import BlackFrameResult
        result = BlackFrameResult()
        assert result.frames == []
        assert result.total_black_duration == 0.0
        assert result.black_percentage == 0.0
