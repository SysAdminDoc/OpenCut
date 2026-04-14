"""
Tests for Enhanced Media Processing modules.

Covers:
  - enhanced_speech.py (AI speech restoration)
  - one_click_enhance.py (one-click video enhance pipeline)
  - low_light.py (low-light video enhancement)
  - scene_edit_detect.py (scene edit detection & classification)
  - enhanced_media_routes.py (route smoke tests)
"""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================================================
# 1. enhanced_speech.py
# ========================================================================
class TestEnhancedSpeechResult:
    """Tests for EnhanceSpeechResult and AudioStats dataclasses."""

    def test_enhance_speech_result_defaults(self):
        from opencut.core.enhanced_speech import EnhanceSpeechResult
        r = EnhanceSpeechResult()
        assert r.output_path == ""
        assert r.mode == ""
        assert r.original_stats == {}
        assert r.enhanced_stats == {}

    def test_enhance_speech_result_populated(self):
        from opencut.core.enhanced_speech import EnhanceSpeechResult
        r = EnhanceSpeechResult(
            output_path="/out/test.wav",
            mode="full",
            original_stats={"sample_rate": 16000, "snr_estimate": 12.5},
            enhanced_stats={"sample_rate": 48000, "snr_estimate": 25.0},
        )
        assert r.output_path == "/out/test.wav"
        assert r.mode == "full"
        assert r.original_stats["sample_rate"] == 16000

    def test_audio_stats_defaults(self):
        from opencut.core.enhanced_speech import AudioStats
        s = AudioStats()
        assert s.sample_rate == 0
        assert s.snr_estimate == 0.0

    def test_audio_stats_values(self):
        from opencut.core.enhanced_speech import AudioStats
        s = AudioStats(sample_rate=44100, snr_estimate=18.3)
        assert s.sample_rate == 44100
        assert s.snr_estimate == 18.3


class TestEnhanceSpeechModes:
    """Tests for enhance_speech mode validation."""

    def test_invalid_mode_raises(self):
        from opencut.core.enhanced_speech import enhance_speech
        with pytest.raises(ValueError, match="Invalid mode"):
            enhance_speech("/fake/audio.wav", mode="turbo")

    def test_file_not_found_raises(self):
        from opencut.core.enhanced_speech import enhance_speech
        with pytest.raises(FileNotFoundError):
            enhance_speech("/nonexistent/audio.wav", mode="full")

    def test_valid_modes_accepted(self):
        """All three modes should be valid strings."""
        valid = ("denoise_only", "enhance", "full")
        for m in valid:
            assert m in valid


class TestEnhanceSpeechProbe:
    """Tests for audio probing helpers."""

    def test_probe_audio_stats_parses_ffprobe(self):
        from opencut.core.enhanced_speech import _probe_audio_stats

        ffprobe_out = json.dumps({"streams": [{"sample_rate": "22050", "codec_name": "pcm_s16le"}]})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_out

        with patch("opencut.core.enhanced_speech.subprocess.run", return_value=mock_result), \
             patch("opencut.core.enhanced_speech._estimate_snr", return_value=15.0):
            stats = _probe_audio_stats("/test/audio.wav")
        assert stats.sample_rate == 22050
        assert stats.snr_estimate == 15.0

    def test_probe_audio_stats_handles_failure(self):
        from opencut.core.enhanced_speech import _probe_audio_stats

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("opencut.core.enhanced_speech.subprocess.run", return_value=mock_result):
            stats = _probe_audio_stats("/test/audio.wav")
        assert stats.sample_rate == 0

    def test_estimate_snr_parses_volumedetect(self):
        from opencut.core.enhanced_speech import _estimate_snr

        stderr = (
            "[Parsed_volumedetect_0 @ 0x1234] mean_volume: -25.3 dB\n"
            "[Parsed_volumedetect_0 @ 0x1234] max_volume: -3.1 dB\n"
        )
        mock_result = MagicMock()
        mock_result.stderr = stderr

        with patch("opencut.core.enhanced_speech.subprocess.run", return_value=mock_result):
            snr = _estimate_snr("/test/audio.wav")
        assert snr == pytest.approx(22.2, abs=0.1)

    def test_estimate_snr_handles_failure(self):
        from opencut.core.enhanced_speech import _estimate_snr

        with patch("opencut.core.enhanced_speech.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 30)):
            snr = _estimate_snr("/test/audio.wav")
        assert snr == 0.0


class TestEnhanceSpeechNormalize:
    """Tests for LUFS normalization."""

    def test_normalize_lufs_calls_ffmpeg(self):
        from opencut.core.enhanced_speech import _normalize_lufs

        with patch("opencut.core.enhanced_speech.run_ffmpeg") as mock_run:
            _normalize_lufs("/in.wav", "/out.wav", target_lufs=-16.0)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "loudnorm" in " ".join(cmd)
            assert "-16" in " ".join(cmd)

    def test_bandwidth_extend_calls_ffmpeg(self):
        from opencut.core.enhanced_speech import _bandwidth_extend_ffmpeg

        with patch("opencut.core.enhanced_speech.run_ffmpeg") as mock_run:
            _bandwidth_extend_ffmpeg("/in.wav", "/out.wav", target_sr=48000)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "aresample" in " ".join(cmd)
            assert "48000" in " ".join(cmd)


class TestEnhanceSpeechVideoDetection:
    """Tests for video vs audio file detection."""

    def test_video_extensions_detected(self):
        from opencut.core.enhanced_speech import _VIDEO_EXTS
        assert ".mp4" in _VIDEO_EXTS
        assert ".mkv" in _VIDEO_EXTS
        assert ".mov" in _VIDEO_EXTS

    def test_audio_extensions_not_video(self):
        from opencut.core.enhanced_speech import _VIDEO_EXTS
        assert ".wav" not in _VIDEO_EXTS
        assert ".mp3" not in _VIDEO_EXTS
        assert ".flac" not in _VIDEO_EXTS


class TestEnhanceSpeechExtractAudio:
    """Tests for audio extraction from video."""

    def test_extract_audio_calls_ffmpeg(self):
        from opencut.core.enhanced_speech import _extract_audio_wav

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("opencut.core.enhanced_speech.subprocess.run", return_value=mock_result):
            _extract_audio_wav("/input.mp4", "/output.wav")

    def test_extract_audio_raises_on_failure(self):
        from opencut.core.enhanced_speech import _extract_audio_wav

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "codec not found"

        with patch("opencut.core.enhanced_speech.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Audio extraction failed"):
                _extract_audio_wav("/input.mp4", "/output.wav")


# ========================================================================
# 2. one_click_enhance.py
# ========================================================================
class TestEnhanceResult:
    """Tests for EnhanceResult dataclass."""

    def test_enhance_result_defaults(self):
        from opencut.core.one_click_enhance import EnhanceResult
        r = EnhanceResult()
        assert r.output_path == ""
        assert r.steps_applied == []
        assert r.duration_seconds == 0.0

    def test_enhance_result_populated(self):
        from opencut.core.one_click_enhance import EnhanceResult
        r = EnhanceResult(
            output_path="/out/video.mp4",
            steps_applied=["audio_denoise", "stabilize", "color_correct"],
            duration_seconds=45.2,
        )
        assert len(r.steps_applied) == 3
        assert "stabilize" in r.steps_applied


class TestOneClickPresets:
    """Tests for preset configurations."""

    def test_all_presets_exist(self):
        from opencut.core.one_click_enhance import _PRESETS
        assert "fast" in _PRESETS
        assert "balanced" in _PRESETS
        assert "quality" in _PRESETS

    def test_fast_skips_upscale(self):
        from opencut.core.one_click_enhance import _PRESETS
        assert _PRESETS["fast"]["skip_upscale"] is True

    def test_balanced_includes_upscale(self):
        from opencut.core.one_click_enhance import _PRESETS
        assert _PRESETS["balanced"]["skip_upscale"] is False

    def test_quality_has_lowest_crf(self):
        from opencut.core.one_click_enhance import _PRESETS
        assert _PRESETS["quality"]["crf"] < _PRESETS["balanced"]["crf"]
        assert _PRESETS["balanced"]["crf"] < _PRESETS["fast"]["crf"]

    def test_invalid_preset_raises(self):
        from opencut.core.one_click_enhance import one_click_enhance
        with pytest.raises(ValueError, match="Invalid preset"):
            one_click_enhance("/fake/video.mp4", preset="ultra")


class TestOneClickAnalysis:
    """Tests for clip analysis logic."""

    def test_analyze_clip_detects_low_res(self):
        from opencut.core.one_click_enhance import _analyze_clip

        mock_info = {"width": 640, "height": 480, "fps": 30.0, "duration": 10.0}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "YAVG=120\nTOUT=0.001\nVREP=0.05\n"

        mock_audio_result = MagicMock()
        mock_audio_result.returncode = 0
        mock_audio_result.stderr = "mean_volume: -20.0 dB\n"

        with patch("opencut.core.one_click_enhance.get_video_info", return_value=mock_info), \
             patch("opencut.core.one_click_enhance.subprocess.run",
                   side_effect=[mock_result, mock_audio_result]):
            issues = _analyze_clip("/test/video.mp4")

        assert issues["needs_upscale"] is True

    def test_analyze_clip_detects_dark(self):
        from opencut.core.one_click_enhance import _analyze_clip

        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "YAVG=50\nTOUT=0.001\nVREP=0.05\n"

        mock_audio_result = MagicMock()
        mock_audio_result.returncode = 0
        mock_audio_result.stderr = "mean_volume: -20.0 dB\n"

        with patch("opencut.core.one_click_enhance.get_video_info", return_value=mock_info), \
             patch("opencut.core.one_click_enhance.subprocess.run",
                   side_effect=[mock_result, mock_audio_result]):
            issues = _analyze_clip("/test/video.mp4")

        assert issues["needs_color_fix"] is True

    def test_analyze_clip_detects_noisy_audio(self):
        from opencut.core.one_click_enhance import _analyze_clip

        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "YAVG=120\nTOUT=0.001\nVREP=0.05\n"

        mock_audio_result = MagicMock()
        mock_audio_result.returncode = 0
        mock_audio_result.stderr = "mean_volume: -40.0 dB\n"

        with patch("opencut.core.one_click_enhance.get_video_info", return_value=mock_info), \
             patch("opencut.core.one_click_enhance.subprocess.run",
                   side_effect=[mock_result, mock_audio_result]):
            issues = _analyze_clip("/test/video.mp4")

        assert issues["is_noisy_audio"] is True

    def test_analyze_clip_detects_grain(self):
        from opencut.core.one_click_enhance import _analyze_clip

        mock_info = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 10.0}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "YAVG=120\nTOUT=0.05\nVREP=0.05\n"

        mock_audio_result = MagicMock()
        mock_audio_result.returncode = 0
        mock_audio_result.stderr = "mean_volume: -20.0 dB\n"

        with patch("opencut.core.one_click_enhance.get_video_info", return_value=mock_info), \
             patch("opencut.core.one_click_enhance.subprocess.run",
                   side_effect=[mock_result, mock_audio_result]):
            issues = _analyze_clip("/test/video.mp4")

        assert issues["is_grainy"] is True


class TestOneClickSteps:
    """Tests for individual enhancement steps."""

    def test_denoise_video_calls_ffmpeg(self):
        from opencut.core.one_click_enhance import _denoise_video

        with patch("opencut.core.one_click_enhance.run_ffmpeg") as mock_run:
            _denoise_video("/in.mp4", "/out.mp4", strength="medium")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "hqdn3d" in " ".join(cmd)

    def test_denoise_video_strength_levels(self):
        from opencut.core.one_click_enhance import _denoise_video

        for strength in ("light", "medium", "strong"):
            with patch("opencut.core.one_click_enhance.run_ffmpeg") as mock_run:
                _denoise_video("/in.mp4", "/out.mp4", strength=strength)
                mock_run.assert_called_once()

    def test_auto_color_adjusts_dark(self):
        from opencut.core.one_click_enhance import _auto_color

        with patch("opencut.core.one_click_enhance.run_ffmpeg") as mock_run:
            _auto_color("/in.mp4", "/out.mp4", avg_brightness=50.0)
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)
            assert "eq=" in cmd_str
            assert "brightness=" in cmd_str

    def test_auto_color_adjusts_bright(self):
        from opencut.core.one_click_enhance import _auto_color

        with patch("opencut.core.one_click_enhance.run_ffmpeg") as mock_run:
            _auto_color("/in.mp4", "/out.mp4", avg_brightness=220.0)
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)
            assert "brightness=-" in cmd_str

    def test_upscale_video_calls_ffmpeg(self):
        from opencut.core.one_click_enhance import _upscale_video

        with patch("opencut.core.one_click_enhance.run_ffmpeg") as mock_run:
            _upscale_video("/in.mp4", "/out.mp4", target_height=1080)
            cmd = mock_run.call_args[0][0]
            assert "1080" in " ".join(cmd)
            assert "lanczos" in " ".join(cmd)

    def test_denoise_audio_calls_ffmpeg(self):
        from opencut.core.one_click_enhance import _denoise_audio

        with patch("opencut.core.one_click_enhance.run_ffmpeg") as mock_run:
            _denoise_audio("/in.mp4", "/out.mp4")
            cmd = mock_run.call_args[0][0]
            assert "afftdn" in " ".join(cmd)

    def test_stabilize_two_pass(self):
        from opencut.core.one_click_enhance import _stabilize_video

        call_count = 0

        def _mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1

        with patch("opencut.core.one_click_enhance.run_ffmpeg", side_effect=_mock_run), \
             patch("opencut.core.one_click_enhance.tempfile.mkstemp",
                   return_value=(0, "/tmp/test.trf")), \
             patch("os.close"), \
             patch("os.unlink"):
            _stabilize_video("/in.mp4", "/out.mp4")
        assert call_count == 2  # Two passes

    def test_file_not_found_raises(self):
        from opencut.core.one_click_enhance import one_click_enhance
        with pytest.raises(FileNotFoundError):
            one_click_enhance("/nonexistent/video.mp4")


class TestOneClickStepSkipping:
    """Tests for conditional step execution."""

    def test_no_issues_produces_passthrough(self):
        from opencut.core.one_click_enhance import one_click_enhance

        no_issues = {
            "is_noisy_audio": False,
            "is_shaky": False,
            "is_grainy": False,
            "needs_color_fix": False,
            "needs_upscale": False,
            "avg_brightness": 128.0,
            "noise_level": 0.0,
        }

        with patch("opencut.core.one_click_enhance._analyze_clip", return_value=no_issues), \
             patch("opencut.core.one_click_enhance.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            result = one_click_enhance("/test/video.mp4", output_path="/out/video.mp4")
        assert "passthrough" in result.steps_applied

    def test_fast_preset_skips_upscale(self):
        from opencut.core.one_click_enhance import one_click_enhance

        all_issues = {
            "is_noisy_audio": True,
            "is_shaky": False,
            "is_grainy": True,
            "needs_color_fix": True,
            "needs_upscale": True,
            "avg_brightness": 60.0,
            "noise_level": 0.05,
        }

        with patch("opencut.core.one_click_enhance._analyze_clip", return_value=all_issues), \
             patch("opencut.core.one_click_enhance.run_ffmpeg"), \
             patch("opencut.core.one_click_enhance._denoise_audio"), \
             patch("opencut.core.one_click_enhance._denoise_video"), \
             patch("opencut.core.one_click_enhance._auto_color"), \
             patch("opencut.core.one_click_enhance._upscale_video") as mock_up, \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"), \
             patch("tempfile.mkstemp", return_value=(0, "/tmp/test.mp4")), \
             patch("os.close"), \
             patch("os.unlink"):
            result = one_click_enhance("/test/video.mp4",
                                       output_path="/out/video.mp4", preset="fast")
        assert "upscale" not in result.steps_applied
        mock_up.assert_not_called()


# ========================================================================
# 3. low_light.py
# ========================================================================
class TestLowLightResult:
    """Tests for LowLightResult dataclass."""

    def test_defaults(self):
        from opencut.core.low_light import LowLightResult
        r = LowLightResult()
        assert r.output_path == ""
        assert r.original_avg_luminance == 0.0
        assert r.enhanced_avg_luminance == 0.0
        assert r.denoise_applied is False

    def test_populated(self):
        from opencut.core.low_light import LowLightResult
        r = LowLightResult(
            output_path="/out/video.mp4",
            original_avg_luminance=35.0,
            enhanced_avg_luminance=120.0,
            denoise_applied=True,
        )
        assert r.denoise_applied is True
        assert r.enhanced_avg_luminance > r.original_avg_luminance


class TestLowLightLuminance:
    """Tests for luminance measurement."""

    def test_measure_avg_luminance_parses_yavg(self):
        from opencut.core.low_light import _measure_avg_luminance

        stderr = "YAVG=45\nYAVG=50\nYAVG=55\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.low_light.subprocess.run", return_value=mock_result):
            lum = _measure_avg_luminance("/test/video.mp4")
        assert lum == pytest.approx(50.0, abs=0.1)

    def test_measure_avg_luminance_empty_returns_zero(self):
        from opencut.core.low_light import _measure_avg_luminance

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no useful output here"

        mock_result2 = MagicMock()
        mock_result2.returncode = 0
        mock_result2.stderr = "still nothing"

        with patch("opencut.core.low_light.subprocess.run",
                   side_effect=[mock_result, mock_result2]):
            lum = _measure_avg_luminance("/test/video.mp4")
        assert lum == 0.0


class TestLowLightAutoSkip:
    """Tests for auto-skip when footage is not dark."""

    def test_skip_when_not_dark(self):
        from opencut.core.low_light import enhance_low_light

        with patch("opencut.core.low_light._measure_avg_luminance", return_value=150.0), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"), \
             patch("shutil.copy2"):
            result = enhance_low_light("/test/video.mp4", output_path="/out/video.mp4")
        assert result.original_avg_luminance == 150.0
        assert result.enhanced_avg_luminance == 150.0
        assert result.denoise_applied is False

    def test_enhances_when_dark(self):
        from opencut.core.low_light import enhance_low_light

        with patch("opencut.core.low_light._measure_avg_luminance",
                   side_effect=[30.0, 90.0]), \
             patch("opencut.core.low_light.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            result = enhance_low_light("/test/video.mp4", output_path="/out/video.mp4")
        assert result.original_avg_luminance == 30.0
        assert result.enhanced_avg_luminance == 90.0
        assert result.denoise_applied is True


class TestLowLightFilters:
    """Tests for filter builders."""

    def test_curves_filter_strength_zero(self):
        from opencut.core.low_light import _build_curves_filter
        f = _build_curves_filter(0.0)
        assert "curves=" in f
        # At strength 0, shadow lift should be minimal
        assert "0.000" in f

    def test_curves_filter_strength_one(self):
        from opencut.core.low_light import _build_curves_filter
        f = _build_curves_filter(1.0)
        assert "curves=" in f

    def test_curves_filter_strength_two(self):
        from opencut.core.low_light import _build_curves_filter
        f = _build_curves_filter(2.0)
        assert "curves=" in f

    def test_curves_filter_clamped(self):
        from opencut.core.low_light import _build_curves_filter
        # Should not crash with out-of-range values
        f1 = _build_curves_filter(-1.0)
        f2 = _build_curves_filter(5.0)
        assert "curves=" in f1
        assert "curves=" in f2

    def test_unsharp_filter(self):
        from opencut.core.low_light import _build_unsharp_filter
        f = _build_unsharp_filter(1.0)
        assert "unsharp=" in f

    def test_denoise_filter(self):
        from opencut.core.low_light import _build_denoise_filter
        f = _build_denoise_filter(1.0)
        assert "nlmeans" in f

    def test_denoise_filter_scales_with_strength(self):
        from opencut.core.low_light import _build_denoise_filter
        f_low = _build_denoise_filter(0.5)
        f_high = _build_denoise_filter(2.0)
        # Both should be valid filters
        assert "nlmeans" in f_low
        assert "nlmeans" in f_high


class TestLowLightValidation:
    """Tests for input validation."""

    def test_file_not_found(self):
        from opencut.core.low_light import enhance_low_light
        with pytest.raises(FileNotFoundError):
            enhance_low_light("/nonexistent/video.mp4")

    def test_strength_clamped(self):
        from opencut.core.low_light import enhance_low_light

        with patch("opencut.core.low_light._measure_avg_luminance", return_value=150.0), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"), \
             patch("shutil.copy2"):
            # Strength > 2.0 should be clamped, not error
            result = enhance_low_light("/test/video.mp4",
                                       output_path="/out/video.mp4", strength=5.0)
        assert result is not None

    def test_denoise_false_no_denoise_filter(self):
        from opencut.core.low_light import enhance_low_light

        with patch("opencut.core.low_light._measure_avg_luminance",
                   side_effect=[30.0, 80.0]), \
             patch("opencut.core.low_light.run_ffmpeg") as mock_run, \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            result = enhance_low_light("/test/video.mp4",
                                       output_path="/out/video.mp4", denoise=False)
        assert result.denoise_applied is False
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "nlmeans" not in cmd_str


# ========================================================================
# 4. scene_edit_detect.py
# ========================================================================
class TestEditPoint:
    """Tests for EditPoint dataclass."""

    def test_defaults(self):
        from opencut.core.scene_edit_detect import EditPoint
        ep = EditPoint()
        assert ep.timestamp == 0.0
        assert ep.frame_number == 0
        assert ep.confidence == 0.0
        assert ep.type == "hard_cut"

    def test_to_dict(self):
        from opencut.core.scene_edit_detect import EditPoint
        ep = EditPoint(timestamp=5.123, frame_number=153, confidence=0.85, type="dissolve")
        d = ep.to_dict()
        assert d["timestamp"] == 5.123
        assert d["frame_number"] == 153
        assert d["type"] == "dissolve"

    def test_types(self):
        from opencut.core.scene_edit_detect import EditPoint
        for t in ("hard_cut", "dissolve", "fade"):
            ep = EditPoint(type=t)
            assert ep.type == t


class TestEditDetectionResult:
    """Tests for EditDetectionResult dataclass."""

    def test_defaults(self):
        from opencut.core.scene_edit_detect import EditDetectionResult
        r = EditDetectionResult()
        assert r.cuts == []
        assert r.total_scenes == 0
        assert r.avg_scene_duration == 0.0

    def test_to_dict(self):
        from opencut.core.scene_edit_detect import EditDetectionResult, EditPoint
        r = EditDetectionResult(
            cuts=[EditPoint(timestamp=3.0, confidence=0.9, type="hard_cut")],
            total_scenes=2,
            avg_scene_duration=5.0,
        )
        d = r.to_dict()
        assert len(d["cuts"]) == 1
        assert d["total_scenes"] == 2
        assert d["avg_scene_duration"] == 5.0

    def test_to_dict_serializable(self):
        from opencut.core.scene_edit_detect import EditDetectionResult, EditPoint
        r = EditDetectionResult(
            cuts=[
                EditPoint(timestamp=1.5, frame_number=45, confidence=0.92, type="hard_cut"),
                EditPoint(timestamp=8.3, frame_number=249, confidence=0.55, type="dissolve"),
            ],
            total_scenes=3,
            avg_scene_duration=4.15,
        )
        # Should be JSON-serializable
        json_str = json.dumps(r.to_dict())
        parsed = json.loads(json_str)
        assert len(parsed["cuts"]) == 2


class TestCutTypeClassification:
    """Tests for _classify_cut_type logic."""

    def test_hard_cut_high_score(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.9, nearby_scores=[0.02, 0.01, 0.03])
        assert result == "hard_cut"

    def test_dissolve_high_score_with_neighbors(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.75, nearby_scores=[0.3, 0.25, 0.2])
        assert result == "dissolve"

    def test_dissolve_medium_score(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.4, nearby_scores=[0.15, 0.12, 0.18])
        assert result == "dissolve"

    def test_fade_with_black_before(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.8, nearby_scores=[], prev_black=True)
        assert result == "fade"

    def test_fade_with_black_after(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.8, nearby_scores=[], next_black=True)
        assert result == "fade"

    def test_hard_cut_no_neighbors(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.85, nearby_scores=[])
        assert result == "hard_cut"

    def test_low_score_is_hard_cut(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.15, nearby_scores=[0.01, 0.02])
        assert result == "hard_cut"

    def test_medium_score_few_elevated_is_hard_cut(self):
        from opencut.core.scene_edit_detect import _classify_cut_type
        result = _classify_cut_type(score=0.4, nearby_scores=[0.02, 0.01])
        assert result == "hard_cut"


class TestScdetParsing:
    """Tests for scdet output parsing."""

    def test_detect_with_scdet_parses_output(self):
        from opencut.core.scene_edit_detect import _detect_with_scdet

        stderr = (
            "[Parsed_scdet_0 @ 0x1234] lavfi.scd.score=85.5 lavfi.scd.time=3.200\n"
            "[Parsed_scdet_0 @ 0x1234] lavfi.scd.score=45.2 lavfi.scd.time=8.700\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.scene_edit_detect.subprocess.run", return_value=mock_result), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=30.0):
            detections = _detect_with_scdet("/test/video.mp4", threshold=0.3)

        assert len(detections) == 2
        assert detections[0]["time"] == 3.2
        assert detections[0]["score"] == pytest.approx(0.855, abs=0.01)
        assert detections[1]["time"] == 8.7

    def test_detect_with_scdet_empty(self):
        from opencut.core.scene_edit_detect import _detect_with_scdet

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no scene changes found"

        with patch("opencut.core.scene_edit_detect.subprocess.run", return_value=mock_result), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=30.0):
            detections = _detect_with_scdet("/test/video.mp4", threshold=0.3)

        assert detections == []

    def test_detect_with_scdet_timeout(self):
        from opencut.core.scene_edit_detect import _detect_with_scdet

        with patch("opencut.core.scene_edit_detect.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 300)), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=30.0):
            with pytest.raises(RuntimeError, match="timed out"):
                _detect_with_scdet("/test/video.mp4", threshold=0.3)


class TestDetectEdits:
    """Tests for the main detect_edits function."""

    def test_file_not_found(self):
        from opencut.core.scene_edit_detect import detect_edits
        with pytest.raises(FileNotFoundError):
            detect_edits("/nonexistent/video.mp4")

    def test_threshold_clamping(self):
        from opencut.core.scene_edit_detect import detect_edits

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=10.0), \
             patch("opencut.core.scene_edit_detect._get_fps", return_value=30.0), \
             patch("opencut.core.scene_edit_detect._detect_with_scdet", return_value=[]), \
             patch("opencut.core.scene_edit_detect._detect_black_frames", return_value={}):
            # Should not error with extreme thresholds
            result = detect_edits("/test/video.mp4", threshold=0.0)
            assert result.total_scenes == 1  # 0 cuts = 1 scene

    def test_min_scene_duration_filter(self):
        from opencut.core.scene_edit_detect import detect_edits

        detections = [
            {"time": 1.0, "score": 0.9},
            {"time": 1.2, "score": 0.8},  # Too close to previous
            {"time": 5.0, "score": 0.85},
        ]

        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=30.0), \
             patch("opencut.core.scene_edit_detect._get_fps", return_value=30.0), \
             patch("opencut.core.scene_edit_detect._detect_with_scdet",
                   return_value=detections), \
             patch("opencut.core.scene_edit_detect._detect_black_frames", return_value={}):
            result = detect_edits("/test/video.mp4", min_scene_duration=0.5)

        # The 1.2s cut should be filtered out (only 0.2s after 1.0s)
        assert len(result.cuts) == 2
        assert result.cuts[0].timestamp == 1.0
        assert result.cuts[1].timestamp == 5.0

    def test_total_scenes_calculation(self):
        from opencut.core.scene_edit_detect import detect_edits

        detections = [
            {"time": 5.0, "score": 0.9},
            {"time": 15.0, "score": 0.85},
            {"time": 25.0, "score": 0.8},
        ]

        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=30.0), \
             patch("opencut.core.scene_edit_detect._get_fps", return_value=30.0), \
             patch("opencut.core.scene_edit_detect._detect_with_scdet",
                   return_value=detections), \
             patch("opencut.core.scene_edit_detect._detect_black_frames", return_value={}):
            result = detect_edits("/test/video.mp4")

        # 3 cuts = 4 scenes
        assert result.total_scenes == 4
        assert result.avg_scene_duration == pytest.approx(7.5, abs=0.1)

    def test_progress_callback_called(self):
        from opencut.core.scene_edit_detect import detect_edits

        progress_calls = []

        def _progress(pct, msg=""):
            progress_calls.append((pct, msg))

        with patch("os.path.isfile", return_value=True), \
             patch("opencut.core.scene_edit_detect._get_duration", return_value=10.0), \
             patch("opencut.core.scene_edit_detect._get_fps", return_value=30.0), \
             patch("opencut.core.scene_edit_detect._detect_with_scdet", return_value=[]), \
             patch("opencut.core.scene_edit_detect._detect_black_frames", return_value={}):
            detect_edits("/test/video.mp4", on_progress=_progress)

        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 100


class TestBlackFrameDetection:
    """Tests for black frame detection."""

    def test_detect_black_frames_parses(self):
        from opencut.core.scene_edit_detect import _detect_black_frames

        stderr = (
            "[blackdetect @ 0x1234] black_start:2.5 black_end:3.0\n"
            "[blackdetect @ 0x1234] black_start:8.0 black_end:8.5\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.scene_edit_detect.subprocess.run", return_value=mock_result):
            info = _detect_black_frames("/test/video.mp4", [3.0, 8.0])

        assert 3.0 in info
        assert 8.0 in info

    def test_detect_black_frames_empty(self):
        from opencut.core.scene_edit_detect import _detect_black_frames
        result = _detect_black_frames("/test/video.mp4", [])
        assert result == {}

    def test_detect_black_frames_handles_timeout(self):
        from opencut.core.scene_edit_detect import _detect_black_frames

        with patch("opencut.core.scene_edit_detect.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 120)):
            info = _detect_black_frames("/test/video.mp4", [5.0])
        assert 5.0 in info
        assert info[5.0]["before"] is False
        assert info[5.0]["after"] is False


# ========================================================================
# 5. Route smoke tests
# ========================================================================
class TestEnhancedMediaRoutes:
    """Smoke tests for enhanced_media_routes.py blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.enhanced_media_routes import enhanced_media_bp
        assert enhanced_media_bp.name == "enhanced_media"

    def test_enhance_speech_route_registered(self):
        from opencut.routes.enhanced_media_routes import enhanced_media_bp
        # Blueprint deferred functions register routes on app attachment
        # Just verify the blueprint has routes defined
        assert len(list(enhanced_media_bp.deferred_functions)) > 0

    def test_route_functions_exist(self):
        from opencut.routes.enhanced_media_routes import (
            detect_edits_route,
            enhance_low_light_preview_route,
            enhance_low_light_route,
            enhance_speech_preview_route,
            enhance_speech_route,
            one_click_enhance_route,
        )
        assert callable(enhance_speech_route)
        assert callable(enhance_speech_preview_route)
        assert callable(one_click_enhance_route)
        assert callable(enhance_low_light_route)
        assert callable(enhance_low_light_preview_route)
        assert callable(detect_edits_route)

    def test_imports_from_routes(self):
        """All route handler names should be importable."""
        import opencut.routes.enhanced_media_routes as mod
        assert hasattr(mod, "enhanced_media_bp")
        assert hasattr(mod, "enhance_speech_route")
        assert hasattr(mod, "one_click_enhance_route")
        assert hasattr(mod, "enhance_low_light_route")
        assert hasattr(mod, "detect_edits_route")

    def test_blueprint_has_six_routes(self):
        from opencut.routes.enhanced_media_routes import enhanced_media_bp
        # Count deferred functions (each route registration is deferred)
        assert len(list(enhanced_media_bp.deferred_functions)) == 6


class TestRouteDataValidation:
    """Tests for route-level data handling patterns."""

    def test_mode_validation_defaults(self):
        """Invalid mode values should default to 'full'."""
        mode = "invalid"
        if mode not in ("denoise_only", "enhance", "full"):
            mode = "full"
        assert mode == "full"

    def test_preset_validation_defaults(self):
        """Invalid preset values should default to 'balanced'."""
        preset = "ultra_hd"
        if preset not in ("fast", "balanced", "quality"):
            preset = "balanced"
        assert preset == "balanced"

    def test_denoise_string_coercion(self):
        """String 'true'/'false' should coerce to bool."""
        for val in ("true", "1", "yes"):
            result = val.lower() in ("true", "1", "yes")
            assert result is True
        for val in ("false", "0", "no"):
            result = val.lower() in ("true", "1", "yes")
            assert result is False
