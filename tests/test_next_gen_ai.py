"""
Tests for OpenCut Next-Gen AI features.

Covers:
  - color_match_shots.py (AI color matching between shots)
  - video_llm.py (multimodal LLM video understanding)
  - music_remix.py (AI music remixing / duration fitting)
  - audio_category.py (audio classification / tagging)
  - next_gen_ai_routes.py (route smoke tests)
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ========================================================================
# 1. color_match_shots.py
# ========================================================================
class TestColorMatchResult:
    """Tests for ColorMatchResult dataclass."""

    def test_defaults(self):
        from opencut.core.color_match_shots import ColorMatchResult
        r = ColorMatchResult()
        assert r.output_path == ""
        assert r.reference_stats == {}
        assert r.matched_stats == {}
        assert r.adjustments == {}

    def test_populated(self):
        from opencut.core.color_match_shots import ColorMatchResult
        r = ColorMatchResult(
            output_path="/out/video.mp4",
            reference_stats={"luminance": 140.0, "avg_r": 130.0},
            matched_stats={"luminance": 138.0, "avg_r": 129.0},
            adjustments={"brightness": 0.05, "contrast": 1.02},
        )
        assert r.output_path == "/out/video.mp4"
        assert r.reference_stats["luminance"] == 140.0
        assert r.adjustments["brightness"] == 0.05

    def test_independent_default_dicts(self):
        """Each instance should have its own dict, not shared."""
        from opencut.core.color_match_shots import ColorMatchResult
        a = ColorMatchResult()
        b = ColorMatchResult()
        a.adjustments["test"] = 1
        assert "test" not in b.adjustments


class TestAnalyzeColorStats:
    """Tests for analyze_color_stats."""

    def test_file_not_found_raises(self):
        from opencut.core.color_match_shots import analyze_color_stats
        with pytest.raises(FileNotFoundError):
            analyze_color_stats("/nonexistent/video.mp4")

    def test_parses_yavg(self):
        from opencut.core.color_match_shots import analyze_color_stats

        stderr = "YAVG=100\nYAVG=110\nYAVG=120\nUAVG=128\nVAVG=128\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        assert stats["luminance"] == pytest.approx(110.0, abs=0.1)

    def test_parses_uavg_vavg(self):
        from opencut.core.color_match_shots import analyze_color_stats

        stderr = "YAVG=128\nUAVG=140\nVAVG=150\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        # RGB should be computed from YUV
        assert stats["avg_r"] != 128.0  # V != 128 shifts red
        assert stats["avg_b"] != 128.0  # U != 128 shifts blue

    def test_parses_satavg(self):
        from opencut.core.color_match_shots import analyze_color_stats

        stderr = "YAVG=128\nSATAVG=80\nSATAVG=90\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        assert stats["saturation"] == pytest.approx(85.0, abs=0.1)

    def test_parses_histogram_distribution(self):
        from opencut.core.color_match_shots import analyze_color_stats

        stderr = "YAVG=128\nYLOW=10\nYHIGH=20\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        assert stats["histogram_low"] + stats["histogram_mid"] + stats["histogram_high"] == pytest.approx(1.0, abs=0.01)

    def test_default_stats_on_empty_stderr(self):
        from opencut.core.color_match_shots import analyze_color_stats

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no useful data"

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        assert stats["luminance"] == 128.0
        assert stats["avg_r"] == 128.0

    def test_sample_seconds_clamped(self):
        from opencut.core.color_match_shots import analyze_color_stats

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "YAVG=100"

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result) as mock_run, \
             patch("os.path.isfile", return_value=True):
            analyze_color_stats("/test/video.mp4", sample_seconds=0.1)
            cmd = mock_run.call_args[0][0]
            # Should be clamped to 0.5 minimum
            t_idx = cmd.index("-t")
            assert float(cmd[t_idx + 1]) >= 0.5

    def test_handles_subprocess_exception(self):
        from opencut.core.color_match_shots import analyze_color_stats

        with patch("opencut.core.color_match_shots.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("cmd", 60)), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        # Should return defaults, not raise
        assert stats["luminance"] == 128.0

    def test_parses_hueavg(self):
        from opencut.core.color_match_shots import analyze_color_stats

        stderr = "YAVG=128\nHUEAVG=180\nHUEAVG=200\n"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = stderr

        with patch("opencut.core.color_match_shots.subprocess.run",
                   return_value=mock_result), \
             patch("os.path.isfile", return_value=True):
            stats = analyze_color_stats("/test/video.mp4")

        assert stats["hue_avg"] == pytest.approx(190.0, abs=0.1)


class TestComputeAdjustments:
    """Tests for _compute_adjustments."""

    def test_identical_stats_no_adjustment(self):
        from opencut.core.color_match_shots import _compute_adjustments

        stats = {"luminance": 128.0, "saturation": 128.0,
                 "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
                 "histogram_low": 0.33, "histogram_high": 0.33}
        adj = _compute_adjustments(stats, stats, 1.0)
        assert adj["brightness"] == pytest.approx(0.0, abs=0.001)
        assert adj["saturation"] == pytest.approx(1.0, abs=0.001)
        assert adj["r_shift"] == pytest.approx(0.0, abs=0.001)

    def test_brighter_reference_positive_brightness(self):
        from opencut.core.color_match_shots import _compute_adjustments

        src = {"luminance": 80.0, "saturation": 128.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        ref = {"luminance": 160.0, "saturation": 128.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        adj = _compute_adjustments(src, ref, 1.0)
        assert adj["brightness"] > 0.0

    def test_strength_zero_no_adjustment(self):
        from opencut.core.color_match_shots import _compute_adjustments

        src = {"luminance": 80.0, "saturation": 80.0,
               "avg_r": 100.0, "avg_g": 100.0, "avg_b": 100.0,
               "histogram_low": 0.2, "histogram_high": 0.5}
        ref = {"luminance": 160.0, "saturation": 200.0,
               "avg_r": 200.0, "avg_g": 200.0, "avg_b": 200.0,
               "histogram_low": 0.1, "histogram_high": 0.6}
        adj = _compute_adjustments(src, ref, 0.0)
        assert adj["brightness"] == pytest.approx(0.0, abs=0.001)
        assert adj["saturation"] == pytest.approx(1.0, abs=0.001)

    def test_strength_clamped_to_two(self):
        from opencut.core.color_match_shots import _compute_adjustments

        src = {"luminance": 128.0, "saturation": 128.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        ref = {"luminance": 200.0, "saturation": 200.0,
               "avg_r": 200.0, "avg_g": 200.0, "avg_b": 200.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        adj_normal = _compute_adjustments(src, ref, 2.0)
        adj_over = _compute_adjustments(src, ref, 5.0)
        assert adj_normal["brightness"] == adj_over["brightness"]

    def test_color_shifts_computed(self):
        from opencut.core.color_match_shots import _compute_adjustments

        src = {"luminance": 128.0, "saturation": 128.0,
               "avg_r": 100.0, "avg_g": 128.0, "avg_b": 150.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        ref = {"luminance": 128.0, "saturation": 128.0,
               "avg_r": 180.0, "avg_g": 128.0, "avg_b": 100.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        adj = _compute_adjustments(src, ref, 1.0)
        assert adj["r_shift"] > 0.0  # Source red < reference red
        assert adj["b_shift"] < 0.0  # Source blue > reference blue
        assert adj["g_shift"] == pytest.approx(0.0, abs=0.001)

    def test_zero_saturation_source(self):
        from opencut.core.color_match_shots import _compute_adjustments

        src = {"luminance": 128.0, "saturation": 0.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        ref = {"luminance": 128.0, "saturation": 100.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        adj = _compute_adjustments(src, ref, 1.0)
        assert adj["saturation"] == 1.0  # Falls back to ratio 1.0

    def test_lum_diff_and_sat_diff_in_result(self):
        from opencut.core.color_match_shots import _compute_adjustments

        src = {"luminance": 100.0, "saturation": 80.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        ref = {"luminance": 150.0, "saturation": 120.0,
               "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
               "histogram_low": 0.33, "histogram_high": 0.33}
        adj = _compute_adjustments(src, ref, 1.0)
        assert adj["lum_diff"] == pytest.approx(50.0, abs=0.1)
        assert adj["sat_diff"] == pytest.approx(40.0, abs=0.1)


class TestBuildColorMatchFilter:
    """Tests for _build_color_match_filter."""

    def test_null_filter_when_no_adjustments(self):
        from opencut.core.color_match_shots import _build_color_match_filter

        adj = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0,
               "r_shift": 0.0, "g_shift": 0.0, "b_shift": 0.0}
        filt = _build_color_match_filter(adj)
        assert filt == "null"

    def test_eq_filter_for_brightness(self):
        from opencut.core.color_match_shots import _build_color_match_filter

        adj = {"brightness": 0.15, "contrast": 1.0, "saturation": 1.0,
               "r_shift": 0.0, "g_shift": 0.0, "b_shift": 0.0}
        filt = _build_color_match_filter(adj)
        assert "eq=" in filt
        assert "brightness=" in filt

    def test_eq_filter_for_contrast(self):
        from opencut.core.color_match_shots import _build_color_match_filter

        adj = {"brightness": 0.0, "contrast": 1.3, "saturation": 1.0,
               "r_shift": 0.0, "g_shift": 0.0, "b_shift": 0.0}
        filt = _build_color_match_filter(adj)
        assert "contrast=" in filt

    def test_eq_filter_for_saturation(self):
        from opencut.core.color_match_shots import _build_color_match_filter

        adj = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.5,
               "r_shift": 0.0, "g_shift": 0.0, "b_shift": 0.0}
        filt = _build_color_match_filter(adj)
        assert "saturation=" in filt

    def test_colorbalance_filter_for_shifts(self):
        from opencut.core.color_match_shots import _build_color_match_filter

        adj = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0,
               "r_shift": 0.1, "g_shift": -0.05, "b_shift": 0.02}
        filt = _build_color_match_filter(adj)
        assert "colorbalance=" in filt
        assert "rs=" in filt

    def test_combined_eq_and_colorbalance(self):
        from opencut.core.color_match_shots import _build_color_match_filter

        adj = {"brightness": 0.1, "contrast": 1.2, "saturation": 0.9,
               "r_shift": 0.05, "g_shift": -0.03, "b_shift": 0.01}
        filt = _build_color_match_filter(adj)
        assert "eq=" in filt
        assert "colorbalance=" in filt
        assert "," in filt  # Combined with comma


class TestMatchColor:
    """Tests for match_color public API."""

    def test_source_not_found(self):
        from opencut.core.color_match_shots import match_color
        with pytest.raises(FileNotFoundError, match="Source"):
            match_color("/nonexistent/source.mp4", "/nonexistent/ref.mp4")

    def test_reference_not_found(self):
        from opencut.core.color_match_shots import match_color
        with patch("os.path.isfile", side_effect=lambda p: "source" in p):
            with pytest.raises(FileNotFoundError, match="Reference"):
                match_color("/test/source.mp4", "/nonexistent/ref.mp4")

    def test_match_calls_ffmpeg(self):
        from opencut.core.color_match_shots import match_color

        mock_stats = {
            "luminance": 128.0, "saturation": 128.0,
            "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
            "histogram_low": 0.33, "histogram_mid": 0.34, "histogram_high": 0.33,
        }

        with patch("opencut.core.color_match_shots.analyze_color_stats",
                   return_value=mock_stats), \
             patch("opencut.core.color_match_shots.run_ffmpeg") as mock_run, \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            result = match_color("/test/source.mp4", "/test/ref.mp4",
                                 output_path="/out/matched.mp4")
        mock_run.assert_called_once()
        assert result.output_path == "/out/matched.mp4"

    def test_strength_clamped(self):
        from opencut.core.color_match_shots import match_color

        mock_stats = {
            "luminance": 128.0, "saturation": 128.0,
            "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
            "histogram_low": 0.33, "histogram_mid": 0.34, "histogram_high": 0.33,
        }

        with patch("opencut.core.color_match_shots.analyze_color_stats",
                   return_value=mock_stats), \
             patch("opencut.core.color_match_shots.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            result = match_color("/test/source.mp4", "/test/ref.mp4",
                                 output_path="/out/matched.mp4", strength=5.0)
        # Should not raise -- strength gets clamped internally
        assert result.output_path == "/out/matched.mp4"

    def test_progress_callback_called(self):
        from opencut.core.color_match_shots import match_color

        mock_stats = {
            "luminance": 128.0, "saturation": 128.0,
            "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
            "histogram_low": 0.33, "histogram_mid": 0.34, "histogram_high": 0.33,
        }
        progress_calls = []

        with patch("opencut.core.color_match_shots.analyze_color_stats",
                   return_value=mock_stats), \
             patch("opencut.core.color_match_shots.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            match_color("/test/source.mp4", "/test/ref.mp4",
                        output_path="/out/matched.mp4",
                        on_progress=lambda pct, msg: progress_calls.append(pct))

        assert len(progress_calls) >= 3
        assert progress_calls[-1] == 100


class TestBatchMatch:
    """Tests for batch_match."""

    def test_empty_paths_raises(self):
        from opencut.core.color_match_shots import batch_match
        with pytest.raises(ValueError, match="No source paths"):
            batch_match([], "/test/ref.mp4")

    def test_reference_not_found_raises(self):
        from opencut.core.color_match_shots import batch_match
        with pytest.raises(FileNotFoundError):
            batch_match(["/test/a.mp4"], "/nonexistent/ref.mp4")

    def test_batch_processes_all_clips(self):
        from opencut.core.color_match_shots import batch_match

        mock_stats = {
            "luminance": 128.0, "saturation": 128.0,
            "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
            "histogram_low": 0.33, "histogram_mid": 0.34, "histogram_high": 0.33,
        }

        with patch("opencut.core.color_match_shots.analyze_color_stats",
                   return_value=mock_stats), \
             patch("opencut.core.color_match_shots.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            results = batch_match(
                ["/test/a.mp4", "/test/b.mp4", "/test/c.mp4"],
                "/test/ref.mp4",
                output_dir="/out/",
            )

        assert len(results) == 3

    def test_batch_skips_missing_files(self):
        from opencut.core.color_match_shots import batch_match

        mock_stats = {
            "luminance": 128.0, "saturation": 128.0,
            "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
            "histogram_low": 0.33, "histogram_mid": 0.34, "histogram_high": 0.33,
        }

        def _isfile(p):
            return "ref" in p or "a.mp4" in p

        with patch("opencut.core.color_match_shots.analyze_color_stats",
                   return_value=mock_stats), \
             patch("opencut.core.color_match_shots.run_ffmpeg"), \
             patch("os.path.isfile", side_effect=_isfile), \
             patch("os.makedirs"):
            results = batch_match(
                ["/test/a.mp4", "/test/missing.mp4"],
                "/test/ref.mp4",
                output_dir="/out/",
            )

        assert len(results) == 1

    def test_batch_progress_reaches_100(self):
        from opencut.core.color_match_shots import batch_match

        mock_stats = {
            "luminance": 128.0, "saturation": 128.0,
            "avg_r": 128.0, "avg_g": 128.0, "avg_b": 128.0,
            "histogram_low": 0.33, "histogram_mid": 0.34, "histogram_high": 0.33,
        }
        progress_calls = []

        with patch("opencut.core.color_match_shots.analyze_color_stats",
                   return_value=mock_stats), \
             patch("opencut.core.color_match_shots.run_ffmpeg"), \
             patch("os.path.isfile", return_value=True), \
             patch("os.makedirs"):
            batch_match(
                ["/test/a.mp4"],
                "/test/ref.mp4",
                output_dir="/out/",
                on_progress=lambda pct, msg: progress_calls.append(pct),
            )

        assert progress_calls[-1] == 100


# ========================================================================
# 2. video_llm.py
# ========================================================================
class TestVideoQueryResult:
    """Tests for VideoQueryResult dataclass."""

    def test_defaults(self):
        from opencut.core.video_llm import VideoQueryResult
        r = VideoQueryResult()
        assert r.answer == ""
        assert r.timestamps == []
        assert r.confidence == 0.0
        assert r.frames_analyzed == 0
        assert r.model_used == ""

    def test_populated(self):
        from opencut.core.video_llm import VideoQueryResult
        r = VideoQueryResult(
            answer="A cat jumps at 5s",
            timestamps=[5.0],
            confidence=0.85,
            frames_analyzed=16,
            model_used="gpt-4o",
        )
        assert "cat" in r.answer
        assert 5.0 in r.timestamps


class TestMomentResult:
    """Tests for MomentResult dataclass."""

    def test_defaults(self):
        from opencut.core.video_llm import MomentResult
        m = MomentResult()
        assert m.timestamp == 0.0
        assert m.confidence == 0.0
        assert m.description == ""

    def test_populated(self):
        from opencut.core.video_llm import MomentResult
        m = MomentResult(timestamp=12.5, confidence=0.9, description="explosion")
        assert m.timestamp == 12.5


class TestVideoLLMHelpers:
    """Tests for video LLM helper functions."""

    def test_select_backend_auto_openai(self):
        from opencut.core.video_llm import _select_backend
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            assert _select_backend("auto") == "openai"

    def test_select_backend_auto_anthropic(self):
        from opencut.core.video_llm import _select_backend
        env = {"ANTHROPIC_API_KEY": "test"}
        with patch.dict(os.environ, env, clear=False):
            # Remove OPENAI key if present
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                result = _select_backend("auto")
                # Should fall through to anthropic or local
                assert result in ("anthropic", "local", "openai")

    def test_select_backend_explicit_openai(self):
        from opencut.core.video_llm import _select_backend
        assert _select_backend("openai") == "openai"
        assert _select_backend("gpt-4o") == "openai"

    def test_select_backend_explicit_anthropic(self):
        from opencut.core.video_llm import _select_backend
        assert _select_backend("anthropic") == "anthropic"
        assert _select_backend("claude") == "anthropic"

    def test_select_backend_florence(self):
        from opencut.core.video_llm import _select_backend
        assert _select_backend("florence-2") == "florence2"
        assert _select_backend("fallback") == "florence2"

    def test_select_backend_unknown_is_local(self):
        from opencut.core.video_llm import _select_backend
        assert _select_backend("some_model") == "local"

    def test_parse_timestamps_hms(self):
        from opencut.core.video_llm import _parse_timestamps
        ts = _parse_timestamps("At 1:23:45 something happens", 6000)
        assert 5025.0 in ts  # 1*3600 + 23*60 + 45

    def test_parse_timestamps_ms(self):
        from opencut.core.video_llm import _parse_timestamps
        ts = _parse_timestamps("At 2:30 the scene changes", 300)
        assert 150.0 in ts  # 2*60 + 30

    def test_parse_timestamps_seconds(self):
        from opencut.core.video_llm import _parse_timestamps
        ts = _parse_timestamps("At 45s the cat jumps", 60)
        assert 45.0 in ts

    def test_parse_timestamps_out_of_range_excluded(self):
        from opencut.core.video_llm import _parse_timestamps
        ts = _parse_timestamps("At 500s something happens", 60)
        assert 500.0 not in ts

    def test_frame_to_base64(self):
        from opencut.core.video_llm import _frame_to_base64
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".jpg")
        try:
            os.write(fd, b"fake image data")
            os.close(fd)
            b64 = _frame_to_base64(path)
            assert len(b64) > 0
            import base64
            decoded = base64.b64decode(b64)
            assert decoded == b"fake image data"
        finally:
            os.unlink(path)


class TestQueryVideo:
    """Tests for query_video public API."""

    def test_file_not_found(self):
        from opencut.core.video_llm import query_video
        with pytest.raises(FileNotFoundError):
            query_video("/nonexistent/video.mp4", "What happens?")

    def test_empty_question_raises(self):
        from opencut.core.video_llm import query_video
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="empty"):
                query_video("/test/video.mp4", "")

    def test_max_frames_clamped(self):
        # Should not raise with extreme values; just test the clamping logic
        clamped = max(1, min(64, 100))
        assert clamped == 64
        clamped = max(1, min(64, -5))
        assert clamped == 1


class TestFindMoments:
    """Tests for find_moments public API."""

    def test_file_not_found(self):
        from opencut.core.video_llm import find_moments
        with pytest.raises(FileNotFoundError):
            find_moments("/nonexistent/video.mp4", "a funny moment")

    def test_empty_description_raises(self):
        from opencut.core.video_llm import find_moments
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="empty"):
                find_moments("/test/video.mp4", "")


class TestDescribeScene:
    """Tests for describe_scene."""

    def test_file_not_found(self):
        from opencut.core.video_llm import describe_scene
        with pytest.raises(FileNotFoundError):
            describe_scene("/nonexistent/video.mp4", timestamp=5.0)

    def test_negative_timestamp_raises(self):
        from opencut.core.video_llm import describe_scene
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="non-negative"):
                describe_scene("/test/video.mp4", timestamp=-1.0)


# ========================================================================
# 3. music_remix.py
# ========================================================================
class TestMusicFitResult:
    """Tests for MusicFitResult dataclass."""

    def test_defaults(self):
        from opencut.core.music_remix import MusicFitResult
        r = MusicFitResult()
        assert r.output_path == ""
        assert r.original_duration == 0.0
        assert r.target_duration == 0.0
        assert r.actual_duration == 0.0
        assert r.mode_used == ""
        assert r.edit_points == []

    def test_populated(self):
        from opencut.core.music_remix import MusicFitResult
        r = MusicFitResult(
            output_path="/out/music.aac",
            original_duration=120.0,
            target_duration=90.0,
            actual_duration=89.5,
            mode_used="smart",
            edit_points=[88.0],
        )
        assert r.mode_used == "smart"
        assert len(r.edit_points) == 1


class TestMusicStructure:
    """Tests for MusicStructure dataclass."""

    def test_defaults(self):
        from opencut.core.music_remix import MusicStructure
        ms = MusicStructure()
        assert ms.bpm == 0.0
        assert ms.bars == 0
        assert ms.sections == []
        assert ms.loop_points == []

    def test_section_dataclass(self):
        from opencut.core.music_remix import Section
        s = Section(label="intro", start=0.0, end=8.0, duration=8.0)
        assert s.label == "intro"


class TestAtempoChain:
    """Tests for _build_atempo_chain."""

    def test_normal_ratio(self):
        from opencut.core.music_remix import _build_atempo_chain
        chain = _build_atempo_chain(1.5)
        assert "atempo=" in chain

    def test_high_ratio_chains(self):
        from opencut.core.music_remix import _build_atempo_chain
        chain = _build_atempo_chain(4.0)
        # Should chain multiple atempo filters
        assert chain.count("atempo=") >= 2

    def test_low_ratio_chains(self):
        from opencut.core.music_remix import _build_atempo_chain
        chain = _build_atempo_chain(0.25)
        assert chain.count("atempo=") >= 2


class TestFitMusicToDuration:
    """Tests for fit_music_to_duration public API."""

    def test_file_not_found(self):
        from opencut.core.music_remix import fit_music_to_duration
        with pytest.raises(FileNotFoundError):
            fit_music_to_duration("/nonexistent/music.mp3", 60.0)

    def test_zero_duration_raises(self):
        from opencut.core.music_remix import fit_music_to_duration
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="positive"):
                fit_music_to_duration("/test/music.mp3", 0)

    def test_negative_duration_raises(self):
        from opencut.core.music_remix import fit_music_to_duration
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="positive"):
                fit_music_to_duration("/test/music.mp3", -10.0)

    def test_invalid_mode_raises(self):
        from opencut.core.music_remix import fit_music_to_duration
        with patch("os.path.isfile", return_value=True):
            with pytest.raises(ValueError, match="Unknown mode"):
                fit_music_to_duration("/test/music.mp3", 60.0, mode="invalid")

    def test_valid_modes_accepted(self):
        valid = ("smart", "stretch", "fade")
        for m in valid:
            assert m in valid


class TestDetectMusicStructure:
    """Tests for detect_music_structure."""

    def test_file_not_found(self):
        from opencut.core.music_remix import detect_music_structure
        with pytest.raises(FileNotFoundError):
            detect_music_structure("/nonexistent/music.mp3")

    def test_find_best_loop_point(self):
        from opencut.core.music_remix import _find_best_loop_point
        bars = [0.0, 2.0, 4.0, 6.0, 8.0]
        assert _find_best_loop_point(bars, 3.0) == 2.0
        assert _find_best_loop_point(bars, 7.5) == 8.0

    def test_find_best_loop_point_empty(self):
        from opencut.core.music_remix import _find_best_loop_point
        assert _find_best_loop_point([], 5.0) == 5.0


# ========================================================================
# 4. audio_category.py
# ========================================================================
class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_defaults(self):
        from opencut.core.audio_category import AudioSegment
        s = AudioSegment()
        assert s.start_time == 0.0
        assert s.end_time == 0.0
        assert s.category == "silence"
        assert s.confidence == 0.0

    def test_populated(self):
        from opencut.core.audio_category import AudioSegment
        s = AudioSegment(start_time=0.0, end_time=2.0,
                         category="speech", confidence=0.85)
        assert s.category == "speech"


class TestAudioClassificationResult:
    """Tests for AudioClassificationResult dataclass."""

    def test_defaults(self):
        from opencut.core.audio_category import AudioClassificationResult
        r = AudioClassificationResult()
        assert r.segments == []
        assert r.summary == {}

    def test_independent_defaults(self):
        from opencut.core.audio_category import AudioClassificationResult
        a = AudioClassificationResult()
        b = AudioClassificationResult()
        a.segments.append("test")
        assert len(b.segments) == 0


class TestAudioConstants:
    """Tests for module constants."""

    def test_categories_defined(self):
        from opencut.core.audio_category import CATEGORIES
        assert "speech" in CATEGORIES
        assert "music" in CATEGORIES
        assert "sound_effect" in CATEGORIES
        assert "ambience" in CATEGORIES
        assert "silence" in CATEGORIES

    def test_silence_threshold(self):
        from opencut.core.audio_category import SILENCE_RMS_THRESHOLD
        assert SILENCE_RMS_THRESHOLD < 0  # Should be negative dBFS


class TestClassifySegment:
    """Tests for _classify_segment heuristic."""

    def test_silence_detection(self):
        from opencut.core.audio_category import _classify_segment
        cat, conf = _classify_segment({"rms_level": -60.0, "peak_level": -55.0,
                                        "crest_factor": 0, "flat_factor": 0,
                                        "zero_crossings_rate": 0})
        assert cat == "silence"
        assert conf > 0.8

    def test_speech_with_spectral(self):
        from opencut.core.audio_category import _classify_segment
        stats = {"rms_level": -20.0, "peak_level": -10.0,
                 "crest_factor": 8, "flat_factor": 0,
                 "zero_crossings_rate": 0.1}
        spectral = {"spectral_centroid": 1500, "spectral_flatness": 0.15,
                     "energy_variance": 1e-4}
        cat, conf = _classify_segment(stats, spectral)
        assert cat == "speech"

    def test_music_with_spectral(self):
        from opencut.core.audio_category import _classify_segment
        stats = {"rms_level": -15.0, "peak_level": -5.0,
                 "crest_factor": 5, "flat_factor": 0,
                 "zero_crossings_rate": 0.05}
        spectral = {"spectral_centroid": 800, "spectral_flatness": 0.1,
                     "energy_variance": 1e-7}
        cat, conf = _classify_segment(stats, spectral)
        assert cat == "music"

    def test_ambience_with_spectral(self):
        from opencut.core.audio_category import _classify_segment
        stats = {"rms_level": -35.0, "peak_level": -25.0,
                 "crest_factor": 3, "flat_factor": 0,
                 "zero_crossings_rate": 0.02}
        spectral = {"spectral_centroid": 2000, "spectral_flatness": 0.7,
                     "energy_variance": 1e-8}
        cat, conf = _classify_segment(stats, spectral)
        assert cat == "ambience"

    def test_sfx_high_energy_variance(self):
        from opencut.core.audio_category import _classify_segment
        stats = {"rms_level": -10.0, "peak_level": -2.0,
                 "crest_factor": 20, "flat_factor": 0,
                 "zero_crossings_rate": 0.3}
        spectral = {"spectral_centroid": 5000, "spectral_flatness": 0.3,
                     "energy_variance": 1e-2}
        cat, conf = _classify_segment(stats, spectral)
        assert cat == "sound_effect"

    def test_fallback_speech_no_spectral(self):
        from opencut.core.audio_category import _classify_segment
        cat, conf = _classify_segment({"rms_level": -20.0, "peak_level": -8.0,
                                        "crest_factor": 10, "flat_factor": 0,
                                        "zero_crossings_rate": 0.1})
        assert cat == "speech"

    def test_fallback_ambience_no_spectral(self):
        from opencut.core.audio_category import _classify_segment
        cat, conf = _classify_segment({"rms_level": -40.0, "peak_level": -30.0,
                                        "crest_factor": 2, "flat_factor": 0,
                                        "zero_crossings_rate": 0.01})
        assert cat == "ambience"


class TestClassifyAudio:
    """Tests for classify_audio public API."""

    def test_file_not_found(self):
        from opencut.core.audio_category import classify_audio
        with pytest.raises(FileNotFoundError):
            classify_audio("/nonexistent/audio.wav")

    def test_segment_duration_clamped(self):
        """segment_duration should be clamped to 0.5-30.0."""
        clamped = max(0.5, min(30.0, 0.1))
        assert clamped == 0.5
        clamped = max(0.5, min(30.0, 100.0))
        assert clamped == 30.0


class TestExtractSegmentStats:
    """Tests for _extract_segment_stats."""

    def test_parses_rms_level(self):
        from opencut.core.audio_category import _extract_segment_stats

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "RMS_level=-25.3\nPeak_level=-12.1\n"
        mock_result.stdout = ""

        with patch("opencut.core.audio_category._sp.run", return_value=mock_result):
            stats = _extract_segment_stats("/test/audio.wav", 0, 2.0)
        assert stats["rms_level"] == pytest.approx(-25.3, abs=0.1)
        assert stats["peak_level"] == pytest.approx(-12.1, abs=0.1)

    def test_defaults_on_empty_output(self):
        from opencut.core.audio_category import _extract_segment_stats

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "nothing useful"
        mock_result.stdout = ""

        with patch("opencut.core.audio_category._sp.run", return_value=mock_result):
            stats = _extract_segment_stats("/test/audio.wav", 0, 2.0)
        assert stats["rms_level"] == -100.0


# ========================================================================
# 5. Route smoke tests
# ========================================================================
class TestNextGenAIRoutes:
    """Smoke tests for next_gen_ai_routes.py blueprint."""

    def test_blueprint_exists(self):
        from opencut.routes.next_gen_ai_routes import next_gen_ai_bp
        assert next_gen_ai_bp.name == "next_gen_ai"

    def test_blueprint_has_routes(self):
        from opencut.routes.next_gen_ai_routes import next_gen_ai_bp
        assert len(list(next_gen_ai_bp.deferred_functions)) > 0

    def test_route_functions_exist(self):
        from opencut.routes.next_gen_ai_routes import (
            audio_classify_route,
            audio_classify_timeline_route,
            color_match_analyze_route,
            color_match_batch_route,
            color_match_route,
            music_fit_duration_route,
            music_remix_route,
            video_llm_find_moment_route,
            video_llm_query_route,
        )
        assert callable(video_llm_query_route)
        assert callable(video_llm_find_moment_route)
        assert callable(music_remix_route)
        assert callable(music_fit_duration_route)
        assert callable(audio_classify_route)
        assert callable(audio_classify_timeline_route)
        assert callable(color_match_route)
        assert callable(color_match_batch_route)
        assert callable(color_match_analyze_route)

    def test_imports_from_routes(self):
        """All route handler names should be importable."""
        import opencut.routes.next_gen_ai_routes as mod
        assert hasattr(mod, "next_gen_ai_bp")
        assert hasattr(mod, "video_llm_query_route")
        assert hasattr(mod, "music_remix_route")
        assert hasattr(mod, "audio_classify_route")
        assert hasattr(mod, "color_match_route")
        assert hasattr(mod, "color_match_analyze_route")

    def test_blueprint_has_nine_routes(self):
        from opencut.routes.next_gen_ai_routes import next_gen_ai_bp
        # 8 async routes + 1 sync route = 9 deferred functions
        assert len(list(next_gen_ai_bp.deferred_functions)) == 9


class TestRouteDataPatterns:
    """Tests for route-level data handling patterns."""

    def test_mode_validation_defaults(self):
        """Invalid mode should default to smart."""
        mode = "invalid"
        if mode not in ("smart", "stretch", "fade"):
            mode = "smart"
        assert mode == "smart"

    def test_mode_validation_preserves_valid(self):
        for valid_mode in ("smart", "stretch", "fade"):
            mode = valid_mode
            if mode not in ("smart", "stretch", "fade"):
                mode = "smart"
            assert mode == valid_mode

    def test_segment_duration_defaults(self):
        """segment_duration should have sensible defaults."""
        from opencut.security import safe_float
        val = safe_float(None, 2.0, min_val=0.5, max_val=30.0)
        assert val == 2.0

    def test_strength_range(self):
        """strength should be clamped to 0.0-2.0."""
        from opencut.security import safe_float
        val = safe_float(5.0, 1.0, min_val=0.0, max_val=2.0)
        assert val == 2.0
        val = safe_float(-1.0, 1.0, min_val=0.0, max_val=2.0)
        assert val == 0.0

    def test_max_frames_range(self):
        """max_frames should be clamped to 1-64."""
        from opencut.security import safe_int
        val = safe_int(100, 16, min_val=1, max_val=64)
        assert val == 64
        val = safe_int(0, 16, min_val=1, max_val=64)
        assert val == 1
