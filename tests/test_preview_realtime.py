"""
Tests for OpenCut Real-Time AI Preview (Category 76)

Covers:
  - live_preview.py — all effects, caching, resolution, effect chains
  - gpu_preview_pipeline.py — GPU/CPU paths, batch, effect chains
  - ab_compare.py — all modes, metrics, wipe frames
  - realtime_scopes.py — all scope types, presets, legal range
  - preview_cache.py — LRU eviction, TTL, invalidation, stats, thread safety
  - preview_realtime_routes.py — route smoke tests via Flask test client

Uses Flask test client.  No real FFmpeg, no GPU, no PIL — all mocked.
"""

import os
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from tests.conftest import csrf_headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_temp_file(suffix=".mp4", content=b"fake video data"):
    """Create a temp file that survives until explicitly removed."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


def _make_temp_jpg(suffix=".jpg"):
    """Create a small valid-ish JPEG temp file."""
    # Minimal JPEG: SOI + EOI markers
    data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    return _make_temp_file(suffix=suffix, content=data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pr_app():
    """Flask app with preview_realtime_bp registered."""
    from opencut.config import OpenCutConfig
    from opencut.server import create_app
    test_config = OpenCutConfig()
    flask_app = create_app(config=test_config)
    flask_app.config["TESTING"] = True
    from opencut.routes.preview_realtime_routes import preview_realtime_bp
    try:
        flask_app.register_blueprint(preview_realtime_bp)
    except ValueError:
        pass
    return flask_app


@pytest.fixture
def pr_client(pr_app):
    return pr_app.test_client()


@pytest.fixture
def pr_csrf(pr_client):
    resp = pr_client.get("/health")
    data = resp.get_json()
    return data.get("csrf_token", "")


@pytest.fixture
def temp_video():
    path = _make_temp_file()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def temp_video_pair():
    a = _make_temp_file(content=b"original video")
    b = _make_temp_file(content=b"processed video")
    yield a, b
    for p in (a, b):
        try:
            os.unlink(p)
        except OSError:
            pass


@pytest.fixture
def temp_jpg():
    path = _make_temp_jpg()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# =====================================================================
# 1. live_preview.py — PreviewResult dataclass
# =====================================================================
class TestPreviewResult:
    def test_defaults(self):
        from opencut.core.live_preview import PreviewResult
        r = PreviewResult()
        assert r.preview_path == ""
        assert r.effect_applied == ""
        assert r.cached is False

    def test_to_dict(self):
        from opencut.core.live_preview import PreviewResult
        r = PreviewResult(preview_path="/tmp/x.jpg", effect_applied="blur",
                          resolution="854x480", processing_time_ms=42.0,
                          cached=True, timestamp=1.5)
        d = r.to_dict()
        assert d["preview_path"] == "/tmp/x.jpg"
        assert d["cached"] is True
        assert d["processing_time_ms"] == 42.0

    def test_with_params(self):
        from opencut.core.live_preview import PreviewResult
        r = PreviewResult(params={"strength": 0.5})
        assert r.params["strength"] == 0.5


# =====================================================================
# 2. live_preview.py — Effect registry
# =====================================================================
class TestEffectRegistry:
    def test_all_effects_registered(self):
        from opencut.core.live_preview import EFFECTS
        expected = {"color_grade", "denoise", "stabilize_frame",
                    "style_transfer", "background_remove", "upscale_preview",
                    "sharpen", "blur", "vignette", "film_grain"}
        assert set(EFFECTS.keys()) == expected

    def test_list_effects(self):
        from opencut.core.live_preview import list_effects
        effects = list_effects()
        assert len(effects) == 10
        ids = {e["id"] for e in effects}
        assert "color_grade" in ids
        assert "blur" in ids

    def test_list_effects_have_names(self):
        from opencut.core.live_preview import list_effects
        for e in list_effects():
            assert "name" in e
            assert "description" in e


# =====================================================================
# 3. live_preview.py — Cache operations
# =====================================================================
class TestPreviewCache:
    def setup_method(self):
        from opencut.core import live_preview
        live_preview._preview_cache.clear()
        live_preview._preview_cache_order.clear()

    def test_cache_key_deterministic(self):
        from opencut.core.live_preview import _cache_key
        k1 = _cache_key("/tmp/a.mp4", 1.0, "blur", {"s": 0.5})
        k2 = _cache_key("/tmp/a.mp4", 1.0, "blur", {"s": 0.5})
        assert k1 == k2

    def test_cache_key_changes_on_effect(self):
        from opencut.core.live_preview import _cache_key
        k1 = _cache_key("/tmp/a.mp4", 1.0, "blur", {})
        k2 = _cache_key("/tmp/a.mp4", 1.0, "sharpen", {})
        assert k1 != k2

    def test_cache_put_get(self):
        from opencut.core.live_preview import _cache_put, _cache_get
        path = _make_temp_file(suffix=".jpg")
        try:
            _cache_put("testkey1", path)
            assert _cache_get("testkey1") == path
        finally:
            os.unlink(path)

    def test_cache_miss(self):
        from opencut.core.live_preview import _cache_get
        assert _cache_get("nonexistent") is None

    def test_cache_stale_entry(self):
        from opencut.core.live_preview import _cache_get, _preview_cache
        _preview_cache["stalekey"] = "/nonexistent/path.jpg"
        assert _cache_get("stalekey") is None

    def test_clear_cache(self):
        from opencut.core.live_preview import _cache_put, clear_preview_cache
        path = _make_temp_file(suffix=".jpg")
        _cache_put("clearkey", path)
        count = clear_preview_cache()
        assert count >= 1

    def test_cache_stats(self):
        from opencut.core.live_preview import preview_cache_stats
        stats = preview_cache_stats()
        assert "entry_count" in stats
        assert "total_size_mb" in stats

    def test_cache_eviction_by_count(self):
        from opencut.core import live_preview
        old_max = live_preview._CACHE_MAX_ENTRIES
        live_preview._CACHE_MAX_ENTRIES = 3
        try:
            paths = []
            for i in range(5):
                p = _make_temp_file(suffix=".jpg")
                paths.append(p)
                live_preview._cache_put(f"evict_{i}", p)
            assert len(live_preview._preview_cache) <= 3
        finally:
            live_preview._CACHE_MAX_ENTRIES = old_max
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# =====================================================================
# 4. live_preview.py — Effect functions (via FFmpeg mock)
# =====================================================================
class TestEffectFunctions:
    @patch("opencut.core.live_preview._sp.run")
    def test_color_grade(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_color_grade
        result = effect_color_grade("/tmp/in.jpg", "/tmp/out.jpg",
                                    {"brightness": 0.1, "contrast": 1.2})
        assert result == "/tmp/out.jpg"
        mock_run.assert_called_once()

    @patch("opencut.core.live_preview._sp.run")
    def test_denoise(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_denoise
        result = effect_denoise("/tmp/in.jpg", "/tmp/out.jpg", {"strength": 0.7})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_stabilize_frame(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_stabilize_frame
        result = effect_stabilize_frame("/tmp/in.jpg", "/tmp/out.jpg", {})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_style_transfer_all_styles(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_style_transfer
        for style in ("oil_painting", "watercolor", "pencil_sketch", "vintage", "noir"):
            result = effect_style_transfer("/tmp/in.jpg", "/tmp/out.jpg",
                                           {"style": style})
            assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_background_remove(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_background_remove
        result = effect_background_remove("/tmp/in.jpg", "/tmp/out.jpg", {})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_upscale_preview(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_upscale_preview
        result = effect_upscale_preview("/tmp/in.jpg", "/tmp/out.jpg",
                                        {"factor": 3})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_sharpen(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_sharpen
        result = effect_sharpen("/tmp/in.jpg", "/tmp/out.jpg",
                                {"strength": 0.8})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_blur(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_blur
        result = effect_blur("/tmp/in.jpg", "/tmp/out.jpg", {"strength": 0.5})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_vignette(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_vignette
        result = effect_vignette("/tmp/in.jpg", "/tmp/out.jpg", {})
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.live_preview._sp.run")
    def test_film_grain(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.live_preview import effect_film_grain
        result = effect_film_grain("/tmp/in.jpg", "/tmp/out.jpg",
                                   {"strength": 0.3})
        assert result == "/tmp/out.jpg"


# =====================================================================
# 5. live_preview.py — generate_live_preview (integrated, mocked FFmpeg)
# =====================================================================
class TestGenerateLivePreview:
    def test_file_not_found(self):
        from opencut.core.live_preview import generate_live_preview
        with pytest.raises(FileNotFoundError):
            generate_live_preview("/nonexistent/video.mp4", "blur")

    def test_unknown_effect(self, temp_video):
        from opencut.core.live_preview import generate_live_preview
        with pytest.raises(ValueError, match="Unknown effect"):
            generate_live_preview(temp_video, "nonexistent_effect")

    @patch("opencut.core.live_preview._sp.run")
    def test_generate_and_cache(self, mock_run, temp_video):
        from opencut.core.live_preview import (
            generate_live_preview, _preview_cache, _preview_cache_order,
        )
        _preview_cache.clear()
        _preview_cache_order.clear()

        # Mock FFmpeg to create output files
        def _side_effect(cmd, **kwargs):
            # Find the output path (last argument before -y)
            out = cmd[-1]
            if out.endswith(".jpg"):
                with open(out, "wb") as f:
                    f.write(b"\xff\xd8fake\xff\xd9")
            return MagicMock(returncode=0)

        mock_run.side_effect = _side_effect
        result = generate_live_preview(temp_video, "blur",
                                       params={"strength": 0.5})
        assert result.effect_applied == "blur"
        assert result.cached is False
        assert result.processing_time_ms >= 0

    @patch("opencut.core.live_preview._sp.run")
    def test_validate_strength(self, mock_run):
        from opencut.core.live_preview import _validate_strength
        assert _validate_strength({}, 0.5) == 0.5
        assert _validate_strength({"strength": "bad"}, 0.3) == 0.3
        assert _validate_strength({"strength": 1.5}, 0.5) == 1.0
        assert _validate_strength({"strength": -0.5}, 0.5) == 0.0


# =====================================================================
# 6. live_preview.py — apply_effect_chain
# =====================================================================
class TestEffectChain:
    def test_empty_chain_raises(self, temp_video):
        from opencut.core.live_preview import apply_effect_chain
        with pytest.raises(ValueError, match="No effects"):
            apply_effect_chain(temp_video, [])

    def test_file_not_found(self):
        from opencut.core.live_preview import apply_effect_chain
        with pytest.raises(FileNotFoundError):
            apply_effect_chain("/nonexistent.mp4", [{"effect": "blur"}])

    @patch("opencut.core.live_preview._sp.run")
    def test_chain_applies_sequentially(self, mock_run, temp_video):
        from opencut.core.live_preview import apply_effect_chain

        def _side_effect(cmd, **kwargs):
            out = cmd[-1]
            if out.endswith(".jpg"):
                with open(out, "wb") as f:
                    f.write(b"\xff\xd8fake\xff\xd9")
            return MagicMock(returncode=0)

        mock_run.side_effect = _side_effect

        chain = [
            {"effect": "blur", "params": {"strength": 0.3}},
            {"effect": "sharpen", "params": {"strength": 0.5}},
        ]
        result = apply_effect_chain(temp_video, chain)
        assert "blur" in result.effect_applied
        assert "sharpen" in result.effect_applied


# =====================================================================
# 7. gpu_preview_pipeline.py — GPU detection
# =====================================================================
class TestGPUDetection:
    def setup_method(self):
        from opencut.core.gpu_preview_pipeline import reset_gpu_detection
        reset_gpu_detection()

    @patch("opencut.core.gpu_preview_pipeline._sp.run")
    def test_nvidia_smi_found(self, mock_run):
        from opencut.core.gpu_preview_pipeline import detect_gpu, reset_gpu_detection
        reset_gpu_detection()
        mock_run.return_value = MagicMock(
            returncode=0, stdout="NVIDIA RTX 4090, 24576\n", stderr=b""
        )
        info = detect_gpu()
        assert info["available"] is True
        assert info["device"] == "cuda"
        assert "4090" in info["name"]
        assert info["method"] == "nvidia-smi"

    @patch("opencut.core.gpu_preview_pipeline._sp.run")
    def test_no_gpu(self, mock_run):
        from opencut.core.gpu_preview_pipeline import detect_gpu, reset_gpu_detection
        reset_gpu_detection()
        mock_run.side_effect = FileNotFoundError
        info = detect_gpu()
        # May still be True if torch.cuda is available, but in test env
        # likely False
        assert "available" in info
        assert "device" in info

    def test_reset_clears_cache(self):
        from opencut.core.gpu_preview_pipeline import reset_gpu_detection
        reset_gpu_detection()
        from opencut.core import gpu_preview_pipeline
        assert gpu_preview_pipeline._gpu_available is None


# =====================================================================
# 8. gpu_preview_pipeline.py — PipelineResult dataclass
# =====================================================================
class TestPipelineResult:
    def test_defaults(self):
        from opencut.core.gpu_preview_pipeline import PipelineResult
        r = PipelineResult()
        assert r.frames == []
        assert r.gpu_used is False
        assert r.total_time_ms == 0.0

    def test_to_dict(self):
        from opencut.core.gpu_preview_pipeline import PipelineResult, FramePreview
        fp = FramePreview(timestamp=1.0, preview_path="/tmp/f.jpg",
                          width=854, height=480)
        r = PipelineResult(frames=[fp], gpu_used=True, device="cuda",
                           total_time_ms=200.0, frames_per_second=5.0)
        d = r.to_dict()
        assert len(d["frames"]) == 1
        assert d["gpu_used"] is True


# =====================================================================
# 9. gpu_preview_pipeline.py — Pipeline singleton
# =====================================================================
class TestPipelineSingleton:
    def test_get_pipeline(self):
        from opencut.core.gpu_preview_pipeline import get_pipeline, reset_pipeline
        reset_pipeline()
        p = get_pipeline(use_gpu=False)
        assert p is not None
        assert p is get_pipeline()
        reset_pipeline()

    def test_pipeline_status(self):
        from opencut.core.gpu_preview_pipeline import get_pipeline, reset_pipeline
        reset_pipeline()
        p = get_pipeline(use_gpu=False)
        s = p.status()
        assert "queue_size" in s
        assert "gpu" in s
        reset_pipeline()


# =====================================================================
# 10. gpu_preview_pipeline.py — render_frame
# =====================================================================
class TestRenderFrame:
    def test_file_not_found(self):
        from opencut.core.gpu_preview_pipeline import PreviewPipeline
        p = PreviewPipeline(use_gpu=False)
        with pytest.raises(FileNotFoundError):
            p.render_frame("/nonexistent.mp4", 0.0)

    @patch("opencut.core.gpu_preview_pipeline._sp.run")
    def test_render_frame_cpu(self, mock_run, temp_video):
        from opencut.core.gpu_preview_pipeline import PreviewPipeline

        def _side_effect(cmd, **kwargs):
            out = cmd[-1]
            if out.endswith(".jpg"):
                with open(out, "wb") as f:
                    f.write(b"\xff\xd8fake\xff\xd9")
            return MagicMock(returncode=0)

        mock_run.side_effect = _side_effect
        p = PreviewPipeline(use_gpu=False)
        fp = p.render_frame(temp_video, 1.0)
        assert fp.timestamp == 1.0
        assert fp.width == 854


# =====================================================================
# 11. gpu_preview_pipeline.py — render_batch
# =====================================================================
class TestRenderBatch:
    @patch("opencut.core.gpu_preview_pipeline._sp.run")
    @patch("opencut.core.gpu_preview_pipeline.get_video_info")
    def test_batch_render(self, mock_info, mock_run, temp_video):
        from opencut.core.gpu_preview_pipeline import PreviewPipeline

        mock_info.return_value = {"width": 1920, "height": 1080,
                                   "fps": 30.0, "duration": 10.0}

        def _side_effect(cmd, **kwargs):
            out = cmd[-1]
            if out.endswith(".jpg"):
                with open(out, "wb") as f:
                    f.write(b"\xff\xd8fake\xff\xd9")
            return MagicMock(returncode=0)

        mock_run.side_effect = _side_effect
        p = PreviewPipeline(use_gpu=False)
        result = p.render_batch(temp_video, num_frames=3)
        assert len(result.frames) == 3
        assert result.gpu_used is False
        assert result.total_time_ms >= 0

    def test_batch_file_not_found(self):
        from opencut.core.gpu_preview_pipeline import PreviewPipeline
        p = PreviewPipeline(use_gpu=False)
        with pytest.raises(FileNotFoundError):
            p.render_batch("/nonexistent.mp4")


# =====================================================================
# 12. ab_compare.py — CompareResult dataclass
# =====================================================================
class TestCompareResult:
    def test_defaults(self):
        from opencut.core.ab_compare import CompareResult
        r = CompareResult()
        assert r.frames == []
        assert r.overall_ssim == 0.0
        assert r.frame_count == 0

    def test_to_dict(self):
        from opencut.core.ab_compare import CompareResult, CompareFrame
        cf = CompareFrame(timestamp=1.0, mode="side_by_side")
        r = CompareResult(frames=[cf], overall_ssim=0.95,
                          overall_psnr=35.0, mode="side_by_side",
                          frame_count=1)
        d = r.to_dict()
        assert d["overall_ssim"] == 0.95
        assert len(d["frames"]) == 1


# =====================================================================
# 13. ab_compare.py — FrameMetrics
# =====================================================================
class TestFrameMetrics:
    def test_defaults(self):
        from opencut.core.ab_compare import FrameMetrics
        m = FrameMetrics()
        assert m.ssim == 0.0
        assert m.psnr == 0.0

    def test_to_dict(self):
        from opencut.core.ab_compare import FrameMetrics
        m = FrameMetrics(ssim=0.98, psnr=40.0, color_delta=2.5, mse=10.0)
        d = m.to_dict()
        assert d["ssim"] == 0.98

    def test_metrics_numpy(self):
        """Test numpy metrics path with synthetic arrays."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        from opencut.core.ab_compare import _metrics_numpy
        a = np.full((10, 10, 3), 128, dtype=np.float64)
        b = np.full((10, 10, 3), 128, dtype=np.float64)
        m = _metrics_numpy(a, b)
        assert m.ssim == 1.0  # identical images
        assert m.psnr == 100.0
        assert m.mse == 0.0

    def test_metrics_numpy_different(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        from opencut.core.ab_compare import _metrics_numpy
        a = np.full((10, 10, 3), 100, dtype=np.float64)
        b = np.full((10, 10, 3), 200, dtype=np.float64)
        m = _metrics_numpy(a, b)
        assert m.ssim < 1.0
        assert m.psnr < 100.0
        assert m.mse > 0


# =====================================================================
# 14. ab_compare.py — Composite functions
# =====================================================================
class TestComposites:
    @patch("opencut.core.ab_compare._sp.run")
    def test_side_by_side(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.ab_compare import _composite_side_by_side
        result = _composite_side_by_side("/tmp/a.jpg", "/tmp/b.jpg",
                                         "/tmp/out.jpg", 854, 480)
        assert result == "/tmp/out.jpg"

    @patch("opencut.core.ab_compare._sp.run")
    def test_overlay_blend(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        from opencut.core.ab_compare import _composite_overlay_blend
        result = _composite_overlay_blend("/tmp/a.jpg", "/tmp/b.jpg",
                                          "/tmp/out.jpg", 854, 480)
        assert result == "/tmp/out.jpg"


# =====================================================================
# 15. ab_compare.py — generate_comparison (mocked)
# =====================================================================
class TestGenerateComparison:
    def test_file_not_found_original(self):
        from opencut.core.ab_compare import generate_comparison
        with pytest.raises(FileNotFoundError):
            generate_comparison("/nonexistent.mp4", "/also_nonexistent.mp4")

    def test_invalid_mode(self, temp_video_pair):
        from opencut.core.ab_compare import generate_comparison
        a, b = temp_video_pair
        with pytest.raises(ValueError, match="Invalid mode"):
            generate_comparison(a, b, mode="invalid_mode")

    def test_list_compare_modes(self):
        from opencut.core.ab_compare import list_compare_modes
        modes = list_compare_modes()
        assert len(modes) == 6
        ids = {m["id"] for m in modes}
        assert "side_by_side" in ids
        assert "checkerboard" in ids


# =====================================================================
# 16. ab_compare.py — get_compare_metrics
# =====================================================================
class TestGetCompareMetrics:
    def test_file_not_found(self):
        from opencut.core.ab_compare import get_compare_metrics
        with pytest.raises(FileNotFoundError):
            get_compare_metrics("/nonexistent.mp4", "/also_nonexistent.mp4")


# =====================================================================
# 17. realtime_scopes.py — ScopeResult dataclass
# =====================================================================
class TestScopeResult:
    def test_defaults(self):
        from opencut.core.realtime_scopes import ScopeResult
        r = ScopeResult()
        assert r.scope_type == ""
        assert r.data == {}
        assert r.legal_range_violations == 0

    def test_to_dict(self):
        from opencut.core.realtime_scopes import ScopeResult
        r = ScopeResult(scope_type="waveform", data={"luma": []},
                        frame_timestamp=1.5)
        d = r.to_dict()
        assert d["scope_type"] == "waveform"


# =====================================================================
# 18. realtime_scopes.py — MultiScopeResult
# =====================================================================
class TestMultiScopeResult:
    def test_defaults(self):
        from opencut.core.realtime_scopes import MultiScopeResult
        r = MultiScopeResult()
        assert r.scopes == {}
        assert r.preset == ""

    def test_to_dict(self):
        from opencut.core.realtime_scopes import MultiScopeResult, ScopeResult
        sr = ScopeResult(scope_type="histogram", data={"bins": 256})
        r = MultiScopeResult(scopes={"histogram": sr}, preset="exposure")
        d = r.to_dict()
        assert "histogram" in d["scopes"]
        assert d["preset"] == "exposure"


# =====================================================================
# 19. realtime_scopes.py — Scope computation functions
# =====================================================================
class TestScopeComputation:
    def _make_pixels(self, w, h, color=(128, 128, 128)):
        return [color] * (w * h)

    def test_compute_histogram(self):
        from opencut.core.realtime_scopes import _compute_histogram
        pixels = self._make_pixels(10, 10, (100, 150, 200))
        data, violations = _compute_histogram(pixels, 10, 10)
        assert "r" in data
        assert "g" in data
        assert "b" in data
        assert "luma" in data
        assert data["bins"] == 256

    def test_compute_histogram_legal_violations(self):
        from opencut.core.realtime_scopes import _compute_histogram
        # All pixels at 0 (below legal black=16)
        pixels = self._make_pixels(10, 10, (0, 0, 0))
        data, violations = _compute_histogram(pixels, 10, 10, check_legal=True)
        assert violations == 100  # all 100 pixels violate

    def test_compute_vectorscope(self):
        from opencut.core.realtime_scopes import _compute_vectorscope
        pixels = self._make_pixels(10, 10, (200, 50, 100))
        data, violations = _compute_vectorscope(pixels, 10, 10)
        assert "grid" in data
        assert data["size"] == 256

    def test_compute_waveform(self):
        from opencut.core.realtime_scopes import _compute_waveform
        pixels = self._make_pixels(10, 10)
        data, violations = _compute_waveform(pixels, 10, 10)
        assert "luma" in data
        assert "columns" in data

    def test_compute_parade(self):
        from opencut.core.realtime_scopes import _compute_parade
        pixels = self._make_pixels(10, 10)
        data, violations = _compute_parade(pixels, 10, 10)
        assert "r" in data
        assert "g" in data
        assert "b" in data

    def test_compute_false_color(self):
        from opencut.core.realtime_scopes import _compute_false_color
        pixels = self._make_pixels(10, 10, (128, 128, 128))
        data, violations = _compute_false_color(pixels, 10, 10)
        assert "zones" in data
        assert "map" in data
        assert len(data["map"]) == 100

    def test_false_color_zones(self):
        from opencut.core.realtime_scopes import _compute_false_color
        # Very dark pixels -> deep_shadow zone
        pixels = self._make_pixels(5, 5, (5, 5, 5))
        data, _ = _compute_false_color(pixels, 5, 5)
        assert data["zones"]["deep_shadow"]["count"] == 25

    def test_false_color_legal_violations(self):
        from opencut.core.realtime_scopes import _compute_false_color
        # Clipped highlights
        pixels = self._make_pixels(5, 5, (255, 255, 255))
        data, violations = _compute_false_color(pixels, 5, 5, check_legal=True)
        assert violations == 25


# =====================================================================
# 20. realtime_scopes.py — generate_scope
# =====================================================================
class TestGenerateScope:
    def test_file_not_found(self):
        from opencut.core.realtime_scopes import generate_scope
        with pytest.raises(FileNotFoundError):
            generate_scope("/nonexistent.mp4", "histogram")

    def test_unknown_scope_type(self, temp_video):
        from opencut.core.realtime_scopes import generate_scope
        with pytest.raises(ValueError, match="Unknown scope"):
            generate_scope(temp_video, "nonexistent_scope")


# =====================================================================
# 21. realtime_scopes.py — list functions
# =====================================================================
class TestScopeListFunctions:
    def test_list_scope_types(self):
        from opencut.core.realtime_scopes import list_scope_types
        types_list = list_scope_types()
        assert len(types_list) == 5
        ids = {t["id"] for t in types_list}
        assert "waveform" in ids
        assert "false_color" in ids

    def test_list_presets(self):
        from opencut.core.realtime_scopes import list_presets
        presets = list_presets()
        assert len(presets) >= 3
        ids = {p["id"] for p in presets}
        assert "colorist" in ids
        assert "exposure" in ids
        assert "broadcast" in ids


# =====================================================================
# 22. realtime_scopes.py — SCOPE_PRESETS
# =====================================================================
class TestScopePresets:
    def test_preset_contents(self):
        from opencut.core.realtime_scopes import SCOPE_PRESETS
        assert "colorist" in SCOPE_PRESETS
        assert "waveform" in SCOPE_PRESETS["colorist"]["scopes"]
        assert "vectorscope" in SCOPE_PRESETS["colorist"]["scopes"]

    def test_broadcast_has_legal_range(self):
        from opencut.core.realtime_scopes import SCOPE_PRESETS
        bc = SCOPE_PRESETS["broadcast"]
        assert bc.get("options", {}).get("legal_range") is True


# =====================================================================
# 23. preview_cache.py — CacheEntry dataclass
# =====================================================================
class TestCacheEntry:
    def test_defaults(self):
        from opencut.core.preview_cache import CacheEntry
        e = CacheEntry()
        assert e.cache_key == ""
        assert e.hit_count == 0
        assert e.ttl == 3600.0

    def test_is_expired(self):
        from opencut.core.preview_cache import CacheEntry
        e = CacheEntry(created_at=time.time() - 7200, ttl=3600.0)
        assert e.is_expired() is True

    def test_not_expired(self):
        from opencut.core.preview_cache import CacheEntry
        e = CacheEntry(created_at=time.time(), ttl=3600.0)
        assert e.is_expired() is False

    def test_to_dict(self):
        from opencut.core.preview_cache import CacheEntry
        e = CacheEntry(cache_key="abc", effect_name="blur")
        d = e.to_dict()
        assert d["cache_key"] == "abc"
        assert d["effect_name"] == "blur"

    def test_from_dict(self):
        from opencut.core.preview_cache import CacheEntry
        d = {"cache_key": "xyz", "effect_name": "sharpen", "ttl": 1800.0}
        e = CacheEntry.from_dict(d)
        assert e.cache_key == "xyz"
        assert e.ttl == 1800.0


# =====================================================================
# 24. preview_cache.py — CacheStats
# =====================================================================
class TestCacheStats:
    def test_defaults(self):
        from opencut.core.preview_cache import CacheStats
        s = CacheStats()
        assert s.hit_count == 0
        assert s.hit_ratio == 0.0

    def test_to_dict(self):
        from opencut.core.preview_cache import CacheStats
        s = CacheStats(hit_count=10, miss_count=5, hit_ratio=0.667,
                       total_size_mb=50.0, entry_count=20)
        d = s.to_dict()
        assert d["hit_count"] == 10
        assert d["entry_count"] == 20


# =====================================================================
# 25. preview_cache.py — PreviewCacheManager
# =====================================================================
class TestPreviewCacheManager:
    def test_make_key_deterministic(self):
        from opencut.core.preview_cache import PreviewCacheManager
        k1 = PreviewCacheManager.make_key("/a.mp4", 100.0, "blur", '{"s":0.5}')
        k2 = PreviewCacheManager.make_key("/a.mp4", 100.0, "blur", '{"s":0.5}')
        assert k1 == k2

    def test_make_key_varies(self):
        from opencut.core.preview_cache import PreviewCacheManager
        k1 = PreviewCacheManager.make_key("/a.mp4", 100.0, "blur", "{}")
        k2 = PreviewCacheManager.make_key("/a.mp4", 100.0, "sharpen", "{}")
        assert k1 != k2

    def test_put_and_get(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       default_ttl=3600, cleanup_interval=9999)
            try:
                f = _make_temp_file(suffix=".jpg")
                mgr.put("testkey", f, source_path="/src.mp4",
                         effect_name="blur")
                result = mgr.get("testkey")
                assert result is not None
                assert result.endswith(".jpg")
            finally:
                mgr.shutdown()

    def test_get_miss(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       cleanup_interval=9999)
            try:
                assert mgr.get("nonexistent") is None
            finally:
                mgr.shutdown()

    def test_get_expired(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       default_ttl=0.001, cleanup_interval=9999)
            try:
                f = _make_temp_file(suffix=".jpg")
                mgr.put("expkey", f, ttl=0.001)
                time.sleep(0.01)
                assert mgr.get("expkey") is None
            finally:
                mgr.shutdown()

    def test_flush(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       cleanup_interval=9999)
            try:
                f = _make_temp_file(suffix=".jpg")
                mgr.put("flushkey", f)
                count = mgr.flush()
                assert count >= 1
                assert mgr.get("flushkey") is None
            finally:
                mgr.shutdown()

    def test_invalidate_by_file(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       cleanup_interval=9999)
            try:
                f = _make_temp_file(suffix=".jpg")
                mgr.put("fkey1", f, source_path="/video/a.mp4")
                count = mgr.invalidate_by_file("/video/a.mp4")
                assert count == 1
            finally:
                mgr.shutdown()

    def test_invalidate_by_effect(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       cleanup_interval=9999)
            try:
                for i in range(3):
                    f = _make_temp_file(suffix=".jpg")
                    mgr.put(f"ekey{i}", f, effect_name="blur")
                count = mgr.invalidate_by_effect("blur")
                assert count == 3
            finally:
                mgr.shutdown()

    def test_stats(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       cleanup_interval=9999)
            try:
                s = mgr.stats()
                assert s.hit_count == 0
                assert s.miss_count == 0
                assert s.entry_count == 0

                mgr.get("miss_test")  # generates a miss
                s = mgr.stats()
                assert s.miss_count == 1
            finally:
                mgr.shutdown()

    def test_lru_eviction(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            # Very small max size to trigger eviction
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=0.0001,
                                       cleanup_interval=9999)
            try:
                for i in range(5):
                    f = _make_temp_file(suffix=".jpg",
                                        content=b"x" * 100)
                    mgr.put(f"lru_{i}", f)
                s = mgr.stats()
                # Should have evicted some entries
                assert s.entry_count < 5
            finally:
                mgr.shutdown()


# =====================================================================
# 26. preview_cache.py — Thread safety
# =====================================================================
class TestCacheThreadSafety:
    def test_concurrent_puts(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=50,
                                       cleanup_interval=9999)
            errors = []

            def _put(idx):
                try:
                    f = _make_temp_file(suffix=".jpg")
                    mgr.put(f"thread_{idx}", f)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=_put, args=(i,))
                       for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            mgr.shutdown()
            assert len(errors) == 0

    def test_concurrent_gets(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=50,
                                       cleanup_interval=9999)
            f = _make_temp_file(suffix=".jpg")
            mgr.put("shared_key", f)
            errors = []

            def _get(idx):
                try:
                    mgr.get("shared_key")
                    mgr.get("nonexistent")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=_get, args=(i,))
                       for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            mgr.shutdown()
            assert len(errors) == 0


# =====================================================================
# 27. preview_cache.py �� Module-level functions
# =====================================================================
class TestCacheModuleFunctions:
    def test_cache_stats(self):
        from opencut.core.preview_cache import cache_stats, reset_cache_manager
        reset_cache_manager()
        s = cache_stats()
        assert "hit_count" in s
        reset_cache_manager()

    def test_cache_flush(self):
        from opencut.core.preview_cache import cache_flush, reset_cache_manager
        reset_cache_manager()
        count = cache_flush()
        assert isinstance(count, int)
        reset_cache_manager()


# =====================================================================
# 28. Route smoke tests — /api/preview/effects
# =====================================================================
class TestEffectsRoute:
    def test_list_effects(self, pr_client):
        resp = pr_client.get("/api/preview/effects")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "effects" in data
        assert len(data["effects"]) == 10


# =====================================================================
# 29. Route smoke tests — /api/preview/scopes/presets
# =====================================================================
class TestScopePresetsRoute:
    def test_list_presets(self, pr_client):
        resp = pr_client.get("/api/preview/scopes/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data
        assert len(data["presets"]) >= 3


# =====================================================================
# 30. Route smoke tests — /api/preview/cache/stats
# =====================================================================
class TestCacheStatsRoute:
    def test_cache_stats(self, pr_client):
        from opencut.core.preview_cache import reset_cache_manager
        reset_cache_manager()
        resp = pr_client.get("/api/preview/cache/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "hit_count" in data
        reset_cache_manager()


# =====================================================================
# 31. Route smoke tests — POST /api/preview/live (validation)
# =====================================================================
class TestLivePreviewRoute:
    def test_missing_filepath(self, pr_client, pr_csrf):
        resp = pr_client.post("/api/preview/live",
                               headers=csrf_headers(pr_csrf),
                               json={"effect": "blur"})
        assert resp.status_code == 400

    def test_missing_effect(self, pr_client, pr_csrf, temp_video):
        resp = pr_client.post("/api/preview/live",
                               headers=csrf_headers(pr_csrf),
                               json={"filepath": temp_video})
        assert resp.status_code == 400

    def test_missing_csrf(self, pr_client, temp_video):
        resp = pr_client.post("/api/preview/live",
                               headers={"Content-Type": "application/json"},
                               json={"filepath": temp_video, "effect": "blur"})
        assert resp.status_code == 403


# =====================================================================
# 32. Route smoke tests — POST /api/preview/compare (validation)
# =====================================================================
class TestCompareRoute:
    def test_missing_paths(self, pr_client, pr_csrf):
        resp = pr_client.post("/api/preview/compare",
                               headers=csrf_headers(pr_csrf),
                               json={})
        assert resp.status_code == 400

    def test_missing_processed(self, pr_client, pr_csrf, temp_video):
        resp = pr_client.post("/api/preview/compare",
                               headers=csrf_headers(pr_csrf),
                               json={"original": temp_video})
        assert resp.status_code == 400


# =====================================================================
# 33. Route smoke tests — POST /api/preview/scopes (validation)
# =====================================================================
class TestScopesRoute:
    def test_missing_filepath(self, pr_client, pr_csrf):
        resp = pr_client.post("/api/preview/scopes",
                               headers=csrf_headers(pr_csrf),
                               json={"scope_type": "histogram"})
        assert resp.status_code == 400

    def test_missing_scope_type(self, pr_client, pr_csrf, temp_video):
        resp = pr_client.post("/api/preview/scopes",
                               headers=csrf_headers(pr_csrf),
                               json={"filepath": temp_video})
        assert resp.status_code == 400


# =====================================================================
# 34. Route smoke tests — DELETE /api/preview/cache
# =====================================================================
class TestDeleteCacheRoute:
    def test_clear_all(self, pr_client, pr_csrf):
        from opencut.core.preview_cache import reset_cache_manager
        reset_cache_manager()
        resp = pr_client.delete("/api/preview/cache",
                                 headers=csrf_headers(pr_csrf))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["scope"] == "all"
        reset_cache_manager()

    def test_clear_missing_csrf(self, pr_client):
        resp = pr_client.delete("/api/preview/cache",
                                 headers={"Content-Type": "application/json"})
        assert resp.status_code == 403


# =====================================================================
# 35. Route smoke tests — GET /api/preview/pipeline/status
# =====================================================================
class TestPipelineStatusRoute:
    def test_pipeline_status(self, pr_client):
        from opencut.core.gpu_preview_pipeline import reset_pipeline
        reset_pipeline()
        resp = pr_client.get("/api/preview/pipeline/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "gpu" in data
        reset_pipeline()


# =====================================================================
# 36. Route smoke tests — GET /api/preview/compare/metrics (validation)
# =====================================================================
class TestCompareMetricsRoute:
    def test_missing_params(self, pr_client):
        resp = pr_client.get("/api/preview/compare/metrics")
        assert resp.status_code == 400


# =====================================================================
# 37. Scope generate_scope_from_frame
# =====================================================================
class TestScopeFromFrame:
    def test_file_not_found(self):
        from opencut.core.realtime_scopes import generate_scope_from_frame
        with pytest.raises(FileNotFoundError):
            generate_scope_from_frame("/nonexistent.jpg", "histogram")

    def test_unknown_type(self, temp_jpg):
        from opencut.core.realtime_scopes import generate_scope_from_frame
        with pytest.raises(ValueError, match="Unknown scope"):
            generate_scope_from_frame(temp_jpg, "nonexistent_scope")


# =====================================================================
# 38. Edge cases and parameter validation
# =====================================================================
class TestEdgeCases:
    def test_validate_strength_boundary(self):
        from opencut.core.live_preview import _validate_strength
        assert _validate_strength({"strength": 0.0}) == 0.0
        assert _validate_strength({"strength": 1.0}) == 1.0

    def test_color_grade_clamps(self):
        """Verify colour grade params are clamped to valid ranges."""
        # We just verify the function builds without crashing
        from opencut.core.live_preview import effect_color_grade
        with patch("opencut.core.live_preview._sp.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            effect_color_grade("/in.jpg", "/out.jpg", {
                "brightness": 5.0,  # should clamp to 1.0
                "contrast": -1.0,   # should clamp to 0.1
                "saturation": 10.0, # should clamp to 3.0
                "gamma": 0.0,       # should clamp to 0.1
                "temperature": 5.0, # should clamp to 1.0
            })
            # Verify FFmpeg was called (no crash)
            assert mock_run.called

    def test_upscale_factor_clamp(self):
        from opencut.core.live_preview import effect_upscale_preview
        with patch("opencut.core.live_preview._sp.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            effect_upscale_preview("/in.jpg", "/out.jpg", {"factor": 10})
            # Factor should be clamped to 4
            cmd = mock_run.call_args[0][0]
            vf_idx = cmd.index("-vf")
            assert "4" in cmd[vf_idx + 1]


# =====================================================================
# 39. Constants validation
# =====================================================================
class TestConstants:
    def test_scope_types(self):
        from opencut.core.realtime_scopes import SCOPE_TYPES
        assert len(SCOPE_TYPES) == 5

    def test_compare_modes(self):
        from opencut.core.ab_compare import COMPARE_MODES
        assert len(COMPARE_MODES) == 6

    def test_legal_range_values(self):
        from opencut.core.realtime_scopes import LEGAL_BLACK, LEGAL_WHITE
        assert LEGAL_BLACK == 16
        assert LEGAL_WHITE == 235


# =====================================================================
# 40. Integration-style: scope presets contain valid scope types
# =====================================================================
class TestPresetIntegrity:
    def test_all_preset_scopes_valid(self):
        from opencut.core.realtime_scopes import SCOPE_PRESETS, SCOPE_TYPES
        for preset_name, preset in SCOPE_PRESETS.items():
            for scope in preset["scopes"]:
                assert scope in SCOPE_TYPES, (
                    f"Preset '{preset_name}' references invalid scope '{scope}'"
                )


# =====================================================================
# 41. Pipeline convenience functions
# =====================================================================
class TestPipelineConvenience:
    @patch("opencut.core.gpu_preview_pipeline._sp.run")
    def test_render_single_preview(self, mock_run, temp_video):
        from opencut.core.gpu_preview_pipeline import (
            render_single_preview, reset_pipeline,
        )
        reset_pipeline()

        def _side_effect(cmd, **kwargs):
            out = cmd[-1]
            if out.endswith(".jpg"):
                with open(out, "wb") as f:
                    f.write(b"\xff\xd8fake\xff\xd9")
            return MagicMock(returncode=0)

        mock_run.side_effect = _side_effect
        result = render_single_preview(temp_video, timestamp=0.5)
        assert "timestamp" in result
        assert result["timestamp"] == 0.5
        reset_pipeline()


# =====================================================================
# 42. Wipe frame generation
# =====================================================================
class TestWipeFrame:
    def test_file_not_found(self):
        from opencut.core.ab_compare import generate_wipe_frame
        with pytest.raises(FileNotFoundError):
            generate_wipe_frame("/nonexistent.mp4", "/also_nonexistent.mp4")


# =====================================================================
# 43. Cache manager metadata persistence
# =====================================================================
class TestCacheMetadata:
    def test_save_and_load(self):
        from opencut.core.preview_cache import PreviewCacheManager
        with tempfile.TemporaryDirectory() as td:
            mgr = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                       cleanup_interval=9999)
            try:
                f = _make_temp_file(suffix=".jpg")
                mgr.put("persist_key", f)
                mgr.save_metadata()

                # Create new manager pointing to same dir
                mgr2 = PreviewCacheManager(cache_dir=td, max_size_mb=10,
                                            cleanup_interval=9999)
                try:
                    result = mgr2.get("persist_key")
                    assert result is not None
                finally:
                    mgr2.shutdown()
            finally:
                mgr.shutdown()


# =====================================================================
# 44. Pipeline with explicit timestamps
# =====================================================================
class TestPipelineTimestamps:
    @patch("opencut.core.gpu_preview_pipeline._sp.run")
    def test_explicit_timestamps(self, mock_run, temp_video):
        from opencut.core.gpu_preview_pipeline import PreviewPipeline

        def _side_effect(cmd, **kwargs):
            out = cmd[-1]
            if out.endswith(".jpg"):
                with open(out, "wb") as f:
                    f.write(b"\xff\xd8fake\xff\xd9")
            return MagicMock(returncode=0)

        mock_run.side_effect = _side_effect
        p = PreviewPipeline(use_gpu=False)
        result = p.render_batch(temp_video, timestamps=[0.5, 1.5, 2.5])
        assert len(result.frames) == 3
        assert result.frames[0].timestamp == 0.5
        assert result.frames[2].timestamp == 2.5


# =====================================================================
# 45. Pure-Python metrics fallback
# =====================================================================
class TestPureMetrics:
    def test_metrics_pure_identical(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        from opencut.core.ab_compare import _metrics_pure
        img = Image.new("RGB", (5, 5), (128, 128, 128))
        m = _metrics_pure(img, img)
        assert m.mse == 0.0
        assert m.psnr == 100.0

    def test_metrics_pure_different(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        from opencut.core.ab_compare import _metrics_pure
        a = Image.new("RGB", (5, 5), (100, 100, 100))
        b = Image.new("RGB", (5, 5), (200, 200, 200))
        m = _metrics_pure(a, b)
        assert m.mse > 0
        assert m.psnr < 100.0
        assert m.color_delta > 0
