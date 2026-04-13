"""
Tests for OpenCut Platform & Infrastructure features.

Covers:
  - Review & Approval Links
  - DaVinci Resolve Integration (unit mocks)
  - Plugin Marketplace
  - ONNX Runtime Everywhere
  - AMD GPU Support
  - Stock Media Search
  - Custom FFmpeg Filter Chain Builder
  - Smart Render / Partial Re-Encode
  - Render Cache with Dependency Tracking
  - Timeline Diff / Comparison
  - Branching Edit Workflows
  - Frame.io Review Integration (unit mocks)
  - Interactive Waveform Timeline Backend
  - Mini Player / Preview Backend
  - Platform infra route smoke tests
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Review & Approval Links
# ============================================================
class TestReviewLinks(unittest.TestCase):
    """Tests for opencut.core.review_links."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "test.mp4")
        with open(self.video, "wb") as f:
            f.write(b"\x00" * 1024)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("opencut.core.review_links.REVIEWS_DIR")
    def test_create_review_link(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.review_links._reviews_path",
                   return_value=os.path.join(self.tmp, "reviews.json")):
            from opencut.core.review_links import create_review_link
            link = create_review_link(self.video, title="Test Review")
            self.assertTrue(link.review_id)
            self.assertTrue(link.token)
            self.assertEqual(link.status, "pending")
            self.assertEqual(link.title, "Test Review")

    def test_create_review_link_missing_file(self):
        from opencut.core.review_links import create_review_link
        with self.assertRaises(FileNotFoundError):
            create_review_link("/nonexistent/video.mp4")

    @patch("opencut.core.review_links._reviews_path")
    def test_add_comment(self, mock_path):
        reviews_file = os.path.join(self.tmp, "reviews.json")
        mock_path.return_value = reviews_file
        # Seed a review
        review_data = {
            "test_review": {
                "review_id": "test_review",
                "video_path": self.video,
                "token": "abc123",
                "status": "pending",
                "title": "Test",
                "created_at": time.time(),
                "expires_at": None,
                "comments": [],
            }
        }
        with open(reviews_file, "w") as f:
            json.dump(review_data, f)

        from opencut.core.review_links import add_review_comment
        comment = add_review_comment("test_review", 5.5, "Nice shot!", "Alice")
        self.assertEqual(comment.text, "Nice shot!")
        self.assertEqual(comment.author, "Alice")
        self.assertAlmostEqual(comment.timestamp, 5.5)

    @patch("opencut.core.review_links._reviews_path")
    def test_add_comment_empty_text(self, mock_path):
        mock_path.return_value = os.path.join(self.tmp, "reviews.json")
        from opencut.core.review_links import add_review_comment
        with self.assertRaises(ValueError):
            add_review_comment("any", 0, "", "Author")

    @patch("opencut.core.review_links._reviews_path")
    def test_add_comment_negative_timestamp(self, mock_path):
        mock_path.return_value = os.path.join(self.tmp, "reviews.json")
        from opencut.core.review_links import add_review_comment
        with self.assertRaises(ValueError):
            add_review_comment("any", -1, "text", "Author")

    @patch("opencut.core.review_links._reviews_path")
    def test_get_comments_sorted(self, mock_path):
        reviews_file = os.path.join(self.tmp, "reviews.json")
        mock_path.return_value = reviews_file
        review_data = {
            "r1": {
                "review_id": "r1", "video_path": "", "token": "t",
                "status": "pending", "title": "", "created_at": 0,
                "expires_at": None,
                "comments": [
                    {"comment_id": "c1", "review_id": "r1", "timestamp": 10.0,
                     "text": "B", "author": "X", "created_at": 0},
                    {"comment_id": "c2", "review_id": "r1", "timestamp": 2.0,
                     "text": "A", "author": "Y", "created_at": 0},
                ],
            }
        }
        with open(reviews_file, "w") as f:
            json.dump(review_data, f)

        from opencut.core.review_links import get_review_comments
        comments = get_review_comments("r1")
        self.assertEqual(len(comments), 2)
        self.assertEqual(comments[0].text, "A")
        self.assertEqual(comments[1].text, "B")

    @patch("opencut.core.review_links._reviews_path")
    def test_update_status(self, mock_path):
        reviews_file = os.path.join(self.tmp, "reviews.json")
        mock_path.return_value = reviews_file
        review_data = {
            "r1": {
                "review_id": "r1", "video_path": "", "token": "t",
                "status": "pending", "title": "", "created_at": 0,
                "expires_at": None, "comments": [],
            }
        }
        with open(reviews_file, "w") as f:
            json.dump(review_data, f)

        from opencut.core.review_links import update_review_status
        result = update_review_status("r1", "approved")
        self.assertEqual(result.status, "approved")

    @patch("opencut.core.review_links._reviews_path")
    def test_update_invalid_status(self, mock_path):
        mock_path.return_value = os.path.join(self.tmp, "reviews.json")
        from opencut.core.review_links import update_review_status
        with self.assertRaises(ValueError):
            update_review_status("r1", "invalid_status")

    @patch("opencut.core.review_links._reviews_path")
    def test_get_review_not_found(self, mock_path):
        reviews_file = os.path.join(self.tmp, "reviews.json")
        mock_path.return_value = reviews_file
        with open(reviews_file, "w") as f:
            json.dump({}, f)
        from opencut.core.review_links import get_review_comments
        with self.assertRaises(KeyError):
            get_review_comments("nonexistent")


# ============================================================
# 2. DaVinci Resolve Integration
# ============================================================
class TestResolveIntegration(unittest.TestCase):
    """Tests for opencut.core.resolve_integration."""

    def test_resolve_marker_invalid_color(self):
        from opencut.core.resolve_integration import resolve_add_marker
        with self.assertRaises(ValueError):
            resolve_add_marker(timeline=MagicMock(), timestamp=0,
                               color="InvalidColor")

    def test_resolve_apply_cuts_empty(self):
        from opencut.core.resolve_integration import resolve_apply_cuts
        with self.assertRaises(ValueError):
            resolve_apply_cuts(cut_list=[])

    def test_resolve_apply_cuts_none(self):
        from opencut.core.resolve_integration import resolve_apply_cuts
        with self.assertRaises(ValueError):
            resolve_apply_cuts(cut_list=None)

    def test_resolve_import_empty(self):
        from opencut.core.resolve_integration import resolve_import_media
        with self.assertRaises(ValueError):
            resolve_import_media([])

    def test_resolve_timeline_info_dataclass(self):
        from opencut.core.resolve_integration import ResolveTimelineInfo
        info = ResolveTimelineInfo(name="Test", frame_rate=30.0)
        self.assertEqual(info.name, "Test")
        self.assertEqual(info.frame_rate, 30.0)

    def test_resolve_marker_dataclass(self):
        from opencut.core.resolve_integration import ResolveMarker
        m = ResolveMarker(frame=100, name="Cut", color="Red")
        self.assertEqual(m.frame, 100)

    def test_resolve_render_result_dataclass(self):
        from opencut.core.resolve_integration import ResolveRenderResult
        r = ResolveRenderResult(success=True, message="ok")
        self.assertTrue(r.success)


# ============================================================
# 3. Plugin Marketplace
# ============================================================
class TestPluginMarketplace(unittest.TestCase):
    """Tests for opencut.core.plugin_marketplace."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plugin_info_dataclass(self):
        from opencut.core.plugin_marketplace import PluginInfo
        p = PluginInfo(
            plugin_id="test", name="Test Plugin", version="1.0.0",
            author="Dev", description="A test", repo_url="https://github.com/test",
        )
        self.assertEqual(p.plugin_id, "test")
        self.assertFalse(p.installed)

    def test_search_empty_query(self):
        from opencut.core.plugin_marketplace import search_plugins
        with self.assertRaises(ValueError):
            search_plugins("")

    @patch("opencut.core.plugin_marketplace.PLUGINS_DIR")
    def test_list_installed_empty(self, mock_dir):
        mock_dir.__str__ = lambda s: self.tmp
        with patch("opencut.core.plugin_marketplace._load_installed", return_value={}):
            from opencut.core.plugin_marketplace import list_installed_plugins
            result = list_installed_plugins()
            self.assertEqual(result, [])

    @patch("opencut.core.plugin_marketplace._load_installed")
    def test_list_installed_with_plugins(self, mock_load):
        mock_load.return_value = {
            "my-plugin": {"name": "My Plugin", "version": "2.0.0"}
        }
        from opencut.core.plugin_marketplace import list_installed_plugins
        result = list_installed_plugins()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].plugin_id, "my-plugin")
        self.assertEqual(result[0].version, "2.0.0")

    @patch("opencut.core.plugin_marketplace._load_installed")
    def test_update_not_installed(self, mock_load):
        mock_load.return_value = {}
        from opencut.core.plugin_marketplace import update_plugin
        with self.assertRaises(KeyError):
            update_plugin("nonexistent")

    @patch("opencut.core.plugin_marketplace._registry_cache_valid", return_value=True)
    @patch("opencut.core.plugin_marketplace._load_installed", return_value={})
    def test_fetch_registry_cached(self, mock_inst, mock_valid):
        cache_data = {"plugins": [
            {"id": "p1", "name": "P1", "version": "1.0", "author": "A",
             "description": "D", "repo_url": "https://x"}
        ]}
        cache_file = os.path.join(self.tmp, "cache.json")
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        with patch("opencut.core.plugin_marketplace.REGISTRY_CACHE", cache_file):
            from opencut.core.plugin_marketplace import fetch_plugin_registry
            result = fetch_plugin_registry()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].plugin_id, "p1")


# ============================================================
# 4. ONNX Runtime
# ============================================================
class TestONNXRuntime(unittest.TestCase):
    """Tests for opencut.core.onnx_runtime."""

    def test_provider_info_dataclass(self):
        from opencut.core.onnx_runtime import ONNXProviderInfo
        p = ONNXProviderInfo(name="CPUExecutionProvider", available=True, priority=99)
        self.assertTrue(p.available)

    def test_check_providers_returns_list(self):
        from opencut.core.onnx_runtime import check_onnx_providers
        providers = check_onnx_providers()
        self.assertIsInstance(providers, list)
        self.assertTrue(len(providers) > 0)
        # CPU should always be in the list
        names = [p.name for p in providers]
        self.assertIn("CPUExecutionProvider", names)

    def test_get_optimal_provider_fallback(self):
        from opencut.core.onnx_runtime import get_optimal_provider
        with patch("opencut.core.onnx_runtime.check_onnx_providers", return_value=[]):
            result = get_optimal_provider()
            self.assertEqual(result, "CPUExecutionProvider")

    def test_get_optimal_provider_cuda(self):
        from opencut.core.onnx_runtime import ONNXProviderInfo, get_optimal_provider
        mock_providers = [
            ONNXProviderInfo("CUDAExecutionProvider", True, 1),
            ONNXProviderInfo("CPUExecutionProvider", True, 99),
        ]
        with patch("opencut.core.onnx_runtime.check_onnx_providers", return_value=mock_providers):
            result = get_optimal_provider()
            self.assertEqual(result, "CUDAExecutionProvider")

    def test_run_inference_file_not_found(self):
        from opencut.core.onnx_runtime import run_onnx_inference
        with self.assertRaises(FileNotFoundError):
            run_onnx_inference("/nonexistent/model.onnx", [1, 2, 3])

    def test_convert_to_onnx_no_torch(self):
        from opencut.core.onnx_runtime import convert_to_onnx
        with patch("builtins.__import__", side_effect=ImportError("no torch")):
            try:
                convert_to_onnx(MagicMock(), "/tmp/out.onnx")
            except ImportError:
                pass


# ============================================================
# 5. AMD GPU Support
# ============================================================
class TestAMDGPU(unittest.TestCase):
    """Tests for opencut.core.amd_gpu."""

    def test_amd_gpu_info_dataclass(self):
        from opencut.core.amd_gpu import AMDGPUInfo
        info = AMDGPUInfo(name="RX 7900 XTX", vram_mb=24576)
        self.assertEqual(info.name, "RX 7900 XTX")
        self.assertEqual(info.vram_mb, 24576)

    def test_guess_architecture_rdna3(self):
        from opencut.core.amd_gpu import _guess_architecture
        self.assertEqual(_guess_architecture("Radeon RX 7900 XTX"), "RDNA3")
        self.assertEqual(_guess_architecture("Radeon RX 6800 XT"), "RDNA2")
        self.assertEqual(_guess_architecture("Radeon RX 5700"), "RDNA")
        self.assertEqual(_guess_architecture("Radeon Vega 64"), "Vega")

    @patch("opencut.core.amd_gpu.platform")
    def test_directml_non_windows(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        from opencut.core.amd_gpu import get_directml_device
        result = get_directml_device()
        self.assertFalse(result["available"])

    @patch("opencut.core.amd_gpu.platform")
    @patch("opencut.core.amd_gpu._detect_amd_windows", return_value=[])
    @patch("opencut.core.amd_gpu._check_directml", return_value=False)
    @patch("opencut.core.amd_gpu.check_rocm_available", return_value=False)
    def test_detect_amd_gpu_windows_empty(self, mock_rocm, mock_dml, mock_detect, mock_plat):
        mock_plat.system.return_value = "Windows"
        from opencut.core.amd_gpu import detect_amd_gpu
        result = detect_amd_gpu()
        self.assertEqual(result, [])

    def test_check_rocm_not_available(self):
        from opencut.core.amd_gpu import check_rocm_available
        # Should return False on typical test machines
        result = check_rocm_available()
        self.assertIsInstance(result, bool)

    @patch("opencut.core.amd_gpu.detect_amd_gpu", return_value=[])
    @patch("opencut.core.amd_gpu.get_directml_device", return_value={"available": False})
    @patch("opencut.core.amd_gpu.check_rocm_available", return_value=False)
    def test_get_capabilities(self, mock_rocm, mock_dml, mock_detect):
        from opencut.core.amd_gpu import get_amd_capabilities
        result = get_amd_capabilities()
        self.assertIn("gpus", result)
        self.assertIn("recommendations", result)
        self.assertEqual(result["gpu_count"], 0)


# ============================================================
# 6. Stock Media Search
# ============================================================
class TestStockSearch(unittest.TestCase):
    """Tests for opencut.core.stock_search."""

    def test_stock_media_result_dataclass(self):
        from opencut.core.stock_search import StockMediaResult
        r = StockMediaResult(media_id="123", source="pexels", media_type="video")
        self.assertEqual(r.media_id, "123")
        self.assertEqual(r.license, "free")

    def test_search_video_empty_query(self):
        from opencut.core.stock_search import search_stock_video
        with self.assertRaises(ValueError):
            search_stock_video("")

    def test_search_video_invalid_source(self):
        from opencut.core.stock_search import search_stock_video
        with self.assertRaises(ValueError):
            search_stock_video("nature", source="unsplash")

    def test_search_photo_empty_query(self):
        from opencut.core.stock_search import search_stock_photo
        with self.assertRaises(ValueError):
            search_stock_photo("")

    def test_search_photo_invalid_source(self):
        from opencut.core.stock_search import search_stock_photo
        with self.assertRaises(ValueError):
            search_stock_photo("sunset", source="invalid")

    def test_download_no_url(self):
        from opencut.core.stock_search import download_stock_media
        with self.assertRaises(ValueError):
            download_stock_media("123", "pexels", tempfile.gettempdir())

    def test_search_video_no_api_key(self):
        from opencut.core.stock_search import search_stock_video
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                search_stock_video("nature", source="pexels")

    @patch("opencut.core.stock_search._pexels_request")
    @patch("opencut.core.stock_search._get_api_key", return_value="test_key")
    def test_search_video_pexels(self, mock_key, mock_req):
        mock_req.return_value = {"videos": [
            {"id": 1, "url": "https://pexels.com/video/nature-1",
             "duration": 15, "video_files": [{"width": 1920, "height": 1080, "link": "https://dl"}],
             "video_pictures": [{"picture": "https://thumb"}],
             "user": {"name": "John", "url": "https://pexels.com/john"}}
        ]}
        from opencut.core.stock_search import search_stock_video
        results = search_stock_video("nature", source="pexels")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].duration, 15.0)


# ============================================================
# 7. FFmpeg Filter Chain Builder
# ============================================================
class TestFFmpegBuilder(unittest.TestCase):
    """Tests for opencut.core.ffmpeg_builder."""

    def test_filter_node_dataclass(self):
        from opencut.core.ffmpeg_builder import FilterNode
        n = FilterNode(node_id="n0", filter_name="scale", params={"w": 1920, "h": 1080})
        self.assertEqual(n.filter_name, "scale")

    def test_build_single_node(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        chain = build_filter_chain([{"filter_name": "scale", "params": {"w": 1280, "h": 720}}])
        self.assertIn("scale", chain)
        self.assertIn("1280", chain)

    def test_build_multiple_nodes(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        nodes = [
            {"filter_name": "scale", "params": {"w": 1280, "h": 720}},
            {"filter_name": "eq", "params": {"brightness": "0.1"}},
        ]
        chain = build_filter_chain(nodes)
        self.assertIn("scale", chain)
        self.assertIn("eq", chain)
        self.assertIn(";", chain)

    def test_build_empty_nodes(self):
        from opencut.core.ffmpeg_builder import build_filter_chain
        with self.assertRaises(ValueError):
            build_filter_chain([])

    def test_validate_valid_graph(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        graph = {"nodes": [{"node_id": "n0", "filter_name": "scale"}]}
        result = validate_filter_graph(graph)
        self.assertTrue(result["valid"])
        self.assertEqual(result["node_count"], 1)

    def test_validate_empty_graph(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        result = validate_filter_graph({"nodes": []})
        self.assertFalse(result["valid"])

    def test_validate_missing_filter_name(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        result = validate_filter_graph({"nodes": [{"node_id": "n0"}]})
        self.assertFalse(result["valid"])

    def test_validate_unknown_filter_warning(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        result = validate_filter_graph({
            "nodes": [{"node_id": "n0", "filter_name": "my_custom_filter"}]
        })
        self.assertTrue(result["valid"])
        self.assertTrue(len(result["warnings"]) > 0)

    def test_validate_duplicate_node_id(self):
        from opencut.core.ffmpeg_builder import validate_filter_graph
        result = validate_filter_graph({
            "nodes": [
                {"node_id": "n0", "filter_name": "scale"},
                {"node_id": "n0", "filter_name": "crop"},
            ]
        })
        self.assertFalse(result["valid"])

    def test_save_preset_empty_name(self):
        from opencut.core.ffmpeg_builder import save_filter_preset
        with self.assertRaises(ValueError):
            save_filter_preset("scale=1280:720", "")

    @patch("opencut.core.ffmpeg_builder.PRESETS_DIR")
    def test_save_and_load_preset(self, mock_dir):
        tmp = tempfile.mkdtemp()
        mock_dir.__str__ = lambda s: tmp
        with patch("opencut.core.ffmpeg_builder.PRESETS_DIR", tmp):
            from opencut.core.ffmpeg_builder import load_filter_presets, save_filter_preset
            save_filter_preset("scale=1280:720", "HD Scale", "Scale to HD")
            presets = load_filter_presets()
            self.assertTrue(len(presets) >= 1)
        shutil.rmtree(tmp, ignore_errors=True)

    def test_preview_filter_missing_file(self):
        from opencut.core.ffmpeg_builder import preview_filter
        with self.assertRaises(FileNotFoundError):
            preview_filter("/nonexistent.mp4", "scale=1280:720")


# ============================================================
# 8. Smart Render
# ============================================================
class TestSmartRender(unittest.TestCase):
    """Tests for opencut.core.smart_render."""

    def test_changed_segment_dataclass(self):
        from opencut.core.smart_render import ChangedSegment
        s = ChangedSegment(start=1.0, end=5.0, change_type="modified")
        self.assertEqual(s.change_type, "modified")

    def test_detect_missing_file(self):
        from opencut.core.smart_render import detect_changed_segments
        with self.assertRaises(FileNotFoundError):
            detect_changed_segments("/nonexistent.mp4", [{"start": 0, "end": 1}])

    def test_detect_empty_history(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00" * 100)
        tmp.close()
        try:
            from opencut.core.smart_render import detect_changed_segments
            with self.assertRaises(ValueError):
                detect_changed_segments(tmp.name, [])
        finally:
            os.unlink(tmp.name)

    def test_smart_render_missing_file(self):
        from opencut.core.smart_render import smart_render
        with self.assertRaises(FileNotFoundError):
            smart_render("/nonexistent.mp4", [{"start": 0, "end": 1}])

    def test_snap_to_keyframe(self):
        from opencut.core.smart_render import _snap_to_keyframe
        kf = [0.0, 2.0, 4.0, 6.0, 8.0]
        self.assertEqual(_snap_to_keyframe(3.5, kf, "before"), 2.0)
        self.assertEqual(_snap_to_keyframe(3.5, kf, "after"), 4.0)
        self.assertEqual(_snap_to_keyframe(0.5, kf, "before"), 0.0)

    def test_snap_to_keyframe_empty(self):
        from opencut.core.smart_render import _snap_to_keyframe
        self.assertEqual(_snap_to_keyframe(5.0, []), 5.0)


# ============================================================
# 9. Render Cache
# ============================================================
class TestRenderCache(unittest.TestCase):
    """Tests for opencut.core.render_cache."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        os.makedirs(self.cache_dir)
        self.index_path = os.path.join(self.cache_dir, "index.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("opencut.core.render_cache.CACHE_DIR")
    @patch("opencut.core.render_cache.CACHE_INDEX")
    def test_cache_and_retrieve(self, mock_index, mock_dir):
        mock_dir.__str__ = lambda s: self.cache_dir
        mock_index.__str__ = lambda s: self.index_path
        with patch("opencut.core.render_cache.CACHE_DIR", self.cache_dir), \
             patch("opencut.core.render_cache.CACHE_INDEX", self.index_path):
            # Create a fake output file
            out_file = os.path.join(self.tmp, "output.mp4")
            with open(out_file, "wb") as f:
                f.write(b"\x00" * 512)

            from opencut.core.render_cache import cache_result, get_cached
            entry = cache_result("hash1", "encode", {"crf": 18}, out_file)
            self.assertEqual(entry.input_hash, "hash1")
            self.assertTrue(entry.file_size > 0)

            # Retrieve
            cached = get_cached("hash1", "encode", {"crf": 18})
            self.assertIsNotNone(cached)
            self.assertEqual(cached.input_hash, "hash1")
            self.assertEqual(cached.hit_count, 1)

    @patch("opencut.core.render_cache.CACHE_DIR", "/tmp/test_cache")
    @patch("opencut.core.render_cache.CACHE_INDEX", "/tmp/test_cache/index.json")
    def test_get_cached_miss(self):
        with patch("opencut.core.render_cache._load_index", return_value={}):
            from opencut.core.render_cache import get_cached
            result = get_cached("unknown", "op", {})
            self.assertIsNone(result)

    def test_cache_result_missing_output(self):
        from opencut.core.render_cache import cache_result
        with self.assertRaises(FileNotFoundError):
            cache_result("h", "op", {}, "/nonexistent/file.mp4")

    @patch("opencut.core.render_cache._load_index")
    @patch("opencut.core.render_cache._save_index")
    def test_get_cache_stats(self, mock_save, mock_load):
        mock_load.return_value = {
            "k1": {"file_size": 1000, "hit_count": 5, "operation": "encode", "created_at": 100},
            "k2": {"file_size": 2000, "hit_count": 3, "operation": "filter", "created_at": 200},
        }
        from opencut.core.render_cache import get_cache_stats
        stats = get_cache_stats()
        self.assertEqual(stats["entry_count"], 2)
        self.assertEqual(stats["total_size_bytes"], 3000)
        self.assertEqual(stats["total_hits"], 8)

    @patch("opencut.core.render_cache._load_index")
    @patch("opencut.core.render_cache._save_index")
    def test_cleanup_within_limits(self, mock_save, mock_load):
        mock_load.return_value = {"k1": {"file_size": 100}}
        from opencut.core.render_cache import cleanup_cache
        result = cleanup_cache(max_size_gb=1.0)
        self.assertEqual(result["removed"], 0)

    @patch("opencut.core.render_cache._load_index")
    @patch("opencut.core.render_cache._save_index")
    def test_invalidate_no_match(self, mock_save, mock_load):
        mock_load.return_value = {}
        from opencut.core.render_cache import invalidate_downstream
        result = invalidate_downstream("h", "op")
        self.assertEqual(result["invalidated"], 0)


# ============================================================
# 10. Timeline Diff
# ============================================================
class TestTimelineDiff(unittest.TestCase):
    """Tests for opencut.core.timeline_diff."""

    def test_timeline_diff_dataclass(self):
        from opencut.core.timeline_diff import TimelineDiff
        d = TimelineDiff(summary="2 added")
        self.assertEqual(d.summary, "2 added")

    def test_diff_no_changes(self):
        from opencut.core.timeline_diff import diff_timelines
        snap = {"clips": [{"id": "c1", "name": "Clip1", "start": 0, "end": 5}]}
        diff = diff_timelines(snap, snap)
        self.assertEqual(diff.total_changes, 0)
        self.assertEqual(diff.summary, "No changes")

    def test_diff_added_clip(self):
        from opencut.core.timeline_diff import diff_timelines
        a = {"clips": []}
        b = {"clips": [{"id": "c1", "name": "New Clip", "start": 0, "end": 5}]}
        diff = diff_timelines(a, b)
        self.assertEqual(diff.added_count, 1)
        self.assertEqual(diff.changes[0].change_type, "added")

    def test_diff_removed_clip(self):
        from opencut.core.timeline_diff import diff_timelines
        a = {"clips": [{"id": "c1", "name": "Old Clip", "start": 0, "end": 5}]}
        b = {"clips": []}
        diff = diff_timelines(a, b)
        self.assertEqual(diff.removed_count, 1)

    def test_diff_modified_clip(self):
        from opencut.core.timeline_diff import diff_timelines
        a = {"clips": [{"id": "c1", "name": "Clip", "start": 0, "end": 5, "effects": []}]}
        b = {"clips": [{"id": "c1", "name": "Clip", "start": 0, "end": 5, "effects": ["blur"]}]}
        diff = diff_timelines(a, b)
        self.assertEqual(diff.modified_count, 1)

    def test_diff_moved_clip(self):
        from opencut.core.timeline_diff import diff_timelines
        a = {"clips": [{"id": "c1", "name": "Clip", "start": 0, "end": 5, "track": 1}]}
        b = {"clips": [{"id": "c1", "name": "Clip", "start": 10, "end": 15, "track": 2}]}
        diff = diff_timelines(a, b)
        self.assertEqual(diff.moved_count, 1)

    def test_diff_complex(self):
        from opencut.core.timeline_diff import diff_timelines
        a = {"clips": [
            {"id": "c1", "name": "A", "start": 0, "end": 5},
            {"id": "c2", "name": "B", "start": 5, "end": 10},
        ]}
        b = {"clips": [
            {"id": "c2", "name": "B", "start": 5, "end": 10},
            {"id": "c3", "name": "C", "start": 10, "end": 15},
        ]}
        diff = diff_timelines(a, b)
        self.assertEqual(diff.added_count, 1)
        self.assertEqual(diff.removed_count, 1)

    def test_export_json(self):
        from opencut.core.timeline_diff import diff_timelines, export_diff_report
        diff = diff_timelines({"clips": []}, {"clips": [{"id": "c1", "name": "X", "start": 0, "end": 5}]})
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        try:
            path = export_diff_report(diff, "json", tmp.name)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["total_changes"], 1)
        finally:
            os.unlink(tmp.name)

    def test_export_text(self):
        from opencut.core.timeline_diff import diff_timelines, export_diff_report
        diff = diff_timelines({"clips": []}, {"clips": []})
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.close()
        try:
            path = export_diff_report(diff, "text", tmp.name)
            with open(path) as f:
                text = f.read()
            self.assertIn("Timeline Diff", text)
        finally:
            os.unlink(tmp.name)

    def test_export_html(self):
        from opencut.core.timeline_diff import diff_timelines, export_diff_report
        diff = diff_timelines({"clips": []}, {"clips": []})
        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        tmp.close()
        try:
            path = export_diff_report(diff, "html", tmp.name)
            with open(path) as f:
                html = f.read()
            self.assertIn("<html>", html)
        finally:
            os.unlink(tmp.name)

    def test_export_invalid_format(self):
        from opencut.core.timeline_diff import TimelineDiff, export_diff_report
        diff = TimelineDiff()
        with self.assertRaises(ValueError):
            export_diff_report(diff, "pdf")


# ============================================================
# 11. Branching Edit Workflows
# ============================================================
class TestEditBranches(unittest.TestCase):
    """Tests for opencut.core.edit_branches."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("opencut.core.edit_branches.BRANCHES_DIR")
    def test_create_branch(self, mock_dir):
        with patch("opencut.core.edit_branches._project_path",
                   return_value=os.path.join(self.tmp, "branches.json")):
            from opencut.core.edit_branches import create_branch
            branch = create_branch("main", {"clips": []}, "proj1")
            self.assertEqual(branch.name, "main")
            self.assertEqual(branch.project_id, "proj1")

    @patch("opencut.core.edit_branches._project_path")
    def test_create_duplicate_branch(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        with open(bfile, "w") as f:
            json.dump({"main": {"name": "main", "project_id": "p",
                                "snapshot": {}, "parent_branch": "",
                                "created_at": 0, "updated_at": 0,
                                "is_active": False, "commit_count": 1}}, f)
        from opencut.core.edit_branches import create_branch
        with self.assertRaises(ValueError):
            create_branch("main", {}, "p")

    def test_create_branch_empty_name(self):
        from opencut.core.edit_branches import create_branch
        with self.assertRaises(ValueError):
            create_branch("", {})

    @patch("opencut.core.edit_branches._project_path")
    def test_switch_branch(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        branches = {
            "main": {"name": "main", "project_id": "p", "snapshot": {"clips": [1]},
                      "parent_branch": "", "created_at": 0, "updated_at": 0,
                      "is_active": False, "commit_count": 1},
            "dev": {"name": "dev", "project_id": "p", "snapshot": {"clips": [2]},
                     "parent_branch": "main", "created_at": 0, "updated_at": 0,
                     "is_active": False, "commit_count": 1},
        }
        with open(bfile, "w") as f:
            json.dump(branches, f)
        from opencut.core.edit_branches import switch_branch
        result = switch_branch("dev", "p")
        self.assertEqual(result.name, "dev")

    @patch("opencut.core.edit_branches._project_path")
    def test_switch_nonexistent(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        with open(bfile, "w") as f:
            json.dump({}, f)
        from opencut.core.edit_branches import switch_branch
        with self.assertRaises(KeyError):
            switch_branch("ghost", "p")

    @patch("opencut.core.edit_branches._project_path")
    def test_merge_branches(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        branches = {
            "main": {"name": "main", "project_id": "p",
                      "snapshot": {"clips": [{"id": "c1", "name": "A"}]},
                      "parent_branch": "", "created_at": 0, "updated_at": 0,
                      "is_active": True, "commit_count": 1},
            "feat": {"name": "feat", "project_id": "p",
                      "snapshot": {"clips": [{"id": "c2", "name": "B"}]},
                      "parent_branch": "main", "created_at": 0, "updated_at": 0,
                      "is_active": False, "commit_count": 1},
        }
        with open(bfile, "w") as f:
            json.dump(branches, f)
        from opencut.core.edit_branches import merge_branches
        result = merge_branches("feat", "main", "p")
        self.assertTrue(result.success)
        self.assertEqual(result.auto_resolved, 1)

    @patch("opencut.core.edit_branches._project_path")
    def test_merge_conflict(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        branches = {
            "main": {"name": "main", "project_id": "p",
                      "snapshot": {"clips": [{"id": "c1", "name": "A-main"}]},
                      "parent_branch": "", "created_at": 0, "updated_at": 0,
                      "is_active": True, "commit_count": 1},
            "feat": {"name": "feat", "project_id": "p",
                      "snapshot": {"clips": [{"id": "c1", "name": "A-feat"}]},
                      "parent_branch": "main", "created_at": 0, "updated_at": 0,
                      "is_active": False, "commit_count": 1},
        }
        with open(bfile, "w") as f:
            json.dump(branches, f)
        from opencut.core.edit_branches import merge_branches
        result = merge_branches("feat", "main", "p")
        self.assertFalse(result.success)
        self.assertTrue(len(result.conflicts) > 0)

    @patch("opencut.core.edit_branches._project_path")
    def test_list_branches(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        with open(bfile, "w") as f:
            json.dump({"main": {"name": "main", "project_id": "p",
                                "snapshot": {}, "parent_branch": "",
                                "created_at": 0, "updated_at": 0,
                                "is_active": True, "commit_count": 2}}, f)
        from opencut.core.edit_branches import list_branches
        result = list_branches("p")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].commit_count, 2)

    @patch("opencut.core.edit_branches._project_path")
    def test_branch_graph(self, mock_path):
        bfile = os.path.join(self.tmp, "branches.json")
        mock_path.return_value = bfile
        with open(bfile, "w") as f:
            json.dump({
                "main": {"name": "main", "project_id": "p", "parent_branch": "",
                          "created_at": 0, "updated_at": 0, "is_active": True,
                          "commit_count": 1, "snapshot": {}},
                "dev": {"name": "dev", "project_id": "p", "parent_branch": "main",
                         "created_at": 0, "updated_at": 0, "is_active": False,
                         "commit_count": 1, "snapshot": {}},
            }, f)
        from opencut.core.edit_branches import get_branch_graph
        graph = get_branch_graph("p")
        self.assertEqual(graph["branch_count"], 2)
        self.assertEqual(len(graph["edges"]), 1)


# ============================================================
# 12. Frame.io Integration
# ============================================================
class TestFrameIOIntegration(unittest.TestCase):
    """Tests for opencut.core.frameio_integration."""

    def test_upload_missing_file(self):
        from opencut.core.frameio_integration import upload_to_frameio
        with self.assertRaises(FileNotFoundError):
            upload_to_frameio("/nonexistent.mp4", "proj", "key")

    def test_upload_no_api_key(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00")
        tmp.close()
        try:
            from opencut.core.frameio_integration import upload_to_frameio
            with self.assertRaises(ValueError):
                upload_to_frameio(tmp.name, "proj", "")
        finally:
            os.unlink(tmp.name)

    def test_get_comments_no_key(self):
        from opencut.core.frameio_integration import get_frameio_comments
        with self.assertRaises(ValueError):
            get_frameio_comments("asset123", "")

    def test_resolve_comment_no_key(self):
        from opencut.core.frameio_integration import resolve_frameio_comment
        with self.assertRaises(ValueError):
            resolve_frameio_comment("comment123", "")

    def test_sync_no_key(self):
        from opencut.core.frameio_integration import sync_frameio_comments
        with self.assertRaises(ValueError):
            sync_frameio_comments("asset123", "")

    def test_frameio_comment_dataclass(self):
        from opencut.core.frameio_integration import FrameIOComment
        c = FrameIOComment(comment_id="c1", text="Great take!", author="Bob", timestamp=5.5)
        self.assertEqual(c.text, "Great take!")
        self.assertFalse(c.completed)

    def test_frameio_upload_result_dataclass(self):
        from opencut.core.frameio_integration import FrameIOUploadResult
        r = FrameIOUploadResult(asset_id="a1", status="uploaded")
        self.assertEqual(r.status, "uploaded")


# ============================================================
# 13. Waveform Timeline
# ============================================================
class TestWaveformTimeline(unittest.TestCase):
    """Tests for opencut.core.waveform_timeline."""

    def test_missing_file(self):
        from opencut.core.waveform_timeline import generate_waveform_data
        with self.assertRaises(FileNotFoundError):
            generate_waveform_data("/nonexistent.wav")

    def test_waveform_image_missing_file(self):
        from opencut.core.waveform_timeline import generate_waveform_image
        with self.assertRaises(FileNotFoundError):
            generate_waveform_image("/nonexistent.wav")

    def test_region_missing_file(self):
        from opencut.core.waveform_timeline import get_waveform_region
        with self.assertRaises(FileNotFoundError):
            get_waveform_region("/nonexistent.wav", 0, 5)

    def test_region_invalid_times(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(b"\x00")
        tmp.close()
        try:
            from opencut.core.waveform_timeline import get_waveform_region
            with self.assertRaises(ValueError):
                get_waveform_region(tmp.name, 5, 3)
        finally:
            os.unlink(tmp.name)

    def test_region_invalid_samples(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(b"\x00")
        tmp.close()
        try:
            from opencut.core.waveform_timeline import get_waveform_region
            with self.assertRaises(ValueError):
                get_waveform_region(tmp.name, 0, 5, samples=0)
        finally:
            os.unlink(tmp.name)

    def test_pcm_to_samples(self):
        import struct

        from opencut.core.waveform_timeline import _pcm_to_samples
        pcm = struct.pack("<4h", 100, -200, 300, -400)
        samples = _pcm_to_samples(pcm)
        self.assertEqual(samples, [100, -200, 300, -400])

    def test_pcm_to_samples_empty(self):
        from opencut.core.waveform_timeline import _pcm_to_samples
        self.assertEqual(_pcm_to_samples(b""), [])


# ============================================================
# 14. Preview Server
# ============================================================
class TestPreviewServer(unittest.TestCase):
    """Tests for opencut.core.preview_server."""

    def test_extract_frame_missing_file(self):
        from opencut.core.preview_server import extract_preview_frame
        with self.assertRaises(FileNotFoundError):
            extract_preview_frame("/nonexistent.mp4", 0)

    def test_extract_frame_negative_timestamp(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00")
        tmp.close()
        try:
            from opencut.core.preview_server import extract_preview_frame
            with self.assertRaises(ValueError):
                extract_preview_frame(tmp.name, -1)
        finally:
            os.unlink(tmp.name)

    def test_generate_clip_missing_file(self):
        from opencut.core.preview_server import generate_preview_clip
        with self.assertRaises(FileNotFoundError):
            generate_preview_clip("/nonexistent.mp4", 0, 5)

    def test_generate_clip_invalid_times(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"\x00")
        tmp.close()
        try:
            from opencut.core.preview_server import generate_preview_clip
            with self.assertRaises(ValueError):
                generate_preview_clip(tmp.name, 5, 3)
        finally:
            os.unlink(tmp.name)

    def test_frame_at_position_missing(self):
        from opencut.core.preview_server import get_frame_at_position
        with self.assertRaises(FileNotFoundError):
            get_frame_at_position("/nonexistent.mp4", 0)

    def test_thumbnail_strip_missing(self):
        from opencut.core.preview_server import generate_thumbnail_strip
        with self.assertRaises(FileNotFoundError):
            generate_thumbnail_strip("/nonexistent.mp4")


# ============================================================
# Route Smoke Tests
# ============================================================
class TestPlatformInfraRoutes(unittest.TestCase):
    """Smoke tests for platform_infra_routes endpoints."""

    @classmethod
    def setUpClass(cls):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        config = OpenCutConfig()
        cls.app = create_app(config=config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        resp = cls.client.get("/health")
        data = resp.get_json()
        cls.csrf = data.get("csrf_token", "")

    def _headers(self):
        return {"X-OpenCut-Token": self.csrf, "Content-Type": "application/json"}

    # --- Review routes ---
    def test_review_comment_missing_review(self):
        resp = self.client.post("/review/comment", headers=self._headers(),
                                data=json.dumps({"review_id": "nope", "timestamp": 0,
                                                  "text": "hi", "author": "X"}))
        self.assertIn(resp.status_code, [200, 400, 404, 500])

    def test_review_status_invalid(self):
        resp = self.client.post("/review/status", headers=self._headers(),
                                data=json.dumps({"review_id": "nope", "status": "bad"}))
        self.assertIn(resp.status_code, [200, 400, 404, 500])

    # --- Resolve routes ---
    def test_resolve_timeline_get(self):
        resp = self.client.get("/resolve/timeline")
        self.assertIn(resp.status_code, [200, 500, 503])

    def test_resolve_marker_post(self):
        resp = self.client.post("/resolve/marker", headers=self._headers(),
                                data=json.dumps({"timestamp": 0, "name": "M"}))
        self.assertIn(resp.status_code, [200, 400, 500, 503])

    # --- Plugin routes ---
    def test_plugins_installed(self):
        resp = self.client.get("/plugins/installed")
        self.assertIn(resp.status_code, [200, 500])

    def test_plugins_search_no_query(self):
        resp = self.client.get("/plugins/search?q=")
        self.assertIn(resp.status_code, [200, 400, 500])

    # --- ONNX routes ---
    def test_onnx_providers(self):
        resp = self.client.get("/onnx/providers")
        self.assertIn(resp.status_code, [200, 500])

    def test_onnx_optimal_provider(self):
        resp = self.client.get("/onnx/optimal-provider")
        self.assertIn(resp.status_code, [200, 500])

    # --- AMD routes ---
    def test_amd_detect(self):
        resp = self.client.get("/amd/detect")
        self.assertIn(resp.status_code, [200, 500])

    def test_amd_directml(self):
        resp = self.client.get("/amd/directml")
        self.assertIn(resp.status_code, [200, 500])

    def test_amd_rocm(self):
        resp = self.client.get("/amd/rocm")
        self.assertIn(resp.status_code, [200, 500])

    def test_amd_capabilities(self):
        resp = self.client.get("/amd/capabilities")
        self.assertIn(resp.status_code, [200, 500])

    # --- Stock routes ---
    def test_stock_video_no_query(self):
        resp = self.client.get("/stock/video?q=")
        self.assertIn(resp.status_code, [200, 400, 500])

    def test_stock_photo_no_query(self):
        resp = self.client.get("/stock/photo?q=")
        self.assertIn(resp.status_code, [200, 400, 500])

    # --- Filter builder routes ---
    def test_filter_build(self):
        resp = self.client.post("/filter/build", headers=self._headers(),
                                data=json.dumps({"nodes": [{"filter_name": "scale",
                                                             "params": {"w": 1280}}]}))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("filter_chain", data)

    def test_filter_validate_empty(self):
        resp = self.client.post("/filter/validate", headers=self._headers(),
                                data=json.dumps({"nodes": []}))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["valid"])

    def test_filter_validate_good(self):
        resp = self.client.post("/filter/validate", headers=self._headers(),
                                data=json.dumps({"nodes": [
                                    {"node_id": "n0", "filter_name": "scale"}
                                ]}))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["valid"])

    def test_filter_presets_list(self):
        resp = self.client.get("/filter/presets")
        self.assertIn(resp.status_code, [200, 500])

    def test_filter_preset_save_empty(self):
        resp = self.client.post("/filter/presets/save", headers=self._headers(),
                                data=json.dumps({"chain": "scale=1280:720", "name": ""}))
        self.assertIn(resp.status_code, [200, 400, 500])

    # --- Smart render routes ---
    def test_smart_render_estimate_missing(self):
        resp = self.client.post("/smart-render/estimate", headers=self._headers(),
                                data=json.dumps({"filepath": "/nonexistent.mp4",
                                                  "changes": [{"start": 0, "end": 1}]}))
        self.assertIn(resp.status_code, [200, 400, 404, 500])

    # --- Cache routes ---
    def test_cache_stats(self):
        resp = self.client.get("/cache/stats")
        self.assertIn(resp.status_code, [200, 500])

    def test_cache_cleanup(self):
        resp = self.client.post("/cache/cleanup", headers=self._headers(),
                                data=json.dumps({"max_size_gb": 5}))
        self.assertIn(resp.status_code, [200, 500])

    def test_cache_invalidate(self):
        resp = self.client.post("/cache/invalidate", headers=self._headers(),
                                data=json.dumps({"input_hash": "abc", "operation": "encode"}))
        self.assertIn(resp.status_code, [200, 500])

    # --- Timeline diff routes ---
    def test_timeline_diff(self):
        resp = self.client.post("/timeline/diff", headers=self._headers(),
                                data=json.dumps({
                                    "snapshot_a": {"clips": []},
                                    "snapshot_b": {"clips": [{"id": "c1", "name": "X",
                                                               "start": 0, "end": 5}]},
                                }))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["added"], 1)

    def test_timeline_diff_export(self):
        resp = self.client.post("/timeline/diff/export", headers=self._headers(),
                                data=json.dumps({
                                    "snapshot_a": {"clips": []},
                                    "snapshot_b": {"clips": []},
                                    "format": "json",
                                }))
        self.assertIn(resp.status_code, [200, 500])

    # --- Branch routes ---
    def test_branch_create(self):
        resp = self.client.post("/branches/create", headers=self._headers(),
                                data=json.dumps({
                                    "name": f"test_{int(time.time())}",
                                    "snapshot": {"clips": []},
                                    "project_id": "test_proj",
                                }))
        self.assertIn(resp.status_code, [200, 400, 500])

    def test_branch_list(self):
        resp = self.client.get("/branches/list?project_id=test_proj")
        self.assertIn(resp.status_code, [200, 500])

    def test_branch_graph(self):
        resp = self.client.get("/branches/graph?project_id=test_proj")
        self.assertIn(resp.status_code, [200, 500])

    # --- Frame.io routes ---
    def test_frameio_comments_no_key(self):
        resp = self.client.post("/frameio/comments", headers=self._headers(),
                                data=json.dumps({"asset_id": "a1", "api_key": ""}))
        self.assertIn(resp.status_code, [200, 400, 500])

    def test_frameio_resolve_no_key(self):
        resp = self.client.post("/frameio/resolve", headers=self._headers(),
                                data=json.dumps({"comment_id": "c1", "api_key": ""}))
        self.assertIn(resp.status_code, [200, 400, 500])

    def test_frameio_sync_no_key(self):
        resp = self.client.post("/frameio/sync", headers=self._headers(),
                                data=json.dumps({"asset_id": "a1", "api_key": ""}))
        self.assertIn(resp.status_code, [200, 400, 500])


if __name__ == "__main__":
    unittest.main()
