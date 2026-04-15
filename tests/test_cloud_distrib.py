"""
Tests for OpenCut Cloud & Distribution features.

Covers:
  - Cloud render (node management, dispatch, failover, local fallback)
  - Platform publish (all platforms, validation, batch export)
  - Content fingerprint (generate, compare, index, search)
  - Render farm (segmentation strategies, aggregation, fault tolerance)
  - Distribution analytics (record, aggregate, cross-platform, export)
  - Cloud & distribution routes (smoke tests)
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# Cloud Render — Node Management
# ============================================================
class TestCloudRenderNodes(unittest.TestCase):
    """Tests for cloud_render node config persistence."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._orig_nodes_file = None

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _patch_nodes_file(self):
        import opencut.core.cloud_render as mod
        self._orig_nodes_file = mod._NODES_FILE
        mod._NODES_FILE = os.path.join(self.tmp_dir, "nodes.json")
        return mod

    def _restore_nodes_file(self, mod):
        if self._orig_nodes_file is not None:
            mod._NODES_FILE = self._orig_nodes_file

    def test_render_node_to_dict(self):
        from opencut.core.cloud_render import RenderNode
        node = RenderNode(name="gpu-1", host="10.0.0.1", port=9090,
                          capabilities=["gpu", "cpu"], max_concurrent=4)
        d = node.to_dict()
        self.assertEqual(d["name"], "gpu-1")
        self.assertEqual(d["host"], "10.0.0.1")
        self.assertIn("gpu", d["capabilities"])

    def test_render_node_from_dict(self):
        from opencut.core.cloud_render import RenderNode
        d = {"name": "node-a", "host": "192.168.1.10", "port": 8080,
             "capabilities": ["cpu"], "max_concurrent": 3}
        node = RenderNode.from_dict(d)
        self.assertEqual(node.name, "node-a")
        self.assertEqual(node.port, 8080)
        self.assertEqual(node.max_concurrent, 3)

    def test_render_node_base_url(self):
        from opencut.core.cloud_render import RenderNode
        node = RenderNode(name="n1", host="10.0.0.5", port=7777)
        self.assertEqual(node.base_url, "http://10.0.0.5:7777")

    def test_save_and_load_nodes(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import RenderNode, load_nodes, save_nodes
            nodes = [
                RenderNode(name="n1", host="host1", port=9090),
                RenderNode(name="n2", host="host2", port=9091),
            ]
            save_nodes(nodes)
            loaded = load_nodes()
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].name, "n1")
            self.assertEqual(loaded[1].name, "n2")
        finally:
            self._restore_nodes_file(mod)

    def test_add_node(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import RenderNode, add_node, load_nodes
            add_node(RenderNode(name="alpha", host="h1"))
            add_node(RenderNode(name="beta", host="h2"))
            nodes = load_nodes()
            self.assertEqual(len(nodes), 2)
            names = [n.name for n in nodes]
            self.assertIn("alpha", names)
            self.assertIn("beta", names)
        finally:
            self._restore_nodes_file(mod)

    def test_add_node_updates_existing(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import RenderNode, add_node, load_nodes
            add_node(RenderNode(name="n1", host="old_host"))
            add_node(RenderNode(name="n1", host="new_host"))
            nodes = load_nodes()
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].host, "new_host")
        finally:
            self._restore_nodes_file(mod)

    def test_remove_node(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import RenderNode, add_node, load_nodes, remove_node
            add_node(RenderNode(name="a", host="h1"))
            add_node(RenderNode(name="b", host="h2"))
            remove_node("a")
            nodes = load_nodes()
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].name, "b")
        finally:
            self._restore_nodes_file(mod)

    def test_remove_nonexistent_node_raises(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import remove_node
            with self.assertRaises(ValueError):
                remove_node("nonexistent")
        finally:
            self._restore_nodes_file(mod)

    def test_load_nodes_empty_file(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import load_nodes
            nodes = load_nodes()
            self.assertEqual(nodes, [])
        finally:
            self._restore_nodes_file(mod)

    def test_get_node(self):
        mod = self._patch_nodes_file()
        try:
            from opencut.core.cloud_render import RenderNode, add_node, get_node
            add_node(RenderNode(name="target", host="h1", port=1234))
            node = get_node("target")
            self.assertIsNotNone(node)
            self.assertEqual(node.port, 1234)
            self.assertIsNone(get_node("missing"))
        finally:
            self._restore_nodes_file(mod)


# ============================================================
# Cloud Render — Node Status & Selection
# ============================================================
class TestCloudRenderStatus(unittest.TestCase):
    """Tests for node health and selection."""

    def test_node_status_to_dict(self):
        from opencut.core.cloud_render import NodeStatus
        st = NodeStatus(name="n1", online=True, active_jobs=2,
                        max_concurrent=4, latency_ms=12.345)
        d = st.to_dict()
        self.assertEqual(d["name"], "n1")
        self.assertTrue(d["online"])
        self.assertEqual(d["latency_ms"], 12.3)

    def test_node_status_load_ratio(self):
        from opencut.core.cloud_render import NodeStatus
        st = NodeStatus(name="n1", active_jobs=3, max_concurrent=6)
        self.assertAlmostEqual(st.load_ratio, 0.5)

    def test_node_status_load_ratio_zero_capacity(self):
        from opencut.core.cloud_render import NodeStatus
        st = NodeStatus(name="n1", active_jobs=1, max_concurrent=0)
        self.assertEqual(st.load_ratio, 1.0)

    def test_cloud_render_result_to_dict(self):
        from opencut.core.cloud_render import CloudRenderResult
        r = CloudRenderResult(node_used="gpu-1", job_id="abc123",
                              status="complete", render_time_ms=5432.1)
        d = r.to_dict()
        self.assertEqual(d["node_used"], "gpu-1")
        self.assertEqual(d["status"], "complete")
        self.assertEqual(d["render_time_ms"], 5432.1)

    def test_get_node_summary_empty(self):
        from opencut.core.cloud_render import get_node_summary
        # Should not crash even with no nodes configured
        summary = get_node_summary()
        self.assertIn("total_nodes", summary)
        self.assertIn("utilization_pct", summary)


# ============================================================
# Cloud Render — Dispatch & Fallback
# ============================================================
class TestCloudRenderDispatch(unittest.TestCase):
    """Tests for render dispatch and local fallback."""

    def test_dispatch_missing_file_raises(self):
        from opencut.core.cloud_render import dispatch_render
        with self.assertRaises(FileNotFoundError):
            dispatch_render("/nonexistent/file.mp4")

    @patch("opencut.core.cloud_render.check_all_nodes")
    @patch("opencut.core.cloud_render._select_node", return_value=None)
    @patch("opencut.core.cloud_render._render_local")
    def test_dispatch_falls_back_to_local(self, mock_local, mock_select, mock_check):
        from opencut.core.cloud_render import dispatch_render
        mock_local.return_value = {"output_path": "/tmp/out.mp4", "status": "complete"}
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            result = dispatch_render(path)
            self.assertTrue(result.fallback_local)
            self.assertEqual(result.status, "complete")
            mock_local.assert_called_once()
        finally:
            os.unlink(path)

    def test_remote_job_tracking(self):
        from opencut.core.cloud_render import clear_completed_jobs, get_remote_jobs
        # Just verify the tracking functions don't crash
        jobs = get_remote_jobs()
        self.assertIsInstance(jobs, list)
        cleared = clear_completed_jobs()
        self.assertIsInstance(cleared, int)


# ============================================================
# Platform Publish — Specs
# ============================================================
class TestPlatformPublishSpecs(unittest.TestCase):
    """Tests for platform specifications."""

    def test_all_platforms_defined(self):
        from opencut.core.platform_publish import PLATFORM_SPECS
        expected = {"youtube", "tiktok", "instagram_reels", "instagram_stories",
                    "instagram_feed", "twitter", "linkedin", "facebook", "vimeo", "podcast"}
        self.assertTrue(expected.issubset(set(PLATFORM_SPECS.keys())))

    def test_each_platform_has_required_keys(self):
        from opencut.core.platform_publish import PLATFORM_SPECS
        required = {"label", "max_width", "max_height", "codec", "audio_codec",
                    "max_file_size_mb", "max_duration_sec", "container"}
        for name, spec in PLATFORM_SPECS.items():
            for key in required:
                self.assertIn(key, spec, f"{name} missing key: {key}")

    def test_list_platforms(self):
        from opencut.core.platform_publish import list_platforms
        platforms = list_platforms()
        self.assertIsInstance(platforms, list)
        self.assertTrue(len(platforms) >= 10)
        keys = [p["key"] for p in platforms]
        self.assertIn("youtube", keys)
        self.assertIn("tiktok", keys)
        self.assertIn("podcast", keys)

    def test_podcast_is_audio_only(self):
        from opencut.core.platform_publish import PLATFORM_SPECS
        podcast = PLATFORM_SPECS["podcast"]
        self.assertEqual(podcast["container"], "mp3")
        self.assertEqual(podcast["max_width"], 0)


# ============================================================
# Platform Publish — Validation
# ============================================================
class TestPlatformPublishValidation(unittest.TestCase):
    """Tests for metadata and video validation."""

    def test_validate_metadata_valid(self):
        from opencut.core.platform_publish import validate_metadata
        errors = validate_metadata("youtube", {
            "title": "My Video",
            "description": "A short description",
            "hashtags": ["#tech", "#coding"],
        })
        blocking = [e for e in errors if e.severity == "error"]
        self.assertEqual(len(blocking), 0)

    def test_validate_metadata_title_too_long(self):
        from opencut.core.platform_publish import validate_metadata
        errors = validate_metadata("youtube", {
            "title": "x" * 200,
        })
        title_errors = [e for e in errors if e.field == "title"]
        self.assertTrue(len(title_errors) > 0)

    def test_validate_metadata_too_many_hashtags(self):
        from opencut.core.platform_publish import validate_metadata
        errors = validate_metadata("twitter", {
            "hashtags": [f"tag{i}" for i in range(50)],
        })
        hashtag_errors = [e for e in errors if e.field == "hashtags" and e.severity == "error"]
        self.assertTrue(len(hashtag_errors) > 0)

    def test_validate_metadata_unknown_platform(self):
        from opencut.core.platform_publish import validate_metadata
        errors = validate_metadata("nonexistent_platform", {})
        self.assertTrue(len(errors) > 0)
        self.assertIn("platform", errors[0].field)

    def test_validate_video_duration_exceeded(self):
        from opencut.core.platform_publish import validate_video_for_platform
        errors = validate_video_for_platform("tiktok", {
            "width": 1080, "height": 1920, "duration": 999,
        })
        dur_errors = [e for e in errors if e.field == "duration"]
        self.assertTrue(len(dur_errors) > 0)

    def test_validate_video_ok(self):
        from opencut.core.platform_publish import validate_video_for_platform
        errors = validate_video_for_platform("youtube", {
            "width": 1920, "height": 1080, "duration": 300,
        })
        blocking = [e for e in errors if e.severity == "error"]
        self.assertEqual(len(blocking), 0)

    def test_validation_error_to_dict(self):
        from opencut.core.platform_publish import ValidationError
        e = ValidationError(field="title", message="Too long", severity="error")
        d = e.to_dict()
        self.assertEqual(d["field"], "title")
        self.assertEqual(d["severity"], "error")

    def test_publish_metadata_to_dict(self):
        from opencut.core.platform_publish import PublishMetadata
        m = PublishMetadata(title="Test", description="Desc", tags=["a"])
        d = m.to_dict()
        self.assertEqual(d["title"], "Test")
        self.assertEqual(d["tags"], ["a"])


# ============================================================
# Platform Publish — Package Preparation
# ============================================================
class TestPlatformPublishPackage(unittest.TestCase):
    """Tests for package preparation."""

    def test_unsupported_platform_raises(self):
        from opencut.core.platform_publish import prepare_publish_package
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                prepare_publish_package(path, "nonexistent")
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        from opencut.core.platform_publish import prepare_publish_package
        with self.assertRaises(FileNotFoundError):
            prepare_publish_package("/no/such/file.mp4", "youtube")

    def test_publish_package_to_dict(self):
        from opencut.core.platform_publish import PublishPackage, ValidationError
        pkg = PublishPackage(
            platform="youtube", video_path="/out.mp4",
            file_size_mb=25.5, resolution="1920x1080",
            validation_errors=[ValidationError("title", "OK", "warning")],
        )
        d = pkg.to_dict()
        self.assertEqual(d["platform"], "youtube")
        self.assertEqual(d["file_size_mb"], 25.5)
        self.assertEqual(len(d["validation_errors"]), 1)

    def test_batch_prepare_no_platforms_raises(self):
        from opencut.core.platform_publish import batch_prepare
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                batch_prepare(path, [])
        finally:
            os.unlink(path)

    def test_batch_prepare_missing_file_raises(self):
        from opencut.core.platform_publish import batch_prepare
        with self.assertRaises(FileNotFoundError):
            batch_prepare("/no/file.mp4", ["youtube"])

    def test_compute_target_resolution(self):
        from opencut.core.platform_publish import PLATFORM_SPECS, _compute_target_resolution
        spec = PLATFORM_SPECS["youtube"]
        w, h = _compute_target_resolution(spec, 1920, 1080)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)
        self.assertEqual(w % 2, 0)
        self.assertEqual(h % 2, 0)

    def test_compute_target_resolution_podcast(self):
        from opencut.core.platform_publish import PLATFORM_SPECS, _compute_target_resolution
        spec = PLATFORM_SPECS["podcast"]
        w, h = _compute_target_resolution(spec, 1920, 1080)
        self.assertEqual(w, 0)
        self.assertEqual(h, 0)

    def test_format_hashtags(self):
        from opencut.core.platform_publish import _format_hashtags
        result = _format_hashtags(["tech", "#coding", "AI"], "youtube")
        self.assertIn("#tech", result)
        self.assertIn("#coding", result)
        self.assertIn("#AI", result)


# ============================================================
# Content Fingerprint — Generation
# ============================================================
class TestContentFingerprint(unittest.TestCase):
    """Tests for fingerprint generation."""

    def test_fingerprint_result_to_dict(self):
        from opencut.core.content_fingerprint import FingerprintResult
        r = FingerprintResult(
            fingerprint_hex="abcdef",
            frame_count=30,
            audio_fingerprint="12345",
            file_hash="sha256hash",
            duration_sec=10.5,
        )
        d = r.to_dict()
        self.assertEqual(d["fingerprint_hex"], "abcdef")
        self.assertEqual(d["frame_count"], 30)
        self.assertEqual(d["duration_sec"], 10.5)

    def test_generate_fingerprint_missing_file(self):
        from opencut.core.content_fingerprint import generate_fingerprint
        with self.assertRaises(FileNotFoundError):
            generate_fingerprint("/nonexistent/file.mp4")

    def test_compute_quick_hash(self):
        from opencut.core.content_fingerprint import _compute_quick_hash
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"test content for hashing" * 100)
            path = f.name
        try:
            h = _compute_quick_hash(path)
            self.assertEqual(len(h), 64)
            # Same content produces same hash
            h2 = _compute_quick_hash(path)
            self.assertEqual(h, h2)
        finally:
            os.unlink(path)

    def test_compute_file_hash(self):
        from opencut.core.content_fingerprint import _compute_file_hash
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"deterministic content")
            path = f.name
        try:
            h = _compute_file_hash(path)
            self.assertEqual(len(h), 64)
        finally:
            os.unlink(path)


# ============================================================
# Content Fingerprint — Perceptual Hashing
# ============================================================
class TestPerceptualHashing(unittest.TestCase):
    """Tests for perceptual hash computation."""

    def test_compute_frame_phash_zeros(self):
        from opencut.core.content_fingerprint import _compute_frame_phash
        # All black frame
        frame = bytes(32 * 32)
        h = _compute_frame_phash(frame, 32, 32)
        self.assertIsInstance(h, int)

    def test_compute_frame_phash_ones(self):
        from opencut.core.content_fingerprint import _compute_frame_phash
        # All white frame
        frame = bytes([255] * (32 * 32))
        h = _compute_frame_phash(frame, 32, 32)
        self.assertIsInstance(h, int)

    def test_identical_frames_same_hash(self):
        from opencut.core.content_fingerprint import _compute_frame_phash
        frame = bytes(range(256)) * 4
        h1 = _compute_frame_phash(frame, 32, 32)
        h2 = _compute_frame_phash(frame, 32, 32)
        self.assertEqual(h1, h2)

    def test_different_frames_different_hash(self):
        from opencut.core.content_fingerprint import _compute_frame_phash
        frame_a = bytes([0] * 512 + [255] * 512)
        frame_b = bytes([255] * 512 + [0] * 512)
        h1 = _compute_frame_phash(frame_a, 32, 32)
        h2 = _compute_frame_phash(frame_b, 32, 32)
        self.assertNotEqual(h1, h2)

    def test_combine_frame_hashes_empty(self):
        from opencut.core.content_fingerprint import _combine_frame_hashes
        result = _combine_frame_hashes([])
        self.assertEqual(result, "")

    def test_combine_frame_hashes_single(self):
        from opencut.core.content_fingerprint import _combine_frame_hashes
        result = _combine_frame_hashes([12345])
        self.assertTrue(len(result) > 0)

    def test_parse_and_combine_roundtrip(self):
        from opencut.core.content_fingerprint import (
            _combine_frame_hashes,
            _parse_fingerprint_hex,
        )
        hashes = [111, 222, 333, 444]
        combined = _combine_frame_hashes(hashes)
        summary, frames = _parse_fingerprint_hex(combined)
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames, hashes)


# ============================================================
# Content Fingerprint — Similarity
# ============================================================
class TestFingerprintSimilarity(unittest.TestCase):
    """Tests for fingerprint comparison."""

    def test_hamming_distance_identical(self):
        from opencut.core.content_fingerprint import _hamming_distance
        self.assertEqual(_hamming_distance(0xFF, 0xFF), 0)

    def test_hamming_distance_all_different(self):
        from opencut.core.content_fingerprint import _hamming_distance
        # 8 bits: 0x00 vs 0xFF
        self.assertEqual(_hamming_distance(0x00, 0xFF), 8)

    def test_hamming_distance_one_bit(self):
        from opencut.core.content_fingerprint import _hamming_distance
        self.assertEqual(_hamming_distance(0b1000, 0b0000), 1)

    def test_visual_similarity_identical(self):
        from opencut.core.content_fingerprint import (
            _combine_frame_hashes,
            compute_visual_similarity,
        )
        hashes = [100, 200, 300]
        fp = _combine_frame_hashes(hashes)
        sim = compute_visual_similarity(fp, fp)
        self.assertAlmostEqual(sim, 100.0)

    def test_visual_similarity_empty(self):
        from opencut.core.content_fingerprint import compute_visual_similarity
        sim = compute_visual_similarity("", "")
        self.assertEqual(sim, 0.0)

    def test_audio_similarity_identical(self):
        from opencut.core.content_fingerprint import compute_audio_similarity
        sim = compute_audio_similarity("abcdef123456", "abcdef123456")
        self.assertAlmostEqual(sim, 100.0)

    def test_audio_similarity_empty(self):
        from opencut.core.content_fingerprint import compute_audio_similarity
        self.assertEqual(compute_audio_similarity("", "test"), 0.0)

    def test_compare_fingerprints(self):
        from opencut.core.content_fingerprint import (
            FingerprintResult,
            _combine_frame_hashes,
            compare_fingerprints,
        )
        fp_hex = _combine_frame_hashes([100, 200, 300])
        a = FingerprintResult(fingerprint_hex=fp_hex, audio_fingerprint="abc",
                              file_path="/a.mp4")
        b = FingerprintResult(fingerprint_hex=fp_hex, audio_fingerprint="abc",
                              file_path="/b.mp4")
        result = compare_fingerprints(a, b)
        self.assertTrue(result.is_duplicate)
        self.assertGreater(result.combined_score, 85)

    def test_similarity_result_to_dict(self):
        from opencut.core.content_fingerprint import SimilarityResult
        r = SimilarityResult(visual_score=90.5, audio_score=85.3,
                             combined_score=88.9, is_duplicate=True)
        d = r.to_dict()
        self.assertEqual(d["visual_score"], 90.5)
        self.assertTrue(d["is_duplicate"])


# ============================================================
# Content Fingerprint — Index
# ============================================================
class TestFingerprintIndex(unittest.TestCase):
    """Tests for fingerprint SQLite index."""

    def setUp(self):
        import opencut.core.content_fingerprint as mod
        self._orig_db = mod._DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        mod._DB_PATH = os.path.join(self.tmp_dir, "test_fp.db")
        mod._init_db()
        self._mod = mod

    def tearDown(self):
        self._mod._DB_PATH = self._orig_db
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_index_and_retrieve(self):
        from opencut.core.content_fingerprint import (
            FingerprintResult,
            get_indexed_fingerprint,
            index_fingerprint,
        )
        fp = FingerprintResult(
            fingerprint_hex="aabbcc",
            file_hash="hash123",
            file_path="/test/video.mp4",
            frame_count=10,
        )
        index_fingerprint(fp)
        retrieved = get_indexed_fingerprint("/test/video.mp4")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.fingerprint_hex, "aabbcc")
        self.assertEqual(retrieved.file_hash, "hash123")

    def test_index_count(self):
        from opencut.core.content_fingerprint import (
            FingerprintResult,
            get_indexed_count,
            index_fingerprint,
        )
        initial = get_indexed_count()
        index_fingerprint(FingerprintResult(
            file_hash="h1", fingerprint_hex="aa", file_path="/a.mp4"))
        index_fingerprint(FingerprintResult(
            file_hash="h2", fingerprint_hex="bb", file_path="/b.mp4"))
        self.assertEqual(get_indexed_count(), initial + 2)

    def test_remove_indexed(self):
        from opencut.core.content_fingerprint import (
            FingerprintResult,
            index_fingerprint,
            remove_indexed,
        )
        index_fingerprint(FingerprintResult(
            file_hash="rem1", fingerprint_hex="cc", file_path="/remove_me.mp4"))
        self.assertTrue(remove_indexed("/remove_me.mp4"))
        self.assertFalse(remove_indexed("/remove_me.mp4"))

    def test_clear_index(self):
        from opencut.core.content_fingerprint import (
            FingerprintResult,
            clear_index,
            get_indexed_count,
            index_fingerprint,
        )
        index_fingerprint(FingerprintResult(
            file_hash="c1", fingerprint_hex="aa", file_path="/c1.mp4"))
        index_fingerprint(FingerprintResult(
            file_hash="c2", fingerprint_hex="bb", file_path="/c2.mp4"))
        cleared = clear_index()
        self.assertGreaterEqual(cleared, 2)
        self.assertEqual(get_indexed_count(), 0)

    def test_search_similar(self):
        from opencut.core.content_fingerprint import (
            FingerprintResult,
            _combine_frame_hashes,
            index_fingerprint,
            search_similar,
        )
        fp_hex = _combine_frame_hashes([100, 200, 300])
        # Index a fingerprint
        index_fingerprint(FingerprintResult(
            file_hash="s1", fingerprint_hex=fp_hex,
            file_path="/indexed.mp4", audio_fingerprint="audio1"))
        # Search with same fingerprint
        query = FingerprintResult(
            fingerprint_hex=fp_hex, file_path="/query.mp4",
            audio_fingerprint="audio1")
        results = search_similar(query, threshold=50)
        self.assertTrue(len(results) >= 1)
        self.assertGreater(results[0].similarity, 50)

    def test_search_result_to_dict(self):
        from opencut.core.content_fingerprint import SearchResult
        r = SearchResult(file_path="/a.mp4", similarity=92.5)
        d = r.to_dict()
        self.assertEqual(d["similarity"], 92.5)


# ============================================================
# Render Farm — Segmentation
# ============================================================
class TestRenderFarmSegmentation(unittest.TestCase):
    """Tests for render farm segmentation strategies."""

    def test_equal_duration_basic(self):
        from opencut.core.render_farm import segment_equal_duration
        ranges = segment_equal_duration(60.0, 4)
        self.assertEqual(len(ranges), 4)
        self.assertAlmostEqual(ranges[0].start, 0.0)
        self.assertAlmostEqual(ranges[0].end, 15.0)
        self.assertAlmostEqual(ranges[3].end, 60.0)

    def test_equal_duration_single(self):
        from opencut.core.render_farm import segment_equal_duration
        ranges = segment_equal_duration(30.0, 1)
        self.assertEqual(len(ranges), 1)
        self.assertAlmostEqual(ranges[0].duration, 30.0)

    def test_equal_duration_zero(self):
        from opencut.core.render_farm import segment_equal_duration
        ranges = segment_equal_duration(0.0, 4)
        self.assertEqual(len(ranges), 0)

    def test_equal_duration_too_many_segments(self):
        from opencut.core.render_farm import segment_equal_duration
        # 10 seconds, 100 segments -> should reduce to avoid tiny segments
        ranges = segment_equal_duration(10.0, 100)
        for r in ranges:
            self.assertGreaterEqual(r.duration, 1.0)

    def test_time_range_to_dict(self):
        from opencut.core.render_farm import TimeRange
        tr = TimeRange(start=10.5, end=25.3)
        d = tr.to_dict()
        self.assertAlmostEqual(d["start"], 10.5)
        self.assertAlmostEqual(d["duration"], 14.8, places=1)

    def test_time_range_duration(self):
        from opencut.core.render_farm import TimeRange
        tr = TimeRange(start=5.0, end=15.0)
        self.assertAlmostEqual(tr.duration, 10.0)

    def test_time_range_negative_duration(self):
        from opencut.core.render_farm import TimeRange
        tr = TimeRange(start=15.0, end=5.0)
        self.assertEqual(tr.duration, 0.0)


# ============================================================
# Render Farm — Segments & Results
# ============================================================
class TestRenderFarmDataClasses(unittest.TestCase):
    """Tests for render farm data classes."""

    def test_farm_segment_to_dict(self):
        from opencut.core.render_farm import FarmSegment, TimeRange
        seg = FarmSegment(
            segment_id="abc",
            time_range=TimeRange(0, 10),
            node="gpu-1",
            status="complete",
            duration_ms=1234.5,
        )
        d = seg.to_dict()
        self.assertEqual(d["segment_id"], "abc")
        self.assertEqual(d["node"], "gpu-1")
        self.assertEqual(d["status"], "complete")

    def test_farm_render_result_to_dict(self):
        from opencut.core.render_farm import FarmRenderResult
        r = FarmRenderResult(
            total_render_time=5000.0,
            speedup_factor=2.5,
            output_path="/out.mp4",
            status="complete",
            total_segments=4,
            completed_segments=4,
        )
        d = r.to_dict()
        self.assertEqual(d["speedup_factor"], 2.5)
        self.assertEqual(d["completed_segments"], 4)

    @patch("opencut.core.render_farm.get_video_info")
    def test_create_segments_zero_duration_raises(self, mock_info):
        from opencut.core.render_farm import create_segments
        mock_info.return_value = {"duration": 0, "width": 1920, "height": 1080, "fps": 30}
        with self.assertRaises(ValueError):
            create_segments("/fake.mp4", "equal_duration", 4)

    @patch("opencut.core.render_farm.get_video_info")
    def test_create_segments_equal(self, mock_info):
        from opencut.core.render_farm import create_segments
        mock_info.return_value = {"duration": 120, "width": 1920, "height": 1080, "fps": 30}
        ranges = create_segments("/fake.mp4", "equal_duration", 4)
        self.assertEqual(len(ranges), 4)

    @patch("opencut.core.render_farm.get_video_info")
    def test_estimate_segments(self, mock_info):
        from opencut.core.render_farm import estimate_segments
        mock_info.return_value = {"duration": 60, "width": 1920, "height": 1080, "fps": 30}
        est = estimate_segments("/fake.mp4", "equal_duration", 3)
        self.assertEqual(est["num_segments"], 3)
        self.assertEqual(est["strategy"], "equal_duration")
        self.assertEqual(len(est["segments"]), 3)


# ============================================================
# Render Farm — Orchestration
# ============================================================
class TestRenderFarmOrchestration(unittest.TestCase):
    """Tests for farm render orchestration."""

    def test_farm_render_missing_file(self):
        from opencut.core.render_farm import farm_render
        with self.assertRaises(FileNotFoundError):
            farm_render("/nonexistent/file.mp4")

    @patch("opencut.core.render_farm.run_ffmpeg")
    @patch("opencut.core.render_farm.get_video_info")
    def test_farm_render_complete(self, mock_info, mock_ffmpeg):
        from opencut.core.render_farm import farm_render
        mock_info.return_value = {"duration": 20, "width": 1920, "height": 1080, "fps": 30}
        mock_ffmpeg.return_value = ""

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            path = f.name
        try:
            result = farm_render(path, strategy="equal_duration", num_segments=2)
            self.assertEqual(result.total_segments, 2)
            self.assertIn(result.status, ("complete", "partial"))
        finally:
            os.unlink(path)

    def test_cleanup_segments(self):
        from opencut.core.render_farm import FarmSegment, TimeRange, cleanup_segments
        segs = [FarmSegment(segment_id="x", output_path="/no/such/seg.mp4",
                            time_range=TimeRange(0, 5))]
        # Should not raise even if files don't exist
        cleanup_segments(segs)


# ============================================================
# Render Farm — Node Assignment
# ============================================================
class TestRenderFarmNodeAssignment(unittest.TestCase):
    """Tests for segment-to-node assignment."""

    def test_assign_nodes_empty(self):
        from opencut.core.render_farm import FarmSegment, TimeRange, _assign_nodes
        segs = [FarmSegment(segment_id="s1", time_range=TimeRange(0, 10))]
        _assign_nodes(segs, None)
        self.assertEqual(segs[0].node, "")

    def test_assign_nodes_round_robin(self):
        from opencut.core.render_farm import FarmSegment, TimeRange, _assign_nodes
        segs = [
            FarmSegment(segment_id="s1", time_range=TimeRange(0, 10)),
            FarmSegment(segment_id="s2", time_range=TimeRange(10, 20)),
            FarmSegment(segment_id="s3", time_range=TimeRange(20, 30)),
        ]
        nodes = [
            {"name": "n1", "capabilities": ["cpu"]},
            {"name": "n2", "capabilities": ["cpu"]},
        ]
        _assign_nodes(segs, nodes)
        self.assertEqual(segs[0].node, "n1")
        self.assertEqual(segs[1].node, "n2")
        self.assertEqual(segs[2].node, "n1")

    def test_assign_gpu_segments_to_gpu_nodes(self):
        from opencut.core.render_farm import FarmSegment, TimeRange, _assign_nodes
        segs = [
            FarmSegment(segment_id="s1", time_range=TimeRange(0, 10), requires_gpu=True),
            FarmSegment(segment_id="s2", time_range=TimeRange(10, 20), requires_gpu=False),
        ]
        nodes = [
            {"name": "cpu-1", "capabilities": ["cpu"]},
            {"name": "gpu-1", "capabilities": ["gpu", "cpu"]},
        ]
        _assign_nodes(segs, nodes)
        self.assertEqual(segs[0].node, "gpu-1")  # GPU segment to GPU node
        self.assertIn(segs[1].node, ["cpu-1", "gpu-1"])


# ============================================================
# Distribution Analytics — Records
# ============================================================
class TestDistributionAnalyticsRecords(unittest.TestCase):
    """Tests for publish record CRUD."""

    def setUp(self):
        import opencut.core.distribution_analytics as mod
        self._orig_db = mod._DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        mod._DB_PATH = os.path.join(self.tmp_dir, "test_analytics.db")
        mod._init_db()
        self._mod = mod

    def tearDown(self):
        self._mod._DB_PATH = self._orig_db
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_and_get_record(self):
        from opencut.core.distribution_analytics import PublishRecord, add_publish_record, get_publish_record
        rec = PublishRecord(
            video_title="Test Video",
            platform="youtube",
            publish_date="2026-01-01",
            category="tech",
        )
        rid = add_publish_record(rec)
        self.assertGreater(rid, 0)
        fetched = get_publish_record(rid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.video_title, "Test Video")
        self.assertEqual(fetched.platform, "youtube")

    def test_list_records(self):
        from opencut.core.distribution_analytics import PublishRecord, add_publish_record, list_publish_records
        add_publish_record(PublishRecord(video_title="V1", platform="youtube"))
        add_publish_record(PublishRecord(video_title="V2", platform="tiktok"))
        add_publish_record(PublishRecord(video_title="V3", platform="youtube"))

        all_recs = list_publish_records()
        self.assertGreaterEqual(len(all_recs), 3)

        yt_recs = list_publish_records(platform="youtube")
        self.assertTrue(all(r.platform == "youtube" for r in yt_recs))

    def test_delete_record(self):
        from opencut.core.distribution_analytics import (
            PublishRecord,
            add_publish_record,
            delete_publish_record,
            get_publish_record,
        )
        rid = add_publish_record(PublishRecord(video_title="Del Me", platform="x"))
        self.assertTrue(delete_publish_record(rid))
        self.assertIsNone(get_publish_record(rid))
        self.assertFalse(delete_publish_record(rid))

    def test_publish_record_to_dict(self):
        from opencut.core.distribution_analytics import PublishRecord
        rec = PublishRecord(
            id=1, video_title="Test", platform="youtube",
            tags=["tag1", "tag2"], duration_sec=120.5,
        )
        d = rec.to_dict()
        self.assertEqual(d["id"], 1)
        self.assertEqual(d["tags"], ["tag1", "tag2"])

    def test_get_record_count(self):
        from opencut.core.distribution_analytics import PublishRecord, add_publish_record, get_record_count
        initial = get_record_count()
        add_publish_record(PublishRecord(video_title="Count1", platform="yt"))
        self.assertEqual(get_record_count(), initial + 1)


# ============================================================
# Distribution Analytics — Metrics
# ============================================================
class TestDistributionAnalyticsMetrics(unittest.TestCase):
    """Tests for metrics recording."""

    def setUp(self):
        import opencut.core.distribution_analytics as mod
        self._orig_db = mod._DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        mod._DB_PATH = os.path.join(self.tmp_dir, "test_analytics.db")
        mod._init_db()
        self._mod = mod

    def tearDown(self):
        self._mod._DB_PATH = self._orig_db
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_record_and_get_metrics(self):
        from opencut.core.distribution_analytics import (
            MetricsEntry,
            PublishRecord,
            add_publish_record,
            get_latest_metrics,
            record_metrics,
        )
        rid = add_publish_record(PublishRecord(video_title="V1", platform="yt"))
        record_metrics(MetricsEntry(record_id=rid, views=1000, likes=50, comments=10))
        latest = get_latest_metrics(rid)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.views, 1000)
        self.assertEqual(latest.likes, 50)

    def test_record_metrics_nonexistent_record(self):
        from opencut.core.distribution_analytics import MetricsEntry, record_metrics
        with self.assertRaises(ValueError):
            record_metrics(MetricsEntry(record_id=99999, views=100))

    def test_metrics_history(self):
        from opencut.core.distribution_analytics import (
            MetricsEntry,
            PublishRecord,
            add_publish_record,
            get_metrics_history,
            record_metrics,
        )
        rid = add_publish_record(PublishRecord(video_title="V1", platform="yt"))
        record_metrics(MetricsEntry(record_id=rid, views=100))
        record_metrics(MetricsEntry(record_id=rid, views=200))
        record_metrics(MetricsEntry(record_id=rid, views=500))
        history = get_metrics_history(rid)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].views, 100)
        self.assertEqual(history[-1].views, 500)

    def test_metrics_entry_to_dict(self):
        from opencut.core.distribution_analytics import MetricsEntry
        m = MetricsEntry(record_id=1, views=1000, likes=50, ctr=0.045)
        d = m.to_dict()
        self.assertEqual(d["views"], 1000)
        self.assertEqual(d["ctr"], 0.045)


# ============================================================
# Distribution Analytics — Aggregation
# ============================================================
class TestDistributionAnalyticsAggregation(unittest.TestCase):
    """Tests for analytics aggregation and reporting."""

    def setUp(self):
        import opencut.core.distribution_analytics as mod
        self._orig_db = mod._DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        mod._DB_PATH = os.path.join(self.tmp_dir, "test_analytics.db")
        mod._init_db()
        self._mod = mod

    def tearDown(self):
        self._mod._DB_PATH = self._orig_db
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _seed_data(self):
        from opencut.core.distribution_analytics import (
            MetricsEntry,
            PublishRecord,
            add_publish_record,
            record_metrics,
        )
        r1 = add_publish_record(PublishRecord(
            video_title="YT Video 1", platform="youtube", category="tech"))
        record_metrics(MetricsEntry(record_id=r1, views=10000, likes=500,
                                    comments=50, shares=100, ctr=0.05))

        r2 = add_publish_record(PublishRecord(
            video_title="YT Video 2", platform="youtube", category="tech"))
        record_metrics(MetricsEntry(record_id=r2, views=5000, likes=200,
                                    comments=30, shares=40, ctr=0.03))

        r3 = add_publish_record(PublishRecord(
            video_title="TT Video 1", platform="tiktok", category="comedy"))
        record_metrics(MetricsEntry(record_id=r3, views=50000, likes=5000,
                                    comments=200, shares=1000, ctr=0.08))

    def test_compute_platform_stats(self):
        from opencut.core.distribution_analytics import compute_platform_stats
        self._seed_data()
        stats = compute_platform_stats()
        self.assertGreaterEqual(len(stats), 2)
        platforms = [s.platform for s in stats]
        self.assertIn("youtube", platforms)
        self.assertIn("tiktok", platforms)

    def test_compute_platform_stats_filter(self):
        from opencut.core.distribution_analytics import compute_platform_stats
        self._seed_data()
        stats = compute_platform_stats(platform="youtube")
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].platform, "youtube")
        self.assertEqual(stats[0].total_videos, 2)

    def test_generate_report(self):
        from opencut.core.distribution_analytics import generate_report
        self._seed_data()
        report = generate_report()
        self.assertIsInstance(report.per_platform_stats, list)
        self.assertGreater(report.total_reach, 0)
        self.assertIn(report.best_platform, ["youtube", "tiktok"])
        self.assertGreater(report.total_videos, 0)

    def test_generate_report_to_dict(self):
        from opencut.core.distribution_analytics import generate_report
        self._seed_data()
        report = generate_report()
        d = report.to_dict()
        self.assertIn("per_platform_stats", d)
        self.assertIn("recommendations", d)
        self.assertIn("total_reach", d)

    def test_content_type_analysis(self):
        from opencut.core.distribution_analytics import content_type_analysis
        self._seed_data()
        analysis = content_type_analysis()
        self.assertIsInstance(analysis, list)
        categories = [a["category"] for a in analysis]
        self.assertIn("tech", categories)

    def test_engagement_rate_computation(self):
        from opencut.core.distribution_analytics import _compute_engagement_rate
        rate = _compute_engagement_rate(1000, 50, 10, 5)
        self.assertAlmostEqual(rate, 0.065)
        self.assertEqual(_compute_engagement_rate(0, 10, 5, 2), 0.0)

    def test_growth_trend(self):
        from opencut.core.distribution_analytics import _compute_growth_trend
        trend = _compute_growth_trend()
        self.assertIn(trend, ["growing", "declining", "stable", "no_data"])

    def test_top_performing(self):
        from opencut.core.distribution_analytics import _get_top_performing
        self._seed_data()
        top = _get_top_performing(limit=5)
        self.assertIsInstance(top, list)
        self.assertGreater(len(top), 0)
        self.assertIn("video_title", top[0])

    def test_recommendations_empty(self):
        from opencut.core.distribution_analytics import _generate_recommendations
        recs = _generate_recommendations([])
        self.assertTrue(len(recs) > 0)


# ============================================================
# Distribution Analytics — Export
# ============================================================
class TestDistributionAnalyticsExport(unittest.TestCase):
    """Tests for analytics export."""

    def setUp(self):
        import opencut.core.distribution_analytics as mod
        self._orig_db = mod._DB_PATH
        self.tmp_dir = tempfile.mkdtemp()
        mod._DB_PATH = os.path.join(self.tmp_dir, "test_analytics.db")
        mod._init_db()
        self._mod = mod

    def tearDown(self):
        self._mod._DB_PATH = self._orig_db
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_csv(self):
        from opencut.core.distribution_analytics import (
            PublishRecord,
            add_publish_record,
            export_csv,
        )
        add_publish_record(PublishRecord(video_title="CSV Test", platform="youtube"))
        csv_path = os.path.join(self.tmp_dir, "export.csv")
        result = export_csv(csv_path)
        self.assertTrue(os.path.isfile(result))
        with open(result, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("CSV Test", content)

    def test_export_json(self):
        from opencut.core.distribution_analytics import (
            PublishRecord,
            add_publish_record,
            export_json,
        )
        add_publish_record(PublishRecord(video_title="JSON Test", platform="tiktok"))
        json_path = os.path.join(self.tmp_dir, "export.json")
        result = export_json(json_path)
        self.assertTrue(os.path.isfile(result))
        with open(result, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("report", data)
        self.assertIn("records", data)


# ============================================================
# Platform Stats Data Class
# ============================================================
class TestPlatformStatsDataClass(unittest.TestCase):
    """Tests for the PlatformStats data class."""

    def test_platform_stats_to_dict(self):
        from opencut.core.distribution_analytics import PlatformStats
        ps = PlatformStats(
            platform="youtube", total_videos=10, total_views=50000,
            avg_engagement_rate=0.065, avg_ctr=0.045,
        )
        d = ps.to_dict()
        self.assertEqual(d["platform"], "youtube")
        self.assertEqual(d["total_videos"], 10)
        self.assertEqual(d["avg_engagement_rate"], 0.065)


# ============================================================
# Route Smoke Tests
# ============================================================
class TestCloudDistribRoutes(unittest.TestCase):
    """Smoke tests for cloud & distribution route registration."""

    def test_blueprint_exists(self):
        from opencut.routes.cloud_distrib_routes import cloud_distrib_bp
        self.assertEqual(cloud_distrib_bp.name, "cloud_distrib")

    def test_blueprint_has_routes(self):
        from opencut.routes.cloud_distrib_routes import cloud_distrib_bp
        # Just verify the blueprint imported without errors
        self.assertIsNotNone(cloud_distrib_bp)

    def test_route_functions_exist(self):
        from opencut.routes import cloud_distrib_routes as mod
        self.assertTrue(hasattr(mod, "cloud_render"))
        self.assertTrue(hasattr(mod, "cloud_nodes_list"))
        self.assertTrue(hasattr(mod, "cloud_nodes_add"))
        self.assertTrue(hasattr(mod, "cloud_nodes_remove"))
        self.assertTrue(hasattr(mod, "publish_prepare"))
        self.assertTrue(hasattr(mod, "publish_platforms"))
        self.assertTrue(hasattr(mod, "publish_validate"))
        self.assertTrue(hasattr(mod, "fingerprint_generate"))
        self.assertTrue(hasattr(mod, "fingerprint_search"))
        self.assertTrue(hasattr(mod, "farm_render_submit"))
        self.assertTrue(hasattr(mod, "farm_status"))
        self.assertTrue(hasattr(mod, "analytics_record"))
        self.assertTrue(hasattr(mod, "analytics_report"))


if __name__ == "__main__":
    unittest.main()
