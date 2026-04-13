"""
Tests for OpenCut content & provenance features.

Covers:
  - Provenance manifest generation & verification
  - Social caption generation (LLM + fallback)
  - Usage analytics recording & querying
  - Podcast RSS feed generation
  - Content routes (smoke tests)
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Provenance
# ============================================================
class TestProvenance(unittest.TestCase):
    """Tests for opencut.core.provenance."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.tmpdir, "sample.mp4")
        with open(self.test_file, "wb") as f:
            f.write(b"fake video content for hashing test")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sha256_file(self):
        """_sha256_file should return a valid hex digest."""
        from opencut.core.provenance import _sha256_file
        digest = _sha256_file(self.test_file)
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_generate_manifest_creates_sidecar(self):
        """generate_provenance_manifest should write a .provenance.json sidecar."""
        from opencut.core.provenance import generate_provenance_manifest
        generate_provenance_manifest(self.test_file)
        sidecar = self.test_file + ".provenance.json"
        self.assertTrue(os.path.isfile(sidecar))
        with open(sidecar, "r") as f:
            data = json.load(f)
        self.assertEqual(data["source_file"]["path"], os.path.abspath(self.test_file))
        self.assertIn("signature", data)
        self.assertIn("opencut_version", data)

    def test_generate_manifest_custom_output(self):
        """generate_provenance_manifest should write to custom output_path."""
        from opencut.core.provenance import generate_provenance_manifest
        custom_out = os.path.join(self.tmpdir, "custom", "manifest.json")
        generate_provenance_manifest(self.test_file, output_path=custom_out)
        self.assertTrue(os.path.isfile(custom_out))

    def test_manifest_structure(self):
        """Manifest must contain required top-level keys."""
        from opencut.core.provenance import generate_provenance_manifest
        m = generate_provenance_manifest(self.test_file)
        for key in ("opencut_version", "generated_at", "source_file",
                     "operations", "output_file", "signature"):
            self.assertIn(key, m, f"Missing key: {key}")
        self.assertIn("hash_sha256", m["source_file"])
        self.assertIn("size_bytes", m["source_file"])

    def test_verify_valid_signature(self):
        """verify_provenance_manifest should return True for untampered manifests."""
        from opencut.core.provenance import (
            generate_provenance_manifest,
            verify_provenance_manifest,
        )
        m = generate_provenance_manifest(self.test_file)
        self.assertTrue(verify_provenance_manifest(m))

    def test_verify_tampered_signature(self):
        """verify_provenance_manifest should return False if manifest is altered."""
        from opencut.core.provenance import (
            generate_provenance_manifest,
            verify_provenance_manifest,
        )
        m = generate_provenance_manifest(self.test_file)
        m["source_file"]["size_bytes"] = 9999999
        self.assertFalse(verify_provenance_manifest(m))

    def test_file_not_found(self):
        """generate_provenance_manifest should raise for missing files."""
        from opencut.core.provenance import generate_provenance_manifest
        with self.assertRaises(FileNotFoundError):
            generate_provenance_manifest("/nonexistent/file.mp4")

    def test_provenance_key_env(self):
        """Custom OPENCUT_PROVENANCE_KEY env var should change signature."""
        from opencut.core.provenance import generate_provenance_manifest
        m1 = generate_provenance_manifest(self.test_file)
        with patch.dict(os.environ, {"OPENCUT_PROVENANCE_KEY": "custom-secret-key"}):
            from opencut.core.provenance import _sign_manifest
            sig_custom = _sign_manifest(m1)
        self.assertNotEqual(m1["signature"], sig_custom)


# ============================================================
# Social Captions
# ============================================================
class TestSocialCaptions(unittest.TestCase):
    """Tests for opencut.core.social_captions."""

    SAMPLE_TRANSCRIPT = (
        "Today we are going to talk about machine learning and artificial intelligence. "
        "Deep learning has revolutionized computer vision and natural language processing. "
        "Neural networks can now generate realistic images and translate languages in real time. "
        "This tutorial covers the basics of training a model using Python and PyTorch."
    )

    def test_fallback_captions_youtube(self):
        """Fallback should produce captions with hashtags and tags for YouTube."""
        from opencut.core.social_captions import _fallback_captions
        result = _fallback_captions(self.SAMPLE_TRANSCRIPT, "youtube")
        self.assertEqual(result.platform, "youtube")
        self.assertTrue(len(result.title) > 0)
        self.assertTrue(len(result.hashtags) > 0)
        self.assertTrue(len(result.tags) > 0)
        self.assertTrue(all(h.startswith("#") for h in result.hashtags))

    def test_fallback_captions_tiktok(self):
        """TikTok fallback should produce 5-8 hashtags."""
        from opencut.core.social_captions import _fallback_captions
        result = _fallback_captions(self.SAMPLE_TRANSCRIPT, "tiktok")
        self.assertEqual(result.platform, "tiktok")
        self.assertLessEqual(len(result.hashtags), 8)

    def test_fallback_captions_instagram(self):
        """Instagram fallback should produce up to 30 hashtags."""
        from opencut.core.social_captions import _fallback_captions
        result = _fallback_captions(self.SAMPLE_TRANSCRIPT, "instagram")
        self.assertEqual(result.platform, "instagram")
        self.assertLessEqual(len(result.hashtags), 30)

    def test_fallback_captions_twitter(self):
        """Twitter fallback should produce max 3 hashtags."""
        from opencut.core.social_captions import _fallback_captions
        result = _fallback_captions(self.SAMPLE_TRANSCRIPT, "twitter")
        self.assertEqual(result.platform, "twitter")
        self.assertLessEqual(len(result.hashtags), 3)

    def test_keyword_extraction(self):
        """_extract_keywords_tfidf should return relevant keywords."""
        from opencut.core.social_captions import _extract_keywords_tfidf
        kw = _extract_keywords_tfidf(self.SAMPLE_TRANSCRIPT, top_n=10)
        self.assertTrue(len(kw) > 0)
        self.assertLessEqual(len(kw), 10)
        # Should find "learning" or "neural" or similar
        combined = " ".join(kw)
        self.assertTrue(
            any(w in combined for w in ("learning", "neural", "language", "deep")),
            f"Expected ML keywords in {kw}"
        )

    def test_keyword_extraction_empty(self):
        """_extract_keywords_tfidf should return empty list for empty input."""
        from opencut.core.social_captions import _extract_keywords_tfidf
        self.assertEqual(_extract_keywords_tfidf(""), [])

    def test_generate_empty_transcript(self):
        """generate_social_captions should return empty result for empty input."""
        from opencut.core.social_captions import generate_social_captions
        result = generate_social_captions("")
        self.assertEqual(result.title, "")
        self.assertEqual(result.hashtags, [])

    def test_generate_unknown_platform_defaults_youtube(self):
        """Unknown platform should fall back to youtube."""
        from opencut.core.social_captions import generate_social_captions
        result = generate_social_captions(self.SAMPLE_TRANSCRIPT, platform="unknown_platform")
        self.assertEqual(result.platform, "youtube")

    def test_parse_llm_response_valid(self):
        """_parse_llm_response should parse valid JSON."""
        from opencut.core.social_captions import _parse_llm_response
        response = json.dumps({
            "title": "Test Title",
            "description": "A description",
            "tags": ["tag1", "tag2"],
            "hashtags": ["#hash1", "hash2"],
        })
        result = _parse_llm_response(response, "youtube")
        self.assertEqual(result.title, "Test Title")
        self.assertEqual(len(result.hashtags), 2)
        self.assertTrue(all(h.startswith("#") for h in result.hashtags))
        self.assertTrue(all(not t.startswith("#") for t in result.tags))

    def test_parse_llm_response_no_json(self):
        """_parse_llm_response should raise on non-JSON response."""
        from opencut.core.social_captions import _parse_llm_response
        with self.assertRaises(ValueError):
            _parse_llm_response("No JSON here at all", "youtube")

    def test_generate_falls_back_when_llm_fails(self):
        """generate_social_captions should fall back to keyword extraction on LLM error."""
        from opencut.core.social_captions import generate_social_captions

        with patch("opencut.core.llm.query_llm", side_effect=RuntimeError("fail")):
            result = generate_social_captions(self.SAMPLE_TRANSCRIPT, platform="youtube")
        self.assertEqual(result.platform, "youtube")
        self.assertTrue(len(result.title) > 0)

    def test_result_dataclass_fields(self):
        """SocialCaptionResult should have all expected fields."""
        from opencut.core.social_captions import SocialCaptionResult
        r = SocialCaptionResult()
        self.assertEqual(r.title, "")
        self.assertEqual(r.description, "")
        self.assertEqual(r.hashtags, [])
        self.assertEqual(r.tags, [])
        self.assertEqual(r.platform, "")


# ============================================================
# Analytics
# ============================================================
class TestAnalytics(unittest.TestCase):
    """Tests for opencut.core.analytics."""

    def setUp(self):
        # Use a temporary database
        import opencut.core.analytics as analytics_mod
        self._orig_db_path = analytics_mod._DB_PATH
        self.tmpdir = tempfile.mkdtemp()
        analytics_mod._DB_PATH = os.path.join(self.tmpdir, "test_analytics.db")
        analytics_mod._INITIALIZED = False
        analytics_mod._LOCAL = __import__("threading").local()

    def tearDown(self):
        import shutil

        import opencut.core.analytics as analytics_mod
        analytics_mod._DB_PATH = self._orig_db_path
        analytics_mod._INITIALIZED = False
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_and_query(self):
        """record_usage + get_usage_stats should return correct counts."""
        from opencut.core.analytics import get_usage_stats, record_usage
        record_usage("/silence", 150, True, "silence")
        record_usage("/silence", 200, True, "silence")
        record_usage("/export", 500, False, "export")

        stats = get_usage_stats(days=1)
        self.assertEqual(stats["total_jobs"], 3)
        self.assertEqual(stats["total_errors"], 1)
        self.assertTrue(stats["avg_duration_ms"] > 0)

        # Top features should have /silence first (2 calls)
        top = stats["top_features"]
        self.assertTrue(len(top) >= 2)
        self.assertEqual(top[0]["endpoint"], "/silence")
        self.assertEqual(top[0]["count"], 2)

    def test_daily_usage(self):
        """get_usage_stats should include daily_usage."""
        from opencut.core.analytics import get_usage_stats, record_usage
        record_usage("/test", 100, True)
        stats = get_usage_stats(days=1)
        self.assertTrue(len(stats["daily_usage"]) >= 1)
        self.assertIn("date", stats["daily_usage"][0])
        self.assertIn("count", stats["daily_usage"][0])

    def test_feature_stats(self):
        """get_feature_stats should return detailed stats for one endpoint."""
        from opencut.core.analytics import get_feature_stats, record_usage
        record_usage("/silence", 100, True)
        record_usage("/silence", 300, True)
        record_usage("/silence", 200, False)

        fs = get_feature_stats("/silence", days=1)
        self.assertEqual(fs["endpoint"], "/silence")
        self.assertEqual(fs["total_calls"], 3)
        self.assertEqual(fs["success_count"], 2)
        self.assertEqual(fs["error_count"], 1)
        self.assertAlmostEqual(fs["error_rate"], 1 / 3, places=3)
        self.assertEqual(fs["min_duration_ms"], 100)
        self.assertEqual(fs["max_duration_ms"], 300)

    def test_feature_stats_empty(self):
        """get_feature_stats for unknown endpoint should return zero counts."""
        from opencut.core.analytics import get_feature_stats
        fs = get_feature_stats("/nonexistent", days=1)
        self.assertEqual(fs["total_calls"], 0)
        self.assertEqual(fs["error_rate"], 0.0)

    def test_empty_stats(self):
        """get_usage_stats with no data should return zeroed stats."""
        from opencut.core.analytics import get_usage_stats
        stats = get_usage_stats(days=1)
        self.assertEqual(stats["total_jobs"], 0)
        self.assertEqual(stats["total_errors"], 0)
        self.assertEqual(stats["avg_duration_ms"], 0.0)
        self.assertEqual(stats["top_features"], [])


# ============================================================
# Podcast RSS
# ============================================================
class TestPodcastRSS(unittest.TestCase):
    """Tests for opencut.core.podcast_rss."""

    def _make_feed_meta(self, **overrides):
        base = {
            "title": "Test Podcast",
            "description": "A test podcast feed",
            "author": "Test Author",
            "language": "en",
            "email": "test@example.com",
            "category": "Technology",
            "image_url": "https://example.com/cover.jpg",
            "website_url": "https://example.com",
        }
        base.update(overrides)
        return base

    def _make_episode(self, **overrides):
        base = {
            "title": "Episode 1",
            "description": "First episode description",
            "audio_path": "https://example.com/ep1.mp3",
            "duration_seconds": 3600,
            "pub_date": time.time(),
        }
        base.update(overrides)
        return base

    def test_generate_valid_xml(self):
        """generate_podcast_rss should return valid XML."""
        from opencut.core.podcast_rss import generate_podcast_rss
        xml = generate_podcast_rss(
            episodes=[self._make_episode()],
            feed_metadata=self._make_feed_meta(),
        )
        self.assertIn("<?xml", xml)
        self.assertIn("<rss", xml)
        self.assertIn("<channel>", xml)
        self.assertIn("<item>", xml)

    def test_itunes_namespace(self):
        """Generated RSS should include iTunes namespace elements."""
        from opencut.core.podcast_rss import generate_podcast_rss
        xml = generate_podcast_rss(
            episodes=[self._make_episode()],
            feed_metadata=self._make_feed_meta(),
        )
        self.assertIn("xmlns:itunes", xml)
        self.assertIn("itunes:author", xml)
        self.assertIn("itunes:duration", xml)

    def test_enclosure_tag(self):
        """Each episode should have an <enclosure> tag."""
        from opencut.core.podcast_rss import generate_podcast_rss
        xml = generate_podcast_rss(
            episodes=[self._make_episode(audio_path="https://example.com/ep.mp3")],
            feed_metadata=self._make_feed_meta(),
        )
        self.assertIn('<enclosure', xml)
        self.assertIn('url="https://example.com/ep.mp3"', xml)
        self.assertIn('type="audio/mpeg"', xml)

    def test_chapter_markers(self):
        """Episodes with chapter_markers should include psc:chapters."""
        from opencut.core.podcast_rss import generate_podcast_rss
        ep = self._make_episode(chapter_markers=[
            {"start": 0, "title": "Intro"},
            {"start": 300, "title": "Main Topic"},
            {"start": 1800, "title": "Conclusion"},
        ])
        xml = generate_podcast_rss(
            episodes=[ep],
            feed_metadata=self._make_feed_meta(),
        )
        self.assertIn("psc:chapters", xml)
        self.assertIn("psc:chapter", xml)
        self.assertIn('title="Intro"', xml)

    def test_multiple_episodes(self):
        """Feed should contain all episodes."""
        from opencut.core.podcast_rss import generate_podcast_rss
        episodes = [
            self._make_episode(title="Episode 1"),
            self._make_episode(title="Episode 2"),
            self._make_episode(title="Episode 3"),
        ]
        xml = generate_podcast_rss(
            episodes=episodes,
            feed_metadata=self._make_feed_meta(),
        )
        self.assertIn("Episode 1", xml)
        self.assertIn("Episode 2", xml)
        self.assertIn("Episode 3", xml)

    def test_write_to_file(self):
        """output_path should cause the RSS to be written to disk."""
        from opencut.core.podcast_rss import generate_podcast_rss
        tmpdir = tempfile.mkdtemp()
        out = os.path.join(tmpdir, "feed.xml")
        try:
            generate_podcast_rss(
                episodes=[self._make_episode()],
                feed_metadata=self._make_feed_meta(),
                output_path=out,
            )
            self.assertTrue(os.path.isfile(out))
            with open(out, "r") as f:
                content = f.read()
            self.assertIn("<rss", content)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_missing_feed_metadata_raises(self):
        """Missing required feed metadata should raise ValueError."""
        from opencut.core.podcast_rss import generate_podcast_rss
        with self.assertRaises(ValueError) as ctx:
            generate_podcast_rss(
                episodes=[self._make_episode()],
                feed_metadata={"title": "Only title"},
            )
        self.assertIn("description", str(ctx.exception))

    def test_missing_episode_title_raises(self):
        """Missing episode title should raise ValueError."""
        from opencut.core.podcast_rss import generate_podcast_rss
        with self.assertRaises(ValueError) as ctx:
            generate_podcast_rss(
                episodes=[{"audio_path": "test.mp3"}],
                feed_metadata=self._make_feed_meta(),
            )
        self.assertIn("title", str(ctx.exception))

    def test_missing_episode_audio_raises(self):
        """Missing episode audio_path should raise ValueError."""
        from opencut.core.podcast_rss import generate_podcast_rss
        with self.assertRaises(ValueError) as ctx:
            generate_podcast_rss(
                episodes=[{"title": "No Audio"}],
                feed_metadata=self._make_feed_meta(),
            )
        self.assertIn("audio_path", str(ctx.exception))

    def test_duration_format(self):
        """_format_duration should produce HH:MM:SS or M:SS."""
        from opencut.core.podcast_rss import _format_duration
        self.assertEqual(_format_duration(65), "1:05")
        self.assertEqual(_format_duration(3661), "1:01:01")
        self.assertEqual(_format_duration(0), "0:00")

    def test_mime_type_guessing(self):
        """_guess_mime_type should return correct MIME types."""
        from opencut.core.podcast_rss import _guess_mime_type
        self.assertEqual(_guess_mime_type("file.mp3"), "audio/mpeg")
        self.assertEqual(_guess_mime_type("file.m4a"), "audio/x-m4a")
        self.assertEqual(_guess_mime_type("file.ogg"), "audio/ogg")
        self.assertEqual(_guess_mime_type("file.wav"), "audio/wav")
        self.assertEqual(_guess_mime_type("file.unknown"), "audio/mpeg")


# ============================================================
# Route smoke tests
# ============================================================
class TestContentRoutes(unittest.TestCase):
    """Smoke tests for content_routes blueprint."""

    @classmethod
    def setUpClass(cls):
        """Create a minimal Flask test app with CSRF token injection."""
        from flask import Flask
        app = Flask(__name__)
        app.config["TESTING"] = True

        from opencut.routes.content_routes import content_bp
        app.register_blueprint(content_bp)

        cls.app = app
        cls.client = app.test_client()

        # Obtain a valid CSRF token by inserting one into the token pool
        from opencut.security import get_csrf_token
        cls.csrf_token = get_csrf_token()

    def _csrf_headers(self):
        return {"X-OpenCut-Token": self.csrf_token, "Content-Type": "application/json"}

    def test_analytics_usage_get(self):
        """GET /analytics/usage should return 200."""
        with patch("opencut.core.analytics.get_usage_stats", return_value={
            "top_features": [], "total_jobs": 0,
            "total_errors": 0, "avg_duration_ms": 0.0, "daily_usage": [],
        }):
            resp = self.client.get("/analytics/usage")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("total_jobs", data)

    def test_analytics_feature_get(self):
        """GET /analytics/feature/silence should return 200."""
        with patch("opencut.core.analytics.get_feature_stats", return_value={
            "endpoint": "/silence", "total_calls": 0, "success_count": 0,
            "error_count": 0, "error_rate": 0.0, "avg_duration_ms": 0.0,
            "min_duration_ms": 0, "max_duration_ms": 0, "daily_usage": [],
        }):
            resp = self.client.get("/analytics/feature/silence")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertIn("endpoint", data)

    def test_podcast_generate_rss_missing_data(self):
        """POST /podcast/generate-rss with no episodes should return 400."""
        resp = self.client.post(
            "/podcast/generate-rss",
            json={"episodes": [], "feed_metadata": {"title": "Test"}},
            headers=self._csrf_headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_podcast_generate_rss_missing_metadata(self):
        """POST /podcast/generate-rss with no feed_metadata should return 400."""
        resp = self.client.post(
            "/podcast/generate-rss",
            json={"episodes": [{"title": "ep", "audio_path": "x.mp3"}]},
            headers=self._csrf_headers(),
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
