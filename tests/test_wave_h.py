"""
Tests for Wave H (v1.25.0) — Commercial Parity & Content-Creator Polish.

Covers the Tier 1 fully-working modules (virality_score, cursor_zoom
sidecar extension, changelog_feed, issue_report, demo_bundle, gist_sync,
onboarding) plus availability checks for Tier 2 / Tier 3 stubs.

Network-dependent code paths (changelog GitHub fetch, gist HTTP) are
stubbed via ``unittest.mock.patch`` so the suite stays hermetic.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# virality_score
# =========================================================================
class TestViralityScore(unittest.TestCase):
    """Pure-Python parts of virality_score (lexicon, weighting, shape)."""

    def test_hook_lexicon_detects_curiosity_gap(self):
        from opencut.core.virality_score import _hook_lexicon_score
        text = (
            "Here's the secret nobody tells you about Python. "
            "Why does everyone get this wrong?"
        )
        score, phrase = _hook_lexicon_score(text)
        self.assertGreater(score, 10.0)
        self.assertIn("secret", phrase.lower())

    def test_hook_lexicon_handles_empty(self):
        from opencut.core.virality_score import _hook_lexicon_score
        self.assertEqual(_hook_lexicon_score(""), (0.0, ""))

    def test_normalise_weights_sums_to_one(self):
        from opencut.core.virality_score import _normalise_weights
        w = _normalise_weights({"audio_energy": 2, "transcript_hook": 2,
                                 "visual_salience": 1})
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertAlmostEqual(w["audio_energy"], 0.4, places=6)

    def test_normalise_weights_rejects_nan_and_negative(self):
        from opencut.core.virality_score import _normalise_weights
        w = _normalise_weights({"audio_energy": float("nan"),
                                 "transcript_hook": -5})
        # Bad values fall back to defaults; total re-normalised.
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)

    def test_result_is_subscriptable(self):
        from opencut.core.virality_score import ViralityResult, ViralitySignals
        r = ViralityResult(
            score=42.0,
            signals=ViralitySignals(audio_energy=10, transcript_hook=20, visual_salience=30),
        )
        # Subscript + keys() so Flask jsonify works.
        self.assertEqual(r["score"], 42.0)
        self.assertIn("hook_phrase", r.keys())

    def test_check_available_requires_ffmpeg(self):
        from opencut.core.virality_score import check_virality_score_available
        # Either ffmpeg is present or not — but the call must not raise.
        self.assertIn(check_virality_score_available(), (True, False))


# =========================================================================
# cursor_zoom sidecar extension
# =========================================================================
class TestCursorZoomSidecar(unittest.TestCase):
    def test_parse_sidecar_clamps_coordinates(self):
        from opencut.core.cursor_zoom import parse_click_sidecar
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "clicks.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({"clicks": [
                    {"t": 1.0, "x": 500, "y": 400},       # in bounds
                    {"t": 2.0, "x": 9999, "y": -50},      # out of bounds
                    {"t": -3.0, "x": 10, "y": 10},        # negative time (drop)
                ]}, fh)
            regions = parse_click_sidecar(p, width=1920, height=1080)
        # Negative-time entry dropped; two in-bounds-after-clamp kept.
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].x, 500)
        self.assertEqual(regions[1].x, 1920)   # clamped to width
        self.assertEqual(regions[1].y, 0)      # clamped to 0

    def test_parse_sidecar_raises_on_missing(self):
        from opencut.core.cursor_zoom import parse_click_sidecar
        with self.assertRaises(FileNotFoundError):
            parse_click_sidecar("/definitely/not/a/real/path.json", 1920, 1080)

    def test_normalise_events_skips_nonfinite(self):
        from opencut.core.cursor_zoom import normalise_click_events
        events = [
            {"t": float("nan"), "x": 10, "y": 10},
            {"t": 1.5, "x": 100, "y": 100},
            {"t": 2.5, "x": float("inf"), "y": 50},
        ]
        regions = normalise_click_events(events, 1920, 1080)
        # NaN dropped; inf-coord → clamped to 0.
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].x, 100)
        self.assertEqual(regions[1].x, 0)

    def test_events_sorted_by_time(self):
        from opencut.core.cursor_zoom import normalise_click_events
        regions = normalise_click_events(
            [{"t": 3.0, "x": 0, "y": 0},
             {"t": 1.0, "x": 0, "y": 0},
             {"t": 2.0, "x": 0, "y": 0}],
            800, 600,
        )
        self.assertEqual([r.timestamp for r in regions], [1.0, 2.0, 3.0])


# =========================================================================
# changelog_feed
# =========================================================================
class TestChangelogFeed(unittest.TestCase):
    def test_fetch_releases_returns_fallback_on_network_error(self):
        from opencut.core import changelog_feed
        with patch("opencut.core.changelog_feed.urllib.request.urlopen",
                   side_effect=OSError("network down")):
            # Clear module cache so the stub is hit.
            with changelog_feed._CACHE_LOCK:
                changelog_feed._CACHE.clear()
                changelog_feed._CACHE["releases"] = None
                changelog_feed._CACHE["expires"] = 0.0
                changelog_feed._CACHE["fetched_at"] = 0.0
            result = changelog_feed.fetch_releases(limit=3)
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["releases"], [])
        self.assertIn("network down", result["note"])

    def test_mark_seen_rejects_empty(self):
        from opencut.core.changelog_feed import mark_seen
        with self.assertRaises(ValueError):
            mark_seen("")

    def test_latest_unseen_filters_by_tag(self):
        from opencut.core import changelog_feed
        fake_feed = {
            "releases": [
                {"tag": "v1.26.0", "name": "", "published_at": "", "url": "", "draft": False, "prerelease": False, "body": ""},
                {"tag": "v1.25.0", "name": "", "published_at": "", "url": "", "draft": False, "prerelease": False, "body": ""},
                {"tag": "v1.24.0", "name": "", "published_at": "", "url": "", "draft": False, "prerelease": False, "body": ""},
            ],
            "source": "cache", "fetched_at": 0, "note": "",
        }
        with patch.object(changelog_feed, "fetch_releases", return_value=fake_feed):
            result = changelog_feed.latest_unseen(last_seen_tag="v1.25.0", limit=5)
        self.assertEqual([r["tag"] for r in result["unseen"]], ["v1.26.0"])


# =========================================================================
# issue_report
# =========================================================================
class TestIssueReport(unittest.TestCase):
    def test_bundle_returns_github_url(self):
        from opencut.core.issue_report import bundle
        result = bundle(title="Smoke", description="hello")
        self.assertIn("github.com", result["url"])
        self.assertIn("Smoke", result["title"])
        self.assertGreater(result["size_bytes"], 0)

    def test_bundle_scrubs_home_paths(self):
        from opencut.core import issue_report
        home = os.path.expanduser("~")
        text_with_home = f"File not found: {home}/secret/project.txt"
        scrubbed = issue_report._scrub_paths(text_with_home)
        self.assertNotIn(home, scrubbed)
        self.assertIn("~", scrubbed)

    def test_bundle_cap_60kb(self):
        from opencut.core.issue_report import bundle
        big_description = "A" * 120_000
        result = bundle(description=big_description)
        self.assertLessEqual(result["size_bytes"], 62_000)


# =========================================================================
# demo_bundle
# =========================================================================
class TestDemoBundle(unittest.TestCase):
    def test_list_assets_handles_missing_dir(self):
        from opencut.core import demo_bundle
        with patch.object(demo_bundle, "_demo_dir",
                          return_value="/does/not/exist"):
            result = demo_bundle.list_assets()
        self.assertEqual(result["assets"], [])
        self.assertIn("demo folder not found", result["note"])

    def test_get_sample_missing(self):
        from opencut.core import demo_bundle
        with patch.object(demo_bundle, "_demo_dir",
                          return_value="/does/not/exist"):
            result = demo_bundle.get_sample()
        self.assertFalse(result["exists"])
        self.assertEqual(result["path"], "")


# =========================================================================
# gist_sync
# =========================================================================
class TestGistSync(unittest.TestCase):
    def test_parse_gist_id_from_url(self):
        from opencut.core.gist_sync import _parse_gist_id
        gid = _parse_gist_id("https://gist.github.com/user/abcdef012345678901234")
        self.assertEqual(gid, "abcdef012345678901234")

    def test_parse_gist_id_bare(self):
        from opencut.core.gist_sync import _parse_gist_id
        gid = _parse_gist_id("abcdef012345678901234")
        self.assertEqual(gid, "abcdef012345678901234")

    def test_parse_gist_id_rejects_invalid(self):
        from opencut.core.gist_sync import _parse_gist_id
        with self.assertRaises(ValueError):
            _parse_gist_id("short")
        with self.assertRaises(ValueError):
            _parse_gist_id("https://evil.example.com/abcdef0123456789")
        with self.assertRaises(ValueError):
            _parse_gist_id("")

    def test_validate_files_rejects_traversal(self):
        from opencut.core.gist_sync import _validate_files
        with self.assertRaises(ValueError):
            _validate_files({"../etc/passwd.json": {}})
        with self.assertRaises(ValueError):
            _validate_files({"secrets.env": {}})   # wrong extension
        with self.assertRaises(ValueError):
            _validate_files({})                    # empty

    def test_validate_files_accepts_good(self):
        from opencut.core.gist_sync import _validate_files
        cleaned = _validate_files({
            "presets.json": {"foo": 1},
            "notes.md": "hello",
        })
        self.assertEqual(set(cleaned.keys()), {"presets.json", "notes.md"})
        self.assertIn('"foo"', cleaned["presets.json"]["content"])

    def test_push_refuses_anonymous_secret(self):
        """Anonymous clients cannot push secret gists."""
        from opencut.core.gist_sync import push
        with patch.dict(os.environ, {"GITHUB_TOKEN": ""}, clear=False):
            if os.environ.get("GITHUB_TOKEN"):
                # Active shell has a token — strip for this test.
                os.environ.pop("GITHUB_TOKEN", None)
            with self.assertRaises(ValueError):
                push({"x.json": {}}, public=False)


# =========================================================================
# onboarding
# =========================================================================
class TestOnboarding(unittest.TestCase):
    def setUp(self):
        # Use a temp "~/.opencut" so we don't clobber the real one.
        self._tmp = tempfile.mkdtemp(prefix="opencut_onboard_test_")
        from opencut import user_data as _u
        self._orig = _u._safe_user_filepath
        def _redirect(name):
            return os.path.join(self._tmp, name)
        _u._safe_user_filepath = _redirect

    def tearDown(self):
        from opencut import user_data as _u
        _u._safe_user_filepath = self._orig
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_default_state_is_unseen(self):
        from opencut.core.onboarding import get_state, reset
        reset()
        state = get_state()
        self.assertFalse(state["seen"])
        self.assertEqual(state["step"], 0)

    def test_set_state_patches_fields(self):
        from opencut.core.onboarding import reset, set_state, get_state
        reset()
        set_state(step=3)
        self.assertEqual(get_state()["step"], 3)
        self.assertFalse(get_state()["seen"])
        set_state(seen=True)
        self.assertTrue(get_state()["seen"])
        self.assertEqual(get_state()["step"], 3)  # step preserved

    def test_step_clamped_to_max(self):
        from opencut.core.onboarding import reset, set_state, MAX_STEP
        reset()
        set_state(step=999)
        from opencut.core.onboarding import get_state
        self.assertEqual(get_state()["step"], MAX_STEP)


# =========================================================================
# Tier 2 / Tier 3 availability checks
# =========================================================================
class TestWaveHStubAvailability(unittest.TestCase):
    """Every Wave H stub's check_X_available() must return without raising."""

    def test_all_stub_checks_return_bool(self):
        import opencut.checks as ck
        names = [
            "check_flashvsr_available", "check_rose_available",
            "check_sammie_available", "check_omnivoice_available",
            "check_reezsynth_available", "check_vidmuse_available",
            "check_video_agent_available", "check_gen_video_cloud_available",
            "check_lipsync_advanced_available",
        ]
        for name in names:
            fn = getattr(ck, name)
            self.assertIn(fn(), (True, False), f"{name} returned non-bool")

    def test_tier3_stubs_always_false(self):
        """Tier 3 check_X_available() must always return False in v1.25.0."""
        from opencut.core import video_agent, gen_video_cloud, lipsync_advanced
        self.assertFalse(video_agent.check_video_agent_available())
        self.assertFalse(gen_video_cloud.check_gen_video_cloud_available())
        self.assertFalse(lipsync_advanced.check_lipsync_advanced_available())


# =========================================================================
# Route integration — Wave H blueprint
# =========================================================================
class TestWaveHRoutes(unittest.TestCase):
    """End-to-end through Flask test client."""

    def setUp(self):
        from opencut.config import OpenCutConfig
        from opencut.server import create_app
        self.app = create_app(config=OpenCutConfig())
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        token_data = self.client.get("/health").get_json() or {}
        self.token = token_data.get("csrf_token", "")

    def _h(self):
        return {"X-OpenCut-Token": self.token,
                "Content-Type": "application/json"}

    def test_issue_report_bundle_get(self):
        r = self.client.get("/system/issue-report/bundle")
        self.assertEqual(r.status_code, 200)
        self.assertIn("url", r.get_json())

    def test_demo_list(self):
        r = self.client.get("/system/demo/list")
        self.assertEqual(r.status_code, 200)
        self.assertIn("assets", r.get_json())

    def test_flashvsr_info_is_available_flag(self):
        r = self.client.get("/video/upscale/flashvsr/info")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("available", body)
        self.assertIn("install_hint", body)

    def test_tier3_cloud_backends_always_unavailable(self):
        r = self.client.get("/generate/cloud/backends")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertFalse(body["available"])
        self.assertEqual(set(body["backends"]), {"hailuo", "seedance"})

    def test_tier3_agent_search_returns_501(self):
        r = self.client.post("/agent/search-footage",
                             json={"query": "foo"}, headers=self._h())
        self.assertEqual(r.status_code, 501)
        self.assertEqual(r.get_json().get("code"), "ROUTE_STUBBED")

    def test_tier2_flashvsr_returns_503(self):
        r = self.client.post("/video/upscale/flashvsr",
                             json={"filepath": "/tmp/x.mp4"},
                             headers=self._h())
        self.assertEqual(r.status_code, 503)
        body = r.get_json()
        self.assertEqual(body.get("code"), "MISSING_DEPENDENCY")

    def test_onboarding_roundtrip(self):
        # GET default
        r = self.client.get("/settings/onboarding")
        self.assertEqual(r.status_code, 200)
        # POST patch
        r = self.client.post("/settings/onboarding",
                             json={"step": 2}, headers=self._h())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("step"), 2)
        # Reset for isolation
        self.client.post("/settings/onboarding",
                         json={"seen": False, "step": 0}, headers=self._h())


if __name__ == "__main__":
    unittest.main()
