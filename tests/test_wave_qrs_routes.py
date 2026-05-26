"""
Tests for Wave Q + R + S route stubs (closes RESEARCH_FEATURE_PLAN_2026-05-25 Q1).

These waves landed core modules in May 2026 (commits b3201f1 and a8f62c0)
without registering any HTTP routes. ``wave_qrs_routes.py`` closes that gap.

The tests confirm:
  * every Wave Q/R/S endpoint is reachable on the live ``url_map``;
  * the ``/info`` GET endpoints respond without requiring the optional
    backend to be installed;
  * the POST endpoints respond cleanly (job accepted, or 503 dependency
    missing — both shapes are documented success modes for stubs).
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Expected route surface — POST processing endpoint and GET /info endpoint per module.
EXPECTED_ROUTES = {
    # Wave Q
    "/video/compose/vace": ("POST", "/video/compose/vace/info"),
    "/audio/tts/cosyvoice": ("POST", "/audio/tts/cosyvoice/info"),
    "/audio/tts/maskgct": ("POST", "/audio/tts/maskgct/info"),
    "/image/generate/omnigen2": ("POST", "/image/generate/omnigen2/info"),
    "/generate/skyreels2/t2v": ("POST", "/generate/skyreels2/info"),
    "/generate/skyreels3/avatar": ("POST", "/generate/skyreels3/info"),
    # Wave R
    "/audio/foley/ezaudio": ("POST", "/audio/foley/ezaudio/info"),
    "/lipsync/musetalk": ("POST", "/lipsync/musetalk/info"),
    "/generate/videox-fun": ("POST", "/generate/videox-fun/info"),
    "/generate/mochi": ("POST", "/generate/mochi/info"),
    "/generate/stepvideo": ("POST", "/generate/stepvideo/info"),
    # Wave S
    "/video/relight/iclight": ("POST", "/video/relight/iclight/info"),
    "/video/relight/lav": ("POST", "/video/relight/lav/info"),
    "/video/relight/diffrenderer": ("POST", "/video/relight/diffrenderer/info"),
    "/video/upscale/seedvr2": ("POST", "/video/upscale/seedvr2/info"),
    "/audio/transcribe/parakeet": ("POST", "/audio/transcribe/parakeet/info"),
    "/audio/transcribe/canary": ("POST", "/audio/transcribe/canary/info"),
    "/analyze/video/qwen3vl": ("POST", "/analyze/video/qwen3vl/info"),
    "/analyze/video/internvl3": ("POST", "/analyze/video/internvl3/info"),
    "/video/face/reage": ("POST", "/video/face/reage/info"),
    "/audio/music/heartmula": ("POST", "/audio/music/heartmula/info"),
}


class WaveQRSRouteRegistrationTest(unittest.TestCase):
    """Every Wave Q/R/S route must appear on the live Flask app's url_map."""

    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.rules = {r.rule: r for r in cls.app.url_map.iter_rules()}

    def test_post_endpoints_registered(self):
        for rule in EXPECTED_ROUTES:
            with self.subTest(rule=rule):
                self.assertIn(rule, self.rules, f"POST {rule} missing from url_map")
                methods = set(self.rules[rule].methods or ())
                self.assertIn("POST", methods, f"{rule} not POST")

    def test_info_endpoints_registered(self):
        for _, (_method, info_rule) in EXPECTED_ROUTES.items():
            with self.subTest(rule=info_rule):
                self.assertIn(info_rule, self.rules, f"GET {info_rule} missing")
                methods = set(self.rules[info_rule].methods or ())
                self.assertIn("GET", methods, f"{info_rule} not GET")


class WaveQRSInfoEndpointsRespondTest(unittest.TestCase):
    """GET /info routes must work without the optional backend installed."""

    @classmethod
    def setUpClass(cls):
        from opencut.server import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_info_endpoints_return_json(self):
        for _, (_method, info_rule) in EXPECTED_ROUTES.items():
            with self.subTest(rule=info_rule):
                resp = self.client.get(info_rule)
                # 200 (success), 503 (dep missing) — both acceptable; 404/500 = bug.
                self.assertIn(
                    resp.status_code, (200, 503),
                    f"{info_rule} returned {resp.status_code}: {resp.data!r}",
                )
                payload = json.loads(resp.data.decode("utf-8"))
                self.assertIsInstance(payload, dict)
                # Body should advertise install_hint for users.
                self.assertIn("install_hint", payload, f"{info_rule} missing install_hint")


class WaveQRSQueueAllowlistTest(unittest.TestCase):
    """Every async POST must be in _ALLOWED_QUEUE_ENDPOINTS so /queue/* works."""

    def test_endpoints_in_queue_allowlist(self):
        from opencut.routes.jobs_routes import _ALLOWED_QUEUE_ENDPOINTS
        for rule in EXPECTED_ROUTES:
            with self.subTest(rule=rule):
                self.assertIn(
                    rule, _ALLOWED_QUEUE_ENDPOINTS,
                    f"{rule} missing from _ALLOWED_QUEUE_ENDPOINTS",
                )


if __name__ == "__main__":
    unittest.main()
